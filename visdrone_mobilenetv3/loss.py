"""Loss module: Dense Gaussian Focal Loss for classification, DIoU Loss for box regression, and Deformable Regularization."""
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from BaseModel.core.registrar import LOSSES
from .utils import cxcywh_to_xyxy, xywh_to_xyxy


def compute_multi_ref_loss(
    ref_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: Optional[List[Dict[str, torch.Tensor]]],
    device: torch.device,
    num_classes: int = 10,
    num_ref_per_obj: int = 4,
    repulsion_weight: float = 0.5,
    scale_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Tính toán Ref Loss đa điểm (Vector hóa 100% trên GPU - không đồng bộ CPU, không all_reduce tránh deadlock):
    1. Nhiều ref point có thể trỏ về cùng 1 object (mỗi object nhận K điểm mục tiêu phân tán trong box).
    2. Các ref point trỏ về cùng 1 object phải có vị trí khác nhau (phạt repulsion nếu quá gần nhau).
    3. Bao quát công bằng mọi object trong ảnh (mỗi object đều có quota K điểm, không bị dồn hết vào 1 object).

    Args:
        ref_tensor: Tensor [B, 2, H, W] hoặc [B, N, 2] tọa độ reference points [0, 1], hoặc Dict của chúng.
        targets: List target dicts.
        device: torch.device.
        num_classes: Số lượng class đối tượng.
        num_ref_per_obj: Số lượng reference points phân bổ cho mỗi object (mặc định 4).
        repulsion_weight: Trọng số phạt đẩy khi các ref của cùng 1 object quá sát nhau (mặc định 0.5).

    Returns:
        torch.Tensor: Scalar loss cho reference points.
    """
    if isinstance(ref_tensor, dict):
        losses = [
            compute_multi_ref_loss(
                ref_tensor=v,
                targets=targets,
                device=device,
                num_classes=num_classes,
                num_ref_per_obj=num_ref_per_obj,
                repulsion_weight=repulsion_weight,
            )
            for v in ref_tensor.values()
        ]
        return sum(losses) / max(len(losses), 1)

    if targets is None or ref_tensor is None:
        return ref_tensor.sum() * 0.0 if ref_tensor is not None else torch.tensor(0.0, device=device)

    batch_size = ref_tensor.shape[0]
    if ref_tensor.ndim == 4:
        ref_flat = ref_tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
    elif ref_tensor.ndim == 3:
        ref_flat = ref_tensor
    else:
        return ref_tensor.sum() * 0.0

    boxes_list = []
    labels_list = []
    batch_idx_list = []

    for b in range(batch_size):
        target_dict = targets[b] if b < len(targets) else None
        if not target_dict:
            continue
        gt_boxes = target_dict.get("boxes", None)
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            continue

        gt_boxes = gt_boxes.to(device)
        if not target_dict.get("_is_cxcywh", False):
            gt_boxes = torch.stack([
                gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0,
                gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0,
                gt_boxes[:, 2],
                gt_boxes[:, 3],
            ], dim=-1)

        gt_labels = target_dict.get("labels", None)
        if gt_labels is not None:
            gt_labels = gt_labels.to(device)

        if scale_range is not None and gt_boxes.shape[0] > 0:
            w = gt_boxes[:, 2]
            h = gt_boxes[:, 3]
            s = torch.sqrt(w * h)
            scale_mask = (s >= scale_range[0]) & (s <= scale_range[1])
            gt_boxes = gt_boxes[scale_mask]
            if gt_labels is not None:
                gt_labels = gt_labels[scale_mask]
            if gt_boxes.shape[0] == 0:
                continue

        boxes_list.append(gt_boxes)
        if gt_labels is not None:
            labels_list.append(gt_labels)
        batch_idx_list.append(torch.full((gt_boxes.shape[0],), b, device=device, dtype=torch.long))

    if not boxes_list:
        return ref_tensor.sum() * 0.0

    all_boxes = torch.cat(boxes_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    total_m = all_boxes.shape[0]
    if total_m == 0:
        return ref_tensor.sum() * 0.0

    dtype = ref_flat.dtype
    if len(labels_list) == len(boxes_list):
        all_labels = torch.cat(labels_list, dim=0)
        valid_f = ((all_labels >= 0) & (all_labels < num_classes)).to(dtype)
    else:
        valid_f = torch.ones(total_m, device=device, dtype=dtype)

    # Phân bố các điểm mục tiêu bên trong bounding box
    if num_ref_per_obj == 1:
        rel_offsets = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
    elif num_ref_per_obj == 4:
        rel_offsets = torch.tensor([
            [-0.25, -0.25],
            [0.25, -0.25],
            [-0.25, 0.25],
            [0.25, 0.25],
        ], device=device, dtype=dtype)
    elif num_ref_per_obj == 5:
        rel_offsets = torch.tensor([
            [0.0, 0.0],
            [-0.25, -0.25],
            [0.25, -0.25],
            [-0.25, 0.25],
            [0.25, 0.25],
        ], device=device, dtype=dtype)
    else:
        thetas = torch.arange(num_ref_per_obj, device=device, dtype=dtype) * (2.0 * math.pi / num_ref_per_obj)
        rel_offsets = torch.stack([thetas.cos() * 0.25, thetas.sin() * 0.25], dim=-1)

    cx = all_boxes[:, 0].clamp(0.0, 0.999)
    cy = all_boxes[:, 1].clamp(0.0, 0.999)
    w = all_boxes[:, 2].clamp(0.001, 1.0)
    h = all_boxes[:, 3].clamp(0.001, 1.0)

    centers = torch.stack([cx, cy], dim=-1)
    wh = torch.stack([w, h], dim=-1)

    target_points = (centers.unsqueeze(1) + rel_offsets.unsqueeze(0) * wh.unsqueeze(1)).clamp(0.0, 1.0)
    obj_refs = ref_flat[batch_idx]

    # Matching 100% trên GPU trong no_grad (tránh CPU-GPU sync)
    with torch.no_grad():
        cost = (target_points.unsqueeze(2) - obj_refs.unsqueeze(1)).abs().sum(dim=-1)
        chosen_indices = []
        for k in range(num_ref_per_obj):
            idx_k = cost[:, k, :].argmin(dim=-1)
            chosen_indices.append(idx_k)
            if k + 1 < num_ref_per_obj:
                cost.scatter_(2, idx_k.view(-1, 1, 1).expand(-1, num_ref_per_obj, 1), float("inf"))
        min_c = torch.stack(chosen_indices, dim=1)

    matched_refs = torch.gather(obj_refs, 1, min_c.unsqueeze(-1).expand(-1, -1, 2))

    pull_per_obj = F.smooth_l1_loss(matched_refs, target_points, beta=0.01, reduction="none").sum(dim=(1, 2))
    total_pull_loss = (pull_per_obj * valid_f).sum()

    if num_ref_per_obj > 1:
        triu_i, triu_j = torch.triu_indices(num_ref_per_obj, num_ref_per_obj, offset=1, device=device)
        pair_dists = torch.norm(matched_refs[:, triu_i] - matched_refs[:, triu_j], p=2, dim=-1)
        obj_scale = torch.sqrt(w * h)
        margins = torch.clamp(0.25 * obj_scale, min=0.02, max=0.15).unsqueeze(1)
        rep_per_obj = F.relu(margins - pair_dists).sum(dim=-1)
        total_rep_loss = (rep_per_obj * valid_f).sum()
    else:
        total_rep_loss = ref_tensor.sum() * 0.0

    # Chuẩn hóa cục bộ trên mỗi GPU (DDP tự động trung bình hóa gradient khi backward)
    norm_pts = (valid_f.sum() * num_ref_per_obj).clamp(min=1.0)
    return (total_pull_loss + repulsion_weight * total_rep_loss) / norm_pts + (ref_tensor.sum() * 0.0)


@LOSSES.register_module(name="MobileNetDetectionLoss")
class MobileNetDetectionLoss(nn.Module):
    """
    Hàm mất mát gồm:
    - Gaussian Focal Loss cho classification heatmap
    - Distance-IoU (DIoU) Loss cho bounding box regression
    - Multi-point Ref Loss phân tán đa điểm, bao quát mọi object
    - Deformable Regularization (nếu sử dụng DeformableHead)
    """

    def __init__(
        self,
        num_classes: int = 10,
        cls_weight: float = 2.0,
        box_weight: float = 1.0,
        ref_weight: float = 10.0,
        obj_weight: float = 1.0,
        obj_pos_weight: float = 5.0,
        deform_weight: float = 0.5,
        alpha: float = 0.25,
        num_ref_per_obj: int = 4,
        ref_repulsion_weight: float = 0.5,
        scale_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.ref_weight = ref_weight
        self.obj_weight = obj_weight
        self.obj_pos_weight = obj_pos_weight
        self.deform_weight = deform_weight
        self.alpha = alpha
        self.num_ref_per_obj = num_ref_per_obj
        self.ref_repulsion_weight = ref_repulsion_weight
        self.scale_ranges = scale_ranges

        # 3x3 Gaussian Kernel
        self.register_buffer(
            "gaussian_k",
            torch.tensor([
                [0.3679, 0.6065, 0.3679],
                [0.6065, 1.0000, 0.6065],
                [0.3679, 0.6065, 0.3679]
            ], dtype=torch.float32)
        )

    def _forward_single_level(
        self,
        cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]],
        device: torch.device,
        sampling_info: Optional[Dict[str, Any]] = None,
        obj_logits: Optional[torch.Tensor] = None,
        scale_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape

        if obj_logits is None and sampling_info is not None and "obj_logits" in sampling_info:
            obj_logits = sampling_info["obj_logits"]

        # Khởi tạo loss gắn với đồ thị tính toán để DDP không bao giờ báo thiếu gradient
        total_box_loss = pred_boxes.sum() * 0.0
        total_deform_in_box = cls_logits.sum() * 0.0
        if sampling_info is not None:
            if "sampling_locations" in sampling_info:
                total_deform_in_box = total_deform_in_box + sampling_info["sampling_locations"].sum() * 0.0
            if "attn_weights" in sampling_info:
                total_deform_in_box = total_deform_in_box + sampling_info["attn_weights"].sum() * 0.0

        num_pos_total = 0

        # Khởi tạo center mask cho toàn bộ batch [B, num_classes, H, W]
        cls_center = torch.zeros(
            (batch_size, num_classes, grid_h, grid_w), device=device, dtype=cls_logits.dtype
        )

        for b in range(batch_size):
            img_box = pred_boxes[b]  # [4, H, W]
            target_dict = targets[b] if targets is not None and b < len(targets) else None
            gt_boxes = (
                target_dict.get("boxes", torch.empty((0, 4), device=device))
                if target_dict
                else torch.empty((0, 4), device=device)
            )
            gt_labels = (
                target_dict.get("labels", torch.empty((0,), device=device, dtype=torch.long))
                if target_dict
                else torch.empty((0,), device=device, dtype=torch.long)
            )
            num_gt = len(gt_boxes)

            if num_gt > 0:
                gt_boxes = gt_boxes.to(device)
                gt_labels = gt_labels.to(device)

                is_cxcywh = target_dict.get("_is_cxcywh", False) if target_dict else False

                if is_cxcywh:
                    gt_cx = gt_boxes[:, 0].clamp(0.0, 0.999)
                    gt_cy = gt_boxes[:, 1].clamp(0.0, 0.999)
                    gt_w = gt_boxes[:, 2].clamp(min=1e-4)
                    gt_h = gt_boxes[:, 3].clamp(min=1e-4)
                    gt_xyxy = cxcywh_to_xyxy(gt_boxes)
                else:
                    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0).clamp(0.0, 0.999)
                    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0).clamp(0.0, 0.999)
                    gt_w = gt_boxes[:, 2].clamp(min=1e-4)
                    gt_h = gt_boxes[:, 3].clamp(min=1e-4)
                    gt_xyxy = xywh_to_xyxy(gt_boxes)

                if scale_range is not None:
                    scale = torch.sqrt(gt_w * gt_h)
                    scale_mask = (scale >= scale_range[0]) & (scale <= scale_range[1])
                    gt_cx = gt_cx[scale_mask]
                    gt_cy = gt_cy[scale_mask]
                    gt_labels = gt_labels[scale_mask]
                    gt_xyxy = gt_xyxy[scale_mask]

                if len(gt_labels) > 0:
                    grid_x = (gt_cx * grid_w).long().clamp(0, grid_w - 1)
                    grid_y = (gt_cy * grid_h).long().clamp(0, grid_h - 1)

                    valid_mask = (gt_labels >= 0) & (gt_labels < num_classes)
                    valid_labels = gt_labels[valid_mask]
                    grid_x = grid_x[valid_mask]
                    grid_y = grid_y[valid_mask]
                    gt_xyxy = gt_xyxy[valid_mask]

                    num_pos = len(valid_labels)
                    num_pos_total += num_pos
                    if num_pos > 0:
                        cls_center[b, valid_labels, grid_y, grid_x] = 1.0

                        # Lấy box dự đoán [4, num_pos] -> [num_pos, 4]
                        p_box = img_box[:, grid_y, grid_x].permute(1, 0)
                        p_xyxy = cxcywh_to_xyxy(p_box)

                        # Distance-IoU (DIoU) Loss
                        box_loss = torchvision.ops.distance_box_iou_loss(
                            p_xyxy, gt_xyxy, reduction="sum"
                        )
                        total_box_loss = total_box_loss + box_loss

                        # Ràng buộc Deformable In-Box Alignment (nếu có sampling_info)
                        if sampling_info is not None and "sampling_locations" in sampling_info and "attn_weights" in sampling_info:
                            b_locs = sampling_info["sampling_locations"][b]
                            b_attn = sampling_info["attn_weights"][b]
                            pos_pts = b_locs[grid_y, grid_x]
                            pos_w = b_attn[grid_y, grid_x]

                            px = pos_pts[..., 0]
                            py = pos_pts[..., 1]

                            x1 = gt_xyxy[:, 0].view(num_pos, 1, 1)
                            y1 = gt_xyxy[:, 1].view(num_pos, 1, 1)
                            x2 = gt_xyxy[:, 2].view(num_pos, 1, 1)
                            y2 = gt_xyxy[:, 3].view(num_pos, 1, 1)

                            dx = F.relu(x1 - px) + F.relu(px - x2)
                            dy = F.relu(y1 - py) + F.relu(py - y2)
                            in_box_dist = dx + dy
                            weighted_dist = (in_box_dist * pos_w).sum(dim=(-2, -1))
                            total_deform_in_box = total_deform_in_box + weighted_dist.sum()

        # 1. Vectorized Gaussian Heatmap Target Generation
        k = self.gaussian_k.to(device=device, dtype=cls_logits.dtype)
        padded = F.pad(cls_center, (1, 1, 1, 1), mode="constant", value=0.0)
        shifts = [
            padded[:, :, dy : dy + grid_h, dx : dx + grid_w] * k[dy, dx]
            for dy in range(3)
            for dx in range(3)
        ]
        cls_target = torch.stack(shifts, dim=0).amax(dim=0)  # [B, num_classes, grid_h, grid_w]

        # 2. Vectorized Dense Gaussian Focal Loss
        alpha = self.alpha
        probs = cls_logits.sigmoid().clamp(min=1e-6, max=1.0 - 1e-6)
        pos_mask = (cls_target >= 0.99).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = -alpha * ((1.0 - probs) ** 2.0) * torch.log(probs) * pos_mask
        neg_weight = (1.0 - alpha) * ((1.0 - cls_target) ** 4.0) * (probs ** 2.0)
        neg_loss = -neg_weight * torch.log(1.0 - probs) * neg_mask

        total_cls_pos = pos_loss.sum()
        total_cls_neg = neg_loss.sum()
        total_cls_loss = total_cls_pos + total_cls_neg

        with torch.no_grad():
            prob_pos_sum = (probs * pos_mask).sum()
            true_bg_mask = cls_target < 0.05
            prob_neg_max = (
                probs[true_bg_mask].max() if true_bg_mask.any() else torch.tensor(0.0, device=device)
            )

        # Multi-Point Reference Points Loss
        ref_tensor = None
        if sampling_info is not None and "ref" in sampling_info:
            ref_tensor = sampling_info["ref"]

        if ref_tensor is not None and targets is not None:
            total_ref_loss = compute_multi_ref_loss(
                ref_tensor=ref_tensor,
                targets=targets,
                device=device,
                num_classes=num_classes,
                num_ref_per_obj=self.num_ref_per_obj,
                repulsion_weight=self.ref_repulsion_weight,
                scale_range=scale_range,
            )
        elif ref_tensor is not None:
            total_ref_loss = ref_tensor.sum() * 0.0
        else:
            total_ref_loss = torch.tensor(0.0, device=device)

        # 3. Objectness Loss
        total_obj_loss = torch.tensor(0.0, device=device)
        if obj_logits is not None:
            obj_target = (cls_center.sum(dim=1, keepdim=True) > 0).float()
            pos_weight = torch.tensor([self.obj_pos_weight], device=device, dtype=obj_logits.dtype)
            total_obj_loss = F.binary_cross_entropy_with_logits(
                obj_logits, obj_target, pos_weight=pos_weight
            )

        # 4. Deformable Regularization
        total_deform_bound = torch.tensor(0.0, device=device)
        total_deform_reg = torch.tensor(0.0, device=device)

        if sampling_info is not None and "sampling_locations" in sampling_info and "offsets" in sampling_info:
            all_locs = sampling_info["sampling_locations"]
            all_offsets = sampling_info["offsets"]

            bound_violation = (
                F.relu(-all_locs[..., 0]) + F.relu(all_locs[..., 0] - 1.0) +
                F.relu(-all_locs[..., 1]) + F.relu(all_locs[..., 1] - 1.0)
            )
            total_deform_bound = bound_violation.mean()
            total_deform_reg = (all_offsets ** 2).mean()

        return {
            "num_pos": num_pos_total,
            "cls_loss_unnorm": total_cls_loss,
            "box_loss_unnorm": total_box_loss,
            "obj_loss": total_obj_loss,
            "ref_loss": total_ref_loss,
            "total_cls_pos": total_cls_pos,
            "total_cls_neg": total_cls_neg,
            "prob_pos_sum": prob_pos_sum,
            "prob_neg_max": prob_neg_max,
            "total_deform_in_box_unnorm": total_deform_in_box,
            "total_deform_bound": total_deform_bound,
            "total_deform_reg": total_deform_reg,
        }

    def forward(
        self,
        cls_logits: Union[torch.Tensor, Dict[str, torch.Tensor]],
        pred_boxes: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]],
        device: torch.device,
        sampling_info: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        obj_logits: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(cls_logits, torch.Tensor):
            cls_dict = {"single": cls_logits}
            boxes_dict = {"single": pred_boxes}
            obj_dict = {"single": obj_logits} if obj_logits is not None else {}
            samp_dict = {"single": sampling_info} if sampling_info is not None else {}
            range_dict = {"single": None}
        else:
            cls_dict = cls_logits
            boxes_dict = pred_boxes
            obj_dict = obj_logits if isinstance(obj_logits, dict) else {}
            samp_dict = sampling_info if isinstance(sampling_info, dict) else {}
            range_dict = self.scale_ranges or {}

        results = {}
        for k in cls_dict.keys():
            k_cls = cls_dict[k]
            k_boxes = boxes_dict[k]
            k_obj = obj_dict.get(k, None)
            k_samp = samp_dict.get(k, None)
            k_range = range_dict.get(k, None)
            results[k] = self._forward_single_level(
                cls_logits=k_cls,
                pred_boxes=k_boxes,
                targets=targets,
                device=device,
                sampling_info=k_samp,
                obj_logits=k_obj,
                scale_range=k_range,
            )

        num_levels = max(len(results), 1)
        total_num_pos = sum(res["num_pos"] for res in results.values())
        norm_pos = max(float(total_num_pos), 1.0)

        total_cls_loss = sum(res["cls_loss_unnorm"] for res in results.values()) / norm_pos
        total_box_loss = sum(res["box_loss_unnorm"] for res in results.values()) / norm_pos
        total_obj_loss = sum(res["obj_loss"] for res in results.values()) / num_levels
        total_ref_loss = sum(res["ref_loss"] for res in results.values()) / num_levels

        total_deform_in_box = sum(res["total_deform_in_box_unnorm"] for res in results.values()) / norm_pos
        total_deform_bound = sum(res["total_deform_bound"] for res in results.values()) / num_levels
        total_deform_reg = sum(res["total_deform_reg"] for res in results.values()) / num_levels
        loss_deform = 1.0 * total_deform_in_box + 0.2 * total_deform_bound + 0.01 * total_deform_reg

        total_loss = (
            self.cls_weight * total_cls_loss
            + self.box_weight * total_box_loss
            + self.ref_weight * total_ref_loss
            + self.obj_weight * total_obj_loss
            + self.deform_weight * loss_deform
        )

        total_cls_pos = sum(res["total_cls_pos"] for res in results.values())
        total_cls_neg = sum(res["total_cls_neg"] for res in results.values())
        prob_pos_sum = sum(res["prob_pos_sum"] for res in results.values())
        prob_neg_max = torch.stack([res["prob_neg_max"] for res in results.values()]).max()

        out = {
            "loss": total_loss,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
            "loss_ref": total_ref_loss,
            "loss_obj": total_obj_loss,
            "loss_cls_pos": (total_cls_pos / norm_pos).detach(),
            "loss_cls_neg": (total_cls_neg / norm_pos).detach(),
            "loss_prob_pos": (prob_pos_sum / norm_pos).detach(),
            "loss_prob_neg_max": prob_neg_max.detach(),
            "loss_deform": loss_deform,
            "loss_deform_in_box": total_deform_in_box,
            "loss_deform_bound": total_deform_bound,
            "loss_deform_reg": total_deform_reg,
        }

        # Bổ sung chi tiết từng level nếu có nhiều hơn 1 level
        if num_levels > 1:
            for k, res in results.items():
                out[f"loss_cls_{k}"] = (res["cls_loss_unnorm"] / norm_pos).detach()
                out[f"loss_box_{k}"] = (res["box_loss_unnorm"] / norm_pos).detach()

        return out
