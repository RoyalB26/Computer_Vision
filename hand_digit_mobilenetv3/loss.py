"""Loss module: Dense Gaussian Focal Loss for classification, DIoU Loss for box regression, and Deformable Regularization."""
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from BaseModel.core.registrar import LOSSES
from .utils import cxcywh_to_xyxy, xywh_to_xyxy


def compute_multi_ref_loss(
    ref_tensor: torch.Tensor,
    targets: Optional[List[Dict[str, torch.Tensor]]],
    device: torch.device,
    num_classes: int = 10,
    num_ref_per_obj: int = 4,
    repulsion_weight: float = 0.5,
) -> torch.Tensor:
    """
    Tính toán Ref Loss đa điểm:
    1. Nhiều ref point có thể trỏ về cùng 1 object (mỗi object nhận K điểm mục tiêu phân tán trong box).
    2. Các ref point trỏ về cùng 1 object phải có vị trí khác nhau (phạt repulsion nếu quá gần nhau).
    3. Bao quát công bằng mọi object trong ảnh (mỗi object đều có quota K điểm, không bị dồn hết vào 1 object).

    Args:
        ref_tensor: Tensor [B, 2, H, W] hoặc [B, N, 2] tọa độ reference points [0, 1].
        targets: List target dicts.
        device: torch.device.
        num_classes: Số lượng class đối tượng.
        num_ref_per_obj: Số lượng reference points phân bổ cho mỗi object (mặc định 4).
        repulsion_weight: Trọng số phạt đẩy khi các ref của cùng 1 object quá sát nhau (mặc định 0.5).

    Returns:
        torch.Tensor: Scalar loss cho reference points.
    """
    if targets is None or ref_tensor is None:
        return torch.tensor(0.0, device=device)

    batch_size = ref_tensor.shape[0]
    if ref_tensor.ndim == 4:
        ref_flat = ref_tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
    elif ref_tensor.ndim == 3:
        ref_flat = ref_tensor
    else:
        return torch.tensor(0.0, device=device)

    total_pull_loss = torch.tensor(0.0, device=device)
    total_rep_loss = torch.tensor(0.0, device=device)
    total_matched_pts = 0

    # Phân bố các điểm mục tiêu bên trong bounding box
    if num_ref_per_obj == 1:
        rel_offsets = torch.tensor([[0.0, 0.0]], device=device)
    elif num_ref_per_obj == 4:
        # 4 góc phần tư bên trong box
        rel_offsets = torch.tensor([
            [-0.25, -0.25],
            [0.25, -0.25],
            [-0.25, 0.25],
            [0.25, 0.25],
        ], device=device)
    elif num_ref_per_obj == 5:
        # Tâm + 4 góc phần tư
        rel_offsets = torch.tensor([
            [0.0, 0.0],
            [-0.25, -0.25],
            [0.25, -0.25],
            [-0.25, 0.25],
            [0.25, 0.25],
        ], device=device)
    else:
        thetas = torch.arange(num_ref_per_obj, device=device, dtype=torch.float32) * (2.0 * math.pi / num_ref_per_obj)
        rel_offsets = torch.stack([thetas.cos() * 0.25, thetas.sin() * 0.25], dim=-1)

    for b in range(batch_size):
        target_dict = targets[b] if b < len(targets) else None
        if not target_dict:
            continue

        gt_boxes = target_dict.get("boxes", torch.empty((0, 4), device=device))
        gt_labels = target_dict.get("labels", torch.empty((0,), device=device, dtype=torch.long))

        if len(gt_boxes) == 0:
            continue

        gt_boxes = gt_boxes.to(device)
        gt_labels = gt_labels.to(device)

        is_cxcywh = target_dict.get("_is_cxcywh", False)
        if is_cxcywh:
            cx = gt_boxes[:, 0].clamp(0.0, 0.999)
            cy = gt_boxes[:, 1].clamp(0.0, 0.999)
            w = gt_boxes[:, 2].clamp(0.001, 1.0)
            h = gt_boxes[:, 3].clamp(0.001, 1.0)
        else:
            cx = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0).clamp(0.0, 0.999)
            cy = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0).clamp(0.0, 0.999)
            w = gt_boxes[:, 2].clamp(0.001, 1.0)
            h = gt_boxes[:, 3].clamp(0.001, 1.0)

        valid_mask = (gt_labels >= 0) & (gt_labels < num_classes)
        cx = cx[valid_mask]
        cy = cy[valid_mask]
        w = w[valid_mask]
        h = h[valid_mask]

        num_gt = len(cx)
        if num_gt == 0:
            continue

        centers = torch.stack([cx, cy], dim=-1)
        wh = torch.stack([w, h], dim=-1)

        # Tạo K điểm mục tiêu khác nhau cho mỗi object: [num_gt, num_ref_per_obj, 2]
        target_points = (centers.unsqueeze(1) + rel_offsets.unsqueeze(0) * wh.unsqueeze(1)).clamp(0.0, 1.0)
        targets_flat = target_points.view(-1, 2)
        obj_ids = torch.arange(num_gt, device=device).unsqueeze(1).expand(num_gt, num_ref_per_obj).reshape(-1)
        obj_scale = torch.sqrt(w * h)

        b_refs = ref_flat[b]  # [N, 2]
        cost_matrix = torch.cdist(targets_flat, b_refs, p=1)
        cost_np = cost_matrix.detach().cpu().numpy()

        try:
            from scipy.optimize import linear_sum_assignment
            r_ind, c_ind = linear_sum_assignment(cost_np)
        except Exception:
            r_ind, c_ind = [], []
            cost_copy = cost_matrix.detach().clone()
            for _ in range(len(targets_flat)):
                min_idx = torch.argmin(cost_copy.view(-1))
                r = (min_idx // cost_copy.shape[1]).item()
                c = (min_idx % cost_copy.shape[1]).item()
                r_ind.append(r)
                c_ind.append(c)
                cost_copy[r, :] = float("inf")
                cost_copy[:, c] = float("inf")

        matched_refs = b_refs[c_ind]
        matched_targets = targets_flat[r_ind]
        matched_obj_ids = obj_ids[r_ind]

        # 1. Pull loss: Kéo các điểm ref về các vị trí đa dạng của object
        pull_loss = F.smooth_l1_loss(matched_refs, matched_targets, beta=0.01, reduction="sum")
        total_pull_loss += pull_loss
        total_matched_pts += len(matched_refs)

        # 2. Diversity / Repulsion loss: Ngăn ngừa các điểm ref của cùng 1 object trùng tọa độ nhau
        if num_ref_per_obj > 1:
            rep_loss_b = torch.tensor(0.0, device=device)
            for m in range(num_gt):
                mask_m = (matched_obj_ids == m)
                refs_m = matched_refs[mask_m]
                n_m = len(refs_m)
                if n_m > 1:
                    dists = torch.cdist(refs_m, refs_m, p=2)
                    margin_m = torch.clamp(0.25 * obj_scale[m], min=0.02, max=0.15)
                    triu_i, triu_j = torch.triu_indices(n_m, n_m, offset=1)
                    pair_dists = dists[triu_i, triu_j]
                    rep_penalties = F.relu(margin_m - pair_dists)
                    rep_loss_b += rep_penalties.sum()
            total_rep_loss += rep_loss_b

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pts_tensor = torch.tensor([total_matched_pts], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(pts_tensor, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        norm_pts = max(float(pts_tensor.item()) / float(world_size), 1.0)
    else:
        norm_pts = max(float(total_matched_pts), 1.0)

    final_loss = (total_pull_loss + repulsion_weight * total_rep_loss) / norm_pts
    return final_loss if total_matched_pts > 0 else (ref_tensor.sum() * 0.0)


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
        self.ref_weight = ref_weight
        self.obj_weight = obj_weight
        self.obj_pos_weight = obj_pos_weight
        self.deform_weight = deform_weight
        self.alpha = alpha

        # 3x3 Gaussian Kernel
        self.register_buffer(
            "gaussian_k",
            torch.tensor([
                [0.3679, 0.6065, 0.3679],
                [0.6065, 1.0000, 0.6065],
                [0.3679, 0.6065, 0.3679]
            ], dtype=torch.float32)
        )

    def forward(
        self,
        cls_logits: torch.Tensor,               # [B, num_classes, H, W]
        pred_boxes: torch.Tensor,               # [B, 4, H, W]
        targets: Optional[List[Dict[str, torch.Tensor]]],
        device: torch.device,
        sampling_info: Optional[Dict[str, torch.Tensor]] = None,
        obj_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape

        if obj_logits is None and sampling_info is not None and "obj_logits" in sampling_info:
            obj_logits = sampling_info["obj_logits"]

        total_box_loss = torch.tensor(0.0, device=device)
        total_ref_loss = torch.tensor(0.0, device=device)
        total_deform_in_box = torch.tensor(0.0, device=device)
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
                    gt_xyxy = cxcywh_to_xyxy(gt_boxes)
                else:
                    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0).clamp(0.0, 0.999)
                    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0).clamp(0.0, 0.999)
                    gt_xyxy = xywh_to_xyxy(gt_boxes)

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
                    total_box_loss += box_loss

                    # Ràng buộc Deformable In-Box Alignment (nếu có sampling_info)
                    if sampling_info is not None and "sampling_locations" in sampling_info and "attn_weights" in sampling_info:
                        b_locs = sampling_info["sampling_locations"][b]  # [H, W, nh, np, 2]
                        b_attn = sampling_info["attn_weights"][b]        # [H, W, nh, np]
                        pos_pts = b_locs[grid_y, grid_x]                 # [num_pos, nh, np, 2]
                        pos_w = b_attn[grid_y, grid_x]                   # [num_pos, nh, np]

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
                        total_deform_in_box += weighted_dist.sum()

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

        # Đồng bộ số lượng positive targets qua các GPU
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            num_pos_tensor = torch.tensor([num_pos_total], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(num_pos_tensor, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
            norm_pos = max(float(num_pos_tensor.item()) / float(world_size), 1.0)
        else:
            norm_pos = max(float(num_pos_total), 1.0)

        total_cls_loss = total_cls_loss / norm_pos
        total_box_loss = (total_box_loss / norm_pos) if num_pos_total > 0 else (total_box_loss * 0.0)

        # Multi-Point Reference Points Loss (nhiều ref trỏ về 1 obj, vị trí phân tán, bao quát mọi obj)
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
            )
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
        loss_deform = torch.tensor(0.0, device=device)
        total_deform_in_box = (total_deform_in_box / norm_pos) if num_pos_total > 0 else (total_deform_in_box * 0.0)
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

            loss_deform = 1.0 * total_deform_in_box + 0.2 * total_deform_bound + 0.01 * total_deform_reg

        total_loss = (
            self.cls_weight * total_cls_loss
            + self.box_weight * total_box_loss
            + self.ref_weight * total_ref_loss
            + self.obj_weight * total_obj_loss
            + self.deform_weight * loss_deform
        )

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
        }

        if sampling_info is not None:
            out["loss_deform"] = loss_deform
            out["loss_deform_in_box"] = total_deform_in_box
            out["loss_deform_bound"] = total_deform_bound
            out["loss_deform_reg"] = total_deform_reg

        return out
