"""Loss module: Dense Gaussian Focal Loss for classification, DIoU Loss for box regression, and Deformable Regularization."""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from BaseModel.core.registrar import LOSSES
from .utils import cxcywh_to_xyxy, xywh_to_xyxy


@LOSSES.register_module(name="MobileNetDetectionLoss")
class MobileNetDetectionLoss(nn.Module):
    """
    Hàm mất mát gồm:
    - Gaussian Focal Loss cho classification heatmap
    - Distance-IoU (DIoU) Loss cho bounding box regression
    - Deformable Regularization (nếu sử dụng DeformableHead)
    """

    def __init__(
        self,
        num_classes: int = 10,
        cls_weight: float = 2.0,
        box_weight: float = 1.0,
        deform_weight: float = 0.5,
        alpha: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
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
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape

        total_box_loss = torch.tensor(0.0, device=device)
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
                    if sampling_info is not None:
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

        # 3. Deformable Regularization
        loss_deform = torch.tensor(0.0, device=device)
        total_deform_in_box = (total_deform_in_box / norm_pos) if num_pos_total > 0 else (total_deform_in_box * 0.0)
        total_deform_bound = torch.tensor(0.0, device=device)
        total_deform_reg = torch.tensor(0.0, device=device)

        if sampling_info is not None:
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
            + self.deform_weight * loss_deform
        )

        out = {
            "loss": total_loss,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
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
