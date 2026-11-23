"""Static-shape, XLA/TPU-optimized losses for VisDrone YOLOv11 + Slot Attention.

Key TPU/XLA optimizations:
  1. 100% static tensor shapes (fixed `max_gt`, zero boolean indexing like `x[fg_mask]`).
  2. Zero `.item()` calls and zero data-dependent Python `if` branches (no CPU-TPU sync).
  3. Fully vectorized Task-Aligned Assigner across batch dimension `B`.
  4. All GT padding and multi-scale `bg_mask`/`cls_mask` rasterization happen on CPU DataLoader workers.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import box_diou


class TPUTaskAlignedAssigner(nn.Module):
    """Fully vectorized, static-shape Task-Aligned Assigner for PyTorch/XLA (TPU)."""

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Static-shape vectorized assignment.

        Args:
            pred_scores: (B, N, C) sigmoid probabilities.
            pred_bboxes: (B, N, 4) decoded xyxy boxes in pixels.
            anchor_points: (N, 2) anchor center coordinates in pixels.
            gt_labels: (B, M, 1) class indices (long).
            gt_bboxes: (B, M, 4) padded GT xyxy boxes in pixels.
            mask_gt: (B, M, 1) float/bool valid GT mask (1.0 = valid, 0.0 = pad).

        Returns:
            target_bboxes: (B, N, 4) float32
            target_scores: (B, N, C) float32
            fg_mask: (B, N) float32 (1.0 = foreground, 0.0 = background)
        """
        pred_scores = pred_scores.float()
        pred_bboxes = pred_bboxes.float()
        anchor_points = anchor_points.float()
        gt_bboxes = gt_bboxes.float()
        gt_labels = gt_labels.long().squeeze(-1).clamp(0, self.num_classes - 1)  # (B, M)
        mask_gt_f = mask_gt.float()  # (B, M, 1)

        bs, num_anchors, _ = pred_scores.shape
        num_gt = gt_bboxes.shape[1]

        # 1. Anchor centers inside GT boxes: (B, M, N)
        # anchor_points: (1, 1, N, 2), gt_bboxes: (B, M, 1, 4)
        ap = anchor_points.view(1, 1, num_anchors, 2)
        lt = ap - gt_bboxes[:, :, None, :2]
        rb = gt_bboxes[:, :, None, 2:] - ap
        is_in_box = (torch.cat([lt, rb], dim=-1).amin(dim=-1) > 0.0).float()  # (B, M, N)

        # 2. Pairwise IoU between gt_bboxes (B, M, 4) and pred_bboxes (B, N, 4) -> (B, M, N)
        ious = self._batch_iou_3d(gt_bboxes, pred_bboxes)

        # 3. Gather predicted scores for GT classes -> (B, M, N)
        # pred_scores.transpose(1, 2): (B, C, N); index: (B, M, N)
        idx_cls = gt_labels.unsqueeze(-1).expand(-1, -1, num_anchors)  # (B, M, N)
        cls_scores = torch.gather(pred_scores.transpose(1, 2), dim=1, index=idx_cls)

        # 4. Joint alignment metric: (B, M, N)
        align_metric = (
            (cls_scores.clamp(min=0.0) ** self.alpha)
            * (ious.clamp(min=0.0) ** self.beta)
            * is_in_box
            * mask_gt_f
        )

        # 5. Top-K per GT along anchor dim
        k = min(self.topk, num_anchors)
        _, topk_indices = torch.topk(align_metric, k=k, dim=-1, largest=True)
        topk_mask = torch.zeros_like(align_metric)
        topk_mask.scatter_(dim=-1, index=topk_indices, value=1.0)

        is_pos = topk_mask * is_in_box * (align_metric > 0.0).float() * mask_gt_f  # (B, M, N)

        # 6. Resolve overlaps: pick GT with max alignment metric per anchor (static!)
        masked_metric = align_metric * is_pos
        anchor_gt_idx = masked_metric.argmax(dim=1)  # (B, N)
        fg_mask = (is_pos.sum(dim=1) > 0.0).float()  # (B, N)

        # 7. Normalize alignment metrics for soft target scores
        max_metrics = align_metric.amax(dim=-1, keepdim=True) + self.eps
        norm_align = (align_metric / max_metrics) * ious  # (B, M, N)

        # Gather assigned boxes: (B, N, 4)
        idx_box = anchor_gt_idx.unsqueeze(-1).expand(-1, -1, 4)
        target_bboxes = torch.gather(gt_bboxes, dim=1, index=idx_box) * fg_mask.unsqueeze(-1)

        # Gather assigned labels & normalized scores: (B, N)
        assigned_labels = torch.gather(gt_labels, dim=1, index=anchor_gt_idx)  # (B, N)
        norm_scores = torch.gather(norm_align, dim=1, index=anchor_gt_idx.unsqueeze(1)).squeeze(1)  # (B, N)
        norm_scores = norm_scores.clamp(0.0, 1.0) * fg_mask

        one_hot = F.one_hot(assigned_labels, num_classes=self.num_classes).float()  # (B, N, C)
        target_scores = one_hot * norm_scores.unsqueeze(-1)

        return target_bboxes, target_scores, fg_mask

    @staticmethod
    def _batch_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Vectorized pairwise IoU for boxes1 (B, M, 4) and boxes2 (B, N, 4) -> (B, M, N)."""
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            boxes1[:, :, 0:1],
            boxes1[:, :, 1:2],
            boxes1[:, :, 2:3],
            boxes1[:, :, 3:4],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            boxes2[:, None, :, 0],
            boxes2[:, None, :, 1],
            boxes2[:, None, :, 2],
            boxes2[:, None, :, 3],
        )

        inter_x1 = torch.maximum(b1_x1, b2_x1)
        inter_y1 = torch.maximum(b1_y1, b2_y1)
        inter_x2 = torch.minimum(b1_x2, b2_x2)
        inter_y2 = torch.minimum(b1_y2, b2_y2)
        inter = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)

        area1 = (b1_x2 - b1_x1).clamp(min=0.0) * (b1_y2 - b1_y1).clamp(min=0.0)
        area2 = (b2_x2 - b2_x1).clamp(min=0.0) * (b2_y2 - b2_y1).clamp(min=0.0)
        union = area1 + area2 - inter + eps
        return inter / union


class TPUYOLOv11DetectionLoss(nn.Module):
    """Static-shape YOLOv11 Detection Loss (DIoU + Focal/BCE + DFL) for TPU/XLA."""

    def __init__(
        self,
        num_classes: int = 10,
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 1.0,
        dfl_weight: float = 1.5,
        focal_alpha: float = -1.0,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_weight = float(box_weight)
        self.cls_weight = float(cls_weight)
        self.dfl_weight = float(dfl_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.assigner = TPUTaskAlignedAssigner(topk=10, num_classes=num_classes)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        mask_gt: torch.Tensor,
        decoded_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred_dist = preds["boxes"].transpose(1, 2).float()      # (B, N, 4*reg_max)
        pred_scores = preds["scores"].transpose(1, 2).float()   # (B, N, C)
        decoded_bboxes = decoded_bboxes.float()                 # (B, N, 4)
        anchor_points = anchor_points.float()                   # (N, 2)
        stride_tensor = stride_tensor.float()                   # (N, 1)

        bs, num_anchors, _ = pred_scores.shape

        # 1. Static vectorized assignment
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores=pred_scores.sigmoid(),
            pred_bboxes=decoded_bboxes,
            anchor_points=anchor_points,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            mask_gt=mask_gt,
        )

        # Tensor normalizer (NO .item()!)
        num_pos = fg_mask.sum().clamp(min=1.0)

        # 2. Classification loss (Focal or standard YOLOv11 BCE when gamma=0, alpha<0)
        ce_loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="none")
        if self.focal_gamma > 0.0:
            p = pred_scores.sigmoid()
            p_t = p * target_scores + (1.0 - p) * (1.0 - target_scores)
            ce_loss = ce_loss * ((1.0 - p_t) ** self.focal_gamma)
        if self.focal_alpha >= 0.0:
            alpha_t = self.focal_alpha * target_scores + (1.0 - self.focal_alpha) * (1.0 - target_scores)
            ce_loss = alpha_t * ce_loss
        loss_focal = (ce_loss.sum() / num_pos) * self.cls_weight

        # 3. DIoU loss over ALL anchors, masked by fg_mask (100% static shape!)
        diou = box_diou(decoded_bboxes, target_bboxes, xywh=False)  # (B, N)
        loss_diou = (((1.0 - diou) * fg_mask).sum() / num_pos) * self.box_weight

        # 4. DFL loss over ALL anchors, masked by fg_mask (100% static shape!)
        ap_exp = anchor_points.unsqueeze(0).expand(bs, -1, -1)       # (B, N, 2)
        st_exp = stride_tensor.unsqueeze(0).expand(bs, -1, -1)       # (B, N, 1)
        lt = ap_exp - target_bboxes[:, :, :2]
        rb = target_bboxes[:, :, 2:] - ap_exp
        target_ltrb = (torch.cat([lt, rb], dim=-1).clamp(min=0.0) / st_exp).clamp(0.0, self.reg_max - 1.01)

        tl = target_ltrb.long()
        tr = tl + 1
        wl = tr.float() - target_ltrb
        wr = 1.0 - wl

        pred_dist_flat = pred_dist.reshape(-1, self.reg_max)
        dfl_raw = (
            F.cross_entropy(pred_dist_flat, tl.reshape(-1), reduction="none") * wl.reshape(-1)
            + F.cross_entropy(pred_dist_flat, tr.reshape(-1), reduction="none") * wr.reshape(-1)
        ).view(bs, num_anchors, 4)
        loss_dfl = ((dfl_raw * fg_mask.unsqueeze(-1)).sum() / num_pos) * self.dfl_weight

        total_loss = loss_focal + loss_diou + loss_dfl
        return {
            "loss": total_loss,
            "loss_diou": loss_diou,
            "loss_focal": loss_focal,
            "loss_dfl": loss_dfl,
        }


class TPUSlotAuxiliaryLoss(nn.Module):
    """Static-shape Slot Auxiliary Loss for TPU/XLA (zero boolean indexing, zero .item())."""

    def __init__(
        self,
        num_classes: int = 10,
        bg_attn_weight: float = 0.5,
        fg_attn_weight: float = 0.5,
        suppress_weight: float = 0.05,
        cls_attn_weight: float = 0.5,
        ortho_weight: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bg_attn_weight = float(bg_attn_weight)
        self.fg_attn_weight = float(fg_attn_weight)
        self.suppress_weight = float(suppress_weight)
        self.cls_attn_weight = float(cls_attn_weight)
        self.ortho_weight = float(ortho_weight)

    def forward(
        self,
        slots: torch.Tensor,
        attn_ori: torch.Tensor,
        obj_queries: torch.Tensor,
        resolutions: Tuple[Tuple[int, int], ...],
        bg_mask: torch.Tensor,
        cls_mask: torch.Tensor,
        gt_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        slots = slots.float()
        attn_ori = attn_ori.float()
        obj_queries = obj_queries.float()
        bg_mask = bg_mask.float()
        cls_mask = cls_mask.float()
        gt_counts = gt_counts.float()

        device = attn_ori.device
        n3 = resolutions[0][0] * resolutions[0][1]
        n4 = resolutions[1][0] * resolutions[1][1]

        def _balanced_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            p = pred.clamp(1e-6, 1.0 - 1e-6)
            l3 = F.binary_cross_entropy(p[..., :n3], target[..., :n3])
            l4 = F.binary_cross_entropy(p[..., n3 : n3 + n4], target[..., n3 : n3 + n4])
            l5 = F.binary_cross_entropy(p[..., n3 + n4 :], target[..., n3 + n4 :])
            return (l3 + l4 + l5) / 3.0

        # 1. Background slot loss
        loss_bg = _balanced_bce(attn_ori[:, -1, :], bg_mask) if self.bg_attn_weight > 0 else torch.zeros((), device=device)

        # 2. Foreground sum loss
        if self.fg_attn_weight > 0:
            fg_attn_sum = attn_ori[:, : self.num_classes, :].sum(dim=1)
            loss_fg = _balanced_bce(fg_attn_sum, 1.0 - bg_mask)
        else:
            loss_fg = torch.zeros((), device=device)

        # 3. Inactive class suppression (masked multiplication — static!)
        if self.suppress_weight > 0:
            inactive = (gt_counts == 0.0).float().unsqueeze(-1)  # (B, K, 1)
            p_cls = attn_ori[:, : self.num_classes, :].clamp(1e-6, 1.0 - 1e-6)
            bce_zero = -torch.log(1.0 - p_cls)
            denom_inact = inactive.sum().clamp(min=1.0)
            l3 = (bce_zero[..., :n3] * inactive).sum() / (denom_inact * n3)
            l4 = (bce_zero[..., n3 : n3 + n4] * inactive).sum() / (denom_inact * n4)
            l5 = (bce_zero[..., n3 + n4 :] * inactive).sum() / (denom_inact * max(p_cls.shape[-1] - n3 - n4, 1))
            loss_suppress = (l3 + l4 + l5) / 3.0
        else:
            loss_suppress = torch.zeros((), device=device)

        # 4. Per-class Dice + Positive CE (masked multiplication — static!)
        if self.cls_attn_weight > 0:
            active = (gt_counts > 0.0).float()  # (B, K)
            p_cls = attn_ori[:, : self.num_classes, :].clamp(1e-6, 1.0 - 1e-6)
            inter = (p_cls * cls_mask).sum(dim=-1)
            card = p_cls.sum(dim=-1) + cls_mask.sum(dim=-1)
            dice = 1.0 - (2.0 * inter + 1.0) / (card + 1.0)
            loss_dice = (dice * active).sum() / active.sum().clamp(min=1.0)

            pos_mask = (cls_mask > 0.5).float() * active.unsqueeze(-1)
            loss_pos = (-torch.log(p_cls) * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)
            loss_cls_attn = loss_dice + 0.5 * loss_pos
        else:
            loss_cls_attn = torch.zeros((), device=device)

        # 5. Orthogonality loss (already static!)
        if self.ortho_weight > 0 and self.num_classes > 1:
            k = self.num_classes
            denom = float(k * (k - 1))
            eye_k = torch.eye(k, device=device, dtype=torch.float32)

            norm_slots = F.normalize(slots[:, :k, :], p=2, dim=-1)
            sim_s = torch.bmm(norm_slots, norm_slots.transpose(1, 2))
            loss_slot_ortho = ((sim_s * (1.0 - eye_k.unsqueeze(0))) ** 2).sum(dim=(-1, -2)).mean() / denom

            norm_q = F.normalize(obj_queries[:, :k, :], p=2, dim=-1)
            sim_q = torch.bmm(norm_q, norm_q.transpose(1, 2))[0] * (1.0 - eye_k)
            loss_q_ortho = (sim_q ** 2).sum() / denom

            loss_ortho = loss_slot_ortho + 0.5 * loss_q_ortho
        else:
            loss_ortho = torch.zeros((), device=device)

        slot_total = (
            self.bg_attn_weight * loss_bg
            + self.fg_attn_weight * loss_fg
            + self.suppress_weight * loss_suppress
            + self.cls_attn_weight * loss_cls_attn
            + self.ortho_weight * loss_ortho
        )

        return {
            "loss_slot": slot_total,
            "loss_bg_attn": loss_bg,
            "loss_fg_attn": loss_fg,
            "loss_suppress": loss_suppress,
            "loss_cls_attn": loss_cls_attn,
            "loss_ortho": loss_ortho,
        }
