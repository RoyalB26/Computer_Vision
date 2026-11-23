"""Loss functions for YOLOv11 on VisDrone dataset: DIoU Loss, Focal Loss, and DFL."""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from BaseModel.core import LOSSES
from BaseModel.loss.base_loss import BaseLoss


# =========================================================================
# 1. DIoU Loss (Distance-IoU)
# =========================================================================

def box_diou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate Distance-IoU (DIoU) between two sets of bounding boxes.

    Args:
        box1: [N, 4] tensor of boxes.
        box2: [N, 4] or [M, 4] tensor of boxes.
        xywh: If True, input is [cx, cy, w, h]; if False, input is [x1, y1, x2, y2].
        eps: Small constant to avoid zero division.

    Returns:
        Tensor of DIoU values in range [-1, 1].
    """
    if xywh:
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2

        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union area
    b1_area = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    b2_area = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union_area = b1_area + b2_area - inter_area + eps
    iou = inter_area / union_area

    # Center distance squared
    c1_x = (b1_x1 + b1_x2) / 2
    c1_y = (b1_y1 + b1_y2) / 2
    c2_x = (b2_x1 + b2_x2) / 2
    c2_y = (b2_y1 + b2_y2) / 2
    rho2 = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2

    # Smallest enclosing box diagonal squared
    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)
    enc_w = (enc_x2 - enc_x1).clamp(min=0)
    enc_h = (enc_y2 - enc_y1).clamp(min=0)
    c2 = enc_w ** 2 + enc_h ** 2 + eps

    # DIoU = IoU - rho^2 / c^2
    diou = iou - (rho2 / c2)
    return diou


@LOSSES.register_module(force=True)
class DIoULoss(BaseLoss):
    """Distance-IoU (DIoU) Loss module.

    Encourages predicted boxes to overlap with ground truth while directly minimizing
    the normalized distance between their center points.

    Args:
        loss_weight: Multiplier weight for this loss component.
        reduction: 'mean', 'sum', or 'none'.
        eps: Small constant to prevent division by zero.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__(loss_weight=loss_weight)
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        xywh: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute DIoU loss: 1.0 - DIoU."""
        if preds.numel() == 0 or targets.numel() == 0:
            return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

        # Try torchvision's native ops if available and format is xyxy
        if not xywh and hasattr(torch, "ops") and hasattr(torch.ops, "torchvision") and hasattr(torch.ops.torchvision, "distance_box_iou_loss"):
            try:
                import torchvision.ops as ops
                return ops.distance_box_iou_loss(preds, targets, reduction=self.reduction) * self.loss_weight
            except Exception:
                pass

        diou = box_diou(preds, targets, xywh=xywh, eps=self.eps)
        loss = 1.0 - diou

        if self.reduction == "mean":
            return loss.mean() * self.loss_weight
        elif self.reduction == "sum":
            return loss.sum() * self.loss_weight
        return loss * self.loss_weight


# =========================================================================
# 2. Focal Loss for Multi-label Classification
# =========================================================================

@LOSSES.register_module(force=True)
class SigmoidFocalLoss(BaseLoss):
    """Sigmoid Focal Loss for dense multi-class / multi-label detection.

    Addresses extreme foreground-background imbalance by dynamically downweighting
    easy examples (background) and focusing on hard negative/positive samples.

    Args:
        loss_weight: Multiplier weight for this loss component.
        alpha: Weighting factor for positive class (default 0.25).
        gamma: Focusing parameter for modulating factor (default 2.0).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "sum",
    ):
        super().__init__(loss_weight=loss_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        normalizer: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute Sigmoid Focal Loss.

        Args:
            preds: Logits tensor of arbitrary shape [B, N, C] or [N, C].
            targets: Target binary scores tensor matching preds shape with values in [0, 1].
            normalizer: Optional scalar denominator to normalize sum loss (e.g., num_positives).
        """
        p = torch.sigmoid(preds)
        ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")

        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if normalizer is not None and normalizer > 0:
            return (loss.sum() / normalizer) * self.loss_weight

        if self.reduction == "mean":
            return loss.mean() * self.loss_weight
        elif self.reduction == "sum":
            return loss.sum() * self.loss_weight
        return loss * self.loss_weight


# =========================================================================
# 3. Distribution Focal Loss (DFL) for Fine-Grained Box Boundary Regression
# =========================================================================

@LOSSES.register_module(force=True)
class DistributionFocalLoss(BaseLoss):
    """Distribution Focal Loss (DFL) from Generalized Focal Loss (GFL).

    Represents bounding box coordinate offsets as a general probability distribution
    over discrete bins instead of single scalar Dirac delta representations.
    """

    def __init__(self, loss_weight: float = 0.5, reg_max: int = 16):
        super().__init__(loss_weight=loss_weight)
        self.reg_max = reg_max

    def forward(
        self,
        pred_dist: torch.Tensor,
        target_dist: torch.Tensor,
        normalizer: Optional[float] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute DFL (Ultralytics-aligned mean over 4 edges with optional per-anchor TAL weight).

        Args:
            pred_dist: [N, 4, reg_max] predicted logits distribution.
            target_dist: [N, 4] continuous ground truth distance in stride units.
            normalizer: Normalization scalar (e.g. target_scores_sum).
            weight: Optional [N] per-anchor TAL quality weight.
        """
        num_pos = target_dist.shape[0]
        target_dist = target_dist.clamp(0, self.reg_max - 1.01)
        tl = target_dist.long()           # left bin integer index
        tr = tl + 1                      # right bin integer index
        wl = tr.float() - target_dist    # left bin weight
        wr = 1.0 - wl                    # right bin weight

        pred_dist = pred_dist.view(-1, self.reg_max)
        tl = tl.view(-1)
        tr = tr.view(-1)
        wl = wl.view(-1)
        wr = wr.view(-1)

        loss = (
            F.cross_entropy(pred_dist, tl, reduction="none") * wl
            + F.cross_entropy(pred_dist, tr, reduction="none") * wr
        ).view(num_pos, 4).mean(dim=-1)  # [N], mean across 4 edges (Ultralytics standard)

        if weight is not None:
            loss = loss * weight

        if normalizer is not None and normalizer > 0:
            return (loss.sum() / normalizer) * self.loss_weight
        return loss.mean() * self.loss_weight


# =========================================================================
# 4. Task-Aligned Assigner (TAL) for Dynamic Positive Anchor Assignment
# =========================================================================

class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner (TAL) used in modern YOLO architectures (YOLOv8/11).

    Selects positive anchors based on a joint alignment metric:
        metric = (pred_score ^ alpha) * (IoU ^ beta)
    within candidate anchors whose center points fall inside ground-truth boxes.

    Args:
        topk: Number of top-scoring anchors selected per ground-truth box (default 10).
        num_classes: Total number of classes.
        alpha: Classification exponent weight.
        beta: Localization IoU exponent weight.
        eps: Small numerical constant.
    """

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
        """Assign anchors to ground truth boxes.

        Args:
            pred_scores: [B, num_anchors, num_classes] sigmoid probabilities.
            pred_bboxes: [B, num_anchors, 4] decoded predicted boxes in xyxy format.
            anchor_points: [num_anchors, 2] center coordinates of each anchor.
            gt_labels: [B, max_num_gt, 1] ground truth class index.
            gt_bboxes: [B, max_num_gt, 4] ground truth boxes in xyxy format.
            mask_gt: [B, max_num_gt, 1] binary mask indicating valid (non-padded) GTs.

        Returns:
            target_bboxes: [B, num_anchors, 4] assigned ground truth boxes.
            target_scores: [B, num_anchors, num_classes] target soft classification scores.
            fg_mask: [B, num_anchors] boolean mask of positive foreground anchors.
        """
        bs, num_anchors, _ = pred_scores.shape
        _, max_num_gt, _ = gt_bboxes.shape

        # Compute assignment in float32 to avoid AMP Half/Float mismatches and numerical underflow
        pred_scores = pred_scores.float()
        pred_bboxes = pred_bboxes.float()
        anchor_points = anchor_points.float()
        gt_bboxes = gt_bboxes.float()

        if max_num_gt == 0 or not mask_gt.any():
            device = pred_scores.device
            return (
                torch.zeros((bs, num_anchors, 4), device=device),
                torch.zeros((bs, num_anchors, self.num_classes), device=device),
                torch.zeros((bs, num_anchors), dtype=torch.bool, device=device),
            )

        target_bboxes = torch.zeros_like(pred_bboxes)
        target_scores = torch.zeros_like(pred_scores)
        fg_mask = torch.zeros((bs, num_anchors), dtype=torch.bool, device=pred_scores.device)

        for b in range(bs):
            b_mask_gt = mask_gt[b].squeeze(-1)  # [max_num_gt]
            if not b_mask_gt.any():
                continue

            b_gt_boxes = gt_bboxes[b][b_mask_gt]     # [M, 4]
            b_gt_labels = gt_labels[b][b_mask_gt].squeeze(-1).long()  # [M]
            M = len(b_gt_boxes)

            b_pred_boxes = pred_bboxes[b]            # [N, 4]
            b_pred_scores = pred_scores[b]           # [N, C]

            # 1. Candidate selection: Anchor centers inside GT boxes
            # b_gt_boxes: [M, 4], anchor_points: [N, 2]
            lt = anchor_points.unsqueeze(0) - b_gt_boxes[:, :2].unsqueeze(1)  # [M, N, 2]
            rb = b_gt_boxes[:, 2:].unsqueeze(1) - anchor_points.unsqueeze(0)  # [M, N, 2]
            is_in_box = torch.cat([lt, rb], dim=-1).min(dim=-1).values > 0.0  # [M, N]

            # 2. Compute IoU between predicted boxes and all GT boxes: [M, N]
            ious = self._batch_iou(b_gt_boxes, b_pred_boxes)  # [M, N]

            # 3. Extract predicted scores for corresponding GT classes: [M, N]
            cls_scores = b_pred_scores[:, b_gt_labels].transpose(0, 1)  # [M, N]

            # 4. Joint Alignment Metric: metric = s^alpha * iou^beta
            align_metric = (cls_scores.clamp(min=0.0) ** self.alpha) * (ious.clamp(min=0.0) ** self.beta)
            align_metric = align_metric * is_in_box.float()

            # 5. Top-K Selection per GT
            k = min(self.topk, num_anchors)
            topk_metrics, topk_indices = torch.topk(align_metric, k=k, dim=-1, largest=True)
            topk_mask = torch.zeros_like(align_metric, dtype=torch.bool)
            topk_mask.scatter_(1, topk_indices, True)

            # Filter candidates: must be in top-k, inside box, and have positive metric
            is_pos = topk_mask & is_in_box & (align_metric > 0)

            # 6. Resolve overlaps: if an anchor is matched to multiple GTs, pick the one with max metric
            # Sum over GTs: how many GTs matched this anchor?
            num_matches = is_pos.sum(dim=0)  # [N]
            if not is_pos.any():
                continue

            # For each anchor, get the GT index with maximum alignment metric
            anchor_gt_idx = align_metric.argmax(dim=0)  # [N]
            anchor_is_fg = num_matches > 0              # [N]

            selected_gt_idx = anchor_gt_idx[anchor_is_fg]  # [num_pos]
            pos_anchor_idx = torch.where(anchor_is_fg)[0]

            # Normalize alignment metrics to form target classification soft labels
            # metric_norm = align_metric * (ious / (align_metric.amax(dim=-1, keepdim=True) + self.eps))
            max_metrics = align_metric.amax(dim=-1, keepdim=True) + self.eps
            norm_align = (align_metric / max_metrics) * ious

            assigned_boxes = b_gt_boxes[selected_gt_idx]
            assigned_labels = b_gt_labels[selected_gt_idx]

            target_bboxes[b, pos_anchor_idx] = assigned_boxes
            fg_mask[b, pos_anchor_idx] = True

            # Populate target class scores
            b_target_scores = torch.zeros_like(b_pred_scores)
            norm_scores = norm_align[selected_gt_idx, pos_anchor_idx]
            b_target_scores[pos_anchor_idx, assigned_labels] = norm_scores.clamp(0, 1.0)
            target_scores[b] = b_target_scores

        return target_bboxes, target_scores, fg_mask

    @staticmethod
    def _batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Compute pairwise IoU between boxes1 [M, 4] and boxes2 [N, 4]."""
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
        area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
        union = area1 + area2 - inter + eps
        return inter / union


# =========================================================================
# 5. Full YOLOv11 Multi-Loss Aggregator
# =========================================================================

@LOSSES.register_module(force=True)
class YOLOv11DetectionLoss(nn.Module):
    """Complete loss engine for YOLOv11 combining DIoU Loss, Focal Loss, and DFL.

    Args:
        num_classes: Number of detection categories (default 10 for VisDrone).
        reg_max: Number of discrete bins per coordinate for DFL (default 16).
        box_weight: Loss weight multiplier for DIoU bounding box regression.
        cls_weight: Loss weight multiplier for Sigmoid Focal Loss classification.
        dfl_weight: Loss weight multiplier for Distribution Focal Loss.
        focal_alpha: Focal loss alpha hyperparameter (default 0.25).
        focal_gamma: Focal loss gamma hyperparameter (default 2.0).
    """

    def __init__(
        self,
        num_classes: int = 10,
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 1.0,
        dfl_weight: float = 1.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

        self.diou_loss = DIoULoss(loss_weight=1.0, reduction="none")
        self.focal_loss = SigmoidFocalLoss(loss_weight=1.0, alpha=focal_alpha, gamma=focal_gamma, reduction="sum")
        self.dfl_loss = DistributionFocalLoss(loss_weight=1.0, reg_max=reg_max)
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)

    def forward(
        self,
        preds: Dict[str, Any],
        targets: Union[List[Dict[str, torch.Tensor]], torch.Tensor],
        decoded_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        img_size: Tuple[int, int] = (640, 640),
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            preds: Dict with 'boxes' [B, 4*reg_max, N] and 'scores' [B, nc, N].
            targets: List of target dicts [{'boxes': tensor, 'labels': tensor}, ...].
            decoded_bboxes: Decoded predicted bounding boxes [B, N, 4] in pixel xyxy format.
            anchor_points: Anchor center coordinates [N, 2].
            stride_tensor: Anchor stride per point [N, 1].
            img_size: Tuple of (height, width) of input batch images.

        Returns:
            Dict containing 'loss', 'loss_diou', 'loss_focal', 'loss_dfl'.
        """
        device = decoded_bboxes.device
        with torch.amp.autocast(device_type=device.type, enabled=False):
            pred_dist = preds["boxes"].transpose(1, 2).float()  # [B, N, 4*reg_max]
            pred_scores = preds["scores"].transpose(1, 2).float()  # [B, N, nc]
            assign_scores = preds.get("assign_scores", preds["scores"]).transpose(1, 2).float()  # [B, N, nc]
            decoded_bboxes = decoded_bboxes.float()
            anchor_points = anchor_points.float()
            stride_tensor = stride_tensor.float()
            bs, num_anchors, _ = pred_scores.shape

            # Format targets into padded tensors [B, max_gt, 4] and [B, max_gt, 1]
            gt_bboxes, gt_labels, mask_gt = self._prepare_targets(targets, img_size, device)

            # Dynamic target assignment (can use decoupled slot-guided assign_scores)
            target_bboxes, target_scores, fg_mask = self.assigner(
                pred_scores=assign_scores.sigmoid(),
                pred_bboxes=decoded_bboxes,
                anchor_points=anchor_points,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                mask_gt=mask_gt,
            )

            # Ultralytics YOLOv11 normalization: weight box/DFL by per-anchor TAL quality `target_scores`
            target_scores_sum = max(float(target_scores.sum().item()), 1.0)

            # 1. Classification: Sigmoid Focal / BCE Loss across ALL anchors
            loss_focal = self.focal_loss(
                pred_scores,
                target_scores,
                normalizer=target_scores_sum,
            ) * self.cls_weight

            # 2. Localization: TAL-weighted DIoU Loss and DFL on FOREGROUND anchors
            if fg_mask.any():
                pos_pred_bboxes = decoded_bboxes[fg_mask]
                pos_target_bboxes = target_bboxes[fg_mask]
                weight = target_scores.sum(dim=-1)[fg_mask]  # [num_pos]

                # Weighted DIoU loss (focuses gradient on high-quality center anchors)
                diou_per_anchor = self.diou_loss(
                    pos_pred_bboxes,
                    pos_target_bboxes,
                    xywh=False,
                )
                loss_diou = ((diou_per_anchor * weight).sum() / target_scores_sum) * self.box_weight

                # Weighted DFL loss (mean over 4 edges, weighted by TAL quality)
                pos_anchor_points = anchor_points.unsqueeze(0).expand(bs, -1, -1)[fg_mask]
                pos_stride = stride_tensor.unsqueeze(0).expand(bs, -1, -1)[fg_mask]

                target_ltrb = self._bbox2dist(pos_anchor_points, pos_target_bboxes) / pos_stride
                pos_pred_dist = pred_dist[fg_mask].view(-1, 4, self.reg_max)

                loss_dfl = self.dfl_loss(
                    pos_pred_dist,
                    target_ltrb,
                    normalizer=target_scores_sum,
                    weight=weight,
                ) * self.dfl_weight
            else:
                loss_diou = torch.tensor(0.0, device=device)
                loss_dfl = torch.tensor(0.0, device=device)

            total_loss = loss_focal + loss_diou + loss_dfl

        return {
            "loss": total_loss,
            "loss_diou": loss_diou,
            "loss_focal": loss_focal,
            "loss_dfl": loss_dfl,
        }

    def _prepare_targets(
        self,
        targets: Union[List[Dict[str, torch.Tensor]], torch.Tensor],
        img_size: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert variable length batch targets into padded tensors for vectorized assignment."""
        H, W = img_size
        batch_size = len(targets) if isinstance(targets, list) else targets.shape[0]

        gt_boxes_list = []
        gt_labels_list = []
        max_gt = 0

        for b in range(batch_size):
            t_dict = targets[b] if isinstance(targets, list) else {}
            boxes = t_dict.get("boxes", torch.empty((0, 4), device=device))
            labels = t_dict.get("labels", torch.empty((0,), device=device, dtype=torch.long))

            if len(boxes) > 0:
                boxes = boxes.to(device).float()
                labels = labels.to(device).long()

                # If coordinates are normalized in [0, 1], scale to pixel coordinates
                if (boxes <= 1.05).all():
                    # Check whether format is [x1, y1, w, h] or [cx, cy, w, h] or [x1, y1, x2, y2]
                    is_cxcywh = t_dict.get("_is_cxcywh", False)
                    if is_cxcywh:
                        cx, cy, w, h = boxes[:, 0] * W, boxes[:, 1] * H, boxes[:, 2] * W, boxes[:, 3] * H
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                    else:
                        # Visdrone default: [x1, y1, w, h] normalized
                        x1 = boxes[:, 0] * W
                        y1 = boxes[:, 1] * H
                        w = boxes[:, 2] * W
                        h = boxes[:, 3] * H
                        boxes_xyxy = torch.stack([x1, y1, x1 + w, y1 + h], dim=-1)
                else:
                    # Already pixel coordinates
                    is_cxcywh = t_dict.get("_is_cxcywh", False)
                    if is_cxcywh:
                        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                        boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
                    elif (boxes[:, 2] < boxes[:, 0]).any() or (boxes[:, 3] < boxes[:, 1]).any():
                        # [x1, y1, w, h] pixels
                        boxes_xyxy = torch.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], dim=-1)
                    else:
                        boxes_xyxy = boxes

                gt_boxes_list.append(boxes_xyxy)
                gt_labels_list.append(labels)
                max_gt = max(max_gt, len(boxes_xyxy))
            else:
                gt_boxes_list.append(torch.empty((0, 4), device=device))
                gt_labels_list.append(torch.empty((0,), device=device, dtype=torch.long))

        if max_gt == 0:
            return (
                torch.zeros((batch_size, 0, 4), device=device),
                torch.zeros((batch_size, 0, 1), device=device),
                torch.zeros((batch_size, 0, 1), dtype=torch.bool, device=device),
            )

        padded_boxes = torch.zeros((batch_size, max_gt, 4), device=device)
        padded_labels = torch.zeros((batch_size, max_gt, 1), device=device)
        mask_gt = torch.zeros((batch_size, max_gt, 1), dtype=torch.bool, device=device)

        for b in range(batch_size):
            num_b = len(gt_boxes_list[b])
            if num_b > 0:
                padded_boxes[b, :num_b] = gt_boxes_list[b]
                padded_labels[b, :num_b, 0] = gt_labels_list[b]
                mask_gt[b, :num_b, 0] = True

        return padded_boxes, padded_labels, mask_gt

    @staticmethod
    def _bbox2dist(anchor_points: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """Transform bbox(xyxy) to distance(ltrb)."""
        x1y1 = bboxes[:, :2]
        x2y2 = bboxes[:, 2:]
        lt = anchor_points - x1y1
        rb = x2y2 - anchor_points
        return torch.cat([lt, rb], dim=-1).clamp(min=0.0)
