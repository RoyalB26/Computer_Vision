"""Standard Computer Vision Loss Functions."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.registrar import LOSSES
from .base_loss import BaseLoss


@LOSSES.register_module()
class CrossEntropyLoss(BaseLoss):
    """Cross Entropy Loss with loss_weight support."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__(loss_weight=loss_weight)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("class_weight", weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        loss = F.cross_entropy(
            preds,
            targets,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        return loss * self.loss_weight


@LOSSES.register_module()
class BCEWithLogitsLoss(BaseLoss):
    """Binary Cross Entropy with Logits and loss_weight support."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__(loss_weight=loss_weight)
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            preds,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        return loss * self.loss_weight


@LOSSES.register_module()
class SmoothL1Loss(BaseLoss):
    """Smooth L1 (Huber) Loss with loss_weight support."""

    def __init__(self, loss_weight: float = 1.0, beta: float = 1.0, reduction: str = "mean"):
        super().__init__(loss_weight=loss_weight)
        self.beta = beta
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        loss = F.smooth_l1_loss(preds, targets, beta=self.beta, reduction=self.reduction)
        return loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(BaseLoss):
    """Focal Loss for addressing class imbalance in dense classification."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__(loss_weight=loss_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        # Cross entropy loss per element
        ce_loss = F.cross_entropy(preds, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean() * self.loss_weight
        elif self.reduction == "sum":
            return focal_loss.sum() * self.loss_weight
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class DiceLoss(BaseLoss):
    """Dice Loss commonly used in Semantic Segmentation tasks."""

    def __init__(self, loss_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__(loss_weight=loss_weight)
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        # preds: (B, C, H, W) or (B, 1, H, W), targets: (B, H, W) or (B, C, H, W)
        probs = torch.softmax(preds, dim=1) if preds.shape[1] > 1 else torch.sigmoid(preds)

        if targets.ndim == 3 and preds.shape[1] > 1:
            targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()

        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()
        return loss * self.loss_weight
