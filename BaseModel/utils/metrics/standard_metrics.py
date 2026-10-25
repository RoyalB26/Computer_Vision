"""Standard Computer Vision Metrics: Accuracy, Precision, Recall, F1, and mIoU."""
from typing import Dict, List, Tuple, Union
import torch
from ...core.registrar import METRICS
from .base_metric import BaseMetric


@METRICS.register_module()
class Accuracy(BaseMetric):
    """Classification Accuracy metric supporting top-k accuracy.

    Args:
        topk: Tuple of top-k accuracies to compute, e.g. (1, 5).
    """

    def __init__(self, topk: Tuple[int, ...] = (1,)):
        self.topk = topk
        self.reset()

    def reset(self) -> None:
        self.correct_counts = {k: 0 for k in self.topk}
        self.total_samples = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(targets, (list, tuple)):
            targets = targets[0]

        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        max_k = max(self.topk)
        batch_size = targets.size(0)

        if preds.ndim == 1:
            # Binary classification probabilities or logits
            pred_indices = (preds > 0.5).long().unsqueeze(1)
        else:
            _, pred_indices = preds.topk(max_k, dim=1, largest=True, sorted=True)

        targets_expanded = targets.view(-1, 1).expand_as(pred_indices)
        correct = pred_indices.eq(targets_expanded)

        for k in self.topk:
            correct_k = correct[:, :k].reshape(-1).float().sum(0).item()
            self.correct_counts[k] += correct_k
        self.total_samples += batch_size

    def compute(self) -> Union[float, Dict[str, float]]:
        if self.total_samples == 0:
            return 0.0 if len(self.topk) == 1 else {f"acc_top{k}": 0.0 for k in self.topk}

        results = {f"acc_top{k}": (self.correct_counts[k] / self.total_samples) * 100.0 for k in self.topk}
        return results["acc_top1"] if len(self.topk) == 1 and 1 in self.topk else results


@METRICS.register_module()
class MeanIoU(BaseMetric):
    """Mean Intersection over Union (mIoU) for Semantic Segmentation.

    Args:
        num_classes: Total number of segmentation classes.
        ignore_index: Class index to ignore during IoU calculation.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        if preds.ndim == 4:
            # (B, C, H, W) -> (B, H, W)
            preds = preds.argmax(dim=1)

        mask = targets != self.ignore_index
        p = preds[mask]
        t = targets[mask]

        indices = self.num_classes * t + p
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        cm = self.confusion_matrix.float()
        intersection = torch.diag(cm)
        union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
        ious = intersection / (union + 1e-7)

        valid_mask = union > 0
        miou = ious[valid_mask].mean().item() * 100.0 if valid_mask.any() else 0.0
        return {"mIoU": miou}
