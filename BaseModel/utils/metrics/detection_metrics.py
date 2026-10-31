"""Detection Evaluation Metrics: Mean Average Precision (mAP: mAP_50, mAP_75, mAP_50:95)."""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np

from ...core.registrar import METRICS
from .base_metric import BaseMetric


def box_iou_cxcywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes in [cx, cy, w, h] format."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)

    # Convert cxcywh to xyxy
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

    inter_x1 = torch.max(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = torch.max(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = torch.min(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = torch.min(b1_y2[:, None], b2_y2[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] * boxes1[:, 3]).clamp(min=0.0)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).clamp(min=0.0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / union_area.clamp(min=1e-7)


@METRICS.register_module(name="MeanAveragePrecision")
@METRICS.register_module(name="mAP")
class MeanAveragePrecision(BaseMetric):
    """Mean Average Precision (mAP) metric for object detection evaluation.

    Computes standard COCO-style metrics:
      - mAP_50: mAP at IoU threshold 0.50
      - mAP_75: mAP at IoU threshold 0.75
      - mAP_50_95: mAP averaged across IoU thresholds [0.50 : 0.95 : 0.05]

    Args:
        iou_thresholds: List or tuple of IoU thresholds (default: 0.50 to 0.95 with step 0.05).
        iou_threshold: Optional single float IoU threshold (backward compatibility).
        num_classes: Optional total number of classes (default: 80).
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        iou_threshold: Optional[float] = None,
        num_classes: Optional[int] = 80,
    ):
        if iou_thresholds is not None:
            self.iou_thresholds = sorted(iou_thresholds)
        elif iou_threshold is not None:
            self.iou_thresholds = [iou_threshold]
        else:
            # Standard 10 COCO IoU thresholds: 0.50 to 0.95 with step 0.05
            self.iou_thresholds = [round(x, 2) for x in np.linspace(0.5, 0.95, 10).tolist()]

        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset accumulated predictions and ground truths."""
        self.predictions: List[Dict[str, torch.Tensor]] = []
        self.ground_truths: List[Dict[str, torch.Tensor]] = []

    def update(self, preds: Any, targets: Any) -> None:
        """Accumulate predictions and ground truths for a batch.

        Args:
            preds: List of dicts, each with keys 'boxes', 'scores', 'labels'.
            targets: List of dicts, each with keys 'boxes', 'labels'.
        """
        if not isinstance(preds, (list, tuple)) or not isinstance(targets, (list, tuple)):
            return

        batch_size = min(len(preds), len(targets))
        for i in range(batch_size):
            p = preds[i]
            t = targets[i]

            pred_dict = {
                "boxes": p.get("boxes", torch.empty((0, 4))).detach().cpu().float(),
                "scores": p.get("scores", torch.empty((0,))).detach().cpu().float(),
                "labels": p.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long(),
            }
            gt_dict = {
                "boxes": t.get("boxes", torch.empty((0, 4))).detach().cpu().float(),
                "labels": t.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long(),
            }

            self.predictions.append(pred_dict)
            self.ground_truths.append(gt_dict)

    def _compute_ap_for_class_and_threshold(
        self,
        class_id: int,
        iou_thresh: float,
        detections: List[Tuple[int, float, torch.Tensor]],
        gt_by_img: Dict[int, torch.Tensor],
        total_gt: int,
    ) -> float:
        """Compute Average Precision for a specific class and IoU threshold."""
        if total_gt == 0 or len(detections) == 0:
            return 0.0

        # Track which ground truth boxes have been matched per image
        matched_gt: Dict[int, set] = {img_idx: set() for img_idx in gt_by_img}

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for d_idx, (img_idx, score, pred_box) in enumerate(detections):
            gt_boxes = gt_by_img.get(img_idx, torch.empty((0, 4)))
            if len(gt_boxes) == 0:
                fp[d_idx] = 1.0
                continue

            ious = box_iou_cxcywh(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, max_idx = ious.max(dim=0)
            max_iou = max_iou.item()
            max_idx = max_idx.item()

            if max_iou >= iou_thresh:
                if max_idx not in matched_gt[img_idx]:
                    tp[d_idx] = 1.0
                    matched_gt[img_idx].add(max_idx)
                else:
                    fp[d_idx] = 1.0  # Duplicate detection
            else:
                fp[d_idx] = 1.0

        # Precision - Recall Curve
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / total_gt
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)

        # Standard 101-point / continuous PR interpolation
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return float(ap)

    def compute(self) -> Dict[str, float]:
        """Compute mAP_50, mAP_75, and mAP_50_95 across all classes."""
        if len(self.predictions) == 0 or len(self.ground_truths) == 0:
            return {
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                "mAP_50_95": 0.0,
            }

        # 1. Determine classes present in dataset
        all_gt_classes = set()
        for gt in self.ground_truths:
            all_gt_classes.update(gt["labels"].tolist())

        if self.num_classes is not None:
            # Only evaluate classes that appear in ground truth
            class_list = [c for c in range(self.num_classes) if c in all_gt_classes]
        else:
            class_list = sorted(all_gt_classes)

        if not class_list:
            return {
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                "mAP_50_95": 0.0,
            }

        # 2. Pre-organize ground truths and detections per class
        # thresholds: ensure 0.50 and 0.75 are evaluated along with 0.50:0.95
        all_eval_thresholds = sorted(set(self.iou_thresholds + [0.50, 0.75]))
        coco_thresholds = [round(x, 2) for x in np.linspace(0.5, 0.95, 10).tolist()]

        # ap_table[thresh][class_id] = ap
        ap_table: Dict[float, List[float]] = {th: [] for th in all_eval_thresholds}

        for c in class_list:
            # Collect GTs for class c
            gt_by_img: Dict[int, torch.Tensor] = {}
            total_gt = 0
            for img_idx, gt in enumerate(self.ground_truths):
                mask = gt["labels"] == c
                gt_boxes = gt["boxes"][mask]
                gt_by_img[img_idx] = gt_boxes
                total_gt += len(gt_boxes)

            if total_gt == 0:
                continue

            # Collect detections for class c
            detections: List[Tuple[int, float, torch.Tensor]] = []
            for img_idx, p in enumerate(self.predictions):
                mask = p["labels"] == c
                boxes = p["boxes"][mask]
                scores = p["scores"][mask]
                for box, score in zip(boxes, scores):
                    detections.append((img_idx, score.item(), box))

            detections.sort(key=lambda x: x[1], reverse=True)

            # Evaluate class c for each threshold
            for th in all_eval_thresholds:
                ap = self._compute_ap_for_class_and_threshold(
                    class_id=c,
                    iou_thresh=th,
                    detections=detections,
                    gt_by_img=gt_by_img,
                    total_gt=total_gt,
                )
                ap_table[th].append(ap)

        # 3. Calculate mean APs across classes (in 0-1)
        map_per_thresh: Dict[float, float] = {}
        for th in all_eval_thresholds:
            aps = ap_table[th]
            map_per_thresh[th] = (float(np.mean(aps))) if len(aps) > 0 else 0.0

        map_50 = map_per_thresh.get(0.50, 0.0)
        map_75 = map_per_thresh.get(0.75, 0.0)

        # Compute mAP 50:95 as average of the 10 standard thresholds
        coco_maps = [map_per_thresh.get(th, 0.0) for th in coco_thresholds]
        map_50_95 = float(np.mean(coco_maps)) if coco_maps else 0.0

        return {
            "mAP_50": round(map_50, 4),
            "mAP_75": round(map_75, 4),
            "mAP_50_95": round(map_50_95, 4),
        }
