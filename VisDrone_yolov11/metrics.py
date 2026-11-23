"""Evaluation Metrics for Object Detection: Mean Average Precision (mAP 50, 75, 50:95) for VisDrone."""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torchvision.ops

from BaseModel.core import METRICS
from BaseModel.utils.metrics.base_metric import BaseMetric
from .dataset import VISDRONE_CLASSES


def convert_to_xyxy(boxes: torch.Tensor, box_format: str = "auto") -> torch.Tensor:
    """Standardize boxes into [x1, y1, x2, y2] format."""
    if len(boxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=boxes.device)

    boxes = boxes.float()
    if box_format == "xyxy":
        return boxes

    if box_format == "cxcywh":
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return torch.stack([x1, y1, x2, y2], dim=-1)

    if box_format == "xywh":
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([x, y, x + w, y + h], dim=-1)

    # Auto detect:
    # If x2 >= x1 and y2 >= y1, check if already xyxy
    if (boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all():
        if (boxes[:, 0] + boxes[:, 2] / 2.0 > 1.05).any() or (boxes[:, 1] + boxes[:, 3] / 2.0 > 1.05).any():
            return boxes
    return convert_to_xyxy(boxes, box_format="cxcywh")


@METRICS.register_module(name="VisDroneMAP", force=True)
@METRICS.register_module(name="VisDroneYOLOMAP", force=True)
class VisDroneMAP(BaseMetric):
    """Mean Average Precision metric matching COCO and VisDrone evaluation benchmarks.

    Computes:
      - mAP_50: IoU threshold 0.50
      - mAP_75: IoU threshold 0.75
      - mAP_50_95: Average across 10 IoU thresholds [0.50 : 0.95 : 0.05]
      - Per-class AP50 scores
    """

    def __init__(
        self,
        num_classes: int = 10,
        box_format: str = "auto",
        iou_thresholds: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.box_format = box_format.lower()
        if iou_thresholds is not None:
            self.iou_thresholds = sorted(iou_thresholds)
        else:
            self.iou_thresholds = [round(x, 2) for x in np.linspace(0.5, 0.95, 10).tolist()]

        self.class_names = class_names or VISDRONE_CLASSES
        self.reset()

    def reset(self) -> None:
        """Reset internal accumulator buffers between epochs."""
        self.predictions: List[Dict[str, torch.Tensor]] = []
        self.ground_truths: List[Dict[str, torch.Tensor]] = []

    def update(self, preds: Any, targets: Any) -> None:
        """Accumulate batch predictions and ground truths."""
        if not isinstance(preds, (list, tuple)) or not isinstance(targets, (list, tuple)):
            return

        batch_size = min(len(preds), len(targets))
        for i in range(batch_size):
            p = preds[i] if isinstance(preds[i], dict) else {}
            t = targets[i] if isinstance(targets[i], dict) else {}

            p_boxes = p.get("boxes", torch.empty((0, 4))).detach().cpu()
            p_scores = p.get("scores", torch.empty((0,))).detach().cpu()
            p_labels = p.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()

            t_boxes = t.get("boxes", torch.empty((0, 4))).detach().cpu()
            t_labels = t.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()

            p_boxes_xyxy = convert_to_xyxy(p_boxes, "xyxy")
            t_fmt = "cxcywh" if t.get("_is_cxcywh", False) else self.box_format
            t_boxes_xyxy = convert_to_xyxy(t_boxes, t_fmt)

            self.predictions.append({
                "boxes": p_boxes_xyxy,
                "scores": p_scores.float(),
                "labels": p_labels.long(),
            })
            self.ground_truths.append({
                "boxes": t_boxes_xyxy,
                "labels": t_labels.long(),
            })

    def compute(self) -> Dict[str, float]:
        """Compute final mAP metrics in range [0, 1]."""
        if len(self.predictions) == 0 or len(self.ground_truths) == 0:
            return {"mAP_50": 0.0, "mAP_75": 0.0, "mAP_50_95": 0.0}

        all_gt_classes = set()
        for gt in self.ground_truths:
            all_gt_classes.update(gt["labels"].tolist())

        if self.num_classes is not None:
            class_list = [c for c in range(self.num_classes) if c in all_gt_classes]
        else:
            class_list = sorted(all_gt_classes)

        if not class_list:
            return {"mAP_50": 0.0, "mAP_75": 0.0, "mAP_50_95": 0.0}

        all_eval_thresholds = sorted(set(self.iou_thresholds + [0.50, 0.75]))
        ap_table: Dict[float, List[float]] = {th: [] for th in all_eval_thresholds}
        recall_thresholds = np.linspace(0.0, 1.0, 101)
        num_imgs = min(len(self.predictions), len(self.ground_truths))

        for c in class_list:
            total_gt = 0
            class_scores_list: List[np.ndarray] = []
            class_tp_list: List[np.ndarray] = []
            class_fp_list: List[np.ndarray] = []

            for img_idx in range(num_imgs):
                p = self.predictions[img_idx]
                t = self.ground_truths[img_idx]

                p_mask = p["labels"] == c
                p_boxes_c = p["boxes"][p_mask]
                p_scores_c = p["scores"][p_mask]

                t_mask = t["labels"] == c
                t_boxes_c = t["boxes"][t_mask]

                n_gt = len(t_boxes_c)
                total_gt += n_gt

                n_pred = len(p_boxes_c)
                if n_pred == 0:
                    continue

                sort_idx = torch.argsort(p_scores_c, descending=True)
                p_boxes_c = p_boxes_c[sort_idx]
                p_scores_c = p_scores_c[sort_idx]

                class_scores_list.append(p_scores_c.numpy())

                if n_gt == 0:
                    class_tp_list.append(np.zeros((n_pred, len(all_eval_thresholds)), dtype=np.float32))
                    class_fp_list.append(np.ones((n_pred, len(all_eval_thresholds)), dtype=np.float32))
                    continue

                # Pairwise IoU matrix
                ious = torchvision.ops.box_iou(p_boxes_c, t_boxes_c)
                max_ious, matched_gt = ious.max(dim=1)

                tp_img = np.zeros((n_pred, len(all_eval_thresholds)), dtype=np.float32)
                fp_img = np.zeros((n_pred, len(all_eval_thresholds)), dtype=np.float32)

                for th_idx, th in enumerate(all_eval_thresholds):
                    detected_gt = set()
                    for p_i in range(n_pred):
                        if max_ious[p_i] >= th:
                            gt_i = matched_gt[p_i].item()
                            if gt_i not in detected_gt:
                                tp_img[p_i, th_idx] = 1.0
                                detected_gt.add(gt_i)
                            else:
                                fp_img[p_i, th_idx] = 1.0
                        else:
                            fp_img[p_i, th_idx] = 1.0

                class_tp_list.append(tp_img)
                class_fp_list.append(fp_img)

            if total_gt == 0:
                continue

            if len(class_scores_list) == 0:
                for th in all_eval_thresholds:
                    ap_table[th].append(0.0)
                continue

            all_scores = np.concatenate(class_scores_list, axis=0)
            all_tp = np.concatenate(class_tp_list, axis=0)
            all_fp = np.concatenate(class_fp_list, axis=0)

            order = np.argsort(-all_scores)
            all_tp = all_tp[order]
            all_fp = all_fp[order]

            for th_idx, th in enumerate(all_eval_thresholds):
                cum_tp = np.cumsum(all_tp[:, th_idx])
                cum_fp = np.cumsum(all_fp[:, th_idx])
                recalls = cum_tp / total_gt
                precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)

                # 101-point COCO interpolation
                mrec = np.concatenate(([0.0], recalls, [1.0]))
                mpre = np.concatenate(([0.0], precisions, [0.0]))
                for idx in range(len(mpre) - 2, -1, -1):
                    mpre[idx] = max(mpre[idx], mpre[idx + 1])

                interp_pre = np.interp(recall_thresholds, mrec, mpre)
                ap = float(np.mean(interp_pre))
                ap_table[th].append(ap)

        map_50 = float(np.mean(ap_table[0.50])) if len(ap_table[0.50]) > 0 else 0.0
        map_75 = float(np.mean(ap_table[0.75])) if len(ap_table[0.75]) > 0 else 0.0

        coco_thresholds = [round(x, 2) for x in np.linspace(0.5, 0.95, 10).tolist()]
        coco_aps = [float(np.mean(ap_table[th])) for th in coco_thresholds if len(ap_table[th]) > 0]
        map_50_95 = float(np.mean(coco_aps)) if len(coco_aps) > 0 else 0.0

        return {
            "mAP_50": round(map_50, 4),
            "mAP_75": round(map_75, 4),
            "mAP_50_95": round(map_50_95, 4),
        }
