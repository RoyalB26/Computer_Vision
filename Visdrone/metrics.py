"""Evaluation Metrics for Object Detection: Mean Average Precision (mAP 50, 75, 50:95).

Hỗ trợ tự động nhận diện cả 2 định dạng Box:
  1. cxcywh: [cx, cy, w, h] (normalized [0, 1] hoặc pixel)
  2. xyxy:   [x1, y1, x2, y2]
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torchvision.ops

from BaseModel.core import METRICS
from BaseModel.utils.metrics.base_metric import BaseMetric


def convert_to_xyxy(boxes: torch.Tensor, box_format: str = "auto") -> torch.Tensor:
    """Chuyển đổi boxes sang định dạng xyxy [x1, y1, x2, y2]."""
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

    # Auto detect: Nếu tọa độ x2 < x1 hoặc w/h có vẻ là size
    # Kiểm tra phần lớn xem x1 <= x2 và y1 <= y2 không
    # Thường trong format cxcywh, nếu coi là xyxy thì [2] là w, [3] là h. Nếu [0]+[2]/2 thì hợp lý.
    # Trong VisDrone/YOLO, cxcywh: box[:, 0] là cx, box[:, 2] là w.
    return convert_to_xyxy(boxes, box_format="cxcywh")


@METRICS.register_module(name="VisDroneMAP")
@METRICS.register_module(name="DetectionMAP")
class VisDroneMAP(BaseMetric):
    """Mean Average Precision metric chuẩn COCO cho mô hình phát hiện vật thể (VisDrone).
    
    Tính toán:
      - mAP_50: IoU threshold 0.50
      - mAP_75: IoU threshold 0.75
      - mAP_50_95: Trung bình trên 10 ngưỡng IoU [0.50 : 0.95 : 0.05]

    Args:
        num_classes: Tổng số lớp của dataset (mặc định 10 cho VisDrone).
        box_format: Định dạng box đầu vào: 'cxcywh' (mặc định) hoặc 'xyxy'.
        iou_thresholds: Danh sách ngưỡng IoU (mặc định 0.50 đến 0.95 step 0.05).
    """

    def __init__(
        self,
        num_classes: int = 10,
        box_format: str = "cxcywh",
        iou_thresholds: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.box_format = box_format.lower()
        if iou_thresholds is not None:
            self.iou_thresholds = sorted(iou_thresholds)
        else:
            self.iou_thresholds = [round(x, 2) for x in np.linspace(0.5, 0.95, 10).tolist()]

        # Tên chuẩn 10 classes của VisDrone
        self.class_names = class_names or [
            "pedestrian",
            "people",
            "bicycle",
            "car",
            "van",
            "truck",
            "tricycle",
            "awning-tricycle",
            "bus",
            "motorcycle",
        ]

        self.reset()

    def reset(self) -> None:
        """Xóa bộ đệm tích lũy giữa các epoch validation."""
        self.predictions: List[Dict[str, torch.Tensor]] = []
        self.ground_truths: List[Dict[str, torch.Tensor]] = []

    def update(self, preds: Any, targets: Any) -> None:
        """Tích lũy kết quả dự đoán và nhãn thực tế cho mỗi batch."""
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

            # Chuẩn hóa về định dạng xyxy
            p_boxes_xyxy = convert_to_xyxy(p_boxes, self.box_format)
            t_boxes_xyxy = convert_to_xyxy(t_boxes, self.box_format)

            self.predictions.append({
                "boxes": p_boxes_xyxy,
                "scores": p_scores.float(),
                "labels": p_labels.long(),
            })
            self.ground_truths.append({
                "boxes": t_boxes_xyxy,
                "labels": t_labels.long(),
            })

    def _compute_ap_for_class_and_threshold(
        self,
        class_id: int,
        iou_thresh: float,
        detections: List[Tuple[int, float, torch.Tensor]],
        gt_by_img: Dict[int, torch.Tensor],
        total_gt: int,
    ) -> float:
        """Tính Average Precision (AP) theo chuẩn COCO 101 điểm cho 1 class và 1 ngưỡng IoU."""
        if total_gt == 0 or len(detections) == 0:
            return 0.0

        matched_gt: Dict[int, set] = {img_idx: set() for img_idx in gt_by_img}
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for d_idx, (img_idx, score, pred_box) in enumerate(detections):
            gt_boxes = gt_by_img.get(img_idx, torch.empty((0, 4)))
            if len(gt_boxes) == 0:
                fp[d_idx] = 1.0
                continue

            # Tính IoU sử dụng torchvision chuẩn xác
            ious = torchvision.ops.box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, max_idx = ious.max(dim=0)
            max_iou = max_iou.item()
            max_idx = max_idx.item()

            if max_iou >= iou_thresh:
                if max_idx not in matched_gt[img_idx]:
                    tp[d_idx] = 1.0
                    matched_gt[img_idx].add(max_idx)
                else:
                    fp[d_idx] = 1.0  # Duplicate detection (đã có box khác match trước đó)
            else:
                fp[d_idx] = 1.0

        # Tính Precision - Recall curve
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / total_gt
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)

        # Chuẩn COCO 101-point interpolation:
        recall_thresholds = np.linspace(0.0, 1.0, 101)
        interpolated_precisions = np.zeros(101)
        for i, r_thresh in enumerate(recall_thresholds):
            prec_after_r = precisions[recalls >= r_thresh]
            if len(prec_after_r) > 0:
                interpolated_precisions[i] = np.max(prec_after_r)
            else:
                interpolated_precisions[i] = 0.0

        ap = float(np.mean(interpolated_precisions))
        return ap

    def compute(self) -> Dict[str, float]:
        """Tính toán mAP_50, mAP_75 và mAP_50_95."""
        if len(self.predictions) == 0 or len(self.ground_truths) == 0:
            return {"mAP_50": 0.0, "mAP_75": 0.0, "mAP_50_95": 0.0}

        # Xác định các class có xuất hiện trong ground truth
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
        coco_thresholds = [round(x, 2) for x in np.linspace(0.5, 0.95, 10).tolist()]

        ap_table: Dict[float, List[float]] = {th: [] for th in all_eval_thresholds}

        # Lưu AP chi tiết của từng class ở ngưỡng IoU 0.50
        per_class_ap50: Dict[int, float] = {}

        for c in class_list:
            # Gom Ground Truths của class c
            gt_by_img: Dict[int, torch.Tensor] = {}
            total_gt = 0
            for img_idx, gt in enumerate(self.ground_truths):
                mask = gt["labels"] == c
                boxes = gt["boxes"][mask]
                gt_by_img[img_idx] = boxes
                total_gt += len(boxes)

            if total_gt == 0:
                continue

            # Gom Detections của class c
            detections: List[Tuple[int, float, torch.Tensor]] = []
            for img_idx, p in enumerate(self.predictions):
                mask = p["labels"] == c
                boxes = p["boxes"][mask]
                scores = p["scores"][mask]
                for box, score in zip(boxes, scores):
                    detections.append((img_idx, score.item(), box))

            # Sắp xếp theo confidence giảm dần
            detections.sort(key=lambda x: x[1], reverse=True)

            # Tính AP cho từng ngưỡng IoU
            for th in all_eval_thresholds:
                ap = self._compute_ap_for_class_and_threshold(
                    class_id=c,
                    iou_thresh=th,
                    detections=detections,
                    gt_by_img=gt_by_img,
                    total_gt=total_gt,
                )
                ap_table[th].append(ap)
                if th == 0.50:
                    per_class_ap50[c] = round(ap * 100.0, 2)

        # In bảng thống kê chi tiết khả năng nhận diện từng class ra terminal
        print("\n" + "=" * 62)
        print(f"{'Class ID':<10} {'Class Name':<28} {'AP@50 (%)':<15}")
        print("-" * 62)
        for c in range(self.num_classes):
            c_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            ap_val = per_class_ap50.get(c, 0.0)
            print(f"{c:<10} {c_name:<28} {ap_val:<15.2f}")
        print("=" * 62 + "\n")

        # Tính mean AP theo phần trăm (0 - 100%)
        map_per_thresh: Dict[float, float] = {}
        for th in all_eval_thresholds:
            aps = ap_table[th]
            map_per_thresh[th] = (float(np.mean(aps))) if len(aps) > 0 else 0.0

        map_50 = map_per_thresh.get(0.50, 0.0)
        map_75 = map_per_thresh.get(0.75, 0.0)

        coco_maps = [map_per_thresh.get(th, 0.0) for th in coco_thresholds]
        map_50_95 = float(np.mean(coco_maps)) if coco_maps else 0.0

        results = {
            "mAP_50": round(map_50, 4),
            "mAP_75": round(map_75, 4),
            "mAP_50_95": round(map_50_95, 4),
        }

        # Lưu per-class AP50 vào results dictionary (ví dụ: 'AP50_car', 'AP50_pedestrian'...)
        for c in range(self.num_classes):
            c_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            # Chuẩn hóa tên bỏ khoảng trắng
            clean_name = c_name.replace(" ", "_")
            results[f"AP50_{clean_name}"] = per_class_ap50.get(c, 0.0)

        return results
