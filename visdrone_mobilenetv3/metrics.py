"""Evaluation Metrics for Object Detection: Mean Average Precision (mAP 50, 75, 50:95).

Hỗ trợ tự động nhận diện cả 3 định dạng Box:
  1. cxcywh: [cx, cy, w, h] (normalized [0, 1] hoặc pixel)
  2. xywh:   [x1, y1, w, h]
  3. xyxy:   [x1, y1, x2, y2]
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

    if box_format == "xywh":
        x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([x1, y1, x1 + w, y1 + h], dim=-1)

    if box_format == "cxcywh":
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return torch.stack([x1, y1, x2, y2], dim=-1)

    # Auto detect: kiểm tra xem tọa độ có đang ở dạng xyxy (x2 >= x1, y2 >= y1 và vượt quá cxcywh thông thường)
    if (boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all():
        if (boxes[:, 0] + boxes[:, 2] / 2.0 > 1.05).any() or (boxes[:, 1] + boxes[:, 3] / 2.0 > 1.05).any():
            return boxes
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
            # Ground truth từ VisdroneDataset mặc định là [x1, y1, w, h] trừ khi đã chuyển sang cxcywh
            t_format = "cxcywh" if t.get("_is_cxcywh", False) else ("xywh" if self.box_format == "cxcywh" else self.box_format)
            t_boxes_xyxy = convert_to_xyxy(t_boxes, t_format)

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
        """Tính toán mAP_50, mAP_75 và mAP_50_95 chuẩn hóa trong khoảng [0, 1]."""
        # Đồng bộ hóa predictions và ground_truths từ tất cả các GPU khi chạy Multi-GPU (DDP)
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        is_main = (not is_dist) or (torch.distributed.get_rank() == 0)

        if is_dist and torch.distributed.get_world_size() > 1:
            world_size = torch.distributed.get_world_size()
            gathered_preds: List[Any] = [None for _ in range(world_size)]
            gathered_gts: List[Any] = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_preds, self.predictions)
            torch.distributed.all_gather_object(gathered_gts, self.ground_truths)
            all_predictions = [item for sublist in gathered_preds if sublist for item in sublist]
            all_ground_truths = [item for sublist in gathered_gts if sublist for item in sublist]
        else:
            all_predictions = self.predictions
            all_ground_truths = self.ground_truths

        if len(all_predictions) == 0 or len(all_ground_truths) == 0:
            return {"mAP_50": 0.0, "mAP_75": 0.0, "mAP_50_95": 0.0}

        # Xác định các class có xuất hiện trong ground truth
        all_gt_classes = set()
        for gt in all_ground_truths:
            all_gt_classes.update(gt["labels"].tolist())

        if self.num_classes is not None:
            class_list = [c for c in range(self.num_classes) if c in all_gt_classes]
        else:
            class_list = sorted(all_gt_classes)

        if not class_list:
            return {"mAP_50": 0.0, "mAP_75": 0.0, "mAP_50_95": 0.0}

        all_eval_thresholds = sorted(set(self.iou_thresholds + [0.50, 0.75]))
        coco_thresholds = [round(x, 2) for x in np.linspace(0.5, 0.95, 10).tolist()]
        num_thresh = len(all_eval_thresholds)
        recall_thresholds = np.linspace(0.0, 1.0, 101)

        ap_table: Dict[float, List[float]] = {th: [] for th in all_eval_thresholds}

        # Lưu AP chi tiết của từng class ở ngưỡng IoU 0.50 (thang [0, 1])
        per_class_ap50: Dict[int, float] = {}

        num_imgs = min(len(all_predictions), len(all_ground_truths))

        for c in class_list:
            total_gt = 0
            class_scores_list: List[np.ndarray] = []
            class_tp_list: List[np.ndarray] = []
            class_fp_list: List[np.ndarray] = []

            # Tính ma trận IoU và matching theo từng ảnh (1 lần box_iou duy nhất/ảnh)
            for img_idx in range(num_imgs):
                gt = all_ground_truths[img_idx]
                gt_mask = gt["labels"] == c
                gt_boxes = gt["boxes"][gt_mask]
                num_g = len(gt_boxes)
                total_gt += num_g

                p = all_predictions[img_idx]
                p_mask = p["labels"] == c
                if not p_mask.any():
                    continue

                p_boxes = p["boxes"][p_mask]
                p_scores = p["scores"][p_mask]

                # Sắp xếp predictions trong ảnh theo confidence giảm dần
                order = torch.argsort(p_scores, descending=True)
                p_boxes = p_boxes[order]
                p_scores_np = p_scores[order].numpy()
                num_p = len(p_boxes)

                tp_img = np.zeros((num_thresh, num_p), dtype=np.float64)
                fp_img = np.ones((num_thresh, num_p), dtype=np.float64)

                if num_g > 0:
                    iou_mat = torchvision.ops.box_iou(p_boxes, gt_boxes).numpy()  # [num_p, num_g]
                    for t_idx, thr in enumerate(all_eval_thresholds):
                        matched_gt = np.zeros(num_g, dtype=bool)
                        for d_idx in range(num_p):
                            ious_d = iou_mat[d_idx]
                            # Chỉ xét các GT chưa được match (chuẩn COCO greedy matching)
                            unmatched_ious = np.where(~matched_gt, ious_d, -1.0)
                            best_gt = int(np.argmax(unmatched_ious))
                            if unmatched_ious[best_gt] >= thr:
                                tp_img[t_idx, d_idx] = 1.0
                                fp_img[t_idx, d_idx] = 0.0
                                matched_gt[best_gt] = True

                class_scores_list.append(p_scores_np)
                class_tp_list.append(tp_img)
                class_fp_list.append(fp_img)

            if total_gt == 0:
                continue

            if len(class_scores_list) == 0:
                for th in all_eval_thresholds:
                    ap_table[th].append(0.0)
                per_class_ap50[c] = 0.0
                continue

            # Gộp toàn bộ detections của class c và sắp xếp theo confidence giảm dần
            all_scores = np.concatenate(class_scores_list, axis=0)
            all_tp = np.concatenate(class_tp_list, axis=1)  # [num_thresh, N_det]
            all_fp = np.concatenate(class_fp_list, axis=1)  # [num_thresh, N_det]

            sort_idx = np.argsort(-all_scores, kind="mergesort")
            all_tp = all_tp[:, sort_idx]
            all_fp = all_fp[:, sort_idx]

            # Vector hóa tích lũy TP, FP cho tất cả ngưỡng IoU cùng lúc
            cum_tp = np.cumsum(all_tp, axis=1)
            cum_fp = np.cumsum(all_fp, axis=1)
            recalls = cum_tp / float(total_gt)
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)

            # Tính AP bằng 101-point interpolation chuẩn COCO cho từng ngưỡng IoU
            for t_idx, th in enumerate(all_eval_thresholds):
                rec = recalls[t_idx]
                prec = precisions[t_idx]
                mprec = np.maximum.accumulate(prec[::-1])[::-1]
                idx = np.searchsorted(rec, recall_thresholds, side="left")
                interp_prec = np.zeros(101, dtype=np.float64)
                valid = idx < len(mprec)
                interp_prec[valid] = mprec[idx[valid]]
                ap = float(np.mean(interp_prec))
                ap_table[th].append(ap)
                if th == 0.50:
                    per_class_ap50[c] = round(ap, 4)

        # In bảng thống kê chi tiết khả năng nhận diện từng class ra terminal (chỉ trên main process)
        if is_main:
            print("\n" + "=" * 62)
            print(f"{'Class ID':<10} {'Class Name':<28} {'AP@50 (0-1)':<15}")
            print("-" * 62)
            for c in range(self.num_classes):
                c_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                ap_val = per_class_ap50.get(c, 0.0)
                print(f"{c:<10} {c_name:<28} {ap_val:<15.4f}")
            print("=" * 62 + "\n")

        # Tính mean AP chuẩn hóa trong khoảng [0, 1]
        map_per_thresh: Dict[float, float] = {}
        for th in all_eval_thresholds:
            aps = ap_table[th]
            map_per_thresh[th] = float(np.mean(aps)) if len(aps) > 0 else 0.0

        map_50 = map_per_thresh.get(0.50, 0.0)
        map_75 = map_per_thresh.get(0.75, 0.0)

        coco_maps = [map_per_thresh.get(th, 0.0) for th in coco_thresholds]
        map_50_95 = float(np.mean(coco_maps)) if coco_maps else 0.0

        results = {
            "mAP_50": round(map_50, 4),
            "mAP_75": round(map_75, 4),
            "mAP_50_95": round(map_50_95, 4),
        }

        # Lưu per-class AP50 vào results dictionary (thang [0, 1])
        for c in range(self.num_classes):
            c_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            clean_name = c_name.replace(" ", "_")
            results[f"AP50_{clean_name}"] = per_class_ap50.get(c, 0.0)

        return results
