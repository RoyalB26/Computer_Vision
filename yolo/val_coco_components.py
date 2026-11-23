#!/usr/bin/env python3
"""Component-by-Component Validation & Feature Inspection of Pretrained YOLOv11 on COCO.

Inspired by:
  - Visdrone/benchmark_blocks.py (multi-scale P3/P4/P5 feature quality: Energy Ratio,
    Full & PCA-3D Fisher Ratio, Inter-Class Cosine Separation, FG Mass %, Visual Grids)
  - Visdrone/ablation_neck_relu.py (activation sign/zero distribution, Neck vs Backbone comparison)
  - Visdrone/aggregate_benchmark_runs.py & aggregate_ablation_runs.py (detailed CSVs,
    multi-chart PNG suite, and interactive HTML dashboard)

Dissects YOLOv11 into 4 inspectable levels:
  1. Backbone Stages  (stage0..stage10, including feat4 //8, feat6 //16, feat10 //32)
  2. PANet Neck       (x13 //16, p3 //8, p4 //16, p5 //32) & Backbone->Neck delta
  3. Detection Head   (cv2 box DFL branch, cv3 cls branch per scale P3/P4/P5, FG mass & scale share)
  4. End-to-End Eval  (NMS, Precision/Recall/F1, mAP@50, mAP@50:95, mAP_S/M/L, per-class AP)

Example:
  python yolo/val_coco_components.py --weights yolo11n.pt --data-dir D:/apps/Datasets/COCO/dataset --max-images 200
  python yolo/val_coco_components.py --weights yolo11n.pt --data-dir D:/apps/Datasets/COCO/dataset
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, nms

from yolo.yolov11 import YOLOv11


COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


# VisDrone (0..9) -> COCO (0..79) mapping for zero-shot detection mAP on shared classes:
# 0:pedestrian->0(person), 1:people->0(person), 2:bicycle->1(bicycle),
# 3:car->2(car), 5:truck->7(truck), 8:bus->5(bus), 9:motorcycle->3(motorcycle)
VISDRONE_TO_COCO_MAP: Dict[int, int] = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    5: 7,
    8: 5,
    9: 3,
}


# =============================================================================
# 1. DATASET & LETTERBOX PREPROCESSING
# =============================================================================

def letterbox_image(
    img_rgb: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image while preserving aspect ratio (standard YOLO letterbox)."""
    h0, w0 = img_rgb.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0

    if (w0, h0) != new_unpad:
        img_resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_rgb

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    if img_padded.shape[0] != new_shape or img_padded.shape[1] != new_shape:
        img_padded = cv2.resize(img_padded, (new_shape, new_shape), interpolation=cv2.INTER_LINEAR)
    return img_padded, r, (dw, dh)


def load_yolo_labels(label_path: Path) -> np.ndarray:
    """Load YOLO txt labels -> (N, 5) float32 [cls, cx, cy, w, h] normalized in [0, 1]."""
    if not label_path.is_file():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                if 0 <= cls_id < 80 and w > 0 and h > 0:
                    rows.append([cls_id, cx, cy, w, h])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def transform_boxes_letterbox(
    boxes_norm: np.ndarray,
    orig_hw: Tuple[int, int],
    imgsz: int,
    ratio: float,
    pad: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert original normalized [cls, cx, cy, w, h] to letterboxed pixel xyxy and norm cxcywh."""
    if len(boxes_norm) == 0:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)

    h0, w0 = orig_hw
    dw, dh = pad
    cls_ids = boxes_norm[:, 0:1]
    cx = boxes_norm[:, 1] * w0 * ratio + dw
    cy = boxes_norm[:, 2] * h0 * ratio + dh
    bw = boxes_norm[:, 3] * w0 * ratio
    bh = boxes_norm[:, 4] * h0 * ratio

    x1 = np.clip(cx - bw / 2.0, 0.0, float(imgsz))
    y1 = np.clip(cy - bh / 2.0, 0.0, float(imgsz))
    x2 = np.clip(cx + bw / 2.0, 0.0, float(imgsz))
    y2 = np.clip(cy + bh / 2.0, 0.0, float(imgsz))

    boxes_xyxy = np.concatenate([cls_ids, x1[:, None], y1[:, None], x2[:, None], y2[:, None]], axis=1)
    boxes_lb_norm = np.concatenate(
        [cls_ids, (cx / imgsz)[:, None], (cy / imgsz)[:, None], (bw / imgsz)[:, None], (bh / imgsz)[:, None]],
        axis=1,
    )
    return boxes_xyxy.astype(np.float32), boxes_lb_norm.astype(np.float32)


def collect_coco_samples(data_dir: str, max_images: int = 0) -> List[Tuple[Path, Path]]:
    """Collect (image_path, label_path) pairs from COCO dataset directory."""
    root = Path(data_dir)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if max_images > 0:
        img_files = img_files[:max_images]

    pairs = []
    for img_p in img_files:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        pairs.append((img_p, lbl_p))
    return pairs


# =============================================================================
# 2. COMPONENT HOOKS & DISSECTION FOR YOLOv11
# =============================================================================

COMPONENT_ORDER: List[str] = [
    # Backbone stages
    "backbone.stage0",
    "backbone.stage1",
    "backbone.stage2",
    "backbone.stage3",
    "backbone.stage4 (feat4 //8)",
    "backbone.stage5",
    "backbone.stage6 (feat6 //16)",
    "backbone.stage7",
    "backbone.stage8",
    "backbone.stage9 (SPPF)",
    "backbone.stage10 (feat10 //32)",
    # Neck stages
    "neck.c3k2_13 (TD //16)",
    "neck.c3k2_16 (P3 //8)",
    "neck.c3k2_19 (P4 //16)",
    "neck.c3k2_22 (P5 //32)",
    # Head branches
    "head.cv2_p3 (Box //8)",
    "head.cv2_p4 (Box //16)",
    "head.cv2_p5 (Box //32)",
    "head.cv3_p3 (Cls //8)",
    "head.cv3_p4 (Cls //16)",
    "head.cv3_p5 (Cls //32)",
]


class YOLOv11ComponentProbe:
    """Attaches forward hooks to dissect every stage of Backbone, Neck, and Head."""

    def __init__(self, model: YOLOv11):
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _make_hook(self, name: str):
        def hook(_mod, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.features[name] = out
        return hook

    def _register_hooks(self):
        b = self.model.backbone
        n = self.model.neck
        h = self.model.head

        mapping = [
            ("backbone.stage0", b.stage0),
            ("backbone.stage1", b.stage1),
            ("backbone.stage2", b.stage2),
            ("backbone.stage3", b.stage3),
            ("backbone.stage4 (feat4 //8)", b.stage4),
            ("backbone.stage5", b.stage5),
            ("backbone.stage6 (feat6 //16)", b.stage6),
            ("backbone.stage7", b.stage7),
            ("backbone.stage8", b.stage8),
            ("backbone.stage9 (SPPF)", b.stage9),
            ("backbone.stage10 (feat10 //32)", b.stage10),
            ("neck.c3k2_13 (TD //16)", n.c3k2_13),
            ("neck.c3k2_16 (P3 //8)", n.c3k2_16),
            ("neck.c3k2_19 (P4 //16)", n.c3k2_19),
            ("neck.c3k2_22 (P5 //32)", n.c3k2_22),
            ("head.cv2_p3 (Box //8)", h.cv2[0]),
            ("head.cv2_p4 (Box //16)", h.cv2[1]),
            ("head.cv2_p5 (Box //32)", h.cv2[2]),
            ("head.cv3_p3 (Cls //8)", h.cv3[0]),
            ("head.cv3_p4 (Cls //16)", h.cv3[1]),
            ("head.cv3_p5 (Cls //32)", h.cv3[2]),
        ]
        for name, mod in mapping:
            self.handles.append(mod.register_forward_hook(self._make_hook(name)))

    def clear(self):
        self.features.clear()

    def remove(self):
        for hd in self.handles:
            hd.remove()
        self.handles.clear()


# =============================================================================
# 3. GPU FEATURE QUALITY METRICS (INSPIRED BY VISDRONE BENCHMARK)
# =============================================================================

def _compute_pca_rgb_gpu(fs: torch.Tensor) -> torch.Tensor:
    """Fast GPU PCA-3D projection for feature map (C, H, W) -> (H*W, 3) in [0, 1]."""
    c, h, w = fs.shape
    flat = fs.reshape(c, -1).transpose(0, 1).float()  # (N, C)
    flat_c = flat - flat.mean(dim=0, keepdim=True)
    if c >= 3:
        cov = (flat_c.transpose(0, 1) @ flat_c) / max(1, flat_c.shape[0] - 1)
        _, eigvecs = torch.linalg.eigh(cov)
        V = eigvecs[:, -3:].flip(1)
        proj = flat_c @ V
    else:
        proj = F.pad(flat_c, (0, 3 - c))
    out = torch.empty_like(proj)
    for ch in range(3):
        col = proj[:, ch]
        lo = torch.quantile(col, 0.02)
        hi = torch.quantile(col, 0.98)
        if hi > lo:
            out[:, ch] = torch.clamp((col - lo) / (hi - lo), 0.0, 1.0)
        else:
            out[:, ch] = 0.5
    return out


def build_masks_at_res_gpu(
    boxes_lb_norm: np.ndarray,
    H: int,
    W: int,
    imgsz: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (all_box_m, small_m, med_m, large_m, cls_m) boolean masks on GPU.

    COCO size thresholds at 640x640:
      small: area < 32^2 = 1024 px^2 (norm < 1024 / imgsz^2 = 0.0025)
      medium: 32^2 <= area < 96^2 = 9216 px^2
      large: area >= 96^2
    """
    all_m = torch.zeros((H, W), dtype=torch.bool, device=device)
    small_m = torch.zeros((H, W), dtype=torch.bool, device=device)
    med_m = torch.zeros((H, W), dtype=torch.bool, device=device)
    large_m = torch.zeros((H, W), dtype=torch.bool, device=device)
    cls_m = torch.zeros((80, H, W), dtype=torch.bool, device=device)

    small_thr = (32.0 / imgsz) ** 2
    large_thr = (96.0 / imgsz) ** 2

    for row in boxes_lb_norm:
        cls_id = int(row[0])
        cx, cy, bw, bh = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        area = bw * bh
        x1 = int(np.clip(np.trunc((cx - bw / 2.0) * W), 0, W - 1))
        y1 = int(np.clip(np.trunc((cy - bh / 2.0) * H), 0, H - 1))
        x2 = max(x1 + 1, int(np.clip(np.ceil((cx + bw / 2.0) * W), 0, W)))
        y2 = max(y1 + 1, int(np.clip(np.ceil((cy + bh / 2.0) * H), 0, H)))

        all_m[y1:y2, x1:x2] = True
        cls_m[cls_id, y1:y2, x1:x2] = True
        if area < small_thr:
            small_m[y1:y2, x1:x2] = True
        elif area < large_thr:
            med_m[y1:y2, x1:x2] = True
        else:
            large_m[y1:y2, x1:x2] = True

    return (
        all_m.reshape(-1),
        small_m.reshape(-1),
        med_m.reshape(-1),
        large_m.reshape(-1),
        cls_m.reshape(80, -1),
    )


def measure_feature_metrics(
    feat: torch.Tensor,
    boxes_lb_norm: np.ndarray,
    imgsz: int,
    is_cls_logits: bool = False,
) -> Dict[str, float]:
    """Measure energy contrast, Fisher ratio, class cos sep, and activation stats for (1, C, H, W)."""
    fs = feat[0].float()
    C_dim, H, W = fs.shape
    device = fs.device
    fs_flat = fs.reshape(C_dim, -1)  # (C, N)

    # Activation health stats (inspired by ablation_neck_relu.py)
    act_mean = float(fs_flat.mean().item())
    act_std = float(fs_flat.std().item())
    neg_frac = float((fs_flat < 0).float().mean().item())
    zero_frac = float((fs_flat.abs() < 1e-4).float().mean().item())

    if is_cls_logits:
        # For head.cv3 classification logits: use max sigmoid probability per spatial cell
        probs = fs_flat.sigmoid().max(dim=0).values  # (N,)
        energy_flat = probs
    else:
        energy_flat = torch.linalg.vector_norm(fs_flat, dim=0)  # (N,)

    all_m, small_m, med_m, large_m, cls_m = build_masks_at_res_gpu(
        boxes_lb_norm, H, W, imgsz, device
    )
    bg_m = ~all_m
    has_fg = bool(all_m.any().item())
    has_bg = bool(bg_m.any().item())

    res: Dict[str, float] = {
        "act_mean": act_mean,
        "act_std": act_std,
        "neg_frac": neg_frac,
        "zero_frac": zero_frac,
        "l2_mean": float(energy_flat.mean().item()),
    }

    if has_fg and has_bg:
        bg_e = float(energy_flat[bg_m].mean().item()) + 1e-6
        res["all_ratio"] = float(energy_flat[all_m].mean().item()) / bg_e
        if bool(small_m.any().item()):
            res["small_ratio"] = float(energy_flat[small_m].mean().item()) / bg_e
        if bool(med_m.any().item()):
            res["medium_ratio"] = float(energy_flat[med_m].mean().item()) / bg_e
        if bool(large_m.any().item()):
            res["large_ratio"] = float(energy_flat[large_m].mean().item()) / bg_e

        res["fg_mass_pct"] = (
            float(energy_flat[all_m].sum().item()) / (float(energy_flat.sum().item()) + 1e-8) * 100.0
        )

        # Full-dim Fisher
        fg_full = fs_flat[:, all_m].transpose(0, 1)
        bg_full = fs_flat[:, bg_m].transpose(0, 1)
        mu_fg = fg_full.mean(dim=0)
        mu_bg = bg_full.mean(dim=0)
        var_fg = fg_full.var(dim=0, correction=0).sum()
        var_bg = bg_full.var(dim=0, correction=0).sum()
        res["full_fisher"] = float((((mu_fg - mu_bg) ** 2).sum() / (var_fg + var_bg + 1e-6)).item())

        # PCA-3D Fisher
        pca_flat = _compute_pca_rgb_gpu(fs)
        fg_p = pca_flat[all_m]
        bg_p = pca_flat[bg_m]
        mu_fg_p = fg_p.mean(dim=0)
        mu_bg_p = bg_p.mean(dim=0)
        var_fg_p = fg_p.var(dim=0, correction=0).sum()
        var_bg_p = bg_p.var(dim=0, correction=0).sum()
        res["pca_fisher"] = float((((mu_fg_p - mu_bg_p) ** 2).sum() / (var_fg_p + var_bg_p + 1e-6)).item())

    # Per-image Inter-class Cosine Separation
    fs_norm = F.normalize(fs_flat, dim=0)
    cnt_c = cls_m.sum(dim=1)  # (80,)
    valid_c = cnt_c >= 2
    k_valid = int(valid_c.sum().item())
    if k_valid >= 2:
        proto_sums = cls_m[valid_c].float() @ fs_norm.transpose(0, 1)
        P_gpu = F.normalize(proto_sums, p=2, dim=1)
        sim_gpu = P_gpu @ P_gpu.transpose(0, 1)
        off_diag = (sim_gpu.sum() - k_valid) / (k_valid * (k_valid - 1))
        res["class_cos_sep"] = float((1.0 - off_diag).item())

    return res


# =============================================================================
# 4. DETECTION DECODING, NMS & COCO mAP CALCULATION
# =============================================================================

def decode_and_nms(
    pred_tensor: torch.Tensor,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    max_det: int = 300,
) -> np.ndarray:
    """Decode Detect output (1, 4+80, N_anchors) -> (M, 6) [x1, y1, x2, y2, conf, cls]."""
    # pred_tensor: (1, 84, 8400) -> transpose to (8400, 84)
    p = pred_tensor[0].transpose(0, 1)  # (N, 4 + nc)
    boxes_xywh = p[:, :4]
    scores = p[:, 4:]  # (N, nc)

    max_scores, cls_ids = scores.max(dim=1)
    keep = max_scores > conf_thres
    if not keep.any():
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xywh = boxes_xywh[keep]
    max_scores = max_scores[keep]
    cls_ids = cls_ids[keep]

    # xywh -> xyxy
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    boxes_xyxy = torch.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=1)

    # Class-aware batched NMS using coordinate offset
    max_wh = 7680.0
    offsets = cls_ids.float().unsqueeze(1) * max_wh
    keep_idx = nms(boxes_xyxy + offsets, max_scores, iou_thres)[:max_det]

    det = torch.cat(
        [boxes_xyxy[keep_idx], max_scores[keep_idx, None], cls_ids[keep_idx, None].float()],
        dim=1,
    )
    return det.cpu().numpy()


def compute_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapezoid(np.interp(x, mrec, mpre), x) if hasattr(np, "trapezoid") else np.trapz(np.interp(x, mrec, mpre), x)
    return float(ap)


def compute_coco_map(
    all_preds: List[np.ndarray],
    all_gts: List[np.ndarray],
    iou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
    area_range: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Compute COCO mAP@50, mAP@50:95, P, R, F1, and per-class AP."""
    nc = 80
    n_iou = len(iou_thresholds)

    # Accumulate per-class stats: list of (correct[N, n_iou], conf[N]) and gt_count
    cls_preds: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {c: [] for c in range(nc)}
    cls_gt_cnt = np.zeros(nc, dtype=np.int64)

    for preds, gts in zip(all_preds, all_gts):
        # gts: (G, 5) [cls, x1, y1, x2, y2]
        if area_range is not None:
            lo_a, hi_a = area_range
            if len(gts) > 0:
                gt_areas = (gts[:, 3] - gts[:, 1]) * (gts[:, 4] - gts[:, 2])
                gts = gts[(gt_areas >= lo_a) & (gt_areas < hi_a)]
            if len(preds) > 0:
                pr_areas = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
                preds = preds[(pr_areas >= lo_a) & (pr_areas < hi_a)]

        for g in gts:
            cls_gt_cnt[int(g[0])] += 1

        if len(preds) == 0:
            continue

        # Sort predictions by confidence descending
        order = np.argsort(-preds[:, 4])
        preds = preds[order]

        for c in range(nc):
            p_c = preds[preds[:, 5] == c]
            if len(p_c) == 0:
                continue
            g_c = gts[gts[:, 0] == c] if len(gts) > 0 else np.zeros((0, 5), dtype=np.float32)
            correct = np.zeros((len(p_c), n_iou), dtype=bool)

            if len(g_c) > 0:
                ious = box_iou(
                    torch.from_numpy(p_c[:, :4]),
                    torch.from_numpy(g_c[:, 1:5]),
                ).numpy()  # (Np, Ng)
                for iou_i, thr in enumerate(iou_thresholds):
                    used_gt = set()
                    for pi in range(len(p_c)):
                        best_gi = int(np.argmax(ious[pi]))
                        if ious[pi, best_gi] >= thr and best_gi not in used_gt:
                            correct[pi, iou_i] = True
                            used_gt.add(best_gi)

            cls_preds[c].append((correct, p_c[:, 4]))

    per_class_rows: List[Dict[str, float]] = []
    ap50_list, ap5095_list = [], []
    p25_list, r25_list = [], []

    for c in range(nc):
        n_gt = int(cls_gt_cnt[c])
        if n_gt == 0:
            continue
        if not cls_preds[c]:
            ap50, ap5095, p_c25, r_c25 = 0.0, 0.0, 0.0, 0.0
        else:
            corr_all = np.concatenate([x[0] for x in cls_preds[c]], axis=0)
            conf_all = np.concatenate([x[1] for x in cls_preds[c]], axis=0)
            idx = np.argsort(-conf_all)
            corr_all = corr_all[idx]
            conf_all = conf_all[idx]

            aps = []
            for iou_i in range(n_iou):
                tp = np.cumsum(corr_all[:, iou_i])
                fp = np.cumsum(~corr_all[:, iou_i])
                rec = tp / (n_gt + 1e-16)
                prec = tp / (tp + fp + 1e-16)
                aps.append(compute_ap_101(rec, prec))
            ap50 = aps[0]
            ap5095 = float(np.mean(aps))

            # Precision & Recall at conf >= 0.25, IoU=0.50
            m25 = conf_all >= 0.25
            tp25 = float(corr_all[m25, 0].sum())
            fp25 = float((~corr_all[m25, 0]).sum())
            p_c25 = tp25 / (tp25 + fp25 + 1e-16) if (tp25 + fp25) > 0 else 0.0
            r_c25 = tp25 / (n_gt + 1e-16)

        ap50_list.append(ap50)
        ap5095_list.append(ap5095)
        p25_list.append(p_c25)
        r25_list.append(r_c25)
        per_class_rows.append({
            "class_id": c,
            "class_name": COCO_CLASSES[c],
            "gt_count": n_gt,
            "ap50": ap50,
            "ap50_95": ap5095,
            "precision_c25": p_c25,
            "recall_c25": r_c25,
        })

    mean_p = float(np.mean(p25_list)) if p25_list else 0.0
    mean_r = float(np.mean(r25_list)) if r25_list else 0.0
    f1 = 2 * mean_p * mean_r / (mean_p + mean_r + 1e-16)
    summary = {
        "mAP50": float(np.mean(ap50_list)) if ap50_list else 0.0,
        "mAP50_95": float(np.mean(ap5095_list)) if ap5095_list else 0.0,
        "precision_c25": mean_p,
        "recall_c25": mean_r,
        "f1_c25": f1,
        "total_gt": int(cls_gt_cnt.sum()),
        "active_classes": len(per_class_rows),
    }
    return summary, per_class_rows


# =============================================================================
# 5. MAIN EVALUATION LOOP (COMPONENT-BY-COMPONENT)
# =============================================================================

@torch.no_grad()
def run_component_validation(
    weights_path: str,
    data_dir: str,
    out_dir: str,
    imgsz: int = 640,
    max_images: int = 0,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    vis_conf: float = 0.25,
    device_str: str = "auto",
    dataset_type: str = "coco",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 80)
    print(f"YOLOv11 COMPONENT-BY-COMPONENT VALIDATION ON {dataset_type.upper()}")
    print(f"  Weights    : {weights_path}")
    print(f"  Dataset    : {data_dir} (type={dataset_type})")
    print(f"  Device     : {device} | ImgSize: {imgsz}")
    print("=" * 80)

    model = YOLOv11(nc=80, weights=weights_path).to(device).eval()
    probe = YOLOv11ComponentProbe(model)

    # Parameter breakdown per major module
    p_backbone = sum(p.numel() for p in model.backbone.parameters()) / 1e6
    p_neck = sum(p.numel() for p in model.neck.parameters()) / 1e6
    p_head = sum(p.numel() for p in model.head.parameters()) / 1e6
    p_total = p_backbone + p_neck + p_head
    print(f"[Model Params] Total: {p_total:.3f}M | Backbone: {p_backbone:.3f}M | Neck: {p_neck:.3f}M | Head: {p_head:.3f}M")

    # Parameter count per component
    comp_params: Dict[str, int] = {
        "backbone.stage0": sum(p.numel() for p in model.backbone.stage0.parameters()),
        "backbone.stage1": sum(p.numel() for p in model.backbone.stage1.parameters()),
        "backbone.stage2": sum(p.numel() for p in model.backbone.stage2.parameters()),
        "backbone.stage3": sum(p.numel() for p in model.backbone.stage3.parameters()),
        "backbone.stage4 (feat4 //8)": sum(p.numel() for p in model.backbone.stage4.parameters()),
        "backbone.stage5": sum(p.numel() for p in model.backbone.stage5.parameters()),
        "backbone.stage6 (feat6 //16)": sum(p.numel() for p in model.backbone.stage6.parameters()),
        "backbone.stage7": sum(p.numel() for p in model.backbone.stage7.parameters()),
        "backbone.stage8": sum(p.numel() for p in model.backbone.stage8.parameters()),
        "backbone.stage9 (SPPF)": sum(p.numel() for p in model.backbone.stage9.parameters()),
        "backbone.stage10 (feat10 //32)": sum(p.numel() for p in model.backbone.stage10.parameters()),
        "neck.c3k2_13 (TD //16)": sum(p.numel() for p in model.neck.c3k2_13.parameters()),
        "neck.c3k2_16 (P3 //8)": sum(p.numel() for p in model.neck.c3k2_16.parameters()),
        "neck.c3k2_19 (P4 //16)": sum(p.numel() for p in model.neck.c3k2_19.parameters()),
        "neck.c3k2_22 (P5 //32)": sum(p.numel() for p in model.neck.c3k2_22.parameters()),
        "head.cv2_p3 (Box //8)": sum(p.numel() for p in model.head.cv2[0].parameters()),
        "head.cv2_p4 (Box //16)": sum(p.numel() for p in model.head.cv2[1].parameters()),
        "head.cv2_p5 (Box //32)": sum(p.numel() for p in model.head.cv2[2].parameters()),
        "head.cv3_p3 (Cls //8)": sum(p.numel() for p in model.head.cv3[0].parameters()),
        "head.cv3_p4 (Cls //16)": sum(p.numel() for p in model.head.cv3[1].parameters()),
        "head.cv3_p5 (Cls //32)": sum(p.numel() for p in model.head.cv3[2].parameters()),
    }

    samples = collect_coco_samples(data_dir, max_images=max_images)
    print(f"[Dataset] Found {len(samples)} images to evaluate.")

    comp_acc: Dict[str, Dict[str, List[float]]] = {
        cname: {
            "act_mean": [], "act_std": [], "neg_frac": [], "zero_frac": [], "l2_mean": [],
            "small_ratio": [], "medium_ratio": [], "large_ratio": [], "all_ratio": [],
            "full_fisher": [], "pca_fisher": [], "class_cos_sep": [], "fg_mass_pct": [],
        }
        for cname in COMPONENT_ORDER
    }

    # Global class prototype accumulation on Neck P3 (for 80x80 pairwise separation)
    p3_dim = 64
    p3_proto_sum = np.zeros((80, p3_dim), dtype=np.float64)
    p3_proto_cnt = np.zeros(80, dtype=np.int64)

    # Head scale share of high-confidence (>=0.25) anchors
    head_scale_share = {"p3": [], "p4": [], "p5": []}

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []
    latencies_ms: List[float] = []
    vis_sample: Optional[Dict] = None

    t0_all = time.perf_counter()
    for idx, (img_p, lbl_p) in enumerate(samples):
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]

        lb_rgb, ratio, pad = letterbox_image(rgb, imgsz)
        img_t = torch.from_numpy(lb_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        raw_boxes = load_yolo_labels(lbl_p)
        gt_xyxy, gt_lb_norm = transform_boxes_letterbox(raw_boxes, (h0, w0), imgsz, ratio, pad)

        probe.clear()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        pred_out, _ = model(img_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

        # 1. Measure every dissected component
        for cname in COMPONENT_ORDER:
            feat = probe.features.get(cname)
            if feat is None:
                continue
            is_cls = cname.startswith("head.cv3")
            m_dict = measure_feature_metrics(feat, gt_lb_norm, imgsz, is_cls_logits=is_cls)
            for k, v in m_dict.items():
                comp_acc[cname][k].append(v)

        # Accumulate global prototypes at Neck P3
        p3_feat = probe.features["neck.c3k2_16 (P3 //8)"][0].float()
        _, h3, w3 = p3_feat.shape
        p3_flat = F.normalize(p3_feat.reshape(p3_dim, -1), dim=0)
        _, _, _, _, cls_m_p3 = build_masks_at_res_gpu(gt_lb_norm, h3, w3, imgsz, device)
        cnt_c = cls_m_p3.sum(dim=1)
        proto_t = cls_m_p3.float() @ p3_flat.transpose(0, 1)
        p3_proto_sum += proto_t.cpu().double().numpy()
        p3_proto_cnt += cnt_c.cpu().numpy()

        # Head scale share (fraction of sigmoid prob mass across P3, P4, P5)
        cv3_masses = []
        for cv3_name in ["head.cv3_p3 (Cls //8)", "head.cv3_p4 (Cls //16)", "head.cv3_p5 (Cls //32)"]:
            pr = probe.features[cv3_name][0].float().sigmoid().max(dim=0).values
            cv3_masses.append(float(pr.sum().item()))
        tot_mass = sum(cv3_masses) + 1e-8
        head_scale_share["p3"].append(cv3_masses[0] / tot_mass * 100.0)
        head_scale_share["p4"].append(cv3_masses[1] / tot_mass * 100.0)
        head_scale_share["p5"].append(cv3_masses[2] / tot_mass * 100.0)

        # 2. Decode predictions & NMS
        det = decode_and_nms(pred_out, conf_thres=conf_thres, iou_thres=iou_thres)
        if dataset_type == "visdrone":
            # Map VisDrone GT classes to shared COCO classes for zero-shot mAP
            mapped_gts = []
            for row in gt_xyxy:
                vc = int(row[0])
                if vc in VISDRONE_TO_COCO_MAP:
                    mapped_gts.append([VISDRONE_TO_COCO_MAP[vc], row[1], row[2], row[3], row[4]])
            gt_for_map = np.array(mapped_gts, dtype=np.float32) if mapped_gts else np.zeros((0, 5), dtype=np.float32)
            shared_coco_ids = set(VISDRONE_TO_COCO_MAP.values())
            det_for_map = det[np.isin(det[:, 5].astype(int), list(shared_coco_ids))] if len(det) > 0 else det
            all_preds.append(det_for_map)
            all_gts.append(gt_for_map)
        else:
            all_preds.append(det)
            all_gts.append(gt_xyxy)

        # Save a rich sample (with >= 4 GT boxes) for visual inspection
        if vis_sample is None and len(gt_xyxy) >= 4:
            vis_sample = {
                "img_name": img_p.name,
                "lb_rgb": lb_rgb,
                "gt_xyxy": gt_xyxy,
                "det": det[det[:, 4] >= vis_conf],
                "feat4": probe.features["backbone.stage4 (feat4 //8)"][0].float().detach(),
                "feat6": probe.features["backbone.stage6 (feat6 //16)"][0].float().detach(),
                "feat10": probe.features["backbone.stage10 (feat10 //32)"][0].float().detach(),
                "p3": probe.features["neck.c3k2_16 (P3 //8)"][0].float().detach(),
                "p4": probe.features["neck.c3k2_19 (P4 //16)"][0].float().detach(),
                "p5": probe.features["neck.c3k2_22 (P5 //32)"][0].float().detach(),
                "cv3_p3": probe.features["head.cv3_p3 (Cls //8)"][0].float().sigmoid().max(dim=0).values.cpu().numpy(),
                "cv3_p4": probe.features["head.cv3_p4 (Cls //16)"][0].float().sigmoid().max(dim=0).values.cpu().numpy(),
                "cv3_p5": probe.features["head.cv3_p5 (Cls //32)"][0].float().sigmoid().max(dim=0).values.cpu().numpy(),
            }

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            print(f"  Processed {idx + 1}/{len(samples)} images...")

    probe.remove()
    total_sec = time.perf_counter() - t0_all

    # 3. Compute Detection Metrics (Overall + Small/Medium/Large)
    overall_det, per_class_rows = compute_coco_map(all_preds, all_gts)
    small_det, _ = compute_coco_map(all_preds, all_gts, area_range=(0.0, 32.0 ** 2))
    med_det, _ = compute_coco_map(all_preds, all_gts, area_range=(32.0 ** 2, 96.0 ** 2))
    large_det, _ = compute_coco_map(all_preds, all_gts, area_range=(96.0 ** 2, 1e9))

    overall_det["mAP50_95_small"] = small_det["mAP50_95"]
    overall_det["mAP50_95_medium"] = med_det["mAP50_95"]
    overall_det["mAP50_95_large"] = large_det["mAP50_95"]
    overall_det["latency_ms"] = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    overall_det["fps"] = 1000.0 / max(1e-6, overall_det["latency_ms"])
    overall_det["total_eval_sec"] = total_sec
    overall_det["params_total_m"] = p_total
    overall_det["params_backbone_m"] = p_backbone
    overall_det["params_neck_m"] = p_neck
    overall_det["params_head_m"] = p_head
    overall_det["head_share_p3_pct"] = float(np.mean(head_scale_share["p3"]))
    overall_det["head_share_p4_pct"] = float(np.mean(head_scale_share["p4"]))
    overall_det["head_share_p5_pct"] = float(np.mean(head_scale_share["p5"]))

    # 4. Aggregate Component Rows
    comp_rows: List[Dict[str, object]] = []
    for cname in COMPONENT_ORDER:
        acc = comp_acc[cname]
        row: Dict[str, object] = {
            "component": cname,
            "module_group": cname.split(".")[0],
            "params_k": round(comp_params.get(cname, 0) / 1000.0, 2),
        }
        for mkey in [
            "act_mean", "act_std", "neg_frac", "zero_frac", "l2_mean",
            "small_ratio", "medium_ratio", "large_ratio", "all_ratio",
            "full_fisher", "pca_fisher", "class_cos_sep", "fg_mass_pct",
        ]:
            vals = acc[mkey]
            row[mkey] = round(float(np.mean(vals)), 4) if vals else np.nan
        comp_rows.append(row)

    # 5. Pairwise Cosine Separation on Neck P3
    valid_cls_ids = [c for c in range(80) if p3_proto_cnt[c] >= 5]
    pw_mat = np.full((len(valid_cls_ids), len(valid_cls_ids)), np.nan, dtype=np.float64)
    if len(valid_cls_ids) >= 2:
        protos = []
        for c in valid_cls_ids:
            v = p3_proto_sum[c] / p3_proto_cnt[c]
            v = v / (np.linalg.norm(v) + 1e-8)
            protos.append(v)
        P_mat = np.stack(protos, axis=0)
        sim_mat = P_mat @ P_mat.T
        pw_mat = 1.0 - sim_mat
        np.fill_diagonal(pw_mat, np.nan)

    # 6. Save all CSVs, Charts, and HTML Dashboard
    save_outputs(
        out_dir=out_dir,
        comp_rows=comp_rows,
        overall_det=overall_det,
        per_class_rows=per_class_rows,
        valid_cls_ids=valid_cls_ids,
        pw_mat=pw_mat,
        vis_sample=vis_sample,
        p3_proto_cnt=p3_proto_cnt,
    )

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY:")
    print(f"  mAP@0.50      : {overall_det['mAP50']:.4f}")
    print(f"  mAP@0.50:0.95 : {overall_det['mAP50_95']:.4f} (S: {overall_det['mAP50_95_small']:.4f} | M: {overall_det['mAP50_95_medium']:.4f} | L: {overall_det['mAP50_95_large']:.4f})")
    print(f"  P / R / F1    : {overall_det['precision_c25']:.4f} / {overall_det['recall_c25']:.4f} / {overall_det['f1_c25']:.4f} (conf>=0.25)")
    print(f"  Latency / FPS : {overall_det['latency_ms']:.2f} ms/img ({overall_det['fps']:.1f} FPS)")
    print(f"  Output saved  : {os.path.abspath(out_dir)}")
    print("=" * 80)


# =============================================================================
# 6. DATA ANALYST CSVs, CHARTS & HTML DASHBOARD
# =============================================================================

def save_outputs(
    out_dir: str,
    comp_rows: List[Dict[str, object]],
    overall_det: Dict[str, float],
    per_class_rows: List[Dict[str, float]],
    valid_cls_ids: List[int],
    pw_mat: np.ndarray,
    vis_sample: Optional[Dict],
    p3_proto_cnt: np.ndarray,
) -> None:
    # 1. CSVs
    comp_csv = os.path.join(out_dir, "component_breakdown.csv")
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comp_rows)

    det_csv = os.path.join(out_dir, "detection_overall_metrics.csv")
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall_det.keys()))
        writer.writeheader()
        writer.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in overall_det.items()})

    cls_csv = os.path.join(out_dir, "per_class_ap.csv")
    if per_class_rows:
        sorted_cls = sorted(per_class_rows, key=lambda r: r["ap50"], reverse=True)
        with open(cls_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sorted_cls[0].keys()))
            writer.writeheader()
            for r in sorted_cls:
                writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    # 2. Chart 01: Layer-by-Layer Component Progression
    _plot_component_progression(comp_rows, os.path.join(out_dir, "chart_01_component_progression.png"))

    # 3. Chart 02: Backbone vs Neck Multi-Scale Comparison (P3, P4, P5)
    _plot_backbone_vs_neck(comp_rows, os.path.join(out_dir, "chart_02_backbone_vs_neck_multiscale.png"))

    # 4. Chart 03: Head Scale Share & Detection AP Summary
    _plot_detection_and_head(overall_det, per_class_rows, os.path.join(out_dir, "chart_03_head_and_detection_summary.png"))

    # 5. Chart 04: Visual Feature Inspection Grid
    if vis_sample is not None:
        _plot_visual_inspection(vis_sample, os.path.join(out_dir, "chart_04_feature_visual_inspection.png"))

    # 6. Chart 05: Pairwise Class Separation Heatmap (Top 15 frequent classes)
    if len(valid_cls_ids) >= 2:
        _plot_pairwise_heatmap(valid_cls_ids, pw_mat, p3_proto_cnt, os.path.join(out_dir, "chart_05_pairwise_class_separation.png"))

    # 7. HTML Dashboard
    _generate_html_dashboard(out_dir, comp_rows, overall_det, per_class_rows)


def _plot_component_progression(comp_rows: List[Dict[str, object]], save_path: str) -> None:
    names = [r["component"].replace("backbone.", "B:").replace("neck.", "N:").replace("head.", "H:") for r in comp_rows]
    groups = [r["module_group"] for r in comp_rows]
    group_colors = {"backbone": "#4C72B0", "neck": "#55A868", "head": "#DD8452"}
    colors = [group_colors.get(g, "#888888") for g in groups]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    specs = [
        ("all_ratio", "FG/BG Energy Ratio (All Boxes) ↑", axes[0, 0]),
        ("small_ratio", "FG/BG Energy Ratio (Small Boxes <32x32) ↑", axes[0, 1]),
        ("full_fisher", "Full-Dim FG/BG Fisher Ratio ↑", axes[1, 0]),
        ("class_cos_sep", "Inter-Class Cosine Separation ↑", axes[1, 1]),
    ]
    for mkey, title, ax in specs:
        vals = [0.0 if (r[mkey] is None or np.isnan(r[mkey])) else float(r[mkey]) for r in comp_rows]
        use_log = mkey in ("all_ratio", "small_ratio") and max(vals) > 20.0
        bars = ax.bar(x, vals, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title + (" [log]" if use_log else ""), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for b, v in zip(bars, vals):
            if v > 0:
                lbl = f"{v:.0f}" if v >= 100 else f"{v:.2f}"
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), lbl, ha="center", va="bottom", fontsize=7)

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="Backbone", markerfacecolor="#4C72B0", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="Neck (PANet)", markerfacecolor="#55A868", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="Head (Detect)", markerfacecolor="#DD8452", markersize=10),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, fontsize=11, bbox_to_anchor=(0.5, 0.99))
    plt.suptitle("YOLOv11 Component-by-Component Feature Quality Progression on COCO", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_backbone_vs_neck(comp_rows: List[Dict[str, object]], save_path: str) -> None:
    by_name = {r["component"]: r for r in comp_rows}
    scales = ["P3 (//8)", "P4 (//16)", "P5 (//32)"]
    bb_keys = ["backbone.stage4 (feat4 //8)", "backbone.stage6 (feat6 //16)", "backbone.stage10 (feat10 //32)"]
    nk_keys = ["neck.c3k2_16 (P3 //8)", "neck.c3k2_19 (P4 //16)", "neck.c3k2_22 (P5 //32)"]

    metrics = [
        ("small_ratio", "Small Object Energy Ratio ↑"),
        ("all_ratio", "All Object Energy Ratio ↑"),
        ("full_fisher", "Full-Dim FG/BG Fisher ↑"),
        ("class_cos_sep", "Inter-Class Cosine Sep ↑"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    x = np.arange(3)
    w = 0.35
    for ax, (mkey, title) in zip(axes, metrics):
        bb_vals = [float(by_name[k][mkey]) for k in bb_keys]
        nk_vals = [float(by_name[k][mkey]) for k in nk_keys]
        b1 = ax.bar(x - w / 2, bb_vals, width=w, label="Backbone", color="#4C72B0", alpha=0.9)
        b2 = ax.bar(x + w / 2, nk_vals, width=w, label="Neck (PANet)", color="#55A868", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(scales, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in list(b1) + list(b2):
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.legend(fontsize=9)

    plt.suptitle("Backbone vs PANet Neck Across Scales (P3 //8, P4 //16, P5 //32)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_detection_and_head(
    overall_det: Dict[str, float],
    per_class_rows: List[Dict[str, float]],
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Overall mAP breakdown
    m_labels = ["mAP@50", "mAP@50:95", "mAP_S", "mAP_M", "mAP_L", "P@0.25", "R@0.25"]
    m_vals = [
        overall_det["mAP50"], overall_det["mAP50_95"],
        overall_det["mAP50_95_small"], overall_det["mAP50_95_medium"], overall_det["mAP50_95_large"],
        overall_det["precision_c25"], overall_det["recall_c25"],
    ]
    bars = axes[0].bar(m_labels, m_vals, color="#4C72B0", alpha=0.9, edgecolor="black")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("COCO Detection Metrics Breakdown", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars, m_vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 2: Head Cls Mass Share across P3, P4, P5
    shares = [overall_det["head_share_p3_pct"], overall_det["head_share_p4_pct"], overall_det["head_share_p5_pct"]]
    axes[1].pie(
        shares,
        labels=["P3 (//8 Small)", "P4 (//16 Medium)", "P5 (//32 Large)"],
        autopct="%1.1f%%",
        colors=["#55A868", "#4C72B0", "#DD8452"],
        startangle=140,
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    axes[1].set_title("Detection Head Cls Probability Mass Share", fontsize=12, fontweight="bold")

    # Panel 3: Top-10 & Bottom-5 Classes by AP@50 (min 5 GTs)
    valid = [r for r in per_class_rows if r["gt_count"] >= 5]
    valid_sorted = sorted(valid, key=lambda r: r["ap50"], reverse=True)
    top_bot = valid_sorted[:8] + valid_sorted[-5:]
    c_names = [f"{r['class_name']} ({int(r['gt_count'])})" for r in top_bot]
    c_aps = [r["ap50"] for r in top_bot]
    c_colors = ["#55A868"] * min(8, len(valid_sorted)) + ["#C44E52"] * min(5, max(0, len(top_bot) - 8))
    y_pos = np.arange(len(c_names))
    axes[2].barh(y_pos, c_aps, color=c_colors, alpha=0.9)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(c_names, fontsize=8.5)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1.05)
    axes[2].set_title("Top-8 & Bottom-5 COCO Classes (AP@50)", fontsize=12, fontweight="bold")
    axes[2].grid(axis="x", linestyle="--", alpha=0.4)
    for yp, v in zip(y_pos, c_aps):
        axes[2].text(v + 0.01, yp, f"{v:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_visual_inspection(vis_sample: Dict, save_path: str) -> None:
    lb_rgb = vis_sample["lb_rgb"]
    H, W = lb_rgb.shape[:2]
    gt_xyxy = vis_sample["gt_xyxy"]
    det = vis_sample["det"]

    fig, axes = plt.subplots(3, 3, figsize=(17, 16))

    # Row 0: Input+GT, Input+Pred, Head cv3 P3 Prob Heatmap
    axes[0, 0].imshow(lb_rgb)
    for g in gt_xyxy:
        c, x1, y1, x2, y2 = int(g[0]), g[1], g[2], g[3], g[4]
        axes[0, 0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="#00FF66", facecolor="none"))
        axes[0, 0].text(x1, max(0, y1 - 3), COCO_CLASSES[c], color="#00FF66", fontsize=7, fontweight="bold")
    axes[0, 0].set_title(f"GT Boxes ({len(gt_xyxy)} objs) | {vis_sample['img_name']}", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(lb_rgb)
    for d in det:
        x1, y1, x2, y2, conf, c = d[0], d[1], d[2], d[3], d[4], int(d[5])
        axes[0, 1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="#FFAA00", facecolor="none"))
        axes[0, 1].text(x1, max(0, y1 - 3), f"{COCO_CLASSES[c]} {conf:.2f}", color="#FFAA00", fontsize=7, fontweight="bold")
    axes[0, 1].set_title(f"YOLOv11 Predictions ({len(det)} dets, conf>=0.25)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    cv3_up = cv2.resize(vis_sample["cv3_p3"], (W, H), interpolation=cv2.INTER_NEAREST)
    axes[0, 2].imshow(lb_rgb, alpha=0.55)
    axes[0, 2].imshow(cv3_up, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Head cv3 P3 Max Class Prob", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # Row 1: Backbone feat4, feat6, feat10 L2-Norm
    for col, (fkey, lbl) in enumerate([("feat4", "Backbone feat4 (//8)"), ("feat6", "Backbone feat6 (//16)"), ("feat10", "Backbone feat10 (//32)")]):
        f_t = vis_sample[fkey]
        l2 = torch.linalg.vector_norm(f_t, dim=0).cpu().numpy()
        l2_up = cv2.resize(l2, (W, H), interpolation=cv2.INTER_NEAREST)
        axes[1, col].imshow(l2_up, cmap="magma")
        for g in gt_xyxy:
            axes[1, col].add_patch(patches.Rectangle((g[1], g[2]), g[3] - g[1], g[4] - g[2], linewidth=0.9, edgecolor="#00FF66", facecolor="none"))
        axes[1, col].set_title(f"{lbl} L2-Norm", fontsize=11, fontweight="bold")
        axes[1, col].axis("off")

    # Row 2: Neck P3, P4, P5 PCA-RGB
    for col, (pkey, lbl) in enumerate([("p3", "Neck P3 (//8)"), ("p4", "Neck P4 (//16)"), ("p5", "Neck P5 (//32)")]):
        f_t = vis_sample[pkey]
        _, hf, wf = f_t.shape
        pca = _compute_pca_rgb_gpu(f_t).reshape(hf, wf, 3).cpu().numpy()
        pca_up = cv2.resize(pca, (W, H), interpolation=cv2.INTER_NEAREST)
        axes[2, col].imshow(pca_up)
        for g in gt_xyxy:
            axes[2, col].add_patch(patches.Rectangle((g[1], g[2]), g[3] - g[1], g[4] - g[2], linewidth=0.9, edgecolor="#FFFFFF", facecolor="none"))
        axes[2, col].set_title(f"{lbl} PCA-RGB", fontsize=11, fontweight="bold")
        axes[2, col].axis("off")

    plt.suptitle("Visual Dissection: Predictions, Head Confidence, Backbone L2-Norm & Neck PCA-RGB", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_pairwise_heatmap(
    valid_cls_ids: List[int],
    pw_mat: np.ndarray,
    p3_proto_cnt: np.ndarray,
    save_path: str,
) -> None:
    # Select top 15 most frequent valid classes for readable heatmap
    counts = [(i, valid_cls_ids[i], p3_proto_cnt[valid_cls_ids[i]]) for i in range(len(valid_cls_ids))]
    counts.sort(key=lambda x: x[2], reverse=True)
    top = counts[:15]
    idxs = [t[0] for t in top]
    names = [COCO_CLASSES[t[1]] for t in top]
    sub = pw_mat[np.ix_(idxs, idxs)]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(sub, cmap="YlOrRd", vmin=0.0, vmax=max(0.5, float(np.nanmax(sub)) if np.isfinite(sub).any() else 1.0))
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Neck P3 Pairwise Inter-Class Cosine Distance (Top 15 Frequent COCO Classes)", fontsize=12, fontweight="bold")
    for r in range(len(names)):
        for c in range(len(names)):
            val = sub[r, c]
            txt = "-" if r == c else (f"{val:.2f}" if np.isfinite(val) else "na")
            ax.text(c, r, txt, ha="center", va="center", fontsize=7.5, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine Distance (1 - cos_sim)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _generate_html_dashboard(
    out_dir: str,
    comp_rows: List[Dict[str, object]],
    overall_det: Dict[str, float],
    per_class_rows: List[Dict[str, float]],
) -> None:
    comp_headers = "".join(f"<th>{html.escape(k)}</th>" for k in comp_rows[0].keys())
    comp_body = ""
    for r in comp_rows:
        tds = "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values())
        comp_body += f"<tr>{tds}</tr>"

    chart_files = [
        ("Component-by-Component Progression", "chart_01_component_progression.png"),
        ("Backbone vs Neck Multi-Scale Comparison", "chart_02_backbone_vs_neck_multiscale.png"),
        ("Head Scale Share & Detection AP Summary", "chart_03_head_and_detection_summary.png"),
        ("Visual Feature Inspection", "chart_04_feature_visual_inspection.png"),
        ("Pairwise Class Cosine Separation (Neck P3)", "chart_05_pairwise_class_separation.png"),
    ]
    charts_html = ""
    for title, fname in chart_files:
        if os.path.isfile(os.path.join(out_dir, fname)):
            charts_html += f"<div class='card'><h3>{html.escape(title)}</h3><img src='{fname}' style='max-width:100%;'></div>"

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>YOLOv11 COCO Component Validation Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f5f7fa; color: #222; }}
  h1, h2 {{ color: #1f3a60; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .kpi {{ background: #fff; padding: 14px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .kpi .lbl {{ font-size: 12px; color: #666; text-transform: uppercase; }}
  .kpi .val {{ font-size: 22px; font-weight: 700; color: #1f3a60; margin-top: 4px; }}
  .card {{ background: #fff; padding: 18px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 24px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #e1e5eb; padding: 6px 10px; text-align: right; }}
  th {{ background: #1f3a60; color: #fff; text-align: center; }}
  td:first-child, td:nth-child(2) {{ text-align: left; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f9fbfd; }}
</style>
</head>
<body>
  <h1>YOLOv11 Component-by-Component Validation on COCO</h1>
  <div class="kpi-grid">
    <div class="kpi"><div class="lbl">mAP@0.50</div><div class="val">{overall_det['mAP50']:.4f}</div></div>
    <div class="kpi"><div class="lbl">mAP@0.50:0.95</div><div class="val">{overall_det['mAP50_95']:.4f}</div></div>
    <div class="kpi"><div class="lbl">mAP S / M / L</div><div class="val">{overall_det['mAP50_95_small']:.3f} / {overall_det['mAP50_95_medium']:.3f} / {overall_det['mAP50_95_large']:.3f}</div></div>
    <div class="kpi"><div class="lbl">Precision / Recall (0.25)</div><div class="val">{overall_det['precision_c25']:.3f} / {overall_det['recall_c25']:.3f}</div></div>
    <div class="kpi"><div class="lbl">Latency / FPS</div><div class="val">{overall_det['latency_ms']:.1f} ms ({overall_det['fps']:.1f})</div></div>
    <div class="kpi"><div class="lbl">Params (B / N / H)</div><div class="val">{overall_det['params_total_m']:.2f}M</div></div>
  </div>
  <div class="card">
    <h2>Component-by-Component Breakdown</h2>
    <table><thead><tr>{comp_headers}</tr></thead><tbody>{comp_body}</tbody></table>
  </div>
  {charts_html}
</body>
</html>"""
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Component-by-component validation of YOLOv11 on COCO.")
    p.add_argument("--weights", type=str, default="yolo11n.pt", help="Path to pretrained YOLOv11 weights (.pt).")
    p.add_argument("--data-dir", type=str, default="D:/apps/Datasets/COCO/dataset", help="Path to COCO dataset dir (with images/ and labels/).")
    p.add_argument("--out-dir", type=str, default="yolo/output_val_coco_yolov11", help="Output directory for CSVs, charts, and dashboard.")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size (letterboxed).")
    p.add_argument("--max-images", type=int, default=0, help="Max images to evaluate (0 = all).")
    p.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold for mAP NMS.")
    p.add_argument("--iou-thres", type=float, default=0.65, help="IoU threshold for NMS.")
    p.add_argument("--vis-conf", type=float, default=0.25, help="Confidence threshold for visual inspection boxes.")
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'.")
    p.add_argument("--dataset-type", type=str, default="coco", choices=["coco", "visdrone"], help="Dataset label schema ('coco' or 'visdrone').")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_component_validation(
        weights_path=args.weights,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        max_images=args.max_images,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        vis_conf=args.vis_conf,
        device_str=args.device,
        dataset_type=args.dataset_type,
    )
