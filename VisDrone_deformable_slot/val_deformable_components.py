#!/usr/bin/env python3
"""Component-by-Component Validation & Deformable High-Res Micro-Patch Diagnostics
for `VisDroneYOLO11DeformableSlot`.

Measures:
  1. End-to-End Detection (mAP@50, mAP@50:95, Tiny <16^2, Small <32^2, Medium, Large,
     Precision/Recall/F1 @0.25, Border vs Interior Recall, per-class AP50 & Small Recall).
  2. Dual Ablation: Full vs Patch-OFF vs (Patch+FiLM)-OFF -> exact `patch_delta_mAP50`,
     `patch_delta_mAP50_tiny`, `patch_delta_mAP50_small`, and `film_delta_mAP50`.
  3. Component Progression (`Backbone -> Neck -> Slot FiLM -> Deformable Patch P3 -> Head`):
     `full_fisher`, `pca_fisher`, `class_cos_sep`, `small_ratio`, `fg_mass_pct`, `l2_mean`.
  4. Deformable Micro-Patch Diagnostics:
     - `patch_fg_hit_rate_pct`: % of K patches landing inside a GT box
     - `patch_tiny_gt_coverage_pct` / `patch_small_gt_coverage_pct` / `patch_border_gt_coverage_pct`
     - `offset_mean_cells` / `offset_max_cells` & `scale_mean_factor` / `scale_std_factor`
     - `p3_box_delta_ratio` / `p3_cls_delta_ratio` & `p3_box_fisher_delta` / `p3_cls_fisher_delta`
     - `mufasa_iter_gate`: learned MUFASA T=2 residual gate `tanh(iter_gate)`
  5. Visual charts (`chart_deformable_patches.png`, `chart_slot_confusion_matrix.png`) + `dashboard.html`.

Example:
  python VisDrone_deformable_slot/val_deformable_components.py --config VisDrone_deformable_slot/config.yaml --weights work_dirs/visdrone_deformable_slot_local/best.pth
  python VisDrone_deformable_slot/val_deformable_components.py --config VisDrone_deformable_slot/config.yaml --max-images 20
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml

from BaseModel.core import MODELS
import VisDrone_deformable_slot.dataset
import VisDrone_deformable_slot.model
from VisDrone_deformable_slot.model import VisDroneYOLO11DeformableSlot
from VisDrone_yolov11.dataset import VISDRONE_CLASSES, VisDroneLMDBDataset
from VisDrone_yolov11.val_glance_focus import _compute_detailed_recalls, _gt_cxcywh_to_xyxy_np
from yolo.val_coco_components import build_masks_at_res_gpu, compute_coco_map, measure_feature_metrics


def load_deformable_model(
    cfg: Dict[str, Any],
    weights_path: Optional[str],
    device: torch.device,
) -> VisDroneYOLO11DeformableSlot:
    """Build VisDroneYOLO11DeformableSlot from YAML config and optionally load a .pth checkpoint."""
    m_cfg = dict(cfg["model"])
    if weights_path and weights_path.endswith(".pth"):
        m_cfg["pretrained"] = None
    model = MODELS.build(m_cfg)

    if weights_path and weights_path.endswith(".pth"):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Load] Loaded {weights_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    return model.to(device).eval()


def _preds_to_np6(det: Dict[str, torch.Tensor]) -> np.ndarray:
    if len(det["boxes"]) == 0:
        return np.empty((0, 6), dtype=np.float32)
    b = det["boxes"].float().cpu().numpy()
    s = det["scores"].float().cpu().numpy()[:, None]
    l = det["labels"].long().cpu().numpy().astype(np.float32)[:, None]
    return np.concatenate([b, s, l], axis=1).astype(np.float32)


def _compute_patch_coverage_stats(
    patch_info: Dict[str, Any],
    gt_arr: np.ndarray,
    img_w: int,
    img_h: int,
    h3: int,
    w3: int,
    base_radius: float,
    border_frac: float = 0.10,
) -> Dict[str, float]:
    """Compute GT hit rate, tiny/small/border GT coverage, and deformable offset/scale stats."""
    ref_xy = patch_info["ref_xy"][0].float().cpu().numpy()        # (K, 2) in [-1, 1]
    cen_xy = patch_info["centers_xy"][0].float().cpu().numpy()    # (K, 2) in [-1, 1]
    rad_xy = patch_info["radii_xy"][0].float().cpu().numpy()      # (K, 2) in [-1, 1] half-width
    k_pts = len(cen_xy)

    # Convert deformable shift to P3 grid cell units
    dx_cells = (cen_xy[:, 0] - ref_xy[:, 0]) * (w3 * 0.5)
    dy_cells = (cen_xy[:, 1] - ref_xy[:, 1]) * (h3 * 0.5)
    shift_cells = np.sqrt(dx_cells ** 2 + dy_cells ** 2)

    scale_fac = rad_xy / max(base_radius, 1e-6)

    # Convert patch boxes to pixel [x1, y1, x2, y2] in (img_w, img_h)
    px_cx = (cen_xy[:, 0] + 1.0) * 0.5 * img_w
    px_cy = (cen_xy[:, 1] + 1.0) * 0.5 * img_h
    px_rx = rad_xy[:, 0] * 0.5 * img_w
    px_ry = rad_xy[:, 1] * 0.5 * img_h
    p_x1, p_y1 = px_cx - px_rx, px_cy - px_ry
    p_x2, p_y2 = px_cx + px_rx, px_cy + px_ry

    stats = {
        "offset_mean_cells": float(shift_cells.mean()) if k_pts > 0 else 0.0,
        "offset_max_cells": float(shift_cells.max()) if k_pts > 0 else 0.0,
        "scale_mean_factor": float(scale_fac.mean()) if k_pts > 0 else 1.0,
        "scale_std_factor": float(scale_fac.std()) if k_pts > 0 else 0.0,
        "patch_fg_hits": 0.0,
        "patch_total": float(k_pts),
        "gt_tiny_cov": 0.0,
        "gt_tiny_tot": 0.0,
        "gt_small_cov": 0.0,
        "gt_small_tot": 0.0,
        "gt_border_cov": 0.0,
        "gt_border_tot": 0.0,
    }

    if len(gt_arr) == 0 or k_pts == 0:
        return stats

    gx1, gy1, gx2, gy2 = gt_arr[:, 1], gt_arr[:, 2], gt_arr[:, 3], gt_arr[:, 4]
    gcx = 0.5 * (gx1 + gx2)
    gcy = 0.5 * (gy1 + gy2)
    g_area = np.maximum(0.0, gx2 - gx1) * np.maximum(0.0, gy2 - gy1)

    # 1. Patch FG hit rate: patch center inside any GT box (or overlaps any GT box)
    # (K, M) center-in-GT check
    in_gt = (
        (px_cx[:, None] >= gx1[None, :])
        & (px_cx[:, None] <= gx2[None, :])
        & (px_cy[:, None] >= gy1[None, :])
        & (px_cy[:, None] <= gy2[None, :])
    )
    stats["patch_fg_hits"] = float(in_gt.any(axis=1).sum())

    # 2. GT coverage: GT center falls inside at least 1 sampled micro-patch rectangle
    gt_in_patch = (
        (gcx[None, :] >= p_x1[:, None])
        & (gcx[None, :] <= p_x2[:, None])
        & (gcy[None, :] >= p_y1[:, None])
        & (gcy[None, :] <= p_y2[:, None])
    ).any(axis=0)  # (M,)

    m_tiny = g_area < (16.0 ** 2)
    m_small = g_area < (32.0 ** 2)
    m_border = (
        (gcx < border_frac * img_w)
        | (gcx > (1.0 - border_frac) * img_w)
        | (gcy < border_frac * img_h)
        | (gcy > (1.0 - border_frac) * img_h)
    )

    stats["gt_tiny_tot"] = float(m_tiny.sum())
    stats["gt_tiny_cov"] = float((m_tiny & gt_in_patch).sum())
    stats["gt_small_tot"] = float(m_small.sum())
    stats["gt_small_cov"] = float((m_small & gt_in_patch).sum())
    stats["gt_border_tot"] = float(m_border.sum())
    stats["gt_border_cov"] = float((m_border & gt_in_patch).sum())
    return stats


def _box_iou_matrix_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix (M, N) for xyxy numpy arrays."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    a2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    return inter / np.maximum(a1[:, None] + a2[None, :] - inter, 1e-7)


def _compute_bottleneck_diagnostics(
    preds_full: List[np.ndarray],
    gts_all: List[np.ndarray],
    iou_thresholds: Tuple[float, ...] = (0.10, 0.30, 0.50),
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Compute IoU-sweep recalls, class-agnostic recalls, and matched GT confidence/confusion stats."""

    # Overall & size-split recall at IoU 0.1, 0.3, 0.5 (conf >= 0.25, class-aware & class-agnostic)
    tot_all = 0
    tot_tiny = 0
    tot_small = 0
    hits_ca: Dict[float, List[int]] = {t: [0, 0, 0] for t in iou_thresholds}   # [all, tiny, small]
    hits_agn: Dict[float, List[int]] = {t: [0, 0, 0] for t in iou_thresholds}

    # Per-class matched GT stats at IoU >= 0.5 (any conf >= 0.001)
    cls_tot = [0] * 10
    cls_best_confs: List[List[float]] = [[] for _ in range(10)]
    cls_agn_hits_05 = [0] * 10
    cls_right_hits_05 = [0] * 10
    cls_right_c25 = [0] * 10

    for pred, gt in zip(preds_full, gts_all):
        if len(gt) == 0:
            continue
        g_boxes = gt[:, 1:5]
        g_cls = gt[:, 0].astype(np.int64)
        g_area = np.maximum(0.0, g_boxes[:, 2] - g_boxes[:, 0]) * np.maximum(0.0, g_boxes[:, 3] - g_boxes[:, 1])
        m_tiny = g_area < (16.0 ** 2)
        m_small = g_area < (32.0 ** 2)

        m_gt = len(gt)
        tot_all += m_gt
        tot_tiny += int(m_tiny.sum())
        tot_small += int(m_small.sum())
        for c in g_cls:
            if 0 <= c < 10:
                cls_tot[c] += 1

        if len(pred) == 0:
            continue

        # 1. All predictions (conf >= 0.001) for matched confidence & class confusion at IoU >= 0.5
        iou_all = _box_iou_matrix_np(g_boxes, pred[:, :4])  # (M, N)
        p_cls_all = pred[:, 5].astype(np.int64)
        p_conf_all = pred[:, 4]
        same_cls = g_cls[:, None] == p_cls_all[None, :]

        # Best IoU regardless of class
        best_iou_agn = iou_all.max(axis=1)
        # Best confidence among predictions with same class & IoU >= 0.5
        valid_same = same_cls & (iou_all >= 0.50)
        best_conf_same = np.where(valid_same, p_conf_all[None, :], -1.0).max(axis=1)

        for gi, c in enumerate(g_cls):
            if not (0 <= c < 10):
                continue
            if best_iou_agn[gi] >= 0.50:
                cls_agn_hits_05[c] += 1
            if best_conf_same[gi] >= 0.0:
                cls_right_hits_05[c] += 1
                cls_best_confs[c].append(float(best_conf_same[gi]))
                if best_conf_same[gi] >= 0.25:
                    cls_right_c25[c] += 1

        # 2. IoU sweep at conf >= 0.25
        p25 = pred[pred[:, 4] >= 0.25]
        if len(p25) == 0:
            continue
        iou_25 = _box_iou_matrix_np(g_boxes, p25[:, :4])
        p_cls_25 = p25[:, 5].astype(np.int64)
        same_25 = g_cls[:, None] == p_cls_25[None, :]
        max_agn = iou_25.max(axis=1)
        max_ca = np.where(same_25, iou_25, 0.0).max(axis=1)

        for t in iou_thresholds:
            ha = max_ca >= t
            hits_ca[t][0] += int(ha.sum())
            hits_ca[t][1] += int((ha & m_tiny).sum())
            hits_ca[t][2] += int((ha & m_small).sum())

            hg = max_agn >= t
            hits_agn[t][0] += int(hg.sum())
            hits_agn[t][1] += int((hg & m_tiny).sum())
            hits_agn[t][2] += int((hg & m_small).sum())

    summary: Dict[str, float] = {}
    for t in iou_thresholds:
        tag = f"{int(round(t * 100)):02d}"
        summary[f"rec_ca_c25_iou{tag}"] = hits_ca[t][0] / max(tot_all, 1)
        summary[f"rec_ca_tiny_c25_iou{tag}"] = hits_ca[t][1] / max(tot_tiny, 1)
        summary[f"rec_ca_small_c25_iou{tag}"] = hits_ca[t][2] / max(tot_small, 1)
        summary[f"rec_agn_c25_iou{tag}"] = hits_agn[t][0] / max(tot_all, 1)
        summary[f"rec_agn_tiny_c25_iou{tag}"] = hits_agn[t][1] / max(tot_tiny, 1)
        summary[f"rec_agn_small_c25_iou{tag}"] = hits_agn[t][2] / max(tot_small, 1)

    per_cls_rows: List[Dict[str, Any]] = []
    for c in range(10):
        confs = np.array(cls_best_confs[c], dtype=np.float32)
        tot = max(cls_tot[c], 1)
        agn_h = cls_agn_hits_05[c]
        r_h = cls_right_hits_05[c]
        wrong_cls_pct = 100.0 * max(0, agn_h - r_h) / max(agn_h, 1) if agn_h > 0 else 0.0
        per_cls_rows.append({
            "class_id": c,
            "class_name": VISDRONE_CLASSES[c],
            "gt_count": cls_tot[c],
            "agn_match_iou50_pct": round(100.0 * agn_h / tot, 2),
            "cls_match_iou50_pct": round(100.0 * r_h / tot, 2),
            "wrong_cls_among_matched_pct": round(wrong_cls_pct, 2),
            "cls_match_c25_pct": round(100.0 * cls_right_c25[c] / tot, 2),
            "matched_conf_mean": round(float(confs.mean()), 4) if len(confs) else 0.0,
            "matched_conf_median": round(float(np.median(confs)), 4) if len(confs) else 0.0,
            "matched_conf_p75": round(float(np.percentile(confs, 75)), 4) if len(confs) else 0.0,
        })
    return summary, per_cls_rows


def _pool_gt_features(
    fmap: torch.Tensor,
    gt_arr: np.ndarray,
    hr_w: int,
    hr_h: int,
) -> Optional[torch.Tensor]:
    """Pool (M, C) feature vectors for each GT box in `gt_arr` from (1, C, Hf, Wf)."""
    if len(gt_arr) == 0:
        return None
    _, c, hf, wf = fmap.shape
    feats_out = []
    for g in gt_arr:
        x1 = max(0, min(wf - 1, int(np.floor(g[1] * wf / float(hr_w)))))
        y1 = max(0, min(hf - 1, int(np.floor(g[2] * hf / float(hr_h)))))
        x2 = max(x1 + 1, min(wf, int(np.ceil(g[3] * wf / float(hr_w)))))
        y2 = max(y1 + 1, min(hf, int(np.ceil(g[4] * hf / float(hr_h)))))
        vec = fmap[0, :, y1:y2, x1:x2].float().mean(dim=(1, 2))
        feats_out.append(vec)
    return torch.stack(feats_out, dim=0)


def _run_linear_probe(
    feats_x: torch.Tensor,
    labels_y: torch.Tensor,
    is_small: torch.Tensor,
    is_tiny: torch.Tensor,
    device: torch.device,
    epochs: int = 80,
) -> Dict[str, float]:
    """Train a fast L2-normalized linear classifier (70/30 split) to measure class separability."""
    n = len(labels_y)
    if n < 50:
        return {"probe_acc_all": 0.0, "probe_acc_small": 0.0, "probe_acc_tiny": 0.0}

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g)
    split = int(0.7 * n)
    tr_idx, va_idx = perm[:split], perm[split:]

    x_tr = F.normalize(feats_x[tr_idx].to(device), dim=-1)
    y_tr = labels_y[tr_idx].to(device)
    x_va = F.normalize(feats_x[va_idx].to(device), dim=-1)
    y_va = labels_y[va_idx].to(device)
    sm_va = is_small[va_idx].to(device)
    tn_va = is_tiny[va_idx].to(device)

    clf = torch.nn.Linear(x_tr.shape[1], 10).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=0.05, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(clf(x_tr), y_tr)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = clf(x_va).argmax(dim=-1)
        correct = pred == y_va
        acc_all = float(correct.float().mean().item())
        acc_sm = float(correct[sm_va].float().mean().item()) if sm_va.any() else 0.0
        acc_tn = float(correct[tn_va].float().mean().item()) if tn_va.any() else 0.0

    return {
        "probe_acc_all": round(acc_all * 100.0, 2),
        "probe_acc_small": round(acc_sm * 100.0, 2),
        "probe_acc_tiny": round(acc_tn * 100.0, 2),
    }


def _plot_deformable_patches(vis_sample: Dict[str, Any], save_path: str) -> None:
    """Save 2x2 visual inspection of Deformable High-Res Micro-Patch sampling & P3 injection."""
    rgb = vis_sample["rgb"]
    h, w = rgb.shape[:2]
    gt_arr = vis_sample["gt_arr"]
    det = vis_sample["det"]
    p_info = vis_sample["patch_info"]
    p3_delta_l2 = vis_sample["p3_delta_l2"]  # (H3, W3)

    ref_xy = p_info["ref_xy"][0].float().cpu().numpy()
    cen_xy = p_info["centers_xy"][0].float().cpu().numpy()
    rad_xy = p_info["radii_xy"][0].float().cpu().numpy()

    rx = (ref_xy[:, 0] + 1.0) * 0.5 * w
    ry = (ref_xy[:, 1] + 1.0) * 0.5 * h
    cx = (cen_xy[:, 0] + 1.0) * 0.5 * w
    cy = (cen_xy[:, 1] + 1.0) * 0.5 * h
    pw = rad_xy[:, 0] * w
    ph = rad_xy[:, 1] * h

    # Classify patches as FG hit vs BG
    hit_mask = np.zeros(len(cx), dtype=bool)
    if len(gt_arr) > 0:
        hit_mask = (
            (cx[:, None] >= gt_arr[None, :, 1])
            & (cx[:, None] <= gt_arr[None, :, 3])
            & (cy[:, None] >= gt_arr[None, :, 2])
            & (cy[:, None] <= gt_arr[None, :, 4])
        ).any(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel 1: All K deformable micro-patch rectangles
    axes[0, 0].imshow(rgb)
    for i in range(len(cx)):
        col = "#00FF66" if hit_mask[i] else "#FF5555"
        axes[0, 0].add_patch(
            patches.Rectangle((cx[i] - 0.5 * pw[i], cy[i] - 0.5 * ph[i]), pw[i], ph[i],
                              linewidth=0.9, edgecolor=col, facecolor="none", alpha=0.75)
        )
    axes[0, 0].set_title(
        f"Deformable High-Res Micro-Patches (K={len(cx)}, FG hits={int(hit_mask.sum())}, green=GT hit)",
        fontsize=11, fontweight="bold",
    )
    axes[0, 0].axis("off")

    # Panel 2: Reference -> Deformable Center shift vectors
    axes[0, 1].imshow(rgb, alpha=0.7)
    axes[0, 1].scatter(rx, ry, s=10, c="#00CCFF", label="Ref (P3 Top-K)", alpha=0.7)
    axes[0, 1].scatter(cx, cy, s=12, c="#FFDD00", label="Deformed Center", alpha=0.85)
    for i in range(len(cx)):
        axes[0, 1].plot([rx[i], cx[i]], [ry[i], cy[i]], color="#FFDD00", linewidth=0.8, alpha=0.8)
    axes[0, 1].legend(loc="upper right", fontsize=9)
    axes[0, 1].set_title("Learned Deformable Offsets (Ref -> Sampled Center)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 3: P3 High-Res Injection Residual L2 Map
    d_norm = p3_delta_l2 / max(float(p3_delta_l2.max()), 1e-6)
    d_up = cv2.resize(d_norm, (w, h), interpolation=cv2.INTER_NEAREST)
    axes[1, 0].imshow(rgb, alpha=0.5)
    axes[1, 0].imshow(d_up, cmap="inferno", alpha=0.65, vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("P3 High-Res Injection Residual ||delta_P3||_2", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 4: Predictions (orange) vs GT (green)
    axes[1, 1].imshow(rgb)
    for g in gt_arr:
        axes[1, 1].add_patch(patches.Rectangle((g[1], g[2]), g[3] - g[1], g[4] - g[2], linewidth=1.0, edgecolor="#00FF66", facecolor="none"))
    for d in det:
        if d[4] >= 0.25:
            axes[1, 1].add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], linewidth=1.2, edgecolor="#FFAA00", facecolor="none"))
    axes[1, 1].set_title(f"GT (green={len(gt_arr)}) vs Preds (orange, conf>=0.25: {int((det[:, 4] >= 0.25).sum()) if len(det) else 0})", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_slot_confusion(conf_mat: np.ndarray, save_path: str) -> None:
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    pct = np.divide(conf_mat, np.maximum(row_sums, 1.0)) * 100.0
    col_names = list(VISDRONE_CLASSES) + ["BG"]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_xticks(range(11))
    ax.set_yticks(range(10))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(VISDRONE_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Slot (Argmax attn_ori)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Ground-Truth Class", fontsize=10, fontweight="bold")
    ax.set_title("Slot Attention Confusion Matrix (% of GT Pixels)", fontsize=12, fontweight="bold")
    for r in range(10):
        for c in range(11):
            v = pct[r, c]
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if v > 50 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% GT Pixels")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _write_dashboard_html(
    out_dir: str,
    det_summary: Dict[str, float],
    ablation_summary: Dict[str, float],
    patch_summary: Dict[str, float],
    comp_rows: List[Dict[str, object]],
) -> None:
    def _kpi_grid(d: Dict[str, float]) -> str:
        return "<div class='kpi-grid'>" + "".join(
            f"<div class='kpi'><div class='lbl'>{html.escape(k)}</div><div class='val'>{v:.4f}</div></div>"
            for k, v in d.items()
        ) + "</div>"

    headers = "".join(f"<th>{html.escape(k)}</th>" for k in comp_rows[0].keys())
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>"
        for r in comp_rows
    )
    charts = [
        ("Deformable High-Res Micro-Patch Inspection", "chart_deformable_patches.png"),
        ("Slot Attention Confusion Matrix", "chart_slot_confusion_matrix.png"),
    ]
    charts_html = "".join(
        f"<div class='card'><h3>{html.escape(t)}</h3><img src='{fn}' style='max-width:100%;'></div>"
        for t, fn in charts if os.path.isfile(os.path.join(out_dir, fn))
    )

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>VisDrone Deformable-Slot Component Validation Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f5f7fa; color: #222; }}
  h1, h2 {{ color: #1f3a60; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .kpi {{ background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .kpi .lbl {{ font-size: 11px; color: #666; text-transform: uppercase; }}
  .kpi .val {{ font-size: 20px; font-weight: 700; color: #1f3a60; margin-top: 4px; }}
  .card {{ background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #e0e6ed; padding: 6px 8px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f0f4f8; }}
</style></head><body>
<h1>VisDroneYOLO11DeformableSlot — Component & Deformable Patch Validation</h1>
<h2>1. End-to-End Detection & Stratified Recall</h2>{_kpi_grid(det_summary)}
<h2>2. Dual Ablation (Deformable Micro-Patch & Slot FiLM)</h2>{_kpi_grid(ablation_summary)}
<h2>3. Deformable High-Res Micro-Patch Diagnostics</h2>{_kpi_grid(patch_summary)}
<div class="card"><h2>4. Component-by-Component Feature Progression</h2><table><tr>{headers}</tr>{body}</table></div>
{charts_html}
</body></html>"""
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(page)


def main() -> None:
    parser = argparse.ArgumentParser(description="Component & Deformable Patch Validation for VisDroneYOLO11DeformableSlot.")
    parser.add_argument("--config", type=str, default="VisDrone_deformable_slot/config.yaml", help="YAML config path")
    parser.add_argument("--weights", type=str, default=None, help="Optional .pth checkpoint path (defaults to config pretrained)")
    parser.add_argument("--out-dir", type=str, default="VisDrone_deformable_slot/output_val_deformable", help="Output folder")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of val images (default: all 548)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence cutoff for mAP")
    parser.add_argument("--iou-thres", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    model = load_deformable_model(cfg, args.weights, device)

    val_ds_cfg = cfg["data"]["val"]["dataset"]
    # Infer high-res size from val pipeline Resize
    hr_size = (1280, 1280)
    for step in val_ds_cfg.get("pipeline", []) or []:
        if isinstance(step, dict) and step.get("type") == "Resize" and "size" in step:
            sz = step["size"]
            hr_size = (int(sz[0]), int(sz[1]))
            break

    val_ds = VisDroneLMDBDataset(
        data_root=val_ds_cfg["data_root"],
        num_classes=int(val_ds_cfg.get("num_classes", 10)),
        include_boxes=True,
        cache_images=None,
        augment=False,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    total_imgs = len(val_ds) if args.max_images is None else min(args.max_images, len(val_ds))
    print(f"[Val Deformable] Evaluating {total_imgs}/{len(val_ds)} val images (HR={hr_size}, BB={model.backbone_size}) on {device}...")

    # Component feature accumulators
    comp_names = [
        "backbone.feat4 (//8)",
        "backbone.feat6 (//16)",
        "backbone.feat10 (//32)",
        "neck.P3 (//8)",
        "neck.P4 (//16)",
        "neck.P5 (//32)",
        "slot.cls_P3 (FiLM)",
        "patch.box_P3 (Deform+HR)",
        "patch.cls_P3 (Deform+HR)",
    ]
    comp_acc: Dict[str, Dict[str, List[float]]] = {
        k: {m: [] for m in ("l2_mean", "small_ratio", "medium_ratio", "large_ratio", "full_fisher", "pca_fisher", "class_cos_sep", "fg_mass_pct")}
        for k in comp_names
    }

    preds_full: List[np.ndarray] = []
    preds_patch_off: List[np.ndarray] = []
    preds_both_off: List[np.ndarray] = []
    gts_all: List[np.ndarray] = []
    img_sizes: List[Tuple[int, int]] = []
    slot_conf_mat = np.zeros((10, 11), dtype=np.float64)

    patch_stats_acc: Dict[str, List[float]] = {
        "offset_mean_cells": [], "offset_max_cells": [],
        "scale_mean_factor": [], "scale_std_factor": [],
        "p3_box_delta_ratio": [], "p3_cls_delta_ratio": [],
    }
    patch_totals = {k: 0.0 for k in ("patch_fg_hits", "patch_total", "gt_tiny_cov", "gt_tiny_tot", "gt_small_cov", "gt_small_tot", "gt_border_cov", "gt_border_tot")}
    tal_stats = {
        "all_tot": 0, "all_in_box": 0, "all_pos": 0,
        "tiny_tot": 0, "tiny_in_box": 0, "tiny_pos": 0,
        "small_tot": 0, "small_in_box": 0, "small_pos": 0,
        "cls_tot": np.zeros(10, dtype=np.int64),
        "cls_pos": np.zeros(10, dtype=np.int64),
        "cls_anchors": np.zeros(10, dtype=np.int64),
    }
    probe_layers = ("backbone.feat4 (//8)", "neck.P3 (//8)", "slot.cls_P3 (FiLM)", "patch.cls_P3 (Deform+HR)")
    probe_feats: Dict[str, List[torch.Tensor]] = {k: [] for k in probe_layers}
    probe_labels: List[torch.Tensor] = []
    probe_small: List[torch.Tensor] = []
    probe_tiny: List[torch.Tensor] = []
    latencies: List[float] = []
    vis_sample: Optional[Dict[str, Any]] = None
    best_vis_gt = -1

    base_rad = model.patch_injector.base_patch_radius if model.patch_injector is not None else 0.025

    t_start = time.perf_counter()
    with torch.no_grad():
        for idx in range(total_imgs):
            sample = val_ds.get_data_info(idx)
            orig_img = sample["image"]
            target = sample["target"]

            hr_img = TF.resize(orig_img, [hr_size[0], hr_size[1]])
            x_hr = TF.to_tensor(hr_img).unsqueeze(0).to(device)
            hr_h, hr_w = hr_size[0], hr_size[1]
            img_sizes.append((hr_w, hr_h))

            gt_arr = _gt_cxcywh_to_xyxy_np(
                target.get("boxes", torch.empty((0, 4))),
                target.get("labels", torch.empty((0,), dtype=torch.long)),
                hr_w, hr_h,
                is_cxcywh=bool(target.get("_is_cxcywh", True)),
            )
            gts_all.append(gt_arr)

            # 1. Full forward (measure latency + capture intermediate features)
            t0 = time.perf_counter()
            x_lr, (lr_h, lr_w) = model._prepare_backbone_input(x_hr)
            feat4, feat6, feat10 = model.net.backbone(x_lr)
            p3, p4, p5 = model.net.neck(feat4, feat6, feat10)
            cls_feats, slot_info = model.slot_refiner([p3, p4, p5])

            p3_box_ref, p3_cls_ref = p3, cls_feats[0]
            patch_info = None
            if model.patch_injector is not None:
                p3_box_ref, p3_cls_ref, patch_info = model.patch_injector(
                    x_hr=x_hr, p3_box=p3, p3_cls=cls_feats[0], attn_ori=slot_info["attn_ori"]
                )
            det_full = model.predict(x_hr, conf_threshold=args.conf_thres, nms_threshold=args.iou_thres, max_det=args.max_det)[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)
            det_full_np = _preds_to_np6(det_full)
            preds_full.append(det_full_np)

            # 2. Ablation A: Patch OFF (FiLM ON)
            saved_inj = model.patch_injector
            model.patch_injector = None
            det_poff = model.predict(x_hr, conf_threshold=args.conf_thres, nms_threshold=args.iou_thres, max_det=args.max_det)[0]
            preds_patch_off.append(_preds_to_np6(det_poff))
            model.patch_injector = saved_inj

            # 3. Ablation B: Both Patch OFF & FiLM OFF (pure YOLOv11n on x_lr)
            # Reuse neck feats directly via base VisDroneYOLO11.predict on x_lr, then scale boxes to hr_size
            det_boff = super(type(model).__mro__[1], model).predict(
                x_lr, conf_threshold=args.conf_thres, nms_threshold=args.iou_thres, max_det=args.max_det
            )[0]
            if len(det_boff["boxes"]) > 0:
                det_boff["boxes"][:, [0, 2]] *= hr_w / float(lr_w)
                det_boff["boxes"][:, [1, 3]] *= hr_h / float(lr_h)
            preds_both_off.append(_preds_to_np6(det_boff))

            # 4. Component feature metrics (normalized [cls, x1, y1, x2, y2] in [0, 1])
            gt_lb_norm = gt_arr.copy()
            if len(gt_lb_norm) > 0:
                gt_lb_norm[:, [1, 3]] /= float(hr_w)
                gt_lb_norm[:, [2, 4]] /= float(hr_h)
            gt_lr = gt_arr.copy()
            if len(gt_lr) > 0:
                gt_lr[:, [1, 3]] *= lr_w / float(hr_w)
                gt_lr[:, [2, 4]] *= lr_h / float(hr_h)

            feat_map_dict = {
                "backbone.feat4 (//8)": feat4,
                "backbone.feat6 (//16)": feat6,
                "backbone.feat10 (//32)": feat10,
                "neck.P3 (//8)": p3,
                "neck.P4 (//16)": p4,
                "neck.P5 (//32)": p5,
                "slot.cls_P3 (FiLM)": cls_feats[0],
                "patch.box_P3 (Deform+HR)": p3_box_ref,
                "patch.cls_P3 (Deform+HR)": p3_cls_ref,
            }
            for cname, ft in feat_map_dict.items():
                fm = measure_feature_metrics(ft, gt_lb_norm, lr_h, is_cls_logits=False)
                for mk in comp_acc[cname].keys():
                    if mk in fm and np.isfinite(fm[mk]):
                        comp_acc[cname][mk].append(float(fm[mk]))

            # 5. Slot confusion matrix on P3
            h3, w3 = slot_info["resolutions"][0]
            n3 = h3 * w3
            attn_p3 = slot_info["attn_ori"][0, :, :n3].float()  # (11, N3)
            pred_s = attn_p3.argmax(dim=0)  # (N3,)
            if len(gt_lr) > 0:
                for g in gt_lr:
                    cid = int(g[0])
                    if 0 <= cid < 10:
                        x1 = max(0, min(w3 - 1, int(g[1] * w3 / lr_w)))
                        y1 = max(0, min(h3 - 1, int(g[2] * h3 / lr_h)))
                        x2 = max(x1 + 1, min(w3, int(np.ceil(g[3] * w3 / lr_w))))
                        y2 = max(y1 + 1, min(h3, int(np.ceil(g[4] * h3 / lr_h))))
                        sl = pred_s.view(h3, w3)[y1:y2, x1:x2].reshape(-1)
                        slot_conf_mat[cid] += torch.bincount(sl, minlength=11).cpu().numpy()

            # 6. Deformable Micro-Patch coverage & modulation diagnostics
            if patch_info is not None:
                ps = _compute_patch_coverage_stats(patch_info, gt_arr, hr_w, hr_h, h3, w3, base_rad)
                for k in ("offset_mean_cells", "offset_max_cells", "scale_mean_factor", "scale_std_factor"):
                    patch_stats_acc[k].append(ps[k])
                for k in patch_totals:
                    patch_totals[k] += ps[k]

                b_delta = float((p3_box_ref - p3).float().norm().item() / max(float(p3.float().norm().item()), 1e-6))
                c_delta = float((p3_cls_ref - cls_feats[0]).float().norm().item() / max(float(cls_feats[0].float().norm().item()), 1e-6))
                patch_stats_acc["p3_box_delta_ratio"].append(b_delta)
                patch_stats_acc["p3_cls_delta_ratio"].append(c_delta)

                if len(gt_arr) > best_vis_gt:
                    best_vis_gt = len(gt_arr)
                    vis_sample = {
                        "rgb": np.array(hr_img, dtype=np.uint8),
                        "gt_arr": gt_arr,
                        "det": det_full_np,
                        "patch_info": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in patch_info.items()},
                        "p3_delta_l2": torch.linalg.vector_norm((p3_box_ref[0] - p3[0]).float(), dim=0).cpu().numpy(),
                    }

            # 7. TAL GT anchor starvation & Linear Probe feature extraction
            if len(gt_arr) > 0:
                _, boxes_raw, scores_raw, slot_info_f, lr_sz_f = model._forward_head_with_slots(x_hr)
                decoded_xyxy, ap_px, _ = model._build_anchors_and_decode(
                    [p3, p4, p5], boxes_raw, slot_info_f, lr_sz_f
                )
                b_ap = ap_px[0] if ap_px.dim() == 3 else ap_px  # (N_total, 2)
                gt_lr_t = torch.from_numpy(gt_lr[:, 1:5]).to(device=device, dtype=torch.float32)
                gt_lbl_t = torch.from_numpy(gt_lr[:, 0]).to(device=device, dtype=torch.long)

                # In-box check per GT
                lt = b_ap.unsqueeze(0) - gt_lr_t[:, :2].unsqueeze(1)
                rb = gt_lr_t[:, 2:].unsqueeze(1) - b_ap.unsqueeze(0)
                in_box_m = (torch.cat([lt, rb], dim=-1).min(dim=-1).values > 0.0).any(dim=1)  # (M,)

                # Actual TAL assignment check per GT
                b_pred_scores = scores_raw.sigmoid().transpose(1, 2)[0]
                b_pred_boxes = decoded_xyxy[0]
                ious_m = model.criterion.assigner._batch_iou(gt_lr_t, b_pred_boxes)
                cls_sc_m = b_pred_scores[:, gt_lbl_t].transpose(0, 1)
                is_in = torch.cat([lt, rb], dim=-1).min(dim=-1).values > 0.0
                align_m = (cls_sc_m.clamp(min=0.0) ** model.criterion.assigner.alpha) * (ious_m.clamp(min=0.0) ** model.criterion.assigner.beta) * is_in.float()
                k_tal = min(model.criterion.assigner.topk, align_m.shape[1])
                _, topk_idx_m = torch.topk(align_m, k=k_tal, dim=-1, largest=True)
                topk_mask_m = torch.zeros_like(align_m, dtype=torch.bool).scatter_(1, topk_idx_m, True)
                pos_cnt_per_gt = (topk_mask_m & is_in & (align_m > 0)).sum(dim=1)  # (M,)
                has_pos_m = pos_cnt_per_gt > 0

                g_area_hr = (gt_arr[:, 3] - gt_arr[:, 1]) * (gt_arr[:, 4] - gt_arr[:, 2])
                m_tn_t = torch.from_numpy(g_area_hr < (16.0 ** 2)).to(device)
                m_sm_t = torch.from_numpy(g_area_hr < (32.0 ** 2)).to(device)

                tal_stats["all_tot"] += len(gt_arr)
                tal_stats["all_in_box"] += int(in_box_m.sum().item())
                tal_stats["all_pos"] += int(has_pos_m.sum().item())
                tal_stats["tiny_tot"] += int(m_tn_t.sum().item())
                tal_stats["tiny_in_box"] += int((in_box_m & m_tn_t).sum().item())
                tal_stats["tiny_pos"] += int((has_pos_m & m_tn_t).sum().item())
                tal_stats["small_tot"] += int(m_sm_t.sum().item())
                tal_stats["small_in_box"] += int((in_box_m & m_sm_t).sum().item())
                tal_stats["small_pos"] += int((has_pos_m & m_sm_t).sum().item())

                for gi in range(len(gt_arr)):
                    cid = int(gt_arr[gi, 0])
                    if 0 <= cid < 10:
                        tal_stats["cls_tot"][cid] += 1
                        if has_pos_m[gi].item():
                            tal_stats["cls_pos"][cid] += 1
                        tal_stats["cls_anchors"][cid] += int(pos_cnt_per_gt[gi].item())

                # Pool GT features for linear probe (sub-sample up to 40 boxes/img for speed)
                step_p = max(1, len(gt_arr) // 40)
                gt_sub = gt_arr[::step_p]
                g_area_sub = (gt_sub[:, 3] - gt_sub[:, 1]) * (gt_sub[:, 4] - gt_sub[:, 2])
                for lname in probe_layers:
                    pv = _pool_gt_features(feat_map_dict[lname], gt_sub, hr_w, hr_h)
                    if pv is not None:
                        probe_feats[lname].append(pv.cpu())
                probe_labels.append(torch.from_numpy(gt_sub[:, 0].astype(np.int64)))
                probe_small.append(torch.from_numpy(g_area_sub < (32.0 ** 2)))
                probe_tiny.append(torch.from_numpy(g_area_sub < (16.0 ** 2)))

            if (idx + 1) % 50 == 0 or (idx + 1) == total_imgs:
                print(f"  [{idx + 1:3d}/{total_imgs}] avg_latency={np.mean(latencies):.1f}ms")

    # Compute Detection & Dual Ablation metrics
    def _eval_all(preds_l: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[int, float], Dict[int, Dict[str, float]]]:
        ov, rows = compute_coco_map(preds_l, gts_all)
        tn, _ = compute_coco_map(preds_l, gts_all, area_range=(0.0, 16.0 ** 2))
        sm, _ = compute_coco_map(preds_l, gts_all, area_range=(0.0, 32.0 ** 2))
        md, _ = compute_coco_map(preds_l, gts_all, area_range=(32.0 ** 2, 96.0 ** 2))
        lg, _ = compute_coco_map(preds_l, gts_all, area_range=(96.0 ** 2, 1e9))
        rec_d, cls_sm = _compute_detailed_recalls(preds_l, gts_all, img_sizes, conf_thr=0.25, iou_thr=0.50)
        res = {
            "mAP50": float(ov.get("mAP50", 0.0)),
            "mAP50_95": float(ov.get("mAP50_95", 0.0)),
            "mAP50_tiny": float(tn.get("mAP50", 0.0)),
            "mAP50_95_tiny": float(tn.get("mAP50_95", 0.0)),
            "mAP50_small": float(sm.get("mAP50", 0.0)),
            "mAP50_95_small": float(sm.get("mAP50_95", 0.0)),
            "mAP50_95_medium": float(md.get("mAP50_95", 0.0)),
            "mAP50_95_large": float(lg.get("mAP50_95", 0.0)),
            **{k: rec_d[k] for k in ("precision_c25", "recall_c25", "f1_c25", "rec_tiny_c25", "rec_small_c25", "rec_border_c25", "rec_interior_c25")},
            "latency_ms": float(np.mean(latencies)),
        }
        ap50_c = {int(r["class_id"]): float(r.get("AP50", r.get("ap50", 0.0))) for r in rows}
        return res, ap50_c, cls_sm

    det_full_m, ap50_full, cls_sm_full = _eval_all(preds_full)
    det_poff_m, _, _ = _eval_all(preds_patch_off)
    det_boff_m, _, _ = _eval_all(preds_both_off)

    ablation_summary = {
        "full_mAP50": det_full_m["mAP50"],
        "patch_off_mAP50": det_poff_m["mAP50"],
        "both_off_mAP50": det_boff_m["mAP50"],
        "patch_delta_mAP50": det_full_m["mAP50"] - det_poff_m["mAP50"],
        "patch_delta_mAP50_tiny": det_full_m["mAP50_tiny"] - det_poff_m["mAP50_tiny"],
        "patch_delta_mAP50_small": det_full_m["mAP50_small"] - det_poff_m["mAP50_small"],
        "film_delta_mAP50": det_poff_m["mAP50"] - det_boff_m["mAP50"],
        "total_delta_mAP50": det_full_m["mAP50"] - det_boff_m["mAP50"],
    }

    # Build component progression table
    comp_rows: List[Dict[str, object]] = []
    for cname in comp_names:
        row: Dict[str, object] = {"component": cname}
        for mk, vals in comp_acc[cname].items():
            row[mk] = round(float(np.mean(vals)), 4) if vals else np.nan
        comp_rows.append(row)

    neck_p3_fish = float(np.mean(comp_acc["neck.P3 (//8)"]["full_fisher"])) if comp_acc["neck.P3 (//8)"]["full_fisher"] else 0.0
    box_p3_fish = float(np.mean(comp_acc["patch.box_P3 (Deform+HR)"]["full_fisher"])) if comp_acc["patch.box_P3 (Deform+HR)"]["full_fisher"] else 0.0
    slot_p3_fish = float(np.mean(comp_acc["slot.cls_P3 (FiLM)"]["full_fisher"])) if comp_acc["slot.cls_P3 (FiLM)"]["full_fisher"] else 0.0
    cls_p3_fish = float(np.mean(comp_acc["patch.cls_P3 (Deform+HR)"]["full_fisher"])) if comp_acc["patch.cls_P3 (Deform+HR)"]["full_fisher"] else 0.0

    ig_val = float(torch.tanh(model.slot_refiner.iter_gate).item()) if getattr(model.slot_refiner, "iter_gate", None) is not None else 0.0

    patch_summary = {
        "patch_fg_hit_rate_pct": 100.0 * patch_totals["patch_fg_hits"] / max(patch_totals["patch_total"], 1.0),
        "patch_tiny_gt_coverage_pct": 100.0 * patch_totals["gt_tiny_cov"] / max(patch_totals["gt_tiny_tot"], 1.0),
        "patch_small_gt_coverage_pct": 100.0 * patch_totals["gt_small_cov"] / max(patch_totals["gt_small_tot"], 1.0),
        "patch_border_gt_coverage_pct": 100.0 * patch_totals["gt_border_cov"] / max(patch_totals["gt_border_tot"], 1.0),
        "offset_mean_cells": float(np.mean(patch_stats_acc["offset_mean_cells"])) if patch_stats_acc["offset_mean_cells"] else 0.0,
        "offset_max_cells": float(np.mean(patch_stats_acc["offset_max_cells"])) if patch_stats_acc["offset_max_cells"] else 0.0,
        "scale_mean_factor": float(np.mean(patch_stats_acc["scale_mean_factor"])) if patch_stats_acc["scale_mean_factor"] else 1.0,
        "scale_std_factor": float(np.mean(patch_stats_acc["scale_std_factor"])) if patch_stats_acc["scale_std_factor"] else 0.0,
        "p3_box_delta_ratio": float(np.mean(patch_stats_acc["p3_box_delta_ratio"])) if patch_stats_acc["p3_box_delta_ratio"] else 0.0,
        "p3_cls_delta_ratio": float(np.mean(patch_stats_acc["p3_cls_delta_ratio"])) if patch_stats_acc["p3_cls_delta_ratio"] else 0.0,
        "p3_box_fisher_delta": box_p3_fish - neck_p3_fish,
        "p3_cls_fisher_delta": cls_p3_fish - slot_p3_fish,
        "mufasa_iter_gate": ig_val,
    }

    # 8. Class-Agnostic mAP, IoU-sweep Recalls, Matched Confidence, TAL Starvation, & Linear Probe
    preds_agn = [
        (np.column_stack([p[:, :5], np.zeros((len(p), 1), dtype=np.float32)]) if len(p) else p)
        for p in preds_full
    ]
    gts_agn = [
        (np.column_stack([np.zeros((len(g), 1), dtype=np.float32), g[:, 1:5]]) if len(g) else g)
        for g in gts_all
    ]
    agn_ov, _ = compute_coco_map(preds_agn, gts_agn)
    agn_tn, _ = compute_coco_map(preds_agn, gts_agn, area_range=(0.0, 16.0 ** 2))
    agn_sm, _ = compute_coco_map(preds_agn, gts_agn, area_range=(0.0, 32.0 ** 2))

    bn_summary, bn_cls_rows = _compute_bottleneck_diagnostics(preds_full, gts_all)
    bn_summary["class_agnostic_mAP50"] = float(agn_ov.get("mAP50", 0.0))
    bn_summary["class_agnostic_mAP50_95"] = float(agn_ov.get("mAP50_95", 0.0))
    bn_summary["class_agnostic_mAP50_tiny"] = float(agn_tn.get("mAP50", 0.0))
    bn_summary["class_agnostic_mAP50_small"] = float(agn_sm.get("mAP50", 0.0))
    bn_summary["tal_all_in_box_pct"] = 100.0 * tal_stats["all_in_box"] / max(tal_stats["all_tot"], 1)
    bn_summary["tal_all_pos_pct"] = 100.0 * tal_stats["all_pos"] / max(tal_stats["all_tot"], 1)
    bn_summary["tal_tiny_in_box_pct"] = 100.0 * tal_stats["tiny_in_box"] / max(tal_stats["tiny_tot"], 1)
    bn_summary["tal_tiny_pos_pct"] = 100.0 * tal_stats["tiny_pos"] / max(tal_stats["tiny_tot"], 1)
    bn_summary["tal_small_in_box_pct"] = 100.0 * tal_stats["small_in_box"] / max(tal_stats["small_tot"], 1)
    bn_summary["tal_small_pos_pct"] = 100.0 * tal_stats["small_pos"] / max(tal_stats["small_tot"], 1)

    for c_row in bn_cls_rows:
        cid = int(c_row["class_id"])
        ct = max(int(tal_stats["cls_tot"][cid]), 1)
        c_row["tal_pos_pct"] = round(100.0 * float(tal_stats["cls_pos"][cid]) / ct, 2)
        c_row["tal_mean_anchors"] = round(float(tal_stats["cls_anchors"][cid]) / ct, 2)

    probe_rows: List[Dict[str, Any]] = []
    if probe_labels:
        y_all = torch.cat(probe_labels, dim=0)
        sm_all = torch.cat(probe_small, dim=0)
        tn_all = torch.cat(probe_tiny, dim=0)
        for lname in probe_layers:
            if probe_feats[lname]:
                x_all = torch.cat(probe_feats[lname], dim=0)
                pr = _run_linear_probe(x_all, y_all, sm_all, tn_all, device=device)
                probe_rows.append({"layer": lname, "num_samples": int(len(y_all)), **pr})

    # Save CSVs
    def _save_dict_csv(path: str, d: Dict[str, float]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(d.keys()))
            w.writeheader()
            w.writerow({k: round(float(v), 5) for k, v in d.items()})

    _save_dict_csv(os.path.join(args.out_dir, "detection_overall_metrics.csv"), det_full_m)
    _save_dict_csv(os.path.join(args.out_dir, "dual_ablation_metrics.csv"), ablation_summary)
    _save_dict_csv(os.path.join(args.out_dir, "deformable_patch_diagnostics.csv"), patch_summary)
    _save_dict_csv(os.path.join(args.out_dir, "bottleneck_summary.csv"), bn_summary)
    with open(os.path.join(args.out_dir, "bottleneck_per_class.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bn_cls_rows[0].keys()))
        w.writeheader()
        w.writerows(bn_cls_rows)
    if probe_rows:
        with open(os.path.join(args.out_dir, "linear_probe.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
            w.writeheader()
            w.writerows(probe_rows)
    with open(os.path.join(args.out_dir, "component_breakdown.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)

    cls_rows = [
        {
            "class_id": c,
            "class_name": VISDRONE_CLASSES[c],
            "AP50": round(ap50_full.get(c, 0.0), 5),
            "small_gt": cls_sm_full[c]["small_gt"],
            "small_rec_c25": round(cls_sm_full[c]["small_rec_c25"], 5) if np.isfinite(cls_sm_full[c]["small_rec_c25"]) else "",
        }
        for c in range(10)
    ]
    with open(os.path.join(args.out_dir, "per_class_ap.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cls_rows[0].keys()))
        w.writeheader()
        w.writerows(cls_rows)

    # Save Charts & HTML Dashboard
    if vis_sample is not None:
        _plot_deformable_patches(vis_sample, os.path.join(args.out_dir, "chart_deformable_patches.png"))
    _plot_slot_confusion(slot_conf_mat, os.path.join(args.out_dir, "chart_slot_confusion_matrix.png"))
    _write_dashboard_html(args.out_dir, det_full_m, ablation_summary, patch_summary, comp_rows)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"DEFORMABLE-SLOT VALIDATION COMPLETE ({total_imgs} imgs, {elapsed:.1f}s)")
    print("=" * 78)
    print(f"  mAP50={det_full_m['mAP50']:.4f} | mAP50:95={det_full_m['mAP50_95']:.4f} | Tiny={det_full_m['mAP50_tiny']:.4f} | Small={det_full_m['mAP50_small']:.4f}")
    print(f"  Class-Agnostic mAP50={bn_summary['class_agnostic_mAP50']:.4f} | Tiny={bn_summary['class_agnostic_mAP50_tiny']:.4f} | Small={bn_summary['class_agnostic_mAP50_small']:.4f}")
    print(f"  TAL GT Pos%: All={bn_summary['tal_all_pos_pct']:.1f}% (in_box={bn_summary['tal_all_in_box_pct']:.1f}%) | Tiny={bn_summary['tal_tiny_pos_pct']:.1f}% (in_box={bn_summary['tal_tiny_in_box_pct']:.1f}%) | Small={bn_summary['tal_small_pos_pct']:.1f}% (in_box={bn_summary['tal_small_in_box_pct']:.1f}%)")
    print(f"  Ablation: patch_delta={ablation_summary['patch_delta_mAP50']:+.4f} | film_delta={ablation_summary['film_delta_mAP50']:+.4f} | total_delta={ablation_summary['total_delta_mAP50']:+.4f}")
    print(
        f"  Patch Coverage: FG_hit={patch_summary['patch_fg_hit_rate_pct']:.1f}% | "
        f"Tiny_cov={patch_summary['patch_tiny_gt_coverage_pct']:.1f}% | "
        f"Small_cov={patch_summary['patch_small_gt_coverage_pct']:.1f}% | "
        f"Border_cov={patch_summary['patch_border_gt_coverage_pct']:.1f}%"
    )
    print(f"  Deformable: offset={patch_summary['offset_mean_cells']:.3f} cells | scale={patch_summary['scale_mean_factor']:.3f} | mufasa_gate={patch_summary['mufasa_iter_gate']:.4f}")
    print(f"[+] Outputs saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
