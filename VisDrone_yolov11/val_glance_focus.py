#!/usr/bin/env python3
"""Evaluate and Visualize Slot-Guided Glance-and-Focus (SG-GF) vs 1-Pass Glance on VisDrone Val.

Example:
  python VisDrone_yolov11/val_glance_focus.py --config VisDrone_yolov11/config_epoch70_sggf.yaml
  python VisDrone_yolov11/val_glance_focus.py --config VisDrone_yolov11/config_epoch70_sggf.yaml --max-images 50 --save-vis 4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from VisDrone_yolov11.dataset import VISDRONE_CLASSES, VisDroneLMDBDataset
from VisDrone_yolov11.slot_glance_focus import SlotGuidedGlanceFocus, load_slot_model_compat
from yolo.val_coco_components import compute_coco_map


def _compute_detailed_recalls(
    preds_list: List[np.ndarray],
    gts_list: List[np.ndarray],
    img_sizes: List[Tuple[int, int]],
    conf_thr: float = 0.25,
    iou_thr: float = 0.50,
    border_frac: float = 0.10,
    num_classes: int = 10,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Compute Precision/Recall/F1 @conf_thr plus stratified Recall for Tiny (<16^2), Small (<32^2),
    Medium (32^2..96^2), Large (>96^2), Border (within 10% of image edge), and per-class Small Recall.
    """
    tp_all = fp_all = fn_all = 0
    size_tp = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
    size_gt = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
    border_tp = border_gt = interior_tp = interior_gt = 0
    cls_small_tp = np.zeros(num_classes, dtype=np.int64)
    cls_small_gt = np.zeros(num_classes, dtype=np.int64)

    for pred, gt, (w, h) in zip(preds_list, gts_list, img_sizes):
        p = pred[pred[:, 4] >= conf_thr] if len(pred) > 0 else np.empty((0, 6), dtype=np.float32)
        n_gt = len(gt)
        gt_matched = np.zeros(n_gt, dtype=bool)

        if len(p) > 0 and n_gt > 0:
            order = np.argsort(-p[:, 4])
            p = p[order]
            gt_cls = gt[:, 0].astype(np.int64)
            gt_b = gt[:, 1:5]
            gt_area = np.maximum(0.0, gt_b[:, 2] - gt_b[:, 0]) * np.maximum(0.0, gt_b[:, 3] - gt_b[:, 1])

            for row in p:
                px1, py1, px2, py2, _, pc = row
                pc_i = int(pc)
                same_c = np.flatnonzero((gt_cls == pc_i) & (~gt_matched))
                if len(same_c) == 0:
                    fp_all += 1
                    continue
                gb = gt_b[same_c]
                ix1 = np.maximum(gb[:, 0], px1)
                iy1 = np.maximum(gb[:, 1], py1)
                ix2 = np.minimum(gb[:, 2], px2)
                iy2 = np.minimum(gb[:, 3], py2)
                inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                p_area = max(0.0, float((px2 - px1) * (py2 - py1)))
                ious = inter / np.maximum(gt_area[same_c] + p_area - inter, 1e-7)
                best_k = int(np.argmax(ious))
                if ious[best_k] >= iou_thr:
                    gt_matched[same_c[best_k]] = True
                    tp_all += 1
                else:
                    fp_all += 1
        elif len(p) > 0:
            fp_all += len(p)

        fn_all += int((~gt_matched).sum())

        if n_gt > 0:
            bw = np.maximum(0.0, gt[:, 3] - gt[:, 1])
            bh = np.maximum(0.0, gt[:, 4] - gt[:, 2])
            areas = bw * bh
            cx = 0.5 * (gt[:, 1] + gt[:, 3])
            cy = 0.5 * (gt[:, 2] + gt[:, 4])
            is_border = (
                (cx < border_frac * w)
                | (cx > (1.0 - border_frac) * w)
                | (cy < border_frac * h)
                | (cy > (1.0 - border_frac) * h)
            )
            border_gt += int(is_border.sum())
            border_tp += int((is_border & gt_matched).sum())
            interior_gt += int((~is_border).sum())
            interior_tp += int(((~is_border) & gt_matched).sum())

            m_tiny = areas < (16.0 ** 2)
            m_small = areas < (32.0 ** 2)
            m_med = (areas >= (32.0 ** 2)) & (areas < (96.0 ** 2))
            m_large = areas >= (96.0 ** 2)

            for key, mask in (("tiny", m_tiny), ("small", m_small), ("medium", m_med), ("large", m_large)):
                size_gt[key] += int(mask.sum())
                size_tp[key] += int((mask & gt_matched).sum())

            g_cls = gt[:, 0].astype(np.int64)
            for c in range(num_classes):
                mc = (g_cls == c) & m_small
                if mc.any():
                    cls_small_gt[c] += int(mc.sum())
                    cls_small_tp[c] += int((mc & gt_matched).sum())

    prec = tp_all / max(tp_all + fp_all, 1)
    rec = tp_all / max(tp_all + fn_all, 1)
    f1 = (2.0 * prec * rec) / max(prec + rec, 1e-8)

    summary = {
        "precision_c25": float(prec),
        "recall_c25": float(rec),
        "f1_c25": float(f1),
        "rec_tiny_c25": float(size_tp["tiny"] / max(size_gt["tiny"], 1)),
        "rec_small_c25": float(size_tp["small"] / max(size_gt["small"], 1)),
        "rec_medium_c25": float(size_tp["medium"] / max(size_gt["medium"], 1)),
        "rec_large_c25": float(size_tp["large"] / max(size_gt["large"], 1)),
        "rec_border_c25": float(border_tp / max(border_gt, 1)),
        "rec_interior_c25": float(interior_tp / max(interior_gt, 1)),
        "gt_tiny_cnt": float(size_gt["tiny"]),
        "gt_small_cnt": float(size_gt["small"]),
        "gt_border_cnt": float(border_gt),
    }
    per_cls_small = {
        c: {
            "small_gt": int(cls_small_gt[c]),
            "small_rec_c25": float(cls_small_tp[c] / max(cls_small_gt[c], 1)) if cls_small_gt[c] > 0 else float("nan"),
        }
        for c in range(num_classes)
    }
    return summary, per_cls_small


def _gt_cxcywh_to_xyxy_np(boxes: torch.Tensor, labels: torch.Tensor, w: int, h: int, is_cxcywh: bool = True) -> np.ndarray:
    """Convert normalized GT boxes to [M, 5] float32 array [cls, x1, y1, x2, y2] in original pixels."""
    if len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)
    b = boxes.float().cpu().numpy()
    l = labels.long().cpu().numpy().astype(np.float32)
    if (b <= 1.05).all():
        if is_cxcywh:
            cx, cy, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
            x1, y1, x2, y2 = cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh
        else:
            x1, y1 = b[:, 0] * w, b[:, 1] * h
            x2, y2 = x1 + b[:, 2] * w, y1 + b[:, 3] * h
    else:
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = np.clip(x1, 0.0, float(w))
    y1 = np.clip(y1, 0.0, float(h))
    x2 = np.clip(x2, 0.0, float(w))
    y2 = np.clip(y2, 0.0, float(h))
    return np.stack([l, x1, y1, x2, y2], axis=-1).astype(np.float32)


def _save_sggf_visual(
    orig_np: np.ndarray,
    gt_arr: np.ndarray,
    res: Dict[str, Any],
    save_path: str,
    vis_conf: float = 0.25,
) -> None:
    h, w = orig_np.shape[:2]
    dbg = res["debug"]
    fmap = dbg["focus_map"]
    fmap_norm = fmap / max(float(fmap.max()), 1e-6)
    fmap_up = cv2.resize(fmap_norm, (w, h), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: M_focus heatmap + chosen Focus crop rectangles
    axes[0].imshow(orig_np, alpha=0.55)
    axes[0].imshow(fmap_up, cmap="jet", alpha=0.50, vmin=0.0, vmax=1.0)
    for k, (cx1, cy1, cx2, cy2) in enumerate(dbg["crops"]):
        axes[0].add_patch(
            patches.Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1, linewidth=2.5, edgecolor="#FF2222", facecolor="none")
        )
        axes[0].text(cx1 + 6, max(16, cy1 + 18), f"Focus #{k + 1}", color="#FFFFFF", fontsize=10, fontweight="bold",
                     bbox=dict(facecolor="#FF2222", alpha=0.85, pad=2, edgecolor="none"))
    axes[0].set_title(f"Slot M_focus + {len(dbg['crops'])} Focus Crops (GT={len(gt_arr)})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: 1-Pass Glance detections
    gb, gs, gl = dbg["glance_boxes"], dbg["glance_scores"], dbg["glance_labels"]
    gm = gs >= vis_conf
    axes[1].imshow(orig_np)
    for b in gb[gm]:
        axes[1].add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1.3, edgecolor="#FFAA00", facecolor="none"))
    axes[1].set_title(f"1-Pass Glance ({int(gm.sum())} dets, conf>={vis_conf})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: SG-GF detections
    fb, fs, fl = res["boxes"], res["scores"], res["labels"]
    fm = fs >= vis_conf
    axes[2].imshow(orig_np)
    for b in fb[fm]:
        axes[2].add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1.3, edgecolor="#00FF66", facecolor="none"))
    axes[2].set_title(f"Slot-Guided Glance-and-Focus ({int(fm.sum())} dets, conf>={vis_conf})", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Slot-Guided Glance-and-Focus vs 1-Pass Glance.")
    parser.add_argument("--config", type=str, default="VisDrone_yolov11/config_epoch70_sggf.yaml", help="YAML config path")
    parser.add_argument("--weights", type=str, default=None, help="Override weights path from config")
    parser.add_argument("--out-dir", type=str, default="VisDrone_yolov11/output_sggf_epoch70", help="Output directory")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of val images (default: all 548)")
    parser.add_argument("--save-vis", type=int, default=4, help="Number of visual comparison PNGs to save")
    parser.add_argument("--vis-conf", type=float, default=0.25, help="Confidence threshold for visualization")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.weights or cfg.get("weights")
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = load_slot_model_compat(weights_path=weights_path, model_cfg=cfg["model"], device=device)

    gf_cfg = cfg.get("glance_focus", {})
    engine = SlotGuidedGlanceFocus(model=model, device=device, **gf_cfg)

    val_ds_cfg = cfg["data"]["val"]["dataset"]
    val_ds = VisDroneLMDBDataset(
        data_root=val_ds_cfg["data_root"],
        num_classes=int(val_ds_cfg.get("num_classes", 10)),
        include_boxes=True,
        cache_images=None,
        augment=False,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    total_imgs = len(val_ds) if args.max_images is None else min(args.max_images, len(val_ds))
    print(f"[SG-GF Eval] Evaluating {total_imgs}/{len(val_ds)} original-resolution val images on {device}...")

    preds_glance: List[np.ndarray] = []
    preds_sggf: List[np.ndarray] = []
    gts_all: List[np.ndarray] = []
    img_sizes: List[Tuple[int, int]] = []
    crop_counts: List[int] = []
    t_glance_list: List[float] = []
    t_total_list: List[float] = []

    t_start = time.perf_counter()
    for idx in range(total_imgs):
        sample = val_ds.get_data_info(idx)
        orig_img = sample["image"]
        orig_w, orig_h = orig_img.size
        img_sizes.append((orig_w, orig_h))
        target = sample["target"]

        gt_arr = _gt_cxcywh_to_xyxy_np(
            target.get("boxes", torch.empty((0, 4))),
            target.get("labels", torch.empty((0,), dtype=torch.long)),
            orig_w,
            orig_h,
            is_cxcywh=bool(target.get("_is_cxcywh", True)),
        )
        gts_all.append(gt_arr)

        res = engine.predict_image(orig_img, return_debug=True)
        dbg = res["debug"]
        crop_counts.append(int(dbg["num_crops"]))
        t_glance_list.append(float(dbg["t_glance_ms"]))
        t_total_list.append(float(dbg["t_glance_ms"] + dbg["t_focus_ms"]))

        # Pack predictions as [N, 6]: [x1, y1, x2, y2, score, cls] for compute_coco_map
        if len(dbg["glance_boxes"]) > 0:
            g_pred = np.concatenate(
                [dbg["glance_boxes"], dbg["glance_scores"][:, None], dbg["glance_labels"][:, None].astype(np.float32)],
                axis=1,
            )
        else:
            g_pred = np.empty((0, 6), dtype=np.float32)
        preds_glance.append(g_pred)

        if len(res["boxes"]) > 0:
            s_pred = np.concatenate(
                [res["boxes"], res["scores"][:, None], res["labels"][:, None].astype(np.float32)],
                axis=1,
            )
        else:
            s_pred = np.empty((0, 6), dtype=np.float32)
        preds_sggf.append(s_pred)

        if idx < args.save_vis:
            vis_path = os.path.join(args.out_dir, f"sggf_vis_{idx:02d}.png")
            _save_sggf_visual(np.array(orig_img, dtype=np.uint8), gt_arr, res, vis_path, vis_conf=args.vis_conf)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_imgs:
            print(
                f"  [{idx + 1:3d}/{total_imgs}] avg_crops={np.mean(crop_counts):.2f} | "
                f"glance={np.mean(t_glance_list):.1f}ms | sggf={np.mean(t_total_list):.1f}ms"
            )

    elapsed = time.perf_counter() - t_start

    def _eval_bundle(preds_list: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[int, float], Dict[int, Dict[str, float]]]:
        ov, rows = compute_coco_map(preds_list, gts_all)
        tn, _ = compute_coco_map(preds_list, gts_all, area_range=(0.0, 16.0 ** 2))
        sm, _ = compute_coco_map(preds_list, gts_all, area_range=(0.0, 32.0 ** 2))
        md, _ = compute_coco_map(preds_list, gts_all, area_range=(32.0 ** 2, 96.0 ** 2))
        lg, _ = compute_coco_map(preds_list, gts_all, area_range=(96.0 ** 2, 1e9))
        rec_det, cls_small = _compute_detailed_recalls(
            preds_list, gts_all, img_sizes, conf_thr=args.vis_conf, iou_thr=0.50
        )
        metrics = {
            "mAP50": float(ov.get("mAP50", 0.0)),
            "mAP50_95": float(ov.get("mAP50_95", 0.0)),
            "mAP50_tiny": float(tn.get("mAP50", 0.0)),
            "mAP50_95_tiny": float(tn.get("mAP50_95", 0.0)),
            "mAP50_small": float(sm.get("mAP50", 0.0)),
            "mAP50_95_small": float(sm.get("mAP50_95", 0.0)),
            "mAP50_95_medium": float(md.get("mAP50_95", 0.0)),
            "mAP50_95_large": float(lg.get("mAP50_95", 0.0)),
            **rec_det,
        }
        ap50_by_cls = {int(r["class_id"]): float(r.get("AP50", r.get("ap50", 0.0))) for r in rows}
        return metrics, ap50_by_cls, cls_small

    m_glance, per_cls_g, cls_sm_g = _eval_bundle(preds_glance)
    m_sggf, per_cls_s, cls_sm_s = _eval_bundle(preds_sggf)

    # Approximate GFLOPs based on measured scaling (960=15.68, 640=6.89, 512=4.40)
    g_gflops = 15.68 * (engine.glance_size / 960.0) ** 2
    f_gflops = 6.89 * (engine.focus_size / 640.0) ** 2
    avg_crops = float(np.mean(crop_counts))
    zero_crop_pct = float(np.mean([c == 0 for c in crop_counts]) * 100.0)
    sggf_gflops = g_gflops + avg_crops * f_gflops

    metric_order = (
        "mAP50", "mAP50_95",
        "mAP50_tiny", "mAP50_95_tiny",
        "mAP50_small", "mAP50_95_small",
        "mAP50_95_medium", "mAP50_95_large",
        "precision_c25", "recall_c25", "f1_c25",
        "rec_tiny_c25", "rec_small_c25", "rec_medium_c25", "rec_large_c25",
        "rec_border_c25", "rec_interior_c25",
    )

    print("\n" + "=" * 82)
    print(f"SLOT-GUIDED GLANCE-AND-FOCUS EVALUATION ({total_imgs} images, {elapsed:.1f}s)")
    print(
        f"GT Counts: tiny(<16^2)={int(m_glance['gt_tiny_cnt'])}, "
        f"small(<32^2)={int(m_glance['gt_small_cnt'])}, "
        f"border(10% edge)={int(m_glance['gt_border_cnt'])}"
    )
    print("=" * 82)
    print(f"{'Metric':<22} | {'1-Pass Glance':>14} | {'SG-GF (WBF)':>14} | {'Delta':>10}")
    print("-" * 82)
    for k in metric_order:
        vg = float(m_glance.get(k, 0.0))
        vs = float(m_sggf.get(k, 0.0))
        print(f"{k:<22} | {vg:14.4f} | {vs:14.4f} | {vs - vg:+10.4f}")
    print("-" * 82)
    print(f"{'Avg Crops / Image':<22} | {0.0:14.2f} | {avg_crops:14.2f} | {'0-crop ' + f'{zero_crop_pct:.1f}%':>10}")
    print(f"{'Est. GFLOPs / Image':<22} | {g_gflops:14.2f} | {sggf_gflops:14.2f} | {sggf_gflops / max(g_gflops, 1e-6):9.2f}x")
    print(f"{'Latency (ms/img)':<22} | {float(np.mean(t_glance_list)):14.1f} | {float(np.mean(t_total_list)):14.1f} | {float(np.mean(t_total_list) - np.mean(t_glance_list)):+10.1f}")
    print("=" * 82)

    # Save summary CSV
    summary_rows = [
        {
            "mode": "1-Pass Glance",
            **{k: round(float(m_glance.get(k, 0.0)), 5) for k in metric_order},
            "avg_crops": 0.0,
            "zero_crop_pct": 100.0,
            "est_gflops": round(g_gflops, 2),
            "latency_ms": round(float(np.mean(t_glance_list)), 2),
        },
        {
            "mode": "Slot-Guided Glance-and-Focus",
            **{k: round(float(m_sggf.get(k, 0.0)), 5) for k in metric_order},
            "avg_crops": round(avg_crops, 2),
            "zero_crop_pct": round(zero_crop_pct, 2),
            "est_gflops": round(sggf_gflops, 2),
            "latency_ms": round(float(np.mean(t_total_list)), 2),
        },
    ]
    with open(os.path.join(args.out_dir, "sggf_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Save per-class AP50 + Small Recall CSV
    cls_rows = []
    print(f"\n{'Class':<18} | {'Gl AP50':>8} | {'GF AP50':>8} | {'dAP50':>8} | {'SmGT':>5} | {'Gl SmRec':>8} | {'GF SmRec':>8} | {'dSmRec':>8}")
    print("-" * 88)
    for c in range(10):
        ag = float(per_cls_g.get(c, 0.0))
        as_ = float(per_cls_s.get(c, 0.0))
        sm_gt = cls_sm_g[c]["small_gt"]
        rg = cls_sm_g[c]["small_rec_c25"]
        rs = cls_sm_s[c]["small_rec_c25"]
        rg_s = f"{rg:8.4f}" if np.isfinite(rg) else "     nan"
        rs_s = f"{rs:8.4f}" if np.isfinite(rs) else "     nan"
        dr_s = f"{rs - rg:+8.4f}" if (np.isfinite(rg) and np.isfinite(rs)) else "     nan"
        print(f"{VISDRONE_CLASSES[c]:<18} | {ag:8.4f} | {as_:8.4f} | {as_ - ag:+8.4f} | {sm_gt:5d} | {rg_s} | {rs_s} | {dr_s}")
        cls_rows.append({
            "class_id": c,
            "class_name": VISDRONE_CLASSES[c],
            "glance_ap50": round(ag, 5),
            "sggf_ap50": round(as_, 5),
            "delta_ap50": round(as_ - ag, 5),
            "small_gt": sm_gt,
            "glance_small_rec_c25": round(rg, 5) if np.isfinite(rg) else "",
            "sggf_small_rec_c25": round(rs, 5) if np.isfinite(rs) else "",
            "delta_small_rec_c25": round(rs - rg, 5) if (np.isfinite(rg) and np.isfinite(rs)) else "",
        })
    with open(os.path.join(args.out_dir, "sggf_per_class_ap.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cls_rows[0].keys()))
        w.writeheader()
        w.writerows(cls_rows)

    print(f"\n[+] Saved results & visualizations to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
