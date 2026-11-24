#!/usr/bin/env python3
"""Component-by-Component Validation & Feature Inspection of VisDrone YOLOv11 / VisDroneYOLO11Slot.

Adapts `yolo/val_coco_components.py` for VisDroneLMDBDataset (10 VisDrone classes) and
supports both `VisDroneYOLO11` and `VisDroneYOLO11Slot` (Backbone, Neck, Slot+FiLM, Head, End-to-End mAP).

Example:
  python VisDrone_yolov11/val_visdrone_components.py --weights "D:/Downloads/Picture/epoch_50.pth" --imgsz 960
  python VisDrone_yolov11/val_visdrone_components.py --weights "D:/Downloads/Picture/epoch_50.pth" --imgsz 960 --max-images 200
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as tv_ops
import torchvision.transforms.functional as TF
import yaml

from VisDrone_yolov11.dataset import VISDRONE_CLASSES, VisDroneLMDBDataset
from VisDrone_yolov11.model import VisDroneYOLO11, VisDroneYOLO11Slot
from yolo.head import dist2bbox, make_anchors
from yolo.val_coco_components import (
    COMPONENT_ORDER as BASE_COMPONENT_ORDER,
    YOLOv11ComponentProbe,
    _compute_pca_rgb_gpu,
    _plot_backbone_vs_neck,
    _plot_component_progression,
    _plot_detection_and_head,
    build_masks_at_res_gpu,
    compute_coco_map,
    measure_feature_metrics,
)


def load_visdrone_model(
    weights_path: str,
    device: torch.device,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.nn.Module, bool]:
    """Auto-detect VisDroneYOLO11Slot vs VisDroneYOLO11 from checkpoint (or YAML model_cfg) and load weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    if weights_path.endswith(".pt"):
        model = VisDroneYOLO11(num_classes=10, pretrained=weights_path)
        return model.to(device).eval(), False

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        sd = ckpt
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    has_slots = any(k.startswith("slot_refiner.") for k in sd.keys())
    if model_cfg is not None:
        cfg_kwargs = {k: v for k, v in model_cfg.items() if k not in ("type", "pretrained")}
        cfg_kwargs["pretrained"] = None
        if has_slots:
            cfg_kwargs.setdefault("use_logit_prior", "slot_refiner.logit_prior_scale" in sd)
            model = VisDroneYOLO11Slot(**cfg_kwargs)
        else:
            base_keys = {
                "num_classes", "pretrained", "box_weight", "cls_weight", "dfl_weight",
                "focal_alpha", "focal_gamma", "conf_threshold", "nms_threshold", "max_det",
            }
            model = VisDroneYOLO11(**{k: v for k, v in cfg_kwargs.items() if k in base_keys})
    else:
        if has_slots:
            has_logit_prior = "slot_refiner.logit_prior_scale" in sd
            model = VisDroneYOLO11Slot(num_classes=10, use_logit_prior=has_logit_prior, assigner_prior_weight=0.0)
        else:
            model = VisDroneYOLO11(num_classes=10)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"[Load] {'VisDroneYOLO11Slot' if has_slots else 'VisDroneYOLO11'} from {weights_path} "
        f"(missing: {len(missing)}, unexpected: {len(unexpected)})"
    )
    return model.to(device).eval(), has_slots


def _plot_visdrone_visual_inspection(vis_sample: Dict[str, Any], save_path: str) -> None:
    lb_rgb = vis_sample["lb_rgb"]
    H, W = lb_rgb.shape[:2]
    gt_xyxy = vis_sample["gt_xyxy"]
    det = vis_sample["det"]

    fig, axes = plt.subplots(3, 3, figsize=(17, 16))

    axes[0, 0].imshow(lb_rgb)
    for g in gt_xyxy:
        c, x1, y1, x2, y2 = int(g[0]), g[1], g[2], g[3], g[4]
        axes[0, 0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor="#00FF66", facecolor="none"))
        if c < len(VISDRONE_CLASSES):
            axes[0, 0].text(x1, max(0, y1 - 3), VISDRONE_CLASSES[c], color="#00FF66", fontsize=6.5, fontweight="bold")
    axes[0, 0].set_title(f"GT Boxes ({len(gt_xyxy)} objs) | {vis_sample['img_name']}", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(lb_rgb)
    for d in det:
        x1, y1, x2, y2, conf, c = d[0], d[1], d[2], d[3], d[4], int(d[5])
        axes[0, 1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor="#FFAA00", facecolor="none"))
        lbl = VISDRONE_CLASSES[c] if c < len(VISDRONE_CLASSES) else f"c{c}"
        axes[0, 1].text(x1, max(0, y1 - 3), f"{lbl} {conf:.2f}", color="#FFAA00", fontsize=6.5, fontweight="bold")
    axes[0, 1].set_title(f"Predictions ({len(det)} dets, conf>=vis_conf)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    cv3_up = cv2.resize(vis_sample["cv3_p3"], (W, H), interpolation=cv2.INTER_NEAREST)
    axes[0, 2].imshow(lb_rgb, alpha=0.55)
    axes[0, 2].imshow(cv3_up, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Head cv3 P3 Max Class Prob", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    for col, (fkey, lbl) in enumerate([
        ("feat4", "Backbone feat4 (//8)"),
        ("feat6", "Backbone feat6 (//16)"),
        ("feat10", "Backbone feat10 (//32)"),
    ]):
        l2 = torch.linalg.vector_norm(vis_sample[fkey], dim=0).cpu().numpy()
        l2_up = cv2.resize(l2, (W, H), interpolation=cv2.INTER_NEAREST)
        axes[1, col].imshow(l2_up, cmap="magma")
        for g in gt_xyxy:
            axes[1, col].add_patch(patches.Rectangle((g[1], g[2]), g[3] - g[1], g[4] - g[2], linewidth=0.8, edgecolor="#00FF66", facecolor="none"))
        axes[1, col].set_title(f"{lbl} L2-Norm", fontsize=11, fontweight="bold")
        axes[1, col].axis("off")

    for col, (pkey, lbl) in enumerate([
        ("p3", "Neck P3 (//8)"),
        ("p4", "Neck P4 (//16)"),
        ("p5", "Neck P5 (//32)"),
    ]):
        f_t = vis_sample[pkey]
        _, hf, wf = f_t.shape
        pca = _compute_pca_rgb_gpu(f_t).reshape(hf, wf, 3).cpu().numpy()
        pca_up = cv2.resize(pca, (W, H), interpolation=cv2.INTER_NEAREST)
        axes[2, col].imshow(pca_up)
        for g in gt_xyxy:
            axes[2, col].add_patch(patches.Rectangle((g[1], g[2]), g[3] - g[1], g[4] - g[2], linewidth=0.8, edgecolor="#FFFFFF", facecolor="none"))
        axes[2, col].set_title(f"{lbl} PCA-RGB", fontsize=11, fontweight="bold")
        axes[2, col].axis("off")

    plt.suptitle("VisDrone Component Dissection: Predictions, Head Confidence, Backbone & Neck", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_visdrone_pairwise_heatmap(
    valid_cls_ids: List[int],
    pw_mat: np.ndarray,
    save_path: str,
    title: str = "Neck P3 Pairwise Inter-Class Cosine Distance (VisDrone)",
) -> None:
    if len(valid_cls_ids) < 2:
        return
    names = [VISDRONE_CLASSES[c] for c in valid_cls_ids]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    vmax = max(0.5, float(np.nanmax(pw_mat)) if np.isfinite(pw_mat).any() else 1.0)
    im = ax.imshow(pw_mat, cmap="YlOrRd", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for r in range(len(names)):
        for c in range(len(names)):
            val = pw_mat[r, c]
            txt = "-" if r == c else (f"{val:.2f}" if np.isfinite(val) else "na")
            ax.text(c, r, txt, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine Distance (1 - cos_sim)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _generate_visdrone_dashboard(
    out_dir: str,
    comp_rows: List[Dict[str, object]],
    overall_det: Dict[str, float],
    slot_summary: Optional[Dict[str, float]],
    box_diag: Optional[Dict[str, float]] = None,
) -> None:
    comp_headers = "".join(f"<th>{html.escape(k)}</th>" for k in comp_rows[0].keys())
    comp_body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>"
        for r in comp_rows
    )

    slot_card = ""
    if slot_summary:
        slot_items = "".join(
            f"<div class='kpi'><div class='lbl'>{html.escape(k)}</div><div class='val'>{v:.4f}</div></div>"
            for k, v in slot_summary.items()
        )
        slot_card = f"<h2>Slot+FiLM Diagnostics</h2><div class='kpi-grid'>{slot_items}</div>"

    box_card = ""
    if box_diag:
        box_items = "".join(
            f"<div class='kpi'><div class='lbl'>{html.escape(k)}</div><div class='val'>{v:.4f}</div></div>"
            for k, v in box_diag.items()
        )
        box_card = f"<h2>Box Learning Mechanism Diagnostics (Boundary, TAL, DFL & Gradient Conflict)</h2><div class='kpi-grid'>{box_items}</div>"

    chart_files = [
        ("Component-by-Component Progression", "chart_01_component_progression.png"),
        ("Backbone vs Neck Multi-Scale Comparison", "chart_02_backbone_vs_neck_multiscale.png"),
        ("Head Scale Share & Detection AP Summary", "chart_03_head_and_detection_summary.png"),
        ("Visual Feature Inspection", "chart_04_feature_visual_inspection.png"),
        ("Pairwise Class Cosine Separation (Neck P3)", "chart_05_pairwise_class_separation.png"),
        ("Slot Attention Confusion Matrix (GT Class vs Predicted Slot)", "chart_06_slot_confusion_matrix.png"),
    ]
    charts_html = "".join(
        f"<div class='card'><h3>{html.escape(title)}</h3><img src='{fname}' style='max-width:100%;'></div>"
        for title, fname in chart_files
        if os.path.isfile(os.path.join(out_dir, fname))
    )

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VisDrone YOLOv11 Component Validation Dashboard</title>
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
  <h1>VisDrone YOLOv11 Component-by-Component Validation</h1>
  <div class="kpi-grid">
    <div class="kpi"><div class="lbl">mAP@0.50</div><div class="val">{overall_det['mAP50']:.4f}</div></div>
    <div class="kpi"><div class="lbl">mAP@0.50:0.95</div><div class="val">{overall_det['mAP50_95']:.4f}</div></div>
    <div class="kpi"><div class="lbl">mAP S / M / L</div><div class="val">{overall_det['mAP50_95_small']:.3f} / {overall_det['mAP50_95_medium']:.3f} / {overall_det['mAP50_95_large']:.3f}</div></div>
    <div class="kpi"><div class="lbl">Precision / Recall (0.25)</div><div class="val">{overall_det['precision_c25']:.3f} / {overall_det['recall_c25']:.3f}</div></div>
    <div class="kpi"><div class="lbl">Latency / FPS</div><div class="val">{overall_det['latency_ms']:.1f} ms ({overall_det['fps']:.1f})</div></div>
    <div class="kpi"><div class="lbl">Total Params</div><div class="val">{overall_det['params_total_m']:.2f}M</div></div>
  </div>
  {slot_card}
  {box_card}
  <div class="card">
    <h2>Component-by-Component Breakdown</h2>
    <table><thead><tr>{comp_headers}</tr></thead><tbody>{comp_body}</tbody></table>
  </div>
  {charts_html}
</body>
</html>"""
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html_str)


def _measure_box_spatial_metrics(
    feat: torch.Tensor,
    gt_lb_norm: np.ndarray,
    imgsz: int,
) -> Dict[str, float]:
    """Measure Boundary Contrast & Intra-Box Spatial Decorrelation on a feature map (1, C, H, W)."""
    f = feat[0].float()  # (C, H, W)
    c, h, w = f.shape
    device = f.device
    if len(gt_lb_norm) == 0 or h < 3 or w < 3:
        return {}

    fg_m, _, _, _, _ = build_masks_at_res_gpu(gt_lb_norm, h, w, imgsz, device)
    fg_2d = fg_m.view(1, 1, h, w).float()
    if fg_2d.sum() < 4:
        return {}

    # 1. Spatial gradient magnitude ||grad_{x,y} F||
    gx = torch.zeros_like(f)
    gy = torch.zeros_like(f)
    gx[:, :, :-1] = f[:, :, 1:] - f[:, :, :-1]
    gy[:, :-1, :] = f[:, 1:, :] - f[:, :-1, :]
    grad_mag = torch.sqrt(gx.pow(2).sum(dim=0) + gy.pow(2).sum(dim=0) + 1e-8)  # (H, W)

    # Border mask = dilate(fg) - erode(fg)
    dilated = F.max_pool2d(fg_2d, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - fg_2d, kernel_size=3, stride=1, padding=1)
    border_m = ((dilated - eroded) > 0.5).squeeze(0).squeeze(0)
    interior_m = (eroded > 0.5).squeeze(0).squeeze(0)
    if not interior_m.any():
        interior_m = (fg_2d > 0.5).squeeze(0).squeeze(0)

    out: Dict[str, float] = {}
    if border_m.any() and interior_m.any():
        out["boundary_contrast"] = float((grad_mag[border_m].mean() / (grad_mag[interior_m].mean() + 1e-8)).item())

    # 2. Intra-Box Spatial Decorrelation: 1 - cos_sim between adjacent cells inside FG
    fn = F.normalize(f, dim=0)
    fg_bool = fg_m.view(h, w) > 0.5
    pair_x = fg_bool[:, :-1] & fg_bool[:, 1:]
    pair_y = fg_bool[:-1, :] & fg_bool[1:, :]
    decorrs = []
    if pair_x.any():
        cos_x = (fn[:, :, :-1] * fn[:, :, 1:]).sum(dim=0)[pair_x]
        decorrs.append(float((1.0 - cos_x).mean().item()))
    if pair_y.any():
        cos_y = (fn[:, :-1, :] * fn[:, 1:, :]).sum(dim=0)[pair_y]
        decorrs.append(float((1.0 - cos_y).mean().item()))
    if decorrs:
        out["intra_box_decorr"] = float(np.mean(decorrs))

    return out


def _measure_tal_and_dfl_metrics(
    model: torch.nn.Module,
    feats: List[torch.Tensor],
    boxes_raw: torch.Tensor,
    scores_raw: torch.Tensor,
    gt_xyxy: np.ndarray,
    imgsz: int,
    device: torch.device,
) -> Dict[str, float]:
    """Measure GT Assignment Coverage, Avg Pos/GT, DFL Clipping Rate, and DFL Entropy."""
    if len(gt_xyxy) == 0:
        return {}

    reg_max = model.net.head.reg_max
    a_raw, s_raw = make_anchors(feats, model.net.head.stride, 0.5)
    anchors, strides = (a.transpose(0, 1).to(device) for a in (a_raw, s_raw))
    anchor_points = (anchors * strides).transpose(0, 1).float()  # (N, 2)
    stride_tensor = strides.transpose(0, 1).float()              # (N, 1)

    pred_dist = model.net.head.dfl(boxes_raw)
    decoded_xyxy = (dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides).transpose(1, 2).float()

    gt_boxes = torch.from_numpy(gt_xyxy[:, 1:5]).to(device=device, dtype=torch.float32)  # (M, 4)
    gt_labels = torch.from_numpy(gt_xyxy[:, 0]).to(device=device, dtype=torch.long)      # (M,)
    M = len(gt_boxes)

    # Check geometric `is_in_box` coverage (how many GTs have >=1 anchor center inside)
    lt = anchor_points.unsqueeze(0) - gt_boxes[:, :2].unsqueeze(1)  # (M, N, 2)
    rb = gt_boxes[:, 2:].unsqueeze(1) - anchor_points.unsqueeze(0)  # (M, N, 2)
    is_in_box = torch.cat([lt, rb], dim=-1).amin(dim=-1) > 0.0      # (M, N)
    covered = is_in_box.any(dim=1)                                   # (M,)

    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    small_m = areas < (32.0 ** 2)

    # Run actual TAL assigner
    target_bboxes, _, fg_mask = model.criterion.assigner(
        pred_scores=scores_raw.transpose(1, 2).float().sigmoid(),
        pred_bboxes=decoded_xyxy,
        anchor_points=anchor_points,
        gt_labels=gt_labels.view(1, M, 1),
        gt_bboxes=gt_boxes.view(1, M, 4),
        mask_gt=torch.ones((1, M, 1), dtype=torch.bool, device=device),
    )

    out: Dict[str, float] = {
        "gt_coverage_all_pct": float(covered.float().mean().item() * 100.0),
        "avg_pos_per_gt": float(fg_mask.sum().item()) / max(M, 1),
    }
    if small_m.any():
        out["gt_coverage_small_pct"] = float(covered[small_m].float().mean().item() * 100.0)

    if fg_mask.any():
        pos_ap = anchor_points.unsqueeze(0)[fg_mask]
        pos_st = stride_tensor.unsqueeze(0)[fg_mask]
        pos_tgt = target_bboxes[fg_mask]
        ltrb = torch.cat([pos_ap - pos_tgt[:, :2], pos_tgt[:, 2:] - pos_ap], dim=-1).clamp(min=0.0) / pos_st
        out["dfl_clipping_rate_pct"] = float((ltrb > (reg_max - 1.0)).float().mean().item() * 100.0)

        # DFL softmax distribution entropy & peak confidence on positive anchors
        pos_logits = boxes_raw.transpose(1, 2).float()[fg_mask].view(-1, 4, reg_max)
        probs = pos_logits.softmax(dim=-1).clamp(min=1e-7)
        entropy = -(probs * probs.log()).sum(dim=-1) / float(np.log(reg_max))  # normalized in [0, 1]
        out["dfl_entropy_pos"] = float(entropy.mean().item())
        out["dfl_top1_prob"] = float(probs.amax(dim=-1).mean().item())

    return out


def _measure_neck_gradient_conflict(
    model: torch.nn.Module,
    has_slots: bool,
    img_t: torch.Tensor,
    target_dict: Dict[str, torch.Tensor],
    imgsz: int,
    device: torch.device,
) -> Dict[str, float]:
    """Measure cos_sim(grad_box, grad_cls+slot) and ||grad_box||/||grad_cls+slot|| at neck P3/P4/P5."""
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}

    with torch.enable_grad():
        with torch.no_grad():
            feat4, feat6, feat10 = model.net.backbone(img_t)
            p3, p4, p5 = model.net.neck(feat4, feat6, feat10)

        p3 = p3.detach().requires_grad_(True)
        p4 = p4.detach().requires_grad_(True)
        p5 = p5.detach().requires_grad_(True)
        feats = [p3, p4, p5]

        if has_slots and hasattr(model, "slot_refiner"):
            cls_feats, slot_info = model.slot_refiner(feats)
        else:
            cls_feats, slot_info = feats, None

        bs = p3.shape[0]
        boxes_raw = torch.cat(
            [model.net.head.cv2[i](feats[i]).view(bs, 4 * model.net.head.reg_max, -1) for i in range(model.net.head.nl)],
            dim=-1,
        )
        scores_raw = torch.cat(
            [model.net.head.cv3[i](cls_feats[i]).view(bs, model.num_classes, -1) for i in range(model.net.head.nl)],
            dim=-1,
        )
        if slot_info is not None and slot_info.get("logit_prior") is not None:
            scores_raw = scores_raw + slot_info["logit_prior"]

        a_raw, s_raw = make_anchors(feats, model.net.head.stride, 0.5)
        anchors, strides = (a.transpose(0, 1).to(device) for a in (a_raw, s_raw))
        pred_dist = model.net.head.dfl(boxes_raw)
        decoded_xyxy = (dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides).transpose(1, 2)
        ap_px = (anchors * strides).transpose(0, 1)
        st_t = strides.transpose(0, 1)

        targets_list = [{
            "boxes": target_dict["boxes"].to(device),
            "labels": target_dict["labels"].to(device),
            "_is_cxcywh": True,
        }]
        det_losses = model.criterion(
            preds={"boxes": boxes_raw, "scores": scores_raw, "feats": feats},
            targets=targets_list,
            decoded_bboxes=decoded_xyxy,
            anchor_points=ap_px,
            stride_tensor=st_t,
            img_size=(imgsz, imgsz),
        )
        loss_box = det_losses["loss_diou"] + det_losses["loss_dfl"]
        loss_cls = det_losses["loss_focal"]

        if has_slots and slot_info is not None and model.slot_loss_weight > 0:
            gt_b, gt_l, m_gt = model.criterion._prepare_targets(targets_list, (imgsz, imgsz), device)
            sl = model.slot_criterion(
                slots=slot_info["slots"],
                attn_ori=slot_info["attn_ori"],
                obj_queries=model.slot_refiner.slot_attention.obj_queries,
                resolutions=slot_info["resolutions"],
                gt_bboxes=gt_b,
                gt_labels=gt_l,
                mask_gt=m_gt,
                img_size=(imgsz, imgsz),
            )
            loss_cls = loss_cls + model.slot_loss_weight * sl["loss_slot"]

        if loss_box.requires_grad and loss_cls.requires_grad:
            g_box = torch.autograd.grad(loss_box, feats, retain_graph=True, allow_unused=True)
            g_cls = torch.autograd.grad(loss_cls, feats, retain_graph=False, allow_unused=True)
            for name, gb, gc in zip(["p3", "p4", "p5"], g_box, g_cls):
                if gb is not None and gc is not None:
                    gb_f = gb.reshape(-1).float()
                    gc_f = gc.reshape(-1).float()
                    nb = float(gb_f.norm().item())
                    nc = float(gc_f.norm().item())
                    if nb > 1e-8 and nc > 1e-8:
                        out[f"grad_cos_box_vs_cls_{name}"] = float(F.cosine_similarity(gb_f.unsqueeze(0), gc_f.unsqueeze(0)).item())
                        out[f"grad_ratio_box_over_cls_{name}"] = nb / (nc + 1e-8)

    if was_training:
        model.train()
    return out


def _plot_slot_confusion_matrix(
    conf_mat: np.ndarray,
    save_path: str,
    title: str = "Slot Attention Confusion Matrix (Row-Normalized %: GT Class -> Pred Slot)",
) -> None:
    """Plot 10x11 row-normalized confusion matrix of GT classes vs predicted argmax slot (0..9 + BG)."""
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    pct = np.divide(conf_mat, np.maximum(row_sums, 1.0)) * 100.0
    row_names = list(VISDRONE_CLASSES)
    col_names = list(VISDRONE_CLASSES) + ["BG"]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(len(row_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_names, fontsize=9)
    ax.set_xlabel("Predicted Slot (argmax attn_ori)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Ground-Truth Class", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    for r in range(len(row_names)):
        for c in range(len(col_names)):
            val = pct[r, c]
            color = "white" if val >= 55.0 else "black"
            ax.text(c, r, f"{val:.1f}" if val >= 0.5 else "", ha="center", va="center", fontsize=7.5, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of GT Pixels")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _measure_slot_overfit_metrics(
    model: torch.nn.Module,
    slot_info: Dict[str, Any],
    gt_xyxy: np.ndarray,
    imgsz: int,
    device: torch.device,
    sa_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute per-image Slot sharpness, entropy, calibration, BG leak, inactive FA, confusion counts,
    plus 4 cls_attn bottleneck diagnostics (per-scale recall/BG leak, T=1 vs T=2 alignment, logit gaps).
    """
    num_classes = 10
    attn_ori = slot_info["attn_ori"][0].float()   # (11, N)
    dots = slot_info["dots"][0].float()           # (11, N)
    resolutions = slot_info["resolutions"]
    (h3, w3), (h4, w4), _ = resolutions
    n3, n4 = h3 * w3, h4 * w4

    conf_mat = np.zeros((num_classes, num_classes + 1), dtype=np.float64)
    cls_inter = np.zeros(num_classes, dtype=np.float64)
    cls_card = np.zeros(num_classes, dtype=np.float64)
    extra: Dict[str, Any] = {
        "scale_correct": np.zeros((3, num_classes), dtype=np.float64),
        "scale_bg_leak": np.zeros((3, num_classes), dtype=np.float64),
        "scale_total": np.zeros((3, num_classes), dtype=np.float64),
        "dot_gt_sum": np.zeros(num_classes, dtype=np.float64),
        "dot_bg_sum": np.zeros(num_classes, dtype=np.float64),
        "dot_other_sum": np.zeros(num_classes, dtype=np.float64),
        "dot_cnt": np.zeros(num_classes, dtype=np.float64),
        "cos_q_k": [[] for _ in range(num_classes)],
        "cos_slot2_k": [[] for _ in range(num_classes)],
        "k_means": [[] for _ in range(num_classes)],
    }

    out: Dict[str, float] = {
        "dots_std": float(dots.std().item()),
        "dots_abs_max": float(dots.abs().max().item()),
    }

    if len(gt_xyxy) == 0:
        return out, conf_mat, cls_inter, cls_card, extra

    gt_boxes = torch.from_numpy(gt_xyxy[:, 1:5]).to(device=device, dtype=torch.float32).unsqueeze(0)
    gt_labels = torch.from_numpy(gt_xyxy[:, 0]).to(device=device, dtype=torch.long).view(1, -1, 1)
    mask_gt = torch.ones((1, len(gt_xyxy), 1), dtype=torch.bool, device=device)

    bg_mask, cls_mask, gt_counts = model.slot_criterion._rasterize_masks_from_xyxy(
        gt_bboxes=gt_boxes,
        gt_labels=gt_labels,
        mask_gt=mask_gt,
        resolutions=resolutions,
        img_size=(imgsz, imgsz),
        num_classes=num_classes,
    )
    bg_m = bg_mask[0] > 0.5         # (N,)
    fg_m = ~bg_m                    # (N,)
    cls_m = cls_mask[0]             # (10, N)
    active = gt_counts[0] > 0       # (10,)
    inactive = ~active              # (10,)

    # Normalized entropy in [0, 1]
    p_safe = attn_ori.clamp(min=1e-7)
    ent = -(p_safe * p_safe.log()).sum(dim=0) / float(np.log(num_classes + 1))  # (N,)
    if fg_m.any():
        out["attn_entropy_fg"] = float(ent[fg_m].mean().item())
        top2 = torch.topk(dots[:, fg_m], k=2, dim=0).values
        out["dots_margin_fg"] = float((top2[0] - top2[1]).mean().item())
        pred_slot_fg = attn_ori[:, fg_m].argmax(dim=0)
        out["slot_bg_leak_pct"] = float((pred_slot_fg == num_classes).float().mean().item() * 100.0)
    if bg_m.any():
        out["attn_entropy_bg"] = float(ent[bg_m].mean().item())

    # Inactive class false-alarm: average attention mass on classes absent from this image
    if inactive.any():
        out["slot_inactive_fa_pct"] = float(attn_ori[:num_classes][inactive].mean().item() * 100.0)

    # Positive CE (all-pixel vs class-balanced)
    neg_log_p = -F.log_softmax(dots, dim=0)[:num_classes, :]  # (10, N)
    pos_m = (cls_m > 0.5).float()
    pos_cnt = pos_m.sum(dim=-1)  # (10,)
    valid_c = active & (pos_cnt > 0)
    if valid_c.any():
        tot_pos = pos_cnt[valid_c].sum().clamp(min=1.0)
        out["slot_pos_ce_all"] = float(((neg_log_p * pos_m)[valid_c].sum() / tot_pos).item())
        per_c_ce = (neg_log_p * pos_m).sum(dim=-1) / pos_cnt.clamp(min=1.0)
        out["slot_pos_ce_balanced"] = float(per_c_ce[valid_c].mean().item())

    # Prepare T=1 static q vs simulated T=2 adapted q2 from slots
    k_tok = sa_cache.get("k")[0].float() if (sa_cache and "k" in sa_cache) else None  # (N, D)
    q_static = sa_cache.get("q")[0].float() if (sa_cache and "q" in sa_cache) else None  # (11, D)
    q_step2 = None
    if k_tok is not None and q_static is not None and hasattr(model, "slot_refiner"):
        sa = model.slot_refiner.slot_attention
        q_step2 = sa.to_q(sa.norm_slots(slot_info["slots"][0].float()))  # (11, D)

    # Per-class Dice, Confusion, Per-Scale Recall/BG Leak, Logit Gaps, and T=1 vs T=2 alignment
    pred_slot_all = attn_ori.argmax(dim=0)  # (N,)
    p_fg = attn_ori[:num_classes]           # (10, N)
    scale_slices = [(0, n3), (n3, n3 + n4), (n3 + n4, dots.shape[1])]

    for c in range(num_classes):
        if not active[c]:
            continue
        cm = cls_m[c] > 0.5
        if not cm.any():
            continue
        cls_inter[c] = float((p_fg[c] * cls_m[c]).sum().item())
        cls_card[c] = float((p_fg[c].sum() + cls_m[c].sum()).item())
        binc = torch.bincount(pred_slot_all[cm], minlength=num_classes + 1).cpu().numpy()
        conf_mat[c] += binc

        # Group 2: Per-scale Recall & BG Leak
        for s_idx, (st, ed) in enumerate(scale_slices):
            cm_s = cm[st:ed]
            tot_s = int(cm_s.sum().item())
            if tot_s > 0:
                ps_s = pred_slot_all[st:ed][cm_s]
                extra["scale_total"][s_idx, c] = tot_s
                extra["scale_correct"][s_idx, c] = float((ps_s == c).sum().item())
                extra["scale_bg_leak"][s_idx, c] = float((ps_s == num_classes).sum().item())

        # Group 4: Logit Gaps on GT pixels of class c
        cnt_px = float(cm.sum().item())
        d_gt = dots[c, cm]
        d_bg = dots[num_classes, cm]
        d_all_c = dots[:, cm].clone()
        d_all_c[c] = -1e9
        d_other = d_all_c.amax(dim=0)
        extra["dot_gt_sum"][c] = float(d_gt.sum().item())
        extra["dot_bg_sum"][c] = float(d_bg.sum().item())
        extra["dot_other_sum"][c] = float(d_other.sum().item())
        extra["dot_cnt"][c] = cnt_px

        # Group 3: T=1 static query vs T=2 adapted slot alignment with mean GT key
        if k_tok is not None and q_static is not None and q_step2 is not None:
            m_proto = getattr(model.slot_refiner.slot_attention, "num_prototypes", 1)
            k_mean = F.normalize(k_tok[cm].mean(dim=0), dim=0)
            q1_sub = F.normalize(q_static[c * m_proto : (c + 1) * m_proto], dim=-1)  # (M, D)
            cos_q1_best = float((q1_sub @ k_mean).max().item())
            q2_n = F.normalize(q_step2[c], dim=0)
            extra["cos_q_k"][c].append(cos_q1_best)
            extra["cos_slot2_k"][c].append(float(torch.dot(q2_n, k_mean).item()))
            extra["k_means"][c].append(k_mean.cpu().numpy())

    return out, conf_mat, cls_inter, cls_card, extra


def _predict_film_off_from_neck(
    model: torch.nn.Module,
    neck_feats: List[torch.Tensor],
    boxes_raw_cat: torch.Tensor,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Run head.cv3 directly on unmodulated neck_feats (FiLM OFF ablation) + NMS."""
    bs = neck_feats[0].shape[0]
    scores_off = torch.cat(
        [model.net.head.cv3[i](neck_feats[i]).view(bs, model.num_classes, -1) for i in range(model.net.head.nl)],
        dim=-1,
    )
    a_raw, s_raw = make_anchors(neck_feats, model.net.head.stride, 0.5)
    anchors, strides = (a.transpose(0, 1).to(device) for a in (a_raw, s_raw))
    pred_dist = model.net.head.dfl(boxes_raw_cat)
    decoded_boxes = (dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides).transpose(1, 2)  # (1, N, 4)
    probs = scores_off.sigmoid().transpose(1, 2)  # (1, N, C)

    b_boxes = decoded_boxes[0]
    b_probs = probs[0]
    max_scores, labels = b_probs.max(dim=-1)
    valid = max_scores >= conf_thres
    if not valid.any():
        return np.zeros((0, 6), dtype=np.float32), scores_off

    cand_boxes = b_boxes[valid]
    cand_scores = max_scores[valid]
    cand_labels = labels[valid]
    keep = tv_ops.batched_nms(cand_boxes, cand_scores, cand_labels, iou_thres)[:max_det]
    det_off = np.column_stack([
        cand_boxes[keep].cpu().numpy(),
        cand_scores[keep].cpu().numpy()[:, None],
        cand_labels[keep].cpu().numpy()[:, None],
    ]).astype(np.float32)
    return det_off, scores_off


@torch.no_grad()
def run_visdrone_component_validation(
    weights_path: str,
    data_root: Optional[str] = None,
    out_dir: str = "VisDrone_yolov11/output_val_epoch50",
    imgsz: Optional[int] = None,
    max_images: int = 0,
    conf_thres: float = 0.001,
    iou_thres: Optional[float] = None,
    vis_conf: float = 0.25,
    split: str = "val",
    split_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    device_str: str = "auto",
    config_path: Optional[str] = None,
) -> None:
    cfg: Optional[Dict[str, Any]] = None
    model_cfg: Optional[Dict[str, Any]] = None
    ds_cfg: Dict[str, Any] = {}

    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = cfg.get("model")
        ds_cfg = cfg.get("data", {}).get(split, {}).get("dataset", {}) or {}
        print(f"[Config] Loaded {config_path} (section: data.{split}.dataset)")

    # Resolve settings (CLI overrides YAML; YAML overrides defaults)
    resolved_data_root = data_root or ds_cfg.get("data_root") or "D:/apps/Datasets/VisDrone/VisDrone_data.lmdb"
    if imgsz is None:
        pipe_imgsz = None
        for step in ds_cfg.get("pipeline", []) or []:
            if isinstance(step, dict) and step.get("type") == "Resize":
                sz = step.get("size")
                if isinstance(sz, (list, tuple)) and len(sz) >= 1:
                    pipe_imgsz = int(sz[0])
                elif isinstance(sz, int):
                    pipe_imgsz = sz
                break
        resolved_imgsz = pipe_imgsz if pipe_imgsz is not None else 960
    else:
        resolved_imgsz = int(imgsz)

    resolved_iou_thres = iou_thres if iou_thres is not None else float((model_cfg or {}).get("nms_threshold", 0.5))
    resolved_seed = seed if seed is not None else int(ds_cfg.get("seed", (cfg or {}).get("seed", 42)))

    if config_path:
        # Respect config's dataset split settings (e.g. None when train_data.lmdb / val_data.lmdb are separate files)
        ds_split = ds_cfg.get("split", None)
        ds_split_ratio = split_ratio if split_ratio is not None else ds_cfg.get("split_ratio", None)
    else:
        ds_split = split
        ds_split_ratio = split_ratio if split_ratio is not None else 0.8

    imgsz = resolved_imgsz
    iou_thres = resolved_iou_thres

    os.makedirs(out_dir, exist_ok=True)
    if device_str == "auto" and cfg and cfg.get("device"):
        device_str = str(cfg["device"])
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else ("cpu" if device_str == "auto" else device_str))

    print("=" * 80)
    print(f"VisDrone YOLOv11 Component-by-Component Validation | Split: {split} | Device: {device} | imgsz: {imgsz}")
    print("=" * 80)

    model, has_slots = load_visdrone_model(weights_path, device, model_cfg=model_cfg)
    probe = YOLOv11ComponentProbe(model.net)

    # Optional hooks on slot_refiner, FiLM modulators, proj_in, to_k, and to_q
    slot_cache: Dict[str, Any] = {}
    film_mod_cache: Dict[int, torch.Tensor] = {}
    proj_cache: Dict[int, torch.Tensor] = {}
    sa_cache: Dict[str, torch.Tensor] = {}
    slot_handle = None
    mod_handles: List[Any] = []
    if has_slots and hasattr(model, "slot_refiner"):
        def _slot_hook(_mod, _inp, out):
            cls_feats, slot_info = out
            slot_cache["cls_feats"] = cls_feats
            slot_cache["slot_info"] = slot_info
        slot_handle = model.slot_refiner.register_forward_hook(_slot_hook)
        for mi, mod in enumerate(model.slot_refiner.modulators):
            def _make_mod_hook(idx_m: int):
                def _mh(_m, _in, out_t):
                    film_mod_cache[idx_m] = out_t.detach()
                return _mh
            mod_handles.append(mod.register_forward_hook(_make_mod_hook(mi)))
        for pi, pr in enumerate(model.slot_refiner.proj_in):
            def _make_proj_hook(idx_p: int):
                def _ph(_m, _in, out_t):
                    proj_cache[idx_p] = out_t.detach()
                return _ph
            mod_handles.append(pr.register_forward_hook(_make_proj_hook(pi)))
        mod_handles.append(model.slot_refiner.slot_attention.to_k.register_forward_hook(
            lambda _m, _in, out_t: sa_cache.__setitem__("k", out_t.detach())
        ))
        mod_handles.append(model.slot_refiner.slot_attention.to_q.register_forward_hook(
            lambda _m, _in, out_t: sa_cache.__setitem__("q", out_t.detach())
        ))

    component_order = list(BASE_COMPONENT_ORDER)
    if has_slots:
        # Insert Slot+FiLM modulated cls features right before head.cv3
        idx_cv3 = component_order.index("head.cv3_p3 (Cls //8)")
        for offset, name in enumerate(["slot.cls_p3 (FiLM //8)", "slot.cls_p4 (FiLM //16)", "slot.cls_p5 (FiLM //32)"]):
            component_order.insert(idx_cv3 + offset, name)

    # Parameter counts
    b, n, h = model.net.backbone, model.net.neck, model.net.head
    comp_modules = {
        "backbone.stage0": b.stage0, "backbone.stage1": b.stage1, "backbone.stage2": b.stage2,
        "backbone.stage3": b.stage3, "backbone.stage4 (feat4 //8)": b.stage4, "backbone.stage5": b.stage5,
        "backbone.stage6 (feat6 //16)": b.stage6, "backbone.stage7": b.stage7, "backbone.stage8": b.stage8,
        "backbone.stage9 (SPPF)": b.stage9, "backbone.stage10 (feat10 //32)": b.stage10,
        "neck.c3k2_13 (TD //16)": n.c3k2_13, "neck.c3k2_16 (P3 //8)": n.c3k2_16,
        "neck.c3k2_19 (P4 //16)": n.c3k2_19, "neck.c3k2_22 (P5 //32)": n.c3k2_22,
        "head.cv2_p3 (Box //8)": h.cv2[0], "head.cv2_p4 (Box //16)": h.cv2[1], "head.cv2_p5 (Box //32)": h.cv2[2],
        "head.cv3_p3 (Cls //8)": h.cv3[0], "head.cv3_p4 (Cls //16)": h.cv3[1], "head.cv3_p5 (Cls //32)": h.cv3[2],
    }
    comp_params = {k: sum(p.numel() for p in m.parameters()) for k, m in comp_modules.items()}
    if has_slots:
        sr_params = sum(p.numel() for p in model.slot_refiner.parameters())
        comp_params["slot.cls_p3 (FiLM //8)"] = sr_params
        comp_params["slot.cls_p4 (FiLM //16)"] = sr_params
        comp_params["slot.cls_p5 (FiLM //32)"] = sr_params

    p_total = sum(p.numel() for p in model.parameters()) / 1e6
    p_backbone = sum(p.numel() for p in model.net.backbone.parameters()) / 1e6
    p_neck = sum(p.numel() for p in model.net.neck.parameters()) / 1e6
    p_head = sum(p.numel() for p in model.net.head.parameters()) / 1e6

    # Load VisDrone dataset (val or train split; never augment during component evaluation)
    ds = VisDroneLMDBDataset(
        data_root=resolved_data_root,
        split=ds_split,
        split_ratio=ds_split_ratio,
        seed=resolved_seed,
        num_classes=int(ds_cfg.get("num_classes", 10)),
        include_boxes=True,
        augment=False,
    )
    num_samples = len(ds) if max_images <= 0 else min(len(ds), max_images)
    print(f"[Dataset] {resolved_data_root} ({split}, split={ds_split}, ratio={ds_split_ratio}): {num_samples} / {len(ds)}")

    metric_keys = [
        "act_mean", "act_std", "neg_frac", "zero_frac", "l2_mean",
        "small_ratio", "medium_ratio", "large_ratio", "all_ratio",
        "full_fisher", "pca_fisher", "class_cos_sep", "fg_mass_pct",
    ]
    comp_acc: Dict[str, Dict[str, List[float]]] = {c: {k: [] for k in metric_keys} for c in component_order}
    head_scale_share: Dict[str, List[float]] = {"p3": [], "p4": [], "p5": []}
    slot_stats: Dict[str, List[float]] = {
        "slot_fg_mass_pct": [], "slot_share_p3": [], "slot_share_p4": [], "slot_share_p5": [], "slot_cos_sep": [],
        "dots_std": [], "dots_abs_max": [], "dots_margin_fg": [],
        "attn_entropy_fg": [], "attn_entropy_bg": [],
        "slot_pos_ce_all": [], "slot_pos_ce_balanced": [],
        "slot_bg_leak_pct": [], "slot_inactive_fa_pct": [],
        "film_gamma_abs_p3": [], "film_gamma_abs_p4": [], "film_gamma_abs_p5": [],
        "film_beta_abs_p3": [], "film_beta_abs_p4": [], "film_beta_abs_p5": [],
        "film_sat_pct_p3": [], "film_sat_pct_p4": [], "film_sat_pct_p5": [],
        "loss_focal_film_on": [], "loss_focal_film_off": [],
    }
    slot_conf_mat = np.zeros((10, 11), dtype=np.float64)
    slot_cls_inter = np.zeros(10, dtype=np.float64)
    slot_cls_card = np.zeros(10, dtype=np.float64)
    slot_scale_correct = np.zeros((3, 10), dtype=np.float64)
    slot_scale_bg_leak = np.zeros((3, 10), dtype=np.float64)
    slot_scale_total = np.zeros((3, 10), dtype=np.float64)
    slot_dot_gt_sum = np.zeros(10, dtype=np.float64)
    slot_dot_bg_sum = np.zeros(10, dtype=np.float64)
    slot_dot_other_sum = np.zeros(10, dtype=np.float64)
    slot_dot_cnt = np.zeros(10, dtype=np.float64)
    slot_cos_q_k: List[List[float]] = [[] for _ in range(10)]
    slot_cos_slot2_k: List[List[float]] = [[] for _ in range(10)]
    slot_k_means: List[List[np.ndarray]] = [[] for _ in range(10)]
    stage_sep_acc: Dict[str, Dict[str, List[float]]] = {
        f"{st}_{sc}": {"full_fisher": [], "class_cos_sep": [], "all_ratio": []}
        for st in ("proj", "key")
        for sc in ("p3", "p4", "p5")
    }
    slot_vecs_list: List[np.ndarray] = []
    all_preds_film_off: List[np.ndarray] = []
    box_diag_acc: Dict[str, List[float]] = {
        "boundary_contrast_p3": [], "boundary_contrast_p4": [], "boundary_contrast_p5": [],
        "intra_box_decorr_p3": [], "intra_box_decorr_p4": [], "intra_box_decorr_p5": [],
        "gt_coverage_all_pct": [], "gt_coverage_small_pct": [], "avg_pos_per_gt": [],
        "dfl_clipping_rate_pct": [], "dfl_entropy_pos": [], "dfl_top1_prob": [],
        "grad_cos_box_vs_cls_p3": [], "grad_cos_box_vs_cls_p4": [], "grad_cos_box_vs_cls_p5": [],
        "grad_ratio_box_over_cls_p3": [], "grad_ratio_box_over_cls_p4": [], "grad_ratio_box_over_cls_p5": [],
    }
    grad_conflict_samples = 0

    p3_dim = 64
    p3_proto_sum = np.zeros((10, p3_dim), dtype=np.float64)
    p3_proto_cnt = np.zeros(10, dtype=np.float64)

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []
    latencies_ms: List[float] = []
    vis_sample: Optional[Dict[str, Any]] = None

    t0_all = time.perf_counter()
    for idx in range(num_samples):
        item = ds.get_data_info(idx)
        pil_img = item["image"]
        target = item["target"]

        resized_pil = TF.resize(pil_img, [imgsz, imgsz])
        img_t = TF.to_tensor(resized_pil).unsqueeze(0).to(device)

        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        if len(boxes) > 0:
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            gt_lb_norm = np.column_stack([labels, cx, cy, bw, bh]).astype(np.float32)
            x1 = np.clip((cx - bw / 2.0) * imgsz, 0.0, float(imgsz))
            y1 = np.clip((cy - bh / 2.0) * imgsz, 0.0, float(imgsz))
            x2 = np.clip((cx + bw / 2.0) * imgsz, 0.0, float(imgsz))
            y2 = np.clip((cy + bh / 2.0) * imgsz, 0.0, float(imgsz))
            gt_xyxy = np.column_stack([labels, x1, y1, x2, y2]).astype(np.float32)
        else:
            gt_lb_norm = np.zeros((0, 5), dtype=np.float32)
            gt_xyxy = np.zeros((0, 5), dtype=np.float32)

        probe.clear()
        slot_cache.clear()
        film_mod_cache.clear()
        proj_cache.clear()
        sa_cache.clear()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        pred_list = model.predict(img_t, conf_threshold=conf_thres, nms_threshold=iou_thres, max_det=300)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

        if has_slots and "cls_feats" in slot_cache:
            cf = slot_cache["cls_feats"]
            probe.features["slot.cls_p3 (FiLM //8)"] = cf[0]
            probe.features["slot.cls_p4 (FiLM //16)"] = cf[1]
            probe.features["slot.cls_p5 (FiLM //32)"] = cf[2]

            s_info = slot_cache["slot_info"]
            attn_ori = s_info["attn_ori"][0].float()  # (11, N)
            (h3, w3), (h4, w4), _ = s_info["resolutions"]
            n3, n4 = h3 * w3, h4 * w4
            fg_attn = attn_ori[:10, :]
            s_p3 = float(fg_attn[:, :n3].sum().item())
            s_p4 = float(fg_attn[:, n3 : n3 + n4].sum().item())
            s_p5 = float(fg_attn[:, n3 + n4 :].sum().item())
            s_tot = s_p3 + s_p4 + s_p5 + 1e-8
            slot_stats["slot_share_p3"].append(s_p3 / s_tot * 100.0)
            slot_stats["slot_share_p4"].append(s_p4 / s_tot * 100.0)
            slot_stats["slot_share_p5"].append(s_p5 / s_tot * 100.0)

            slots_fg = F.normalize(s_info["slots"][0, :10, :].float(), dim=-1)
            slot_vecs_list.append(slots_fg.cpu().numpy())
            sim = slots_fg @ slots_fg.T
            off_diag = sim[~torch.eye(10, dtype=torch.bool, device=sim.device)]
            slot_stats["slot_cos_sep"].append(float((1.0 - off_diag).mean().item()))

            # Slot overfit + cls_attn bottleneck diagnostics
            so_m, c_mat, c_int, c_card, so_extra = _measure_slot_overfit_metrics(
                model, s_info, gt_xyxy, imgsz, device, sa_cache=sa_cache
            )
            for k, v in so_m.items():
                slot_stats[k].append(v)
            slot_conf_mat += c_mat
            slot_cls_inter += c_int
            slot_cls_card += c_card
            slot_scale_correct += so_extra["scale_correct"]
            slot_scale_bg_leak += so_extra["scale_bg_leak"]
            slot_scale_total += so_extra["scale_total"]
            slot_dot_gt_sum += so_extra["dot_gt_sum"]
            slot_dot_bg_sum += so_extra["dot_bg_sum"]
            slot_dot_other_sum += so_extra["dot_other_sum"]
            slot_dot_cnt += so_extra["dot_cnt"]
            for c_i in range(10):
                slot_cos_q_k[c_i].extend(so_extra["cos_q_k"][c_i])
                slot_cos_slot2_k[c_i].extend(so_extra["cos_slot2_k"][c_i])
                slot_k_means[c_i].extend(so_extra["k_means"][c_i])

            # Group 1 bottleneck: measure separability at proj_in (32d) and Key (32d) per scale
            if len(gt_lb_norm) > 0:
                (h3, w3), (h4, w4), (h5, w5) = s_info["resolutions"]
                n3_s, n4_s = h3 * w3, h4 * w4
                for pi, sc_name in enumerate(["p3", "p4", "p5"]):
                    pf = proj_cache.get(pi)
                    if pf is not None:
                        pm = measure_feature_metrics(pf, gt_lb_norm, imgsz, is_cls_logits=False)
                        for mk in ("full_fisher", "class_cos_sep", "all_ratio"):
                            if mk in pm:
                                stage_sep_acc[f"proj_{sc_name}"][mk].append(pm[mk])
                if "k" in sa_cache:
                    kt = sa_cache["k"][0].float().transpose(0, 1)  # (D, N)
                    d_k = kt.shape[0]
                    k_maps = [
                        ("p3", kt[:, :n3_s].view(1, d_k, h3, w3)),
                        ("p4", kt[:, n3_s : n3_s + n4_s].view(1, d_k, h4, w4)),
                        ("p5", kt[:, n3_s + n4_s :].view(1, d_k, h5, w5)),
                    ]
                    for sc_name, km in k_maps:
                        k_met = measure_feature_metrics(km, gt_lb_norm, imgsz, is_cls_logits=False)
                        for mk in ("full_fisher", "class_cos_sep", "all_ratio"):
                            if mk in k_met:
                                stage_sep_acc[f"key_{sc_name}"][mk].append(k_met[mk])

            # FiLM modulation magnitude & tanh saturation per scale
            if model.slot_refiner.fusion_mode == "film":
                for mi, pname in enumerate(["p3", "p4", "p5"]):
                    mod_out = film_mod_cache.get(mi)
                    if mod_out is not None:
                        dg, db = mod_out.float().chunk(2, dim=1)
                        tg, tb = torch.tanh(dg), torch.tanh(db)
                        slot_stats[f"film_gamma_abs_{pname}"].append(float((0.5 * tg.abs()).mean().item()))
                        slot_stats[f"film_beta_abs_{pname}"].append(float((0.5 * tb.abs()).mean().item()))
                        sat = ((tg.abs() > 0.9).float().mean() + (tb.abs() > 0.9).float().mean()) * 50.0
                        slot_stats[f"film_sat_pct_{pname}"].append(float(sat.item()))

        for cname in component_order:
            feat = probe.features.get(cname)
            if feat is None:
                continue
            is_cls = cname.startswith("head.cv3")
            m_dict = measure_feature_metrics(feat, gt_lb_norm, imgsz, is_cls_logits=is_cls)
            for k, v in m_dict.items():
                comp_acc[cname][k].append(v)

        p3_feat = probe.features["neck.c3k2_16 (P3 //8)"][0].float()
        _, h3, w3 = p3_feat.shape
        p3_flat = F.normalize(p3_feat.reshape(p3_dim, -1), dim=0)
        _, _, _, _, cls_m_p3 = build_masks_at_res_gpu(gt_lb_norm, h3, w3, imgsz, device)
        cls_m_10 = cls_m_p3[:10]
        cnt_c = cls_m_10.sum(dim=1)
        proto_t = cls_m_10.float() @ p3_flat.transpose(0, 1)
        p3_proto_sum += proto_t.cpu().double().numpy()
        p3_proto_cnt += cnt_c.cpu().numpy()

        cv3_masses = []
        for cv3_name in ["head.cv3_p3 (Cls //8)", "head.cv3_p4 (Cls //16)", "head.cv3_p5 (Cls //32)"]:
            pr = probe.features[cv3_name][0].float().sigmoid().max(dim=0).values
            cv3_masses.append(float(pr.sum().item()))
        tot_mass = sum(cv3_masses) + 1e-8
        head_scale_share["p3"].append(cv3_masses[0] / tot_mass * 100.0)
        head_scale_share["p4"].append(cv3_masses[1] / tot_mass * 100.0)
        head_scale_share["p5"].append(cv3_masses[2] / tot_mass * 100.0)

        # Box learning mechanism diagnostics: Boundary Contrast & Intra-Box Decorrelation
        for pname, cname in [
            ("p3", "neck.c3k2_16 (P3 //8)"),
            ("p4", "neck.c3k2_19 (P4 //16)"),
            ("p5", "neck.c3k2_22 (P5 //32)"),
        ]:
            sp_m = _measure_box_spatial_metrics(probe.features[cname], gt_lb_norm, imgsz)
            if "boundary_contrast" in sp_m:
                box_diag_acc[f"boundary_contrast_{pname}"].append(sp_m["boundary_contrast"])
            if "intra_box_decorr" in sp_m:
                box_diag_acc[f"intra_box_decorr_{pname}"].append(sp_m["intra_box_decorr"])

        # TAL coverage, DFL entropy/clipping, and FiLM ON/OFF ablation
        neck_feats = [
            probe.features["neck.c3k2_16 (P3 //8)"],
            probe.features["neck.c3k2_19 (P4 //16)"],
            probe.features["neck.c3k2_22 (P5 //32)"],
        ]
        boxes_raw_cat = torch.cat([
            probe.features["head.cv2_p3 (Box //8)"].flatten(2),
            probe.features["head.cv2_p4 (Box //16)"].flatten(2),
            probe.features["head.cv2_p5 (Box //32)"].flatten(2),
        ], dim=-1)
        scores_raw_cat = torch.cat([
            probe.features["head.cv3_p3 (Cls //8)"].flatten(2),
            probe.features["head.cv3_p4 (Cls //16)"].flatten(2),
            probe.features["head.cv3_p5 (Cls //32)"].flatten(2),
        ], dim=-1)
        if has_slots and "slot_info" in slot_cache and slot_cache["slot_info"].get("logit_prior") is not None:
            scores_raw_cat = scores_raw_cat + slot_cache["slot_info"]["logit_prior"]

        if has_slots:
            det_off, scores_off = _predict_film_off_from_neck(
                model, neck_feats, boxes_raw_cat, imgsz, conf_thres, iou_thres, 300, device
            )
            all_preds_film_off.append(det_off)
            if len(gt_xyxy) > 0:
                a_raw, s_raw = make_anchors(neck_feats, model.net.head.stride, 0.5)
                anchors, strides = (a.transpose(0, 1).to(device) for a in (a_raw, s_raw))
                pred_dist = model.net.head.dfl(boxes_raw_cat)
                dec_xyxy = (dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides).transpose(1, 2)
                ap_px = (anchors * strides).transpose(0, 1)
                st_t = strides.transpose(0, 1)
                t_list = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device), "_is_cxcywh": True}]
                l_on = model.criterion(
                    preds={"boxes": boxes_raw_cat, "scores": scores_raw_cat, "feats": neck_feats},
                    targets=t_list, decoded_bboxes=dec_xyxy, anchor_points=ap_px, stride_tensor=st_t, img_size=(imgsz, imgsz),
                )
                l_off = model.criterion(
                    preds={"boxes": boxes_raw_cat, "scores": scores_off, "feats": neck_feats},
                    targets=t_list, decoded_bboxes=dec_xyxy, anchor_points=ap_px, stride_tensor=st_t, img_size=(imgsz, imgsz),
                )
                slot_stats["loss_focal_film_on"].append(float(l_on["loss_focal"].item()))
                slot_stats["loss_focal_film_off"].append(float(l_off["loss_focal"].item()))

        if len(gt_xyxy) > 0:
            tal_dfl_m = _measure_tal_and_dfl_metrics(model, neck_feats, boxes_raw_cat, scores_raw_cat, gt_xyxy, imgsz, device)
            for k, v in tal_dfl_m.items():
                box_diag_acc[k].append(v)

            # Gradient conflict cos_sim(grad_box, grad_cls+slot) on first 20 val images with GTs
            if grad_conflict_samples < 20:
                gc_m = _measure_neck_gradient_conflict(model, has_slots, img_t, target, imgsz, device)
                for k, v in gc_m.items():
                    box_diag_acc[k].append(v)
                grad_conflict_samples += 1

        p0 = pred_list[0]
        if len(p0["boxes"]) > 0:
            det = np.column_stack([
                p0["boxes"].cpu().numpy(),
                p0["scores"].cpu().numpy()[:, None],
                p0["labels"].cpu().numpy()[:, None],
            ]).astype(np.float32)
        else:
            det = np.zeros((0, 6), dtype=np.float32)

        all_preds.append(det)
        all_gts.append(gt_xyxy)

        if vis_sample is None and len(gt_xyxy) >= 4:
            vis_sample = {
                "img_name": f"val_idx_{idx:05d}",
                "lb_rgb": np.asarray(resized_pil, dtype=np.uint8),
                "gt_xyxy": gt_xyxy,
                "det": det[det[:, 4] >= vis_conf] if len(det) > 0 else det,
                "feat4": probe.features["backbone.stage4 (feat4 //8)"][0].float().detach(),
                "feat6": probe.features["backbone.stage6 (feat6 //16)"][0].float().detach(),
                "feat10": probe.features["backbone.stage10 (feat10 //32)"][0].float().detach(),
                "p3": probe.features["neck.c3k2_16 (P3 //8)"][0].float().detach(),
                "p4": probe.features["neck.c3k2_19 (P4 //16)"][0].float().detach(),
                "p5": probe.features["neck.c3k2_22 (P5 //32)"][0].float().detach(),
                "cv3_p3": probe.features["head.cv3_p3 (Cls //8)"][0].float().sigmoid().max(dim=0).values.cpu().numpy(),
            }

        if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
            print(f"  Processed {idx + 1}/{num_samples} images...")

    probe.remove()
    if slot_handle is not None:
        slot_handle.remove()
    for mh in mod_handles:
        mh.remove()
    total_sec = time.perf_counter() - t0_all

    overall_det, per_class_rows = compute_coco_map(all_preds, all_gts)
    small_det, _ = compute_coco_map(all_preds, all_gts, area_range=(0.0, 32.0 ** 2))
    med_det, _ = compute_coco_map(all_preds, all_gts, area_range=(32.0 ** 2, 96.0 ** 2))
    large_det, _ = compute_coco_map(all_preds, all_gts, area_range=(96.0 ** 2, 1e9))

    for r in per_class_rows:
        cid = int(r["class_id"])
        r["class_name"] = VISDRONE_CLASSES[cid] if cid < len(VISDRONE_CLASSES) else f"class_{cid}"

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

    comp_rows: List[Dict[str, object]] = []
    for cname in component_order:
        acc = comp_acc[cname]
        row: Dict[str, object] = {
            "component": cname,
            "module_group": cname.split(".")[0],
            "params_k": round(comp_params.get(cname, 0) / 1000.0, 2),
        }
        for mkey in metric_keys:
            vals = acc[mkey]
            row[mkey] = round(float(np.mean(vals)), 4) if vals else np.nan
        comp_rows.append(row)

    valid_cls_ids = [c for c in range(10) if p3_proto_cnt[c] >= 5]
    pw_mat = np.full((len(valid_cls_ids), len(valid_cls_ids)), np.nan, dtype=np.float64)
    if len(valid_cls_ids) >= 2:
        protos = [p3_proto_sum[c] / p3_proto_cnt[c] for c in valid_cls_ids]
        protos = [v / (np.linalg.norm(v) + 1e-8) for v in protos]
        P_mat = np.stack(protos, axis=0)
        pw_mat = 1.0 - (P_mat @ P_mat.T)
        np.fill_diagonal(pw_mat, np.nan)

    slot_summary = None
    slot_per_class_rows: List[Dict[str, object]] = []
    if has_slots and slot_stats["slot_cos_sep"]:
        slot_summary = {
            "slot_cos_sep": round(float(np.mean(slot_stats["slot_cos_sep"])), 4),
            "slot_share_p3_pct": round(float(np.mean(slot_stats["slot_share_p3"])), 2),
            "slot_share_p4_pct": round(float(np.mean(slot_stats["slot_share_p4"])), 2),
            "slot_share_p5_pct": round(float(np.mean(slot_stats["slot_share_p5"])), 2),
        }

        # Group 1: FiLM ON vs OFF Ablation
        if all_preds_film_off:
            film_off_det, _ = compute_coco_map(all_preds_film_off, all_gts)
            slot_summary["film_on_mAP50"] = round(float(overall_det["mAP50"]), 4)
            slot_summary["film_off_mAP50"] = round(float(film_off_det["mAP50"]), 4)
            slot_summary["film_delta_mAP50"] = round(float(overall_det["mAP50"] - film_off_det["mAP50"]), 4)
            slot_summary["film_off_mAP50_95"] = round(float(film_off_det["mAP50_95"]), 4)
            slot_summary["film_delta_mAP50_95"] = round(float(overall_det["mAP50_95"] - film_off_det["mAP50_95"]), 4)
        if slot_stats["loss_focal_film_on"] and slot_stats["loss_focal_film_off"]:
            f_on = float(np.mean(slot_stats["loss_focal_film_on"]))
            f_off = float(np.mean(slot_stats["loss_focal_film_off"]))
            slot_summary["loss_focal_film_on"] = round(f_on, 4)
            slot_summary["loss_focal_film_off"] = round(f_off, 4)
            slot_summary["loss_focal_film_delta"] = round(f_on - f_off, 4)

        # Group 2: Attention sharpness, entropy, calibration, BG leak, inactive FA
        for k in [
            "dots_std", "dots_abs_max", "dots_margin_fg",
            "attn_entropy_fg", "attn_entropy_bg",
            "slot_pos_ce_all", "slot_pos_ce_balanced",
            "slot_bg_leak_pct", "slot_inactive_fa_pct",
            "film_gamma_abs_p3", "film_gamma_abs_p4", "film_gamma_abs_p5",
            "film_beta_abs_p3", "film_beta_abs_p4", "film_beta_abs_p5",
            "film_sat_pct_p3", "film_sat_pct_p4", "film_sat_pct_p5",
        ]:
            if slot_stats[k]:
                slot_summary[k] = round(float(np.mean(slot_stats[k])), 4)

        # Group 4: Query orthogonality, cross-image slot variance, and submodule weight L2 norms
        q_fg = F.normalize(model.slot_refiner.slot_attention.obj_queries[0, :10, :].detach().float(), dim=-1)
        q_sim = q_fg @ q_fg.T
        q_off = q_sim[~torch.eye(10, dtype=torch.bool, device=q_sim.device)]
        slot_summary["query_cos_sep"] = round(float((1.0 - q_off).mean().item()), 4)
        if len(slot_vecs_list) >= 2:
            sv = np.stack(slot_vecs_list, axis=0)  # (N_img, 10, D)
            slot_summary["slot_cross_img_std"] = round(float(sv.std(axis=0).mean()), 4)

        def _wnorm(mod: torch.nn.Module) -> float:
            sq = sum(float(p.detach().float().pow(2).sum().item()) for p in mod.parameters())
            return round(float(np.sqrt(sq)), 4)

        slot_summary["wnorm_proj_in"] = _wnorm(model.slot_refiner.proj_in)
        slot_summary["wnorm_slot_attn"] = _wnorm(model.slot_refiner.slot_attention)
        slot_summary["wnorm_modulators"] = _wnorm(model.slot_refiner.modulators)

        # Group 4 (static): ||q_c||_2 for all 10 FG classes (mean over M prototypes) + BG
        sa = model.slot_refiner.slot_attention
        m_proto = getattr(sa, "num_prototypes", 1)
        q_all_static = sa.to_q(sa.norm_slots(sa.obj_queries.detach().float()))[0]  # (K*M + 1, D)
        q_norms_raw = torch.linalg.vector_norm(q_all_static, dim=-1).cpu().numpy()
        q_norms_fg = q_norms_raw[: 10 * m_proto].reshape(10, m_proto).mean(axis=1)
        q_norms_11 = np.concatenate([q_norms_fg, q_norms_raw[-1:]])  # (11,)
        slot_summary["q_norm_fg_mean"] = round(float(q_norms_11[:10].mean()), 4)
        slot_summary["q_norm_fg_min"] = round(float(q_norms_11[:10].min()), 4)
        slot_summary["q_norm_fg_max"] = round(float(q_norms_11[:10].max()), 4)
        slot_summary["q_norm_bg"] = round(float(q_norms_11[10]), 4)

        # Group 1: Neck -> proj_in (32d) -> Key (32d) separability table
        neck_comp_map = {
            "p3": "neck.c3k2_16 (P3 //8)",
            "p4": "neck.c3k2_19 (P4 //16)",
            "p5": "neck.c3k2_22 (P5 //32)",
        }
        stage_rows: List[Dict[str, object]] = []
        for sc in ("p3", "p4", "p5"):
            nc_acc = comp_acc[neck_comp_map[sc]]
            n_fish = float(np.mean(nc_acc["full_fisher"])) if nc_acc["full_fisher"] else np.nan
            n_cos = float(np.mean(nc_acc["class_cos_sep"])) if nc_acc["class_cos_sep"] else np.nan
            p_fish = float(np.mean(stage_sep_acc[f"proj_{sc}"]["full_fisher"])) if stage_sep_acc[f"proj_{sc}"]["full_fisher"] else np.nan
            p_cos = float(np.mean(stage_sep_acc[f"proj_{sc}"]["class_cos_sep"])) if stage_sep_acc[f"proj_{sc}"]["class_cos_sep"] else np.nan
            k_fish = float(np.mean(stage_sep_acc[f"key_{sc}"]["full_fisher"])) if stage_sep_acc[f"key_{sc}"]["full_fisher"] else np.nan
            k_cos = float(np.mean(stage_sep_acc[f"key_{sc}"]["class_cos_sep"])) if stage_sep_acc[f"key_{sc}"]["class_cos_sep"] else np.nan
            slot_summary[f"fisher_neck_{sc}"] = round(n_fish, 4) if np.isfinite(n_fish) else np.nan
            slot_summary[f"fisher_proj_{sc}"] = round(p_fish, 4) if np.isfinite(p_fish) else np.nan
            slot_summary[f"fisher_key_{sc}"] = round(k_fish, 4) if np.isfinite(k_fish) else np.nan
            stage_rows.append({
                "scale": sc.upper(),
                "neck_fisher": round(n_fish, 4), "proj_fisher": round(p_fish, 4), "key_fisher": round(k_fish, 4),
                "neck_cos_sep": round(n_cos, 4), "proj_cos_sep": round(p_cos, 4), "key_cos_sep": round(k_cos, 4),
            })
        with open(os.path.join(out_dir, "slot_bottleneck_stages.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
            w.writeheader()
            w.writerows(stage_rows)

        # Group 2, 3, 4: Per-class Slot Dice, Recall (overall + P3/P4/P5), BG leak (overall + P3/P4/P5),
        # T=1 vs T=2 alignment, key variance, ||q_c||, and logit gaps
        col_names_11 = list(VISDRONE_CLASSES) + ["BG"]
        dices_valid = []
        cos_q1_all, cos_q2_all = [], []
        for c in range(10):
            tot_px = float(slot_conf_mat[c].sum())
            dice_c = (2.0 * slot_cls_inter[c] / max(slot_cls_card[c], 1e-8)) if slot_cls_card[c] > 0 else np.nan
            rec_c = (slot_conf_mat[c, c] / tot_px * 100.0) if tot_px > 0 else np.nan
            bg_leak_c = (slot_conf_mat[c, 10] / tot_px * 100.0) if tot_px > 0 else np.nan
            wrong = slot_conf_mat[c].copy()
            wrong[c] = -1.0
            top_w_idx = int(np.argmax(wrong)) if tot_px > 0 else -1
            top_w_str = f"{col_names_11[top_w_idx]} ({wrong[top_w_idx] / tot_px * 100.0:.1f}%)" if (tot_px > 0 and wrong[top_w_idx] > 0) else "-"

            rec_sc = [
                (slot_scale_correct[s, c] / slot_scale_total[s, c] * 100.0) if slot_scale_total[s, c] > 0 else np.nan
                for s in range(3)
            ]
            bgl_sc = [
                (slot_scale_bg_leak[s, c] / slot_scale_total[s, c] * 100.0) if slot_scale_total[s, c] > 0 else np.nan
                for s in range(3)
            ]
            d_cnt = max(slot_dot_cnt[c], 1.0)
            dot_gt_m = slot_dot_gt_sum[c] / d_cnt if slot_dot_cnt[c] > 0 else np.nan
            dot_bg_m = slot_dot_bg_sum[c] / d_cnt if slot_dot_cnt[c] > 0 else np.nan
            dot_ot_m = slot_dot_other_sum[c] / d_cnt if slot_dot_cnt[c] > 0 else np.nan
            cos_q1_m = float(np.mean(slot_cos_q_k[c])) if slot_cos_q_k[c] else np.nan
            cos_q2_m = float(np.mean(slot_cos_slot2_k[c])) if slot_cos_slot2_k[c] else np.nan
            k_std_m = float(np.stack(slot_k_means[c], axis=0).std(axis=0).mean()) if len(slot_k_means[c]) >= 2 else np.nan

            if np.isfinite(dice_c):
                dices_valid.append(dice_c)
            if np.isfinite(cos_q1_m):
                cos_q1_all.append(cos_q1_m)
            if np.isfinite(cos_q2_m):
                cos_q2_all.append(cos_q2_m)

            slot_per_class_rows.append({
                "class_id": c,
                "class_name": VISDRONE_CLASSES[c],
                "gt_pixels": int(tot_px),
                "slot_dice": round(float(dice_c), 4) if np.isfinite(dice_c) else np.nan,
                "slot_recall_pct": round(float(rec_c), 2) if np.isfinite(rec_c) else np.nan,
                "rec_p3_pct": round(float(rec_sc[0]), 1) if np.isfinite(rec_sc[0]) else np.nan,
                "rec_p4_pct": round(float(rec_sc[1]), 1) if np.isfinite(rec_sc[1]) else np.nan,
                "rec_p5_pct": round(float(rec_sc[2]), 1) if np.isfinite(rec_sc[2]) else np.nan,
                "bg_leak_pct": round(float(bg_leak_c), 2) if np.isfinite(bg_leak_c) else np.nan,
                "bg_leak_p3_pct": round(float(bgl_sc[0]), 1) if np.isfinite(bgl_sc[0]) else np.nan,
                "bg_leak_p4_pct": round(float(bgl_sc[1]), 1) if np.isfinite(bgl_sc[1]) else np.nan,
                "bg_leak_p5_pct": round(float(bgl_sc[2]), 1) if np.isfinite(bgl_sc[2]) else np.nan,
                "q_norm": round(float(q_norms_11[c]), 3),
                "dot_gt": round(float(dot_gt_m), 3) if np.isfinite(dot_gt_m) else np.nan,
                "gap_vs_bg": round(float(dot_gt_m - dot_bg_m), 3) if np.isfinite(dot_gt_m) else np.nan,
                "gap_vs_other": round(float(dot_gt_m - dot_ot_m), 3) if np.isfinite(dot_gt_m) else np.nan,
                "cos_q1_k": round(cos_q1_m, 3) if np.isfinite(cos_q1_m) else np.nan,
                "cos_q2_k": round(cos_q2_m, 3) if np.isfinite(cos_q2_m) else np.nan,
                "k_intra_std": round(k_std_m, 3) if np.isfinite(k_std_m) else np.nan,
                "top_confused_slot": top_w_str,
            })
        if dices_valid:
            slot_summary["slot_mean_dice"] = round(float(np.mean(dices_valid)), 4)
        if cos_q1_all and cos_q2_all:
            slot_summary["cos_q1_vs_k_mean"] = round(float(np.mean(cos_q1_all)), 4)
            slot_summary["cos_q2_vs_k_mean"] = round(float(np.mean(cos_q2_all)), 4)

        with open(os.path.join(out_dir, "slot_overfit_diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(slot_summary.keys()))
            w.writeheader()
            w.writerow(slot_summary)

        with open(os.path.join(out_dir, "slot_per_class_diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(slot_per_class_rows[0].keys()))
            w.writeheader()
            w.writerows(slot_per_class_rows)

        if slot_conf_mat.sum() > 0:
            _plot_slot_confusion_matrix(slot_conf_mat, os.path.join(out_dir, "chart_06_slot_confusion_matrix.png"))

    box_diag = {
        k: round(float(np.mean(v)), 4)
        for k, v in box_diag_acc.items()
        if len(v) > 0
    }
    if box_diag:
        with open(os.path.join(out_dir, "box_learning_diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(box_diag.keys()))
            w.writeheader()
            w.writerow(box_diag)

    # Save CSVs & Plots
    with open(os.path.join(out_dir, "component_breakdown.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)

    with open(os.path.join(out_dir, "detection_overall_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overall_det.keys()))
        w.writeheader()
        w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in overall_det.items()})

    if per_class_rows:
        sorted_cls = sorted(per_class_rows, key=lambda r: r["ap50"], reverse=True)
        with open(os.path.join(out_dir, "per_class_ap.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sorted_cls[0].keys()))
            w.writeheader()
            for r in sorted_cls:
                w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    _plot_component_progression(comp_rows, os.path.join(out_dir, "chart_01_component_progression.png"))
    _plot_backbone_vs_neck(comp_rows, os.path.join(out_dir, "chart_02_backbone_vs_neck_multiscale.png"))
    _plot_detection_and_head(overall_det, per_class_rows, os.path.join(out_dir, "chart_03_head_and_detection_summary.png"))
    if vis_sample is not None:
        _plot_visdrone_visual_inspection(vis_sample, os.path.join(out_dir, "chart_04_feature_visual_inspection.png"))
    _plot_visdrone_pairwise_heatmap(valid_cls_ids, pw_mat, os.path.join(out_dir, "chart_05_pairwise_class_separation.png"))
    _generate_visdrone_dashboard(out_dir, comp_rows, overall_det, slot_summary, box_diag=box_diag)

    print("\n" + "=" * 80)
    print("VISDRONE COMPONENT VALIDATION SUMMARY:")
    print(f"  mAP@0.50      : {overall_det['mAP50']:.4f}")
    print(f"  mAP@0.50:0.95 : {overall_det['mAP50_95']:.4f} (S: {overall_det['mAP50_95_small']:.4f} | M: {overall_det['mAP50_95_medium']:.4f} | L: {overall_det['mAP50_95_large']:.4f})")
    print(f"  P / R / F1    : {overall_det['precision_c25']:.4f} / {overall_det['recall_c25']:.4f} / {overall_det['f1_c25']:.4f} (conf>=0.25)")
    if slot_summary:
        print("  --- Slot Overfit & FiLM Diagnostics ---")
        print(f"  Slot Cos Sep / Query Sep / Std : {slot_summary.get('slot_cos_sep', 0):.4f} / {slot_summary.get('query_cos_sep', 0):.4f} / {slot_summary.get('slot_cross_img_std', 0):.4f}")
        print(f"  Slot Share P3/P4/P5            : {slot_summary.get('slot_share_p3_pct', 0):.1f}% / {slot_summary.get('slot_share_p4_pct', 0):.1f}% / {slot_summary.get('slot_share_p5_pct', 0):.1f}%")
        if "film_off_mAP50" in slot_summary:
            print(f"  FiLM Ablation mAP@50 (ON/OFF)  : {slot_summary['film_on_mAP50']:.4f} / {slot_summary['film_off_mAP50']:.4f} (delta: {slot_summary['film_delta_mAP50']:+.4f})")
        if "loss_focal_film_on" in slot_summary:
            print(f"  FiLM Ablation Focal  (ON/OFF)  : {slot_summary['loss_focal_film_on']:.4f} / {slot_summary['loss_focal_film_off']:.4f} (delta: {slot_summary['loss_focal_film_delta']:+.4f})")
        print(f"  Dots Std / AbsMax / FG Margin  : {slot_summary.get('dots_std', 0):.3f} / {slot_summary.get('dots_abs_max', 0):.3f} / {slot_summary.get('dots_margin_fg', 0):.3f}")
        print(f"  Attn Entropy (FG / BG)         : {slot_summary.get('attn_entropy_fg', 0):.3f} / {slot_summary.get('attn_entropy_bg', 0):.3f}")
        print(f"  Slot PosCE (All / Balanced)    : {slot_summary.get('slot_pos_ce_all', 0):.3f} / {slot_summary.get('slot_pos_ce_balanced', 0):.3f} | Mean Dice: {slot_summary.get('slot_mean_dice', 0):.4f}")
        print(f"  BG Leak / Inactive FA          : {slot_summary.get('slot_bg_leak_pct', 0):.2f}% / {slot_summary.get('slot_inactive_fa_pct', 0):.3f}%")
        print(f"  FiLM |d_gamma| P3/P4/P5        : {slot_summary.get('film_gamma_abs_p3', 0):.3f} / {slot_summary.get('film_gamma_abs_p4', 0):.3f} / {slot_summary.get('film_gamma_abs_p5', 0):.3f}")
        print(f"  FiLM |d_beta|  P3/P4/P5        : {slot_summary.get('film_beta_abs_p3', 0):.3f} / {slot_summary.get('film_beta_abs_p4', 0):.3f} / {slot_summary.get('film_beta_abs_p5', 0):.3f}")
        print(f"  FiLM Sat(>0.9) P3/P4/P5        : {slot_summary.get('film_sat_pct_p3', 0):.2f}% / {slot_summary.get('film_sat_pct_p4', 0):.2f}% / {slot_summary.get('film_sat_pct_p5', 0):.2f}%")
        print(f"  Weight Norms (proj/attn/mod)   : {slot_summary.get('wnorm_proj_in', 0):.2f} / {slot_summary.get('wnorm_slot_attn', 0):.2f} / {slot_summary.get('wnorm_modulators', 0):.2f}")
        print(f"  ||q|| Norm (FG min/mean/max|BG): {slot_summary.get('q_norm_fg_min', 0):.2f} / {slot_summary.get('q_norm_fg_mean', 0):.2f} / {slot_summary.get('q_norm_fg_max', 0):.2f} | BG={slot_summary.get('q_norm_bg', 0):.2f}")
        print(f"  Bottleneck Fisher P3 (N/P/K)   : {slot_summary.get('fisher_neck_p3', 0):.3f} / {slot_summary.get('fisher_proj_p3', 0):.3f} / {slot_summary.get('fisher_key_p3', 0):.3f}")
        print(f"  Bottleneck Fisher P4 (N/P/K)   : {slot_summary.get('fisher_neck_p4', 0):.3f} / {slot_summary.get('fisher_proj_p4', 0):.3f} / {slot_summary.get('fisher_key_p4', 0):.3f}")
        print(f"  Bottleneck Fisher P5 (N/P/K)   : {slot_summary.get('fisher_neck_p5', 0):.3f} / {slot_summary.get('fisher_proj_p5', 0):.3f} / {slot_summary.get('fisher_key_p5', 0):.3f}")
        print(f"  T=1 vs T=2 Cos(q, k_gt) Mean   : {slot_summary.get('cos_q1_vs_k_mean', 0):+.3f} -> {slot_summary.get('cos_q2_vs_k_mean', 0):+.3f}")
    if box_diag:
        print("  --- Box Learning Mechanism Diagnostics ---")
        print(f"  Boundary Contrast (P3/P4/P5)   : {box_diag.get('boundary_contrast_p3', 0):.3f} / {box_diag.get('boundary_contrast_p4', 0):.3f} / {box_diag.get('boundary_contrast_p5', 0):.3f}")
        print(f"  Intra-Box Decorr  (P3/P4/P5)   : {box_diag.get('intra_box_decorr_p3', 0):.3f} / {box_diag.get('intra_box_decorr_p4', 0):.3f} / {box_diag.get('intra_box_decorr_p5', 0):.3f}")
        print(f"  TAL GT Coverage (All / Small)  : {box_diag.get('gt_coverage_all_pct', 0):.1f}% / {box_diag.get('gt_coverage_small_pct', 0):.1f}% | Avg Pos/GT: {box_diag.get('avg_pos_per_gt', 0):.2f}")
        print(f"  DFL Entropy / Top1 / Clipping  : {box_diag.get('dfl_entropy_pos', 0):.3f} / {box_diag.get('dfl_top1_prob', 0):.3f} / {box_diag.get('dfl_clipping_rate_pct', 0):.2f}%")
        print(f"  Grad Cos(Box, Cls+Slot) P3/4/5 : {box_diag.get('grad_cos_box_vs_cls_p3', 0):+.3f} / {box_diag.get('grad_cos_box_vs_cls_p4', 0):+.3f} / {box_diag.get('grad_cos_box_vs_cls_p5', 0):+.3f}")
        print(f"  Grad Ratio ||Box||/||Cls||     : {box_diag.get('grad_ratio_box_over_cls_p3', 0):.2f} / {box_diag.get('grad_ratio_box_over_cls_p4', 0):.2f} / {box_diag.get('grad_ratio_box_over_cls_p5', 0):.2f}")
    print(f"  Latency / FPS : {overall_det['latency_ms']:.2f} ms/img ({overall_det['fps']:.1f} FPS)")
    print(f"  Output saved  : {os.path.abspath(out_dir)}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Component-by-component validation of VisDrone YOLOv11.")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config path (e.g. VisDrone_yolov11/config.yaml).")
    p.add_argument("--weights", type=str, default=r"D:\Downloads\Picture\epoch_50.pth", help="Path to checkpoint (.pth or .pt).")
    p.add_argument("--data-root", type=str, default=None, help="Path to VisDrone LMDB (overrides --config if set).")
    p.add_argument("--out-dir", type=str, default="VisDrone_yolov11/output_val_epoch50", help="Output directory.")
    p.add_argument("--imgsz", type=int, default=None, help="Input image size (defaults to config Resize or 960).")
    p.add_argument("--max-images", type=int, default=0, help="Max images to evaluate (0 = all).")
    p.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold for mAP.")
    p.add_argument("--iou-thres", type=float, default=None, help="IoU threshold for NMS (defaults to config nms_threshold or 0.5).")
    p.add_argument("--vis-conf", type=float, default=0.25, help="Confidence threshold for visual inspection.")
    p.add_argument("--split", type=str, default="val", choices=["val", "train"], help="Dataset split to evaluate ('val' or 'train').")
    p.add_argument("--split-ratio", type=float, default=None, help="Train/val split ratio (overrides --config if set).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for split (defaults to config seed or 42).")
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visdrone_component_validation(
        weights_path=args.weights,
        data_root=args.data_root,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        max_images=args.max_images,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        vis_conf=args.vis_conf,
        split=args.split,
        split_ratio=args.split_ratio,
        seed=args.seed,
        device_str=args.device,
        config_path=args.config,
    )
