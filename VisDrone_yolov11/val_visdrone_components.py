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
import torchvision.transforms.functional as TF

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


def load_visdrone_model(weights_path: str, device: torch.device) -> Tuple[torch.nn.Module, bool]:
    """Auto-detect VisDroneYOLO11Slot vs VisDroneYOLO11 from checkpoint and load weights."""
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
    model = VisDroneYOLO11Slot(num_classes=10) if has_slots else VisDroneYOLO11(num_classes=10)
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


@torch.no_grad()
def run_visdrone_component_validation(
    weights_path: str,
    data_root: str,
    out_dir: str,
    imgsz: int = 960,
    max_images: int = 0,
    conf_thres: float = 0.001,
    iou_thres: float = 0.5,
    vis_conf: float = 0.25,
    split_ratio: float = 0.8,
    seed: int = 42,
    device_str: str = "auto",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else ("cpu" if device_str == "auto" else device_str))

    print("=" * 80)
    print(f"VisDrone YOLOv11 Component-by-Component Validation | Device: {device} | imgsz: {imgsz}")
    print("=" * 80)

    model, has_slots = load_visdrone_model(weights_path, device)
    probe = YOLOv11ComponentProbe(model.net)

    # Optional hook on slot_refiner to inspect modulated cls_feats & slot attention
    slot_cache: Dict[str, Any] = {}
    slot_handle = None
    if has_slots and hasattr(model, "slot_refiner"):
        def _slot_hook(_mod, _inp, out):
            cls_feats, slot_info = out
            slot_cache["cls_feats"] = cls_feats
            slot_cache["slot_info"] = slot_info
        slot_handle = model.slot_refiner.register_forward_hook(_slot_hook)

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

    # Load VisDrone val dataset
    ds = VisDroneLMDBDataset(data_root=data_root, split="val", split_ratio=split_ratio, seed=seed, include_boxes=True)
    num_samples = len(ds) if max_images <= 0 else min(len(ds), max_images)
    print(f"[Dataset] VisDrone val samples: {num_samples} / {len(ds)}")

    metric_keys = [
        "act_mean", "act_std", "neg_frac", "zero_frac", "l2_mean",
        "small_ratio", "medium_ratio", "large_ratio", "all_ratio",
        "full_fisher", "pca_fisher", "class_cos_sep", "fg_mass_pct",
    ]
    comp_acc: Dict[str, Dict[str, List[float]]] = {c: {k: [] for k in metric_keys} for c in component_order}
    head_scale_share: Dict[str, List[float]] = {"p3": [], "p4": [], "p5": []}
    slot_stats: Dict[str, List[float]] = {"slot_fg_mass_pct": [], "slot_share_p3": [], "slot_share_p4": [], "slot_share_p5": [], "slot_cos_sep": []}
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
            sim = slots_fg @ slots_fg.T
            off_diag = sim[~torch.eye(10, dtype=torch.bool, device=sim.device)]
            slot_stats["slot_cos_sep"].append(float((1.0 - off_diag).mean().item()))

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

        # TAL coverage & DFL entropy/clipping
        if len(gt_xyxy) > 0:
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
    if has_slots and slot_stats["slot_cos_sep"]:
        slot_summary = {
            "slot_cos_sep": round(float(np.mean(slot_stats["slot_cos_sep"])), 4),
            "slot_share_p3_pct": round(float(np.mean(slot_stats["slot_share_p3"])), 2),
            "slot_share_p4_pct": round(float(np.mean(slot_stats["slot_share_p4"])), 2),
            "slot_share_p5_pct": round(float(np.mean(slot_stats["slot_share_p5"])), 2),
        }

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
        print(f"  Slot Cos Sep  : {slot_summary['slot_cos_sep']:.4f} | Share P3/P4/P5: {slot_summary['slot_share_p3_pct']:.1f}% / {slot_summary['slot_share_p4_pct']:.1f}% / {slot_summary['slot_share_p5_pct']:.1f}%")
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
    p.add_argument("--weights", type=str, default=r"D:\Downloads\Picture\epoch_50.pth", help="Path to checkpoint (.pth or .pt).")
    p.add_argument("--data-root", type=str, default="D:/apps/Datasets/VisDrone/VisDrone_data.lmdb", help="Path to VisDrone LMDB.")
    p.add_argument("--out-dir", type=str, default="VisDrone_yolov11/output_val_epoch50", help="Output directory.")
    p.add_argument("--imgsz", type=int, default=960, help="Input image size.")
    p.add_argument("--max-images", type=int, default=0, help="Max val images (0 = all).")
    p.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold for mAP.")
    p.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for NMS.")
    p.add_argument("--vis-conf", type=float, default=0.25, help="Confidence threshold for visual inspection.")
    p.add_argument("--split-ratio", type=float, default=0.8, help="Train/val split ratio.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split.")
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
        split_ratio=args.split_ratio,
        seed=args.seed,
        device_str=args.device,
    )
