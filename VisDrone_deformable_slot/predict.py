"""Inference & Deformable Micro-Patch Visualization Script for VisDroneYOLO11DeformableSlot.

Runs end-to-end inference on a single image, a directory of images, or an LMDB dataset
and saves for each image:
  1. `pred_<stem>.jpg`    : Predicted bounding boxes, labels, and confidence scores.
  2. `patches_<stem>.jpg` : Deformable micro-patches on the original image:
                            cyan dot = Slot ref center (`ref_xy`),
                            yellow arrow = learned offset (`ref_xy -> centers_xy`),
                            lime box = final deformed patch (`centers_xy +/- radii_xy`),
                            optional dashed cyan box = initial ref box (`--show-init-box`).
  3. `compare_<stem>.png` : Side-by-side 1x2 comparison (Predictions | Deformable Ref->Patch).
  4. (Optional `--save-attn`) `attn_<stem>.png` & `all_attn_<stem>.png`.

Example usage:
  python VisDrone_deformable_slot/predict.py \
      --weights work_dirs/visdrone_deformable_slot_kaggle/best.pth \
      --source path/to/image.jpg \
      --output VisDrone_deformable_slot/predict_out

  # Or directly from val_data.lmdb:
  python VisDrone_deformable_slot/predict.py \
      --weights work_dirs/visdrone_deformable_slot_kaggle/best.pth \
      --source D:/apps/Datasets/VisDrone/val_data.lmdb \
      --index 0,5,12 \
      --output VisDrone_deformable_slot/predict_out
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.ops
import yaml

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from yolo.head import dist2bbox, make_anchors
from VisDrone_yolov11.dataset import VISDRONE_CLASSES
from VisDrone_yolov11.predict import (
    draw_detections,
    get_spatial_attention_maps,
    visualize_all_slots_overlay,
    visualize_attention_maps,
)
from VisDrone_deformable_slot.model import VisDroneYOLO11DeformableSlot


def _resolve_config_path(weights_path: str, config_arg: Optional[str]) -> Optional[str]:
    """Auto-discover config_archived.yaml next to weights if --config is not specified."""
    if config_arg:
        return config_arg
    w_dir = Path(weights_path).resolve().parent
    for cand in ("config_archived.yaml", "config.yaml", "config_kaggle.yaml"):
        p = w_dir / cand
        if p.exists():
            return str(p)
    default_cfg = Path(__file__).resolve().parent / "config_kaggle.yaml"
    return str(default_cfg) if default_cfg.exists() else None


def load_deformable_model(
    weights_path: str,
    config_path: Optional[str],
    device: torch.device,
) -> Tuple[VisDroneYOLO11DeformableSlot, int]:
    """Build VisDroneYOLO11DeformableSlot from config and load checkpoint weights."""
    cfg_path = _resolve_config_path(weights_path, config_path)
    model_kwargs: Dict[str, Any] = {}
    img_size = 1280

    if cfg_path and os.path.exists(cfg_path):
        print(f"[*] Loading config from: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_cfg = dict(cfg.get("model", {}))
        model_cfg.pop("type", None)
        model_cfg["pretrained"] = None
        model_kwargs.update(model_cfg)

        # Read high-res input size from val pipeline if available
        val_pipe = cfg.get("data", {}).get("val", {}).get("dataset", {}).get("pipeline", [])
        for step in val_pipe:
            if isinstance(step, dict) and step.get("type") == "Resize" and "size" in step:
                sz = step["size"]
                img_size = int(sz[0] if isinstance(sz, (list, tuple)) else sz)
                break

    model = VisDroneYOLO11DeformableSlot(**model_kwargs)

    print(f"[*] Loading weights from: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
    if missing or unexpected:
        print(f"[!] State dict note: {len(missing)} missing, {len(unexpected)} unexpected keys.")

    model.to(device).eval()
    return model, img_size


@torch.no_grad()
def run_inference_with_patch_info(
    model: VisDroneYOLO11DeformableSlot,
    img_rgb: np.ndarray,
    img_size: int,
    device: torch.device,
    conf_threshold: float,
    nms_threshold: float,
    max_det: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Single forward pass returning both NMS detections and `slot_info` (including `patch_info`)."""
    orig_h, orig_w = img_rgb.shape[:2]

    # Resize to high-res model input (e.g. 1280x1280) using OpenCV bilinear (matches dataset.py)
    resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    inp = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device)

    feats, boxes_raw, scores_raw, slot_info, (lr_h, lr_w) = model._forward_head_with_slots(inp)

    decoded_boxes, _, _ = model._build_anchors_and_decode(
        feats, boxes_raw, slot_info, (lr_h, lr_w)
    )
    decoded_boxes = decoded_boxes[0]                 # (N_total, 4) in lr_size coordinates
    probs = scores_raw.sigmoid().transpose(1, 2)[0]  # (N_total, C)

    max_scores, labels = probs.max(dim=-1)
    valid_mask = max_scores >= conf_threshold

    if not valid_mask.any():
        det = {
            "boxes": torch.empty((0, 4), device=device),
            "scores": torch.empty((0,), device=device),
            "labels": torch.empty((0,), dtype=torch.long, device=device),
        }
        return det, slot_info

    cand_boxes = decoded_boxes[valid_mask]
    cand_scores = max_scores[valid_mask]
    cand_labels = labels[valid_mask]

    if len(cand_scores) > 1000:
        topk_idx = torch.topk(cand_scores, 1000).indices
        cand_boxes = cand_boxes[topk_idx]
        cand_scores = cand_scores[topk_idx]
        cand_labels = cand_labels[topk_idx]

    # Scale boxes directly from lr_size to original image pixel coordinates
    scale_x = float(orig_w) / float(lr_w)
    scale_y = float(orig_h) / float(lr_h)
    cand_boxes[:, [0, 2]] = (cand_boxes[:, [0, 2]] * scale_x).clamp(0.0, float(orig_w))
    cand_boxes[:, [1, 3]] = (cand_boxes[:, [1, 3]] * scale_y).clamp(0.0, float(orig_h))

    keep = torchvision.ops.batched_nms(cand_boxes, cand_scores, cand_labels, nms_threshold)
    keep = keep[:max_det]

    det = {
        "boxes": cand_boxes[keep],
        "scores": cand_scores[keep],
        "labels": cand_labels[keep],
    }
    return det, slot_info


def draw_deformable_patches(
    img_rgb: np.ndarray,
    patch_info: Dict[str, Any],
    base_patch_radius: float = 0.035,
    max_patches: int = 0,
    show_init_box: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Render Deformable Ref Centers, Offset Arrows, and Shifted Patch Boxes on the RGB image."""
    orig_h, orig_w = img_rgb.shape[:2]
    canvas = img_rgb.copy()

    ref_xy = patch_info["ref_xy"][0].detach().cpu().numpy()          # (K, 2) in [-1, 1]
    centers_xy = patch_info["centers_xy"][0].detach().cpu().numpy()  # (K, 2) in [-1, 1]
    radii_xy = patch_info["radii_xy"][0].detach().cpu().numpy()      # (K, 2) half-size in [-1, 1]
    top_scores = patch_info["top_scores"][0].detach().cpu().numpy()  # (K,) in [0, 1]
    h3, w3 = patch_info["grid_hw"]

    k_total = len(ref_xy)
    order = np.argsort(-top_scores)
    if max_patches > 0 and max_patches < k_total:
        order = order[:max_patches]

    # Compute offset stats in P3 cell units & scale stats
    dx_cells = (centers_xy[:, 0] - ref_xy[:, 0]) * (w3 * 0.5)
    dy_cells = (centers_xy[:, 1] - ref_xy[:, 1]) * (h3 * 0.5)
    offset_cells = np.sqrt(dx_cells ** 2 + dy_cells ** 2)
    scale_fac = radii_xy / max(base_patch_radius, 1e-6)

    stats = {
        "k_total": float(k_total),
        "k_drawn": float(len(order)),
        "offset_mean_cells": float(offset_cells.mean()),
        "offset_max_cells": float(offset_cells.max()),
        "scale_mean": float(scale_fac.mean()),
    }

    # Draw from lowest-score to highest-score so top patches are on top
    for idx in reversed(order):
        rx_n, ry_n = ref_xy[idx]
        cx_n, cy_n = centers_xy[idx]
        rad_x_n, rad_y_n = radii_xy[idx]

        # Map [-1, 1] -> [0, orig_w/orig_h]
        rx_px = (rx_n + 1.0) * 0.5 * orig_w
        ry_px = (ry_n + 1.0) * 0.5 * orig_h
        cx_px = (cx_n + 1.0) * 0.5 * orig_w
        cy_px = (cy_n + 1.0) * 0.5 * orig_h

        half_w = rad_x_n * 0.5 * orig_w
        half_h = rad_y_n * 0.5 * orig_h

        # Optional initial unshifted Ref Box (dashed cyan)
        if show_init_box:
            init_hw = base_patch_radius * 0.5 * orig_w
            init_hh = base_patch_radius * 0.5 * orig_h
            ix1 = int(max(0, round(rx_px - init_hw)))
            iy1 = int(max(0, round(ry_px - init_hh)))
            ix2 = int(min(orig_w - 1, round(rx_px + init_hw)))
            iy2 = int(min(orig_h - 1, round(ry_px + init_hh)))
            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (0, 220, 255), 1, cv2.LINE_AA)

        # Final deformed sampling box (lime green)
        x1 = int(max(0, round(cx_px - half_w)))
        y1 = int(max(0, round(cy_px - half_h)))
        x2 = int(min(orig_w - 1, round(cx_px + half_w)))
        y2 = int(min(orig_h - 1, round(cy_px + half_h)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 255, 80), 1, cv2.LINE_AA)

        # Offset arrow: Ref center -> Shifted center (yellow)
        p_ref = (int(round(rx_px)), int(round(ry_px)))
        p_ctr = (int(round(cx_px)), int(round(cy_px)))
        cv2.arrowedLine(canvas, p_ref, p_ctr, (255, 230, 0), 1, cv2.LINE_AA, tipLength=0.35)

        # Initial Slot Ref point (cyan dot)
        cv2.circle(canvas, p_ref, 2, (0, 220, 255), -1, cv2.LINE_AA)

    return canvas, stats


def save_side_by_side(
    pred_img_rgb: np.ndarray,
    patch_img_rgb: np.ndarray,
    det: Dict[str, torch.Tensor],
    stats: Dict[str, float],
    out_path: str,
    show_init_box: bool,
) -> None:
    """Save a crisp 1x2 side-by-side comparison of Predictions vs Deformable Ref/Patch Boxes."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    n_det = len(det["boxes"])
    axes[0].imshow(pred_img_rgb)
    axes[0].set_title(f"Predicted Detections ({n_det} boxes)", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")

    axes[1].imshow(patch_img_rgb)
    axes[1].set_title(
        f"Deformable Micro-Patches ({int(stats['k_drawn'])}/{int(stats['k_total'])} patches | "
        f"offset={stats['offset_mean_cells']:.2f} P3 cells | scale={stats['scale_mean']:.2f}x)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    axes[1].axis("off")

    legend_handles = [
        mpatches.Patch(color=(0.0, 0.86, 1.0), label="Ref Center (Slot P3 peak)"),
        mpatches.Patch(color=(1.0, 0.90, 0.0), label="Learned Offset (ref -> center)"),
        mpatches.Patch(color=(0.16, 1.0, 0.31), label="Deformed Patch Box (32x32 sample)"),
    ]
    if show_init_box:
        legend_handles.insert(1, mpatches.Patch(edgecolor=(0.0, 0.86, 1.0), facecolor="none", label="Initial Ref Box (unshifted)"))
    axes[1].legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_routing_comparison(
    old_patch_rgb: np.ndarray,
    unc_patch_rgb: np.ndarray,
    old_stats: Dict[str, float],
    unc_stats: Dict[str, float],
    out_path: str,
    floor: float,
) -> None:
    """Save 1x2 comparison of Old Saliency Routing vs Uncertainty-Guided Routing."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    axes[0].imshow(old_patch_rgb)
    axes[0].set_title(
        f"Old Routing: fg_sal * (0.25 + w_cls)\n"
        f"({int(old_stats['k_drawn'])}/{int(old_stats['k_total'])} patches | "
        f"offset={old_stats['offset_mean_cells']:.2f} cells | scale={old_stats['scale_mean']:.2f}x)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    axes[0].axis("off")

    axes[1].imshow(unc_patch_rgb)
    axes[1].set_title(
        f"Uncertainty Routing: floor={floor:.2f} + {1.0 - floor:.2f} * max(u_fg, fg*u_cls)\n"
        f"({int(unc_stats['k_drawn'])}/{int(unc_stats['k_total'])} patches | "
        f"offset={unc_stats['offset_mean_cells']:.2f} cells | scale={unc_stats['scale_mean']:.2f}x)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _iter_inputs(source: str, indices: Optional[List[int]]) -> List[Tuple[str, np.ndarray]]:
    """Load RGB numpy images from a single file, a folder, or a VisDrone LMDB database."""
    src_path = Path(source)
    items: List[Tuple[str, np.ndarray]] = []

    if src_path.is_dir() and (src_path / "data.mdb").exists():
        import lmdb
        import pickle

        env = lmdb.open(str(src_path), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            length_raw = txn.get(b"__len__")
            total = int(length_raw.decode("utf-8")) if length_raw else 0
            pick_ids = indices if indices else [0]
            for idx in pick_ids:
                if idx < 0 or idx >= total:
                    print(f"[!] Skip out-of-range LMDB index {idx} (total={total})")
                    continue
                raw = txn.get(f"{idx:08d}".encode("ascii"))
                if raw is None:
                    raw = txn.get(str(idx).encode("ascii"))
                if raw is None:
                    print(f"[!] Key for index {idx} not found in LMDB.")
                    continue
                sample = pickle.loads(raw)
                img_data = sample["image"] if isinstance(sample, dict) else sample[0]
                if isinstance(img_data, (bytes, bytearray)):
                    arr = np.frombuffer(img_data, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif isinstance(img_data, Image.Image):
                    rgb = np.array(img_data.convert("RGB"))
                elif isinstance(img_data, np.ndarray):
                    rgb = img_data
                else:
                    raise TypeError(f"Unsupported LMDB image type: {type(img_data)}")
                items.append((f"lmdb_{idx:04d}", rgb))
        env.close()
        return items

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if src_path.is_file():
        img = np.array(Image.open(src_path).convert("RGB"))
        items.append((src_path.stem, img))
    elif src_path.is_dir():
        files = sorted(p for p in src_path.iterdir() if p.suffix.lower() in valid_exts)
        if indices:
            files = [files[i] for i in indices if 0 <= i < len(files)]
        for p in files:
            items.append((p.stem, np.array(Image.open(p).convert("RGB"))))
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="VisDroneYOLO11DeformableSlot Inference & Patch Visualizer")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None, help="Path to config (.yaml). Auto-detected if omitted.")
    parser.add_argument("--source", type=str, required=True, help="Image file, folder, or VisDrone LMDB directory.")
    parser.add_argument("--index", type=str, default=None, help="Comma-separated indices for LMDB/folder (e.g. '0,5,12').")
    parser.add_argument("--output", type=str, default="VisDrone_deformable_slot/predict_out", help="Output directory.")
    parser.add_argument("--img-size", type=int, default=0, help="Override high-res input size (0 = read from config).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--nms", type=float, default=0.5, help="NMS IoU threshold (default: 0.5).")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image (default: 300).")
    parser.add_argument("--max-patches", type=int, default=0, help="Max patches to draw (0 = draw all K patches).")
    parser.add_argument("--show-init-box", action="store_true", help="Also draw initial unshifted ref boxes (cyan).")
    parser.add_argument("--uncertainty-routing", action="store_true", help="Use Uncertainty-Guided Routing for patch selection.")
    parser.add_argument("--uncertainty-floor", type=float, default=0.2, help="Floor weight for clear objects in uncertainty routing (default: 0.2).")
    parser.add_argument("--compare-routing", action="store_true", help="Save side-by-side routing_compare_<stem>.png (Old vs Uncertainty).")
    parser.add_argument("--save-attn", action="store_true", help="Also save 3x4 slot attention grid & all-slots overlay.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    model, cfg_img_size = load_deformable_model(args.weights, args.config, device)
    img_size = args.img_size if args.img_size > 0 else cfg_img_size
    base_radius = model.patch_injector.base_patch_radius if model.patch_injector is not None else 0.035

    if model.patch_injector is not None:
        model.patch_injector.use_uncertainty_routing = bool(args.uncertainty_routing)
        model.patch_injector.uncertainty_floor = float(args.uncertainty_floor)

    indices = [int(x.strip()) for x in args.index.split(",") if x.strip()] if args.index else None
    samples = _iter_inputs(args.source, indices)
    routing_tag = f"uncertainty(floor={args.uncertainty_floor})" if args.uncertainty_routing else "old"
    print(f"[*] Running inference on {len(samples)} image(s) at HR={img_size}, backbone={model.backbone_size}, conf={args.conf}, routing={routing_tag}...")

    for stem, img_rgb in samples:
        det, slot_info = run_inference_with_patch_info(
            model=model,
            img_rgb=img_rgb,
            img_size=img_size,
            device=device,
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            max_det=args.max_det,
        )

        # 1. Save predicted detections image
        pil_pred = draw_detections(Image.fromarray(img_rgb), det, line_width=2)
        pred_path = os.path.join(args.output, f"pred_{stem}.jpg")
        pil_pred.save(pred_path, quality=95)

        # 2. Save deformable ref + shifted patches image & 1x2 comparison
        if "patch_info" in slot_info:
            patch_img_rgb, stats = draw_deformable_patches(
                img_rgb=img_rgb,
                patch_info=slot_info["patch_info"],
                base_patch_radius=base_radius,
                max_patches=args.max_patches,
                show_init_box=args.show_init_box,
            )
            patch_path = os.path.join(args.output, f"patches_{stem}.jpg")
            Image.fromarray(patch_img_rgb).save(patch_path, quality=95)

            compare_path = os.path.join(args.output, f"compare_{stem}.png")
            save_side_by_side(
                pred_img_rgb=np.array(pil_pred),
                patch_img_rgb=patch_img_rgb,
                det=det,
                stats=stats,
                out_path=compare_path,
                show_init_box=args.show_init_box,
            )
            print(
                f"[+] {stem}: {len(det['boxes'])} detections | "
                f"{int(stats['k_drawn'])}/{int(stats['k_total'])} patches "
                f"(offset={stats['offset_mean_cells']:.2f} cells, scale={stats['scale_mean']:.2f}x) -> {compare_path}"
            )

            # Optional side-by-side Old vs Uncertainty routing comparison
            if args.compare_routing and model.patch_injector is not None:
                prev_mode = model.patch_injector.use_uncertainty_routing
                model.patch_injector.use_uncertainty_routing = not prev_mode
                _, alt_slot_info = run_inference_with_patch_info(
                    model=model,
                    img_rgb=img_rgb,
                    img_size=img_size,
                    device=device,
                    conf_threshold=args.conf,
                    nms_threshold=args.nms,
                    max_det=args.max_det,
                )
                model.patch_injector.use_uncertainty_routing = prev_mode
                alt_patch_rgb, alt_stats = draw_deformable_patches(
                    img_rgb=img_rgb,
                    patch_info=alt_slot_info["patch_info"],
                    base_patch_radius=base_radius,
                    max_patches=args.max_patches,
                    show_init_box=args.show_init_box,
                )
                old_rgb, old_st = (alt_patch_rgb, alt_stats) if prev_mode else (patch_img_rgb, stats)
                unc_rgb, unc_st = (patch_img_rgb, stats) if prev_mode else (alt_patch_rgb, alt_stats)
                rc_path = os.path.join(args.output, f"routing_compare_{stem}.png")
                save_routing_comparison(old_rgb, unc_rgb, old_st, unc_st, rc_path, args.uncertainty_floor)
                print(f"    -> Saved routing comparison: {rc_path}")
        else:
            print(f"[+] {stem}: {len(det['boxes'])} detections (deformable patch disabled) -> {pred_path}")

        # 3. Optional Slot Attention maps
        if args.save_attn and "attn_ori" in slot_info and "resolutions" in slot_info:
            orig_h, orig_w = img_rgb.shape[:2]
            attn_maps = get_spatial_attention_maps(
                slot_info["attn_ori"],
                slot_info["resolutions"],
                target_size=(orig_h, orig_w),
            )[0]
            visualize_attention_maps(
                original_img=img_rgb,
                slot_attn_maps=attn_maps,
                out_path=os.path.join(args.output, f"attn_{stem}.png"),
                overlay=True,
            )
            visualize_all_slots_overlay(
                original_img=img_rgb,
                class_attn_maps=attn_maps[:10],
                out_path=os.path.join(args.output, f"all_attn_{stem}.png"),
                threshold=0.25,
            )

    print(f"[*] Done! Results saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
