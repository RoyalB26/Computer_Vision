"""Visualize 4-image Class-Aware Object-Centered Mosaic samples with GT bounding boxes.

Example:
  python VisDrone_yolov11/visualize_mosaic.py --num 4
  python VisDrone_yolov11/visualize_mosaic.py --config VisDrone_yolov11/config.yaml --num 6 --raw-mosaic
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from VisDrone_yolov11.dataset import VISDRONE_CLASSES, VisDroneLMDBDataset
from VisDrone_yolov11.predict import COLOR_PALETTE

RARE_CLASS_IDS = {2, 6, 7, 8}  # bicycle, tricycle, awning-tricycle, bus


def draw_gt_boxes(img_tensor: torch.Tensor, target: dict) -> Image.Image:
    """Convert [3, H, W] float tensor in [0, 1] to PIL Image and draw GT boxes (normalized cxcywh)."""
    arr = (img_tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    boxes = target.get("boxes", torch.empty((0, 4)))
    labels = target.get("labels", torch.empty((0,), dtype=torch.long))

    for i in range(len(boxes)):
        cx, cy, bw, bh = [float(v) for v in boxes[i].tolist()]
        cls_id = int(labels[i].item())
        x1 = max(0.0, min(w - 1.0, (cx - 0.5 * bw) * w))
        y1 = max(0.0, min(h - 1.0, (cy - 0.5 * bh) * h))
        x2 = max(0.0, min(w - 1.0, (cx + 0.5 * bw) * w))
        y2 = max(0.0, min(h - 1.0, (cy + 0.5 * bh) * h))
        if x2 <= x1 or y2 <= y1:
            continue

        is_rare = cls_id in RARE_CLASS_IDS
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        lw = 3 if is_rare else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        # Draw text tag for rare classes (avoid cluttering dense car/pedestrian crowds)
        if is_rare:
            name = VISDRONE_CLASSES[cls_id] if cls_id < len(VISDRONE_CLASSES) else f"c{cls_id}"
            tb = draw.textbbox((x1, y1), name, font=font) if hasattr(draw, "textbbox") else (x1, y1, x1 + 50, y1 + 12)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0.0, y1 - th - 3)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 3], fill=color)
            draw.text((x1 + 2, ty + 1), name, fill=(0, 0, 0), font=font)

    return pil_img


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Class-Aware Mosaic4 augmented training samples.")
    parser.add_argument("--config", type=str, default="VisDrone_yolov11/config.yaml", help="Path to YAML config")
    parser.add_argument("--num", type=int, default=4, help="Number of Mosaic images to generate")
    parser.add_argument("--out-dir", type=str, default="VisDrone_yolov11/mosaic_preview", help="Output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rare-prob", type=float, default=None, help="Override mosaic_rare_prob (default from config)")
    parser.add_argument("--raw-mosaic", action="store_true", help="Show pure Mosaic4 without Flip/Scale/ColorJitter")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg["data"]["train"]["dataset"]

    rare_prob = float(args.rare_prob) if args.rare_prob is not None else float(ds_cfg.get("mosaic_rare_prob", 0.85))
    ds_type = str(ds_cfg.get("type", "VisDroneLMDBDataset"))
    ds_kwargs = dict(
        data_root=ds_cfg["data_root"],
        pipeline=ds_cfg.get("pipeline"),
        num_classes=int(ds_cfg.get("num_classes", 10)),
        include_boxes=True,
        cache_images=None,  # Fast startup: don't cache all 6,471 images for a quick preview
        augment=True,
        mosaic_prob=1.0,    # Force 100% Mosaic for preview
        mosaic_rare_prob=rare_prob,
    )
    if ds_type == "VisDroneDualResLMDBDataset":
        from VisDrone_deformable_slot.dataset import VisDroneDualResLMDBDataset

        for k in ("drone_aug", "hsv_prob", "blur_prob", "haze_gamma_prob", "noise_prob"):
            if k in ds_cfg:
                ds_kwargs[k] = ds_cfg[k]
        ds = VisDroneDualResLMDBDataset(**ds_kwargs)
    else:
        ds = VisDroneLMDBDataset(**ds_kwargs)

    os.makedirs(args.out_dir, exist_ok=True)
    n_total = len(ds)
    chosen_indices = random.sample(range(n_total), min(args.num, n_total))

    grid_imgs = []
    grid_titles = []

    for i, idx in enumerate(chosen_indices):
        if args.raw_mosaic:
            if hasattr(ds, "_apply_mosaic4_uint8"):
                rgb, target = ds._apply_mosaic4_uint8(idx)
                img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
            else:
                img_t, target = ds._apply_mosaic4(idx)
        else:
            sample = ds.prepare_data(idx)
            img_t, target = sample["image"], sample["target"]

        vis_img = draw_gt_boxes(img_t, target)
        out_path = os.path.join(args.out_dir, f"mosaic_{i:02d}.png")
        vis_img.save(out_path)

        cnts = target["counts"].int().tolist()
        rare_summary = ", ".join(
            f"{VISDRONE_CLASSES[c]}:{cnts[c]}" for c in sorted(RARE_CLASS_IDS) if cnts[c] > 0
        ) or "none"
        title = f"#{i} (idx={idx}) | total={len(target['boxes'])} boxes | rare: {rare_summary}"
        print(f"[+] Saved {os.path.abspath(out_path)} | {title}")
        grid_imgs.append(np.array(vis_img))
        grid_titles.append(title)

    # Save 2x2 (or Nx2) overview grid with class color legend
    cols = 2
    rows = max(1, (len(grid_imgs) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, img_np, ttl in zip(axes, grid_imgs, grid_titles):
        ax.imshow(img_np)
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(grid_imgs):]:
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(
            color=tuple(c / 255.0 for c in COLOR_PALETTE[cid]),
            label=f"{cid}: {VISDRONE_CLASSES[cid]}" + (" [RARE]" if cid in RARE_CLASS_IDS else ""),
        )
        for cid in range(len(VISDRONE_CLASSES))
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    grid_path = os.path.join(args.out_dir, "mosaic_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved summary grid: {os.path.abspath(grid_path)}")


if __name__ == "__main__":
    main()
