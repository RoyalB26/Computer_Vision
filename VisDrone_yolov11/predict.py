"""Inference and Visualization Script for VisDrone YOLOv11 / VisDroneYOLO11Slot.

Allows running inference on single images or entire image folders:
  - Draws predicted bounding boxes, labels, and confidence scores (pred_<name>).
  - For VisDroneYOLO11Slot:
      * attn_map_result.png: 3x4 grid showing Original RGB image in panel 0,
        10 class slot attention heatmaps on panels 1-10, and background slot on panel 11.
      * all_attn.png: Composite image rendering all 10 class slots with distinct colors and legend.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from VisDrone_yolov11.model import VisDroneYOLO11, VisDroneYOLO11Slot
from VisDrone_yolov11.dataset import VISDRONE_CLASSES


# Distinct color palette for 10 VisDrone categories (uint8 RGB for PIL bboxes)
COLOR_PALETTE = [
    (255, 56, 56),    # pedestrian (red)
    (255, 157, 151),  # people (light red)
    (255, 112, 31),   # bicycle (orange)
    (255, 178, 29),   # car (amber)
    (207, 210, 49),   # van (yellow-green)
    (72, 249, 10),    # truck (green)
    (146, 204, 23),   # tricycle (lime)
    (61, 219, 134),   # awning-tricycle (teal)
    (26, 147, 243),   # bus (blue)
    (171, 71, 188),   # motorcycle (purple)
]

# 10-class RGB palette in [0, 1] for all_attn.png (synced with Visdrone/inference.py)
CLASS_COLORS_RGB = [
    (1.0, 0.2, 0.2),    # 0: pedestrian
    (1.0, 0.6, 0.0),    # 1: people
    (0.9, 0.9, 0.1),    # 2: bicycle
    (0.1, 0.8, 0.2),    # 3: car
    (0.1, 0.9, 0.8),    # 4: van
    (0.2, 0.4, 1.0),    # 5: truck
    (0.6, 0.2, 0.9),    # 6: tricycle
    (0.9, 0.2, 0.7),    # 7: awning-tricycle
    (0.6, 0.3, 0.1),    # 8: bus
    (0.8, 0.8, 0.8),    # 9: motorcycle
]


def get_spatial_attention_maps(
    attn_ori: torch.Tensor,
    resolutions: Tuple[Tuple[int, int], ...],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Reconstruct and fuse multi-scale slot attention maps (P3/P4/P5) to target_size (H, W).

    Args:
        attn_ori: (B, S, N) per-pixel slot posterior.
        resolutions: ((H3, W3), (H4, W4), (H5, W5)).
        target_size: (H, W) output spatial resolution.

    Returns:
        (B, S, H, W) fused spatial attention tensor.
    """
    bs, num_slots, _ = attn_ori.shape
    fused = None
    offset = 0
    for h, w in resolutions:
        n_i = h * w
        attn_i = attn_ori[:, :, offset : offset + n_i].view(bs, num_slots, h, w).float()
        offset += n_i
        up_i = F.interpolate(attn_i, size=target_size, mode="bilinear", align_corners=False)
        fused = up_i if fused is None else (fused + up_i)
    return fused / float(len(resolutions))


def visualize_attention_maps(
    original_img: np.ndarray,
    slot_attn_maps: torch.Tensor,
    out_path: str = "attn_map_result.png",
    overlay: bool = True,
) -> None:
    """Save 3x4 grid: panel 0 = Original RGB, panels 1-10 = 10 class heatmaps, panel 11 = background slot."""
    gray_img = np.dot(original_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    gray_rgb = np.stack([gray_img] * 3, axis=-1)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    # Panel 0: Original RGB
    axes[0].imshow(original_img)
    axes[0].set_title("Input Image (RGB)", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")

    attn_np = slot_attn_maps.detach().cpu().numpy()  # (10 or 11, H, W)

    # Panels 1-10: 10 Class Slot Attention Maps
    for idx in range(10):
        ax = axes[idx + 1]
        c_name = VISDRONE_CLASSES[idx]

        heatmap = attn_np[idx]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            norm_heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            norm_heatmap = np.zeros_like(heatmap)

        if overlay:
            ax.imshow(gray_rgb)
            ax.imshow(norm_heatmap, cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(norm_heatmap, cmap="inferno", vmin=0.0, vmax=1.0)

        ax.set_title(f"Slot {idx}: {c_name} (max p={h_max:.2f})", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

    # Panel 11: Background slot (if present)
    ax_bg = axes[11]
    if attn_np.shape[0] > 10:
        bg_map = attn_np[10]
        b_min, b_max = bg_map.min(), bg_map.max()
        norm_bg = (bg_map - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(bg_map)
        if overlay:
            ax_bg.imshow(gray_rgb)
            ax_bg.imshow(norm_bg, cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
        else:
            ax_bg.imshow(norm_bg, cmap="inferno", vmin=0.0, vmax=1.0)
        ax_bg.set_title(f"Slot 10: background (max p={b_max:.2f})", fontsize=11, fontweight="bold", pad=6)
    ax_bg.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved 3x4 slot attention map grid: {os.path.abspath(out_path)}")


def visualize_all_slots_overlay(
    original_img: np.ndarray,
    class_attn_maps: torch.Tensor,
    out_path: str = "all_attn.png",
    threshold: float = 0.25,
) -> None:
    """Composite image rendering all 10 class slots with distinct colors and legend."""
    H, W, _ = original_img.shape
    gray_img = np.dot(original_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    gray_rgb = np.stack([gray_img] * 3, axis=-1).astype(np.float32) / 255.0

    attn_np = class_attn_maps.detach().cpu().numpy()  # (10, H, W)
    composite = gray_rgb.copy()

    legend_patches = []
    for idx in range(10):
        color = CLASS_COLORS_RGB[idx]
        c_name = VISDRONE_CLASSES[idx]

        h = attn_np[idx]
        h_min, h_max = h.min(), h.max()
        norm_h = (h - h_min) / (h_max - h_min + 1e-8)

        mask = (norm_h > threshold).astype(np.float32)
        weight = (norm_h * mask)[..., None]  # (H, W, 1)

        color_layer = np.zeros((H, W, 3), dtype=np.float32)
        for ch in range(3):
            color_layer[..., ch] = color[ch]

        alpha = 0.55
        composite = composite * (1.0 - weight * alpha) + color_layer * (weight * alpha)
        legend_patches.append(mpatches.Patch(color=color, label=f"{idx}: {c_name}"))

    composite = np.clip(composite, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(composite)
    ax.set_title("VisDrone Multi-Class Slot Attention Overlay (All 10 Classes)", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.28, 1.0),
        fontsize=10,
        title="10 VisDrone Classes",
        title_fontsize=11,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved composite all-slots attention overlay: {os.path.abspath(out_path)}")


def draw_detections(
    image: Image.Image,
    detections: dict,
    class_names=VISDRONE_CLASSES,
    line_width: int = 3,
) -> Image.Image:
    """Draw bounding boxes and class labels on a PIL Image."""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    orig_w, orig_h = image.size

    boxes = detections["boxes"].cpu()
    scores = detections["scores"].cpu()
    labels = detections["labels"].cpu()

    # Load default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(len(boxes)):
        box = boxes[i].tolist()
        score = float(scores[i])
        cls_id = int(labels[i])

        x1, y1, x2, y2 = box
        # Ensure x1 <= x2, y1 <= y2 and clamp to image boundaries
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 = max(0.0, min(float(orig_w - 1), x1))
        y1 = max(0.0, min(float(orig_h - 1), y1))
        x2 = max(0.0, min(float(orig_w - 1), x2))
        y2 = max(0.0, min(float(orig_h - 1), y2))
        if x2 <= x1 or y2 <= y1:
            continue

        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        caption = f"{cls_name} {score:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw text label background
        text_bbox = draw.textbbox((x1, y1), caption, font=font) if hasattr(draw, "textbbox") else (x1, y1, x1 + 60, y1 + 12)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, max(0, y1 - text_h - 4), x1 + text_w + 4, max(text_h + 4, y1)], fill=color)
        draw.text((x1 + 2, max(0, y1 - text_h - 2)), caption, fill=(255, 255, 255), font=font)

    return draw_img


def predict_image(
    model: VisDroneYOLO11,
    image_path: str,
    img_size: int = 960,
    device: str = "cuda",
    conf_th: float = 0.25,
    nms_th: float = 0.5,
) -> Tuple[Image.Image, dict, np.ndarray, Optional[torch.Tensor]]:
    """Preprocess, predict, scale detections back to original size, and extract slot attention maps if available."""
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    orig_np = np.array(orig_img, dtype=np.uint8)

    # Resize and convert to tensor
    resized_img = TF.resize(orig_img, [img_size, img_size])
    tensor_img = TF.to_tensor(resized_img).unsqueeze(0).to(device)

    # Capture slot_info via lightweight hook if model has slot_refiner
    slot_cache: Dict[str, Any] = {}
    hook_handle = None
    if hasattr(model, "slot_refiner"):
        def _slot_hook(_mod, _inp, out):
            _, slot_info = out
            slot_cache["slot_info"] = slot_info
        hook_handle = model.slot_refiner.register_forward_hook(_slot_hook)

    try:
        results = model.predict(tensor_img, conf_threshold=conf_th, nms_threshold=nms_th)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    det = results[0]

    # Scale boxes back to original image size
    if len(det["boxes"]) > 0:
        scale_x = orig_w / float(img_size)
        scale_y = orig_h / float(img_size)
        det["boxes"][:, [0, 2]] *= scale_x
        det["boxes"][:, [1, 3]] *= scale_y

    annotated = draw_detections(orig_img, det)

    slot_attn_maps: Optional[torch.Tensor] = None
    if "slot_info" in slot_cache:
        s_info = slot_cache["slot_info"]
        slot_attn_maps = get_spatial_attention_maps(
            s_info["attn_ori"], s_info["resolutions"], target_size=(orig_h, orig_w)
        )[0]  # (11, orig_h, orig_w)

    return annotated, det, orig_np, slot_attn_maps


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 / VisDroneYOLO11Slot prediction & slot attention visualization.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Path to checkpoint weights (.pt or .pth)")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Output directory to save predictions")
    parser.add_argument("--conf-th", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms-th", type=float, default=0.5, help="NMS threshold")
    parser.add_argument("--img-size", type=int, default=960, help="Inference image resolution")
    parser.add_argument("--attn-threshold", type=float, default=0.25, help="Threshold for all_attn.png slot overlay")
    parser.add_argument("--no-overlay", action="store_true", help="Disable grayscale background overlay on attn_map_result.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Initializing VisDrone YOLOv11 on device: {args.device}...")
    if os.path.exists(args.weights) and args.weights.endswith(".pth"):
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
        else:
            sd = ckpt
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        has_slots = any(k.startswith("slot_refiner.") for k in sd.keys())
        if has_slots:
            has_logit_prior = "slot_refiner.logit_prior_scale" in sd
            model = VisDroneYOLO11Slot(num_classes=10, use_logit_prior=has_logit_prior, assigner_prior_weight=0.0)
        else:
            model = VisDroneYOLO11(num_classes=10)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"Loaded {'VisDroneYOLO11Slot' if has_slots else 'VisDroneYOLO11'} weights from {args.weights} "
            f"(missing: {len(missing)}, unexpected: {len(unexpected)})"
        )
    else:
        model = VisDroneYOLO11(num_classes=10)
        if os.path.exists(args.weights) and args.weights.endswith(".pt"):
            model.load_pretrained(args.weights)
        elif not os.path.exists(args.weights):
            print(f"Notice: weights path {args.weights} not found. Running with initial weights.")

    model.to(args.device)
    model.eval()

    # Collect images
    if os.path.isdir(args.image):
        img_paths = [
            os.path.join(args.image, f)
            for f in sorted(os.listdir(args.image))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    else:
        img_paths = [args.image]

    multi = len(img_paths) > 1
    print(f"Found {len(img_paths)} images to process.")
    for path in img_paths:
        annotated_img, det, orig_np, class_attn_maps = predict_image(
            model=model,
            image_path=path,
            img_size=args.img_size,
            device=args.device,
            conf_th=args.conf_th,
            nms_th=args.nms_th,
        )
        out_name = os.path.basename(path)
        stem = os.path.splitext(out_name)[0]
        save_path = os.path.join(args.out_dir, f"pred_{out_name}")
        annotated_img.save(save_path)
        print(f"Processed: {path} -> {len(det['boxes'])} objects detected -> Saved to: {save_path}")

        # Slot attention visualizations (attn_map_result.png & all_attn.png)
        if class_attn_maps is not None:
            attn_out = os.path.join(args.out_dir, f"{stem}_attn_map_result.png" if multi else "attn_map_result.png")
            all_out = os.path.join(args.out_dir, f"{stem}_all_attn.png" if multi else "all_attn.png")
            visualize_attention_maps(
                original_img=orig_np,
                slot_attn_maps=class_attn_maps,
                out_path=attn_out,
                overlay=not args.no_overlay,
            )
            visualize_all_slots_overlay(
                original_img=orig_np,
                class_attn_maps=class_attn_maps[:10],
                out_path=all_out,
                threshold=args.attn_threshold,
            )


if __name__ == "__main__":
    main()
