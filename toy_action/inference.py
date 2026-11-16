"""Inference and Deformable Spatial Attention Visualization script for Toy Action.

Loads a sample folder containing 16 frames (or .mp4), runs `DeformableMambaActionModel`,
overlays the 10x10 Mamba spatial attention heatmap (and optional deformable sampling offsets
from `DeformableFeatureHead`) onto the 16 frames, arranges them into a 4x4 montage grid,
and prints predicted action classes.

Usage:
    python toy_action/inference.py --input-dir toy_action/sample_output
    python toy_action/inference.py --input-dir toy_action/sample_output --checkpoint work_dirs/toy_action_deformable_mamba_exp/best.pth
"""

import argparse
import glob
import importlib
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from BaseModel.model import build_model
from BaseModel.utils import Config
from toy_action.dataset import IDX_TO_CLASS


def load_input_frames_and_masks(
    input_dir: str,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[str]]:
    """Loads 16 RGB frames and optional 16 bg_masks from input directory."""
    png_files = sorted(glob.glob(os.path.join(input_dir, "frame_*.png")))
    if len(png_files) >= 16:
        png_files = png_files[:16]
        frames = []
        for p in png_files:
            bgr = cv2.imread(p)
            if bgr is None:
                raise ValueError(f"Unable to read image: {p}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        mp4_files = glob.glob(os.path.join(input_dir, "*.mp4"))
        if not mp4_files:
            raise FileNotFoundError(f"No 16 frame_*.png or .mp4 found in {input_dir}")
        video_path = mp4_files[0]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < 16:
            ret, bgr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) < 16:
            raise ValueError(f"Video {video_path} has fewer than 16 frames ({len(frames)})")

    bg_masks = None
    mask_files = sorted(glob.glob(os.path.join(input_dir, "bg_mask_*.png")))
    if len(mask_files) >= 16:
        bg_masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_files[:16]]

    gt_label = None
    label_path = os.path.join(input_dir, "label.txt")
    if os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Label:"):
                    gt_label = line.replace("Label:", "").strip()
                    break

    return frames, bg_masks, gt_label


def overlay_spatial_heatmap(
    rgb_frame: np.ndarray,
    attn_2d: np.ndarray,
    gt_bg_mask_2d: Optional[np.ndarray] = None,
    filter_background: bool = False,
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlays the 10x10 Mamba spatial attention heatmap onto a 224x224 RGB frame."""
    h, w, _ = rgb_frame.shape
    heat = cv2.resize(attn_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    h_min, h_max = float(heat.min()), float(heat.max())
    heat_norm = (heat - h_min) / (h_max - h_min + 1e-6) if h_max - h_min > 1e-6 else np.zeros((h, w), dtype=np.float32)

    heatmap_bgr = cv2.applyColorMap((heat_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    base = rgb_frame.astype(np.float32)
    if filter_background and gt_bg_mask_2d is not None:
        if gt_bg_mask_2d.shape != (h, w):
            gt_bg_mask_2d = cv2.resize(gt_bg_mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
        fg_alpha = (gt_bg_mask_2d < 128).astype(np.float32)[..., None]
        base = base * fg_alpha + 255.0 * (1.0 - fg_alpha)

    blend_w = np.clip(heat_norm[..., None], 0.0, 1.0) * alpha
    blended = base * (1.0 - blend_w) + heatmap_rgb * blend_w
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_16_frame_grid(
    frames_overlaid: List[np.ndarray],
    summary_text: str,
    rows: int = 4,
    cols: int = 4,
    cell_size: int = 224,
) -> np.ndarray:
    """Combines 16 frames into a 4x4 grid with frame indices and a bottom prediction bar."""
    grid_h = rows * cell_size
    grid_w = cols * cell_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames_overlaid[:16]):
        r, c = idx // cols, idx % cols
        img = cv2.resize(frame, (cell_size, cell_size))
        cv2.rectangle(img, (4, 4), (58, 26), (0, 0, 0), -1)
        cv2.putText(img, f"F{idx:02d}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        grid[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = img

    footer = np.full((64, grid_w, 3), 245, dtype=np.uint8)
    cv2.putText(
        footer,
        "DeformableFeatureHead (10x10 -> 100 pts) + Mamba Spatial Attention:",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        f"Pred: {summary_text[:95]}",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([grid, footer])


def run_inference(
    input_dir: str,
    config_path: str = "toy_action/config.yaml",
    checkpoint_path: Optional[str] = None,
    output_path: str = "toy_action/sample_output/deformable_mamba_inference_grid.png",
    threshold: float = 0.35,
    filter_background: bool = False,
    device_str: str = "cuda",
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Loading configuration from: {config_path}")
    cfg = Config.fromfile(config_path)

    custom_imports = getattr(cfg, "custom_imports", None)
    if custom_imports:
        for mod in custom_imports:
            importlib.import_module(mod)

    print(f"Building model: {cfg.model.type} ...")
    model = build_model(cfg.model)
    model.to(device)
    model.eval()

    ckpt = checkpoint_path or getattr(cfg, "checkpoint", None)
    if ckpt and os.path.isfile(ckpt):
        print(f"Loading checkpoint weights from: {ckpt}")
        checkpoint_data = torch.load(ckpt, map_location=device)
        state_dict = checkpoint_data.get("state_dict", checkpoint_data)
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint specified or found. Using model default/pretrained weights.")

    frames_rgb, gt_bg_masks, gt_label = load_input_frames_and_masks(input_dir)
    print(f"Loaded {len(frames_rgb)} frames from: {input_dir}")
    if gt_label:
        print(f"Ground-Truth Label: {gt_label}")

    resized_frames = [cv2.resize(f, (224, 224)) for f in frames_rgb]
    video_np = np.stack(resized_frames, axis=0)
    video_tensor = torch.from_numpy(video_np).float().div(255.0).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        preds_dict = model.predict(video_tensor, threshold=threshold)

    probs = preds_dict["probs"].squeeze(0).cpu().numpy()
    pred_labels = preds_dict.get("label_strings", [""])[0]
    top_indices = np.argsort(probs)[::-1][:8]

    print("=" * 60)
    print("[DEFORMABLE-MAMBA INFERENCE PREDICTIONS]")
    print(f"Active Predicted Labels (threshold >= {threshold}):")
    print(f"  --> {pred_labels if pred_labels else '(None above threshold)'}")
    print("\nTop 8 Highest Probability Classes:")
    for rank, idx in enumerate(top_indices, 1):
        cls_name = IDX_TO_CLASS[idx]
        conf = probs[idx] * 100.0
        status = "[DETECTED]" if probs[idx] >= threshold else "          "
        print(f"  {rank}. {status} {cls_name:<16} : {conf:6.2f}%")
    print("=" * 60)

    spatial_attn = preds_dict.get("spatial_attn", None)
    if spatial_attn is not None:
        attn_2d = spatial_attn.squeeze(0).cpu().numpy()  # (10, 10)
    else:
        attn_2d = np.ones((10, 10), dtype=np.float32)

    overlaid_frames = [
        overlay_spatial_heatmap(
            rgb_frame=resized_frames[t],
            attn_2d=attn_2d,
            gt_bg_mask_2d=gt_bg_masks[t] if gt_bg_masks is not None else None,
            filter_background=filter_background,
        )
        for t in range(16)
    ]

    summary = pred_labels if pred_labels else ", ".join(f"{IDX_TO_CLASS[i]}({probs[i]*100:.0f}%)" for i in top_indices[:3])
    grid_img = create_16_frame_grid(overlaid_frames, summary_text=summary, rows=4, cols=4, cell_size=224)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    print(f"Saved inference grid to: {output_path} ({grid_img.shape[1]}x{grid_img.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description="Deformable-Mamba Toy Action Inference & Visualization")
    parser.add_argument("--input-dir", type=str, default="toy_action/sample_output")
    parser.add_argument("--config", type=str, default="toy_action/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="toy_action/sample_output/deformable_mamba_inference_grid.png")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--filter-bg", action="store_true", help="Filter background using gt bg_mask if available")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold=args.threshold,
        filter_background=args.filter_bg,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
