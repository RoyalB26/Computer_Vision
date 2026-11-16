"""Inference and Multi-Slot Attention Map Visualization script for Toy Action.

Loads a sample folder containing 16 frames (or .mp4), runs model inference,
removes/filters out the background (so only foreground objects and their slot
attentions are visible on a clean neutral canvas, or using slot background mask),
assigns a distinct vibrant color to each active slot attention map,
overlays the multi-colored slot heatmaps onto the 16 frames, arranges them
in a single 4x4 montage grid with frame indices and a color legend, and
prints model predictions.

Usage:
    python toy_action/inference.py --input-dir toy_action/sample_output
    python toy_action/inference.py --input-dir toy_action/sample_output --filter-bg
    python toy_action/inference.py --input-dir toy_action/sample_output --checkpoint work_dirs/toy_action_slot_exp/best.pth
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from BaseModel.utils import Config
from BaseModel.model import build_model
from toy_action.dataset import ALL_CLASSES, IDX_TO_CLASS, SYMBOLS, SYMBOL_VARIANTS

# Distinct vibrant color palette for up to 32 slots (RGB)
SLOT_PALETTE: List[Tuple[int, int, int]] = [
    (230, 25, 75),    # 0: Red
    (60, 180, 75),    # 1: Green
    (255, 225, 25),   # 2: Yellow
    (0, 130, 200),    # 3: Blue
    (245, 130, 48),   # 4: Orange
    (145, 30, 180),   # 5: Purple
    (70, 240, 240),   # 6: Cyan
    (240, 50, 230),   # 7: Magenta
    (210, 245, 60),   # 8: Lime
    (250, 190, 212),  # 9: Pink
    (0, 128, 128),    # 10: Teal
    (220, 190, 255),  # 11: Lavender
    (170, 110, 40),   # 12: Brown
    (255, 250, 200),  # 13: Beige
    (128, 0, 0),      # 14: Maroon
    (170, 255, 195),  # 15: Mint
    (128, 128, 0),    # 16: Olive
    (255, 215, 180),  # 17: Coral
    (0, 0, 128),      # 18: Navy
    (255, 105, 180),  # 19: Hot Pink
    (50, 205, 50),    # 20: Lime Green
    (255, 140, 0),    # 21: Dark Orange
    (138, 43, 226),   # 22: Blue Violet
    (0, 255, 127),    # 23: Spring Green
    (192, 192, 192),  # 24: Background Slot (Silver)
]


def load_input_frames_and_masks(
    input_dir: str,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[str]]:
    """Loads 16 RGB frames and optional 16 bg_masks from input directory."""
    # 1. Individual frame_*.png images
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
        # Fallback to .mp4
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

    # 2. Check for ground-truth bg_mask_*.png (255: background, 0: foreground object)
    bg_masks = None
    mask_files = sorted(glob.glob(os.path.join(input_dir, "bg_mask_*.png")))
    if len(mask_files) >= 16:
        bg_masks = []
        for p in mask_files[:16]:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            bg_masks.append(m)

    # 3. Read ground truth label
    gt_label = None
    label_path = os.path.join(input_dir, "label.txt")
    if os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Label:"):
                    gt_label = line.replace("Label:", "").strip()
                    break

    return frames, bg_masks, gt_label


def overlay_multi_slot_attention(
    rgb_frame: np.ndarray,
    slot_attns_2d: np.ndarray,
    active_slot_indices: List[int],
    bg_slot_attn_2d: Optional[np.ndarray] = None,
    gt_bg_mask_2d: Optional[np.ndarray] = None,
    filter_background: bool = True,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.70,
) -> np.ndarray:
    """Overlays multi-colored slot attention maps with background filtering.

    When filter_background=True:
        - The distracting multi-region background is filtered out to a clean white canvas.
        - Foreground objects and their assigned slot colors are prominently displayed.

    Args:
        rgb_frame: uint8 array of shape (H, W, 3), RGB.
        slot_attns_2d: float array of shape (num_slots, H_feat, W_feat).
        active_slot_indices: list of slot indices to visualize.
        bg_slot_attn_2d: attention map of background slot (if available from model).
        gt_bg_mask_2d: ground-truth bg_mask (255: background, 0: foreground object).
        filter_background: whether to filter out background pixels.
        bg_color: clean neutral color for filtered background.
        alpha: blending weight for slot attention.
    """
    h, w, _ = rgb_frame.shape
    num_slots = slot_attns_2d.shape[0]

    # Resize all active slot attention maps to (H, W)
    resized_attns = []
    for s_idx in active_slot_indices:
        a_map = slot_attns_2d[s_idx]
        if a_map.shape != (h, w):
            a_map = cv2.resize(a_map, (w, h), interpolation=cv2.INTER_LINEAR)
        resized_attns.append(a_map)

    # Compute normalized slot attention maps
    norm_attns = []
    for a in resized_attns:
        a_min, a_max = float(a.min()), float(a.max())
        if a_max - a_min > 1e-6:
            norm_attns.append((a - a_min) / (a_max - a_min))
        else:
            norm_attns.append(np.zeros_like(a))
    norm_attns = np.stack(norm_attns, axis=0) if norm_attns else np.zeros((0, h, w))

    # Determine Foreground vs Background Mask
    # Priority 1: Ground-truth bg_mask if provided (0: foreground object, 255: background)
    if gt_bg_mask_2d is not None:
        if gt_bg_mask_2d.shape != (h, w):
            gt_bg_mask_2d = cv2.resize(gt_bg_mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
        # fg_prob is 1.0 where object is present, 0.0 at background
        fg_prob = (gt_bg_mask_2d < 128).astype(np.float32)
        # Slightly dilate mask so slot attention around object boundary remains clearly visible
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_prob = cv2.dilate(fg_prob, kernel, iterations=1)
    elif bg_slot_attn_2d is not None:
        # Priority 2: Use model's predicted background slot attention map
        if bg_slot_attn_2d.shape != (h, w):
            bg_slot_attn_2d = cv2.resize(bg_slot_attn_2d, (w, h), interpolation=cv2.INTER_LINEAR)
        b_min, b_max = float(bg_slot_attn_2d.min()), float(bg_slot_attn_2d.max())
        norm_bg = (bg_slot_attn_2d - b_min) / (b_max - b_min + 1e-6)
        fg_prob = np.clip(1.0 - norm_bg, 0.0, 1.0)
    else:
        # Priority 3: Max attention across all foreground slots
        if norm_attns.shape[0] > 0:
            fg_prob = np.max(norm_attns, axis=0)
        else:
            fg_prob = np.ones((h, w), dtype=np.float32)

    # Base Canvas: either original frame or background-filtered frame
    if filter_background:
        # Filter noisy/multi-region background to white canvas while preserving foreground objects
        white_bg = np.full((h, w, 3), bg_color, dtype=np.float32)
        # fg_alpha: 1.0 on foreground object pixels, 0.0 on background
        fg_alpha = np.clip(fg_prob[..., None], 0.0, 1.0)
        base_canvas = rgb_frame.astype(np.float32) * fg_alpha + white_bg * (1.0 - fg_alpha)
    else:
        base_canvas = rgb_frame.astype(np.float32)

    # Composite color layer for active slots
    color_layer = np.zeros((h, w, 3), dtype=np.float32)
    weight_layer = np.zeros((h, w, 1), dtype=np.float32)

    for i, s_idx in enumerate(active_slot_indices):
        color = np.array(SLOT_PALETTE[s_idx % len(SLOT_PALETTE)], dtype=np.float32)
        attn_weight = norm_attns[i, ..., None]  # (H, W, 1)

        # Apply soft threshold to suppress low background noise
        effective_weight = np.clip((attn_weight - 0.10) / 0.90, 0.0, 1.0)

        color_layer += color.reshape(1, 1, 3) * effective_weight
        weight_layer += effective_weight

    # Blend multi-slot colors onto canvas
    blend_mask = np.clip(weight_layer, 0.0, 1.0) * alpha
    color_normalized = np.divide(
        color_layer,
        np.maximum(weight_layer, 1e-6),
        out=np.zeros_like(color_layer),
        where=(weight_layer > 1e-6),
    )

    blended = base_canvas * (1.0 - blend_mask) + color_normalized * blend_mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_16_frame_grid_with_legend(
    frames_overlaid: List[np.ndarray],
    active_slot_indices: List[int],
    slot_label_names: Dict[int, str],
    rows: int = 4,
    cols: int = 4,
    cell_size: int = 224,
) -> np.ndarray:
    """Combines 16 frames into a 4x4 grid, labels each frame (F00..F15),

    and adds a bottom legend showing the assigned color for each slot.
    """
    grid_h = rows * cell_size
    grid_w = cols * cell_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames_overlaid[:16]):
        r = idx // cols
        c = idx % cols

        img = cv2.resize(frame, (cell_size, cell_size))

        # Draw frame badge / index (top-left)
        text = f"F{idx:02d}"
        cv2.rectangle(img, (4, 4), (58, 26), (0, 0, 0), -1)
        cv2.putText(
            img,
            text,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        grid[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = img

    # Add Color Legend at bottom
    legend_entries = [
        (s_idx, slot_label_names.get(s_idx, f"Slot {s_idx}"))
        for s_idx in active_slot_indices
    ]

    max_label_len = max((len(name) for _, name in legend_entries), default=0)
    cols_legend = 2 if max_label_len > 14 else 4
    rows_legend = max(1, (len(legend_entries) + cols_legend - 1) // cols_legend)
    legend_h = 40 + rows_legend * 28
    legend_bar = np.full((legend_h, grid_w, 3), 245, dtype=np.uint8)

    cv2.putText(
        legend_bar,
        "Active Slot Color Mapping (Background Filtered to White):",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    col_w = grid_w // cols_legend
    for i, (s_idx, s_name) in enumerate(legend_entries):
        col = i % cols_legend
        row = i // cols_legend
        x = 12 + col * col_w
        y = 52 + row * 28

        color = SLOT_PALETTE[s_idx % len(SLOT_PALETTE)]
        # Draw color patch
        cv2.rectangle(legend_bar, (x, y - 14), (x + 22, y + 2), color, -1)
        cv2.rectangle(legend_bar, (x, y - 14), (x + 22, y + 2), (50, 50, 50), 1)

        # Draw slot text label
        cv2.putText(
            legend_bar,
            f"S{s_idx:02d}: {s_name}",
            (x + 28, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    # Concatenate vertically
    full_image = np.vstack([grid, legend_bar])
    return full_image


def _map_class_to_slot(cls_idx: int, num_obj_slots: int) -> int:
    """Map an action class index (0..47) to its object/action slot index."""
    if num_obj_slots == len(SYMBOLS):
        # 6 object slots: slot s covers single + group (+) actions (8 classes per symbol)
        return cls_idx // 8
    if num_obj_slots == 24:
        # 24 symbol-direction slots: (6 symbols * 4 directions)
        sym_idx = cls_idx // 8
        act_idx = (cls_idx % 8) % 4
        return sym_idx * 4 + act_idx
    if num_obj_slots >= len(ALL_CLASSES):
        return cls_idx
    return min(cls_idx // max(1, len(ALL_CLASSES) // max(1, num_obj_slots)), num_obj_slots - 1)


def run_inference(
    input_dir: str,
    config_path: str = "toy_action/config.yaml",
    checkpoint_path: Optional[str] = None,
    output_path: str = "toy_action/sample_output/attn_inference_grid.png",
    threshold: float = 0.35,
    top_k_slots: int = 6,
    filter_background: bool = True,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Loading configuration from: {config_path}")
    cfg = Config.fromfile(config_path)

    # Import custom modules defined in config
    custom_imports = getattr(cfg, "custom_imports", None)
    if custom_imports:
        import importlib
        for mod in custom_imports:
            importlib.import_module(mod)

    # Build model
    print(f"Building model: {cfg.model.type} ...")
    model = build_model(cfg.model)
    model.to(device)
    model.eval()

    # Load checkpoint if available
    ckpt = checkpoint_path or getattr(cfg, "checkpoint", None)
    if ckpt and os.path.isfile(ckpt):
        print(f"Loading checkpoint weights from: {ckpt}")
        checkpoint_data = torch.load(ckpt, map_location=device)
        state_dict = checkpoint_data.get("state_dict", checkpoint_data)
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint specified or found. Using model default/pretrained weights.")

    # Load 16 input frames and optional bg_masks
    print(f"Loading 16 frames from: {input_dir} ...")
    frames_rgb, gt_bg_masks, gt_label = load_input_frames_and_masks(input_dir)
    print(f"Successfully loaded {len(frames_rgb)} frames.")
    if gt_bg_masks:
        print(f"Found {len(gt_bg_masks)} ground-truth background masks (bg_mask_*.png).")
    if gt_label:
        print(f"Ground-Truth Label: {gt_label}")

    # Prepare input tensor: (1, 3, 16, 224, 224) normalized in [0, 1]
    resized_frames = [cv2.resize(f, (224, 224)) for f in frames_rgb]
    video_np = np.stack(resized_frames, axis=0)  # (16, 224, 224, 3)
    video_tensor = torch.from_numpy(video_np).float().div(255.0)
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, 3, 16, 224, 224)

    # Forward pass
    with torch.no_grad():
        preds_dict = model.predict(video_tensor, threshold=threshold)

    probs = preds_dict["probs"].squeeze(0).cpu().numpy()  # (48,)
    pred_labels = preds_dict.get("label_strings", [""])[0]

    # Show top predicted classes
    top_indices = np.argsort(probs)[::-1][:8]
    print("=" * 60)
    print("[MODEL INFERENCE PREDICTIONS]")
    print(f"Active Predicted Labels (threshold >= {threshold}):")
    print(f"  --> {pred_labels if pred_labels else '(None above threshold)'}")
    print("\nTop Highest Probability Classes:")
    for rank, idx in enumerate(top_indices, 1):
        cls_name = IDX_TO_CLASS[idx]
        conf = probs[idx] * 100.0
        status = "[DETECTED]" if probs[idx] >= threshold else "          "
        print(f"  {rank}. {status} {cls_name:<16} : {conf:6.2f}%")

    # Show auxiliary predictions if available (per-frame object counts or video object classes)
    obj_counts = preds_dict.get("obj_counts", None)
    if obj_counts is not None:
        counts_np = obj_counts.squeeze(0).cpu().numpy()  # (16, 6)
        mean_counts = counts_np.mean(axis=0)
        count_strs = [f"{sym}:{mean_counts[i]:.2f}" for i, sym in enumerate(SYMBOLS)]
        print("\nPredicted Mean Object Counts (per frame):")
        print("  " + " | ".join(count_strs))

    obj_cls_probs = preds_dict.get("obj_cls_probs", None)
    if obj_cls_probs is not None:
        cls_p = obj_cls_probs.squeeze(0).cpu().numpy()  # (12,)
        present = [f"{SYMBOL_VARIANTS[i]}({cls_p[i]*100:.0f}%)" for i in range(len(SYMBOL_VARIANTS)) if cls_p[i] >= threshold]
        print("\nPredicted Video Object Classes:")
        print("  " + (", ".join(present) if present else "(None above threshold)"))
    print("=" * 60)

    # Extract Multi-Slot Attention Maps:
    attn_tensor = preds_dict.get("attn", None)
    active_slot_indices: List[int] = []
    slot_label_names: Dict[int, str] = {}
    bg_slot_attn = None

    if attn_tensor is not None:
        # attn_tensor shape: (1, 16, actual_slots, H, W)
        attn = attn_tensor.squeeze(0).cpu().numpy()  # (16, actual_slots, H, W)
        total_slots = attn.shape[1]
        model_obj_slots = getattr(model, "num_slots", 6)

        # Last slot is background slot if bg_slot is enabled and extra slot is present
        has_bg_slot = getattr(model, "bg_slot", True) and total_slots > model_obj_slots
        num_obj_slots = total_slots - 1 if has_bg_slot else min(total_slots, model_obj_slots)
        if has_bg_slot:
            bg_slot_attn = attn[:, -1, :, :]  # (16, H, W)

        # Group classes by object slot
        slot_to_classes: Dict[int, List[int]] = {}
        for c_idx in range(len(probs)):
            s_idx = _map_class_to_slot(c_idx, num_obj_slots)
            if 0 <= s_idx < num_obj_slots:
                slot_to_classes.setdefault(s_idx, []).append(c_idx)

        # Order slots by highest class confidence, keeping detected slots + meaningful candidates (>= 5%)
        slot_scores = [
            (s_idx, float(max(probs[c] for c in c_list)))
            for s_idx, c_list in slot_to_classes.items()
        ]
        slot_scores.sort(key=lambda x: x[1], reverse=True)

        selected = [
            (s_idx, score)
            for s_idx, score in slot_scores
            if score >= threshold or (s_idx in {_map_class_to_slot(int(i), num_obj_slots) for i in top_indices} and score >= 0.05)
        ]
        if not selected:
            selected = slot_scores[: min(top_k_slots, 3)]

        for s_idx, _ in selected[: min(top_k_slots, num_obj_slots)]:
            c_list = sorted(slot_to_classes[s_idx], key=lambda c: probs[c], reverse=True)
            detected_c = [c for c in c_list if probs[c] >= threshold]
            chosen_c = detected_c if detected_c else c_list[:1]
            label_parts = [f"{IDX_TO_CLASS[c]} ({probs[c]*100:.0f}%)" for c in chosen_c[:2]]
            if len(chosen_c) > 2:
                label_parts.append(f"+{len(chosen_c) - 2}")
            active_slot_indices.append(s_idx)
            slot_label_names[s_idx] = ", ".join(label_parts)

        if not active_slot_indices:
            active_slot_indices = list(range(min(4, num_obj_slots)))
            for s in active_slot_indices:
                slot_label_names[s] = SYMBOLS[s] if num_obj_slots == len(SYMBOLS) else f"Slot_{s}"

        print(f"\n[SLOT COLOR ASSIGNMENTS]")
        for s_idx in active_slot_indices:
            color = SLOT_PALETTE[s_idx % len(SLOT_PALETTE)]
            name = slot_label_names.get(s_idx, f"Slot_{s_idx}")
            sym_tag = f" [{SYMBOLS[s_idx]}]" if num_obj_slots == len(SYMBOLS) else ""
            print(f"  Slot {s_idx:02d}{sym_tag} ({name:<28}) -> RGB Color: {color}")
    else:
        attn = np.zeros((16, 1, 224, 224), dtype=np.float32)
        active_slot_indices = [0]
        slot_label_names = {0: "Default"}

    # Overlay multi-colored slot attention maps with background filtering
    overlaid_frames = []
    for t in range(16):
        overlaid = overlay_multi_slot_attention(
            rgb_frame=resized_frames[t],
            slot_attns_2d=attn[t],
            active_slot_indices=active_slot_indices,
            bg_slot_attn_2d=bg_slot_attn[t] if bg_slot_attn is not None else None,
            gt_bg_mask_2d=gt_bg_masks[t] if gt_bg_masks is not None else None,
            filter_background=filter_background,
            alpha=0.75,
        )
        overlaid_frames.append(overlaid)

    # Create 4x4 Grid Montage with Color Legend
    grid_img = create_16_frame_grid_with_legend(
        frames_overlaid=overlaid_frames,
        active_slot_indices=active_slot_indices,
        slot_label_names=slot_label_names,
        rows=4,
        cols=4,
        cell_size=224,
    )

    # Save output image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))

    print(f"\n[BACKGROUND-FILTERED MULTI-COLOR ATTENTION GRID SAVED]")
    print(f"Output Image : {output_path}")
    print(f"Resolution   : {grid_img.shape[1]}x{grid_img.shape[0]} (4x4 Grid + Bottom Slot Legend)")
    print(f"Background   : Filtered to clean white canvas. Only moving objects & slot colors shown.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Multi-Slot Attention Map Visualization with Background Filtering")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="toy_action/sample_output",
        help="Path to folder containing sample.mp4 or frame_00..15.png",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="toy_action/config.yaml",
        help="Path to model config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model weights checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="toy_action/sample_output/attn_inference_grid.png",
        help="Destination path for the composite attention visualization image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Probability threshold for active label prediction",
    )
    parser.add_argument(
        "--top-k-slots",
        type=int,
        default=6,
        help="Maximum number of active slot attention maps to colorize",
    )
    parser.add_argument(
        "--no-filter-bg",
        action="store_true",
        help="Disable background filtering (keep original background)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda' or 'cpu')",
    )

    args = parser.parse_args()
    run_inference(
        input_dir=args.input_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold=args.threshold,
        top_k_slots=args.top_k_slots,
        filter_background=(not args.no_filter_bg),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
