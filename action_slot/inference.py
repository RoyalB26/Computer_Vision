"""Inference and Multi-Slot Attention Map Visualization script for Action Slot (TACO).

Loads a scenario RGB folder (e.g. D:/apps/Datasets/TACO/ap_Town01/t10/varient_scenario/54/rgb/front),
samples 16 strided frames following the exact sampling logic of `ACTION_SLOT_LMDBDataset`,
runs `ActionSlotModel` inference, overlays multi-colored slot attention heatmaps directly onto
the original camera frames (preserving the original scene background), arranges them into
a 4x4 montage grid (3:1 aspect ratio) with a slot-to-color legend, and prints ego & actor predictions.

Usage:
    python action_slot/inference.py --input-dir "D:/apps/Datasets/TACO/ap_Town01/t10/varient_scenario/54/rgb/front"
    python action_slot/inference.py --input-dir "..." --checkpoint work_dirs/action_slot/best.pth
    python action_slot/inference.py --input-dir "..." --checkpoint best.pth --show-bg-slot
"""

import argparse
import glob
import importlib
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure both project root and action_slot dir are on sys.path
action_slot_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(action_slot_dir, ".."))
for p in (project_root, action_slot_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from BaseModel.model import build_model
from BaseModel.utils import Config
from action_slot.metrics.taco_labels import TACO_ACTOR_LABELS, TACO_EGO_LABELS

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Slot -> (Group Name, RGB Color) for ActionSlotModel (actions_per_slot=[24, 24, 16] + optional bg slot)
SLOT_PALETTE: List[Tuple[int, int, int]] = [
    (230, 25, 75),    # Slot 0: Red    -> Car (c / c+)
    (60, 200, 75),    # Slot 1: Green  -> Bike / Motorbike (b / b+)
    (0, 175, 255),    # Slot 2: Cyan   -> Pedestrian (p / p+)
    (180, 180, 180),  # Slot 3: Silver -> Background Slot (bg)
    (255, 215, 25),   # Extra Slot 4
    (245, 130, 48),   # Extra Slot 5
    (145, 30, 180),   # Extra Slot 6
]

SLOT_GROUP_NAMES: Dict[int, str] = {
    0: "Car (c/c+)",
    1: "Bike (b/b+)",
    2: "Ped (p/p+)",
    3: "Background (bg)",
}


def _parse_frame_number(path: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return None


def _list_sorted_frames(input_dir: str) -> List[str]:
    files = [
        f for f in glob.glob(os.path.join(input_dir, "*"))
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]
    if not files:
        return []
    # Sort numerically by filename stem when possible (e.g. 00000001.jpg -> 1)
    if all(_parse_frame_number(f) is not None for f in files):
        files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    else:
        files.sort()
    return files


def sample_strided_frames(
    frame_paths: List[str],
    sequence_len: int = 16,
    clip_idx: Optional[int] = None,
) -> Tuple[List[str], str]:
    """Sample `sequence_len` strided frame paths matching `ACTION_SLOT_LMDBDataset._build_scenario_clips`.

    Defaults to selecting the middle clip (`len(clips) // 2`), matching validation sampling.
    """
    n_rgb = len(frame_paths)
    if n_rgb == 0:
        raise ValueError("No image frames found to sample.")

    clips: List[List[str]] = []
    max_num = 50
    case_type = "unknown"

    parsed_nums = [_parse_frame_number(p) for p in frame_paths]
    all_numeric = all(num is not None for num in parsed_nums)
    if all_numeric:
        start_frame = int(parsed_nums[0])
        end_frame = int(parsed_nums[-1])
        num_frame = end_frame - start_frame + 1
        num_to_path = {int(num): p for num, p in zip(parsed_nums, frame_paths)}
    else:
        start_frame, end_frame, num_frame = 0, n_rgb - 1, n_rgb
        num_to_path = {i: p for i, p in enumerate(frame_paths)}

    # Case 1: Standard TACO strided sampling when num_frame >= 50 and n_rgb >= 50
    if num_frame >= 50 and n_rgb >= 50:
        step = max(1, num_frame // sequence_len)
        for m in range(max_num):
            start = start_frame + m
            if start_frame + (sequence_len - 1) * step > end_frame:
                break
            seq: List[str] = []
            for frame_num in range(start, end_frame + 1, step):
                if frame_num in num_to_path:
                    seq.append(num_to_path[frame_num])
                if len(seq) == sequence_len:
                    break
            if len(seq) == sequence_len:
                clips.append(seq)
        if clips:
            case_type = f"case1_ge50 (total={n_rgb}, step={step})"

    # Case 2: 16 <= n_rgb < 50 (or sparse numbering fallback)
    if not clips and n_rgb >= sequence_len:
        step_idx = max(1, n_rgb // sequence_len)
        max_start = n_rgb - (sequence_len - 1) * step_idx
        for m in range(min(max_num, max(1, max_start))):
            clips.append([frame_paths[m + k * step_idx] for k in range(sequence_len)])
        case_type = f"case2_16to49 (total={n_rgb}, step_idx={step_idx})"

    # Case 3: 0 < n_rgb < 16 -> uniform temporal interpolation
    if not clips:
        idxs = np.linspace(0, n_rgb - 1, sequence_len).round().astype(int).tolist()
        clips.append([frame_paths[i] for i in idxs])
        case_type = f"case3_lt16 (total={n_rgb}, interpolated)"

    chosen = len(clips) // 2 if clip_idx is None else max(0, min(clip_idx, len(clips) - 1))
    return clips[chosen], f"{case_type}, clip={chosen}/{len(clips)}"


def load_strided_frames(
    input_dir: str,
    sequence_len: int = 16,
    clip_idx: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[str], str]:
    """Loads 16 RGB frames from either an image folder or a video file."""
    if os.path.isfile(input_dir):
        mp4_files = [input_dir]
        frame_files = []
    elif os.path.isdir(input_dir):
        frame_files = _list_sorted_frames(input_dir)
        mp4_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    if frame_files:
        sampled_paths, sampling_info = sample_strided_frames(
            frame_files, sequence_len=sequence_len, clip_idx=clip_idx
        )
        frames: List[np.ndarray] = []
        for p in sampled_paths:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Unable to read image: {p}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frame_tags = [os.path.splitext(os.path.basename(p))[0] for p in sampled_paths]
        return frames, frame_tags, sampling_info

    if mp4_files:
        cap = cv2.VideoCapture(mp4_files[0])
        all_frames: List[np.ndarray] = []
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not all_frames:
            raise ValueError(f"Video {mp4_files[0]} contains 0 frames.")
        idxs = np.linspace(0, len(all_frames) - 1, sequence_len).round().astype(int).tolist()
        frames = [all_frames[i] for i in idxs]
        frame_tags = [f"{i:06d}" for i in idxs]
        return frames, frame_tags, f"video_uniform (total={len(all_frames)})"

    raise FileNotFoundError(f"No image frames ({IMAGE_EXTS}) or .mp4 found in: {input_dir}")


def overlay_multi_slot_attention(
    rgb_frame: np.ndarray,
    slot_attns_2d: np.ndarray,
    active_slot_indices: List[int],
    alpha: float = 0.60,
    attn_floor: float = 0.12,
) -> np.ndarray:
    """Overlays multi-colored slot attention heatmaps onto the original RGB frame."""
    h, w, _ = rgb_frame.shape
    base_canvas = rgb_frame.astype(np.float32)

    if not active_slot_indices:
        return rgb_frame.copy()

    norm_attns = []
    for s_idx in active_slot_indices:
        a_map = slot_attns_2d[s_idx]
        if a_map.shape != (h, w):
            a_map = cv2.resize(a_map, (w, h), interpolation=cv2.INTER_LINEAR)
        a_min, a_max = float(a_map.min()), float(a_map.max())
        # Preserve full dynamic range for active slots while preventing near-dead slots (max < 0.2) from flooding
        scale_denom = max(a_max - a_min, 0.20)
        if a_max - a_min > 1e-6:
            norm_attns.append(np.clip((a_map - a_min) / scale_denom, 0.0, 1.0))
        else:
            norm_attns.append(np.zeros((h, w), dtype=np.float32))

    color_layer = np.zeros((h, w, 3), dtype=np.float32)
    weight_layer = np.zeros((h, w, 1), dtype=np.float32)
    denom = max(1e-6, 1.0 - attn_floor)

    for i, s_idx in enumerate(active_slot_indices):
        color = np.array(SLOT_PALETTE[s_idx % len(SLOT_PALETTE)], dtype=np.float32)
        attn_weight = norm_attns[i][..., None]
        effective_weight = np.clip((attn_weight - attn_floor) / denom, 0.0, 1.0)
        color_layer += color.reshape(1, 1, 3) * effective_weight
        weight_layer += effective_weight

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
    frame_tags: List[str],
    active_slot_indices: List[int],
    slot_label_names: Dict[int, str],
    rows: int = 4,
    cols: int = 4,
    cell_size: Tuple[int, int] = (384, 128),
) -> np.ndarray:
    """Combines 16 frames into a 4x4 grid (preserving 3:1 aspect ratio) with frame badges and legend."""
    cell_w, cell_h = cell_size
    grid_h = rows * cell_h
    grid_w = cols * cell_w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames_overlaid[: rows * cols]):
        r = idx // cols
        c = idx % cols
        img = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA if frame.shape[1] > cell_w else cv2.INTER_LINEAR)

        tag = frame_tags[idx] if idx < len(frame_tags) else f"{idx:02d}"
        text = f"F{idx:02d} ({tag})"
        box_w = min(cell_w - 8, 12 + len(text) * 8)
        cv2.rectangle(img, (4, 4), (box_w, 24), (0, 0, 0), -1)
        cv2.putText(
            img,
            text,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = img

    legend_entries = [
        (s_idx, slot_label_names.get(s_idx, SLOT_GROUP_NAMES.get(s_idx, f"Slot {s_idx}")))
        for s_idx in active_slot_indices
    ]
    cols_legend = 2 if len(legend_entries) <= 4 else 3
    rows_legend = max(1, (len(legend_entries) + cols_legend - 1) // cols_legend)
    legend_h = 40 + rows_legend * 28
    legend_bar = np.full((legend_h, grid_w, 3), 245, dtype=np.uint8)

    cv2.putText(
        legend_bar,
        "Action-Slot Attention Overlay (Original Frame Background):",
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
        cv2.rectangle(legend_bar, (x, y - 14), (x + 22, y + 2), color, -1)
        cv2.rectangle(legend_bar, (x, y - 14), (x + 22, y + 2), (50, 50, 50), 1)
        cv2.putText(
            legend_bar,
            f"S{s_idx:02d}: {s_name}",
            (x + 28, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    return np.vstack([grid, legend_bar])


def _map_class_to_slot(cls_idx: int, actions_per_slot: List[int]) -> int:
    """Map an actor class index (0..63) to its object slot based on `actions_per_slot`."""
    cursor = 0
    for s_idx, count in enumerate(actions_per_slot):
        if cls_idx < cursor + count:
            return s_idx
        cursor += count
    return max(0, len(actions_per_slot) - 1)


def run_inference(
    input_dir: str,
    config_path: str = "action_slot/config.yaml",
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None,
    threshold: float = 0.35,
    alpha: float = 0.60,
    clip_idx: Optional[int] = None,
    show_bg_slot: bool = False,
    device_str: str = "cuda",
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Loading configuration from: {config_path}")
    cfg = Config.fromfile(config_path)

    custom_imports = getattr(cfg, "custom_imports", None)
    if custom_imports:
        for mod in custom_imports:
            importlib.import_module(mod)

    ckpt = checkpoint_path or getattr(cfg, "checkpoint", None)
    # Avoid failing on Kaggle/Colab pretrained_backbone_path when loading a full model checkpoint locally
    if ckpt and os.path.isfile(ckpt):
        pb_path = cfg.model.get("pretrained_backbone_path", None)
        if pb_path and not os.path.isfile(pb_path):
            cfg.model["pretrained_backbone"] = False
            cfg.model.pop("pretrained_backbone_path", None)

    print(f"Building model: {cfg.model.type} ...")
    model = build_model(cfg.model)
    model.to(device)
    model.eval()

    if ckpt and os.path.isfile(ckpt):
        print(f"Loading checkpoint weights from: {ckpt}")
        try:
            checkpoint_data = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            checkpoint_data = torch.load(ckpt, map_location=device)
        state_dict = checkpoint_data.get("state_dict", checkpoint_data.get("model_state_dict", checkpoint_data))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded (missing={len(missing)}, unexpected={len(unexpected)}).")
    else:
        print("No checkpoint specified or found. Using model default/pretrained weights.")

    # Resolve target spatial/temporal dimensions from config
    sequence_len = int(cfg.data["val"]["dataset"].get("sequence_len", 16))
    video_size = (256, 768)
    val_pipeline = cfg.data["val"]["dataset"].get("pipeline", [])
    if val_pipeline and isinstance(val_pipeline[0], dict) and "video_size" in val_pipeline[0]:
        video_size = tuple(val_pipeline[0]["video_size"])
    target_h, target_w = int(video_size[0]), int(video_size[1])

    print(f"Sampling {sequence_len} strided frames from: {input_dir}")
    frames_rgb, frame_tags, sampling_info = load_strided_frames(
        input_dir=input_dir, sequence_len=sequence_len, clip_idx=clip_idx
    )
    print(f"Sampling strategy: {sampling_info}")
    print(f"Sampled frame IDs : {frame_tags}")

    resized_rgb = [
        f if f.shape[:2] == (target_h, target_w) else cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for f in frames_rgb
    ]
    # Pass uint8 tensor (1, T, 3, H, W) so ActionSlotModel applies its GPU normalization identically to train/val
    video_np = np.stack(resized_rgb, axis=0)  # (T, H, W, 3)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).contiguous().to(device)

    with torch.no_grad():
        preds = model.predict(video_tensor)

    actor_probs = torch.sigmoid(preds["actor"]).squeeze(0).cpu().numpy()  # (64,)
    ego_logits = preds.get("ego")
    ego_probs = torch.softmax(ego_logits, dim=-1).squeeze(0).cpu().numpy() if ego_logits is not None else None

    top_indices = np.argsort(actor_probs)[::-1][:10]
    detected = [i for i in range(len(actor_probs)) if actor_probs[i] >= threshold]

    print("=" * 68)
    print("[MODEL INFERENCE PREDICTIONS]")
    if ego_probs is not None:
        ego_idx = int(np.argmax(ego_probs))
        ego_name = TACO_EGO_LABELS[ego_idx] if ego_idx < len(TACO_EGO_LABELS) else f"Ego_{ego_idx}"
        print(f"Predicted Ego Action : {ego_name} ({ego_probs[ego_idx] * 100:.2f}%)")

    detected_str = ", ".join(f"{TACO_ACTOR_LABELS[i]} ({actor_probs[i]*100:.1f}%)" for i in detected)
    print(f"Detected Actor Labels (threshold >= {threshold}):")
    print(f"  --> {detected_str if detected_str else '(None above threshold)'}")
    print("\nTop 10 Actor Probabilities:")
    for rank, idx in enumerate(top_indices, 1):
        cls_name = TACO_ACTOR_LABELS[idx] if idx < len(TACO_ACTOR_LABELS) else f"Class_{idx}"
        conf = actor_probs[idx] * 100.0
        mark = "[DETECTED]" if actor_probs[idx] >= threshold else "          "
        print(f"  {rank:2d}. {mark} {cls_name:<12} : {conf:6.2f}%")
    print("=" * 68)

    # Extract slot attention maps: (1, T, actual_slots, H_feat, W_feat)
    attn_tensor = preds.get("attn")
    active_slot_indices: List[int] = []
    slot_label_names: Dict[int, str] = {}

    if attn_tensor is not None:
        attn = attn_tensor.squeeze(0).cpu().numpy()  # (T, actual_slots, H_feat, W_feat)
        total_slots = attn.shape[1]
        model_obj_slots = int(getattr(model, "num_slots", 3))
        actions_per_slot = list(getattr(getattr(model, "head", None), "actions_per_slot", [24, 24, 16]))

        num_obj_slots = min(model_obj_slots, total_slots)
        has_bg_slot = bool(getattr(model, "bg_slot", True)) and total_slots > num_obj_slots

        slot_to_classes: Dict[int, List[int]] = {}
        for c_idx in range(len(actor_probs)):
            s_idx = _map_class_to_slot(c_idx, actions_per_slot)
            if 0 <= s_idx < num_obj_slots:
                slot_to_classes.setdefault(s_idx, []).append(c_idx)

        for s_idx in range(num_obj_slots):
            c_list = sorted(slot_to_classes.get(s_idx, []), key=lambda c: actor_probs[c], reverse=True)
            det_c = [c for c in c_list if actor_probs[c] >= threshold]
            chosen_c = det_c if det_c else c_list[:1]
            top_text = ", ".join(f"{TACO_ACTOR_LABELS[c]} ({actor_probs[c]*100:.0f}%)" for c in chosen_c[:2])
            if len(chosen_c) > 2:
                top_text += f" +{len(chosen_c) - 2}"
            group_prefix = SLOT_GROUP_NAMES.get(s_idx, f"Slot_{s_idx}")
            slot_label_names[s_idx] = f"{group_prefix}: {top_text}" if top_text else group_prefix
            active_slot_indices.append(s_idx)

        if show_bg_slot and has_bg_slot:
            bg_idx = total_slots - 1
            active_slot_indices.append(bg_idx)
            slot_label_names[bg_idx] = SLOT_GROUP_NAMES.get(3, f"Background ({bg_idx})")

        print("\n[SLOT COLOR ASSIGNMENTS & RAW ATTENTION STATS]")
        for s_idx in active_slot_indices:
            s_raw = attn[:, s_idx]
            stat_str = f"min={s_raw.min():.3f}, mean={s_raw.mean():.3f}, max={s_raw.max():.3f}"
            print(f"  Slot {s_idx:02d} ({slot_label_names[s_idx]:<36}) -> RGB: {SLOT_PALETTE[s_idx % len(SLOT_PALETTE)]} | {stat_str}")
        if has_bg_slot and not show_bg_slot:
            bg_raw = attn[:, total_slots - 1]
            print(f"  Slot {total_slots - 1:02d} (Background slot - hidden          ) | min={bg_raw.min():.3f}, mean={bg_raw.mean():.3f}, max={bg_raw.max():.3f}")
    else:
        attn = np.zeros((len(resized_rgb), 1, target_h, target_w), dtype=np.float32)
        active_slot_indices = [0]
        slot_label_names = {0: "Default"}

    overlaid_frames = [
        overlay_multi_slot_attention(
            rgb_frame=resized_rgb[t],
            slot_attns_2d=attn[t],
            active_slot_indices=active_slot_indices,
            alpha=alpha,
        )
        for t in range(len(resized_rgb))
    ]

    grid_img = create_16_frame_grid_with_legend(
        frames_overlaid=overlaid_frames,
        frame_tags=frame_tags,
        active_slot_indices=active_slot_indices,
        slot_label_names=slot_label_names,
        rows=4,
        cols=4,
        cell_size=(384, 128),
    )

    if not output_path:
        output_path = os.path.join("work_dirs", "action_slot", "attn_inference_grid.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))

    print(f"\n[ORIGINAL-BACKGROUND SLOT ATTENTION GRID SAVED]")
    print(f"Output Image : {output_path}")
    print(f"Resolution   : {grid_img.shape[1]}x{grid_img.shape[0]} (4x4 Grid + Slot Legend)")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Action-Slot (TACO) Inference & Attention Visualization")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"D:\apps\Datasets\TACO\ap_Town01\t10\varient_scenario\54\rgb\front",
        help="Path to scenario RGB folder (e.g. .../varient_scenario/54/rgb/front) or .mp4 file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="action_slot/config.yaml",
        help="Path to Action-Slot config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="work_dirs/action_slot/attn_inference_grid.png",
        help="Output path for 4x4 montage grid image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Probability threshold for detected actor labels",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.60,
        help="Blending opacity of slot attention overlay on the original frame",
    )
    parser.add_argument(
        "--clip-idx",
        type=int,
        default=None,
        help="Optional strided clip offset index (defaults to middle clip len//2 like val)",
    )
    parser.add_argument(
        "--show-bg-slot",
        action="store_true",
        help="Also overlay background slot attention on top of the original frame",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference ('cuda' or 'cpu')",
    )

    args = parser.parse_args()
    run_inference(
        input_dir=args.input_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold=args.threshold,
        alpha=args.alpha,
        clip_idx=args.clip_idx,
        show_bg_slot=args.show_bg_slot,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
