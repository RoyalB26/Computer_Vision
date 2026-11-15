"""Toy Action Dataset Generator.

Generates a toy video dataset consisting of symbols ('x', 'y', 'o', 'T', '8', 'O')
moving continuously in 4 directions ('gotop', 'gobottom', 'goleft', 'goright')
across 16 frames.

Features:
- Canvas: 224x224 RGB.
- Object size: random between 8x8 and 32x32.
- Symbol 'O' has filled interior and a different colored border.
- Objects have random colors, and different parts of an object have different colors.
- Multi-region randomized background (Voronoi cells, split regions, quadrants, gradients).
- Objects can spawn at the same or different frames (t_start in [0, 8]).
- Frame-wise background mask (bg_mask) for all 16 frames (255: background, 0: foreground object).
- Grouping label logic:
    - Single object: '<symbol>:<action>' (e.g. 'x:gotop')
    - Multiple objects with same symbol and same action: '<symbol>+:<action>' (e.g. 'x+:gotop')

CLI Usage:
    # 1. Generate 1 sample video, 16 frame images, and 16 bg_mask images:
    python toy_action/generate_data.py --sample --sample-dir toy_action/sample_output

    # 2. Generate LMDB dataset with 1000 train samples and 200 val samples (including bg_mask):
    python toy_action/generate_data.py --lmdb --num-samples 1000 --val-samples 200
"""

import argparse
from collections import Counter
import io
import math
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import cv2
import lmdb
import numpy as np
from PIL import Image, ImageDraw

# Constants
CANVAS_SIZE = 224
NUM_FRAMES = 16
MIN_OBJ_SIZE = 8
MAX_OBJ_SIZE = 32

SYMBOLS = ["x", "y", "o", "T", "8", "O"]
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOLS)}
ACTIONS = ["gotop", "gobottom", "goleft", "goright"]

# 12 symbol variants (single and group '+')
SYMBOL_VARIANTS = []
for s in SYMBOLS:
    SYMBOL_VARIANTS.append(s)
    SYMBOL_VARIANTS.append(f"{s}+")

ALL_CLASSES = [f"{s}:{a}" for s in SYMBOL_VARIANTS for a in ACTIONS]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)  # 48 classes


def random_distinct_color(exclude_color: Optional[Tuple[int, int, int]] = None, min_dist: float = 60.0) -> Tuple[int, int, int]:
    """Generate a random vibrant RGB color, ensuring distance from an excluded color."""
    while True:
        c = (random.randint(20, 245), random.randint(20, 245), random.randint(20, 245))
        if exclude_color is None:
            return c
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c, exclude_color)))
        if dist >= min_dist:
            return c


def render_symbol(symbol: str, size: int) -> Image.Image:
    """Render a symbol into an RGBA PIL Image of size (size, size).

    Different parts of the symbol have different colors.
    Symbol 'O' specifically has a filled interior and a different colored border.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    w = max(1, size // 7)
    margin = max(1, size // 10)

    # Base colors for the symbol parts
    c1 = (*random_distinct_color(), 255)
    c2 = (*random_distinct_color(exclude_color=c1[:3], min_dist=70), 255)
    c3 = (*random_distinct_color(exclude_color=c2[:3], min_dist=70), 255)

    if symbol == "O":
        # 'O' (capital): filled interior with one color and outline of a distinct different color
        fill_color = c1
        border_color = c2
        border_width = max(2, size // 7)
        draw.ellipse(
            [margin, margin, size - 1 - margin, size - 1 - margin],
            fill=fill_color,
            outline=border_color,
            width=border_width,
        )

    elif symbol == "o":
        # 'o' (lowercase): hollow ring with color c1
        border_width = max(1, size // 7)
        draw.ellipse(
            [margin + 1, margin + 1, size - 1 - margin - 1, size - 1 - margin - 1],
            fill=None,
            outline=c1,
            width=border_width,
        )

    elif symbol == "x":
        # 'x': two crossing diagonals with two distinct colors
        draw.line([(margin, margin), (size - 1 - margin, size - 1 - margin)], fill=c1, width=w)
        draw.line([(size - 1 - margin, margin), (margin, size - 1 - margin)], fill=c2, width=w)

    elif symbol == "y":
        # 'y': left branch (c1), right branch (c2), and stem (c3)
        mid_x = size // 2
        mid_y = int(size * 0.48)
        draw.line([(margin, margin), (mid_x, mid_y)], fill=c1, width=w)
        draw.line([(size - 1 - margin, margin), (mid_x, mid_y)], fill=c2, width=w)
        draw.line([(mid_x, mid_y), (mid_x, size - 1 - margin)], fill=c3, width=w)

    elif symbol == "T":
        # 'T': horizontal top bar (c1) and vertical stem (c2)
        mid_x = size // 2
        bar_y = margin + w // 2
        draw.line([(margin, bar_y), (size - 1 - margin, bar_y)], fill=c1, width=w)
        draw.line([(mid_x, bar_y), (mid_x, size - 1 - margin)], fill=c2, width=w)

    elif symbol == "8":
        # '8': top loop (c1) and bottom loop (c2)
        mid_y = size // 2
        w_loop = max(1, size // 8)
        draw.ellipse(
            [margin + 1, margin, size - 1 - margin - 1, mid_y + 1],
            fill=None,
            outline=c1,
            width=w_loop,
        )
        draw.ellipse(
            [margin, mid_y - 1, size - 1 - margin, size - 1 - margin],
            fill=None,
            outline=c2,
            width=w_loop,
        )

    return img


def generate_multi_region_background(height: int = CANVAS_SIZE, width: int = CANVAS_SIZE) -> np.ndarray:
    """Generate a 224x224 RGB background with multiple random color regions."""
    mode = random.choice(["voronoi", "split_h", "split_v", "quadrants", "gradient", "patches"])

    if mode == "voronoi":
        # 3 to 6 Voronoi color cells
        num_seeds = random.randint(3, 6)
        seed_points = np.random.randint(0, [height, width], size=(num_seeds, 2))
        seed_colors = [np.array(random_distinct_color(), dtype=np.float32) for _ in range(num_seeds)]

        y_coords, x_coords = np.indices((height, width))
        grid = np.stack([y_coords, x_coords], axis=-1)  # (H, W, 2)

        # Distances to each seed
        dists = [np.sum((grid - seed) ** 2, axis=-1) for seed in seed_points]  # list of (H, W)
        nearest = np.argmin(np.stack(dists, axis=-1), axis=-1)  # (H, W)

        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for i, color in enumerate(seed_colors):
            bg[nearest == i] = color.astype(np.uint8)
        # Apply smooth blur to region boundaries
        if random.random() < 0.6:
            bg = cv2.GaussianBlur(bg, (9, 9), 4)

    elif mode == "split_h":
        # Horizontal split with 2-3 color regions
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        splits = sorted(random.sample(range(40, height - 40), random.choice([1, 2])))
        splits = [0] + splits + [height]
        for i in range(len(splits) - 1):
            color = random_distinct_color()
            bg[splits[i] : splits[i + 1], :] = color

    elif mode == "split_v":
        # Vertical split with 2-3 color regions
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        splits = sorted(random.sample(range(40, width - 40), random.choice([1, 2])))
        splits = [0] + splits + [width]
        for i in range(len(splits) - 1):
            color = random_distinct_color()
            bg[:, splits[i] : splits[i + 1]] = color

    elif mode == "quadrants":
        # 4 quadrants with 4 distinct colors
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        my = height // 2 + random.randint(-20, 20)
        mx = width // 2 + random.randint(-20, 20)
        bg[0:my, 0:mx] = random_distinct_color()
        bg[0:my, mx:width] = random_distinct_color()
        bg[my:height, 0:mx] = random_distinct_color()
        bg[my:height, mx:width] = random_distinct_color()

    elif mode == "gradient":
        # Smooth 2-color or 4-color gradient
        c_tl = np.array(random_distinct_color(), dtype=np.float32)
        c_tr = np.array(random_distinct_color(), dtype=np.float32)
        c_bl = np.array(random_distinct_color(), dtype=np.float32)
        c_br = np.array(random_distinct_color(), dtype=np.float32)

        tx = np.linspace(0, 1, width, dtype=np.float32).reshape(1, width, 1)
        ty = np.linspace(0, 1, height, dtype=np.float32).reshape(height, 1, 1)

        top = c_tl * (1 - tx) + c_tr * tx
        bot = c_bl * (1 - tx) + c_br * tx
        bg = (top * (1 - ty) + bot * ty).astype(np.uint8)

    else:  # patches
        bg = np.full((height, width, 3), random_distinct_color(), dtype=np.uint8)
        for _ in range(random.randint(2, 5)):
            x1 = random.randint(0, width - 40)
            y1 = random.randint(0, height - 40)
            w_box = random.randint(30, 90)
            h_box = random.randint(30, 90)
            bg[y1 : min(height, y1 + h_box), x1 : min(width, x1 + w_box)] = random_distinct_color()

    return bg


class MovingObject:
    """An object moving across 16 frames in one of 4 directions."""

    def __init__(
        self,
        symbol: str,
        action: str,
        size: int,
        start_frame: int,
        speed: float,
        start_x: float,
        start_y: float,
    ):
        self.symbol = symbol
        self.action = action
        self.size = size
        self.start_frame = start_frame
        self.speed = speed
        self.start_x = start_x
        self.start_y = start_y
        self.sprite = render_symbol(symbol, size)

        # Direction velocities
        if action == "gotop":
            self.dx = 0.0
            self.dy = -speed
        elif action == "gobottom":
            self.dx = 0.0
            self.dy = speed
        elif action == "goleft":
            self.dx = -speed
            self.dy = 0.0
        elif action == "goright":
            self.dx = speed
            self.dy = 0.0
        else:
            raise ValueError(f"Unknown action: {action}")

    def get_position(self, frame_idx: int) -> Tuple[float, float]:
        """Compute (x, y) at a given frame_idx."""
        dt = frame_idx - self.start_frame
        return self.start_x + self.dx * dt, self.start_y + self.dy * dt

    def is_active(self, frame_idx: int) -> bool:
        """Whether the object has spawned by frame_idx."""
        return frame_idx >= self.start_frame


def paste_sprite_rgba(
    canvas: np.ndarray,
    bg_mask: np.ndarray,
    sprite: Image.Image,
    x: int,
    y: int,
) -> bool:
    """Alpha-blend an RGBA sprite onto canvas and update bg_mask in-place.

    Args:
        canvas: RGB image array of shape (H, W, 3), modified in-place.
        bg_mask: Grayscale binary mask of shape (H, W), uint8.
                 255 represents background, 0 represents foreground object.
        sprite: RGBA PIL sprite image of shape (sw, sh).
        x, y: Canvas coordinates where sprite top-left corner is positioned.

    Returns:
        True if at least part of the sprite was rendered inside the canvas, False otherwise.
    """
    h, w, _ = canvas.shape
    sw, sh = sprite.size
    sprite_arr = np.array(sprite)  # (sh, sw, 4)

    # Intersect coordinates
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + sw)
    y2 = min(h, y + sh)

    if x1 >= x2 or y1 >= y2:
        return False

    # Sprite slice
    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    sprite_crop = sprite_arr[sy1:sy2, sx1:sx2]
    rgb = sprite_crop[..., :3].astype(np.float32)
    alpha = (sprite_crop[..., 3:] / 255.0).astype(np.float32)

    # Blend onto canvas
    target_slice = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = target_slice * (1.0 - alpha) + rgb * alpha
    canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # Update bg_mask: wherever alpha > 0.05, it is foreground (bg_mask becomes 0)
    fg_mask_region = (alpha[..., 0] > 0.05)
    mask_slice = bg_mask[y1:y2, x1:x2]
    mask_slice[fg_mask_region] = 0
    bg_mask[y1:y2, x1:x2] = mask_slice
    return True


def generate_single_sample() -> Tuple[List[np.ndarray], List[np.ndarray], str, List[Dict], np.ndarray]:
    """Generate a 16-frame clip, 16 background masks, ground-truth label, and per-frame object counts.

    Returns:
        frames: List of 16 RGB numpy arrays of shape (224, 224, 3)
        bg_masks: List of 16 uint8 numpy arrays of shape (224, 224) (255: background, 0: object)
        label_str: Ground-truth label string (e.g. 'x+:gotop, y:gobottom')
        objects_meta: List of object metadata dicts
        obj_counts: Float32 numpy array of shape (16, 6) giving visible object counts per symbol per frame
    """
    # 1. Background
    bg = generate_multi_region_background(CANVAS_SIZE, CANVAS_SIZE)

    # 2. Decide objects
    num_objects = random.randint(1, 4)
    objects: List[MovingObject] = []
    objects_meta: List[Dict] = []

    # Induce duplicate (symbol, action) with high probability (50%) to demonstrate group '+'
    make_group = random.random() < 0.5 and num_objects >= 2

    chosen_symbols = [random.choice(SYMBOLS) for _ in range(num_objects)]
    chosen_actions = [random.choice(ACTIONS) for _ in range(num_objects)]

    if make_group:
        # Force the first two objects to share the same symbol and action
        group_sym = random.choice(SYMBOLS)
        group_act = random.choice(ACTIONS)
        chosen_symbols[0] = group_sym
        chosen_symbols[1] = group_sym
        chosen_actions[0] = group_act
        chosen_actions[1] = group_act

    for sym, act in zip(chosen_symbols, chosen_actions):
        size = random.randint(MIN_OBJ_SIZE, MAX_OBJ_SIZE)
        speed = random.uniform(5.0, 10.0)
        # Appear at frame 0 or later (up to frame 7)
        start_frame = random.choice([0, 0, 0, random.randint(1, 7)])

        # Calculate travel span
        travel_span = speed * (NUM_FRAMES - 1 - start_frame)

        # Place initial (x0, y0) so object stays predominantly visible
        if act == "gotop":
            min_y = min(CANVAS_SIZE - size - 10, int(travel_span) + 10)
            start_y = random.randint(min_y, CANVAS_SIZE - size - 5)
            start_x = random.randint(5, CANVAS_SIZE - size - 5)
        elif act == "gobottom":
            max_y = max(5, CANVAS_SIZE - size - int(travel_span) - 10)
            start_y = random.randint(5, max_y)
            start_x = random.randint(5, CANVAS_SIZE - size - 5)
        elif act == "goleft":
            min_x = min(CANVAS_SIZE - size - 10, int(travel_span) + 10)
            start_x = random.randint(min_x, CANVAS_SIZE - size - 5)
            start_y = random.randint(5, CANVAS_SIZE - size - 5)
        else:  # goright
            max_x = max(5, CANVAS_SIZE - size - int(travel_span) - 10)
            start_x = random.randint(5, max_x)
            start_y = random.randint(5, CANVAS_SIZE - size - 5)

        obj = MovingObject(
            symbol=sym,
            action=act,
            size=size,
            start_frame=start_frame,
            speed=speed,
            start_x=float(start_x),
            start_y=float(start_y),
        )
        objects.append(obj)
        objects_meta.append(
            {
                "symbol": sym,
                "action": act,
                "size": size,
                "start_frame": start_frame,
                "speed": speed,
                "start_pos": (start_x, start_y),
            }
        )

    # 3. Compute Ground-Truth Labels:
    # Rule 1: Object accompanied by its action (e.g. x:gotop)
    # Rule 2: If a group of objects of the same type share the same action -> add '+' (e.g. x+:gotop)
    counts = Counter((obj.symbol, obj.action) for obj in objects)
    labels = []
    for (sym, act), cnt in counts.items():
        if cnt > 1:
            labels.append(f"{sym}+:{act}")
        else:
            labels.append(f"{sym}:{act}")

    labels.sort()
    label_str = ", ".join(labels)

    # 4. Render 16 frames, corresponding 16 bg_masks, and per-frame object counts (16, 6)
    frames: List[np.ndarray] = []
    bg_masks: List[np.ndarray] = []
    obj_counts = np.zeros((NUM_FRAMES, len(SYMBOLS)), dtype=np.float32)

    for t in range(NUM_FRAMES):
        frame = bg.copy()
        # Initialize bg_mask with 255 (background)
        bg_mask = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)

        for obj in objects:
            if obj.is_active(t):
                x, y = obj.get_position(t)
                rendered = paste_sprite_rgba(frame, bg_mask, obj.sprite, int(round(x)), int(round(y)))
                if rendered:
                    obj_counts[t, SYMBOL_TO_IDX[obj.symbol]] += 1.0

        frames.append(frame)
        bg_masks.append(bg_mask)

    return frames, bg_masks, label_str, objects_meta, obj_counts


def export_sample(
    output_dir: str = "toy_action/sample_output",
    video_name: str = "sample.mp4",
    mask_video_name: str = "sample_bg_mask.mp4",
    fps: float = 8.0,
) -> Tuple[str, str]:
    """Generate 1 sample and write .mp4 video, 16 frame images, and 16 bg_mask images."""
    os.makedirs(output_dir, exist_ok=True)
    frames, bg_masks, label_str, meta, obj_counts = generate_single_sample()

    # 1. Write RGB video
    video_path = os.path.join(output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (CANVAS_SIZE, CANVAS_SIZE))

    # 2. Write bg_mask video
    mask_video_path = os.path.join(output_dir, mask_video_name)
    out_mask = cv2.VideoWriter(mask_video_path, fourcc, fps, (CANVAS_SIZE, CANVAS_SIZE))

    for i, (frame, mask) in enumerate(zip(frames, bg_masks)):
        # cv2 expects BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)

        # Mask as 3-channel for video
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out_mask.write(mask_bgr)

        # Save individual frame image
        img_path = os.path.join(output_dir, f"frame_{i:02d}.png")
        cv2.imwrite(img_path, bgr)

        # Save individual bg_mask image (255: background, 0: object)
        mask_path = os.path.join(output_dir, f"bg_mask_{i:02d}.png")
        cv2.imwrite(mask_path, mask)

    out.release()
    out_mask.release()

    # Save label text file
    label_file = os.path.join(output_dir, "label.txt")
    with open(label_file, "w", encoding="utf-8") as f:
        f.write(f"Label: {label_str}\n")
        f.write("Mask explanation: 255 = Background, 0 = Foreground Object\n")
        f.write("Objects:\n")
        for m in meta:
            f.write(f"  - {m}\n")

    print(f"==================================================")
    print(f"[SAMPLE CREATED]")
    print(f"Label            : {label_str}")
    print(f"Video file       : {video_path}")
    print(f"Mask video file  : {mask_video_path}")
    print(f"Frames saved     : {output_dir}/frame_00.png .. frame_15.png")
    print(f"BG Masks saved   : {output_dir}/bg_mask_00.png .. bg_mask_15.png")
    print(f"Metadata         : {label_file}")
    print(f"==================================================")
    return video_path, label_str


def export_lmdb(
    lmdb_path: str,
    num_samples: int = 1000,
    map_size: int = 5 * 1024 * 1024 * 1024,
    jpeg_quality: int = 95,
) -> None:
    """Generate n samples into an LMDB database with frames, bg_masks, and labels."""
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)

    print(f"Generating {num_samples} samples into LMDB: {lmdb_path} ...")
    with env.begin(write=True) as txn:
        for idx in range(num_samples):
            frames, bg_masks, label_str, _, obj_counts = generate_single_sample()

            # Encode 16 frames as high-quality JPEGs
            encoded_frames = [
                cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes()
                for f in frames
            ]

            # Encode 16 bg_masks as lossless PNGs
            encoded_masks = [
                cv2.imencode(".png", m)[1].tobytes()
                for m in bg_masks
            ]

            video_payload = pickle.dumps(encoded_frames)
            mask_payload = pickle.dumps(encoded_masks)
            counts_payload = pickle.dumps(obj_counts)

            txn.put(f"video-{idx:09d}".encode("utf-8"), video_payload)
            txn.put(f"bg_mask-{idx:09d}".encode("utf-8"), mask_payload)
            txn.put(f"label-{idx:09d}".encode("utf-8"), label_str.encode("utf-8"))
            txn.put(f"obj_counts-{idx:09d}".encode("utf-8"), counts_payload)

            if (idx + 1) % 100 == 0 or idx == num_samples - 1:
                print(f"  Processed [{idx + 1}/{num_samples}] samples...")

        txn.put(b"num-samples", str(num_samples).encode("utf-8"))

    env.sync()
    env.close()
    print(f"LMDB successfully created at: {lmdb_path} ({num_samples} samples)")


def parse_args():
    parser = argparse.ArgumentParser(description="Toy Action Video Dataset Generator")
    parser.add_argument("--sample", action="store_true", help="Generate 1 sample (.mp4 + 16 images + 16 bg_masks)")
    parser.add_argument("--lmdb", action="store_true", help="Generate samples into LMDB database")
    parser.add_argument("--num-samples", "-n", type=int, default=1000, help="Number of samples to generate in LMDB")
    parser.add_argument("--val-samples", type=int, default=0, help="Number of validation samples to generate in val LMDB")
    parser.add_argument("--lmdb-path", type=str, default="toy_action/data/train_lmdb", help="LMDB destination path")
    parser.add_argument("--val-lmdb-path", type=str, default="toy_action/data/val_lmdb", help="Validation LMDB path")
    parser.add_argument("--sample-dir", type=str, default="toy_action/sample_output", help="Output directory for --sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not args.sample and not args.lmdb:
        print("No mode specified. Defaulting to --sample mode.")
        args.sample = True

    if args.sample:
        export_sample(output_dir=args.sample_dir)

    if args.lmdb:
        export_lmdb(lmdb_path=args.lmdb_path, num_samples=args.num_samples)
        if args.val_samples > 0:
            export_lmdb(lmdb_path=args.val_lmdb_path, num_samples=args.val_samples)


if __name__ == "__main__":
    main()
