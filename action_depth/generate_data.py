"""Perspective Toy Road Dataset Generator (2.5D Ground Plane Oxz).

Generates a synthetic dataset of symbols ('x', 'y', 'o', 'T', '8', 'O')
moving continuously in a 2.5D perspective ground world:
    - Ground plane Oxz:
        X in [-1.0 (left), 1.0 (right)]
        Z in [0.1 (near camera), 1.0 (far / horizon)]
    - Actions on Oxz:
        'far' (moving +Z: shrinks & rises towards horizon)
        'near' (moving -Z: grows & descends towards bottom)
        'left' (moving -X: lateral movement to left)
        'right' (moving +X: lateral movement to right)
    - Ground truth:
        1. Multi-label 48 classes: '<symbol>:<action>' and '<symbol>+:<action>'
        2. Continuous Ground-Truth 2D Oxz Trajectories: (T=16, S=6, 2) in [X, Z]
        3. Perspective RGB video (16, 224, 224, 3)
        4. bg_mask (16, 224, 224) (255: road/sky background, 0: object)
        5. Ground-Truth 2D Oxz BEV Map (16, 6, Z_res=32, X_res=32)
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

CANVAS_SIZE = 224
NUM_FRAMES = 16
HORIZON_Y = int(0.40 * CANVAS_SIZE)  # Y coordinate of road horizon in image (~90px)

SYMBOLS = ["x", "y", "o", "T", "8", "O"]
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOLS)}
# Actions defined in Oxz ground-plane!
ACTIONS = ["far", "near", "left", "right"]

SYMBOL_VARIANTS = []
for s in SYMBOLS:
    SYMBOL_VARIANTS.append(s)
    SYMBOL_VARIANTS.append(f"{s}+")

ALL_CLASSES = [f"{s}:{a}" for s in SYMBOL_VARIANTS for a in ACTIONS]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)  # 48 classes


def random_distinct_color(exclude_color: Optional[Tuple[int, int, int]] = None, min_dist: float = 60.0) -> Tuple[int, int, int]:
    while True:
        c = (random.randint(25, 240), random.randint(25, 240), random.randint(25, 240))
        if exclude_color is None:
            return c
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c, exclude_color)))
        if dist >= min_dist:
            return c


def render_symbol(symbol: str, size: int) -> Image.Image:
    size = max(6, int(round(size)))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    w = max(1, size // 7)
    margin = max(1, size // 10)

    c1 = (*random_distinct_color(), 255)
    c2 = (*random_distinct_color(exclude_color=c1[:3], min_dist=70), 255)
    c3 = (*random_distinct_color(exclude_color=c2[:3], min_dist=70), 255)

    if symbol == "O":
        draw.ellipse([margin, margin, size - 1 - margin, size - 1 - margin], fill=c1, outline=c2, width=max(2, size // 7))
    elif symbol == "o":
        draw.ellipse([margin + 1, margin + 1, size - 1 - margin - 1, size - 1 - margin - 1], fill=None, outline=c1, width=max(1, size // 7))
    elif symbol == "x":
        draw.line([(margin, margin), (size - 1 - margin, size - 1 - margin)], fill=c1, width=w)
        draw.line([(size - 1 - margin, margin), (margin, size - 1 - margin)], fill=c2, width=w)
    elif symbol == "y":
        mid_x = size // 2
        mid_y = int(size * 0.48)
        draw.line([(margin, margin), (mid_x, mid_y)], fill=c1, width=w)
        draw.line([(size - 1 - margin, margin), (mid_x, mid_y)], fill=c2, width=w)
        draw.line([(mid_x, mid_y), (mid_x, size - 1 - margin)], fill=c3, width=w)
    elif symbol == "T":
        mid_x = size // 2
        bar_y = margin + w // 2
        draw.line([(margin, bar_y), (size - 1 - margin, bar_y)], fill=c1, width=w)
        draw.line([(mid_x, bar_y), (mid_x, size - 1 - margin)], fill=c2, width=w)
    elif symbol == "8":
        mid_y = size // 2
        w_loop = max(1, size // 8)
        draw.ellipse([margin + 1, margin, size - 1 - margin - 1, mid_y + 1], fill=None, outline=c1, width=w_loop)
        draw.ellipse([margin, mid_y - 1, size - 1 - margin, size - 1 - margin], fill=None, outline=c2, width=w_loop)

    return img


def draw_perspective_road_background(h: int = CANVAS_SIZE, w: int = CANVAS_SIZE, horizon_y: int = HORIZON_Y) -> np.ndarray:
    """Draw a 2.5D road perspective canvas: sky above horizon, road + grass/curb below."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Sky / Background scenery above horizon
    sky_c1 = np.array(random_distinct_color(), dtype=np.float32)
    sky_c2 = np.array(random_distinct_color(exclude_color=tuple(sky_c1.astype(int))), dtype=np.float32)
    for y in range(horizon_y):
        ratio = y / max(1, horizon_y)
        canvas[y, :] = (sky_c1 * (1 - ratio) + sky_c2 * ratio).astype(np.uint8)

    # 2. Road surface below horizon
    road_c = (55, 55, 60)
    canvas[horizon_y:, :] = road_c

    # Trapezoid road perspective borders
    top_road_w = int(w * 0.22)
    bot_road_w = int(w * 0.85)
    cx = w // 2
    road_pts = np.array([
        [cx - top_road_w // 2, horizon_y],
        [cx + top_road_w // 2, horizon_y],
        [cx + bot_road_w // 2, h - 1],
        [cx - bot_road_w // 2, h - 1],
    ], dtype=np.int32)
    # Lighter tarmac
    cv2.fillPoly(canvas, [road_pts], (75, 75, 80))

    # Lane dividing dashes
    for t_frac in [0.15, 0.35, 0.55, 0.75, 0.92]:
        v = int(horizon_y + t_frac * (h - horizon_y))
        dash_len = max(2, int(8 * t_frac))
        cv2.line(canvas, (cx, v - dash_len), (cx, v + dash_len), (220, 220, 220), max(1, int(2 * t_frac)))

    return canvas


class PerspectiveObject:
    """An object moving on ground-plane Oxz projected to camera view using perspective equations."""

    def __init__(self, symbol: str, action: str, start_frame: int = 0):
        self.symbol = symbol
        self.action = action
        self.start_frame = start_frame

        # Speed on Oxz plane (units per frame, ground span ~ 0.8)
        speed = random.uniform(0.035, 0.055)

        # Actions on ground Oxz:
        # X in [-0.85, 0.85], Z in [0.15 (near), 0.85 (far)]
        if action == "far":
            self.x0 = random.uniform(-0.6, 0.6)
            self.z0 = random.uniform(0.15, 0.35)
            self.vx = random.uniform(-0.005, 0.005)
            self.vz = speed
        elif action == "near":
            self.x0 = random.uniform(-0.6, 0.6)
            self.z0 = random.uniform(0.70, 0.90)
            self.vx = random.uniform(-0.005, 0.005)
            self.vz = -speed
        elif action == "left":
            self.x0 = random.uniform(0.35, 0.75)
            self.z0 = random.uniform(0.25, 0.75)
            self.vx = -speed
            self.vz = random.uniform(-0.005, 0.005)
        elif action == "right":
            self.x0 = random.uniform(-0.75, -0.35)
            self.z0 = random.uniform(0.25, 0.75)
            self.vx = speed
            self.vz = random.uniform(-0.005, 0.005)

        # Base nominal size at Z = 0.5
        self.base_size = random.randint(18, 26)
        self.sprite = render_symbol(symbol, self.base_size)

    def get_ground_pos(self, frame_idx: int) -> Tuple[float, float]:
        """Compute ground-plane coordinates (X, Z) at frame_idx."""
        dt = frame_idx - self.start_frame
        x = float(np.clip(self.x0 + self.vx * dt, -0.95, 0.95))
        z = float(np.clip(self.z0 + self.vz * dt, 0.10, 0.95))
        return x, z

    def project_to_image(self, x: float, z: float, h: int = CANVAS_SIZE, w: int = CANVAS_SIZE, horizon_y: int = HORIZON_Y) -> Tuple[int, int, int]:
        """Perspective projection from (X, Z) ground coordinates to (u_pixel, v_foot_pixel, sprite_size).
        
        Physics / Geometry:
            z in [0 (near/bottom), 1 (far/horizon)]
            dy_from_horizon = (1 - z)^1.8 * (h - horizon_y)
            v_foot = horizon_y + dy_from_horizon
            u = w/2 + (x * fan_scale) * (w / 2)
            sprite_size proportional to 1 / (z + 0.35)
        """
        # Vertical ground coordinate in image:
        dy = ((1.0 - z) ** 1.6) * (h - 1 - horizon_y)
        v_foot = int(round(horizon_y + dy))

        # Horizontal coordinate in image (perspective widening at near Z):
        fan_scale = 0.25 + 0.75 * (1.0 - z)
        u_center = int(round((w // 2) + x * fan_scale * (w * 0.44)))

        # Apparent size shrinks with depth Z:
        scale_factor = 0.35 + 0.85 * (1.0 - z)
        pixel_size = max(7, int(round(self.base_size * scale_factor)))

        return u_center, v_foot, pixel_size

    def is_active(self, frame_idx: int) -> bool:
        return frame_idx >= self.start_frame


def paste_sprite_perspective(
    canvas: np.ndarray,
    bg_mask: np.ndarray,
    sprite: Image.Image,
    u_center: int,
    v_foot: int,
    pixel_size: int,
) -> bool:
    h, w, _ = canvas.shape
    if pixel_size != sprite.size[0]:
        resized_sprite = sprite.resize((pixel_size, pixel_size), Image.BILINEAR)
    else:
        resized_sprite = sprite

    sw, sh = resized_sprite.size
    top_left_x = u_center - sw // 2
    top_left_y = v_foot - sh  # Sprite touches ground plane at v_foot!

    x1 = max(0, top_left_x)
    y1 = max(0, top_left_y)
    x2 = min(w, top_left_x + sw)
    y2 = min(h, top_left_y + sh)

    if x1 >= x2 or y1 >= y2:
        return False

    sx1 = max(0, -top_left_x)
    sy1 = max(0, -top_left_y)
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    arr = np.array(resized_sprite)
    sprite_crop = arr[sy1:sy2, sx1:sx2]
    rgb = sprite_crop[..., :3].astype(np.float32)
    alpha = (sprite_crop[..., 3:] / 255.0).astype(np.float32)

    target_slice = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = target_slice * (1.0 - alpha) + rgb * alpha
    canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    fg_region = (alpha[..., 0] > 0.05)
    mask_slice = bg_mask[y1:y2, x1:x2]
    mask_slice[fg_region] = 0
    bg_mask[y1:y2, x1:x2] = mask_slice
    return True


def generate_single_perspective_sample() -> Tuple[
    List[np.ndarray], List[np.ndarray], str, np.ndarray, np.ndarray
]:
    """Generates:
        frames: 16 RGB images (224, 224, 3)
        bg_masks: 16 masks (224, 224) (255: bg, 0: object)
        label_str: e.g. 'x:far, y:left'
        gt_oxz_coords: (16, 6, 2) Ground-truth (X, Z) per slot per frame
        gt_oxz_bev: (16, 6, 32, 32) Ground-truth 2D Oxz density heatmap
    """
    bg_canvas = draw_perspective_road_background(CANVAS_SIZE, CANVAS_SIZE, HORIZON_Y)

    num_objects = random.randint(1, 4)
    chosen_symbols = [random.choice(SYMBOLS) for _ in range(num_objects)]
    chosen_actions = [random.choice(ACTIONS) for _ in range(num_objects)]

    make_group = (random.random() < 0.45) and (num_objects >= 2)
    if make_group:
        grp_s = random.choice(SYMBOLS)
        grp_a = random.choice(ACTIONS)
        chosen_symbols[0] = grp_s
        chosen_symbols[1] = grp_s
        chosen_actions[0] = grp_a
        chosen_actions[1] = grp_a

    objects: List[PerspectiveObject] = []
    for s, a in zip(chosen_symbols, chosen_actions):
        st = random.choice([0, 0, random.randint(1, 4)])
        objects.append(PerspectiveObject(s, a, start_frame=st))

    counts = Counter((obj.symbol, obj.action) for obj in objects)
    labels = []
    for (sym, act), cnt in counts.items():
        if cnt > 1:
            labels.append(f"{sym}+:{act}")
        else:
            labels.append(f"{sym}:{act}")
    labels.sort()
    label_str = ", ".join(labels)

    frames: List[np.ndarray] = []
    bg_masks: List[np.ndarray] = []

    # (16, 6, 2) Ground truth (X in [-1, 1], Z in [0, 1])
    gt_oxz_coords = np.zeros((NUM_FRAMES, len(SYMBOLS), 2), dtype=np.float32)
    # (16, 6, 32, 32) Ground truth 2D Oxz BEV heatmap (Z_res=32, X_res=32)
    bev_res = 32
    gt_oxz_bev = np.zeros((NUM_FRAMES, len(SYMBOLS), bev_res, bev_res), dtype=np.float32)

    for t in range(NUM_FRAMES):
        frame = bg_canvas.copy()
        bg_mask = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)

        for obj in objects:
            if obj.is_active(t):
                x_g, z_g = obj.get_ground_pos(t)
                s_idx = SYMBOL_TO_IDX[obj.symbol]
                gt_oxz_coords[t, s_idx] = [x_g, z_g]

                u, v_foot, psize = obj.project_to_image(x_g, z_g, CANVAS_SIZE, CANVAS_SIZE, HORIZON_Y)
                rendered = paste_sprite_perspective(frame, bg_mask, obj.sprite, u, v_foot, psize)
                if rendered:
                    # Place a Gaussian blob on 2D Oxz map:
                    # x in [-1, 1] -> col in [0, bev_res-1]
                    # z in [0, 1] -> row in [0, bev_res-1] (row=0 is near Z=0, row=31 is far Z=1)
                    col = int(np.clip(round((x_g * 0.5 + 0.5) * (bev_res - 1)), 0, bev_res - 1))
                    row = int(np.clip(round(z_g * (bev_res - 1)), 0, bev_res - 1))
                    gt_oxz_bev[t, s_idx, row, col] = 1.0

        for s_idx in range(len(SYMBOLS)):
            if gt_oxz_bev[t, s_idx].max() > 0:
                gt_oxz_bev[t, s_idx] = cv2.GaussianBlur(gt_oxz_bev[t, s_idx], (5, 5), 1.2)
                gt_oxz_bev[t, s_idx] /= (gt_oxz_bev[t, s_idx].max() + 1e-6)

        frames.append(frame)
        bg_masks.append(bg_mask)

    return frames, bg_masks, label_str, gt_oxz_coords, gt_oxz_bev


def generate_lmdb_dataset(output_dir: str, num_samples: int = 500):
    os.makedirs(output_dir, exist_ok=True)
    map_size = int(num_samples * 224 * 224 * 3 * 16 * 1.5)
    env = lmdb.open(output_dir, map_size=max(map_size, 1024 * 1024 * 512))

    with env.begin(write=True) as txn:
        txn.put(b"num-samples", str(num_samples).encode("utf-8"))
        for i in range(num_samples):
            frames, bg_masks, label_str, gt_coords, gt_bev = generate_single_perspective_sample()

            encoded_frames = [cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))[1].tobytes() for f in frames]
            encoded_masks = [cv2.imencode(".png", m)[1].tobytes() for m in bg_masks]

            txn.put(f"video-{i:09d}".encode("utf-8"), pickle.dumps(encoded_frames))
            txn.put(f"label-{i:09d}".encode("utf-8"), label_str.encode("utf-8"))
            txn.put(f"bg_mask-{i:09d}".encode("utf-8"), pickle.dumps(encoded_masks))
            txn.put(f"oxz_coords-{i:09d}".encode("utf-8"), pickle.dumps(gt_coords))
            txn.put(f"oxz_bev-{i:09d}".encode("utf-8"), pickle.dumps(gt_bev))

            if (i + 1) % 50 == 0 or i == num_samples - 1:
                print(f"[LMDB] Generated {i + 1}/{num_samples} samples into {output_dir}")

    env.close()
    print(f"[SUCCESS] LMDB created at {output_dir} with {num_samples} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", action="store_true")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--val-samples", type=int, default=60)
    parser.add_argument("--sample", action="store_true", default=True)
    args = parser.parse_args()

    if args.lmdb:
        generate_lmdb_dataset("action_depth/data/train_lmdb", num_samples=args.num_samples)
        generate_lmdb_dataset("action_depth/data/val_lmdb", num_samples=args.val_samples)
    else:
        frames, bg_masks, label_str, gt_coords, gt_bev = generate_single_perspective_sample()
        print(f"Sample generated! Label: {label_str}")
        print(f"Video shape: {len(frames)} frames, size: {frames[0].shape}")
        print(f"gt_oxz_coords shape: {gt_coords.shape}, gt_oxz_bev shape: {gt_bev.shape}")
