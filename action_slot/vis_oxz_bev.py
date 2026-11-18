"""Standalone Full-Frame 2D Oxz Ground-Plane & Ego-Pose Trajectory Visualizer for TACO LMDB.

Features:
  1. Runs across ALL frames of a scenario by default (`--num-frames 0`).
  2. Estimates per-frame Ego Camera Position `(X_ego(t), Z_ego(t))` and Heading/Yaw Angle
     `theta_ego(t)` on the 2D Oxz plane using:
       - Distant background (`bg_seg > 0.5`) optical flow -> camera yaw rate `d_theta(t)`.
       - Static road/curb (`bg_seg <= 0.5` & `relu == 0`) rotation-compensated downward flow
         -> camera forward translation `ds(t)`.
  3. Projects foreground objects (`ReLU(LayerNorm(exp(5 * |x - local_mean|))) > 0`) onto both:
       - Ego-Centric 2D Oxz Map (camera frame)
       - World-Stabilized 2D Oxz Map (compensated for ego position & yaw rotation), showing
         camera frustums/heading arrows along the ego trajectory.
  4. Builds the 2D Conv2d feature channels (`Max Occupancy` & `Chronological Time Ramp`).

Usage:
    python action_slot/vis_oxz_bev.py --input-dir "D:/apps/Datasets/TACO/data/ap_Town07_lmdb" --index 0
"""

import argparse
from collections import defaultdict
import json
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-frame 2D Oxz ground trajectory & ego-pose (position + heading) visualizer."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="D:/apps/Datasets/TACO/data/ap_Town07_lmdb",
        help="Path to LMDB directory containing rgb_front and mask_background.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Scenario index (0-based) or starting frame index if --mode=frame.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="scenario",
        choices=["scenario", "frame"],
        help="'scenario': load full frames of scenario #index; 'frame': load frames starting at #index.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Number of frames to process. 0 (default) = ALL frames in the scenario.",
    )
    parser.add_argument(
        "--bev-h",
        type=int,
        default=340,
        help="Height (Z resolution) of each rendered 2D Oxz map.",
    )
    parser.add_argument(
        "--bev-w",
        type=int,
        default=360,
        help="Width (X resolution) of each rendered 2D Oxz map.",
    )
    parser.add_argument(
        "--hfov-deg",
        type=float,
        default=120.0,
        help="Approximate horizontal FOV in degrees for yaw estimation (default: 120.0).",
    )
    parser.add_argument(
        "--horizon-ratio",
        type=float,
        default=0.43,
        help="Approximate vertical horizon ratio in image (default: 0.43).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save output visualization PNG.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Optional path to save full-frame animated MP4 video of the 2D Oxz tracking.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip cv2.imshow window (for headless runs).",
    )
    return parser.parse_args()


def _decode_bytes(raw: bytes) -> bytes:
    try:
        obj = pickle.loads(raw)
        if isinstance(obj, dict) and "image" in obj:
            return obj["image"]
        return obj
    except Exception:
        return raw


def load_sequence_from_lmdb(
    input_dir: str,
    index: int,
    num_frames: int = 0,
    mode: str = "scenario",
) -> Tuple[List[np.ndarray], List[np.ndarray], str, str]:
    """Load full (or `num_frames`) RGB frames and `bg_seg` masks from an LMDB scenario."""
    rgb_dir = os.path.join(input_dir, "rgb_front")
    bg_dir = os.path.join(input_dir, "mask_background")
    jsonl_path = os.path.join(input_dir, "retrieve_gt_all.jsonl")

    if not os.path.exists(rgb_dir) or not os.path.exists(bg_dir):
        raise FileNotFoundError(f"Missing rgb_front or mask_background in {input_dir}")

    gt_map: Dict[str, str] = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sp = d.get("scenario_path", "")
                parts = sp.split("/", 1)
                short_sp = parts[1] if len(parts) > 1 else sp
                content = d.get("content", "").replace("\n", ", ").strip(", ")
                gt_map[short_sp] = content

    env_bg = lmdb.open(bg_dir, readonly=True, lock=False)
    env_rgb = lmdb.open(rgb_dir, readonly=True, lock=False)

    with env_bg.begin() as txn_bg, env_rgb.begin() as txn_rgb:
        cur = txn_bg.cursor()
        if mode == "scenario":
            scen_dict: Dict[str, List[str]] = defaultdict(list)
            for k, _ in cur:
                ks = k.decode("utf-8")
                scen = ks.split("/mask/background/")[0]
                scen_dict[scen].append(ks)
            scen_list = list(scen_dict.items())
            if index < 0 or index >= len(scen_list):
                raise IndexError(f"Scenario index {index} out of range [0, {len(scen_list) - 1}]")
            scen_name, all_bg_keys = scen_list[index]
            all_bg_keys.sort()
            if num_frames is not None and num_frames > 0 and num_frames < len(all_bg_keys):
                idxs = np.linspace(0, len(all_bg_keys) - 1, num_frames).round().astype(int)
                selected_bg_keys = [all_bg_keys[i] for i in idxs]
            else:
                selected_bg_keys = all_bg_keys
        else:
            all_bg_keys = [k.decode("utf-8") for k, _ in cur]
            if index < 0 or index >= len(all_bg_keys):
                raise IndexError(f"Frame index {index} out of range [0, {len(all_bg_keys) - 1}]")
            scen_name = all_bg_keys[index].split("/mask/background/")[0]
            same_scen = [k for k in all_bg_keys[index:] if k.startswith(scen_name + "/")]
            if num_frames is not None and num_frames > 0:
                selected_bg_keys = same_scen[:num_frames]
            else:
                selected_bg_keys = same_scen

        rgb_frames: List[np.ndarray] = []
        bg_masks: List[np.ndarray] = []
        for bg_k in selected_bg_keys:
            bg_raw = txn_bg.get(bg_k.encode("utf-8"))
            rgb_k = bg_k.replace("/mask/background/", "/rgb/front/").rsplit(".", 1)[0] + ".jpg"
            rgb_raw = txn_rgb.get(rgb_k.encode("utf-8"))
            if rgb_raw is None or bg_raw is None:
                continue

            rgb_bgr = cv2.imdecode(np.frombuffer(_decode_bytes(rgb_raw), np.uint8), cv2.IMREAD_COLOR)
            bg_gray = cv2.imdecode(np.frombuffer(_decode_bytes(bg_raw), np.uint8), cv2.IMREAD_GRAYSCALE)
            if rgb_bgr is not None and bg_gray is not None:
                rgb_frames.append(rgb_bgr)
                bg_masks.append(bg_gray)

    gt_label = gt_map.get(scen_name, "N/A")
    return rgb_frames, bg_masks, scen_name, gt_label


def estimate_ego_pose_oxz(
    rgb_frames: List[np.ndarray],
    bg_masks: List[np.ndarray],
    fg_relus: List[torch.Tensor],
    work_hw: Tuple[int, int] = (256, 768),
    hfov_deg: float = 120.0,
    horizon_ratio: float = 0.43,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate Ego Camera trajectory `(X_ego(t), Z_ego(t), theta_ego(t))` across all T frames.

    Returns:
        ego_pos: (T, 2) array of `(x_ego, z_ego)` in normalized world units where (0, 0) is start.
        ego_yaw: (T,) array of camera heading angles `theta_ego` in radians (`0` = +Z initial heading,
                 `> 0` = turned right, `< 0` = turned left).
        ego_shifts_px: (T, 2) array of `[du_bg_px, dv_bg_px]` per frame for background motion compensation.
    """
    wh, ww = work_hw
    num_t = len(rgb_frames)
    hfov_rad = math.radians(float(hfov_deg))
    v_horizon = float(horizon_ratio * wh)

    grays: List[np.ndarray] = []
    bgs: List[np.ndarray] = []
    for t in range(num_t):
        small_bgr = cv2.resize(rgb_frames[t], (ww, wh), interpolation=cv2.INTER_AREA)
        grays.append(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY))
        bgs.append(cv2.resize(bg_masks[t], (ww, wh), interpolation=cv2.INTER_NEAREST))

    d_yaw = np.zeros(num_t, dtype=np.float32)
    d_fwd = np.zeros(num_t, dtype=np.float32)
    ego_shifts_px = np.zeros((num_t, 2), dtype=np.float32)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    for t in range(1, num_t):
        prev_g = grays[t - 1]
        curr_g = grays[t]
        bg_mask_prev = (bgs[t - 1] > 127).astype(np.uint8) * 255
        # Focus on horizon/building band (avoid blank top sky)
        bg_mask_prev[: int(wh * 0.08), :] = 0

        # 1. Estimate Yaw Rotation d_theta from distant background (skyline/buildings/trees)
        pts_bg = cv2.goodFeaturesToTrack(
            prev_g, maxCorners=180, qualityLevel=0.01, minDistance=8, mask=bg_mask_prev
        )
        du_bg = 0.0
        dv_bg = 0.0
        if pts_bg is not None and len(pts_bg) >= 6:
            pts_next, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pts_bg, None, **lk_params)
            valid = (st.squeeze(-1) == 1)
            if np.any(valid):
                flow_bg = (pts_next[valid] - pts_bg[valid]).reshape(-1, 2)
                # Filter out outliers
                med_u = float(np.median(flow_bg[:, 0]))
                med_v = float(np.median(flow_bg[:, 1]))
                inliers = np.abs(flow_bg[:, 0] - med_u) < 6.0
                if np.any(inliers):
                    du_bg = float(np.median(flow_bg[inliers, 0]))
                    dv_bg = float(np.median(flow_bg[inliers, 1]))

        ego_shifts_px[t, 0] = du_bg
        ego_shifts_px[t, 1] = dv_bg
        # When camera turns right (d_yaw > 0), distant scene shifts left (du_bg < 0)
        d_yaw[t] = -float(du_bg / float(ww)) * hfov_rad

        # 2. Estimate Forward Translation ds from static road/curb points
        # (where bg_seg <= 127 AND foreground vehicle score is low)
        fg_prev = fg_relus[t - 1].numpy()
        road_static_mask = ((bgs[t - 1] <= 127) & (fg_prev < 0.35)).astype(np.uint8) * 255
        road_static_mask[: int(v_horizon + 8), :] = 0

        pts_rd = cv2.goodFeaturesToTrack(
            prev_g, maxCorners=160, qualityLevel=0.008, minDistance=7, mask=road_static_mask
        )
        step_fwd = 0.0
        if pts_rd is not None and len(pts_rd) >= 6:
            pts_rd_next, st_r, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pts_rd, None, **lk_params)
            valid_r = (st_r.squeeze(-1) == 1)
            if np.any(valid_r):
                p0 = pts_rd[valid_r].reshape(-1, 2)
                p1 = pts_rd_next[valid_r].reshape(-1, 2)
                dv_road = p1[:, 1] - p0[:, 1] - dv_bg
                du_road = p1[:, 0] - p0[:, 0] - du_bg

                # For forward camera motion, road points below horizon move downward (dv_road > 0)
                # and radially outward from center ((u - W/2) * du_road >= 0).
                dy = np.maximum(p0[:, 1] - v_horizon, 12.0) / float(wh - v_horizon)
                # Weight normalized downward flow
                fwd_samples = dv_road / (dy * 35.0 + 4.0)
                med_fwd = float(np.median(fwd_samples))
                # Suppress jitter when stationary
                if abs(med_fwd) > 0.012:
                    step_fwd = float(np.clip(med_fwd * 0.035, -0.02, 0.045))

        d_fwd[t] = step_fwd

    # Smooth yaw rates and forward speeds with a 5-frame moving average
    kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float32)
    d_yaw_smooth = np.convolve(d_yaw, kernel, mode="same")
    d_fwd_smooth = np.convolve(d_fwd, kernel, mode="same")

    ego_yaw = np.cumsum(d_yaw_smooth)  # (T,)
    ego_pos = np.zeros((num_t, 2), dtype=np.float32)  # (T, 2): [X_ego, Z_ego]
    for t in range(1, num_t):
        half_yaw = 0.5 * (ego_yaw[t - 1] + ego_yaw[t])
        ego_pos[t, 0] = ego_pos[t - 1, 0] + d_fwd_smooth[t] * math.sin(half_yaw)
        ego_pos[t, 1] = ego_pos[t - 1, 1] + d_fwd_smooth[t] * math.cos(half_yaw)

    return ego_pos, ego_yaw, ego_shifts_px


def compute_sequence_relu_ln_foreground(
    rgb_frames: List[np.ndarray],
    bg_masks: List[np.ndarray],
    work_hw: Tuple[int, int] = (256, 768),
    exp_factor: float = 5.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Compute per-frame raw ReLU(LayerNorm(exp(5 * |x - local_mean|))) and ego-compensated clean foreground."""
    wh, ww = work_hw
    num_t = len(rgb_frames)

    xs: List[torch.Tensor] = []
    bms: List[torch.Tensor] = []
    raw_relus: List[torch.Tensor] = []

    for t in range(num_t):
        rgb_small = cv2.resize(rgb_frames[t], (ww, wh), interpolation=cv2.INTER_AREA)
        rgb_rgb = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
        bg_small = cv2.resize(bg_masks[t], (ww, wh), interpolation=cv2.INTER_NEAREST)

        x = torch.from_numpy(rgb_rgb).float() / 255.0
        black_mask = torch.from_numpy(bg_small <= 127)
        xs.append(x)
        bms.append(black_mask)

        if black_mask.any():
            local_mean = x[black_mask].mean(dim=0, keepdim=True).unsqueeze(0)
        else:
            local_mean = x.mean(dim=(0, 1), keepdim=True)

        score_map = torch.exp(exp_factor * torch.abs(x - local_mean)).mean(dim=-1)
        fg_relu = torch.zeros_like(score_map)
        if black_mask.any():
            vals = score_map[black_mask]
            norm_vals = (vals - vals.mean()) / (vals.std() + 1e-6)
            fg_relu[black_mask] = torch.relu(norm_vals)
        raw_relus.append(fg_relu)

    return raw_relus, bms, xs


def refine_foreground_with_ego_shift(
    raw_relus: List[torch.Tensor],
    bms: List[torch.Tensor],
    xs: List[torch.Tensor],
    ego_shifts_px: np.ndarray,
) -> List[torch.Tensor]:
    """Use ego-shift compensated temporal difference to suppress static road lane paint."""
    num_t = len(raw_relus)
    if num_t <= 1:
        return raw_relus

    h, w, _ = xs[0].shape
    fg_clean_list: List[torch.Tensor] = []

    for t in range(num_t):
        t_prev = max(0, t - 1)
        t_next = min(num_t - 1, t + 1)
        du = float(ego_shifts_px[t, 0]) if t > 0 else float(ego_shifts_px[t_next, 0])
        dv = float(ego_shifts_px[t, 1]) if t > 0 else float(ego_shifts_px[t_next, 1])

        # Warp previous RGB frame by background shift (du, dv)
        M = np.float32([[1, 0, du], [0, 1, dv]])
        prev_np = xs[t_prev].numpy()
        warped_prev = cv2.warpAffine(prev_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        diff = torch.from_numpy(np.abs(xs[t].numpy() - warped_prev).mean(axis=-1))
        diff_smooth = F.avg_pool2d(diff.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze()

        fg_t = raw_relus[t]
        bm_t = bms[t]
        if bm_t.any():
            m_mean = diff_smooth[bm_t].mean() + 1e-5
            m_gate = torch.clamp(diff_smooth / (1.2 * m_mean), 0.2, 2.0)
            fg_t = fg_t * m_gate

        fg_smooth = F.avg_pool2d(fg_t.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze()
        fg_clean = torch.where(fg_smooth > 0.45, fg_t, torch.zeros_like(fg_t))
        fg_clean_list.append(fg_clean)

    return fg_clean_list


def project_frame_to_oxz(
    fg_relu: torch.Tensor,
    black_mask: torch.Tensor,
    ego_xy: Tuple[float, float],
    ego_theta: float,
    bev_h: int = 340,
    bev_w: int = 360,
    horizon_ratio: float = 0.43,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
]:
    """Project 2D camera foreground (H, W) into both Ego-Centric Oxz and World-Stabilized Oxz maps.

    Returns:
        bev_ego_map: (bev_h, bev_w) in Ego-centric coordinates
        bev_world_map: (bev_h, bev_w) in World coordinates (rotated by ego_theta & translated by ego_xy)
        c_ego_px: (col_x, row_z) centroid on Ego map
        c_world_px: (col_x, row_z) centroid on World map
        c_img_px: (u, v_foot) ground-contact point in image
    """
    h, w = fg_relu.shape
    fg_np = fg_relu.numpy()
    road_np = black_mask.numpy()

    bev_ego = np.zeros((bev_h, bev_w), dtype=np.float32)
    bev_world = np.zeros((bev_h, bev_w), dtype=np.float32)

    road_rows = np.where(road_np.any(axis=1))[0]
    v_horizon = float(
        min(
            int(horizon_ratio * h),
            int(road_rows.min()) if len(road_rows) > 0 else int(horizon_ratio * h),
        )
    )
    v_horizon = max(1.0, min(v_horizon, h - 10.0))

    col_energy = fg_np.sum(axis=0)
    col_max = fg_np.max(axis=0)
    active_cols = np.where((col_max > 0.55) & (col_energy > 2.0))[0]

    if len(active_cols) == 0:
        return bev_ego, bev_world, None, None, None

    v_indices = np.arange(h, dtype=np.float32)
    row_bottom_bias = np.linspace(0.5, 2.2, h, dtype=np.float32)[:, None]
    weighted_fg = (fg_np ** 1.5) * row_bottom_bias

    # World origin (Ego at t=0) is placed at (col = bev_w / 2, row = bev_h * 0.85)
    origin_col = bev_w * 0.5
    origin_row = bev_h * 0.85
    scale_z = bev_h * 0.78
    scale_x = bev_w * 0.44

    cos_t = math.cos(ego_theta)
    sin_t = math.sin(ego_theta)
    ex, ez = ego_xy

    ego_cols, ego_rows = [], []
    wld_cols, wld_rows = [], []
    img_us, img_vs = [], []
    weights = []

    for u in active_cols:
        w_col = weighted_fg[:, u]
        w_sum = w_col.sum()
        if w_sum <= 1e-5:
            continue
        v_foot = float((w_col * v_indices).sum() / w_sum)
        v_foot = max(v_horizon + 2.0, min(float(h - 1), v_foot))

        dy_norm = (v_foot - v_horizon) / max(1.0, (h - 1.0 - v_horizon))
        dy_norm = float(np.clip(dy_norm, 0.02, 1.0))

        # Local camera ground coordinates: z_cam in [0.04, 0.96], x_cam in [-1, 1]
        z_cam = float(np.clip(1.0 - (dy_norm ** 0.55), 0.04, 0.96))
        u_signed = (float(u) / float(w - 1)) * 2.0 - 1.0
        fan_scale = 0.28 + 0.72 * z_cam
        x_cam = u_signed * fan_scale

        # 1. Ego-centric map coordinates
        c_ego = origin_col + x_cam * scale_x
        r_ego = origin_row - z_cam * scale_z

        # 2. World-stabilized Oxz coordinates: rotate (x_cam, z_cam) by ego_theta and add (ex, ez)
        # Convention: +X is right, +Z is forward; theta > 0 is clockwise (right turn)
        x_world = ex + (x_cam * cos_t + z_cam * sin_t)
        z_world = ez + (-x_cam * sin_t + z_cam * cos_t)

        c_wld = origin_col + x_world * scale_x
        r_wld = origin_row - z_world * scale_z

        strength = float(col_max[u] * np.sqrt(col_energy[u]))

        re_i = int(np.clip(round(r_ego), 0, bev_h - 1))
        ce_i = int(np.clip(round(c_ego), 0, bev_w - 1))
        bev_ego[re_i, ce_i] += strength

        rw_i = int(np.clip(round(r_wld), 0, bev_h - 1))
        cw_i = int(np.clip(round(c_wld), 0, bev_w - 1))
        bev_world[rw_i, cw_i] += strength

        ego_cols.append(c_ego)
        ego_rows.append(r_ego)
        wld_cols.append(c_wld)
        wld_rows.append(r_wld)
        img_us.append(float(u))
        img_vs.append(v_foot)
        weights.append(strength)

    if bev_ego.max() > 0:
        bev_ego = cv2.GaussianBlur(bev_ego, (17, 17), 3.5)
        bev_ego /= bev_ego.max() + 1e-6
    if bev_world.max() > 0:
        bev_world = cv2.GaussianBlur(bev_world, (17, 17), 3.5)
        bev_world /= bev_world.max() + 1e-6

    if not weights:
        return bev_ego, bev_world, None, None, None

    w_arr = np.asarray(weights, dtype=np.float32)
    w_arr /= w_arr.sum() + 1e-6

    c_ego_pt = (float((np.asarray(ego_cols) * w_arr).sum()), float((np.asarray(ego_rows) * w_arr).sum()))
    c_wld_pt = (float((np.asarray(wld_cols) * w_arr).sum()), float((np.asarray(wld_rows) * w_arr).sum()))
    c_img_pt = (float((np.asarray(img_us) * w_arr).sum()), float((np.asarray(img_vs) * w_arr).sum()))

    return bev_ego, bev_world, c_ego_pt, c_wld_pt, c_img_pt


def time_color_bgr(t: int, total_t: int) -> Tuple[int, int, int]:
    val = int(round(255.0 * t / max(1, total_t - 1)))
    lut_in = np.array([[val]], dtype=np.uint8)
    bgr = cv2.applyColorMap(lut_in, cv2.COLORMAP_TURBO)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_oxz_world_grid(
    bev_h: int,
    bev_w: int,
    ego_pos: np.ndarray,
    ego_yaw: np.ndarray,
    draw_ego_pose_trail: bool = True,
) -> np.ndarray:
    """Draw 2D Oxz coordinate grid + Ego Camera trajectory & heading frustum arrows."""
    canvas = np.full((bev_h, bev_w, 3), (24, 20, 18), dtype=np.uint8)
    origin_col = bev_w * 0.5
    origin_row = bev_h * 0.85
    scale_z = bev_h * 0.78
    scale_x = bev_w * 0.44

    # Grid lines
    for frac in [0.15, 0.35, 0.55, 0.75, 0.85]:
        r = int(bev_h * frac)
        cv2.line(canvas, (0, r), (bev_w - 1, r), (48, 44, 40), 1, cv2.LINE_AA)
    for frac in [0.2, 0.5, 0.8]:
        c = int(bev_w * frac)
        cv2.line(canvas, (c, 0), (c, bev_h - 1), (48, 44, 40), 1, cv2.LINE_AA)

    # Initial camera FOV cone at t=0
    ego0 = (int(round(origin_col)), int(round(origin_row)))
    left_top = (int(bev_w * 0.04), int(bev_h * 0.05))
    right_top = (int(bev_w * 0.96), int(bev_h * 0.05))
    cv2.line(canvas, ego0, left_top, (75, 70, 62), 1, cv2.LINE_AA)
    cv2.line(canvas, ego0, right_top, (75, 70, 62), 1, cv2.LINE_AA)

    # Zone labels
    cv2.putText(canvas, "Z3 (Far Ahead)", (int(origin_col) - 48, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 180, 130), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Z1 (Left)", (10, int(bev_h * 0.48)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 180, 130), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Z2 (Right)", (bev_w - 76, int(bev_h * 0.48)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 180, 130), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Z4 (Ego Start)", (int(origin_col) + 12, bev_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 220), 1, cv2.LINE_AA)

    if not draw_ego_pose_trail:
        cv2.circle(canvas, ego0, 5, (0, 255, 255), -1, cv2.LINE_AA)
        return canvas

    # Draw Ego Camera trajectory (X_ego(t), Z_ego(t)) and camera heading frustums/arrows
    num_t = len(ego_pos)
    ego_px = []
    for t in range(num_t):
        cx = int(round(origin_col + ego_pos[t, 0] * scale_x))
        rz = int(round(origin_row - ego_pos[t, 1] * scale_z))
        ego_px.append((cx, rz))

    # Draw ego path line in bright cyan/white
    for t in range(num_t - 1):
        cv2.line(canvas, ego_px[t], ego_px[t + 1], (255, 220, 0), 2, cv2.LINE_AA)

    # Draw camera heading frustum wedges & arrows at 5 sampled steps along the sequence
    sample_ts = np.linspace(0, num_t - 1, min(6, num_t)).round().astype(int).tolist()
    for t in sample_ts:
        cx, rz = ego_px[t]
        th = float(ego_yaw[t])
        c_bgr = time_color_bgr(t, num_t)

        # Heading unit vector in image coords: +Z is up (-row), +X is right (+col)
        hx = math.sin(th)
        hz = -math.cos(th)
        tip = (int(round(cx + hx * 24)), int(round(rz + hz * 24)))

        # Mini FOV rays (+/- 35 deg around heading)
        for d_ang in (-0.6, 0.6):
            rx = math.sin(th + d_ang)
            rz_d = -math.cos(th + d_ang)
            r_end = (int(round(cx + rx * 32)), int(round(rz + rz_d * 32)))
            cv2.line(canvas, (cx, rz), r_end, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.arrowedLine(canvas, (cx, rz), tip, c_bgr, 2, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(canvas, (cx, rz), 3, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas


def overlay_trajectory_on_canvas(
    base_canvas: np.ndarray,
    bev_arr: np.ndarray,
    centroids: List[Optional[Tuple[float, float]]],
) -> np.ndarray:
    """Blend temporal BEV heatmaps (t=0..T-1) and centroid path onto a background canvas."""
    num_t = len(bev_arr)
    out = base_canvas.astype(np.float32)

    for t in range(num_t):
        b_map = bev_arr[t]
        if b_map.max() <= 1e-4:
            continue
        color = np.array(time_color_bgr(t, num_t), dtype=np.float32)
        alpha = np.clip(b_map * 0.65, 0.0, 0.85)[..., None]
        out = out * (1.0 - alpha) + color * alpha

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Smooth valid centroid path gently for clean trajectory line
    valid = [(t, pt) for t, pt in enumerate(centroids) if pt is not None]
    for k in range(len(valid) - 1):
        t0, (x0, z0) = valid[k]
        t1, (x1, z1) = valid[k + 1]
        c_bgr = time_color_bgr(t1, num_t)
        p0 = (int(round(x0)), int(round(z0)))
        p1 = (int(round(x1)), int(round(z1)))
        if k == len(valid) - 2:
            cv2.arrowedLine(out, p0, p1, c_bgr, 2, cv2.LINE_AA, tipLength=0.35)
        else:
            cv2.line(out, p0, p1, c_bgr, 2, cv2.LINE_AA)

    if valid:
        for idx_v in [0, len(valid) // 2, len(valid) - 1]:
            t, pt = valid[idx_v]
            c_bgr = time_color_bgr(t, num_t)
            p = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, p, 3, c_bgr, -1, cv2.LINE_AA)
            cv2.putText(
                out,
                f"t={t}",
                (p[0] + 6, max(14, p[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return out


def draw_ego_pose_telemetry_panel(
    panel_h: int,
    panel_w: int,
    ego_pos: np.ndarray,
    ego_yaw: np.ndarray,
    bev_world_arr: np.ndarray,
) -> np.ndarray:
    """Render a dedicated telemetry & Conv2d Time-Ramp panel showing Ego Yaw(t), Speed, and Time-Ramp."""
    num_t = len(ego_yaw)
    t_weights = np.linspace(0.1, 1.0, num_t, dtype=np.float32)[:, None, None]
    max_occ = bev_world_arr.max(axis=0)
    sum_occ = bev_world_arr.sum(axis=0)
    time_ramp = (bev_world_arr * t_weights).sum(axis=0) / np.maximum(sum_occ, 1e-4)
    time_ramp_masked = time_ramp * (max_occ > 0.08).astype(np.float32)

    ramp_vis = cv2.applyColorMap((np.clip(time_ramp_masked, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    ramp_vis[max_occ <= 0.08] = (20, 20, 20)
    ramp_vis = cv2.resize(ramp_vis, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

    # Draw a mini telemetry inset at the bottom of panel 3 showing Yaw angle (deg) & displacement
    inset_h = 88
    ramp_vis[panel_h - inset_h :, :] = (
        ramp_vis[panel_h - inset_h :, :].astype(np.float32) * 0.25 + np.array([28, 22, 18], dtype=np.float32) * 0.75
    ).astype(np.uint8)
    cv2.line(ramp_vis, (0, panel_h - inset_h), (panel_w - 1, panel_h - inset_h), (90, 90, 90), 1)

    yaw_deg = np.degrees(ego_yaw)
    total_yaw = float(yaw_deg[-1] - yaw_deg[0]) if num_t > 0 else 0.0
    total_dist = float(np.linalg.norm(ego_pos[-1] - ego_pos[0])) if num_t > 0 else 0.0

    if abs(total_yaw) < 4.0 and total_dist < 0.05:
        ego_state = "STOPPED / WAITING"
    elif total_yaw > 8.0:
        ego_state = f"TURN RIGHT (+{total_yaw:.1f} deg)"
    elif total_yaw < -8.0:
        ego_state = f"TURN LEFT ({total_yaw:.1f} deg)"
    else:
        ego_state = f"GO STRAIGHT (yaw={total_yaw:+.1f} deg)"

    cv2.putText(
        ramp_vis,
        f"Estimated Ego State ({num_t} full frames): {ego_state}",
        (10, panel_h - inset_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        ramp_vis,
        f"Ego dX={ego_pos[-1,0]:+.2f}, dZ={ego_pos[-1,1]:+.2f} | Final Heading={yaw_deg[-1]:+.1f} deg",
        (10, panel_h - inset_h + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Plot Yaw curve over t = 0..T-1
    gx0, gx1 = 14, panel_w - 14
    gy_mid = panel_h - 18
    cv2.line(ramp_vis, (gx0, gy_mid), (gx1, gy_mid), (80, 80, 80), 1)
    max_abs_y = max(15.0, float(np.max(np.abs(yaw_deg))) + 2.0)
    for t in range(num_t - 1):
        x0 = int(round(gx0 + (gx1 - gx0) * (t / max(1, num_t - 1))))
        x1 = int(round(gx0 + (gx1 - gx0) * ((t + 1) / max(1, num_t - 1))))
        y0 = int(round(gy_mid - (yaw_deg[t] / max_abs_y) * 18.0))
        y1 = int(round(gy_mid - (yaw_deg[t + 1] / max_abs_y) * 18.0))
        cv2.line(ramp_vis, (x0, y0), (x1, y1), time_color_bgr(t, num_t), 2, cv2.LINE_AA)

    return ramp_vis


def main():
    args = parse_args()
    rgb_frames, bg_masks, scen_name, gt_label = load_sequence_from_lmdb(
        args.input_dir, args.index, num_frames=args.num_frames, mode=args.mode
    )
    num_t = len(rgb_frames)
    print(f"[INFO] Loaded scenario '{scen_name}' with FULL {num_t} frames | GT: {gt_label}")

    work_h, work_w = 256, 768
    bev_h, bev_w = args.bev_h, args.bev_w

    # 1. Compute initial color-difference ReLU(LayerNorm(exp(5*|x - local_mean|)))
    raw_relus, bm_tensors, xs = compute_sequence_relu_ln_foreground(
        rgb_frames, bg_masks, work_hw=(work_h, work_w)
    )

    # 2. Estimate Ego Camera Pose (X_ego(t), Z_ego(t)) and Yaw Heading theta_ego(t) across all frames
    ego_pos, ego_yaw, ego_shifts_px = estimate_ego_pose_oxz(
        rgb_frames,
        bg_masks,
        raw_relus,
        work_hw=(work_h, work_w),
        hfov_deg=args.hfov_deg,
        horizon_ratio=args.horizon_ratio,
    )
    print(
        f"[INFO] Estimated Ego Pose over {num_t} frames: "
        f"final (X, Z)=({ego_pos[-1, 0]:+.3f}, {ego_pos[-1, 1]:+.3f}), "
        f"final Yaw={math.degrees(ego_yaw[-1]):+.2f} deg"
    )

    # 3. Refine foreground using ego-shift compensated frame difference
    fg_tensors = refine_foreground_with_ego_shift(raw_relus, bm_tensors, xs, ego_shifts_px)

    # 4. Project every frame t = 0..T-1 onto both Ego-Centric Oxz and World-Stabilized Oxz maps
    bev_ego_stack: List[np.ndarray] = []
    bev_world_stack: List[np.ndarray] = []
    ego_centroids: List[Optional[Tuple[float, float]]] = []
    world_centroids: List[Optional[Tuple[float, float]]] = []
    img_centroids: List[Optional[Tuple[float, float]]] = []

    for t in range(num_t):
        b_ego, b_wld, c_ego, c_wld, c_img = project_frame_to_oxz(
            fg_tensors[t],
            bm_tensors[t],
            ego_xy=(float(ego_pos[t, 0]), float(ego_pos[t, 1])),
            ego_theta=float(ego_yaw[t]),
            bev_h=bev_h,
            bev_w=bev_w,
            horizon_ratio=args.horizon_ratio,
        )
        bev_ego_stack.append(b_ego)
        bev_world_stack.append(b_wld)
        ego_centroids.append(c_ego)
        world_centroids.append(c_wld)
        img_centroids.append(c_img)

    bev_ego_arr = np.stack(bev_ego_stack, axis=0)      # (T_full, bev_h, bev_w)
    bev_world_arr = np.stack(bev_world_stack, axis=0)  # (T_full, bev_h, bev_w)

    # =========================================================================
    # Render 3 Bottom 2D Oxz Panels:
    #   Panel 1: World-Stabilized 2D Oxz Map (with Ego Camera Trajectory + Yaw Arrows + Object Trail)
    #   Panel 2: Ego-Centric 2D Oxz Map (Object Trail relative to camera)
    #   Panel 3: Conv2d Time-Ramp Map on World Oxz + Ego Pose Telemetry Curve
    # =========================================================================
    world_bg = draw_oxz_world_grid(bev_h, bev_w, ego_pos, ego_yaw, draw_ego_pose_trail=True)
    world_map_vis = overlay_trajectory_on_canvas(world_bg, bev_world_arr, world_centroids)

    ego_bg = draw_oxz_world_grid(bev_h, bev_w, ego_pos, ego_yaw, draw_ego_pose_trail=False)
    ego_map_vis = overlay_trajectory_on_canvas(ego_bg, bev_ego_arr, ego_centroids)

    # Render 4 keyframes across the full sequence at the top
    kf_indices = np.linspace(0, num_t - 1, 4).round().astype(int).tolist()
    kf_panels = []
    panel_w, panel_h = 360, 125

    for t in kf_indices:
        rgb_small = cv2.resize(rgb_frames[t], (work_w, work_h), interpolation=cv2.INTER_AREA)
        fg_t = fg_tensors[t].numpy()
        bm_t = bm_tensors[t].numpy()

        fg_norm = np.clip(fg_t / max(float(fg_t.max()), 1e-4), 0.0, 1.0)
        heat = cv2.applyColorMap((fg_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = rgb_small.copy()
        active_px = (fg_t > 0.45) & bm_t
        overlay[active_px] = cv2.addWeighted(rgb_small[active_px], 0.35, heat[active_px], 0.65, 0)
        overlay[~bm_t] = (overlay[~bm_t].astype(np.float32) * 0.55).astype(np.uint8)

        if img_centroids[t] is not None:
            cu, cv_foot = img_centroids[t]
            c_bgr = time_color_bgr(t, num_t)
            pt_img = (int(round(cu)), int(round(cv_foot)))
            cv2.circle(overlay, pt_img, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, pt_img, 5, c_bgr, -1, cv2.LINE_AA)

        yaw_deg_t = math.degrees(float(ego_yaw[t]))
        cv2.putText(
            overlay,
            f"t={t:02d}/{num_t-1:02d} | Ego({ego_pos[t,0]:+.2f},{ego_pos[t,1]:+.2f}) Yaw:{yaw_deg_t:+.1f}d",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            time_color_bgr(t, num_t),
            2,
            cv2.LINE_AA,
        )
        kf_panels.append(cv2.resize(overlay, (panel_w, panel_h), interpolation=cv2.INTER_AREA))

    top_strip = np.hstack(kf_panels)
    total_w = top_strip.shape[1]

    map_w = total_w // 3
    map_h = 380
    p_world = cv2.resize(world_map_vis, (map_w, map_h), interpolation=cv2.INTER_LINEAR)
    p_ego = cv2.resize(ego_map_vis, (map_w, map_h), interpolation=cv2.INTER_LINEAR)
    p_ramp = draw_ego_pose_telemetry_panel(map_h, map_w, ego_pos, ego_yaw, bev_world_arr)

    cv2.putText(
        p_world,
        "1. World 2D Oxz Map (Ego Pose/Yaw + Object Trail)",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        p_ego,
        "2. Camera-Centric 2D Oxz Map (Uncompensated)",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        p_ramp,
        "3. World Oxz Time-Ramp + Ego Yaw/Pose Telemetry",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    bottom_maps = np.hstack([p_world, p_ego, p_ramp])

    banner = np.full((44, total_w, 3), (15, 23, 42), dtype=np.uint8)
    cv2.putText(
        banner,
        f"Scenario [{args.index}]: {scen_name} ({num_t} Full Frames)  |  GT: {gt_label}  |  2D Oxz + Ego Pose & Heading",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    dashboard = np.vstack([banner, top_strip, bottom_maps])

    if args.save_path is None:
        save_dir = r"C:\Users\hbaog\.gemini\antigravity\brain\a3bfaf0a-3396-4e7f-99fb-957f36dc6ae3"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"oxz_full_scen_{args.index}.png")
    else:
        save_path = args.save_path

    cv2.imwrite(save_path, dashboard)
    print(f"[SUCCESS] Saved Full-Frame 2D Oxz + Ego Pose visualization to: {save_path}")

    if not args.no_gui:
        try:
            cv2.imshow(f"Full-Frame 2D Oxz & Ego Pose - Scenario {args.index}", dashboard)
            print("[INFO] Press any key in the image window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[NOTICE] GUI display skipped ({e}). View saved image at: {save_path}")


if __name__ == "__main__":
    main()
