import os
import sys
import argparse
import pickle
import lmdb
import cv2
import torch
import torch.nn as nn
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Filter and visualize road foreground using bg_seg, color distance, LayerNorm, and ReLU.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="D:/apps/Datasets/TACO/data/ap_Town07_lmdb",
        help="Path to LMDB directory containing rgb_front and mask_background folders.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the frame/scenario to inspect (0-based).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save result image. Default saves to artifact directory.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not attempt cv2.imshow (useful for headless/remote runs).",
    )
    parser.add_argument(
        "--mean-mode",
        type=str,
        default="road",
        choices=["road", "image", "spatial"],
        help="How local_mean is computed: 'road' (mean color of black region), 'image' (mean of entire image), or 'spatial' (local moving average filter).",
    )
    parser.add_argument(
        "--norm-scope",
        type=str,
        default="road",
        choices=["road", "full"],
        help="Scope of LayerNorm: 'road' (normalizes statistics of the black road region) or 'full' (nn.LayerNorm over full [H, W]).",
    )
    return parser.parse_args()


def load_sample_from_lmdb(input_dir: str, index: int):
    rgb_dir = os.path.join(input_dir, "rgb_front")
    bg_dir = os.path.join(input_dir, "mask_background")

    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"RGB folder not found: {rgb_dir}")
    if not os.path.exists(bg_dir):
        raise FileNotFoundError(f"mask_background folder not found: {bg_dir}")

    env_bg = lmdb.open(bg_dir, readonly=True, lock=False)
    env_rgb = lmdb.open(rgb_dir, readonly=True, lock=False)

    with env_bg.begin() as txn_bg, env_rgb.begin() as txn_rgb:
        cur = txn_bg.cursor()
        total_bg = txn_bg.stat()["entries"]
        if index < 0 or index >= total_bg:
            raise IndexError(f"Index {index} out of range [0, {total_bg - 1}]")

        cur.first()
        for _ in range(index):
            if not cur.next():
                raise IndexError(f"Cannot seek to index {index}")

        bg_key = cur.key().decode("utf-8")
        bg_raw = cur.value()

        # Derive RGB key corresponding to BG key
        rgb_key = bg_key.replace("/mask/background/", "/rgb/front/").rsplit(".", 1)[0] + ".jpg"
        rgb_raw = txn_rgb.get(rgb_key.encode("utf-8"))

        if rgb_raw is None:
            cur_rgb = txn_rgb.cursor()
            cur_rgb.first()
            for _ in range(index):
                cur_rgb.next()
            rgb_key = cur_rgb.key().decode("utf-8")
            rgb_raw = cur_rgb.value()

    try:
        rgb_obj = pickle.loads(rgb_raw)
        rgb_bytes = rgb_obj["image"] if isinstance(rgb_obj, dict) and "image" in rgb_obj else rgb_obj
    except Exception:
        rgb_bytes = rgb_raw

    try:
        bg_obj = pickle.loads(bg_raw)
        bg_bytes = bg_obj["image"] if isinstance(bg_obj, dict) and "image" in bg_obj else bg_obj
    except Exception:
        bg_bytes = bg_raw

    rgb_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
    bg_gray = cv2.imdecode(np.frombuffer(bg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    if rgb_bgr is None or bg_gray is None:
        raise ValueError(f"Failed to decode image or mask for index {index}")

    return rgb_bgr, bg_gray, bg_key, rgb_key


def main():
    args = parse_args()
    print(f"[INFO] Loading sample index {args.index} from {args.input_dir}...")
    rgb_bgr, bg_gray, bg_key, rgb_key = load_sample_from_lmdb(args.input_dir, args.index)
    print(f"  BG Key : {bg_key} (shape={bg_gray.shape})")
    print(f"  RGB Key: {rgb_key} (shape={rgb_bgr.shape})")

    # 1. Align mask resolution to RGB image resolution
    h_rgb, w_rgb = rgb_bgr.shape[:2]
    if bg_gray.shape[:2] != (h_rgb, w_rgb):
        bg_aligned = cv2.resize(bg_gray, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
    else:
        bg_aligned = bg_gray

    # 2. Get black region (bg_seg <= 127) -> Road + Vehicles / Pedestrians
    black_mask_np = (bg_aligned <= 127)
    black_ratio = np.mean(black_mask_np)
    print(f"[INFO] Black region ratio (bg_seg <= 127): {black_ratio * 100:.2f}% of image")

    # 3. Convert RGB to PyTorch tensor in [0, 1]
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb_rgb).float() / 255.0  # (H, W, 3)
    black_mask = torch.from_numpy(black_mask_np)   # (H, W), bool

    # 4. Compute local_mean according to mode
    if args.mean_mode == "road":
        if black_mask.any():
            local_mean = x[black_mask].mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, 3)
        else:
            local_mean = x.mean(dim=(0, 1), keepdim=True)
    elif args.mean_mode == "image":
        local_mean = x.mean(dim=(0, 1), keepdim=True)  # (1, 1, 3)
    else:  # spatial moving average
        x_4d = x.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        k_size = 31
        local_mean = torch.nn.functional.avg_pool2d(
            x_4d, kernel_size=k_size, stride=1, padding=k_size // 2
        ).squeeze(0).permute(1, 2, 0)  # (H, W, 3)

    # 5. Apply the color difference formula:
    # torch.exp(5 * torch.abs(x - local_mean))
    color_diff = torch.abs(x - local_mean)
    exp_diff = torch.exp(5.0 * color_diff)
    score_map = exp_diff.mean(dim=-1)  # (H, W)

    # 6. Apply LayerNorm followed by ReLU
    if args.norm_scope == "road":
        # Standardize strictly over the road/black region distribution
        vals = score_map[black_mask]
        mean_val = vals.mean()
        std_val = vals.std()
        normalized_score = torch.zeros_like(score_map)
        normalized_score[black_mask] = (score_map[black_mask] - mean_val) / (std_val + 1e-6)
    else:
        # Standard PyTorch nn.LayerNorm over full spatial dimensions [H, W]
        ln = nn.LayerNorm([score_map.shape[0], score_map.shape[1]], elementwise_affine=False)
        score_input = torch.where(black_mask, score_map, torch.zeros_like(score_map))
        normalized_score = ln(score_input)

    # Apply ReLU activation
    relu_score = torch.relu(normalized_score)

    # Ensure sky/buildings (non-black regions) remain zeroed out
    filtered_score = torch.where(black_mask, relu_score, torch.zeros_like(relu_score))

    active_road_ratio = (filtered_score[black_mask] > 0).float().mean().item()
    print(f"[INFO] Active non-zero foreground ratio within road after ReLU: {active_road_ratio * 100:.2f}%")
    print(f"       (The remaining {100 - active_road_ratio * 100:.2f}% of flat road background was filtered to 0)")

    # 7. Normalize active response to [0, 255] for visual heatmap
    max_val = filtered_score.max().item()
    if max_val > 0:
        norm_score = torch.clamp(filtered_score / max_val, 0.0, 1.0)
    else:
        norm_score = filtered_score

    score_gray = (norm_score.numpy() * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(score_gray, cv2.COLORMAP_JET)
    heatmap_bgr[~black_mask_np] = 0  # Black out non-road areas

    # Also zero out areas where ReLU produced 0 (flat road background) to show pure objects
    road_zero_mask = (score_gray == 0) & black_mask_np
    heatmap_bgr[road_zero_mask] = (20, 20, 20)  # Very dark gray for zeroed road

    # Masked RGB image: keep original RGB on black region only
    masked_rgb_bgr = rgb_bgr.copy()
    masked_rgb_bgr[~black_mask_np] = 0

    # Composite comparison panel
    # [1] Original RGB | [2] bg_seg mask | [3] Black region only | [4] LayerNorm -> ReLU
    bg_vis_bgr = cv2.cvtColor(bg_aligned, cv2.COLOR_GRAY2BGR)

    cv2.putText(rgb_bgr, f"Index {args.index}: Original RGB", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(bg_vis_bgr, "bg_seg (Black: Road+Vehicles, White: Sky/Bldg)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(masked_rgb_bgr, "Masked Road Region (bg_seg <= 127)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(heatmap_bgr, "ReLU(LayerNorm(exp(5*|x-mean|)))", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    top_row = np.hstack([rgb_bgr, bg_vis_bgr])
    bottom_row = np.hstack([masked_rgb_bgr, heatmap_bgr])
    combined_img = np.vstack([top_row, bottom_row])

    # Save to disk
    if args.save_path is None:
        save_dir = "C:/Users/hbaog/.gemini/antigravity/brain/a3bfaf0a-3396-4e7f-99fb-957f36dc6ae3"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"tmp_result_idx_{args.index}_ln_relu.png")
    else:
        save_path = args.save_path

    cv2.imwrite(save_path, combined_img)
    print(f"[SUCCESS] Saved filtered output visualization to: {save_path}")

    # Display on screen if GUI is supported and not disabled
    if not args.no_gui:
        try:
            screen_w = 1600
            scale = screen_w / combined_img.shape[1]
            disp_h = int(combined_img.shape[0] * scale)
            resized_disp = cv2.resize(combined_img, (screen_w, disp_h), interpolation=cv2.INTER_AREA)

            cv2.imshow(f"TACO bg_seg LN+ReLU - Index {args.index}", resized_disp)
            print("[INFO] Displaying image on screen. Press any key in the window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[NOTICE] cv2.imshow not available in current environment: {e}")
            print(f"[NOTICE] Image has been saved to {save_path}")


if __name__ == "__main__":
    main()
