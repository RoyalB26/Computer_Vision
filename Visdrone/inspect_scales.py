"""Quan sát và phân tích đầu ra đa quy mô P3 (//8), P4 (//16), P5 (//32) đối chiếu với GT Bounding Boxes.

Xuất ra:
  1. p3_p4_p5_features.png       : L2-Norm Activation & PCA-RGB của feat_p3, feat_p4, feat_p5 có vẽ đè GT boxes.
  2. p3_p4_p5_fg_bg_attn.png     : Attention tách riêng 3 tầng của Foreground (10 class gộp) và Background slot + GT boxes.
  3. p3_p4_p5_per_class_attn.png : Attention tách riêng P3 | P4 | P5 cho từng class có trong ảnh + GT boxes của đúng class đó.
  4. Bảng thống kê định lượng ra terminal (Contrast Ratio In/Out Box, Mean Attn trong Box, % Mass rơi trúng Box).

Ví dụ chạy:
    python Visdrone/inspect_scales.py --config Visdrone/config.yaml --weights work_dirs/visdrone_counting/best.pth --image "D:/apps/Datasets/VisDrone/dataset/images/0000001_02999_d_0000005.jpg"
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from BaseModel.utils.config import Config
from Visdrone.dataset import VISDRONE_CLASSES
from Visdrone.inference import load_image, load_model
from Visdrone.tmp import derive_label_path, load_boxes_from_txt


def compute_pca_rgb(feat_chw: torch.Tensor) -> np.ndarray:
    """Chiếu feature map (C, H, W) xuống 3 kênh RGB bằng SVD/PCA để quan sát ngữ nghĩa."""
    c, h, w = feat_chw.shape
    flat = feat_chw.detach().cpu().float().permute(1, 2, 0).reshape(-1, c)  # (H*W, C)
    flat = flat - flat.mean(dim=0, keepdim=True)
    # Dùng torch.pca_lowrank nhanh và ổn định cho ma trận (14400, 256)
    _, _, V = torch.pca_lowrank(flat, q=3, center=False)
    proj = flat @ V[:, :3]  # (H*W, 3)
    proj = proj.reshape(h, w, 3).numpy()
    # Chuẩn hóa theo percentile [2%, 98%] để tránh outlier làm tối ảnh
    out = np.zeros_like(proj)
    for ch in range(3):
        p_lo, p_hi = np.percentile(proj[..., ch], [2, 98])
        if p_hi > p_lo:
            out[..., ch] = np.clip((proj[..., ch] - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        else:
            out[..., ch] = 0.5
    return out


def build_box_mask_at_res(
    boxes_norm: List[Tuple[int, float, float, float, float]],
    H: int,
    W: int,
    class_filter: Optional[int] = None,
) -> np.ndarray:
    """Tạo mask nhị phân (H, W) bool: True bên trong bounding box."""
    mask = np.zeros((H, W), dtype=bool)
    for cls_id, cx, cy, w, h in boxes_norm:
        if class_filter is not None and cls_id != class_filter:
            continue
        x1 = int(np.clip(np.trunc((cx - w / 2.0) * W), 0, W - 1))
        y1 = int(np.clip(np.trunc((cy - h / 2.0) * H), 0, H - 1))
        x2 = max(x1 + 1, int(np.clip(np.ceil((cx + w / 2.0) * W), 0, W)))
        y2 = max(y1 + 1, int(np.clip(np.ceil((cy + h / 2.0) * H), 0, H)))
        mask[y1:y2, x1:x2] = True
    return mask


def load_boxes_normalized(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Đọc file nhãn YOLO <class_id> <cx> <cy> <w> <h> dạng chuẩn hóa [0, 1]."""
    if not label_path or not os.path.isfile(label_path):
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((
                    int(float(parts[0])),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ))
    return boxes


def draw_boxes_on_ax(
    ax: plt.Axes,
    boxes_px: List[Tuple[int, int, int, int, int]],
    class_filter: Optional[int] = None,
    color: str = "#00FF66",
    linewidth: float = 0.9,
) -> None:
    """Vẽ khung bounding box lên matplotlib Axes."""
    from matplotlib.patches import Rectangle

    for cls_id, x1, y1, x2, y2 in boxes_px:
        if class_filter is not None and cls_id != class_filter:
            continue
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def inspect_scales(
    config_path: str,
    image_path: str,
    weights_path: Optional[str] = None,
    label_path: Optional[str] = None,
    out_dir: str = "Visdrone/output_inspect_scales",
    device: str = "cuda",
    selected_classes: Optional[List[int]] = None,
) -> None:
    if not torch.cuda.is_available() and device == "cuda":
        print("[!] CUDA không khả dụng, tự động chuyển về CPU.")
        device = "cpu"
    print(f"[*] Thiết bị suy luận: {device.upper()}")

    os.makedirs(out_dir, exist_ok=True)
    model = load_model(config_path=config_path, weights_path=weights_path, device=device)

    # Đọc target_size từ config
    target_size = (960, 960)
    if os.path.isfile(config_path):
        cfg = Config.fromfile(config_path)
        if hasattr(cfg, "data") and "train" in cfg.data:
            for p in cfg.data.train.get("pipeline", []):
                if p.get("type") == "Resize" and "size" in p:
                    s = p["size"]
                    target_size = (s[0], s[1]) if isinstance(s, (list, tuple)) else (s, s)
                    break

    H_img, W_img = target_size
    resized_img, tensor_bchw = load_image(image_path, target_size=target_size)
    tensor_bchw = tensor_bchw.to(device)

    if label_path is None:
        label_path = derive_label_path(image_path)
    boxes_norm = load_boxes_normalized(label_path)
    boxes_px = load_boxes_from_txt(label_path, img_w=W_img, img_h=H_img)
    print(f"[*] Ảnh: {image_path} ({W_img}x{H_img})")
    print(f"[*] Nhãn: {label_path} -> {len(boxes_px)} GT bounding boxes")

    # Chạy forward lấy riêng feat_p3, feat_p4, feat_p5 (sau PANetNeck) và attn_ori
    with torch.no_grad():
        raw_feats = model.backbone(tensor_bchw)
        feat_p3, feat_p4, feat_p5 = model.neck(raw_feats["p3"], raw_feats["p4"], raw_feats["p5"])
        slots, attn_ori, resolutions = model.extract_slots(tensor_bchw)
        pred_counts = model._predict_counts(slots, attn_ori, resolutions)[0].cpu().numpy()

    (H3, W3), (H4, W4), (H5, W5) = resolutions
    n3, n4 = H3 * W3, H4 * W4
    attn_0 = attn_ori[0]  # (11, 18900)
    attn_scales = {
        "P3 (//8, 120x120)": attn_0[:, :n3].view(11, H3, W3).cpu().numpy(),
        "P4 (//16, 60x60)": attn_0[:, n3 : n3 + n4].view(11, H4, W4).cpu().numpy(),
        "P5 (//32, 30x30)": attn_0[:, n3 + n4 :].view(11, H5, W5).cpu().numpy(),
    }
    feat_scales = [
        ("P3 (//8, 120x120)", feat_p3[0], (H3, W3)),
        ("P4 (//16, 60x60)", feat_p4[0], (H4, W4)),
        ("P5 (//32, 30x30)", feat_p5[0], (H5, W5)),
    ]

    # =========================================================================
    # 1. FEATURE MAPS (L2-Norm Energy + PCA-RGB) VẼ ĐÈ GT BOUNDING BOXES
    # =========================================================================
    print("\n" + "=" * 82)
    print("  1. THỐNG KÊ CƯỜNG ĐỘ ĐẶC TRƯNG (L2-NORM) TRONG BOX VS NGOÀI BOX TẠI P3, P4, P5")
    print("=" * 82)
    print(f"{'Tầng':<20} | {'Mean In-Box':<13} | {'Mean Out-Box':<13} | {'Tỉ lệ In/Out':<13} | {'% Ô là Box':<10}")
    print("-" * 82)

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    for col, (title, feat_chw, (fh, fw)) in enumerate(feat_scales):
        l2_map = torch.linalg.vector_norm(feat_chw.float(), dim=0).cpu().numpy()  # (fh, fw)
        box_mask = build_box_mask_at_res(boxes_norm, fh, fw)

        in_val = float(l2_map[box_mask].mean()) if box_mask.any() else 0.0
        out_val = float(l2_map[~box_mask].mean()) if (~box_mask).any() else 1e-6
        ratio = in_val / max(out_val, 1e-6)
        pct_box = float(box_mask.mean()) * 100.0
        print(f"{title:<20} | {in_val:<13.4f} | {out_val:<13.4f} | {ratio:<13.2f}x | {pct_box:<9.2f}%")

        # Upsample nearest để giữ rõ cấu trúc từng ô lưới của P3/P4/P5
        l2_up = cv2.resize(l2_map, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        pca_rgb = compute_pca_rgb(feat_chw)
        pca_up = cv2.resize(pca_rgb, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        # Hàng 0: L2-Norm + GT Boxes
        ax0 = axes[0, col]
        ax0.imshow(l2_up, cmap="magma")
        draw_boxes_on_ax(ax0, boxes_px, color="#00FF66", linewidth=0.8)
        ax0.set_title(f"{title}\nL2-Norm Energy (In/Out: {ratio:.2f}x)", fontsize=12, fontweight="bold")
        ax0.axis("off")

        # Hàng 1: PCA-RGB + GT Boxes
        ax1 = axes[1, col]
        ax1.imshow(pca_up)
        draw_boxes_on_ax(ax1, boxes_px, color="#FFFFFF", linewidth=0.8)
        ax1.set_title(f"{title}\nPCA-RGB Semantic Projection + GT Boxes", fontsize=12, fontweight="bold")
        ax1.axis("off")

    print("=" * 82)
    plt.tight_layout()
    path_feat = os.path.join(out_dir, "p3_p4_p5_features.png")
    plt.savefig(path_feat, dpi=180, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 2. FOREGROUND VS BACKGROUND ATTENTION TÁCH RIÊNG 3 TẦNG + GT BOXES
    # =========================================================================
    gray_img = np.dot(resized_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    gray_rgb = np.stack([gray_img] * 3, axis=-1)

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    for col, (title, attn_scale) in enumerate(attn_scales.items()):
        fh, fw = attn_scale.shape[1], attn_scale.shape[2]
        box_mask = build_box_mask_at_res(boxes_norm, fh, fw)

        fg_map = attn_scale[:10].sum(axis=0)  # (fh, fw)
        bg_map = attn_scale[10]               # (fh, fw)

        fg_hit = float(fg_map[box_mask].mean()) if box_mask.any() else 0.0
        fg_out = float(fg_map[~box_mask].mean()) if (~box_mask).any() else 0.0
        bg_on_bg = float(bg_map[~box_mask].mean()) if (~box_mask).any() else 0.0
        bg_on_box = float(bg_map[box_mask].mean()) if box_mask.any() else 0.0

        fg_up = cv2.resize(fg_map, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        bg_up = cv2.resize(bg_map, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        ax_fg = axes[0, col]
        ax_fg.imshow(gray_rgb)
        ax_fg.imshow(fg_up, cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
        draw_boxes_on_ax(ax_fg, boxes_px, color="#00FF66", linewidth=0.8)
        ax_fg.set_title(
            f"{title} — FG Attn (10 slots)\nIn-Box: {fg_hit:.3f} | Out-Box: {fg_out:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax_fg.axis("off")

        ax_bg = axes[1, col]
        ax_bg.imshow(gray_rgb)
        ax_bg.imshow(bg_up, cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
        draw_boxes_on_ax(ax_bg, boxes_px, color="#FF3333", linewidth=0.8)
        ax_bg.set_title(
            f"{title} — BG Slot Attn\nOn-BG: {bg_on_bg:.3f} | On-Box: {bg_on_box:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax_bg.axis("off")

    plt.tight_layout()
    path_fg_bg = os.path.join(out_dir, "p3_p4_p5_fg_bg_attn.png")
    plt.savefig(path_fg_bg, dpi=180, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 3. ATTENTION TỪNG CLASS TÁCH RIÊNG P3 | P4 | P5 + GT BOXES CỦA ĐÚNG CLASS ĐÓ
    # =========================================================================
    gt_counts = [sum(1 for b in boxes_norm if b[0] == c) for c in range(10)]
    if selected_classes is not None:
        classes_to_show = [c for c in selected_classes if 0 <= c < 10]
    else:
        classes_to_show = [c for c in range(10) if gt_counts[c] > 0 or pred_counts[c] >= 0.5]
        if not classes_to_show:
            classes_to_show = list(range(10))

    print("\n" + "=" * 105)
    print("  2. ĐỘ BÁM BOUNDING BOX CỦA TỪNG CLASS SLOT TRÊN P3, P4, P5 (Mean Attn In-Box / % Mass In-Box)")
    print("=" * 105)
    print(
        f"{'ID':<3} | {'Class':<16} | {'GT':<4} | {'Pred':<5} | "
        f"{'P3 InBox (Mass%)':<20} | {'P4 InBox (Mass%)':<20} | {'P5 InBox (Mass%)':<20}"
    )
    print("-" * 105)

    n_rows = len(classes_to_show)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.8 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    scale_items = list(attn_scales.items())
    for row_idx, c in enumerate(classes_to_show):
        c_name = VISDRONE_CLASSES[c]
        gt_c = gt_counts[c]
        pred_c = float(pred_counts[c])

        # Cột 0: Ảnh RGB + GT Boxes của riêng class c
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(resized_img)
        draw_boxes_on_ax(ax_orig, boxes_px, class_filter=c, color="#00FFFF", linewidth=1.4)
        ax_orig.set_title(f"Class {c}: {c_name}\nGT: {gt_c} | Pred: {pred_c:.2f}", fontsize=11, fontweight="bold")
        ax_orig.axis("off")

        scale_strs = []
        for s_idx, (s_title, attn_scale) in enumerate(scale_items):
            fh, fw = attn_scale.shape[1], attn_scale.shape[2]
            c_map = attn_scale[c]  # (fh, fw)
            c_mask = build_box_mask_at_res(boxes_norm, fh, fw, class_filter=c)

            if c_mask.any():
                in_mean = float(c_map[c_mask].mean())
                total_mass = float(c_map.sum()) + 1e-8
                mass_in_pct = float(c_map[c_mask].sum()) / total_mass * 100.0
                scale_strs.append(f"{in_mean:.3f} ({mass_in_pct:5.1f}%)")
                subtitle = f"InBox: {in_mean:.3f} | MassIn: {mass_in_pct:.1f}%"
            else:
                scale_strs.append(f"no GT (max={c_map.max():.2f})")
                subtitle = f"No GT | Max: {c_map.max():.3f}"

            c_up = cv2.resize(c_map, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
            ax_s = axes[row_idx, s_idx + 1]
            ax_s.imshow(gray_rgb)
            # Scale màu tương đối theo max của tầng đó (hoặc tối thiểu 0.1 để không phóng đại nhiễu 0)
            vmax = max(float(c_map.max()), 0.1)
            ax_s.imshow(c_up, cmap="jet", alpha=0.6, vmin=0.0, vmax=vmax)
            draw_boxes_on_ax(ax_s, boxes_px, class_filter=c, color="#00FFFF", linewidth=1.2)
            ax_s.set_title(f"{s_title}\n{subtitle}", fontsize=10, fontweight="bold")
            ax_s.axis("off")

        print(
            f"{c:<3} | {c_name:<16} | {gt_c:<4} | {pred_c:<5.2f} | "
            f"{scale_strs[0]:<20} | {scale_strs[1]:<20} | {scale_strs[2]:<20}"
        )

    print("=" * 105)
    plt.tight_layout()
    path_cls = os.path.join(out_dir, "p3_p4_p5_per_class_attn.png")
    plt.savefig(path_cls, dpi=160, bbox_inches="tight")
    plt.close()

    print("\n[+] Đã lưu các ảnh phân tích P3/P4/P5 thành công:")
    print(f"    1. Feature L2-Norm & PCA-RGB : {os.path.abspath(path_feat)}")
    print(f"    2. FG vs BG Attn (3 tầng)    : {os.path.abspath(path_fg_bg)}")
    print(f"    3. Per-Class Attn (3 tầng)   : {os.path.abspath(path_cls)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect P3, P4, P5 features & attention vs GT bounding boxes")
    parser.add_argument("--config", type=str, default="Visdrone/config.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument(
        "--image",
        type=str,
        default=r"D:\apps\Datasets\VisDrone\dataset\images\0000001_02999_d_0000005.jpg",
    )
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="Visdrone/output_inspect_scales")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu", action="store_true", help="Chạy trên CPU")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Chỉ định ID các class cần xem")
    args = parser.parse_args()

    device = "cpu" if args.cpu else args.device
    inspect_scales(
        config_path=args.config,
        image_path=args.image,
        weights_path=args.weights,
        label_path=args.labels,
        out_dir=args.out_dir,
        device=device,
        selected_classes=args.classes,
    )


if __name__ == "__main__":
    main()
