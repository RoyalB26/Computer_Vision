"""Inference script for VisDrone Object Counting using MobileNetV3 + Slot Attention.

Outputs:
  - Terminal table with predicted object counts for all 10 VisDrone classes.
  - attn_map_result.png: 3x4 grid showing Original RGB image in panel 0 and 10 class heatmaps on panels 1-10.
  - all_attn.png: Composite image rendering all 10 class slots with distinct colors and legend.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from BaseModel.utils.config import Config
from Visdrone.dataset import VISDRONE_CLASSES
from Visdrone.model import VisDroneDetector

# Bảng màu 10 class tương ứng cho all_attn.png (RGB trong thang [0, 1])
CLASS_COLORS_RGB = [
    (1.0, 0.2, 0.2),    # 0: pedestrian - Đỏ tươi
    (1.0, 0.6, 0.0),    # 1: people - Cam
    (0.9, 0.9, 0.1),    # 2: bicycle - Vàng
    (0.1, 0.8, 0.2),    # 3: car - Lục tươi
    (0.1, 0.9, 0.8),    # 4: van - Cyan
    (0.2, 0.4, 1.0),    # 5: truck - Lam
    (0.6, 0.2, 0.9),    # 6: tricycle - Tím
    (0.9, 0.2, 0.7),    # 7: awning-tricycle - Hồng cánh sen
    (0.6, 0.3, 0.1),    # 8: bus - Nâu
    (0.8, 0.8, 0.8),    # 9: motorcycle - Xám sáng
]


def load_image(image_path: str, target_size: Tuple[int, int] = (960, 960)) -> Tuple[np.ndarray, torch.Tensor]:
    """Đọc ảnh từ đường dẫn, resize về target_size và chuyển thành tensor BCHW chuẩn hóa."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    resized_pil = pil_img.resize(target_size, Image.BILINEAR)
    resized_np = np.array(resized_pil, dtype=np.uint8)  # (H, W, 3)

    # Tensor BCHW normalized [0, 1]
    tensor = torch.from_numpy(resized_np).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # (1, 3, H, W)
    return resized_np, tensor


def load_model(
    config_path: str,
    weights_path: Optional[str] = None,
    device: str = "cuda",
) -> VisDroneDetector:
    """Khởi tạo mô hình VisDroneDetector từ config và nạp trọng số đã huấn luyện."""
    print(f"[*] Đang nạp cấu hình mô hình từ: {config_path}")
    cfg = Config.fromfile(config_path)

    # Khởi tạo mô hình
    model_cfg = cfg.model.copy()
    model_cfg.pop("type", None)
    model = VisDroneDetector(**model_cfg)
    model.to(device)

    # Tự động tìm checkpoint nếu weights_path không được chỉ định
    if weights_path is None:
        work_dir = getattr(cfg, "work_dir", "./work_dirs/visdrone_counting")
        candidate_best = os.path.join(work_dir, "best.pth")
        candidate_last = os.path.join(work_dir, "last.pth")
        if os.path.isfile(candidate_best):
            weights_path = candidate_best
            print(f"[i] Tự động tìm thấy checkpoint: {weights_path}")
        elif os.path.isfile(candidate_last):
            weights_path = candidate_last
            print(f"[i] Tự động tìm thấy checkpoint: {weights_path}")

    if weights_path and os.path.isfile(weights_path):
        print(f"[*] Đang nạp checkpoint từ: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Lọc bỏ tiền tố module. nếu lưu từ DDP / DataParallel
        model_dict = model.state_dict()
        filtered_state_dict = {}
        mismatched_keys = []
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            if clean_k in model_dict:
                if model_dict[clean_k].shape == v.shape:
                    filtered_state_dict[clean_k] = v
                else:
                    mismatched_keys.append((clean_k, tuple(v.shape), tuple(model_dict[clean_k].shape)))
            else:
                filtered_state_dict[clean_k] = v

        if mismatched_keys:
            print(f"[!] Bỏ qua {len(mismatched_keys)} layer do lệch kích thước tensor (do thay đổi kiến trúc/mass/pos):")
            for mk, ckpt_sh, mod_sh in mismatched_keys:
                print(f"    - {mk}: checkpoint {ckpt_sh} != model {mod_sh}")

        model.load_state_dict(filtered_state_dict, strict=False)
        print("[+] Nạp trọng số thành công!")
    else:
        if weights_path:
            print(f"[!] Không tìm thấy file weights: {weights_path}.")
        print("[i] Khởi tạo mô hình với trọng số ban đầu.")

    model.eval()
    return model


def print_prediction_table(image_path: str, counts_raw: np.ndarray):
    """In bảng dự đoán số lượng từng class kèm tên label chuẩn ra console."""
    print("\n" + "=" * 70)
    print("      KẾT QUẢ DỰ ĐOÁN ĐẾM SỐ LƯỢNG CLASS VISDRONE (SLOT ATTENTION)")
    print("=" * 70)
    print(f"Ảnh đầu vào: {image_path}")
    print("-" * 70)
    print(f"{'ID':<4} | {'Tên Class':<28} | {'Dự đoán (Raw)':<15} | {'Làm tròn':<10}")
    print("-" * 70)

    total_raw = 0.0
    total_rounded = 0
    for idx, name in enumerate(VISDRONE_CLASSES):
        raw = float(counts_raw[idx])
        rounded = max(0, int(round(raw)))
        total_raw += raw
        total_rounded += rounded
        print(f"{idx:<4} | {name:<28} | {raw:<15.2f} | {rounded:<10}")

    print("-" * 70)
    print(f"{'TỔNG':<4} | {'Tất cả các đối tượng':<28} | {total_raw:<15.2f} | {total_rounded:<10}")
    print("=" * 70 + "\n")


def visualize_attention_maps(
    original_img: np.ndarray,
    class_attn_maps: torch.Tensor,
    counts_raw: np.ndarray,
    out_path: str = "attn_map_result.png",
    overlay: bool = True,
):
    """Xuất hình ảnh lưới 3x4 thể hiện kết quả suy luận:

    - Ô đầu tiên (vị trí 0, 0): Ảnh gốc dạng RGB đầy đủ màu sắc.
    - 10 ô tiếp theo (vị trí 1 đến 10): 10 class slots VisDrone với heatmap chồng lên ảnh grayscale.
    - Ô thứ 12 (vị trí 2, 3): Bảng tổng hợp số lượng.
    """
    H, W, _ = original_img.shape
    # Tạo ảnh grayscale cho nền của các ô heatmap
    gray_img = np.dot(original_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    gray_rgb = np.stack([gray_img] * 3, axis=-1)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    # Panel 0: Ảnh gốc RGB
    axes[0].imshow(original_img)
    axes[0].set_title("Input Image (RGB)", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")

    attn_np = class_attn_maps.detach().cpu().numpy()  # (10, H, W)

    # Panels 1-10: 10 Class Slot Attention Maps
    for idx in range(10):
        ax = axes[idx + 1]
        c_name = VISDRONE_CLASSES[idx]
        raw_val = float(counts_raw[idx])
        round_val = max(0, int(round(raw_val)))

        heatmap = attn_np[idx]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            norm_heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            norm_heatmap = np.zeros_like(heatmap)

        if overlay:
            ax.imshow(gray_rgb)
            im = ax.imshow(norm_heatmap, cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(norm_heatmap, cmap="inferno", vmin=0.0, vmax=1.0)

        title_str = f"Slot {idx}: {c_name}\nCount: {round_val} (raw: {raw_val:.2f})"
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

    # Panel 11: Summary statistics
    ax_sum = axes[11]
    ax_sum.axis("off")
    summary_lines = ["COUNT SUMMARY:"]
    total_cnt = 0
    for idx, name in enumerate(VISDRONE_CLASSES):
        r_val = max(0, int(round(float(counts_raw[idx]))))
        total_cnt += r_val
        summary_lines.append(f"• {name:<16}: {r_val}")
    summary_lines.append(f"\n=> TOTAL: {total_cnt} objects")

    ax_sum.text(
        0.05, 0.95,
        "\n".join(summary_lines),
        transform=ax_sum.transAxes,
        fontsize=10.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0", edgecolor="#aaaaaa"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu ảnh attention map thành công tại:\n    {os.path.abspath(out_path)}")


def visualize_all_slots_overlay(
    original_img: np.ndarray,
    class_attn_maps: torch.Tensor,
    counts_raw: np.ndarray,
    out_path: str = "all_attn.png",
    threshold: float = 0.25,
):
    """Vẽ toàn bộ 10 class slots lên cùng 1 bức ảnh, mỗi slot một màu kèm Legend chuẩn."""
    H, W, _ = original_img.shape
    gray_img = np.dot(original_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    gray_rgb = np.stack([gray_img] * 3, axis=-1).astype(np.float32) / 255.0

    attn_np = class_attn_maps.detach().cpu().numpy()  # (10, H, W)
    composite = gray_rgb.copy()

    # Chuẩn hóa từng slot và cộng màu phủ tương ứng
    legend_patches = []
    for idx in range(10):
        color = CLASS_COLORS_RGB[idx]
        c_name = VISDRONE_CLASSES[idx]
        round_cnt = max(0, int(round(float(counts_raw[idx]))))

        h = attn_np[idx]
        h_min, h_max = h.min(), h.max()
        norm_h = (h - h_min) / (h_max - h_min + 1e-8)

        # Mặt nạ những điểm có độ chú ý vượt ngưỡng threshold
        mask = (norm_h > threshold).astype(np.float32)
        weight = (norm_h * mask)[..., None]  # (H, W, 1)

        color_layer = np.zeros((H, W, 3), dtype=np.float32)
        for ch in range(3):
            color_layer[..., ch] = color[ch]

        alpha = 0.55
        composite = composite * (1.0 - weight * alpha) + color_layer * (weight * alpha)

        patch = mpatches.Patch(
            color=color,
            label=f"{idx}: {c_name} ({round_cnt})"
        )
        legend_patches.append(patch)

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
    print(f"[+] Đã lưu ảnh toàn bộ slot attention thành công tại:\n    {os.path.abspath(out_path)}")


def run_inference(
    config_path: str,
    image_path: str,
    weights_path: Optional[str] = None,
    out_path: str = "attn_map_result.png",
    out_all_path: str = "all_attn.png",
    device: str = "cuda",
    overlay: bool = True,
):
    """Quy trình Inference hoàn chỉnh: Tải ảnh, chạy qua mạng, in kết quả và lưu ảnh."""
    if not torch.cuda.is_available() and device == "cuda":
        print("[!] CUDA không khả dụng, tự động chuyển về cpu.")
        device = "cpu"
    print(f"[*] Thiết bị suy luận: {device.upper()}")

    model = load_model(config_path=config_path, weights_path=weights_path, device=device)

    # Đọc target resolution từ config nếu có
    target_size = (960, 960)
    if os.path.isfile(config_path):
        cfg = Config.fromfile(config_path)
        if hasattr(cfg, "data") and "train" in cfg.data:
            pipeline = cfg.data.train.get("pipeline", [])
            for p in pipeline:
                if p.get("type") == "Resize" and "size" in p:
                    s = p["size"]
                    target_size = (s[0], s[1]) if isinstance(s, (list, tuple)) else (s, s)
                    break

    # 1. Đọc và tiền xử lý ảnh
    print(f"[*] Đang đọc ảnh: {image_path} (Target size: {target_size})")
    resized_img, tensor_bchw = load_image(image_path, target_size=target_size)
    tensor_bchw = tensor_bchw.to(device)

    # 2. Suy luận qua Slot Attention
    with torch.no_grad():
        slots, attn_ori, resolutions = model.extract_slots(tensor_bchw)
        pred_counts = model._predict_counts(slots, attn_ori, resolutions)[0]  # (10,)
        counts_raw = pred_counts.cpu().numpy()

        # Tái cấu trúc spatial attention maps đa quy mô gộp từ //8, //16, //32 về target_size
        attn_spatial = model.get_spatial_attention_maps(attn_ori, resolutions, target_size=target_size)[0]
        # Chỉ lấy 10 class slots (bỏ qua slot 11 là background slot)
        class_attn_maps = attn_spatial[:10]  # (10, H, W)

    # 3. In kết quả dạng bảng ra màn hình
    print_prediction_table(image_path, counts_raw)

    # 4. Xuất ảnh hiển thị Attention Maps từng class slot riêng biệt (3x4 grid)
    visualize_attention_maps(
        original_img=resized_img,
        class_attn_maps=class_attn_maps,
        counts_raw=counts_raw,
        out_path=out_path,
        overlay=overlay,
    )

    # 5. Xuất ảnh vẽ toàn bộ slot lên cùng một ảnh, mỗi slot một màu (all_attn.png)
    visualize_all_slots_overlay(
        original_img=resized_img,
        class_attn_maps=class_attn_maps,
        counts_raw=counts_raw,
        out_path=out_all_path,
    )


def main():
    parser = argparse.ArgumentParser(description="VisDrone Class Counting & Attention Map Inference")
    parser.add_argument("--config", type=str, default="Visdrone/config.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh đầu vào (mặc định lấy ảnh mẫu)")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn checkpoint .pth (tự động tìm nếu None)")
    parser.add_argument("--out", type=str, default="Visdrone/attn_map_result.png", help="Đường dẫn lưu ảnh 3x4 attention map")
    parser.add_argument("--out-all", type=str, default="Visdrone/all_attn.png", help="Đường dẫn lưu ảnh gộp tất cả slot đa màu")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị: 'cuda' hoặc 'cpu'")
    parser.add_argument("--cpu", action="store_true", help="Chạy suy luận hoàn toàn trên CPU (tương đương --device cpu)")
    parser.add_argument("--no-overlay", action="store_true", help="Không phủ attention map lên ảnh gốc")
    args = parser.parse_args()

    device = "cpu" if args.cpu else args.device

    # Tìm ảnh mặc định nếu người dùng không truyền --image
    image_path = args.image
    if image_path is None:
        candidate_images = [
            r"D:\apps\Datasets\VisDrone\dataset\images\0000001_02999_d_0000005.jpg",
            r"D:\apps\Datasets\VisDrone\dataset\images\0000002_00050_d_0000001.jpg",
            r"D:\apps\Datasets\VisDrone\dataset\images\0000002_00450_d_0000002.jpg",
        ]
        for c_img in candidate_images:
            if os.path.isfile(c_img):
                image_path = c_img
                break

    if image_path is None or not os.path.isfile(image_path):
        print("[!] Không tìm thấy ảnh mẫu trên máy. Đang tạo ảnh tổng hợp giả lập để kiểm thử...")
        image_path = "Visdrone/dummy_sample.jpg"
        dummy_arr = np.random.randint(50, 200, size=(960, 960, 3), dtype=np.uint8)
        Image.fromarray(dummy_arr).save(image_path)

    run_inference(
        config_path=args.config,
        image_path=image_path,
        weights_path=args.weights,
        out_path=args.out,
        out_all_path=args.out_all,
        device=device,
        overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()
