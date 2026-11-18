"""Inference script for VisDrone Class Counting and Attention Slot Visualization."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Ensure root dir is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from BaseModel.core import MODELS
from BaseModel.model import BaseModel, build_model
from BaseModel.utils.config import Config
from Visdrone.model import VisDroneDetector

# Mapping id sang tên label chuẩn như trong ảnh
VISDRONE_CLASSES: Dict[int, str] = {
    0: "pedestrian",
    1: "person riding or sitting",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle without canopy",
    7: "tricycle with canopy",
    8: "bus",
    9: "motorcycle",
}

# Màu sắc đại diện cho từng class (khớp bảng màu trực quan)
CLASS_COLORS: Dict[int, str] = {
    0: "#1A44FF",   # blue
    1: "#00E5FF",   # cyan
    2: "#FFFFFF",   # white
    3: "#00E6A8",   # teal
    4: "#283593",   # dark blue
    5: "#FF4081",   # pink
    6: "#FF3D3D",   # coral / red
    7: "#CCFF00",   # lime
    8: "#00E639",   # green
    9: "#AA00FF",   # purple
}


def load_image(image_path: str, target_size: Tuple[int, int] = (960, 960)) -> Tuple[Image.Image, torch.Tensor]:
    """Đọc ảnh từ đường dẫn, resize về 960x960 và chuyển thành Tensor BCHW [0, 1]."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {path.resolve()}")

    raw_img = Image.open(path).convert("RGB")
    resized_img = raw_img.resize(target_size, Image.Resampling.BILINEAR)

    tensor_chw = TF.to_tensor(resized_img)
    tensor_bchw = tensor_chw.unsqueeze(0)  # (1, 3, H, W)
    return resized_img, tensor_bchw


def load_model(
    config_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    device: str = "cuda",
) -> VisDroneDetector:
    """Khởi tạo mô hình VisDroneDetector từ file config .yaml và nạp trọng số checkpoint."""
    if config_path and os.path.isfile(config_path):
        print(f"[*] Đang nạp cấu hình mô hình từ: {config_path}")
        cfg = Config.fromfile(config_path)
        model = build_model(cfg.model).to(device)
    else:
        print("[i] Sử dụng cấu hình mặc định (VisDroneDetector, dim=32).")
        model = VisDroneDetector(num_classes=10, dim=32).to(device)

    # Nếu người dùng chưa truyền weights nhưng trong config có work_dir, tự tìm best.pth hoặc last.pth
    if not weights_path and config_path and os.path.isfile(config_path):
        cfg = Config.fromfile(config_path)
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
        # Loại bỏ tiền tố module. nếu lưu từ DDP / DataParallel
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
    total_round = 0
    for c_id in range(10):
        c_name = VISDRONE_CLASSES.get(c_id, f"class_{c_id}")
        raw_val = float(counts_raw[c_id])
        round_val = int(round(raw_val))
        total_raw += raw_val
        total_round += round_val
        print(f"{c_id:<4} | {c_name:<28} | {raw_val:<15.2f} | {round_val:<10}")

    print("-" * 70)
    print(f"{'TỔNG':<4} | {'Tất cả các đối tượng':<28} | {total_raw:<15.2f} | {total_round:<10}")
    print("=" * 70 + "\n")


def visualize_attention_maps(
    original_img: Image.Image,
    class_attn_maps: torch.Tensor,
    counts_raw: np.ndarray,
    out_path: str,
    overlay: bool = True,
):
    """Xuất ảnh hiện attention map cho 10 class slots (KHÔNG hiện background slot).

    Hiển thị dạng lưới 3x4:
      - Ô 1: Ảnh gốc
      - Ô 2-11: 10 class slots (heatmaps overlay hoặc attention maps)
      - Ô 12: Biểu đồ cột tổng hợp số lượng đếm được theo từng class.
    """
    # Ảnh gốc RGB giữ nguyên màu sắc
    img_rgb = np.array(original_img)  # (H, W, 3)
    H, W, _ = img_rgb.shape

    # Ảnh thang xám trắng đen dùng làm nền cho các ô overlay heatmap
    gray_pil = original_img.convert("L")
    img_gray = np.array(gray_pil)  # (H, W)

    # class_attn_maps shape: (10, H_feat, W_feat)
    attn_cpu = class_attn_maps.detach().cpu().float().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.patch.set_facecolor("#121212")
    axes = axes.flatten()

    # Ô 0: Ảnh gốc màu sắc RGB nguyên bản
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image (RGB 960x960)", color="white", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")
    # Viền xám cho ô ảnh gốc
    rect = patches.Rectangle((0, 0), W - 1, H - 1, linewidth=2, edgecolor="#888888", facecolor="none")
    axes[0].add_patch(rect)

    # Ô 1 đến 10: 10 Class Slots Attention Map
    for c_id in range(10):
        ax = axes[c_id + 1]
        c_name = VISDRONE_CLASSES[c_id]
        c_color = CLASS_COLORS[c_id]
        raw_val = float(counts_raw[c_id])
        round_val = int(round(raw_val))

        attn_slot = attn_cpu[c_id]  # (H_feat, W_feat)
        # Chuẩn hóa min-max attention để hiển thị tương phản rõ nét
        attn_min, attn_max = attn_slot.min(), attn_slot.max()
        if attn_max > attn_min:
            attn_norm = (attn_slot - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = attn_slot

        # Hiển thị ảnh nền trắng đen rồi overlay heatmap lên trên để màu nổi bật nhất
        if overlay:
            ax.imshow(img_gray, cmap="gray")
            im = ax.imshow(attn_norm, cmap="jet", alpha=0.55, extent=[0, W, H, 0])
        else:
            im = ax.imshow(attn_norm, cmap="inferno")

        # Tiêu đề mang màu sắc nhận diện của class
        title = f"[{c_id}] {c_name}\nCount: {raw_val:.2f} (~{round_val})"
        ax.set_title(title, color=c_color, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

        # Viền bao quanh ô theo màu sắc đặc trưng của class
        rect = patches.Rectangle((0, 0), W - 1, H - 1, linewidth=2.5, edgecolor=c_color, facecolor="none")
        ax.add_patch(rect)

    # Ô 11: Biểu đồ cột tổng hợp số lượng (Count Bar Chart)
    ax_bar = axes[11]
    ax_bar.set_facecolor("#1e1e1e")
    class_ids = list(range(10))
    counts = [float(counts_raw[c]) for c in class_ids]
    colors = [CLASS_COLORS[c] for c in class_ids]

    bars = ax_bar.barh(class_ids, counts, color=colors, edgecolor="white", height=0.7)
    ax_bar.set_yticks(class_ids)
    ax_bar.set_yticklabels([f"[{c}] {VISDRONE_CLASSES[c]}" for c in class_ids], color="white", fontsize=9)
    ax_bar.set_xlabel("Predicted Count", color="white", fontsize=11)
    ax_bar.set_title("Class Counts Summary", color="white", fontsize=13, fontweight="bold", pad=8)
    ax_bar.invert_yaxis()  # Lớp 0 trên cùng
    ax_bar.tick_params(colors="white")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.3, color="white")

    # In giá trị số lượng lên đầu mỗi cột
    for bar in bars:
        w = bar.get_width()
        ax_bar.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}",
                    va="center", ha="left", color="white", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Đã lưu ảnh attention map thành công tại:\n    {os.path.abspath(out_path)}")


def visualize_all_slots_combined(
    original_img: Image.Image,
    class_attn_maps: torch.Tensor,
    counts_raw: np.ndarray,
    out_path: str = "D:\\Study\\Projects\\Computer_Vision\\Visdrone\\all_attn.png",
    alpha: float = 0.9,
):
    """Vẽ attention của toàn bộ 10 slot lên cùng một ảnh, mỗi slot một màu đặc trưng.

    Xuất file: all_attn.png
    """
    import matplotlib.colors as mcolors

    # Chuyển ảnh gốc sang ảnh trắng đen (Grayscale 3 kênh RGB đồng nhất)
    gray_pil = original_img.convert("L")
    gray_np = np.array(gray_pil).astype(np.float32) / 255.0  # (H, W) in [0, 1]
    H, W = gray_np.shape
    img_gray_3ch = np.stack([gray_np, gray_np, gray_np], axis=-1)  # (H, W, 3)

    # class_attn_maps: (10, H_feat, W_feat)
    attn_cpu = class_attn_maps.detach().cpu().float().numpy()

    # Chuyển bảng màu HEX sang RGB [0, 1]
    slot_colors = np.array([mcolors.to_rgb(CLASS_COLORS[i]) for i in range(10)])  # (10, 3)

    # 1. Tính toán Weighted Color Composite: Mỗi pixel là tổ hợp màu của các slot theo trọng số attention
    attn_weights = np.maximum(attn_cpu, 0.0)
    total_weight = attn_weights.sum(axis=0, keepdims=True) + 1e-6  # (1, H, W)
    norm_weights = attn_weights / total_weight  # (10, H, W)

    # Tổng hợp màu RGB theo trọng số: (H, W, 3)
    color_composite = np.einsum("chw,cd->hwd", norm_weights, slot_colors)

    # Trọng số hiển thị (alpha map): dựa trên độ nổi bật của attention
    max_attn = attn_weights.max(axis=0)  # (H, W)
    if max_attn.max() > max_attn.min():
        salience = (max_attn - max_attn.min()) / (max_attn.max() - max_attn.min() + 1e-6)
    else:
        salience = max_attn
    effective_alpha = np.clip(salience * alpha + 0.15, 0.0, 0.85)[..., None]

    # Hòa trộn (alpha blending) với nền ảnh trắng đen để màu của từng slot nổi bật tối đa
    blended = (1.0 - effective_alpha) * img_gray_3ch + effective_alpha * color_composite
    blended = np.clip(blended, 0.0, 1.0)

    # 2. Tạo hình vẽ Widescreen có Legend chi tiết khớp chính xác như ảnh mẫu
    fig, (ax_img, ax_legend) = plt.subplots(
        1, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [3.2, 1.2]}
    )
    fig.patch.set_facecolor("#121212")

    # Panel chính: Ảnh vẽ toàn bộ Slot Attention với các màu tương ứng
    ax_img.imshow(blended)
    ax_img.set_title(
        "All Slots Attention Map Overlay (Each Slot Has A Unique Color)",
        color="white",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax_img.axis("off")
    rect = patches.Rectangle((0, 0), W - 1, H - 1, linewidth=2, edgecolor="#555555", facecolor="none")
    ax_img.add_patch(rect)

    # Panel phụ: Legend chi tiết với chấm tròn màu, ID và tên class
    ax_legend.set_facecolor("#181818")
    ax_legend.axis("off")
    ax_legend.set_xlim(0, 10)
    ax_legend.set_ylim(-0.5, 10.5)

    ax_legend.text(
        0.5, 10.1, "SLOT COLOR LEGEND", color="white", fontsize=14, fontweight="bold", va="center"
    )
    ax_legend.axhline(9.5, color="#333333", linewidth=1.5)

    for c_id in range(10):
        y_pos = 9.0 - c_id * 0.95
        c_name = VISDRONE_CLASSES[c_id]
        c_color = CLASS_COLORS[c_id]
        raw_val = float(counts_raw[c_id])
        round_val = int(round(raw_val))

        # Chấm tròn màu (Color Dot)
        circle = patches.Circle((0.8, y_pos), radius=0.25, facecolor=c_color, edgecolor="white", linewidth=0.8)
        ax_legend.add_patch(circle)

        # ID
        ax_legend.text(1.5, y_pos, f"{c_id}", color="#CCCCCC", fontsize=12, fontweight="bold", va="center")

        # Tên Class
        ax_legend.text(2.2, y_pos, c_name, color="white", fontsize=11, fontweight="bold", va="center")

        # Số lượng đếm
        ax_legend.text(9.5, y_pos, f"{raw_val:.2f} (~{round_val})", color=c_color, fontsize=10, ha="right", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Đã lưu ảnh toàn bộ slot attention thành công tại:\n    {os.path.abspath(out_path)}")


def run_inference(
    image_path: str,
    config_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    out_path: str = "D:\\Study\\Projects\\Computer_Vision\\Visdrone\\attn_map_result.png",
    out_all_path: str = "D:\\Study\\Projects\\Computer_Vision\\Visdrone\\all_attn.png",
    device: Optional[str] = None,
    overlay: bool = True,
):
    """Hàm chạy toàn bộ quy trình suy luận từ ảnh -> dự đoán số lượng -> xuất ảnh attention map."""
    # Lấy device từ config nếu chưa truyền
    if device is None and config_path and os.path.isfile(config_path):
        cfg = Config.fromfile(config_path)
        cfg_dev = getattr(cfg, "device", None)
        if cfg_dev:
            device = cfg_dev if (cfg_dev == "cpu" or torch.cuda.is_available()) else "cpu"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] Thiết bị sử dụng: {device}")
    model = load_model(config_path=config_path, weights_path=weights_path, device=device)

    # Lấy kích thước ảnh từ pipeline trong config nếu có
    target_size = (960, 960)
    if config_path and os.path.isfile(config_path):
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
        pred_counts = model._predict_counts(slots, attn_ori)[0]  # (10,)
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
    visualize_all_slots_combined(
        original_img=resized_img,
        class_attn_maps=class_attn_maps,
        counts_raw=counts_raw,
        out_path=out_all_path,
    )


def main():
    parser = argparse.ArgumentParser(description="VisDrone Class Counting & Slot Attention Map Inference")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Đường dẫn đến file cấu hình (.yaml) của mô hình",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=r"D:\apps\Datasets\VisDrone\dataset\images\0000001_02999_d_0000005.jpg",
        help="Đường dẫn đến file ảnh cần suy luận",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Đường dẫn đến file checkpoint (.pth) đã huấn luyện (tùy chọn)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=r"D:\Study\Projects\Computer_Vision\Visdrone\attn_map_result.png",
        help="Đường dẫn file ảnh xuất ra chứa attention maps của từng slot",
    )
    parser.add_argument(
        "--out-all",
        type=str,
        default=r"D:\Study\Projects\Computer_Vision\Visdrone\all_attn.png",
        help="Đường dẫn file ảnh xuất ra vẽ toàn bộ slot lên cùng 1 ảnh (all_attn.png)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị chạy (cuda hoặc cpu)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Không overlay heatmap lên ảnh gốc mà hiển thị heatmap thuần",
    )

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        config_path=args.config,
        weights_path=args.weights,
        out_path=args.out,
        out_all_path=args.out_all,
        device=args.device,
        overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()
