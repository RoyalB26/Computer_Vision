"""Tạo Background Segmentation Mask (bg_seg) từ Bounding Box của VisDrone.

Sử dụng:
    python Visdrone/tmp.py
    python Visdrone/tmp.py "D:/apps/Datasets/VisDrone/dataset/images/0000001_02999_d_0000005.jpg"
    python Visdrone/tmp.py --image "D:/apps/Datasets/VisDrone/dataset/images/0000001_02999_d_0000005.jpg" --target-size 960 960
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def derive_label_path(image_path: str) -> str:
    """Tự động suy ra đường dẫn file nhãn .txt từ đường dẫn ảnh."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.dirname(os.path.abspath(image_path))

    # Trường hợp 1: Thay thư mục 'images' thành 'labels'
    parent_dir = os.path.dirname(img_dir)
    candidate_label = os.path.join(parent_dir, "labels", f"{base_name}.txt")
    if os.path.isfile(candidate_label):
        return candidate_label

    # Trường hợp 2: File .txt nằm cùng thư mục ảnh
    candidate_same = os.path.join(img_dir, f"{base_name}.txt")
    if os.path.isfile(candidate_same):
        return candidate_same

    # Trường hợp 3: Thay chuỗi 'images' trực tiếp trong đường dẫn
    candidate_replace = image_path.replace("images", "labels")
    candidate_replace = os.path.splitext(candidate_replace)[0] + ".txt"
    if os.path.isfile(candidate_replace):
        return candidate_replace

    return candidate_label


def load_boxes_from_txt(label_path: str, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, int]]:
    """Đọc bounding boxes từ file nhãn VisDrone.
    
    Hỗ trợ định dạng YOLO: <class_id> <cx> <cy> <w> <h> (chuẩn hóa [0, 1]).
    Trả về danh sách: [(class_id, x1, y1, x2, y2), ...]
    """
    if not os.path.isfile(label_path):
        print(f"[!] Cảnh báo: Không tìm thấy file nhãn tại {label_path}")
        return []

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Chuyển đổi từ normalized (cx, cy, w, h) sang pixel (x1, y1, x2, y2)
            x1 = int(round((cx - w / 2.0) * img_w))
            y1 = int(round((cy - h / 2.0) * img_h))
            x2 = int(round((cx + w / 2.0) * img_w))
            y2 = int(round((cy + h / 2.0) * img_h))

            # Giới hạn tọa độ trong biên ảnh
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))

            if x2 > x1 and y2 > y1:
                boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def create_bg_segmentation_mask(
    img_h: int,
    img_w: int,
    boxes: List[Tuple[int, int, int, int, int]],
) -> np.ndarray:
    """Tạo mask nền (bg_mask):
    - Giá trị 255 (trắng): Background (Nền).
    - Giá trị 0 (đen): Foreground (Bên trong bounding box của vật thể).
    """
    # Khởi tạo toàn bộ ảnh là Background (255)
    bg_mask = np.ones((img_h, img_w), dtype=np.uint8) * 255

    # Gán vùng chứa bounding box thành 0 (Foreground)
    for _, x1, y1, x2, y2 in boxes:
        bg_mask[y1:y2, x1:x2] = 0

    return bg_mask


def process_image_bg_seg(
    image_path: str,
    label_path: str | None = None,
    target_size: Tuple[int, int] | None = None,
    out_dir: str = "Visdrone/output_bg_seg",
):
    """Quy trình tạo mask nền và áp dụng lên ảnh RGB."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

    if label_path is None:
        label_path = derive_label_path(image_path)

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Đọc ảnh gốc bằng OpenCV (BGR -> RGB)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Không thể đọc file ảnh bằng OpenCV: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Resize nếu có yêu cầu target_size (ví dụ: 960x960)
    if target_size is not None:
        img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        H, W = target_size[1], target_size[0]
        print(f"[*] Đã resize ảnh từ ({orig_w}x{orig_h}) sang ({W}x{H})")
    else:
        H, W = orig_h, orig_w

    print(f"[*] Ảnh đầu vào: {image_path} (Kích thước: {W}x{H})")
    print(f"[*] File nhãn:   {label_path}")

    # 2. Đọc bounding boxes
    boxes = load_boxes_from_txt(label_path, img_w=W, img_h=H)
    print(f"[+] Tìm thấy tổng cộng {len(boxes)} bounding boxes vật thể.")

    # 3. Tạo Mask Background (Trắng 255 = Nền, Đen 0 = Vật thể)
    bg_mask = create_bg_segmentation_mask(img_h=H, img_w=W, boxes=boxes)

    # 4. Áp dụng Mask lên ảnh RGB: Giữ lại nền, che đen các vật thể
    bg_applied_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=bg_mask)

    # 5. Mask Foreground (Ngược lại: Trắng 255 = Vật thể, Đen 0 = Nền)
    fg_mask = 255 - bg_mask
    fg_applied_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=fg_mask)

    # Thống kê tỷ lệ diện tích
    total_pixels = H * W
    bg_pixels = int(np.sum(bg_mask == 255))
    fg_pixels = total_pixels - bg_pixels
    print(f"[*] Thống kê phân vùng:")
    print(f"    - Diện tích Nền (Background):  {bg_pixels:,} px ({bg_pixels / total_pixels * 100:.2f}%)")
    print(f"    - Diện tích Vật thể (Objects): {fg_pixels:,} px ({fg_pixels / total_pixels * 100:.2f}%)")

    # 6. Lưu các file ảnh kết quả
    path_bg_mask = os.path.join(out_dir, f"{base_name}_bg_mask.png")
    path_bg_rgb = os.path.join(out_dir, f"{base_name}_bg_applied_rgb.png")
    path_fg_rgb = os.path.join(out_dir, f"{base_name}_fg_applied_rgb.png")
    path_preview = os.path.join(out_dir, f"{base_name}_preview_comparison.png")

    # Lưu mask nhị phân
    cv2.imwrite(path_bg_mask, bg_mask)

    # Lưu ảnh RGB áp dụng mask (BGR khi ghi bằng OpenCV)
    cv2.imwrite(path_bg_rgb, cv2.cvtColor(bg_applied_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(path_fg_rgb, cv2.cvtColor(fg_applied_rgb, cv2.COLOR_RGB2BGR))

    # 7. Xuất hình ảnh so sánh tổng quan trực quan (2x2 grid)
    # Vẽ bounding box lên bản sao ảnh gốc để kiểm chứng
    img_with_boxes = img_rgb.copy()
    for _, x1, y1, x2, y2 in boxes:
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 50, 50), 2)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    axes[0, 0].imshow(img_with_boxes)
    axes[0, 0].set_title(f"1. Ảnh gốc + Bounding Boxes ({len(boxes)} boxes)", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(bg_mask, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"2. Mask Nền (Trắng: Nền {bg_pixels/total_pixels*100:.1f}%, Đen: Box)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(bg_applied_rgb)
    axes[1, 0].set_title("3. Ảnh RGB đã Apply Mask Nền (Che vật thể)", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(fg_applied_rgb)
    axes[1, 1].set_title("4. Ảnh RGB bóc tách Vật thể (Che nền)", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(path_preview, dpi=180, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 70)
    print("      KẾT QUẢ ĐÃ ĐƯỢC LƯU THÀNH CÔNG TẠI THƯ MỤC:")
    print("=" * 70)
    print(f"1. Mask Nền nhị phân:        {os.path.abspath(path_bg_mask)}")
    print(f"2. Ảnh RGB apply Mask Nền:   {os.path.abspath(path_bg_rgb)}")
    print(f"3. Ảnh RGB bóc tách Vật thể: {os.path.abspath(path_fg_rgb)}")
    print(f"4. Ảnh tổng hợp 4 Panel:     {os.path.abspath(path_preview)}")
    print("=" * 70 + "\n")

    return {
        "bg_mask_path": path_bg_mask,
        "bg_applied_rgb_path": path_bg_rgb,
        "fg_applied_rgb_path": path_fg_rgb,
        "preview_path": path_preview,
    }


def main():
    parser = argparse.ArgumentParser(description="Tạo Background Segmentation Mask theo Bounding Box")
    parser.add_argument(
        "image_pos",
        nargs="?",
        default=None,
        help="Đường dẫn file ảnh (tham số trực tiếp không cần --image)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Đường dẫn file ảnh đầu vào",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Đường dẫn file nhãn .txt (tự động suy ra nếu không truyền)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        help="Kích thước resize (ví dụ: --target-size 960 960). Mặc định giữ nguyên kích thước gốc.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Visdrone/output_bg_seg",
        help="Thư mục lưu ảnh đầu ra",
    )
    args = parser.parse_args()

    # Ưu tiên lấy đường dẫn ảnh từ tham số trực tiếp hoặc qua --image
    image_path = args.image or args.image_pos
    if image_path is None:
        # Đường dẫn mặc định theo ví dụ của bạn
        default_img = r"D:\apps\Datasets\VisDrone\dataset\images\0000001_02999_d_0000005.jpg"
        if os.path.isfile(default_img):
            image_path = default_img
        else:
            print("[!] Vui lòng cung cấp đường dẫn ảnh. Ví dụ:")
            print('    python Visdrone/tmp.py "D:/apps/Datasets/VisDrone/dataset/images/0000001_02999_d_0000005.jpg"')
            sys.exit(1)

    target_size = tuple(args.target_size) if args.target_size else None

    process_image_bg_seg(
        image_path=image_path,
        label_path=args.labels,
        target_size=target_size,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
