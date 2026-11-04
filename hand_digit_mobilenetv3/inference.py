"""
Inference script for Hand Digit Detection with HandDigitMobileNetV3Detector (DeformableHead + PANet).

Usage:
    python hand_digit_mobilenetv3/inference.py --image path/to/image.jpg
    python hand_digit_mobilenetv3/inference.py --image path/to/folder_of_images/ --conf 0.25
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Thêm root workspace vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
import yaml

from hand_digit_mobilenetv3.model import HandDigitMobileNetV3Detector


# Bảng màu trực quan cho 10 class chữ số (0 -> 9)
CLASS_COLORS = [
    (0, 255, 0),     # 0: Xanh lá cây
    (255, 0, 0),     # 1: Xanh dương
    (0, 0, 255),     # 2: Đỏ
    (255, 255, 0),   # 3: Xanh lơ (Cyan)
    (0, 255, 255),   # 4: Vàng
    (255, 0, 255),   # 5: Tím hồng (Magenta)
    (255, 128, 0),   # 6: Xanh da trời nhạt
    (0, 165, 255),   # 7: Cam
    (180, 105, 255), # 8: Hồng
    (128, 0, 128),   # 9: Tím đậm
]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Hand Digit MobileNetV3 Detector")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Đường dẫn tới file ảnh hoặc thư mục chứa ảnh cần inference.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=r"D:\Study\Projects\Computer_Vision\work_dirs\hand_digit_mobilenetv3\last.pth",
        help="Đường dẫn tới checkpoint weights (.pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=r"D:\Study\Projects\Computer_Vision\hand_digit_mobilenetv3\config.yaml",
        help="Đường dẫn tới file config.yaml.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Ngưỡng Confidence score để lọc bounding boxes (mặc định 0.15).",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.5,
        help="Ngưỡng NMS IoU (mặc định 0.5).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=320,
        help="Kích thước ảnh đầu vào mô hình (mặc định 320x320).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị chạy model: 'cuda' hoặc 'cpu'.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=r"D:\Study\Projects\Computer_Vision\work_dirs\hand_digit_mobilenetv3\inference_results",
        help="Thư mục lưu ảnh kết quả.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị ảnh kết quả bằng OpenCV GUI.",
    )
    return parser.parse_args()


def load_model(weights_path: str, config_path: str, device: torch.device) -> HandDigitMobileNetV3Detector:
    """Khởi tạo và load trọng số cho mô hình HandDigitMobileNetV3Detector."""
    print(f"\n[INFO] Đang khởi tạo mô hình...")

    # Đọc cấu hình nếu file config tồn tại
    model_cfg = {
        "num_classes": 10,
        "dim": 32,
        "head_type": "DeformableHead",
        "backbone": "NormalBackbone",
        "neck": "PANet",
        "num_heads": 4,
        "num_points": 32,
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if "model" in cfg:
                    m_cfg = cfg["model"]
                    for k in ["num_classes", "dim", "head_type", "backbone", "neck", "num_heads", "num_points"]:
                        if k in m_cfg:
                            model_cfg[k] = m_cfg[k]
            print(f"[INFO] Đã load cấu hình từ: {config_path}")
        except Exception as e:
            print(f"[WARN] Lỗi đọc config: {e}. Sử dụng cấu hình mặc định.")

    # Khởi tạo detector
    model = HandDigitMobileNetV3Detector(
        num_classes=model_cfg.get("num_classes", 10),
        dim=model_cfg.get("dim", 32),
        head_type=model_cfg.get("head_type", "DeformableHead"),
        backbone=model_cfg.get("backbone", "NormalBackbone"),
        neck=model_cfg.get("neck", "PANet"),
        num_heads=model_cfg.get("num_heads", 4),
        num_points=model_cfg.get("num_points", 32),
    )

    # Load weights checkpoint
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {weights_path}")

    print(f"[INFO] Đang tải checkpoint: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    res = model.load_state_dict(state_dict, strict=False)
    if len(res.missing_keys) > 0:
        print(f"[WARN] Missing keys ({len(res.missing_keys)}): {res.missing_keys[:3]}")
    if len(res.unexpected_keys) > 0:
        print(f"[WARN] Unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys[:3]}")

    model.to(device)
    model.eval()
    print(f"[INFO] Load model thành công trên thiết bị: {device}\n")
    return model


def preprocess_image(image_bgr: np.ndarray, target_size: int = 320, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Tiền xử lý ảnh: BGR -> RGB, Resize 320x320, Chuẩn hóa [0, 1], tạo Batch Tensor [1, 3, H, W]."""
    # Resize về kích thước đầu vào của model
    resized = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # Chuyển BGR sang RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Chuẩn hóa về [0, 1] và permute (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def draw_detections(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Vẽ bounding boxes, label class và confidence scores lên ảnh gốc."""
    vis_img = image_bgr.copy()
    h, w = vis_img.shape[:2]

    for box, score, label in zip(boxes, scores, labels):
        cx, cy, bw, bh = box
        # Đổi tọa độ chuẩn hóa [cx, cy, w, h] [0, 1] về pixel gốc [x1, y1, x2, y2]
        x1 = int(round((cx - bw / 2.0) * w))
        y1 = int(round((cy - bh / 2.0) * h))
        x2 = int(round((cx + bw / 2.0) * w))
        y2 = int(round((cy + bh / 2.0) * h))

        # Clamp trong kích thước ảnh
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        # Chọn màu theo class
        color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]

        # Vẽ Bounding Box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness=2)

        # Chuẩn bị nhãn text
        text = f"{int(label)}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Vẽ nền cho text
        text_y1 = max(0, y1 - text_h - 4)
        text_y2 = y1
        text_x1 = x1
        text_x2 = x1 + text_w + 4

        cv2.rectangle(vis_img, (text_x1, text_y1), (text_x2, text_y2), color, -1)
        cv2.putText(
            vis_img,
            text,
            (text_x1 + 2, text_y2 - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return vis_img


def run_inference_on_file(
    image_path: str,
    model: HandDigitMobileNetV3Detector,
    args,
    device: torch.device,
    save_dir: str,
):
    """Chạy inference cho 1 file ảnh cụ thể và lưu kết quả."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Không tìm thấy ảnh: {image_path}")
        return

    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        print(f"[ERROR] Không thể đọc file ảnh (hỏng hoặc không đúng định dạng): {image_path}")
        return

    # Tiền xử lý
    inputs = preprocess_image(orig_bgr, target_size=args.img_size, device=device)

    # Dự đoán
    t_start = time.perf_counter()
    with torch.no_grad():
        preds = model.predict(
            inputs,
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            max_det=100,
        )
    infer_time_ms = (time.perf_counter() - t_start) * 1000.0

    det = preds[0]
    boxes = det["boxes"].detach().cpu().numpy()
    scores = det["scores"].detach().cpu().numpy()
    labels = det["labels"].detach().cpu().numpy()

    num_det = len(boxes)
    img_name = Path(image_path).name
    print(f"-> Ảnh: {img_name} | Phát hiện: {num_det} chữ số | Thời gian: {infer_time_ms:.1f} ms")

    if num_det > 0:
        for idx in range(num_det):
            print(f"   [{idx+1}] Chữ số: {int(labels[idx])} | Confidence: {scores[idx]:.3f} | Box (cx,cy,w,h): {boxes[idx].round(3)}")
    else:
        print(f"   (Không có chữ số nào vượt qua ngưỡng confidence {args.conf})")

    # Vẽ kết quả
    vis_img = draw_detections(orig_bgr, boxes, scores, labels)

    # Lưu ảnh ra thư mục đầu ra
    out_path = os.path.join(save_dir, f"pred_{img_name}")
    cv2.imwrite(out_path, vis_img)
    print(f"   => Đã lưu kết quả tại: {out_path}\n")

    if args.show:
        cv2.imshow(f"Detection: {img_name}", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Tạo thư mục lưu kết quả
    os.makedirs(args.save_dir, exist_ok=True)

    # Tải mô hình
    model = load_model(args.weights, args.config, device)

    # Kiểm tra đường dẫn ảnh là file hay folder
    target_path = Path(args.image)
    if target_path.is_file():
        image_list = [str(target_path)]
    elif target_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_list = [str(p) for p in target_path.iterdir() if p.suffix.lower() in valid_exts]
        print(f"[INFO] Tìm thấy {len(image_list)} ảnh trong thư mục: {args.image}")
    else:
        print(f"[ERROR] Đường dẫn không tồn tại: {args.image}")
        return

    if len(image_list) == 0:
        print(f"[WARN] Không có file ảnh hợp lệ nào để inference.")
        return

    # Lặp qua từng ảnh để inference
    for img_file in image_list:
        run_inference_on_file(img_file, model, args, device, args.save_dir)

    print("[SUCCESS] Hoàn thành toàn bộ quá trình inference!")


if __name__ == "__main__":
    main()
