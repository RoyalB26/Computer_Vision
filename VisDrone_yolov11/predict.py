"""Inference and Visualization Script for VisDrone YOLOv11.

Allows running inference on single images or entire image folders,
drawing predicted bounding boxes, labels, and confidence scores.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms.functional as TF

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from VisDrone_yolov11.model import VisDroneYOLO11
from VisDrone_yolov11.dataset import VISDRONE_CLASSES


# Distinct color palette for 10 VisDrone categories
COLOR_PALETTE = [
    (255, 56, 56),    # pedestrian (red)
    (255, 157, 151),  # people (light red)
    (255, 112, 31),   # bicycle (orange)
    (255, 178, 29),   # car (amber)
    (207, 210, 49),   # van (yellow-green)
    (72, 249, 10),    # truck (green)
    (146, 204, 23),   # tricycle (lime)
    (61, 219, 134),   # awning-tricycle (teal)
    (26, 147, 243),   # bus (blue)
    (171, 71, 188),   # motorcycle (purple)
]


def draw_detections(
    image: Image.Image,
    detections: dict,
    class_names=VISDRONE_CLASSES,
    line_width: int = 3,
) -> Image.Image:
    """Draw bounding boxes and class labels on a PIL Image."""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    orig_w, orig_h = image.size

    boxes = detections["boxes"].cpu()
    scores = detections["scores"].cpu()
    labels = detections["labels"].cpu()

    # Load default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(len(boxes)):
        box = boxes[i].tolist()
        score = float(scores[i])
        cls_id = int(labels[i])

        x1, y1, x2, y2 = box
        # Clamp to image boundaries
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        caption = f"{cls_name} {score:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw text label background
        text_bbox = draw.textbbox((x1, y1), caption, font=font) if hasattr(draw, "textbbox") else (x1, y1, x1 + 60, y1 + 12)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, max(0, y1 - text_h - 4), x1 + text_w + 4, max(text_h + 4, y1)], fill=color)
        draw.text((x1 + 2, max(0, y1 - text_h - 2)), caption, fill=(255, 255, 255), font=font)

    return draw_img


def predict_image(
    model: VisDroneYOLO11,
    image_path: str,
    img_size: int = 640,
    device: str = "cuda",
    conf_th: float = 0.25,
    nms_th: float = 0.5,
) -> Tuple[Image.Image, dict]:
    """Preprocess, predict, and scale detections back to original image size."""
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Resize and convert to tensor
    resized_img = TF.resize(orig_img, [img_size, img_size])
    tensor_img = TF.to_tensor(resized_img).unsqueeze(0).to(device)

    # Predict
    results = model.predict(tensor_img, conf_threshold=conf_th, nms_threshold=nms_th)
    det = results[0]

    # Scale boxes back to original image size
    if len(det["boxes"]) > 0:
        scale_x = orig_w / float(img_size)
        scale_y = orig_h / float(img_size)
        det["boxes"][:, [0, 2]] *= scale_x
        det["boxes"][:, [1, 3]] *= scale_y

    annotated = draw_detections(orig_img, det)
    return annotated, det


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 prediction on VisDrone images.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Path to checkpoint weights (.pt or .pth)")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Output directory to save predictions")
    parser.add_argument("--conf-th", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms-th", type=float, default=0.5, help="NMS threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Initializing VisDrone YOLOv11 on device: {args.device}...")
    model = VisDroneYOLO11(num_classes=10)
    if os.path.exists(args.weights):
        if args.weights.endswith(".pt"):
            model.load_pretrained(args.weights)
        elif args.weights.endswith(".pth"):
            ckpt = torch.load(args.weights, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            model.load_state_dict(sd, strict=False)
            print(f"Loaded weights from {args.weights}")
    else:
        print(f"Notice: weights path {args.weights} not found. Running with initial weights.")

    model.to(args.device)
    model.eval()

    # Collect images
    if os.path.isdir(args.image):
        img_paths = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    else:
        img_paths = [args.image]

    print(f"Found {len(img_paths)} images to process.")
    for path in img_paths:
        annotated_img, det = predict_image(
            model=model,
            image_path=path,
            img_size=args.img_size,
            device=args.device,
            conf_th=args.conf_th,
            nms_th=args.nms_th,
        )
        out_name = os.path.basename(path)
        save_path = os.path.join(args.out_dir, f"pred_{out_name}")
        annotated_img.save(save_path)
        print(f"Processed: {path} -> {len(det['boxes'])} objects detected -> Saved to: {save_path}")


if __name__ == "__main__":
    main()
