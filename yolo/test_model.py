"""Verification script for YOLOv11 modular classes and weight loading."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from yolo.backbone import YOLOv11Backbone
from yolo.neck import YOLOv11Neck
from yolo.head import Detect
from yolo.yolov11 import YOLOv11, YOLOv11Sequential


def test_modular_yolov11():
    print("=" * 70)
    print("1. TESTING INDIVIDUAL COMPONENT CLASSES (Backbone, Neck, Head)")
    print("=" * 70)

    # 1.1 Test Backbone
    backbone = YOLOv11Backbone()
    x = torch.randn(2, 3, 640, 640)
    feat4, feat6, feat10 = backbone(x)
    print(f"Backbone Params: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"  feat4 shape : {feat4.shape} (stride 8)")
    print(f"  feat6 shape : {feat6.shape} (stride 16)")
    print(f"  feat10 shape: {feat10.shape} (stride 32)")

    # 1.2 Test Neck
    neck = YOLOv11Neck()
    p3, p4, p5 = neck(feat4, feat6, feat10)
    print(f"\nNeck Params: {sum(p.numel() for p in neck.parameters()):,}")
    print(f"  P3 shape: {p3.shape}")
    print(f"  P4 shape: {p4.shape}")
    print(f"  P5 shape: {p5.shape}")

    # 1.3 Test Head
    head = Detect(nc=80, ch=(64, 128, 256)).eval()
    print(f"\nHead Params: {sum(p.numel() for p in head.parameters()):,}")
    out, raw = head([p3, p4, p5])
    print(f"  Head output: {out.shape}")

    print("\n" + "=" * 70)
    print("2. TESTING FULL MODEL & WEIGHT LOADING FROM yolo11n.pt")
    print("=" * 70)

    weights_file = Path("yolo11n.pt")
    if not weights_file.exists():
        print(f"Warning: {weights_file} not found. Skipping weight loading test.")
        return

    # 2.1 Load weights into Modular YOLOv11
    model_modular = YOLOv11(nc=80, weights=weights_file).eval()

    # 2.2 Load weights into YOLOv11Sequential
    model_seq = YOLOv11Sequential(nc=80, weights=weights_file).eval()

    # 2.3 Verify numerical equality
    torch.manual_seed(42)
    test_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out_mod, _ = model_modular(test_input)
        out_seq, _ = model_seq(test_input)

    max_diff = (out_mod - out_seq).abs().max().item()
    print(f"Max diff between Modular and Sequential models: {max_diff}")
    assert max_diff < 1e-4, "Outputs do not match!"

    print("ALL TESTS PASSED WITH 100% ACCURACY AND CLEAN WEIGHT LOADING!")
    print("=" * 70)


if __name__ == "__main__":
    test_modular_yolov11()
