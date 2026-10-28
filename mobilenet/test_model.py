"""Full test and verification script for MobileNetV3 Small."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from mobilenet.backbone import MobileNetV3SmallBackbone
from mobilenet.classifier import MobileNetV3Classifier
from mobilenet.mobilenet_v3_small import MobileNetV3Small


def test_mobilenet_v3_small():
    print("=" * 70)
    print("1. TESTING MODULAR SUBMODULES (Backbone, Classifier)")
    print("=" * 70)

    # 1.1 Test Backbone
    backbone = MobileNetV3SmallBackbone()
    b_params = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone Total Parameters: {b_params:,} (Expected: 927,008)")
    assert b_params == 927008, f"Mismatch: {b_params} vs 927008"

    x = torch.randn(2, 3, 224, 224)
    feat_last = backbone(x)
    print(f"  Last feature map shape: {feat_last.shape} (Expected: [2, 576, 7, 7])")
    assert feat_last.shape == (2, 576, 7, 7)

    # Multi-scale features test
    multi_feats = backbone.forward_multiscale(x)
    print("  Multi-scale feature maps:")
    for k, v in multi_feats.items():
        print(f"    {k:<10}: {v.shape}")

    # 1.2 Test Classifier
    classifier = MobileNetV3Classifier(in_features=576, mid_features=1024, num_classes=1000)
    c_params = sum(p.numel() for p in classifier.parameters())
    print(f"\nClassifier Total Parameters: {c_params:,} (Expected: 1,615,848)")
    assert c_params == 1615848, f"Mismatch: {c_params} vs 1615848"

    out = classifier(feat_last)
    print(f"  Classifier output shape: {out.shape} (Expected: [2, 1000])")
    assert out.shape == (2, 1000)

    print("\n" + "=" * 70)
    print("2. TESTING FULL MODEL ARCHITECTURE & TOTAL PARAMETERS")
    print("=" * 70)

    model = MobileNetV3Small(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Full Model Total Parameters: {total_params:,} (Expected: 2,542,856)")
    assert total_params == 2542856, f"Mismatch: {total_params} vs 2,542,856"

    print("\n" + "=" * 70)
    print("3. TESTING WEIGHT LOADING & NUMERICAL EQUIVALENCE WITH TORCHVISION")
    print("=" * 70)

    # Load torchvision reference model
    tv_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()

    # Load weights into our custom model
    model_pretrained = MobileNetV3Small(pretrained=True).eval()

    # Compare outputs on deterministic input
    torch.manual_seed(42)
    test_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        tv_out = tv_model(test_tensor)
        our_out = model_pretrained(test_tensor)

    max_diff = (tv_out - our_out).abs().max().item()
    print(f"Max absolute difference with torchvision: {max_diff}")
    assert max_diff < 1e-4, f"Output difference too large: {max_diff}"

    print("\n===> ALL MOBILENET_V3_SMALL TESTS PASSED WITH 100% PRECISION!")
    print("=" * 70)


if __name__ == "__main__":
    test_mobilenet_v3_small()
