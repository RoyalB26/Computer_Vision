"""Unit tests and full pipeline verification for VisDrone Counting Model.

Tests:
  1. filter_color_bg functionality and tensor shapes.
  2. PositionEmbeddingSine (2D Sine-Cosine Positional Encoding).
  3. SlotAttention module.
  4. VisDroneDetector forward (train, val, predict), loss computation, and backward gradient pass.
  5. ClassCountingMetric accumulation, MAE, RMSE, and round-accuracy computation.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from Visdrone.dataset import VISDRONE_CLASSES
from Visdrone.metrics import ClassCountingMetric
from Visdrone.model import (
    PositionEmbeddingSine,
    SlotAttention,
    VisDroneDetector,
    filter_color_bg,
)


def test_filter_color_bg():
    print("--- 1. Testing filter_color_bg ---")
    dummy_images = torch.rand((2, 3, 960, 960), dtype=torch.float32)
    mask = filter_color_bg(dummy_images, target_res=(120, 120), exp_factor=5.0)
    assert mask.shape == (2, 120, 120), f"Expected shape (2, 120, 120), got {mask.shape}"
    assert ((mask == 0.0) | (mask == 1.0)).all(), "Mask must contain only 0.0 and 1.0"
    print(f"filter_color_bg PASSED! Mask shape: {mask.shape}")


def test_pos_embedding():
    print("--- 2. Testing PositionEmbeddingSine ---")
    pos_enc = PositionEmbeddingSine(num_pos_feats=16)  # dim = 32
    dummy_feat = torch.zeros((2, 32, 60, 60))
    pos = pos_enc(dummy_feat)
    assert pos.shape == (2, 32, 60, 60), f"Expected (2, 32, 60, 60), got {pos.shape}"
    print(f"PositionEmbeddingSine PASSED! Pos shape: {pos.shape}")


def test_slot_attention():
    print("--- 3. Testing SlotAttention Module ---")
    dim = 32
    num_classes = 10
    num_slots = num_classes + 1
    sa = SlotAttention(dim=dim, num_class=num_classes)
    dummy_tokens = torch.randn((2, 100, dim))  # (B, N, dim)
    slots, attn_ori, attn_norm = sa(dummy_tokens)
    assert slots.shape == (2, num_slots, dim), f"Expected slots shape {(2, num_slots, dim)}, got {slots.shape}"
    assert attn_ori.shape == (2, num_slots, 100), f"Expected attn_ori shape {(2, num_slots, 100)}, got {attn_ori.shape}"
    print(f"SlotAttention PASSED! Slots: {slots.shape}, Attn: {attn_ori.shape}")


def test_visdrone_detector():
    print("--- 4. Testing VisDroneDetector Forward, Backward & Loss ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisDroneDetector(
        num_classes=10,
        dim=32,
        pretrained=False,
        exp_factor=5.0,
        count_loss_weight=1.0,
        bg_attn_weight=0.5,
        fg_attn_weight=0.5,
        suppress_weight=0.05,
    ).to(device)

    # 4.1 Forward Train with Loss
    dummy_x = torch.rand((2, 3, 960, 960), device=device)
    dummy_targets = torch.tensor([
        [2.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ], device=device)

    model.train()
    loss_dict = model(dummy_x, targets=dummy_targets)
    assert "loss" in loss_dict, "Loss dictionary must contain 'loss'"
    assert "loss_count" in loss_dict, "Loss dictionary must contain 'loss_count'"
    assert "loss_bg_attn" in loss_dict, "Loss dictionary must contain 'loss_bg_attn'"
    assert "loss_fg_attn" in loss_dict, "Loss dictionary must contain 'loss_fg_attn'"
    print(f"Loss dict: {{k: round(v.item(), 4) for k, v in loss_dict.items()}}")

    # 4.2 Backward Pass & Gradients
    total_loss = loss_dict["loss"]
    total_loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and "backbone" not in name:
            assert param.grad is not None, f"Parameter {name} has no gradient!"

    print("Backward pass & gradients verified successfully!")

    # 4.3 Forward Validation
    model.eval()
    with torch.no_grad():
        val_out = model(dummy_x, targets=dummy_targets)
        assert "preds" in val_out, "Validation output must contain 'preds'"
        assert val_out["preds"].shape == (2, 10), f"Expected preds shape (2, 10), got {val_out['preds'].shape}"
        print(f"Forward Val PASSED! Preds shape: {val_out['preds'].shape}")

    # 4.4 Inference Predict
    with torch.no_grad():
        pred_counts = model.predict(dummy_x)
        assert pred_counts.shape == (2, 10), f"Expected predict shape (2, 10), got {pred_counts.shape}"
        print(f"Inference Predict PASSED! Shape: {pred_counts.shape}")


def test_metric():
    print("--- 5. Testing ClassCountingMetric ---")
    metric = ClassCountingMetric(num_classes=10)
    metric.reset()

    # Giả lập 2 batch dự đoán và ground truth
    preds_b1 = torch.tensor([[2.1, 0.9, 0.0, 5.2, 0.1, 1.0, 0.0, 0.0, 0.0, 3.1]])
    targets_b1 = torch.tensor([[2.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0]])

    preds_b2 = torch.tensor([[0.0, 0.2, 1.0, 2.1, 0.9, 0.0, 0.0, 1.1, 0.0, 0.0]])
    targets_b2 = torch.tensor([[0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

    metric.update(preds_b1, targets_b1)
    metric.update(preds_b2, targets_b2)

    results = metric.compute()
    assert "val_mae" in results, "Metric output must contain 'val_mae'"
    assert "val_rmse" in results, "Metric output must contain 'val_rmse'"
    assert "val_acc" in results, "Metric output must contain 'val_acc'"
    assert "mae_pedestrian" in results, "Metric output must contain 'mae_pedestrian'"
    print(f"Metric Results: {results}")
    print("ClassCountingMetric PASSED!")


def main():
    print("=" * 70)
    print("      RUNNING VISDRONE PIPELINE VERIFICATION SUITE")
    print("=" * 70)
    test_filter_color_bg()
    test_pos_embedding()
    test_slot_attention()
    test_visdrone_detector()
    test_metric()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! The pipeline is fully functional and ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
