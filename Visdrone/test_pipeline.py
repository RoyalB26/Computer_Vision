"""Test script to thoroughly verify the VisDrone Attention Slot counting pipeline."""
import os
import sys
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Visdrone.model import VisDroneDetector, SlotAttention, filter_color_bg
from Visdrone.metrics import ClassCountingMetric


def test_filter_color_bg():
    print("--- 1. Testing filter_color_bg ---")
    B, C, H, W = 2, 3, 960, 960
    images = torch.rand(B, C, H, W)
    mask = filter_color_bg(images, target_res=(960, 960), exp_factor=5.0)
    assert mask.shape == (B, 960, 960), f"Expected {(B, 960, 960)}, got {mask.shape}"
    assert set(mask.unique().tolist()).issubset({0.0, 1.0}), f"Mask values should be in {{0, 1}}, got {mask.unique()}"
    print(f"filter_color_bg PASSED! Mask shape: {mask.shape}")


def test_slot_attention():
    print("--- 2. Testing SlotAttention Module ---")
    B, N, dim, num_class = 2, 100, 32, 10
    slot_attn = SlotAttention(dim=dim, num_class=num_class)
    inputs = torch.randn(B, N, dim)
    slots, attn_ori, attn_norm = slot_attn(inputs)
    assert slots.shape == (B, num_class + 1, dim), f"Expected {(B, num_class + 1, dim)}, got {slots.shape}"
    assert attn_ori.shape == (B, num_class + 1, N), f"Expected {(B, num_class + 1, N)}, got {attn_ori.shape}"
    print(f"SlotAttention PASSED! Slots shape: {slots.shape}, Attn shape: {attn_ori.shape}")


def test_visdrone_detector():
    print("--- 3. Testing VisDroneDetector Forward, Backward & Loss ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, C, H, W = 2, 3, 960, 960
    dim = 32
    num_classes = 10
    model = VisDroneDetector(num_classes=num_classes, dim=dim, stride=1).to(device)

    # 1. Forward Train
    inputs = torch.rand(B, C, H, W, device=device)
    targets = torch.randint(0, 15, (B, num_classes), device=device).float()  # (B, 10)

    model.train()
    loss_dict = model(inputs, targets=targets)
    assert "loss" in loss_dict, "Missing 'loss' in outputs"
    assert "loss_count" in loss_dict, "Missing 'loss_count' in outputs"
    assert "loss_bg_attn" in loss_dict, "Missing 'loss_bg_attn' in outputs"
    assert "loss_fg_attn" in loss_dict, "Missing 'loss_fg_attn' in outputs"
    print(f"Loss dict: { {k: round(v.item(), 4) for k, v in loss_dict.items()} }")

    # 2. Backward
    loss_dict["loss"].backward()
    assert model.proj_p3.weight.grad is not None, "proj_p3 grad is None"
    assert model.proj_p4.weight.grad is not None, "proj_p4 grad is None"
    assert model.proj_p5.weight.grad is not None, "proj_p5 grad is None"
    assert model.level_embed.grad is not None, "level_embed grad is None"
    assert model.slot_attention.obj_queries.grad is not None, "obj_queries grad is None"
    for i in range(num_classes):
        assert model.fc_actor[i][0].weight.grad is not None, f"fc_actor[{i}] grad is None"
    print("Backward pass & gradients verified successfully!")

    # 3. Forward Val
    model.eval()
    with torch.no_grad():
        val_outputs = model(inputs, targets=targets)
        assert "preds" in val_outputs, "Missing 'preds' in val_outputs"
        preds = val_outputs["preds"]
        assert preds.shape == (B, num_classes), f"Expected preds shape {(B, num_classes)}, got {preds.shape}"
        assert (preds >= 0).all(), "Predicted counts should be non-negative"
        print(f"Forward Val PASSED! Preds shape: {preds.shape}, sample prediction: {preds[0].cpu().numpy()}")

    # 4. Predict
    with torch.no_grad():
        preds_infer = model.predict(inputs)
        assert preds_infer.shape == (B, num_classes)
        print(f"Inference Predict PASSED! Shape: {preds_infer.shape}")


def test_counting_metric():
    print("--- 4. Testing ClassCountingMetric ---")
    metric = ClassCountingMetric(num_classes=10)
    preds = torch.tensor([[1.2, 0.1, 5.0, 0.0, 2.8, 0.0, 0.0, 1.1, 0.0, 3.9],
                          [0.0, 2.1, 0.0, 4.2, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 4.0],
                            [0.0, 2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0]])
    metric.update(preds, targets)
    results = metric.compute()
    print("Metric Results:", {k: round(v, 4) for k, v in results.items()})
    assert "val_mae" in results and "val_rmse" in results and "val_acc" in results
    assert results["val_mae"] < 0.2
    assert results["val_acc"] == 1.0  # Tất cả đều làm tròn đúng
    print("ClassCountingMetric PASSED!")


if __name__ == "__main__":
    test_filter_color_bg()
    test_slot_attention()
    test_visdrone_detector()
    test_counting_metric()
    print("\nALL TESTS PASSED SUCCESSFULLY! The pipeline is fully functional and ready.")
