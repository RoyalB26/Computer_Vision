"""Unit tests and verification suite for VisDrone YOLOv11."""
import os
import sys
from pathlib import Path
import torch

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from VisDrone_yolov11.loss import DIoULoss, SigmoidFocalLoss, YOLOv11DetectionLoss
from VisDrone_yolov11.model import VisDroneYOLO11
from VisDrone_yolov11.metrics import VisDroneMAP
from VisDrone_yolov11.dataset import VISDRONE_CLASSES


def test_losses():
    print("=" * 60)
    print("TEST 1: LOSS FUNCTIONS (DIoU Loss & Focal Loss)")
    print("=" * 60)

    # 1.1 Test DIoU Loss
    diou_loss_fn = DIoULoss(loss_weight=1.0)
    # Perfect match: IoU=1.0, rho=0 -> DIoU=1.0 -> Loss=0.0
    box_a = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    box_b = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    loss_same = diou_loss_fn(box_a, box_b, xywh=False).item()
    print(f"DIoU Loss (Identical boxes): {loss_same:.6f} (expected ~0.0)")
    assert abs(loss_same) < 1e-4, f"Expected 0.0, got {loss_same}"

    # Disjoint boxes: IoU=0.0, center distance penalty added
    box_far = torch.tensor([[100.0, 100.0, 150.0, 150.0]])
    loss_far = diou_loss_fn(box_a, box_far, xywh=False).item()
    print(f"DIoU Loss (Far apart boxes): {loss_far:.4f} (expected > 1.0 due to distance penalty)")
    assert loss_far > 1.0, f"Expected > 1.0, got {loss_far}"

    # 1.2 Test Sigmoid Focal Loss
    focal_loss_fn = SigmoidFocalLoss(loss_weight=1.0, alpha=0.25, gamma=2.0)
    logits = torch.randn(2, 100, 10)
    targets = torch.zeros(2, 100, 10)
    targets[0, 5, 2] = 1.0
    targets[1, 10, 8] = 1.0
    loss_focal = focal_loss_fn(logits, targets, normalizer=2.0)
    print(f"Focal Loss: {loss_focal.item():.4f}")
    assert not torch.isnan(loss_focal) and loss_focal.item() > 0.0

    print(">> Loss functions test PASSED!\n")


def test_model_forward_and_loss():
    print("=" * 60)
    print("TEST 2: MODEL ARCHITECTURE, WEIGHT LOADING, FORWARD & BACKWARD")
    print("=" * 60)

    weights_path = Path(root_dir) / "yolo11n.pt"
    pretrained = str(weights_path) if weights_path.exists() else None

    model = VisDroneYOLO11(num_classes=10, pretrained=pretrained)
    model.train()

    # Create dummy batch of 2 images (640x640)
    images = torch.randn(2, 3, 640, 640)
    targets = [
        {
            "boxes": torch.tensor([
                [0.1, 0.1, 0.2, 0.2],   # [x1, y1, w, h] normalized
                [0.5, 0.4, 0.15, 0.2],
            ]),
            "labels": torch.tensor([0, 3]),  # pedestrian, car
        },
        {
            "boxes": torch.tensor([
                [0.3, 0.2, 0.2, 0.3],
            ]),
            "labels": torch.tensor([8]),     # bus
        },
    ]

    # Forward Train
    loss_dict = model.forward_train(images, targets)
    print("Forward Train Loss Outputs:")
    for k, v in loss_dict.items():
        print(f"  - {k}: {v.item():.4f}")
        assert not torch.isnan(v), f"NaN detected in {k}!"

    # Backward Pass
    loss = loss_dict["loss"]
    loss.backward()

    # Check gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Backward Pass successful: Gradients computed = {has_grad}")
    assert has_grad, "No gradients were computed!"

    print(">> Forward train and backward test PASSED!\n")


def test_prediction_and_metrics():
    print("=" * 60)
    print("TEST 3: PREDICTION, NMS & METRIC EVALUATION (VisDroneMAP)")
    print("=" * 60)

    model = VisDroneYOLO11(num_classes=10).eval()
    images = torch.randn(2, 3, 640, 640)

    # Test predict
    results = model.predict(images, conf_threshold=0.01, nms_threshold=0.5, max_det=50)
    assert len(results) == 2, f"Expected 2 predictions, got {len(results)}"
    for i, res in enumerate(results):
        print(f"Image {i}: detected {len(res['boxes'])} boxes")
        assert "boxes" in res and "scores" in res and "labels" in res

    # Test metric
    metric = VisDroneMAP(num_classes=10)
    metric.reset()

    # Synthetic perfect match
    gt_targets = [
        {
            "boxes": torch.tensor([[50.0, 50.0, 100.0, 100.0]]),
            "labels": torch.tensor([3]),
        }
    ]
    pred_results = [
        {
            "boxes": torch.tensor([[50.0, 50.0, 100.0, 100.0]]),
            "scores": torch.tensor([0.98]),
            "labels": torch.tensor([3]),
        }
    ]

    metric.update(pred_results, gt_targets)
    metrics_res = metric.compute()
    print("VisDroneMAP Results (Perfect Match):")
    for k, v in metrics_res.items():
        print(f"  - {k}: {v}")
    assert metrics_res["mAP_50"] >= 0.99, f"Expected mAP_50 >= 0.99, got {metrics_res['mAP_50']}"

    print(">> Prediction and metric evaluation test PASSED!\n")


if __name__ == "__main__":
    test_losses()
    test_model_forward_and_loss()
    test_prediction_and_metrics()
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
