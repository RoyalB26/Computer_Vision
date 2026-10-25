"""Mathematical and geometric utilities for Computer Vision tasks."""
from typing import Tuple, Union
import torch


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) to (x_center, y_center, width, height)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    x_c = (x1 + x2) * 0.5
    y_c = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x_c, y_c, w, h], dim=-1)


def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute Intersection over Union (IoU) between two sets of bounding boxes in (x1, y1, x2, y2) format.

    Args:
        boxes1: Tensor of shape (N, 4).
        boxes2: Tensor of shape (M, 4).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Tensor of shape (N, M) containing IoU values.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)


def one_hot_encoding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert discrete class indices to one-hot encoded tensor."""
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
