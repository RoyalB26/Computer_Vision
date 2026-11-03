"""Utility functions for bounding box conversions and coordinate transformations."""
from typing import List, Tuple
import torch
from torch import Tensor


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Chuyển đổi boxes từ [cx, cy, w, h] sang [x1, y1, x2, y2]."""
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Chuyển đổi boxes từ [x1, y1, w, h] sang [x1, y1, x2, y2]."""
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    return torch.stack([x1, y1, x1 + w, y1 + h], dim=-1)


def xywh_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Chuyển đổi boxes từ [x1, y1, w, h] sang [cx, cy, w, h]."""
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    return torch.stack([x1 + w / 2.0, y1 + h / 2.0, w, h], dim=-1)
