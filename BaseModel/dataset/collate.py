"""Custom Collate Functions for specialized Computer Vision tasks."""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from ..core.registrar import COLLATE_FUNCTIONS
from .base_dataset import BaseDataset


@COLLATE_FUNCTIONS.register_module()
class SmartCollate:
    """Standard intelligent collate function from BaseDataset.

    Stacks matching shape tensors, leaves variable-length objects as python lists.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return BaseDataset.collate_fn(batch)


@COLLATE_FUNCTIONS.register_module()
class PadCollate:
    """Pads variable height/width image tensors in a batch to the maximum dimensions.

    Crucial for Scene Text Recognition (OCR), document analysis, or multi-scale training
    where image sizes within the same batch vary.

    Args:
        pad_value: Value to pad with (default: 0.0).
        size_divisor: If set (e.g. 32), pads max_h and max_w to be divisible by this number.
    """

    def __init__(self, pad_value: float = 0.0, size_divisor: Optional[int] = None):
        self.pad_value = pad_value
        self.size_divisor = size_divisor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}

        collated = {}
        # Collect images
        images = [b["image"] for b in batch if "image" in b]

        if images and isinstance(images[0], torch.Tensor):
            max_c = max(img.shape[0] for img in images)
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)

            if self.size_divisor is not None:
                max_h = ((max_h + self.size_divisor - 1) // self.size_divisor) * self.size_divisor
                max_w = ((max_w + self.size_divisor - 1) // self.size_divisor) * self.size_divisor

            padded_images = []
            for img in images:
                c, h, w = img.shape
                pad_bottom = max_h - h
                pad_right = max_w - w
                # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
                padded = F.pad(img, (0, pad_right, 0, pad_bottom), value=self.pad_value)
                padded_images.append(padded)

            collated["image"] = torch.stack(padded_images, dim=0)

        # Collate remaining keys
        for key in batch[0]:
            if key == "image":
                continue
            values = [b[key] for b in batch]
            first_val = values[0]
            if isinstance(first_val, torch.Tensor) and all(v.shape == first_val.shape for v in values):
                collated[key] = torch.stack(values, dim=0)
            elif isinstance(first_val, (int, float)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = values

        return collated


@COLLATE_FUNCTIONS.register_module()
class DetectionCollate:
    """Collate function for Object Detection and Instance Segmentation.

    Stacks images of identical dimensions into a single (B, C, H, W) tensor,
    while keeping variable-length bounding boxes and class labels as lists of tensors.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}

        collated = {}
        # 1. Images
        if "image" in batch[0]:
            images = [b["image"] for b in batch]
            if isinstance(images[0], torch.Tensor):
                collated["image"] = torch.stack(images, dim=0)
            else:
                collated["image"] = images

        # 2. Targets (Bounding boxes & labels)
        if "target" in batch[0]:
            collated["target"] = [b["target"] for b in batch]

        # 3. Other metadata
        for key in batch[0]:
            if key in ["image", "target"]:
                continue
            values = [b[key] for b in batch]
            first_val = values[0]
            if isinstance(first_val, torch.Tensor) and all(v.shape == first_val.shape for v in values):
                collated[key] = torch.stack(values, dim=0)
            elif isinstance(first_val, (int, float)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = values

        return collated
