"""Dual-Resolution LMDB Dataset for VisDroneYOLO11DeformableSlot.

Outputs high-resolution tensors (e.g. 1280x1280) with all augmentations (HFlip, Scale/Crop,
ColorJitter, Class-Aware Mosaic4) applied once in high-res space. Inside the model, the
backbone downsamples on GPU to `backbone_size` (e.g. 640 or 960) while `DeformableHighResPatchInjector`
samples crisp micro-patches directly from the high-res tensor with 100% pixel alignment.
"""
from __future__ import annotations

from BaseModel.core import DATASETS
from VisDrone_yolov11.dataset import (
    VISDRONE_CLASSES,
    CloseMosaicHook,
    VisDroneLMDBDataset,
    WarmupLRHook,
)


@DATASETS.register_module(name="VisDroneDualResLMDBDataset", force=True)
class VisDroneDualResLMDBDataset(VisDroneLMDBDataset):
    """High-res LMDB dataset alias tailored for Deformable High-Res Micro-Patch models.

    Supports `cache_images: "jpeg"` (~0.8 GB RAM for 6,471 full-res JPEGs) or `"ram"`
    (shared uint8 tensor at pipeline Resize resolution).
    """

    pass


__all__ = [
    "VISDRONE_CLASSES",
    "VisDroneDualResLMDBDataset",
    "WarmupLRHook",
    "CloseMosaicHook",
]
