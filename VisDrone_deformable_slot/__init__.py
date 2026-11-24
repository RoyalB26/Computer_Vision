"""VisDrone YOLOv11 + Slot-Guided Deformable High-Res Micro-Patch Injection."""
from .deformable_patch import DeformableHighResPatchInjector
from .model import VisDroneYOLO11DeformableSlot
from .dataset import VisDroneDualResLMDBDataset

__all__ = [
    "DeformableHighResPatchInjector",
    "VisDroneYOLO11DeformableSlot",
    "VisDroneDualResLMDBDataset",
]
