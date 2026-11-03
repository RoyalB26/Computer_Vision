"""VisDrone YOLOv11 Package.

Includes YOLOv11 detection architecture trained with DIoU Loss and Focal Loss on VisDrone dataset,
built on top of BaseModel modular framework.
"""
from .model import VisDroneYOLO11
from .loss import DIoULoss, SigmoidFocalLoss, DistributionFocalLoss, YOLOv11DetectionLoss
from .dataset import VisDroneLMDBDataset, VisDroneFolderDataset, CustomLMDBDataset, VISDRONE_CLASSES
from .metrics import VisDroneMAP

__all__ = [
    "VisDroneYOLO11",
    "DIoULoss",
    "SigmoidFocalLoss",
    "DistributionFocalLoss",
    "YOLOv11DetectionLoss",
    "VisDroneLMDBDataset",
    "VisDroneFolderDataset",
    "CustomLMDBDataset",
    "VISDRONE_CLASSES",
    "VisDroneMAP",
]
