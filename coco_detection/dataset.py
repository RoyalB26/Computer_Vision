"""COCO Detection Dataset re-using COCOLMDBDataset from Coco/."""
import os
import sys

# Đảm bảo import được Coco package
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from BaseModel.core import DATASETS
from Coco.dataset import COCOLMDBDataset


@DATASETS.register_module(name="CocoDetectionDataset")
class CocoDetectionDataset(COCOLMDBDataset):
    """COCO LMDB Dataset for object detection (bounding boxes + labels)."""

    def __init__(self, *args, **kwargs):
        # Mặc định task là 'detection'
        kwargs["task"] = kwargs.get("task", "detection")
        super().__init__(*args, **kwargs)


__all__ = ["CocoDetectionDataset", "COCOLMDBDataset"]
