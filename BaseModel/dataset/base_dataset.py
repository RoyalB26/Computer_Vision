"""Base Dataset Class for Computer Vision tasks.

Provides data caching, transform pipeline integration, and standardized output format.
"""
from abc import ABC, abstractmethod
import os
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from .transforms.compose import Compose


class BaseDataset(Dataset, ABC):
    """Abstract Base Dataset for Computer Vision datasets.

    Subclasses only need to implement `load_annotations()` and optionally `get_data_info()`.
    All transform pipelining, caching, indexing, and collate logic are handled automatically.

    Args:
        data_root: Root directory of dataset.
        ann_file: Path to annotation file (json, txt, csv, etc.).
        pipeline: Data transform pipeline (Compose instance or list of transform configs).
        test_mode: Whether dataset is used in test/evaluation mode.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        ann_file: Optional[str] = None,
        pipeline: Optional[Union[List[Union[Dict[str, Any], Callable]], Compose]] = None,
        test_mode: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.ann_file = os.path.join(data_root, ann_file) if (data_root and ann_file and not os.path.isabs(ann_file)) else ann_file
        self.test_mode = test_mode

        # Pipeline setup
        if pipeline is None:
            self.pipeline = Compose([])
        elif isinstance(pipeline, Compose):
            self.pipeline = pipeline
        else:
            self.pipeline = Compose(pipeline)

        # Load annotation data list
        self.data_infos: List[Dict[str, Any]] = self.load_annotations()

    @abstractmethod
    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load annotations from ann_file or data_root.

        Returns:
            List of dictionaries where each dict represents a single sample's metadata.
            Example:
                [
                    {'img_path': 'path/to/img1.jpg', 'target': 0},
                    {'img_path': 'path/to/img2.jpg', 'target': 1},
                    ...
                ]
        """
        pass

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Fetch raw sample data info by index.

        Subclasses can override to lazy-load images or annotations.
        """
        return self.data_infos[idx].copy()

    def prepare_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Fetch data info and apply transform pipeline."""
        data_info = self.get_data_info(idx)

        # If data_info contains an img_path and not loaded image, lazy-load image
        if "image" not in data_info and "img_path" in data_info:
            from PIL import Image
            img_path = data_info["img_path"]
            if self.data_root and not os.path.isabs(img_path):
                img_path = os.path.join(self.data_root, img_path)
            data_info["image"] = Image.open(img_path).convert("RGB")

        # Apply transforms pipeline
        data = self.pipeline(data_info)
        return data

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.prepare_data(idx)
        if data is None:
            # Fallback to random sample if corrupt sample encountered
            import random
            return self.__getitem__(random.randint(0, len(self) - 1))
        return data

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Smart collate function that handles both stackable tensors and variable-sized objects.

        - Tensors of matching shapes are stacked along dimension 0.
        - Scalars and integers are converted to torch.Tensor.
        - Variable length sequences or bounding box lists are preserved as python lists.
        """
        if not batch:
            return {}

        elem = batch[0]
        collated = {}

        for key in elem:
            values = [b[key] for b in batch]

            first_val = values[0]
            if isinstance(first_val, torch.Tensor):
                # Check if all shapes match
                if all(v.shape == first_val.shape for v in values):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    collated[key] = values
            elif isinstance(first_val, (int, float)):
                collated[key] = torch.tensor(values)
            elif isinstance(first_val, str):
                collated[key] = values
            else:
                try:
                    collated[key] = torch.tensor(values)
                except Exception:
                    collated[key] = values

        return collated
