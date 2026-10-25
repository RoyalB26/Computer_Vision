"""Toy/Synthetic classification dataset inheriting from BaseDataset."""
from typing import Any, Dict, List
import numpy as np
import torch
from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset


@DATASETS.register_module()
class SyntheticImageDataset(BaseDataset):
    """Synthetic dataset for testing and demonstration purposes.

    Args:
        num_samples: Number of synthetic images to generate.
        img_size: Image resolution (H, W).
        num_classes: Number of distinct target classes.
    """

    def __init__(
        self,
        num_samples: int = 100,
        img_size: tuple = (32, 32),
        num_classes: int = 5,
        pipeline=None,
        test_mode: bool = False,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        super().__init__(pipeline=pipeline, test_mode=test_mode)

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Generate synthetic metadata."""
        data_infos = []
        for i in range(self.num_samples):
            label = i % self.num_classes
            data_infos.append({"idx": i, "target": label})
        return data_infos

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Generate a synthetic (C, H, W) RGB tensor image for sample."""
        info = self.data_infos[idx].copy()
        # Create random image as float tensor [0, 1]
        img = torch.rand(3, self.img_size[0], self.img_size[1])
        info["image"] = img
        return info
