"""Toy/Synthetic classification dataset inheriting from BaseDataset."""
from typing import Any, Dict, List
import numpy as np
import torch
from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset
import json

@DATASETS.register_module()
class Dataset(BaseDataset):
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
        split: str = "train",
        ratio: float = 0.8,
        num_classes: int = 5,
        pipeline=None,
        test_mode: bool = False,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.split = split
        self.ratio = ratio
        super().__init__(pipeline=pipeline, test_mode=test_mode)

    def load_shape_dataset(self, npy_path: str = "toy_detection/data.npy", json_path: str = "toy_detection/labels.json"):
        """
        Đọc dữ liệu từ file .npy và .json.
        
        Returns:
            images: np.ndarray kích thước (N, H, W) dtype uint8 (hoặc float32 chuẩn hóa)
            labels: list dict chứa metadata từng mẫu
            targets: np.ndarray chứa nhãn dạng số (integer class id)
        """
        # 1. Đọc mảng ảnh
        images = np.load(npy_path)

        # 2. Đọc metadata từ JSON
        with open(json_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        # 3. Tạo mapping nhãn chữ sang dạng số nguyên
        label_to_idx = {
            "empty": 0,
            "plus": 1,
            "minus": 2,
            "cross": 3,
            "circle": 4
        }
        
        # Trích xuất nhãn dạng số nguyên
        targets = np.array([label_to_idx[item["label"]] for item in labels], dtype=np.int64)

        return images, labels, targets

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Generate synthetic metadata."""
        data_infos = []
        images, labels, targets= self.load_shape_dataset()
        start = 0 if self.split == 'train' else int(len(images) * self.ratio)
        end = len(images) if self.split == 'val' else int(len(images) * self.ratio)
        for i in range(start, end):
            # label = i % self.num_classes
            data_infos.append({"image": images[i], "idx": i, "target": targets[i], "label": labels[i]})
        return data_infos

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Generate a synthetic (C, H, W) RGB tensor image for sample."""
        info = self.data_infos[idx].copy()
        return info
