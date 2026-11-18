import io
import os
import pickle
import random
from typing import Any, Dict, List, Optional
import torch
from PIL import Image
from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset, open_lmdb

VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motorcycle",
]


@DATASETS.register_module(name="CustomLMDBDataset", force=True)
@DATASETS.register_module(name="VisDroneLMDBDataset", force=True)
class CustomLMDBDataset(BaseDataset):
    """Dataset LMDB cho VisDrone chuẩn hóa nhãn về đếm số lượng 10 class: shape (10,).

    Khi gom batch qua SmartCollate, label sẽ có shape (B, 10).
    """

    def __init__(
        self,
        data_root: str,
        pipeline=None,
        split: str = "train",        # 'train' hoặc 'val'
        split_ratio: float = 0.8,      # 80% train, 20% val
        seed: int = 42,              # Seed đảm bảo train và val không bị lẫn
        test_mode: bool = False,
        num_classes: int = 10,
        return_dict: bool = False,   # Nếu True thì trả về dict {'counts', 'boxes', 'labels'}
        **kwargs,
    ):
        self.split = split.lower()
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_classes = num_classes
        self.return_dict = return_dict
        self.env = None
        self.txn = None
        super().__init__(data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    def load_annotations(self) -> List[Dict[str, Any]]:
        # 1. Đọc tổng số mẫu trong file LMDB
        if not os.path.exists(self.data_root):
            return []

        env, txn = open_lmdb(self.data_root, readonly=True)
        total_len_bytes = txn.get(b"__len__")
        if total_len_bytes is None:
            total = env.stat()["entries"]
        else:
            total = int(total_len_bytes.decode("ascii"))
        # 2. Tạo danh sách index và xáo trộn đồng nhất bằng seed
        indices = list(range(total))
        rng = random.Random(self.seed)
        rng.shuffle(indices)

        # 3. Chia danh sách theo tỉ lệ split_ratio
        split_point = int(total * self.split_ratio)
        if self.split == "train":
            selected_indices = indices[:split_point]
        else:  # 'val' hoặc 'test'
            selected_indices = indices[split_point:]

        # Lưu lại index thực tế trong LMDB
        return [{"lmdb_idx": idx} for idx in selected_indices]

    def _init_db(self):
        if self.env is None:
            self.env, self.txn = open_lmdb(self.data_root, readonly=True)

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        self._init_db()
        real_idx = self.data_infos[idx]["lmdb_idx"]

        byteflow = self.txn.get(f"{real_idx:08d}".encode("ascii"))
        data = pickle.loads(byteflow)
        buffer = io.BytesIO(data["image"])
        image = Image.open(buffer).convert("RGB")

        raw_labels = data.get("labels", [])
        raw_boxes = data.get("boxes", [])

        # Chuẩn hóa nhãn về vector đếm số lượng từng class (10,)
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(raw_labels) > 0:
            lbl_tensor = torch.tensor(raw_labels, dtype=torch.int64)
            for c in range(self.num_classes):
                counts[c] = (lbl_tensor == c).sum().float()

        if self.return_dict:
            target = {
                "counts": counts,
                "boxes": (
                    torch.tensor(raw_boxes, dtype=torch.float32)
                    if len(raw_boxes) > 0
                    else torch.empty((0, 4), dtype=torch.float32)
                ),
                "labels": (
                    torch.tensor(raw_labels, dtype=torch.int64)
                    if len(raw_labels) > 0
                    else torch.empty((0,), dtype=torch.int64)
                ),
            }
        else:
            # Mặc định trả về counts (10,) để SmartCollate gom thành Tensor (B, 10)
            target = counts

        return {"image": image, "target": target}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"], state["txn"] = None, None
        return state
