import io
import pickle
import random
import torch
from PIL import Image
from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset, open_lmdb


@DATASETS.register_module()
class COCOLMDBDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        pipeline=None,
        split="train",        # 'train' hoặc 'val'
        split_ratio=0.8,      # 80% train, 20% val
        seed=42,              # Seed đảm bảo train và val không bị lẫn
        num_classes=80,       # Số lượng classes của COCO
        task="classification", # "classification" (multi-label) hoặc "detection" (boxes + labels)
        test_mode=False,
        **kwargs,
    ):
        self.split = split.lower()
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_classes = num_classes
        self.task = task.lower()
        self.env = None
        self.txn = None
        super().__init__(data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    def load_annotations(self):
        # 1. Đọc tổng số mẫu trong file LMDB
        env, txn = open_lmdb(self.data_root, readonly=True)
        length_bytes = txn.get(b"num-samples")
        total = int(length_bytes.decode()) if length_bytes else txn.stat()["entries"] // 2

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

    def get_data_info(self, idx: int):
        self._init_db()
        # Lấy chỉ số gốc trong LMDB đã được chia
        real_idx = self.data_infos[idx]["lmdb_idx"]

        byteflow = self.txn.get(f"{real_idx:08d}".encode("ascii"))
        data = pickle.loads(byteflow)
        buffer = io.BytesIO(data["image"])
        image = Image.open(buffer).convert("RGB")

        if self.task == "detection":
            boxes = (
                torch.tensor(data["boxes"], dtype=torch.float32)
                if len(data.get("boxes", [])) > 0
                else torch.empty((0, 4), dtype=torch.float32)
            )
            labels = (
                torch.tensor(data["labels"], dtype=torch.int64)
                if len(data.get("labels", [])) > 0
                else torch.empty((0,), dtype=torch.int64)
            )
            target = {"boxes": boxes, "labels": labels}
        else:
            # Multi-label binary vector (one-hot / multi-hot) kích thước (C,)
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for label in data.get("labels", []):
                if 0 <= label < self.num_classes:
                    target[label] = 1.0

        return {"image": image, "target": target}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"], state["txn"] = None, None
        return state
