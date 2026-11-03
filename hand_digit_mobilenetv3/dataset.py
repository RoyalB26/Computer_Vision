import io
import pickle
import random
import torch
from PIL import Image
from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset, open_lmdb


@DATASETS.register_module(name="Hand_Digit_LMDBDataset")
class Hand_Digit_LMDBDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        pipeline=None,
        split="train",        # 'train' hoặc 'val'
        split_ratio=0.8,      # 80% train, 20% val
        seed=42,              # Seed đảm bảo train và val không bị lẫn
        num_classes=10,       # Số lượng classes digits
        task="detection",     # "detection" (boxes + labels)
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
        env, txn = open_lmdb(self.data_root, readonly=True)
        length_bytes = txn.get(b"num-samples")
        total = 3000

        indices = list(range(total))
        rng = random.Random(self.seed)
        rng.shuffle(indices)

        split_point = int(total * self.split_ratio)
        if self.split == "train":
            selected_indices = indices[:split_point]
        else:
            selected_indices = indices[split_point:]

        return [{"lmdb_idx": idx} for idx in selected_indices]

    def _init_db(self):
        if self.env is None:
            self.env, self.txn = open_lmdb(self.data_root, readonly=True)

    def get_data_info(self, idx: int):
        self._init_db()
        real_idx = self.data_infos[idx]["lmdb_idx"]

        byteflow = self.txn.get(f"{real_idx:08d}".encode("ascii"))
        data = pickle.loads(byteflow)
        image = Image.frombytes(mode='L', size=data['size'], data=data['image']).convert("RGB")

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
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for label in data.get("labels", []):
                if 0 <= label < self.num_classes:
                    target[label] = 1.0

        return {"image": image, "target": target}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"], state["txn"] = None, None
        return state
