"""VisDrone Dataset loaders: LMDB format (inherited from Visdrone) and standard directory formats."""
import io
import os
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch

from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset, open_lmdb


# VisDrone 10 standard categories
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


@DATASETS.register_module(name="VisDroneLMDBDataset", force=True)
@DATASETS.register_module(name="CustomLMDBDataset", force=True)
class VisDroneLMDBDataset(BaseDataset):
    """LMDB-backed VisDrone Dataset, directly inheriting and refining the logic from Visdrone.

    Reads serialized pickle objects from an LMDB database containing images and bounding box annotations,
    supporting deterministic train/val splitting.

    Args:
        data_root: Path to the LMDB database folder or file.
        pipeline: Data augmentation pipeline (e.g. Resize, ToTensor).
        split: 'train' or 'val'.
        split_ratio: Ratio for training set (e.g. 0.8 for 80% train, 20% val).
        seed: Random seed ensuring disjoint train/val splits across runs.
        test_mode: Whether in evaluation mode.
    """

    def __init__(
        self,
        data_root: str,
        pipeline: Optional[Union[List[Dict[str, Any]], Callable]] = None,
        split: str = "train",
        split_ratio: float = 0.8,
        seed: int = 42,
        test_mode: bool = False,
        **kwargs,
    ):
        self.split = split.lower()
        self.split_ratio = split_ratio
        self.seed = seed
        self.env = None
        self.txn = None
        super().__init__(data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Scan LMDB total entries and generate split indices."""
        if not os.path.exists(self.data_root):
            # If path does not exist on disk, return empty list (prevents crash during import or testing)
            return []

        env, txn = open_lmdb(self.data_root, readonly=True)
        total_len_bytes = txn.get(b"__len__")
        if total_len_bytes is None:
            # Fallback if __len__ is not stored as ascii key
            total = env.stat()["entries"]
        else:
            total = int(total_len_bytes.decode("ascii"))

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
        """Thread/worker-safe lazy initialization of the LMDB connection."""
        if self.env is None:
            self.env, self.txn = open_lmdb(self.data_root, readonly=True)

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Fetch a single sample from the LMDB database."""
        self._init_db()
        real_idx = self.data_infos[idx]["lmdb_idx"]

        byteflow = self.txn.get(f"{real_idx:08d}".encode("ascii"))
        data = pickle.loads(byteflow)

        buffer = io.BytesIO(data["image"])
        image = Image.open(buffer).convert("RGB")

        boxes = data.get("boxes", [])
        labels = data.get("labels", [])

        target = {
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if len(boxes) > 0
                else torch.empty((0, 4), dtype=torch.float32)
            ),
            "labels": (
                torch.tensor(labels, dtype=torch.int64)
                if len(labels) > 0
                else torch.empty((0,), dtype=torch.int64)
            ),
        }
        return {"image": image, "target": target}

    def __getstate__(self):
        """Enable safe pickle serialization across multiprocessing DataLoader workers."""
        state = self.__dict__.copy()
        state["env"], state["txn"] = None, None
        return state


# Backward-compatible alias matching Visdrone/dataset.py
CustomLMDBDataset = VisDroneLMDBDataset


@DATASETS.register_module(name="VisDroneFolderDataset", force=True)
@DATASETS.register_module(name="VisDroneDataset", force=True)
class VisDroneFolderDataset(BaseDataset):
    """File-system VisDrone Dataset reading images and txt annotation files.

    Supports standard VisDrone structure:
        data_root/
        ├── images/
        │   ├── 0000001_00001_d_0000001.jpg
        │   └── ...
        └── annotations/
            ├── 0000001_00001_d_0000001.txt
            └── ...

    VisDrone annotation format per line:
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    where object_category:
        1: pedestrian, 2: people, 3: bicycle, 4: car, 5: van,
        6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor.
        (Categories 0 and 11 are ignored).
    """

    # VisDrone 1-indexed category to 0-indexed category mapping
    CATEGORY_MAPPING = {
        1: 0,   # pedestrian
        2: 1,   # people
        3: 2,   # bicycle
        4: 3,   # car
        5: 4,   # van
        6: 5,   # truck
        7: 6,   # tricycle
        8: 7,   # awning-tricycle
        9: 8,   # bus
        10: 9,  # motorcycle
    }

    def __init__(
        self,
        data_root: str,
        pipeline: Optional[Union[List[Dict[str, Any]], Callable]] = None,
        img_folder: str = "images",
        ann_folder: str = "annotations",
        normalize_boxes: bool = True,
        test_mode: bool = False,
        **kwargs,
    ):
        self.img_folder = os.path.join(data_root, img_folder)
        self.ann_folder = os.path.join(data_root, ann_folder)
        self.normalize_boxes = normalize_boxes
        super().__init__(data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Collect all image paths and their corresponding annotation files."""
        if not os.path.isdir(self.img_folder):
            return []

        data_infos = []
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        for fname in sorted(os.listdir(self.img_folder)):
            if fname.lower().endswith(valid_exts):
                img_path = os.path.join(self.img_folder, fname)
                base_name = os.path.splitext(fname)[0]
                ann_path = os.path.join(self.ann_folder, f"{base_name}.txt")
                data_infos.append({
                    "img_path": img_path,
                    "ann_path": ann_path if os.path.exists(ann_path) else None,
                })
        return data_infos

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        info = self.data_infos[idx]
        image = Image.open(info["img_path"]).convert("RGB")
        w_img, h_img = image.size

        boxes = []
        labels = []

        if info["ann_path"] and os.path.isfile(info["ann_path"]):
            with open(info["ann_path"], "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue
                    try:
                        x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                        score = float(parts[4])
                        cat = int(parts[5])

                        # In VisDrone, score==0 indicates ignored region or evaluate==0
                        if score == 0 or cat not in self.CATEGORY_MAPPING:
                            continue
                        if w <= 0 or h <= 0:
                            continue

                        mapped_cat = self.CATEGORY_MAPPING[cat]

                        if self.normalize_boxes:
                            # Normalized [x1, y1, w, h] in [0, 1]
                            boxes.append([x / w_img, y / h_img, w / w_img, h / h_img])
                        else:
                            boxes.append([x, y, w, h])

                        labels.append(mapped_cat)
                    except ValueError:
                        continue

        target = {
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if len(boxes) > 0
                else torch.empty((0, 4), dtype=torch.float32)
            ),
            "labels": (
                torch.tensor(labels, dtype=torch.int64)
                if len(labels) > 0
                else torch.empty((0,), dtype=torch.int64)
            ),
        }
        return {"image": image, "target": target}
