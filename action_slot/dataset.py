import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import io
import json
import pickle
import random
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch

from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset, open_lmdb


@DATASETS.register_module()
class ACTION_SLOT_LMDBDataset(BaseDataset):
    """LMDB-based Action Slot dataset supporting single or multiple data_root paths.

    Args:
        data_root: Single directory path (str) or list of directory paths (List[str]).
        data_roots: Optional alias for data_root when configuring multiple roots.
        pipeline: Data transform pipeline.
        split: 'train' or 'val'.
        ann_file: Optional path to custom annotation json file.
        seed: Random seed for sample indexing.
        sequence_len: Number of video frames in clip.
        mask_every_frame: Frequency of mask extraction.
        num_samples: Optional limit on total samples loaded.
        test_mode: Whether running in evaluation mode.
    """

    def __init__(
        self,
        data_root: Optional[Union[str, List[str]]] = None,
        data_roots: Optional[Union[str, List[str]]] = None,
        pipeline=None,
        split: str = "train",
        ann_file: Optional[str] = None,
        seed: int = 42,
        sequence_len: int = 16,
        mask_every_frame: int = 4,
        num_samples: Optional[int] = None,
        test_mode: bool = False,
        **kwargs,
    ):
        self.split = split.lower()
        self.seed = seed
        self.sequence_len = sequence_len
        self.mask_every_frame = mask_every_frame
        self.num_samples = num_samples

        # 1. Normalize data_root(s) into a unified list of paths
        raw_roots = data_roots if data_roots is not None else data_root
        if raw_roots is None:
            self.data_roots: List[str] = []
        elif isinstance(raw_roots, (list, tuple)):
            self.data_roots = [str(r) for r in raw_roots]
        else:
            self.data_roots = [str(raw_roots)]

        # Keep primary data_root string for BaseDataset compatibility
        primary_root = self.data_roots[0] if self.data_roots else ""
        self.data_root = primary_root

        # Internal stores and LMDB environment cache
        self.label: Dict[str, Any] = {}
        self.videos_path: List[Dict[str, Any]] = []
        self.env_cache: Dict[str, Any] = {}

        super().__init__(
            data_root=primary_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
        )

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Scan annotation entries and resolve file paths across all configured data_roots."""
        # Determine annotation file path
        if getattr(self, "ann_file", None) and os.path.exists(self.ann_file):
            data_path = self.ann_file
        elif self.split == "train":
            data_path = "action_slot/taco_train_label.json"
        else:
            data_path = "action_slot/taco_val_label.json"

        # Resolve relative to root_dir if needed
        if not os.path.isabs(data_path) and not os.path.exists(data_path):
            alt_path = os.path.join(root_dir, data_path)
            if os.path.exists(alt_path):
                data_path = alt_path

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Annotation file not found at: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.videos_path = []
        self.label = {}
        matched_roots = set()

        for k, v in data.items():
            split_path = k.split("/")
            folder_name = split_path[0] + "_lmdb"
            raw_folder = split_path[0]

            rgb_prefix = os.path.join(split_path[1], "variant_scenario", split_path[2], "rgb", "front")
            bg_prefix = os.path.join(split_path[1], "variant_scenario", split_path[2], "mask", "background")
            obj_prefix = os.path.join(split_path[1], "variant_scenario", split_path[2], "mask", "object")

            matched_rgb = None
            matched_bg = None
            matched_obj = None

            # Search across all specified data roots
            for root in self.data_roots:
                # 1. Standard structure: root / <folder_name> / rgb_front
                c_rgb = os.path.join(root, folder_name, "rgb_front")
                c_bg = os.path.join(root, folder_name, "mask_background")
                c_obj = os.path.join(root, folder_name, "mask_object")
                if os.path.exists(c_rgb):
                    matched_rgb, matched_bg, matched_obj = c_rgb, c_bg, c_obj
                    matched_roots.add(root)
                    break

                # 2. Root is directly the LMDB folder (e.g. root = ".../ap_Town07_lmdb")
                if os.path.basename(os.path.normpath(root)) in (folder_name, raw_folder):
                    c_rgb = os.path.join(root, "rgb_front")
                    c_bg = os.path.join(root, "mask_background")
                    c_obj = os.path.join(root, "mask_object")
                    if os.path.exists(c_rgb):
                        matched_rgb, matched_bg, matched_obj = c_rgb, c_bg, c_obj
                        matched_roots.add(root)
                        break

                # 3. Structure without _lmdb suffix: root / <raw_folder> / rgb_front
                c_rgb = os.path.join(root, raw_folder, "rgb_front")
                c_bg = os.path.join(root, raw_folder, "mask_background")
                c_obj = os.path.join(root, raw_folder, "mask_object")
                if os.path.exists(c_rgb):
                    matched_rgb, matched_bg, matched_obj = c_rgb, c_bg, c_obj
                    matched_roots.add(root)
                    break

            if matched_rgb is not None:
                self.videos_path.append({
                    "rgb_path": matched_rgb,
                    "mask_bg_path": matched_bg,
                    "mask_obj_path": matched_obj,
                    "rgb_prefix": rgb_prefix,
                    "bg_prefix": bg_prefix,
                    "obj_prefix": obj_prefix,
                })
                self.label[rgb_prefix] = v

        total = len(self.videos_path)
        indices = list(range(total))
        rng = random.Random(self.seed)
        rng.shuffle(indices)

        if self.num_samples:
            indices = indices[:self.num_samples]

        return [{"lmdb_idx": idx} for idx in indices]

    def _get_lmdb_env(self, path: str):
        """Lazily initialize and cache LMDB environment for a path."""
        if path not in self.env_cache or self.env_cache[path] is None:
            self.env_cache[path], _ = open_lmdb(path, readonly=True)
        return self.env_cache[path]

    def get_first_n_images_by_prefix(self, env, prefix_path: str, num_images: int = 16):
        """Retrieve up to num_images from LMDB matching prefix."""
        clean_prefix = prefix_path.replace("\\", "/")
        results = []

        with env.begin() as txn:
            cursor = txn.cursor()
            key_bytes = clean_prefix.encode("utf-8")
            if cursor.set_range(key_bytes):
                for _ in range(num_images):
                    key, val = cursor.item()
                    key_str = key.decode("utf-8")

                    if clean_prefix in key_str:
                        sample = pickle.loads(val)
                        results.append(sample)
                    else:
                        break

                    if not cursor.next():
                        break

        return results

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Fetch sample data and ground-truth targets."""
        real_idx = self.data_infos[idx]["lmdb_idx"]
        path = self.videos_path[real_idx]
        label = self.label[path["rgb_prefix"]]

        rgb_env = self._get_lmdb_env(path["rgb_path"])
        bg_env = self._get_lmdb_env(path["mask_bg_path"])
        obj_env = self._get_lmdb_env(path["mask_obj_path"])

        videos = self.get_first_n_images_by_prefix(rgb_env, path["rgb_prefix"], num_images=self.sequence_len)
        bg_masks = self.get_first_n_images_by_prefix(bg_env, path["bg_prefix"], num_images=self.sequence_len)
        obj_masks = self.get_first_n_images_by_prefix(obj_env, path["obj_prefix"], num_images=self.sequence_len)

        target: Dict[str, Any] = {
            "bg_seg": [],
            "obj_masks": [],
            "raw": [],
            "ego": torch.tensor(label["ego"]),
            "actor": torch.FloatTensor(label["agents"]),
            "path": path,
        }
        data = []

        if videos:
            for sample in videos:
                img = Image.open(io.BytesIO(sample["image"])).convert("RGB")
                data.append(img)

        if bg_masks:
            for i, sample in enumerate(bg_masks):
                if i % self.mask_every_frame == 0:
                    img = Image.open(io.BytesIO(sample["image"])).convert("L")
                    target["bg_seg"].append(img)

        if obj_masks:
            for sample in obj_masks:
                mask = self.get_obj_mask(sample)
                target["obj_masks"].append(mask)

        return {"inputs": data, "targets": target}

    def get_obj_mask(self, obj_masks):
        """Process object segmentation mask and pad to 64 slots."""
        if hasattr(obj_masks, "shape") and obj_masks.shape[0] == 0:
            obj_masks = torch.zeros([64, 32, 96], dtype=torch.int32)
        else:
            obj_masks = torch.from_numpy(np.stack(obj_masks, 0))

        obj_masks = obj_masks.type(torch.int)
        pad_num = 64 - obj_masks.shape[0]
        if pad_num > 0:
            obj_masks = torch.cat((obj_masks, torch.zeros([pad_num, 32, 96], dtype=torch.int32)), dim=0)
        obj_masks = obj_masks.type(torch.float32)

        return obj_masks

    def __getstate__(self):
        """Clear unpicklable LMDB environments for DataLoader multiprocessing."""
        state = self.__dict__.copy()
        state["env_cache"] = {}
        return state

    def __setstate__(self, state):
        """Restore dataset state in worker processes."""
        self.__dict__.update(state)
        self.env_cache = {}


if __name__ == "__main__":
    data_roots = [r"D:/apps/Datasets/TACO/data"]
    ds = ACTION_SLOT_LMDBDataset(data_root=data_roots, split="val")
    print(f"Loaded {len(ds)} samples across roots: {data_roots}")
    if len(ds) > 0:
        sample = ds.get_data_info(0)
        print("Sample 0 path:", sample["targets"]["path"]["rgb_path"])
        print("Num video frames:", len(sample["inputs"]))
