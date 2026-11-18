"""Dataset module for Action Depth (2.5D Perspective Toy Action LMDB)."""

import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pickle
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset
from BaseModel.dataset.helpers import open_lmdb, read_lmdb_bytes

from .generate_data import ALL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES, SYMBOLS, ACTIONS


def encode_label(label_str: str) -> torch.Tensor:
    target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    if not label_str:
        return target
    tokens = [tok.strip() for tok in label_str.split(",") if tok.strip()]
    for token in tokens:
        if token in CLASS_TO_IDX:
            target[CLASS_TO_IDX[token]] = 1.0
    return target


@DATASETS.register_module()
class ActionDepthLMDBDataset(BaseDataset):
    """LMDB Dataset for 2.5D Perspective Ground-Plane Actions."""

    def __init__(
        self,
        lmdb_path: Optional[str] = None,
        data_root: Optional[str] = None,
        ann_file: Optional[str] = None,
        pipeline=None,
        test_mode: bool = False,
    ):
        path = lmdb_path or ann_file or data_root
        if path is None:
            raise ValueError("Must specify lmdb_path.")
        self.lmdb_path = os.path.abspath(path)

        super().__init__(
            data_root=data_root,
            ann_file=self.lmdb_path,
            pipeline=pipeline,
            test_mode=test_mode,
        )

    def load_annotations(self) -> List[Dict[str, Any]]:
        env, txn = open_lmdb(self.lmdb_path, readonly=True)
        count_bytes = read_lmdb_bytes(txn, b"num-samples")
        if count_bytes is None:
            raise KeyError("num-samples not found")
        num_samples = int(count_bytes.decode("utf-8"))
        return [{"idx": i} for i in range(num_samples)]

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        env, txn = open_lmdb(self.lmdb_path, readonly=True)
        video_bytes = read_lmdb_bytes(txn, f"video-{idx:09d}")
        label_bytes = read_lmdb_bytes(txn, f"label-{idx:09d}")
        mask_bytes = read_lmdb_bytes(txn, f"bg_mask-{idx:09d}")
        coords_bytes = read_lmdb_bytes(txn, f"oxz_coords-{idx:09d}")
        bev_bytes = read_lmdb_bytes(txn, f"oxz_bev-{idx:09d}")

        label_str = label_bytes.decode("utf-8") if label_bytes is not None else ""
        encoded_frames = pickle.loads(video_bytes)
        frames = [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in encoded_frames]
        # (16, 224, 224, 3) -> (3, 16, 224, 224) (C, T, H, W)
        video_tensor = torch.from_numpy(np.stack(frames, axis=0)).float().permute(3, 0, 1, 2) / 255.0

        target_dict: Dict[str, Any] = {
            "actor": encode_label(label_str),
        }

        if mask_bytes is not None:
            encoded_masks = pickle.loads(mask_bytes)
            masks = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in encoded_masks]
            mask_t = torch.from_numpy(np.stack(masks, axis=0)).float() / 255.0  # (16, 224, 224)
            target_dict["bg_mask"] = mask_t.unsqueeze(0)  # (1, 16, 224, 224)

        if coords_bytes is not None:
            target_dict["oxz_coords"] = torch.from_numpy(pickle.loads(coords_bytes)).float()  # (16, 6, 2)

        if bev_bytes is not None:
            target_dict["oxz_bev"] = torch.from_numpy(pickle.loads(bev_bytes)).float()  # (16, 6, 32, 32)

        return {
            "idx": idx,
            "image": video_tensor,
            "target": target_dict,
            "label_str": label_str,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = BaseDataset.collate_fn(batch)
        return collated
