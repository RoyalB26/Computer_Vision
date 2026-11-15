"""Dataset module for Toy Action LMDB dataset.

Inherits from BaseModel's BaseDataset, integrating with BaseModel's Registry system,
smart collation, and data pipelines.
"""

import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


from collections import Counter
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset
from BaseModel.dataset.helpers import open_lmdb, read_lmdb_bytes

# Symbol and Action Vocabularies
SYMBOLS = ["x", "y", "o", "T", "8", "O"]
ACTIONS = ["gotop", "gobottom", "goleft", "goright"]

SYMBOL_VARIANTS = []
for s in SYMBOLS:
    SYMBOL_VARIANTS.append(s)
    SYMBOL_VARIANTS.append(f"{s}+")

ALL_CLASSES = [f"{s}:{a}" for s in SYMBOL_VARIANTS for a in ACTIONS]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)  # 48 classes


def encode_label(label_str: str) -> torch.Tensor:
    """Encode comma-separated label string into multi-hot binary tensor of shape (NUM_CLASSES,).

    Example:
        'x+:gotop, y:gobottom' -> tensor with 1.0 at corresponding indices, 0.0 elsewhere.
    """
    target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    if not label_str:
        return target
    tokens = [tok.strip() for tok in label_str.split(",") if tok.strip()]
    for token in tokens:
        if token in CLASS_TO_IDX:
            target[CLASS_TO_IDX[token]] = 1.0
    return target


def decode_target(target: Union[torch.Tensor, np.ndarray], threshold: float = 0.5) -> List[str]:
    """Decode a multi-hot tensor or probability vector into a list of label token strings."""
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    indices = np.where(target >= threshold)[0]
    return [IDX_TO_CLASS[idx] for idx in indices if idx in IDX_TO_CLASS]


@DATASETS.register_module()
class ToyActionLMDBDataset(BaseDataset):
    """LMDB-backed Dataset for Toy Action videos, compatible with BaseModel.

    Each sample contains:
        - 16 frames of size (224, 224, 3)
        - Multi-label target encoding of active (symbol, action) pairs.

    Args:
        lmdb_path: Path to the LMDB database folder.
        data_root: Optional root directory (can prefix lmdb_path if relative).
        ann_file: Optional alias for lmdb_path.
        channel_first: If True, returns video tensor of shape (3, 16, 224, 224) (C, T, H, W).
                       If False, returns (16, 3, 224, 224) (T, C, H, W).
        pipeline: Data transform pipeline (optional).
        test_mode: Whether running in evaluation/test mode.
    """

    ALL_CLASSES = ALL_CLASSES
    CLASS_TO_IDX = CLASS_TO_IDX
    IDX_TO_CLASS = IDX_TO_CLASS
    NUM_CLASSES = NUM_CLASSES

    def __init__(
        self,
        lmdb_path: Optional[str] = None,
        data_root: Optional[str] = None,
        ann_file: Optional[str] = None,
        channel_first: bool = True,
        pipeline: Optional[Union[List[Union[Dict[str, Any], Callable]], Any]] = None,
        test_mode: bool = False,
    ):
        # Resolve target path
        path = lmdb_path or ann_file or data_root
        if path is None:
            raise ValueError("Either 'lmdb_path', 'ann_file', or 'data_root' must be specified.")
        if data_root and not os.path.isabs(path):
            path = os.path.join(data_root, path)

        self.lmdb_path = os.path.abspath(path)
        self.channel_first = channel_first

        # Initialize BaseDataset (which triggers self.load_annotations())
        super().__init__(
            data_root=data_root,
            ann_file=self.lmdb_path,
            pipeline=pipeline,
            test_mode=test_mode,
        )

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Query total samples from LMDB and return list of index metadata."""
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB database directory does not exist: {self.lmdb_path}")

        env, txn = open_lmdb(self.lmdb_path, readonly=True)
        count_bytes = read_lmdb_bytes(txn, b"num-samples")
        if count_bytes is None:
            raise KeyError(f"'num-samples' key not found in LMDB at {self.lmdb_path}")

        num_samples = int(count_bytes.decode("utf-8"))
        return [{"idx": i} for i in range(num_samples)]

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Fetch video clip and ground-truth label from LMDB by index."""
        env, txn = open_lmdb(self.lmdb_path, readonly=True)

        video_key = f"video-{idx:09d}"
        label_key = f"label-{idx:09d}"

        video_bytes = read_lmdb_bytes(txn, video_key)
        label_bytes = read_lmdb_bytes(txn, label_key)

        if video_bytes is None:
            raise KeyError(f"Key '{video_key}' not found in LMDB.")

        label_str = label_bytes.decode("utf-8") if label_bytes is not None else ""

        # Decode 16 JPEG frames
        encoded_frames = pickle.loads(video_bytes)
        frames: List[np.ndarray] = []
        for b in encoded_frames:
            # Decode JPEG buffer to BGR
            bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)

        # Shape: (16, 224, 224, 3), uint8
        video_np = np.stack(frames, axis=0)

        # Convert to float32 tensor normalized in [0, 1]
        video_tensor = torch.from_numpy(video_np).float() / 255.0

        if self.channel_first:
            # (16, 224, 224, 3) -> (3, 16, 224, 224) (C, T, H, W)
            video_tensor = video_tensor.permute(3, 0, 1, 2).contiguous()
        else:
            # (16, 224, 224, 3) -> (16, 3, 224, 224) (T, C, H, W)
            video_tensor = video_tensor.permute(0, 3, 1, 2).contiguous()

        # Multi-hot target tensor (NUM_CLASSES,)
        target_tensor = encode_label(label_str)

        data_info = {
            "idx": idx,
            "image": video_tensor,  # Standard key expected by BaseModel trainer
            "target": target_tensor,  # Standard target expected by BaseModel trainer
            "label_str": label_str,
        }

        # Check and decode bg_mask if stored in LMDB
        mask_key = f"bg_mask-{idx:09d}"
        mask_bytes = read_lmdb_bytes(txn, mask_key)
        if mask_bytes is not None:
            encoded_masks = pickle.loads(mask_bytes)
            masks = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in encoded_masks]
            masks_np = np.stack(masks, axis=0)  # (16, 224, 224), uint8
            mask_t = torch.from_numpy(masks_np).float() / 255.0  # 1.0: background, 0.0: object
            if self.channel_first:
                # (1, 16, 224, 224) (C, T, H, W)
                data_info["bg_mask"] = mask_t.unsqueeze(0).contiguous()
            else:
                # (16, 1, 224, 224) (T, C, H, W)
                data_info["bg_mask"] = mask_t.unsqueeze(1).contiguous()

        return data_info

if __name__ == "__main__":
    toy_dataset= ToyActionLMDBDataset("toy_action/data/train_lmdb")
    data = toy_dataset.get_data_info(0)
    print(data['image'].shape)
    print(data['target'].shape)
    print(data['bg_mask'].shape)