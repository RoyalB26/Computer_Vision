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
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset, open_lmdb


@DATASETS.register_module()
class ACTION_SLOT_LMDBDataset(BaseDataset):
    """LMDB-based Action Slot dataset supporting single or multiple data_root paths.

    Matches the temporal strided frame sampling of the original TACO dataset:
      - Computes `step = num_frame // sequence_len` across the full scenario duration.
      - Builds up to `max_num = 50` strided 16-frame clips per scenario.
      - During `train`: randomly samples one of the strided clips (`random.randint`).
      - During `val`: selects the middle strided clip (`len(videos) // 2`).

    Args:
        data_root: Single directory path (str) or list of directory paths (List[str]).
        data_roots: Optional alias for data_root when configuring multiple roots.
        pipeline: Data transform pipeline.
        split: 'train' or 'val'.
        ann_file: Optional path to custom annotation json file.
        seed: Random seed for sample indexing.
        sequence_len: Number of video frames in clip.
        mask_every_frame: Frequency of mask extraction.
        min_frames: Minimum number of frames required for a valid scenario (default 50).
        use_obj_mask: Whether to load object instance masks from LMDB (default False).
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
        min_frames: int = 50,
        use_obj_mask: bool = False,
        num_samples: Optional[int] = None,
        test_mode: bool = False,
        **kwargs,
    ):
        self.split = split.lower()
        self.seed = seed
        self.sequence_len = sequence_len
        self.mask_every_frame = mask_every_frame
        self.min_frames = min_frames
        self.use_obj_mask = use_obj_mask
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

    def _get_lmdb_env(self, path: str):
        """Lazily initialize and cache LMDB environment for a path."""
        if path not in self.env_cache or self.env_cache[path] is None:
            self.env_cache[path], _ = open_lmdb(path, readonly=True)
        return self.env_cache[path]

    @staticmethod
    def _get_first_last_frame(env, prefix_path: str) -> Optional[Tuple[int, int]]:
        """Find (start_frame, end_frame) for a scenario prefix in O(log N) via LMDB B-tree seeks."""
        p = prefix_path.replace("\\", "/").rstrip("/") + "/"
        p_end = p[:-1] + "0"  # ASCII '0' (0x30) immediately follows '/' (0x2f)
        with env.begin() as txn:
            cursor = txn.cursor()
            if not cursor.set_range(p.encode("utf-8")):
                return None
            first_key = cursor.key().decode("utf-8")
            if not first_key.startswith(p):
                return None

            if cursor.set_range(p_end.encode("utf-8")):
                if not cursor.prev():
                    return None
            else:
                if not cursor.last():
                    return None
            last_key = cursor.key().decode("utf-8")
            if not last_key.startswith(p):
                return None

        try:
            start_frame = int(first_key.split("/")[-1].split(".")[0])
            end_frame = int(last_key.split("/")[-1].split(".")[0])
            return start_frame, end_frame
        except (ValueError, IndexError):
            return None

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Scan annotation entries, resolve LMDB paths, and build strided frame sequences."""
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
        max_required_seg_idx = ((self.sequence_len - 1) // self.mask_every_frame) * self.mask_every_frame

        for k, v in data.items():
            split_path = k.split("/")
            folder_name = split_path[0] + "_lmdb"
            raw_folder = split_path[0]

            rgb_prefix = os.path.join(split_path[1], "variant_scenario", split_path[2], "rgb", "front")
            bg_prefix = os.path.join(split_path[1], "variant_scenario", split_path[2], "mask", "background")
            obj_prefix = os.path.join(split_path[1], "variant_scenario", split_path[2], "mask", "object")

            clean_rgb_prefix = rgb_prefix.replace("\\", "/").rstrip("/")
            clean_bg_prefix = bg_prefix.replace("\\", "/").rstrip("/")

            matched_rgb = None
            matched_bg = None
            matched_obj = None
            rgb_range = None
            bg_range = None

            # Search across all specified data roots
            for root in self.data_roots:
                candidates = [
                    (
                        os.path.join(root, folder_name, "rgb_front"),
                        os.path.join(root, folder_name, "mask_background"),
                        os.path.join(root, folder_name, "mask_object"),
                    ),
                    (
                        os.path.join(root, raw_folder, "rgb_front"),
                        os.path.join(root, raw_folder, "mask_background"),
                        os.path.join(root, raw_folder, "mask_object"),
                    ),
                ]
                if os.path.basename(os.path.normpath(root)) in (folder_name, raw_folder):
                    candidates.insert(
                        1,
                        (
                            os.path.join(root, "rgb_front"),
                            os.path.join(root, "mask_background"),
                            os.path.join(root, "mask_object"),
                        ),
                    )

                for c_rgb, c_bg, c_obj in candidates:
                    if os.path.exists(c_rgb) and os.path.exists(c_bg):
                        rgb_env = self._get_lmdb_env(c_rgb)
                        r_range = self._get_first_last_frame(rgb_env, clean_rgb_prefix)
                        if r_range is not None:
                            bg_env = self._get_lmdb_env(c_bg)
                            b_range = self._get_first_last_frame(bg_env, clean_bg_prefix)
                            matched_rgb, matched_bg, matched_obj = c_rgb, c_bg, c_obj
                            rgb_range, bg_range = r_range, b_range
                            break
                if matched_rgb is not None:
                    break

            if matched_rgb is None or rgb_range is None:
                continue

            start_frame, end_frame = rgb_range
            num_frame = end_frame - start_frame + 1
            if num_frame < self.min_frames or num_frame < self.sequence_len:
                continue

            if bg_range is not None:
                bg_start_frame, bg_end_frame = bg_range
            else:
                bg_start_frame, bg_end_frame = start_frame, end_frame

            step = max(1, num_frame // self.sequence_len)
            videos: List[List[str]] = []
            segs: List[List[str]] = []
            max_num = 50

            for m in range(max_num):
                start = start_frame + m
                if start_frame + (self.sequence_len - 1) * step > end_frame:
                    break
                videos_temp: List[str] = []
                seg_temp: List[str] = []
                for i in range(start, end_frame + 1, step):
                    if start_frame <= i <= end_frame:
                        videos_temp.append(f"{clean_rgb_prefix}/{str(i).zfill(8)}.jpg")
                    if bg_start_frame <= i <= bg_end_frame:
                        seg_temp.append(f"{clean_bg_prefix}/{str(i).zfill(8)}.png")
                    if len(videos_temp) == self.sequence_len:
                        break
                if len(videos_temp) == self.sequence_len and len(seg_temp) > max_required_seg_idx:
                    videos.append(videos_temp)
                    segs.append(seg_temp)

            if not videos:
                continue

            self.videos_path.append({
                "scenario_key": k,
                "rgb_path": matched_rgb,
                "mask_bg_path": matched_bg,
                "mask_obj_path": matched_obj,
                "rgb_prefix": rgb_prefix,
                "bg_prefix": bg_prefix,
                "obj_prefix": obj_prefix,
                "videos": videos,
                "segs": segs,
                "label": v,
            })
            self.label[k] = v

        # Clear cached LMDB handles before DataLoader forks/spawns workers
        self.env_cache.clear()

        total = len(self.videos_path)
        indices = list(range(total))
        rng = random.Random(self.seed)
        rng.shuffle(indices)

        if self.num_samples:
            indices = indices[:self.num_samples]

        return [{"lmdb_idx": idx} for idx in indices]

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

    @staticmethod
    def _fetch_samples_by_keys(env, keys: List[str]) -> List[Any]:
        """Fetch and unpickle exact keys from LMDB in a single read transaction."""
        results = []
        with env.begin() as txn:
            for k in keys:
                val = txn.get(k.encode("utf-8"))
                if val is not None:
                    results.append(pickle.loads(val))
        return results

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Fetch strided clip frames and ground-truth targets."""
        real_idx = self.data_infos[idx]["lmdb_idx"]
        path = self.videos_path[real_idx]
        label = path.get("label") or self.label[path["scenario_key"]]

        rgb_env = self._get_lmdb_env(path["rgb_path"])
        bg_env = self._get_lmdb_env(path["mask_bg_path"])

        videos_clips = path.get("videos", [])
        segs_clips = path.get("segs", [])

        if videos_clips:
            if self.split == "train":
                sample_idx = random.randint(0, len(videos_clips) - 1)
            else:
                sample_idx = len(videos_clips) // 2

            seq_video_keys = videos_clips[sample_idx]
            seq_seg_all_keys = segs_clips[sample_idx]
            seq_seg_keys = [
                seq_seg_all_keys[i]
                for i in range(0, self.sequence_len, self.mask_every_frame)
                if i < len(seq_seg_all_keys)
            ]

            videos = self._fetch_samples_by_keys(rgb_env, seq_video_keys)
            bg_masks = self._fetch_samples_by_keys(bg_env, seq_seg_keys)
        else:
            videos = self.get_first_n_images_by_prefix(rgb_env, path["rgb_prefix"], num_images=self.sequence_len)
            raw_bg = self.get_first_n_images_by_prefix(bg_env, path["bg_prefix"], num_images=self.sequence_len)
            bg_masks = [s for i, s in enumerate(raw_bg) if i % self.mask_every_frame == 0]

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
            for sample in bg_masks:
                img = Image.open(io.BytesIO(sample["image"])).convert("L")
                target["bg_seg"].append(img)

        if self.use_obj_mask and path.get("mask_obj_path"):
            obj_env = self._get_lmdb_env(path["mask_obj_path"])
            obj_masks = self.get_first_n_images_by_prefix(obj_env, path["obj_prefix"], num_images=self.sequence_len)
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
