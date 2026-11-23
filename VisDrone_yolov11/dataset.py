"""VisDrone Dataset loaders: LMDB format (inherited from Visdrone) and standard directory formats."""
import io
import os
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
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
        num_classes: int = 10,
        bg_lmdb_path: Optional[str] = None,
        include_boxes: bool = True,
        cache_images: Optional[str] = None,   # None | "ram" (shared uint8 tensor) | "jpeg" (compressed bytes)
        cache_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.split = split.lower()
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_classes = num_classes
        self.bg_lmdb_path = bg_lmdb_path
        self.include_boxes = include_boxes
        self.cache_images = cache_images.lower() if isinstance(cache_images, str) else None
        self.cache_size = tuple(cache_size) if cache_size is not None else self._infer_resize_from_pipeline(pipeline)
        self.env = None
        self.txn = None
        self.bg_env = None
        self.bg_txn = None
        self._bg_num_tokens = 18900
        self._bg_format = "packbits_uint8"
        self._ram_images: Optional[torch.Tensor] = None
        self._ram_targets: Optional[List[Dict[str, Any]]] = None
        self._jpeg_cache: Optional[List[Tuple[bytes, Dict[str, Any]]]] = None
        super().__init__(data_root=data_root, pipeline=pipeline, test_mode=test_mode)

        if self.cache_images in ("ram", "jpeg") and len(self.data_infos) > 0:
            self._build_ram_cache()

    @staticmethod
    def _infer_resize_from_pipeline(pipeline: Optional[Any]) -> Tuple[int, int]:
        if isinstance(pipeline, list):
            for step in pipeline:
                if isinstance(step, dict) and step.get("type") == "Resize" and "size" in step:
                    sz = step["size"]
                    return (int(sz[0]), int(sz[1]))
        return (960, 960)

    def _build_ram_cache(self) -> None:
        """Pre-load all samples into shared RAM once to eliminate Disk/LMDB + Resize overhead."""
        n = len(self.data_infos)
        self._init_db()

        if self.cache_images == "ram":
            h, w = self.cache_size
            gb = (n * 3 * h * w) / (1024 ** 3)
            print(f"[VisDroneLMDBDataset:{self.split}] Caching {n:,} resized images ({h}x{w} uint8) into shared RAM (~{gb:.2f} GB)...")
            buf = torch.empty((n, 3, h, w), dtype=torch.uint8).share_memory_()
            targets_cache: List[Dict[str, Any]] = []
            for idx in range(n):
                info = self._read_lmdb_sample(idx)
                img_resized = info["image"].resize((w, h), Image.BILINEAR)
                arr = np.array(img_resized, dtype=np.uint8, copy=True)  # (H, W, 3) writable
                buf[idx] = torch.from_numpy(arr).permute(2, 0, 1)
                # Store targets as NumPy arrays (0 FDs) to avoid Python 3.13 forkserver `too many fds`
                t = info["target"]
                t_np: Dict[str, Any] = {"_is_cxcywh": t.get("_is_cxcywh", True)}
                for k in ("counts", "boxes", "labels", "bg_mask"):
                    if k in t:
                        t_np[k] = t[k].numpy() if isinstance(t[k], torch.Tensor) else np.asarray(t[k])
                targets_cache.append(t_np)
            self._ram_images = buf
            self._ram_targets = targets_cache
            print(f"[VisDroneLMDBDataset:{self.split}] RAM cache ready ({gb:.2f} GB shared memory).")

        elif self.cache_images == "jpeg":
            print(f"[VisDroneLMDBDataset:{self.split}] Caching {n:,} compressed JPEG bytes into RAM...")
            jpeg_cache: List[Tuple[bytes, Dict[str, Any]]] = []
            total_bytes = 0
            for idx in range(n):
                real_idx = self.data_infos[idx]["lmdb_idx"]
                key = f"{real_idx:08d}".encode("ascii")
                data = pickle.loads(self.txn.get(key))
                img_bytes = data["image"]
                total_bytes += len(img_bytes)
                target = self._build_target(data, key)
                jpeg_cache.append((img_bytes, target))
            self._jpeg_cache = jpeg_cache
            print(f"[VisDroneLMDBDataset:{self.split}] JPEG RAM cache ready (~{total_bytes / (1024 ** 3):.2f} GB).")

    @staticmethod
    def _load_or_scan_class_counts(data_root: str, txn, total: int, num_classes: int) -> np.ndarray:
        """Đọc hoặc quét và cache số lượng box của từng class cho toàn bộ ảnh trong LMDB."""
        cache_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(cache_dir, ".lmdb_class_counts_cache.npz")

        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path)
                if cached["counts"].shape == (total, num_classes):
                    return cached["counts"]
            except Exception:
                pass

        all_counts = np.zeros((total, num_classes), dtype=np.float64)
        for i in range(total):
            bf = txn.get(f"{i:08d}".encode("ascii"))
            if bf is None:
                continue
            d = pickle.loads(bf)
            raw_labels = d.get("labels", [])
            if len(raw_labels) == 0:
                continue
            lbl_arr = np.asarray(raw_labels, dtype=np.int64)
            lbl_arr = lbl_arr[(lbl_arr >= 0) & (lbl_arr < num_classes)]
            if len(lbl_arr) > 0:
                all_counts[i] = np.bincount(lbl_arr, minlength=num_classes)

        try:
            np.savez_compressed(cache_path, counts=all_counts)
        except Exception:
            pass
        return all_counts

    @staticmethod
    def _stratified_train_val_split(
        all_counts: np.ndarray,
        split_ratio: float,
        seed: int,
    ) -> Tuple[List[int], List[int]]:
        """Multi-label Iterative Stratification + Local Swap Refinement (đồng bộ 100% với Visdrone/dataset.py)."""
        total, num_classes = all_counts.shape
        n_train = int(total * split_ratio)
        n_val = total - n_train

        rng = np.random.default_rng(seed)

        val_ratio = 1.0 - split_ratio
        global_box_sum = all_counts.sum(axis=0)
        target_val_boxes = global_box_sum * val_ratio
        target_val_pres = (all_counts > 0).sum(axis=0) * val_ratio

        class_rarity_order = np.argsort(global_box_sum + 1e-6)
        assigned = np.zeros(total, dtype=bool)
        is_val = np.zeros(total, dtype=bool)

        cur_val_boxes = np.zeros(num_classes, dtype=np.float64)
        cur_val_pres = np.zeros(num_classes, dtype=np.float64)
        cur_val_count = 0

        inv_box_target = 1.0 / np.maximum(target_val_boxes, 1.0)
        inv_pres_target = 1.0 / np.maximum(target_val_pres, 1.0)

        for c in class_rarity_order:
            cand_indices = np.flatnonzero((~assigned) & (all_counts[:, c] > 0))
            if len(cand_indices) == 0:
                continue

            perm = rng.permutation(len(cand_indices))
            cand_indices = cand_indices[perm]

            for idx in cand_indices:
                assigned[idx] = True
                rem_total = total - int(assigned.sum())
                rem_val_needed = n_val - cur_val_count

                if rem_val_needed <= 0:
                    is_val[idx] = False
                    continue
                if rem_val_needed > rem_total:
                    is_val[idx] = True
                    cur_val_count += 1
                    cur_val_boxes += all_counts[idx]
                    cur_val_pres += (all_counts[idx] > 0).astype(np.float64)
                    continue

                img_boxes = all_counts[idx]
                img_pres = (img_boxes > 0).astype(np.float64)

                c_box_fill = cur_val_boxes[c] * inv_box_target[c]
                c_pres_fill = cur_val_pres[c] * inv_pres_target[c]
                c_fill_ratio = 0.75 * c_box_fill + 0.25 * c_pres_fill

                active_m = img_boxes > 0
                co_fill_ratio = (
                    0.75 * (cur_val_boxes[active_m] * inv_box_target[active_m]).mean()
                    + 0.25 * (cur_val_pres[active_m] * inv_pres_target[active_m]).mean()
                )

                if c_fill_ratio < 1.0 and co_fill_ratio < 1.08:
                    is_val[idx] = True
                    cur_val_count += 1
                    cur_val_boxes += img_boxes
                    cur_val_pres += img_pres
                else:
                    is_val[idx] = False

        unassigned_indices = np.flatnonzero(~assigned)
        if len(unassigned_indices) > 0:
            rng.shuffle(unassigned_indices)
            rem_val_needed = max(0, n_val - cur_val_count)
            for i_u, idx in enumerate(unassigned_indices):
                if i_u < rem_val_needed:
                    is_val[idx] = True
                    cur_val_count += 1
                else:
                    is_val[idx] = False

        global_dist = global_box_sum / (global_box_sum.sum() + 1e-8)
        weights = 1.0 / (global_dist + 0.005)
        val_indices_arr = np.flatnonzero(is_val)
        train_indices_arr = np.flatnonzero(~is_val)

        cur_val_total = float(cur_val_boxes.sum()) + 1e-8
        cur_val_dist = cur_val_boxes / cur_val_total
        best_loss = float(((cur_val_dist - global_dist) ** 2 * weights).sum())

        for _ in range(400):
            if len(val_indices_arr) == 0 or len(train_indices_arr) == 0:
                break
            v_pos = int(rng.integers(0, len(val_indices_arr)))
            t_pos = int(rng.integers(0, len(train_indices_arr)))
            v_idx = val_indices_arr[v_pos]
            t_idx = train_indices_arr[t_pos]

            diff_boxes = all_counts[t_idx] - all_counts[v_idx]
            if not np.any(diff_boxes != 0):
                continue

            cand_val_boxes = cur_val_boxes + diff_boxes
            cand_total = float(cand_val_boxes.sum()) + 1e-8
            cand_dist = cand_val_boxes / cand_total
            cand_loss = float(((cand_dist - global_dist) ** 2 * weights).sum())

            if cand_loss < best_loss:
                best_loss = cand_loss
                cur_val_boxes = cand_val_boxes
                val_indices_arr[v_pos] = t_idx
                train_indices_arr[t_pos] = v_idx

        train_list = train_indices_arr.tolist()
        val_list = val_indices_arr.tolist()
        rng.shuffle(train_list)
        rng.shuffle(val_list)
        return train_list, val_list

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Scan LMDB total entries and generate stratified train/val split indices."""
        if not os.path.exists(self.data_root):
            return []

        env, txn = open_lmdb(self.data_root, readonly=True)
        total_len_bytes = txn.get(b"__len__")
        if total_len_bytes is None:
            total = env.stat()["entries"]
        else:
            total = int(total_len_bytes.decode("ascii"))

        all_counts = self._load_or_scan_class_counts(self.data_root, txn, total, self.num_classes)
        train_indices, val_indices = self._stratified_train_val_split(
            all_counts=all_counts,
            split_ratio=self.split_ratio,
            seed=self.seed,
        )

        if self.split == "train":
            selected_indices = train_indices
        else:
            selected_indices = val_indices

        return [{"lmdb_idx": idx} for idx in selected_indices]

    def _init_db(self):
        """Thread/worker-safe lazy initialization of the LMDB connection (and optional sidecar bg LMDB)."""
        if self.env is None:
            self.env, self.txn = open_lmdb(self.data_root, readonly=True)
        if self.bg_lmdb_path and os.path.exists(self.bg_lmdb_path) and self.bg_env is None:
            self.bg_env, self.bg_txn = open_lmdb(self.bg_lmdb_path, readonly=True)
            num_tok_bytes = self.bg_txn.get(b"__num_tokens__")
            if num_tok_bytes is not None:
                self._bg_num_tokens = int(num_tok_bytes.decode("ascii"))
            fmt_bytes = self.bg_txn.get(b"__format__")
            if fmt_bytes is not None:
                self._bg_format = fmt_bytes.decode("ascii")

    def _build_target(self, data: Dict[str, Any], key: bytes) -> Dict[str, Any]:
        raw_boxes = data.get("boxes", [])
        raw_labels = data.get("labels", [])

        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(raw_labels) > 0:
            lbl_tensor = torch.tensor(raw_labels, dtype=torch.int64)
            for c in range(self.num_classes):
                counts[c] = (lbl_tensor == c).sum().float()

        target: Dict[str, Any] = {
            "counts": counts,
            "_is_cxcywh": True,
        }

        if self.bg_txn is not None:
            bg_bytes = self.bg_txn.get(key)
            if bg_bytes is not None:
                if self._bg_format == "packbits_uint8":
                    unpacked = np.unpackbits(
                        np.frombuffer(bg_bytes, dtype=np.uint8)
                    )[: self._bg_num_tokens]
                    target["bg_mask"] = torch.from_numpy(unpacked.copy())
                else:
                    target["bg_mask"] = torch.from_numpy(
                        np.frombuffer(bg_bytes, dtype=np.uint8).copy()
                    )

        if self.include_boxes or "bg_mask" not in target:
            target["boxes"] = (
                torch.tensor(raw_boxes, dtype=torch.float32)
                if len(raw_boxes) > 0
                else torch.empty((0, 4), dtype=torch.float32)
            )
            target["labels"] = (
                torch.tensor(raw_labels, dtype=torch.int64)
                if len(raw_labels) > 0
                else torch.empty((0,), dtype=torch.int64)
            )
        return target

    def _read_lmdb_sample(self, idx: int) -> Dict[str, Any]:
        self._init_db()
        real_idx = self.data_infos[idx]["lmdb_idx"]
        key = f"{real_idx:08d}".encode("ascii")

        byteflow = self.txn.get(key)
        data = pickle.loads(byteflow)

        buffer = io.BytesIO(data["image"])
        image = Image.open(buffer).convert("RGB")
        target = self._build_target(data, key)
        return {"image": image, "target": target}

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Fetch a single sample from RAM cache (if enabled) or LMDB database."""
        if self._jpeg_cache is not None:
            img_bytes, target = self._jpeg_cache[idx]
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return {"image": image, "target": target}
        return self._read_lmdb_sample(idx)

    def prepare_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Fast path when resized uint8 images are cached in shared RAM."""
        if self._ram_images is not None and self._ram_targets is not None:
            img = self._ram_images[idx].to(dtype=torch.float32).div_(255.0)
            t_np = self._ram_targets[idx]
            target: Dict[str, Any] = {"_is_cxcywh": t_np.get("_is_cxcywh", True)}
            if "counts" in t_np:
                target["counts"] = torch.from_numpy(t_np["counts"])
            if "boxes" in t_np:
                target["boxes"] = torch.from_numpy(t_np["boxes"])
            if "labels" in t_np:
                target["labels"] = torch.from_numpy(t_np["labels"])
            if "bg_mask" in t_np:
                target["bg_mask"] = torch.from_numpy(t_np["bg_mask"])
            return {"image": img, "target": target}
        return super().prepare_data(idx)

    def __getstate__(self):
        """Enable safe pickle serialization across multiprocessing DataLoader workers."""
        state = self.__dict__.copy()
        state["env"], state["txn"] = None, None
        state["bg_env"], state["bg_txn"] = None, None
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
