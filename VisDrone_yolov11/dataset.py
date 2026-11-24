"""VisDrone Dataset loaders: LMDB format (inherited from Visdrone) and standard directory formats."""
import hashlib
import io
import os
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from BaseModel.core import DATASETS, HOOKS
from BaseModel.core.hooks.base_hook import BaseHook
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
        split: Optional 'train', 'val', or None. When None (or 'all'), loads 100% of data_root.
        split_ratio: Ratio for training set (e.g. 0.8 for 80% train, 20% val) when splitting a single LMDB.
        seed: Random seed ensuring disjoint train/val splits across runs.
        test_mode: Whether in evaluation mode.
    """

    def __init__(
        self,
        data_root: str,
        pipeline: Optional[Union[List[Dict[str, Any]], Callable]] = None,
        split: Optional[str] = None,
        split_ratio: Optional[float] = None,
        seed: int = 42,
        test_mode: bool = False,
        num_classes: int = 10,
        bg_lmdb_path: Optional[str] = None,
        include_boxes: bool = True,
        cache_images: Optional[str] = None,   # None | "ram" (shared uint8 tensor) | "jpeg" (compressed bytes)
        cache_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,                # Fast tensor augmentation (HFlip + ColorJitter + Scale/Crop)
        mosaic_prob: float = 0.0,             # 4-image full-resolution Mosaic probability (active when augment=True)
        **kwargs,
    ):
        self.split = split.lower() if isinstance(split, str) else None
        self.split_ratio = float(split_ratio) if split_ratio is not None else None
        self.seed = seed
        self.num_classes = num_classes
        self.bg_lmdb_path = bg_lmdb_path
        self.include_boxes = include_boxes
        self.cache_images = cache_images.lower() if isinstance(cache_images, str) else None
        self.cache_size = tuple(cache_size) if cache_size is not None else self._infer_resize_from_pipeline(pipeline)
        self.augment = bool(augment) and (self.split not in ("val", "test")) and (not test_mode)
        self.mosaic_prob = float(mosaic_prob) if self.augment else 0.0
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

        if self.augment:
            m_str = f" + mosaic4 (p={self.mosaic_prob:.2f})" if self.mosaic_prob > 0.0 else ""
            print(f"[VisDroneLMDBDataset:{self.split}] Data augmentation ENABLED (flip + scale/crop + color jitter{m_str}).")
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
        root_hash = hashlib.md5(os.path.abspath(data_root).encode("utf-8")).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f".lmdb_class_counts_cache_{root_hash}.npz")

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
        assigned_count = 0

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
                assigned_count += 1
                rem_total = total - assigned_count
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
        """Scan LMDB total entries and optionally apply stratified train/val split."""
        if not os.path.exists(self.data_root):
            return []

        env, txn = open_lmdb(self.data_root, readonly=True)
        total_len_bytes = txn.get(b"__len__")
        if total_len_bytes is None:
            total = env.stat()["entries"]
        else:
            total = int(total_len_bytes.decode("ascii"))

        should_split = (
            self.split in ("train", "val")
            and (self.split_ratio is None or 0.0 < self.split_ratio < 1.0)
        )
        if not should_split:
            return [{"lmdb_idx": idx} for idx in range(total)]

        ratio = 0.8 if self.split_ratio is None else self.split_ratio
        all_counts = self._load_or_scan_class_counts(self.data_root, txn, total, self.num_classes)
        train_indices, val_indices = self._stratified_train_val_split(
            all_counts=all_counts,
            split_ratio=ratio,
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

    def _apply_tensor_augment(
        self, img: torch.Tensor, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Fast GPU/CPU-friendly tensor augmentation compatible with `cache_images='ram'`.

        Applies:
          1. RandomHorizontalFlip (p=0.5)
          2. RandomScale + RandomCrop/Pad (p=0.5, scale in [0.8, 1.2])
          3. Random ColorJitter: brightness, contrast, saturation (p=0.8)
        """
        _, h, w = img.shape
        boxes = target.get("boxes", torch.empty((0, 4), dtype=torch.float32)).clone()
        labels = target.get("labels", torch.empty((0,), dtype=torch.int64)).clone()

        # 1. Random Horizontal Flip (p=0.5)
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            if len(boxes) > 0:
                boxes[:, 0] = (1.0 - boxes[:, 0]).clamp(0.0, 1.0)

        # 2. Random Scale + Crop/Pad (p=0.5)
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            nh, nw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
            img_s = F.interpolate(img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)

            if scale >= 1.0:
                # Random crop (h, w) from larger (nh, nw)
                y0 = random.randint(0, nh - h)
                x0 = random.randint(0, nw - w)
                img = img_s[:, y0 : y0 + h, x0 : x0 + w]
                if len(boxes) > 0:
                    cx = (boxes[:, 0] * nw - x0) / w
                    cy = (boxes[:, 1] * nh - y0) / h
                    bw = boxes[:, 2] * (nw / w)
                    bh = boxes[:, 3] * (nh / h)
                    x1 = (cx - bw / 2.0).clamp(0.0, 1.0)
                    y1 = (cy - bh / 2.0).clamp(0.0, 1.0)
                    x2 = (cx + bw / 2.0).clamp(0.0, 1.0)
                    y2 = (cy + bh / 2.0).clamp(0.0, 1.0)
                    nbw, nbh = x2 - x1, y2 - y1
                    keep = (nbw > (2.0 / w)) & (nbh > (2.0 / h))
                    boxes = torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, nbw, nbh], dim=-1)[keep]
                    labels = labels[keep]
            else:
                # Random pad (nh, nw) into canvas (h, w) with neutral gray (0.447)
                canvas = torch.full((3, h, w), 0.447, dtype=img.dtype)
                y0 = random.randint(0, h - nh)
                x0 = random.randint(0, w - nw)
                canvas[:, y0 : y0 + nh, x0 : x0 + nw] = img_s
                img = canvas
                if len(boxes) > 0:
                    cx = (boxes[:, 0] * nw + x0) / w
                    cy = (boxes[:, 1] * nh + y0) / h
                    bw = boxes[:, 2] * (nw / w)
                    bh = boxes[:, 3] * (nh / h)
                    boxes = torch.stack([cx, cy, bw, bh], dim=-1)

        # 3. Random Photometric Jitter (p=0.8)
        if random.random() < 0.8:
            b_fac = random.uniform(0.8, 1.2)
            c_fac = random.uniform(0.8, 1.2)
            s_fac = random.uniform(0.8, 1.2)
            img = img * b_fac
            mean = img.mean(dim=(1, 2), keepdim=True)
            img = (img - mean) * c_fac + mean
            gray = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
            img = (gray + (img - gray) * s_fac).clamp_(0.0, 1.0)

        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(labels) > 0:
            valid = labels[(labels >= 0) & (labels < self.num_classes)]
            if len(valid) > 0:
                counts = torch.bincount(valid, minlength=self.num_classes).float()

        new_target: Dict[str, Any] = {
            "_is_cxcywh": True,
            "counts": counts,
            "boxes": boxes,
            "labels": labels,
        }
        return img, new_target

    def _load_raw_tensor_sample(self, idx: int) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Load a single un-augmented (img [3, H, W] float32 in [0, 1], target dict) from RAM cache or LMDB."""
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
            if "bg_mask" in t_np and not self.augment:
                target["bg_mask"] = torch.from_numpy(t_np["bg_mask"])
            return img, target

        data = super().prepare_data(idx)
        if data is None:
            return None, {}
        return data.get("image"), data.get("target", {})

    def _apply_mosaic4(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """4-image full-resolution Mosaic (Ultralytics-style) preserving small VisDrone object resolution.

        Aligns 4 images at a random split point (xc, yc) in [0.25, 0.75] of (W, H) without shrinking
        objects, clamps each quadrant's boxes strictly inside its quadrant, and filters sliver boxes.
        """
        n_total = len(self.data_infos)
        indices = [idx] + [random.randint(0, n_total - 1) for _ in range(3)]
        samples = [self._load_raw_tensor_sample(i) for i in indices]

        img0, t0 = samples[0]
        _, h, w = img0.shape
        yc = random.randint(int(0.25 * h), int(0.75 * h))
        xc = random.randint(int(0.25 * w), int(0.75 * w))

        canvas = torch.full((3, h, w), 0.447, dtype=img0.dtype)
        all_boxes: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        # (src_y1, src_y2, src_x1, src_x2, dst_y1, dst_y2, dst_x1, dst_x2, dx, dy)
        quads = [
            (h - yc, h, w - xc, w, 0, yc, 0, xc, xc - w, yc - h),       # Top-Left
            (h - yc, h, 0, w - xc, 0, yc, xc, w, xc, yc - h),           # Top-Right
            (0, h - yc, w - xc, w, yc, h, 0, xc, xc - w, yc),           # Bottom-Left
            (0, h - yc, 0, w - xc, yc, h, xc, w, xc, yc),               # Bottom-Right
        ]

        for (img_q, t_q), (sy1, sy2, sx1, sx2, dy1, dy2, dx1, dx2, dx, dy) in zip(samples, quads):
            if img_q is None:
                continue
            # Ensure matching spatial size if loaded from non-RAM path
            if img_q.shape[1] != h or img_q.shape[2] != w:
                img_q = F.interpolate(img_q.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
            canvas[:, dy1:dy2, dx1:dx2] = img_q[:, sy1:sy2, sx1:sx2]

            boxes_q = t_q.get("boxes", torch.empty((0, 4), dtype=torch.float32))
            labels_q = t_q.get("labels", torch.empty((0,), dtype=torch.int64))
            if len(boxes_q) == 0:
                continue

            cx = boxes_q[:, 0] * w
            cy = boxes_q[:, 1] * h
            bw = boxes_q[:, 2] * w
            bh = boxes_q[:, 3] * h
            orig_area = (bw * bh).clamp(min=1.0)

            bx1 = (cx - 0.5 * bw + dx).clamp(float(dx1), float(dx2))
            by1 = (cy - 0.5 * bh + dy).clamp(float(dy1), float(dy2))
            bx2 = (cx + 0.5 * bw + dx).clamp(float(dx1), float(dx2))
            by2 = (cy + 0.5 * bh + dy).clamp(float(dy1), float(dy2))

            nbw = bx2 - bx1
            nbh = by2 - by1
            new_area = nbw * nbh
            keep = (nbw >= 2.5) & (nbh >= 2.5) & ((new_area / orig_area) >= 0.15)
            if keep.any():
                ncx = ((bx1 + bx2) * 0.5) / float(w)
                ncy = ((by1 + by2) * 0.5) / float(h)
                nbw_n = nbw / float(w)
                nbh_n = nbh / float(h)
                all_boxes.append(torch.stack([ncx, ncy, nbw_n, nbh_n], dim=-1)[keep])
                all_labels.append(labels_q[keep])

        if all_boxes:
            boxes_cat = torch.cat(all_boxes, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
        else:
            boxes_cat = torch.empty((0, 4), dtype=torch.float32)
            labels_cat = torch.empty((0,), dtype=torch.int64)

        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(labels_cat) > 0:
            valid = labels_cat[(labels_cat >= 0) & (labels_cat < self.num_classes)]
            if len(valid) > 0:
                counts = torch.bincount(valid, minlength=self.num_classes).float()

        target: Dict[str, Any] = {
            "_is_cxcywh": True,
            "counts": counts,
            "boxes": boxes_cat,
            "labels": labels_cat,
        }
        return canvas, target

    def prepare_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Fast path when resized uint8 images are cached in shared RAM, with optional Mosaic + tensor augment."""
        if self.augment and self.mosaic_prob > 0.0 and len(self.data_infos) >= 4 and random.random() < self.mosaic_prob:
            img, target = self._apply_mosaic4(idx)
            img, target = self._apply_tensor_augment(img, target)
            return {"image": img, "target": target}

        img, target = self._load_raw_tensor_sample(idx)
        if img is None:
            return None
        if self.augment and isinstance(img, torch.Tensor):
            img, target = self._apply_tensor_augment(img, target)
        return {"image": img, "target": target}

    def __getstate__(self):
        """Enable safe pickle serialization across multiprocessing DataLoader workers."""
        state = self.__dict__.copy()
        state["env"], state["txn"] = None, None
        state["bg_env"], state["bg_txn"] = None, None
        return state


@HOOKS.register_module(name="CloseMosaicHook", force=True)
class CloseMosaicHook(BaseHook):
    """Automatically disables Mosaic augmentation during the final `close_epochs` of training.

    Helps the detector & presence_head converge cleanly to the true single-image distribution.
    """

    def __init__(self, close_epochs: int = 15):
        self.close_epochs = max(0, int(close_epochs))
        self._closed = False

    def before_epoch(self, trainer) -> None:
        if self._closed or self.close_epochs <= 0:
            return
        if trainer.epoch >= trainer.max_epochs - self.close_epochs:
            ds = getattr(trainer.train_loader, "dataset", None)
            # Unwrap Subset / wrapper datasets if any
            while ds is not None and not hasattr(ds, "mosaic_prob") and hasattr(ds, "dataset"):
                ds = ds.dataset
            if ds is not None and hasattr(ds, "mosaic_prob") and ds.mosaic_prob > 0.0:
                ds.mosaic_prob = 0.0
                self._closed = True
                msg = (
                    f"[CloseMosaicHook] Disabled Mosaic augmentation at epoch "
                    f"{trainer.epoch + 1}/{trainer.max_epochs} (final {self.close_epochs} epochs)."
                )
                if getattr(trainer, "logger", None) is not None:
                    trainer.logger.info(msg)
                else:
                    print(msg)


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
