import io
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

    Hỗ trợ nạp trực tiếp ground-truth multi-scale background mask từ Sidecar LMDB (`bg_lmdb_path`)
    để loại bỏ hoàn toàn chi phí tính toán rasterize bounding box trên CPU trong lúc train.
    """

    def __init__(
        self,
        data_root: str,
        pipeline=None,
        split: str = "train",            # 'train' hoặc 'val'
        split_ratio: float = 0.8,        # 80% train, 20% val
        seed: int = 42,                  # Seed đảm bảo train và val không bị lẫn
        test_mode: bool = False,
        num_classes: int = 10,
        return_dict: bool = True,        # Mặc định True để trả về dict {'counts', 'bg_mask', ...}
        bg_lmdb_path: Optional[str] = None,  # Đường dẫn Sidecar LMDB (bg_seg.lmdb)
        include_boxes: bool = False,     # Luôn kèm boxes/labels kể cả khi đã có bg_mask
        **kwargs,
    ):
        self.split = split.lower()
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_classes = num_classes
        self.return_dict = return_dict
        self.bg_lmdb_path = bg_lmdb_path
        self.include_boxes = include_boxes
        self.env = None
        self.txn = None
        self.bg_env = None
        self.bg_txn = None
        self._bg_num_tokens = 18900
        self._bg_format = "packbits_uint8"
        super().__init__(data_root=data_root, pipeline=pipeline, test_mode=test_mode)

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
        """Chia toàn bộ dataset thành 2 tập Train và Val theo đúng tỉ lệ split_ratio
        sao cho cả Train và Val đều cân bằng và bám sát phân phối các class của toàn bộ dữ liệu
        (Multi-label Iterative Stratification kết hợp cân bằng mật độ box).
        """
        total, num_classes = all_counts.shape
        n_train = int(total * split_ratio)
        n_val = total - n_train

        rng = np.random.default_rng(seed)

        # Tổng số box mục tiêu cần phân bổ vào tập Val cho từng class (tỷ lệ 1 - split_ratio)
        val_ratio = 1.0 - split_ratio
        global_box_sum = all_counts.sum(axis=0)  # (C,)
        target_val_boxes = global_box_sum * val_ratio
        target_val_pres = (all_counts > 0).sum(axis=0) * val_ratio

        # Sắp xếp thứ tự ưu tiên phân bổ từ các ảnh chứa class hiếm nhất -> phổ biến hơn
        # (Ý tưởng cốt lõi của Iterative Stratification: Sechidis et al. 2011)
        class_rarity_order = np.argsort(global_box_sum + 1e-6)  # Từ hiếm nhất tới nhiều nhất
        assigned = np.zeros(total, dtype=bool)
        is_val = np.zeros(total, dtype=bool)

        cur_val_boxes = np.zeros(num_classes, dtype=np.float64)
        cur_val_pres = np.zeros(num_classes, dtype=np.float64)
        cur_val_count = 0

        # Trọng số chuẩn hóa theo từng class để sai số tương đối giữa class hiếm và phổ biến cân bằng nhau
        inv_box_target = 1.0 / np.maximum(target_val_boxes, 1.0)
        inv_pres_target = 1.0 / np.maximum(target_val_pres, 1.0)

        # Duyệt từng class từ hiếm nhất đến phổ biến nhất
        for c in class_rarity_order:
            cand_indices = np.flatnonzero((~assigned) & (all_counts[:, c] > 0))
            if len(cand_indices) == 0:
                continue

            # Xáo trộn ngẫu nhiên các ảnh có chứa class c (kết hợp sắp xếp theo số lượng box của class c)
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

                # Đánh giá độ khớp quota nếu đưa ảnh idx vào Val vs đưa vào Train
                img_boxes = all_counts[idx]
                img_pres = (img_boxes > 0).astype(np.float64)

                # Tỷ lệ phần trăm hoàn thành quota riêng của class hiếm c đang xét (ưu tiên số 1)
                c_box_fill = cur_val_boxes[c] * inv_box_target[c]
                c_pres_fill = cur_val_pres[c] * inv_pres_target[c]
                c_fill_ratio = 0.75 * c_box_fill + 0.25 * c_pres_fill

                # Tỷ lệ phần trăm hoàn thành quota của các class khác đồng xuất hiện trong ảnh idx
                active_m = img_boxes > 0
                co_fill_ratio = (
                    0.75 * (cur_val_boxes[active_m] * inv_box_target[active_m]).mean()
                    + 0.25 * (cur_val_pres[active_m] * inv_pres_target[active_m]).mean()
                )

                # Nếu riêng class c chưa đủ quota (< 1.0) thì ưu tiên nạp vào Val cho đủ đúng tỷ lệ val_ratio
                if c_fill_ratio < 1.0 and co_fill_ratio < 1.08:
                    is_val[idx] = True
                    cur_val_count += 1
                    cur_val_boxes += img_boxes
                    cur_val_pres += img_pres
                else:
                    is_val[idx] = False

        # Phân bổ nốt các ảnh không có box nào (nếu có) theo đúng số lượng còn thiếu
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

        # Bước tinh chỉnh hoán đổi cục bộ (Local Swap Refinement) để tối thiểu hóa Total Variation Distance
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
        # 1. Đọc tổng số mẫu trong file LMDB
        if not os.path.exists(self.data_root):
            return []

        env, txn = open_lmdb(self.data_root, readonly=True)
        total_len_bytes = txn.get(b"__len__")
        if total_len_bytes is None:
            total = env.stat()["entries"]
        else:
            total = int(total_len_bytes.decode("ascii"))

        # 2. Quét (hoặc đọc cache) phân phối class toàn cục và chia Train/Val cân bằng theo phân phối gốc
        all_counts = self._load_or_scan_class_counts(self.data_root, txn, total, self.num_classes)
        train_indices, val_indices = self._stratified_train_val_split(
            all_counts=all_counts,
            split_ratio=self.split_ratio,
            seed=self.seed,
        )

        # 3. Chọn danh sách index theo split ('train' hoặc 'val')
        if self.split == "train":
            selected_indices = train_indices
        else:  # 'val' hoặc 'test'
            selected_indices = val_indices

        # Lưu lại index thực tế trong LMDB
        return [{"lmdb_idx": idx} for idx in selected_indices]

    def _init_db(self):
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

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        self._init_db()
        real_idx = self.data_infos[idx]["lmdb_idx"]
        key = f"{real_idx:08d}".encode("ascii")

        byteflow = self.txn.get(key)
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
            target: Dict[str, Any] = {"counts": counts}

            # Đọc nhanh pre-generated bg_mask từ sidecar LMDB nếu có
            bg_mask: Optional[torch.Tensor] = None
            if self.bg_txn is not None:
                bg_bytes = self.bg_txn.get(key)
                if bg_bytes is not None:
                    if self._bg_format == "packbits_uint8":
                        unpacked = np.unpackbits(
                            np.frombuffer(bg_bytes, dtype=np.uint8)
                        )[: self._bg_num_tokens]
                        bg_mask = torch.from_numpy(unpacked.copy())  # uint8 (18900,)
                    else:
                        bg_mask = torch.from_numpy(
                            np.frombuffer(bg_bytes, dtype=np.uint8).copy()
                        )
                    target["bg_mask"] = bg_mask

            # Chỉ tạo và truyền boxes/labels khi chưa có bg_mask hoặc được yêu cầu tường minh
            if bg_mask is None or self.include_boxes:
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
        else:
            target = counts

        return {"image": image, "target": target}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"], state["txn"] = None, None
        state["bg_env"], state["bg_txn"] = None, None
        return state
