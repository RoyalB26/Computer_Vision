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
from tqdm import tqdm

from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset, close_lmdb, open_lmdb


@DATASETS.register_module()
class ACTION_SLOT_LMDBDataset(BaseDataset):
    """LMDB-based Action Slot dataset supporting single or multiple data_root paths.

    Matches the temporal strided frame sampling of the original TACO dataset:
      - Computes `step = num_frame // sequence_len` across the full scenario duration.
      - Builds up to `max_num = 50` strided 16-frame clips per scenario using verified existing LMDB keys.
      - Gracefully handles scenarios with `num_frame < 50` (including `16 <= num_frame < 50` and `num_frame < 16`)
        or non-contiguous frame numbers so every sample always yields exact `sequence_len` frames.
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
        min_frames: Minimum number of frames required for a valid scenario (default 1).
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
        min_frames: int = 1,
        filter_non_contiguous: bool = False,
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
        self.filter_non_contiguous = filter_non_contiguous
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
            env, txn = open_lmdb(
                path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            txn.abort()
            self.env_cache[path] = env
        return self.env_cache[path]

    @staticmethod
    def _get_prefix_keys(env, prefix_path: str) -> List[str]:
        """Return all keys in LMDB starting with prefix_path using key-only B-tree cursor iteration."""
        p = prefix_path.replace("\\", "/").rstrip("/") + "/"
        keys: List[str] = []
        with env.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_range(p.encode("utf-8")):
                while True:
                    k_str = cursor.key().decode("utf-8")
                    if k_str.startswith(p):
                        keys.append(k_str)
                    else:
                        break
                    if not cursor.next():
                        break
        return keys

    @staticmethod
    def _match_bg_keys(
        video_keys: List[str],
        clean_bg_prefix: str,
        bg_set: set,
        bg_keys: List[str],
    ) -> List[str]:
        """Match each selected RGB frame key to its corresponding background mask key (or closest existing key)."""
        seg_temp: List[str] = []
        n_vid = len(video_keys)
        n_bg = len(bg_keys)
        for idx, vk in enumerate(video_keys):
            stem = vk.split("/")[-1].split(".")[0]
            cand = f"{clean_bg_prefix}/{stem}.png"
            if cand in bg_set:
                seg_temp.append(cand)
            else:
                bg_idx = min(n_bg - 1, int(round(idx * (n_bg - 1) / max(1, n_vid - 1))))
                seg_temp.append(bg_keys[bg_idx])
        return seg_temp

    def _build_scenario_clips(
        self,
        rgb_keys: List[str],
        bg_keys: List[str],
        clean_rgb_prefix: str,
        clean_bg_prefix: str,
    ) -> Tuple[List[List[str]], List[List[str]], str]:
        """Build strided clips of length `self.sequence_len` for both `num_frame >= 50` and `num_frame < 50`.
        
        Returns:
            (videos, segs, case_type) where case_type is one of:
              - 'case1_ge50' (num_frame >= 50 and n_rgb >= 50)
              - 'case2_16to49' (16 <= n_rgb < 50)
              - 'case3_lt16' (n_rgb < 16)
        """
        videos: List[List[str]] = []
        segs: List[List[str]] = []
        if not rgb_keys or not bg_keys:
            return videos, segs, "unknown"

        rgb_set = set(rgb_keys)
        bg_set = set(bg_keys)
        max_num = 50
        n_rgb = len(rgb_keys)
        ext = os.path.splitext(rgb_keys[0])[1] or ".jpg"
        case_type = "unknown"

        try:
            start_frame = int(rgb_keys[0].split("/")[-1].split(".")[0])
            end_frame = int(rgb_keys[-1].split("/")[-1].split(".")[0])
            num_frame = end_frame - start_frame + 1
        except (ValueError, IndexError):
            start_frame, end_frame, num_frame = 0, n_rgb - 1, n_rgb

        # 1. Standard TACO strided sampling when num_frame >= 50 and n_rgb >= 50
        if num_frame >= 50 and n_rgb >= 50:
            step = max(1, num_frame // self.sequence_len)
            for m in range(max_num):
                start = start_frame + m
                if start_frame + (self.sequence_len - 1) * step > end_frame:
                    break
                videos_temp: List[str] = []
                for i in range(start, end_frame + 1, step):
                    rgb_k = f"{clean_rgb_prefix}/{str(i).zfill(8)}{ext}"
                    if rgb_k in rgb_set:
                        videos_temp.append(rgb_k)
                    if len(videos_temp) == self.sequence_len:
                        break
                if len(videos_temp) == self.sequence_len:
                    videos.append(videos_temp)
                    segs.append(self._match_bg_keys(videos_temp, clean_bg_prefix, bg_set, bg_keys))
            if videos:
                case_type = "case1_ge50"

        # 2. Fallback for scenarios with num_frame < 50 (or sparse frame numbering where `videos` is still empty)
        if not videos:
            if n_rgb >= self.sequence_len:
                case_type = "case2_16to49"
                step_idx = max(1, n_rgb // self.sequence_len)
                max_start = n_rgb - (self.sequence_len - 1) * step_idx
                for m in range(min(max_num, max(1, max_start))):
                    videos_temp = [rgb_keys[m + k * step_idx] for k in range(self.sequence_len)]
                    videos.append(videos_temp)
                    segs.append(self._match_bg_keys(videos_temp, clean_bg_prefix, bg_set, bg_keys))
            else:
                # 0 < n_rgb < sequence_len: uniform temporal interpolation to exact `sequence_len` frames
                case_type = "case3_lt16"
                idxs = np.linspace(0, n_rgb - 1, self.sequence_len).round().astype(int).tolist()
                videos_temp = [rgb_keys[i] for i in idxs]
                videos.append(videos_temp)
                segs.append(self._match_bg_keys(videos_temp, clean_bg_prefix, bg_set, bg_keys))

        return videos, segs, case_type

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

        # 0. Kiểm tra và nạp từ cache file nếu đã được tạo từ trước (load tức thì trong 0.05 giây)
        cache_candidates = [
            os.path.join("./work_dirs", f".cache_taco_{self.split}.pkl"),
            os.path.join(os.path.dirname(data_path), f".cache_taco_{self.split}.pkl"),
            f".cache_taco_{self.split}.pkl",
        ]
        for cp in cache_candidates:
            if os.path.exists(cp):
                try:
                    with open(cp, "rb") as f:
                        cached = pickle.load(f)
                    if isinstance(cached, dict) and "videos_path" in cached and len(cached["videos_path"]) > 0:
                        # Kiểm tra xem cache có đúng với data_root hiện tại hay không (tránh xung đột khi chuyển giữa Local / Colab / Kaggle)
                        cached_roots = cached.get("data_roots", [])
                        current_roots = [os.path.abspath(r) for r in self.data_roots]
                        if cached_roots and cached_roots != current_roots:
                            continue

                        # Kiểm tra tính tồn tại của tệp trên máy hiện tại
                        sample_rgb = cached["videos_path"][0].get("rgb_path", "")
                        if not os.path.exists(sample_rgb):
                            continue
                        self.videos_path = cached["videos_path"]
                        self.label = cached.get("label", {})
                        case_counts = cached.get("case_counts", {"case1_ge50": len(self.videos_path), "case2_16to49": 0, "case3_lt16": 0})
                        non_contiguous_count = cached.get("non_contiguous_count", 0)
                        print(f"[{self.split.upper()} DATASET] Nạp siêu tốc {len(self.videos_path)} scenarios từ file cache: {cp}")
                        total = len(self.videos_path)
                        indices = list(range(total))
                        rng = random.Random(self.seed)
                        rng.shuffle(indices)
                        if self.num_samples:
                            indices = indices[:self.num_samples]
                        return [{"lmdb_idx": idx} for idx in indices]
                except Exception:
                    pass

        # 1. Khám phá toàn bộ các thư mục Town LMDB qua các data_roots (chỉ 1 pass duy nhất, tránh gọi os.path.exists hàng ngàn lần trên Kaggle)
        town_info: Dict[str, Dict[str, Any]] = {}
        for root in self.data_roots:
            if not os.path.exists(root):
                continue

            root_rgb = os.path.join(root, "rgb_front")
            root_bg = os.path.join(root, "mask_background")
            root_obj = os.path.join(root, "mask_object")
            if os.path.exists(root_rgb) and os.path.exists(root_bg):
                base_name = os.path.basename(os.path.normpath(root))
                norm_name = base_name.replace("_lmdb", "")
                t_info = {
                    "rgb": root_rgb,
                    "bg": root_bg,
                    "obj": root_obj if os.path.exists(root_obj) else None,
                }
                town_info[norm_name] = t_info
                town_info[base_name] = t_info

            if os.path.isdir(root):
                try:
                    for item in os.listdir(root):
                        item_dir = os.path.join(root, item)
                        if not os.path.isdir(item_dir):
                            continue
                        sub_rgb = os.path.join(item_dir, "rgb_front")
                        sub_bg = os.path.join(item_dir, "mask_background")
                        sub_obj = os.path.join(item_dir, "mask_object")
                        if os.path.exists(sub_rgb) and os.path.exists(sub_bg):
                            norm_name = item.replace("_lmdb", "")
                            t_info = {
                                "rgb": sub_rgb,
                                "bg": sub_bg,
                                "obj": sub_obj if os.path.exists(sub_obj) else None,
                            }
                            town_info[norm_name] = t_info
                            town_info[item] = t_info
                except Exception:
                    pass

        # 2. Cơ chế quét nhanh toàn bộ key trong LMDB (Key-Only Sweep) theo từng Town, lưu dictionary in-memory
        from collections import defaultdict
        town_key_index: Dict[str, Dict[str, List[str]]] = {}

        def get_town_prefix_keys(lmdb_path: str, target_prefix: str) -> List[str]:
            if lmdb_path not in town_key_index:
                pdict = defaultdict(list)
                try:
                    env = self._get_lmdb_env(lmdb_path)
                    with env.begin(write=False) as txn:
                        cur = txn.cursor()
                        if cur.first():
                            while True:
                                k_str = cur.key().decode("utf-8")
                                pfx = "/".join(k_str.split("/")[:-1])
                                pdict[pfx].append(k_str)
                                if not cur.next():
                                    break
                except Exception:
                    pass
                town_key_index[lmdb_path] = pdict
            return town_key_index[lmdb_path].get(target_prefix, [])

        self.videos_path = []
        self.label = {}
        case_counts = {"case1_ge50": 0, "case2_16to49": 0, "case3_lt16": 0}
        non_contiguous_count = 0

        for k, v in tqdm(data.items(), desc=f"[{self.split.upper()} DATASET] Loading scenarios", dynamic_ncols=True):
            split_path = k.split("/")
            town_key = split_path[0]
            norm_town = town_key.replace("_lmdb", "")
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
            rgb_keys: List[str] = []
            bg_keys: List[str] = []

            # Tra cứu siêu tốc O(1) qua pre-discovered town_info
            if norm_town in town_info:
                info = town_info[norm_town]
                r_keys = get_town_prefix_keys(info["rgb"], clean_rgb_prefix)
                if r_keys:
                    b_keys = get_town_prefix_keys(info["bg"], clean_bg_prefix)
                    if b_keys:
                        matched_rgb = info["rgb"]
                        matched_bg = info["bg"]
                        matched_obj = info["obj"]
                        rgb_keys = r_keys
                        bg_keys = b_keys

            # Fallback tra cứu truyền thống nếu town chưa có trong pre-discovery
            if matched_rgb is None:
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
                            r_keys = self._get_prefix_keys(rgb_env, clean_rgb_prefix)
                            if r_keys:
                                bg_env = self._get_lmdb_env(c_bg)
                                b_keys = self._get_prefix_keys(bg_env, clean_bg_prefix)
                                if b_keys:
                                    matched_rgb, matched_bg, matched_obj = c_rgb, c_bg, c_obj
                                    rgb_keys, bg_keys = r_keys, b_keys
                                    break
                    if matched_rgb is not None:
                        break

            if matched_rgb is None or len(rgb_keys) < self.min_frames or not bg_keys:
                continue

            # Kiểm tra xem video có bị khuyết số frame ở giữa hay không
            is_non_contiguous = False
            try:
                s_frame = int(rgb_keys[0].split("/")[-1].split(".")[0])
                e_frame = int(rgb_keys[-1].split("/")[-1].split(".")[0])
                if len(rgb_keys) != (e_frame - s_frame + 1):
                    is_non_contiguous = True
                    non_contiguous_count += 1
            except Exception:
                pass

            # Loại bỏ các video bị khiếm khuyết nếu filter_non_contiguous=True
            if self.filter_non_contiguous and is_non_contiguous:
                continue

            videos, segs, case_type = self._build_scenario_clips(
                rgb_keys=rgb_keys,
                bg_keys=bg_keys,
                clean_rgb_prefix=clean_rgb_prefix,
                clean_bg_prefix=clean_bg_prefix,
            )

            if not videos:
                continue

            if case_type in case_counts:
                case_counts[case_type] += 1

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

        # Close all LMDB handles in parent process before DataLoader forks worker processes
        close_lmdb()
        self.env_cache.clear()

        # Tự động lưu cache index để tái sử dụng ngay lập tức ở lần chạy sau (chỉ mất 0.05 giây)
        if len(self.videos_path) > 0:
            for save_dir in ["./work_dirs", os.path.dirname(data_path)]:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    cache_file = os.path.join(save_dir, f".cache_taco_{self.split}.pkl")
                    with open(cache_file, "wb") as f:
                        pickle.dump({
                            "data_roots": [os.path.abspath(r) for r in self.data_roots],
                            "videos_path": self.videos_path,
                            "label": self.label,
                            "case_counts": case_counts,
                            "non_contiguous_count": non_contiguous_count,
                        }, f)
                    print(f"[{self.split.upper()} DATASET] Đã lưu file cache: {cache_file}")
                    break
                except Exception:
                    pass

        # In thống kê số lượng video thuộc mỗi trường hợp ra terminal
        total = len(self.videos_path)
        print(f"[{self.split.upper()} DATASET] Thống kê số lượng video:")
        print(f"  + Tổng số scenario hợp lệ : {total}")
        print(f"  + Trường hợp 1 (>=50 frames liên tục) : {case_counts['case1_ge50']}")
        print(f"  + Trường hợp 2 (16 - 49 frames thực tế) : {case_counts['case2_16to49']}")
        print(f"  + Trường hợp 3 (<16 frames, nội suy đều): {case_counts['case3_lt16']}")
        if self.filter_non_contiguous:
            print(f"  + Đã loại bỏ số video bị khuyết frame ở giữa: {non_contiguous_count}")
        else:
            print(f"  + Số video bị khuyết số frame ở giữa  : {non_contiguous_count}")

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

        # Guarantee exact sequence length and background mask count across all batch items
        if data:
            while len(data) < self.sequence_len:
                data.append(data[-1].copy())
            if len(data) > self.sequence_len:
                data = data[:self.sequence_len]

        expected_bg_len = max(1, self.sequence_len // self.mask_every_frame)
        if target["bg_seg"]:
            while len(target["bg_seg"]) < expected_bg_len:
                target["bg_seg"].append(target["bg_seg"][-1].copy())
            if len(target["bg_seg"]) > expected_bg_len:
                target["bg_seg"] = target["bg_seg"][:expected_bg_len]

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
