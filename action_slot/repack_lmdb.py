"""Repack TACO LMDB datasets at exact training resolution `(256, 768)` and `(32, 96)`.

Loại bỏ lớp bọc `pickle` và giảm kích thước ảnh từ `1536x512` xuống thẳng `(256, 768)`
giúp giảm 4x-6x dung lượng đĩa/FUSE I/O và tăng tốc giải mã trong `ACTION_SLOT_LMDBDataset`.

Hỗ trợ 2 định dạng đầu ra (đều được `ACTION_SLOT_LMDBDataset` đọc tự động):
  - `jpeg` (mặc định): Lưu raw JPEG bytes (256x768) cho `rgb_front` và raw PNG bytes (32x96)
    cho `mask_background`. Giảm ~4x-6x dung lượng, dễ nhét toàn bộ vào `/tmp` trên Kaggle.
  - `raw`: Lưu tensor `uint8` thô có header `RAW0` (không nén). Tốn đĩa hơn `jpeg` nhưng
    chi phí giải mã CPU lúc train gần như bằng 0.

Ví dụ:
    python action_slot/repack_lmdb.py --src D:/apps/Datasets/TACO/data --dst D:/apps/Datasets/TACO_256x768
    python action_slot/repack_lmdb.py --src /kaggle/input/datasets/royalb26/taco-lmdb-part2 --dst /tmp/taco-lmdb-part2 --format jpeg
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import os
import pickle
import shutil
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm


LMDB_KINDS_DEFAULT = ("rgb_front", "mask_background")


def _decode_source_payload(val: bytes) -> Optional[Dict[str, object]]:
    """Decode legacy pickled entry or already-repacked RAW0/JPEG/PNG entry."""
    if len(val) >= 9 and val[:4] == b"RAW0":
        h, w, c = struct.unpack("<HHB", val[4:9])
        arr = np.frombuffer(val, dtype=np.uint8, offset=9)
        if arr.size == h * w * c:
            shape = (h, w) if c == 1 else (h, w, c)
            return {"raw_array": arr.reshape(shape)}
    if val[:3] == b"\xff\xd8\xff" or val[:8] == b"\x89PNG\r\n\x1a\n":
        return {"image": val}
    return pickle.loads(val)


def _pack_raw0(arr: np.ndarray) -> bytes:
    """Pack uint8 array `(H, W)` or `(H, W, C)` with 9-byte `RAW0` header."""
    if arr.ndim == 2:
        h, w = arr.shape
        c = 1
    elif arr.ndim == 3:
        h, w, c = arr.shape
    else:
        raise ValueError(f"Unsupported array ndim={arr.ndim} for RAW0 packing")
    header = b"RAW0" + struct.pack("<HHB", int(h), int(w), int(c))
    return header + np.ascontiguousarray(arr, dtype=np.uint8).tobytes()


def _transform_rgb_entry(
    key_val: Tuple[bytes, bytes],
    rgb_hw: Tuple[int, int],
    out_format: str,
    jpeg_quality: int,
) -> Optional[Tuple[bytes, bytes]]:
    key, val = key_val
    target_h, target_w = rgb_hw
    sample = _decode_source_payload(val)
    if sample is None:
        return None

    if "raw_array" in sample:
        rgb = np.asarray(sample["raw_array"], dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif "image" in sample:
        buf = np.frombuffer(sample["image"], dtype=np.uint8)
        # Dùng DCT 1/2 nếu ảnh nguồn >= 2x target để tăng tốc độ repack
        bgr = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2)
        if bgr is None or bgr.shape[0] < target_h or bgr.shape[1] < target_w:
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            pil = Image.open(io.BytesIO(sample["image"])).convert("RGB")
            bgr = cv2.cvtColor(np.asarray(pil, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    else:
        return None

    if bgr.shape[0] != target_h or bgr.shape[1] != target_w:
        interp = cv2.INTER_AREA if (bgr.shape[0] > target_h or bgr.shape[1] > target_w) else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=interp)

    if out_format == "raw":
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return key, _pack_raw0(rgb)

    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode key={key!r}")
    return key, enc.tobytes()


def _transform_mask_entry(
    key_val: Tuple[bytes, bytes],
    mask_hw: Tuple[int, int],
    out_format: str,
) -> Optional[Tuple[bytes, bytes]]:
    key, val = key_val
    target_h, target_w = mask_hw
    sample = _decode_source_payload(val)
    if sample is None:
        return None

    if "raw_array" in sample:
        mask = np.asarray(sample["raw_array"], dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
    elif "image" in sample:
        buf = np.frombuffer(sample["image"], dtype=np.uint8)
        mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            pil = Image.open(io.BytesIO(sample["image"])).convert("L")
            mask = np.asarray(pil, dtype=np.uint8)
    else:
        return None

    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    if out_format == "raw":
        return key, _pack_raw0(mask)

    ok, enc = cv2.imencode(".png", mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError(f"Failed to PNG-encode key={key!r}")
    return key, enc.tobytes()


def _discover_lmdb_jobs(src_roots: List[str], dst_root: str, include_obj_mask: bool) -> List[Tuple[str, str, str]]:
    """Find `(kind, src_lmdb_dir, dst_lmdb_dir)` jobs matching TACO folder structure."""
    kinds = list(LMDB_KINDS_DEFAULT) + (["mask_object"] if include_obj_mask else [])
    jobs: List[Tuple[str, str, str]] = []
    seen_dst = set()

    for root in src_roots:
        abs_root = os.path.abspath(root)
        if not os.path.isdir(abs_root):
            print(f"[Warn] Bỏ qua thư mục nguồn không tồn tại: {root}")
            continue

        # Trường hợp root chính là thư mục town (chứa thẳng rgb_front/data.mdb)
        for kind in kinds:
            cand = os.path.join(abs_root, kind)
            if os.path.isfile(os.path.join(cand, "data.mdb")):
                town_name = os.path.basename(abs_root)
                dst_dir = os.path.join(dst_root, town_name, kind)
                if dst_dir not in seen_dst:
                    seen_dst.add(dst_dir)
                    jobs.append((kind, cand, dst_dir))

        # Trường hợp root chứa nhiều thư mục town con
        for entry in sorted(os.listdir(abs_root)):
            town_dir = os.path.join(abs_root, entry)
            if not os.path.isdir(town_dir):
                continue
            for kind in kinds:
                cand = os.path.join(town_dir, kind)
                if os.path.isfile(os.path.join(cand, "data.mdb")):
                    dst_dir = os.path.join(dst_root, entry, kind)
                    if dst_dir not in seen_dst:
                        seen_dst.add(dst_dir)
                        jobs.append((kind, cand, dst_dir))

    return jobs


def _estimate_initial_map_size(src_mdb_path: str, kind: str, out_format: str) -> int:
    src_bytes = os.path.getsize(src_mdb_path)
    if kind == "rgb_front":
        scale = 2.0 if out_format == "raw" else 0.45
    else:
        scale = 1.25
    est = int(src_bytes * scale) + 64 * 1024 * 1024
    # Làm tròn lên bội số 16MB và tối thiểu 64MB
    align = 16 * 1024 * 1024
    return max(64 * 1024 * 1024, ((est + align - 1) // align) * align)


def _commit_batch_with_resize(dst_env: lmdb.Environment, batch: List[Tuple[bytes, bytes]]) -> None:
    """Write `(key, val)` batch to LMDB, dynamically growing `map_size` on `MapFullError`."""
    while True:
        try:
            with dst_env.begin(write=True) as txn:
                for k, v in batch:
                    txn.put(k, v)
            return
        except lmdb.MapFullError:
            cur_size = dst_env.info()["map_size"]
            new_size = max(cur_size * 2, cur_size + 256 * 1024 * 1024)
            dst_env.set_mapsize(new_size)


def repack_single_lmdb(
    kind: str,
    src_dir: str,
    dst_dir: str,
    rgb_hw: Tuple[int, int],
    mask_hw: Tuple[int, int],
    out_format: str,
    jpeg_quality: int,
    workers: int,
    commit_interval: int,
    overwrite: bool,
) -> Tuple[int, int, int]:
    """Repack one LMDB directory. Returns `(num_entries, src_bytes, dst_bytes)`."""
    src_mdb = os.path.join(src_dir, "data.mdb")
    src_bytes = os.path.getsize(src_mdb)

    if os.path.exists(dst_dir):
        if overwrite:
            shutil.rmtree(dst_dir)
        elif os.path.isfile(os.path.join(dst_dir, "data.mdb")):
            dst_bytes = os.path.getsize(os.path.join(dst_dir, "data.mdb"))
            print(f"[Skip] Đã tồn tại {dst_dir} (dùng --overwrite để ghi đè).")
            return 0, src_bytes, dst_bytes

    os.makedirs(dst_dir, exist_ok=True)
    src_env = lmdb.open(src_dir, readonly=True, lock=False, readahead=True, meminit=False)
    init_map_size = _estimate_initial_map_size(src_mdb, kind, out_format)
    dst_env = lmdb.open(dst_dir, map_size=init_map_size, sync=False, map_async=True, writemap=False)

    with src_env.begin(write=False) as txn:
        total_entries = txn.stat()["entries"]

    def _worker_fn(kv: Tuple[bytes, bytes]) -> Optional[Tuple[bytes, bytes]]:
        if kind == "rgb_front":
            return _transform_rgb_entry(kv, rgb_hw, out_format, jpeg_quality)
        if kind == "mask_background":
            return _transform_mask_entry(kv, mask_hw, out_format)
        # Với mask_object giữ nguyên payload gốc (tần suất dùng thấp, cấu trúc list mask riêng)
        return kv

    written = 0
    chunk_size = max(128, commit_interval)
    desc = f"{os.path.basename(os.path.dirname(src_dir))}/{kind}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, tqdm(
        total=total_entries, desc=desc, dynamic_ncols=True, unit="img"
    ) as pbar:
        with src_env.begin(write=False) as src_txn:
            cursor = src_txn.cursor()
            raw_chunk: List[Tuple[bytes, bytes]] = []

            for k, v in cursor:
                raw_chunk.append((bytes(k), bytes(v)))
                if len(raw_chunk) >= chunk_size:
                    processed = [res for res in pool.map(_worker_fn, raw_chunk) if res is not None]
                    if processed:
                        _commit_batch_with_resize(dst_env, processed)
                        written += len(processed)
                    pbar.update(len(raw_chunk))
                    raw_chunk.clear()

            if raw_chunk:
                processed = [res for res in pool.map(_worker_fn, raw_chunk) if res is not None]
                if processed:
                    _commit_batch_with_resize(dst_env, processed)
                    written += len(processed)
                pbar.update(len(raw_chunk))
                raw_chunk.clear()

    dst_env.sync()
    dst_env.close()
    src_env.close()

    dst_bytes = os.path.getsize(os.path.join(dst_dir, "data.mdb"))
    return written, src_bytes, dst_bytes


def main():
    parser = argparse.ArgumentParser(description="Repack TACO LMDBs at target resolution without pickle overhead.")
    parser.add_argument("--src", nargs="+", required=True, help="Một hoặc nhiều thư mục data_root nguồn.")
    parser.add_argument("--dst", required=True, help="Thư mục gốc lưu LMDB đã repack.")
    parser.add_argument("--rgb-size", type=int, nargs=2, default=[256, 768], metavar=("H", "W"))
    parser.add_argument("--mask-size", type=int, nargs=2, default=[32, 96], metavar=("H", "W"))
    parser.add_argument("--format", choices=["jpeg", "raw"], default="jpeg", help="Định dạng lưu trữ đầu ra.")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="Chất lượng JPEG (chỉ dùng với --format jpeg).")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="Số luồng xử lý song song.")
    parser.add_argument("--commit-interval", type=int, default=1000, help="Số entry mỗi transaction ghi LMDB.")
    parser.add_argument("--include-obj-mask", action="store_true", help="Copy thêm thư mục mask_object nếu có.")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè LMDB đích nếu đã tồn tại.")
    args = parser.parse_args()

    jobs = _discover_lmdb_jobs(args.src, os.path.abspath(args.dst), args.include_obj_mask)
    if not jobs:
        print("[Error] Không tìm thấy thư mục LMDB hợp lệ nào trong --src.")
        sys.exit(1)

    print(
        f"Tìm thấy {len(jobs)} LMDB jobs | Format={args.format} | "
        f"RGB={tuple(args.rgb_size)} | Mask={tuple(args.mask_size)} | Workers={args.workers}"
    )

    t0 = time.time()
    total_src, total_dst, total_items = 0, 0, 0
    for kind, src_dir, dst_dir in jobs:
        n_items, sb, db = repack_single_lmdb(
            kind=kind,
            src_dir=src_dir,
            dst_dir=dst_dir,
            rgb_hw=(args.rgb_size[0], args.rgb_size[1]),
            mask_hw=(args.mask_size[0], args.mask_size[1]),
            out_format=args.format,
            jpeg_quality=args.jpeg_quality,
            workers=args.workers,
            commit_interval=args.commit_interval,
            overwrite=args.overwrite,
        )
        total_items += n_items
        total_src += sb
        total_dst += db

    elapsed = max(1e-6, time.time() - t0)
    src_gb = total_src / (1024 ** 3)
    dst_gb = total_dst / (1024 ** 3)
    ratio = total_src / max(1, total_dst)
    print(
        f"[Done] {total_items:,} entries trong {elapsed:.1f}s | "
        f"Dung lượng: {src_gb:.2f} GB -> {dst_gb:.2f} GB (giảm {ratio:.2f}x)"
    )


if __name__ == "__main__":
    main()
