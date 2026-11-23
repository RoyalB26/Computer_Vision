"""Pre-generate Ground Truth Multi-Scale Background Segmentation Masks into a Sidecar LMDB.

Reads bounding boxes from the source VisDrone LMDB (e.g. train_data.lmdb) without modifying it,
rasterizes exact multi-scale background masks for (P3: 120x120, P4: 60x60, P5: 30x30),
bit-packs them with np.packbits (~2.3 KB/sample, ~25 MB total for 8,626 images),
and writes them into a standalone sidecar LMDB (bg_seg.lmdb) for zero-overhead training.

Usage:
    python Visdrone/build_bg_seg_lmdb.py
    python Visdrone/build_bg_seg_lmdb.py --src D:/apps/Datasets/VisDrone/train_data.lmdb --dst D:/apps/Datasets/VisDrone/bg_seg.lmdb
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time
from typing import List, Tuple

import lmdb
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def rasterize_multiscale_bg_mask(
    boxes: np.ndarray,
    resolutions: Tuple[Tuple[int, int], ...] = ((120, 120), (60, 60), (30, 30)),
) -> np.ndarray:
    """Tạo vector multi-scale background mask (uint8: 1 = background, 0 = bounding box).

    Logic làm tròn tọa độ đồng nhất tuyệt đối với `VisDroneDetector._create_bg_masks_from_boxes`:
      - x1 = clip(trunc((cx - w/2) * W), 0, W - 1)
      - y1 = clip(trunc((cy - h/2) * H), 0, H - 1)
      - x2 = max(x1 + 1, clip(ceil((cx + w/2) * W), 0, W))
      - y2 = max(y1 + 1, clip(ceil((cy + h/2) * H), 0, H))
    """
    has_boxes = boxes is not None and len(boxes) > 0
    if has_boxes:
        bxs = np.asarray(boxes, dtype=np.float32)
        if bxs.ndim == 1:
            bxs = bxs[None, :]
        cx = bxs[:, 0]
        cy = bxs[:, 1]
        w = bxs[:, 2]
        h = bxs[:, 3]

    flat_masks: List[np.ndarray] = []
    for H, W in resolutions:
        mask = np.ones((H, W), dtype=np.uint8)
        if has_boxes:
            x1 = np.clip(np.trunc((cx - w / 2.0) * W).astype(np.int64), 0, W - 1)
            y1 = np.clip(np.trunc((cy - h / 2.0) * H).astype(np.int64), 0, H - 1)
            x2 = np.maximum(x1 + 1, np.clip(np.ceil((cx + w / 2.0) * W).astype(np.int64), 0, W))
            y2 = np.maximum(y1 + 1, np.clip(np.ceil((cy + h / 2.0) * H).astype(np.int64), 0, H))
            for i in range(len(x1)):
                mask[y1[i] : y2[i], x1[i] : x2[i]] = 0
        flat_masks.append(mask.reshape(-1))

    return np.concatenate(flat_masks, axis=0)


def build_bg_seg_lmdb(
    src_lmdb: str,
    dst_lmdb: str,
    input_size: Tuple[int, int] = (960, 960),
    strides: Tuple[int, ...] = (8, 16, 32),
    commit_interval: int = 500,
    overwrite: bool = True,
) -> None:
    """Đọc toàn bộ bounding boxes từ src_lmdb và tạo file sidecar dst_lmdb."""
    if not os.path.exists(src_lmdb):
        raise FileNotFoundError(f"Không tìm thấy LMDB nguồn: {src_lmdb}")

    resolutions = tuple((input_size[0] // s, input_size[1] // s) for s in strides)
    total_tokens = sum(h * w for h, w in resolutions)
    packed_bytes_per_sample = (total_tokens + 7) // 8

    print("=" * 72)
    print("  PRE-GENERATE GROUND TRUTH BG_SEG SIDECAR LMDB (CÁCH A)")
    print("=" * 72)
    print(f"[*] LMDB nguồn (readonly): {src_lmdb}")
    print(f"[*] LMDB đích (sidecar) : {dst_lmdb}")
    print(f"[*] Kích thước ảnh      : {input_size[0]}x{input_size[1]}")
    print(f"[*] Các tầng quy mô     : {resolutions} -> tổng {total_tokens:,} tokens")
    print(f"[*] Nén bit (packbits)  : {packed_bytes_per_sample:,} bytes / mẫu")

    # Mở LMDB nguồn ở chế độ chỉ đọc (tuyệt đối an toàn cho dữ liệu gốc)
    src_env = lmdb.open(src_lmdb, readonly=True, lock=False, readahead=True, meminit=False)
    with src_env.begin(write=False) as src_txn:
        total_len_bytes = src_txn.get(b"__len__")
        total = int(total_len_bytes.decode("ascii")) if total_len_bytes else src_env.stat()["entries"]

    print(f"[*] Tổng số mẫu cần xử lý: {total:,}")

    # Xử lý thư mục đích
    if os.path.exists(dst_lmdb):
        if overwrite:
            if os.path.isdir(dst_lmdb):
                shutil.rmtree(dst_lmdb)
            else:
                os.remove(dst_lmdb)
        else:
            raise FileExistsError(f"Thư mục đích đã tồn tại: {dst_lmdb}. Truyền --overwrite để ghi đè.")

    os.makedirs(dst_lmdb, exist_ok=True)

    # Cấp phát map_size gọn gàng cho Windows (không bị phình file trống hàng chục GB)
    # Ước lượng: total * packed_bytes * 2.5 + 32 MB đệm B+tree
    map_size = int(total * packed_bytes_per_sample * 2.5) + 32 * 1024 * 1024
    print(f"[*] Cấp phát LMDB map_size: {map_size / (1024 * 1024):.1f} MB")

    dst_env = lmdb.open(dst_lmdb, map_size=map_size, readonly=False, meminit=False, map_async=True)

    t0 = time.perf_counter()
    bg_ratio_sum = 0.0
    processed = 0

    src_txn = src_env.begin(write=False)
    dst_txn = dst_env.begin(write=True)

    try:
        for idx in tqdm(range(total), desc="Building bg_seg.lmdb", unit="img"):
            key = f"{idx:08d}".encode("ascii")
            byteflow = src_txn.get(key)
            if byteflow is None:
                continue

            data = pickle.loads(byteflow)
            raw_boxes = data.get("boxes", [])

            mask_flat = rasterize_multiscale_bg_mask(raw_boxes, resolutions=resolutions)
            bg_ratio_sum += float(mask_flat.mean())

            # Nén 8 bit vào 1 byte -> giảm 8x dung lượng & băng thông IO
            packed = np.packbits(mask_flat).tobytes()
            dst_txn.put(key, packed)
            processed += 1

            if processed % commit_interval == 0:
                dst_txn.commit()
                dst_txn = dst_env.begin(write=True)

        # Ghi metadata tự mô tả
        dst_txn.put(b"__len__", str(total).encode("ascii"))
        dst_txn.put(b"__num_tokens__", str(total_tokens).encode("ascii"))
        dst_txn.put(b"__resolutions__", pickle.dumps(resolutions))
        dst_txn.put(b"__format__", b"packbits_uint8")
        dst_txn.commit()
    finally:
        src_txn.abort()
        src_env.close()
        dst_env.sync()
        dst_env.close()

    elapsed = time.perf_counter() - t0
    data_mdb = os.path.join(dst_lmdb, "data.mdb")
    file_size_mb = os.path.getsize(data_mdb) / (1024 * 1024) if os.path.isfile(data_mdb) else 0.0
    avg_bg_pct = (bg_ratio_sum / max(processed, 1)) * 100.0

    print("\n" + "=" * 72)
    print("  HOÀN TẤT TẠO BG_SEG.LMDB!")
    print("=" * 72)
    print(f"[+] Đã xử lý        : {processed:,} / {total:,} ảnh trong {elapsed:.2f}s ({processed / max(elapsed, 1e-6):.0f} img/s)")
    print(f"[+] Tỷ lệ nền TB    : {avg_bg_pct:.2f}% (Foreground: {100.0 - avg_bg_pct:.2f}%)")
    print(f"[+] Dung lượng file : {file_size_mb:.2f} MB ({data_mdb})")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Pre-generate bg_seg sidecar LMDB from VisDrone train_data.lmdb")
    parser.add_argument(
        "--src",
        type=str,
        default="D:/apps/Datasets/VisDrone/train_data.lmdb",
        help="Đường dẫn LMDB gốc chứa ảnh và bounding boxes",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="D:/apps/Datasets/VisDrone/bg_seg.lmdb",
        help="Đường dẫn LMDB đích lưu mặt nạ nền (sidecar)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[960, 960],
        help="Kích thước ảnh đầu vào (H W), mặc định 960 960",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Không ghi đè nếu thư mục đích đã tồn tại",
    )
    args = parser.parse_args()

    build_bg_seg_lmdb(
        src_lmdb=args.src,
        dst_lmdb=args.dst,
        input_size=(args.size[0], args.size[1]),
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
