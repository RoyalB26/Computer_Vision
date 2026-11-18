import io
import os
import sys
import pickle

sys.path.insert(0, "D:/Study/Projects/Computer_Vision")
import numpy as np
from BaseModel.dataset import open_lmdb

data_root = "D:/apps/Datasets/VisDrone/train_data.lmdb"
env, txn = open_lmdb(data_root, readonly=True)

total_len_bytes = txn.get(b"__len__")
total = int(total_len_bytes.decode("ascii")) if total_len_bytes else env.stat()["entries"]

num_samples = min(500, total)
print(f"Tổng số ảnh trong LMDB: {total}. Đang quét {num_samples} ảnh mẫu ngẫu nhiên...")

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

stats = {c: {"w": [], "h": [], "area": []} for c in range(10)}
per_image_total_area = {c: [] for c in range(10)}
per_image_counts = {c: [] for c in range(10)}

TARGET_W, TARGET_H = 960, 960
TOTAL_PIXELS = TARGET_W * TARGET_H

for idx in range(num_samples):
    byteflow = txn.get(f"{idx:08d}".encode("ascii"))
    if byteflow is None:
        continue
    data = pickle.loads(byteflow)
    boxes = data.get("boxes", [])
    labels = data.get("labels", [])

    img_class_area = {c: 0.0 for c in range(10)}
    img_class_count = {c: 0 for c in range(10)}

    for b, l in zip(boxes, labels):
        if 0 <= l < 10 and len(b) >= 4:
            w_px = float(b[2]) * TARGET_W
            h_px = float(b[3]) * TARGET_H
            area_px = w_px * h_px
            stats[l]["w"].append(w_px)
            stats[l]["h"].append(h_px)
            stats[l]["area"].append(area_px)
            img_class_area[l] += area_px
            img_class_count[l] += 1

    for c in range(10):
        per_image_total_area[c].append(img_class_area[c])
        per_image_counts[c].append(img_class_count[c])

print("\n" + "=" * 115)
print(f"THỐNG KÊ KÍCH THƯỚC VÀ DIỆN TÍCH TỪNG ĐỐI TƯỢNG (1 OBJECT) TRÊN ẢNH {TARGET_W}x{TARGET_H} (TỔNG {TOTAL_PIXELS:,} px)")
print("=" * 115)
print(
    f"{'Class Name':<18} | {'Số lượng':<8} | {'Rộng x Cao tb':<15} | {'Min Area':<9} | "
    f"{'Median Area':<11} | {'Mean Area':<10} | {'Max Area':<10} | {'% ẢNH (1 obj)':<14}"
)
print("-" * 115)

for c, name in enumerate(VISDRONE_CLASSES):
    cnt = len(stats[c]["area"])
    if cnt == 0:
        print(f"{name:<18} | {0:<8} | {'N/A':<15} | {'N/A':<9} | {'N/A':<11} | {'N/A':<10} | {'N/A':<10} | {'N/A':<14}")
        continue
    mean_w = np.mean(stats[c]["w"])
    mean_h = np.mean(stats[c]["h"])
    min_a = np.min(stats[c]["area"])
    med_a = np.median(stats[c]["area"])
    mean_a = np.mean(stats[c]["area"])
    max_a = np.max(stats[c]["area"])
    pct_img = (mean_a / TOTAL_PIXELS) * 100

    print(
        f"{name:<18} | {cnt:<8} | {mean_w:5.1f} x {mean_h:5.1f} px  | {min_a:7.0f} px | "
        f"{med_a:9.0f} px | {mean_a:8.0f} px | {max_a:8.0f} px | {pct_img:9.4f}%"
    )

print("=" * 115)

print("\n" + "=" * 115)
print(f"THỐNG KÊ TỔNG DIỆN TÍCH CỦA CẢ CLASS TRONG 1 KHUNG HÌNH (GỒM TẤT CẢ VẬT THỂ CÙNG CLASS)")
print("=" * 115)
print(
    f"{'Class Name':<18} | {'Số lượng TB/ảnh':<16} | {'Tổng diện tích TB':<18} | "
    f"{'% Diện tích cả ảnh':<20} | {'Tương đương số ô //16':<20}"
)
print("-" * 115)

for c, name in enumerate(VISDRONE_CLASSES):
    mean_count = np.mean(per_image_counts[c])
    mean_total_area = np.mean(per_image_total_area[c])
    pct_total = (mean_total_area / TOTAL_PIXELS) * 100
    # Một ô trên feature map //16 có diện tích 16x16 = 256 pixel ảnh
    token_16_equiv = mean_total_area / 256.0

    print(
        f"{name:<18} | {mean_count:8.2f} obj/ảnh  | {mean_total_area:12.0f} px      | "
        f"{pct_total:12.4f}%         | {token_16_equiv:12.1f} ô (token)"
    )

print("=" * 115)
