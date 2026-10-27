import numpy as np
import json
import random

def generate_shape_dataset(
    num_samples: int = 1000,
    h: int = 64,
    w: int = 64,
    output_npy: str = "images.npy",
    output_json: str = "labels.json",
    seed: int = 42
):
    """
    Sinh dataset nhị phân (0 và 255) kích thước (N, H, W).
    - output_npy: file .npy chứa tensor (N, H, W) dtype uint8
    - output_json: file .json chứa class label và tọa độ đặt mẫu 5x5
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Định nghĩa ma trận template 5x5 cho từng hình dạng (1: nét vẽ, 0: nền)
    templates = {
        "plus": np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=np.uint8),

        "minus": np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8),

        "cross": np.array([
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1]
        ], dtype=np.uint8),

        "circle": np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ], dtype=np.uint8)
    }

    shape_classes = list(templates.keys()) + ["empty"]
    
    # 2. Khởi tạo mảng ảnh N x H x W toàn số 0 kiểu uint8
    images = np.zeros((num_samples, h, w), dtype=np.uint8)
    labels = []

    # 3. Random vị trí và chèn vào canvas
    for idx in range(num_samples):
        chosen_shape = random.choice(shape_classes)
        sample_meta = {
            "id": idx,
            "label": chosen_shape,
            "coord_top_left": None  # [row_top, col_left]
        }

        if chosen_shape != "empty":
            # Chọn ngẫu nhiên tọa độ hợp lệ để mẫu 5x5 nằm hoàn toàn trong ảnh
            top = random.randint(0, h - 5)
            left = random.randint(0, w - 5)
            
            # Gán giá trị 255 cho các pixel nét vẽ
            images[idx, top:top+5, left:left+5] = templates[chosen_shape] * 255
            sample_meta["coord_top_left"] = [int(top), int(left)]

        labels.append(sample_meta)

    # 4. Xuất file .npy và .json
    np.save(output_npy, images)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"-> Đã lưu tensor ảnh vào: {output_npy} | Shape: {images.shape} | Dtype: {images.dtype}")
    print(f"-> Giá trị pixel trong mảng: {np.unique(images)}")
    print(f"-> Đã lưu nhãn vào: {output_json}")

if __name__ == "__main__":
    generate_shape_dataset(
        num_samples=10000,
        h=16,
        w=16,
        output_npy="dataset.npy",
        output_json="labels.json"
    )