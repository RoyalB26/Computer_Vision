# Toy Action: Synthetic Spatio-Temporal Video Dataset & Action-Slot Architecture

Dự án sinh tập dữ liệu hành động tổng hợp (toy action video dataset) và huấn luyện mô hình nhận diện video kết hợp kiến trúc **Action-Slot** (Slot Attention 3D + X3D-M) trên khung xương chuẩn hóa **BaseModel**.

---

## 1. Cấu trúc thư mục

```
toy_action/
├── __init__.py           # Package exports
├── generate_data.py      # Bộ sinh dữ liệu CLI (--sample và --lmdb) kèm bg_mask
├── dataset.py            # ToyActionLMDBDataset (kế thừa BaseDataset của BaseModel)
├── model.py              # ActionSlotModel & ToyActionNet & MultiLabelMetric
├── loss.py               # ActionSlotLoss (CrossEntropy trên 24 slots + Attention Loss)
├── hooks.py              # TrainToyActionMAPHook (tính và ghi nhận train action mAP)
├── inference.py          # Script suy luận, phủ Attention Heatmap đỏ lên 16 frame & ghép lưới 4x4
├── config.yaml           # Cấu hình huấn luyện tích hợp BaseModel/scripts/train.py
├── README.md             # Tài liệu hướng dẫn sử dụng và thông số kỹ thuật
├── model_parts/          # Các khối thành phần của kiến trúc Action-Slot
│   ├── __init__.py
│   ├── backbones.py      # X3DBackbone và YoloV11Backbone (đầu ra P5 sau Neck, load checkpoint pretrained)
│   ├── slot_attention.py # Spatiotemporal 3D Slot Attention & SoftPositionEmbed3D
│   ├── heads.py          # ReduceAllocatedHead (phân loại slot: single, group+, bg)
│   └── x3d_baseline.py   # Baseline X3D chuẩn không dùng slot attention
├── metrics/              # Bộ đo lường đánh giá hiệu năng
│   ├── __init__.py
│   ├── toy_labels.py     # 48 nhãn hành động chuẩn hóa
│   └── toy_metric.py     # ToyActionMetric đo mAP tổng thể và mAP theo nhóm
├── data/                 # Thư mục chứa cơ sở dữ liệu LMDB (train/val)
└── sample_output/        # Thư mục xuất mẫu khi chạy với tag --sample (.mp4 + 16 ảnh + 16 bg_masks + attn_grid)
```

---

## 2. Đặc tả dữ liệu Toy Action

- **Số frame**: 16 frame liên tục cho mỗi mẫu video clip.
- **Kích thước khung hình**: 224 x 224 pixels (RGB).
- **Kích thước vật thể**: Ngẫu nhiên trong khoảng từ 8x8 đến 32x32 pixels.
- **Tập ký hiệu (6 loại)**:
  - `x`: Dấu chéo với 2 nét chéo mang 2 màu khác nhau.
  - `y`: Dấu chữ y với 2 nhánh trên và nhánh dưới mang màu khác nhau.
  - `o`: Chữ o thường viền tròn rỗng.
  - `T`: Dấu chữ T với thanh ngang và thanh dọc mang màu khác nhau.
  - `8`: Số 8 với 2 vòng tròn trên/dưới mang màu khác nhau.
  - `O`: Chữ O hoa **tô kín phần trong (fill) và có màu viền ngoài (outline) khác biệt rõ rệt**.
- **Nền chuyển động (Background)**:
  - Nền biến đổi ngẫu nhiên theo clip với 6 chế độ: phân vùng Voronoi, chia cắt ngang/dọc, 4 góc (quadrants), dải màu mượt (gradient), hoặc các mảng chữ nhật.
- **Mặt nạ nền (`bg_mask`)**:
  - Mỗi mẫu sinh ra 16 `bg_mask` tương ứng chính xác với vị trí vật thể di chuyển qua 16 frame.
  - `255` (hoặc `1.0` trên tensor): điểm ảnh thuộc nền (Background).
  - `0` (hoặc `0.0` trên tensor): điểm ảnh thuộc vật thể (Foreground Object).
- **4 Hướng chuyển động**:
  - `gotop` (lên trên), `gobottom` (xuống dưới), `goleft` (trái), `goright` (phải).
- **Quy tắc gán nhãn**:
  1. Vật thể đi cùng hành động: ví dụ `x:gotop`, `y:gobottom`.
  2. Nếu có nhóm vật thể cùng loại và cùng hành động (xuất hiện cùng lúc hoặc khác thời điểm) thì thêm dấu `+`: ví dụ `x+:gotop`, `y+:gobottom`.
  3. Không gian nhãn gồm **48 lớp multi-label** ($12 \text{ biến thể ký hiệu} \times 4 \text{ hướng chuyển động}$).

---

## 3. Kiến trúc mô hình Action-Slot

Mô hình `ActionSlotModel` được thiết kế chuyên biệt cho nhận diện hành động và phân rã vật thể:

1. **Backbone (`X3DBackbone`)**:
   - Sử dụng backbone chuẩn video 3D: **X3D-M** pretrained.
   - Gồm 5 khối: `[0: Stem, 1: ResStage1, 2: ResStage2, 3: ResStage3, 4: ResStage4]`.
   - **Chiến lược đóng băng (Freezing Strategy)**:
     - `finetune_last_n_stages: 2`: Khóa toàn bộ tham số của 3 khối đầu (Stem, Stage1, Stage2) với `requires_grad = False` và giữ BatchNorm ở chế độ `eval()`.
     - Chỉ mở khóa huấn luyện (unfreeze) **2 tầng cuối (Stage3 và Stage4)**.
2. **Spatiotemporal Feature Projection**:
   - Khối `Conv3d` $1 \times 1 \times 1$ chiếu đặc trưng từ 192 chiều lên 256 chiều (`slot_dim`).
3. **Slot Attention 3D**:
   - Nhúng toạ độ thời không `SoftPositionEmbed3D` kích thước $(16, 7, 7)$.
   - Gom cụm thông tin thành **24 semantic slots** (tương ứng 6 ký hiệu $\times$ 4 hành động) + 1 background slot.
4. **Classification Head (`ReduceAllocatedHead`)**:
   - Mỗi slot trong 24 slot dự đoán 3 logits: `[0: single, 1: group+, 2: background]`.
   - Hiệu chỉnh logit theo background: `fg_logits = pred[:, :, :2] - pred[:, :, 2:3]`.
   - Chuyển đổi đối xứng 2 chiều về đúng vector 48 nhãn `ALL_CLASSES`.
5. **Loss Function (`ActionSlotLoss`)**:
   - `loss_actor`: CrossEntropyLoss trên 24 slots.
   - `loss_bg_attn`: Giám sát attention map của background slot bằng `bg_mask`.
   - `loss_attn_suppress`: Triệt tiêu attention map của các slot không hoạt động.

---

## 4. Hướng dẫn sử dụng & Dòng lệnh (CLI)

### A. Sinh dữ liệu (`generate_data.py`)

#### 1. Xuất 1 mẫu thử nghiệm trực quan (`--sample`)
Tạo 1 file `.mp4`, 1 file video mask `.mp4`, 16 ảnh frame và 16 ảnh `bg_mask` tại `toy_action/sample_output/`:
```powershell
python toy_action/generate_data.py --sample
```

#### 2. Sinh cơ sở dữ liệu LMDB (`--lmdb`)
Tạo $N$ mẫu video nén JPEG chất lượng cao và nhãn đi kèm trong LMDB (khuyến nghị 10.000 train, 2.000 val):
```powershell
python toy_action/generate_data.py --lmdb --num-samples 10000 --val-samples 2000 --lmdb-path toy_action/data/train_lmdb --val-lmdb-path toy_action/data/val_lmdb
```

---

### B. Huấn luyện & Đánh giá với BaseModel

```powershell
# 1. Kiểm tra thống kê cấu hình mô hình & đóng băng layer
python BaseModel/scripts/train.py --config toy_action/config.yaml --show

# 2. Huấn luyện mô hình ActionSlotModel (Backbone đóng băng, chỉ mở 2 tầng cuối)
python BaseModel/scripts/train.py --config toy_action/config.yaml

# 3. Ghi đè tham số qua CLI (ví dụ: huấn luyện 20 epochs, batch size 8)
python BaseModel/scripts/train.py --config toy_action/config.yaml --cfg-options train_cfg.max_epochs=20 data.train.batch_size=8

# 4. Chạy đánh giá checkpoint (Evaluation Only)
python BaseModel/scripts/train.py --config toy_action/config.yaml --eval-only
```

---

### C. Sử dụng Backbone YoloV11 (`YoloV11Backbone`)

Mô hình hỗ trợ chuyển đổi sang backbone YOLOv11 (kèm PANet Neck, đầu ra đặc trưng $P5$ kích thước $256 \times 16 \times 7 \times 7$):
- Nạp trọng số checkpoint pretrained từ `D:\Study\Projects\Computer_Vision\yolo11n.pt`.
- Cho phép đóng băng / mở huấn luyện linh hoạt theo số tầng (`finetune_last_n_stages`, `freeze_stages`, `unfreeze_all`).

Cấu hình trong `config.yaml`:
```yaml
model:
  type: "ActionSlotModel"
  backbone_type: "yolov11"
  pretrained_backbone_path: "D:/Study/Projects/Computer_Vision/yolo11n.pt"
  finetune_last_n_stages: 2
```

---

### D. Suy luận & Trực quan hóa Attention Heatmap (`inference.py`)

File `inference.py` nạp 16 frame từ thư mục mẫu, chạy qua mô hình, phủ bản đồ chú ý (màu đỏ càng đậm khi attention càng cao), đánh số thứ tự từng frame và ghép thành 1 ảnh lưới $4 \times 4$:

```powershell
# Suy luận trên thư mục sample_output với trọng số mặc định:
python toy_action/inference.py --input-dir toy_action/sample_output

# Suy luận với checkpoint đã huấn luyện:
python toy_action/inference.py --input-dir toy_action/sample_output --checkpoint work_dirs/toy_action_slot_exp/best.pth

# Tùy chỉnh threshold phát hiện và đường dẫn file ảnh xuất ra:
python toy_action/inference.py --input-dir toy_action/sample_output --threshold 0.4 --output toy_action/sample_output/attn_inference_grid.png
```

Kết quả hiển thị:
- **Terminal**: In ra nhãn phát hiện vượt ngưỡng và danh sách Top-6 lớp có xác suất cao nhất.
- **File ảnh**: Lưu tại `toy_action/sample_output/attn_inference_grid.png` kích thước $896 \times 896$ gồm 16 frame phủ Attention Heatmap đỏ và đánh số `F00` .. `F15`.
