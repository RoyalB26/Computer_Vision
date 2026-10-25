# 06. Hướng dẫn Thiết lập Cấu hình YAML (Configuration System)

Hệ thống cấu hình của `BaseModel` được thiết kế theo nguyên lý **Configuration-Driven**. Mọi thành phần của vòng đời huấn luyện (Mô hình, Dữ liệu, Trình tối ưu, Hàm mất mát, Lịch trình học, Chỉ số đánh giá và Tiện ích theo dõi) đều có thể được thiết lập hoàn toàn thông qua file YAML hoặc JSON.

---

## 1. Cấu trúc một File Cấu hình YAML Hoàn chỉnh

```yaml
# ==========================================
# 0. THÔNG TIN THỰC THI & PHẦN CỨNG
# ==========================================
work_dir: "./work_dirs/visdrone_detection"
device: "cuda"            # 'cuda', 'cuda:0', 'mps', 'cpu', hoặc 'tpu'
seed: 42
deterministic: false      # Đảm bảo tái lập thí nghiệm 100% (bật benchmark=False)
resume_from: null         # Đường dẫn checkpoint .pth để tiếp tục train nếu bị ngắt

# Tiện ích trực quan & Tăng tốc
tqdm: true                # Bật thanh tiến trình trực quan
accelerate: false         # Bật Hugging Face Accelerate cho phân tán đa GPU
tensorboard: true         # Tự động ghi log sự kiện TensorBoard (<work_dir>/tf_logs)
wandb: false              # Bật đồng bộ trực tuyến lên Weights & Biases
project_name: "VisDrone_Exp"

# Nạp động các module tùy biến vào Registry
custom_imports:
  - "Visdrone.model"
  - "Visdrone.dataset"

# ==========================================
# 1. MÔ HÌNH (MODEL)
# ==========================================
model:
  type: "VisDroneDetector"
  backbone_type: "simple_cnn"   # 'simple_cnn' hoặc 'resnet18'
  in_channels: 3
  num_classes: 10
  feat_channels: 64

# ==========================================
# 2. DỮ LIỆU & DATALOADER
# ==========================================
data:
  train:
    dataset:
      type: "CustomLMDBDataset"
      data_root: "D:/apps/Datasets/VisDrone/VisDrone_data.lmdb"
      split: "train"
      split_ratio: 0.8
      seed: 42
      pipeline:
        - type: "Resize"
          size: [224, 224]
        - type: "ToTensor"
        - type: "Normalize"
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
    batch_size: 16
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    collate_fn: null            # 'PadCollate', 'DetectionCollate' hoặc null

  val:
    dataset:
      type: "CustomLMDBDataset"
      data_root: "D:/apps/Datasets/VisDrone/VisDrone_data.lmdb"
      split: "val"
      split_ratio: 0.8
      seed: 42
      pipeline:
        - type: "Resize"
          size: [224, 224]
        - type: "ToTensor"
        - type: "Normalize"
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
    batch_size: 16
    shuffle: false
    num_workers: 2
    pin_memory: true
    drop_last: false

# ==========================================
# 3. BỘ TỐI ƯU HÓA (OPTIMIZER)
# ==========================================
optimizer:
  type: "AdamW"                 # SGD, Adam, AdamW, RMSprop, Adagrad
  lr: 0.0005
  weight_decay: 0.0001
  # momentum: 0.9               # (Nếu dùng SGD)

# ==========================================
# 4. LỊCH TRÌNH HỌC (LR SCHEDULER)
# ==========================================
lr_scheduler:
  type: "CosineAnnealingLR"     # StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
  T_max: 50
  eta_min: 0.00001

# ==========================================
# 5. CHỈ SỐ ĐO LƯỜNG (METRICS)
# ==========================================
metrics:
  acc:
    type: "Accuracy"
    topk: [1]

# ==========================================
# 6. ENGINE HUẤN LUYỆN (TRAIN ENGINE)
# ==========================================
train_cfg:
  max_epochs: 50
  val_interval: 1               # Đánh giá validation sau mỗi 1 epoch
  log_interval: 20              # In thông tin loss/lr ra màn hình sau mỗi N batch/iter
  grad_clip: 1.0                # Cắt tỉa gradient tránh exploding gradients (null nếu tắt)
  use_amp: true                 # Bật Automatic Mixed Precision 16-bit
```

---

## 2. Chi tiết Toàn bộ các Khóa Cấu hình có thể Thiết lập

Dưới đây là danh mục tra cứu đầy đủ tất cả các trường cấu hình theo từng nhóm tính năng:

### 2.1. Nhóm Cấu hình Chung & Môi trường thực thi (Root Level)

| Tên Khóa | Kiểu | Mặc định | Mô tả chi tiết |
| :--- | :---: | :---: | :--- |
| `work_dir` | `str` | `"./work_dirs"` | Thư mục lưu toàn bộ log (`train.log`), checkpoint (`best.pth`, `last.pth`) và file cấu hình lưu trữ. |
| `device` | `str` | `"cuda"` | Thiết bị phần cứng: `"cuda"`, `"cuda:0"`, `"cpu"`, `"mps"` (Apple Silicon), hoặc `"tpu"` (Google Cloud TPU). |
| `seed` | `int` | `42` | Cố định seed cho Python, NumPy, cuDNN nhằm đảm bảo tái lập thí nghiệm đồng nhất. |
| `deterministic` | `bool` | `false` | Nếu `true`, buộc cuDNN dùng giải thuật xác định (`cudnn.benchmark = False`). |
| `resume_from` | `str` | `null` | Đường dẫn file `.pth` để khôi phục toàn bộ trạng thái epoch, optimizer, scheduler khi train dở dang. |
| `custom_imports` | `list[str]` | `[]` | Danh sách các file/module Python người dùng cần import để nạp Model/Dataset vào Registry. |

---

### 2.2. Nhóm Cấu hình Tiện ích Theo dõi & Tăng tốc (Monitoring & Acceleration)

| Tên Khóa | Kiểu | Mặc định | Mô tả chi tiết |
| :--- | :---: | :---: | :--- |
| `tqdm` | `bool` | `false` | Hiển thị thanh tiến trình trực quan `tqdm` kèm tốc độ xử lý `it/s`, thời gian còn lại và loss tức thời. |
| `accelerate` | `bool` | `false` | Kích hoạt Hugging Face Accelerate cho huấn luyện đa GPU (DDP) và tối ưu hóa bộ nhớ phần cứng. |
| `tensorboard` | `bool` | `false` | Tự động ghi biểu đồ loss, learning rate và validation metrics vào thư mục `<work_dir>/tf_logs`. |
| `wandb` | `bool` | `false` | Đồng bộ toàn bộ thông số thí nghiệm và chỉ số đánh giá lên nền tảng Weights & Biases trực tuyến. |
| `project_name` | `str` | `"BaseModel_CV"` | Tên dự án hiển thị trên bảng điều khiển Weights & Biases (khi bật `wandb`). |

---

### 2.3. Nhóm Cấu hình Mô hình (`model`)

| Tên Khóa | Kiểu | Mô tả chi tiết |
| :--- | :---: | :--- |
| `type` | `str` | Tên class của mô hình đã đăng ký trong `MODELS` (ví dụ: `"VisDroneDetector"`, `"ModularModel"`). |
| `*` (các tham số con) | Tùy biến | Toàn bộ các tham số khởi tạo sẽ được truyền trực tiếp vào hàm `__init__` của class mô hình (ví dụ: `num_classes`, `in_channels`, `feat_channels`, `backbone_type`). |

---

### 2.4. Nhóm Cấu hình Dữ liệu (`data.train` và `data.val`)

Mỗi tập dữ liệu (`train` / `val`) hỗ trợ các khóa thiết lập:

| Tên Khóa | Kiểu | Mặc định | Mô tả chi tiết |
| :--- | :---: | :---: | :--- |
| `dataset` | `dict` | *(Bắt buộc)* | Cấu hình dataset gồm: `type` (tên class dataset), `data_root`, `pipeline`, và các tham số riêng. |
| `dataset.pipeline` | `list` | `[]` | Danh sách các phép biến đổi ảnh: `Resize`, `ToTensor`, `Normalize`, `Compose`,... |
| `batch_size` | `int` | `16` | Kích thước batch cho mỗi lần nạp dữ liệu. |
| `shuffle` | `bool` | `true`/`false` | Xáo trộn mẫu ngẫu nhiên (`true` cho train, `false` cho val). |
| `num_workers` | `int` | `4` | Số luồng CPU đa tiến trình nạp dữ liệu song song (đặt `0` trên Windows nếu debug). |
| `pin_memory` | `bool` | `true` | Khóa bộ nhớ đệm giúp tăng tốc độ chuyển tensor từ RAM sang VRAM GPU. |
| `drop_last` | `bool` | `true`/`false` | Bỏ qua batch cuối cùng nếu số lượng mẫu không đủ chia hết cho `batch_size`. |
| `collate_fn` | `str` / `dict` | `null` | Hàm gom nhóm batch tùy biến: `"SmartCollate"`, `"PadCollate"`, `"DetectionCollate"` hoặc custom. |

---

### 2.5. Nhóm Cấu hình Bộ Tối ưu hóa (`optimizer`)

| Tên Khóa | Kiểu | Mô tả chi tiết |
| :--- | :---: | :--- |
| `type` | `str` | Tên bộ tối ưu: `"SGD"`, `"Adam"`, `"AdamW"`, `"RMSprop"`, `"Adagrad"`. |
| `lr` | `float` | Tốc độ học cơ sở (Base Learning Rate), ví dụ: `0.001`, `0.0005`. |
| `weight_decay` | `float` | Hệ số suy giảm trọng số (L2 regularization), ví dụ: `0.0001`. |
| `momentum` | `float` | Hệ số quán tính (thường dùng cho `"SGD"`, ví dụ: `0.9`). |

---

### 2.6. Nhóm Cấu hình Điều chỉnh Tốc độ học (`lr_scheduler`)

| Tên Khóa | Kiểu | Mô tả chi tiết |
| :--- | :---: | :--- |
| `type` | `str` | Tên giải thuật scheduler: `"CosineAnnealingLR"`, `"StepLR"`, `"MultiStepLR"`, `"ReduceLROnPlateau"`. |
| `T_max` | `int` | Số chu kỳ tối đa cho Cosine Annealing. |
| `eta_min` | `float` | Tốc độ học tối thiểu khi kết thúc lịch trình (ví dụ: `1e-6`). |
| `step_size` / `gamma`| `int` / `float`| Chu kỳ giảm và hệ số nhân giảm bước (dành cho `StepLR`). |

---

### 2.7. Nhóm Cấu hình Vòng lặp Huấn luyện (`train_cfg`)

| Tên Khóa | Kiểu | Mặc định | Mô tả chi tiết |
| :--- | :---: | :---: | :--- |
| `max_epochs` | `int` | `100` | Tổng số epoch cần huấn luyện. |
| `val_interval` | `int` | `1` | Chu kỳ chạy đánh giá tập Validation (sau mỗi $N$ epoch). |
| `log_interval` | `int` | `20` | Tần suất in thông số loss, learning rate ra terminal sau mỗi $N$ iteration (batch). |
| `grad_clip` | `float` | `null` | Giá trị chuẩn cực đại để cắt tỉa gradient (chống bùng nổ gradient, ví dụ: `1.0`). |
| `use_amp` | `bool` | `false` | Bật chế độ tự động tính toán độ chính xác hỗn hợp (Automatic Mixed Precision 16-bit). |

---

### 2.8. Nhóm Cấu hình Hooks Tùy biến (`hooks`)

Cho phép khai báo thêm các hook mở rộng trực tiếp trong file cấu hình:

```yaml
hooks:
  - type: "LoggerHook"
    interval: 50                # Chu kỳ in log sau mỗi 50 iter
    log_keys: ["loss", "loss_cls", "lr"]  # Chỉ in các chỉ số mong muốn (bỏ loss_box, loss_obj,...)
    show_time: true             # Có hiển thị time hay không (true/false)
    show_lr: true               # Có hiển thị lr hay không (true/false)
  - type: "CheckpointHook"
    interval: 1
    save_best: "acc"            # Theo dõi metric tốt nhất để lưu best.pth
    mode: "max"
  - type: "EarlyStoppingHook"
    monitor: "val_loss"
    patience: 10                # Dừng train nếu sau 10 epoch loss không giảm
    mode: "min"
```
