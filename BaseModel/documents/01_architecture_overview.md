# 01. Tổng quan kiến trúc & Design Patterns

## 1. Mục tiêu kiến trúc

Trong phát triển các dự án Computer Vision (CV), việc viết đi viết lại mã nguồn xử lý DataLoader, huấn luyện epoch, lưu checkpoint, log kết quả, tính toán độ đo, bật tắt mixed precision... dẫn đến sự phân mảnh mã nguồn, dễ phát sinh lỗi và khó tái sử dụng.

`BaseModel` được thiết kế như một **Meta-framework** (tương tự triết lý của OpenMMLab / Detectron2 / PyTorch Lightning):
- **Phân tách hoàn toàn (Decoupling)** giữa giải thuật (Algorithm/Model) và hạ tầng huấn luyện (Training Infrastructure).
- **Tính đa hình (Polymorphism)**: Mọi mô hình thị giác máy tính đều tuân theo hợp đồng giao tiếp chuẩn hóa.
- **Dựa trên cấu hình (Configuration-Driven)**: Mọi thành phần đều có thể lắp ghép thông qua cấu hình YAML/JSON.

---

## 2. Bản đồ phân bổ trách nhiệm các thư mục

```
BaseModel/
├── core/       # HẠ TẦNG CỐT LÕI:
│               # - Registry: Quản lý đăng ký động các Class bằng decorator
│               # - BaseTrainer: Vòng lặp huấn luyện, validation, AMP, GPU management
│               # - Hooks: Checkpoint, Logger, Scheduler, EarlyStopping
│
├── dataset/    # DỮ LIỆU & TIỀN XỬ LÝ:
│               # - BaseDataset: Chuẩn hóa nạp dữ liệu, lazy-loading, collate_fn
│               # - helpers: load_image, find_images, parse_csv, open_lmdb, read_lmdb_bytes, create_lmdb
│               # - collate: SmartCollate, PadCollate, DetectionCollate
│               # - transforms: Pipeline Compose, ToTensor, Normalize, Resize
│
├── model/      # KIẾN TRÚC MẠNG:
│               # - BaseModel: Lớp cơ sở với 3 nhánh thực thi train / val / predict
│               # - builder: Khởi tạo mô hình động từ Registry
│
├── loss/       # HÀM MẤT MÁT:
│               # - BaseLoss: Cơ sở cho các loss có trọng số (loss_weight)
│               # - LossAggregator: Gom nhóm nhiều loss thành composite objective
│               # - standard_losses: CrossEntropy, BCE, SmoothL1, Focal, Dice
│
├── utils/      # CÔNG CỤ BỔ TRỢ:
│               # - config: ConfigParser hỗ trợ dot-notation và ghi đè CLI
│               # - metrics: BaseMetric, Accuracy, MeanIoU
│               # - logger, seed, device, checkpoint, math_tools
│
└── scripts/    # ENTRY POINTS THỰC THI:
                # - train.py: Huấn luyện hoàn toàn tự động
                # - val.py: Benchmark & đánh giá checkpoint
                # - test.py: Thực thi suy luận (inference)
```

---

## 3. Các Design Patterns cốt lõi

### 3.1. Registry Pattern (`core/registry.py`, `core/registrar.py`)
Mẫu thiết kế ánh xạ giữa chuỗi định danh (string name) và class/hàm Python:
```python
from BaseModel.core import MODELS
import torch.nn as nn

@MODELS.register_module()
class ResNetBackbone(nn.Module):
    def __init__(self, depth=50):
        super().__init__()
        ...
```
Nhờ Registry Pattern, ta có thể xây dựng đối tượng từ điển cấu hình:
```python
cfg = {"type": "ResNetBackbone", "depth": 50}
model = MODELS.build(cfg)
```

### 3.2. Template Method Pattern (`core/trainer.py`, `dataset/base_dataset.py`)
Định hình cấu trúc thuật toán ở lớp cha và cho phép lớp con ghi đè các bước cụ thể:
- Trong `BaseTrainer`:
  ```python
  def train(self):
      self.call_hook("before_train")
      for epoch in range(self.max_epochs):
          self.call_hook("before_epoch")
          self.train_epoch()
          if self.should_validate(epoch):
              self.val_epoch()
          self.call_hook("after_epoch")
      self.call_hook("after_train")
  ```
- Lớp con chỉ cần override `train_step(batch, batch_idx)` nếu có logic phức tạp (ví dụ: huấn luyện GAN 2 bước Generator / Discriminator).

### 3.3. Observer / Hook Pattern (`core/hooks/`)
Mỗi sự kiện trong vòng lặp (bắt đầu train, sau mỗi batch, sau mỗi epoch, kết thúc validation) được phát tán tới các `Hook`.
- `CheckpointHook`: Tự động tìm model có điểm số cao nhất để lưu `best.pth`.
- `LoggerHook`: Ghi nhận tốc độ xử lý, tiến độ epoch, loss và learning rate.
- `EarlyStoppingHook`: Ngắt sớm quá trình huấn luyện khi loss không còn giảm để tránh lãng phí GPU.

### 3.4. Strategy Pattern (`loss/`, `utils/metrics/`)
Tách rời giải thuật tính toán hàm mất mát và giải thuật đo lường hiệu năng thành các chiến lược (Strategy) riêng biệt, dễ dàng thay đổi qua cấu hình.

### 3.5. Composite Pattern (`loss/base_loss.py: LossAggregator`)
Cho phép xử lý một tập hợp nhiều hàm mất mát theo cùng một giao diện như một hàm mất mát đơn lẻ:
$$\text{Total Loss} = w_1 \cdot \text{Loss}_1 + w_2 \cdot \text{Loss}_2 + \dots$$
