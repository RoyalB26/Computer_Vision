# BaseModel: Modular & Extensible Computer Vision Framework

`BaseModel` là một framework thị giác máy tính (Computer Vision) chuẩn hóa, áp dụng sâu sắc các nguyên lý **OOP**, **Polymorphism (Tính đa hình)** và các **Software Design Patterns** (Registry, Template Method, Observer/Hook, Strategy, Composite).

Mục tiêu cốt lõi của `BaseModel` là:
> **Xây dựng khung xương (skeleton) chuẩn hóa cao nhất.** Khi phát triển một bài toán thị giác máy tính mới (Image Classification, Object Detection, Semantic Segmentation, Pose Estimation,...), lập trình viên **chỉ cần kế thừa BaseModel & BaseDataset**, khai báo kiến trúc mạng và định dạng nhãn riêng của bài toán. Toàn bộ vòng lặp Training, Validation, Checkpoint, Metric Tracking, Learning Rate Scheduling, AMP (Mixed Precision), CLI argument parsing... đều được tự động hóa 100%.

---

## 1. Cấu trúc thư mục

```
BaseModel/
├── __init__.py
├── README.md                      # Tài liệu hướng dẫn kiến trúc & sử dụng
├── core/                          # Modular registry & training engine
│   ├── __init__.py
│   ├── registry.py                # Lớp Registry ánh xạ chuỗi cấu hình -> Class (Registry Pattern)
│   ├── registrar.py               # Các Registry toàn cục: MODELS, DATASETS, LOSSES, METRICS, HOOKS...
│   ├── trainer.py                 # BaseTrainer: Training engine cốt lõi (Template Method Pattern)
│   └── hooks/                     # Hệ thống Hook (Observer Pattern)
│       ├── __init__.py
│       ├── base_hook.py           # Interface BaseHook
│       ├── checkpoint_hook.py     # Tự động lưu checkpoint định kỳ & checkpoint tốt nhất (best.pth)
│       ├── logger_hook.py         # Ghi log tiến độ huấn luyện ra màn hình & file log
│       ├── lr_scheduler_hook.py   # Tự động cập nhật learning rate theo epoch/iter
│       └── early_stopping_hook.py # Dừng huấn luyện sớm nếu metric validation không cải thiện
├── dataset/                       # Chuẩn hóa nạp dữ liệu, Collate & Helpers
│   ├── __init__.py
│   ├── base_dataset.py            # Lớp trừu tượng BaseDataset (Template Method Pattern)
│   ├── helpers.py                 # Bộ hàm hỗ trợ: load_image, find_images, parse_csv, open_lmdb,...
│   ├── collate.py                 # Bộ gom batch tùy biến: SmartCollate, PadCollate, DetectionCollate
│   ├── builder.py                 # build_dataset, build_dataloader, build_collate_fn
│   └── transforms/                # Pipeline xử lý & tăng cường dữ liệu (Compose Pattern)
│       ├── __init__.py
│       └── compose.py             # Compose, ToTensor, Normalize, Resize
├── model/                         # Kiến trúc mô hình chuẩn
│   ├── __init__.py
│   ├── base_model.py              # Interface trừu tượng BaseModel (Polymorphism: train/val/predict)
│   └── builder.py                 # build_model từ cấu hình Registry
├── loss/                          # Hàm mất mát mô-đun hóa
│   ├── __init__.py
│   ├── base_loss.py               # BaseLoss & LossAggregator (Composite Pattern cho đa hàm mất mát)
│   ├── standard_losses.py         # CrossEntropy, BCEWithLogits, SmoothL1, FocalLoss, DiceLoss
│   └── builder.py                 # build_loss từ cấu hình Registry
├── utils/                         # Công cụ bổ trợ
│   ├── __init__.py
│   ├── config.py                  # Trình nạp cấu hình YAML/JSON với cú pháp dot-notation & ghi đè CLI
│   ├── logger.py                  # Trình ghi log chuẩn
│   ├── seed.py                    # Khởi tạo seed đồng bộ (torch, numpy, cudnn)
│   ├── device.py                  # Quản lý thiết bị tính toán (CUDA / MPS / CPU)
│   ├── checkpoint.py              # Tiện ích lưu / nạp trọng số (.pth)
│   ├── math_tools.py              # Các phép tính hình học/toán học: IoU, xywh2xyxy, one-hot...
│   └── metrics/                   # Hệ thống đo lường metrics (Strategy Pattern)
│       ├── __init__.py
│       ├── base_metric.py         # Interface BaseMetric (update, compute, reset)
│       ├── standard_metrics.py    # Accuracy (top-k), MeanIoU (mIoU)
│       └── builder.py             # build_metric từ Registry
└── scripts/                       # Entry-points thực thi
    ├── __init__.py
    ├── parser.py                  # Trình phân tích tham số dòng lệnh CLI tiêu chuẩn
    ├── train.py                   # Script huấn luyện tự động đa năng
    ├── val.py                     # Script benchmark & đánh giá checkpoint
    └── test.py                    # Script chạy suy luận (Inference)
```

---

## 2. Các Design Patterns áp dụng

1. **Registry Pattern (`core/registry.py`, `core/registrar.py`)**:
   - Tách rời hoàn toàn sự phụ thuộc giữa mã nguồn và cấu hình.
   - Bất kỳ Model, Dataset, Loss, Metric hay Hook nào chỉ cần gắn decorator `@MODELS.register_module()` hoặc `@DATASETS.register_module()`.
   - File cấu hình `.yaml` chỉ cần chỉ định tên chuỗi (ví dụ: `type: "ResNet18"`), hệ thống sẽ tự động tìm và khởi tạo đối tượng tương ứng.

2. **Template Method Pattern (`core/trainer.py`, `dataset/base_dataset.py`)**:
   - Trong `BaseTrainer`: Khung vòng lặp chính `train()` cố định các bước:
     `before_train` -> lặp epoch -> `train_step()` -> `val_step()` -> cập nhật scheduler -> lưu checkpoint -> `after_train`.
     Các bài toán đặc thù chỉ việc override `train_step()` nếu cần.
   - Trong `BaseDataset`: Quy trình nạp ảnh và áp dụng biến đổi được chuẩn hóa qua `prepare_data(idx)` và `collate_fn()`. Lớp con chỉ cần hiện thực duy nhất `load_annotations()`.

3. **Observer / Hook Pattern (`core/hooks/`)**:
   - Tách rời các logic bổ trợ (lưu model, ghi log console, ghi nhận TensorBoard, early stopping) ra khỏi vòng lặp train.
   - Dễ dàng bật/tắt hoặc viết thêm Hook mới mà không cần chỉnh sửa một dòng code nào trong `BaseTrainer`.

4. **Strategy Pattern (`loss/`, `utils/metrics/`)**:
   - Các hàm tính loss (`BaseLoss`) và metric (`BaseMetric`) có thể hoán đổi linh hoạt thông qua cấu hình mà không làm thay đổi luồng điều khiển của `BaseTrainer`.

5. **Composite Pattern (`loss/base_loss.py: LossAggregator`)**:
   - Cho phép kết hợp nhiều hàm loss với trọng số khác nhau (ví dụ: `0.7 * Classification_Loss + 0.3 * Regression_Loss`) thành một đối tượng duy nhất, tự động tổng hợp loss dictionary.

6. **Polymorphic Model Interface (`model/base_model.py`)**:
   - `BaseModel` định nghĩa 3 chế độ thực thi chuẩn hóa:
     - `forward_train(inputs, targets)`: Trả về loss dictionary phục vụ gradient descent.
     - `forward_val(inputs, targets)`: Trả về predictions + validation loss phục vụ tính metric.
     - `predict(inputs)`: Thực hiện inference với post-processing (argmax, softmax, NMS).

---

## 3. Hướng dẫn kế thừa cho một bài toán mới

Khi giải quyết một bài toán mới (ví dụ: `projects/dog_cat_classification/`):

### Bước 1: Kế thừa `BaseDataset`
```python
# projects/dog_cat_classification/dataset.py
from BaseModel.core import DATASETS
from BaseModel.dataset import BaseDataset

@DATASETS.register_module()
class DogCatDataset(BaseDataset):
    def load_annotations(self):
        # Đọc dữ liệu từ file csv/json/thư mục
        data_infos = []
        # Duyệt và trả về list dict: [{'img_path': 'path/to/cat.jpg', 'target': 0}, ...]
        return data_infos
```

### Bước 2: Kế thừa `BaseModel`
```python
# projects/dog_cat_classification/model.py
import torch.nn as nn
from BaseModel.core import MODELS
from BaseModel.model import BaseModel

@MODELS.register_module()
class DogCatNet(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(...)
        self.classifier = nn.Linear(512, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward_train(self, inputs, targets=None, **kwargs):
        logits = self.classifier(self.backbone(inputs))
        loss = self.criterion(logits, targets)
        return {"loss": loss, "loss_cls": loss}

    def predict(self, inputs, **kwargs):
        return self.classifier(self.backbone(inputs))
```

### Bước 3: Tạo file `config.yaml`
```yaml
work_dir: "./work_dirs/dog_cat_exp1"
device: "cuda"

custom_imports:
  - "projects.dog_cat_classification.model"
  - "projects.dog_cat_classification.dataset"

model:
  type: "DogCatNet"
  num_classes: 2

data:
  train:
    dataset:
      type: "DogCatDataset"
      data_root: "./data/train"
      pipeline:
        - type: "Resize"
          size: [224, 224]
        - type: "ToTensor"
        - type: "Normalize"
    batch_size: 32
    shuffle: true

  val:
    dataset:
      type: "DogCatDataset"
      data_root: "./data/val"
      pipeline:
        - type: "Resize"
          size: [224, 224]
        - type: "ToTensor"
        - type: "Normalize"
    batch_size: 32

optimizer:
  type: "AdamW"
  lr: 0.0005
  weight_decay: 0.01

metrics:
  acc:
    type: "Accuracy"
    topk: [1]

train_cfg:
  max_epochs: 30
  val_interval: 1
```

### Bước 4: Chạy huấn luyện & đánh giá trực tiếp!
Không cần viết lại bất kỳ hàm train loop nào, chỉ cần gọi CLI:

```bash
# Huấn luyện mô hình
python BaseModel/scripts/train.py --config projects/dog_cat_classification/config.yaml

# Ghi đè tham số qua CLI
python BaseModel/scripts/train.py --config projects/dog_cat_classification/config.yaml --cfg-options train_cfg.max_epochs=50 optimizer.lr=0.001

# Đánh giá checkpoint
python BaseModel/scripts/val.py --config projects/dog_cat_classification/config.yaml --checkpoint work_dirs/dog_cat_exp1/best.pth
```
