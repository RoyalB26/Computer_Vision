# Handwritten Classification (BaseModel Architecture)

Dự án phân loại chữ số viết tay được xây dựng đúng theo chuẩn kiến trúc của framework [BaseModel](file:///D:/Study/Projects/Computer_Vision/BaseModel).

---

## 1. Cấu trúc thư mục dự án

```text
D:\Study\Projects\Computer_Vision\hand_written_classifcation\
├── __init__.py           # Export Dataset và Model
├── dataset.py            # LMDBHandwrittenDataset kế thừa BaseDataset (@DATASETS.register_module())
├── model.py              # SimpleConvClassifier kế thừa BaseModel (@MODELS.register_module())
├── config.yaml           # Cấu hình huấn luyện định dạng YAML chuẩn BaseModel
├── train.py              # Entry-point huấn luyện (kế thừa pipeline BaseModel)
└── README.md             # Tài liệu dự án
```

---

## 2. Các thành phần chuẩn theo BaseModel

1. **Dataset (`dataset.py`)**:
   - Kế thừa [`BaseDataset`](file:///D:/Study/Projects/Computer_Vision/BaseModel/dataset/base_dataset.py) và đăng ký qua `@DATASETS.register_module()`.
   - Đọc dữ liệu từ file LMDB (`train.lmdb` & `test.lmdb`) với kích thước gốc **1200x900**.
   - Pipeline nhẹ nhàng chỉ cần chuyển ảnh sang Tensor, **toàn bộ phép biến đổi downsample phức tạp được đẩy sang GPU**.
   - Nhãn `target` được encode dưới dạng One-Hot FloatTensor cho loss `BCEWithLogitsLoss`.

2. **Model (`model.py`) & `ConvKernelDownsampleModule` trên GPU**:
   - Kế thừa [`BaseModel`](file:///D:/Study/Projects/Computer_Vision/BaseModel/model/base_model.py) và đăng ký qua `@MODELS.register_module()`.
   - **`ConvKernelDownsampleModule`**: Tích hợp trực tiếp ở đầu vào của Model. Sử dụng $n$ tầng Conv2d (depthwise kernel 3x3 toàn 1, stride=2, padding=1) chạy song song trên GPU để nén ảnh gốc $1200 \times 900$ về $64 \times 64$ cực nhanh.
   - Các lớp trích xuất đặc trưng Conv2d + BatchNorm + ReLU.
   - Head phân loại `Linear(num_classes, num_classes)` kết hợp `BCEWithLogitsLoss`.

3. **Huấn luyện qua Config (`config.yaml`)**:
   - Chạy với engine `BaseTrainer` có sẵn của `BaseModel`, hỗ trợ tự động Device management, Logging, Checkpoint, Mixed Precision (AMP), gradient accumulation,...
   - Đã tích hợp metric `Accuracy` (đo lường `acc_top1` và `acc_top5`) tự động trong quá trình validation.

---

## 3. Cách chạy huấn luyện

### Cách 1: Chạy trực tiếp qua train.py của dự án
```bash
cd D:\Study\Projects\Computer_Vision
python hand_written_classifcation/train.py --config hand_written_classifcation/config.yaml
```

### Cách 2: Chạy thông qua BaseModel CLI script
```bash
python BaseModel/scripts/train.py --config hand_written_classifcation/config.yaml --cfg-options train_cfg.max_epochs=10
```
