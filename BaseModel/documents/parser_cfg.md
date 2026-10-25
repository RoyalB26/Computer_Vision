# Tài liệu Tra cứu Tham số Dòng lệnh (CLI Arguments) & Cấu hình

Tài liệu này tổng hợp toàn bộ các tham số dòng lệnh (CLI arguments) được định nghĩa trong hệ thống `BaseModel` (bao gồm `scripts/parser.py`, `scripts/train.py`, `scripts/val.py`, `scripts/test.py`) cùng quy tắc ghi đè cấu hình thông qua `--cfg-options`.

---

## 1. Bảng tra cứu tổng quan các Tham số CLI

| Tham số CLI | Kiểu dữ liệu | Mặc định | Áp dụng cho | Mục đích & Ý nghĩa |
| :--- | :---: | :---: | :---: | :--- |
| `--config` | `str` | *(Bắt buộc)* | `train`, `val`, `test` | Đường dẫn đến file cấu hình (`.yaml` hoặc `.json`). |
| `--work-dir` | `str` | `None` | `train`, `val` | Thư mục lưu trữ logs, file cấu hình lưu trữ và các checkpoint trọng số (`.pth`). Ghi đè khóa `work_dir` trong file YAML. |
| `--resume` | `str` | `None` | `train` | Đường dẫn tới checkpoint (`.pth`) để tiếp tục huấn luyện dở dang (khôi phục epoch, optimizer, lr_scheduler, scaler). |
| `--checkpoint` | `str` | *(Bắt buộc)* | `val`, `test` | Đường dẫn tới file trọng số (`.pth`) cần nạp để đánh giá benchmark hoặc chạy suy luận. |
| `--device` | `str` | `None` | `train`, `val`, `test` | Chỉ định phần cứng tính toán (`cuda`, `cuda:0`, `cpu`, `mps`, `tpu`). Ghi đè khóa `device` trong YAML. |
| `--tpu` | `flag` | `False` | `train`, `val`, `test` | Cờ kích hoạt chạy trên **Google Cloud TPU** thông qua PyTorch XLA (`torch_xla`). Tự động cấu hình `xm.xla_device()`, `xm.optimizer_step()` và `xm.save()`. |
| `--accelerate` | `flag` | `False` | `train` | Kích hoạt **Hugging Face Accelerate** cho phân tán đa GPU (DDP), gradient accumulation, và tối ưu hóa bộ nhớ phần cứng. |
| `--wandb` | `flag` | `False` | `train` | Kích hoạt theo dõi thí nghiệm trực tuyến trên **Weights & Biases (WandB)**: ghi log loss, learning rate, metrics theo thời gian thực. |
| `--tqdm` | `flag` | `False` | `train` | Hiển thị thanh tiến trình trực quan **tqdm** kèm thông số `loss`, `iter/s` trong từng epoch train và validation. |
| `--tensorboard` | `flag` | `False` | `train` | Tự động ghi lại biểu đồ loss, learning rate và scalar metrics vào tệp sự kiện **TensorBoard** (`<work_dir>/tf_logs`). |
| `--seed` | `int` | `None` | `train` | Giá trị hạt giống ngẫu nhiên (seed) cho Python, NumPy, PyTorch CPU/CUDA nhằm đảm bảo tái lập thí nghiệm. Ghi đè `seed` trong YAML. |
| `--eval-only` | `flag` | `False` | `train` | Chỉ thực hiện đánh giá tập validation một lần duy nhất rồi thoát, không chạy vòng lặp huấn luyện. |
| `--input` | `str` | `None` | `test` | Đường dẫn tới file ảnh đơn lẻ hoặc thư mục ảnh cần thực hiện suy luận (inference). |
| `--custom-imports` | `list[str]` | `None` | `train`, `val`, `test` | Danh sách tên module/package Python tùy biến cần import trước khi xây dựng mô hình/dataset để tự động đăng ký vào Registry. |
| `--cfg-options` | `list[str]` | `None` | `train` | Danh sách các cặp `key=value` dùng để ghi đè trực tiếp bất kỳ giá trị nào trong file cấu hình YAML mà không cần sửa file. |

---

## 2. Chi tiết và Cơ chế hoạt động của từng Tham số

### 2.1. `--config`
* **Ý nghĩa**: Xác định tệp cấu hình trung tâm định nghĩa toàn bộ mô hình, dataset, optimizer, scheduler, metrics và hooks.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/resnet50_cifar.yaml
  ```

---

### 2.2. `--work-dir`
* **Ý nghĩa**: Định vị thư mục xuất dữ liệu huấn luyện. Khi truyền tham số này, toàn bộ file `train.log`, `last.pth`, `best.pth`, và bản sao `config_archived.yaml` sẽ được lưu vào thư mục chỉ định.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml --work-dir ./work_dirs/run_gpu_v100
  ```

---

### 2.3. `--resume`
* **Ý nghĩa**: Phục hồi trạng thái huấn luyện khi tiến trình bị gián đoạn (mất điện, hết thời lượng GPU Colab). Khác với việc chỉ nạp trọng số mô hình, `--resume` nạp lại:
  1. Trọng số mô hình (`model.state_dict()`).
  2. Trạng thái bộ tối ưu (`optimizer.state_dict()`) bao gồm momentum, running averages.
  3. Trạng thái lịch trình Learning rate (`lr_scheduler.state_dict()`).
  4. Số thứ tự epoch và iter tiếp theo (`epoch = checkpoint['epoch'] + 1`).
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml --resume work_dirs/exp1/last.pth
  ```

---

### 2.4. `--checkpoint`
* **Ý nghĩa**: Dùng trong `scripts/val.py` và `scripts/test.py`. Chỉ nạp trọng số mô hình (`weights`) để đánh giá độ chính xác hoặc chạy dự đoán, không can thiệp vào optimizer.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/val.py --config configs/exp1.yaml --checkpoint work_dirs/exp1/best.pth
  ```

---

### 2.5. `--device` và `--tpu`
* **Ý nghĩa**:
  - `--device`: Thiết lập thủ công thiết bị phần cứng (`cuda:0`, `cuda:1`, `mps` trên Apple Silicon, hoặc `cpu`).
  - `--tpu`: Bật chế độ tăng tốc trên Google TPU (v2/v3/v4/v5). Khi bật cờ này:
    - Thiết bị tự động chuyển sang `xm.xla_device()`.
    - Bước cập nhật gradient chuyển sang `xm.optimizer_step(optimizer)` và `xm.mark_step()`.
    - Checkpoint được lưu an toàn cho môi trường phân tán qua `xm.save()`.
* **Ví dụ**:
  ```bash
  # Chạy trên GPU 1
  python BaseModel/scripts/train.py --config configs/exp1.yaml --device cuda:1

  # Chạy trên Google TPU
  python BaseModel/scripts/train.py --config configs/exp1.yaml --tpu
  ```

---

### 2.6. `--eval-only`
* **Ý nghĩa**: Tận dụng pipeline của `scripts/train.py` để chạy một lượt validation đầy đủ (tính metric và val_loss) mà không bước vào vòng lặp train. Rất hữu ích để kiểm tra nhanh hiệu năng của model ngay trước khi train tiếp.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml --eval-only
  ```

---

### 2.7. `--custom-imports`
* **Ý nghĩa**: Khi bạn viết Model hoặc Dataset trong thư mục dự án riêng mà chưa import trong code framework, tham số này ra lệnh cho Python nạp các module đó để decorator `@MODELS.register_module()` hoặc `@DATASETS.register_module()` tự động chạy.
* **Cú pháp**: Nhận danh sách các module ngăn cách bởi dấu cách.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml \
      --custom-imports projects.face_recognition.model projects.face_recognition.dataset
  ```
  *(Lưu ý: Bạn cũng có thể khai báo danh sách này trong trường `custom_imports:` của file YAML).*

---

### 2.8. `--cfg-options` (Cực kỳ mạnh mẽ)
* **Ý nghĩa**: Cho phép ghi đè bất kỳ trường nào trong cấu trúc phân tầng của file YAML thông qua cú pháp **dot-notation** (`parent.child=value`).
* **Quy tắc ép kiểu tự động (Auto Type Casting)**:
  - Kiểu số nguyên: `train_cfg.max_epochs=100` -> `int(100)`
  - Kiểu số thực: `optimizer.lr=0.0005` -> `float(0.0005)`
  - Kiểu boolean: `train_cfg.use_amp=True` -> `bool(True)`
  - Kiểu None: `loss=None` -> `None`
  - Kiểu danh sách/Dict JSON: `metrics.acc.topk="[1, 5]"` -> `list([1, 5])`
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml \
      --cfg-options \
      optimizer.lr=0.0002 \
      optimizer.weight_decay=0.05 \
      data.train.batch_size=64 \
      train_cfg.max_epochs=200 \
      train_cfg.val_interval=5 \
      train_cfg.use_amp=True
  ```

---

### 2.9. `--accelerate`
* **Ý nghĩa**: Tích hợp thư viện **Hugging Face Accelerate** vào `BaseTrainer`.
* **Cơ chế**:
  - Tự động chuẩn bị `model, optimizer, train_loader, val_loader` qua `accelerator.prepare()`.
  - Tự động hỗ trợ huấn luyện phân tán đa GPU (Distributed Data Parallel - DDP) hoặc Single-GPU mà không cần thay đổi code.
  - Lan truyền ngược chuẩn hóa qua `accelerator.backward(loss)` và cắt tỉa gradient an toàn.
  - Tự động unwrap model khi lưu checkpoint để đảm bảo tương thích hoàn toàn khi nạp lại.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml --accelerate
  ```

---

### 2.10. `--wandb`
* **Ý nghĩa**: Tự động kích hoạt Hook kết nối và ghi nhận toàn bộ thông số thí nghiệm lên nền tảng **Weights & Biases (WandB)**.
* **Cơ chế**:
  - Tự động khởi tạo phiên làm việc với `wandb.init()` nạp toàn bộ cấu hình siêu tham số.
  - Ghi log liên tục các giá trị `loss`, `learning rate` theo từng iter (`train/loss`, `train/lr`).
  - Ghi log các chỉ số đánh giá `val/acc`, `val/mIoU` theo từng epoch.
  - Đóng phiên an toàn qua `wandb.finish()` khi kết thúc huấn luyện.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml --wandb
  ```

---

### 2.11. `--tqdm`
* **Ý nghĩa**: Bọc `DataLoader` bằng thanh tiến trình trực quan `tqdm` trên Console thay vì chỉ in log dạng text định kỳ.
* **Cơ chế**:
  - Hiển thị tỷ lệ phần trăm hoàn thành, số lượng batch đã chạy/tổng batch trong từng epoch.
  - Hiển thị postfix cập nhật loss tức thời theo thời gian thực (realtime).
  - Tự động chuyển đổi mượt mà giữa thanh tiến trình của tập Train và tập Val.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml --tqdm
  ```

---

### 2.12. `--tensorboard`
* **Ý nghĩa**: Kích hoạt `TensorboardHook` lưu trữ đồ thị sự kiện (event logs) để trực quan hóa bằng TensorBoard.
* **Cơ chế**:
  - Tạo tệp sự kiện tại `<work_dir>/tf_logs`.
  - Ghi lại đường cong mất mát (`train/loss`), lịch trình học (`train/lr`) và chỉ số kiểm thử (`val/<metric>`).
  - Có thể mở giao diện xem trực tiếp bằng lệnh: `tensorboard --logdir ./work_dirs/my_exp/tf_logs`.
* **Ví dụ**:
  ```bash
  python BaseModel/scripts/train.py --config configs/exp1.yaml --tensorboard
  ```

---

## 3. Bảng đối chiếu Cờ CLI và Khóa trong file YAML

Các cờ dòng lệnh có độ ưu tiên cao nhất (**High Priority**), sẽ ghi đè lên các giá trị tương ứng được định nghĩa trong file YAML:

| Khóa trong file YAML | Tham số dòng lệnh CLI tương ứng | Thứ tự ưu tiên |
| :--- | :--- | :---: |
| `work_dir: "./work_dirs/exp1"` | `--work-dir ./work_dirs/override` | CLI > YAML |
| `device: "cuda"` | `--device cpu` hoặc `--tpu` | CLI > YAML |
| `seed: 42` | `--seed 1234` | CLI > YAML |
| `custom_imports: [...]` | `--custom-imports module1 module2` | CLI > YAML |
| Bất kỳ trường nào (`model.num_classes`, `optimizer.lr`,...) | `--cfg-options path.to.key=value` | CLI > YAML |

---

## 4. Bảng tổng hợp các lệnh thông dụng (Cheat Sheet)

```bash
# 1. Huấn luyện cơ bản
python BaseModel/scripts/train.py --config configs/baseline.yaml

# 2. Huấn luyện trên Google TPU
python BaseModel/scripts/train.py --config configs/baseline.yaml --tpu

# 3. Huấn luyện ghi đè LR và Batch size
python BaseModel/scripts/train.py --config configs/baseline.yaml \
    --cfg-options optimizer.lr=0.0001 data.train.batch_size=32

# 4. Tiếp tục huấn luyện sau khi bị gián đoạn
python BaseModel/scripts/train.py --config configs/baseline.yaml \
    --resume work_dirs/baseline/last.pth

# 5. Đánh giá checkpoint tốt nhất trên tập Validation
python BaseModel/scripts/val.py --config configs/baseline.yaml \
    --checkpoint work_dirs/baseline/best.pth

# 6. Đánh giá checkpoint trên TPU
python BaseModel/scripts/val.py --config configs/baseline.yaml \
    --checkpoint work_dirs/baseline/best.pth --tpu

# 7. Chạy dự đoán suy luận trên 1 ảnh
python BaseModel/scripts/test.py --config configs/baseline.yaml \
    --checkpoint work_dirs/baseline/best.pth \
    --input ./data/test_samples/sample_01.jpg

# 8. Huấn luyện kèm thanh tiến trình tqdm và Accelerate
python BaseModel/scripts/train.py --config configs/baseline.yaml --accelerate --tqdm

# 9. Huấn luyện ghi log TensorBoard và WandB cùng lúc
python BaseModel/scripts/train.py --config configs/baseline.yaml --tensorboard --wandb --tqdm
```
