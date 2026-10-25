# 05. Hàm mất mát & Chỉ số đo lường (Loss & Metrics)

## 1. Hệ thống Hàm mất mát (`BaseModel/loss/`)

Mọi hàm loss trong framework đều kế thừa từ `BaseLoss`, hỗ trợ sẵn hệ số trọng số `loss_weight`.

### 1.1. Các hàm Loss có sẵn

1. **`CrossEntropyLoss`**: Phân loại đa lớp với trọng số lớp (`class_weight`) và `ignore_index`.
2. **`BCEWithLogitsLoss`**: Phân loại nhị phân và đa nhãn (multi-label) kèm `pos_weight`.
3. **`SmoothL1Loss`**: Hồi quy tọa độ bounding box trong phát hiện đối tượng, ít nhạy cảm với ngoại lai hơn MSE.
4. **`FocalLoss`**: Giải quyết mất cân bằng lớp trầm trọng trong bài toán phát hiện hoặc phân đoạn vật thể nhỏ:
   $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
5. **`DiceLoss`**: Tối ưu trực tiếp chỉ số giao trên hợp cho phân đoạn ảnh (Semantic Segmentation).

---

### 1.2. `LossAggregator` (Composite Pattern cho đa mục tiêu)

Khi bài toán cần kết hợp nhiều hàm loss với trọng số khác nhau, ta sử dụng `LossAggregator`:

```yaml
loss:
  type: "LossAggregator"
  losses:
    loss_cls:
      type: "FocalLoss"
      alpha: 0.25
      gamma: 2.0
      loss_weight: 1.0
    loss_seg:
      type: "DiceLoss"
      loss_weight: 0.5
```
Trong quá trình huấn luyện, `LossAggregator` sẽ tính toán từng thành phần và trả về:
```python
{
    "loss": total_loss,      # 1.0 * FocalLoss + 0.5 * DiceLoss
    "loss_cls": focal_loss,
    "loss_seg": dice_loss
}
```

---

## 2. Hệ thống Đo lường Chỉ số (Metrics)

Chỉ số đo lường (Metric) kế thừa từ `BaseMetric` và hoạt động theo cơ chế tích lũy qua 3 phương thức:
- **`reset()`**: Xóa bộ đệm ở đầu mỗi lượt đánh giá (validation epoch).
- **`update(preds, targets)`**: Tích lũy số lượng mẫu đúng và tổng số mẫu từ mỗi batch.
- **`compute()`**: Tính toán điểm số phần trăm cuối cùng sau khi đã duyệt qua toàn bộ tập dữ liệu.

### 2.1. Các Metric tích hợp sẵn
- **`Accuracy`**: Hỗ trợ Top-k accuracy (ví dụ `topk: [1, 5]`).
- **`MeanIoU`**: Đo lường mIoU cho phân đoạn ảnh với hỗ trợ bỏ qua nhãn nền `ignore_index`.

---

### 2.2. Cách viết Metric tùy biến (Custom Metric)

Ví dụ: Viết một Metric đo lường độ tương đồng **F1-Score**:

```python
# custom_metrics.py
import torch
from BaseModel.core import METRICS
from BaseModel.utils.metrics import BaseMetric

@METRICS.register_module()
class BinaryF1Score(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(preds)
        pred_binary = (probs > self.threshold).long()

        self.tp += ((pred_binary == 1) & (targets == 1)).sum().item()
        self.fp += ((pred_binary == 1) & (targets == 0)).sum().item()
        self.fn += ((pred_binary == 0) & (targets == 1)).sum().item()

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return {
            "f1_score": f1 * 100.0,
            "precision": precision * 100.0,
            "recall": recall * 100.0
        }
```

Khai báo metric trong cấu hình YAML:
```yaml
metrics:
  f1:
    type: "BinaryF1Score"
    threshold: 0.5
```
Trainer sẽ tự động cập nhật và log `f1_score`, `precision`, `recall` ở mỗi lượt validation.

---

## 3. Tự thiết lập các Chỉ số tùy biến với TensorBoard & WandB

Cả **TensorBoard** và **Weights & Biases (WandB)** trong framework đều hỗ trợ 100% việc tự do thiết lập thêm bất kỳ chỉ số mong muốn nào thông qua 3 cơ chế:

### Cơ chế 1: Tự động ghi nhận từ Custom Metric (Tập Validation)
Mọi Metric kế thừa từ `BaseMetric` khi trả về `float` hoặc `Dict[str, float]` trong hàm `compute()` sẽ **tự động** được đẩy lên cả TensorBoard (`val/<tên_metric>`) và WandB:
```python
# Trong compute() của BaseMetric:
return {
    "mAP_50": 85.4,
    "mAP_75": 62.1,
    "psnr": 32.5
}
# -> Tự động xuất hiện trên TensorBoard: val/mAP_50, val/mAP_75, val/psnr
```

### Cơ chế 2: Tự động ghi nhận từ `forward_train` (Tập Huấn luyện)
Trong hàm `forward_train` của mô hình, bạn có thể trả về bất kỳ chỉ số theo dõi nào cùng với loss:
```python
def forward_train(self, inputs, targets=None, **kwargs):
    logits = self.classifier(self.backbone(inputs))
    loss = self.loss_fn(logits, targets)
    
    # Tính thêm độ chính xác batch hoặc chỉ số mong muốn
    batch_acc = (logits.argmax(dim=-1) == targets).float().mean() * 100.0

    return {
        "loss": loss,
        "batch_acc": batch_acc,     # Tự động log lên train/batch_acc
        "loss_cls": loss
    }
```

### Cơ chế 3: Ghi log Hình ảnh & Chỉ số thủ công qua Hook
Các Hook `TensorboardHook` và `WandbHook` cung cấp sẵn các phương thức tiện ích:
- `hook.log_scalar(tag, value, step)`: Ghi nhận một giá trị số bất kỳ.
- `hook.log_image(tag, image, step)`: Ghi nhận ảnh dự đoán, mask hoặc bounding box trực quan lên bảng điều khiển.
