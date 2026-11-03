# VisDrone YOLOv11 (DIoU Loss + Focal Loss)

Dự án phát hiện vật thể (Object Detection) trên tập dữ liệu ảnh từ thiết bị bay không người lái (**VisDrone**) sử dụng kiến trúc **YOLO11** kết hợp **DIoU Loss** (Distance-IoU) và **Sigmoid Focal Loss**, kế thừa toàn bộ khung chuẩn hóa và design pattern từ framework `BaseModel` và cơ chế nạp dữ liệu từ `Visdrone`.

---

## 1. Kiến trúc & Công thức hàm mất mát (Loss)

### 1.1 Bounding Box Loss: Distance-IoU (DIoU Loss)
Đối với ảnh flycam/drone (VisDrone), các vật thể thường có kích thước rất nhỏ và mật độ dày đặc. DIoU bổ sung hình phạt về khoảng cách Euclidean giữa tâm của hộp dự đoán và hộp nhãn thực tế, chuẩn hóa theo đường chéo của hộp bao nhỏ nhất:

$$\mathcal{L}_{DIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2}$$

Trong đó:
- $IoU = \frac{|b \cap b^{gt}|}{|b \cup b^{gt}|}$
- $\rho(b, b^{gt}) = \sqrt{(cx - cx^{gt})^2 + (cy - cy^{gt})^2}$ là khoảng cách giữa 2 tâm bounding box.
- $c = \sqrt{(w^c)^2 + (h^c)^2}$ là độ dài đường chéo của convex hull bao bọc cả hai box.

### 1.2 Classification Loss: Sigmoid Focal Loss
VisDrone gặp phải sự mất cân bằng cực lớn giữa background và foreground (hàng nghìn anchor points trên bầu trời, mặt đất chỉ có vài chục vật thể nhỏ). Focal Loss triệt tiêu gradient của các mẫu dễ (easy negatives / background) và tập trung vào các mẫu khó (hard examples):

$$\mathcal{L}_{Focal} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$

Trong đó:
- $p_t = p \cdot y + (1 - p) \cdot (1 - y)$
- $\alpha_t = \alpha \cdot y + (1 - \alpha) \cdot (1 - y)$ với $\alpha = 0.25, \gamma = 2.0$.

### 1.3 Target Assignment: Task-Aligned Assigner (TAL)
Sử dụng bộ gán nhãn động TAL (tương tự chuẩn YOLOv8/11):
- Đánh giá anchor nằm trong GT Box kết hợp chỉ số tương đồng:

$$t = s^\alpha \times IoU^\beta \quad (\alpha=0.5, \beta=6.0)$$

- Tự động chọn Top-K anchor tốt nhất cho từng bounding box và gán nhãn soft score.

---

## 2. Cấu trúc thư mục `VisDrone_yolov11/`

```
VisDrone_yolov11/
├── __init__.py           # Export các module và đăng ký vào Registries
├── config.yaml           # Cấu hình huấn luyện chuẩn theo BaseModel
├── dataset.py            # Kế thừa CustomLMDBDataset từ Visdrone + hỗ trợ Folder/Txt dataset
├── loss.py               # Hiện thực DIoULoss, SigmoidFocalLoss, DFL và YOLOv11DetectionLoss
├── model.py              # Lớp VisDroneYOLO11 kế thừa BaseModel, nạp yolo11n.pt, forward_train/val/predict
├── metrics.py            # Đo lường mAP_50, mAP_75, mAP_50_95 theo chuẩn COCO/VisDrone
├── run.py                # Entry-point huấn luyện CLI tích hợp BaseModel
├── predict.py            # Script suy luận, vẽ bounding box và lưu ảnh kết quả
├── test_model.py         # Bộ kiểm thử đơn vị (Unit Tests) tự động cho model, loss và metric
└── README.md             # Tài liệu hướng dẫn chi tiết
```

---

## 3. Các lớp đối tượng trong tập dữ liệu VisDrone (10 Classes)

| ID | Tên Lớp | Màu hiển thị |
|:---|:---|:---|
| 0 | `pedestrian` (Người đi bộ) | Đỏ |
| 1 | `people` (Người đứng/ngồi) | Đỏ nhạt |
| 2 | `bicycle` (Xe đạp) | Cam |
| 3 | `car` (Xe ô tô con) | Hổ phách |
| 4 | `van` (Xe tải nhỏ/bán tải) | Vàng chanh |
| 5 | `truck` (Xe tải lớn) | Xanh lá |
| 6 | `tricycle` (Xe ba bánh) | Xanh nõn chuối |
| 7 | `awning-tricycle` (Xe ba bánh có mui) | Xanh lục lam |
| 8 | `bus` (Xe buýt) | Xanh dương |
| 9 | `motorcycle` (Xe máy) | Tím |

---

## 4. Hướng dẫn sử dụng

### 4.1 Chạy Unit Test kiểm tra toàn bộ hệ thống
```bash
python VisDrone_yolov11/test_model.py
```

### 4.2 Huấn luyện mô hình (Training)
Chạy trực tiếp file `run.py` (tự động nạp file `config.yaml`):
```bash
python VisDrone_yolov11/run.py
```

Ghi đè đường dẫn dữ liệu hoặc tham số thông qua CLI:
```bash
python VisDrone_yolov11/run.py --cfg-options data.train.dataset.data_root="D:/Datasets/visdrone.lmdb" data.train.batch_size=8 train_cfg.max_epochs=50
```

Hoặc gọi thông qua script tổng của framework `BaseModel`:
```bash
python BaseModel/scripts/train.py --config VisDrone_yolov11/config.yaml
```

### 4.3 Đánh giá mô hình đã huấn luyện (Validation / Benchmark)
```bash
python BaseModel/scripts/val.py --config VisDrone_yolov11/config.yaml --checkpoint work_dirs/visdrone_yolov11/best.pth
```

### 4.4 Chạy dự đoán & trực quan hóa (Inference & Visualization)
Chạy dự đoán trên 1 bức ảnh:
```bash
python VisDrone_yolov11/predict.py --image path/to/drone_image.jpg --weights yolo11n.pt --out-dir ./outputs --conf-th 0.25
```

Chạy dự đoán trên toàn bộ thư mục ảnh:
```bash
python VisDrone_yolov11/predict.py --image path/to/image_folder/ --weights work_dirs/visdrone_yolov11/best.pth --out-dir ./outputs
```
