Dưới đây là bản tổng hợp toàn bộ các giả định cốt lõi, hiện tượng thực nghiệm và kết luận mang tính nguyên lý đã được phân tích và chứng minh qua các chuỗi thử nghiệm:

---

### I. Bài toán thực nghiệm gốc

* **Đầu vào:** Ảnh kích thước $64 \times 64$, chứa một trong 5 lớp đối tượng: Dấu cộng ($+$), Dấu trừ ($-$), Dấu chéo ($x$), Hình tròn rỗng ($o$) kích thước nhỏ $5 \times 5$, và Lớp ảnh rỗng (`empty`).
* **Đặc tính dữ liệu:** Vị trí xuất hiện ngẫu nhiên trên canvas; chuyển dịch từ ảnh Grayscale ($0$ và $1$) sang ảnh RGB 3 kênh với màu nền và nét vẽ ngẫu nhiên.
* **Mục tiêu:** Tìm ra kiến trúc tối giản nhất về mặt tham số nhưng đạt độ chính xác 100% dựa trên đúng tiên nghiệm toán học (Inductive Bias).

---

### II. Các Giả định tiên nghiệm (Inductive Biases) & Kiến trúc

| Thành phần kiến trúc | Giả định tiên nghiệm & Cơ chế toán học | Kết luận qua thực nghiệm |
| :--- | :--- | :--- |
| **Vùng cảm thụ (Receptive Field - RF)** | Vật thể có kích thước $5 \times 5$. Để phân biệt các nét, mô hình cần RF lý thuyết $\ge 5 \times 5$. | • **1 lớp Conv $3 \times 3$** ($\text{RF}=3$): Thất bại vì các mẩu cục bộ của 4 hình bị trùng lặp.<br>• **2 lớp Conv $3 \times 3$ liên tiếp**: Cho $\text{RF} = 1 + 2 + 2 = 5 \times 5$, bao trọn toàn bộ hình học đối tượng. |
| **Flatten + Linear** | Gắn cứng từng pixel tại $(x, y)$ vào một trọng số cố định của ma trận chiếu. | • **Thất bại hoàn toàn:** Dẫn đến học vẹt tọa độ (Overfitting), `val_loss` nổ lên $16 - 18$, `acc` chỉ đạt mức ngẫu nhiên $\approx 40\%$. |
| **Global Max Pooling (`AdaptiveMaxPool2d(1)`)** | Tín hiệu vật thể mang tính thưa thớt (Sparse feature: $5 \times 5$ chỉ chiếm $0.6\%$ diện tích canvas). | • **Cốt lõi để đạt 100%:** Gom giá trị kích hoạt cực đại trên toàn bộ không gian, triệt tiêu hoàn toàn sự phụ thuộc tọa độ, khôi phục tính bất biến dịch chuyển (**Translation Invariance**). |
| **Global Average Pooling (`AdaptiveAvgPool2d(1)`)** | Tính trung bình toàn bộ không gian $(H \times W)$. | • **Thất bại (Acc tụt về 45%):** Gây loãng tín hiệu (Signal Dilution). Giá trị nét vẽ bị chia đều cho 4096 pixel, kéo biên độ về gần 0 và xóa nhòa ranh giới giữa các lớp. |
| **Conv 1x1 (Pointwise)** | Phép biến đổi Affine độc lập trên từng pixel, không can thiệp không gian. | • Đóng vai trò như một **Color Space Projection/Adapter**, biến đổi tổ hợp 3 kênh RGB thành các kênh trừu tượng tối ưu độ tương phản. |
| **Batch Normalization (`BatchNorm2d`)** | Chuẩn hóa mini-batch: $\hat{x} = \frac{x - \mu}{\sigma}$. | • **Tối quan trọng cho RGB đổi màu:** Nền chiếm $>99\%$ diện tích, nên $\mu_B$ chính là màu nền. Phép trừ $\mu_B$ tự động đưa nền về mức $0$, biến nét vẽ thành outlier nổi bật. |
| **Hàm phi tuyến (Activation: ReLU / Hardswish)** | Cung cấp khả năng phân tách phi tuyến giữa các lớp. | • Nếu thiếu Activation giữa 2 lớp Conv $3 \times 3$, hai lớp này suy biến thành một phép tích chập tuyến tính đơn lẻ $\implies$ kẹt ở mức $90\%$. Kích hoạt phi tuyến giúp chạm mốc $100\%$ tại Epoch 60. |
| **Residual Connections ($x + \mathcal{F}(x)$)** | Tạo đường dẫn gradient tắt cho mạng sâu ($\ge 20$ layers). | • **Không cần thiết cho mạng nông 2–3 lớp:** Thậm chí có thể gây hại do cộng ngược thông tin màu nền thô chưa lọc vào feature map. |
| **MinPool** | Lấy giá trị cực tiểu: $\min(x) = -\max(-x)$. | • Hữu ích khi kết hợp $\text{Max} - \text{Min}$ để đo dải động/độ tương phản cục bộ, nhưng MaxPool kết hợp BatchNorm đã đủ để giải quyết bài toán. |

---

### III. Về không gian màu và Hình học đối tượng

1. **Sự trùng lặp đặc trưng ở kernel $3 \times 3$:**
   * Một kernel $3 \times 3$ đơn lẻ không thể phân biệt được:
     * Cạnh ngang của Dấu cộng ($+$) với toàn bộ Dấu trừ ($-$).
     * Góc chéo của Dấu chéo ($x$) với góc uốn của Hình tròn ($o$).
     * Cạnh đứng của Dấu cộng ($+$) với hai vách đứng của Hình tròn ($o$).
     * Tâm rỗng của Hình tròn ($o$) với Ảnh rỗng (`empty`).
   * **Kết luận:** Bắt buộc phải có tầng thứ 2 để tổng hợp ngữ cảnh tương đối giữa các mảnh $3 \times 3$ này.

2. **Vùng cảm thụ thực tế (ERF) và Hình dạng:**
   * Theoretical RF là cố định theo công thức toán, nhưng Effective RF (vùng đóng góp gradient chính) tuân theo phân phối Gauss tập trung ở tâm.
   * Khi hidden dimension bị bóp nghẹt ($C_1=3, C_2=5$), ERF bị co cụm lại:
     * Dấu trừ ($-$) ít bị ảnh hưởng nhất do tập trung trên một trục ngang đơn giản.
     * Dấu cộng ($+$), Dấu chéo ($x$), và Hình tròn ($o$) dễ bị suy giảm khả năng nhận diện nhất vì các nét vươn dài ra biên và phân tán ra 4 góc xa của vùng cảm thụ.

---

### IV. Kết luận triết lý thiết kế mạng nơ-ron

> **"Mô hình lớn không quyết định thành công bằng việc mô hình có bắt đúng bản chất bài toán hay không."**

* Một mô hình hàng triệu tham số với giả định sai (Flattening trên canvas thưa) sẽ học vẹt và thất bại hoàn toàn.
* Một mô hình **chưa tới 1.000 tham số** với cấu hình:

$$\text{Input (3 kênh)} \to \text{Conv 1x1} \to \text{BatchNorm} \to \text{Act} \to \text{Conv 3x3} \to \text{Act} \to \text{Conv 3x3} \to \text{AdaptiveMaxPool2d(1)} \to \text{Linear}$$

đã giải quyết bài toán một cách triệt để, đạt **100% Validation Accuracy** và loss tiệm cận 0 (`0.1495` tại Epoch 60), nhờ thỏa mãn đồng thời:
1. Khử nền động bằng chuẩn hóa thống kê batch.
2. Bao phủ trọn vẹn hình học đối tượng bằng Receptive Field vừa đủ $5 \times 5$.
3. Bất biến dịch chuyển tuyệt đối bằng Global Max Pooling.
