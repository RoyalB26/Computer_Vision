# Tổng Hợp Các Giả Định & Bản Chất Hình Học Của Attention, Feature Representation và Overfitting

Tài liệu này tổng hợp lại chuỗi thảo luận chuyên sâu về bản chất toán học (đại số tuyến tính, hình học không gian) của Attention, Slot Attention, Biến đổi đặc trưng và Hiện tượng Overfitting từ câu hỏi:
*"Để một slot có attention cao về các token của một object thì các token đó có phải cùng số liệu về vector không?"*

---

## 1. Giả Định 1: Các Token Của Cùng Một Object Không Cần Cùng Số Liệu Vector

### Vấn đề đặt ra
Để một slot (ví dụ slot `truck`) có attention score cao lên toàn bộ các token thuộc về cùng một vật thể (ví dụ các token phủ đầu xe, bánh xe, thùng xe), liệu các token đó có cần mang vector đặc trưng giống hệt nhau không?

### Bản chất toán học & Hình học
- Attention trong Slot Attention được tính qua tích vô hướng:
  $$\text{dots}_{i,j} = \frac{q_i^\top k_j}{\sqrt{d}}, \quad \text{attn}_{i,j} = \text{softmax}_i(\text{dots}_{i,j})$$
- Để hai token $j_1$ và $j_2$ của cùng một vật thể đều nhận attention cao từ slot $i$ (ví dụ `truck`), điều kiện cần và đủ là:
  $$q_i^\top k_{j_1} > q_s^\top k_{j_1} \quad \text{và} \quad q_i^\top k_{j_2} > q_s^\top k_{j_2} \quad (\forall s \neq i)$$
- **Hình học không gian $d$ chiều ($d = 256$):**
  - Mọi vector $k_j$ đều có thể phân rã thành hai thành phần:
    $$k_j = k_{j, \parallel q_i} + k_{j, \perp q_i}$$
    với $k_{j, \perp q_i}$ nằm trong không gian con vuông góc (trực giao) với $q_i$, tức $q_i^\top k_{j, \perp q_i} = 0$.
  - Hai token $j_1, j_2$ **chỉ cần có thành phần chiếu dương lớn theo hướng của $q_i$**.
  - Trên **$d - 1$ chiều trực giao còn lại**, các token hoàn toàn có thể mang giá trị khác nhau (vị trí tọa độ $x, y$, tầng đặc trưng $P_3/P_4/P_5$, màu sắc, góc cạnh cục bộ, kết cấu từng bộ phận).
- **Phân định vai trò Key và Value:**
  - `to_k` (Key): Học phép lọc bỏ sự khác biệt cục bộ (tọa độ, chi tiết lẻ) để giữ lại vector hướng chung của class nhằm so khớp attention.
  - `to_v` (Value): Giữ lại các đặc trưng chi tiết đa dạng để khi tổng hợp có đầy đủ ngữ cảnh phục vụ head dự đoán.

---

## 2. Giả Định 2: Bản Chất Trực Giao & Phân Rã Vector Cơ Sở (Unit Vectors)

### Trực giao (Orthogonal)
- Trực giao trong không gian $n$ chiều chính là khái niệm **vuông góc** mở rộng: tích vô hướng giữa hai vector bằng 0 ($a^\top b = 0$).

### Phân rã theo Unit Vectors
- Mọi vector $k \in \mathbb{R}^n$ đều được cấu tạo từ $n$ unit vector đôi một trực giao và hệ số co giãn tương ứng:
  $$k = c_1 e_1 + c_2 e_2 + \dots + c_n e_n$$
- Giả sử có $M = 11$ queries slot:
  - Hệ 11 vector query này chỉ chiếm tối đa một không gian con **11 chiều**.
  - Nếu chọn hệ trục tọa độ sao cho $e_1, \dots, e_{11}$ phủ không gian của 11 queries, thì chỉ có 11 hệ số $(c_1, \dots, c_{11})$ quyết định điểm attention.
  - Còn $n - 11$ hệ số còn lại $(c_{12}, \dots, c_n)$ thuộc các chiều trực giao tự do biến thiên giữa các token mà **không làm thay đổi điểm attention** với bất kỳ query nào.

---

## 3. Giả Định 3: Về Mặt Lý Thuyết Chỉ Cần 11 Chiều Để Phân Tách 11 Slot?

### Về mặt lý thuyết
- **Đúng.** Về mặt tích vô hướng và phân loại 11 lớp, chỉ cần tối thiểu 10 chiều (đơn hình đều $\mathbb{R}^{10}$) hoặc 11 chiều (hệ trục trực chuẩn $\mathbb{R}^{11}$) là đủ để 11 vector queries phân tách rạch ròi.

### Vì sao thực tế vẫn cần chiều cao hơn (ví dụ $d = 128$ hoặc $256$)?
1. **Tránh nghẽn cổ chai thông tin (Information Bottleneck):** Trước khi vào Slot Attention, mỗi token phải mang thông tin thị giác phức tạp, kết hợp với Positional Encoding 2D ($x, y$) và Level Embeddings. Nếu ép xuống quá thấp (ví dụ 11 chiều), mạng sẽ mất sạch chi tiết không gian.
2. **Không gian biểu diễn cho Value vector:** Vector slot sau khi gom tụ phải giữ đủ thông tin phong phú về kích thước, mật độ, ngữ cảnh để head `fc_actor` hồi quy số lượng đếm.
3. **Cảnh quan tối ưu hóa (Optimization Landscape):** Không gian số chiều cao giúp Gradient Descent dễ tìm thấy nghiệm tách biệt tốt hơn nhiều so với việc bị bó hẹp trong không gian quá nhỏ.

---

## 4. Giả Định 4: Phép Biến Đổi Tuyến Tính Đầu Ra Backbone Về Số Chiều Nhỏ (ví dụ 32) Liệu Có Đủ Tốt?

### Câu hỏi
Liệu một phép chiếu tuyến tính đưa tensor $(B, C, N)$ về $(B, 32, N)$ có đủ để phân tách và có đảm bảo biểu diễn tốt không?

### Phân tích
1. **Nếu chỉ dùng 1 phép chiếu tuyến tính `Conv 1x1` trực tiếp từ Backbone:**
   - **Không đủ tốt**, vì phép chiếu tuyến tính không tạo thêm ranh giới phi tuyến mới và không có receptive field để gom ngữ cảnh vùng xung quanh.
   - Khi nén mạnh (ví dụ $P_5$ từ $576 \to 32$), mạng dễ làm mất các đặc trưng phân biệt giữa các class tương đồng (car vs van vs truck).
2. **Nếu thông qua Neck phi tuyến có ngữ cảnh (như PANet):**
   - Phép biến đổi gồm các khối tích chập $3\times3$, BatchNorm, ReLU kết hợp Top-Down và Bottom-Up giúp lan truyền ngữ nghĩa cấp cao xuống tầng nông và chi tiết không gian lên tầng sâu.
   - Nhờ đó, trước khi vào bước chiếu, các đặc trưng đã được chuyển hóa sang trạng thái dễ phân tách tuyến tính hơn rất nhiều.
3. **Riêng với phép chiếu `to_q` và `to_k`:**
   - Chiếu $Q$ và $K$ xuống không gian trung gian 32 chiều bằng ma trận tuyến tính `Linear(C, 32)` là **hoàn toàn đủ và tối ưu**:
     $$Q K^\top = (S W_Q)(X W_K)^\top = S (W_Q W_K^\top) X^\top$$
   - Ma trận tương quan có rank lên tới 32 ($> 11$ slots), không làm mất bậc tự do phân tách nào nhưng giảm mạnh chi phí tính toán attention trên $N = 18,900$ tokens.

---

## 5. Giả Định 5: Hệ Tọa Độ Của Slot Query Và Các Token Có Giống Nhau Không?

1. **Trước phép chiếu (`to_q`, `to_k`): KHÔNG giống nhau.**
   - Token là biểu diễn từ Backbone + PANet + Positional Encoding.
   - Slot là vector tham số học tự do (`nn.Parameter`), nằm ở hệ tọa độ riêng.
2. **Sau phép chiếu (`to_q`, `to_k`): CHUNG một hệ tọa độ.**
   - Bản chất của $W_Q$ và $W_K$ chính là **hai ánh xạ biến đổi tọa độ** đưa cả hai về chung một không gian đo lường để phép nhân vô hướng $q^\top k$ có ý nghĩa so sánh.
3. **Sau khi nhân với Value:**
   - Slot chuyển sang hệ tọa độ của không gian `Value` (`to_v`) trước khi được đưa vào MLP và head dự đoán.

---

## 6. Giả Định 6: Head Tuyến Tính Cuối Cùng Lựa Chọn Unit Vector Đặc Trưng Của Object

### Bản chất hình học của lớp Linear cuối
- Lớp tuyến tính cuối cùng $\text{Linear}(d, 1)$ có dạng:
  $$y = w^\top h + b = \|w\| \cdot (\hat{w}^\top h) + b, \quad \text{với } \hat{w} = \frac{w}{\|w\|}$$
- **$\hat{w}$ chính là một unit vector (hướng đặc trưng)** trong không gian $d$ chiều đại diện cho thuộc tính cần tìm.
- **$\hat{w}^\top h$** là độ dài hình chiếu của vector biểu diễn $h$ lên hướng đặc trưng đó.
- Toàn bộ các tầng trước (Backbone, Neck, Attention, MLP) làm nhiệm vụ **uốn cong và dịch chuyển không gian** sao cho đặc trưng cốt lõi của object tập trung dọc theo trục $\hat{w}$, giúp phép chiếu vô hướng tách biệt được mục tiêu.

---

## 7. Giả Định 7: Bản Chất Của Overfitting Dưới Góc Nhìn Hình Học

### Định nghĩa hình học của Overfit
- **Overfitting xảy ra khi mạng tìm ra một hướng unit vector $\hat{w}$ phản ánh "đặc trưng ăn may / đường tắt" (shortcut/spurious feature) trên tập train thay vì bản chất thật của object.**
- *Ví dụ:* Mạng học nhận diện `truck` dựa vào bề mặt đường rộng (bối cảnh) thay vì cấu trúc hình học của xe tải, vì đường tắt này giúp giảm loss nhanh nhất trên tập train.
- Khi sang tập val/test, bối cảnh thay đổi, vector $h$ chiếu lên $\hat{w}$ không còn chính xác.
- Ngược lại, **khái quát hóa tốt (Generalization)** là khi:
  - $\hat{w}$ trùng với hướng đặc trưng bất biến của vật thể.
  - Các yếu tố biến thiên ngẫu nhiên (nền, ánh sáng, góc chụp) bị triệt tiêu bằng cách đẩy hết vào **không gian trực giao** với $\hat{w}$.

---

## 8. Giả Định 8: Đầu Ra Của Backbone Không Tự Động Đảm Bảo Tính Bất Biến

- Mạng CNN thuần túy chỉ có tính đẳng biến tịnh tiến cục bộ, **không tự động bất biến** với:
  - Scale (tỉ lệ kích thước).
  - Góc xoay, góc nhìn 3D (đặc biệt trong ảnh drone từ trên cao).
  - Độ sáng, màu sắc nền, bóng đổ.
- Tính bất biến **không phải là thuộc tính tự sinh**, mà là kết quả của việc ép mạng phải học thông qua:
  - Data Augmentation đa dạng.
  - Cấu trúc đa quy mô (FPN/PANet).
  - Hàm loss ràng buộc (như Background/Foreground Mask Loss).
  - Regularization (Weight decay, Dropout).

---
