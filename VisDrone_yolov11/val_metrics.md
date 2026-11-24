# Tài liệu Toàn bộ Chỉ số Đánh giá Validation (`val_visdrone_components.py`)

Tài liệu này tổng hợp đầy đủ **tất cả các chỉ số** được tính toán và xuất ra bởi [val_visdrone_components.py](val_visdrone_components.py), phân theo từng file CSV đầu ra:

1. `detection_overall_metrics.csv` & `per_class_ap.csv` — Hiệu năng phát hiện End-to-End (COCO mAP, P/R/F1, tốc độ, tham số).
2. `component_breakdown.csv` — Sức khỏe đặc trưng theo từng khối Backbone / Neck / Slot+FiLM / Head.
3. `box_learning_diagnostics.csv` — Cơ chế học Bounding Box (Boundary, TAL, DFL, Xung đột Gradient).
4. `slot_overfit_diagnostics.csv` — Chẩn đoán Overfit & Đóng góp của Slot Attention + FiLM.
5. `slot_bottleneck_stages.csv` — So sánh độ phân tách lớp qua 3 bước `Neck → proj_in → Key`.
6. `slot_per_class_diagnostics.csv` — Chẩn đoán chi tiết từng lớp VisDrone (0..9) theo từng tầng quy mô `P3/P4/P5`.

---

## 1. Hiệu năng Phát hiện End-to-End (`detection_overall_metrics.csv` & `per_class_ap.csv`)

| Chỉ số | Công thức | Ý nghĩa | Khoảng giá trị & Mốc tốt |
|---|---|---|---|
| **`mAP50`** | $\frac{1}{K}\sum_{c=0}^{K-1} \text{AP}_{c}^{\text{IoU}=0.50}$ (COCO 101-point) | Độ chính xác phát hiện trung bình 10 lớp tại ngưỡng trùng khớp $\text{IoU} \ge 0.50$. | **`[0, 1]`**. Càng cao càng tốt (VisDrone YOLOv11n 960px tốt quanh `0.32 – 0.42+`). |
| **`mAP50_95`** | $\frac{1}{10}\sum_{t \in \{0.50, 0.55, \dots, 0.95\}} \text{mAP}^{t}$ | mAP chuẩn COCO trung bình trên 10 ngưỡng IoU, đo cả phân loại lẫn độ khít box. | **`[0, 1]`**. Càng cao càng tốt (tốt quanh `0.16 – 0.24+`). |
| **`mAP50_95_small`** | $\text{mAP}_{50:95}$ lọc riêng GT có diện tích $A < 32^2\text{ px}$ | Hiệu năng trên vật thể nhỏ (nhóm chiếm đa số trên ảnh drone VisDrone). | **`[0, 1]`**. Càng cao càng tốt (`> 0.10` là khá). |
| **`mAP50_95_medium`** | $\text{mAP}_{50:95}$ lọc riêng GT có $32^2 \le A < 96^2\text{ px}$ | Hiệu năng trên vật thể kích thước trung bình. | **`[0, 1]`**. Càng cao càng tốt (`> 0.20` là khá). |
| **`mAP50_95_large`** | $\text{mAP}_{50:95}$ lọc riêng GT có $A \ge 96^2\text{ px}$ | Hiệu năng trên vật thể lớn (`bus`, `truck` gần camera). | **`[0, 1]`**. Càng cao càng tốt. |
| **`precision_c25`** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ tại $\text{conf} \ge 0.25,\ \text{IoU} \ge 0.50$ | Tỷ lệ dự đoán đúng trong tổng số box model xuất ra ở ngưỡng tin cậy 0.25. | **`[0, 1]`**. Càng cao càng ít báo giả (`> 0.55` là tốt). |
| **`recall_c25`** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ tại $\text{conf} \ge 0.25,\ \text{IoU} \ge 0.50$ | Tỷ lệ vật thể thật được phát hiện ra ở ngưỡng tin cậy 0.25. | **`[0, 1]`**. Càng cao càng ít bỏ sót. |
| **`f1_c25`** | $\frac{2 \cdot P \cdot R}{P + R}$ tại $\text{conf} \ge 0.25$ | Trung bình điều hòa cân bằng giữa Precision và Recall. | **`[0, 1]`**. Càng cao càng tốt. |
| **`total_gt` / `active_classes`** | Tổng số GT box / số lớp có GT trong tập đánh giá | Xác nhận quy mô tập đánh giá (VisDrone2019-DET-val đủ 548 ảnh có `38,759` GT, `10` lớp). | Số nguyên dương. |
| **`latency_ms` / `fps`** | Thời gian suy luận trung bình mỗi ảnh (ms) / $1000 / \text{latency}$ | Tốc độ inference thực tế trên GPU/CPU hiện tại (đã đồng bộ CUDA). | `ms > 0`, `FPS > 0`. Real-time khi `FPS ≥ 30` (`latency ≤ 33 ms`). |
| **`params_total_m` / `backbone` / `neck` / `head`** | Tổng số tham số chia cho $10^6$ (triệu tham số) | Kích thước từng phân hệ của mô hình. | `> 0` (đơn vị: triệu params, vd `2.64M`). |
| **`head_share_p3 / p4 / p5_pct`** | $\frac{\sum \max_c \sigma(\text{cv3}_s)}{\sum_{s' \in \{3,4,5\}} \sum \max_c \sigma(\text{cv3}_{s'})} \times 100\%$ | Tỷ trọng xác suất phát hiện đến từ từng tầng đầu ra `P3 (//8)`, `P4 (//16)`, `P5 (//32)`. | **`[0%, 100%]`**, tổng 3 tầng = `100%`. VisDrone nên có P3 hoạt động mạnh (`> 20%`). |
| **`ap50` / `ap50_95` (`per_class_ap.csv`)** | $\text{AP}_{c}^{0.50}$ và $\text{AP}_{c}^{50:95}$ riêng từng lớp $c$ | Hiệu năng phát hiện chi tiết của từng trong 10 lớp VisDrone. | **`[0, 1]`**. Dùng phát hiện lớp yếu / lệch phân phối. |

---

## 2. Phân tích Đặc trưng Từng Thành phần (`component_breakdown.csv`)

Đo trên từng tầng từ `backbone.stage0 → stage10`, `neck.c3k2_13..22`, `slot.cls_p3..p5`, `head.cv2_p3..p5`, đến `head.cv3_p3..p5`.

| Chỉ số | Công thức | Ý nghĩa | Khoảng giá trị & Mốc tốt |
|---|---|---|---|
| **`params_k`** | Số tham số của module chia cho $10^3$ | Dung lượng tham số của thành phần (đơn vị: nghìn params). | `≥ 0`. |
| **`act_mean` / `act_std`** | $\text{mean}(F)$, $\text{std}(F)$ | Giá trị trung bình và độ lệch chuẩn của activation map. | $(-\infty, +\infty)$ / $[0, +\infty)$. Kiểm tra bùng nổ hay triệt tiêu activation. |
| **`neg_frac` / `zero_frac`** | $\text{mean}(F < 0)$, $\text{mean}(\|F\| < 10^{-6})$ | Tỷ lệ phần tử âm / chết bằng 0 (dead neurons). | **`[0, 1]`**. `zero_frac` quá cao (`> 0.8`) = chết neuron; `SiLU` thường có `neg_frac ~ 0.3–0.5`. |
| **`l2_mean`** | $\text{mean}_{h,w}\|F_{:,h,w}\|_2$ | Độ lớn năng lượng đặc trưng trung bình tại mỗi vị trí không gian. | $[0, +\infty)$. Ổn định, không tăng vọt bất thường giữa các tầng. |
| **`all_ratio`** | $\frac{\text{mean}_{i \in \text{FG}}(E_i)}{\text{mean}_{i \in \text{BG}}(E_i)}$ (năng lượng $E_i = \|F_i\|_2$ hoặc $\max\sigma(\text{cv3})$) | Tỷ lệ tương phản tín hiệu Vật thể / Nền (FG-to-BG contrast). | $[0, +\infty)$. **`> 1.0`** là tập trung vào vật thể; ở `head.cv3` càng lớn càng tốt (`> 100`). |
| **`small / medium / large_ratio`** | $\frac{\text{mean}_{i \in \text{FG}_{\text{size}}}(E_i)}{\text{mean}_{i \in \text{BG}}(E_i)}$ | Tỷ lệ tương phản FG/BG tách riêng theo kích thước vật thể (nhỏ / vừa / lớn). | $[0, +\infty)$. `small_ratio` phải cao ở P3, `large_ratio` cao ở P5. |
| **`fg_mass_pct`** | $\frac{\sum_{i \in \text{FG}} E_i}{\sum_{\text{all } i} E_i} \times 100\%$ | Phần trăm tổng năng lượng đặc trưng (hoặc xác suất `cv3`) rơi đúng vào trong GT boxes. | **`[0%, 100%]`**. Ở `head.cv3` càng cao càng tốt (`> 70%` ở P3, `> 85%` ở P4/P5). |
| **`full_fisher`** | $\frac{\text{Var}_{\text{between-class}}(F)}{\text{Var}_{\text{within-class}}(F)}$ trên toàn bộ không gian kênh | Chỉ số Fisher đo khả năng phân tách 10 lớp của feature map. | $[0, +\infty)$. Càng **cao** các lớp càng tách biệt rõ. |
| **`pca_fisher`** | Fisher Ratio tính trên 3 thành phần chính đầu tiên (PCA-3D) | Khả năng phân tách lớp trong không gian nén thấp chiều nhất. | $[0, +\infty)$. Càng cao càng dễ phân loại tuyến tính. |
| **`class_cos_sep`** | $1 - \text{mean}_{c_1 \neq c_2}\cos(\mu_{c_1}, \mu_{c_2})$ | Khoảng cách góc cosine trung bình giữa vector tâm các lớp. | **`[0, 2]`**. Càng lớn (hướng tới `1.0`) các lớp càng trực giao. |

---

## 3. Chẩn đoán Cơ chế Học Box (`box_learning_diagnostics.csv`)

| Chỉ số | Công thức | Ý nghĩa | Khoảng giá trị & Mốc tốt |
|---|---|---|---|
| **`boundary_contrast_p3 / p4 / p5`** | $\frac{\text{mean}_{i \in \text{border}}\|\nabla_{x,y} F_i\|_2}{\text{mean}_{i \in \text{interior}}\|\nabla_{x,y} F_i\|_2}$ | Độ sắc nét biên vật thể so với ruột vật thể trên Neck P3/P4/P5 (`border = dilate(FG) - erode(FG)`). | $[0, +\infty)$. **`> 1.0`** là biên sắc nét hơn ruột (giúp định vị box chính xác). |
| **`intra_box_decorr_p3 / p4 / p5`** | $\text{mean}_{(i,j) \in \text{adj}(\text{FG})}\big(1 - \cos(\hat{F}_i, \hat{F}_j)\big)$ | Mức độ khác biệt giữa 2 ô kề nhau *bên trong* cùng 1 GT box. | **`[0, 2]`**. Quá gần `0` = bị làm phẳng mất tọa độ tương đối; quá cao (`> 0.7`) = nhiễu. |
| **`gt_coverage_all / small_pct`** | $\frac{\|\{g \in \text{GT}:\ \exists \text{ anchor center} \in g\}\|}{\|\text{GT}\|} \times 100\%$ | Tỷ lệ GT (tất cả / vật thể nhỏ) chứa ít nhất 1 tâm anchor hợp lệ về mặt hình học. | **`[0%, 100%]`**. Tốt **`> 95%`** (nếu thấp ở `small` → cần tăng `imgsz` hoặc thêm P2 stride-4). |
| **`avg_pos_per_gt`** | $\frac{\sum \text{fg\_mask}}{M}$ | Số anchor dương tính trung bình mà `TaskAlignedAssigner` gán cho mỗi GT box. | $[0, \text{topk}]$. Khỏe mạnh quanh **`5 – 9`** (`< 2` là thiếu tín hiệu học box). |
| **`dfl_clipping_rate_pct`** | $\text{mean}\big(\text{ltrb}_{\text{target}} > \text{reg\_max} - 1\big) \times 100\%$ | Tỷ lệ cạnh GT vượt quá giới hạn `reg_max - 1 = 15` bước stride của DFL (bị cắt ngọn). | **`[0%, 100%]`**. Tốt **`< 1%`** (`≈ 0%` là hoàn hảo). |
| **`dfl_entropy_pos`** | $-\frac{1}{\ln 16}\sum_{b=0}^{15} p_b \ln p_b$ trên positive anchors | Độ không chắc chắn của phân phối DFL 16 bin cho 4 cạnh box. | **`[0, 1]`**. Càng **thấp** (`< 0.35`) phân phối cạnh càng tập trung/chắc chắn. |
| **`dfl_top1_prob`** | $\text{mean}(\max_b p_b)$ trên positive anchors | Xác suất bin cao nhất của phân phối DFL cho 4 cạnh box. | **`[1/16, 1]`**. Càng **cao** (`> 0.60`) dự đoán tọa độ cạnh càng sắc nét. |
| **`grad_cos_box_vs_cls_p3 / p4 / p5`** | $\cos\big(\nabla_{F_s}\mathcal{L}_{\text{box}},\ \nabla_{F_s}(\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{slot}})\big)$ | Độ tương đồng góc giữa gradient nhánh Box và nhánh Cls+Slot chảy về Neck P3/P4/P5. | **`[-1, +1]`**. **`≥ 0`** là hài hòa / trực giao; **`< -0.1`** là dấu hiệu xung đột gradient kéo ngược nhau. |
| **`grad_ratio_box_over_cls_p3 / p4 / p5`** | $\frac{\|\nabla_{F_s}\mathcal{L}_{\text{box}}\|_2}{\|\nabla_{F_s}(\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{slot}})\|_2}$ | Tỷ lệ độ lớn lực kéo gradient của Box so với Cls+Slot trên Neck. | $[0, +\infty)$. Cân bằng quanh **`0.2 – 2.0`** (quá nhỏ `< 0.05` = Box bị Cls át). |

---

## 4. Chẩn đoán Overfit & Đóng góp của Slot + FiLM (`slot_overfit_diagnostics.csv`)

| Chỉ số | Công thức | Ý nghĩa | Khoảng giá trị & Mốc tốt |
|---|---|---|---|
| **`slot_share_p3 / p4 / p5_pct`** | $\frac{\sum_{i \in S_s, c<10} p_{c,i}}{\sum_{\text{all } i, c<10} p_{c,i}} \times 100\%$ | Tỷ trọng tổng xác suất FG slot phân bổ trên từng tầng `P3 / P4 / P5`. | **`[0%, 100%]`**, tổng = `100%`. |
| **`film_on / off_mAP50`** | $\text{mAP}_{50}$ khi bật FiLM vs tắt FiLM (`cls_feats = neck_feats`) | So sánh trực tiếp hiệu quả khi có và không có điều chế FiLM trên cùng weights. | **`[0, 1]`**. |
| **`film_delta_mAP50` / `mAP50_95`** | $\text{mAP}(\text{ON}) - \text{mAP}(\text{OFF})$ | Lợi ích ròng của Slot+FiLM. | `[-1, +1]`. **`> 0`** = FiLM có ích; `< 0` = FiLM gây hại/overfit. |
| **`loss_focal_film_on / off / delta`** | $\mathcal{L}_{\text{focal}}(\text{ON})$, $\mathcal{L}_{\text{focal}}(\text{OFF})$, hiệu `ON − OFF` | Tác động của FiLM lên Classification loss (`head.cv3`). | `delta < 0` là tốt (FiLM giúp giảm loss phân loại). |
| **`dots_std` / `dots_abs_max`** | $\text{std}(D)$, $\max \|D\|$ với $D = q k^\top / \sqrt{d}$ | Biên độ logit trước `softmax`. | $[0, +\infty)$. Quá lớn (`> 15`) = quá tự tin (overconfident). |
| **`dots_margin_fg`** | $\text{mean}_{i \in \text{FG}}(D_{(1),i} - D_{(2),i})$ | Biên cách giữa slot thắng nhất và nhì trên pixel vật thể. | $[0, +\infty)$. Tốt quanh **`1.0 – 3.0`**. |
| **`attn_entropy_fg` / `bg`** | $-\frac{1}{\ln 11}\sum_{s=0}^{10} p_{s,i}\ln p_{s,i}$ | Độ phân tán xác suất `attn_ori` trên FG / BG. | **`[0, 1]`**. BG càng nhỏ (`< 0.1`) càng sạch; FG đẹp quanh **`0.20 – 0.45`**. |
| **`slot_pos_ce_all` / `balanced`** | $-\text{mean}\ln p_{c,i}$ (trung bình theo pixel / theo lớp active) | Cross-entropy trên pixel GT lớp active (`balanced` tránh bị `car` che lấp lớp hiếm). | $[0, +\infty)$. Càng **thấp** càng tốt (`< 1.5`). |
| **`slot_mean_dice`** | $\frac{1}{K}\sum_c \frac{2\sum p_{c,i} m_{c,i}}{\sum p_{c,i} + \sum m_{c,i}}$ | Dice mềm trung bình 10 lớp giữa attention map và GT box. | **`[0, 1]`**. Càng **cao** càng tốt (`> 0.35`). |
| **`slot_bg_leak_pct`** | $\frac{\|\{i \in \text{FG}:\ \arg\max_s p_{s,i} = \text{BG}\}\|}{\|\text{FG}\|} \times 100\%$ | Tỷ lệ pixel vật thể bị BG slot nuốt nhầm. | **`[0%, 100%]`**. Càng **thấp** càng tốt (`< 15%`). |
| **`slot_inactive_fa_pct`** | $\text{mean}_{c \notin \text{img},\, i}(p_{c,i}) \times 100\%$ | Xác suất rò rỉ trung bình vào các lớp *không có mặt* trong ảnh. | **`[0%, 100%]`**. Càng gần **`0%`** (`< 0.5%`) càng tốt. |
| **`film_gamma_abs` / `beta_abs_p3..p5`** | $\text{mean}\|0.5\tanh(\Delta\gamma)\|$, $\text{mean}\|0.5\tanh(\Delta\beta)\|$ | Độ mạnh thực tế của phép nhân (`gamma`) và cộng (`beta`) của FiLM từng tầng. | **`[0, 0.5]`**. Khỏe mạnh quanh **`0.05 – 0.25`**. |
| **`film_sat_pct_p3..p5`** | $\text{mean}(\|\tanh(\Delta)\| > 0.9) \times 100\%$ | Tỷ lệ phần tử FiLM bị bão hòa sát biên `±0.5`. | **`[0%, 100%]`**. Tốt **`< 5%`**. |
| **`query_cos_sep` / `slot_cos_sep`** | $1 - \text{mean}_{i \neq j}\cos(u_i, u_j)$ | Độ trực giao giữa 10 `obj_queries` / 10 `slots` đầu ra. | **`[0, 2]`**. **`≈ 1.0`** là lý tưởng; `< 0.5` = slot collapse. |
| **`slot_cross_img_std`** | $\text{mean}_{c,d}\big(\text{std}_{\text{img}}(\hat{s}_{c,d})\big)$ | Độ linh hoạt của vector slot khi đổi ảnh. | **`[0, ~0.3]`**. Quá nhỏ (`< 0.03`) = slot tĩnh/cứng nhắc. |
| **`wnorm_proj_in` / `slot_attn` / `modulators`** | $\sqrt{\sum \|\theta\|_2^2}$ | Chuẩn L2 trọng số từng phần của `slot_refiner`. | $[0, +\infty)$. Theo dõi mức phình trọng số khi train dài. |
| **`q_norm_fg_min / mean / max` & `q_norm_bg`** | $\min / \text{mean} / \max \|q_c\|_2$ và $\|q_{\text{BG}}\|_2$ | Kiểm tra cân bằng độ lớn vector query giữa các lớp và BG. | $[0, +\infty)$. `max / min < 1.5` là cân bằng tốt. |
| **`fisher_neck / proj / key_p3..p5`** | Fisher Ratio tại `Neck`, `proj_in`, `to_k` cho P3/P4/P5 | Tóm tắt mức tăng/giảm độ phân tách lớp qua cổ chai `slot_dim`. | $[0, +\infty)$. `key > proj ≥ neck` là khỏe mạnh. |
| **`cos_q1_vs_k_mean` / `cos_q2_vs_k_mean`** | Trung bình 10 lớp của `cos_q1_k` và `cos_q2_k` | So sánh độ khớp trung bình của query tĩnh `T=1` vs query lặp `T=2`. | **`[-1, +1]`**. `cos_q1_vs_k_mean` càng cao (`> 0.60`) càng tốt. |

---

## 5. So sánh Nút thắt qua 3 Tầng Chiếu (`slot_bottleneck_stages.csv`)

| Chỉ số | Công thức | Ý nghĩa | Khoảng giá trị & Mốc tốt |
|---|---|---|---|
| **`scale`** | `P3`, `P4`, `P5` | Tầng tỷ lệ đang đo (`//8`, `//16`, `//32`). | Danh mục. |
| **`neck_fisher` / `proj_fisher` / `key_fisher`** | Fisher Ratio tại `Neck` → `proj_in` → `to_k` | Xác định xem phép nén `proj_in` (vd `256 → 32`) có làm mất thông tin phân loại hay không. | $[0, +\infty)$. Nếu `proj_fisher ≪ neck_fisher` → cổ chai `slot_dim`. |
| **`neck_cos_sep` / `proj_cos_sep` / `key_cos_sep`** | Cosine Separation giữa tâm các lớp tại 3 bước | Mức độ mở rộng góc tách giữa các lớp khi đi vào không gian Key của Slot. | **`[0, 2]`**. Tăng dần `neck → proj → key` (`key > 0.4`) là rất tốt. |

---

## 6. Chẩn đoán Chi tiết Từng Lớp (`slot_per_class_diagnostics.csv`)

| Chỉ số | Công thức | Ý nghĩa | Khoảng giá trị & Mốc tốt |
|---|---|---|---|
| **`class_id` / `class_name`** | `0..9` (`pedestrian` → `motorcycle`) | Định danh 10 lớp VisDrone. | Danh mục. |
| **`gt_pixels`** | Tổng số token GT của lớp $c$ cộng gộp trên cả 3 tầng và toàn bộ tập đánh giá | Đo độ hiếm/phổ biến về diện tích pixel của lớp $c$. | Số nguyên $\ge 0$. |
| **`slot_dice`** | $\frac{2\sum p_{c,i} m_{c,i}}{\sum p_{c,i} + \sum m_{c,i}}$ của riêng lớp $c$ | Độ trùng khớp mềm của slot lớp $c$ với GT box lớp $c$. | **`[0, 1]`**. Càng cao càng tốt (`car` thường `> 0.60`). |
| **`slot_recall_pct`** | $\frac{\|\{i \in \text{GT}_c:\ \arg\max p_i = c\}\|}{\|\text{GT}_c\|} \times 100\%$ | Tỷ lệ pixel GT của lớp $c$ mà slot $c$ thắng vị trí số 1. | **`[0%, 100%]`**. Càng cao càng tốt (`> 50%`). |
| **`rec_p3 / p4 / p5_pct`** | `slot_recall_pct` tính riêng trên từng tầng `P3`, `P4`, `P5` | Phân tách lỗi theo kích thước (vật nhỏ tụt ở P4/P5, vật lớn tụt ở P3). | **`[0%, 100%]`**. |
| **`bg_leak_pct`** | $\frac{\|\{i \in \text{GT}_c:\ \arg\max p_i = \text{BG}\}\|}{\|\text{GT}_c\|} \times 100\%$ | Tỷ lệ pixel của lớp $c$ bị BG slot nuốt nhầm. | **`[0%, 100%]`**. Càng thấp càng tốt (`< 15%`). |
| **`bg_leak_p3 / p4 / p5_pct`** | `bg_leak_pct` tính riêng trên từng tầng `P3`, `P4`, `P5` | Xác định tầng nào đang bị BG nuốt vật thể nặng nhất. | **`[0%, 100%]`**. |
| **`q_norm`** | $\|q_c\|_2$ | Độ lớn vector query của riêng lớp $c$. | $[0, +\infty)$. Nên đồng đều giữa các lớp. |
| **`dot_gt`** | $\text{mean}_{i \in \text{GT}_c}(D_{c,i})$ | Logit trung bình của slot $c$ trên đúng pixel GT của nó. | $(-\infty, +\infty)$. Càng dương lớn (`> 2.5`) càng tốt. |
| **`gap_vs_bg`** | $\text{mean}_{i \in \text{GT}_c}(D_{c,i} - D_{\text{BG},i})$ | Mức thắng/thua logit của lớp $c$ so với BG slot. | $(-\infty, +\infty)$. **Bắt buộc `> 0`** (tốt `> +1.5`); `< 0` = thua thẳng BG. |
| **`gap_vs_other`** | $\text{mean}_{i \in \text{GT}_c}\big(D_{c,i} - \max_{j \neq c} D_{j,i}\big)$ | Mức thắng/thua logit của lớp $c$ so với đối thủ mạnh nhất. | $(-\infty, +\infty)$. **`> 0`** = thắng đa số pixel; `< 0` (với `gap_vs_bg > 0`) = nhầm sang lớp họ hàng. |
| **`cos_q1_k`** | $\text{mean}_{\text{img}}\cos\big(\hat{q}_c,\, \widehat{\text{mean}}(k_{\text{GT}_c})\big)$ | Độ khớp giữa query tĩnh `T=1` với key trung bình của lớp $c$ trên Val. | **`[-1, +1]`**. Tốt **`> +0.60`**; `< +0.30` = 1 query đơn lẻ không bao phủ nổi lớp. |
| **`cos_q2_k`** | $\text{mean}_{\text{img}}\cos\big(\widehat{W_q\text{LN}(s_c)},\, \widehat{\text{mean}}(k_{\text{GT}_c})\big)$ | Độ khớp khi đưa slot đầu ra `s_c` qua lại `to_q` (chỉ có ý nghĩa trực tiếp khi train `T ≥ 2`). | **`[-1, +1]`**. |
| **`k_intra_std`** | $\text{mean}_d\big(\text{std}_{\text{img}}(\widehat{\text{mean}}(k_{\text{GT}_c})_d)\big)$ | Độ phân tán hình dạng/ngữ cảnh nội lớp của Key giữa các ảnh Val. | **`[0, ~0.2]`**. Thấp (`< 0.06`) = đồng nhất; **cao (`> 0.10`)** = đa hình dạng (multi-modal, nên dùng multi-prototype). |
| **`top_confused_slot`** | $\arg\max_{j \neq c}\text{ConfMat}[c, j]$ kèm `%` | Slot sai bị lớp $c$ gán nhầm nhiều nhất (`0..9` hoặc `BG`). | Chuỗi mô tả, vd `car (30.5%)` hoặc `BG (41.1%)`. |
