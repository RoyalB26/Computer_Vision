# Cài Đặt Kiến Trúc Mamba (Selective State Space Model)

Dự án cài đặt kiến trúc **Mamba** (*"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"* của Albert Gu và Tri Dao) thuần PyTorch.

Mã nguồn được phân tách thành từng mô-đun nhỏ gọn, chú thích chi tiết, kèm theo khối `if __name__ == "__main__":` trong từng file để bạn có thể chạy kiểm thử độc lập dữ liệu đầu vào (Input), đầu ra (Output), Gradient, và tính toán số học.

---

## 📁 Cấu Trúc Mã Nguồn

```text
Mamba/
├── configs.py             # Cấu hình siêu tham số (MambaConfig)
├── norm.py                # Chuẩn hóa RMSNorm
├── selective_scan.py      # Thuật toán Selective Scan (Lõi toán học S6)
├── ssm.py                 # Module SelectiveSSM (S6 layer)
├── mamba_block.py         # Khối Mamba Block (Conv1d + SiLU + S6 + Gating)
├── mamba_layer.py         # Tầng ResidualBlock (Pre-RMSNorm + Mixer + Residual)
├── model.py               # Mô hình hoàn chỉnh Mamba & MambaLMHeadModel (Kèm sinh token O(1))
├── test_all.py            # Chạy tự động toàn bộ test suite
└── __init__.py            # Khởi tạo package
```

---

## 🧠 Sơ Đồ Kiến Trúc

### 1. Khối Mamba Block (`mamba_block.py`)
```
             x (B, L, d_model)
                   │
                in_proj
               /       \
              u         z
              │         │
           Conv1d       │
              │         │
            SiLU        │
              │         │
             SSM      SiLU
              \       /
               \     /
               Multiply (Gating)
                   │
                out_proj
                   │
             y (B, L, d_model)
```

### 2. Thuật toán Selective Scan S6 (`selective_scan.py`)
- **Mô hình liên tục:**
  $$\begin{aligned}
  h'(t) &= A h(t) + B x(t) \\
  y(t) &= C h(t) + D x(t)
  \end{aligned}$$
- **Rời rạc hóa (Zero-Order Hold với bước $\Delta$):**
  $$\bar{A} = \exp(\Delta \cdot A), \quad \bar{B} = \Delta \cdot B$$
- **Đệ quy có chọn lọc (Selective Recurrence):**
  $$\begin{aligned}
  h_t &= \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot u_t \\
  y_t &= \sum_{n} h_{t, :, n} \cdot C_{t, n} + D \odot u_t
  \end{aligned}$$

---

## 🚀 Hướng Dẫn Chạy & Kiểm Thử Từng File

Bạn có thể chạy kiểm thử từng file riêng lẻ bằng lệnh:

```bash
# 1. Kiểm tra cấu hình và tính toán kích thước chiều ẩn
python configs.py

# 2. Kiểm tra chuẩn hóa RMSNorm (kiểm tra RMS ~ 1.0, gradient, xử lý tensor 0)
python norm.py

# 3. Kiểm tra thuật toán Selective Scan (độ khớp tuyệt đối giữa scan chuỗi và scan từng bước)
python selective_scan.py

# 4. Kiểm tra module SelectiveSSM (khởi tạo ma trận A HiPPO, Softplus delta)
python ssm.py

# 5. Kiểm tra MambaBlock (Causal Conv1d, Gating, tính tương đương giữa forward và step)
python mamba_block.py

# 6. Kiểm tra ResidualBlock (Pre-Norm + Skip connection)
python mamba_layer.py

# 7. Kiểm tra mô hình hoàn chỉnh MambaLMHeadModel (Loss, Backward, Sinh token O(1) Cache vs No-Cache)
python model.py

# Chạy kiểm thử toàn bộ dự án cùng lúc:
python test_all.py
```
