"""
selective_scan.py
-----------------
Cài đặt thuật toán Selective Scan (S6) - Trái tim toán học của kiến trúc Mamba.

Toán học:
    Mô hình không gian trạng thái liên tục (Continuous-time SSM):
        h'(t) = A h(t) + B x(t)
        y(t)  = C h(t) + D x(t)

    Rời rạc hóa (Discretization) bằng phương pháp Zero-Order Hold (ZOH) với bước thời gian Delta (dt):
        A_bar = exp(Delta * A)
        B_bar = Delta * B   (xấp xỉ bậc nhất trong Mamba)

    Hệ phương trình đệ quy có chọn lọc (Selective Recurrence):
        h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
        y_t = C_t * h_t + D * u_t

    Điểm đột phá của Mamba (Selection Mechanism):
        Khác với S4 (Linear Time-Invariant - LTI), trong Mamba:
        B, C và Delta đều phụ thuộc vào dữ liệu đầu vào u_t tại từng bước thời gian:
            B_t = Linear_B(u_t)
            C_t = Linear_C(u_t)
            Delta_t = Softplus(Linear_Delta(u_t))
        Nhờ đó mô hình có thể chủ động "ghi nhớ" hoặc "quên" thông tin dựa trên ngữ cảnh.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def selective_scan_step(
    u_t: torch.Tensor,
    delta_t: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    prev_state: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Thực hiện 1 bước đệ quy SSM cho 1 token (dùng cho quá trình sinh sinh mã / inference với độ phức tạp O(1)).

    Kích thước:
        u_t        : (B, S, D)     - Slots hiện tại
        delta_t    : (B, D)     - Bước thời gian Delta được chọn lọc
        A          : (S, D, N)     - Ma trận chuyển trạng thái
        B_t        : (B, S, N)     - Ma trận đầu vào được chọn lọc
        C_t        : (B, N)     - Ma trận quan sát được chọn lọc
        D          : (D,)       - Trọng số skip-connection (tùy chọn)
        prev_state : (B, D, N)  - Trạng thái ẩn tại bước trước h_{t-1}

    Trả về:
        y_t        : (B, D)     - Đầu ra tại bước t
        next_state : (B, D, N)  - Trạng thái ẩn mới h_t
    """
    batch_size, slots, d_inner = u_t.shape
    d_state = A.shape[-1]

    if prev_state is None:
        prev_state = torch.zeros(batch_size, slots, d_inner, d_state, device=u_t.device, dtype=u_t.dtype)

    # 1. Rời rạc hóa ma trận A: A_bar = exp(Delta * A)
    # delta_t: (B, S, D) hoặc (B, D) -> (B, S, D, 1), A: (1, S, D, N) -> deltaA: (B, S, D, N)
    if delta_t.dim() == 2:
        delta_t = delta_t.unsqueeze(1)
    deltaA = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))

    # 2. Rời rạc hóa ma trận B và nhân với u_t:
    # (Delta * u_t): (B, S, D, 1), B_t: (B, S, 1, N) -> deltaB_u: (B, S, D, N)
    deltaB_u = (delta_t * u_t).unsqueeze(-1) * B_t.unsqueeze(2)

    # 3. Cập nhật trạng thái ẩn: h_t = A_bar * h_{t-1} + B_bar * u_t
    next_state = deltaA * prev_state + deltaB_u # (B, S, D, N)

    # 4. Tính toán đầu ra: y_t = C_t * h_t
    # next_state: (B, S, D, N), C_t: (B, S, 1, N) -> sum theo N -> y_t: (B, S, D)
    y_t = (next_state * C_t.unsqueeze(2)).sum(dim=-1)

    # 5. Thêm nhánh skip-connection nếu có vector D
    if D is not None:
        y_t = y_t + u_t * D

    return y_t, next_state


def selective_scan_seq(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Thực hiện Selective Scan trên toàn bộ chuỗi có độ dài L (dùng khi huấn luyện và tiền xử lý prompt).

    Kích thước:
        u            : (B, L, S, D)    - Chuỗi đầu vào
        delta        : (B, L, D)    - Chuỗi bước thời gian
        A            : (S, D, N)       - Ma trận chuyển trạng thái
        B            : (B, L, S, N)    - Chuỗi ma trận B
        C            : (B, L, S, N)    - Chuỗi ma trận C
        D            : (D,)         - Trọng số skip-connection
        initial_state: (B, S, D, N)    - Trạng thái ban đầu (mặc định bằng 0)

    Trả về:
        ys           : (B, L, S, D)    - Toàn bộ chuỗi đầu ra
        final_state  : (B, S, D, N)    - Trạng thái ẩn cuối cùng
    """
    batch_size, seq_len, num_slot, d_inner = u.shape
    d_state = A.shape[-1]

    # Khởi tạo trạng thái ban đầu h_0
    state = (
        torch.zeros(batch_size, num_slot, d_inner, d_state, device=u.device, dtype=u.dtype)
        if initial_state is None
        else initial_state
    )

    ys = []

    # Quét tuần tự qua từng bước thời gian t = 0 ... L-1
    for t in range(seq_len):
        u_t = u[:, t, :]
        delta_t = delta[:, t, :]
        B_t = B[:, t, :]
        C_t = C[:, t, :]

        y_t, state = selective_scan_step(
            u_t=u_t,
            delta_t=delta_t,
            A=A,
            B_t=B_t,
            C_t=C_t,
            D=D,
            prev_state=state
        )
        ys.append(y_t)


    # Gom các đầu ra y_t thành tensor (B, L, S, D)
    ys = torch.stack(ys, dim=1)
    return ys, state


if __name__ == "__main__":
    print("=" * 60)
    print("TEST THUẬT TOÁN SELECTIVE SCAN (selective_scan.py)")
    print("=" * 60)

    

    # Thiết lập tham số test
    batch_size = 2
    seq_len = 5
    num_slot = 24
    d_inner = 16
    d_state = 8

    # Tạo dữ liệu giả lập
    torch.manual_seed(42)
    # slots= 

    u = torch.randn(batch_size, seq_len, num_slot, d_inner, requires_grad=True)
    delta = F.softplus(torch.randn(batch_size, seq_len, d_inner))
    # Ma trận A thường có giá trị âm để đảm bảo hệ thống SSM ổn định (decay over time)
    A = -torch.rand(num_slot, d_inner, d_state)
    B = torch.randn(batch_size, seq_len, num_slot, d_state)
    C = torch.randn(batch_size, seq_len, num_slot, d_state)
    D = torch.randn(d_inner)


    print(f"Tham số đầu vào:")
    print(f"  - u     shape : {tuple(u.shape)}     (B, L, D)")
    print(f"  - delta shape : {tuple(delta.shape)} (B, L, D)")
    print(f"  - A     shape : {tuple(A.shape)}         (D, N)")
    print(f"  - B     shape : {tuple(B.shape)}     (B, L, N)")
    print(f"  - C     shape : {tuple(C.shape)}     (B, L, N)")
    print(f"  - D     shape : {tuple(D.shape)}           (D,)")

    # 1. Chạy quét toàn bộ chuỗi
    ys_seq, final_state_seq = selective_scan_seq(u, delta, A, B, C, D)
    print(f"\n[OK] Chạy quét chuỗi thành công:")
    print(f"  - ys shape         : {tuple(ys_seq.shape)} (B, L, D)")
    print(f"  - final_state shape: {tuple(final_state_seq.shape)} (B, D, N)")

    # 2. Kiểm tra tính tương đương: Lặp từng bước qua selective_scan_step
    print(f"\n[OK] Kiểm tra tính tương thích giữa scan chuỗi và scan từng bước:")
    state = None
    step_ys = []
    for t in range(seq_len):
        y_t, state = selective_scan_step(
            u[:, t, :], delta[:, t, :], A, B[:, t, :], C[:, t, :], D, prev_state=state
        )
        step_ys.append(y_t)
    ys_steps = torch.stack(step_ys, dim=1)

    diff = (ys_seq - ys_steps).abs().max().item()
    print(f"  - Độ lệch lớn nhất giữa quét chuỗi và từng bước: {diff:.2e}")
    assert torch.allclose(ys_seq, ys_steps, atol=1e-6), "Scan chuỗi và scan từng bước không khớp!"

    # 3. Kiểm tra tính toán Gradient (Backward pass)
    loss = ys_seq.sum()
    loss.backward()
    assert u.grad is not None, "Gradient của u bị None!"
    print(f"  - Gradient norm của u: {u.grad.norm().item():.5f}")

    print("\n=> [PASS] selective_scan.py kiểm tra hoàn tất thành công!")
