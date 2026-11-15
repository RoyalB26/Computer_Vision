"""
test_all.py
-----------
Script tổng hợp chạy kiểm thử tự động toàn bộ các thành phần trong thư mục Mamba:
1. configs.py
2. norm.py
3. selective_scan.py
4. ssm.py
5. mamba_block.py
6. mamba_layer.py
7. model.py
"""

import subprocess
import sys
import time

FILES_TO_TEST = [
    ("configs.py", "Cấu hình siêu tham số MambaConfig"),
    ("norm.py", "Chuẩn hóa RMSNorm"),
    ("selective_scan.py", "Thuật toán Selective Scan (S6 Core Math)"),
    ("ssm.py", "Lớp SelectiveSSM"),
    ("mamba_block.py", "Khối Mamba Block (Conv1d + S6 + Gating)"),
    ("mamba_layer.py", "Tầng ResidualBlock (Pre-RMSNorm + Residual)"),
    ("model.py", "Mô hình hoàn chỉnh Mamba & MambaLMHeadModel"),
]


def run_test(filename: str, description: str) -> bool:
    print("\n" + "=" * 70)
    print(f"Đang chạy kiểm thử: {filename} ({description})")
    print("=" * 70)
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(result.stdout)
        print(f"--> [PASS] {filename} hoàn thành trong {elapsed:.2f} giây.")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        print(f"--> [FAIL] {filename} thất bại sau {elapsed:.2f} giây!")
        return False


def main():
    print("*" * 70)
    print("       BẮT ĐẦU CHẠY BỘ KIỂM THỬ TOÀN DIỆN KIẾN TRÚC MAMBA")
    print(f"       Python: {sys.executable}")
    print("*" * 70)

    results = []
    for filename, desc in FILES_TO_TEST:
        passed = run_test(filename, desc)
        results.append((filename, desc, passed))

    print("\n" + "*" * 70)
    print("                   BẢNG TỔNG KẾT KIỂM THỬ")
    print("*" * 70)
    all_passed = True
    for filename, desc, passed in results:
        status_str = "[PASS] ĐẠT" if passed else "[FAIL] THẤT BẠI"
        if not passed:
            all_passed = False
        print(f" {status_str:<12} | {filename:<18} | {desc}")
    print("*" * 70)

    if all_passed:
        print("\n🎉 CHÚC MỪNG: TẤT CẢ CÁC FILE ĐỀU ĐẠT CHUẨN VÀ HOẠT ĐỘNG HOÀN HẢO!\n")
    else:
        print("\n❌ CẢNH BÁO: CÓ FILE KIỂM THỬ KHÔNG THÀNH CÔNG!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
