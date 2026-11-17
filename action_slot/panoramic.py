"""Script trực quan hóa và giải thích `PanoramicRayPositionEncoding` trên một ảnh panoramic (256x768 -> grid 16x48).

Cách dùng:
    # Tự động lấy 1 frame thật từ LMDB TACO (hoặc tạo ảnh đường phố mẫu nếu không có LMDB)
    python action_slot/tmp.py

    # Hoặc chỉ định một ảnh bất kỳ
    python action_slot/tmp.py --image path/to/frame.jpg --show
"""

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Bảo đảm import được action_slot.model dù chạy từ thư mục gốc hay trong action_slot/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in (PROJECT_ROOT, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_slot.model import (  # noqa: E402
    PanoramicRayPositionEncoding,
    build_2d_sincos_position_embedding,
)


DEFAULT_LMDB_PATH = "D:/apps/Datasets/TACO/data/ap_Town07_lmdb/rgb_front"
DEFAULT_OUT_PATH = os.path.join(CURRENT_DIR, "panoramic_ray_pe_explained.png")


def load_sample_image(
    image_path: Optional[str] = None,
    lmdb_path: str = DEFAULT_LMDB_PATH,
    lmdb_idx: int = 10,
    target_hw: Tuple[int, int] = (256, 768),
) -> Tuple[np.ndarray, str]:
    """Đọc ảnh RGB kích thước (H, W, 3) uint8 từ file, LMDB TACO, hoặc ảnh giả lập."""
    th, tw = target_hw

    if image_path:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Không đọc được ảnh tại: {image_path}")
        rgb = cv2.cvtColor(cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        return rgb, f"Image file: {image_path}"

    if lmdb_path and os.path.exists(lmdb_path):
        try:
            import lmdb

            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                if cursor.first():
                    for _ in range(max(0, lmdb_idx)):
                        if not cursor.next():
                            cursor.first()
                            break
                    key, val = cursor.item()
                    sample = pickle.loads(val)
                    buf = np.frombuffer(sample["image"], dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        rgb = cv2.cvtColor(
                            cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR),
                            cv2.COLOR_BGR2RGB,
                        )
                        key_name = key.decode("utf-8", errors="ignore")
                        env.close()
                        return rgb, f"TACO LMDB ({key_name})"
            env.close()
        except Exception as exc:
            print(f"[Info] Không đọc được LMDB ({exc}), chuyển sang ảnh đường phố mẫu.")

    # Fallback: tạo ảnh đường phố panoramic 3:1 có đường chân trời và các xe trái/giữa/phải
    rgb = np.zeros((th, tw, 3), dtype=np.uint8)
    horizon_y = int(th * 0.45)
    rgb[:horizon_y] = (145, 190, 235)  # Trời
    rgb[horizon_y:] = (72, 76, 82)     # Mặt đường
    # Vạch kẻ đường phối cảnh hội tụ về tâm chân trời
    cx = tw // 2
    for dx_bot, col in [(-320, (220, 220, 220)), (-110, (240, 200, 70)), (110, (240, 200, 70)), (320, (220, 220, 220))]:
        cv2.line(rgb, (cx, horizon_y), (cx + dx_bot, th - 1), col, 2)
    # Xe bên trái gần, xe phía trước xa, xe bên phải trung bình
    cv2.rectangle(rgb, (75, 165), (165, 225), (210, 60, 60), -1)
    cv2.rectangle(rgb, (365, 118), (403, 142), (60, 175, 90), -1)
    cv2.rectangle(rgb, (585, 145), (660, 195), (45, 130, 225), -1)
    return rgb, "Synthetic 3:1 Panoramic Road"


def compute_cosine_similarity_map(emb: torch.Tensor, row: int, col: int) -> np.ndarray:
    """Tính cosine similarity từ điểm (row, col) tới toàn bộ grid (H, W). emb shape: (1, D, H, W)."""
    feat = emb.squeeze(0).float()  # (D, H, W)
    norm_feat = F.normalize(feat, dim=0)
    ref = norm_feat[:, row, col].view(-1, 1, 1)
    sim = (norm_feat * ref).sum(dim=0)  # (H, W)
    return sim.detach().cpu().numpy()


def print_probe_explanation(
    probes: List[Tuple[str, int, int]],
    geom: np.ndarray,
    sincos: np.ndarray,
    ray_emb: np.ndarray,
    full_pe: np.ndarray,
    hfov_deg: float,
    vfov_deg: float,
    horizon_ratio: float,
    grid_hw: Tuple[int, int],
) -> None:
    """In bảng phân tích chi tiết các đại lượng hình học tại 5 điểm thăm dò."""
    H, W = grid_hw
    print("=" * 102)
    print(f"PHÂN TÍCH PanoramicRayPositionEncoding | Feature Grid: (H={H}, W={W}) | HFOV={hfov_deg}° | VFOV={vfov_deg}° | Horizon={horizon_ratio}")
    print("=" * 102)
    header = (
        f"{'Điểm':<24} | {'(r,c)':<7} | {'theta':>7} | {'phi':>6} | "
        f"{'Tia 3D (rx, ry, rz)':<23} | {'invD':>5} | {'logD':>5} | {'||sincos||':>10} | {'||PE||':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, r, c in probes:
        rx, ry, rz, u, v, inv_d, log_d = geom[:, r, c]
        theta_deg = u * 0.5 * hfov_deg
        phi_deg = v * 0.5 * vfov_deg
        norm_sc = float(np.linalg.norm(sincos[:, r, c]))
        norm_pe = float(np.linalg.norm(full_pe[:, r, c]))
        ray_str = f"({rx:+.2f}, {ry:+.2f}, {rz:+.2f})"
        print(
            f"{label:<24} | ({r:>2},{c:>2}) | {theta_deg:>+6.1f}° | {phi_deg:>+5.1f}° | "
            f"{ray_str:<23} | {inv_d:>5.2f} | {log_d:>5.2f} | {norm_sc:>10.2f} | {norm_pe:>8.2f}"
        )
    print("-" * len(header))
    print(
        "Ghi chú:\n"
        "  • (rx, ry, rz) luôn có chuẩn Euclid = 1: rx < 0 là bên trái, rx > 0 là bên phải, rz lớn là hướng thẳng phía trước.\n"
        "  • invD (ground_inv_depth): bằng 0 ở đường chân trời & bầu trời (r/H <= horizon_ratio), tăng dần về 1 ở đáy ảnh (sát xe ego).\n"
        "  • logD (ground_log_depth): thang nén log khoảng cách mặt đường (lớn ở chân trời = xa, nhỏ ở đáy ảnh = gần).\n"
        "  • PE cuối = sincos + ray_scale * ray_mlp(geom), giúp SlotAttention có cả tần số góc cầu lẫn hình học 3D.\n"
    )


def visualize_panoramic_ray_pe(
    rgb: np.ndarray,
    source_desc: str,
    grid_hw: Tuple[int, int] = (16, 48),
    embed_dim: int = 256,
    hfov_deg: float = 180.0,
    vfov_deg: float = 60.0,
    horizon_ratio: float = 0.45,
    out_path: str = DEFAULT_OUT_PATH,
    show: bool = False,
) -> None:
    H, W = grid_hw
    img_h, img_w = rgb.shape[:2]

    ray_pe = PanoramicRayPositionEncoding(
        embed_dim=embed_dim,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        horizon_ratio=horizon_ratio,
        default_hw=(H, W),
    )
    ray_pe.eval()

    with torch.no_grad():
        sincos_t, geom_t = ray_pe._build_spherical_grid(H, W, dtype=torch.float32)
        ray_emb_t = ray_pe.ray_mlp(geom_t)
        full_pe_t = ray_pe(H, W, dtype=torch.float32)
        old_2d_pe_t = build_2d_sincos_position_embedding(H, W, embed_dim, dtype=torch.float32)

    geom = geom_t.squeeze(0).numpy()       # (7, H, W)
    sincos = sincos_t.squeeze(0).numpy()   # (D, H, W)
    ray_emb = ray_emb_t.squeeze(0).numpy() # (D, H, W)
    full_pe = full_pe_t.squeeze(0).numpy() # (D, H, W)

    # 5 điểm thăm dò trên lưới (H=16, W=48)
    r_horizon = min(H - 1, max(0, int(round(horizon_ratio * H))))
    probes = [
        ("A: Trai gan", min(H - 1, int(0.80 * H)), int(0.12 * W)),
        ("B: Giua chan troi", r_horizon, W // 2),
        ("C: Giua gan", min(H - 1, int(0.80 * H)), W // 2),
        ("D: Phai gan", min(H - 1, int(0.80 * H)), int(0.88 * W)),
        ("E: Bau troi", max(0, int(0.18 * H)), int(0.65 * W)),
    ]
    probe_colors = ["#ff4d4d", "#ffd23f", "#00e5ff", "#52e368", "#d580ff"]

    print_probe_explanation(
        probes, geom, sincos, ray_emb, full_pe, hfov_deg, vfov_deg, horizon_ratio, grid_hw
    )

    # Cosine similarity từ điểm C (Giữa gần) để so sánh PE cũ vs PE mới
    _, ref_r, ref_c = probes[2]
    sim_old = compute_cosine_similarity_map(old_2d_pe_t, ref_r, ref_c)
    sim_new = compute_cosine_similarity_map(full_pe_t, ref_r, ref_c)

    # Vẽ figure giải thích trực quan
    fig = plt.figure(figsize=(18, 12), facecolor="#111418")
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.0, 1.0, 1.15], hspace=0.42, wspace=0.16)
    plt.rcParams["text.color"] = "#eef2f6"
    plt.rcParams["axes.edgecolor"] = "#3a414d"

    def style_ax(ax, title: str):
        ax.set_title(title, fontsize=11, fontweight="bold", color="#f5f7fa", pad=8)
        ax.tick_params(colors="#b8c0cc", labelsize=8)

    # --- Panel 1 (trên cùng, trải rộng 2 cột): Ảnh gốc + lưới (16x48) + chân trời + 5 điểm ---
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(rgb)
    # Vẽ lưới feature stride-16 mờ
    step_x = img_w / W
    step_y = img_h / H
    for gx in np.arange(0, img_w, step_x):
        ax0.axvline(gx, color="white", alpha=0.12, lw=0.6)
    for gy in np.arange(0, img_h, step_y):
        ax0.axhline(gy, color="white", alpha=0.12, lw=0.6)
    # Đường chân trời
    hy = horizon_ratio * img_h
    ax0.axhline(hy, color="#ffd23f", linestyle="--", lw=1.8, label=f"Horizon (y={horizon_ratio:.2f})")
    # Đánh dấu 5 điểm thăm dò
    label_offsets = [
        (8, -10, "left"),     # A: Trái gần
        (8, -10, "left"),     # B: Giữa chân trời
        (0, -18, "center"),   # C: Giữa gần (đặt phía trên điểm để không đè vào D)
        (-8, 12, "right"),    # D: Phải gần (đặt phía dưới-trái điểm)
        (8, -8, "left"),      # E: Bầu trời
    ]
    for (lbl, r, c), col, (dx_lbl, dy_lbl, ha) in zip(probes, probe_colors, label_offsets):
        px = (c + 0.5) * step_x
        py = (r + 0.5) * step_y
        rx, ry, rz, u, v, inv_d, _ = geom[:, r, c]
        th_deg = u * 0.5 * hfov_deg
        ax0.scatter([px], [py], s=95, color=col, edgecolors="black", linewidths=1.3, zorder=5)
        ax0.text(
            px + dx_lbl,
            py + dy_lbl,
            f"{lbl} ({th_deg:+.0f}°, invD={inv_d:.2f})",
            color=col,
            fontsize=8.5,
            fontweight="bold",
            ha=ha,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1117", alpha=0.82, edgecolor=col, lw=0.8),
        )
    ax0.set_xticks(np.linspace(0, img_w, 7))
    ax0.set_xticklabels([f"{ang:+.0f}°" for ang in np.linspace(-hfov_deg / 2, hfov_deg / 2, 7)])
    ax0.set_xlabel("Góc phương vị ngang theta (Azimuth)", color="#b8c0cc", fontsize=9)
    ax0.set_ylabel("Pixel Y", color="#b8c0cc", fontsize=9)
    ax0.legend(loc="upper left", facecolor="#161b22", edgecolor="#3a414d", fontsize=9)
    style_ax(ax0, f"1. Ảnh đầu vào ({img_h}x{img_w}) + Feature Grid ({H}x{W}) | Nguồn: {source_desc}")

    # --- Panel 2: ray_x (trái <-> phải) ---
    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(geom[0], cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax1.axhline(horizon_ratio * H - 0.5, color="yellow", ls="--", lw=1.2)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)
    cb1.ax.tick_params(colors="#b8c0cc", labelsize=8)
    style_ax(ax1, "2a. Camera Ray X = sin(theta)*cos(phi)  [-1: Trái | 0: Thẳng | +1: Phải]")

    # --- Panel 3: ground_inv_depth (chiều sâu mặt đường) ---
    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.imshow(geom[5], cmap="inferno", vmin=0.0, vmax=1.0, aspect="auto")
    ax2.axhline(horizon_ratio * H - 0.5, color="cyan", ls="--", lw=1.2)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
    cb2.ax.tick_params(colors="#b8c0cc", labelsize=8)
    style_ax(ax2, "2b. Ground Inverse-Depth Proxy  [0: Trời/Chân trời | 1: Mặt đường sát xe]")

    # --- Panel 4: Spherical Sin-Cos (tần số thấp vs tần số cao theo góc phương vị theta) ---
    ax3 = fig.add_subplot(gs[2, 0])
    # Ghép 2 nửa trên/dưới để quan sát cùng lúc: trên là sin(1*theta) (thô), dưới là sin(W*theta) (tinh)
    dim_per_axis = embed_dim // 2
    num_bands = dim_per_axis // 2
    band_low = sincos[0]                   # sin(freq_0 * theta)
    band_high = sincos[num_bands - 1]      # sin(freq_max * theta)
    split_vis = np.vstack([band_low[: H // 2], band_high[H // 2 :]])
    im3 = ax3.imshow(split_vis, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax3.axhline(H // 2 - 0.5, color="white", lw=1.5)
    ax3.text(1.5, 2.0, "Nửa trên: sin(freq_thấp * theta) -> vùng Trái/Giữa/Phải", color="black", fontsize=8,
             bbox=dict(facecolor="white", alpha=0.75, pad=1.5))
    ax3.text(1.5, H - 2.2, "Nửa dưới: sin(freq_cao * theta) -> phân giải từng cột góc", color="black", fontsize=8,
             bbox=dict(facecolor="white", alpha=0.75, pad=1.5))
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)
    cb3.ax.tick_params(colors="#b8c0cc", labelsize=8)
    style_ax(ax3, "3a. Spherical Sin-Cos Bands theo góc phương vị theta")

    # --- Panel 5: Bird's-Eye View minh họa tia 3D (ray_x, ray_z) của 5 điểm thăm dò ---
    ax4 = fig.add_subplot(gs[2, 1], facecolor="#0d1117")
    arc_t = np.linspace(-0.5 * np.radians(hfov_deg), 0.5 * np.radians(hfov_deg), 200)
    ax4.plot(np.sin(arc_t), np.cos(arc_t), color="#4b5563", ls="--", lw=1.2, label="Vòng cung FOV")
    ax4.scatter([0.0], [0.0], color="white", s=70, zorder=5, label="Camera Ego (0,0)")
    for (lbl, r, c), col in zip(probes, probe_colors):
        rx, ry, rz, _, _, inv_d, _ = geom[:, r, c]
        # Độ dài tia minh họa: gần xe thì ngắn, gần chân trời thì dài
        vis_len = 0.35 + 0.65 * (1.0 - inv_d) if r >= r_horizon else 1.05
        ax4.annotate(
            "",
            xy=(rx * vis_len, rz * vis_len),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color=col, lw=2.0),
        )
        ax4.text(rx * vis_len * 1.06, rz * vis_len * 1.06, lbl.split(":")[0], color=col, fontsize=9, fontweight="bold")
    ax4.set_xlim(-1.25, 1.25)
    ax4.set_ylim(-0.15, 1.25)
    ax4.set_xlabel("ray_x (Trái <-> Phải)", color="#b8c0cc", fontsize=8.5)
    ax4.set_ylabel("ray_z (Hướng tiến phía trước)", color="#b8c0cc", fontsize=8.5)
    ax4.grid(color="#222831", ls=":", lw=0.8)
    ax4.legend(loc="lower right", facecolor="#161b22", edgecolor="#3a414d", fontsize=8)
    style_ax(ax4, "3b. Góc nhìn BEV từ trên xuống: Hướng tia 3D & tầm xa ước lượng của A..E")

    # --- Panel 6 & 7: So sánh Cosine Similarity từ điểm C (Giữa gần) ---
    for ax, sim_map, title in [
        (fig.add_subplot(gs[3, 0]), sim_old, "4a. Cosine Similarity từ điểm C: 2D Sin-Cos CŨ"),
        (fig.add_subplot(gs[3, 1]), sim_new, "4b. Cosine Similarity từ điểm C: PanoramicRayPE MỚI"),
    ]:
        ax.imshow(rgb)
        sim_up = cv2.resize(sim_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        im_s = ax.imshow(sim_up, cmap="jet", vmin=-0.3, vmax=1.0, alpha=0.52)
        px = (ref_c + 0.5) * step_x
        py = (ref_r + 0.5) * step_y
        ax.scatter([px], [py], s=110, color="#00e5ff", edgecolors="black", linewidths=1.5, zorder=5)
        ax.axhline(hy, color="#ffd23f", ls="--", lw=1.2)
        cb = fig.colorbar(im_s, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(colors="#b8c0cc", labelsize=8)
        style_ax(ax, title)

    fig.suptitle(
        "Giải thích trực quan PanoramicRayPositionEncoding trên một ảnh 256x768",
        fontsize=14,
        fontweight="bold",
        color="#ffffff",
        y=0.99,
    )
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] Đã lưu hình minh họa tại: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa PanoramicRayPositionEncoding cho 1 ảnh.")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn ảnh đầu vào (tùy chọn).")
    parser.add_argument("--lmdb", type=str, default=DEFAULT_LMDB_PATH, help="Thư mục LMDB nếu không truyền --image.")
    parser.add_argument("--lmdb-idx", type=int, default=10, help="Index mẫu lấy từ LMDB.")
    parser.add_argument("--hfov", type=float, default=180.0, help="Horizontal FOV (độ).")
    parser.add_argument("--vfov", type=float, default=60.0, help="Vertical FOV (độ).")
    parser.add_argument("--horizon", type=float, default=0.45, help="Tỉ lệ chiều cao đường chân trời [0..1].")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_PATH, help="Đường dẫn file PNG xuất ra.")
    parser.add_argument("--show", action="store_true", help="Mở cửa sổ matplotlib hiển thị trực tiếp.")
    args = parser.parse_args()

    rgb, src_desc = load_sample_image(
        image_path=args.image,
        lmdb_path=args.lmdb,
        lmdb_idx=args.lmdb_idx,
    )
    visualize_panoramic_ray_pe(
        rgb=rgb,
        source_desc=src_desc,
        hfov_deg=args.hfov,
        vfov_deg=args.vfov,
        horizon_ratio=args.horizon,
        out_path=args.out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
