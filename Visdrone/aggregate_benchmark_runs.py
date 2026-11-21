#!/usr/bin/env python3
"""Data Analyst Aggregation & Visualization Suite for Backbone Benchmark Runs.

Đầu vào:
  - 1 hoặc nhiều thư mục benchmark dạng `output_benchmark_<X>` chứa các thư mục con `run_1`, `run_2`, ...
    (hoặc chứa trực tiếp file `benchmark_results.csv` nếu chỉ có 1 run).

Đầu ra:
  - Thư mục `summary_benchmark_<X>` (hoặc theo `--out-dir`) gồm:
      1. Các file CSV phân tích chi tiết:
         - `benchmark_ranking_summary.csv`: Bảng xếp hạng các kiến trúc theo Val MAE (Mean ± Std).
         - `benchmark_detailed_stats.csv`: Đầy đủ Mean, Std, Min, Max, Median, CV% cho tất cả chỉ số.
         - `benchmark_raw_runs.csv`: Dữ liệu thô của từng run được gắn thẻ run và folder.
         - `benchmark_delta_vs_baseline.csv`: Chênh lệch và % cải thiện so với baseline (mặc định: resnet hoặc #1 rank).
         - `benchmark_pairwise_slots_stats.csv`: Thống kê 45 cặp lớp trên Final Slots qua các run.
         - `benchmark_pairwise_p3_stats.csv`: Thống kê 45 cặp lớp tại P3 qua các run.
         - `benchmark_class_level_separation.csv`: Tổng hợp độ phân tách trung bình theo từng class.
         - `benchmark_metrics_mae_impact.csv`: Đánh giá tác động (Tương quan Pearson/Spearman, Ridge, RF) của TẤT CẢ các chỉ số đặc trưng tới Val MAE.
         - `benchmark_pairwise_mae_impact.csv`: Tác động của từng cặp phân tách lớp (45 cặp) tới Val MAE.
         - `benchmark_class_level_mae_impact.csv`: Tổng hợp mức độ quan trọng/nhạy cảm của từng class tới Val MAE.
      2. Bộ biểu đồ trực quan chất lượng cao (PNG 180 DPI) CÓ ĐẦY ĐỦ CHÚ THÍCH MÀU SẮC CHO TỪNG BACKBONE:
         - `chart_01_mae_ranking_and_distribution.png`: Xếp hạng Val MAE (Mean ± Std) + Phân phối run + Legend.
         - `chart_02_accuracy_vs_efficiency_scatter.png`: Đánh giá Pareto: MAE vs Params (M) & MAE vs Train Time (s) + Legend.
         - `chart_03_accuracy_vs_stability_scatter.png`: Độ chính xác vs Độ ổn định (Mean MAE vs Std MAE) + Legend.
         - `chart_04_key_metrics_grouped_bars.png`: So sánh các chỉ số then chốt qua các kiến trúc + Legend.
         - `chart_05_multiscale_p3_p4_p5_profile.png`: Đa tầng P3/P4/P5 (Energy, Fisher, Cosine Sep, Slot Share %).
         - `chart_06_metrics_normalized_heatmap.png`: Ma trận chuẩn hóa toàn diện Scorecard Matrix [0, 1].
         - `chart_07_run_consistency_lines.png`: Xu hướng và độ ổn định của các block qua run_1 -> run_N + Legend.
         - `chart_08_pairwise_slots_heatmap.png`: Ma trận 10x10 Cosine Separation cho các kiến trúc hàng đầu.
         - `chart_09_pairwise_top_classes.png`: Top cặp lớp phân tách tốt nhất vs thách thức nhất.
         - `chart_10_all_metrics_correlation_with_mae.png`: Tương quan của TẤT CẢ các chỉ số đặc trưng kiến trúc tới Val MAE.
         - `chart_11_top_metric_scatter_regressions.png`: Đồ thị hồi quy thực nghiệm Top chỉ số tác động mạnh nhất tới MAE + Legend.
         - `chart_12_pairwise_sep_mae_correlation.png`: Tương quan 45 cặp phân tách & mức độ nhạy cảm cấp Class tới MAE.
      3. `dashboard.html`: Trang tổng hợp toàn bộ bảng số liệu, phân tích tương quan và biểu đồ để mở trực tiếp trên trình duyệt.

Ví dụ chạy:
  python Visdrone/aggregate_benchmark_runs.py --dirs Visdrone/output_benchmark_40
  python Visdrone/aggregate_benchmark_runs.py --auto
"""
from __future__ import annotations

import argparse
import glob
import html
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    has_sklearn = True
except ImportError:
    has_sklearn = False

from Visdrone.dataset import VISDRONE_CLASSES


# Bảng màu trực quan phân loại theo họ kiến trúc
BLOCK_COLORS = {
    # Plain & Classical
    "vgg": "#8c564b",
    "inception": "#e377c2",
    "resnet": "#4c72b0",
    "resnext": "#3b528b",
    "densenet": "#55a868",
    # Attention & Channel Enhancements
    "senet": "#dd8452",
    "cbam": "#c44e52",
    "cbamv2": "#d62728",
    # Efficient & Mobile
    "inverted": "#8172b3",
    "csp": "#937860",
    "repvgg": "#bcbd22",
    # Modern ConvNets
    "convnext": "#17becf",
    # Vision Transformers & Hybrid Attn
    "window_attn": "#2ca02c",
    "mobilevit": "#9467bd",
    "fastvit": "#ff7f0e",
    "area_attn": "#1f77b4",
    # State Space Models (SSM)
    "vmamba": "#009688",
}

KEY_METRICS_SPEC = [
    ("val_mae", "Val MAE", False),
    ("params_m", "Params (M)", False),
    ("train_sec", "Train Time (s)", False),
    ("slot_class_cos_sep", "Slot Class Cos Sep", True),
    ("slot_global_45pair_sep", "Slot 45-Pair Sep", True),
    ("slot_act_inact_ratio", "Slot Act/Inact Ratio", True),
    ("slot_full_fisher", "Slot Full Fisher", True),
    ("slot_pca_fisher", "Slot PCA Fisher", True),
    ("slot_fg_mass_pct", "Slot FG Mass (%)", True),
    ("p3_small_ratio", "P3 Small Ratio", True),
    ("p3_full_fisher", "P3 Full Fisher", True),
    ("p3_class_cos_sep", "P3 Class Cos Sep", True),
    ("p4_small_ratio", "P4 Small Ratio", True),
    ("p4_full_fisher", "P4 Full Fisher", True),
    ("p5_small_ratio", "P5 Small Ratio", True),
    ("p5_full_fisher", "P5 Full Fisher", True),
]


def make_backbone_legend_handles(blocks: List[str]) -> List[plt.Line2D]:
    """Tạo legend handle rõ ràng kết nối Tên Backbone với Màu sắc."""
    return [
        plt.Line2D([0], [0], marker="s", color="w", label=b,
                   markerfacecolor=BLOCK_COLORS.get(b, "#4C72B0"),
                   markeredgecolor="black", markersize=8)
        for b in blocks
    ]


def extract_benchmark_name(folder_path: str) -> str:
    """Trích xuất tên ngắn gọn từ tên thư mục (VD: output_benchmark_40 -> benchmark_40)."""
    base = os.path.basename(os.path.normpath(folder_path))
    prefix = "output_"
    if base.startswith(prefix):
        return base[len(prefix):]
    return base


def find_run_subdirs(parent_dir: str) -> List[str]:
    """Tìm tất cả các thư mục run_X bên trong parent_dir sắp xếp theo thứ tự số."""
    if not os.path.isdir(parent_dir):
        return []
    subdirs = []
    for item in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, item)
        if os.path.isdir(full_path) and re.match(r"^run_?\d+$", item, re.IGNORECASE):
            subdirs.append(full_path)

    def sort_key(p: str) -> int:
        m = re.search(r"(\d+)$", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted(subdirs, key=sort_key)


def load_benchmark_folder_data(
    folder_path: str,
) -> Tuple[str, str, List[Tuple[str, pd.DataFrame]], List[Tuple[str, pd.DataFrame]], List[Tuple[str, pd.DataFrame]]]:
    """Đọc dữ liệu benchmark kết quả tổng và các bảng pairwise từ tất cả các run trong folder_path."""
    folder_name = os.path.basename(os.path.normpath(folder_path))
    bench_name = extract_benchmark_name(folder_path)
    run_dirs = find_run_subdirs(folder_path)

    metric_runs: List[Tuple[str, pd.DataFrame]] = []
    pairwise_slots_runs: List[Tuple[str, pd.DataFrame]] = []
    pairwise_p3_runs: List[Tuple[str, pd.DataFrame]] = []

    def _read_if_exists(path: str) -> Optional[pd.DataFrame]:
        if os.path.isfile(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"[!] Lỗi đọc {path}: {e}")
        return None

    if not run_dirs:
        m_df = _read_if_exists(os.path.join(folder_path, "benchmark_results.csv"))
        if m_df is not None:
            metric_runs.append(("run_1", m_df))
        p_slots = _read_if_exists(os.path.join(folder_path, "pairwise_cosine_sep_slots.csv"))
        if p_slots is not None:
            pairwise_slots_runs.append(("run_1", p_slots))
        p_p3 = _read_if_exists(os.path.join(folder_path, "pairwise_cosine_sep.csv"))
        if p_p3 is not None:
            pairwise_p3_runs.append(("run_1", p_p3))
    else:
        for r_dir in run_dirs:
            run_label = os.path.basename(r_dir)
            m_df = _read_if_exists(os.path.join(r_dir, "benchmark_results.csv"))
            if m_df is not None:
                metric_runs.append((run_label, m_df))

            p_slots = _read_if_exists(os.path.join(r_dir, "pairwise_cosine_sep_slots.csv"))
            if p_slots is not None:
                pairwise_slots_runs.append((run_label, p_slots))

            p_p3 = _read_if_exists(os.path.join(r_dir, "pairwise_cosine_sep.csv"))
            if p_p3 is not None:
                pairwise_p3_runs.append((run_label, p_p3))

    return folder_name, bench_name, metric_runs, pairwise_slots_runs, pairwise_p3_runs


def build_raw_runs_df(
    bench_name: str,
    folder_name: str,
    metric_runs: List[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Gộp dữ liệu thô từng run thành 1 DataFrame chuẩn có cột run và folder."""
    frames = []
    for run_label, df in metric_runs:
        d = df.copy()
        d.insert(0, "run", run_label)
        d.insert(0, "folder", folder_name)
        d.insert(0, "benchmark", bench_name)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_detailed_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Tính mean, std, min, max, median, cv_pct cho từng block qua các run."""
    if raw_df.empty:
        return pd.DataFrame()

    meta_cols = {"benchmark", "folder", "run", "block"}
    numeric_cols = [c for c in raw_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(raw_df[c])]

    rows = []
    for (benchmark, folder, block), grp in raw_df.groupby(["benchmark", "folder", "block"], sort=False):
        n_runs = len(grp)
        ddof = 1 if n_runs > 1 else 0
        row: Dict[str, object] = {
            "benchmark": benchmark,
            "folder": folder,
            "block": block,
            "n_runs": n_runs,
        }
        for col in numeric_cols:
            vals = grp[col].dropna()
            mean_v = float(vals.mean()) if len(vals) else np.nan
            std_v = float(vals.std(ddof=ddof)) if len(vals) else 0.0
            min_v = float(vals.min()) if len(vals) else np.nan
            max_v = float(vals.max()) if len(vals) else np.nan
            med_v = float(vals.median()) if len(vals) else np.nan
            cv_v = (std_v / abs(mean_v) * 100.0) if (mean_v and not np.isnan(mean_v)) else 0.0

            row[f"{col}_mean"] = mean_v
            row[f"{col}_std"] = std_v
            row[f"{col}_min"] = min_v
            row[f"{col}_max"] = max_v
            row[f"{col}_median"] = med_v
            row[f"{col}_cv_pct"] = cv_v
        rows.append(row)

    df_stats = pd.DataFrame(rows)
    df_stats = df_stats.sort_values(by=["val_mae_mean", "val_mae_std"], ascending=[True, True]).reset_index(drop=True)
    df_stats.insert(0, "rank", df_stats.index + 1)
    return df_stats


def build_ranking_display_df(df_stats: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng xếp hạng gọn gàng với chuỗi 'Mean ± Std' và các chỉ số then chốt."""
    cols_to_format = [
        "val_mae",
        "params_m",
        "train_sec",
        "slot_class_cos_sep",
        "slot_global_45pair_sep",
        "slot_act_inact_ratio",
        "slot_full_fisher",
        "slot_pca_fisher",
        "slot_fg_mass_pct",
        "p3_small_ratio",
        "p3_full_fisher",
        "p3_class_cos_sep",
    ]
    rows = []
    for _, r in df_stats.iterrows():
        out: Dict[str, object] = {
            "rank": int(r["rank"]),
            "block": r["block"],
            "n_runs": int(r["n_runs"]),
            "val_mae_mean": round(float(r["val_mae_mean"]), 4),
            "val_mae_std": round(float(r["val_mae_std"]), 4),
            "val_mae_min": round(float(r["val_mae_min"]), 4),
            "val_mae_max": round(float(r["val_mae_max"]), 4),
            "val_mae_cv_pct": round(float(r["val_mae_cv_pct"]), 2),
        }
        for c in cols_to_format:
            if f"{c}_mean" in r and f"{c}_std" in r:
                m = float(r[f"{c}_mean"])
                s = float(r[f"{c}_std"])
                decimals = 2 if c in ("slot_fg_mass_pct", "train_sec", "params_m") else 4
                out[f"{c}_mean_std"] = f"{m:.{decimals}f} ± {s:.{decimals}f}"
        rows.append(out)
    return pd.DataFrame(rows)


def compute_delta_vs_baseline(
    df_stats: pd.DataFrame,
    baseline_block: Optional[str] = None,
) -> pd.DataFrame:
    """Tính chênh lệch tuyệt đối và % thay đổi của từng kiến trúc so với baseline."""
    if df_stats.empty:
        return pd.DataFrame()

    all_blocks = list(df_stats["block"].unique())
    if baseline_block is None or baseline_block not in all_blocks:
        baseline_block = "resnet" if "resnet" in all_blocks else all_blocks[0]

    base_row_df = df_stats[df_stats["block"] == baseline_block]
    if base_row_df.empty:
        return pd.DataFrame()
    base_row = base_row_df.iloc[0]

    metrics = [m for m, _, _ in KEY_METRICS_SPEC if f"{m}_mean" in df_stats.columns]
    rows = []
    for _, r in df_stats.iterrows():
        bname = r["block"]
        row: Dict[str, object] = {
            "rank": int(r["rank"]),
            "block": bname,
            "baseline": baseline_block,
        }
        for m in metrics:
            bv = float(base_row[f"{m}_mean"])
            cv = float(r[f"{m}_mean"])
            diff = cv - bv
            pct = (diff / abs(bv) * 100.0) if abs(bv) > 1e-12 else 0.0
            row[f"{m}_mean"] = round(cv, 4)
            row[f"{m}_diff"] = round(diff, 4)
            row[f"{m}_pct_change"] = round(pct, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_pairwise_stats_across_runs(
    pairwise_runs: List[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Tính Mean, Std, Min, Max qua các run cho ma trận 45 cặp class."""
    if not pairwise_runs:
        return pd.DataFrame()

    n_runs = len(pairwise_runs)
    ddof = 1 if n_runs > 1 else 0

    first_df = pairwise_runs[0][1]
    meta_cols = {"class_i", "class_j", "pair", "mean_blocks"}
    block_cols = [c for c in first_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(first_df[c])]

    all_dfs = [df for _, df in pairwise_runs]
    combined = pd.concat(all_dfs, ignore_index=True)

    pair_rows = []
    for (ci, cj, pair), grp in combined.groupby(["class_i", "class_j", "pair"], sort=False):
        row: Dict[str, object] = {
            "class_i": ci,
            "class_j": cj,
            "pair": pair,
            "n_runs": len(grp),
        }
        block_means = []
        for b in block_cols:
            if b in grp.columns:
                vals = grp[b].dropna()
                mean_v = float(vals.mean()) if len(vals) else np.nan
                std_v = float(vals.std(ddof=ddof)) if len(vals) else 0.0
                row[f"{b}_mean"] = mean_v
                row[f"{b}_std"] = std_v
                if not np.isnan(mean_v):
                    block_means.append(mean_v)
        row["all_blocks_mean"] = float(np.mean(block_means)) if block_means else np.nan
        pair_rows.append(row)

    df_pair_res = pd.DataFrame(pair_rows)
    return df_pair_res


def compute_class_level_separation(df_pair_stats: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp độ phân tách trung bình của từng class."""
    if df_pair_stats.empty:
        return pd.DataFrame()

    classes = sorted(list(set(df_pair_stats["class_i"]).union(set(df_pair_stats["class_j"]))))
    mean_cols = [c for c in df_pair_stats.columns if c.endswith("_mean") and c != "all_blocks_mean"]

    rows = []
    for c in classes:
        sub = df_pair_stats[(df_pair_stats["class_i"] == c) | (df_pair_stats["class_j"] == c)]
        row: Dict[str, object] = {
            "class": c,
            "num_pairs": len(sub),
            "all_blocks_avg_sep": round(float(sub["all_blocks_mean"].mean()), 4) if "all_blocks_mean" in sub else np.nan,
        }
        for mc in mean_cols:
            row[mc] = round(float(sub[mc].mean()), 4)
        rows.append(row)

    df_cls = pd.DataFrame(rows).sort_values(by="all_blocks_avg_sep", ascending=False).reset_index(drop=True)
    return df_cls


# =============================================================================
# CHỨC NĂNG MỞ RỘNG: ĐÁNH GIÁ TÁC ĐỘNG CỦA TẤT CẢ CÁC CHỈ SỐ ĐẾN VAL MAE
# =============================================================================

def compute_all_metrics_mae_impact(
    df_stats: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Tính tương quan Pearson, Spearman và tầm quan trọng đặc trưng (Ridge / RF)
    của TẤT CẢ các chỉ số đặc trưng kiến trúc (ngoại trừ params_m, train_sec, metadata)
    với Val MAE trên tập dữ liệu tổng hợp.
    """
    if df_stats.empty or "val_mae_mean" not in df_stats.columns:
        return pd.DataFrame()

    # Danh sách các cột bị loại bỏ (chỉ là tham số kích thước/thời gian/metadata)
    exclude_prefixes = {
        "block", "folder", "benchmark", "rank", "n_runs", "val_mae",
        "params_m", "train_sec", "val_mae_mean", "val_mae_std", "val_mae_min",
        "val_mae_max", "val_mae_median", "val_mae_cv_pct"
    }

    # Lấy danh sách các metric gốc từ df_stats có đuôi _mean
    metric_candidates = []
    for col in df_stats.columns:
        if col.endswith("_mean"):
            base_name = col[:-5]
            if base_name not in exclude_prefixes and not any(base_name.startswith(p) for p in ["params", "train", "val_mae"]):
                metric_candidates.append(base_name)

    metric_candidates = sorted(list(dict.fromkeys(metric_candidates)))
    y_mean = df_stats["val_mae_mean"].values.astype(float)

    impact_rows = []
    X_matrix_dict = {}

    for m in metric_candidates:
        mean_col = f"{m}_mean"
        if mean_col not in df_stats.columns:
            continue

        x_vals = df_stats[mean_col].values.astype(float)
        # Bỏ qua nếu giá trị hằng số hoặc toàn NaN
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_mean)
        if valid_mask.sum() < 4 or np.std(x_vals[valid_mask]) < 1e-9:
            continue

        X_matrix_dict[m] = x_vals

        # 1. Tương quan ở cấp độ Kiến trúc (Block Mean, N = số block)
        r_p, p_p = stats.pearsonr(x_vals[valid_mask], y_mean[valid_mask])
        r_s, p_s = stats.spearmanr(x_vals[valid_mask], y_mean[valid_mask])

        # 2. Tương quan ở cấp độ Mẫu (Tất cả các run độc lập, N = số run * số block)
        r_sample, p_sample = np.nan, np.nan
        if not raw_df.empty and m in raw_df.columns and "val_mae" in raw_df.columns:
            sub_raw = raw_df[[m, "val_mae"]].dropna()
            if len(sub_raw) >= 5 and sub_raw[m].std() > 1e-9:
                r_sample, p_sample = stats.pearsonr(sub_raw[m].values, sub_raw["val_mae"].values)

        # Diễn giải xu hướng
        if r_p <= -0.4:
            interpretation = "Cải thiện mạnh MAE (Tương quan nghịch sâu)"
        elif -0.4 < r_p <= -0.15:
            interpretation = "Cải thiện vừa MAE (Tương quan nghịch)"
        elif -0.15 < r_p < 0.15:
            interpretation = "Ảnh hưởng yếu / Không tương quan"
        elif 0.15 <= r_p < 0.4:
            interpretation = "Tăng nhẹ MAE (Tương quan thuận)"
        else:
            interpretation = "Tăng mạnh MAE (Đánh đổi hiệu năng)"

        impact_rows.append({
            "metric": m,
            "mean_val": float(np.mean(x_vals[valid_mask])),
            "std_val": float(np.std(x_vals[valid_mask])),
            "pearson_r": r_p,
            "pearson_p": p_p,
            "spearman_rho": r_s,
            "spearman_p": p_s,
            "abs_pearson": abs(r_p),
            "sample_pearson_r": r_sample,
            "sample_pearson_p": p_sample,
            "interpretation": interpretation,
        })

    df_impact = pd.DataFrame(impact_rows)
    if df_impact.empty:
        return df_impact

    # 3. Tính hệ số hồi quy đa biến Ridge & Random Forest Importance nếu có sklearn
    if has_sklearn and len(X_matrix_dict) > 0:
        X_df = pd.DataFrame(X_matrix_dict)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df.fillna(X_df.mean()))
        y_scaled = (y_mean - y_mean.mean()) / (y_mean.std() + 1e-8)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y_scaled)
        ridge_coefs = dict(zip(X_df.columns, ridge.coef_))

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y_scaled)
        rf_importances = dict(zip(X_df.columns, rf.feature_importances_))

        df_impact["ridge_std_coef"] = df_impact["metric"].map(ridge_coefs)
        df_impact["rf_importance"] = df_impact["metric"].map(rf_importances)
    else:
        df_impact["ridge_std_coef"] = np.nan
        df_impact["rf_importance"] = np.nan

    # Sắp xếp theo mức độ tương quan tuyệt đối
    df_impact = df_impact.sort_values(by="abs_pearson", ascending=False).reset_index(drop=True)
    df_impact.insert(0, "impact_rank", df_impact.index + 1)
    return df_impact


def compute_pairwise_mae_impact(
    df_pair_stats: pd.DataFrame,
    df_stats: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Đánh giá tác động của 45 cặp khoảng cách Cosine Separation (trên Final Slots) tới Val MAE."""
    if df_pair_stats.empty or df_stats.empty:
        return pd.DataFrame(), pd.DataFrame()

    val_mae_dict = dict(zip(df_stats["block"], df_stats["val_mae_mean"]))
    common_blocks = [b for b in df_stats["block"] if f"{b}_mean" in df_pair_stats.columns]

    if len(common_blocks) < 4:
        return pd.DataFrame(), pd.DataFrame()

    y_mae = pd.Series([val_mae_dict[b] for b in common_blocks], index=common_blocks)

    pair_results = []
    X_dict = {}

    for _, row in df_pair_stats.iterrows():
        pair_name = row["pair"]
        ci = row.get("class_i", "")
        cj = row.get("class_j", "")
        sep_vals = pd.Series([row[f"{b}_mean"] for b in common_blocks], index=common_blocks, dtype=float)

        if sep_vals.std() < 1e-9:
            continue

        r_p, p_p = stats.pearsonr(sep_vals, y_mae)
        r_s, p_s = stats.spearmanr(sep_vals, y_mae)

        X_dict[pair_name] = sep_vals.values

        if r_p <= -0.35:
            interp = "Phân tách giúp giảm mạnh MAE"
        elif -0.35 < r_p <= -0.15:
            interp = "Phân tách giúp giảm nhẹ MAE"
        elif -0.15 < r_p < 0.15:
            interp = "Ít ảnh hưởng tới MAE"
        else:
            interp = "Tương quan thuận với MAE (Đánh đổi)"

        pair_results.append({
            "pair": pair_name,
            "class_i": ci,
            "class_j": cj,
            "mean_sep": float(sep_vals.mean()),
            "std_sep": float(sep_vals.std()),
            "pearson_r": r_p,
            "pearson_p": p_p,
            "spearman_rho": r_s,
            "spearman_p": p_s,
            "abs_pearson": abs(r_p),
            "interpretation": interp,
        })

    df_pair_impact = pd.DataFrame(pair_results)
    if df_pair_impact.empty:
        return pd.DataFrame(), pd.DataFrame()

    if has_sklearn and len(X_dict) > 0:
        X_df = pd.DataFrame(X_dict, index=common_blocks)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_df.fillna(X_df.mean()))
        y_sc = (y_mae.values - y_mae.mean()) / (y_mae.std() + 1e-8)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_sc, y_sc)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_sc, y_sc)

        df_pair_impact["ridge_std_coef"] = df_pair_impact["pair"].map(dict(zip(X_df.columns, ridge.coef_)))
        df_pair_impact["rf_importance"] = df_pair_impact["pair"].map(dict(zip(X_df.columns, rf.feature_importances_)))
    else:
        df_pair_impact["ridge_std_coef"] = np.nan
        df_pair_impact["rf_importance"] = np.nan

    df_pair_impact = df_pair_impact.sort_values(by="abs_pearson", ascending=False).reset_index(drop=True)
    df_pair_impact.insert(0, "pair_rank", df_pair_impact.index + 1)

    # Tổng hợp mức độ quan trọng theo từng Class
    all_classes = sorted(list(set(df_pair_impact["class_i"]).union(set(df_pair_impact["class_j"]))))
    class_summary = []
    for c in all_classes:
        c_sub = df_pair_impact[(df_pair_impact["class_i"] == c) | (df_pair_impact["class_j"] == c)]
        class_summary.append({
            "class": c,
            "num_pairs": len(c_sub),
            "avg_abs_pearson": float(c_sub["abs_pearson"].mean()),
            "avg_pearson_r": float(c_sub["pearson_r"].mean()),
            "min_pearson_r": float(c_sub["pearson_r"].min()),  # Cặp có đóng góp giảm MAE lớn nhất
            "max_pearson_r": float(c_sub["pearson_r"].max()),
        })
    df_class_impact = pd.DataFrame(class_summary).sort_values(by="avg_abs_pearson", ascending=False).reset_index(drop=True)
    df_class_impact.insert(0, "class_rank", df_class_impact.index + 1)

    return df_pair_impact, df_class_impact


# =============================================================================
# VISUALIZATION SUITE (BỔ SUNG CHÚ THÍCH MÀU SẮC CHO TỪNG BIỂU ĐỒ)
# =============================================================================

def _setup_mpl_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.edgecolor": "#CCCCCC",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })


def plot_01_mae_ranking_and_distribution(
    df_stats: pd.DataFrame,
    raw_df: pd.DataFrame,
    out_dir: str,
) -> str:
    """Biểu đồ thanh xếp hạng ngang Val MAE (Mean ± Std) + Boxplot và phân phối từng run + Legend màu sắc."""
    _setup_mpl_style()
    n_rows = len(df_stats)
    fig_h = max(5.5, 0.45 * n_rows + 2.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, fig_h), sharey=True)

    names = [r["block"] for _, r in df_stats.iterrows()]
    labels = [f"#{int(r['rank'])} {r['block'].upper()}" for _, r in df_stats.iterrows()]
    y_pos = np.arange(n_rows)
    colors = [BLOCK_COLORS.get(r["block"], "#4C72B0") for _, r in df_stats.iterrows()]

    means = df_stats["val_mae_mean"].values
    stds = df_stats["val_mae_std"].values

    ax1.barh(y_pos, means, xerr=stds, capsize=4, color=colors, alpha=0.88, edgecolor="#333333", linewidth=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Val MAE (Mean ± Std) — Thấp hơn là tốt hơn", fontsize=10)
    ax1.set_title("1A. Xếp hạng Val MAE (Mean ± Std)", fontsize=11, fontweight="bold")

    x_min = max(0.0, float(np.min(means - stds)) * 0.90)
    x_max = float(np.max(means + stds)) * 1.12
    ax1.set_xlim(x_min, x_max)

    for idx, (m, s) in enumerate(zip(means, stds)):
        ax1.text(m + s + (x_max - x_min) * 0.012, idx, f"{m:.4f} ± {s:.4f}", va="center", fontsize=8.5, fontweight="bold")

    # Right panel: Boxplot + individual runs scatter
    run_series = []
    for _, r in df_stats.iterrows():
        sub = raw_df[raw_df["block"] == r["block"]]["val_mae"].dropna().values
        run_series.append(sub)

    bp = ax2.boxplot(run_series, positions=y_pos, orientation="horizontal", widths=0.5, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5),
                     whiskerprops=dict(color="#555555"),
                     capprops=dict(color="#555555"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(42)
    for idx, vals in enumerate(run_series):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax2.scatter(vals, np.full_like(vals, idx, dtype=float) + jitter, color=colors[idx],
                    edgecolor="black", linewidth=0.5, s=42, zorder=5)

    ax2.set_xlabel("Val MAE (Phân phối qua các run)", fontsize=10)
    ax2.set_title("1B. Phân phối phân tán giữa các Run", fontsize=11, fontweight="bold")

    # Chú thích màu sắc cho toàn bộ các kiến trúc ở đáy biểu đồ
    fig.legend(handles=make_backbone_legend_handles(names), loc="lower center", ncol=min(9, len(names)),
               fontsize=8.5, title="Màu sắc tương ứng các Kiến trúc Backbone", framealpha=0.92)

    fig.suptitle("Backbone Block Ranking & Cross-Run Val MAE Distribution", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_01_mae_ranking_and_distribution.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_02_accuracy_vs_efficiency_scatter(df_stats: pd.DataFrame, out_dir: str) -> str:
    """Đồ thị phân tán Pareto: Val MAE vs Tham số Params (M) và Val MAE vs Thời gian huấn luyện + Legend."""
    _setup_mpl_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 7.5))
    names = [r["block"] for _, r in df_stats.iterrows()]

    for ax, x_col, x_label, title in [
        (ax1, "params_m_mean", "Số lượng tham số (Params M) — Thấp hơn là nhẹ hơn", "2A. Val MAE vs. Model Parameters (Pareto Frontier)"),
        (ax2, "train_sec_mean", "Thời gian huấn luyện mỗi Run (giây) — Thấp hơn là nhanh hơn", "2B. Val MAE vs. Training Speed (s)"),
    ]:
        if x_col not in df_stats.columns:
            ax.axis("off")
            continue

        xs = df_stats[x_col].values
        ys = df_stats["val_mae_mean"].values
        errs_y = df_stats["val_mae_std"].values

        for _, r in df_stats.iterrows():
            c = BLOCK_COLORS.get(r["block"], "#4C72B0")
            x_v = float(r[x_col])
            y_v = float(r["val_mae_mean"])
            y_err = float(r["val_mae_std"])

            ax.errorbar(x_v, y_v, yerr=y_err, fmt="o", color=c, ecolor="#999999", elinewidth=1.0,
                        capsize=3, markersize=9, markeredgecolor="black", markeredgewidth=0.7, zorder=4)

            lbl = f"#{int(r['rank'])} {r['block']}"
            ax.annotate(lbl, (x_v, y_v), textcoords="offset points", xytext=(6, 4),
                        fontsize=8.5, fontweight="bold")

        # Đường Pareto Optimal Frontier (Min X và Min Y)
        sorted_indices = np.argsort(xs)
        xs_sorted = xs[sorted_indices]
        ys_sorted = ys[sorted_indices]
        pareto_x, pareto_y = [xs_sorted[0]], [ys_sorted[0]]
        cur_min_y = ys_sorted[0]
        for xi, yi in zip(xs_sorted[1:], ys_sorted[1:]):
            if yi < cur_min_y:
                pareto_x.append(xi)
                pareto_y.append(yi)
                cur_min_y = yi

        ax.step(pareto_x, pareto_y, where="post", color="#e74c3c", linestyle=":", linewidth=1.5,
                alpha=0.75, label="Pareto Frontier (Đường tối ưu)")

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Val MAE (Mean ± Std) — Thấp hơn là tốt hơn", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper right")

    # Legend màu sắc từng backbone
    fig.legend(handles=make_backbone_legend_handles(names), loc="lower center", ncol=min(9, len(names)),
               fontsize=8.5, title="Màu sắc tương ứng các Kiến trúc Backbone", framealpha=0.92)

    fig.suptitle("Trade-off Analysis: Accuracy vs. Efficiency & Complexity", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_02_accuracy_vs_efficiency_scatter.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_03_accuracy_vs_stability_scatter(df_stats: pd.DataFrame, out_dir: str) -> str:
    """Đồ thị phân tán: Mean Val MAE vs Std Val MAE + Legend màu sắc rõ ràng bên cạnh."""
    _setup_mpl_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    names = [r["block"] for _, r in df_stats.iterrows()]

    for _, r in df_stats.iterrows():
        c = BLOCK_COLORS.get(r["block"], "#4C72B0")
        m_x = float(r["val_mae_mean"])
        s_y = float(r["val_mae_std"])
        cv = float(r["val_mae_cv_pct"])

        bubble_size = max(70, min(320, cv * 25))
        ax.scatter(m_x, s_y, color=c, s=bubble_size, alpha=0.88, edgecolor="black", linewidth=0.8, zorder=5)

        lbl = f"#{int(r['rank'])} {r['block']} (CV:{cv:.1f}%)"
        ax.annotate(lbl, (m_x, s_y), textcoords="offset points", xytext=(8, 4),
                    fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Mean Val MAE (Thấp hơn = Độ chính xác cao hơn)", fontsize=10.5)
    ax.set_ylabel("Std Val MAE (Thấp hơn = Tính ổn định và khả năng lặp lại cao hơn)", fontsize=10.5)
    ax.set_title("Accuracy vs. Stability: Mean Val MAE vs. Cross-Run Std Val MAE\n(Góc dưới bên trái: Tối ưu nhất về cả độ chính xác lẫn độ ổn định)", fontsize=11.5, fontweight="bold")

    # Legend đặt bên ngoài góc phải
    ax.legend(handles=make_backbone_legend_handles(names), title="Kiến trúc Backbone",
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "chart_03_accuracy_vs_stability_scatter.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_04_key_metrics_grouped_bars(df_stats: pd.DataFrame, out_dir: str) -> str:
    """Lưới biểu đồ cột so sánh các chỉ số hiệu năng then chốt + Legend."""
    _setup_mpl_style()
    metrics = [
        ("val_mae", "Val MAE (↓ Tốt hơn)"),
        ("slot_class_cos_sep", "Slot Class Cos Sep (↑ Tốt hơn)"),
        ("slot_global_45pair_sep", "Slot Global 45-Pair Sep (↑ Tốt hơn)"),
        ("slot_act_inact_ratio", "Slot Act/Inact Ratio (↑ Tốt hơn)"),
        ("slot_full_fisher", "Slot Full Fisher (↑ Tốt hơn)"),
        ("p3_small_ratio", "P3 Small Ratio (↑ Tốt hơn)"),
    ]
    metrics = [(m, t) for m, t in metrics if f"{m}_mean" in df_stats.columns]

    n_m = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    names = [r["block"] for _, r in df_stats.iterrows()]
    x = np.arange(len(names))
    colors = [BLOCK_COLORS.get(b, "#4C72B0") for b in names]

    for idx, (m_col, title) in enumerate(metrics):
        ax = axes[idx]
        means = df_stats[f"{m_col}_mean"].values
        stds = df_stats[f"{m_col}_std"].values

        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.88, edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5, fontweight="bold")
        ax.set_title(title, fontsize=10.5, fontweight="bold")

        for b, mv in zip(bars, means):
            dec = 3 if mv < 1.0 else (2 if mv < 10.0 else 1)
            ax.text(b.get_x() + b.get_width() / 2.0, mv, f"{mv:.{dec}f}",
                    ha="center", va="bottom", fontsize=7.2, rotation=0)

    for j in range(n_m, len(axes)):
        axes[j].axis("off")

    fig.legend(handles=make_backbone_legend_handles(names), loc="lower center", ncol=min(9, len(names)),
               fontsize=8.5, title="Màu sắc tương ứng các Kiến trúc Backbone", framealpha=0.92)

    fig.suptitle("Key Benchmark Metrics Comparison Across All Backbone Blocks (Mean ± Std)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_04_key_metrics_grouped_bars.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_05_multiscale_profile(df_stats: pd.DataFrame, out_dir: str) -> str:
    """Hồ sơ đa tầng P3/P4/P5 (Energy, Fisher, Cosine Sep) và Tỷ trọng phân bổ Attention Share %."""
    _setup_mpl_style()
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))

    names = [r["block"] for _, r in df_stats.iterrows()]
    x = np.arange(len(names))
    w = 0.26

    scale_panels = [
        (axes[0, 0], ("p3_small_ratio", "p4_small_ratio", "p5_small_ratio"), "Small Object Energy Ratio (P3 / P4 / P5)"),
        (axes[0, 1], ("p3_full_fisher", "p4_full_fisher", "p5_full_fisher"), "FG/BG Full Fisher Ratio (P3 / P4 / P5)"),
        (axes[1, 0], ("p3_class_cos_sep", "p4_class_cos_sep", "p5_class_cos_sep"), "Inter-Class Cosine Separation (P3 / P4 / P5)"),
    ]
    scale_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    scale_names = ["P3 (//8)", "P4 (//16)", "P5 (//32)"]

    for ax, cols, title in scale_panels:
        for si, (col, sc, sn) in enumerate(zip(cols, scale_colors, scale_names)):
            if f"{col}_mean" not in df_stats.columns:
                continue
            vals = df_stats[f"{col}_mean"].values
            errs = df_stats[f"{col}_std"].values
            ax.bar(x + (si - 1) * w, vals, width=w * 0.92, yerr=errs, capsize=2, label=sn, color=sc, alpha=0.88, edgecolor="#333", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5, rotation=35, ha="right", fontweight="bold")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=8.5)

    ax_s = axes[1, 1]
    share_cols = ("slot_share_p3_pct_mean", "slot_share_p4_pct_mean", "slot_share_p5_pct_mean")
    if all(c in df_stats.columns for c in share_cols):
        p3 = df_stats[share_cols[0]].values
        p4 = df_stats[share_cols[1]].values
        p5 = df_stats[share_cols[2]].values
        ax_s.bar(x, p3, label="P3 Share %", color="#1f77b4", alpha=0.88, edgecolor="#333", linewidth=0.5)
        ax_s.bar(x, p4, bottom=p3, label="P4 Share %", color="#ff7f0e", alpha=0.88, edgecolor="#333", linewidth=0.5)
        ax_s.bar(x, p5, bottom=p3 + p4, label="P5 Share %", color="#2ca02c", alpha=0.88, edgecolor="#333", linewidth=0.5)
        for i in range(len(x)):
            if p3[i] > 12:
                ax_s.text(x[i], p3[i] / 2, f"{p3[i]:.0f}%", ha="center", va="center", color="white", fontsize=7.2, fontweight="bold")
            if p4[i] > 12:
                ax_s.text(x[i], p3[i] + p4[i] / 2, f"{p4[i]:.0f}%", ha="center", va="center", color="white", fontsize=7.2, fontweight="bold")
        ax_s.set_ylim(0, 108)
        ax_s.set_ylabel("Tỷ trọng Attention (%)", fontsize=10)
        ax_s.set_xticks(x)
        ax_s.set_xticklabels(names, fontsize=8.5, rotation=35, ha="right", fontweight="bold")
        ax_s.set_title("Slot Attention Mass Share Across Scales (P3 / P4 / P5)", fontsize=10.5, fontweight="bold")
        ax_s.legend(fontsize=8.5, loc="upper right")
    else:
        ax_s.axis("off")

    fig.suptitle("Multi-Scale (P3 / P4 / P5) Feature Contrast & Slot Attention Profile", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_05_multiscale_p3_p4_p5_profile.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_06_normalized_heatmap(df_stats: pd.DataFrame, out_dir: str) -> str:
    """Ma trận bảng điểm chuẩn hóa [0, 1] cho toàn bộ các kiến trúc trên các chỉ số then chốt."""
    _setup_mpl_style()
    spec = [(m, lbl, hib) for m, lbl, hib in KEY_METRICS_SPEC if f"{m}_mean" in df_stats.columns]
    n_rows = len(df_stats)
    n_cols = len(spec)

    norm_mat = np.zeros((n_rows, n_cols), dtype=float)
    text_mat: List[List[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for ci, (m, _, hib) in enumerate(spec):
        vals = df_stats[f"{m}_mean"].values.astype(float)
        stds = df_stats[f"{m}_std"].values.astype(float)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
        scaled = (vals - vmin) / denom
        if not hib:
            scaled = 1.0 - scaled
        norm_mat[:, ci] = scaled
        for ri in range(n_rows):
            dec = 1 if (m.endswith("_pct") or m in ("params_m", "train_sec")) else (3 if vals[ri] < 1.0 else 2)
            text_mat[ri][ci] = f"{vals[ri]:.{dec}f}\n±{stds[ri]:.{dec}f}"

    fig_w = max(13.5, n_cols * 1.25)
    fig_h = max(6.0, n_rows * 0.48 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(norm_mat, cmap="YlGn", aspect="auto", vmin=0.0, vmax=1.0)

    row_labels = [f"#{int(r['rank'])} {r['block'].upper()}" for _, r in df_stats.iterrows()]
    col_labels = [f"{lbl}\n({'↑' if hib else '↓'})" for _, lbl, hib in spec]

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8.5, rotation=25, ha="right")
    ax.grid(False)

    for ri in range(n_rows):
        for ci in range(n_cols):
            val = norm_mat[ri, ci]
            txt_color = "white" if val > 0.65 else "#111111"
            ax.text(ci, ri, text_mat[ri][ci], ha="center", va="center", color=txt_color, fontsize=7.5, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Điểm chuẩn hóa cột (1.0 = Tốt nhất ở cột đó)", fontsize=9)
    ax.set_title("Normalized Metrics Scorecard Matrix (Số liệu trong ô là Mean ± Std thực tế)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "chart_06_metrics_normalized_heatmap.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_07_run_consistency_lines(raw_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    """Biểu đồ đường theo dõi độ biến thiên qua từng run của từng block + Legend màu sắc."""
    if raw_df.empty:
        return None
    _setup_mpl_style()

    runs_sorted = sorted(raw_df["run"].unique(), key=lambda s: int(re.search(r"(\d+)$", s).group(1)) if re.search(r"(\d+)$", s) else 0)
    if len(runs_sorted) <= 1:
        return None

    blocks = list(raw_df["block"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.8))
    x = np.arange(len(runs_sorted))

    for pi, (ax, col, ylabel, title) in enumerate([
        (ax1, "val_mae", "Val MAE (↓ Thấp hơn là tốt hơn)", "7A. Quỹ đạo Val MAE qua các Run"),
        (ax2, "slot_class_cos_sep", "Slot Class Cos Sep (↑ Cao hơn là tốt hơn)", "7B. Quỹ đạo Slot Cosine Separation qua các Run"),
    ]):
        if col not in raw_df.columns:
            ax.axis("off")
            continue
        for b in blocks:
            sub_b = raw_df[raw_df["block"] == b].set_index("run")
            ys = [float(sub_b.loc[r, col]) if r in sub_b.index else np.nan for r in runs_sorted]
            c = BLOCK_COLORS.get(b, "#4C72B0")
            ax.plot(x, ys, marker="o", linewidth=1.8, markersize=5.5, label=b, color=c, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(runs_sorted, fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.legend(handles=make_backbone_legend_handles(blocks), loc="lower center", ncol=min(9, len(blocks)),
               fontsize=8.5, title="Màu sắc tương ứng các Kiến trúc Backbone", framealpha=0.92)

    fig.suptitle("Run-to-Run Consistency & Stability Across Random Seeds", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_07_run_consistency_lines.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_08_pairwise_slots_heatmap(
    df_pair_stats: pd.DataFrame,
    df_stats: pd.DataFrame,
    out_dir: str,
    top_k: int = 6,
) -> Optional[str]:
    """Lưới ma trận nhiệt 10x10 Cosine Separation cho Top-K kiến trúc có MAE tốt nhất."""
    if df_pair_stats.empty or df_stats.empty:
        return None

    _setup_mpl_style()
    classes = VISDRONE_CLASSES
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    short_labels = ["ped", "ppl", "bic", "car", "van", "trk", "tri", "awn", "bus", "mot"]

    top_blocks = df_stats.head(top_k)["block"].tolist()
    valid_blocks = [b for b in top_blocks if f"{b}_mean" in df_pair_stats.columns]
    if not valid_blocks:
        return None

    n_b = len(valid_blocks)
    ncols = min(3, n_b)
    nrows = int(np.ceil(n_b / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    all_means = [df_pair_stats[f"{b}_mean"].values for b in valid_blocks]
    vmax = max(0.08, float(np.nanmax(all_means)))

    for idx, b in enumerate(valid_blocks):
        ax = axes[idx]
        mat = np.zeros((n_cls, n_cls), dtype=float)
        mcol = f"{b}_mean"

        for _, r in df_pair_stats.iterrows():
            i, j = cls_to_idx[r["class_i"]], cls_to_idx[r["class_j"]]
            val = float(r[mcol])
            mat[i, j] = val
            mat[j, i] = val

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_xticks(np.arange(n_cls))
        ax.set_yticks(np.arange(n_cls))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.grid(False)

        mean_val = float(df_pair_stats[mcol].mean())
        rank_val = int(df_stats[df_stats["block"] == b]["rank"].iloc[0])
        ax.set_title(f"#{rank_val} {b.upper()}\nMean Separation: {mean_val:.4f}", fontsize=10, fontweight="bold")

        for i in range(n_cls):
            for j in range(n_cls):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center", color="#888888", fontsize=7)
                else:
                    v = mat[i, j]
                    txt_c = "white" if v > vmax * 0.62 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_c, fontsize=6.5)

    for j in range(n_b, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Pairwise 10x10 Cosine Separation on Final Slots (Top Backbone Architectures)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.colorbar(im, ax=list(axes[:n_b]), fraction=0.02, pad=0.02, label="Khoảng cách Cosine (1 - cos_sim)")
    out_path = os.path.join(out_dir, "chart_08_pairwise_slots_heatmap.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_09_pairwise_top_classes(
    df_pair_stats: pd.DataFrame,
    df_stats: pd.DataFrame,
    out_dir: str,
    top_k: int = 12,
) -> Optional[str]:
    """Top các cặp lớp phân tách tốt nhất vs thách thức nhất trên tập Final Slots."""
    if df_pair_stats.empty or "all_blocks_mean" not in df_pair_stats.columns:
        return None

    _setup_mpl_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    df_hard = df_pair_stats.sort_values(by="all_blocks_mean", ascending=True).head(top_k)
    y1 = np.arange(len(df_hard))
    vals1 = df_hard["all_blocks_mean"].values
    ax1.barh(y1, vals1, color="#d62728", alpha=0.85, edgecolor="#333", linewidth=0.6)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(df_hard["pair"].values, fontsize=8.5, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Cosine Distance (Thấp = Các lớp gần nhau, dễ nhầm lẫn)")
    ax1.set_title(f"Top {top_k} Cặp lớp khó phân tách nhất (Dễ nhầm lẫn)")
    for i, v in enumerate(vals1):
        ax1.text(v + max(vals1) * 0.02, i, f"{v:.4f}", va="center", fontsize=8, fontweight="bold")

    df_easy = df_pair_stats.sort_values(by="all_blocks_mean", ascending=False).head(top_k)
    y2 = np.arange(len(df_easy))
    vals2 = df_easy["all_blocks_mean"].values
    ax2.barh(y2, vals2, color="#2ca02c", alpha=0.85, edgecolor="#333", linewidth=0.6)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(df_easy["pair"].values, fontsize=8.5, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Cosine Distance (Cao = Không gian ngữ nghĩa phân tách rõ rệt)")
    ax2.set_title(f"Top {top_k} Cặp lớp phân tách tốt nhất")
    for i, v in enumerate(vals2):
        ax2.text(v + max(vals2) * 0.02, i, f"{v:.4f}", va="center", fontsize=8, fontweight="bold")

    fig.suptitle("Cross-Architecture Semantic Separation: Most Challenging vs. Best-Separated Class Pairs", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_09_pairwise_top_classes.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_10_all_metrics_correlation_with_mae(
    df_metric_impact: pd.DataFrame,
    out_dir: str,
) -> Optional[str]:
    """Biểu đồ thanh phân kỳ biểu diễn hệ số tương quan Pearson của TẤT CẢ các chỉ số đặc trưng tới Val MAE."""
    if df_metric_impact.empty:
        return None

    _setup_mpl_style()
    df_sorted = df_metric_impact.sort_values(by="pearson_r", ascending=True).copy()
    n_metrics = len(df_sorted)

    fig_h = max(6.0, 0.38 * n_metrics + 2.0)
    fig, ax = plt.subplots(figsize=(14.5, fig_h))

    y_pos = np.arange(n_metrics)
    r_vals = df_sorted["pearson_r"].values
    p_vals = df_sorted["pearson_p"].values
    metric_labels = df_sorted["metric"].values

    # Màu sắc: Xanh lá cây (r < 0: chỉ số tăng giúp giảm MAE), Đỏ (r > 0: chỉ số tăng làm tăng MAE)
    bar_colors = ["#2ca02c" if r < 0 else "#d62728" for r in r_vals]

    bars = ax.barh(y_pos, r_vals, color=bar_colors, alpha=0.88, edgecolor="#333333", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="-")
    ax.axvline(-0.3, color="#2ca02c", linewidth=0.8, linestyle="--", alpha=0.6, label="Ngưỡng tương quan nghịch vừa (r = -0.3)")
    ax.axvline(0.3, color="#d62728", linewidth=0.8, linestyle="--", alpha=0.6, label="Ngưỡng tương quan thuận vừa (r = +0.3)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_labels, fontsize=9.0, fontweight="bold")
    ax.invert_yaxis()

    ax.set_xlabel("Hệ số tương quan Pearson (r) với Val MAE\n← [r < 0]: Chỉ số tăng giúp GIẢM SAI SỐ MAE (Tốt)  |  [r > 0]: Chỉ số tăng làm TĂNG SAI SỐ MAE →", fontsize=10.5)
    ax.set_title("Correlation Profile: Architectural Features vs. Validation MAE", fontsize=12, fontweight="bold", pad=12)

    lim = max(0.65, float(np.max(np.abs(r_vals))) * 1.25)
    ax.set_xlim(-lim, lim)

    for i, (r, p) in enumerate(zip(r_vals, p_vals)):
        ha = "left" if r >= 0 else "right"
        pad = lim * 0.02 if r >= 0 else -lim * 0.02
        sig_star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        ax.text(r + pad, i, f"r={r:+.3f} (p={p:.3f}){sig_star}", va="center", ha=ha, fontsize=8, fontweight="bold")

    ax.legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "chart_10_all_metrics_correlation_with_mae.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_11_top_metric_scatter_regressions(
    df_metric_impact: pd.DataFrame,
    df_stats: pd.DataFrame,
    out_dir: str,
    top_k: int = 6,
) -> Optional[str]:
    """Lưới đồ thị phân tán kèm đường hồi quy thực nghiệm cho Top chỉ số tác động mạnh nhất tới Val MAE + Legend đầy đủ."""
    if df_metric_impact.empty or df_stats.empty:
        return None

    _setup_mpl_style()
    top_df = df_metric_impact.head(top_k)
    metrics_to_plot = top_df["metric"].tolist()
    n_plots = len(metrics_to_plot)
    if n_plots == 0:
        return None

    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.8 * nrows + 1.2))
    axes = np.atleast_1d(axes).ravel()

    names = [r["block"] for _, r in df_stats.iterrows()]
    y_mae = df_stats["val_mae_mean"].values

    for idx, m in enumerate(metrics_to_plot):
        ax = axes[idx]
        mean_col = f"{m}_mean"
        if mean_col not in df_stats.columns:
            ax.axis("off")
            continue

        x_vals = df_stats[mean_col].values
        r_val = float(top_df[top_df["metric"] == m]["pearson_r"].iloc[0])
        p_val = float(top_df[top_df["metric"] == m]["pearson_p"].iloc[0])

        for _, r in df_stats.iterrows():
            c = BLOCK_COLORS.get(r["block"], "#4C72B0")
            xv = float(r[mean_col])
            yv = float(r["val_mae_mean"])
            ax.scatter(xv, yv, color=c, s=55, edgecolor="black", linewidth=0.7, zorder=4)
            ax.annotate(r["block"], (xv, yv), textcoords="offset points", xytext=(4, 3), fontsize=7.5)

        # Đường hồi quy tuyến tính
        valid_m = np.isfinite(x_vals) & np.isfinite(y_mae)
        if valid_m.sum() >= 3:
            slope, intercept, r_value, p_value, _ = stats.linregress(x_vals[valid_m], y_mae[valid_m])
            x_line = np.linspace(x_vals[valid_m].min(), x_vals[valid_m].max(), 50)
            y_line = slope * x_line + intercept
            line_color = "#2ca02c" if slope < 0 else "#d62728"
            ax.plot(x_line, y_line, color=line_color, linestyle="--", linewidth=1.5,
                    label=f"Fit (R²={r_value**2:.2f})")

        ax.set_xlabel(m, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Val MAE (Mean)", fontsize=9.5)
        ax.set_title(f"#{idx+1} {m}\nr = {r_val:+.3f} (p = {p_val:.3f})", fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=8, loc="best")

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    fig.legend(handles=make_backbone_legend_handles(names), loc="lower center", ncol=min(9, len(names)),
               fontsize=8.5, title="Màu sắc tương ứng các Kiến trúc Backbone", framealpha=0.92)

    fig.suptitle("Regression Analysis: Top Features with Strongest Correlation to Val MAE", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_11_top_metric_scatter_regressions.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_12_pairwise_sep_mae_correlation(
    df_pair_impact: pd.DataFrame,
    df_class_impact: pd.DataFrame,
    out_dir: str,
    top_k_pairs: int = 12,
) -> Optional[str]:
    """Biểu đồ tương quan 45 cặp phân tách lớp và mức độ nhạy cảm của từng Class tới Val MAE."""
    if df_pair_impact.empty or df_class_impact.empty:
        return None

    _setup_mpl_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 7.0))

    # Panel 1: Top cặp có tương quan nghịch mạnh nhất tới MAE (Tách tốt cặp này giúp giảm lỗi đếm nhiều nhất)
    df_top_neg = df_pair_impact.sort_values(by="pearson_r", ascending=True).head(top_k_pairs)
    y1 = np.arange(len(df_top_neg))
    r1 = df_top_neg["pearson_r"].values
    ax1.barh(y1, r1, color="#2ca02c", alpha=0.88, edgecolor="#333", linewidth=0.6)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(df_top_neg["pair"].values, fontsize=8.5, fontweight="bold")
    ax1.invert_yaxis()
    ax1.axvline(0.0, color="black", linewidth=0.9)
    ax1.set_xlabel("Hệ số tương quan Pearson (r) với Val MAE\n← Càng âm: Phân tách cặp này càng tốt thì MAE càng giảm mạnh")
    ax1.set_title(f"Top {top_k_pairs} Cặp lớp đóng góp giảm MAE nhiều nhất", fontsize=11, fontweight="bold")
    for i, (r, p) in enumerate(zip(r1, df_top_neg["pearson_p"].values)):
        ax1.text(r - 0.02, i, f"r={r:+.3f} (p={p:.2f})", va="center", ha="right", fontsize=8, fontweight="bold")
    ax1.set_xlim(min(r1) * 1.35, 0.05)

    # Panel 2: Mức độ nhạy cảm của từng Class tới MAE (avg_abs_pearson)
    y2 = np.arange(len(df_class_impact))
    avg_abs = df_class_impact["avg_abs_pearson"].values
    avg_r = df_class_impact["avg_pearson_r"].values
    ax2.barh(y2, avg_abs, color="#3498db", alpha=0.88, edgecolor="#333", linewidth=0.6)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([f"#{r['class_rank']} {r['class']}" for _, r in df_class_impact.iterrows()],
                        fontsize=9.0, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Độ nhạy cảm trung bình (|r|) với Val MAE\nCao hơn = Class này là nút thắt quyết định sự chênh lệch MAE giữa các backbone")
    ax2.set_title("Class-Level Impact Ranking (Độ nhạy cảm cấp Class)", fontsize=11, fontweight="bold")
    for i, (val, ar) in enumerate(zip(avg_abs, avg_r)):
        ax2.text(val + max(avg_abs) * 0.02, i, f"Avg |r|={val:.3f} (Avg r={ar:+.3f})", va="center", fontsize=8, fontweight="bold")
    ax2.set_xlim(0, max(avg_abs) * 1.38)

    fig.suptitle("Impact of Semantic Class Separation on Final Counting Error (Val MAE)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_12_pairwise_sep_mae_correlation.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# HTML DASHBOARD GENERATOR
# =============================================================================

def _df_to_html_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "<p>No data.</p>"
    d = df.head(max_rows) if max_rows else df
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body_rows = []
    for _, row in d.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                s = f"{val:.4f}" if not np.isnan(val) else "-"
            else:
                s = str(val)
            cells.append(f"<td>{html.escape(s)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def generate_html_dashboard(
    summary_title: str,
    out_dir: str,
    ranking_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    class_sep_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    metric_impact_df: pd.DataFrame,
    pair_impact_df: pd.DataFrame,
    class_impact_df: pd.DataFrame,
    chart_files: List[str],
    csv_files: List[str],
) -> str:
    """Tạo trang HTML trực quan tổng hợp toàn bộ bảng, thống kê tương quan và biểu đồ."""
    chart_cards = []
    for cf in chart_files:
        fname = os.path.basename(cf)
        chart_cards.append(
            f"<div class='card'><h3>{html.escape(fname)}</h3>"
            f"<a href='{html.escape(fname)}' target='_blank'><img src='{html.escape(fname)}' alt='{html.escape(fname)}'></a></div>"
        )

    csv_links = "".join(
        f"<li><a href='{html.escape(os.path.basename(f))}'>{html.escape(os.path.basename(f))}</a></li>"
        for f in csv_files
    )

    delta_section = ""
    if not delta_df.empty:
        delta_section = f"<section><h2>2. So sánh Chênh lệch Delta & % Cải thiện so với Baseline</h2>{_df_to_html_table(delta_df)}</section>"

    impact_section = ""
    if not metric_impact_df.empty:
        impact_section = (
            f"<section><h2>3. Phân tích Tác động của TẤT CẢ các Chỉ số Đặc trưng tới Val MAE</h2>"
            f"<p class='meta'>Tương quan Pearson r < 0 biểu thị chỉ số tăng giúp giảm sai số đếm MAE (cải thiện hiệu năng). p < 0.05 là có ý nghĩa thống kê.</p>"
            f"{_df_to_html_table(metric_impact_df)}</section>"
        )

    cls_impact_section = ""
    if not class_impact_df.empty:
        cls_impact_section = (
            f"<section><h2>4. Đánh giá Mức độ Nhạy cảm & Tầm quan trọng cấp Class tới Val MAE</h2>"
            f"<p class='meta'>Class có Avg |r| cao nhất là nút thắt nhạy cảm nhất quyết định sự chênh lệch MAE giữa các kiến trúc.</p>"
            f"{_df_to_html_table(class_impact_df)}</section>"
        )

    pair_section = ""
    if not pairwise_df.empty:
        disp_cols = [c for c in pairwise_df.columns if c in ("pair", "n_runs", "all_blocks_mean") or c.endswith("_mean")]
        pair_section = (
            f"<section><h2>5. Bảng chi tiết 45 cặp lớp trên Final Slots</h2>"
            f"<details><summary><strong>Nhấn để mở rộng bảng 45 cặp class</strong></summary>"
            f"{_df_to_html_table(pairwise_df[disp_cols])}</details></section>"
        )

    best_block = ranking_df["block"].iloc[0] if not ranking_df.empty else "N/A"
    best_mae = ranking_df["val_mae_mean_std"].iloc[0] if not ranking_df.empty else "N/A"

    top_beneficial_metric = metric_impact_df.sort_values(by="pearson_r").iloc[0] if not metric_impact_df.empty else None
    top_metric_str = f"{top_beneficial_metric['metric']} (r = {top_beneficial_metric['pearson_r']:+.3f})" if top_beneficial_metric is not None else "N/A"

    top_critical_class = class_impact_df.iloc[0] if not class_impact_df.empty else None
    top_class_str = f"{top_critical_class['class']} (Avg |r| = {top_critical_class['avg_abs_pearson']:.3f})" if top_critical_class is not None else "N/A"

    html_doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>{html.escape(summary_title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f4f6f9; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .kpi-row {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .kpi-card {{ background: #fff; border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 220px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #4C72B0; }}
  .kpi-card h4 {{ margin: 0 0 6px 0; color: #666; font-size: 0.88em; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-card .val {{ font-size: 1.45em; font-weight: bold; color: #111; }}
  section {{ background: #fff; border-radius: 8px; padding: 18px 20px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  h2 {{ margin-top: 0; border-bottom: 2px solid #eef1f5; padding-bottom: 6px; font-size: 1.2em; }}
  .meta {{ color: #555; margin-bottom: 12px; font-size: 0.92em; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #e1e5eb; padding: 6px 10px; text-align: right; white-space: nowrap; }}
  th {{ background: #f0f3f8; font-weight: 600; text-align: center; position: sticky; top: 0; }}
  td:nth-child(1), td:nth-child(2) {{ text-align: left; }}
  tr:nth-child(even) {{ background: #fafbfd; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
  .card {{ background: #fff; border: 1px solid #e5e9f0; border-radius: 8px; padding: 14px; }}
  .card h3 {{ margin: 0 0 10px 0; font-size: 1em; color: #333; }}
  .card img {{ max-width: 100%; height: auto; display: block; border-radius: 4px; }}
  details {{ margin-top: 10px; }}
  summary {{ cursor: pointer; padding: 8px; background: #f0f3f8; border-radius: 4px; }}
  ul.files {{ columns: 2; font-size: 13.5px; }}
</style>
</head>
<body>
  <h1>{html.escape(summary_title)}</h1>
  <div class="kpi-row">
    <div class="kpi-card">
      <h4>KIẾN TRÚC XẾP HẠNG #1</h4>
      <div class="val">{html.escape(best_block.upper())}</div>
    </div>
    <div class="kpi-card">
      <h4>VAL MAE THẤP NHẤT</h4>
      <div class="val">{html.escape(str(best_mae))}</div>
    </div>
    <div class="kpi-card">
      <h4>CHỈ SỐ GIẢM MAE MẠNH NHẤT</h4>
      <div class="val" style="font-size: 1.15em; color: #2ca02c;">{html.escape(top_metric_str)}</div>
    </div>
    <div class="kpi-card">
      <h4>CLASS NHẠY CẢM NHẤT</h4>
      <div class="val" style="font-size: 1.15em; color: #3498db;">{html.escape(top_class_str)}</div>
    </div>
  </div>

  <section>
    <h2>0. Danh sách các File CSV Báo cáo đã xuất</h2>
    <ul class="files">{csv_links}</ul>
  </section>

  <section>
    <h2>1. Bảng Xếp hạng Tổng thể (Mean ± Std qua các Run)</h2>
    {_df_to_html_table(ranking_df)}
  </section>

  {delta_section}
  {impact_section}
  {cls_impact_section}
  {pair_section}

  <section>
    <h2>6. Bộ Biểu đồ Trực quan hóa (Có Đầy đủ Chú thích Màu sắc)</h2>
    <div class="grid">
      {''.join(chart_cards)}
    </div>
  </section>
</body>
</html>
"""
    dash_path = os.path.join(out_dir, "dashboard.html")
    with open(dash_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return dash_path


# =============================================================================
# CONSOLE PRINTER & MAIN
# =============================================================================

def print_console_ranking(df_stats: pd.DataFrame):
    print("\n" + "=" * 145)
    print("  BẢNG XẾP HẠNG BACKBONE BENCHMARK (Ưu tiên 1: Mean Val MAE asc | Ưu tiên 2: Std Val MAE asc)")
    print("=" * 145)
    print(
        f"{'Rank':<5} | {'Block':<15} | {'Runs':<4} | {'Params':<8} | "
        f"{'Val MAE (Mean±Std)':<20} | {'CV%':<6} | {'Slot ClsSep':<14} | {'Slot Act/Inact':<14} | {'P3 Small':<12}"
    )
    print("-" * 145)
    for _, r in df_stats.iterrows():
        mae_s = f"{r['val_mae_mean']:.4f} ± {r['val_mae_std']:.4f}"
        cv_s = f"{r['val_mae_cv_pct']:.2f}%"
        params_s = f"{r.get('params_m_mean', 0):.2f}M"
        cls_s = f"{r.get('slot_class_cos_sep_mean', 0):.4f}±{r.get('slot_class_cos_sep_std', 0):.4f}"
        act_s = f"{r.get('slot_act_inact_ratio_mean', 0):.2f}±{r.get('slot_act_inact_ratio_std', 0):.2f}x"
        p3_s = f"{r.get('p3_small_ratio_mean', 0):.2f}±{r.get('p3_small_ratio_std', 0):.2f}x"
        print(
            f"{int(r['rank']):<5} | {r['block']:<15} | {int(r['n_runs']):<4} | {params_s:<8} | "
            f"{mae_s:<20} | {cv_s:<6} | {cls_s:<14} | {act_s:<14} | {p3_s:<12}"
        )
    print("=" * 145)


def print_console_impact(df_impact: pd.DataFrame):
    if df_impact.empty:
        return
    print("\n" + "=" * 145)
    print("  TOP CÁC CHỈ SỐ ĐẶC TRƯNG KIẾN TRÚC CÓ TƯƠNG QUAN MẠNH NHẤT TỚI VAL MAE")
    print("=" * 145)
    print(
        f"{'Rank':<5} | {'Metric Name':<28} | {'Pearson r':<11} | {'p-value':<9} | "
        f"{'Spearman rho':<13} | {'Ridge Coef':<11} | {'Diễn giải xu hướng':<38}"
    )
    print("-" * 145)
    for _, r in df_impact.head(10).iterrows():
        ridge_s = f"{r.get('ridge_std_coef', 0):+.4f}" if pd.notna(r.get('ridge_std_coef')) else "N/A"
        print(
            f"{int(r['impact_rank']):<5} | {r['metric']:<28} | {r['pearson_r']:>+9.4f}  | {r['pearson_p']:>8.4f}  | "
            f"{r['spearman_rho']:>+11.4f}  | {ridge_s:>10}  | {r['interpretation']:<38}"
        )
    print("=" * 145)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-run benchmark results into summary_benchmark_<X> with CSVs, correlation analysis, charts, and HTML dashboard."
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="*",
        default=["Visdrone/output_benchmark_40"],
        help="Danh sách đường dẫn folder kết quả benchmark (Mặc định: Visdrone/output_benchmark_40)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Tự động quét tất cả các folder output_benchmark* trong --base-dir",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="Visdrone",
        help="Thư mục cơ sở (mặc định: Visdrone)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Ghi đè đường dẫn thư mục xuất kết quả (mặc định: <base-dir>/summary_<benchmark_name>)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Kiến trúc baseline để so sánh delta (mặc định: resnet nếu có, hoặc rank #1)",
    )
    args = parser.parse_args()

    target_dirs = list(args.dirs)
    if args.auto:
        matched = sorted(glob.glob(os.path.join(args.base_dir, "output_benchmark*")))
        target_dirs = [d for d in matched if os.path.isdir(d)]
        if not target_dirs:
            print(f"[!] Không tìm thấy thư mục dạng 'output_benchmark*' trong '{args.base_dir}'.")
            return

    target_dirs = [os.path.normpath(d) for d in target_dirs if os.path.isdir(d)]
    if not target_dirs:
        print("[!] Không có thư mục hợp lệ nào để xử lý.")
        return

    for f_idx, folder_path in enumerate(target_dirs):
        folder_name, bench_name, metric_runs, pairwise_slots_runs, pairwise_p3_runs = load_benchmark_folder_data(folder_path)

        if not metric_runs:
            print(f"[!] Thư mục '{folder_path}' không chứa file 'benchmark_results.csv' nào. Bỏ qua.")
            continue

        print("\n" + "#" * 120)
        print(f"  ĐANG TỔNG HỢP BENCHMARK: {bench_name.upper()} ({len(metric_runs)} runs tìm thấy)")
        print(f"  Đường dẫn nguồn: {os.path.abspath(folder_path)}")
        print("#" * 120)

        # 1. Thư mục xuất
        if args.out_dir is not None:
            out_dir = args.out_dir if len(target_dirs) == 1 else os.path.join(args.out_dir, f"summary_{bench_name}")
        else:
            out_dir = os.path.join(args.base_dir, f"summary_{bench_name}")
        os.makedirs(out_dir, exist_ok=True)

        # 2. Xây dựng DataFrames thống kê
        raw_df = build_raw_runs_df(bench_name, folder_name, metric_runs)
        df_stats = compute_detailed_stats(raw_df)
        ranking_df = build_ranking_display_df(df_stats)
        delta_df = compute_delta_vs_baseline(df_stats, baseline_block=args.baseline)

        df_pair_slots = compute_pairwise_stats_across_runs(pairwise_slots_runs)
        df_pair_p3 = compute_pairwise_stats_across_runs(pairwise_p3_runs)
        df_class_sep = compute_class_level_separation(df_pair_slots)

        # 3. Phân tích tác động của TẤT CẢ các chỉ số và các cặp phân tách tới Val MAE
        print("[*] Đang phân tích tương quan và mức độ tác động của các chỉ số kiến trúc tới Val MAE...")
        df_metric_impact = compute_all_metrics_mae_impact(df_stats, raw_df)
        df_pair_impact, df_class_impact = compute_pairwise_mae_impact(df_pair_slots, df_stats)

        # In bảng console
        print_console_ranking(df_stats)
        print_console_impact(df_metric_impact)

        # 4. Xuất các file CSV
        csv_files = []
        raw_csv = os.path.join(out_dir, "benchmark_raw_runs.csv")
        raw_df.to_csv(raw_csv, index=False)
        csv_files.append(raw_csv)

        stats_csv = os.path.join(out_dir, "benchmark_detailed_stats.csv")
        df_stats.to_csv(stats_csv, index=False)
        csv_files.append(stats_csv)

        ranking_csv = os.path.join(out_dir, "benchmark_ranking_summary.csv")
        ranking_df.to_csv(ranking_csv, index=False)
        csv_files.append(ranking_csv)

        if not delta_df.empty:
            delta_csv = os.path.join(out_dir, "benchmark_delta_vs_baseline.csv")
            delta_df.to_csv(delta_csv, index=False)
            csv_files.append(delta_csv)

        if not df_pair_slots.empty:
            pair_slots_csv = os.path.join(out_dir, "benchmark_pairwise_slots_stats.csv")
            df_pair_slots.to_csv(pair_slots_csv, index=False)
            csv_files.append(pair_slots_csv)

        if not df_pair_p3.empty:
            pair_p3_csv = os.path.join(out_dir, "benchmark_pairwise_p3_stats.csv")
            df_pair_p3.to_csv(pair_p3_csv, index=False)
            csv_files.append(pair_p3_csv)

        if not df_class_sep.empty:
            cls_sep_csv = os.path.join(out_dir, "benchmark_class_level_separation.csv")
            df_class_sep.to_csv(cls_sep_csv, index=False)
            csv_files.append(cls_sep_csv)

        if not df_metric_impact.empty:
            metric_impact_csv = os.path.join(out_dir, "benchmark_metrics_mae_impact.csv")
            df_metric_impact.to_csv(metric_impact_csv, index=False)
            csv_files.append(metric_impact_csv)

        if not df_pair_impact.empty:
            pair_impact_csv = os.path.join(out_dir, "benchmark_pairwise_mae_impact.csv")
            df_pair_impact.to_csv(pair_impact_csv, index=False)
            csv_files.append(pair_impact_csv)

        if not df_class_impact.empty:
            class_impact_csv = os.path.join(out_dir, "benchmark_class_level_mae_impact.csv")
            df_class_impact.to_csv(class_impact_csv, index=False)
            csv_files.append(class_impact_csv)

        # 5. Tạo bộ biểu đồ trực quan
        chart_files = []
        print("[*] Đang vẽ bộ biểu đồ trực quan hóa (có đầy đủ chú thích màu sắc)...")

        c1 = plot_01_mae_ranking_and_distribution(df_stats, raw_df, out_dir)
        chart_files.append(c1)

        c2 = plot_02_accuracy_vs_efficiency_scatter(df_stats, out_dir)
        chart_files.append(c2)

        c3 = plot_03_accuracy_vs_stability_scatter(df_stats, out_dir)
        chart_files.append(c3)

        c4 = plot_04_key_metrics_grouped_bars(df_stats, out_dir)
        chart_files.append(c4)

        c5 = plot_05_multiscale_profile(df_stats, out_dir)
        chart_files.append(c5)

        c6 = plot_06_normalized_heatmap(df_stats, out_dir)
        chart_files.append(c6)

        c7 = plot_07_run_consistency_lines(raw_df, out_dir)
        if c7:
            chart_files.append(c7)

        c8 = plot_08_pairwise_slots_heatmap(df_pair_slots, df_stats, out_dir)
        if c8:
            chart_files.append(c8)

        c9 = plot_09_pairwise_top_classes(df_pair_slots, df_stats, out_dir)
        if c9:
            chart_files.append(c9)

        c10 = plot_10_all_metrics_correlation_with_mae(df_metric_impact, out_dir)
        if c10:
            chart_files.append(c10)

        c11 = plot_11_top_metric_scatter_regressions(df_metric_impact, df_stats, out_dir)
        if c11:
            chart_files.append(c11)

        c12 = plot_12_pairwise_sep_mae_correlation(df_pair_impact, df_class_impact, out_dir)
        if c12:
            chart_files.append(c12)

        # 6. Tạo Dashboard HTML
        dash_path = generate_html_dashboard(
            summary_title=f"Backbone Benchmark Analysis Dashboard — {bench_name.upper()}",
            out_dir=out_dir,
            ranking_df=ranking_df,
            delta_df=delta_df,
            class_sep_df=df_class_sep,
            pairwise_df=df_pair_slots,
            metric_impact_df=df_metric_impact,
            pair_impact_df=df_pair_impact,
            class_impact_df=df_class_impact,
            chart_files=chart_files,
            csv_files=csv_files,
        )

        print("\n[+] ĐÃ HOÀN TẤT TỔNG HỢP VÀ PHÂN TÍCH TÁC ĐỘNG TỚI VAL MAE:")
        print(f"    - Thư mục báo cáo           : {os.path.abspath(out_dir)}")
        print(f"    - Trang Dashboard           : {os.path.abspath(dash_path)}")
        print(f"    - CSV Xếp hạng              : {os.path.abspath(ranking_csv)}")
        print(f"    - CSV Tác động các chỉ số   : {os.path.abspath(os.path.join(out_dir, 'benchmark_metrics_mae_impact.csv'))}")
        print(f"    - CSV Tác động 45 cặp class : {os.path.abspath(os.path.join(out_dir, 'benchmark_pairwise_mae_impact.csv'))}")
        print(f"    - CSV Tác động cấp Class    : {os.path.abspath(os.path.join(out_dir, 'benchmark_class_level_mae_impact.csv'))}")


if __name__ == "__main__":
    main()
