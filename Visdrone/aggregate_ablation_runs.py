#!/usr/bin/env python3
"""Data Analyst Aggregation & Visualization Suite for Neck ReLU Ablation Runs.

Đầu vào:
  - 1 hoặc nhiều thư mục dạng `output_ablation_neck_relu_<backbone>` chứa `run_1`, `run_2`, ...

Đầu ra:
  - Thư mục `summary_<backboneA>_<backboneB>_...` gồm:
      1. Các file CSV thống kê chi tiết (Ranking, Full Mean/Std/Min/Max/Median/CV%, Raw Runs, Delta vs Baseline, Pairwise 45 Pairs, Backbone Summary).
      2. Bộ biểu đồ trực quan chất lượng cao (PNG 180 DPI):
         - chart_01_mae_ranking_and_distribution.png
         - chart_02_mae_vs_stability_scatter.png
         - chart_03_key_metrics_grouped_bars.png
         - chart_04_multiscale_p3_p4_p5_profile.png
         - chart_05_metrics_normalized_heatmap.png
         - chart_06_delta_vs_with_relu.png
         - chart_07_run_consistency_lines.png
         - chart_08_pairwise_slots_heatmap_<backbone>.png
         - chart_09_pairwise_top_diff_<backbone>.png
      3. `dashboard.html`: Trang tổng hợp toàn bộ bảng số liệu và biểu đồ để mở trực tiếp trên trình duyệt.

Ví dụ chạy:
  python Visdrone/aggregate_ablation_runs.py --dirs Visdrone/output_ablation_neck_relu_senet
  python Visdrone/aggregate_ablation_runs.py --dirs Visdrone/output_ablation_neck_relu_senet Visdrone/output_ablation_neck_relu_resnet
  python Visdrone/aggregate_ablation_runs.py --auto
"""
from __future__ import annotations

import argparse
import glob
import html
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = [
    "neck_with_relu",
    "neck_no_last_relu",
    "neck_preserve_bu_relu",
    "neck_silu",
]

VARIANT_COLORS = {
    "neck_with_relu": "#4C72B0",
    "neck_no_last_relu": "#DD8452",
    "neck_preserve_bu_relu": "#55A868",
    "neck_silu": "#C44E52",
}

KEY_METRICS_SPEC = [
    ("val_mae", "Val MAE", False),
    ("slot_act_inact_ratio", "Slot Act/Inact Ratio", True),
    ("slot_class_cos_sep", "Slot Class Cos Sep", True),
    ("slot_global_45pair_sep", "Slot Global 45-Pair Sep", True),
    ("slot_full_fisher", "Slot Full Fisher", True),
    ("slot_fg_mass_pct", "Slot FG Mass (%)", True),
    ("p3_small_ratio", "P3 Small Ratio", True),
    ("p3_full_fisher", "P3 Full Fisher", True),
    ("p3_class_cos_sep", "P3 Class Cos Sep", True),
]


def extract_backbone_name(folder_path: str) -> str:
    """Trích xuất tên backbone ngắn gọn từ tên thư mục."""
    base = os.path.basename(os.path.normpath(folder_path))
    prefix = "output_ablation_neck_relu_"
    if base.startswith(prefix):
        return base[len(prefix):]
    return base


def find_run_subdirs(parent_dir: str) -> List[str]:
    """Tìm tất cả các thư mục run_X hoặc runX bên trong parent_dir."""
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


def load_folder_data(
    folder_path: str,
) -> Tuple[str, str, List[Tuple[str, pd.DataFrame]], List[Tuple[str, pd.DataFrame]]]:
    """Đọc toàn bộ file metrics và pairwise CSV từ các run trong folder_path."""
    folder_name = os.path.basename(os.path.normpath(folder_path))
    backbone_name = extract_backbone_name(folder_path)
    run_dirs = find_run_subdirs(folder_path)

    metric_runs: List[Tuple[str, pd.DataFrame]] = []
    pairwise_runs: List[Tuple[str, pd.DataFrame]] = []

    if not run_dirs:
        direct_csv = os.path.join(folder_path, "ablation_neck_relu_results.csv")
        if os.path.isfile(direct_csv):
            metric_runs.append(("run_1", pd.read_csv(direct_csv)))
        direct_pair = os.path.join(folder_path, "pairwise_cosine_sep_slots.csv")
        if os.path.isfile(direct_pair):
            pairwise_runs.append(("run_1", pd.read_csv(direct_pair)))
    else:
        for r_dir in run_dirs:
            run_label = os.path.basename(r_dir)
            csv_path = os.path.join(r_dir, "ablation_neck_relu_results.csv")
            if os.path.isfile(csv_path):
                try:
                    metric_runs.append((run_label, pd.read_csv(csv_path)))
                except Exception as e:
                    print(f"[!] Lỗi đọc {csv_path}: {e}")

            pair_path = os.path.join(r_dir, "pairwise_cosine_sep_slots.csv")
            if os.path.isfile(pair_path):
                try:
                    pairwise_runs.append((run_label, pd.read_csv(pair_path)))
                except Exception as e:
                    print(f"[!] Lỗi đọc {pair_path}: {e}")

    return folder_name, backbone_name, metric_runs, pairwise_runs


def build_raw_runs_df(
    backbone_name: str,
    folder_name: str,
    metric_runs: List[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Gộp dữ liệu thô từng run thành 1 DataFrame chuẩn."""
    frames = []
    for run_label, df in metric_runs:
        d = df.copy()
        d.insert(0, "run", run_label)
        d.insert(0, "folder", folder_name)
        d.insert(0, "backbone", backbone_name)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_detailed_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Tính mean, std, min, max, median, cv_pct cho từng cặp (backbone, variant)."""
    if raw_df.empty:
        return pd.DataFrame()

    meta_cols = {"backbone", "folder", "run", "variant"}
    numeric_cols = [c for c in raw_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(raw_df[c])]

    rows = []
    for (backbone, folder, variant), grp in raw_df.groupby(["backbone", "folder", "variant"], sort=False):
        n_runs = len(grp)
        ddof = 1 if n_runs > 1 else 0
        row: Dict[str, object] = {
            "backbone": backbone,
            "folder": folder,
            "variant": variant,
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
    # Xếp hạng: ưu tiên 1 = val_mae_mean tăng dần, ưu tiên 2 = val_mae_std tăng dần
    df_stats = df_stats.sort_values(by=["val_mae_mean", "val_mae_std"], ascending=[True, True]).reset_index(drop=True)
    df_stats.insert(0, "rank", df_stats.index + 1)
    return df_stats


def build_ranking_display_df(df_stats: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng xếp hạng gọn gàng với cả chuỗi 'Mean ± Std' lẫn số thực."""
    cols_to_format = [
        "val_mae",
        "slot_act_inact_ratio",
        "slot_class_cos_sep",
        "slot_global_45pair_sep",
        "slot_full_fisher",
        "slot_fg_mass_pct",
        "p3_small_ratio",
        "p3_full_fisher",
        "p3_class_cos_sep",
        "params_m",
        "train_sec",
    ]
    rows = []
    for _, r in df_stats.iterrows():
        out: Dict[str, object] = {
            "rank": int(r["rank"]),
            "backbone": r["backbone"],
            "variant": r["variant"],
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


def compute_delta_vs_baseline(df_stats: pd.DataFrame, baseline_variant: str = "neck_with_relu") -> pd.DataFrame:
    """Tính chênh lệch tuyệt đối và % thay đổi của từng biến thể so với baseline trong cùng backbone."""
    metrics = [m for m, _, _ in KEY_METRICS_SPEC if f"{m}_mean" in df_stats.columns]
    rows = []
    for backbone, grp in df_stats.groupby("backbone", sort=False):
        base_rows = grp[grp["variant"] == baseline_variant]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        for _, r in grp.iterrows():
            row: Dict[str, object] = {
                "backbone": backbone,
                "variant": r["variant"],
                "baseline": baseline_variant,
            }
            for m in metrics:
                bv = float(base[f"{m}_mean"])
                cv = float(r[f"{m}_mean"])
                diff = cv - bv
                pct = (diff / abs(bv) * 100.0) if abs(bv) > 1e-12 else 0.0
                row[f"{m}_mean"] = round(cv, 4)
                row[f"{m}_diff"] = round(diff, 4)
                row[f"{m}_pct_change"] = round(pct, 2)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_backbone_agg_df(df_stats: pd.DataFrame) -> pd.DataFrame:
    """Thống kê tổng quan ở cấp độ Backbone (tổng hợp qua các biến thể Neck)."""
    rows = []
    for backbone, grp in df_stats.groupby("backbone", sort=False):
        best_row = grp.sort_values(by=["val_mae_mean", "val_mae_std"], ascending=[True, True]).iloc[0]
        rows.append({
            "backbone": backbone,
            "n_variants": len(grp),
            "n_runs_per_variant": int(grp["n_runs"].iloc[0]),
            "best_variant_by_mae": best_row["variant"],
            "best_val_mae_mean": round(float(best_row["val_mae_mean"]), 4),
            "best_val_mae_std": round(float(best_row["val_mae_std"]), 4),
            "avg_val_mae_across_variants": round(float(grp["val_mae_mean"].mean()), 4),
            "avg_val_mae_std_across_variants": round(float(grp["val_mae_std"].mean()), 4),
            "avg_slot_class_cos_sep": round(float(grp["slot_class_cos_sep_mean"].mean()), 4) if "slot_class_cos_sep_mean" in grp else np.nan,
            "avg_p3_small_ratio": round(float(grp["p3_small_ratio_mean"].mean()), 4) if "p3_small_ratio_mean" in grp else np.nan,
            "params_m": round(float(grp["params_m_mean"].mean()), 4) if "params_m_mean" in grp else np.nan,
            "avg_train_sec": round(float(grp["train_sec_mean"].mean()), 2) if "train_sec_mean" in grp else np.nan,
        })
    df_bb = pd.DataFrame(rows)
    if not df_bb.empty:
        df_bb = df_bb.sort_values(by=["best_val_mae_mean", "best_val_mae_std"], ascending=[True, True]).reset_index(drop=True)
        df_bb.insert(0, "backbone_rank", df_bb.index + 1)
    return df_bb


def compute_pairwise_stats(
    backbone_name: str,
    pairwise_runs: List[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Tính Mean, Std, Min, Max cho 45 cặp lớp từ pairwise_cosine_sep_slots.csv."""
    if not pairwise_runs:
        return pd.DataFrame()
    all_pairs = pd.concat([df for _, df in pairwise_runs], ignore_index=True)
    meta_cols = ["class_i", "class_j", "pair"]
    num_cols = [c for c in all_pairs.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(all_pairs[c])]
    n_runs = len(pairwise_runs)
    ddof = 1 if n_runs > 1 else 0

    rows = []
    for (ci, cj, pair), grp in all_pairs.groupby(meta_cols, sort=False):
        row: Dict[str, object] = {
            "backbone": backbone_name,
            "class_i": ci,
            "class_j": cj,
            "pair": pair,
            "n_runs": len(grp),
        }
        for col in num_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=ddof)) if len(vals) else 0.0
            row[f"{col}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{col}_max"] = float(vals.max()) if len(vals) else np.nan
        if "neck_with_relu_mean" in row and "neck_no_last_relu_mean" in row:
            base_v = float(row["neck_with_relu_mean"])
            no_v = float(row["neck_no_last_relu_mean"])
            row["diff_no_minus_with"] = no_v - base_v
            row["pct_no_vs_with"] = ((no_v - base_v) / (abs(base_v) + 1e-8)) * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION SUITE (PURE OBJECTIVE DATA CHARTS)
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


def plot_01_mae_ranking_and_distribution(df_stats: pd.DataFrame, raw_df: pd.DataFrame, out_dir: str) -> str:
    """Horizontal ranked bar chart (Mean ± Std) + Run-level boxplot & scatter."""
    _setup_mpl_style()
    n_rows = len(df_stats)
    fig_h = max(4.5, 0.55 * n_rows + 1.8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_h), sharey=True)

    labels = [f"#{int(r['rank'])} {r['backbone']} | {r['variant']}" for _, r in df_stats.iterrows()]
    y_pos = np.arange(n_rows)
    colors = [VARIANT_COLORS.get(r["variant"], "#666666") for _, r in df_stats.iterrows()]

    means = df_stats["val_mae_mean"].values
    stds = df_stats["val_mae_std"].values

    bars = ax1.barh(y_pos, means, xerr=stds, capsize=4, color=colors, alpha=0.88, edgecolor="#333333", linewidth=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Val MAE (Mean ± Std) — Lower is better")
    ax1.set_title("1A. Ranked Val MAE (Mean ± Std)")

    x_min = max(0.0, float(np.min(means - stds)) * 0.92)
    x_max = float(np.max(means + stds)) * 1.12
    ax1.set_xlim(x_min, x_max)

    for idx, (m, s) in enumerate(zip(means, stds)):
        ax1.text(m + s + (x_max - x_min) * 0.012, idx, f"{m:.4f} ± {s:.4f}", va="center", fontsize=8.5, fontweight="bold")

    # Right panel: Boxplot + individual run points
    run_series = []
    for _, r in df_stats.iterrows():
        sub = raw_df[(raw_df["backbone"] == r["backbone"]) & (raw_df["variant"] == r["variant"])]["val_mae"].dropna().values
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
                    edgecolor="black", linewidth=0.5, s=36, zorder=5)

    ax2.set_xlabel("Val MAE (Individual Runs Distribution)")
    ax2.set_title("1B. Val MAE Distribution Across Runs")

    fig.suptitle("Val MAE Ranking & Cross-Run Distribution", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_01_mae_ranking_and_distribution.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_02_mae_vs_stability_scatter(df_stats: pd.DataFrame, out_dir: str) -> str:
    """Scatter plot: Mean Val MAE (X) vs Std Val MAE (Y)."""
    _setup_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 6.5))

    backbones = list(df_stats["backbone"].unique())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    bb_marker = {bb: markers[i % len(markers)] for i, bb in enumerate(backbones)}

    for _, r in df_stats.iterrows():
        c = VARIANT_COLORS.get(r["variant"], "#555555")
        m = bb_marker[r["backbone"]]
        ax.scatter(r["val_mae_mean"], r["val_mae_std"], color=c, marker=m, s=130, edgecolor="black", linewidth=0.8, zorder=5)
        lbl = f"#{int(r['rank'])} {r['backbone']}/{r['variant'].replace('neck_', '')}"
        ax.annotate(lbl, (r["val_mae_mean"], r["val_mae_std"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8.5)

    # Legend for variants and backbones
    var_handles = [plt.Line2D([0], [0], marker="o", color="w", label=v, markerfacecolor=VARIANT_COLORS.get(v, "#555"),
                              markeredgecolor="black", markersize=8) for v in df_stats["variant"].unique()]
    bb_handles = [plt.Line2D([0], [0], marker=bb_marker[b], color="w", label=f"backbone: {b}", markerfacecolor="#DDDDDD",
                             markeredgecolor="black", markersize=8) for b in backbones]
    ax.legend(handles=var_handles + bb_handles, loc="best", fontsize=8.5, framealpha=0.9)

    ax.set_xlabel("Mean Val MAE (Lower = Better Accuracy)")
    ax.set_ylabel("Std Val MAE (Lower = Higher Stability)")
    ax.set_title("Accuracy vs. Stability: Mean Val MAE vs. Std Val MAE")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "chart_02_mae_vs_stability_scatter.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_03_key_metrics_grouped_bars(df_stats: pd.DataFrame, out_dir: str) -> str:
    """2x3 grid of grouped bar charts (Mean ± Std) for 6 primary metrics."""
    _setup_mpl_style()
    metrics = [
        ("val_mae", "Val MAE (↓)"),
        ("slot_act_inact_ratio", "Slot Act/Inact Ratio (↑)"),
        ("slot_class_cos_sep", "Slot Class Cos Sep (↑)"),
        ("slot_global_45pair_sep", "Slot Global 45-Pair Sep (↑)"),
        ("p3_small_ratio", "P3 Small Ratio (↑)"),
        ("slot_full_fisher", "Slot Full Fisher (↑)"),
    ]
    metrics = [(m, t) for m, t in metrics if f"{m}_mean" in df_stats.columns]
    backbones = list(df_stats["backbone"].unique())
    variants = [v for v in VARIANT_ORDER if v in df_stats["variant"].unique()]
    for v in df_stats["variant"].unique():
        if v not in variants:
            variants.append(v)

    n_m = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(len(backbones))
    bar_w = 0.78 / max(1, len(variants))

    for idx, (m_col, title) in enumerate(metrics):
        ax = axes[idx]
        for vi, var in enumerate(variants):
            means, stds = [], []
            for bb in backbones:
                sub = df_stats[(df_stats["backbone"] == bb) & (df_stats["variant"] == var)]
                if not sub.empty:
                    means.append(float(sub[f"{m_col}_mean"].iloc[0]))
                    stds.append(float(sub[f"{m_col}_std"].iloc[0]))
                else:
                    means.append(0.0)
                    stds.append(0.0)
            offset = (vi - (len(variants) - 1) / 2.0) * bar_w
            bars = ax.bar(x + offset, means, width=bar_w * 0.92, yerr=stds, capsize=3,
                          label=var, color=VARIANT_COLORS.get(var, "#777777"), edgecolor="#333333", linewidth=0.6, alpha=0.9)
            for b, mv, sv in zip(bars, means, stds):
                if mv != 0:
                    ax.text(b.get_x() + b.get_width() / 2.0, mv + sv, f"{mv:.3f}",
                            ha="center", va="bottom", fontsize=7.5, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontweight="bold")
        ax.set_title(title)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(n_m, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Key Metrics Comparison Across Backbones & Neck Variants (Mean ± Std)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_03_key_metrics_grouped_bars.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_04_multiscale_profile(df_stats: pd.DataFrame, out_dir: str) -> str:
    """Multi-scale P3/P4/P5 comparison + Slot Attention Share stacked bars."""
    _setup_mpl_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    labels = [f"{r['backbone']}\n{r['variant'].replace('neck_', '')}" for _, r in df_stats.iterrows()]
    x = np.arange(len(labels))
    w = 0.25

    scale_panels = [
        (axes[0, 0], ("p3_small_ratio", "p4_small_ratio", "p5_small_ratio"), "Small Object Energy Ratio (P3 / P4 / P5)"),
        (axes[0, 1], ("p3_full_fisher", "p4_full_fisher", "p5_full_fisher"), "FG/BG Full Fisher Ratio (P3 / P4 / P5)"),
        (axes[1, 0], ("p3_class_cos_sep", "p4_class_cos_sep", "p5_class_cos_sep"), "Class Cosine Separation (P3 / P4 / P5)"),
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
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # Bottom-right: Stacked Slot Attention Share %
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
            ax_s.text(x[i], p3[i] / 2, f"{p3[i]:.1f}%", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
            ax_s.text(x[i], p3[i] + p4[i] / 2, f"{p4[i]:.1f}%", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
        ax_s.set_ylim(0, 108)
        ax_s.set_ylabel("Percentage (%)")
        ax_s.set_xticks(x)
        ax_s.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax_s.set_title("Slot Attention Mass Share Across Scales (P3 / P4 / P5)")
        ax_s.legend(fontsize=8, loc="upper right")
    else:
        ax_s.axis("off")

    fig.suptitle("Multi-Scale Feature & Slot Attention Profile (Ordered by MAE Rank)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_04_multiscale_p3_p4_p5_profile.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_05_normalized_heatmap(df_stats: pd.DataFrame, out_dir: str) -> str:
    """Normalized [0,1] performance matrix heatmap with Mean ± Std annotations."""
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
            dec = 1 if m.endswith("_pct") else 3
            text_mat[ri][ci] = f"{vals[ri]:.{dec}f}\n±{stds[ri]:.{dec}f}"

    fig, ax = plt.subplots(figsize=(max(11, n_cols * 1.45), max(4.2, n_rows * 0.72 + 1.8)))
    im = ax.imshow(norm_mat, cmap="YlGn", aspect="auto", vmin=0.0, vmax=1.0)

    row_labels = [f"#{int(r['rank'])} {r['backbone']} | {r['variant']}" for _, r in df_stats.iterrows()]
    col_labels = [f"{lbl}\n({'↑' if hib else '↓'})" for _, lbl, hib in spec]

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8.5, rotation=20, ha="right")
    ax.grid(False)

    for ri in range(n_rows):
        for ci in range(n_cols):
            val = norm_mat[ri, ci]
            txt_color = "white" if val > 0.68 else "#111111"
            ax.text(ci, ri, text_mat[ri][ci], ha="center", va="center", color=txt_color, fontsize=8, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Column-Normalized Score (1.0 = Best in Column)", fontsize=8.5)
    ax.set_title("Normalized Metrics Scorecard Matrix (Raw Mean ± Std Inside Cells)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "chart_05_metrics_normalized_heatmap.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_06_delta_vs_baseline(df_delta: pd.DataFrame, out_dir: str) -> Optional[str]:
    """Diverging bar chart of % change vs neck_with_relu baseline."""
    if df_delta.empty:
        return None
    sub = df_delta[df_delta["variant"] != df_delta["baseline"]].copy()
    if sub.empty:
        return None

    _setup_mpl_style()
    metrics = [
        ("val_mae_pct_change", "Val MAE (%Δ, ↓ better)"),
        ("slot_act_inact_ratio_pct_change", "Slot Act/Inact (%Δ, ↑ better)"),
        ("slot_class_cos_sep_pct_change", "Slot Cls Cos Sep (%Δ, ↑ better)"),
        ("slot_global_45pair_sep_pct_change", "Slot 45-Pair Sep (%Δ, ↑ better)"),
        ("p3_small_ratio_pct_change", "P3 Small Ratio (%Δ, ↑ better)"),
        ("slot_full_fisher_pct_change", "Slot Full Fisher (%Δ, ↑ better)"),
    ]
    metrics = [(c, t) for c, t in metrics if c in sub.columns]
    if not metrics:
        return None

    labels = [f"{r['backbone']} | {r['variant'].replace('neck_', '')}" for _, r in sub.iterrows()]
    y = np.arange(len(labels))

    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, max(4.0, len(labels) * 0.45 + 1.5) * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        vals = sub[col].values.astype(float)
        # Color: green for improvement, red for degradation
        is_mae = col.startswith("val_mae")
        bar_colors = []
        for v in vals:
            improved = (v < 0) if is_mae else (v > 0)
            bar_colors.append("#2ca02c" if improved else "#d62728")

        ax.barh(y, vals, color=bar_colors, alpha=0.85, edgecolor="#333", linewidth=0.6)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("% Change vs. neck_with_relu")

        lim = max(5.0, float(np.max(np.abs(vals))) * 1.25)
        ax.set_xlim(-lim, lim)
        for i, v in enumerate(vals):
            ha = "left" if v >= 0 else "right"
            pad = lim * 0.03 if v >= 0 else -lim * 0.03
            ax.text(v + pad, i, f"{v:+.1f}%", va="center", ha=ha, fontsize=8, fontweight="bold")

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Relative % Change vs. Baseline (neck_with_relu)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_06_delta_vs_with_relu.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_07_run_consistency_lines(raw_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    """Run-by-run line plots of Val MAE and Slot Class Cos Sep for each backbone."""
    if raw_df.empty:
        return None
    _setup_mpl_style()
    backbones = list(raw_df["backbone"].unique())
    nrows = len(backbones)
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 4.2 * nrows), squeeze=False)

    for bi, bb in enumerate(backbones):
        sub_bb = raw_df[raw_df["backbone"] == bb]
        runs_sorted = sorted(sub_bb["run"].unique(), key=lambda s: int(re.search(r"(\d+)$", s).group(1)) if re.search(r"(\d+)$", s) else 0)
        x = np.arange(len(runs_sorted))

        for pi, (col, ylabel) in enumerate([("val_mae", "Val MAE (↓)"), ("slot_class_cos_sep", "Slot Class Cos Sep (↑)")]):
            ax = axes[bi, pi]
            if col not in sub_bb.columns:
                ax.axis("off")
                continue
            for var in [v for v in VARIANT_ORDER if v in sub_bb["variant"].unique()]:
                sub_v = sub_bb[sub_bb["variant"] == var].set_index("run")
                ys = [float(sub_v.loc[r, col]) if r in sub_v.index else np.nan for r in runs_sorted]
                ax.plot(x, ys, marker="o", linewidth=2.0, markersize=6, label=var, color=VARIANT_COLORS.get(var, "#555"))
            ax.set_xticks(x)
            ax.set_xticklabels(runs_sorted)
            ax.set_ylabel(ylabel)
            ax.set_title(f"[{bb.upper()}] {ylabel} Across Runs")
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("Run-to-Run Trajectory & Consistency", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "chart_07_run_consistency_lines.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_08_pairwise_heatmaps(backbone_name: str, df_pair: pd.DataFrame, out_dir: str) -> Optional[str]:
    """10x10 symmetric cosine separation heatmaps (Mean across runs) for each Neck variant."""
    if df_pair.empty:
        return None
    mean_cols = [c for c in df_pair.columns if c.endswith("_mean") and c.startswith("neck_")]
    if not mean_cols:
        return None

    _setup_mpl_style()
    classes = list(dict.fromkeys(list(df_pair["class_i"].unique()) + list(df_pair["class_j"].unique())))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)

    n_vars = len(mean_cols)
    ncols = min(2, n_vars)
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 6.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    vmax = max(0.05, float(df_pair[mean_cols].max().max()))

    for idx, mcol in enumerate(mean_cols):
        ax = axes[idx]
        var_name = mcol[:-5]
        mat = np.zeros((n_cls, n_cls), dtype=float)
        for _, r in df_pair.iterrows():
            i, j = cls_to_idx[r["class_i"]], cls_to_idx[r["class_j"]]
            val = float(r[mcol])
            mat[i, j] = val
            mat[j, i] = val

        im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_xticks(np.arange(n_cls))
        ax.set_yticks(np.arange(n_cls))
        ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.grid(False)
        ax.set_title(f"{var_name} (Mean over {int(df_pair['n_runs'].iloc[0])} runs)")

        for i in range(n_cls):
            for j in range(n_cls):
                if i == j:
                    continue
                val = mat[i, j]
                txt_c = "white" if val < vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_c, fontsize=6.5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    for j in range(n_vars, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"[{backbone_name.upper()}] Pairwise Slot Cosine Separation Matrices (Mean)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, f"chart_08_pairwise_slots_heatmap_{backbone_name}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_09_pairwise_top_diff(backbone_name: str, df_pair: pd.DataFrame, out_dir: str, top_k: int = 15) -> Optional[str]:
    """Top-K pairwise deltas (neck_no_last_relu vs neck_with_relu) + Top-K highest separated pairs."""
    if df_pair.empty or "diff_no_minus_with" not in df_pair.columns:
        return None
    _setup_mpl_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    # Panel 1: Top-K largest absolute diff (no_last_relu - with_relu)
    df_diff = df_pair.sort_values(by="diff_no_minus_with", ascending=False).head(top_k)
    y1 = np.arange(len(df_diff))
    diffs = df_diff["diff_no_minus_with"].values
    colors1 = ["#2ca02c" if v >= 0 else "#d62728" for v in diffs]
    ax1.barh(y1, diffs, color=colors1, alpha=0.88, edgecolor="#333", linewidth=0.6)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(df_diff["pair"].values, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.axvline(0, color="black", linewidth=0.9)
    ax1.set_xlabel("Δ Cosine Separation (no_last_relu - with_relu)")
    ax1.set_title(f"Top {top_k} Positive Δ (no_last_relu vs with_relu)")
    for i, (d, p) in enumerate(zip(diffs, df_diff["pct_no_vs_with"].values)):
        ax1.text(d + max(diffs) * 0.015, i, f"{d:+.4f} ({p:+.0f}%)", va="center", fontsize=8)
    ax1.set_xlim(min(0, float(np.min(diffs)) * 1.15), float(np.max(diffs)) * 1.32)

    # Panel 2: Top-K highest mean separation in neck_no_last_relu (with error bars comparing with_relu)
    df_top = df_pair.sort_values(by="neck_no_last_relu_mean", ascending=False).head(top_k)
    y2 = np.arange(len(df_top))
    h = 0.38
    ax2.barh(y2 - h / 2, df_top["neck_with_relu_mean"].values, height=h, xerr=df_top["neck_with_relu_std"].values,
             capsize=2, label="neck_with_relu", color=VARIANT_COLORS["neck_with_relu"], alpha=0.88, edgecolor="#333", linewidth=0.5)
    ax2.barh(y2 + h / 2, df_top["neck_no_last_relu_mean"].values, height=h, xerr=df_top["neck_no_last_relu_std"].values,
             capsize=2, label="neck_no_last_relu", color=VARIANT_COLORS["neck_no_last_relu"], alpha=0.88, edgecolor="#333", linewidth=0.5)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(df_top["pair"].values, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Cosine Separation (± Std)")
    ax2.set_title(f"Top {top_k} Highest-Separation Class Pairs")
    ax2.legend(fontsize=8.5, loc="lower right")

    fig.suptitle(f"[{backbone_name.upper()}] Pairwise Slot Cosine Separation Highlights", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, f"chart_09_pairwise_top_diff_{backbone_name}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# HTML DASHBOARD GENERATOR (OBJECTIVE TABLES + CHARTS, NO COMMENTARY)
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
    backbone_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    pairwise_by_bb: Dict[str, pd.DataFrame],
    chart_files: List[str],
    csv_files: List[str],
) -> str:
    """Tạo trang HTML trực quan tổng hợp toàn bộ bảng và biểu đồ (thuần dữ liệu)."""
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

    bb_section = ""
    if not backbone_df.empty and len(backbone_df) > 1:
        bb_section = f"<section><h2>2. Backbone-Level Aggregate Table</h2>{_df_to_html_table(backbone_df)}</section>"

    delta_section = ""
    if not delta_df.empty:
        delta_section = f"<section><h2>3. Delta & % Change vs. Baseline (neck_with_relu)</h2>{_df_to_html_table(delta_df)}</section>"

    pair_sections = []
    for bb, pdf in pairwise_by_bb.items():
        if not pdf.empty:
            # Show concise display table
            disp_cols = [c for c in pdf.columns if c in ("pair", "n_runs", "diff_no_minus_with", "pct_no_vs_with") or c.endswith("_mean") or c.endswith("_std")]
            pair_sections.append(
                f"<details><summary><strong>Pairwise 45-Class-Pair Table — {html.escape(bb.upper())}</strong> (click to expand)</summary>"
                f"{_df_to_html_table(pdf[disp_cols])}</details>"
            )
    pair_html = f"<section><h2>4. Pairwise Slot Cosine Separation Tables</h2>{''.join(pair_sections)}</section>" if pair_sections else ""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(summary_title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f4f6f9; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #555; margin-bottom: 20px; font-size: 0.95em; }}
  section {{ background: #fff; border-radius: 8px; padding: 18px 20px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  h2 {{ margin-top: 0; border-bottom: 2px solid #eef1f5; padding-bottom: 6px; font-size: 1.2em; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #e1e5eb; padding: 6px 10px; text-align: right; white-space: nowrap; }}
  th {{ background: #f0f3f8; font-weight: 600; text-align: center; position: sticky; top: 0; }}
  td:nth-child(1), td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
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
  <div class="meta">Ranking Rule: Priority 1 = Mean Val MAE (ascending), Priority 2 = Std Val MAE (ascending).</div>

  <section>
    <h2>0. Generated CSV Files</h2>
    <ul class="files">{csv_links}</ul>
  </section>

  <section>
    <h2>1. Overall Ranking Table (Mean ± Std)</h2>
    {_df_to_html_table(ranking_df)}
  </section>

  {bb_section}
  {delta_section}
  {pair_html}

  <section>
    <h2>5. Visual Charts</h2>
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
# CONSOLE TABLE & MAIN
# =============================================================================

def print_console_ranking(df_stats: pd.DataFrame):
    print("\n" + "=" * 145)
    print("  RANKING TABLE (Priority 1: Mean Val MAE asc | Priority 2: Std Val MAE asc)")
    print("=" * 145)
    print(
        f"{'Rank':<5} | {'Backbone':<18} | {'Variant':<22} | {'Runs':<4} | "
        f"{'Val MAE (Mean±Std)':<20} | {'CV%':<6} | {'Act/Inact':<13} | {'Slot ClsSep':<13} | {'P3 Small':<13}"
    )
    print("-" * 145)
    for _, r in df_stats.iterrows():
        mae_s = f"{r['val_mae_mean']:.4f} ± {r['val_mae_std']:.4f}"
        cv_s = f"{r['val_mae_cv_pct']:.2f}%"
        act_s = f"{r.get('slot_act_inact_ratio_mean', 0):.2f}±{r.get('slot_act_inact_ratio_std', 0):.2f}x"
        cls_s = f"{r.get('slot_class_cos_sep_mean', 0):.4f}±{r.get('slot_class_cos_sep_std', 0):.4f}"
        p3_s = f"{r.get('p3_small_ratio_mean', 0):.2f}±{r.get('p3_small_ratio_std', 0):.2f}x"
        print(
            f"{int(r['rank']):<5} | {r['backbone']:<18} | {r['variant']:<22} | {int(r['n_runs']):<4} | "
            f"{mae_s:<20} | {cv_s:<6} | {act_s:<13} | {cls_s:<13} | {p3_s:<13}"
        )
    print("=" * 145)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ablation runs into summary_<backboneA>_<backboneB>_... with CSVs, charts, and HTML dashboard."
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="*",
        default=[],
        help="Danh sách đường dẫn folder kết quả (VD: Visdrone/output_ablation_neck_relu_senet Visdrone/output_ablation_neck_relu_resnet)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Tự động quét tất cả các folder output_ablation_neck_relu* trong --base-dir",
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
        help="Ghi đè đường dẫn thư mục xuất (mặc định: <base-dir>/summary_<backboneA>_<backboneB>_...)",
    )
    args = parser.parse_args()

    target_dirs = list(args.dirs)
    if args.auto or not target_dirs:
        matched = sorted(glob.glob(os.path.join(args.base_dir, "output_ablation_neck_relu*")))
        target_dirs = [d for d in matched if os.path.isdir(d)]
        if not target_dirs:
            print(f"[!] Không tìm thấy thư mục dạng 'output_ablation_neck_relu*' trong '{args.base_dir}'.")
            return

    raw_frames: List[pd.DataFrame] = []
    pairwise_by_bb: Dict[str, pd.DataFrame] = {}
    backbones_loaded: List[str] = []

    for d in target_dirs:
        folder_name, backbone_name, metric_runs, pair_runs = load_folder_data(d)
        if not metric_runs:
            print(f"[-] Bỏ qua '{d}': không tìm thấy ablation_neck_relu_results.csv.")
            continue
        print(f"[+] Loaded '{backbone_name}': {len(metric_runs)} metric runs, {len(pair_runs)} pairwise runs ({d})")
        backbones_loaded.append(backbone_name)
        raw_frames.append(build_raw_runs_df(backbone_name, folder_name, metric_runs))
        if pair_runs:
            pairwise_by_bb[backbone_name] = compute_pairwise_stats(backbone_name, pair_runs)

    if not raw_frames:
        print("[!] Không có dữ liệu hợp lệ để tổng hợp.")
        return

    # Tạo tên thư mục summary_<backboneA>_<backboneB>_...
    summary_folder_name = "summary_" + "_".join(backbones_loaded)
    out_dir = args.out_dir if args.out_dir is not None else os.path.join(args.base_dir, summary_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    raw_df = pd.concat(raw_frames, ignore_index=True)
    df_stats = compute_detailed_stats(raw_df)
    ranking_df = build_ranking_display_df(df_stats)
    delta_df = compute_delta_vs_baseline(df_stats)
    backbone_df = compute_backbone_agg_df(df_stats)

    print_console_ranking(df_stats)

    # 1. Save CSVs
    csv_files: List[str] = []

    p1 = os.path.join(out_dir, "01_ranking_summary.csv")
    ranking_df.to_csv(p1, index=False, encoding="utf-8-sig")
    csv_files.append(p1)

    p2 = os.path.join(out_dir, "02_all_metrics_mean_std.csv")
    df_stats.to_csv(p2, index=False, encoding="utf-8-sig")
    csv_files.append(p2)

    p3 = os.path.join(out_dir, "03_all_runs_raw.csv")
    raw_df.to_csv(p3, index=False, encoding="utf-8-sig")
    csv_files.append(p3)

    if not delta_df.empty:
        p4 = os.path.join(out_dir, "04_delta_vs_baseline.csv")
        delta_df.to_csv(p4, index=False, encoding="utf-8-sig")
        csv_files.append(p4)

    if pairwise_by_bb:
        all_pair_df = pd.concat(list(pairwise_by_bb.values()), ignore_index=True)
        p5 = os.path.join(out_dir, "05_pairwise_slots_mean_std.csv")
        all_pair_df.to_csv(p5, index=False, encoding="utf-8-sig")
        csv_files.append(p5)

    if len(backbones_loaded) > 1 and not backbone_df.empty:
        p6 = os.path.join(out_dir, "06_backbone_agg_summary.csv")
        backbone_df.to_csv(p6, index=False, encoding="utf-8-sig")
        csv_files.append(p6)

    # 2. Generate Charts
    chart_files: List[str] = []
    chart_files.append(plot_01_mae_ranking_and_distribution(df_stats, raw_df, out_dir))
    chart_files.append(plot_02_mae_vs_stability_scatter(df_stats, out_dir))
    chart_files.append(plot_03_key_metrics_grouped_bars(df_stats, out_dir))
    chart_files.append(plot_04_multiscale_profile(df_stats, out_dir))
    chart_files.append(plot_05_normalized_heatmap(df_stats, out_dir))

    c6 = plot_06_delta_vs_baseline(delta_df, out_dir)
    if c6:
        chart_files.append(c6)

    c7 = plot_07_run_consistency_lines(raw_df, out_dir)
    if c7:
        chart_files.append(c7)

    for bb, pdf in pairwise_by_bb.items():
        c8 = plot_08_pairwise_heatmaps(bb, pdf, out_dir)
        if c8:
            chart_files.append(c8)
        c9 = plot_09_pairwise_top_diff(bb, pdf, out_dir)
        if c9:
            chart_files.append(c9)

    # 3. Generate HTML Dashboard
    dash_path = generate_html_dashboard(
        summary_title=f"Ablation Summary: {', '.join(b.upper() for b in backbones_loaded)}",
        out_dir=out_dir,
        ranking_df=ranking_df,
        backbone_df=backbone_df,
        delta_df=delta_df,
        pairwise_by_bb=pairwise_by_bb,
        chart_files=chart_files,
        csv_files=csv_files,
    )

    print(f"\n[+] Output Directory: {os.path.abspath(out_dir)}")
    print(f"    - CSV tables : {len(csv_files)} files")
    print(f"    - PNG charts : {len(chart_files)} files")
    print(f"    - Dashboard  : {os.path.abspath(dash_path)}")


if __name__ == "__main__":
    main()
