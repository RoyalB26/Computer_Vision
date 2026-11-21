import os
import pandas as pd
import numpy as np
from scipy import stats
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    has_sklearn = True
except ImportError:
    has_sklearn = False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(base_dir, "pairwise_cosine_sep.xlsx")
    csv_path = os.path.join(base_dir, "benchmark_results.csv")
    
    print(f"Reading Excel: {excel_path}")
    print(f"Reading CSV:   {csv_path}\n")
    
    df_excel = pd.read_excel(excel_path)
    df_csv = pd.read_csv(csv_path)
    
    # Identify block columns that appear in both files
    meta_cols = {'class_i', 'class_j', 'pair', 'mean_blocks'}
    block_cols = [c for c in df_excel.columns if c not in meta_cols]
    
    # Filter CSV to only matching blocks
    df_csv_matching = df_csv[df_csv['block'].isin(block_cols)].copy()
    val_mae_dict = dict(zip(df_csv_matching['block'], df_csv_matching['val_mae']))
    
    # Common blocks in order
    common_blocks = [b for b in block_cols if b in val_mae_dict]
    val_mae_series = pd.Series([val_mae_dict[b] for b in common_blocks], index=common_blocks)
    
    print(f"Loaded {len(df_excel)} pairwise separation features across {len(common_blocks)} blocks.")
    print("=" * 80)
    print("1. TƯƠNG QUAN GIỮA TỪNG CẶP LỚP (PAIRWISE SEPARATION) VÀ VAL_MAE")
    print("=" * 80)
    
    pair_results = []
    
    for _, row in df_excel.iterrows():
        pair_name = row['pair']
        class_i = row.get('class_i', '')
        class_j = row.get('class_j', '')
        sep_values = pd.Series([row[b] for b in common_blocks], index=common_blocks, dtype=float)
        
        # Pearson correlation (tuyến tính)
        r_pearson, p_pearson = stats.pearsonr(sep_values, val_mae_series)
        # Spearman correlation (đơn điệu/xếp hạng)
        r_spearman, p_spearman = stats.spearmanr(sep_values, val_mae_series)
        
        pair_results.append({
            'pair': pair_name,
            'class_i': class_i,
            'class_j': class_j,
            'mean_sep': sep_values.mean(),
            'std_sep': sep_values.std(),
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_rho': r_spearman,
            'spearman_p': p_spearman,
            'abs_pearson': abs(r_pearson)
        })
        
    df_pair_eval = pd.DataFrame(pair_results)
    df_pair_eval.sort_values(by='abs_pearson', ascending=False, inplace=True)
    
    print("\n--- TOP 10 CẶP LỚP CÓ TƯƠNG QUAN MẠNH NHẤT TỚI VAL_MAE ---")
    top_10 = df_pair_eval.head(10)[['pair', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p', 'mean_sep']]
    print(top_10.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("2. TỔNG HỢP THEO TỪNG CLASS RIÊNG LẺ (CLASS-LEVEL IMPACT)")
    print("=" * 80)
    # Gom tất cả các cặp có chứa class_k để xem class nào ảnh hưởng nhiều nhất đến val_mae
    classes = set(df_excel['class_i']).union(set(df_excel['class_j']))
    class_summary = []
    
    for c in sorted(classes):
        c_pairs = df_pair_eval[(df_pair_eval['class_i'] == c) | (df_pair_eval['class_j'] == c)]
        avg_abs_corr = c_pairs['abs_pearson'].mean()
        avg_corr = c_pairs['pearson_r'].mean()
        class_summary.append({
            'class': c,
            'num_pairs': len(c_pairs),
            'avg_abs_pearson': avg_abs_corr,
            'avg_pearson': avg_corr
        })
        
    df_class_eval = pd.DataFrame(class_summary).sort_values(by='avg_abs_pearson', ascending=False)
    print(df_class_eval.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("3. PHÂN TÍCH HỒI QUY ĐÁNH GIÁ TẦM QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCE)")
    print("=" * 80)
    
    # Chuẩn bị ma trận X (samples: blocks, features: pairs), y: val_mae
    X = pd.DataFrame(index=common_blocks)
    for _, row in df_excel.iterrows():
        X[row['pair']] = [row[b] for b in common_blocks]
        
    y = val_mae_series.values
    
    if has_sklearn:
        # Ridge Regression để xử lý hiện tượng p > n (45 features > 12 samples)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        ridge_coefs = pd.Series(ridge.coef_, index=X.columns)
        
        # Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
        
        importance_df = pd.DataFrame({
            'pair': X.columns,
            'ridge_coef': ridge_coefs.values,
            'ridge_abs_coef': ridge_coefs.abs().values,
            'rf_importance': rf_importances.values
        }).sort_values(by='ridge_abs_coef', ascending=False)
        
        print("\n--- TOP 10 CẶP CÓ HỆ SỐ RIDGE REGRESSION LỚN NHẤT ---")
        print(importance_df.head(10)[['pair', 'ridge_coef', 'ridge_abs_coef', 'rf_importance']].to_string(index=False))
    else:
        print("scikit-learn chưa được cài đặt, bỏ qua phần mô hình hồi quy Ridge / RF.")

    # Xuất báo cáo ra file Excel và CSV
    out_eval_csv = os.path.join(base_dir, "cls_sep_val_mae_impact.csv")
    df_pair_eval.to_csv(out_eval_csv, index=False)
    
    out_class_csv = os.path.join(base_dir, "cls_level_impact.csv")
    df_class_eval.to_csv(out_class_csv, index=False)
    
    print(f"\n=> Kết quả phân tích chi tiết đã được lưu tại:")
    print(f"   - {out_eval_csv}")
    print(f"   - {out_class_csv}")

if __name__ == "__main__":
    main()
