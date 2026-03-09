import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# ============================================================================
# 数据准备 - 从CSV读取
# ============================================================================

# 读取CSV文件
df_all = pd.read_csv(r'D:\PythonProgramme\c_paper\【新数据库】CSV 16+16最新 20260103.csv', encoding='gbk')


# 数据预处理
def clean_numeric(series):
    if series.dtype == 'object':
        cleaned = series.str.replace('h', '', regex=False)
        cleaned = cleaned.str.replace('°C', '', regex=False)
        cleaned = cleaned.str.replace('℃', '', regex=False)
        cleaned = cleaned.str.replace('ml', '', regex=False)
        cleaned = cleaned.str.replace('%', '', regex=False)
        cleaned = cleaned.str.replace('nm', '', regex=False)
        cleaned = cleaned.str.strip()
        return pd.to_numeric(cleaned, errors='coerce')
    return series


# 清理数值列
for col in ['Time', 'Temperature', 'Molar Ratio', 'Volume of H2SO4', 'QY', 'FWHM']:
    if col in df_all.columns:
        df_all[col] = clean_numeric(df_all[col])

# 重命名列名
df_all = df_all.rename(columns={'Volume of H2SO4': 'H2SO4 Volume'})

print("=" * 80)
print("c) 完全拟合后的R²和MAE（使用全部32个样本）")
print("=" * 80)

# ============================================================================
# 训练完整模型
# ============================================================================

feature_cols = ['Time', 'Temperature', 'Molar Ratio', 'H2SO4 Volume']

X_all = df_all[feature_cols].values
y_all_qy = df_all['QY'].values
y_all_fwhm = df_all['FWHM'].values

# 训练完整模型
model_qy_full = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
model_fwhm_full = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)

model_qy_full.fit(X_all, y_all_qy)
model_fwhm_full.fit(X_all, y_all_fwhm)

# 预测
pred_qy_full = model_qy_full.predict(X_all)
pred_fwhm_full = model_fwhm_full.predict(X_all)

# 计算R²和MAE
r2_qy = r2_score(y_all_qy, pred_qy_full)
r2_fwhm = r2_score(y_all_fwhm, pred_fwhm_full)
mae_qy = mean_absolute_error(y_all_qy, pred_qy_full)
mae_fwhm = mean_absolute_error(y_all_fwhm, pred_fwhm_full)

print(f"\nQY模型:")
print(f"  R² = {r2_qy:.4f}")
print(f"  MAE = {mae_qy:.4f}")

print(f"\nFWHM模型:")
print(f"  R² = {r2_fwhm:.4f}")
print(f"  MAE = {mae_fwhm:.2f}")

# ============================================================================
# 保存数据到Excel - 用于手绘
# ============================================================================

# 创建数据表格
results_df = pd.DataFrame({
    'Serial_Number': df_all['Serial Number'],
    'Actual_QY': y_all_qy,
    'Predicted_QY': pred_qy_full,
    'Actual_FWHM': y_all_fwhm,
    'Predicted_FWHM': pred_fwhm_full
})

# 保存到Excel
with pd.ExcelWriter('pic_output1/prediction_data_for_plotting.xlsx', engine='openpyxl') as writer:
    # Sheet 1: QY数据
    qy_data = pd.DataFrame({
        'Serial_Number': df_all['Serial Number'],
        'Actual_QY': y_all_qy,
        'Predicted_QY': pred_qy_full,
        'Error': y_all_qy - pred_qy_full
    })
    qy_data.to_excel(writer, sheet_name='QY_Data', index=False)

    # Sheet 2: FWHM数据
    fwhm_data = pd.DataFrame({
        'Serial_Number': df_all['Serial Number'],
        'Actual_FWHM': y_all_fwhm,
        'Predicted_FWHM': pred_fwhm_full,
        'Error': y_all_fwhm - pred_fwhm_full
    })
    fwhm_data.to_excel(writer, sheet_name='FWHM_Data', index=False)

    # Sheet 3: 指标汇总
    metrics_summary = pd.DataFrame({
        'Target': ['QY', 'FWHM'],
        'R²': [r2_qy, r2_fwhm],
        'MAE': [mae_qy, mae_fwhm]
    })
    metrics_summary.to_excel(writer, sheet_name='Metrics_Summary', index=False)

    # Sheet 4: 完整数据（包含特征）
    full_data = df_all[['Serial Number'] + feature_cols + ['QY', 'FWHM']].copy()
    full_data['Predicted_QY'] = pred_qy_full
    full_data['Predicted_FWHM'] = pred_fwhm_full
    full_data.to_excel(writer, sheet_name='Full_Data', index=False)

print("\n✓ 数据已保存到: pic_output1/prediction_data_for_plotting.xlsx")

# ============================================================================
# 绘图1: QY 预测值 vs 实际值
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(8, 8))

# QY散点图
ax1.scatter(y_all_qy, pred_qy_full, s=100, alpha=0.7, color='#2166AC',
            edgecolors='black', linewidth=1)

# 完美预测线
min_val = min(y_all_qy.min(), pred_qy_full.min())
max_val = max(y_all_qy.max(), pred_qy_full.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
         label='Perfect Prediction')

# 设置标签和标题
ax1.set_xlabel('Actual QY (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Predicted QY (%)', fontsize=14, fontweight='bold')
ax1.set_title(f'QY: R²={r2_qy:.4f}, MAE={mae_qy:.4f}',
              fontsize=16, fontweight='bold', pad=15)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# 调整刻度字体大小
ax1.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('pic_output1/c1_QY_prediction.png', dpi=300, bbox_inches='tight')
print("✓ 图1已保存: pic_output1/c1_QY_prediction.png")
plt.close()

# ============================================================================
# 绘图2: FWHM 预测值 vs 实际值
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(8, 8))

# FWHM散点图
ax2.scatter(y_all_fwhm, pred_fwhm_full, s=100, alpha=0.7, color='#B2182B',
            edgecolors='black', linewidth=1)

# 完美预测线
min_val = min(y_all_fwhm.min(), pred_fwhm_full.min())
max_val = max(y_all_fwhm.max(), pred_fwhm_full.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
         label='Perfect Prediction')

# 设置标签和标题
ax2.set_xlabel('Actual FWHM (nm)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Predicted FWHM (nm)', fontsize=14, fontweight='bold')
ax2.set_title(f'FWHM: R²={r2_fwhm:.4f}, MAE={mae_fwhm:.2f}',
              fontsize=16, fontweight='bold', pad=15)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# 调整刻度字体大小
ax2.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('pic_output1/c2_FWHM_prediction.png', dpi=300, bbox_inches='tight')
print("✓ 图2已保存: pic_output1/c2_FWHM_prediction.png")
plt.close()

# ============================================================================
# 打印数据表格预览
# ============================================================================

print("\n" + "=" * 80)
print("数据表格预览（用于手绘）")
print("=" * 80)

print("\n【QY数据】前10行:")
print(qy_data.head(10).to_string(index=False))

print("\n【FWHM数据】前10行:")
print(fwhm_data.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("完成！生成文件：")
print("  1. pic_output1/c1_QY_prediction.png")
print("  2. pic_output1/c2_FWHM_prediction.png")
print("  3. pic_output1/prediction_data_for_plotting.xlsx")
print("=" * 80)