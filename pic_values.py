import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# ============================================================================
# 数据准备
# ============================================================================

df = pd.read_csv(r'D:\PythonProgramme\c_paper\【新数据库】CSV 16+16最新 20260103.csv', encoding='gbk')


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


for col in ['Time', 'Temperature', 'Molar Ratio', 'Volume of H2SO4', 'QY', 'FWHM']:
    if col in df.columns:
        df[col] = clean_numeric(df[col])

df = df.rename(columns={'Volume of H2SO4': 'H2SO4 Volume'})

# ============================================================================
# 计算迭代过程中的效用值
# ============================================================================

feature_cols = ['Time', 'Temperature', 'Molar Ratio', 'H2SO4 Volume']

# 初始Origin数据（1-16）
origin_data = df[df['Serial Number'] <= 16].copy()
iterated_data = df[df['Serial Number'] > 16].copy()

X_origin = origin_data[feature_cols].values
y_QY_origin = origin_data['QY'].values
y_FWHM_origin = origin_data['FWHM'].values

X_iterated = iterated_data[feature_cols].values
y_QY_iterated = iterated_data['QY'].values
y_FWHM_iterated = iterated_data['FWHM'].values

# 标准化
scaler_X = StandardScaler()
scaler_QY = StandardScaler()
scaler_FWHM = StandardScaler()

X_origin_scaled = scaler_X.fit_transform(X_origin)
X_iterated_scaled = scaler_X.transform(X_iterated)

y_QY_origin_scaled = scaler_QY.fit_transform(y_QY_origin.reshape(-1, 1)).ravel()
y_FWHM_origin_scaled = scaler_FWHM.fit_transform(y_FWHM_origin.reshape(-1, 1)).ravel()

# 模拟迭代过程，每次迭代4个样本
n_iterations = 4  # 迭代4次，每次4个样本
samples_per_iteration = 4

utility_qy_list = []
utility_fwhm_list = []
iteration_labels = []

# 初始训练数据
X_train = X_origin_scaled.copy()
y_QY_train = y_QY_origin_scaled.copy()
y_FWHM_train = y_FWHM_origin_scaled.copy()

kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

print("=" * 80)
print("计算每次迭代的效用值")
print("=" * 80)
print("\n效用值计算方法：")
print("- QY (最大化):  Utility = (预测均值 - 当前最优值) + κ * 预测不确定性")
print("- FWHM (最小化): Utility = (当前最优值 - 预测均值) + κ * 预测不确定性")
print("  其中 κ=2 (探索-利用平衡参数)")
print("=" * 80)

for i in range(n_iterations):
    # 训练模型
    gp_QY = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp_FWHM = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

    gp_QY.fit(X_train, y_QY_train)
    gp_FWHM.fit(X_train, y_FWHM_train)

    # 获取当前迭代的新样本
    start_idx = i * samples_per_iteration
    end_idx = (i + 1) * samples_per_iteration
    X_new = X_iterated_scaled[start_idx:end_idx]

    # 预测均值和标准差
    y_QY_pred_scaled, y_QY_std = gp_QY.predict(X_new, return_std=True)
    y_FWHM_pred_scaled, y_FWHM_std = gp_FWHM.predict(X_new, return_std=True)

    # 当前最优值
    best_QY = np.max(y_QY_train)  # QY越大越好
    best_FWHM = np.min(y_FWHM_train)  # FWHM越小越好

    # 计算效用值 (Upper Confidence Bound, UCB)
    # 参数κ控制探索vs利用的平衡，通常取1-3
    kappa = 2.0

    # QY: 最大化，所以效用 = (预测值 - 最优值) + κ * 不确定性
    # 取平均效用值作为这次迭代的效用
    utility_QY = np.mean((y_QY_pred_scaled - best_QY) + kappa * y_QY_std)

    # FWHM: 最小化，所以效用 = (最优值 - 预测值) + κ * 不确定性
    utility_FWHM = np.mean((best_FWHM - y_FWHM_pred_scaled) + kappa * y_FWHM_std)

    utility_qy_list.append(utility_QY)
    utility_fwhm_list.append(utility_FWHM)
    iteration_labels.append(str(i))

    print(f"\nIteration {i}:")
    print(f"  Utility(QY)  = {utility_QY:.4f}")
    print(f"  Utility(FWHM) = {utility_FWHM:.4f}")

    # 将新样本加入训练集
    y_QY_new_true = scaler_QY.transform(y_QY_iterated[start_idx:end_idx].reshape(-1, 1)).ravel()
    y_FWHM_new_true = scaler_FWHM.transform(y_FWHM_iterated[start_idx:end_idx].reshape(-1, 1)).ravel()

    X_train = np.vstack([X_train, X_new])
    y_QY_train = np.append(y_QY_train, y_QY_new_true)
    y_FWHM_train = np.append(y_FWHM_train, y_FWHM_new_true)

# ============================================================================
# 保存数据到Excel
# ============================================================================

utility_df = pd.DataFrame({
    'Iteration': range(n_iterations),
    'Utility_QY': utility_qy_list,
    'Utility_FWHM': utility_fwhm_list
})

with pd.ExcelWriter('pic_output2/utility_values_data.xlsx', engine='openpyxl') as writer:
    utility_df.to_excel(writer, sheet_name='Utility_Values', index=False)

    # 添加说明sheet
    explanation = pd.DataFrame({
        'Item': ['计算方法', 'QY效用值', 'FWHM效用值', '参数κ', '迭代方式'],
        'Description': [
            'Upper Confidence Bound (UCB) 采集函数',
            'Utility = (预测均值 - 当前最优) + κ * 标准差',
            'Utility = (当前最优 - 预测均值) + κ * 标准差',
            '2.0 (探索-利用平衡参数)',
            '每次迭代4个新样本，共4次迭代'
        ]
    })
    explanation.to_excel(writer, sheet_name='Explanation', index=False)

print("\n✓ 数据已保存到: pic_output1/utility_values_data.xlsx")

# ============================================================================
# 绘图1: 两个目标的效用值在同一张图（1C）
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(10, 6))

iterations_x = range(n_iterations)

# QY效用值
ax1.plot(iterations_x, utility_qy_list, 'o-', color='#2166AC',
         linewidth=2.5, markersize=10, label='Utility (QY)',
         markeredgewidth=1.5, markeredgecolor='darkblue')

# FWHM效用值
ax1.plot(iterations_x, utility_fwhm_list, 's-', color='#B2182B',
         linewidth=2.5, markersize=10, label='Utility (FWHM)',
         markeredgewidth=1.5, markeredgecolor='darkred')

ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax1.set_ylabel('Utility Value', fontsize=14, fontweight='bold')
ax1.set_title('Acquisition Utility Values vs Iteration', fontsize=16, fontweight='bold', pad=15)
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(iterations_x)
ax1.set_xticklabels(iteration_labels)
ax1.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('pic_output2/utility_1C_combined.png', dpi=300, bbox_inches='tight')
print("✓ 图1已保存: pic_output2/utility_1C_combined.png")
plt.close()

# ============================================================================
# 绘图2: QY的效用值（2A - 第1张）
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(iterations_x, utility_qy_list, 'o-', color='#2166AC',
         linewidth=2.5, markersize=12,
         markeredgewidth=1.5, markeredgecolor='darkblue')

ax2.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax2.set_ylabel('Utility Value for QY', fontsize=14, fontweight='bold')
ax2.set_title('QY: Acquisition Utility vs Iteration', fontsize=16, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(iterations_x)
ax2.set_xticklabels(iteration_labels)
ax2.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('pic_output2/utility_2A_QY.png', dpi=300, bbox_inches='tight')
print("✓ 图2已保存: pic_output2/utility_2A_QY.png")
plt.close()

# ============================================================================
# 绘图3: FWHM的效用值（2A - 第2张）
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.plot(iterations_x, utility_fwhm_list, 's-', color='#B2182B',
         linewidth=2.5, markersize=12,
         markeredgewidth=1.5, markeredgecolor='darkred')

ax3.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax3.set_ylabel('Utility Value for FWHM', fontsize=14, fontweight='bold')
ax3.set_title('FWHM: Acquisition Utility vs Iteration', fontsize=16, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(iterations_x)
ax3.set_xticklabels(iteration_labels)
ax3.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('pic_output2/utility_2A_FWHM.png', dpi=300, bbox_inches='tight')
print("✓ 图3已保存: pic_output2/utility_2A_FWHM.png")
plt.close()

# ============================================================================
# 打印数据预览
# ============================================================================

print("\n" + "=" * 80)
print("效用值数据表格")
print("=" * 80)
print(utility_df.to_string(index=False))

print("\n" + "=" * 80)
print("完成！生成文件：")
print("  1. pic_output1/utility_1C_combined.png     - 两个目标效用值（1C）")
print("  2. pic_output1/utility_2A_QY.png           - QY效用值（2A之一）")
print("  3. pic_output1/utility_2A_FWHM.png         - FWHM效用值（2A之二）")
print("  4. pic_output1/utility_values_data.xlsx    - 所有数据点")
print("=" * 80)
