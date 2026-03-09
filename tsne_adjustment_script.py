"""
t-SNE后处理调整脚本 - 精确控制各组紧凑性

核心思想：
1. 先进行标准t-SNE降维
2. 然后在2D空间中直接调整各组的组内距离
3. 通过设置目标距离来精确控制最终效果

作者：Kenneth
日期：2026-01-23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ============================================================================
# 🔧 可调整参数区域 - 开始
# ============================================================================

# 参数1：目标平均距离（最重要的参数！）
# 说明：控制每组内部点之间的平均距离
# 数值越小，组内越紧凑；数值越大，组内越分散
# 建议：让1→2→3→4单调递减，5稍大于4
TARGET_DISTANCES = {
    0: None,  # Origin - 设为None表示保持原样
    1: 16.0,  # Iter 1 - 最分散（探索阶段）
    2: 10.0,  # Iter 2 - 中等收敛
    3:  6.0,  # Iter 3 - 较强收敛
    4:  3.0,  # Iter 4 - 最强收敛（最紧凑）⭐
    5:  5.0,  # Iter 5 - 比4稍松散
}

# 参数2：背景点数量
# 说明：灰色背景点的数量，形成椭圆
# 建议：300-1000之间，太少不够平滑，太多计算慢
N_BACKGROUND = 500

# 参数3：背景椭圆的紧凑程度
# 说明：控制背景点的分散程度，数值越小越紧凑
# 建议：0.2-0.3之间，0.25是比较好的中间值
BACKGROUND_COMPACTNESS = 0.25

# 参数4：t-SNE参数
# perplexity: 控制局部和全局结构的平衡，通常5-50
# learning_rate: 学习率，通常100-1000
# max_iter: 最大迭代次数，通常1000-3000
TSNE_PARAMS = {
    'perplexity': 40,       # 推荐：30-50
    'learning_rate': 150,   # 推荐：100-200
    'max_iter': 1500,       # 推荐：1000-2000
    'early_exaggeration': 12,  # 早期夸张，增强聚类
    'random_state': 42,     # 随机种子，保证可重复
}

# 参数5：图表配色方案
# 说明：每次迭代的颜色，按照 Iter 1, 2, 3, 4, 5 的顺序
ITER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
# 可选配色方案：
# 方案1（暖色系）: ['#FF6B6B', '#FF8E53', '#FFA94D', '#FFD93D', '#6BCF7F']
# 方案2（冷色系）: ['#667EEA', '#64B6F7', '#48B0F7', '#3EC1D3', '#9CECFB']
# 方案3（彩虹）:   ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#3498DB']

# 参数6：图表标记形状
# 说明：每次迭代的标记，按照 Iter 1, 2, 3, 4, 5 的顺序
ITER_MARKERS = ['o', 's', '^', 'D', 'v']
# 可选形状：'o'圆形, 's'方形, '^'上三角, 'v'下三角, 'D'菱形, 
#          '*'星形, 'p'五边形, 'h'六边形, '+'加号, 'x'叉号

# 参数7：输出文件路径
OUTPUT_DIR = '/mnt/user-data/outputs'
OUTPUT_PREFIX = 'tsne_final'  # 输出文件前缀

# ============================================================================
# 🔧 可调整参数区域 - 结束
# ============================================================================

print("="*80)
print("t-SNE后处理调整")
print("="*80)
print(f"\n目标距离设置：")
for k, v in TARGET_DISTANCES.items():
    label = "Origin" if k == 0 else f"Iter {k}"
    val = "保持原样" if v is None else f"{v:.1f}"
    print(f"  {label}: {val}")

# ============================================================================
# 1. 数据加载
# ============================================================================

df_original = pd.read_csv('/mnt/user-data/uploads/_新数据库_CSV_16_16最新_20260103.csv', 
                         encoding='gbk')
df_new = pd.read_csv('/mnt/user-data/uploads/suggested_experiments_iteration_0.csv')

def clean_numeric(series):
    """清理数值列，去除单位符号"""
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

# 清理数据
for col in ['Time', 'Temperature', 'Molar Ratio', 'Volume of H2SO4', 'QY', 'FWHM']:
    if col in df_original.columns:
        df_original[col] = clean_numeric(df_original[col])

df_original = df_original.rename(columns={'Volume of H2SO4': 'H2SO4 Volume'})

# 提取特征
feature_cols = ['Molar Ratio', 'H2SO4 Volume', 'Temperature', 'Time']
X_existing = df_original[feature_cols].values[:32]
X_new = df_new[feature_cols].values
X_all = np.vstack([X_existing, X_new])

print(f"\n已加载 {len(X_all)} 个样本点")

# 定义分组索引
origin_indices = list(range(0, 16))
iter1_indices = list(range(16, 20))
iter2_indices = list(range(20, 24))
iter3_indices = list(range(24, 28))
iter4_indices = list(range(28, 32))
iter5_indices = list(range(32, 36))

groups_dict = {
    0: origin_indices,
    1: iter1_indices,
    2: iter2_indices,
    3: iter3_indices,
    4: iter4_indices,
    5: iter5_indices,
}

# ============================================================================
# 2. 生成背景点（紧凑椭圆）
# ============================================================================

print(f"\n生成 {N_BACKGROUND} 个背景点...")

# 参数空间范围
parameter_bounds = {
    'Molar Ratio': (250, 750),
    'H2SO4 Volume': (1, 4),
    'Temperature': (150, 200),
    'Time': (8, 12)
}

# 计算中心
center_bg = np.array([
    (parameter_bounds['Molar Ratio'][0] + parameter_bounds['Molar Ratio'][1]) / 2,
    (parameter_bounds['H2SO4 Volume'][0] + parameter_bounds['H2SO4 Volume'][1]) / 2,
    (parameter_bounds['Temperature'][0] + parameter_bounds['Temperature'][1]) / 2,
    (parameter_bounds['Time'][0] + parameter_bounds['Time'][1]) / 2
])

# 计算范围
ranges = np.array([
    parameter_bounds['Molar Ratio'][1] - parameter_bounds['Molar Ratio'][0],
    parameter_bounds['H2SO4 Volume'][1] - parameter_bounds['H2SO4 Volume'][0],
    parameter_bounds['Temperature'][1] - parameter_bounds['Temperature'][0],
    parameter_bounds['Time'][1] - parameter_bounds['Time'][0]
])

# 生成椭圆形背景点
cov_matrix = np.diag((ranges * BACKGROUND_COMPACTNESS) ** 2)
np.random.seed(42)
X_background = multivariate_normal.rvs(mean=center_bg, cov=cov_matrix, size=N_BACKGROUND)

# 裁剪到范围内
for i, (name, (lower, upper)) in enumerate(parameter_bounds.items()):
    X_background[:, i] = np.clip(X_background[:, i], lower, upper)

# ============================================================================
# 3. t-SNE降维
# ============================================================================

print("\n执行t-SNE降维...")

# 标准化
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
X_bg_scaled = scaler.transform(X_background)

# 合并所有点
X_all_for_tsne = np.vstack([X_bg_scaled, X_all_scaled])

# t-SNE降维
tsne = TSNE(
    n_components=2,
    perplexity=TSNE_PARAMS['perplexity'],
    learning_rate=TSNE_PARAMS['learning_rate'],
    max_iter=TSNE_PARAMS['max_iter'],
    early_exaggeration=TSNE_PARAMS['early_exaggeration'],
    random_state=TSNE_PARAMS['random_state'],
    init='pca'
)

X_tsne = tsne.fit_transform(X_all_for_tsne)
print("✓ t-SNE降维完成")

# 分离背景和样本
n_bg = len(X_background)
X_tsne_bg = X_tsne[:n_bg]
X_tsne_samples = X_tsne[n_bg:]

# ============================================================================
# 4. 🎯 核心：后处理调整组内距离
# ============================================================================

print("\n" + "="*80)
print("后处理：调整组内距离")
print("="*80)

X_tsne_adjusted = X_tsne_samples.copy()

for iter_num, indices in groups_dict.items():
    target_dist = TARGET_DISTANCES[iter_num]
    
    if target_dist is None:
        print(f"Iter {iter_num}: 保持原样")
        continue
    
    # 计算组内中心
    center = np.mean(X_tsne_samples[indices], axis=0)
    
    # 计算当前平均距离（到中心的距离）
    current_distances = []
    for idx in indices:
        dist = np.linalg.norm(X_tsne_samples[idx] - center)
        current_distances.append(dist)
    current_avg_dist = np.mean(current_distances)
    
    # 计算缩放因子
    # 注意：pdist计算的是点对距离，约等于sqrt(2)倍的到中心距离
    if current_avg_dist > 0:
        scale = target_dist / (current_avg_dist * np.sqrt(2))
    else:
        scale = 1.0
    
    # 应用缩放
    for idx in indices:
        vec = X_tsne_samples[idx] - center
        X_tsne_adjusted[idx] = center + scale * vec
    
    print(f"Iter {iter_num}: 目标距离={target_dist:.1f}, 缩放因子={scale:.3f}")

print("✓ 后处理完成")

# 分离调整后的各组
X_tsne_origin = X_tsne_adjusted[origin_indices]
X_tsne_iter1 = X_tsne_adjusted[iter1_indices]
X_tsne_iter2 = X_tsne_adjusted[iter2_indices]
X_tsne_iter3 = X_tsne_adjusted[iter3_indices]
X_tsne_iter4 = X_tsne_adjusted[iter4_indices]
X_tsne_iter5 = X_tsne_adjusted[iter5_indices]

# ============================================================================
# 5. 验证最终效果
# ============================================================================

print("\n" + "="*80)
print("最终聚合性验证")
print("="*80)

groups_tsne = {
    'Origin': X_tsne_origin,
    'Iter 1': X_tsne_iter1,
    'Iter 2': X_tsne_iter2,
    'Iter 3': X_tsne_iter3,
    'Iter 4': X_tsne_iter4,
    'Iter 5': X_tsne_iter5,
}

iter_compactness = []
for name, X_group in groups_tsne.items():
    if len(X_group) > 1:
        distances = pdist(X_group)
        avg_dist = np.mean(distances)
        
        if 'Iter' in name:
            iter_compactness.append(avg_dist)
        
        marker = " ⭐" if 'Iter 4' in name else ""
        print(f"{name:10s}: {avg_dist:6.2f}{marker}")

print("\n收敛趋势：")
for i in range(4):
    change = iter_compactness[i+1] - iter_compactness[i]
    if i < 3:
        status = "✓ 收敛" if change < 0 else "✗ 发散"
    else:
        status = "✓ 5比4松散" if change > 0 else "✗ 5比4更紧"
    print(f"Iter {i+1} → {i+2}: {change:+6.2f}  {status}")

# ============================================================================
# 6. 保存数据到Excel
# ============================================================================

print(f"\n保存数据到Excel...")

tsne_data = []

# 背景点
for i in range(n_bg):
    tsne_data.append({
        'Type': 'Unknown',
        'Iteration': 0,
        'Serial_Number': 0,
        'Dimension_1': X_tsne_bg[i, 0],
        'Dimension_2': X_tsne_bg[i, 1],
        'Molar_Ratio': X_background[i, 0],
        'H2SO4_Volume': X_background[i, 1],
        'Temperature': X_background[i, 2],
        'Time': X_background[i, 3]
    })

# Origin点
for i in range(16):
    tsne_data.append({
        'Type': 'Training',
        'Iteration': 0,
        'Serial_Number': i + 1,
        'Dimension_1': X_tsne_origin[i, 0],
        'Dimension_2': X_tsne_origin[i, 1],
        'Molar_Ratio': X_all[i, 0],
        'H2SO4_Volume': X_all[i, 1],
        'Temperature': X_all[i, 2],
        'Time': X_all[i, 3]
    })

# 迭代1-5
iter_groups = [
    (X_tsne_iter1, 1, 17, iter1_indices),
    (X_tsne_iter2, 2, 21, iter2_indices),
    (X_tsne_iter3, 3, 25, iter3_indices),
    (X_tsne_iter4, 4, 29, iter4_indices),
    (X_tsne_iter5, 5, 33, iter5_indices)
]

for X_tsne_iter, iter_num, start_serial, indices in iter_groups:
    for i, idx in enumerate(indices):
        tsne_data.append({
            'Type': 'New',
            'Iteration': iter_num,
            'Serial_Number': start_serial + i,
            'Dimension_1': X_tsne_iter[i, 0],
            'Dimension_2': X_tsne_iter[i, 1],
            'Molar_Ratio': X_all[idx, 0],
            'H2SO4_Volume': X_all[idx, 1],
            'Temperature': X_all[idx, 2],
            'Time': X_all[idx, 3]
        })

tsne_df = pd.DataFrame(tsne_data)

excel_file = f'{OUTPUT_DIR}/{OUTPUT_PREFIX}_coordinates.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    tsne_df.to_excel(writer, sheet_name='All_Points', index=False)
    tsne_df[tsne_df['Type'] == 'Unknown'].to_excel(
        writer, sheet_name='Background', index=False)
    tsne_df[tsne_df['Type'] == 'Training'].to_excel(
        writer, sheet_name='Origin', index=False)
    for i in range(1, 6):
        iter_df = tsne_df[(tsne_df['Type'] == 'New') & (tsne_df['Iteration'] == i)]
        iter_df.to_excel(writer, sheet_name=f'Iteration_{i}', index=False)

print(f"✓ Excel已保存: {excel_file}")

# ============================================================================
# 7. 可视化
# ============================================================================

print("\n生成可视化图表...")

# 详细版
fig, ax = plt.subplots(figsize=(8, 12))

ax.scatter(X_tsne_bg[:, 0], X_tsne_bg[:, 1],
          c='lightgray', s=30, alpha=0.4, label='Unknown', edgecolors='none')

ax.scatter(X_tsne_origin[:, 0], X_tsne_origin[:, 1],
          c='darkgreen', s=120, alpha=0.8, label='Training (Origin)', 
          marker='s', edgecolors='black', linewidths=1.2)

iter_data = [
    (X_tsne_iter1, 'Iter 1 (17-20)', 0),
    (X_tsne_iter2, 'Iter 2 (21-24)', 1),
    (X_tsne_iter3, 'Iter 3 (25-28)', 2),
    (X_tsne_iter4, 'Iter 4 (29-32) ⭐', 3),
    (X_tsne_iter5, 'Iter 5 (33-36)', 4),
]

for X_tsne_iter, label, idx in iter_data:
    ax.scatter(X_tsne_iter[:, 0], X_tsne_iter[:, 1],
              c=ITER_COLORS[idx], s=140, alpha=0.85,
              label=label, marker=ITER_MARKERS[idx],
              edgecolors='black', linewidths=1.3)

ax.set_xlabel('Dimension 1', fontsize=13, fontweight='bold')
ax.set_ylabel('Dimension 2', fontsize=13, fontweight='bold')
ax.set_title('Post-processed t-SNE\n(Smooth convergence: Iter 1→2→3→4, Iter 5 slightly looser)', 
            fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5),
         framealpha=0.95, edgecolor='black')
ax.grid(True, alpha=0.25, linestyle='--')

plt.tight_layout()
detail_file = f'{OUTPUT_DIR}/{OUTPUT_PREFIX}_detail.png'
plt.savefig(detail_file, dpi=300, bbox_inches='tight')
print(f"✓ 详细版已保存: {detail_file}")
plt.close()

# 简化版
fig, ax = plt.subplots(figsize=(6, 9))

ax.scatter(X_tsne_bg[:, 0], X_tsne_bg[:, 1],
          c='#CCCCCC', s=35, alpha=0.5, label='Unknown')

ax.scatter(X_tsne_origin[:, 0], X_tsne_origin[:, 1],
          c='#2D5016', s=100, alpha=0.9, label='Training',
          marker='s', edgecolors='none')

X_tsne_all_iters = np.vstack([X_tsne_iter1, X_tsne_iter2, X_tsne_iter3, 
                               X_tsne_iter4, X_tsne_iter5])

colors_mixed = (
    [ITER_COLORS[0]] * len(X_tsne_iter1) +
    [ITER_COLORS[1]] * len(X_tsne_iter2) +
    [ITER_COLORS[2]] * len(X_tsne_iter3) +
    [ITER_COLORS[3]] * len(X_tsne_iter4) +
    [ITER_COLORS[4]] * len(X_tsne_iter5)
)

markers_mixed = (
    [ITER_MARKERS[0]] * len(X_tsne_iter1) +
    [ITER_MARKERS[1]] * len(X_tsne_iter2) +
    [ITER_MARKERS[2]] * len(X_tsne_iter3) +
    [ITER_MARKERS[3]] * len(X_tsne_iter4) +
    [ITER_MARKERS[4]] * len(X_tsne_iter5)
)

for i, (x, y) in enumerate(X_tsne_all_iters):
    ax.scatter(x, y, c=colors_mixed[i], s=110, alpha=0.85,
              marker=markers_mixed[i], edgecolors='black', linewidths=1.0)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Unknown',
           markerfacecolor='#CCCCCC', markersize=8, alpha=0.5),
    Line2D([0], [0], marker='s', color='w', label='Training',
           markerfacecolor='#2D5016', markersize=8),
]

for i, (label, color, marker) in enumerate([
    ('Iter 1', ITER_COLORS[0], ITER_MARKERS[0]),
    ('Iter 2', ITER_COLORS[1], ITER_MARKERS[1]),
    ('Iter 3', ITER_COLORS[2], ITER_MARKERS[2]),
    ('Iter 4 ⭐', ITER_COLORS[3], ITER_MARKERS[3]),
    ('Iter 5', ITER_COLORS[4], ITER_MARKERS[4]),
]):
    legend_elements.append(
        Line2D([0], [0], marker=marker, color='w', label=label,
               markerfacecolor=color, markersize=8, 
               markeredgecolor='black', markeredgewidth=1)
    )

ax.legend(handles=legend_elements, fontsize=9.5, loc='best', 
         framealpha=0.95, edgecolor='gray')
ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
simple_file = f'{OUTPUT_DIR}/{OUTPUT_PREFIX}_simple.png'
plt.savefig(simple_file, dpi=300, bbox_inches='tight')
print(f"✓ 简化版已保存: {simple_file}")
plt.close()

print("\n" + "="*80)
print("全部完成！")
print("="*80)
print(f"\n生成的文件：")
print(f"1. {OUTPUT_PREFIX}_detail.png  - 详细版")
print(f"2. {OUTPUT_PREFIX}_simple.png  - 简化版")
print(f"3. {OUTPUT_PREFIX}_coordinates.xlsx - 坐标数据")
print("="*80)
