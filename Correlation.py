import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import os

os.makedirs("pic_output", exist_ok=True)


plt.rcParams['font.family'] = 'Times New Roman'

# Read CSV file with proper encoding
try:
    df = pd.read_csv("20260103.csv", encoding='gbk')
except:
    try:
        df = pd.read_csv('/mnt/user-data/uploads/_新数据库_CSV_16_16最新_20260103.csv', encoding='gb2312')
    except:
        df = pd.read_csv('/mnt/user-data/uploads/_新数据库_CSV_16_16最新_20260103.csv', encoding='latin1')

# Data preprocessing: extract numerical values
def clean_numeric(series):
    """Clean data and extract numerical values"""
    if series.dtype == 'object':
        # Remove units and convert to numeric
        cleaned = series.str.replace('h', '', regex=False)
        cleaned = cleaned.str.replace('°C', '', regex=False)
        cleaned = cleaned.str.replace('℃', '', regex=False)
        cleaned = cleaned.str.replace('ml', '', regex=False)
        cleaned = cleaned.str.replace('%', '', regex=False)
        cleaned = cleaned.str.replace('nm', '', regex=False)
        cleaned = cleaned.str.strip()
        return pd.to_numeric(cleaned, errors='coerce')
    return series

# Select numerical columns and clean
numeric_columns = ['Time', 'Temperature', 'Molar Ratio', 'Volume of H2SO4', 'QY', 'FWHM']
df_numeric = df[numeric_columns].copy()

# Clean each column
for col in numeric_columns:
    df_numeric[col] = clean_numeric(df_numeric[col])

# Rename columns to short names
column_names = {
    'Time': 't',
    'Temperature': 'T',
    'Molar Ratio': 'M',
    'Volume of H2SO4': 'V',
    'QY': 'QY',
    'FWHM': 'FWHM'
}
df_numeric.rename(columns=column_names, inplace=True)

# Calculate correlation matrix
correlation_matrix = df_numeric.corr()


colors = ['#695EAA','#FFFFFF','#F15638']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)


# ============================================================================
# VERSION 1: WITH NUMBERS (放大版本)
# ============================================================================
plt.figure(figsize=(12, 10))

# Plot heatmap with numbers
sns.heatmap(correlation_matrix,
            annot=True,  # Show correlation values
            fmt='.3f',   # 3 decimal places
            annot_kws={'size': 20},  # 放大数字
            cmap=cmap,   # Custom color map
            center=0,    # Set 0 as center (white)
            vmin=-1,     # Minimum value is -1
            vmax=1,      # Maximum value is 1
            square=True, # Square cells
            linewidths=1,  # Grid line width
            linecolor='gray',  # Grid line color
            cbar_kws={'shrink': 1.0})  # Color bar same height as plot

# Remove color bar ticks and label
cbar = plt.gca().collections[0].colorbar

# Get colorbar
cbar = plt.gca().collections[0].colorbar

# 1) 指定刻度位置（必须在 vmin=-1, vmax=1 的前提下）
ticks = [1, 0.5, 0, -0.5, -1]
cbar.set_ticks(ticks)

# 2) 指定显示文字（带正号）
cbar.set_ticklabels(["+1", "+0.5", "0", "-0.5", "-1"])

# 3) 把刻度/文字放到 colorbar 左侧
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
cbar.ax.tick_params(labelleft=True, labelright=False, length=0)  # length=0 可选：去掉小短线

# 关键：length>0 才会显示“刻度线”
cbar.ax.tick_params(
    labelleft=True, labelright=False,
    left=True, right=False,
    length=3, width=2,  # 这里控制刻度线长度/粗细
    direction='out',    # 朝外（左侧）画
    pad=1.5               # 文字和刻度线间距
)

# （可选）控制字号
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(20)


# Set title and labels
plt.title('Correlation matrix', fontsize=20, fontweight='bold', pad=10)
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=18, rotation=45, ha='right')
plt.yticks(fontsize=18, rotation=45)

# 删除坐标轴旁边的刻度标记
plt.tick_params(axis='both', which='both', length=2)

# Adjust layout
plt.tight_layout()

# Save figure with numbers
plt.savefig('pic_output/correlation_heatmap_with_numbers.png', dpi=1200, bbox_inches='tight')
print("Saved: correlation_heatmap_with_numbers.png")
plt.close()


# ============================================================================
# VERSION 2: WITHOUT NUMBERS (无数字版本)
# ============================================================================
plt.figure(figsize=(12, 10))

# Plot heatmap without numbers
sns.heatmap(correlation_matrix,
            annot=False,  # Hide correlation values
            cmap=cmap,   # Custom color map
            center=0,    # Set 0 as center (white)
            vmin=-1,     # Minimum value is -1
            vmax=1,      # Maximum value is 1
            square=True, # Square cells
            linewidths=1,  # Grid line width
            linecolor='gray',  # Grid line color
            cbar_kws={'shrink': 1.0})  # Color bar same height as plot

# Remove color bar ticks and label
cbar = plt.gca().collections[0].colorbar
cbar.set_ticks([])  # Remove ticks/numbers
cbar.set_label('')  # Remove label

# Set title and labels
plt.title('', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=18, rotation=45, ha='right')
plt.yticks(fontsize=18, rotation=0)

# 删除坐标轴旁边的刻度标记
plt.tick_params(axis='both', which='both', length=0)

# Adjust layout
plt.tight_layout()

# Save figure without numbers
plt.savefig('pic_output/correlation_heatmap_without_numbers.png', dpi=300, bbox_inches='tight')
print("Saved: correlation_heatmap_without_numbers.png")
plt.close()


# Display correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Find strongest positive and negative correlations
correlation_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        correlation_pairs.append({
            'Variable 1': correlation_matrix.columns[i],
            'Variable 2': correlation_matrix.columns[j],
            'Correlation': correlation_matrix.iloc[i, j]
        })

correlation_df = pd.DataFrame(correlation_pairs)
correlation_df = correlation_df.sort_values('Correlation', ascending=False)

print("\nStrongest Positive Correlations:")
print(correlation_df.head(5))

print("\nStrongest Negative Correlations:")
print(correlation_df.tail(5))

print("\n" + "="*60)
print("Generated 2 versions:")
print("1. correlation_heatmap_with_numbers.png (有数字)")
print("2. correlation_heatmap_without_numbers.png (无数字)")
print("="*60)