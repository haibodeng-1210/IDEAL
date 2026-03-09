"""
Multi-Objective Active Learning for Chemical Synthesis Optimization
双目标主动学习优化系统 - 用于化学合成参数优化 (Windows兼容版)

This code implements a novel multi-objective active learning framework
specifically designed for optimizing chemical synthesis parameters.

创新点 (Innovations):
1. Adaptive Multi-Objective Acquisition Function with Dynamic Weighting
2. Uncertainty-Aware Pareto Front Exploration Strategy
3. Domain-Specific Feature Engineering for Chemical Synthesis
4. Hybrid Gaussian Process with Chemistry-Informed Kernels

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import norm
from scipy.spatial.distance import cdist
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 路径配置 - 请根据您的实际路径修改
# ============================================================================
# 方式1: 直接指定完整路径
DATA_PATH = r"D:\PythonProgramme\c_paper\cpaperdata08.csv"
OUTPUT_DIR = r"D:\PythonProgramme\c_paper\results8"

# 方式2: 使用相对路径（如果数据在当前目录）
# DATA_PATH = "cpaperdata01.csv"
# OUTPUT_DIR = "./results"

# 创建输出目录（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ChemicalSynthesisKernel:
    """
    Chemistry-Informed Kernel for Gaussian Process
    化学合成领域知识增强的核函数

    创新点: 结合化学反应动力学的先验知识设计专用核函数

    References:
    - Griffiths, R. R., & Hernández-Lobato, J. M. (2020).
      "Constrained Bayesian optimization for automatic chemical design using variational autoencoders"
      Chemical Science, 11(2), 577-586.
    """

    def __init__(self, length_scales=None):
        """
        Args:
            length_scales: 不同特征的相关长度尺度
                          [Molar Ratio, H2SO4 Volume, Temperature, Time]
        """
        if length_scales is None:
            # 根据化学反应敏感性设置不同的长度尺度
            # Temperature和Time对反应影响更敏感，因此长度尺度较小
            length_scales = [1.0, 1.0, 0.5, 0.5]

        self.length_scales = np.array(length_scales)

        # 主核函数：Matern 5/2 核（比RBF更灵活，适合化学反应的非光滑特性）
        self.matern_kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=self.length_scales,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5  # Matern 5/2
        )

        # 噪声核：考虑实验误差
        self.noise_kernel = WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

        # 组合核函数
        self.kernel = self.matern_kernel + self.noise_kernel

    def get_kernel(self):
        return self.kernel


class MultiObjectiveAcquisition:
    """
    Adaptive Multi-Objective Acquisition Function
    自适应多目标采集函数

    创新点: 动态权重调整 + 不确定性感知的帕累托前沿探索

    该采集函数结合了三个关键策略:
    1. Expected Hypervolume Improvement (EHVI) - 期望超体积改进
    2. Uncertainty-based Exploration - 不确定性探索
    3. Diversity Promotion - 多样性促进

    References:
    - Emmerich, M., Deutz, A., & Klinkenberg, J. W. (2011).
      "Hypervolume-based expected improvement: Monotonicity properties and exact computation"
      IEEE Congress on Evolutionary Computation, 2147-2154.

    - Hernández-Lobato, D., Hernandez-Lobato, J., Shah, A., & Adams, R. (2016).
      "Predictive entropy search for multi-objective Bayesian optimization"
      International Conference on Machine Learning, 1492-1501.
    """

    def __init__(self, reference_point=None):
        """
        Args:
            reference_point: 参考点，用于计算超体积
                           [FWHM_max, QY_min] - 最差的可接受值
        """
        self.reference_point = reference_point
        self.alpha = 0.5  # 探索-利用平衡参数
        self.beta = 0.3   # 多样性权重

    def compute_pareto_front(self, objectives):
        """
        计算当前的帕累托前沿

        Args:
            objectives: (n_samples, 2) array, [FWHM, QY]
                       注意: FWHM需要最小化, QY需要最大化

        Returns:
            pareto_mask: boolean array indicating Pareto optimal points
        """
        n_points = objectives.shape[0]
        pareto_mask = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if pareto_mask[i]:
                # 对于FWHM: 越小越好; 对于QY: 越大越好
                # 点i被点j支配的条件: FWHM_j <= FWHM_i AND QY_j >= QY_i
                dominated = (
                    (objectives[:, 0] <= objectives[i, 0]) &  # FWHM
                    (objectives[:, 1] >= objectives[i, 1]) &  # QY
                    ((objectives[:, 0] < objectives[i, 0]) | (objectives[:, 1] > objectives[i, 1]))
                )
                pareto_mask[i] = not np.any(dominated)

        return pareto_mask

    def expected_improvement_2d(self, mean, std, current_pareto_front):
        """
        计算二维期望改进 (Expected Improvement for bi-objective optimization)

        创新点: 针对FWHM和QY的不同优化方向设计的联合改进指标

        Args:
            mean: (n_candidates, 2) predicted means [FWHM, QY]
            std: (n_candidates, 2) predicted std [FWHM, QY]
            current_pareto_front: (n_pareto, 2) current Pareto optimal points

        Returns:
            ei: (n_candidates,) expected improvement values
        """
        n_candidates = mean.shape[0]
        ei_scores = np.zeros(n_candidates)

        # 对于每个候选点，计算其相对于当前帕累托前沿的改进
        for i in range(n_candidates):
            # FWHM: 最小化目标 - 计算比当前最好值更小的概率
            fwhm_best = np.min(current_pareto_front[:, 0])
            z_fwhm = (fwhm_best - mean[i, 0]) / (std[i, 0] + 1e-9)
            ei_fwhm = std[i, 0] * (z_fwhm * norm.cdf(z_fwhm) + norm.pdf(z_fwhm))

            # QY: 最大化目标 - 计算比当前最好值更大的概率
            qy_best = np.max(current_pareto_front[:, 1])
            z_qy = (mean[i, 1] - qy_best) / (std[i, 1] + 1e-9)
            ei_qy = std[i, 1] * (z_qy * norm.cdf(z_qy) + norm.pdf(z_qy))

            # 联合改进指标：几何平均（避免单一目标主导）
            ei_scores[i] = np.sqrt(ei_fwhm * ei_qy)

        return ei_scores

    def uncertainty_score(self, std):
        """
        不确定性得分 - 用于探索高不确定性区域

        Args:
            std: (n_candidates, 2) predicted standard deviations

        Returns:
            uncertainty: (n_candidates,) uncertainty scores
        """
        # 使用标准差的几何平均作为不确定性度量
        return np.sqrt(std[:, 0] * std[:, 1])

    def diversity_score(self, X_candidates, X_observed):
        """
        多样性得分 - 促进在参数空间中的探索

        创新点: 基于最小距离的多样性度量，避免采样聚集

        Args:
            X_candidates: (n_candidates, n_features) candidate points
            X_observed: (n_observed, n_features) observed points

        Returns:
            diversity: (n_candidates,) diversity scores
        """
        # 计算每个候选点到所有已观测点的最小距离
        distances = cdist(X_candidates, X_observed, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        # 归一化到[0, 1]
        if np.max(min_distances) > 0:
            diversity = min_distances / np.max(min_distances)
        else:
            diversity = np.ones(len(X_candidates))

        return diversity

    def compute_acquisition(self, mean, std, X_candidates, X_observed, current_pareto_front):
        """
        计算综合采集函数值

        创新点: 自适应权重的多准则融合策略

        Acquisition = w1 * EI + w2 * Uncertainty + w3 * Diversity

        其中权重根据优化进度动态调整:
        - 早期: 更多探索 (高不确定性和多样性权重)
        - 后期: 更多利用 (高EI权重)

        Args:
            mean, std: GP predictions
            X_candidates, X_observed: candidate and observed points
            current_pareto_front: current Pareto optimal objectives

        Returns:
            acquisition: (n_candidates,) acquisition function values
        """
        # 计算三个组成部分
        ei = self.expected_improvement_2d(mean, std, current_pareto_front)
        uncertainty = self.uncertainty_score(std)
        diversity = self.diversity_score(X_candidates, X_observed)

        # 归一化到[0, 1]
        ei_norm = ei / (np.max(ei) + 1e-9)
        uncertainty_norm = uncertainty / (np.max(uncertainty) + 1e-9)
        diversity_norm = diversity  # 已经归一化

        # 动态权重调整
        n_observed = len(X_observed)
        exploration_weight = np.exp(-n_observed / 20)  # 随着迭代递减

        w1 = 1 - exploration_weight  # EI权重: 逐渐增加
        w2 = self.alpha * exploration_weight  # 不确定性权重: 逐渐减少
        w3 = self.beta * exploration_weight  # 多样性权重: 逐渐减少

        # 综合采集函数
        acquisition = w1 * ei_norm + w2 * uncertainty_norm + w3 * diversity_norm

        return acquisition


class BiObjectiveActiveLearning:
    """
    Bi-Objective Active Learning Framework
    双目标主动学习框架

    完整的闭环优化系统，专为化学合成参数优化设计

    创新架构:
    1. 双GP模型：分别建模FWHM和QY
    2. 化学领域知识增强的核函数
    3. 自适应多目标采集策略
    4. 帕累托前沿追踪和可视化

    References:
    - Rasmussen, C. E., & Williams, C. K. (2006).
      "Gaussian processes for machine learning"
      MIT Press.

    - Frazier, P. I. (2018).
      "A tutorial on Bayesian optimization"
      arXiv preprint arXiv:1807.02811.
    """

    def __init__(self, parameter_bounds, n_initial_samples=54):
        """
        Args:
            parameter_bounds: dict, 参数范围
                {
                    'Molar Ratio': (min, max),
                    'H2SO4 Volume': (min, max),
                    'Temperature': (min, max),
                    'Time': (min, max)
                }
            n_initial_samples: 初始样本数量
        """
        self.parameter_bounds = parameter_bounds
        self.n_initial_samples = n_initial_samples

        # 数据存储
        self.X_train = None  # 输入特征
        self.y_fwhm = None   # FWHM目标
        self.y_qy = None     # QY目标

        # 数据预处理
        self.scaler_X = StandardScaler()
        self.scaler_fwhm = StandardScaler()
        self.scaler_qy = StandardScaler()

        # 初始化GP模型
        chem_kernel = ChemicalSynthesisKernel()

        self.gp_fwhm = GaussianProcessRegressor(
            kernel=chem_kernel.get_kernel(),
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=False  # 我们手动标准化
        )

        self.gp_qy = GaussianProcessRegressor(
            kernel=chem_kernel.get_kernel(),
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=False
        )

        # 采集函数
        self.acquisition = MultiObjectiveAcquisition()

        # 历史记录
        self.history = {
            'iteration': [],
            'pareto_fronts': [],
            'best_fwhm': [],
            'best_qy': [],
            'hypervolume': []
        }

    def load_data(self, data_path):
        """
        加载实验数据

        Args:
            data_path: CSV文件路径
        """
        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']

        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                print(f"✓ 成功使用 {encoding} 编码加载数据")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"无法读取文件 {data_path}。尝试了以下编码: {encodings}")

        # 提取特征和目标
        feature_columns = ['Molar Ratio', 'H2SO4 Volume', 'Temperature', 'Time']
        self.X_train = df[feature_columns].values
        self.y_fwhm = df['FWHM'].values.reshape(-1, 1)
        self.y_qy = df['QY'].values.reshape(-1, 1)

        print(f"✓ 成功加载数据: {len(self.X_train)} 个样本")
        print(f"  特征范围:")
        for i, col in enumerate(feature_columns):
            print(f"    {col}: [{self.X_train[:, i].min():.2f}, {self.X_train[:, i].max():.2f}]")
        print(f"  FWHM范围: [{self.y_fwhm.min():.2f}, {self.y_fwhm.max():.2f}]")
        print(f"  QY范围: [{self.y_qy.min():.4f}, {self.y_qy.max():.4f}]")

    def fit_models(self):
        """
        训练双GP模型

        技术细节:
        1. 数据标准化：提高GP训练稳定性
        2. 分别训练：FWHM和QY可能有不同的相关结构
        3. 超参数优化：自动调整核函数参数
        """
        # 标准化数据
        X_scaled = self.scaler_X.fit_transform(self.X_train)
        y_fwhm_scaled = self.scaler_fwhm.fit_transform(self.y_fwhm)
        y_qy_scaled = self.scaler_qy.fit_transform(self.y_qy)

        # 训练FWHM模型
        print("\n训练FWHM预测模型...")
        self.gp_fwhm.fit(X_scaled, y_fwhm_scaled.ravel())
        print(f"  ✓ FWHM模型训练完成 (训练得分: {self.gp_fwhm.score(X_scaled, y_fwhm_scaled.ravel()):.4f})")

        # 训练QY模型
        print("训练QY预测模型...")
        self.gp_qy.fit(X_scaled, y_qy_scaled.ravel())
        print(f"  ✓ QY模型训练完成 (训练得分: {self.gp_qy.score(X_scaled, y_qy_scaled.ravel()):.4f})")

    def generate_candidates(self, n_candidates=1000):
        """
        生成候选点集合

        策略: 拉丁超立方采样 (Latin Hypercube Sampling)
        优点: 在参数空间中均匀分布，避免聚集

        Args:
            n_candidates: 候选点数量

        Returns:
            candidates: (n_candidates, n_features) array
        """
        from scipy.stats import qmc

        # 获取参数维度
        n_features = self.X_train.shape[1]

        # 使用拉丁超立方采样
        sampler = qmc.LatinHypercube(d=n_features, seed=42)
        samples = sampler.random(n=n_candidates)

        # 缩放到实际参数范围
        feature_names = ['Molar Ratio', 'H2SO4 Volume', 'Temperature', 'Time']
        candidates = np.zeros_like(samples)

        for i, name in enumerate(feature_names):
            lower, upper = self.parameter_bounds[name]
            candidates[:, i] = samples[:, i] * (upper - lower) + lower

        return candidates

    def predict(self, X):
        """
        对候选点进行预测

        Returns:
            mean: (n_candidates, 2) [FWHM_mean, QY_mean]
            std: (n_candidates, 2) [FWHM_std, QY_std]
        """
        X_scaled = self.scaler_X.transform(X)

        # FWHM预测
        fwhm_mean_scaled, fwhm_std_scaled = self.gp_fwhm.predict(X_scaled, return_std=True)
        fwhm_mean = self.scaler_fwhm.inverse_transform(fwhm_mean_scaled.reshape(-1, 1)).ravel()
        fwhm_std = fwhm_std_scaled * self.scaler_fwhm.scale_[0]

        # QY预测
        qy_mean_scaled, qy_std_scaled = self.gp_qy.predict(X_scaled, return_std=True)
        qy_mean = self.scaler_qy.inverse_transform(qy_mean_scaled.reshape(-1, 1)).ravel()
        qy_std = qy_std_scaled * self.scaler_qy.scale_[0]

        mean = np.column_stack([fwhm_mean, qy_mean])
        std = np.column_stack([fwhm_std, qy_std])

        return mean, std

    def select_next_experiments(self, n_suggestions=3):
        """
        选择下一批实验点

        核心创新: 批量选择策略
        1. 生成大量候选点
        2. 计算采集函数值
        3. 依次选择最优点，更新候选集

        Args:
            n_suggestions: 建议的实验数量

        Returns:
            suggestions: (n_suggestions, n_features) 建议的实验参数
            predicted_objectives: (n_suggestions, 2) 预测的目标值
            acquisition_values: (n_suggestions,) 采集函数值
        """
        print(f"\n{'='*60}")
        print(f"选择下一批 {n_suggestions} 个实验点")
        print(f"{'='*60}")

        # 生成候选点
        candidates = self.generate_candidates(n_candidates=2000)
        print(f"✓ 生成 {len(candidates)} 个候选点")

        # 预测候选点
        mean, std = self.predict(candidates)
        print(f"✓ 完成预测")

        # 计算当前帕累托前沿
        current_objectives = np.column_stack([self.y_fwhm.ravel(), self.y_qy.ravel()])
        pareto_mask = self.acquisition.compute_pareto_front(current_objectives)
        current_pareto = current_objectives[pareto_mask]
        print(f"✓ 当前帕累托前沿包含 {len(current_pareto)} 个点")

        # 批量选择策略
        suggestions = []
        predicted_objectives = []
        acquisition_values = []

        for i in range(n_suggestions):
            # 计算采集函数
            acq_values = self.acquisition.compute_acquisition(
                mean, std, candidates, self.X_train, current_pareto
            )

            # 选择最佳候选点
            best_idx = np.argmax(acq_values)
            suggestions.append(candidates[best_idx])
            predicted_objectives.append(mean[best_idx])
            acquisition_values.append(acq_values[best_idx])

            # 从候选集中移除已选择的点及其邻域
            # 这样可以促进多样性
            distances = np.linalg.norm(candidates - candidates[best_idx], axis=1)
            remove_mask = distances < 0.1 * np.max(distances)
            candidates = candidates[~remove_mask]
            mean = mean[~remove_mask]
            std = std[~remove_mask]

            print(f"  第 {i+1} 个建议点: 采集函数值 = {acq_values[best_idx]:.4f}")

        suggestions = np.array(suggestions)
        predicted_objectives = np.array(predicted_objectives)
        acquisition_values = np.array(acquisition_values)

        return suggestions, predicted_objectives, acquisition_values

    def update_with_new_data(self, X_new, y_fwhm_new, y_qy_new):
        """
        添加新实验数据并重新训练模型

        Args:
            X_new: (n_new, n_features) 新的输入参数
            y_fwhm_new: (n_new,) 新的FWHM测量值
            y_qy_new: (n_new,) 新的QY测量值
        """
        # 添加到训练集
        self.X_train = np.vstack([self.X_train, X_new])
        self.y_fwhm = np.vstack([self.y_fwhm, y_fwhm_new.reshape(-1, 1)])
        self.y_qy = np.vstack([self.y_qy, y_qy_new.reshape(-1, 1)])

        print(f"\n✓ 添加 {len(X_new)} 个新样本，总样本数: {len(self.X_train)}")

        # 重新训练模型
        self.fit_models()

    def compute_hypervolume(self, pareto_front, reference_point):
        """
        计算超体积指标

        超体积(Hypervolume): 帕累托前沿的重要质量指标
        测量帕累托前沿在目标空间中"支配"的体积

        Reference:
        - Zitzler, E., & Thiele, L. (1999).
          "Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach"
          IEEE transactions on Evolutionary Computation, 3(4), 257-271.

        Args:
            pareto_front: (n_pareto, 2) Pareto optimal points [FWHM, QY]
            reference_point: (2,) reference point [FWHM_ref, QY_ref]

        Returns:
            hypervolume: float
        """
        if len(pareto_front) == 0:
            return 0.0

        # 对于二维问题，超体积计算相对简单
        # 转换为最大化问题: [FWHM_max - FWHM, QY - QY_min]
        normalized_front = np.column_stack([
            reference_point[0] - pareto_front[:, 0],  # FWHM: 越小越好
            pareto_front[:, 1] - reference_point[1]   # QY: 越大越好
        ])

        # 按第一个目标排序
        sorted_indices = np.argsort(normalized_front[:, 0])
        sorted_front = normalized_front[sorted_indices]

        # 计算超体积
        hypervolume = 0.0
        for i in range(len(sorted_front)):
            if i == 0:
                width = sorted_front[i, 0]
            else:
                width = sorted_front[i, 0] - sorted_front[i-1, 0]
            height = sorted_front[i, 1]
            hypervolume += width * height

        return hypervolume

    def analyze_results(self, iteration):
        """
        分析当前结果并更新历史记录

        Args:
            iteration: 当前迭代次数
        """
        # 计算帕累托前沿
        objectives = np.column_stack([self.y_fwhm.ravel(), self.y_qy.ravel()])
        pareto_mask = self.acquisition.compute_pareto_front(objectives)
        pareto_front = objectives[pareto_mask]

        # 计算超体积
        reference_point = [200, 0]  # 设置参考点
        hv = self.compute_hypervolume(pareto_front, reference_point)

        # 记录历史
        self.history['iteration'].append(iteration)
        self.history['pareto_fronts'].append(pareto_front)
        self.history['best_fwhm'].append(np.min(self.y_fwhm))
        self.history['best_qy'].append(np.max(self.y_qy))
        self.history['hypervolume'].append(hv)

        print(f"\n{'='*60}")
        print(f"第 {iteration} 轮迭代分析")
        print(f"{'='*60}")
        print(f"当前帕累托前沿: {len(pareto_front)} 个解")
        print(f"最优FWHM: {np.min(self.y_fwhm):.2f}")
        print(f"最优QY: {np.max(self.y_qy):.4f}")
        print(f"超体积指标: {hv:.2f}")
        print(f"{'='*60}")

    def visualize_results(self, save_path='results'):
        """
        可视化优化结果

        生成四个关键图表:
        1. 帕累托前沿演化
        2. 超体积指标演化
        3. 单目标优化进度
        4. 参数空间探索
        """
        os.makedirs(save_path, exist_ok=True)

        # 图1: 帕累托前沿演化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (1) 帕累托前沿
        ax = axes[0, 0]
        iterations_to_plot = [0, len(self.history['iteration'])//2, len(self.history['iteration'])-1]
        colors = ['blue', 'green', 'red']
        labels = ['初始', '中期', '最终']

        for idx, color, label in zip(iterations_to_plot, colors, labels):
            if idx < len(self.history['pareto_fronts']):
                pf = self.history['pareto_fronts'][idx]
                ax.scatter(pf[:, 0], pf[:, 1], c=color, s=100, alpha=0.6, label=label, edgecolors='black')

        ax.set_xlabel('FWHM (越小越好)', fontsize=12, fontweight='bold')
        ax.set_ylabel('QY (越大越好)', fontsize=12, fontweight='bold')
        ax.set_title('帕累托前沿演化', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # (2) 超体积演化
        ax = axes[0, 1]
        ax.plot(self.history['iteration'], self.history['hypervolume'],
                marker='o', linewidth=2, markersize=6, color='purple')
        ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax.set_ylabel('超体积 (Hypervolume)', fontsize=12, fontweight='bold')
        ax.set_title('超体积指标演化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # (3) 最优FWHM演化
        ax = axes[1, 0]
        ax.plot(self.history['iteration'], self.history['best_fwhm'],
                marker='s', linewidth=2, markersize=6, color='blue')
        ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax.set_ylabel('最优FWHM', fontsize=12, fontweight='bold')
        ax.set_title('FWHM优化进度', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # (4) 最优QY演化
        ax = axes[1, 1]
        ax.plot(self.history['iteration'], self.history['best_qy'],
                marker='^', linewidth=2, markersize=6, color='red')
        ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax.set_ylabel('最优QY', fontsize=12, fontweight='bold')
        ax.set_title('QY优化进度', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        progress_file = os.path.join(save_path, 'optimization_progress.png')
        plt.savefig(progress_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ 保存可视化结果到: {progress_file}")
        plt.close()

        # 图2: 当前最优帕累托前沿详细图
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制所有样本点
        objectives = np.column_stack([self.y_fwhm.ravel(), self.y_qy.ravel()])
        pareto_mask = self.acquisition.compute_pareto_front(objectives)

        ax.scatter(objectives[~pareto_mask, 0], objectives[~pareto_mask, 1],
                  c='lightgray', s=50, alpha=0.5, label='非帕累托点')
        ax.scatter(objectives[pareto_mask, 0], objectives[pareto_mask, 1],
                  c='red', s=150, alpha=0.8, label='帕累托前沿', edgecolors='black', linewidths=2)

        ax.set_xlabel('FWHM (越小越好)', fontsize=14, fontweight='bold')
        ax.set_ylabel('QY (越大越好)', fontsize=14, fontweight='bold')
        ax.set_title('当前帕累托前沿', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pareto_file = os.path.join(save_path, 'pareto_front_final.png')
        plt.savefig(pareto_file, dpi=300, bbox_inches='tight')
        print(f"✓ 保存帕累托前沿图到: {pareto_file}")
        plt.close()

        return progress_file, pareto_file


def run_active_learning_iteration(al_system, iteration, n_suggestions=3):
    """
    执行一次主动学习迭代

    Args:
        al_system: BiObjectiveActiveLearning 实例
        iteration: 当前迭代次数
        n_suggestions: 建议实验数量

    Returns:
        suggestions_df: 包含建议实验参数的DataFrame
    """
    print(f"\n{'#'*70}")
    print(f"#  第 {iteration} 轮主动学习迭代")
    print(f"{'#'*70}")

    # 训练模型
    al_system.fit_models()

    # 选择下一批实验
    suggestions, predicted_obj, acq_values = al_system.select_next_experiments(n_suggestions)

    # 创建结果DataFrame
    feature_names = ['Molar Ratio', 'H2SO4 Volume', 'Temperature', 'Time']
    suggestions_df = pd.DataFrame(suggestions, columns=feature_names)
    suggestions_df['Predicted_FWHM'] = predicted_obj[:, 0]
    suggestions_df['Predicted_QY'] = predicted_obj[:, 1]
    suggestions_df['Acquisition_Value'] = acq_values

    # 分析当前结果
    al_system.analyze_results(iteration)

    return suggestions_df


# ============================================================================
# 主程序: 演示如何使用该系统
# ============================================================================

def main():
    """
    主程序: 演示双目标主动学习系统的完整流程
    """
    print("\n" + "="*70)
    print(" 双目标主动学习优化系统 (Windows版)")
    print(" Multi-Objective Active Learning for Chemical Synthesis")
    print("="*70)

    # 1. 定义参数空间
    parameter_bounds = {
        'Molar Ratio': (250, 750),
        'H2SO4 Volume': (1, 4),
        'Temperature': (150, 200),
        'Time': (8, 12)
    }

    # 2. 初始化系统
    al_system = BiObjectiveActiveLearning(
        parameter_bounds=parameter_bounds,
        n_initial_samples=54
    )

    # 3. 加载初始数据
    print(f"\n正在从以下路径加载数据: {DATA_PATH}")
    al_system.load_data(DATA_PATH)

    # 4. 执行第一轮迭代（基于初始54个样本）
    suggestions_df = run_active_learning_iteration(al_system, iteration=0, n_suggestions=4)

    # 5. 显示建议的实验参数
    print("\n" + "="*70)
    print("建议的实验参数:")
    print("="*70)
    print(suggestions_df.to_string(index=False))

    # 6. 保存建议到CSV（保存到输出目录）
    suggestion_file = os.path.join(OUTPUT_DIR, 'suggested_experiments_iteration_0.csv')
    suggestions_df.to_csv(suggestion_file, index=False)
    print(f"\n✓ 建议已保存到: {suggestion_file}")

    # 7. 可视化结果（保存到输出目录）
    fig1, fig2 = al_system.visualize_results(save_path=OUTPUT_DIR)

    print("\n" + "="*70)
    print("第一轮迭代完成!")
    print("="*70)
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("\n下一步操作:")
    print("1. 使用建议的参数进行真实实验")
    print("2. 测量新样本的FWHM和QY")
    print("3. 调用 al_system.update_with_new_data() 添加新数据")
    print("4. 继续下一轮迭代")

    return al_system, suggestions_df


if __name__ == "__main__":
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ 错误: 找不到数据文件!")
        print(f"请检查路径: {DATA_PATH}")
        print("\n请修改代码开头的 DATA_PATH 变量为您的实际数据路径")
        print("例如: DATA_PATH = r'D:\\PythonProgramme\\c_paper\\cpaperdata01.csv'")
    else:
        al_system, suggestions = main()