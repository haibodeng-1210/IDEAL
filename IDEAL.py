"""
IDEAL: Interpretable Dynamic Experimental Active Learning
可解释动态实验主动学习框架

A generalized N-objective active learning framework for chemical synthesis
optimization. Supports single-objective, bi-objective, and multi-objective
optimization with minimal configuration changes.

通用N目标主动学习优化框架，支持单目标、双目标、多目标优化。
切换目标数量只需修改下方"用户配置区"，无需改动其他代码。

Framework Architecture:
1. Adaptive Multi-Objective Acquisition Function with Dynamic Weighting
2. Uncertainty-Aware Pareto Front Exploration Strategy
3. Domain-Specific Feature Engineering for Chemical Synthesis
4. Hybrid Gaussian Process with Chemistry-Informed Kernels

References:
- Griffiths & Hernández-Lobato (2020). Chemical Science, 11(2), 577-586.
- Emmerich et al. (2011). IEEE Congress on Evolutionary Computation.
- Hernández-Lobato et al. (2016). ICML, 1492-1501.
- Rasmussen & Williams (2006). Gaussian Processes for Machine Learning, MIT Press.
- Frazier (2018). arXiv:1807.02811.

Author: Research Team
Version: 2.0 (IDEAL Framework)
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import norm
from scipy.spatial.distance import cdist
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# ██╗   ██╗███████╗███████╗██████╗      ██████╗ ██████╗ ███╗   ██╗███████╗
# ██║   ██║██╔════╝██╔════╝██╔══██╗    ██╔════╝██╔═══██╗████╗  ██║██╔════╝
# ██║   ██║███████╗█████╗  ██████╔╝    ██║     ██║   ██║██╔██╗ ██║█████╗
# ██║   ██║╚════██║██╔══╝  ██╔══██╗    ██║     ██║   ██║██║╚██╗██║██╔══╝
# ╚██████╔╝███████║███████╗██║  ██║    ╚██████╗╚██████╔╝██║ ╚████║██║
#  ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝
#
#   IDEAL 用户配置区 — 只需修改这里，其余代码不用动
# ============================================================================

# ----------------------------------------------------------------------------
# [1] 路径配置
# ----------------------------------------------------------------------------
DATA_PATH  = r"D:\PythonProgramme\c_paper\cpaperdatan16_16.csv"
OUTPUT_DIR = r"D:\PythonProgramme\c_paper\resultsn16_16"

# ----------------------------------------------------------------------------
# [2] 合成条件描述符（输入特征 / Synthesis Condition Descriptors）
#     列出CSV中作为输入的列名，顺序随意
# ----------------------------------------------------------------------------
CONDITION_DESCRIPTORS = [
    'Molar Ratio',
    'H2SO4 Volume',
    'Temperature',
    'Time',
]

# ----------------------------------------------------------------------------
# [3] 优化目标描述符（Performance Descriptors）
#     每个目标需指定：
#       - 'column'    : CSV中的列名
#       - 'direction' : 'minimize' (越小越好) 或 'maximize' (越大越好)
#       - 'ref'       : 参考点（用于超体积计算；minimize时取较大值，maximize时取较小值）
#
#   想切换目标数量，直接增删下面的字典条目即可
#   示例1 — 单目标:  只保留一个字典
#   示例2 — 双目标:  保留两个字典  ← 当前配置
#   示例3 — 三目标:  增加一个字典，如 {'column': 'PL_peak', 'direction': 'minimize', 'ref': 700}
# ----------------------------------------------------------------------------
OBJECTIVE_DESCRIPTORS = [
    {'column': 'FWHM', 'direction': 'minimize', 'ref': 200},
    {'column': 'QY',   'direction': 'maximize', 'ref': 0  },
    # {'column': 'PL_peak', 'direction': 'minimize', 'ref': 700},   # ← 解注释即为三目标
]

# ----------------------------------------------------------------------------
# [4] 其他运行参数（一般不需要改）
# ----------------------------------------------------------------------------
N_INITIAL_SAMPLES = 16      # 初始样本数
N_SUGGESTIONS     = 4       # 每轮建议的实验数
N_CANDIDATES      = 20000    # 候选点采样数（越大越精细，也越慢）

# ============================================================================
#   配置区结束 ↑  以下为 IDEAL 核心代码，通常不需要修改
# ============================================================================


os.makedirs(OUTPUT_DIR, exist_ok=True)

# 自动推断目标数
N_OBJECTIVES = len(OBJECTIVE_DESCRIPTORS)
assert N_OBJECTIVES >= 1, "至少需要定义一个优化目标"


# ============================================================================
# Core Module 1: Chemistry-Informed Gaussian Process Kernel
# ============================================================================

class ChemicalSynthesisKernel:
    """
    Chemistry-Informed Kernel for Gaussian Process Regression.

    Uses Matern 5/2 kernel (more flexible than RBF for non-smooth chemical
    responses) with per-feature length scales, plus a WhiteKernel for
    experimental noise.

    Length scales are set shorter for condition descriptors that are
    typically more sensitive (e.g., Temperature, Time).
    """

    def __init__(self, n_features):
        """
        Args:
            n_features (int): Number of input (condition) features.
        """
        length_scales = [1.0] * n_features
        # Heuristic: last two features often represent Temperature & Time
        # which are typically more sensitive — use smaller length scales
        for i in range(max(0, n_features - 2), n_features):
            length_scales[i] = 0.5

        self.kernel = (
            C(1.0, (1e-3, 1e3)) *
            Matern(length_scale=length_scales,
                   length_scale_bounds=(1e-2, 1e2),
                   nu=2.5)
            + WhiteKernel(noise_level=1e-5,
                          noise_level_bounds=(1e-10, 1e-1))
        )

    def get_kernel(self):
        return self.kernel


# ============================================================================
# Core Module 2: Generalized N-Objective Acquisition Function
# ============================================================================

class IDEALAcquisition:
    """
    IDEAL Acquisition Function — generalized for N objectives.

    Strategy (same as original bi-objective design, now extended to N):
      Acquisition = w1 * EI_combined + w2 * Uncertainty + w3 * Diversity

    Weights are dynamically adjusted:
      - Early iterations: higher exploration (uncertainty + diversity)
      - Later iterations: higher exploitation (EI)

    For single-objective mode, falls back to standard Expected Improvement.
    For multi-objective mode, uses the generalized Pareto dominance check
    and joint EI across all objectives.
    """

    def __init__(self, objective_descriptors, alpha=0.5, beta=0.3):
        """
        Args:
            objective_descriptors: list of dicts, each with keys
                'column', 'direction' ('minimize'|'maximize'), 'ref'
            alpha: weight coefficient for uncertainty term
            beta:  weight coefficient for diversity term
        """
        self.objectives = objective_descriptors
        self.n_obj = len(objective_descriptors)
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------
    # Pareto utilities (generalized to N dimensions)
    # ------------------------------------------------------------------

    def _dominates(self, a, b):
        """
        Return True if point a dominates point b under the configured
        optimization directions.

        a dominates b if:
          - a is no worse than b in ALL objectives, AND
          - a is strictly better than b in AT LEAST ONE objective
        """
        no_worse   = True
        at_least_one_better = False

        for k, obj in enumerate(self.objectives):
            if obj['direction'] == 'minimize':
                # smaller is better
                if a[k] > b[k]:
                    no_worse = False
                    break
                if a[k] < b[k]:
                    at_least_one_better = True
            else:
                # larger is better
                if a[k] < b[k]:
                    no_worse = False
                    break
                if a[k] > b[k]:
                    at_least_one_better = True

        return no_worse and at_least_one_better

    def compute_pareto_front(self, objectives_array):
        """
        Compute the Pareto-optimal mask for N objectives.

        Args:
            objectives_array: (n_samples, n_obj) array

        Returns:
            pareto_mask: (n_samples,) boolean array
        """
        n = objectives_array.shape[0]
        pareto_mask = np.ones(n, dtype=bool)

        for i in range(n):
            if pareto_mask[i]:
                for j in range(n):
                    if j != i and pareto_mask[j]:
                        if self._dominates(objectives_array[j], objectives_array[i]):
                            pareto_mask[i] = False
                            break

        return pareto_mask

    # ------------------------------------------------------------------
    # EI (generalized to N objectives)
    # ------------------------------------------------------------------

    def _best_reference(self, pareto_front):
        """
        Compute per-objective reference (best observed value) from the
        current Pareto front.
        """
        refs = []
        for k, obj in enumerate(self.objectives):
            if obj['direction'] == 'minimize':
                refs.append(np.min(pareto_front[:, k]))
            else:
                refs.append(np.max(pareto_front[:, k]))
        return refs

    def expected_improvement_nd(self, mean, std, current_pareto_front):
        """
        Generalized joint Expected Improvement for N objectives.

        For each objective independently computes EI against the best
        observed value, then combines via geometric mean (prevents any
        single objective from dominating).

        Args:
            mean: (n_candidates, n_obj)
            std:  (n_candidates, n_obj)
            current_pareto_front: (n_pareto, n_obj)

        Returns:
            ei: (n_candidates,) combined EI scores
        """
        n_candidates = mean.shape[0]
        best_refs = self._best_reference(current_pareto_front)

        ei_per_obj = np.zeros((n_candidates, self.n_obj))

        for k, obj in enumerate(self.objectives):
            if obj['direction'] == 'minimize':
                z = (best_refs[k] - mean[:, k]) / (std[:, k] + 1e-9)
            else:
                z = (mean[:, k] - best_refs[k]) / (std[:, k] + 1e-9)

            ei_per_obj[:, k] = std[:, k] * (z * norm.cdf(z) + norm.pdf(z))

        # Geometric mean across objectives
        if self.n_obj == 1:
            return ei_per_obj[:, 0]
        else:
            # Clip negatives to 0 before geometric mean
            ei_clipped = np.clip(ei_per_obj, 0, None)
            return np.power(np.prod(ei_clipped + 1e-30, axis=1),
                            1.0 / self.n_obj)

    # ------------------------------------------------------------------
    # Uncertainty & Diversity
    # ------------------------------------------------------------------

    def uncertainty_score(self, std):
        """
        Geometric mean of std across all objectives.

        Args:
            std: (n_candidates, n_obj)

        Returns:
            uncertainty: (n_candidates,)
        """
        if self.n_obj == 1:
            return std[:, 0]
        return np.power(np.prod(std + 1e-30, axis=1), 1.0 / self.n_obj)

    def diversity_score(self, X_candidates, X_observed):
        """
        Minimum-distance diversity score in condition space.

        Encourages sampling away from already-observed points.

        Args:
            X_candidates: (n_candidates, n_features)
            X_observed:   (n_observed,   n_features)

        Returns:
            diversity: (n_candidates,) normalized to [0, 1]
        """
        distances    = cdist(X_candidates, X_observed, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        max_dist = np.max(min_distances)
        if max_dist > 0:
            return min_distances / max_dist
        return np.ones(len(X_candidates))

    # ------------------------------------------------------------------
    # Combined acquisition
    # ------------------------------------------------------------------

    def compute_acquisition(self, mean, std, X_candidates,
                            X_observed, current_pareto_front):
        """
        Compute the combined IDEAL acquisition value for each candidate.

        Acquisition = w1 * EI_norm + w2 * Uncertainty_norm + w3 * Diversity

        Weights are dynamically adjusted by exploration_weight which
        decays as more observations are collected.

        Args:
            mean, std:           GP predictions (n_candidates, n_obj)
            X_candidates:        (n_candidates, n_features)
            X_observed:          (n_observed,   n_features)
            current_pareto_front:(n_pareto,      n_obj)

        Returns:
            acquisition: (n_candidates,)
        """
        ei          = self.expected_improvement_nd(mean, std, current_pareto_front)
        uncertainty = self.uncertainty_score(std)
        diversity   = self.diversity_score(X_candidates, X_observed)

        # Normalize EI and uncertainty to [0, 1]
        ei_norm          = ei          / (np.max(ei)          + 1e-9)
        uncertainty_norm = uncertainty / (np.max(uncertainty) + 1e-9)

        # Dynamic weight schedule
        n_obs = len(X_observed)
        exploration_weight = np.exp(-n_obs / 20)

        w1 = 1.0 - exploration_weight                   # EI weight
        w2 = self.alpha * exploration_weight            # Uncertainty weight
        w3 = self.beta  * exploration_weight            # Diversity weight

        return w1 * ei_norm + w2 * uncertainty_norm + w3 * diversity


# ============================================================================
# Core Module 3: IDEAL System
# ============================================================================

class IDEALSystem:
    """
    IDEAL: Interpretable Dynamic Experimental Active Learning System

    A closed-loop, N-objective Bayesian active learning framework for
    chemical synthesis optimization.

    Key design:
    - One GP model per objective (independent modelling)
    - Generalized Pareto front tracking
    - Adaptive multi-objective acquisition function
    - Configurable via OBJECTIVE_DESCRIPTORS (see user config above)

    Usage:
        system = IDEALSystem(parameter_bounds=PARAMETER_BOUNDS)
        system.load_data(DATA_PATH)
        suggestions_df = system.run_iteration(iteration=0)
    """

    def __init__(self, parameter_bounds, n_initial_samples=N_INITIAL_SAMPLES):
        """
        Args:
            parameter_bounds: dict  {feature_name: (min, max)}
            n_initial_samples: int  number of initial training samples
        """
        self.parameter_bounds   = parameter_bounds
        self.n_initial_samples  = n_initial_samples
        self.n_features         = len(CONDITION_DESCRIPTORS)
        self.n_objectives       = N_OBJECTIVES
        self.obj_cfg            = OBJECTIVE_DESCRIPTORS

        # Data storage: X_train and one y array per objective
        self.X_train  = None
        self.Y_list   = [None] * self.n_objectives   # list of (n, 1) arrays

        # Scalers
        self.scaler_X = StandardScaler()
        self.scalers_Y = [StandardScaler() for _ in range(self.n_objectives)]

        # One GP per objective
        self.gp_models = [
            GaussianProcessRegressor(
                kernel=ChemicalSynthesisKernel(self.n_features).get_kernel(),
                n_restarts_optimizer=10,
                alpha=1e-6,
                normalize_y=False
            )
            for _ in range(self.n_objectives)
        ]

        # Acquisition function
        self.acquisition = IDEALAcquisition(self.obj_cfg)

        # History
        self.history = {
            'iteration':    [],
            'pareto_fronts': [],
            'hypervolume':  [],
        }
        # Best value per objective
        for obj in self.obj_cfg:
            self.history[f"best_{obj['column']}"] = []

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_data(self, data_path):
        """
        Load experiment data from CSV.

        Reads CONDITION_DESCRIPTORS as X and each OBJECTIVE_DESCRIPTORS
        column as a separate Y target.

        Args:
            data_path (str): path to CSV file
        """
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(data_path, encoding=enc)
                print(f"✓ 数据加载成功 (编码: {enc})")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"无法读取文件: {data_path}")

        self.X_train = df[CONDITION_DESCRIPTORS].values

        for k, obj in enumerate(self.obj_cfg):
            col = obj['column']
            self.Y_list[k] = df[col].values.reshape(-1, 1)

        print(f"✓ 样本数: {len(self.X_train)}")
        print(f"  合成条件描述符: {CONDITION_DESCRIPTORS}")
        print(f"  优化目标描述符:")
        for obj in self.obj_cfg:
            vals = df[obj['column']].values
            direction_cn = '↓ 最小化' if obj['direction'] == 'minimize' else '↑ 最大化'
            print(f"    {obj['column']} [{direction_cn}]: "
                  f"[{vals.min():.4f}, {vals.max():.4f}]")

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def fit_models(self):
        """
        Fit one GP model per objective on standardized data.
        """
        X_scaled = self.scaler_X.fit_transform(self.X_train)

        for k, obj in enumerate(self.obj_cfg):
            Y_scaled = self.scalers_Y[k].fit_transform(self.Y_list[k])
            self.gp_models[k].fit(X_scaled, Y_scaled.ravel())
            score = self.gp_models[k].score(X_scaled, Y_scaled.ravel())
            print(f"  ✓ GP [{obj['column']}] 训练完成  R² = {score:.4f}")

    # ------------------------------------------------------------------
    # Candidate generation & prediction
    # ------------------------------------------------------------------

    def generate_candidates(self, n_candidates=N_CANDIDATES):
        """
        Generate candidate points via Latin Hypercube Sampling.

        Args:
            n_candidates (int)

        Returns:
            candidates: (n_candidates, n_features)
        """
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=self.n_features, seed=42)
        samples  = sampler.random(n=n_candidates)

        candidates = np.zeros_like(samples)
        for i, name in enumerate(CONDITION_DESCRIPTORS):
            lo, hi = self.parameter_bounds[name]
            candidates[:, i] = samples[:, i] * (hi - lo) + lo

        return candidates

    def predict(self, X):
        """
        Predict objectives for candidate points.

        Args:
            X: (n_candidates, n_features)

        Returns:
            mean: (n_candidates, n_obj)
            std:  (n_candidates, n_obj)
        """
        X_scaled = self.scaler_X.transform(X)
        means, stds = [], []

        for k in range(self.n_objectives):
            m_scaled, s_scaled = self.gp_models[k].predict(X_scaled,
                                                             return_std=True)
            m = self.scalers_Y[k].inverse_transform(
                    m_scaled.reshape(-1, 1)).ravel()
            s = s_scaled * self.scalers_Y[k].scale_[0]
            means.append(m)
            stds.append(s)

        return np.column_stack(means), np.column_stack(stds)

    # ------------------------------------------------------------------
    # Experiment selection
    # ------------------------------------------------------------------

    def select_next_experiments(self, n_suggestions=N_SUGGESTIONS):
        """
        Select the next batch of experiments using the IDEAL acquisition.

        Sequential greedy selection: after each pick, remove the chosen
        point and its neighbours from the candidate pool (diversity).

        Args:
            n_suggestions (int)

        Returns:
            suggestions:           (n_suggestions, n_features)
            predicted_objectives:  (n_suggestions, n_obj)
            acquisition_values:    (n_suggestions,)
        """
        print(f"\n{'='*60}")
        print(f"  IDEAL — 选择下一批 {n_suggestions} 个实验点")
        print(f"{'='*60}")

        candidates = self.generate_candidates()
        print(f"  候选点数: {len(candidates)}")

        mean, std = self.predict(candidates)

        # Current Pareto front
        current_obj = np.column_stack(
            [self.Y_list[k].ravel() for k in range(self.n_objectives)]
        )
        pareto_mask    = self.acquisition.compute_pareto_front(current_obj)
        current_pareto = current_obj[pareto_mask]
        print(f"  当前 Pareto 前沿: {pareto_mask.sum()} 个点")

        suggestions, pred_objs, acq_vals = [], [], []

        for i in range(n_suggestions):
            acq = self.acquisition.compute_acquisition(
                mean, std, candidates, self.X_train, current_pareto
            )
            best = np.argmax(acq)
            suggestions.append(candidates[best])
            pred_objs.append(mean[best])
            acq_vals.append(acq[best])

            # Remove chosen point and neighbourhood
            dists = np.linalg.norm(candidates - candidates[best], axis=1)
            keep  = dists >= 0.1 * np.max(dists)
            candidates = candidates[keep]
            mean = mean[keep]
            std  = std[keep]

            print(f"  建议 {i+1}: 采集函数值 = {acq_vals[-1]:.4f}")

        return (np.array(suggestions),
                np.array(pred_objs),
                np.array(acq_vals))

    # ------------------------------------------------------------------
    # Hypervolume (generalized to N objectives)
    # ------------------------------------------------------------------

    def compute_hypervolume(self, pareto_front):
        """
        Compute hypervolume indicator.

        For 2 objectives: exact sweep algorithm.
        For 1 or 3+ objectives: uses pygmo if available, else Monte Carlo
        approximation as fallback.

        Args:
            pareto_front: (n_pareto, n_obj) array

        Returns:
            hv: float
        """
        if len(pareto_front) == 0:
            return 0.0

        ref = np.array([obj['ref'] for obj in self.obj_cfg], dtype=float)

        # Convert everything to a maximization problem
        normalized = np.zeros_like(pareto_front)
        for k, obj in enumerate(self.obj_cfg):
            if obj['direction'] == 'minimize':
                normalized[:, k] = ref[k] - pareto_front[:, k]
            else:
                normalized[:, k] = pareto_front[:, k] - ref[k]

        if self.n_objectives == 1:
            return float(np.max(normalized[:, 0]))

        elif self.n_objectives == 2:
            # Exact 2D sweep
            idx    = np.argsort(normalized[:, 0])
            front  = normalized[idx]
            hv = 0.0
            for i in range(len(front)):
                width = front[i, 0] if i == 0 else front[i, 0] - front[i-1, 0]
                hv   += width * front[i, 1]
            return float(hv)

        else:
            # Try pygmo (exact), fall back to Monte Carlo
            try:
                import pygmo as pg
                hv_obj = pg.hypervolume(normalized)
                return float(hv_obj.compute(np.zeros(self.n_objectives)))
            except ImportError:
                # Monte Carlo approximation
                n_mc  = 100_000
                lo    = normalized.min(axis=0)
                hi    = normalized.max(axis=0) * 1.1
                pts   = np.random.uniform(lo, hi, (n_mc, self.n_objectives))
                vol   = float(np.prod(hi - lo))
                dominated = np.any(
                    np.all(pts[:, None, :] <= normalized[None, :, :], axis=2),
                    axis=1
                )
                return vol * np.mean(dominated)

    # ------------------------------------------------------------------
    # Analysis & Update
    # ------------------------------------------------------------------

    def analyze_results(self, iteration):
        """
        Compute Pareto front and hypervolume, print summary, update history.

        Args:
            iteration (int)
        """
        current_obj = np.column_stack(
            [self.Y_list[k].ravel() for k in range(self.n_objectives)]
        )
        pareto_mask    = self.acquisition.compute_pareto_front(current_obj)
        pareto_front   = current_obj[pareto_mask]
        hv             = self.compute_hypervolume(pareto_front)

        self.history['iteration'].append(iteration)
        self.history['pareto_fronts'].append(pareto_front)
        self.history['hypervolume'].append(hv)

        for k, obj in enumerate(self.obj_cfg):
            col = obj['column']
            y   = self.Y_list[k].ravel()
            best = np.min(y) if obj['direction'] == 'minimize' else np.max(y)
            self.history[f"best_{col}"].append(best)

        print(f"\n{'='*60}")
        print(f"  第 {iteration} 轮 IDEAL 分析")
        print(f"{'='*60}")
        print(f"  Pareto 前沿点数:  {len(pareto_front)}")
        for k, obj in enumerate(self.obj_cfg):
            y    = self.Y_list[k].ravel()
            best = np.min(y) if obj['direction'] == 'minimize' else np.max(y)
            direction_cn = '最小' if obj['direction'] == 'minimize' else '最大'
            print(f"  {direction_cn} {obj['column']}: {best:.4f}")
        print(f"  超体积指标:       {hv:.4f}")
        print(f"{'='*60}")

    def update_with_new_data(self, X_new, Y_new_list):
        """
        Append new experimental results and retrain all GP models.

        Args:
            X_new (array): (n_new, n_features)
            Y_new_list (list of arrays): one (n_new,) array per objective,
                in the same order as OBJECTIVE_DESCRIPTORS
        """
        self.X_train = np.vstack([self.X_train, X_new])
        for k in range(self.n_objectives):
            y_new = np.array(Y_new_list[k]).reshape(-1, 1)
            self.Y_list[k] = np.vstack([self.Y_list[k], y_new])

        print(f"\n✓ 新增 {len(X_new)} 个样本，总样本数: {len(self.X_train)}")
        self.fit_models()

    # ------------------------------------------------------------------
    # Visualization (auto-adapts to objective count)
    # ------------------------------------------------------------------

    def visualize_results(self, save_path=OUTPUT_DIR):
        """
        Generate and save diagnostic plots.

        - Hypervolume evolution
        - Best-value evolution per objective
        - Pareto front (2D only; for 3+ objectives, shows pairwise scatter)
        """
        os.makedirs(save_path, exist_ok=True)

        n_rows = 2
        n_cols = max(2, self.n_objectives)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 5 * n_rows))
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        iters = self.history['iteration']

        # Row 0: best value per objective
        for k, obj in enumerate(self.obj_cfg):
            ax  = axes[0, k]
            col = obj['column']
            ax.plot(iters, self.history[f"best_{col}"],
                    marker='o', linewidth=2, markersize=6)
            direction_cn = '越小越好' if obj['direction'] == 'minimize' else '越大越好'
            ax.set_xlabel('迭代次数', fontsize=11)
            ax.set_ylabel(f'{col}', fontsize=11)
            ax.set_title(f'{col} 优化进度 ({direction_cn})', fontsize=12)
            ax.grid(True, alpha=0.3)

        # Row 0 extra: hypervolume
        ax_hv = axes[0, n_cols - 1] if self.n_objectives < n_cols else axes[1, n_cols - 1]
        ax_hv.plot(iters, self.history['hypervolume'],
                   marker='s', linewidth=2, markersize=6, color='purple')
        ax_hv.set_xlabel('迭代次数', fontsize=11)
        ax_hv.set_ylabel('超体积', fontsize=11)
        ax_hv.set_title('超体积指标演化', fontsize=12)
        ax_hv.grid(True, alpha=0.3)

        # Row 1: Pareto front visualization
        current_obj = np.column_stack(
            [self.Y_list[k].ravel() for k in range(self.n_objectives)]
        )
        pareto_mask = self.acquisition.compute_pareto_front(current_obj)

        if self.n_objectives == 1:
            ax = axes[1, 0]
            ax.scatter(range(len(current_obj)),
                       current_obj[:, 0], c='steelblue', s=40, alpha=0.5)
            best_idx = (np.argmin(current_obj[:, 0])
                        if self.obj_cfg[0]['direction'] == 'minimize'
                        else np.argmax(current_obj[:, 0]))
            ax.axhline(current_obj[best_idx, 0], color='red',
                       linestyle='--', label='最优值')
            ax.set_xlabel('样本序号'); ax.set_ylabel(self.obj_cfg[0]['column'])
            ax.set_title('单目标优化结果'); ax.legend(); ax.grid(True, alpha=0.3)

        elif self.n_objectives == 2:
            ax = axes[1, 0]
            ax.scatter(current_obj[~pareto_mask, 0],
                       current_obj[~pareto_mask, 1],
                       c='lightgray', s=40, alpha=0.5, label='其他点')
            ax.scatter(current_obj[pareto_mask, 0],
                       current_obj[pareto_mask, 1],
                       c='red', s=150, alpha=0.8, edgecolors='black',
                       linewidths=1.5, label='Pareto 前沿')
            x_lbl = (f"{self.obj_cfg[0]['column']} "
                     f"({'↓' if self.obj_cfg[0]['direction']=='minimize' else '↑'})")
            y_lbl = (f"{self.obj_cfg[1]['column']} "
                     f"({'↓' if self.obj_cfg[1]['direction']=='minimize' else '↑'})")
            ax.set_xlabel(x_lbl, fontsize=11)
            ax.set_ylabel(y_lbl, fontsize=11)
            ax.set_title('当前 Pareto 前沿', fontsize=12)
            ax.legend(); ax.grid(True, alpha=0.3)

        else:
            # N ≥ 3: pairwise scatter (first vs second, etc.)
            pairs = [(0, 1), (0, 2), (1, 2)][:n_cols]
            for col_idx, (i, j) in enumerate(pairs):
                if col_idx >= n_cols:
                    break
                ax = axes[1, col_idx]
                ax.scatter(current_obj[~pareto_mask, i],
                           current_obj[~pareto_mask, j],
                           c='lightgray', s=30, alpha=0.4)
                ax.scatter(current_obj[pareto_mask, i],
                           current_obj[pareto_mask, j],
                           c='red', s=100, alpha=0.8, edgecolors='black')
                ax.set_xlabel(self.obj_cfg[i]['column'])
                ax.set_ylabel(self.obj_cfg[j]['column'])
                ax.set_title(f'Pareto 投影: '
                             f'{self.obj_cfg[i]["column"]} vs '
                             f'{self.obj_cfg[j]["column"]}')
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'IDEAL 优化结果 ({self.n_objectives} 目标)',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()

        out_file = os.path.join(save_path, 'IDEAL_results.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ 可视化结果已保存: {out_file}")
        plt.close()
        return out_file

    # ------------------------------------------------------------------
    # Convenience: one-call iteration
    # ------------------------------------------------------------------

    def run_iteration(self, iteration=0, n_suggestions=N_SUGGESTIONS):
        """
        Execute one full IDEAL iteration:
          1. Train GP models
          2. Select next experiments
          3. Analyse current results
          4. Return suggestions as a DataFrame

        Args:
            iteration    (int): current round number
            n_suggestions(int): number of experiments to suggest

        Returns:
            suggestions_df (pd.DataFrame)
        """
        print(f"\n{'#'*70}")
        print(f"  IDEAL — 第 {iteration} 轮主动学习迭代")
        print(f"  模式: {self.n_objectives} 目标优化")
        print(f"{'#'*70}")

        self.fit_models()

        suggestions, pred_objs, acq_vals = self.select_next_experiments(
            n_suggestions
        )

        # Build output DataFrame
        df = pd.DataFrame(suggestions, columns=CONDITION_DESCRIPTORS)
        for k, obj in enumerate(self.obj_cfg):
            df[f"Predicted_{obj['column']}"] = pred_objs[:, k]
        df['Acquisition_Value'] = acq_vals

        self.analyze_results(iteration)

        return df


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  IDEAL: Interpretable Dynamic Experimental Active Learning")
    print(f"  模式: {N_OBJECTIVES} 目标优化")
    print(f"  合成条件描述符: {CONDITION_DESCRIPTORS}")
    print(f"  优化目标描述符: {[o['column'] for o in OBJECTIVE_DESCRIPTORS]}")
    print("="*70)

    # 1. 参数空间（与 CONDITION_DESCRIPTORS 对应）
    parameter_bounds = {
        'Molar Ratio':   (250, 750),
        'H2SO4 Volume':  (1,   4),
        'Temperature':   (150, 200),
        'Time':          (8,   12),
    }

    # 2. 初始化系统
    system = IDEALSystem(
        parameter_bounds=parameter_bounds,
        n_initial_samples=N_INITIAL_SAMPLES
    )

    # 3. 加载数据
    print(f"\n加载数据: {DATA_PATH}")
    system.load_data(DATA_PATH)

    # 4. 第一轮迭代
    suggestions_df = system.run_iteration(iteration=0,
                                          n_suggestions=N_SUGGESTIONS)

    # 5. 打印建议
    print("\n" + "="*70)
    print("  建议的实验参数:")
    print("="*70)
    print(suggestions_df.to_string(index=False))

    # 6. 保存建议
    out_csv = os.path.join(OUTPUT_DIR, 'IDEAL_suggestions_iteration_0.csv')
    suggestions_df.to_csv(out_csv, index=False)
    print(f"\n✓ 建议已保存: {out_csv}")

    # 7. 可视化
    system.visualize_results(save_path=OUTPUT_DIR)

    print("\n" + "="*70)
    print("  第 0 轮迭代完成!")
    print(f"  所有结果保存至: {OUTPUT_DIR}")
    print("="*70)
    print("\n下一步操作:")
    print("  1. 按建议参数进行实验，获取新测量值")
    print("  2. 调用 system.update_with_new_data(X_new, Y_new_list) 更新数据")
    print("     Y_new_list 格式: 按 OBJECTIVE_DESCRIPTORS 顺序排列的列表")
    print("     例 (双目标): Y_new_list = [fwhm_array, qy_array]")
    print("  3. 继续调用 system.run_iteration(iteration=1) 进行下一轮")

    return system, suggestions_df


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ 找不到数据文件: {DATA_PATH}")
        print("   请修改文件顶部用户配置区的 DATA_PATH")
    else:
        system, suggestions = main()
