# -*- coding: utf-8 -*-
"""
Reviewer 2 - Question 1
Candidate-pool-size sensitivity test for TOP-4 active-learning batch selection.

This script tests whether 2000 Latin-hypercube candidate points are sufficient
for stable TOP-4 batch selection in the four-dimensional synthesis space.

Important:
The original acquisition function normalizes EI, uncertainty, and diversity
inside each candidate pool. Therefore raw acquisition values from different
pool sizes should not be compared directly. For a fair sensitivity analysis,
this script uses the original pool-wise acquisition function for selection,
then evaluates the selected TOP-4 batch using a fixed 20000-point reference
pool as a high-resolution approximation of the candidate space.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import qmc


warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parents[1]
MODEL_PATH = ROOT / "multi_objective_active_learning.py"

FEATURE_COLUMNS = ["Molar Ratio", "H2SO4 Volume", "Temperature", "Time"]
PARAMETER_BOUNDS = {
    "Molar Ratio": (250, 750),
    "H2SO4 Volume": (1, 4),
    "Temperature": (150, 200),
    "Time": (8, 12),
}
AL_STATES = [
    ("cycle_1_start_n16", "cpaperdatan16.csv"),
    ("cycle_2_start_n20", "cpaperdatan16_4.csv"),
    ("cycle_3_start_n24", "cpaperdatan16_8.csv"),
    ("cycle_4_start_n28", "cpaperdatan16_12.csv"),
    ("cycle_5_start_n32", "cpaperdatan16_16.csv"),
]
POOL_SIZES = [250, 500, 1000, 2000, 5000, 10000, 20000]
SEEDS = [11, 22, 33]
REFERENCE_POOL_SIZE = 20000
REFERENCE_SEED = 20260531


def safe_to_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUT_DIR / filename
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        updated = path.with_name(f"{path.stem}_updated{path.suffix}")
        df.to_csv(updated, index=False, encoding="utf-8-sig")
        return updated


def safe_write_text(filename: str, text: str) -> Path:
    path = OUT_DIR / filename
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        updated = path.with_name(f"{path.stem}_updated{path.suffix}")
        updated.write_text(text, encoding="utf-8")
        return updated


def load_model_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("moal", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latin_hypercube_candidates(n_candidates: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(FEATURE_COLUMNS), seed=seed)
    unit = sampler.random(n=n_candidates)
    candidates = np.zeros_like(unit)
    for idx, name in enumerate(FEATURE_COLUMNS):
        lower, upper = PARAMETER_BOUNDS[name]
        candidates[:, idx] = unit[:, idx] * (upper - lower) + lower
    return candidates


def fit_al_system(module, csv_name: str):
    df = pd.read_csv(ROOT / csv_name)
    system = module.BiObjectiveActiveLearning(PARAMETER_BOUNDS, n_initial_samples=16)
    system.X_train = df[FEATURE_COLUMNS].values
    system.y_fwhm = df["FWHM"].values.reshape(-1, 1)
    system.y_qy = df["QY"].values.reshape(-1, 1)
    system.fit_models()
    return system


def current_pareto(system) -> np.ndarray:
    objectives = np.column_stack([system.y_fwhm.ravel(), system.y_qy.ravel()])
    mask = system.acquisition.compute_pareto_front(objectives)
    return objectives[mask]


def select_top4_by_original_algorithm(system, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mimic select_next_experiments() without printing and with supplied candidates."""
    candidates = candidates.copy()
    mean, std = system.predict(candidates)
    pareto = current_pareto(system)

    selected = []
    selected_mean = []
    selected_acq = []
    for _ in range(4):
        acq = system.acquisition.compute_acquisition(
            mean, std, candidates, system.X_train, pareto
        )
        best_idx = int(np.argmax(acq))
        selected.append(candidates[best_idx])
        selected_mean.append(mean[best_idx])
        selected_acq.append(float(acq[best_idx]))

        distances = np.linalg.norm(candidates - candidates[best_idx], axis=1)
        remove_mask = distances < 0.1 * float(np.max(distances))
        candidates = candidates[~remove_mask]
        mean = mean[~remove_mask]
        std = std[~remove_mask]
        if len(candidates) == 0:
            break

    return np.asarray(selected), np.asarray(selected_mean), np.asarray(selected_acq)


def fixed_reference_context(system, reference_candidates: np.ndarray) -> dict:
    """Build fixed normalization constants from a 20000-point reference pool."""
    ref_mean, ref_std = system.predict(reference_candidates)
    pareto = current_pareto(system)

    ei = system.acquisition.expected_improvement_2d(ref_mean, ref_std, pareto)
    uncertainty = system.acquisition.uncertainty_score(ref_std)
    distances = cdist(reference_candidates, system.X_train)
    diversity = np.min(distances, axis=1)

    n_observed = len(system.X_train)
    exploration_weight = np.exp(-n_observed / 20)
    return {
        "pareto": pareto,
        "ei_max": max(float(np.max(ei)), 1e-12),
        "uncertainty_max": max(float(np.max(uncertainty)), 1e-12),
        "diversity_max": max(float(np.max(diversity)), 1e-12),
        "w1": 1 - exploration_weight,
        "w2": 0.5 * exploration_weight,
        "w3": 0.3 * exploration_weight,
    }


def fixed_reference_score(system, points: np.ndarray, context: dict) -> np.ndarray:
    """Evaluate arbitrary points using fixed reference-pool normalization."""
    mean, std = system.predict(points)
    ei = system.acquisition.expected_improvement_2d(mean, std, context["pareto"])
    uncertainty = system.acquisition.uncertainty_score(std)
    diversity = np.min(cdist(points, system.X_train), axis=1)

    ei_norm = ei / context["ei_max"]
    uncertainty_norm = uncertainty / context["uncertainty_max"]
    diversity_norm = diversity / context["diversity_max"]

    return (
        context["w1"] * ei_norm
        + context["w2"] * uncertainty_norm
        + context["w3"] * diversity_norm
    )


def standardized_nearest_distance(points: np.ndarray, reference_points: np.ndarray) -> float:
    """Distance in [0, 1]-scaled synthesis space to nearest reference TOP-4 point."""
    lowers = np.array([PARAMETER_BOUNDS[name][0] for name in FEATURE_COLUMNS])
    uppers = np.array([PARAMETER_BOUNDS[name][1] for name in FEATURE_COLUMNS])
    span = uppers - lowers
    scaled_points = (points - lowers) / span
    scaled_ref = (reference_points - lowers) / span
    dists = cdist(scaled_points, scaled_ref)
    return float(np.mean(np.min(dists, axis=1)))


def main():
    module = load_model_module()
    detail_records = []
    reference_records = []

    for state_label, csv_name in AL_STATES:
        print(f"\n=== {state_label} ({csv_name}) ===")
        system = fit_al_system(module, csv_name)

        reference_candidates = latin_hypercube_candidates(
            REFERENCE_POOL_SIZE, REFERENCE_SEED
        )
        context = fixed_reference_context(system, reference_candidates)
        ref_start = time.perf_counter()
        ref_points, ref_mean, ref_pool_acq = select_top4_by_original_algorithm(
            system, reference_candidates
        )
        ref_seconds = time.perf_counter() - ref_start
        ref_fixed_score = fixed_reference_score(system, ref_points, context)
        ref_mean_score = float(np.mean(ref_fixed_score))

        reference_records.append(
            {
                "state": state_label,
                "training_n": len(system.X_train),
                "reference_pool_size": REFERENCE_POOL_SIZE,
                "reference_mean_fixed_score": ref_mean_score,
                "reference_runtime_seconds": ref_seconds,
                "reference_mean_pred_FWHM": float(np.mean(ref_mean[:, 0])),
                "reference_mean_pred_PLQY_percent": float(np.mean(ref_mean[:, 1]) * 100),
            }
        )

        for pool_size in POOL_SIZES:
            for seed in SEEDS:
                candidates = latin_hypercube_candidates(pool_size, seed)
                start = time.perf_counter()
                selected, selected_mean, selected_pool_acq = select_top4_by_original_algorithm(
                    system, candidates
                )
                runtime = time.perf_counter() - start
                fixed_score = fixed_reference_score(system, selected, context)

                detail_records.append(
                    {
                        "state": state_label,
                        "training_n": len(system.X_train),
                        "pool_size": pool_size,
                        "seed": seed,
                        "runtime_seconds": runtime,
                        "mean_poolwise_acquisition": float(np.mean(selected_pool_acq)),
                        "mean_fixed_reference_score": float(np.mean(fixed_score)),
                        "relative_to_20000_reference_percent": 100
                        * float(np.mean(fixed_score))
                        / ref_mean_score,
                        "nearest_distance_to_reference_top4_scaled": standardized_nearest_distance(
                            selected, ref_points
                        ),
                        "mean_pred_FWHM": float(np.mean(selected_mean[:, 0])),
                        "mean_pred_PLQY_percent": float(np.mean(selected_mean[:, 1]) * 100),
                    }
                )

        print(f"reference mean fixed score: {ref_mean_score:.4f}")

    details = pd.DataFrame(detail_records)
    references = pd.DataFrame(reference_records)
    summary = (
        details.groupby("pool_size")
        .agg(
            mean_relative_efficiency_percent=(
                "relative_to_20000_reference_percent",
                "mean",
            ),
            sd_relative_efficiency_percent=(
                "relative_to_20000_reference_percent",
                "std",
            ),
            mean_distance_to_reference_top4=(
                "nearest_distance_to_reference_top4_scaled",
                "mean",
            ),
            sd_distance_to_reference_top4=(
                "nearest_distance_to_reference_top4_scaled",
                "std",
            ),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            sd_runtime_seconds=("runtime_seconds", "std"),
        )
        .reset_index()
    )

    safe_to_csv(details, "candidate_pool_size_sensitivity_details.csv")
    safe_to_csv(references, "candidate_pool_size_reference_top4.csv")
    safe_to_csv(summary, "candidate_pool_size_sensitivity_summary.csv")

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=220)

    x = summary["pool_size"].to_numpy()
    y = summary["mean_relative_efficiency_percent"].to_numpy()
    yerr = summary["sd_relative_efficiency_percent"].to_numpy()
    axes[0].plot(x, y, marker="o", linewidth=2.2, color="#1f77b4")
    axes[0].fill_between(x, y - yerr, y + yerr, color="#1f77b4", alpha=0.15)
    axes[0].axvline(2000, color="#d62728", linestyle="--", linewidth=1.6)
    axes[0].text(2150, min(108, max(y) + 2), "2000", color="#d62728", fontsize=9)
    axes[0].set_xscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(v) for v in x], rotation=35)
    axes[0].set_xlabel("Candidate pool size")
    axes[0].set_ylabel("Relative TOP-4 score vs. 20000-point reference (%)")
    axes[0].set_title("Search benefit increases with candidate-pool size")
    axes[0].grid(True, linestyle=":", alpha=0.55)

    rt = summary["mean_runtime_seconds"].to_numpy()
    rt_err = summary["sd_runtime_seconds"].to_numpy()
    axes[1].plot(x, rt, marker="s", linewidth=2.2, color="#2ca02c")
    axes[1].fill_between(x, rt - rt_err, rt + rt_err, color="#2ca02c", alpha=0.15)
    axes[1].axvline(2000, color="#d62728", linestyle="--", linewidth=1.6)
    axes[1].text(2150, max(0, max(rt) * 0.08), "2000", color="#d62728", fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(v) for v in x], rotation=35)
    axes[1].set_xlabel("Candidate pool size")
    axes[1].set_ylabel("Mean TOP-4 selection runtime (s)")
    axes[1].set_title("Computation cost increases with pool size")
    axes[1].grid(True, linestyle=":", alpha=0.55)

    fig.suptitle(
        "Candidate-pool-size sensitivity for LHS-based TOP-4 active-learning selection",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Selection used the original pool-wise acquisition function; evaluation used fixed normalization from a 20000-point reference pool.",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    png_path = OUT_DIR / "candidate_pool_size_sensitivity.png"
    fig.savefig(png_path, bbox_inches="tight")

    row_2000 = summary[summary["pool_size"] == 2000].iloc[0]
    row_5000 = summary[summary["pool_size"] == 5000].iloc[0]
    row_10000 = summary[summary["pool_size"] == 10000].iloc[0]
    row_20000 = summary[summary["pool_size"] == 20000].iloc[0]

    explanation = [
        "第一问补充实验说明：候选池规模敏感性测试",
        "",
        "一、实验目的",
        "审稿人询问每轮主动学习中模型预测了多少个候选点，并质疑为什么每轮选择 4 个点。本实验补充回答其中的候选池规模问题：为什么使用 2000 个 LHS 候选点，而不是更大的 5000、10000 或 20000 个。",
        "",
        "二、实验逻辑",
        "候选池规模不是模型物理参数，而是对连续四维合成空间进行离散化搜索的计算设置。",
        "原程序的 acquisition function 会在每个候选池内部对 EI、不确定度和多样性分别归一化，因此不同候选池的原始 acquisition score 不能直接比较。",
        "为避免不公平比较，本实验采用如下流程：",
        "1. 对每个 AL 起始状态分别训练 GP 模型，包括 n=16、20、24、28、32 五个状态。",
        "2. 对每个状态构建一个 20000 点 LHS 候选池，作为高分辨率参考候选池。",
        "3. 对 250、500、1000、2000、5000、10000、20000 个候选点分别运行原始 TOP-4 选点算法。",
        "4. 用固定的 20000 点参考池归一化标准，重新评价不同候选池选出 TOP-4 的综合得分。",
        "5. 同时记录不同候选池规模下的 TOP-4 选择耗时。",
        "",
        "三、主要结果",
        f"当候选池规模达到 2000 时，TOP-4 批次的相对参考得分为 {row_2000['mean_relative_efficiency_percent']:.2f}% ± {row_2000['sd_relative_efficiency_percent']:.2f}%。",
        f"进一步增加到 5000、10000 和 20000 时，相对参考得分别为 {row_5000['mean_relative_efficiency_percent']:.2f}%、{row_10000['mean_relative_efficiency_percent']:.2f}% 和 {row_20000['mean_relative_efficiency_percent']:.2f}%。",
        f"与此同时，平均选点耗时从 2000 点的 {row_2000['mean_runtime_seconds']:.3f} s 增加到 10000 点的 {row_10000['mean_runtime_seconds']:.3f} s 和 20000 点的 {row_20000['mean_runtime_seconds']:.3f} s。",
        "",
        "四、结果解释",
        "这个结果不能证明 2000 是数学意义上的最优候选池规模。相反，随着候选池从 2000 增加到 5000、10000 和 20000，参考归一化 TOP-4 得分仍然继续提高，说明更大的候选池确实可以提供更充分的搜索。",
        "因此，修稿时不能写“2000 个候选点最好”或“继续增加候选池不会改变结果”。更稳妥的说法是：2000 个候选点比 250、500 和 1000 个候选点更稳定，同时计算时间明显低于 10000 或 20000 个候选点，所以它是原程序中计算效率和搜索质量之间的 practical compromise。",
        "如果希望给审稿人一个更强的候选池规模依据，根据本实验，5000 个候选点比 2000 个更接近高分辨率参考候选池，同时计算成本仍低于 10000 和 20000。因此，修订版代码也可以考虑把候选池规模从 2000 提高到 5000；但如果保持 2000，则只能解释为效率折中，而不能称为最优。",
        "",
        "五、可写入 SI 或回复中的表述",
        "Candidate-pool-size sensitivity tests were performed using 250-20000 LHS candidates. Increasing the pool size improved the reference-normalized TOP-4 acquisition score but also increased the candidate-ranking time. The 2000-candidate pool provided a practical computational compromise compared with smaller pools, although larger pools such as 5000 candidates gave higher reference-normalized scores. Therefore, the candidate-pool size should be described as a computational setting balancing search quality and efficiency, rather than as a physically optimized model parameter.",
        "",
        "六、输出文件",
        "1. candidate_pool_size_sensitivity_experiment.py：可提交或上传 GitHub 的候选池规模敏感性实验代码。",
        "2. candidate_pool_size_sensitivity.png：候选池规模收益图和计算耗时图。",
        "3. candidate_pool_size_sensitivity_summary.csv：不同候选池规模的汇总指标。",
        "4. candidate_pool_size_sensitivity_details.csv：所有 AL 状态、候选池规模和随机种子的详细结果。",
        "5. candidate_pool_size_reference_top4.csv：20000 点参考候选池的基准 TOP-4 信息。",
    ]
    safe_write_text("第一问_候选池规模实验说明_代码解释_结果解释.txt", "\n".join(explanation))

    print("\nSummary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved figure: {png_path}")


if __name__ == "__main__":
    main()
