# -*- coding: utf-8 -*-
"""
Reviewer 2 - Question 7
SHAP-style model interpretability analysis for PLQY and FWHM.

The external "shap" package is not required. Because the model uses only four
synthesis descriptors, exact model-agnostic Shapley values are calculated by
enumerating all feature subsets. Missing features are marginalized with the
empirical 36-sample background distribution.

This analysis is intended as supplementary model interpretation. It should be
reported together with the Pearson/Spearman correlation table, because pairwise
linear correlation and model-based feature contribution answer different
questions.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


OUT_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = ["Time", "Temperature", "Molar Ratio", "H2SO4 Volume"]
FEATURE_LABELS = {
    "Time": "Time",
    "Temperature": "Temperature",
    "Molar Ratio": "Molar ratio",
    "H2SO4 Volume": "H2SO4 amount",
}
TARGETS = [
    ("PLQY", "PLQY_percent", "percentage points"),
    ("FWHM", "FWHM_nm", "nm"),
]


def build_final_dataset() -> pd.DataFrame:
    rows = [
        (1, 10.00, 150.0, 400.0, 3.00, 21.83, 62),
        (2, 10.00, 150.0, 400.0, 4.00, 23.19, 64),
        (3, 10.00, 150.0, 600.0, 3.00, 15.87, 69),
        (4, 10.00, 150.0, 600.0, 4.00, 9.50, 74),
        (5, 10.00, 170.0, 400.0, 3.00, 10.91, 68),
        (6, 10.00, 170.0, 400.0, 4.00, 14.86, 58),
        (7, 10.00, 170.0, 600.0, 3.00, 15.76, 72),
        (8, 10.00, 170.0, 600.0, 4.00, 13.73, 85),
        (9, 12.00, 150.0, 400.0, 3.00, 14.44, 65),
        (10, 12.00, 150.0, 400.0, 4.00, 26.37, 76),
        (11, 12.00, 150.0, 600.0, 3.00, 9.73, 82),
        (12, 12.00, 150.0, 600.0, 4.00, 17.75, 87),
        (13, 12.00, 170.0, 400.0, 3.00, 9.44, 67),
        (14, 12.00, 170.0, 400.0, 4.00, 14.37, 64),
        (15, 12.00, 170.0, 600.0, 3.00, 16.60, 75),
        (16, 12.00, 170.0, 600.0, 4.00, 15.23, 71),
        (17, 10.91, 152.3, 396.1, 3.75, 24.19, 57),
        (18, 9.02, 150.3, 359.9, 3.92, 18.99, 58),
        (19, 8.40, 199.5, 250.2, 1.69, 15.32, 64),
        (20, 9.15, 196.4, 749.3, 2.53, 17.18, 55),
        (21, 10.68, 153.9, 350.2, 3.97, 27.44, 45),
        (22, 10.41, 155.3, 436.5, 3.84, 22.79, 44),
        (23, 10.78, 153.2, 473.9, 3.99, 20.32, 53),
        (24, 10.93, 151.8, 305.1, 3.54, 21.63, 48),
        (25, 11.78, 154.8, 346.1, 3.77, 26.32, 38),
        (26, 10.68, 159.1, 268.4, 3.92, 25.91, 52),
        (27, 8.79, 155.1, 415.1, 2.29, 21.23, 40),
        (28, 8.29, 152.6, 466.6, 2.69, 20.59, 51),
        (29, 9.75, 154.1, 272.1, 1.89, 32.05, 29),
        (30, 10.49, 155.4, 368.1, 1.51, 36.63, 31),
        (31, 11.84, 156.1, 325.6, 1.82, 19.31, 47),
        (32, 11.85, 155.9, 410.7, 1.59, 18.39, 48),
        (33, 10.01, 153.3, 293.9, 1.37, 23.85, 41),
        (34, 9.67, 155.4, 374.5, 1.12, 33.58, 61),
        (35, 10.04, 162.3, 476.6, 1.02, 19.45, 36),
        (36, 8.36, 155.7, 448.7, 1.03, 28.33, 36),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Serial_Number",
            "Time",
            "Temperature",
            "Molar Ratio",
            "H2SO4 Volume",
            "PLQY_percent",
            "FWHM_nm",
        ],
    )
    df["AL_cycle"] = np.where(df["Serial_Number"] <= 16, 0, ((df["Serial_Number"] - 17) // 4) + 1)
    return df


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def subsets_without(feature_idx: int, n_features: int):
    others = [idx for idx in range(n_features) if idx != feature_idx]
    for subset_size in range(n_features):
        for subset in combinations(others, subset_size):
            yield tuple(subset)


def marginalized_prediction(model: RandomForestRegressor, x: np.ndarray, background: np.ndarray, known_subset: tuple[int, ...]) -> float:
    imputed = background.copy()
    if known_subset:
        imputed[:, list(known_subset)] = x[list(known_subset)]
    return float(model.predict(imputed).mean())


def exact_model_agnostic_shap(model: RandomForestRegressor, X: np.ndarray, background: np.ndarray) -> np.ndarray:
    n_samples, n_features = X.shape
    shap_values = np.zeros((n_samples, n_features), dtype=float)
    normalizer = factorial(n_features)

    for row_idx, x in enumerate(X):
        coalition_prediction: dict[tuple[int, ...], float] = {}
        for subset_size in range(n_features + 1):
            for subset in combinations(range(n_features), subset_size):
                coalition_prediction[tuple(subset)] = marginalized_prediction(model, x, background, tuple(subset))

        for feature_idx in range(n_features):
            contribution = 0.0
            for subset in subsets_without(feature_idx, n_features):
                subset_with_feature = tuple(sorted(subset + (feature_idx,)))
                weight = factorial(len(subset)) * factorial(n_features - len(subset) - 1) / normalizer
                pred_with = coalition_prediction[subset_with_feature]
                pred_without = coalition_prediction[subset]
                contribution += weight * (pred_with - pred_without)
            shap_values[row_idx, feature_idx] = contribution
    return shap_values


def train_model(df: pd.DataFrame, target_col: str) -> tuple[RandomForestRegressor, pd.DataFrame]:
    X = df[FEATURE_COLUMNS].values
    y = df[target_col].values
    model = RandomForestRegressor(n_estimators=500, random_state=42, max_depth=15)
    model.fit(X, y)
    pred = model.predict(X)
    metrics = pd.DataFrame(
        [
            {
                "target": target_col,
                "n": len(df),
                "R2": r2_score(y, pred),
                "MAE": mean_absolute_error(y, pred),
            }
        ]
    )
    return model, metrics


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name, target_col, unit in TARGETS:
        for feature in FEATURE_COLUMNS:
            rows.append(
                {
                    "target": target_name,
                    "feature": feature,
                    "pearson_r": df[[feature, target_col]].corr(method="pearson").iloc[0, 1],
                    "spearman_r": df[[feature, target_col]].corr(method="spearman").iloc[0, 1],
                    "target_unit": unit,
                }
            )
    corr = pd.DataFrame(rows)
    corr["abs_pearson_r"] = corr["pearson_r"].abs()
    corr["abs_spearman_r"] = corr["spearman_r"].abs()
    return corr.sort_values(["target", "abs_pearson_r"], ascending=[True, False]).reset_index(drop=True)


def make_importance_plot(summary: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=300)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    for ax, (target_name, _, unit) in zip(axes, TARGETS):
        sub = summary[summary["target"] == target_name].sort_values("mean_abs_shap", ascending=True)
        ax.barh(sub["feature_label"], sub["mean_abs_shap"], color=colors[: len(sub)])
        ax.set_xlabel(f"Mean |SHAP value| ({unit})")
        ax.set_title(target_name, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.35)

    fig.suptitle("Model-based descriptor importance from exact SHAP analysis", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "SHAP_mean_abs_importance.png", bbox_inches="tight")


def make_summary_dot_plot(values_long: pd.DataFrame, df: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=300)

    for ax, (target_name, _, unit) in zip(axes, TARGETS):
        sub = values_long[values_long["target"] == target_name].copy()
        order = (
            sub.groupby("feature")["shap_value"]
            .apply(lambda x: np.mean(np.abs(x)))
            .sort_values(ascending=True)
            .index.tolist()
        )
        y_positions = {feature: idx for idx, feature in enumerate(order)}
        for feature in order:
            feature_values = df[feature].values
            denom = feature_values.max() - feature_values.min()
            normalized = (feature_values - feature_values.min()) / denom if denom > 0 else np.zeros_like(feature_values)
            rows = sub[sub["feature"] == feature]
            jitter = np.linspace(-0.18, 0.18, len(rows))
            ax.scatter(
                rows["shap_value"],
                y_positions[feature] + jitter,
                c=normalized,
                cmap="viridis",
                s=30,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.88,
            )
        ax.axvline(0, color="#333333", lw=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([FEATURE_LABELS[f] for f in order])
        ax.set_xlabel(f"SHAP value ({unit})")
        ax.set_title(target_name, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.35)

    fig.suptitle("SHAP value distribution; color indicates relative descriptor value", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "SHAP_summary_dot_plot.png", bbox_inches="tight")


def write_explanation(model_metrics: pd.DataFrame, shap_summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    plqy_rank = shap_summary[shap_summary["target"] == "PLQY"].sort_values("mean_abs_shap", ascending=False)
    fwhm_rank = shap_summary[shap_summary["target"] == "FWHM"].sort_values("mean_abs_shap", ascending=False)
    plqy_corr = corr[corr["target"] == "PLQY"].sort_values("abs_pearson_r", ascending=False)
    fwhm_corr = corr[corr["target"] == "FWHM"].sort_values("abs_pearson_r", ascending=False)

    lines = [
        "第七问 SHAP 补充分析说明",
        "",
        "一、为什么补 SHAP",
        "审稿人指出 Fig. 3f 的 Pearson 相关性热图与正文表述不一致：热图中 PLQY 和 FWHM 的最高绝对 Pearson 相关系数出现在摩尔比描述符上，而正文写成温度和时间最强。",
        "Pearson 相关性只描述单变量线性关系。为了补充模型层面的解释，我们增加 SHAP 分析，用来评估训练模型中各合成描述符对 PLQY 和 FWHM 预测的平均贡献。",
        "",
        "二、方法",
        "由于外部 shap 包没有安装，本代码没有调用第三方 SHAP 库，而是直接计算精确的模型无关 Shapley values。",
        "本研究只有四个描述符，因此可以枚举全部特征子集；缺失特征通过 36 个样品的经验背景分布进行边际化。",
        "代理模型为 RandomForestRegressor(n_estimators=500, random_state=42, max_depth=15)，分别拟合 PLQY 和 FWHM。",
        "该分析是补充性的模型解释，不是因果证明。",
        "",
        "三、模型回拟合指标",
        model_metrics.round(4).to_string(index=False),
        "",
        "四、SHAP mean absolute importance 排名",
        "PLQY:",
        plqy_rank[["feature", "mean_abs_shap", "relative_importance_percent"]].round(4).to_string(index=False),
        "",
        "FWHM:",
        fwhm_rank[["feature", "mean_abs_shap", "relative_importance_percent"]].round(4).to_string(index=False),
        "",
        "五、Pearson 相关性最高项",
        f"PLQY 的最高绝对 Pearson 相关描述符为 {plqy_corr.iloc[0]['feature']}，|r| = {plqy_corr.iloc[0]['abs_pearson_r']:.4f}。",
        f"FWHM 的最高绝对 Pearson 相关描述符为 {fwhm_corr.iloc[0]['feature']}，|r| = {fwhm_corr.iloc[0]['abs_pearson_r']:.4f}。",
        "",
        "六、建议写法",
        "正文中应承认 Pearson 热图显示摩尔比具有最高的绝对线性相关性。",
        "随后说明 Pearson correlation 只反映 pairwise linear association，而 SHAP 反映训练模型中的平均预测贡献。",
        "如果 SHAP 排名与 Pearson 不完全相同，可以解释为非线性关系和参数交互导致；不能再写成温度和时间在 Pearson 热图中最强。",
        "",
        "七、输出文件",
        "1. shap_interpretability_analysis.py：分析代码。",
        "2. shap_dataset_36.csv：36 个样品数据。",
        "3. shap_model_metrics.csv：两个随机森林模型的回拟合指标。",
        "4. shap_importance_summary.csv：PLQY 和 FWHM 的 mean absolute SHAP 排名。",
        "5. shap_values_PLQY.csv 和 shap_values_FWHM.csv：逐样品 SHAP 值。",
        "6. pearson_spearman_correlation_summary.csv：Pearson/Spearman 相关性对照表。",
        "7. SHAP_mean_abs_importance.png：可放入 SI 的 SHAP 条形图。",
        "8. SHAP_summary_dot_plot.png：SHAP 分布图。",
    ]
    (OUT_DIR / "第七问_SHAP实验说明_代码解释_结果解释.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_final_dataset()
    save_csv(df, "shap_dataset_36.csv")

    X = df[FEATURE_COLUMNS].values
    background = X.copy()
    all_metrics = []
    all_summary = []
    all_values_long = []

    for target_name, target_col, unit in TARGETS:
        model, metrics = train_model(df, target_col)
        metrics["target_name"] = target_name
        metrics["unit"] = unit
        all_metrics.append(metrics)

        shap_values = exact_model_agnostic_shap(model, X, background)
        values_df = pd.DataFrame(shap_values, columns=[f"SHAP_{feature}" for feature in FEATURE_COLUMNS])
        values_df.insert(0, "Serial_Number", df["Serial_Number"])
        values_df.insert(1, "target", target_name)
        save_csv(values_df, f"shap_values_{target_name}.csv")

        expected_value = float(np.mean(model.predict(background)))
        predicted = model.predict(X)
        reconstructed = expected_value + shap_values.sum(axis=1)
        check_df = pd.DataFrame(
            {
                "Serial_Number": df["Serial_Number"],
                "target": target_name,
                "prediction": predicted,
                "expected_value_plus_sum_SHAP": reconstructed,
                "absolute_reconstruction_error": np.abs(predicted - reconstructed),
            }
        )
        save_csv(check_df, f"shap_additivity_check_{target_name}.csv")

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        total = mean_abs.sum()
        for feature, importance in zip(FEATURE_COLUMNS, mean_abs):
            all_summary.append(
                {
                    "target": target_name,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "mean_abs_shap": importance,
                    "relative_importance_percent": importance / total * 100 if total > 0 else 0,
                    "unit": unit,
                }
            )

        for sample_idx, serial in enumerate(df["Serial_Number"]):
            for feature_idx, feature in enumerate(FEATURE_COLUMNS):
                all_values_long.append(
                    {
                        "Serial_Number": serial,
                        "target": target_name,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "feature_value": df.loc[sample_idx, feature],
                        "shap_value": shap_values[sample_idx, feature_idx],
                        "unit": unit,
                    }
                )

    model_metrics = pd.concat(all_metrics, ignore_index=True)
    model_metrics = model_metrics[["target_name", "target", "n", "R2", "MAE", "unit"]].round(6)
    shap_summary = pd.DataFrame(all_summary).sort_values(["target", "mean_abs_shap"], ascending=[True, False]).reset_index(drop=True)
    values_long = pd.DataFrame(all_values_long)
    corr = correlation_table(df)

    save_csv(model_metrics, "shap_model_metrics.csv")
    save_csv(shap_summary.round(6), "shap_importance_summary.csv")
    save_csv(values_long.round(6), "shap_values_long_format.csv")
    save_csv(corr.round(6), "pearson_spearman_correlation_summary.csv")

    make_importance_plot(shap_summary)
    make_summary_dot_plot(values_long, df)
    write_explanation(model_metrics, shap_summary, corr)

    print("Model metrics")
    print(model_metrics.round(4).to_string(index=False))
    print("\nSHAP mean absolute importance")
    print(shap_summary.round(4).to_string(index=False))
    print("\nPearson/Spearman")
    print(corr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
