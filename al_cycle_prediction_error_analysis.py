# -*- coding: utf-8 -*-
"""
Reviewer 2 - Question 4
Per-cycle prediction error analysis for AL-selected batches.

This script compares the prediction values saved at each AL recommendation
step with the subsequently measured experimental FWHM and PLQY values.

The primary table is prospective: for AL cycle k, the predictions are those
available before running that experimental batch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parents[1]

FEATURE_COLUMNS = ["Time", "Temperature", "Molar Ratio", "H2SO4 Volume"]
ROUND_FILES = [
    (1, "resultsn16/suggested_experiments_iteration_0.csv"),
    (2, "resultsn16_4/suggested_experiments_iteration_0.csv"),
    (3, "resultsn16_8/suggested_experiments_iteration_0.csv"),
    (4, "resultsn16_12/suggested_experiments_iteration_0.csv"),
    (5, "resultsn16_16/suggested_experiments_iteration_0.csv"),
]


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


def build_final_dataset() -> pd.DataFrame:
    """Use the manuscript/SI 36-point dataset, with PLQY as fraction."""
    rows = [
        (1, 10.00, 150.0, 400.0, 3.00, 0.2183, 62),
        (2, 10.00, 150.0, 400.0, 4.00, 0.2319, 64),
        (3, 10.00, 150.0, 600.0, 3.00, 0.1587, 69),
        (4, 10.00, 150.0, 600.0, 4.00, 0.0950, 74),
        (5, 10.00, 170.0, 400.0, 3.00, 0.1091, 68),
        (6, 10.00, 170.0, 400.0, 4.00, 0.1486, 58),
        (7, 10.00, 170.0, 600.0, 3.00, 0.1576, 72),
        (8, 10.00, 170.0, 600.0, 4.00, 0.1373, 85),
        (9, 12.00, 150.0, 400.0, 3.00, 0.1444, 65),
        (10, 12.00, 150.0, 400.0, 4.00, 0.2637, 76),
        (11, 12.00, 150.0, 600.0, 3.00, 0.0973, 82),
        (12, 12.00, 150.0, 600.0, 4.00, 0.1775, 87),
        (13, 12.00, 170.0, 400.0, 3.00, 0.0944, 67),
        (14, 12.00, 170.0, 400.0, 4.00, 0.1437, 64),
        (15, 12.00, 170.0, 600.0, 3.00, 0.1660, 75),
        (16, 12.00, 170.0, 600.0, 4.00, 0.1523, 71),
        (17, 10.91, 152.3, 396.1, 3.75, 0.2419, 57),
        (18, 9.02, 150.3, 359.9, 3.92, 0.1899, 58),
        (19, 8.40, 199.5, 250.2, 1.69, 0.1532, 64),
        (20, 9.15, 196.4, 749.3, 2.53, 0.1718, 55),
        (21, 10.68, 153.9, 350.2, 3.97, 0.2744, 45),
        (22, 10.41, 155.3, 436.5, 3.84, 0.2279, 44),
        (23, 10.78, 153.2, 473.9, 3.99, 0.2032, 53),
        (24, 10.93, 151.8, 305.1, 3.54, 0.2163, 48),
        (25, 11.78, 154.8, 346.1, 3.77, 0.2632, 38),
        (26, 10.68, 159.1, 268.4, 3.92, 0.2591, 52),
        (27, 8.79, 155.1, 415.1, 2.29, 0.2123, 40),
        (28, 8.29, 152.6, 466.6, 2.69, 0.2059, 51),
        (29, 9.75, 154.1, 272.1, 1.89, 0.3205, 29),
        (30, 10.49, 155.4, 368.1, 1.51, 0.3663, 31),
        (31, 11.84, 156.1, 325.6, 1.82, 0.1931, 47),
        (32, 11.85, 155.9, 410.7, 1.59, 0.1839, 48),
        (33, 10.01, 153.3, 293.9, 1.37, 0.2385, 41),
        (34, 9.67, 155.4, 374.5, 1.12, 0.3358, 61),
        (35, 10.04, 162.3, 476.6, 1.02, 0.1945, 36),
        (36, 8.36, 155.7, 448.7, 1.03, 0.2833, 36),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Serial Number",
            "Time",
            "Temperature",
            "Molar Ratio",
            "H2SO4 Volume",
            "PLQY",
            "FWHM",
        ],
    )
    df["PLQY_percent"] = df["PLQY"] * 100
    df["AL_cycle"] = np.where(df["Serial Number"] <= 16, 0, ((df["Serial Number"] - 17) // 4) + 1)
    return df


def read_prospective_predictions(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, rel_path in ROUND_FILES:
        pred_df = pd.read_csv(ROOT / rel_path)
        serials = list(range(17 + (cycle - 1) * 4, 21 + (cycle - 1) * 4))
        for row_idx, serial in enumerate(serials):
            exp = final_df[final_df["Serial Number"] == serial].iloc[0]
            pred = pred_df.iloc[row_idx]
            rows.append(
                {
                    "AL_cycle": cycle,
                    "batch_point": row_idx + 1,
                    "Serial Number": serial,
                    "Time": exp["Time"],
                    "Temperature": exp["Temperature"],
                    "Molar Ratio": exp["Molar Ratio"],
                    "H2SO4 Volume": exp["H2SO4 Volume"],
                    "Predicted_FWHM": pred["Predicted_FWHM"],
                    "Experimental_FWHM": exp["FWHM"],
                    "FWHM_error": exp["FWHM"] - pred["Predicted_FWHM"],
                    "Predicted_PLQY_percent": pred["Predicted_QY"] * 100,
                    "Experimental_PLQY_percent": exp["PLQY_percent"],
                    "PLQY_error_percentage_points": exp["PLQY_percent"] - pred["Predicted_QY"] * 100,
                    "Acquisition_Value": pred["Acquisition_Value"],
                }
            )
    return pd.DataFrame(rows)


def summarize_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, group in predictions.groupby("AL_cycle"):
        rows.append(
            {
                "AL_cycle": cycle,
                "n": len(group),
                "FWHM_RMSE_nm": mean_squared_error(
                    group["Experimental_FWHM"], group["Predicted_FWHM"]
                )
                ** 0.5,
                "FWHM_MAE_nm": mean_absolute_error(
                    group["Experimental_FWHM"], group["Predicted_FWHM"]
                ),
                "PLQY_RMSE_percentage_points": mean_squared_error(
                    group["Experimental_PLQY_percent"],
                    group["Predicted_PLQY_percent"],
                )
                ** 0.5,
                "PLQY_MAE_percentage_points": mean_absolute_error(
                    group["Experimental_PLQY_percent"],
                    group["Predicted_PLQY_percent"],
                ),
                "mean_acquisition_value": group["Acquisition_Value"].mean(),
                "mean_experimental_FWHM_nm": group["Experimental_FWHM"].mean(),
                "mean_experimental_PLQY_percent": group["Experimental_PLQY_percent"].mean(),
            }
        )

    all_rows = {
        "AL_cycle": "All AL cycles",
        "n": len(predictions),
        "FWHM_RMSE_nm": mean_squared_error(
            predictions["Experimental_FWHM"], predictions["Predicted_FWHM"]
        )
        ** 0.5,
        "FWHM_MAE_nm": mean_absolute_error(
            predictions["Experimental_FWHM"], predictions["Predicted_FWHM"]
        ),
        "PLQY_RMSE_percentage_points": mean_squared_error(
            predictions["Experimental_PLQY_percent"],
            predictions["Predicted_PLQY_percent"],
        )
        ** 0.5,
        "PLQY_MAE_percentage_points": mean_absolute_error(
            predictions["Experimental_PLQY_percent"],
            predictions["Predicted_PLQY_percent"],
        ),
        "mean_acquisition_value": predictions["Acquisition_Value"].mean(),
        "mean_experimental_FWHM_nm": predictions["Experimental_FWHM"].mean(),
        "mean_experimental_PLQY_percent": predictions["Experimental_PLQY_percent"].mean(),
    }
    rows.append(all_rows)
    return pd.DataFrame(rows)


def retrospective_fit_metrics(final_df: pd.DataFrame) -> pd.DataFrame:
    """Optional model-fit metrics on all 36 points for context, not AL-time prediction."""
    X = final_df[FEATURE_COLUMNS].values
    rows = []
    for target, y_col, unit in [
        ("FWHM", "FWHM", "nm"),
        ("PLQY", "PLQY_percent", "percentage points"),
    ]:
        y = final_df[y_col].values
        model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
        model.fit(X, y)
        pred = model.predict(X)
        rows.append(
            {
                "evaluation_type": "retrospective_final_fit_not_AL_time",
                "target": target,
                "n": len(y),
                "R2": r2_score(y, pred),
                "RMSE": mean_squared_error(y, pred) ** 0.5,
                "MAE": mean_absolute_error(y, pred),
                "unit": unit,
            }
        )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, predictions: pd.DataFrame):
    plot_df = summary[summary["AL_cycle"] != "All AL cycles"].copy()
    plot_df["AL_cycle"] = plot_df["AL_cycle"].astype(int)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=220)
    axes[0].plot(plot_df["AL_cycle"], plot_df["FWHM_RMSE_nm"], marker="o", label="RMSE", color="#1f77b4")
    axes[0].plot(plot_df["AL_cycle"], plot_df["FWHM_MAE_nm"], marker="s", label="MAE", color="#ff7f0e")
    axes[0].set_xlabel("AL cycle")
    axes[0].set_ylabel("FWHM error (nm)")
    axes[0].set_title("Prospective FWHM prediction error")
    axes[0].grid(True, linestyle=":", alpha=0.55)
    axes[0].legend(frameon=False)

    axes[1].plot(plot_df["AL_cycle"], plot_df["PLQY_RMSE_percentage_points"], marker="o", label="RMSE", color="#1f77b4")
    axes[1].plot(plot_df["AL_cycle"], plot_df["PLQY_MAE_percentage_points"], marker="s", label="MAE", color="#ff7f0e")
    axes[1].set_xlabel("AL cycle")
    axes[1].set_ylabel("PLQY error (percentage points)")
    axes[1].set_title("Prospective PLQY prediction error")
    axes[1].grid(True, linestyle=":", alpha=0.55)
    axes[1].legend(frameon=False)

    fig.suptitle("Per-cycle prospective prediction errors for AL-selected batches", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "al_cycle_prediction_error_evolution.png", bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=220)
    cycles = predictions["AL_cycle"].values
    scatter = axes[0].scatter(
        predictions["Experimental_FWHM"],
        predictions["Predicted_FWHM"],
        c=cycles,
        cmap="viridis",
        s=55,
        edgecolor="black",
        linewidth=0.4,
    )
    min_v = min(predictions["Experimental_FWHM"].min(), predictions["Predicted_FWHM"].min())
    max_v = max(predictions["Experimental_FWHM"].max(), predictions["Predicted_FWHM"].max())
    axes[0].plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    axes[0].set_xlabel("Experimental FWHM (nm)")
    axes[0].set_ylabel("Predicted FWHM (nm)")
    axes[0].set_title("FWHM parity")
    axes[0].grid(True, linestyle=":", alpha=0.45)

    axes[1].scatter(
        predictions["Experimental_PLQY_percent"],
        predictions["Predicted_PLQY_percent"],
        c=cycles,
        cmap="viridis",
        s=55,
        edgecolor="black",
        linewidth=0.4,
    )
    min_v = min(predictions["Experimental_PLQY_percent"].min(), predictions["Predicted_PLQY_percent"].min())
    max_v = max(predictions["Experimental_PLQY_percent"].max(), predictions["Predicted_PLQY_percent"].max())
    axes[1].plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    axes[1].set_xlabel("Experimental PLQY (%)")
    axes[1].set_ylabel("Predicted PLQY (%)")
    axes[1].set_title("PLQY parity")
    axes[1].grid(True, linestyle=":", alpha=0.45)

    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("AL cycle")
    fig.suptitle("Prospective predictions for all 20 AL-selected samples", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.96, 0.93])
    fig.savefig(OUT_DIR / "al_cycle_prediction_parity.png", bbox_inches="tight")


def main():
    final_df = build_final_dataset()
    safe_to_csv(final_df, "al_cycle_prediction_dataset_36.csv")

    predictions = read_prospective_predictions(final_df)
    summary = summarize_errors(predictions)
    retrospective = retrospective_fit_metrics(final_df)

    safe_to_csv(predictions, "al_cycle_prospective_predictions_vs_experiment.csv")
    safe_to_csv(summary, "al_cycle_prediction_error_summary.csv")
    safe_to_csv(retrospective, "retrospective_final_fit_metrics_context.csv")

    make_figures(summary, predictions)

    all_row = summary[summary["AL_cycle"] == "All AL cycles"].iloc[0]
    best_fwhm_cycle = summary[summary["AL_cycle"] != "All AL cycles"].sort_values("FWHM_MAE_nm").iloc[0]
    best_plqy_cycle = summary[summary["AL_cycle"] != "All AL cycles"].sort_values("PLQY_MAE_percentage_points").iloc[0]

    lines = [
        "第四问补充实验说明：每轮 AL 的预测误差指标",
        "",
        "一、实验目的",
        "审稿人建议在每次 AL 迭代中加入 PLQY 和 FWHM 的误差指标，例如预测数据与实验数据之间的 RMSE 和 MAE。",
        "本实验使用每轮模型推荐时已经保存的预测值，与随后真实实验测得的 FWHM 和 PLQY 进行比较。",
        "",
        "二、重要口径",
        "这里的主结果是 prospective AL-cycle prediction error：即第 k 轮实验开始之前，模型对本轮 TOP-4 候选点给出的预测，与实验完成后的真实值之间的误差。",
        "这不同于最终使用全部数据重新拟合后的 retrospective fit。前者更能真实回答审稿人的问题，后者通常更好看，但不能代表 AL 选点当时的预测能力。",
        "",
        "三、代码说明",
        "代码文件：al_cycle_prediction_error_analysis.py",
        "预测值来源：每轮 results*/suggested_experiments_iteration_0.csv 中保存的 Predicted_FWHM、Predicted_QY 和 Acquisition_Value。",
        "实验值来源：整理后的 36 点最终闭环数据集，其中 PLQY 以百分数形式计算误差。",
        "输出指标：每轮 TOP-4 的 FWHM RMSE、FWHM MAE、PLQY RMSE 和 PLQY MAE。",
        "",
        "四、主要结果",
        f"全部 20 个 AL 推荐样品的 prospective FWHM RMSE/MAE = {all_row['FWHM_RMSE_nm']:.2f}/{all_row['FWHM_MAE_nm']:.2f} nm。",
        f"全部 20 个 AL 推荐样品的 prospective PLQY RMSE/MAE = {all_row['PLQY_RMSE_percentage_points']:.2f}/{all_row['PLQY_MAE_percentage_points']:.2f} 个百分点。",
        f"FWHM MAE 最低的是第 {int(best_fwhm_cycle['AL_cycle'])} 轮，MAE = {best_fwhm_cycle['FWHM_MAE_nm']:.2f} nm。",
        f"PLQY MAE 最低的是第 {int(best_plqy_cycle['AL_cycle'])} 轮，MAE = {best_plqy_cycle['PLQY_MAE_percentage_points']:.2f} 个百分点。",
        "",
        "五、结果解释",
        "逐轮误差并不严格单调下降，这在主动学习中是合理的。因为 acquisition function 不只是选择模型最有把握的点，而是同时考虑 expected improvement、prediction uncertainty 和 diversity。",
        "换句话说，AL 会有意选择一部分高潜力但模型不确定度较高的点，因此这些点的前瞻预测误差可能较大。",
        "建议在 SI 中报告该 prospective error table，同时在正文中说明这些误差用于表征每轮选点时模型预测行为，而不是最终模型拟合精度。",
        "",
        "六、可放入 SI 的英文表述",
        "To evaluate model behavior during the closed-loop process, we calculated prospective prediction errors for each AL cycle by comparing the FWHM and PLQY values predicted before experimental validation with the subsequently measured experimental values. The per-cycle RMSE and MAE values are summarized in Table Sx. These errors do not decrease monotonically because the acquisition function intentionally balances expected improvement, predictive uncertainty, and diversity, rather than selecting only the candidates with the lowest model uncertainty.",
        "",
        "七、输出文件",
        "1. al_cycle_prediction_error_analysis.py：第四问实验代码。",
        "2. al_cycle_prospective_predictions_vs_experiment.csv：20 个 AL 样品的逐点预测值、实验值和误差。",
        "3. al_cycle_prediction_error_summary.csv：可放入 SI 的逐轮 RMSE/MAE 汇总表。",
        "4. al_cycle_prediction_error_evolution.png：逐轮误差变化图。",
        "5. al_cycle_prediction_parity.png：20 个 AL 样品的预测-实验 parity plot。",
        "6. retrospective_final_fit_metrics_context.csv：最终回拟合指标，仅作内部参考或补充，不能替代逐轮 prospective error。",
    ]
    safe_write_text("第四问_逐轮预测误差实验说明_代码解释_结果解释.txt", "\n".join(lines))

    print("Prospective AL-cycle error summary:")
    print(summary.round(4).to_string(index=False))
    print("\nRetrospective context:")
    print(retrospective.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
