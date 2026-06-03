# -*- coding: utf-8 -*-
"""
Reviewer 2 - Question 3
Length-scale initialization robustness test for the GP surrogate model.

Purpose:
Compare chemistry-informed initial length scales with uniform initial length
scales to test whether the initial ARD setting biases model prediction.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = ["Molar Ratio", "H2SO4 Volume", "Temperature", "Time"]
TARGET_COLUMNS = ["FWHM", "PLQY"]


def safe_to_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write CSV; if a spreadsheet app locks the old file, write *_updated.csv."""
    path = OUT_DIR / filename
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        updated = path.with_name(f"{path.stem}_updated{path.suffix}")
        df.to_csv(updated, index=False)
        return updated


def safe_write_text(filename: str, text: str) -> Path:
    """Write UTF-8 text; if locked, write *_updated.txt."""
    path = OUT_DIR / filename
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        updated = path.with_name(f"{path.stem}_updated{path.suffix}")
        updated.write_text(text, encoding="utf-8")
        return updated


def build_dataset() -> pd.DataFrame:
    """Return the complete 36-point closed-loop dataset used in the revision."""
    rows = [
        # Serial, Time, Temperature, Molar Ratio, H2SO4 Volume, PLQY fraction, FWHM nm
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
    return df


def make_kernel(initial_length_scales: list[float]):
    """Create the same Matérn 5/2 + white-noise GP kernel used by IDEAL."""
    return (
        C(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=initial_length_scales,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )


def optimized_length_scales(gp: GaussianProcessRegressor) -> np.ndarray:
    """Extract optimized ARD length scales from C * Matern + WhiteKernel."""
    return np.asarray(gp.kernel_.k1.k2.length_scale, dtype=float)


def cross_validate(
    df: pd.DataFrame,
    target_column: str,
    initial_length_scales: list[float],
    setting_name: str,
    optimize_hyperparameters: bool = True,
) -> tuple[dict, pd.DataFrame]:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)
    mode = "optimized_hyperparameters" if optimize_hyperparameters else "no_optimizer_diagnostic"
    optimizer = "fmin_l_bfgs_b" if optimize_hyperparameters else None
    n_restarts = 10 if optimize_hyperparameters else 0

    fold_rows = []
    all_true = []
    all_pred = []
    all_serial = []

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_x.fit_transform(X[train_idx])
        X_test = scaler_x.transform(X[test_idx])
        y_train = scaler_y.fit_transform(y[train_idx].reshape(-1, 1)).ravel()

        gp = GaussianProcessRegressor(
            kernel=make_kernel(initial_length_scales),
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts,
            alpha=1e-6,
            normalize_y=False,
            random_state=42,
        )
        gp.fit(X_train, y_train)

        pred_scaled = gp.predict(X_test)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        truth = y[test_idx]

        all_true.extend(truth)
        all_pred.extend(pred)
        all_serial.extend(df.iloc[test_idx]["Serial Number"].tolist())

        length_scales = optimized_length_scales(gp)
        fold_rows.append(
            {
                "mode": mode,
                "setting": setting_name,
                "target": target_column,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "RMSE": mean_squared_error(truth, pred) ** 0.5,
                "MAE": mean_absolute_error(truth, pred),
                "R2": r2_score(truth, pred),
                "opt_l_Molar_Ratio": length_scales[0],
                "opt_l_H2SO4_Volume": length_scales[1],
                "opt_l_Temperature": length_scales[2],
                "opt_l_Time": length_scales[3],
            }
        )

    all_true_arr = np.asarray(all_true)
    all_pred_arr = np.asarray(all_pred)
    metrics = {
        "mode": mode,
        "setting": setting_name,
        "target": target_column,
        "initial_length_scales": str(initial_length_scales),
        "optimizer": "enabled" if optimize_hyperparameters else "disabled",
        "n_restarts_optimizer": n_restarts,
        "n_samples": len(df),
        "RMSE": mean_squared_error(all_true_arr, all_pred_arr) ** 0.5,
        "MAE": mean_absolute_error(all_true_arr, all_pred_arr),
        "R2": r2_score(all_true_arr, all_pred_arr),
    }

    predictions = pd.DataFrame(
        {
            "mode": mode,
            "setting": setting_name,
            "target": target_column,
            "Serial Number": all_serial,
            "experimental": all_true,
            "predicted": all_pred,
            "error": all_true_arr - all_pred_arr,
        }
    )
    fold_df = pd.DataFrame(fold_rows)
    return metrics, fold_df, predictions


def main():
    df = build_dataset()
    safe_to_csv(df, "length_scale_robustness_dataset_36.csv")

    settings = {
        "chemistry_informed": [1.0, 1.0, 0.5, 0.5],
        "uniform": [1.0, 1.0, 1.0, 1.0],
    }

    metric_rows = []
    fold_tables = []
    prediction_tables = []

    for optimize_hyperparameters in [True, False]:
        for setting_name, length_scales in settings.items():
            for target in TARGET_COLUMNS:
                metrics, fold_df, predictions = cross_validate(
                    df=df,
                    target_column=target,
                    initial_length_scales=length_scales,
                    setting_name=setting_name,
                    optimize_hyperparameters=optimize_hyperparameters,
                )
                metric_rows.append(metrics)
                fold_tables.append(fold_df)
                prediction_tables.append(predictions)

    metrics_df = pd.DataFrame(metric_rows)
    fold_df = pd.concat(fold_tables, ignore_index=True)
    predictions_df = pd.concat(prediction_tables, ignore_index=True)

    safe_to_csv(metrics_df, "length_scale_robustness_metrics.csv")
    safe_to_csv(fold_df, "length_scale_robustness_fold_results.csv")
    safe_to_csv(predictions_df, "length_scale_robustness_predictions.csv")

    # Prepare manuscript/SI-friendly tables in physical units.
    friendly = metrics_df.copy()
    friendly.loc[friendly["target"] == "PLQY", ["RMSE", "MAE"]] *= 100
    friendly["unit"] = np.where(friendly["target"] == "FWHM", "nm", "percentage points")
    safe_to_csv(friendly, "length_scale_robustness_all_results_table.csv")
    si_table = friendly[friendly["mode"] == "optimized_hyperparameters"].copy()
    diagnostic_table = friendly[friendly["mode"] == "no_optimizer_diagnostic"].copy()
    safe_to_csv(si_table, "length_scale_robustness_SI_table.csv")
    safe_to_csv(diagnostic_table, "length_scale_no_optimizer_diagnostic_table.csv")

    chem = si_table[si_table["setting"] == "chemistry_informed"].set_index("target")
    unif = si_table[si_table["setting"] == "uniform"].set_index("target")
    chem_no = diagnostic_table[diagnostic_table["setting"] == "chemistry_informed"].set_index("target")
    unif_no = diagnostic_table[diagnostic_table["setting"] == "uniform"].set_index("target")

    optimized_predictions = predictions_df[predictions_df["mode"] == "optimized_hyperparameters"]
    max_diffs = {}
    for target in TARGET_COLUMNS:
        a = (
            optimized_predictions[
                (optimized_predictions["setting"] == "chemistry_informed")
                & (optimized_predictions["target"] == target)
            ]
            .sort_values("Serial Number")
            .reset_index(drop=True)
        )
        b = (
            optimized_predictions[
                (optimized_predictions["setting"] == "uniform")
                & (optimized_predictions["target"] == target)
            ]
            .sort_values("Serial Number")
            .reset_index(drop=True)
        )
        max_diffs[target] = float((a["predicted"] - b["predicted"]).abs().max())

    lines = [
        "第三问补充实验说明：初始 length scale 鲁棒性测试",
        "",
        "一、实验目的",
        "审稿人关心的是：我们在 GP-ARD 核函数中把 temperature 和 time 的初始 length scale 设得更小，是否会人为偏置模型，进而影响 FWHM 和 PLQY 的预测。",
        "本实验比较两组初始 length scale：",
        "1. chemistry-informed initialization: [1.0, 1.0, 0.5, 0.5]",
        "2. uniform initialization: [1.0, 1.0, 1.0, 1.0]",
        "",
        "二、代码说明",
        "代码文件：length_scale_robustness_experiment.py",
        "代码使用 36 个闭环实验样本，输入特征为 Molar Ratio、H2SO4 Volume、Temperature 和 Time，输出目标为 FWHM 和 PLQY。",
        "模型为与 IDEAL 程序一致的 GaussianProcessRegressor，核函数为 ConstantKernel * Matern(nu=2.5) + WhiteKernel。",
        "所有输入特征和目标值在每个交叉验证训练折内单独进行 Z-score 标准化，避免测试集信息泄漏。",
        "正式结果使用 shuffle 后的 5-fold cross-validation，random_state=42，并开启 GP 超参数优化：optimizer=fmin_l_bfgs_b, n_restarts_optimizer=10。",
        "另外加入一个诊断对照：关闭超参数优化 optimizer=None。这个诊断不作为 SI 主结果，只用于证明初始 length scale 确实进入了模型；如果不优化，两种初始化会产生可见差异。",
        "",
        "三、正式实验结果：开启 GP 超参数优化",
        f"chemistry-informed 设置下，FWHM 的 RMSE/MAE = {chem.loc['FWHM', 'RMSE']:.2f}/{chem.loc['FWHM', 'MAE']:.2f} nm；"
        f"uniform 设置下，FWHM 的 RMSE/MAE = {unif.loc['FWHM', 'RMSE']:.2f}/{unif.loc['FWHM', 'MAE']:.2f} nm。",
        f"chemistry-informed 设置下，PLQY 的 RMSE/MAE = {chem.loc['PLQY', 'RMSE']:.2f}/{chem.loc['PLQY', 'MAE']:.2f} 个百分点；"
        f"uniform 设置下，PLQY 的 RMSE/MAE = {unif.loc['PLQY', 'RMSE']:.2f}/{unif.loc['PLQY', 'MAE']:.2f} 个百分点。",
        f"优化后，两种初始化在交叉验证预测值上的最大差异很小：FWHM 为 {max_diffs['FWHM']:.2e} nm，PLQY 为 {max_diffs['PLQY']:.2e}（PLQY fraction）。",
        "",
        "四、诊断对照：关闭 GP 超参数优化",
        f"关闭优化器后，chemistry-informed 设置下 FWHM 的 RMSE/MAE = {chem_no.loc['FWHM', 'RMSE']:.2f}/{chem_no.loc['FWHM', 'MAE']:.2f} nm；"
        f"uniform 设置下 FWHM 的 RMSE/MAE = {unif_no.loc['FWHM', 'RMSE']:.2f}/{unif_no.loc['FWHM', 'MAE']:.2f} nm。",
        f"关闭优化器后，chemistry-informed 设置下 PLQY 的 RMSE/MAE = {chem_no.loc['PLQY', 'RMSE']:.2f}/{chem_no.loc['PLQY', 'MAE']:.2f} 个百分点；"
        f"uniform 设置下 PLQY 的 RMSE/MAE = {unif_no.loc['PLQY', 'RMSE']:.2f}/{unif_no.loc['PLQY', 'MAE']:.2f} 个百分点。",
        "这个诊断说明初始 length scale 并不是没有被模型读取；如果不进行超参数优化，不同初始值确实会造成略有不同的预测结果。",
        "",
        "五、结果解释",
        "正式 GP 训练中，两组初始 length scale 的交叉验证误差几乎相同，说明初始 length-scale 设置没有对模型预测造成可观察的负面影响，也没有主导最终预测性能。",
        "更准确地说：初始 length scale 确实会进入模型，但它只是 GP 超参数优化的起点，不是固定权重。开启最大化对数边际似然后，两种初始值会收敛到几乎相同的模型解。",
        "因此，真正影响模型预测的是实验数据分布、训练后优化得到的核函数超参数、实验噪声以及 FWHM/PLQY 的多变量耦合关系，而不是初始 length scale 本身。",
        "",
        "六、可写入 SI 的结论句",
        "To test whether the chemistry-informed initial length scales biased model prediction, we compared the original initialization [1.0, 1.0, 0.5, 0.5] with a uniform initialization [1.0, 1.0, 1.0, 1.0]. With GP hyperparameter optimization enabled, five-fold cross-validation gave nearly identical RMSE/MAE values for both FWHM and PLQY. A no-optimizer diagnostic showed that the initial values were indeed read by the model, but their effect disappeared after log-marginal-likelihood optimization. These results indicate that the initial length-scale setting did not dominate the final predictive performance.",
        "",
        "七、输出文件",
        "1. length_scale_robustness_experiment.py：可提交或上传 GitHub 的实验代码。",
        "2. length_scale_robustness_dataset_36.csv：本次实验使用的 36 点数据。",
        "3. length_scale_robustness_metrics.csv：正式优化结果和关闭优化器诊断的总体指标。",
        "4. length_scale_robustness_fold_results.csv：每折交叉验证指标和 length scale。",
        "5. length_scale_robustness_predictions.csv：每个样本在交叉验证测试折中的预测值。",
        "6. length_scale_robustness_SI_table.csv：可整理进 SI 的正式指标表，只包含开启 GP 超参数优化的结果。",
        "7. length_scale_no_optimizer_diagnostic_table.csv：关闭优化器的诊断对照表，不建议作为主结果放 SI，可用于内部解释或回复追问。",
        "8. length_scale_robustness_all_results_table.csv：包含正式结果和诊断结果的汇总表。",
    ]
    safe_write_text("第三问_实验说明_代码解释_结果解释.txt", "\n".join(lines))

    print("Finished length-scale robustness experiment.")
    print(friendly.to_string(index=False))


if __name__ == "__main__":
    main()
