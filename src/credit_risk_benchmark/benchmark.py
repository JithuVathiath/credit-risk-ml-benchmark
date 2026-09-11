"""End-to-end SBA 7(a) benchmark orchestration and artifact generation."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from .data import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    CohortConfig,
    build_vintage_summary,
    discover_csv_files,
    load_sba_model_cohort,
    temporal_split,
)
from .metrics import (
    binary_classification_metrics,
    bootstrap_metric_interval,
    calibration_table,
    select_cost_threshold,
    subgroup_report,
)
from .models import catboost_model, prepare_catboost_frame, sklearn_models


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime controls for the reproducible experiment."""

    mode: str = "full"
    random_state: int = 42
    false_positive_cost: float = 1.0
    false_negative_cost: float = 5.0
    bootstrap_resamples: int = 200
    train_end_year: int = 2014
    validation_end_year: int = 2017
    include_catboost: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"quick", "full"}:
            raise ValueError("mode must be 'quick' or 'full'")


def _stratified_limit(frame: pd.DataFrame, limit: int | None, random_state: int) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    sample, _ = train_test_split(
        frame,
        train_size=limit,
        stratify=frame["target"],
        random_state=random_state,
    )
    return sample.sort_values("approval_date").reset_index(drop=True)


def _partition_limits(mode: str) -> tuple[int | None, int | None, int | None]:
    return (80_000, 30_000, 30_000) if mode == "quick" else (None, None, None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_isotonic(y_true: pd.Series, probabilities: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(probabilities, y_true.to_numpy())
    return calibrator


def _evaluate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float | int]:
    metrics = binary_classification_metrics(y_true, probabilities, threshold=threshold)
    relative_cost = (
        false_positive_cost * int(metrics["false_positive"])
        + false_negative_cost * int(metrics["false_negative"])
    )
    metrics["cost_per_100_loans"] = 100.0 * relative_cost / len(y_true)
    return metrics


def _model_comparison_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = metrics["model"].str.replace("_", " ").str.title()
    positions = np.arange(len(metrics))
    width = 0.35
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(positions - width / 2, metrics["roc_auc"], width, label="ROC-AUC", color="#153D64")
    axis.bar(positions + width / 2, metrics["pr_auc"], width, label="PR-AUC", color="#17A6A1")
    baseline = float(metrics["positive_rate"].iloc[0])
    axis.axhline(baseline, color="#D97706", linestyle="--", linewidth=1.5, label="PR baseline")
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Untouched temporal test performance")
    axis.set_title("SBA 7(a) five-year charge-off benchmark")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _calibration_plot(calibration: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    axis.plot([0, 1], [0, 1], "--", color="#64748B", label="Perfect calibration")
    palette = ["#153D64", "#17A6A1", "#D97706", "#7C3AED", "#BE123C"]
    for color, (model, group) in zip(palette, calibration.groupby("model"), strict=False):
        axis.plot(
            group["mean_predicted_probability"],
            group["observed_charge_off_rate"],
            marker="o",
            linewidth=1.7,
            markersize=4,
            label=model.replace("_", " ").title(),
            color=color,
        )
    axis.set_xlim(0, max(0.35, float(calibration["mean_predicted_probability"].max()) * 1.08))
    axis.set_ylim(0, max(0.35, float(calibration["observed_charge_off_rate"].max()) * 1.08))
    axis.set_xlabel("Mean calibrated probability")
    axis.set_ylabel("Observed five-year charge-off rate")
    axis.set_title("Probability reliability on future vintages")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _vintage_plot(vintage: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.plot(
        vintage["approval_year"],
        100 * vintage["five_year_charge_off_rate"],
        color="#153D64",
        marker="o",
        markersize=3.5,
        linewidth=2,
    )
    axis.axvspan(2010, 2014, alpha=0.09, color="#2563EB", label="Train")
    axis.axvspan(2015, 2017, alpha=0.10, color="#D97706", label="Validation")
    axis.axvspan(2018, 2020, alpha=0.10, color="#059669", label="Test")
    axis.set_xlim(int(vintage["approval_year"].min()), int(vintage["approval_year"].max()))
    axis.set_xticks(np.arange(1991, 2022, 5))
    axis.set_xlabel("SBA approval fiscal year")
    axis.set_ylabel("Charge-offs within five years (%)")
    axis.set_title("Material vintage drift makes random splitting inappropriate")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _importance_plot(importance: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot = importance.sort_values("mean_absolute_shap").tail(12)
    figure, axis = plt.subplots(figsize=(8.5, 6))
    axis.barh(
        plot["feature"].str.replace("_", " "),
        plot["mean_absolute_shap"],
        color="#17A6A1",
    )
    axis.set_xlabel("Mean |SHAP value| on log-odds scale")
    axis.set_title("CatBoost global explanations (held-out test sample)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _executive_dashboard(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    vintage: pd.DataFrame,
    best_model: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    best = metrics.loc[metrics["model"].eq(best_model)].iloc[0]
    best_calibration = calibration.loc[calibration["model"].eq(best_model)]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("SBA 7(a) Credit-Risk Benchmark — Executive View", fontsize=18, weight="bold")

    axis = axes[0, 0]
    ordered = metrics.sort_values("pr_auc")
    axis.barh(ordered["model"].str.replace("_", " "), ordered["pr_auc"], color="#153D64")
    axis.axvline(best["positive_rate"], color="#D97706", linestyle="--", label="No-skill baseline")
    axis.set_title("Precision-recall performance")
    axis.set_xlabel("PR-AUC")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)

    axis = axes[0, 1]
    axis.plot([0, 1], [0, 1], "--", color="#64748B")
    axis.plot(
        best_calibration["mean_predicted_probability"],
        best_calibration["observed_charge_off_rate"],
        marker="o",
        color="#17A6A1",
    )
    limit = max(0.25, float(best_calibration["observed_charge_off_rate"].max()) * 1.1)
    axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="Predicted", ylabel="Observed")
    axis.set_title(f"{best_model.replace('_', ' ').title()} calibration")
    axis.grid(alpha=0.2)

    axis = axes[1, 0]
    axis.plot(vintage["approval_year"], 100 * vintage["five_year_charge_off_rate"], color="#153D64")
    axis.axvspan(2018, 2020, color="#059669", alpha=0.12, label="Test vintages")
    axis.set_xlim(int(vintage["approval_year"].min()), int(vintage["approval_year"].max()))
    axis.set_xticks(np.arange(1991, 2022, 5))
    axis.set(xlabel="Approval year", ylabel="Five-year charge-off rate (%)")
    axis.set_title("Outcome drift across mature vintages")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)

    axis = axes[1, 1]
    matrix = np.array(
        [
            [best["true_negative"], best["false_positive"]],
            [best["false_negative"], best["true_positive"]],
        ]
    )
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{int(matrix[row, column]):,}", ha="center", va="center")
    axis.set_xticks([0, 1], ["Not flagged", "Flagged"])
    axis.set_yticks([0, 1], ["No charge-off", "Charge-off"])
    axis.set_title(f"Cost-selected threshold: {best['threshold']:.3f}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.text(
        0.5,
        0.015,
        f"Untouched FY2018–FY2020 test | {int(best['n']):,} loans | "
        f"ROC-AUC {best['roc_auc']:.3f} | PR-AUC {best['pr_auc']:.3f} | "
        f"Brier {best['brier_score']:.4f}",
        ha="center",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _catboost_explanations(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from catboost import Pool

    sample = features.sample(n=min(2_000, len(features)), random_state=random_state)
    prepared = prepare_catboost_frame(sample)
    pool = Pool(prepared, cat_features=CATEGORICAL_FEATURES)
    shap_matrix = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    shap_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_absolute_shap": np.abs(shap_matrix).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)

    permutation_sample = features.sample(
        n=min(5_000, len(features)), random_state=random_state + 1
    )
    permutation_target = target.loc[permutation_sample.index].to_numpy()
    prepared_permutation = prepare_catboost_frame(permutation_sample)
    # CatBoost cannot accept permuted object columns through sklearn's generic helper reliably,
    # so score one shuffled column at a time with an explicit, deterministic implementation.
    baseline = model.predict_proba(prepared_permutation)[:, 1]
    from sklearn.metrics import roc_auc_score

    baseline_auc = roc_auc_score(permutation_target, baseline)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []
    for feature in MODEL_FEATURES:
        drops: list[float] = []
        for _ in range(3):
            permuted = prepared_permutation.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            score = roc_auc_score(
                permutation_target, model.predict_proba(permuted)[:, 1]
            )
            drops.append(float(baseline_auc - score))
        rows.append(
            {
                "feature": feature,
                "mean_auc_decrease": float(np.mean(drops)),
                "std_auc_decrease": float(np.std(drops, ddof=1)),
            }
        )
    permutation = pd.DataFrame(rows).sort_values("mean_auc_decrease", ascending=False)
    return shap_importance, permutation


def _subgroup_outputs(
    test: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dimension in ["business_type", "business_age_band", "approval_year"]:
        report = subgroup_report(
            test["target"], probabilities, test[dimension].astype(str), threshold=threshold
        )
        report.insert(0, "dimension", dimension)
        frames.append(report)
    return pd.concat(frames, ignore_index=True)


def _error_analysis(
    test: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> pd.DataFrame:
    frame = test[["business_type", "business_age_band", "approval_year", "target"]].copy()
    frame["probability"] = probabilities
    frame["prediction"] = (probabilities >= threshold).astype(int)
    frame["error_type"] = np.select(
        [
            frame["target"].eq(1) & frame["prediction"].eq(0),
            frame["target"].eq(0) & frame["prediction"].eq(1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )
    rows: list[pd.DataFrame] = []
    for dimension in ["business_type", "business_age_band", "approval_year"]:
        grouped = (
            frame.groupby([dimension, "error_type"], dropna=False)
            .agg(loans=("target", "size"), mean_probability=("probability", "mean"))
            .reset_index()
            .rename(columns={dimension: "segment"})
        )
        grouped.insert(0, "dimension", dimension)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _write_summary_markdown(
    path: Path,
    metrics: pd.DataFrame,
    partitions: dict[str, dict[str, float | int]],
    best_model: str,
    config: BenchmarkConfig,
) -> None:
    best = metrics.loc[metrics["model"].eq(best_model)].iloc[0]
    train_n = f"{partitions['train']['loans']:,}"
    train_rate = f"{partitions['train']['charge_off_rate']:.2%}"
    validation_n = f"{partitions['validation']['loans']:,}"
    validation_rate = f"{partitions['validation']['charge_off_rate']:.2%}"
    test_n = f"{partitions['test']['loans']:,}"
    test_rate = f"{partitions['test']['charge_off_rate']:.2%}"
    roc_interval = (
        f"{best['roc_auc_ci_lower']:.3f}–{best['roc_auc_ci_upper']:.3f}"
    )
    content = f"""# Reproducible Benchmark Summary

Generated from the official SBA 7(a) snapshot using `{config.mode}` mode and random seed
`{config.random_state}`. The test set was not used for fitting, calibration, model selection,
or threshold selection.

## Cohort

| Partition | Approval vintages | Loans | Five-year charge-off rate |
|---|---:|---:|---:|
| Train | FY2010–FY2014 | {train_n} | {train_rate} |
| Validation | FY2015–FY2017 | {validation_n} | {validation_rate} |
| Test | FY2018–FY2020 | {test_n} | {test_rate} |

## Headline result

`{best_model}` ranked first by validation PR-AUC. On the untouched temporal test it achieved:

- ROC-AUC: **{best['roc_auc']:.3f}** (95% bootstrap CI {roc_interval})
- PR-AUC: **{best['pr_auc']:.3f}** (no-skill baseline {best['positive_rate']:.3f})
- Brier score after isotonic calibration: **{best['brier_score']:.4f}**
- Recall at the validation-selected {config.false_negative_cost:g}:1 cost threshold:
  **{best['recall']:.1%}**
- Precision at that threshold: **{best['precision']:.1%}**

The relative cost ratio is an analytical scenario, not an SBA policy or production decision rule.

## Interpretation boundary

This estimates charge-off within five years among already-approved and disbursed SBA 7(a) loans.
It does not estimate approval likelihood, establish causality, or validate an underwriting system.
The source does not include protected-class attributes, so the subgroup report is performance
monitoring—not a demographic fairness assessment.
"""
    path.write_text(content, encoding="utf-8")


def run_benchmark(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    config: BenchmarkConfig | None = None,
) -> dict[str, object]:
    """Run the complete temporal benchmark and write only aggregate public artifacts."""
    config = config or BenchmarkConfig()
    started = time.perf_counter()
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    figures_dir = output / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_csv_files(data_dir)
    cohort_config = CohortConfig(first_model_year=2010, last_model_year=2020)
    cohort = load_sba_model_cohort(paths, config=cohort_config)
    train, validation, test = temporal_split(
        cohort,
        train_end_year=config.train_end_year,
        validation_end_year=config.validation_end_year,
    )
    limits = _partition_limits(config.mode)
    train = _stratified_limit(train, limits[0], config.random_state)
    validation = _stratified_limit(validation, limits[1], config.random_state + 1)
    test = _stratified_limit(test, limits[2], config.random_state + 2)

    partitions = {
        name: {"loans": len(frame), "charge_off_rate": float(frame["target"].mean())}
        for name, frame in [("train", train), ("validation", validation), ("test", test)]
    }
    x_train, y_train = train[MODEL_FEATURES], train["target"]
    x_validation, y_validation = validation[MODEL_FEATURES], validation["target"]
    x_test, y_test = test[MODEL_FEATURES], test["target"]

    predictions: dict[str, np.ndarray] = {}
    validation_scores: dict[str, float] = {}
    fitted_models: dict[str, object] = {}
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []

    candidates: dict[str, object] = dict(sklearn_models(random_state=config.random_state))
    if config.include_catboost:
        candidates["catboost"] = catboost_model(random_state=config.random_state)

    for model_name, model in candidates.items():
        fit_started = time.perf_counter()
        if model_name == "catboost":
            cat_train = prepare_catboost_frame(x_train)
            cat_validation = prepare_catboost_frame(x_validation)
            cat_test = prepare_catboost_frame(x_test)
            model.fit(
                cat_train,
                y_train,
                cat_features=CATEGORICAL_FEATURES,
                eval_set=(cat_validation, y_validation),
                early_stopping_rounds=60,
                use_best_model=True,
            )
            raw_validation = model.predict_proba(cat_validation)[:, 1]
            inference_started = time.perf_counter()
            raw_test = model.predict_proba(cat_test)[:, 1]
        else:
            model.fit(x_train, y_train)
            raw_validation = model.predict_proba(x_validation)[:, 1]
            inference_started = time.perf_counter()
            raw_test = model.predict_proba(x_test)[:, 1]
        inference_seconds = time.perf_counter() - inference_started
        train_seconds = inference_started - fit_started

        calibrator = _fit_isotonic(y_validation, raw_validation)
        calibrated_validation = np.asarray(calibrator.predict(raw_validation))
        calibrated_test = np.asarray(calibrator.predict(raw_test))
        threshold, threshold_table = select_cost_threshold(
            y_validation,
            calibrated_validation,
            false_positive_cost=config.false_positive_cost,
            false_negative_cost=config.false_negative_cost,
        )
        test_metrics = _evaluate(
            y_test,
            calibrated_test,
            threshold=threshold,
            false_positive_cost=config.false_positive_cost,
            false_negative_cost=config.false_negative_cost,
        )
        raw_metrics = binary_classification_metrics(y_test, raw_test, threshold=threshold)
        validation_metrics = binary_classification_metrics(y_validation, calibrated_validation)
        validation_scores[model_name] = float(validation_metrics["pr_auc"])
        predictions[model_name] = calibrated_test
        fitted_models[model_name] = model
        metric_rows.append(
            {
                "model": model_name,
                **test_metrics,
                "raw_brier_score": raw_metrics["brier_score"],
                "raw_expected_calibration_error": raw_metrics["expected_calibration_error"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "roc_auc_ci_lower": np.nan,
                "roc_auc_ci_upper": np.nan,
                "pr_auc_ci_lower": np.nan,
                "pr_auc_ci_upper": np.nan,
                "train_seconds": train_seconds,
                "inference_ms_per_1000": 1_000_000 * inference_seconds / len(test),
            }
        )
        reliability = calibration_table(y_test, calibrated_test, n_bins=10)
        reliability.insert(0, "model", model_name)
        calibration_rows.append(reliability)
        threshold_table.insert(0, "model", model_name)
        threshold_rows.append(threshold_table)

    metrics = pd.DataFrame(metric_rows).sort_values("validation_pr_auc", ascending=False)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    thresholds = pd.concat(threshold_rows, ignore_index=True)
    best_model = max(validation_scores, key=validation_scores.get)
    best_row = metrics.loc[metrics["model"].eq(best_model)].iloc[0]
    best_probabilities = predictions[best_model]
    best_threshold = float(best_row["threshold"])
    best_mask = metrics["model"].eq(best_model)
    metrics.loc[best_mask, ["roc_auc_ci_lower", "roc_auc_ci_upper"]] = bootstrap_metric_interval(
        y_test,
        best_probabilities,
        metric="roc_auc",
        n_resamples=config.bootstrap_resamples,
        random_state=config.random_state,
    )
    metrics.loc[best_mask, ["pr_auc_ci_lower", "pr_auc_ci_upper"]] = bootstrap_metric_interval(
        y_test,
        best_probabilities,
        metric="pr_auc",
        n_resamples=config.bootstrap_resamples,
        random_state=config.random_state + 1,
    )
    best_row = metrics.loc[best_mask].iloc[0]

    vintage = build_vintage_summary(paths, config=cohort_config)
    subgroup = _subgroup_outputs(test, best_probabilities, best_threshold)
    errors = _error_analysis(test, best_probabilities, best_threshold)
    annual_performance_rows: list[dict[str, object]] = []
    for year, indices in test.groupby("approval_year").groups.items():
        annual = binary_classification_metrics(
            test.loc[indices, "target"],
            best_probabilities[np.asarray(indices)],
            threshold=best_threshold,
        )
        annual_performance_rows.append({"approval_year": int(year), **annual})
    annual_performance = pd.DataFrame(annual_performance_rows)

    shap_importance = pd.DataFrame(columns=["feature", "mean_absolute_shap"])
    permutation = pd.DataFrame(columns=["feature", "mean_auc_decrease", "std_auc_decrease"])
    if "catboost" in fitted_models:
        shap_importance, permutation = _catboost_explanations(
            fitted_models["catboost"], x_test, y_test, config.random_state
        )

    metrics.to_csv(metrics_dir / "model_metrics.csv", index=False)
    calibration.to_csv(metrics_dir / "calibration.csv", index=False)
    thresholds.to_csv(metrics_dir / "validation_thresholds.csv", index=False)
    vintage.to_csv(metrics_dir / "vintage_drift.csv", index=False)
    subgroup.to_csv(metrics_dir / "subgroup_performance.csv", index=False)
    errors.to_csv(metrics_dir / "error_analysis.csv", index=False)
    annual_performance.to_csv(metrics_dir / "annual_test_performance.csv", index=False)
    shap_importance.to_csv(metrics_dir / "shap_importance.csv", index=False)
    permutation.to_csv(metrics_dir / "permutation_importance.csv", index=False)

    _model_comparison_plot(metrics, figures_dir / "model_comparison.png")
    _calibration_plot(calibration, figures_dir / "calibration_curves.png")
    _vintage_plot(vintage, figures_dir / "vintage_drift.png")
    if not shap_importance.empty:
        _importance_plot(shap_importance, figures_dir / "shap_importance.png")
    _executive_dashboard(
        metrics, calibration, vintage, best_model, figures_dir / "executive_dashboard.png"
    )

    _write_summary_markdown(
        output / "benchmark-summary.md", metrics, partitions, best_model, config
    )
    metadata = {
        "config": asdict(config),
        "cohort_definition": asdict(cohort_config),
        "partitions": partitions,
        "cohort_counts": {
            "mature_outcome_loans_fy1991_present": int(vintage["loans"].sum()),
            "model_cohort_loans_fy2010_fy2020": int(len(cohort)),
        },
        "best_model_by_validation_pr_auc": best_model,
        "data_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in paths
        ],
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata
