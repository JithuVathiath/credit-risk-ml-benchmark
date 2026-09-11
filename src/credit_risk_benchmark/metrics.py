"""Metrics used by the credit-risk benchmark.

The positive class is generic: callers must document whether it represents
default, repayment, approval, or another outcome. This module deliberately
avoids assigning lending semantics to a numeric label.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated_binary_inputs(
    y_true: Iterable[int], y_probability: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(list(y_true))
    probabilities = np.asarray(list(y_probability), dtype=float)

    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and y_probability must be one-dimensional")
    if labels.size == 0:
        raise ValueError("inputs must not be empty")
    if labels.size != probabilities.size:
        raise ValueError("y_true and y_probability must have equal length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("y_probability must contain only finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie in the closed interval [0, 1]")

    return labels.astype(int), probabilities


def _validated_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0 <= value <= 1:
        raise ValueError("threshold must lie in the closed interval [0, 1]")
    return value


def expected_calibration_error(
    y_true: Iterable[int], y_probability: Iterable[float], *, n_bins: int = 10
) -> float:
    """Return the weighted absolute calibration gap across equal-width bins."""
    labels, probabilities = _validated_binary_inputs(y_true, y_probability)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    total = labels.size
    error = 0.0

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        observed_rate = float(labels[mask].mean())
        mean_probability = float(probabilities[mask].mean())
        error += float(mask.sum() / total) * abs(observed_rate - mean_probability)

    return float(error)


def binary_classification_metrics(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
) -> dict[str, float | int]:
    """Calculate discrimination, calibration, and threshold metrics."""
    labels, probabilities = _validated_binary_inputs(y_true, y_probability)
    cutoff = _validated_threshold(threshold)
    predictions = (probabilities >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    has_both_classes = np.unique(labels).size == 2

    return {
        "n": int(labels.size),
        "positive_rate": float(labels.mean()),
        "threshold": cutoff,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else np.nan,
        "pr_auc": (
            float(average_precision_score(labels, probabilities)) if has_both_classes else np.nan
        ),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(
            labels, probabilities, n_bins=n_calibration_bins
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions)) if has_both_classes else np.nan
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def select_cost_threshold(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select the validation threshold that minimises documented relative cost."""
    labels, probabilities = _validated_binary_inputs(y_true, y_probability)
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("misclassification costs must be non-negative")
    candidates = (
        np.linspace(0.005, 0.5, 200) if thresholds is None else np.asarray(list(thresholds))
    )
    report = threshold_report(labels, probabilities, thresholds=candidates)
    report["relative_cost"] = (
        false_positive_cost * report["false_positive"]
        + false_negative_cost * report["false_negative"]
    )
    report["cost_per_100_loans"] = 100.0 * report["relative_cost"] / report["n"]
    best_row = report.sort_values(
        ["relative_cost", "threshold"], ascending=[True, False]
    ).iloc[0]
    return float(best_row["threshold"]), report


def bootstrap_metric_interval(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    metric: str = "roc_auc",
    n_resamples: int = 200,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return a deterministic stratified bootstrap confidence interval."""
    labels, probabilities = _validated_binary_inputs(y_true, y_probability)
    if metric not in {"roc_auc", "pr_auc", "brier_score"}:
        raise ValueError("unsupported bootstrap metric")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    if not len(positive_indices) or not len(negative_indices):
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    values = np.empty(n_resamples, dtype=float)
    for position in range(n_resamples):
        sample_indices = np.concatenate(
            [
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
            ]
        )
        sample_labels = labels[sample_indices]
        sample_probabilities = probabilities[sample_indices]
        if metric == "roc_auc":
            values[position] = roc_auc_score(sample_labels, sample_probabilities)
        elif metric == "pr_auc":
            values[position] = average_precision_score(sample_labels, sample_probabilities)
        else:
            values[position] = brier_score_loss(sample_labels, sample_probabilities)

    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def calibration_table(
    y_true: Iterable[int], y_probability: Iterable[float], *, n_bins: int = 10
) -> pd.DataFrame:
    """Return quantile-binned reliability data suitable for plotting."""
    labels, probabilities = _validated_binary_inputs(y_true, y_probability)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    try:
        bins = pd.qcut(probabilities, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(probabilities, bins=n_bins, include_lowest=True)
    frame = pd.DataFrame({"target": labels, "probability": probabilities, "bin": bins})
    return (
        frame.groupby("bin", observed=True)
        .agg(
            loans=("target", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_charge_off_rate=("target", "mean"),
        )
        .reset_index(drop=True)
    )


def threshold_report(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Return one row of metrics per decision threshold."""
    labels, probabilities = _validated_binary_inputs(y_true, y_probability)
    rows = [
        binary_classification_metrics(labels, probabilities, threshold=threshold)
        for threshold in thresholds
    ]
    if not rows:
        raise ValueError("thresholds must contain at least one value")
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def subgroup_report(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    groups: Iterable[object],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return transparent per-group metrics without declaring a fairness verdict."""
    labels, probabilities = _validated_binary_inputs(y_true, y_probability)
    group_values = np.asarray(list(groups), dtype=object)
    cutoff = _validated_threshold(threshold)

    if group_values.ndim != 1 or group_values.size != labels.size:
        raise ValueError("groups must be one-dimensional and match y_true length")
    if pd.isna(group_values).any():
        raise ValueError("groups must not contain missing values")

    rows: list[dict[str, object]] = []
    for group in pd.unique(group_values):
        mask = group_values == group
        predictions = (probabilities[mask] >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels[mask], predictions, labels=[0, 1]).ravel()
        actual_positives = tp + fn
        actual_negatives = tn + fp
        metrics = binary_classification_metrics(
            labels[mask], probabilities[mask], threshold=cutoff
        )
        rows.append(
            {
                "group": group,
                **metrics,
                "predicted_positive_rate": float(predictions.mean()),
                "true_positive_rate": float(tp / actual_positives) if actual_positives else np.nan,
                "false_positive_rate": float(fp / actual_negatives) if actual_negatives else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("group", key=lambda s: s.astype(str)).reset_index(
        drop=True
    )
