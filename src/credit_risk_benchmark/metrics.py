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
    brier_score_loss,
    confusion_matrix,
    f1_score,
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
        "expected_calibration_error": expected_calibration_error(
            labels, probabilities, n_bins=n_calibration_bins
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


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

