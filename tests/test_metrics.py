import numpy as np
import pytest

from credit_risk_benchmark.metrics import (
    binary_classification_metrics,
    bootstrap_metric_interval,
    calibration_table,
    expected_calibration_error,
    select_cost_threshold,
    subgroup_report,
    threshold_report,
)


def test_perfect_probabilities_have_perfect_discrimination() -> None:
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["true_positive"] == 2
    assert metrics["true_negative"] == 2


def test_calibration_error_is_zero_for_exact_bin_rates() -> None:
    assert expected_calibration_error([0, 1], [0.5, 0.5], n_bins=2) == pytest.approx(0.0)


def test_threshold_report_orders_thresholds() -> None:
    report = threshold_report([0, 1, 1], [0.2, 0.6, 0.9], thresholds=[0.8, 0.3])

    assert report["threshold"].tolist() == [0.3, 0.8]


def test_subgroup_report_includes_rates_and_sizes() -> None:
    report = subgroup_report(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.7, 0.8],
        ["A", "A", "B", "B"],
    )

    assert report["group"].tolist() == ["A", "B"]
    assert report["n"].tolist() == [2, 2]
    assert np.isfinite(report["predicted_positive_rate"]).all()


def test_cost_threshold_returns_auditable_report() -> None:
    threshold, report = select_cost_threshold(
        [0, 0, 0, 1, 1], [0.01, 0.10, 0.30, 0.40, 0.90], thresholds=[0.2, 0.5]
    )

    assert threshold in {0.2, 0.5}
    assert {"relative_cost", "cost_per_100_loans"}.issubset(report.columns)


def test_bootstrap_interval_and_calibration_table_are_reproducible() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    probabilities = [0.05, 0.20, 0.40, 0.55, 0.70, 0.95]

    first = bootstrap_metric_interval(labels, probabilities, n_resamples=20, random_state=7)
    second = bootstrap_metric_interval(labels, probabilities, n_resamples=20, random_state=7)
    reliability = calibration_table(labels, probabilities, n_bins=3)

    assert first == second
    assert first[0] <= first[1]
    assert reliability["loans"].sum() == len(labels)


@pytest.mark.parametrize(
    ("labels", "probabilities"),
    [([], []), ([0, 1], [0.2]), ([0, 2], [0.2, 0.8]), ([0, 1], [-0.1, 0.8])],
)
def test_invalid_inputs_raise(labels: list[int], probabilities: list[float]) -> None:
    with pytest.raises(ValueError):
        binary_classification_metrics(labels, probabilities)
