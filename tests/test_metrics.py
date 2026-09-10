import numpy as np
import pytest

from credit_risk_benchmark.metrics import (
    binary_classification_metrics,
    expected_calibration_error,
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


@pytest.mark.parametrize(
    ("labels", "probabilities"),
    [([], []), ([0, 1], [0.2]), ([0, 2], [0.2, 0.8]), ([0, 1], [-0.1, 0.8])],
)
def test_invalid_inputs_raise(labels: list[int], probabilities: list[float]) -> None:
    with pytest.raises(ValueError):
        binary_classification_metrics(labels, probabilities)

