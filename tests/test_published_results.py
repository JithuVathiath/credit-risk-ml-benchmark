import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]


def test_published_results_match_declared_selection_rule() -> None:
    metrics = pd.read_csv(ROOT / "reports" / "metrics" / "model_metrics.csv")
    metadata = json.loads((ROOT / "reports" / "run_metadata.json").read_text())
    best = metrics.sort_values("validation_pr_auc", ascending=False).iloc[0]

    assert len(metrics) == 5
    assert metrics["model"].is_unique
    assert best["model"] == metadata["best_model_by_validation_pr_auc"] == "catboost"
    assert int(best["n"]) == metadata["partitions"]["test"]["loans"] == 136_027
    assert best["roc_auc_ci_lower"] < best["roc_auc"] < best["roc_auc_ci_upper"]
    assert best["pr_auc_ci_lower"] < best["pr_auc"] < best["pr_auc_ci_upper"]
    assert best["pr_auc"] > 8 * best["positive_rate"]


def test_published_cohort_counts_are_internally_consistent() -> None:
    metadata = json.loads((ROOT / "reports" / "run_metadata.json").read_text())
    partitions = metadata["partitions"]
    partition_total = sum(partition["loans"] for partition in partitions.values())

    assert partition_total == metadata["cohort_counts"]["model_cohort_loans_fy2010_fy2020"]
    assert partition_total == 514_825
    assert partitions["test"]["charge_off_rate"] == pytest.approx(0.0402053, abs=1e-7)


def test_public_artifacts_are_aggregate_only() -> None:
    metrics_dir = ROOT / "reports" / "metrics"
    forbidden_fragments = {"borrower", "borrname", "street", "zipcode", "loan_number"}

    for path in metrics_dir.glob("*.csv"):
        columns = {column.lower() for column in pd.read_csv(path, nrows=0).columns}
        assert columns.isdisjoint(forbidden_fragments), path.name
