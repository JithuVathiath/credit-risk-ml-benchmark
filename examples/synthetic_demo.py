"""Exercise the evaluation utilities on deterministic synthetic data."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk_benchmark import (
    binary_classification_metrics,
    repeated_group_holdout,
    subgroup_report,
)


def main() -> None:
    features, labels = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=6,
        weights=[0.75, 0.25],
        class_sep=0.8,
        random_state=17,
    )
    entity_groups = np.repeat(np.arange(200), 3)
    audit_groups = np.where(features[:, 0] >= np.median(features[:, 0]), "segment_a", "segment_b")
    train_indices, test_indices = next(
        repeated_group_holdout(entity_groups, n_splits=1, test_size=0.25, random_state=17)
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000))
    model.fit(features[train_indices], labels[train_indices])
    probabilities = model.predict_proba(features[test_indices])[:, 1]

    print("Synthetic demonstration only - not a real credit benchmark")
    print(binary_classification_metrics(labels[test_indices], probabilities))
    print(subgroup_report(labels[test_indices], probabilities, audit_groups[test_indices]))


if __name__ == "__main__":
    main()

