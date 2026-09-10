"""Leakage-aware validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def repeated_group_holdout(
    groups: Iterable[object],
    *,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield reproducible train/test indices with disjoint groups."""
    group_values = np.asarray(list(groups), dtype=object)
    if group_values.ndim != 1 or group_values.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional sequence")
    if len(np.unique(group_values)) < 2:
        raise ValueError("at least two distinct groups are required")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must lie strictly between 0 and 1")

    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    observations = np.zeros(group_values.size)
    for train_indices, test_indices in splitter.split(observations, groups=group_values):
        train_groups = set(group_values[train_indices])
        test_groups = set(group_values[test_indices])
        if train_groups & test_groups:
            raise RuntimeError("group leakage detected in generated split")
        yield train_indices, test_indices

