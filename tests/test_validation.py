import numpy as np
import pytest

from credit_risk_benchmark.validation import repeated_group_holdout


def test_group_holdout_has_no_group_overlap() -> None:
    groups = np.repeat(np.arange(20), 3)

    splits = list(repeated_group_holdout(groups, n_splits=4, test_size=0.25))

    assert len(splits) == 4
    for train_indices, test_indices in splits:
        assert set(groups[train_indices]).isdisjoint(set(groups[test_indices]))


def test_group_holdout_is_reproducible() -> None:
    groups = np.repeat(np.arange(10), 2)
    first = list(repeated_group_holdout(groups, n_splits=2, random_state=7))
    second = list(repeated_group_holdout(groups, n_splits=2, random_state=7))

    for (first_train, first_test), (second_train, second_test) in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_test, second_test)


def test_group_holdout_requires_multiple_groups() -> None:
    with pytest.raises(ValueError):
        list(repeated_group_holdout(["only", "only"]))

