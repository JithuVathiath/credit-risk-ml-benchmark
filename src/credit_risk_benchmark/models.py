"""Model definitions and consistent data preparation for the benchmark."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _one_hot_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def _ordinal_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def sklearn_models(*, random_state: int = 42) -> Mapping[str, Pipeline]:
    """Return deterministic baselines spanning four common model families."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", _one_hot_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=600, solver="lbfgs", random_state=random_state),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", _ordinal_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=240,
                        max_depth=16,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", _ordinal_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.07,
                        max_iter=300,
                        max_leaf_nodes=31,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "neural_network": Pipeline(
            [
                ("preprocessor", _one_hot_preprocessor()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        alpha=0.001,
                        batch_size=512,
                        learning_rate_init=0.001,
                        max_iter=60,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=6,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def prepare_catboost_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Make categorical values acceptable to CatBoost without data leakage."""
    output = features.copy()
    for column in CATEGORICAL_FEATURES:
        output[column] = output[column].astype("string").fillna("UNKNOWN").astype(str)
    for column in NUMERIC_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def catboost_model(*, random_state: int = 42):
    """Construct the optional categorical gradient-boosting baseline."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'CatBoost is required for the full benchmark; install with ".[benchmark]"'
        ) from exc

    return CatBoostClassifier(
        iterations=700,
        depth=7,
        learning_rate=0.06,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=random_state,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
