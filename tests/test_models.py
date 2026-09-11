from credit_risk_benchmark.data import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from credit_risk_benchmark.models import sklearn_models


def test_model_registry_covers_prespecified_families() -> None:
    models = sklearn_models(random_state=7)

    assert set(models) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "neural_network",
    }
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(MODEL_FEATURES)


def test_models_receive_a_preprocessor_before_classifier() -> None:
    for model in sklearn_models().values():
        assert list(model.named_steps) == ["preprocessor", "classifier"]
