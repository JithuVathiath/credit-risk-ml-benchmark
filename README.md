# Credit Risk ML Benchmark

A rigorous, reproducible framework for comparing credit-risk classifiers across discrimination, calibration, threshold behaviour, subgroup performance, and explainability.

> **Status:** Active development. This repository does not yet report benchmark results. Metrics will be added only after the selected public dataset and evaluation protocol have been validated.

## Why this project exists

Credit models are often compared using a single headline score. That is insufficient for lending decisions, where probability calibration, class imbalance, operational thresholds, subgroup behaviour, and validation design can materially affect conclusions.

This project will compare interpretable baselines, tree ensembles, and a neural baseline under one controlled protocol. It is a decision-support and research project, not a lending product and not financial advice.

## Current capabilities

- Validated binary-classification metrics: ROC-AUC, PR-AUC, precision, recall, F1, and Brier score
- Expected calibration error with explicit binning
- Threshold-by-threshold performance reports
- Subgroup performance reports with sample-size visibility
- Repeated group-aware holdout splits that prevent group leakage
- Deterministic synthetic example for testing the workflow
- Unit tests and automated quality checks

## Planned benchmark

The completed benchmark will include:

1. logistic-regression baseline;
2. decision-tree and random-forest baselines;
3. gradient-boosted tree models;
4. a compact neural-network baseline;
5. calibration comparisons;
6. threshold and cost analysis;
7. SHAP and permutation-based interpretation;
8. subgroup performance and fairness analysis;
9. error analysis and sensitivity checks;
10. reproducible experiment tracking.

## Evaluation principles

- Keep train, validation, and test responsibilities separate.
- Use group-aware or temporal validation when the dataset structure requires it.
- Fit preprocessing only on training data.
- Report uncertainty and sample sizes alongside point estimates.
- Treat explanations as descriptive, not causal.
- Never manufacture or copy performance results.

See [docs/evaluation-protocol.md](docs/evaluation-protocol.md) for the planned protocol.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python examples/synthetic_demo.py
```

The synthetic example verifies the mechanics of the evaluation code. It is not evidence of real-world credit performance.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── docs/evaluation-protocol.md
├── examples/synthetic_demo.py
├── src/credit_risk_benchmark/
│   ├── __init__.py
│   ├── metrics.py
│   └── validation.py
├── tests/
│   ├── test_metrics.py
│   └── test_validation.py
├── LICENSE
└── pyproject.toml
```

## Data

No private dissertation or participant data will be used. A suitable public credit-risk dataset will be selected only after its licence, provenance, target definition, time structure, and known limitations have been documented.

## Limitations

- The repository currently provides evaluation foundations, not a completed model benchmark.
- Fairness metrics do not determine whether a lending system is fair, lawful, or suitable for deployment.
- Synthetic data is used only for software tests and examples.
- Any eventual benchmark will be dataset-specific and will not establish real-world underwriting validity.

## Licence

Code is released under the MIT License. Dataset and model licences will be documented separately and may impose additional conditions.

