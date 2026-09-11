# Model Card: CatBoost Five-Year SBA 7(a) Charge-Off Benchmark

## Model details

- **Version:** 1.0.0
- **Evaluation date:** 10 September 2026
- **Model family:** categorical gradient-boosted decision trees
- **Selection rule:** highest validation PR-AUC among five prespecified model families
- **Prediction point:** first loan disbursement
- **Outcome:** charge-off within 1,826 days
- **Population:** approved, disbursed SBA 7(a) loans with a complete outcome window

The fitted model itself is not distributed. This card documents a reproducible research
benchmark; the source, hyperparameters, seed, and exact aggregate results are versioned.

## Data partitions

| Partition | Fiscal years | Loans | Charge-off rate |
|---|---:|---:|---:|
| Train | FY2010–FY2014 | 210,671 | 3.81% |
| Validation | FY2015–FY2017 | 168,127 | 4.13% |
| Test | FY2018–FY2020 | 136,027 | 4.02% |

## Untouched temporal-test performance

| Measure | Result |
|---|---:|
| ROC-AUC | 0.913 |
| ROC-AUC 95% bootstrap CI | 0.909–0.917 |
| PR-AUC | 0.385 |
| PR-AUC 95% bootstrap CI | 0.376–0.398 |
| No-skill PR baseline | 0.040 |
| Brier score | 0.0291 |
| Expected calibration error | 0.0114 |
| Validation-selected threshold | 0.194 |
| Precision at threshold | 37.0% |
| Recall at threshold | 69.1% |

The threshold minimises validation cost under an illustrative 5:1 relative penalty for a missed
charge-off versus a false alert. It is not a policy recommendation.

## Intended uses

- Education and portfolio demonstration of temporal credit-risk evaluation
- Reproducible comparison of tabular classification model families
- Research on calibration, threshold sensitivity, vintage drift, and model explanation

## Out-of-scope uses

- Production underwriting, approval, pricing, adverse action, or account management
- Consumer credit or non-SBA portfolio scoring
- Individual lending decisions or causal claims
- Demographic fairness or regulatory-compliance claims

## Limitations and risks

- Only approved loans are observed, creating selection bias relative to applicants.
- FY2018–FY2020 is future data relative to training, but not evidence of performance today.
- Macroeconomic conditions and SBA policies can change beyond the observed vintages.
- The source lacks credit bureau, financial-statement, protected-class, and payment-sequence data.
- Isotonic calibration and threshold selection share the large validation partition.
- Explanations reflect the fitted model and correlated inputs; they are not causal reasons.

## Monitoring expectations

A real implementation would monitor outcome maturity, missingness, category novelty, score drift,
calibration, discrimination, threshold workload, error costs, and authorised protected-group
metrics. Material change would trigger recalibration, challenger review, or withdrawal.
