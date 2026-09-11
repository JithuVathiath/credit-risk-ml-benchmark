# Reproducible Benchmark Summary

Generated from the official SBA 7(a) snapshot using `full` mode and random seed
`42`. The test set was not used for fitting, calibration, model selection,
or threshold selection.

## Cohort

| Partition | Approval vintages | Loans | Five-year charge-off rate |
|---|---:|---:|---:|
| Train | FY2010–FY2014 | 210,671 | 3.81% |
| Validation | FY2015–FY2017 | 168,127 | 4.13% |
| Test | FY2018–FY2020 | 136,027 | 4.02% |

## Headline result

`catboost` ranked first by validation PR-AUC. On the untouched temporal test it achieved:

- ROC-AUC: **0.913** (95% bootstrap CI 0.909–0.917)
- PR-AUC: **0.385** (no-skill baseline 0.040)
- Brier score after isotonic calibration: **0.0291**
- Recall at the validation-selected 5:1 cost threshold: **69.1%**
- Precision at that threshold: **37.0%**

The relative cost ratio is an analytical scenario, not an SBA policy or production decision rule.

## Interpretation boundary

This estimates charge-off within five years among already-approved and disbursed SBA 7(a) loans.
It does not estimate approval likelihood, establish causality, or validate an underwriting system.
The source does not include protected-class attributes, so the subgroup report is performance
monitoring—not a demographic fairness assessment.
