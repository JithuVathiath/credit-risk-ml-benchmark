# Evaluation Protocol

## Estimand

Estimate the probability that an already-approved and disbursed SBA 7(a) loan is charged off
within 1,826 days of first disbursement. This is a fixed-horizon performance benchmark—not an
approval model, survival model, or underwriting policy.

## Cohort construction

1. Read only the fields required for cohort construction, modelling, or aggregate auditing.
2. Exclude cancelled and undisbursed commitments.
3. Require a valid first-disbursement date and a complete 1,826-day follow-up window at the
   30 June 2026 snapshot date.
4. Exclude charged-off records without a usable charge-off date because their event horizon
   cannot be determined.
5. Label charge-off on or before day 1,826 as `1`; otherwise label the loan `0`, including a
   charge-off after the horizon.
6. Restrict model development to FY2010–FY2020 because key origination fields are absent from
   older extracts. Earlier mature vintages remain in the drift analysis.

## Leakage controls

The following lifecycle fields are available only for label construction and never enter a
feature matrix:

- current loan status;
- paid-in-full date;
- charge-off date;
- gross charge-off amount;
- snapshot date; and
- secondary-market sale indicator, which can change after origination.

Borrower names and addresses are never loaded. Preprocessing is fit only on training data.

## Temporal validation

| Partition | SBA approval fiscal years | Purpose |
|---|---|---|
| Train | FY2010–FY2014 | Fit preprocessing and model parameters |
| Validation | FY2015–FY2017 | Fit isotonic calibration, select model and threshold |
| Test | FY2018–FY2020 | One final future-vintage evaluation |

No random split can substitute for the temporal test because vintage charge-off rates and
feature vocabularies change materially over time.

## Models

- Logistic regression: interpretable linear baseline on one-hot encoded features
- Random forest: bagged tree ensemble
- Histogram gradient boosting: boosted-tree baseline
- CatBoost: categorical gradient boosting with early stopping
- Multilayer perceptron: compact neural baseline

The winning model is selected by validation PR-AUC. This metric is primary because charge-offs
are uncommon; ROC-AUC, probability quality, and operational metrics remain essential secondary
evidence.

## Calibration and decisions

Each raw score is mapped to a probability using isotonic regression fit on the validation
partition. A threshold is then selected on validation under an illustrative 5:1 relative cost
for a missed charge-off versus a false alert. This ratio is a sensitivity scenario, not an SBA
policy or an economic estimate.

## Required outputs

- ROC-AUC and PR-AUC, including a 95% stratified bootstrap interval for the selected model
- Brier score, log loss, expected calibration error, and reliability curves
- Precision, recall, F1, balanced accuracy, and confusion matrix at the selected threshold
- Threshold-cost curves and timing measurements
- Test performance by fiscal year and non-protected operational subgroup
- Vintage outcome drift, SHAP importance, permutation importance, and aggregate error analysis
- Exact data checksums, package versions, random seed, and runtime metadata

## Interpretation safeguards

Predictive performance does not establish causality, fairness, legality, or deployment fitness.
Explanations describe model behaviour and must not be presented as reasons inherent to a
business or borrower. The public files contain approved loans only, so selection into SBA
lending is outside the estimand.
