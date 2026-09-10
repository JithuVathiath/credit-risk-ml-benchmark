# Evaluation Protocol

## Objective

Compare candidate credit-risk models under a single, documented protocol that separates predictive discrimination, probability quality, decision thresholds, subgroup behaviour, and operational considerations.

## Dataset acceptance gate

Before modelling, document:

- source and licence;
- unit of observation;
- outcome definition and observation window;
- feature availability time;
- missing-data mechanisms;
- repeated applicants or other grouping;
- time range and temporal drift;
- protected or sensitive attributes and their permitted research use;
- known collection and representation limitations.

## Splitting strategy

The final choice must follow the dataset's structure:

- temporal holdout when future performance is the target;
- group-aware splitting for repeated entities;
- a fixed untouched test set for final reporting;
- cross-validation only within training and validation data;
- preprocessing fitted inside each training fold.

## Model families

- Logistic regression as the primary interpretable baseline
- Decision tree and random forest
- Gradient-boosted trees
- Compact multilayer perceptron
- Any modern tabular model only when compute and licensing permit fair comparison

## Required outputs

- ROC-AUC and PR-AUC
- Precision, recall, and F1 at documented thresholds
- Brier score, reliability curves, and expected calibration error
- Confusion matrices and threshold tables
- Bootstrap confidence intervals where justified
- Subgroup sample sizes and performance
- Error analysis
- Explanation method and stability analysis
- Training and inference time
- Reproducibility metadata

## Interpretation safeguards

Predictive performance does not establish causality, fairness, legality, or deployment suitability. Subgroup metrics depend on dataset construction and sample size. Explanation methods describe model behaviour and must not be presented as reasons inherent to an applicant.

