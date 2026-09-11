# Responsible-AI and Fairness Scope

## What can be assessed

This project reports predictive performance and errors across approval fiscal year, business
type, and harmonised business-age band. These are operational subgroup checks used to expose
instability and sparse segments.

## What cannot be assessed

The SBA FOIA extract used here does not provide protected-class attributes. Consequently, this
repository does **not** claim demographic fairness, equal opportunity, disparate-impact
compliance, or fairness of SBA approval decisions. The source contains approved loans only, so
even protected attributes would not identify disparities in who was denied or never applied.

Calling the available subgroup table a demographic fairness audit would be misleading. The
repository therefore treats missing fairness evidence as a documented limitation, not a metric
to invent or proxy through geography.

## Prohibited interpretations

- Do not use project state, bank state, NAICS sector, or lender identifier as proxies for race,
  ethnicity, sex, disability, or another protected characteristic.
- Do not infer that an explanation is causal or that a feature is an appropriate underwriting
  reason.
- Do not deploy the reported threshold: its 5:1 cost ratio is illustrative and excludes capital,
  recovery, pricing, legal, and customer-impact considerations.
- Do not apply the model to applicants, consumer credit, or non-SBA portfolios.

## A real deployment would additionally require

Representative current data; explicit legal review; protected-class analysis using authorised
data and suitable methodology; outcome and selection-bias analysis; adverse-action governance;
challenger monitoring; human oversight; security review; and a documented rollback process.
