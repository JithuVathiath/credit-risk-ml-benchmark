# Contributing

Contributions that improve reproducibility, validation, documentation, or responsible use are
welcome.

1. Open an issue describing the proposed change and its effect on the estimand or protocol.
2. Create a focused branch and add tests for behavioural changes.
3. Run `make test` before submitting a pull request.
4. Do not commit raw SBA files, borrower identifiers, row-level predictions, or proprietary data.
5. Regenerated benchmark results must include updated provenance and must never be fabricated.

Changes to the label, temporal partitions, costs, or model-selection rule are protocol changes and
must be clearly separated from comparisons to the existing published run.
