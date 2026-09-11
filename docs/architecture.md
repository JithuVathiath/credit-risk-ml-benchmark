# Architecture

```mermaid
flowchart LR
    A[Official SBA snapshot<br/>4 CSV files] --> B[Checksum verification]
    B --> C[Chunked, privacy-minimised ingestion]
    C --> D[Five-year outcome construction]
    D --> E[Temporal partitions]
    E --> F[Train<br/>FY2010-2014]
    E --> G[Validation<br/>FY2015-2017]
    E --> H[Test<br/>FY2018-2020]
    F --> I[Five model families]
    G --> J[Isotonic calibration<br/>model + threshold selection]
    I --> J
    H --> K[Untouched final evaluation]
    J --> K
    K --> L[Metrics and confidence intervals]
    K --> M[Drift and subgroup monitoring]
    K --> N[SHAP + permutation explanations]
    L --> O[Versioned aggregate reports]
    M --> O
    N --> O
```

## Design choices

- **Chunked reads:** all four files are scanned without loading direct identifiers or the entire
  raw schema.
- **One cohort function:** tests exercise the same eligibility and label code used by the full
  benchmark.
- **Separate partitions:** fitting, calibration/selection, and final evaluation have explicit
  responsibilities.
- **Aggregate artifacts:** no row-level predictions are written to disk.
- **Pinned provenance:** a fresh SBA quarterly release cannot silently replace the benchmarked
  snapshot.
