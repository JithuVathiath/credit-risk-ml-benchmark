# Reproducibility Guide

## Full benchmark

Requirements: Python 3.11+, approximately 1 GB for the source snapshot, and several minutes of
CPU time.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,benchmark]"
python scripts/download_data.py
credit-risk-benchmark --mode full --bootstrap-resamples 200
```

The published run used random seed 42. Exact input hashes and runtime versions are recorded in
`reports/run_metadata.json`. Parallel numerical libraries and future package releases can create
small floating-point differences; conclusions should be compared with reasonable tolerance.

## Quick rehearsal

Quick mode uses deterministic, class-stratified limits of 80,000 training, 30,000 validation,
and 30,000 test loans while preserving the temporal boundaries:

```bash
credit-risk-benchmark --mode quick --bootstrap-resamples 30 --output-dir artifacts/quick-run
```

Quick results are diagnostic and are not the results reported in the README.

## Software-only verification

The unit tests and synthetic demonstration require no SBA data:

```bash
pytest
python examples/synthetic_demo.py
```

## Updating the data snapshot

Treat a quarterly update as a new experiment. Update URLs, sizes, record counts, and checksums in
`data/sources.json`; review schema changes; rerun tests; regenerate all reports; and document any
cohort or result movement. Never overwrite provenance while retaining old metrics.
