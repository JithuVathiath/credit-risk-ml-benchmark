.PHONY: install test download quick benchmark

install:
	python -m pip install -e ".[dev,benchmark]"

test:
	python -m ruff check .
	python -m pytest --cov=credit_risk_benchmark --cov-report=term-missing
	python examples/synthetic_demo.py

download:
	python scripts/download_data.py

quick:
	credit-risk-benchmark --mode quick --bootstrap-resamples 30 --output-dir artifacts/quick-run

benchmark:
	credit-risk-benchmark --mode full --bootstrap-resamples 200 --output-dir reports
