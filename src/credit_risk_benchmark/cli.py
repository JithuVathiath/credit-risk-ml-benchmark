"""Command-line entry point for data verification and benchmark execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import BenchmarkConfig, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk-benchmark",
        description="Run a leakage-safe SBA 7(a) five-year charge-off benchmark.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--without-catboost", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_benchmark(
        args.data_dir,
        args.output_dir,
        config=BenchmarkConfig(
            mode=args.mode,
            random_state=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            include_catboost=not args.without_catboost,
        ),
    )
    partitions = metadata["partitions"]
    print(
        "Benchmark complete:",
        f"{partitions['train']['loans']:,} train /",
        f"{partitions['validation']['loans']:,} validation /",
        f"{partitions['test']['loans']:,} test loans;",
        f"best model = {metadata['best_model_by_validation_pr_auc']}",
    )


if __name__ == "__main__":
    main()
