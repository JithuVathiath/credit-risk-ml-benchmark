from credit_risk_benchmark.cli import build_parser


def test_cli_defaults_to_full_reproducible_run() -> None:
    args = build_parser().parse_args([])

    assert args.mode == "full"
    assert args.seed == 42
    assert args.bootstrap_resamples == 200
    assert not args.without_catboost
