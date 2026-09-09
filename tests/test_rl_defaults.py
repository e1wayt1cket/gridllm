"""Unit tests for train_rl CLI defaults."""

import pytest


def test_deviation_penalty_defaults():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.bid_dev_penalty == 5.0
    assert args.offer_dev_penalty == 0.5


def test_algo_and_bounds_defaults():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.algo == "matd3"
    assert args.bid_mult_low is None
    assert args.bid_mult_high is None
    assert args.diff_reward is True
    assert args.noise_anneal_steps == 5000


def test_default_training_scenario_is_fixed():
    """Default training uses a single fixed scenario, not rotation."""
    from train_rl import _resolve_train_scenarios, DEFAULT_TRAIN_SCENARIO
    assert _resolve_train_scenarios(None) == [DEFAULT_TRAIN_SCENARIO]


def test_explicit_scenario_list_parses():
    """An explicit --scenarios list restores rotation across that subset."""
    from train_rl import _resolve_train_scenarios
    assert _resolve_train_scenarios("baseline,peak_load") == \
        ["baseline", "peak_load"]
    assert _resolve_train_scenarios("peak_load") == ["peak_load"]


def test_qmatd3_is_accepted_algo_choice():
    from train_rl import build_parser
    args = build_parser().parse_args(["--algo", "qmatd3"])
    assert args.algo == "qmatd3"


def test_n_quantiles_default():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.n_quantiles == 32


def test_masac_is_accepted_algo_choice():
    from train_rl import build_parser
    args = build_parser().parse_args(["--algo", "masac"])
    assert args.algo == "masac"


def test_masac_hyperparameter_defaults():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.alpha_init == 0.1
    assert args.target_entropy == -2.0
    assert args.physics is False


def test_capacity_default_and_parse():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.capacity is None
    args2 = build_parser().parse_args(["--capacity", "1.0"])
    assert args2.capacity == 1.0


def test_market_impact_penalty_default_and_parse():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.market_impact_penalty == 0.0
    args2 = build_parser().parse_args(["--market-impact-penalty", "0.3"])
    assert args2.market_impact_penalty == 0.3
