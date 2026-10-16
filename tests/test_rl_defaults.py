"""Unit tests for train_rl CLI defaults."""

import pytest


def test_deviation_penalty_defaults_to_the_calibrated_weight():
    """The default is the frozen calibrated weight, not 0.0.

    It used to default to 0.0 so the coefficient had to be chosen explicitly by
    every caller -- which left any run that omitted the flag silently
    unregularized, with nothing in its artifacts to say so. The calibrated
    weight now lives in one place and both trainers default to it, so a formal
    run cannot start without one.

    The retired pair (5.0 / 0.5) is still gone: it was tuned for a penalty
    anchored at the box midpoint, while this one is measured from the truthful
    point, which for storage is a box corner.
    """
    from train_rl import build_parser
    from rl_td3 import CALIBRATED_DEV_PENALTY

    args = build_parser().parse_args([])
    assert args.dev_penalty == CALIBRATED_DEV_PENALTY
    assert CALIBRATED_DEV_PENALTY > 0.0
    assert not hasattr(args, "bid_dev_penalty")
    assert not hasattr(args, "offer_dev_penalty")


def test_algo_and_bounds_defaults():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.algo == "matd3"
    assert args.bid_mult_low is None
    assert args.bid_mult_high is None
    assert args.diff_reward is True
    assert args.noise_anneal_steps == 5000


def test_a_run_must_name_its_training_scenarios():
    """There is no default scenario. Which one a run trains on depends on
    which data exists and what is being studied, so it is stated."""
    from train_rl import _resolve_train_scenarios
    import pytest as _pytest
    with _pytest.raises(SystemExit, match="no --scenarios"):
        _resolve_train_scenarios(None)


def test_explicit_scenario_list_must_be_a_formal_subset():
    """An explicit --scenarios list is accepted only when every name is in the
    declared formal set. A run over a scenario priced by a generated series
    would produce a curve that is a result about that generator."""
    import scenarios
    from train_rl import _resolve_train_scenarios
    formal = scenarios.formal_scenarios()
    assert _resolve_train_scenarios(",".join(formal)) == formal
    with pytest.raises(ValueError, match="not in the formal scenario set"):
        _resolve_train_scenarios("baseline,peak_load")


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
