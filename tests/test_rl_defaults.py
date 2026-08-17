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
    assert args.algo == "td3"
    assert args.bid_mult_low is None
    assert args.bid_mult_high is None
    assert args.diff_reward is True
    assert args.noise_anneal_steps == 5000
