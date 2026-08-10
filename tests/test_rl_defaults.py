"""Unit tests for train_rl CLI defaults."""

import pytest


def test_deviation_penalty_defaults():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.bid_dev_penalty == 5.0
    assert args.offer_dev_penalty == 0.5
