# test_autumn_pairing_audit.py
"""The autumn audit's distance statistics, bottleneck and certificate."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_autumn_pairing as aa


def test_stats_counts_both_tails():
    pairs = {"2022-07-04": "2025-07-04",     # 0 days
             "2022-07-05": "2025-07-26",     # 21 days
             "2022-07-06": "2025-09-15"}     # >30 days
    result = aa.stats(pairs)
    assert result["n"] == 3
    assert result["max"] == 71
    assert result["median"] == 21
    assert result["over_14"] == 2
    assert result["over_30"] == 1


def test_autumn_pool_drops_the_reserved_national_day_dates():
    coverage = ([{"date": d, "complete": "True"}
                 for d in ("2025-10-01", "2025-10-02", "2025-10-03",
                           "2025-10-05", "2025-10-08", "2025-11-20")]
                + [{"date": "2025-10-04", "complete": "False"}])
    pool = aa.autumn_pool(coverage)
    assert "2025-10-03" not in pool
    assert "2025-10-05" not in pool
    assert "2025-10-04" not in pool          # incomplete
    assert "2025-10-08" in pool


def test_the_bottleneck_is_the_smallest_feasible_limit():
    loads = ["2022-07-04", "2022-07-05"]     # Monday, Tuesday
    prices = ["2025-07-04", "2025-07-07", "2025-07-08"]  # Fri, Mon, Tue
    # The best matching gives the Monday load its own date (0 days) and the
    # Tuesday load 2025-07-07 (2 days); no matching keeps both within 1 day.
    assert aa.bottleneck(loads, prices) == 2


def test_min_max_respects_the_limit_and_the_type():
    loads = ["2022-07-05"]                   # Tuesday
    prices = ["2025-07-05", "2025-07-08"]    # Saturday, Tuesday
    assignment = aa.min_max(loads, prices, limit=3)
    assert assignment["2022-07-05"] == "2025-07-08"


def test_the_certificate_matches_the_linear_assignment():
    loads = ["2022-07-04", "2022-07-05", "2022-07-06"]
    prices = ["2025-07-04", "2025-07-07", "2025-07-08", "2025-07-09"]
    cost = aa._cost(loads, prices)
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost)
    lap = float(cost[rows, cols].sum())

    certificate = aa.milp_certificate(loads, prices)
    assert certificate["available"] is True
    assert certificate["optimal"] is True
    assert certificate["objective"] == pytest.approx(lap)
    assert certificate["mip_gap"] == pytest.approx(0.0)
