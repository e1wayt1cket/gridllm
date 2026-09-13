# tests/test_counterfactual_eval.py
"""The paired benefit comparison.

The number this produces is the paper's headline, so the tests here are mostly
about what the comparison refuses to do: compare arms that saw different days,
or average in a day whose clearing degraded.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import counterfactual_eval as ce


# --------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------
def test_benefit_is_the_difference_in_settled_profit():
    assert ce.benefit(150.0, 100.0) == pytest.approx(50.0)
    assert ce.benefit(-50.0, -100.0) == pytest.approx(50.0)


def test_benefit_rate_uses_the_baseline_magnitude():
    # Improving on a money-losing baseline is a positive rate, not a negative
    # one: a battery whose day does not cover its degradation still gained.
    assert ce.benefit_rate(150.0, 100.0) == pytest.approx(0.5)
    assert ce.benefit_rate(-50.0, -100.0) > 0
    assert ce.benefit_rate(-50.0, -100.0) == pytest.approx(0.5, rel=1e-4)


def test_benefit_rate_survives_a_zero_baseline():
    rate = ce.benefit_rate(10.0, 0.0)
    assert np.isfinite(rate) and rate > 0


# --------------------------------------------------------------------------
# Distribution
# --------------------------------------------------------------------------
def test_summarize_reports_the_shape_of_the_gain():
    values = [10.0, -5.0, 20.0, -2.0, 30.0, 5.0]
    s = ce.summarize(values)
    assert s["n"] == 6
    assert s["mean"] == pytest.approx(np.mean(values))
    assert s["median"] == pytest.approx(np.median(values))
    assert s["p05"] <= s["p25"] <= s["p50"] <= s["p75"] <= s["p95"]
    assert s["positive_rate"] == pytest.approx(4 / 6)


def test_summarize_separates_a_stable_gain_from_a_lucky_one():
    # Same mean, different findings: one is reliable, the other is not. Two
    # large gains paid for by three losses average out to the same number and
    # mean something entirely different.
    stable = ce.summarize([10.0, 11.0, 9.0, 10.5, 9.5])
    lucky = ce.summarize([100.0, 90.0, -40.0, -45.0, -55.0])
    assert stable["mean"] == pytest.approx(lucky["mean"])
    assert stable["positive_rate"] == 1.0
    assert lucky["positive_rate"] == pytest.approx(2 / 5)
    assert lucky["std"] > stable["std"]


def test_summarize_of_nothing_is_nan_not_zero():
    s = ce.summarize([])
    assert all(np.isnan(s[c]) for c in ce.STAT_COLUMNS if c != "n")


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------
def test_paired_arms_must_share_the_price_day():
    day = {"arm": "truthful", "wholesale": np.array([1.0, 2.0]),
           "seed": 1}
    other = {"arm": "rl", "wholesale": np.array([1.0, 2.0]), "seed": 1}
    ce.assert_paired(day, other)          # identical days pass

    different = {"arm": "rl", "wholesale": np.array([1.0, 9.0]), "seed": 1}
    with pytest.raises(ValueError, match="same price day"):
        ce.assert_paired(day, different)

    shorter = {"arm": "rl", "wholesale": np.array([1.0]), "seed": 1}
    with pytest.raises(ValueError, match="same price day"):
        ce.assert_paired(day, shorter)


# --------------------------------------------------------------------------
# Headline
# --------------------------------------------------------------------------
def test_headline_leads_with_the_three_numbers_the_question_needs():
    result = {
        "n_days": 3, "seeds_dropped": [9],
        "fleet": {"benefit": {"mean": 42.0, "positive_rate": 2 / 3},
                  "benefit_rate": {"mean": 0.25}},
    }
    h = ce.headline(result)
    assert h["mean_benefit"] == pytest.approx(42.0)
    assert h["mean_benefit_rate"] == pytest.approx(0.25)
    assert h["positive_rate"] == pytest.approx(2 / 3)
    assert h["n_days_dropped"] == 1


def test_write_csv_puts_the_headline_columns_first(tmp_path):
    result = {
        "scenario": "baseline", "ai_arm": "rl", "baseline_arm": "truthful",
        "n_days": 2, "seeds_dropped": [],
        "fleet": {q: ce.summarize([1.0, 2.0]) for q in
                  ("baseline_profit", "ai_profit", "benefit", "benefit_rate")},
        "per_agent": {"ESS2": {q: ce.summarize([1.0, 2.0]) for q in
                               ("baseline_profit", "ai_profit", "benefit",
                                "benefit_rate")}},
    }
    path = str(tmp_path / "benefit.csv")
    ce.write_csv(result, path)
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        body = fh.read().strip().splitlines()
    assert header[:7] == ["scenario", "ai_arm", "baseline_arm", "agent",
                          "n_days", "n_days_dropped", "mean_benefit"]
    assert len(body) == 2                      # the fleet row and one agent
    assert body[0].startswith("baseline,rl,truthful,ALL")
