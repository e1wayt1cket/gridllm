# test_lmp_fallback_report.py
"""A day whose prices are partly not nodal has to say so.

`lmp_fallbacks` reported how many price cells came off the wholesale curve and
nothing else: not which periods, not why, and it was absent altogether on the
paths that never extracted a dual -- so a reader of `lmp` could not tell a
nodal price field from a mixed one. These tests pin the report, and pin the
rule that a mixed field is not a settled result.
"""

from __future__ import annotations

import numpy as np
import pytest

from counterfactual_eval import is_degraded
from dispatch_core import (LMP_FALLBACK_NO_DUAL, LMP_FALLBACK_UNSOLVED,
                           lmp_fallback_report)
from market import clear_da_rolling
from models import MarketConfig
from scenarios import get_scenario


def _clear(T, opf_mode="socp"):
    config = MarketConfig(opf_mode=opf_mode, verbose=False)
    agents, wholesale = get_scenario("baseline", T=T)
    from market import clear_market
    result = clear_market(agents, T, "DA", {}, config,
                          wholesale=wholesale, horizon_type="full_day")
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------

def test_a_nodal_day_reports_itself_as_nodal():
    report = lmp_fallback_report(0, [], LMP_FALLBACK_NO_DUAL)
    assert report["has_wholesale_fallback"] is False
    assert report["lmp_price_source"] == "nodal"
    assert report["lmp_fallback_periods"] == []
    assert report["lmp_fallback_reason"] == ""


def test_a_mixed_day_names_the_periods_and_the_cause():
    report = lmp_fallback_report(3, [7, 2, 7], LMP_FALLBACK_NO_DUAL)
    assert report["has_wholesale_fallback"] is True
    assert report["lmp_price_source"] == "mixed_wholesale_fallback"
    # Distinct periods, in order, and the count is what the caller measured --
    # it counts cells, which is what the field has always meant.
    assert report["lmp_fallback_periods"] == [2, 7]
    assert report["lmp_fallbacks"] == 3
    assert report["lmp_fallback_reason"] == LMP_FALLBACK_NO_DUAL


def test_the_two_causes_are_different_statements():
    """A missing dual and an unsolved period call for different responses."""
    assert LMP_FALLBACK_NO_DUAL != LMP_FALLBACK_UNSOLVED


# ---------------------------------------------------------------------------
# Every clearing path carries it
# ---------------------------------------------------------------------------

def test_the_batch_engine_reports_the_marker_on_a_clean_day():
    result = _clear(16)
    for key in ("lmp_fallbacks", "lmp_fallback_periods",
                "lmp_fallback_reason", "has_wholesale_fallback",
                "lmp_price_source"):
        assert key in result, key
    # The reported field must describe the prices it ships with, whichever way
    # this day happened to clear.
    assert result["has_wholesale_fallback"] == bool(result["lmp_fallbacks"] > 0)
    assert result["lmp_price_source"] == (
        "mixed_wholesale_fallback" if result["has_wholesale_fallback"]
        else "nodal")


def test_the_per_period_path_reports_it_too():
    """This path used to emit no `lmp_fallbacks` at all.

    A reader that used `.get(..., 0)` on it read zero -- "every price is
    nodal" -- on a path whose prices are carried forward from the period
    before whenever a solve fails.
    """
    result = _clear(16, opf_mode="dc")
    for key in ("lmp_fallbacks", "lmp_fallback_periods",
                "lmp_fallback_reason", "has_wholesale_fallback",
                "lmp_price_source"):
        assert key in result, key
    # And the fields describe the same field of prices rather than each other.
    assert result["has_wholesale_fallback"] == bool(
        result["lmp_fallback_periods"])
    assert result["lmp_fallbacks"] >= 0


def test_a_period_without_a_dual_is_reported_as_not_nodal(monkeypatch):
    """The condition the report exists for, forced at the dual itself.

    A missing dual is what a fallback actually is, so the failure is injected
    there rather than by handing the engine a result dict. Nothing else about
    the clear changes.
    """
    import money

    real = money.price_from_balance_dual
    seen = {"n": 0}

    def flaky(dual, dt=money.DT_HOURS):
        seen["n"] += 1
        if seen["n"] % 3 == 0:
            raise AttributeError("no dual for this balance constraint")
        return real(dual, dt)

    monkeypatch.setattr(money, "price_from_balance_dual", flaky)
    result = _clear(16)
    assert result["has_wholesale_fallback"] is True
    assert result["lmp_fallback_periods"], "the fallback was not located"
    assert result["lmp_price_source"] == "mixed_wholesale_fallback"
    assert result["lmp_fallback_reason"] == LMP_FALLBACK_NO_DUAL
    # And the periods are the ones whose prices actually came off the curve.
    assert max(result["lmp_fallback_periods"]) < 16


def test_a_rolling_clear_carries_the_marker():
    """A rolling day is assembled from windows; the marker has to survive."""
    config = MarketConfig(opf_mode="socp", verbose=False)
    agents, wholesale = get_scenario("baseline", T=8)
    config.rt.da_rolling_enabled = True
    config.rt.da_window_length = 4
    config.rt.da_window_step = 2
    result = clear_da_rolling(agents, 8, {}, config, wholesale=wholesale)
    assert result is not None
    assert "has_wholesale_fallback" in result
    assert isinstance(result["lmp_fallback_periods"], list)
    assert result["has_wholesale_fallback"] == bool(
        result["lmp_fallback_periods"])


# ---------------------------------------------------------------------------
# What a mixed field is allowed to be called
# ---------------------------------------------------------------------------

def test_a_mixed_price_field_is_not_a_settled_result():
    assert is_degraded({"has_wholesale_fallback": True}) is True
    assert is_degraded({"fell_back": True}) is True
    assert is_degraded(None) is True


def test_a_nodal_result_is_a_settled_result():
    assert is_degraded({"lmp_fallbacks": 0, "has_wholesale_fallback": False}) \
        is False
    assert is_degraded({}) is False


def test_the_audit_keeps_which_periods_and_why_not_just_how_many():
    """`clear_day` hands the whole report on, not a reduced count."""
    import market_opportunity as mo
    from config_loader import reload_defaults
    import wholesale_price as wp

    reload_defaults()
    wp.reset_cache()
    try:
        spec = mo.ensemble_specs(1, 5)[0]
        cleared = mo.clear_day(spec, with_fleet=False, T=96)
        for key in ("lmp_fallbacks", "lmp_fallback_periods",
                    "lmp_fallback_reason", "lmp_price_source",
                    "has_wholesale_fallback"):
            assert key in cleared, key
        assert cleared["has_wholesale_fallback"] == bool(
            cleared["lmp_fallback_periods"])
        assert cleared["lmp_price_source"] == (
            "mixed_wholesale_fallback" if cleared["has_wholesale_fallback"]
            else "nodal")
    finally:
        reload_defaults()
        wp.reset_cache()


def test_the_per_bus_table_carries_the_price_source():
    import market_opportunity as mo

    unit = mo.audited_unit(9)
    price = np.full(96, 400.0)
    row = mo.bus_day_metrics(price, unit, 100.0, day=0, bus=9,
                             lmp_price_source="mixed_wholesale_fallback",
                             lmp_fallback_periods=[3, 4])
    assert row["lmp_price_source"] == "mixed_wholesale_fallback"
    assert row["lmp_fallback_periods"] == [3, 4]
    assert "lmp_price_source" in mo.REPORT_COLUMNS
    assert "lmp_fallback_periods" in mo.REPORT_COLUMNS
    assert set(row) == set(mo.REPORT_COLUMNS)
