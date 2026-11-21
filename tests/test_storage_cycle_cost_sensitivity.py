# test_storage_cycle_cost_sensitivity.py
"""Cost, energy and break-even checks for the cycle-cost snapshot.

These tests do not clear a market. They pin the absolute cost mapping, the
MW-to-MWh conversion, the two-leg cycle-cost accounting, the monotonicity of
the arbitrage LP in the cycle cost, and the corrected breakeven ratio.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arbitrage_lp
import config_loader as cfg
import participant_payoff
from diagnose_typical_day import _window, round_trip_margin_cny_mwh
from models import Agent, MarketConfig, StorageSpec
from storage_cycle_cost_sensitivity import COSTS, _case_config, _schedule_stats


def _storage() -> StorageSpec:
    return StorageSpec(
        e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
        eta_ch=0.92, eta_dis=0.92,
        soc0=0.5, soc_min=0.10, soc_max=0.90)


def _storage_agent() -> Agent:
    zeros = np.zeros(3)
    return Agent(
        name="ESS9", bus=9, is_prosumer=False,
        load_forecast=zeros.copy(), pv_forecast=zeros.copy(),
        load_real=zeros.copy(), pv_real=zeros.copy(),
        bid_value=0.0, offer_cost=0.0, storage=_storage(),
        participant_type="storage")


def test_costs_are_absolute_80_90_100():
    cfg.reload_defaults()
    cfg.reload_scenarios()
    assert COSTS == {"C80": 80.0, "C90": 90.0, "C100": 100.0}

    base = MarketConfig.from_yaml()
    cheaper = _case_config(80.0)
    assert base.storage.cycle_cost == pytest.approx(100.0)
    assert cheaper.storage.cycle_cost == pytest.approx(80.0)

    base_fields = dataclasses.asdict(base)
    cheaper_fields = dataclasses.asdict(cheaper)
    base_fields["storage"]["cycle_cost"] = 80.0
    assert cheaper_fields == base_fields


def test_schedule_stats_convert_power_to_quarter_hour_energy():
    schedule = {
        "p_ch": np.array([0.4, 0.0, 0.0, 0.0]),
        "p_dis": np.array([0.0, 0.4, 0.0, 0.0]),
        "p_buy": np.array([0.4, 0.0, 0.0, 0.0]),
        "p_sell": np.array([0.0, 0.4, 0.0, 0.0]),
        "soc": np.array([0.55, 0.50, 0.50, 0.50]),
        "soc_boundary": np.array([0.50, 0.55, 0.50, 0.50, 0.50]),
    }
    stats = _schedule_stats(schedule, _storage())
    assert stats["charge_mwh"] == pytest.approx(0.4 * 0.25)
    assert stats["discharge_mwh"] == pytest.approx(0.4 * 0.25)
    assert stats["charge_periods"] == 1
    assert stats["discharge_periods"] == 1
    assert stats["idle_periods"] == 2
    assert stats["idle_share"] == pytest.approx(0.5)
    assert stats["equivalent_cycles"] == pytest.approx(
        (0.1 + 0.1) / (2.0 * 1.6))
    assert stats["soc_initial"] == pytest.approx(0.50)
    assert stats["soc_terminal"] == pytest.approx(0.50)
    assert stats["soc_min"] == pytest.approx(0.50)
    assert stats["soc_max"] == pytest.approx(0.55)
    assert stats["max_charge_mw"] == pytest.approx(0.4)
    assert stats["max_discharge_mw"] == pytest.approx(0.4)
    assert stats["simultaneous_periods"] == 0


def test_cycle_cost_is_charged_on_both_legs_per_period():
    lmp = np.array([100.0, 200.0, 150.0])
    ch = np.array([1.0, 0.0, 0.0])
    dis = np.array([0.0, 0.5, 0.0])
    zeros = np.zeros(3)
    schedule = {
        "served": zeros.copy(), "unserved": zeros.copy(),
        "pv_used": zeros.copy(), "wind_used": zeros.copy(),
        "p_buy": ch.copy(), "p_sell": dis.copy(),
        "p_ch": ch.copy(), "p_dis": dis.copy(), "soc": zeros.copy(),
    }
    config = MarketConfig()
    config.storage.cycle_cost = 80.0
    breakdown = participant_payoff.participant_payoff(
        schedule, lmp, _storage_agent(), config)

    assert breakdown.revenue == pytest.approx(200.0 * 0.5 * 0.25)
    assert breakdown.purchase_cost == pytest.approx(100.0 * 1.0 * 0.25)
    assert breakdown.degradation_cost == pytest.approx(
        80.0 * (1.0 + 0.5) * 0.25)
    assert breakdown.generation_cost == 0.0
    assert breakdown.penalty == 0.0
    assert breakdown.other_cost == 0.0
    assert breakdown.energy_sold_mwh == pytest.approx(0.5 * 0.25)
    assert breakdown.energy_bought_mwh == pytest.approx(1.0 * 0.25)
    assert breakdown.total == pytest.approx(
        breakdown.revenue - breakdown.purchase_cost
        - breakdown.degradation_cost)


def test_arbitrage_profit_is_non_increasing_in_cycle_cost():
    unit = arbitrage_lp.UnitSpec(
        e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
        eta_ch=0.92, eta_dis=0.92,
        soc0=0.5, soc_min=0.10, soc_max=0.90)
    price = np.array([100.0] * 24 + [250.0] * 24
                     + [400.0] * 24 + [250.0] * 24)
    margins = {
        cost: arbitrage_lp.arbitrage_lp(price, unit, cost).margin_cny
        for cost in (80.0, 90.0, 100.0)}
    assert margins[80.0] >= margins[90.0] - 1e-6
    assert margins[90.0] >= margins[100.0] - 1e-6
    for cost in (80.0, 90.0, 100.0):
        assert margins[cost] > 0.0

    required = {
        cost: arbitrage_lp.required_margin(cost, 0.92, 0.92)
        for cost in (80.0, 90.0, 100.0)}
    assert required[80.0] < required[90.0] < required[100.0]


def test_window_ratio_matches_the_cny_per_mwh_margin():
    price = np.array([225.90, 445.00])
    window = _window(price, 0.92, 0.92, 100.0)
    rtm = round_trip_margin_cny_mwh(price, 0.92, 0.92, 100.0)

    required = arbitrage_lp.required_margin(100.0, 0.92, 0.92)
    corrected = ((225.90 / 0.92 + required) / (0.92 * 225.90))
    old = ((225.90 / (0.92 * 0.92) + required) / (0.92 * 225.90))

    assert window["breakeven_ratio"] == pytest.approx(corrected)
    assert window["breakeven_ratio"] != pytest.approx(old)
    assert window["margin"] * 0.92 * 225.90 == pytest.approx(
        rtm, abs=1e-9)
    assert window["margin"] * 0.92 * 225.90 == pytest.approx(
        -36.8391, abs=1e-3)
    assert (old - corrected) * 0.92 * 225.90 == pytest.approx(
        21.3516, abs=1e-3)


@pytest.mark.parametrize("price", [
    np.array([100.0, 200.0]),
    np.array([50.0, 400.0]),
    np.array([225.90, 445.00]),
    np.array([300.0, 300.0]),
])
def test_window_and_round_trip_margin_agree_in_sign(price):
    window = _window(price, 0.92, 0.92, 100.0)
    rtm = round_trip_margin_cny_mwh(price, 0.92, 0.92, 100.0)
    assert np.sign(window["margin"]) == np.sign(rtm)
