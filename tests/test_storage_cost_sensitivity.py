# test_storage_cost_sensitivity.py
"""The C0/C1/C2 mapping and the accounting the sensitivity report depends on.

These tests do not clear a market. They pin the cost mapping, the break-even
margin, the SOC summary and the fleet aggregation, so a wrong multiplier or a
summed-wrong row is caught before a Gurobi run is spent on it.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arbitrage_lp
import storage_cost_sensitivity as scs
from models import MarketConfig


def test_cases_are_the_declared_multipliers():
    assert scs.CASES == {"C0": 1.0, "C1": 0.9, "C2": 0.8}
    assert scs.resolve_case("C0", 100.0) == 100.0
    assert scs.resolve_case("C1", 100.0) == pytest.approx(90.0)
    assert scs.resolve_case("c2", 100.0) == pytest.approx(80.0)


def test_an_unknown_case_is_refused():
    with pytest.raises(ValueError, match="unknown storage-cost case"):
        scs.resolve_case("C3", 100.0)


def test_required_margin_moves_with_the_cycle_cost():
    assert arbitrage_lp.required_margin(100.0, 0.92, 0.92) == \
        pytest.approx(200.695652, rel=1e-6)
    assert arbitrage_lp.required_margin(90.0, 0.92, 0.92) == \
        pytest.approx(180.626087, rel=1e-6)
    assert arbitrage_lp.required_margin(80.0, 0.92, 0.92) == \
        pytest.approx(160.556522, rel=1e-6)


def test_the_config_fingerprint_changes_with_the_cycle_cost():
    from counterfactual_eval import _config_fingerprint

    base = MarketConfig.from_yaml()
    cheaper = dataclasses.replace(
        base, storage=dataclasses.replace(base.storage, cycle_cost=90.0))
    assert _config_fingerprint(base) != _config_fingerprint(cheaper)


def test_soc_summary_reports_bounds_and_terminal():
    import numpy as np

    stats = scs._soc_stats(np.array([0.10, 0.50, 0.90]), 0.10, 0.90)
    assert stats["soc_min"] == pytest.approx(0.10)
    assert stats["soc_max"] == pytest.approx(0.90)
    assert stats["soc_mean"] == pytest.approx(0.50)
    assert stats["soc_terminal"] == pytest.approx(0.90)
    assert stats["soc_bound_periods"] == 2


def test_fleet_row_sums_throughput_and_profit():
    rows = [
        {"case": "C0", "cycle_cost_cny_mwh": 100.0,
         "required_margin_cny_mwh": 217.4, "agent": "ESS9", "e_max_mwh": 1.6,
         "p_dis_max_mw": 0.4, "charge_mwh": 1.0, "discharge_mwh": 0.8,
         "equivalent_cycles": 0.5625, "cycles_from_discharge": 0.5,
         "utilization": 0.0833, "soc_min": 0.1, "soc_max": 0.9,
         "soc_mean": 0.5, "soc_terminal": 0.5, "soc_bound_periods": 2,
         "profit_cny": 10.0, "gross_margin_cny": 12.0,
         "degradation_cost_cny": 2.0, "revenue_cny": 12.0,
         "purchase_cost_cny": 0.0, "energy_sold_mwh": 0.8,
         "energy_bought_mwh": 1.0, "clean": True},
        {"case": "C0", "cycle_cost_cny_mwh": 100.0,
         "required_margin_cny_mwh": 217.4, "agent": "ESS14", "e_max_mwh": 1.6,
         "p_dis_max_mw": 0.4, "charge_mwh": 2.0, "discharge_mwh": 1.6,
         "equivalent_cycles": 1.125, "cycles_from_discharge": 1.0,
         "utilization": 0.1667, "soc_min": 0.2, "soc_max": 0.8,
         "soc_mean": 0.5, "soc_terminal": 0.5, "soc_bound_periods": 0,
         "profit_cny": 20.0, "gross_margin_cny": 24.0,
         "degradation_cost_cny": 4.0, "revenue_cny": 24.0,
         "purchase_cost_cny": 0.0, "energy_sold_mwh": 1.6,
         "energy_bought_mwh": 2.0, "clean": True},
    ]
    fleet = scs._fleet_row(rows)
    assert fleet["agent"] == "fleet"
    assert fleet["charge_mwh"] == pytest.approx(3.0)
    assert fleet["discharge_mwh"] == pytest.approx(2.4)
    assert fleet["profit_cny"] == pytest.approx(30.0)
    assert fleet["gross_margin_cny"] == pytest.approx(36.0)
    assert fleet["degradation_cost_cny"] == pytest.approx(6.0)
    assert fleet["e_max_mwh"] == pytest.approx(3.2)
    assert fleet["soc_min"] == pytest.approx(0.1)
    assert fleet["soc_max"] == pytest.approx(0.9)
    assert fleet["soc_terminal"] == pytest.approx(0.5)
    assert fleet["clean"] is True
