# test_v2_cy_jn_market.py
"""The support knobs behind `line50_q2.0` and the experiment configuration."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import grid
import models
from screen_v2_cy_jn_2025 import build_config


@pytest.fixture
def network_cache():
    """Keep these tests from leaving a scaled network in the module cache."""
    saved = (grid._net_cache, grid._net_cache_cfg)
    yield
    grid._net_cache, grid._net_cache_cfg = saved


def test_the_support_knobs_default_off():
    config = models.MarketConfig.from_yaml()
    assert config.network.line_impedance_scale == 1.0
    assert config.network.reactive_fraction_of_load_q == 0.0


def test_the_support_assumption_scales_impedance_but_not_ampacity(network_cache):
    plain = models.MarketConfig.from_yaml()
    scaled = models.MarketConfig.from_yaml()
    scaled.network.line_impedance_scale = 0.5
    base = grid.build_base_network(plain)
    half = grid.build_base_network(scaled)
    np.testing.assert_allclose(
        half.line["r_ohm_per_km"].to_numpy(),
        base.line["r_ohm_per_km"].to_numpy() * 0.5)
    np.testing.assert_allclose(
        half.line["x_ohm_per_km"].to_numpy(),
        base.line["x_ohm_per_km"].to_numpy() * 0.5)
    # Ampacity is the installed conductor's and must not follow the scale.
    np.testing.assert_allclose(half.line["max_i_ka"].to_numpy(),
                               base.line["max_i_ka"].to_numpy())


def test_the_cache_does_not_hand_a_scaled_network_to_a_plain_request(
        network_cache):
    scaled = models.MarketConfig.from_yaml()
    scaled.network.line_impedance_scale = 0.5
    half_r = grid.build_base_network(scaled).line["r_ohm_per_km"].sum()
    plain = models.MarketConfig.from_yaml()
    plain_r = grid.build_base_network(plain).line["r_ohm_per_km"].sum()
    np.testing.assert_allclose(plain_r, 2.0 * half_r)


def test_the_experiment_clears_the_support_assumption_and_the_cycle_cost():
    config = build_config(80.0)
    assert config.opf_mode == "socp"
    assert config.network.line_impedance_scale == 0.5
    assert config.network.reactive_fraction_of_load_q == 2.0
    assert config.storage.cycle_cost == 80.0
    assert config.storage.terminal_soc_equal is True
    # The PCC constraints are the project's, not this experiment's.
    assert config.network.reverse_power_limit_mw == 2.5


_CY = Path(__file__).resolve().parents[1] / "data" / "processed" / "v2_cy_jn_2025"


@pytest.mark.skipif(not (_CY / "load_price_pairing.csv").exists(),
                    reason="V2-CY-JN-2025 artifacts are not built")
def test_one_paired_day_closes_both_balance_rebuilds():
    """The result must carry the terms a reader needs to verify both balances.

    Without the line Q flow, the loss terms and the slack Q, a balance check
    can only trust that the constraint was written. This closes both from the
    published result alone and requires them to agree with the clearing.
    """
    import csv

    import market
    from screen_v2_cy_jn_2025 import _checks, build_agents

    with (_CY / "load_price_pairing.csv").open(encoding="utf-8") as fh:
        pair = next(csv.DictReader(fh))
    curves = np.load(_CY / "price_curves.npz")
    loads = np.load(_CY / "bus_loads_96x33.npz")
    price = np.asarray(curves[f"south/{pair['price_date']}"], dtype=float)
    load = np.asarray(loads[f"load/{pair['load_date']}"], dtype=float)
    config = build_config(100.0)
    net, agents = build_agents(config, load, price, [9, 14, 17, 27])
    result = market.clear_market(agents, 96, "DA", {}, config,
                                 wholesale=price, horizon_type="full_day")
    checks = _checks(result, agents, config, price, net)
    assert checks["q_balance_checked"] is True
    assert checks["p_balance_max_residual_mw"] < 1e-5
    assert checks["q_balance_max_residual_mvar"] < 1e-5
    assert checks["implied_loss_mwh"] > 0
