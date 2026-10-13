# tests/test_phase7_strategy_comparison.py
"""Contracts of the threshold-greedy baseline and the strategy comparison.

Three things this pins, each of which a plausible-looking wrong answer would
also satisfy:

* **The greedy reads one period at a time.** Its quote at period `t` must not
  move when only later prices change. Nothing in the implementation looks
  forward, but "it happens to match on this price series" is not the same claim,
  and the test perturbs the future rather than trusting the shape of the code.

* **The greedy declares; it does not dispatch.** Its output is a pair of
  reservation prices, and what the unit finally trades is the clearing's
  answer. A strategy that wrote `p_ch` would pass every output-shape check and
  be doing something entirely different.

* **A zero is only a zero if the window is shut.** On all four real curves the
  perfect-foresight bound is exactly zero. That is a strong claim, and an LP
  that silently returned nothing would produce it too -- so a control with a
  wide price band is asserted to return a large positive number beside it.

All four curves are **spatial** means, so none of them represents a bus or a
zone. On the temporal axis `261009` is a single day and the other three are
window means, read from `temporal_statistic` -- the retired `is_measured` flag
answers a different question ("is the source real measured data") and is true
for all four. One day plus three window means is not a series, so nothing here
is a cross-day statistic and no test constructs one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from config_loader import get_default, reload_defaults, set_default
from ess import build_ess_fleet
from models import MarketConfig
from rl_spec import action_spec_for
from strategies.greedy_bidding import (THRESHOLD_SOURCES, greedy_quotes,
                                       resolve_thresholds)

SHEETS = ("261009", "7days", "30days", "halfyear")


@pytest.fixture
def unit_spec():
    reload_defaults()
    fleet = build_ess_fleet(96, MarketConfig.from_yaml())
    return action_spec_for(fleet[0])


@pytest.fixture
def storage_spec():
    """The storage quote space: the box the corners are taken from."""
    reload_defaults()
    from rl_spec import ACTION_STORAGE_QUOTE_V2
    return ACTION_STORAGE_QUOTE_V2


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_the_thresholds_come_from_configuration_and_are_reproducible():
    reload_defaults()
    assert get_default("greedy.threshold_source") == "fixed"
    assert get_default("greedy.low_threshold") == 300.0
    assert get_default("greedy.high_threshold") == 400.0
    assert get_default("greedy.threshold_unit") == "CNY/MWh"
    first = resolve_thresholds()
    for _ in range(5):
        assert resolve_thresholds() == first
    assert first == (300.0, 400.0)


def test_the_thresholds_do_not_move_with_the_curve_being_evaluated():
    """A threshold fitted to the evaluated curve would be look-ahead."""
    reload_defaults()
    before = resolve_thresholds()
    for sheet in SHEETS:
        # Reading a curve must not touch the configured thresholds.
        from dianyicha_price import read_curves
        read_curves()[sheet].stats()
        assert resolve_thresholds() == before, sheet


def test_a_misordered_threshold_pair_is_refused():
    reload_defaults()
    try:
        set_default("greedy.low_threshold", 500.0)
        set_default("greedy.high_threshold", 400.0)
        with pytest.raises(ValueError, match="low < high"):
            resolve_thresholds()
    finally:
        reload_defaults()


def test_an_unknown_threshold_source_is_refused():
    reload_defaults()
    try:
        set_default("greedy.threshold_source", "tuned_on_results")
        with pytest.raises(ValueError, match="threshold_source"):
            resolve_thresholds()
    finally:
        reload_defaults()


def test_the_declared_sources_are_exactly_the_supported_ones():
    assert set(THRESHOLD_SOURCES) == {"fixed", "historical_mean"}


# ---------------------------------------------------------------------------
# One period at a time
# ---------------------------------------------------------------------------

def test_a_periods_quote_does_not_depend_on_later_prices(storage_spec):
    """The look-ahead test: perturb the future, the past must not move."""
    price = np.full(96, 350.0)
    price[5] = 250.0          # charge intent, early
    price[60] = 450.0         # discharge intent, late
    base = greedy_quotes(price, 300.0, 400.0, storage_spec)

    perturbed = price.copy()
    perturbed[11:] = 12345.0  # change everything after period 10
    alt = greedy_quotes(perturbed, 300.0, 400.0, storage_spec)

    for name in ("charge_price", "discharge_price"):
        np.testing.assert_array_equal(base[name][:11], alt[name][:11],
                                      err_msg=name)
    # ...and the perturbation did land, so the test is not vacuous.
    assert not np.array_equal(base["charge_price"][11:],
                              alt["charge_price"][11:])


def test_the_rule_is_a_pure_function_of_the_current_price(storage_spec):
    """Reordering the future does not change any quote."""
    price = np.linspace(250.0, 480.0, 96)
    base = greedy_quotes(price, 300.0, 400.0, storage_spec)
    shuffled = price.copy()
    shuffled[48:] = shuffled[48:][::-1]
    alt = greedy_quotes(shuffled, 300.0, 400.0, storage_spec)
    np.testing.assert_array_equal(base["charge_price"][:48],
                                  alt["charge_price"][:48])


def test_each_quote_follows_from_its_own_period(storage_spec):
    price = np.array([100.0, 350.0, 500.0])
    out = greedy_quotes(price, 300.0, 400.0, storage_spec)
    c_lo, c_hi = storage_spec.bounds["charge_price"]
    d_lo, d_hi = storage_spec.bounds["discharge_price"]
    # below low -> wants to charge
    assert (out["charge_price"][0], out["discharge_price"][0]) == (c_lo, d_hi)
    # inside the band -> neutral, which is the truthful point
    assert (out["charge_price"][1], out["discharge_price"][1]) == (0.0, 0.0)
    # above high -> wants to discharge
    assert (out["charge_price"][2], out["discharge_price"][2]) == (c_hi, d_lo)


def test_the_corners_match_the_documented_convention(storage_spec):
    """Ask little for the leg you want, much for the leg you do not."""
    out = greedy_quotes(np.array([100.0, 500.0]), 300.0, 400.0, storage_spec)
    # charging: charge_price low, discharge_price high
    assert out["charge_price"][0] == storage_spec.bounds["charge_price"][0]
    assert out["discharge_price"][0] == storage_spec.bounds["discharge_price"][1]
    # discharging: the other corner
    assert out["charge_price"][1] == storage_spec.bounds["charge_price"][1]
    assert out["discharge_price"][1] == storage_spec.bounds["discharge_price"][0]


def test_a_non_finite_price_signal_is_refused(storage_spec):
    bad = np.full(96, 350.0)
    bad[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        greedy_quotes(bad, 300.0, 400.0, storage_spec)


# ---------------------------------------------------------------------------
# Quotes, not dispatch
# ---------------------------------------------------------------------------

def test_the_strategy_returns_quotes_in_each_agents_own_action_space():
    import wholesale_price as wp
    from scenarios import get_scenario
    from strategies import adaptive_bidding

    reload_defaults()
    wp.reset_cache()
    config = MarketConfig.from_yaml()
    agents, _ = get_scenario("typical_day_industry", T=96, config=config,
                             day_id=0)
    actions = adaptive_bidding(agents, config, strategy="greedy_threshold",
                               T=96)
    assert set(actions) == {a.name for a in agents}
    for a in agents:
        spec = action_spec_for(a)
        assert set(actions[a.name]) == set(spec.action_names), a.name
        for name in spec.action_names:
            arr = np.asarray(actions[a.name][name])
            lo, hi = spec.bounds[name]
            assert arr.shape == (96,), (a.name, name)
            assert np.all(arr >= lo - 1e-12) and np.all(arr <= hi + 1e-12)
            if a.storage is None or "charge_price" not in spec.action_names:
                # Non-storage participants are left alone.
                np.testing.assert_allclose(arr, spec.truthful[name])
    reload_defaults()
    wp.reset_cache()


def test_the_strategy_declares_no_power_variables():
    """A quote is not a dispatch: writing p_ch/p_dis would bypass the market."""
    import wholesale_price as wp
    from scenarios import get_scenario
    from strategies import adaptive_bidding

    reload_defaults()
    wp.reset_cache()
    config = MarketConfig.from_yaml()
    agents, _ = get_scenario("typical_day_industry", T=96, config=config,
                             day_id=0)
    actions = adaptive_bidding(agents, config, strategy="greedy_threshold", T=96)
    for name, payload in actions.items():
        assert "p_ch" not in payload and "p_dis" not in payload, name
        assert "soc" not in payload, name
    reload_defaults()
    wp.reset_cache()


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------

def test_the_lp_is_not_silently_returning_nothing():
    """A control, so the real curves' zero cannot be a broken LP.

    On a wide band the same LP returns a large positive number; without this,
    `bound == 0` on the real curves would be indistinguishable from an LP that
    had stopped working.
    """
    from arbitrage_lp import arbitrage_lp, spec_from_storage

    reload_defaults()
    fleet = build_ess_fleet(96, MarketConfig.from_yaml())
    unit = spec_from_storage(fleet[0].storage)

    wide = arbitrage_lp(np.linspace(120.0, 1000.0, 96), unit, 100.0)
    assert wide.margin_cny > 100.0
    assert wide.chargeable_energy_mwh > 0.0

    trough = np.full(96, 500.0)
    trough[40:48] = 200.0
    small = arbitrage_lp(trough, unit, 100.0)
    assert small.margin_cny > 0.0
    assert small.margin_cny < wide.margin_cny


def test_the_real_curves_close_the_arbitrage_window():
    """The finding, stated as an inequality rather than as a number.

    A unit clears a cycle only when its round-trip margin exceeds what the
    clearing charges it. That requirement is `arbitrage_lp.required_margin` --
    derived from the objective as `cycle_cost * (1/eta_ch + eta_dis)`, **not**
    the lossless-round-trip `2 * cycle_cost`, which is 0.6957 CNY/MWh too
    permissive here. On every one of these four curves the margin falls short.
    """
    from arbitrage_lp import arbitrage_lp, required_margin, spec_from_storage
    from dianyicha_price import read_curves

    reload_defaults()
    config = MarketConfig.from_yaml()
    fleet = build_ess_fleet(96, config)
    unit = spec_from_storage(fleet[0].storage)
    cycle_cost = float(config.storage.cycle_cost)
    need = required_margin(cycle_cost, unit.eta_ch, unit.eta_dis)

    for sheet, curve in read_curves().items():
        arr = curve.array()
        margin = unit.eta_dis * arr.max() - arr.min() / unit.eta_ch
        assert margin < need, (
            f"{sheet}: the round-trip margin {margin:.2f} now clears the "
            f"{need:.4f} required; the conclusion in "
            f"reports/phase7_arbitrage_strategy_comparison.md needs revisiting")
        res = arbitrage_lp(arr, unit, cycle_cost)
        assert res.margin_cny == pytest.approx(0.0, abs=1e-9), sheet


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pair():
    """The same curve run under two strategies, each freshly built."""
    import phase7_strategy_comparison as p7

    reload_defaults()
    try:
        yield {s: p7.run_strategy(s, 0, 96) for s in ("fixed", "greedy_threshold")}
    finally:
        p7._fresh()


def test_every_strategy_is_settled_by_the_same_path(pair):
    """Same entry point, same clean flag, same reconciliation."""
    for strategy, run in pair.items():
        assert run["settled_clean"] is True, strategy
        assert all(b["matches_settlement"] for b in run["breakdown"].values())
        assert run["fleet_profit"] == pytest.approx(
            sum(run["breakdown"][k]["total"] for k in run["breakdown"]))


def test_the_declared_intent_is_read_from_the_quotes_actually_made(pair):
    """`fixed` never leaves the truthful corner; the greedy does."""
    for unit, counts in pair["fixed"]["declared_intent"].items():
        assert counts["periods_charge_intent"] == 0, unit
        assert counts["periods_discharge_intent"] == 0, unit
        assert counts["periods_neutral"] == 96, unit

    greedy = pair["greedy_threshold"]["declared_intent"]
    assert any(c["periods_discharge_intent"] > 0 for c in greedy.values())
    for unit, counts in greedy.items():
        total = (counts["periods_charge_intent"]
                 + counts["periods_discharge_intent"]
                 + counts["periods_neutral"])
        assert total == 96, unit


def test_a_strategy_can_declare_intent_the_clearing_refuses(pair):
    """The greedy asks; the market decides. On this curve it refuses.

    This is the honest statement of the result: the quotes are placed and the
    units still do not move, because the clearing prices the cycle above what
    the spread covers.
    """
    greedy = pair["greedy_threshold"]
    declared = sum(c["periods_charge_intent"] + c["periods_discharge_intent"]
                   for c in greedy["declared_intent"].values())
    traded = sum(t["charge_mwh"] + t["discharge_mwh"]
                 for t in greedy["traded"].values())
    assert declared > 0
    assert traded < 1e-4, (
        "the greedy found trade on this curve; the zero-throughput finding in "
        "the phase-7 report no longer holds")


def test_the_price_scenario_is_not_taken_from_a_stale_cache():
    """Two curves back to back must price two different series."""
    import phase7_strategy_comparison as p7
    import wholesale_price as wp
    from scenarios import get_scenario

    try:
        p7._fresh()
        config = MarketConfig.from_yaml()
        _, first = get_scenario("typical_day_industry", T=96, config=config,
                                day_id=0)
        from dianyicha_price import read_curves
        curves = read_curves()
        assert np.allclose(first, curves["261009"].array())

        p7._fresh()
        config = MarketConfig.from_yaml()
        _, third = get_scenario("typical_day_industry", T=96, config=config,
                                day_id=3)
        assert np.allclose(third, curves["halfyear"].array())
        assert not np.allclose(first, third)
    finally:
        p7._fresh()
        wp.reset_cache()


def test_an_unclean_settlement_is_never_reported_as_zero(monkeypatch):
    """A degraded clear must produce a reason, not a pay-off of 0.0."""
    import counterfactual_eval as ce
    import phase7_strategy_comparison as p7

    monkeypatch.setattr(ce, "settle_day",
                        lambda *a, **k: ({}, False, None))
    run = p7.run_strategy("fixed", 0, 96)
    assert run["settled_clean"] is False
    assert run["profits"] is None
    assert "zero" in run["reason"]
    p7._fresh()


# ---------------------------------------------------------------------------
# Nothing older moved
# ---------------------------------------------------------------------------

def test_the_registered_strategies_are_exactly_the_expected_set():
    import strategies
    assert {"fixed", "mpc", "greedy_threshold"} <= set(strategies.STRATEGY_REGISTRY)


def test_phase_1_to_6_interfaces_are_intact():
    import dianyicha_price as dp
    import industry_load as il
    assert il.N_SLOTS == 96
    assert callable(il.load_shapes_csv)
    assert dp.SHEETS == SHEETS
    for name, c in dp.read_curves().items():
        # Taxonomy sanity, not a single-day claim: every curve is real measured
        # data and declares a closed-vocabulary temporal statistic.
        assert c.source_kind == "real_market_data", name
        assert c.temporal_statistic in dp.TEMPORAL_STATISTICS, name

    reload_defaults()
    import wholesale_price as wp
    wp.reset_cache()
    from scenarios import get_scenario
    agents, _ = get_scenario("typical_day_industry", T=96,
                             config=MarketConfig.from_yaml(), day_id=0)
    loads = [a for a in agents if a.load_type != "storage"]
    peak = float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())
    assert peak == pytest.approx(3.715, abs=1e-6)
    assert [a.name for a in agents if a.participant_type == "storage"] == \
        ["ESS9", "ESS14", "ESS17", "ESS27"]
    reload_defaults()
    wp.reset_cache()
