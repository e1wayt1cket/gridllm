"""The single-unit arbitrage bound: what a perfect forecaster could earn.

The bound is the instrument that decides whether a market has arbitrage in it,
so the tests here are about whether it is *money* and whether it is *reachable*,
not about whether it returns a particular number. Two of them pin failure modes
that a plausible-looking implementation actually exhibits: a terminal pin that
overwrites the last transition, which lets a unit discharge energy it never
charged, and an objective that is an optimizer's score rather than a settlement.

None of these need Gurobi, a network, or a config file.
"""

import os

import numpy as np
import pytest

import arbitrage_lp as al
from models import Agent, MarketConfig, StorageSpec
from participant_payoff import participant_payoff


def unit(**overrides) -> al.UnitSpec:
    """The fleet's own unit: 0.4 MW, 1.6 MWh, eta 0.92, band [0.10, 0.90]."""
    fields = dict(e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
                  eta_ch=0.92, eta_dis=0.92, soc0=0.5,
                  soc_min=0.10, soc_max=0.90)
    fields.update(overrides)
    return al.UnitSpec(**fields)


# ---------------------------------------------------------------------------
# The degenerate cases
# ---------------------------------------------------------------------------

def test_a_flat_price_day_earns_nothing():
    """No spread, no trade. A unit that cycled here would pay degradation twice."""
    res = al.arbitrage_lp(np.full(96, 500.0), unit(), cycle_cost=100.0)
    assert res.margin_cny == pytest.approx(0.0, abs=1e-9)
    assert np.max(res.ch_mw) == pytest.approx(0.0, abs=1e-9)
    assert np.max(res.dis_mw) == pytest.approx(0.0, abs=1e-9)


def test_a_near_degenerate_price_vector_still_solves():
    """A real vector that used to make the bound unavailable.

    Taken from the native library's day 20 at bus 17, whose nodal prices carry
    a near-tie of 5e-6 CNY/MWh between two periods. HiGHS presolve returned an
    Unknown status on this instance while reporting a feasible point, so the
    bound for that bus-day simply did not exist -- and the audit would have
    dropped the day rather than reporting a wrong number, which is better but
    still a hole in the measurement.

    The fixture is the measured prices rather than a synthesized approximation,
    because a synthesized near-tie does not reproduce the failure: the trigger
    is this vector's structure, not near-degeneracy on its own.
    """
    path = os.path.join(os.path.dirname(__file__), "data",
                        "lmp_near_degenerate_day20_bus17.txt")
    with open(path, "r", encoding="utf-8") as fh:
        price = np.array([float(line) for line in fh
                          if line.strip() and not line.startswith("#")])
    assert price.size == 96
    # The near-tie is what makes this vector the one that broke, so it is
    # asserted rather than assumed -- a fixture that lost it would still pass.
    gaps = np.diff(np.sort(price))
    assert gaps.min() < 1e-5

    res = al.arbitrage_lp(price, unit(), cycle_cost=100.0)
    assert res.margin_cny > 0.0
    assert np.isfinite(res.adjusted_spread_cny_mwh)
    assert res.ch_mw.size == 96


def test_a_zero_cycle_cost_is_refused():
    """At zero cost the LP is free to churn both legs at once for a zero.

    A charge and a discharge in the same period cancel in the objective, so the
    optimum is degenerate and the schedule it returns is not a schedule. The
    module refuses the input rather than answering it.
    """
    with pytest.raises(ValueError, match="cycle_cost"):
        al.arbitrage_lp(np.full(96, 500.0), unit(), cycle_cost=0.0)
    with pytest.raises(ValueError, match="cycle_cost"):
        al.arbitrage_lp(np.full(96, 500.0), unit(), cycle_cost=-1.0)


def test_a_start_outside_the_band_is_refused():
    with pytest.raises(ValueError, match="soc0"):
        al.arbitrage_lp(np.full(96, 500.0), unit(soc0=1.5), cycle_cost=100.0)


def test_a_non_finite_price_is_refused():
    price = np.full(96, 500.0)
    price[40] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        al.arbitrage_lp(price, unit(), cycle_cost=100.0)


# ---------------------------------------------------------------------------
# The state of charge
# ---------------------------------------------------------------------------

def test_the_soc_identity_holds_on_every_period():
    """The transition must be the clearing's, not a lookalike.

    Checked period by period against the same expression `dispatch_socp` writes,
    including the `DT_HOURS / e_max` factor. A bound computed on a battery that
    is not this battery is not a bound on this market.
    """
    price = 400.0 + 200.0 * np.sin(np.arange(96) / 96 * 2 * np.pi)
    u = unit()
    res = al.arbitrage_lp(price, u, cycle_cost=100.0)
    dt = 0.25
    expected = res.soc[:-1] + (u.eta_ch * res.ch_mw
                               - res.dis_mw / u.eta_dis) * dt / u.e_max
    assert np.allclose(res.soc[1:], expected, atol=1e-9)


def test_the_endpoints_are_pinned_to_the_initial_state():
    """A whole-day figure must not be flattered by banked energy."""
    price = 400.0 + 200.0 * np.sin(np.arange(96) / 96 * 2 * np.pi)
    u = unit()
    res = al.arbitrage_lp(price, u, cycle_cost=100.0)
    assert res.soc[0] == pytest.approx(u.soc0, abs=1e-9)
    assert res.soc[-1] == pytest.approx(u.soc0, abs=1e-9)


def test_soc_stays_inside_the_band_every_period():
    price = 300.0 + 400.0 * np.sin(np.arange(96) / 96 * 2 * np.pi)
    u = unit()
    res = al.arbitrage_lp(price, u, cycle_cost=100.0)
    assert np.min(res.soc) >= u.soc_min - 1e-9
    assert np.max(res.soc) <= u.soc_max + 1e-9


def test_the_unit_cannot_discharge_energy_it_never_charged():
    """The bound on delivered energy from the state of charge alone.

    This is the regression for the terminal pin: writing `soc[T] == soc[0]` onto
    the last transition row instead of its own row silently drops the final
    period's energy balance, and the LP then discharges from nowhere. It is a
    failure that reads as a very good day, so it is asserted rather than
    commented.

    A price that rises all day makes discharging as early and as hard as
    possible the optimum, which is exactly when the missing row pays best.
    """
    u = unit()
    price = np.linspace(100.0, 2000.0, 96)
    res = al.arbitrage_lp(price, u, cycle_cost=100.0)
    delivered_ceiling = (u.eta_ch * u.eta_dis * res.chargeable_energy_mwh
                         + (u.soc0 - u.soc_min) * u.e_max * u.eta_dis)
    assert res.dischargeable_energy_mwh <= delivered_ceiling + 1e-6, (
        f"delivered {res.dischargeable_energy_mwh:.4f} MWh against a ceiling of "
        f"{delivered_ceiling:.4f} MWh")


def test_a_spread_below_the_floor_produces_no_dispatch():
    """The floor is what the whole audit turns on, so it is asserted directly.

    A day whose peak-to-valley swing cannot cover the round trip has nothing in
    it for the unit: the optimum is to sit still, and the margin is zero rather
    than a small positive number the objective would be chasing. A day like this
    is what "below the degradation floor" means, stated as a test.
    """
    u = unit()
    # 450 valley against 550 peak, so a 100 CNY/MWh gross swing against a
    # 200 CNY/MWh floor before conversion losses are counted.
    price = 500.0 + 50.0 * np.sin(np.arange(96) / 96 * 2 * np.pi)
    res = al.arbitrage_lp(price, u, cycle_cost=100.0)

    # The price-side fact, read off the day's extremes rather than off the
    # schedule: the day is below the floor, which is a statement about the
    # market and holds whether or not the unit trades.
    extremes = u.eta_dis * price.max() - price.min() / u.eta_ch - 2.0 * 100.0
    assert extremes < 0.0

    # And the schedule-side consequence: it does nothing.
    assert res.margin_cny == pytest.approx(0.0, abs=1e-9)
    assert res.chargeable_energy_mwh == pytest.approx(0.0, abs=1e-9)
    assert res.dischargeable_energy_mwh == pytest.approx(0.0, abs=1e-9)
    assert res.feasible_charge_periods == 0


def test_a_day_the_unit_sits_out_reports_no_spread_rather_than_zero():
    """NaN, not 0, when there is no trade to price.

    A quoted spread of zero on a day the unit never ran would read as "the
    spread was zero", which is a claim about the market. NaN says the schedule
    carries no charge or discharge price to average, which is a claim about the
    schedule -- and those are different findings.
    """
    u = unit()
    res = al.arbitrage_lp(np.full(96, 500.0), u, cycle_cost=100.0)
    assert np.isnan(res.best_charge_price)
    assert np.isnan(res.best_discharge_price)
    assert np.isnan(res.adjusted_spread_cny_mwh)
    assert np.isnan(res.margin_per_mwh_charged)


# ---------------------------------------------------------------------------
# The optimum
# ---------------------------------------------------------------------------

def test_a_two_step_spike_is_captured_exactly():
    """A hand-computable case, so the objective is checked against arithmetic.

    One cheap period, one dear one, unit efficiency so the closed form has no
    conversion term in it. Charging a full power-period and discharging a full
    power-period is the whole opportunity, and the answer is the spread less
    both legs of degradation.
    """
    u = al.UnitSpec(e_max=1.0, p_ch_max=1.0, p_dis_max=1.0,
                    eta_ch=1.0, eta_dis=1.0, soc0=0.5,
                    soc_min=0.0, soc_max=1.0)
    p_lo, p_hi, cc = 100.0, 1000.0, 100.0
    res = al.arbitrage_lp(np.array([p_lo, p_hi]), u, cycle_cost=cc)
    expected = 0.25 * (p_hi - p_lo - 2.0 * cc)
    assert res.margin_cny == pytest.approx(expected, rel=1e-9)
    assert res.chargeable_energy_mwh == pytest.approx(0.25, rel=1e-9)
    assert res.dischargeable_energy_mwh == pytest.approx(0.25, rel=1e-9)


def test_simultaneous_charge_and_discharge_is_never_optimal():
    """At a positive cycle cost the two legs cannot both run in one period.

    Their price terms cancel exactly, so only the degradation survives, and it
    is a cost. This is why the LP needs no binaries and no exclusivity cap.
    """
    rng = np.random.RandomState(7)
    price = 300.0 + rng.normal(0.0, 150.0, 96)
    res = al.arbitrage_lp(price, unit(), cycle_cost=100.0)
    assert np.max(np.minimum(res.ch_mw, res.dis_mw)) <= al.DEFAULT_TOL_MW


def test_the_bound_never_beats_the_day_extremes():
    """The LP trades at prices it can reach; the extremes are not always reachable.

    `best_charge_price` is the mean over the periods the unit actually charged,
    so it sits at or above the day's minimum and the discharge price at or below
    its maximum. The adjusted spread must therefore not exceed the one computed
    on the extremes -- and on the native day it is strictly less, which is the
    overstatement this decomposition exists to expose.
    """
    u = unit()
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    res = al.arbitrage_lp(price, u, cycle_cost=100.0)
    extremes = (u.eta_dis * price.max() - price.min() / u.eta_ch - 2.0 * 100.0)
    assert res.adjusted_spread_cny_mwh <= extremes + 1e-9
    assert res.adjusted_spread_cny_mwh < extremes


def test_the_spread_decomposition_adds_up():
    """Gross less its two named costs is the adjusted spread, exactly."""
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    res = al.arbitrage_lp(price, unit(), cycle_cost=100.0)
    u = unit()
    # The degradation floor is the derived one, not the lossless-round-trip
    # `2 * cycle_cost`: the charge leg draws 1/eta_ch MWh from the grid per MWh
    # stored and the discharge leg returns eta_dis, so the two legs are not
    # equal in energy. See `arbitrage_lp.required_margin`.
    assert res.degradation_cost_cny_mwh == pytest.approx(
        al.required_margin(100.0, u.eta_ch, u.eta_dis))
    assert res.degradation_cost_cny_mwh == pytest.approx(200.6957, abs=1e-4)
    assert res.adjusted_spread_cny_mwh == pytest.approx(
        res.gross_spread_cny_mwh
        - res.efficiency_loss_cny_mwh
        - res.degradation_cost_cny_mwh, rel=1e-9)


def test_the_required_margin_reduces_to_the_lossless_case():
    """At unit efficiency the derived floor is the familiar `2 * cycle_cost`."""
    assert al.required_margin(100.0, 1.0, 1.0) == pytest.approx(200.0)


def test_the_required_margin_is_what_the_lp_actually_breaks_even_on():
    """The formula is checked against the optimiser, not asserted.

    Two prices are chosen so the margin equals the derived floor exactly; the
    LP's optimum must be zero there and positive just above it.
    """
    u = unit()
    cc, ec, ed = 100.0, u.eta_ch, u.eta_dis
    req = al.required_margin(cc, ec, ed)
    p_lo = 300.0
    for delta, expect_positive in ((0.0, False), (0.5, True), (2.0, True)):
        p_hi = (req + delta + p_lo / ec) / ed
        price = np.full(96, p_lo)
        price[44:52] = p_hi
        res = al.arbitrage_lp(price, u, cc)
        if expect_positive:
            assert res.margin_cny > 0, delta
        else:
            assert res.margin_cny == pytest.approx(0.0, abs=1e-9), delta
    # ...and the lossless form would have called this day profitable.
    p_hi = (2.0 * cc + 0.5 + p_lo / ec) / ed
    price = np.full(96, p_lo)
    price[44:52] = p_hi
    assert al.arbitrage_lp(price, u, cc).margin_cny == pytest.approx(0.0, abs=1e-9)


def test_the_same_series_gives_a_bit_identical_answer_twice():
    """A reported figure has to be reproducible run to run."""
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    a = al.arbitrage_lp(price, unit(), cycle_cost=100.0)
    b = al.arbitrage_lp(price, unit(), cycle_cost=100.0)
    assert a.margin_cny == b.margin_cny
    assert np.array_equal(a.ch_mw, b.ch_mw)
    assert np.array_equal(a.dis_mw, b.dis_mw)


def test_charged_energy_cannot_exceed_one_band_swing_at_the_documented_prices():
    """A single-humped day admits one fill, so the swing is bounded.

    The unit starts at `soc0`, may rise to `soc_max` and fall to `soc_min`, and
    must return. Charging the whole band at the bus therefore costs
    `(soc_max - soc_min) * e_max / eta_ch`, and on a day with one valley the
    optimum cannot exceed it. A richer price series with several valleys would
    admit more, which is why this is asserted against *this* shape rather than
    claimed as a general ceiling.
    """
    u = unit()
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    res = al.arbitrage_lp(price, u, cycle_cost=100.0)
    swing = (u.soc_max - u.soc_min) * u.e_max / u.eta_ch
    assert 0.0 < res.chargeable_energy_mwh <= swing + 1e-9
    assert res.margin_per_mwh_charged > 0.0


def test_a_single_humped_day_offers_one_cycle():
    """The claim the audit rests on for why the native market is thin.

    One valley and one peak means one fill and one empty per day, so the unit
    turns its whole usable window over once at most. Degradation is charged per
    MWh through the battery, so the number of cycles is the number of times the
    unit can pay that floor -- and one is not many.
    """
    u = unit()
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    res = al.arbitrage_lp(price, u, cycle_cost=100.0)
    assert res.equivalent_cycles <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# The objective is money
# ---------------------------------------------------------------------------

def _storage_agent() -> Agent:
    zeros = np.zeros(96)
    return Agent(
        name="ESS_AUDIT", bus=9, is_prosumer=False,
        load_forecast=zeros, load_real=zeros,
        pv_forecast=zeros.copy(), pv_real=zeros.copy(),
        wind_forecast=None, wind_real=None,
        bid_value=0.0, offer_cost=0.0,
        storage=StorageSpec(e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
                            eta_ch=0.92, eta_dis=0.92, soc0=0.5,
                            soc_min=0.10, soc_max=0.90),
        load_type="storage", pv_capacity=0.0, wind_capacity=0.0,
        participant_type="storage")


def test_the_objective_equals_the_settled_payoff():
    """The keystone: the LP's score is the repo's own settlement, to the cent.

    The LP maximizes a linear expression; `participant_payoff` settles a
    dispatch at the nodal price with generation cost, load value and penalty
    terms that are all zero for a battery. If those two ever disagree, the audit
    is reporting an optimizer's score under the name of money.
    """
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    cc = 100.0
    res = al.arbitrage_lp(price, unit(), cycle_cost=cc)
    config = MarketConfig(verbose=False)
    config.storage.cycle_cost = cc
    settled = participant_payoff(al.lp_schedule(res), price,
                                 _storage_agent(), config)
    assert settled.total == pytest.approx(res.margin_cny, abs=1e-6)
    assert settled.degradation_cost == pytest.approx(
        cc * (res.chargeable_energy_mwh + res.dischargeable_energy_mwh), abs=1e-9)
    assert settled.energy_sold_mwh == pytest.approx(
        res.dischargeable_energy_mwh, abs=1e-9)
    assert settled.energy_bought_mwh == pytest.approx(
        res.chargeable_energy_mwh, abs=1e-9)


def test_the_settlement_agrees_with_a_cycle_cost_the_config_did_not_set():
    """The cross-check must follow the cycle cost it was run with.

    `MarketConfig` carries its own `cycle_cost`; if the audit ever ran at a
    sensitivity value while the settlement kept the configured one, the two
    would agree on the default and silently diverge everywhere else.
    """
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    cc = 40.0
    res = al.arbitrage_lp(price, unit(), cycle_cost=cc)
    config = MarketConfig(verbose=False)
    config.storage.cycle_cost = cc
    settled = participant_payoff(al.lp_schedule(res), price,
                                 _storage_agent(), config)
    assert settled.total == pytest.approx(res.margin_cny, abs=1e-6)


# ---------------------------------------------------------------------------
# The schedule bridge
# ---------------------------------------------------------------------------

def test_the_schedule_carries_every_key_a_payoff_model_reads():
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    res = al.arbitrage_lp(price, unit(), cycle_cost=100.0)
    sched = al.lp_schedule(res)
    for key in ("served", "unserved", "pv_used", "wind_used",
                "p_ch", "p_dis", "p_buy", "p_sell", "soc"):
        assert key in sched, key
        assert len(sched[key]) == 96, key


def test_the_schedule_never_buys_and_sells_in_the_same_period():
    """One net flow split two ways, as `split_power` splits it."""
    price = 450.0 + 350.0 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.0)
    res = al.arbitrage_lp(price, unit(), cycle_cost=100.0)
    sched = al.lp_schedule(res)
    assert np.min(sched["p_buy"] * sched["p_sell"]) >= 0.0
    assert np.max(np.minimum(sched["p_buy"], sched["p_sell"])) <= 1e-12


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

def test_the_adapter_copies_every_field_of_the_spec():
    spec = StorageSpec(e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
                       eta_ch=0.92, eta_dis=0.92, soc0=0.5,
                       soc_min=0.10, soc_max=0.90)
    u = al.spec_from_storage(spec)
    assert u == al.UnitSpec(e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
                            eta_ch=0.92, eta_dis=0.92, soc0=0.5,
                            soc_min=0.10, soc_max=0.90)


def test_the_adapter_can_override_the_initial_state():
    """The audit varies the starting fill without rebuilding the spec."""
    spec = StorageSpec(e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
                       eta_ch=0.92, eta_dis=0.92, soc0=0.5,
                       soc_min=0.10, soc_max=0.90)
    assert al.spec_from_storage(spec, soc0=0.9).soc0 == pytest.approx(0.9)
    assert al.spec_from_storage(spec).soc0 == pytest.approx(0.5)
