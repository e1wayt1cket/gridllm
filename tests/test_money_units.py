# tests/test_money_units.py
"""One money convention across the simulator: money = price x power x period.

Before this convention was unified the objective mixed two units — its base
economic terms valued a period's power as if a period were an hour, while the
carbon, RE and terminal-SOC terms were already energy times a price. These tests
pin the convention, and pin the invariance that justifies the change: where the
objective is purely energy-valued, unifying the unit is a positive rescale and
must leave the dispatch and the nodal prices untouched.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types

import numpy as np
import pytest

import money
from models import MarketConfig
from market import clear_market
from scenarios import get_scenario


def _baseline_clear(T, dt_scaled, seed=7, opf_mode="socp",
                    use_nodal_price=False, **storage_overrides):
    """Clear one seeded baseline day with the money convention switched."""
    np.random.seed(seed)
    agents, wholesale = get_scenario("baseline", T=T)
    config = MarketConfig(opf_mode=opf_mode, verbose=False)
    config.dt_scaled_money = dt_scaled
    config.storage.self_schedule = False
    config.storage.use_nodal_price = use_nodal_price
    # Isolate the base ledger: the terminal SOC value is already money and the
    # multi-objective terms already carry the period length, so leaving either
    # in place would make the rescale non-uniform by design.
    config.storage.terminal_value = 0.0
    config.market_design.enable_multi_objective = False
    config.market_design.use_constraint_multi_obj = False
    for key, value in storage_overrides.items():
        setattr(config.storage, key, value)
    result = clear_market(agents, T, "DA", {}, config, wholesale=wholesale)
    return agents, config, result


def test_dt_hours_has_a_single_source():
    """The period length is one number, not several that happen to agree."""
    from dispatch_core import DT_HOURS as core_dt
    assert money.DT_HOURS == 0.25
    assert core_dt == money.DT_HOURS
    assert money.PERIODS_PER_DAY == 96


def test_hand_computed_energy_value():
    """Four periods of 1 MW at 100 CNY/MWh is 100 CNY, not 400."""
    power = np.ones(4)
    price = np.full(4, 100.0)
    assert money.total_money(price, power) == pytest.approx(100.0)
    assert money.energy(power).sum() == pytest.approx(1.0)


def test_price_from_balance_dual_divides_by_period_length():
    """A balance dual is per-MW-over-a-period; a price is per MWh."""
    assert money.price_from_balance_dual(-400.0) == pytest.approx(1600.0)
    assert money.price_from_balance_dual(-400.0, dt=1.0) == pytest.approx(400.0)


@pytest.mark.slow
@pytest.mark.parametrize("opf_mode", ["socp", "lindistflow"])
@pytest.mark.parametrize("use_nodal_price", [False, True])
def test_pure_ledger_objective_is_invariant_to_the_unit_fix(opf_mode,
                                                            use_nodal_price):
    """Unifying the unit must not move the dispatch or the prices.

    With the terminal SOC value at zero and the multi-objective terms off, every
    remaining objective term is energy-valued, so scaling them by the period
    length is a positive rescale: the argmax is unchanged and the de-scaled
    duals are unchanged. This is what makes the change a unit correction rather
    than a behaviour change, and it is the check that catches a missed de-scale
    at any dual-extraction site.

    Both batch solvers are covered, and both the first pass and the nodal-price
    re-solve, because each builds its own objective and its own price
    extraction; scaling one and not the other leaves the solver silently mixing
    units, which is exactly how this was missed the first time.
    """
    T = 16
    kwargs = dict(opf_mode=opf_mode, use_nodal_price=use_nodal_price)
    agents_old, _, old = _baseline_clear(T, dt_scaled=False, **kwargs)
    agents_new, _, new = _baseline_clear(T, dt_scaled=True, **kwargs)
    assert old is not None and new is not None

    label = f"{opf_mode} nodal_reprice={use_nodal_price}"

    # Day-level energy flows must be identical: the same model, rescaled, has
    # the same optimal value and so moves the same quantity of energy. This is
    # asserted for every solver, because it holds whichever optimum the solver
    # lands on.
    #
    # Only the quantities the clearing decides are checked. p_buy and p_sell are
    # derived afterwards by splitting each agent's net flow, which is not linear
    # in the decision, so their totals are not preserved when two solves land on
    # different optima of a degenerate problem.
    def totals(result):
        t = {}
        for key in ("served", "p_ch", "p_dis", "pv_used", "wind_used"):
            t[key] = sum(float(np.sum(np.asarray(result["schedules"][a.name][key],
                                                 dtype=float)))
                         for a in agents_new)
        return t

    tot_new, tot_old = totals(new), totals(old)
    for key, value in tot_new.items():
        assert value == pytest.approx(tot_old[key], rel=1e-4, abs=1e-3), (
            f"[{label}] total {key} moved with the unit fix: "
            f"{value:.4f} vs {tot_old[key]:.4f}")

    if opf_mode != "socp":
        # LinDistFlow is not a single solve: it iterates an outer approximation
        # to line losses — and again over nodal prices when repricing is on —
        # and its optimum is degenerate, reaching the same objective by
        # dispatching the same energy in a different order. So neither its
        # per-period dispatch, nor its duals, nor its reported objective value is
        # reproducible across two solves of the same model. The day-level energy
        # totals asserted above are what the rescale guarantees, and they hold.
        #
        # SOCP is a single solve and is the path training and evaluation use, so
        # the tighter checks below apply to it.
        return

    # SOCP: one solve, so the same optimum is reached and the dispatch, the
    # de-scaled duals and the objective all have to line up.
    assert new["objective"] == pytest.approx(0.25 * old["objective"], rel=1e-5), (
        f"[{label}] the unified objective should be the legacy one rescaled by "
        f"the period length: {new['objective']:.4f} vs "
        f"0.25 * {old['objective']:.4f}")
    for a in agents_new:
        n = new["schedules"][a.name]
        o = old["schedules"][a.name]
        for key in ("served", "p_ch", "p_dis", "pv_used", "wind_used"):
            assert np.allclose(n[key], o[key], atol=1e-4), (
                f"[{label}] {a.name}: {key} dispatch moved with the unit fix")

    lmp_new = np.asarray(new["lmp"], dtype=float)
    lmp_old = np.asarray(old["lmp"], dtype=float)
    assert np.allclose(lmp_new, lmp_old, rtol=1e-3, atol=0.05), (
        f"[{label}] nodal prices moved with the unit fix; a dual-extraction "
        "site is missing its de-scale (worst "
        f"{np.abs(lmp_new - lmp_old).max():.3f} CNY/MWh)")


def test_all_four_profit_implementations_agree():
    """The four copies of the profit formula must stay on one convention.

    The formula is duplicated across the metrics layer, evaluation, the
    environment reward and the decomposition; each was rescaled separately by
    hand, so this pins them together. Any one of them missing the period length
    would silently put a training signal on a different scale from the benefit
    it is later reported against.
    """
    import types

    import diagnose_profit
    import eval_agents
    import surplus_metrics
    from rl_env import BiddingEnv

    cfg = MarketConfig(verbose=False)
    cfg.storage.cycle_cost = 100.0
    cfg.market_design.penalty_unserved = 2500.0
    agent = types.SimpleNamespace(
        bid_value=350.0, offer_cost=180.0, storage=object())
    sched = {
        "p_sell": np.array([1.0, 0.0, 2.0]),
        "p_buy": np.array([0.0, 3.0, 0.0]),
        "served": np.array([4.0, 4.0, 4.0]),
        "pv_used": np.array([1.0, 0.0, 2.0]),
        "wind_used": np.array([0.0, 1.0, 0.0]),
        "p_ch": np.array([0.0, 2.0, 0.0]),
        "p_dis": np.array([1.0, 0.0, 2.0]),
        "unserved": np.array([0.0, 1.0, 0.0]),
    }
    lmp_node = np.array([100.0, 110.0, 90.0])

    metrics = surplus_metrics.agent_profit(sched, lmp_node, agent, cfg)
    evaluation = eval_agents.compute_agent_profit(sched, lmp_node, agent, cfg,
                                                  n_periods=3)
    diagnostics = diagnose_profit.agent_profit(sched, lmp_node, agent, cfg)
    # `_agent_block_profit` reads only config off the instance, so a stand-in
    # avoids building a whole environment for a formula check.
    env_stub = types.SimpleNamespace(config=cfg)
    reward = BiddingEnv._agent_block_profit(env_stub, sched, lmp_node, agent, 3)

    assert evaluation == pytest.approx(metrics)
    assert diagnostics == pytest.approx(metrics)
    assert reward == pytest.approx(metrics)

    # And the value is money, not a per-period power sum: hand-computed term by
    # term and scaled once by the 0.25 h period.
    market = (1.0 * 100 + 0.0 * 110 + 2.0 * 90) \
        - (0.0 * 100 + 3.0 * 110 + 0.0 * 90)          # sell - buy
    consumption = 350.0 * (4.0 + 4.0 + 4.0)
    generation = 180.0 * (1.0 + 0.0 + 2.0 + 0.0 + 1.0 + 0.0)
    penalty = 2500.0 * 1.0
    degradation = 100.0 * (0.0 + 2.0 + 0.0 + 1.0 + 0.0 + 2.0)
    manual = (market + consumption - generation - penalty
              - degradation) * 0.25
    assert manual == pytest.approx(107.5)
    assert metrics == pytest.approx(manual)


def test_ratio_metrics_are_invariant_to_the_time_scale():
    """Ratios of two money figures stay comparable across the unification."""
    import surplus_metrics

    # T_SCALE is now the period length, so it is no longer 1.0; anything that
    # relied on the old value is a ratio and cancels it.
    assert surplus_metrics.T_SCALE == money.DT_HOURS

    a = types.SimpleNamespace(name="A", bus=0, bid_value=50.0,
                              load_forecast=np.ones(3), load_real=None)
    sched = {"A": {"served": np.array([1.0, 2.0, 3.0]),
                   "p_buy": np.array([1.0, 2.0, 3.0]),
                   "p_sell": np.zeros(3),
                   "pv_used": np.zeros(3), "wind_used": np.zeros(3),
                   "p_ch": np.zeros(3), "p_dis": np.zeros(3),
                   "unserved": np.zeros(3)}}
    lmp = np.array([[100.0], [140.0], [120.0]])
    wholesale = np.full(3, 120.0)
    markup = surplus_metrics.load_payment_weighted_markup(
        sched, lmp, wholesale, [a])
    # Same ratio computed without any time scale: it must not move.
    power_weighted = ((100 - 120) * 1 + (140 - 120) * 2
                      + (120 - 120) * 3) / (120.0 * 6)
    assert markup == pytest.approx(power_weighted)


@pytest.mark.slow
def test_reported_prices_stay_in_the_legacy_range():
    """A unit slip shows up as prices off by 4x, so bound them directly."""
    T = 16
    _, _, result = _baseline_clear(T, dt_scaled=True)
    lmp = np.asarray(result["lmp"], dtype=float)
    assert np.all(np.isfinite(lmp)), "prices must be finite"
    # This network clears in the hundreds of CNY/MWh; a 4x slip in either
    # direction leaves that band immediately.
    assert 100.0 < lmp.mean() < 2000.0, f"mean price {lmp.mean():.2f} out of band"
    assert result["lmp_fallbacks"] == 0, (
        f"{result['lmp_fallbacks']} periods fell back to the wholesale curve "
        "instead of the power-balance dual")
