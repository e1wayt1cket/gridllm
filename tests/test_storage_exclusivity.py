# tests/test_storage_exclusivity.py
"""A battery must never charge and discharge in the same period.

`ch` and `dis` are independent continuous variables in every batch solver, so
nothing arithmetically keeps them apart. What keeps them apart is the market
rule in `StorageConfig.churn_free_quotes`: a unit's declared charge bid may not
exceed its declared discharge offer by more than twice the degradation cost,
which makes the objective coefficient of a simultaneous charge/discharge
non-positive at every period and therefore never optimal. These tests pin that outcome, prove they can still see
the defect when the rule is removed, and cross-check the rule against the exact
binary formulation.

Every clear here is handed an explicit wholesale curve, because `clear_market`
draws a fresh one when given none and two clears of "the same" day would then
not be comparable.

These tests solve real day-ahead clears, so they are slow.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from models import MarketConfig
from market import clear_market
from scenarios import get_scenario

# Solver output is a float; anything below this is numerical noise rather than
# an actual simultaneous dispatch.
EXCLUSIVITY_TOL = 1e-4


def _socp_config(**storage_overrides):
    """Default SOCP config, with optional StorageConfig field overrides."""
    config = MarketConfig(opf_mode="socp", verbose=False)
    for key, value in storage_overrides.items():
        setattr(config.storage, key, value)
    return config


def _rl_path_config(**storage_overrides):
    """The config the RL training and evaluation paths use.

    `train_rl` disables the MPC pre-scheduling and the nodal re-solve so the
    declared bids are what drive storage dispatch.
    """
    kwargs = dict(self_schedule=False, use_nodal_price=False)
    kwargs.update(storage_overrides)
    return _socp_config(**kwargs)


def _clear(T, config, bid_mult=1.0, seed=7):
    """Clear one seeded day and return (agents, wholesale, result)."""
    np.random.seed(seed)
    agents, wholesale = get_scenario("baseline", T=T)
    actions = {}
    if bid_mult != 1.0:
        actions = {a.name: {"bid_mult": bid_mult, "offer_adder": 0.0}
                   for a in agents if a.storage is not None}
    result = clear_market(agents, T, "DA", actions, config,
                          wholesale=wholesale)
    return agents, wholesale, result


def simultaneity_rows(agents, result):
    """Per storage agent: simultaneous MW, affected periods, ch/dis totals."""
    rows = []
    for a in agents:
        if a.storage is None:
            continue
        sched = result["schedules"].get(a.name)
        if sched is None:
            continue
        ch = np.asarray(sched["p_ch"], dtype=float)
        dis = np.asarray(sched["p_dis"], dtype=float)
        overlap = np.minimum(ch, dis)
        rows.append({
            "agent": a.name,
            "simultaneous_mw": float(np.sum(overlap)),
            "periods_affected": int(np.sum(overlap > EXCLUSIVITY_TOL)),
            "charge_mw": float(np.sum(ch)),
            "discharge_mw": float(np.sum(dis)),
        })
    return rows


def total_simultaneous(rows):
    return sum(r["simultaneous_mw"] for r in rows)


def format_report(rows):
    header = (f"{'agent':<26}{'simul MW':>11}{'periods':>9}"
              f"{'charge MW':>12}{'disch MW':>12}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['agent']:<26}{r['simultaneous_mw']:>11.4f}"
            f"{r['periods_affected']:>9d}{r['charge_mw']:>12.3f}"
            f"{r['discharge_mw']:>12.3f}")
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL':<26}{total_simultaneous(rows):>11.4f}")
    return "\n".join(lines)


def _assert_no_simultaneity(agents, result, context):
    rows = simultaneity_rows(agents, result)
    assert rows, "baseline scenario should contain storage agents"
    total = total_simultaneous(rows)
    assert total < EXCLUSIVITY_TOL, (
        f"{context}: storage charged and discharged in the same period, total "
        f"{total:.4f} MW:\n{format_report(rows)}")


@pytest.mark.slow
def test_no_simultaneous_charge_discharge_default_config():
    """The default SOCP clear must not charge and discharge one battery at once."""
    agents, _, result = _clear(96, _socp_config())
    assert result is not None, "baseline SOCP clear should succeed"
    _assert_no_simultaneity(agents, result, "default config")


@pytest.mark.slow
def test_no_simultaneous_charge_discharge_rl_path():
    """The RL path (no MPC pre-schedule, no nodal re-solve) must hold too."""
    agents, _, result = _clear(96, _rl_path_config())
    assert result is not None, "baseline SOCP clear should succeed"
    _assert_no_simultaneity(agents, result, "RL path")


@pytest.mark.slow
def test_no_simultaneous_charge_discharge_under_shaded_bid():
    """A shaded bid must not open a simultaneity incentive.

    The clearing objective credits the agent's declared charge bid, so the
    incentive to cycle one battery both ways scales with `bid_mult`. This is the
    range the RL action space reaches, so exclusivity has to hold here and not
    only at truthful bidding.
    """
    agents, _, result = _clear(96, _rl_path_config(), bid_mult=1.8)
    assert result is not None, "baseline SOCP clear should succeed"
    _assert_no_simultaneity(agents, result, "RL path, bid_mult=1.8")


@pytest.mark.slow
def test_market_rule_is_what_prevents_simultaneity():
    """Controlled comparison: the market rule, not the bid anchor, is the guard.

    Two mechanisms independently suppress simultaneity. Pricing storage from an
    economic anchor leaves the truthful bid non-crossing, so at ``bid_mult=1``
    the defect is already gone and a control at that setting would test the
    anchor rather than the rule. Shading the bid defeats the anchor, which is
    what isolates the rule: without it the defect returns, with it the dispatch
    stays physically possible.
    """
    T = 16
    agents_off, _, result_off = _clear(
        T, _rl_path_config(churn_free_quotes=False), bid_mult=1.8)
    assert result_off is not None
    rows_off = simultaneity_rows(agents_off, result_off)
    total_off = total_simultaneous(rows_off)
    assert total_off > EXCLUSIVITY_TOL, (
        "expected a simultaneous charge/discharge without the market rule, but "
        "found none; the invariant tests can no longer detect the defect:\n"
        f"{format_report(rows_off)}")

    agents_on, _, result_on = _clear(
        T, _rl_path_config(churn_free_quotes=True), bid_mult=1.8)
    assert result_on is not None
    total_on = total_simultaneous(simultaneity_rows(agents_on, result_on))
    assert total_on < EXCLUSIVITY_TOL, (
        f"the market rule should remove the {total_off:.4f} MW of simultaneity "
        f"the shaded bid induces, but {total_on:.4f} MW remains")


@pytest.mark.slow
def test_lindistflow_is_structurally_churn_free():
    """Pin that the LinDistFlow solver needs no exclusivity rule.

    LinDistFlow values storage as ``wholesale * (dis - ch)`` — symmetric
    opposite coefficients — so its objective coefficient for simultaneous
    charge/discharge is ``-2 * cycle_cost`` and the defect cannot arise. The
    no-crossing rule is therefore SOCP-only; this records that difference
    between the two batch solvers so it cannot drift unnoticed.
    """
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    config.storage.self_schedule = False
    agents, _, result = _clear(16, config, bid_mult=1.8)
    assert result is not None, "baseline LinDistFlow clear should succeed"
    _assert_no_simultaneity(agents, result, "LinDistFlow")


@pytest.mark.slow
def test_binary_exclusivity_agrees_with_the_market_rule():
    """Cross-check: the exact binary form reaches the same clearing outcome.

    The market rule is an incentive argument, so it is worth confirming that an
    explicit exclusive-or on charge/discharge — exact by construction — leaves
    the dispatch, the nodal prices and the objective value unchanged.
    """
    T = 16
    config_rule = _rl_path_config(churn_free_quotes=True)
    agents_rule, _, result_rule = _clear(T, config_rule)
    config_bin = _rl_path_config(churn_free_quotes=True,
                                 exclusive_mode="binary")
    agents_bin, _, result_bin = _clear(T, config_bin)
    assert result_rule is not None and result_bin is not None

    _assert_no_simultaneity(agents_bin, result_bin, "binary exclusivity")

    assert result_bin["welfare"] == pytest.approx(result_rule["welfare"],
                                                  rel=1e-6), (
        "binary exclusivity and the market rule should describe the same "
        f"model; objective {result_bin['welfare']:.4f} vs "
        f"{result_rule['welfare']:.4f}")

    # Duals of a degenerate LP are not unique, so this is a relative check: the
    # two formulations agree to a fraction of a percent (measured worst case
    # 0.84 CNY/MWh on ~489, and 15 of 528 entries above 0.01) while the
    # objective matches to 3e-7. The failure being guarded against is the
    # wholesale fallback, which would move whole rows by hundreds of CNY/MWh.
    lmp_bin = np.asarray(result_bin["lmp"], dtype=float)
    lmp_rule = np.asarray(result_rule["lmp"], dtype=float)
    assert np.allclose(lmp_bin, lmp_rule, rtol=5e-3, atol=0.05), (
        "nodal prices differ between the market rule and the exact binary "
        f"formulation (worst {np.abs(lmp_bin - lmp_rule).max():.3f} CNY/MWh); "
        "the binary path has likely lost its power-balance duals")
    # A wholesale fallback would also flatten the spatial spread within a
    # period, so check the nodal structure directly.
    assert (lmp_bin.max(axis=1) - lmp_bin.min(axis=1)).max() > 1.0, (
        "every bus shares one price within a period, which is the wholesale "
        "fallback rather than a nodal solution")

    # Per-period dispatch is deliberately not asserted equal. The optimum is
    # degenerate — many churn-free dispatches reach the same objective — and the
    # two formulations land on different vertices of that optimal face (e.g. one
    # charges 0.015 MW in a period where the other charges nothing, at an
    # identical objective). A tight per-period comparison would fail for a
    # reason that has nothing to do with exclusivity.
    _assert_no_simultaneity(agents_rule, result_rule, "market rule")
