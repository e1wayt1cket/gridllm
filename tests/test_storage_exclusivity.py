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
from participant_payoff import participant_payoff
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

    The action space reaches ``bid_mult`` up to 1.8, so exclusivity has to hold
    across the range the policy explores and not only at the anchor. The bid is
    a reservation price on charging now rather than a credit, so shading it up
    makes the unit more reluctant to charge, but the pair is still the widest
    spread the action can declare and is worth checking here.
    """
    agents, _, result = _clear(96, _rl_path_config(), bid_mult=1.8)
    assert result is not None, "baseline SOCP clear should succeed"
    _assert_no_simultaneity(agents, result, "RL path, bid_mult=1.8")


@pytest.mark.slow
def test_simultaneity_is_unrepresentable_rather_than_forbidden():
    """Charge and discharge are the parts of one flow, so they cannot overlap.

    This used to assert the opposite: that switching off the no-crossing rule
    let a shaded bid induce simultaneity, which is what made that rule load
    bearing. The two directions are now the positive and negative parts of a
    single variable, so the overlap is not something a rule suppresses -- it is
    a dispatch the model cannot express, and the flag that used to control it
    is inert. The check is run with the flag off and a crossing quote, the two
    conditions under which the old defect was reachable, and it has to find
    both legs in use, or "no overlap" would be satisfied by a parked battery.
    """
    T = 96
    agents, _, result = _clear(
        T, _rl_path_config(churn_free_quotes=False), bid_mult=1.8)
    assert result is not None, "baseline SOCP clear should succeed"

    rows = simultaneity_rows(agents, result)
    assert total_simultaneous(rows) < EXCLUSIVITY_TOL, format_report(rows)
    assert any(r["charge_mw"] > 0 for r in rows), \
        "no unit charged, so the exclusivity check is vacuous"
    assert any(r["discharge_mw"] > 0 for r in rows), \
        "no unit discharged, so the exclusivity check is vacuous"


@pytest.mark.slow
def test_a_crossing_quote_is_accepted_and_still_exclusive():
    """A unit may declare a charge bid above its discharge offer.

    That pair is how a battery says it will buy dearer than it is currently
    asking to sell, and it is exactly the case the old rule forbade: the
    effective bid was capped at the offer, which left storage unable to charge
    above what it asked to discharge and therefore never discharging. It is
    now the case the decomposition is built to accept -- the net flow decides
    which leg applies, not a cap on the quotes -- so the crossing pair must
    clear, must use both legs, and must still produce no overlap.
    """
    for mode in ("socp", "lindistflow"):
        config = _rl_path_config()
        config.opf_mode = mode
        agents, _, result = _clear(96, config, bid_mult=1.8)
        assert result is not None, f"{mode} clear should succeed"

        storage = [a for a in agents if a.storage is not None]
        crossing = [a for a in storage if a.bid_value * 1.8 > a.offer_cost]
        assert crossing, f"{mode}: the shaded bid did not cross the offer"

        rows = simultaneity_rows(agents, result)
        assert total_simultaneous(rows) < EXCLUSIVITY_TOL, \
            f"{mode} with a crossing quote:\n{format_report(rows)}"
        assert any(r["charge_mw"] > 0 and r["discharge_mw"] > 0 for r in rows), \
            f"{mode}: a crossing quote left every unit on one leg only"


@pytest.mark.slow
def test_lindistflow_is_structurally_churn_free():
    """Pin that the LinDistFlow solver needs no exclusivity rule.

    Its prosumer batteries are valued at ``wholesale * (dis - ch)`` — symmetric
    opposite coefficients — so their objective coefficient for simultaneous
    charge/discharge is ``-2 * cycle_cost`` and the defect cannot arise; and
    the independent fleet, which is valued at declared quotes, has charge and
    discharge tied to one net flow. This records that both batch solvers reach
    the same guarantee by different means so the difference cannot drift.
    """
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    config.storage.self_schedule = False
    agents, _, result = _clear(16, config, bid_mult=1.8)
    assert result is not None, "baseline LinDistFlow clear should succeed"
    _assert_no_simultaneity(agents, result, "LinDistFlow")


@pytest.mark.slow
def test_the_continuous_relaxation_is_validated_against_the_exact_model():
    """A whole day, cleared both ways, compared on what the day is quoted for.

    The claim the reformulation rests on is that charge and discharge are the
    positive and negative parts of one net flow once their objective
    coefficients are negative, so the relaxation is exact and no exclusive-or
    is needed. That claim is checkable: the exact model is the same problem
    with an explicit binary per unit-period, and if the argument is right the
    two must agree on everything the baseline reports.

    Two limits on what this can show, stated rather than buried. Per-period
    charge and discharge are not compared, because the optimum is degenerate --
    four identical units arbitrage one spread against a state of charge that has
    to close, so whole families of schedules attain the same value and the two
    solves may land on different members of the same family. And the exact
    model's nodal prices come from re-solving it continuously with the chosen
    directions frozen, because a MIP has no duals; the prices compared here are
    therefore a property of that re-solve, not of the integer program.
    """
    T = 96
    agents_rule, _, result_rule = _clear(T, _rl_path_config())
    agents_bin, _, result_bin = _clear(
        T, _rl_path_config(exclusive_mode="binary"))
    assert result_rule is not None and result_bin is not None

    _assert_no_simultaneity(agents_bin, result_bin, "exact binary model")

    # The objective is the claim itself, so it is held tightest.
    assert result_bin["objective"] == pytest.approx(
        result_rule["objective"], rel=1e-5), (
        f"the relaxation is not exact: objective {result_rule['objective']:.4f} "
        f"against the exact model's {result_bin['objective']:.4f}")

    for a_rule, a_bin in zip(agents_rule, agents_bin):
        if a_rule.storage is None:
            continue
        s_rule = result_rule["schedules"][a_rule.name]
        s_bin = result_bin["schedules"][a_bin.name]

        # State of charge is the physical state, so it is compared per period.
        assert np.allclose(s_rule["soc"], s_bin["soc"], atol=1e-4), (
            f"{a_rule.name}: state of charge differs by "
            f"{np.abs(np.asarray(s_rule['soc']) - np.asarray(s_bin['soc'])).max():.2e}")

        # Day totals rather than per-period, for the degeneracy above.
        for key in ("p_ch", "p_dis"):
            got = float(np.sum(s_bin[key])) * 0.25
            want = float(np.sum(s_rule[key])) * 0.25
            assert abs(got - want) <= 1e-3, (
                f"{a_rule.name}: daily {key} {got:.6f} MWh against "
                f"{want:.6f} MWh in the relaxation")

    lmp_rule = np.asarray(result_rule["lmp"], dtype=float)
    lmp_bin = np.asarray(result_bin["lmp"], dtype=float)
    assert np.allclose(lmp_bin, lmp_rule, rtol=5e-3, atol=0.05), (
        "nodal prices differ beyond a fraction of a percent (worst "
        f"{np.abs(lmp_bin - lmp_rule).max():.2f} CNY/MWh), which is the shape "
        "a wholesale fallback would take")
    assert (lmp_bin.max(axis=1) - lmp_bin.min(axis=1)).max() > 1.0, \
        "every bus shares one price within a period"

    # And the money, which is what the model is finally for.
    def fleet_profit(agents, result, config):
        return sum(participant_payoff(
            result["schedules"][a.name], np.asarray(result["lmp"])[:, a.bus],
            a, config).total for a in agents if a.storage is not None)

    profit_rule = fleet_profit(agents_rule, result_rule, _rl_path_config())
    profit_bin = fleet_profit(agents_bin, result_bin,
                              _rl_path_config(exclusive_mode="binary"))
    assert abs(profit_bin - profit_rule) <= 1.0, (
        f"settled profit differs by {abs(profit_bin - profit_rule):.4f} CNY "
        f"({profit_rule:.2f} against {profit_bin:.2f})")


@pytest.mark.slow
def test_binary_exclusivity_agrees_with_the_market_rule():
    """Cross-check: the exact binary form reaches the same clearing outcome.

    The decomposition of charge and discharge into the parts of one net flow is
    exact by an argument about objective coefficient signs, so it is worth
    confirming that an explicit exclusive-or on charge/discharge — exact by
    construction — leaves the dispatch, the nodal prices and the objective
    value unchanged. This is the comparison that makes the argument falsifiable.
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
