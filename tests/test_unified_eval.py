# tests/test_unified_eval.py
"""Contracts of the unified strategy evaluation interface.

The interface exists because the two comparison paths did not meet: one produced
declared quotes without money, the other produced settled profit for a smaller
set of arms. The failure modes this file guards are the ones that make a
comparison lie rather than crash:

* an arm failing on a hard scenario and being quietly dropped, so its average
  improves by failing;
* two arms "compared" on days that were not the same day;
* a training reward reported where a settled profit belongs;
* an evaluation that silently re-fitted the normalization scale it is normalized
  by.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

import numpy as np
import pytest

import unified_eval as ue
from models import MarketConfig

#: These tests exercise an *interface*: the strategy registry, the settlement
#: path, the paired evaluator. Their subject is not the load, so they run on the
#: native benchmark day -- `typical_day_industry` now carries the metered
#: industry load, which this feeder can serve only a few percent of and which
#: would make every one of them a test of the shedding rather than of the
#: interface. The price is still the real curve, set per-run on the scenario the
#: same way `compare_price_sources` sets it.
SCENARIO = "typical_day"
PRICE_SOURCE = "dianyicha"


@pytest.fixture(autouse=True)
def _real_price_curve():
    """Price the native day from the real curve, for this module only."""
    from config_loader import reload_scenarios, set_scenario_param
    reload_scenarios()
    set_scenario_param(SCENARIO, "wholesale_source", PRICE_SOURCE)
    yield
    reload_scenarios()
T = 96


def _config():
    return MarketConfig.from_yaml()


@pytest.fixture(scope="module")
def config():
    return _config()


def _scen(day_id=0, split="test"):
    return ue.ScenarioRef(scenario=SCENARIO, day_id=day_id, split=split,
                          family="fam_a", data_digest="deadbeef")


def _arm(name="fixed", label=None, family="whole_day"):
    return ue.ArmRef(label=label or name, family=family, name=name)


# ---------------------------------------------------------------------------
# 1. Repeatability
# ---------------------------------------------------------------------------

def test_the_same_arm_scenario_and_seed_repeats(config):
    """Same inputs, same result. The clearing is a deterministic MILP/LP and the
    environment is seeded from the day, so this is bit-identical rather than
    merely close -- if a solver ever stops being deterministic here, this test
    is where that becomes visible."""
    a = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=7)
    b = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=7)
    assert a["settled_net_profit_cny"] == b["settled_net_profit_cny"]
    assert a["scenario_freeze"]["wholesale_sha256"] == \
        b["scenario_freeze"]["wholesale_sha256"]
    assert a["settled_clean"] == b["settled_clean"]


# ---------------------------------------------------------------------------
# 2. Same scenario across arms
# ---------------------------------------------------------------------------

def test_arms_share_the_scenario_they_are_compared_on(config):
    """Price series, initial SOC and economics must be identical across arms --
    read off the recorded provenance, not taken on trust."""
    scen = _scen()
    recs = [ue.evaluate_arm(_arm("fixed"), scen, config, seed=11),
            ue.evaluate_arm(_arm("mpc"), scen, config, seed=11)]
    a, b = (r["scenario_freeze"] for r in recs)
    assert a["wholesale_sha256"] == b["wholesale_sha256"]
    assert a["soc0"] == b["soc0"]
    assert a["cycle_cost_per_leg"] == b["cycle_cost_per_leg"]
    ue.assert_comparable(recs)                 # must not raise


def test_a_comparison_across_different_days_is_refused(config):
    """`assert_comparable` must actually bite; otherwise it is decoration."""
    good = ue.evaluate_arm(_arm("fixed"), _scen(day_id=0), config, seed=3)
    tampered = copy.deepcopy(good)
    tampered["scenario_id"] = good["scenario_id"]
    tampered["scenario_freeze"]["wholesale_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="different price series"):
        ue.assert_comparable([good, tampered])


# ---------------------------------------------------------------------------
# 3. Evaluation mode
# ---------------------------------------------------------------------------

def test_evaluation_does_not_update_policy_parameters(config):
    """No gradient step and no exploration noise: an evaluation that trained
    would report a policy that no longer exists."""
    import torch
    import rl_td3
    from strategies import rl_bidding

    policies = {}
    for name in ("ESS9", "ESS14"):
        actor = rl_td3.Actor(12, 2, torch.tensor([[0.0, 0.0], [150.0, 300.0]]))
        policies[name] = actor
    before = {k: [p.detach().clone() for p in v.parameters()]
              for k, v in policies.items()}

    arm = ue.ArmRef(label="rl", family="per_block", name="rl",
                    policies=policies, version="untrained-stub")
    ue.evaluate_arm(arm, _scen(), config, seed=5,
                    rl_names=["ESS9", "ESS14"])

    for k, v in policies.items():
        for p_before, p_after in zip(before[k], v.parameters()):
            assert torch.equal(p_before, p_after.detach()), k


def test_results_are_marked_as_evaluation_mode(config):
    rec = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=1)
    assert rec["mode"] == "evaluation"


# ---------------------------------------------------------------------------
# 4. The frozen scale is not re-fitted by evaluation
# ---------------------------------------------------------------------------

def test_evaluation_does_not_recalibrate_the_price_reference(config):
    """A normalization scale re-derived from the data it normalizes is a leak.
    The result records the frozen constant and its artefact, and evaluating
    leaves both unchanged."""
    import rl_env
    from pathlib import Path
    import hashlib

    artifact = Path(ue.ss.PRICE_REF_CALIBRATION_FILE)
    before = hashlib.sha256(artifact.read_bytes()).hexdigest() \
        if artifact.exists() else None
    frozen = rl_env.PRICE_REF_CNY_MWH

    for day in (0, 1):
        rec = ue.evaluate_arm(_arm("fixed"), _scen(day_id=day), config, seed=2)
        assert rec["price_ref_cny_per_mwh"] == frozen
        assert rec["price_ref_artifact"] == ue.ss.PRICE_REF_CALIBRATION_FILE

    assert rl_env.PRICE_REF_CNY_MWH == frozen
    after = hashlib.sha256(artifact.read_bytes()).hexdigest() \
        if artifact.exists() else None
    assert after == before


# ---------------------------------------------------------------------------
# 5. Failures are identifiable, never faked
# ---------------------------------------------------------------------------

def test_a_solver_failure_is_recorded_not_zeroed(config, monkeypatch):
    """A failed arm must not look like an arm that earned nothing."""
    def boom(*a, **k):
        raise RuntimeError("simulated solver failure")
    monkeypatch.setattr(ue.ce, "settle_day", boom)

    rec = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=1)
    assert rec["settled_clean"] is False
    assert rec["settled_net_profit_cny"] is None
    assert rec["included"] is False
    assert rec["exclusion_reason"] == ue.EXCLUDED_SOLVER_FAILURE
    assert "failure" in rec
    assert rec["settled_net_profit_cny"] != 0.0


def test_a_soc_violation_excludes_the_record(config, monkeypatch):
    """SOC infeasibility is a failed run, not a number to average in."""
    real = ue._soc_report
    monkeypatch.setattr(ue, "_soc_report",
                        lambda *a, **k: {"per_unit": {}, "all_feasible": False})
    rec = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=1)
    assert rec["included"] is False
    assert rec["exclusion_reason"] == ue.EXCLUDED_SOC_INFEASIBLE
    monkeypatch.setattr(ue, "_soc_report", real)


def test_an_unbuildable_scenario_is_recorded_as_incomplete(config):
    rec = ue.evaluate_arm(_arm("fixed"),
                          ue.ScenarioRef(scenario="no_such_scenario", day_id=0,
                                         split="test"),
                          config, seed=1)
    assert rec["settled_clean"] is False
    assert rec["included"] is False
    assert rec["exclusion_reason"] == ue.EXCLUDED_INCOMPLETE
    assert "failure" in rec


# ---------------------------------------------------------------------------
# 6. Idle versus degenerate zero revenue
# ---------------------------------------------------------------------------

def test_idle_and_degenerate_zero_revenue_days_are_distinguished():
    """Both are `no_arbitrage`; the schedules differ, and the diagnostic must
    say which is which. Restoring "zero revenue means no schedule" would erase
    the distinction."""
    import synthetic_scenarios as ss
    from arbitrage_lp import arbitrage_lp

    flat = np.full(T, 500.0)
    R = ss.required_margin_cny_per_mwh()
    p_c = 200.0
    p_d = (R + p_c / ss.REFERENCE_UNIT.eta_ch) / ss.REFERENCE_UNIT.eta_dis
    at_be = np.full(T, p_c)
    at_be[T // 2:] = p_d

    idle = ue._regime_of(flat)
    degen = ue._regime_of(at_be)
    assert idle["regime"] == degen["regime"] == ss.NO_ARBITRAGE
    assert idle["schedule_class"] == ss.SCHEDULE_NONE
    assert degen["schedule_class"] == ss.SCHEDULE_DEGENERATE_ZERO_REVENUE
    assert arbitrage_lp(at_be, ss.REFERENCE_UNIT,
                        ss.CYCLE_COST_PER_LEG).margin_cny <= ss.IDLE_TOL_CNY


def test_the_regime_is_measured_on_the_curve_not_inherited(config):
    rec = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=1)
    reg = rec["economic_regime"]
    assert reg["regime"] in list(ue.ss.REGIMES) + [None]
    assert "diagnostic" in reg


# ---------------------------------------------------------------------------
# 7. Profit and reward are separate fields
# ---------------------------------------------------------------------------

def test_profit_and_reward_are_separate_fields(config):
    rec = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=1)
    assert "settled_net_profit_cny" in rec
    assert "training_reward" in rec
    # A whole-day arm has no reward at all; it must not be invented from profit.
    assert rec["training_reward"] is None


def test_a_per_block_arm_records_reward_apart_from_profit(config):
    rec = ue.evaluate_arm(ue.ArmRef(label="truthful", family="per_block",
                                    name="truthful"),
                          _scen(), config, seed=1, n_blocks=3)
    assert "settled_net_profit_cny" in rec
    if rec["training_reward"] is not None:
        assert set(rec["training_reward"]) <= set(rec["per_agent_profit_cny"])


# ---------------------------------------------------------------------------
# 8. A shared scenario list stays shared
# ---------------------------------------------------------------------------

def test_a_shared_scenario_list_is_identical_for_every_arm(config):
    scens = [_scen(day_id=d) for d in (0, 1)]
    out = ue.evaluate([_arm("fixed"), _arm("mpc")], scens, config, seed=4)
    assert out["scenario_ids"] == [s.scenario_id for s in scens]
    assert out["n_scenarios"] == 2
    # Every arm saw every scenario, in the same order.
    per_arm = {}
    for r in out["records"]:
        per_arm.setdefault(r["arm_label"], []).append(r["scenario_id"])
    assert len(per_arm) == 2
    assert all(v == out["scenario_ids"] for v in per_arm.values())


def test_an_arm_that_fails_a_scenario_does_not_lose_it(config, monkeypatch):
    """The scenario list is not filtered per arm: failing one must not shrink
    that arm's list and improve its average."""
    calls = {"n": 0}
    real = ue.ce.settle_day

    def sometimes(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated failure")
        return real(*a, **k)

    monkeypatch.setattr(ue.ce, "settle_day", sometimes)
    scens = [_scen(day_id=0), _scen(day_id=1)]
    out = ue.evaluate([_arm("fixed")], scens, config, seed=1)
    assert [r["scenario_id"] for r in out["records"]] == \
        [s.scenario_id for s in scens]
    assert out["summary"]["fixed"]["n_records"] == 2
    assert out["summary"]["fixed"]["n_excluded"] == 1


# ---------------------------------------------------------------------------
# 9. Summary handles failures with a stated denominator
# ---------------------------------------------------------------------------

def test_the_summary_states_its_denominator_and_lists_exclusions(config,
                                                                 monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated")
    calls = {"n": 0}
    real = ue.ce.settle_day

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return boom()
        return real(*a, **k)

    monkeypatch.setattr(ue.ce, "settle_day", flaky)
    out = ue.evaluate([_arm("fixed")],
                      [_scen(day_id=0), _scen(day_id=1)], config, seed=1)
    s = out["summary"]["fixed"]
    assert s["n_records"] == 2
    assert s["n_included"] + s["n_excluded"] == s["n_records"]
    assert "n_included" in s["denominator"]
    assert s["exclusion_reasons"].get(ue.EXCLUDED_SOLVER_FAILURE) == 1
    assert len(s["excluded_scenario_ids"]) == 1
    assert s["settled_net_profit_cny"] is not None


def test_a_wholly_failed_arm_reports_no_mean_rather_than_zero(monkeypatch,
                                                              config):
    monkeypatch.setattr(ue.ce, "settle_day",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    out = ue.evaluate([_arm("fixed")], [_scen()], config, seed=1)
    s = out["summary"]["fixed"]
    assert s["n_included"] == 0
    assert s["settled_net_profit_cny"] is None      # not 0.0
    assert s["n_excluded"] == 1


def test_excluded_records_are_retained_not_deleted(config, monkeypatch):
    monkeypatch.setattr(ue.ce, "settle_day",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    out = ue.evaluate([_arm("fixed")], [_scen()], config, seed=1)
    assert len(out["records"]) == 1
    assert out["records"][0]["exclusion_reason"] is not None


# ---------------------------------------------------------------------------
# Contracts on the inputs themselves
# ---------------------------------------------------------------------------

def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="not a whole-day strategy"):
        ue.ArmRef(label="x", family="whole_day", name="not_a_strategy")
    with pytest.raises(ValueError, match="not a per-block kind"):
        ue.ArmRef(label="x", family="per_block", name="not_a_kind")
    with pytest.raises(ValueError, match="unknown arm family"):
        ue.ArmRef(label="x", family="nonsense", name="fixed")


def test_an_empty_evaluation_is_refused(config):
    with pytest.raises(ValueError, match="no arms"):
        ue.evaluate([], [_scen()], config, seed=1)
    with pytest.raises(ValueError, match="no scenarios"):
        ue.evaluate([_arm("fixed")], [], config, seed=1)


def test_results_serialise(config):
    out = ue.evaluate([_arm("fixed")], [_scen()], config, seed=1)
    text = ue.to_json(out)
    assert "settled_net_profit_cny" in text
    assert '"mode": "evaluation"' in text


# ---------------------------------------------------------------------------
# The exogenous-input digest
# ---------------------------------------------------------------------------

def test_the_digest_covers_inputs_beyond_price(config):
    """Price, load, renewables, agents, economics and network each get their own
    component, so `assert_comparable` is not a price-only check."""
    rec = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=1)
    d = rec["scenario_input_digest"]
    for key in ("price", "load", "renewables", "agents", "economics",
                "network", "combined"):
        assert key in d, key
        assert isinstance(d[key], str) and len(d[key]) == 64, key
    # The combined value is not simply the price digest.
    assert d["combined"] != d["price"]


def test_a_load_difference_with_identical_prices_is_refused(config):
    """The failure the price-only check could not catch: same price curve, same
    SOC, same economics -- different load. A profit difference there would be
    the day's, not the strategy's."""
    rec = ue.evaluate_arm(_arm("fixed"), _scen(), config, seed=1)
    tampered = copy.deepcopy(rec)
    # Move ONLY the load component, leaving price/SOC/economics identical.
    tampered["scenario_input_digest"]["load"] = "f" * 64
    tampered["scenario_input_digest"]["combined"] = "e" * 64
    assert tampered["scenario_freeze"]["wholesale_sha256"] == \
        rec["scenario_freeze"]["wholesale_sha256"]
    with pytest.raises(AssertionError, match="load"):
        ue.assert_comparable([rec, tampered])


def test_the_digest_excludes_strategy_state(config):
    """Behavioural, not a source grep: strategies that declare *different*
    quotes on the same day must still share one digest, or the digest would
    confuse a strategy difference with a scenario difference.

    Checked across three arms whose declarations differ substantially, and
    against a per-block arm too -- if anything strategy-derived had been folded
    in, these would not agree.
    """
    scen = _scen()
    digests = {}
    for label in ("fixed", "mpc", "greedy_threshold"):
        digests[label] = ue.evaluate_arm(_arm(label), scen, config,
                                         seed=1)["scenario_input_digest"]
    digests["truthful"] = ue.evaluate_arm(
        ue.ArmRef(label="truthful", family="per_block", name="truthful"),
        scen, config, seed=1)["scenario_input_digest"]
    combined = {v["combined"] for v in digests.values()}
    assert len(combined) == 1, digests
    # NOTE: a "the arms must have earned different amounts" sanity check does
    # NOT work on this scenario, and its failure would be informative rather
    # than a bug: on these prices the arbitrage window is shut, so every arm --
    # including `greedy_threshold`, which declares heavily and is filled in
    # none -- settles at ~0.0 CNY. Equal profit here is not evidence that the
    # arms behaved alike, so profit is not used as the discriminator. The
    # component digests are compared directly above, which is the property under
    # test: nothing strategy-derived reached the fingerprint.


def test_arms_on_the_same_scenario_share_one_digest(config):
    scen = _scen()
    a = ue.evaluate_arm(_arm("fixed"), scen, config, seed=9)
    b = ue.evaluate_arm(_arm("mpc"), scen, config, seed=9)
    assert a["scenario_input_digest"] == b["scenario_input_digest"]


# ---------------------------------------------------------------------------
# Paired statistics on the common valid set
# ---------------------------------------------------------------------------

def test_the_pairing_runs_on_the_common_valid_set(config):
    scens = [_scen(day_id=d) for d in (0, 1, 2)]
    out = ue.evaluate([_arm("fixed"), _arm("mpc")], scens, config, seed=1)
    p = out["paired"]
    assert p["comparable"] is True
    assert p["n_common"] == 3
    assert p["denominator"].startswith("n_common")
    assert set(p["per_scenario_delta_cny"]) == set(p["common_scenario_ids"])
    d = p["delta_settled_net_profit_cny"]
    assert d["n_positive"] + d["n_negative"] + d["n_zero"] == 3


def test_different_failure_sets_do_not_produce_a_ranking(config, monkeypatch):
    """The selection effect the pairing exists to remove: when the two arms keep
    different days, the per-arm means are over different denominators and their
    difference is partly an artefact. Below the threshold, no ranking."""
    real = ue.ce.settle_day
    state = {"n": 0}

    def fail_third(*a, **k):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("simulated")
        return real(*a, **k)

    monkeypatch.setattr(ue.ce, "settle_day", fail_third)
    # arm A: 3 scenarios, one fails -> 2 included. arm B (mpc) runs clean.
    out = ue.evaluate([_arm("fixed"), _arm("mpc")],
                      [_scen(day_id=d) for d in (0, 1, 2)], config, seed=1)
    p = out["paired"]
    assert p["n_included_a"] != p["n_included_b"]
    if p["n_common"] < ue.MIN_COMMON_SCENARIOS_FOR_RANKING:
        assert p["comparable"] is False
        assert p["delta_settled_net_profit_cny"] is None
        assert "denominator" not in p


def test_an_empty_common_set_refuses_a_ranking(monkeypatch, config):
    import copy as _copy
    recs = []
    for label in ("fixed", "mpc"):
        r = ue.evaluate_arm(_arm(label), _scen(), config, seed=1)
        r = _copy.deepcopy(r)
        r["arm_label"] = label
        # Make each arm fail a DIFFERENT scenario so the intersection is empty.
        r["included"] = False
        r["settled_net_profit_cny"] = None
        r["exclusion_reason"] = ue.EXCLUDED_SOLVER_FAILURE
        recs.append(r)
    p = ue.paired_summary(recs, "fixed", "mpc")
    assert p["n_common"] == 0
    assert p["comparable"] is False
    assert p["delta_settled_net_profit_cny"] is None
    assert "no ranking" in p["reason"] or "No ranking" in p["reason"]


def test_failures_are_counted_beside_the_pairing(config, monkeypatch):
    """Runnability is reported separately from return, and neither is hidden."""
    monkeypatch.setattr(ue.ce, "settle_day",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    out = ue.evaluate([_arm("fixed"), _arm("mpc")], [_scen()], config, seed=1)
    p = out["paired"]
    assert p["n_failed_a"] == 1 and p["n_failed_b"] == 1
    assert p["comparable"] is False


def test_a_single_arm_run_has_no_pairing(config):
    out = ue.evaluate([_arm("fixed")], [_scen()], config, seed=1)
    assert "paired" not in out
