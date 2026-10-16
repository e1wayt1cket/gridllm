# tests/test_storage_economics_audit.py
"""Contracts of the storage-economics accounting layer.

Three defects were found by audit and are each pinned here as a falsifiable
regression, plus the accounting invariants that must hold whichever way the
code is arranged.

The three, briefly:

1. **The evaluation CLI reported a learning signal as a payoff.** `main()`
   summed the episode's per-block rewards and published the result under
   `profit_delta`, which is the same column name `evaluate_policy_dir` uses for
   the endpoint-pinned settled payoff. The two are not close: on the native day
   the rolling sum runs two to five times the settled payoff *and ranks the
   units differently*.
2. **The baseline cache was keyed by `(scenario, day_id, T)`.** A capacity or
   siting sweep passes the same triple and would have been paired against a
   baseline belonging to another fleet, silently, because the only check
   (`assert_paired`) compares the price curve, which a fleet change does not
   move.
3. **The trained-policy cache was keyed by agent name.** Independent-storage
   names are identical in every scenario, so a policy trained for one day
   applied to another without a word.

The tests that would have caught each are the ones that state the failure
directly: the CLI's number equals the settled one and not the rolling one; the
key moves when the fleet moves; the second load wins and a foreign spec is
refused.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses

import numpy as np
import pytest

import counterfactual_eval as ce
import eval_agents as ea
import rl_bidding as rb
import wholesale_price as wp
from config_loader import get_default, reload_defaults, set_default
from models import MarketConfig
from scenarios import get_scenario

RL_NAMES = ["ESS9", "ESS14", "ESS17", "ESS27"]


# ---------------------------------------------------------------------------
# Fixtures: one native day, built and run once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def day():
    """The native day's agents, and a full-day truthful episode captured."""
    reload_defaults()
    wp.reset_cache()
    config = MarketConfig.from_yaml()
    agents, wholesale = get_scenario("typical_day", T=96, config=config)

    from rl_env import BiddingEnv
    np.random.seed(1007)
    env = BiddingEnv(agents, config)
    result = ea.run_episode(env, policy=None, capture=True)
    return {"agents": agents, "config": config, "wholesale": wholesale,
            "capture": result}


# ---------------------------------------------------------------------------
# 1. One settled payoff, two entry points
# ---------------------------------------------------------------------------

def test_settle_capture_is_the_shared_settlement(day):
    """The CLI's helper must BE `settle_day`, not a second implementation."""
    profits, clean = ea.settle_capture(day["agents"], day["config"], RL_NAMES,
                                       day["capture"])
    expected, expected_clean, _ = ce.settle_day(
        day["agents"], day["config"], RL_NAMES,
        day["capture"]["declared"], day["capture"]["wholesale"], 96)
    assert clean == expected_clean
    for nm in RL_NAMES:
        assert profits[nm] == pytest.approx(expected[nm])


def test_the_settled_payoff_is_not_the_rolling_reward(day):
    """The defect, stated as a number.

    If these were equal the old CLI column would have been harmless. They are
    not: the rolling sum is a per-block cash tally with no endpoint pin, so a
    unit that ends the day with an emptier battery scores higher on it.
    """
    settled, clean = ea.settle_capture(day["agents"], day["config"], RL_NAMES,
                                       day["capture"])
    assert clean
    rolling = day["capture"]["profits"]
    assert any(abs(rolling[nm] - settled[nm]) > 1.0 for nm in RL_NAMES), (
        "rolling and settled agreed on every unit; this test no longer "
        "demonstrates anything")


def test_the_rolling_reward_is_not_the_research_kpi(day):
    """Reporting the rolling sum as the payoff would overstate every unit."""
    settled, _ = ea.settle_capture(day["agents"], day["config"], RL_NAMES,
                                   day["capture"])
    rolling = day["capture"]["profits"]
    assert sum(rolling[nm] for nm in RL_NAMES) > \
        sum(settled[nm] for nm in RL_NAMES)


def test_a_degraded_capture_settles_to_nan_not_zero(day):
    """A fallen-back block must be reported as degraded, never averaged in."""
    broken = dict(day["capture"])
    broken["usable"] = False
    profits, clean = ea.settle_capture(day["agents"], day["config"], RL_NAMES,
                                       broken)
    assert clean is False
    assert all(np.isnan(v) for v in profits.values())


def test_both_money_families_are_present_and_separately_named():
    """Rolling reward and settled payoff must not share a column."""
    money_columns = ["baseline_profit", "rl_profit", "profit_delta",
                     "baseline_rolling_reward", "rl_rolling_reward",
                     "rolling_reward_delta"]
    for col in money_columns:
        assert col in ea.MONEY_COLUMN_MEANINGS, col
    assert "rolling" not in ea.MONEY_COLUMN_MEANINGS["profit_delta"]
    assert "settled" in ea.MONEY_COLUMN_MEANINGS["profit_delta"]
    # The two families are disjoint: no column carries both meanings.
    settled_only = {"baseline_profit", "rl_profit", "profit_delta"}
    rolling_only = {"baseline_rolling_reward", "rl_rolling_reward",
                    "rolling_reward_delta"}
    assert settled_only & rolling_only == set()


# ---------------------------------------------------------------------------
# 2. The baseline cache key
# ---------------------------------------------------------------------------

def _key(day, **over):
    kwargs = {"scenario": "typical_day", "day_id": 11, "T": 96,
              "config": day["config"], "rl_names": RL_NAMES}
    kwargs.update(over)
    return ce.baseline_cache_key(**kwargs)


def test_identical_inputs_produce_an_identical_key(day):
    assert _key(day) == _key(day)


def test_different_config_objects_with_equal_content_share_a_key(day):
    """Content, not object identity: a deep copy must not defeat the cache.

    The copy must match the fixture's config *as it stands*, not a freshly
    read one: building the scenario applies its `override_config` to the object
    in place, so the fixture's config legitimately differs from the file.
    """
    import copy as _copy
    twin = _copy.deepcopy(day["config"])
    assert twin is not day["config"]
    assert _key(day, config=twin) == _key(day)


def test_a_config_differing_in_content_moves_the_key(day):
    """The other half: equal content shares, different content does not."""
    altered = dataclasses.replace(
        day["config"],
        market_design=dataclasses.replace(day["config"].market_design,
                                          penalty_unserved=9999.0))
    assert _key(day, config=altered) != _key(day)


def test_a_capacity_change_moves_the_key(day):
    reload_defaults()
    before = _key(day)
    set_default("independent_storage.capacity_mwh", 3.2)
    try:
        assert _key(day) != before, (
            "a fleet capacity change left the cache key unchanged; a capacity "
            "sweep would be paired against the previous fleet's baseline")
    finally:
        reload_defaults()


def test_a_siting_change_moves_the_key(day):
    reload_defaults()
    before = _key(day)
    set_default("independent_storage.buses", [2, 12, 13, 19])
    try:
        assert _key(day) != before
    finally:
        reload_defaults()


def test_a_price_source_change_moves_the_key(day):
    import pathlib
    import yaml
    root = pathlib.Path(__file__).resolve().parent.parent
    path = root / "config" / "scenarios.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    before = _key(day)
    # Read the scenario's declared source rather than the live global.
    assert ce.baseline_cache_key(
        "typical_day_industry", 11, 96, day["config"], RL_NAMES) != before
    assert raw["scenarios"]["typical_day_industry"]["wholesale_source"] \
        == "dianyicha"


def test_day_T_and_agent_set_all_move_the_key(day):
    base = _key(day)
    assert _key(day, day_id=12) != base
    assert _key(day, T=48) != base
    assert _key(day, rl_names=["ESS9"]) != base
    assert _key(day, scenario="baseline") != base


def test_the_spec_identity_moves_the_key(day):
    from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V3
    with_specs = _key(day, obs_spec=OBS_V3, action_spec=ACTION_STORAGE_QUOTE_V2)
    assert with_specs != _key(day)
    assert with_specs != _key(day, obs_spec=None, action_spec=ACTION_STORAGE_QUOTE_V2)


def test_the_key_does_not_read_a_mutable_global(day):
    """The price source is taken from the scenario, not the live global.

    `_build_scenario` rewrites that global on every build, so keying on it
    would make the key depend on which scenario happened to be built last.
    """
    reload_defaults()
    first = _key(day)
    set_default("wholesale_market.source", "dianyicha")
    try:
        assert _key(day) == first
    finally:
        reload_defaults()


def test_a_cached_baseline_is_reused_for_identical_input_and_not_for_a_new_fleet(
        day, monkeypatch):
    """The end-to-end statement: same inputs reuse, a moved fleet does not.

    Runs `validation_kpis` with the cache on, counts how many times the
    truthful arm is actually cleared, and checks the count against the fleet.
    """
    reload_defaults()
    wp.reset_cache()
    calls = {"n": 0}
    real_run_arm = ce.run_arm

    def counting_run_arm(*args, **kwargs):
        calls["n"] += 1
        return real_run_arm(*args, **kwargs)

    monkeypatch.setattr(ce, "run_arm", counting_run_arm)
    ce.reset_baseline_cache()

    config = MarketConfig.from_yaml()
    from rl_env import BiddingEnv
    env = BiddingEnv(day["agents"], config, rl_agent_names=RL_NAMES)
    actors = _actors(RL_NAMES, obs_dim=env.get_state_dim(),
                     bounds=env.get_action_bounds())
    kwargs = dict(scenario="typical_day", rl_names=RL_NAMES, day_ids=[11],
                  obs_spec=None, action_spec=None)
    try:
        ce.validation_kpis(actors, config, use_cache=True, **kwargs)
        first = calls["n"]
        assert first == 2, "expected one baseline arm and one AI arm"

        # Same inputs again: the baseline must come from the cache, so only the
        # AI arm clears.
        ce.validation_kpis(actors, config, use_cache=True, **kwargs)
        assert calls["n"] == first + 1, (
            "the cached baseline was not reused for identical inputs")

        # A different fleet must NOT reuse it.
        set_default("independent_storage.capacity_mwh", 3.2)
        ce.validation_kpis(actors, config, use_cache=True, **kwargs)
        assert calls["n"] == first + 3, (
            "a capacity change reused the previous fleet's cached baseline")
    finally:
        reload_defaults()
        ce.reset_baseline_cache()


# ---------------------------------------------------------------------------
# 3. Trained-policy provenance
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_policy_cache():
    rb.reset_trained_policies()
    yield
    rb.reset_trained_policies()


def _actors(names, obs_dim=12, act_dim=2, bounds=None):
    import torch
    from rl_td3 import Actor
    if bounds is None:
        bounds = torch.tensor([[-1.0] * act_dim, [1.0] * act_dim])
    return {nm: Actor(obs_dim, int(bounds.shape[1]), bounds) for nm in names}


def _save(path, names, obs_spec, action_spec):
    rb.save_policies(_actors(names), str(path), obs_spec=obs_spec,
                     action_spec=action_spec)


def test_the_cache_records_where_each_policy_came_from(tmp_path):
    from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V3
    path = tmp_path / "a.pt"
    _save(path, ["ESS9", "ESS14"], OBS_V3, ACTION_STORAGE_QUOTE_V2)
    rb.load_policies(str(path), 12, _actors(["ESS9"])["ESS9"]._bounds()
                     if hasattr(_actors(["ESS9"])["ESS9"], "_bounds")
                     else __import__("torch").tensor([[-1.0, -1.0], [1.0, 1.0]]),
                     obs_spec=OBS_V3, action_spec=ACTION_STORAGE_QUOTE_V2)
    prov = rb.policy_provenance("ESS9")
    assert prov is not None
    assert prov["source"] == str(path)
    assert prov["obs_spec"] == OBS_V3.name


def test_a_second_file_wins_for_the_same_agent_name(tmp_path):
    """Loading B after A must leave B's network under the shared name."""
    import torch
    from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V3
    bounds = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])

    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    _save(a, ["ESS9"], OBS_V3, ACTION_STORAGE_QUOTE_V2)
    _save(b, ["ESS9"], OBS_V3, ACTION_STORAGE_QUOTE_V2)
    # Make B's weights distinguishable from A's.
    for p in _actors(["ESS9"])["ESS9"].parameters():
        pass
    state = torch.load(str(b), map_location="cpu", weights_only=False)
    state["ESS9"]["net.0.weight"] = torch.full_like(
        state["ESS9"]["net.0.weight"], 0.5)
    torch.save(state, str(b))

    rb.load_policies(str(a), 12, bounds, obs_spec=OBS_V3,
                     action_spec=ACTION_STORAGE_QUOTE_V2)
    assert rb.policy_provenance("ESS9")["source"] == str(a)
    rb.load_policies(str(b), 12, bounds, obs_spec=OBS_V3,
                     action_spec=ACTION_STORAGE_QUOTE_V2)
    assert rb.policy_provenance("ESS9")["source"] == str(b), (
        "the second load did not take over the name")

    w = rb._TRAINED_POLICIES["ESS9"].net[0].weight
    assert torch.allclose(w, torch.full_like(w, 0.5)), (
        "the cached network is still the first file's")


def test_reset_clears_the_cache_and_the_provenance_together():
    rb._TRAINED_POLICIES["ESS9"] = _actors(["ESS9"])["ESS9"]
    rb._record_provenance(["ESS9"], source="somewhere")
    assert rb.policy_provenance("ESS9") is not None
    rb.reset_trained_policies()
    assert not rb._TRAINED_POLICIES
    assert rb.policy_provenance("ESS9") is None


def test_a_foreign_observation_spec_is_refused(day):
    """`ESS9` here is not `ESS9` trained somewhere else."""
    from strategies.rl_bidding import _reject_foreign_cached_policies
    rb._TRAINED_POLICIES["ESS9"] = _actors(["ESS9"])["ESS9"]
    rb._record_provenance(["ESS9"], source="policies/other/ESS9.pt",
                          meta={"obs_spec": {"name": "v2_old_encoding"}})
    with pytest.raises(ValueError, match="observation spec"):
        _reject_foreign_cached_policies(rb._TRAINED_POLICIES, day["agents"])


def test_a_policy_with_no_recorded_provenance_is_not_refused(day):
    """An older checkpoint predates the record; the loader checked its spec."""
    from strategies.rl_bidding import _reject_foreign_cached_policies
    rb._TRAINED_POLICIES["ESS9"] = _actors(["ESS9"])["ESS9"]
    _reject_foreign_cached_policies(rb._TRAINED_POLICIES, day["agents"])


def test_a_matching_spec_is_accepted(day):
    """A policy cached under the environment's own observation spec is applied;
    one cached under a different spec is refused.

    The refusal half is the load-bearing one. `OBS_V3` fed
    `avg_lmp_norm` from the whole day's realized price, so a V3 actor was
    trained on a vector this environment no longer produces; applying it
    silently would be a contract error dressed as a performance result.
    """
    from rl_spec import OBS_V6
    from strategies.rl_bidding import _reject_foreign_cached_policies
    try:
        rb._TRAINED_POLICIES["ESS9"] = _actors(["ESS9"])["ESS9"]
        rb._record_provenance(["ESS9"], source="x",
                              meta={"obs_spec": {"name": OBS_V6.name}})
        _reject_foreign_cached_policies(rb._TRAINED_POLICIES, day["agents"])
    finally:
        rb._TRAINED_POLICIES.pop("ESS9", None)


def test_a_v3_policy_is_refused_by_the_v4_environment(day):
    """The retired contract is recorded, not reinterpreted.

    This is what makes the surviving `policies/formal_v0_1_seed*` checkpoints
    safe to keep on disk: they are identifiable as V3, and nothing can quietly
    run them under a layout they were not trained on.
    """
    import pytest
    from rl_spec import OBS_V3, OBS_V6
    from strategies.rl_bidding import _reject_foreign_cached_policies
    assert OBS_V3.name != OBS_V6.name
    try:
        rb._TRAINED_POLICIES["ESS9"] = _actors(["ESS9"])["ESS9"]
        rb._record_provenance(["ESS9"], source="x",
                              meta={"obs_spec": {"name": OBS_V3.name}})
        with pytest.raises(ValueError, match="observation spec"):
            _reject_foreign_cached_policies(rb._TRAINED_POLICIES,
                                           day["agents"])
    finally:
        rb._TRAINED_POLICIES.pop("ESS9", None)


def test_the_v3_checkpoint_dirs_are_recorded_as_v3(tmp_path):
    """A policy file saved under the retired spec cannot be loaded as V4.

    Asserted on a file rather than only on the in-memory cache: the on-disk
    checkpoints in `policies/` are what this protects.
    """
    import torch
    from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V3, OBS_V6
    path = tmp_path / "v3.pt"
    _save(path, ["ESS9"], OBS_V3, ACTION_STORAGE_QUOTE_V2)
    bounds = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="observation spec"):
        rb.load_policies(str(path), 12, bounds,
                         obs_spec=OBS_V6, action_spec=ACTION_STORAGE_QUOTE_V2)


def test_an_action_spec_mismatch_on_load_raises(tmp_path):
    from rl_spec import ACTION_BID_OFFER_V1, ACTION_STORAGE_QUOTE_V2, OBS_V3
    import torch
    path = tmp_path / "a.pt"
    _save(path, ["ESS9"], OBS_V3, ACTION_STORAGE_QUOTE_V2)
    with pytest.raises(ValueError, match="action spec"):
        rb.load_policies(str(path), 12,
                         torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
                         obs_spec=OBS_V3, action_spec=ACTION_BID_OFFER_V1)


def test_an_observation_spec_mismatch_on_load_raises(tmp_path):
    from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V3
    import torch
    path = tmp_path / "a.pt"
    _save(path, ["ESS9"], OBS_V3, ACTION_STORAGE_QUOTE_V2)
    # `to_dict` carries derived keys the constructor does not take, so the
    # altered spec is built by replacement rather than round-tripped.
    older = dataclasses.replace(OBS_V3, name="v0_old")
    with pytest.raises(ValueError, match="observation spec"):
        rb.load_policies(str(path), 12,
                         torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
                         obs_spec=older, action_spec=ACTION_STORAGE_QUOTE_V2)


# ---------------------------------------------------------------------------
# 4. Accounting invariants
# ---------------------------------------------------------------------------

def test_the_period_is_a_quarter_hour():
    import money
    assert money.DT_HOURS == 0.25
    assert money.PERIODS_PER_DAY == 96


def test_the_payoff_breakdown_reconciles(day):
    """Revenue less every cost equals the reported total, term by term."""
    import participant_payoff
    settled_result = None
    profits, clean, result = ce.settle_day(
        day["agents"], day["config"], RL_NAMES,
        day["capture"]["declared"], day["capture"]["wholesale"], 96)
    assert clean and result is not None

    for nm in RL_NAMES:
        agent = next(a for a in day["agents"] if a.name == nm)
        sched = result["schedules"][nm]
        bd = participant_payoff.participant_payoff(
            sched, result["lmp"][:, agent.bus], agent, day["config"])
        parts = (bd.revenue - bd.purchase_cost - bd.generation_cost
                 - bd.degradation_cost - bd.other_cost - bd.penalty)
        assert bd.total == pytest.approx(parts, abs=1e-6), nm
        assert profits[nm] == pytest.approx(bd.total, abs=1e-6), nm


def test_a_storage_unit_is_not_charged_for_load_or_generation(day):
    """An independent battery serves no load and owns no generator."""
    import participant_payoff
    _, _, result = ce.settle_day(
        day["agents"], day["config"], RL_NAMES,
        day["capture"]["declared"], day["capture"]["wholesale"], 96)
    for nm in RL_NAMES:
        agent = next(a for a in day["agents"] if a.name == nm)
        bd = participant_payoff.participant_payoff(
            result["schedules"][nm], result["lmp"][:, agent.bus], agent,
            day["config"])
        assert bd.generation_cost == 0.0, nm
        assert bd.penalty == 0.0, nm


def test_other_cost_is_zero_and_not_claimed_otherwise():
    """Confirmed, not fixed: the field exists and is never assigned.

    No cost is invented here. The point of the assertion is that a report
    cannot claim the breakdown covers items it does not.
    """
    import participant_payoff
    bd = participant_payoff.PayoffBreakdown(
        revenue=100.0, purchase_cost=40.0, generation_cost=0.0,
        degradation_cost=10.0, penalty=0.0)
    assert bd.other_cost == 0.0
    assert bd.total == pytest.approx(50.0)


def test_both_arms_settle_with_the_endpoints_pinned(day):
    """A paired comparison is only paired if the SOC rule is the same."""
    import dataclasses as dc
    _, _, result = ce.settle_day(
        day["agents"], day["config"], RL_NAMES,
        day["capture"]["declared"], day["capture"]["wholesale"], 96)
    for nm in RL_NAMES:
        soc = np.asarray(result["schedules"][nm]["soc"])
        agent = next(a for a in day["agents"] if a.name == nm)
        assert soc[0] == pytest.approx(agent.storage.soc0, abs=1e-6), nm
        assert soc[-1] == pytest.approx(soc[0], abs=1e-5), nm


def test_the_settlement_pins_the_endpoints_regardless_of_the_config(day):
    """`_settle_full_day` forces `terminal_soc_equal`; it is not optional."""
    loose = dataclasses.replace(
        day["config"],
        storage=dataclasses.replace(day["config"].storage,
                                    terminal_soc_equal=False))
    settled, clean, result = ce.settle_day(
        day["agents"], loose, RL_NAMES,
        day["capture"]["declared"], day["capture"]["wholesale"], 96)
    assert clean
    for nm in RL_NAMES:
        soc = np.asarray(result["schedules"][nm]["soc"])
        assert soc[-1] == pytest.approx(soc[0], abs=1e-5), nm


# ---------------------------------------------------------------------------
# 5. The phases below still hold
# ---------------------------------------------------------------------------

def test_phase_1_interface_is_intact():
    import industry_load as il
    assert il.N_SLOTS == 96
    assert callable(il.load_shapes_csv)
    assert callable(il.reading_to_mw)


def test_phase_2_interface_is_intact():
    import dianyicha_price as dp
    assert dp.SHEETS == ("261009", "7days", "30days", "halfyear")
    curves = dp.read_curves()
    for name, c in curves.items():
        # Taxonomy sanity, not a single-day claim.
        assert c.source_kind == "real_market_data", name
        assert c.temporal_statistic in dp.TEMPORAL_STATISTICS, name


def test_phase_4_scenario_is_intact():
    """The scenario still builds the same *facilities*; its load is the ledger's.

    The peak used to be pinned to the benchmark's 3.715 MW. It is not any more:
    this scenario now takes its per-bus MW from the node-load ledger, so the
    peak is the metered one. Renewable capacity is unaffected -- it is declared
    in the scenario, not derived from the load.
    """
    reload_defaults()
    wp.reset_cache()
    agents, wholesale = get_scenario("typical_day_industry", T=96)
    loads = [a for a in agents if a.load_type != "storage"]
    peak = float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())
    assert peak > 100.0, "the ledger's load, not the benchmark's 3.715 MW"
    assert peak != pytest.approx(3.715, abs=1e-6)
    assert sum(float(a.pv_capacity or 0.0) for a in agents) == pytest.approx(0.6)
    reload_defaults()


def test_an_existing_scenario_still_builds_unchanged():
    reload_defaults()
    wp.reset_cache()
    agents, _ = get_scenario("typical_day", T=96)
    loads = [a for a in agents if a.load_type != "storage"]
    peak = float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())
    assert peak == pytest.approx(3.715, abs=1e-6)
    assert [a.name for a in agents if a.participant_type == "storage"] == RL_NAMES
    reload_defaults()
