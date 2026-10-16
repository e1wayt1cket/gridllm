# counterfactual_eval.py
"""The economic gain of an assisted bidding policy over a baseline policy.

This module owns the research question's key performance indicators. They are
defined on one participant's settled profit over one day:

    Pi_AI          the assisted arm's settled profit
    Pi_baseline    the baseline arm's settled profit
    DeltaPi        = Pi_AI - Pi_baseline
    DeltaPi_pct    = DeltaPi / (|Pi_baseline| + eps)
    P(DeltaPi > 0) how often the assisted arm came out ahead

`Pi` is a *participant payoff*, computed by `participant_payoff` from what the
unit was actually settled at, and never the clearing objective: the objective
values a unit at the quotes it declared, while settlement happens at the nodal
price, and only the second is money the unit keeps.

Everything here exists to make the difference mean what it says. Each arm of a
pair is run with the same RNG seed over the same scenario, so load, on-site
generation, the initial state of charge, the network and the price realization
are identical and the only thing that differs is the bidding policy. Both arms
are then settled the same way -- one joint clear of the whole day with the
state of charge pinned at both ends -- so neither can win by ending the day
with a fuller or emptier battery. The pair is *checked*, not assumed: if the
two arms did not see the same day, the comparison is refused rather than
reported.

The baseline is a whole-day policy applied to every strategic agent. That is a
different counterfactual from the one the training reward subtracts, which is
taken inside a single window and holds the other agents at their learned bids
(see rl_env.step). A training reward is not a payoff and must never be read as
one.

`delta_pi_percent` divides by the magnitude of the baseline, so a baseline that
loses money -- a battery whose day's spread does not cover its degradation --
still yields a rate that reads the right way round.
"""

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from models import MarketConfig
from rl_env import BiddingEnv, N_BLOCKS
from scenarios import get_scenario

# Guard against a zero denominator for an exactly break-even baseline.
RATE_EPSILON = 1e-6

# Statistics reported per agent and for the fleet, for every quantity.
# `positive_rate` is the generic form of P(DeltaPi > 0): the share of days the
# quantity was above zero. It is a statistic of whichever quantity it is
# attached to, so the delta's share is read as `delta_pi_positive_rate`.
STAT_COLUMNS = ("n", "mean", "std", "median", "p05", "p25", "p50", "p75",
                "p95", "positive_rate")


def delta_pi(profit_ai: float, profit_baseline: float) -> float:
    """How much more the assisted policy earned (CNY)."""
    return float(profit_ai) - float(profit_baseline)


def delta_pi_percent(profit_ai: float, profit_baseline: float) -> float:
    """The benefit relative to the baseline's magnitude.

    The denominator is the magnitude, not the signed value, so that improving
    on a baseline that lost money is a positive rate rather than a negative one.
    """
    return ((float(profit_ai) - float(profit_baseline))
            / (abs(float(profit_baseline)) + RATE_EPSILON))


def summarize(values: Sequence[float]) -> dict:
    """Distribution of a per-day quantity across days.

    Percentiles rather than only a mean, because the question the research asks
    is whether the gain is *stable*: a mean lifted by two good days and a
    positive_rate near one half are different findings, and only the pair
    distinguishes them.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {c: float("nan") for c in STAT_COLUMNS}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "positive_rate": float(np.mean(arr > 0)),
    }


@dataclass(frozen=True)
class ArmSpec:
    """One arm of a comparison."""

    name: str
    #: "truthful" / "rule" / "myopic", or "rl" to use the loaded actors.
    kind: str
    policies: Optional[Dict[str, object]] = None


def is_degraded(result) -> bool:
    """Whether a cleared day may not be reported as a payoff.

    Two ways a clear stops being a result. It fell back -- the whole path
    degraded to the single-period solver. Or its price field is mixed: some
    periods are not nodal prices at all, and the payoff is defined on the nodal
    price. The second was invisible until the fallback report named it, and a
    day settled on a partly-wholesale price field was reported as clean.

    One definition, used by the block-level reward loop and the day-level
    settlement, so the two cannot disagree about which days count.
    """
    return bool(result is None
                or result.get("fell_back")
                or result.get("has_wholesale_fallback"))



def run_arm(scenario: str, config: MarketConfig, arm: ArmSpec,
            rl_names: Sequence[str], day_seed: int, T: int = 96,
            n_blocks: int = N_BLOCKS, settle_full_day: bool = True,
            day_id: Optional[int] = None, obs_spec=None,
            action_spec=None) -> dict:
    """Run one policy over one day and return what each agent settled.

    ``day_seed`` is required and applied before anything else, so every arm of a
    pair starts from the same RNG state and therefore sees the same day. Making
    it mandatory is the point: an unseeded arm cannot be paired with anything.

    Returns per-agent profit, the price curve the day was built from (which is
    what the pairing check compares), and whether the run was clean.
    """
    import baselines
    from eval_agents import actor_predict

    if arm.kind != "rl":
        policy = baselines.make_baseline_policy(arm.kind, config)
    else:
        if not arm.policies:
            raise ValueError(f"arm {arm.name!r} needs loaded policies")
        policy = None

    np.random.seed(day_seed)
    # `day_id` selects which day of the wholesale library this arm is priced on.
    # It is orthogonal to `day_seed`: the seed pairs the two arms' residual
    # noise, the day chooses the price. Both arms of a pair get the same one of
    # each, which is what keeps the wholesale arrays comparable.
    agents, wholesale = get_scenario(scenario, T=T, config=config,
                                     day_id=day_id)
    # `obs_spec`/`action_spec` must match the environment the policy was
    # TRAINED in. A validation env built on the defaults would put a trained
    # actor in a differently shaped observation or a differently bounded action
    # space, and would produce a plausible, wrong profit rather than an error.
    env = BiddingEnv(agents, config, rl_agent_names=list(rl_names),
                     obs_spec=obs_spec, action_spec=action_spec)

    profits: Dict[str, float] = {nm: 0.0 for nm in rl_names}
    clean = True
    obs = env.reset()
    for _ in range(n_blocks):
        if arm.kind == "rl":
            # Trained actors take a torch batch; a baseline takes the raw
            # observation. Both come back as the same action vector.
            acts = {nm: actor_predict(arm.policies[nm], obs[nm])
                    for nm in rl_names}
        else:
            acts = {nm: np.asarray(policy(obs[nm]), dtype=np.float32)
                    for nm in rl_names}
        obs, _rewards, done, info = env.step(acts)
        # A block whose clearing degraded or fell back holds zeros; counting it
        # would silently deflate this arm's profit relative to a clean one.
        if is_degraded(env._last_result):
            clean = False
        for nm in rl_names:
            profits[nm] += float(info.get("raw_profit", {}).get(nm, 0.0))
        if done:
            break

    rolling_profits = dict(profits)
    day_result = None
    if settle_full_day:
        profits, settled, day_result = _settle_full_day(
            agents, config, rl_names, env.current_actions, wholesale, T,
            spec_for=env.spec_for)
        clean = clean and settled

    return {"arm": arm.name, "profits": profits,
            "rolling_profits": rolling_profits,
            "settled": bool(settle_full_day), "wholesale": wholesale,
            "seed": day_seed, "clean": clean, "day_result": day_result}


def _settle_full_day(agents, config: MarketConfig, rl_names, actions, wholesale,
                     T: int, spec_for=None) -> tuple:
    """Re-clear the whole day at once with the state of charge pinned.

    A day that finishes with a fuller battery has banked energy it did not sell,
    so settling each block's cash and adding them up rewards whichever policy
    ran the battery down. Clearing the horizon in one go with the endpoints tied
    makes the two arms comparable on the day itself, with no terminal price to
    choose. The action sequence is the one the policy emitted causally, block by
    block; only the settlement is joint.

    Returns (per-agent profit, clean, result). `result` is the cleared day
    itself -- network state, prices, schedules -- and is None when the clear
    degraded.
    """
    import dataclasses

    import participant_payoff
    from market import clear_market

    settle_config = dataclasses.replace(
        config,
        storage=dataclasses.replace(config.storage, terminal_soc_equal=True))
    if spec_for is None:
        from rl_spec import action_spec_for as spec_for

    settle_actions = {}
    for nm in rl_names:
        act = actions.get(nm)
        if act is None:
            continue
        agent = next(a for a in agents if a.name == nm)
        # The keys come from the agent's action space. Naming the retired pair
        # here would hand a storage unit `quote_arrays` refuses, turning a
        # settlement into a fallback the arm would then report as zero profit.
        settle_actions[nm] = {
            name: np.asarray(act[name], dtype=float)[:T]
            for name in spec_for(agent).action_names
        }

    try:
        result = clear_market(agents, T, "DA", settle_actions, settle_config,
                              wholesale=wholesale, horizon_type="full_day")
    except Exception:
        result = None
    if result is None or result.get("fell_back"):
        return {nm: 0.0 for nm in rl_names}, False, None
    if result.get("has_wholesale_fallback"):
        # The payoff is defined on the nodal price, and this day's is not one
        # for every period. The cleared day comes back with its marker rather
        # than being thrown away, so a caller that wants to look at it can --
        # what it may not do is quote this day's numbers as a settlement.
        return {nm: 0.0 for nm in rl_names}, False, result

    profits = {}
    for nm in rl_names:
        agent = next((a for a in agents if a.name == nm), None)
        sched = result["schedules"].get(nm)
        if agent is None or sched is None:
            profits[nm] = 0.0
            continue
        profits[nm] = participant_payoff.participant_payoff(
            sched, result["lmp"][:, agent.bus], agent, settle_config).total
    return profits, True, result


#: The truthful arm does not depend on the policy under test, so its settled
#: profit on a validation day is computed once per process and reused.
#:
#: The key has to name everything the baseline's settled payoff depends on, not
#: just the day. It once held `(scenario, day_id, T)` alone, which was enough
#: while nothing varied the fleet -- and wrong the moment something did: a
#: capacity or siting sweep passes the same `(scenario, day_id, T)` and gets a
#: baseline belonging to a different fleet. `assert_paired` cannot catch it,
#: because it compares only the wholesale array, which a capacity change does
#: not move. See `baseline_cache_key`.
_BASELINE_CACHE: Dict[tuple, dict] = {}


def _config_fingerprint(config) -> str:
    """A stable digest of the whole configuration.

    Built from `dataclasses.asdict` rather than `repr` or `id`, so two equal
    configurations digest the same and the same configuration digests the same
    across processes. Every bound field is covered at once, which is what makes
    this survive a new field being added later.
    """
    import dataclasses
    import hashlib
    import json

    payload = json.dumps(dataclasses.asdict(config), sort_keys=True,
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


#: `defaults.yaml` blocks that the clearing reads but that are NOT bound to
#: `MarketConfig`, so a config digest alone would miss them. `profiles` is the
#: load shape and level, `independent_storage` is the whole fleet, and
#: `wholesale_market` is which market prices the day. A capacity or siting
#: sweep moves the second one and nothing else.
_FINGERPRINT_BLOCKS = ("profiles", "independent_storage", "prosumers",
                       "load_types", "non_prosumer_storage", "price_curve",
                       "wholesale_market", "network")


#: Keys inside those blocks that are rewritten by the build machinery rather
#: than chosen by the study, and so must not enter the digest.
#:
#: `wholesale_market.source` is written by `_build_scenario` on every build and
#: `wholesale_market.day_id` is the sticky caller knob, so reading either here
#: would make the key depend on which scenario happened to be built last rather
#: than on the one being keyed. Both are carried explicitly instead: the source
#: through the scenario's own declaration, the day through the `day_id`
#: argument. Everything else in the block -- the floor, the cap, the library
#: identity -- is a real input and stays in.
_VOLATILE_DEFAULT_KEYS = {"wholesale_market": ("source", "day_id")}


def _defaults_fingerprint() -> str:
    """A stable digest of the configuration the clearing actually reads.

    Read live from `config_loader`'s cache, so `set_default` mutations -- which
    is how a sweep moves the fleet -- are picked up rather than assumed away.
    """
    import hashlib
    import json

    from config_loader import load_defaults

    defaults = load_defaults()
    blocks = {}
    for block in _FINGERPRINT_BLOCKS:
        node = defaults.get(block)
        volatile = _VOLATILE_DEFAULT_KEYS.get(block)
        if isinstance(node, dict) and volatile:
            node = {k: v for k, v in node.items() if k not in volatile}
        blocks[block] = node
    payload = json.dumps(blocks, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _scenario_fingerprint(scenario: str) -> str:
    """A stable digest of the scenario's own declaration.

    Covers the scale, the load shape, the renewable placement, the multipliers
    and the `industry_map`, so a scenario edited in the file cannot reuse a
    baseline computed for its earlier self.
    """
    import hashlib
    import json

    from config_loader import get_scenario_cfg

    payload = json.dumps(get_scenario_cfg(scenario), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def baseline_cache_key(scenario: str, day_id, T: int, config,
                       rl_names: Sequence[str], obs_spec=None,
                       action_spec=None) -> tuple:
    """Everything the truthful arm's settled payoff depends on.

    A cached baseline is sound only if an identical key would recompute the
    same number. The key therefore carries:

    * the scenario, its declaration (scale, load shape, renewables, industry
      map) and its market;
    * the configuration object, whole;
    * the `defaults.yaml` blocks the clearing reads but the config object does
      not carry -- which is where the storage fleet lives, so a capacity or
      siting sweep lands here;
    * `T`, `day_id`, the agents being settled, and the obs/action spec the
      environment is built with.

    The price source is read from the scenario's own declaration rather than
    from the live global: `_build_scenario` writes that global on every build,
    so reading it here would make the key depend on which scenario happened to
    be built last rather than on this one.
    """
    from wholesale_price import get_scenario_cfg_source

    return (
        str(scenario),
        int(day_id) if day_id is not None else None,
        int(T),
        _config_fingerprint(config),
        _defaults_fingerprint(),
        _scenario_fingerprint(scenario),
        str(get_scenario_cfg_source(scenario)),
        tuple(sorted(str(n) for n in rl_names)),
        getattr(obs_spec, "name", None),
        getattr(action_spec, "name", None),
    )


def day_seed_for(day_id: int) -> int:
    """The RNG seed an arm is run with on a given library day.

    Derived from the day rather than drawn, and that is what makes the baseline
    cache sound: the cache stores one arm's result for a day, and it is only
    comparable to a later AI arm if both were run under the same seed. Drawing
    a fresh seed each evaluation would pair a cached baseline against a
    differently realised day, and `assert_paired` would be comparing two days
    while looking like it compared two policies.
    """
    return int(day_id) + 1_000


def _prepare_config(config: MarketConfig, scenario: str) -> None:
    """Put the config into the state the scenario's own build will leave it in.

    `_build_scenario` applies the scenario's `override_config` and its power
    factor to the config object **in place**. A cache key computed before the
    first build therefore describes a different configuration from one computed
    after it, and the baseline cache never hits -- which is how this was found:
    the second identical `validation_kpis` call missed its own cached baseline.

    Applying the same preparation first makes the key describe the
    configuration the arm will actually run under. `apply_scenario_overrides`
    is idempotent, so the build's own application afterwards changes nothing.
    """
    from config_loader import get_scenario_cfg
    from scenarios import apply_scenario_overrides

    cfg = get_scenario_cfg(scenario)
    override = dict(cfg.get("override_config", {}) or {})
    # `custom_loads` is a per-bus directive the builder pops before applying
    # the rest, because it is not a configuration field. Same reason here:
    # `apply_scenario_overrides` refuses a key that names no field.
    override.pop("custom_loads", None)
    apply_scenario_overrides(config, override)
    if "load_power_factor" in cfg:
        config.network.load_power_factor = float(cfg["load_power_factor"])


def validation_kpis(actors: Dict[str, object], config: MarketConfig,
                    scenario: str, rl_names: Sequence[str],
                    day_ids: Sequence[int], T: int = 96,
                    n_blocks: int = N_BLOCKS, obs_spec=None,
                    action_spec=None, use_cache: bool = True) -> dict:
    """The paired counterfactual on held-out days, as one number to select on.

    Returns mean Delta_Pi and the two pieces it is made of, plus `days`: one
    entry per retained day carrying `day_id`, `pi_ai`, `pi_baseline`,
    `delta_pi` and `delta_pi_percent`. The means are what a cycle is selected
    on; the per-day entries are what a reader checks the selection against.

    Only the AI arm moves between cycles -- the truthful baseline's settled
    profit is a function of the day and the configuration, not of the policy --
    so the baseline is computed once and cached, which halves the cost of
    evaluating every cycle without changing the result.

    The validation days must come from `day_rotation.validation_days()`. The
    test days are not reachable from here, and nothing in the training path
    passes them.
    """
    ai_arm = ArmSpec(name="ai", kind="rl", policies=dict(actors))
    base_arm = ArmSpec(name="baseline", kind="truthful")

    # Before anything is keyed: the config must already be in the state the
    # build will leave it in, or the key describes a configuration no arm ran.
    _prepare_config(config, scenario)

    pi_ai: List[float] = []
    pi_base: List[float] = []
    kept: List[int] = []
    dropped: List[int] = []

    for day_id in day_ids:
        seed = day_seed_for(day_id)

        # The key names the fleet and the configuration, not only the day. If it
        # cannot be constructed, the cache is DISABLED for this call rather than
        # keyed on something partial: a baseline reused across two fleets is a
        # wrong number that looks like a right one, and nothing downstream could
        # tell.
        try:
            key = baseline_cache_key(scenario, day_id, T, config, rl_names,
                                     obs_spec, action_spec)
        except Exception:
            key = None
        cacheable = bool(use_cache) and key is not None

        cached = _BASELINE_CACHE.get(key) if cacheable else None
        if cached is None:
            base = run_arm(scenario, config, base_arm, rl_names, seed, T,
                           n_blocks, day_id=day_id, obs_spec=obs_spec,
                           action_spec=action_spec)
            if cacheable:
                _BASELINE_CACHE[key] = base
        else:
            base = cached

        ai = run_arm(scenario, config, ai_arm, rl_names, seed, T, n_blocks,
                     day_id=day_id, obs_spec=obs_spec, action_spec=action_spec)

        # The two arms must have been priced on the same day, or the difference
        # between them is the difference between two days.
        assert_paired(base, ai)

        if not (base.get("clean") and ai.get("clean")):
            dropped.append(int(day_id))
            continue

        kept.append(int(day_id))
        pi_base.append(float(sum(base["profits"].values())))
        pi_ai.append(float(sum(ai["profits"].values())))

    deltas = [a - b for a, b in zip(pi_ai, pi_base)]
    # The per-day pairs, not only the mean over them. The mean is what the
    # cycle is selected on, but a mean alone cannot show whether the gain held
    # across the ten days or came from one -- and once this call returns, the
    # per-day profits are gone. Emitting them costs bookkeeping, not a solve.
    days = [
        {"day_id": day, "pi_ai": a, "pi_baseline": b,
         "delta_pi": delta_pi(a, b),
         "delta_pi_percent": delta_pi_percent(a, b)}
        for day, a, b in zip(kept, pi_ai, pi_base)
    ]
    return {
        "days": days,
        "mean_delta_pi": float(np.mean(deltas)) if deltas else float("nan"),
        "mean_pi_ai": float(np.mean(pi_ai)) if pi_ai else float("nan"),
        "mean_pi_baseline": float(np.mean(pi_base)) if pi_base else float("nan"),
        "p_delta_pi_positive": (float(np.mean([d > 0 for d in deltas]))
                                if deltas else float("nan")),
        "n_days": len(deltas),
        "n_days_dropped": len(dropped),
        "dropped_days": dropped,
    }


def reset_baseline_cache() -> None:
    """Drop the cached baseline arm, so a config change is picked up."""
    _BASELINE_CACHE.clear()


def settle_day(agents, config: MarketConfig, rl_names, declared: dict,
               wholesale, T: int = 96) -> tuple:
    """Settle one arm's day the way a pair of arms must be settled.

    Public entry to the endpoint-pinned settlement, so the evaluation CLI and
    the baseline ladder cannot settle the same day two different ways. See
    `_settle_full_day` for why the endpoints are pinned.

    Returns (per-agent profit, clean, result).
    """
    return _settle_full_day(agents, config, rl_names, declared, wholesale, T)


def assert_paired(base: dict, ai: dict) -> None:
    """Refuse a comparison whose arms did not run the same day.

    The two arms are run with the same seed, so their price curves must be equal
    element for element. Comparing arms that differ here would attribute the
    difference between two days to the bidding policy.
    """
    wb = np.asarray(base["wholesale"], dtype=float)
    wa = np.asarray(ai["wholesale"], dtype=float)
    if wb.shape != wa.shape or not np.allclose(wb, wa):
        raise ValueError(
            f"arms {base['arm']!r} and {ai['arm']!r} did not see the same "
            f"price day (seed {base['seed']} vs {ai['seed']}); their "
            "difference cannot be attributed to the bidding policy")


def evaluate(scenario: str, config: MarketConfig, ai_arm: ArmSpec,
             baseline_arm: ArmSpec, rl_names: Sequence[str],
             seeds: Sequence[int], T: int = 96,
             n_blocks: int = N_BLOCKS) -> dict:
    """Paired comparison over a set of days.

    Returns per-agent and fleet distributions of the baseline profit, the
    assisted profit, the benefit and the benefit rate, over the days that ran
    cleanly. Days where either arm degraded are dropped and counted, never
    averaged in.
    """
    per_agent: Dict[str, Dict[str, List[float]]] = {
        nm: {"pi_baseline": [], "pi_ai": [], "delta_pi": [],
             "delta_pi_percent": []}
        for nm in rl_names}
    fleet = {k: [] for k in ("pi_baseline", "pi_ai", "delta_pi",
                             "delta_pi_percent")}
    dropped = []

    for seed in seeds:
        base = run_arm(scenario, config, baseline_arm, rl_names, seed, T,
                       n_blocks)
        ai = run_arm(scenario, config, ai_arm, rl_names, seed, T, n_blocks)
        assert_paired(base, ai)
        if not (base["clean"] and ai["clean"]):
            dropped.append(seed)
            continue

        base_total = ai_total = 0.0
        for nm in rl_names:
            b = base["profits"][nm]
            a = ai["profits"][nm]
            base_total += b
            ai_total += a
            per_agent[nm]["pi_baseline"].append(b)
            per_agent[nm]["pi_ai"].append(a)
            per_agent[nm]["delta_pi"].append(delta_pi(a, b))
            per_agent[nm]["delta_pi_percent"].append(delta_pi_percent(a, b))
        fleet["pi_baseline"].append(base_total)
        fleet["pi_ai"].append(ai_total)
        fleet["delta_pi"].append(delta_pi(ai_total, base_total))
        fleet["delta_pi_percent"].append(delta_pi_percent(ai_total, base_total))

    return {
        "scenario": scenario,
        "ai_arm": ai_arm.name,
        "baseline_arm": baseline_arm.name,
        "seeds_requested": list(seeds),
        "seeds_dropped": dropped,
        "n_days": len(fleet["delta_pi"]),
        "fleet": {k: summarize(v) for k, v in fleet.items()},
        "per_agent": {nm: {k: summarize(v) for k, v in stats.items()}
                      for nm, stats in per_agent.items()},
    }


#: Headline columns, in the order a report should read them: how much was
#: gained, how large that is relative to the baseline, and how often it was
#: gained at all.
HEADLINE_COLUMNS = ("mean_delta_pi", "mean_delta_pi_percent",
                    "p_delta_pi_positive")


def headline(result: dict) -> dict:
    """The numbers the research question is answered with.

    `p_delta_pi_positive` is P(DeltaPi > 0), the share of days the assisted arm
    earned more. It sits beside the mean on purpose: a mean lifted by two good
    days and a share near one half are different findings, and neither number
    alone distinguishes them.
    """
    fleet = result["fleet"]["delta_pi"]
    rate = result["fleet"]["delta_pi_percent"]
    return {
        "mean_delta_pi": fleet["mean"],
        "mean_delta_pi_percent": rate["mean"],
        "p_delta_pi_positive": fleet["positive_rate"],
        "n_days": result["n_days"],
        "n_days_dropped": len(result["seeds_dropped"]),
    }


def write_csv(result: dict, path: str) -> None:
    """One row per agent plus a fleet row, with the headline columns first."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    base_cols = ["scenario", "ai_arm", "baseline_arm", "agent", "n_days",
                 "n_days_dropped", "mean_delta_pi", "mean_delta_pi_percent",
                 "p_delta_pi_positive"]
    stat_cols = [f"{q}_{s}" for q in ("pi_baseline", "pi_ai",
                                      "delta_pi", "delta_pi_percent")
                 for s in STAT_COLUMNS]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=base_cols + stat_cols)
        writer.writeheader()
        rows = [("ALL", result["fleet"])]
        rows += sorted(result["per_agent"].items())
        for agent, stats in rows:
            row = {
                "scenario": result["scenario"], "ai_arm": result["ai_arm"],
                "baseline_arm": result["baseline_arm"], "agent": agent,
                "n_days": result["n_days"],
                "n_days_dropped": len(result["seeds_dropped"]),
                "mean_delta_pi": stats["delta_pi"]["mean"],
                "mean_delta_pi_percent": stats["delta_pi_percent"]["mean"],
                "p_delta_pi_positive": stats["delta_pi"]["positive_rate"],
            }
            for quantity, values in stats.items():
                for stat in STAT_COLUMNS:
                    row[f"{quantity}_{stat}"] = values[stat]
            writer.writerow(row)
