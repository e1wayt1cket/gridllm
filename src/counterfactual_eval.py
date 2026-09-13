# counterfactual_eval.py
"""Benefit of an assisted bidding policy, measured against a baseline policy.

The quantity the research is about is a difference between two runs of the same
day:

    benefit_i = profit_i(assisted) - profit_i(baseline)

Everything here exists to make that difference mean what it says. Each arm of a
pair is run with the same RNG seed over the same scenario, so load, on-site
generation, the initial state of charge, the network and the price realization
are identical and the only thing that differs is the bidding policy. The pair is
then *checked*, not assumed: if the two arms did not see the same day, the
comparison is refused rather than reported.

The baseline is a whole-day policy applied to every strategic agent. That is a
different counterfactual from the one the training reward subtracts, which is
taken inside a single window and holds the other agents at their learned bids
(see rl_env.step). A training reward is not a benefit.

`benefit_rate` divides by the magnitude of the baseline, so a baseline that
loses money — a battery whose day's spread does not cover its degradation — still
yields a rate that reads the right way round.
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

# Benefit statistics reported per agent and for the fleet.
STAT_COLUMNS = ("n", "mean", "std", "median", "p05", "p25", "p50", "p75",
                "p95", "positive_rate")


def benefit(profit_ai: float, profit_baseline: float) -> float:
    """How much more the assisted policy earned (CNY)."""
    return float(profit_ai) - float(profit_baseline)


def benefit_rate(profit_ai: float, profit_baseline: float) -> float:
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


def run_arm(scenario: str, config: MarketConfig, arm: ArmSpec,
            rl_names: Sequence[str], day_seed: int, T: int = 96,
            n_blocks: int = N_BLOCKS, settle_full_day: bool = True) -> dict:
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
    agents, wholesale = get_scenario(scenario, T=T, config=config)
    env = BiddingEnv(agents, config, rl_agent_names=list(rl_names))

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
        if env._last_result is None or env._last_result.get("fell_back"):
            clean = False
        for nm in rl_names:
            profits[nm] += float(info.get("raw_profit", {}).get(nm, 0.0))
        if done:
            break

    rolling_profits = dict(profits)
    if settle_full_day:
        profits, settled = _settle_full_day(agents, config, rl_names,
                                            env.current_actions, wholesale, T)
        clean = clean and settled

    return {"arm": arm.name, "profits": profits,
            "rolling_profits": rolling_profits,
            "settled": bool(settle_full_day), "wholesale": wholesale,
            "seed": day_seed, "clean": clean}


def _settle_full_day(agents, config: MarketConfig, rl_names, actions, wholesale,
                     T: int) -> tuple:
    """Re-clear the whole day at once with the state of charge pinned.

    A day that finishes with a fuller battery has banked energy it did not sell,
    so settling each block's cash and adding them up rewards whichever policy
    ran the battery down. Clearing the horizon in one go with the endpoints tied
    makes the two arms comparable on the day itself, with no terminal price to
    choose. The action sequence is the one the policy emitted causally, block by
    block; only the settlement is joint.

    Returns (per-agent profit, clean).
    """
    import dataclasses

    import participant_payoff
    from market import clear_market

    settle_config = dataclasses.replace(
        config,
        storage=dataclasses.replace(config.storage, terminal_soc_equal=True))
    settle_actions = {}
    for nm in rl_names:
        act = actions.get(nm)
        if act is None:
            continue
        settle_actions[nm] = {
            "bid_mult": np.asarray(act["bid_mult"], dtype=float)[:T],
            "offer_adder": np.asarray(act["offer_adder"], dtype=float)[:T],
        }

    try:
        result = clear_market(agents, T, "DA", settle_actions, settle_config,
                              wholesale=wholesale)
    except Exception:
        result = None
    if result is None or result.get("fell_back"):
        return {nm: 0.0 for nm in rl_names}, False

    profits = {}
    for nm in rl_names:
        agent = next((a for a in agents if a.name == nm), None)
        sched = result["schedules"].get(nm)
        if agent is None or sched is None:
            profits[nm] = 0.0
            continue
        profits[nm] = participant_payoff.participant_payoff(
            sched, result["lmp"][:, agent.bus], agent, settle_config).total
    return profits, True


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
        nm: {"baseline_profit": [], "ai_profit": [], "benefit": [],
             "benefit_rate": []}
        for nm in rl_names}
    fleet = {k: [] for k in ("baseline_profit", "ai_profit", "benefit",
                             "benefit_rate")}
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
            per_agent[nm]["baseline_profit"].append(b)
            per_agent[nm]["ai_profit"].append(a)
            per_agent[nm]["benefit"].append(benefit(a, b))
            per_agent[nm]["benefit_rate"].append(benefit_rate(a, b))
        fleet["baseline_profit"].append(base_total)
        fleet["ai_profit"].append(ai_total)
        fleet["benefit"].append(benefit(ai_total, base_total))
        fleet["benefit_rate"].append(benefit_rate(ai_total, base_total))

    return {
        "scenario": scenario,
        "ai_arm": ai_arm.name,
        "baseline_arm": baseline_arm.name,
        "seeds_requested": list(seeds),
        "seeds_dropped": dropped,
        "n_days": len(fleet["benefit"]),
        "fleet": {k: summarize(v) for k, v in fleet.items()},
        "per_agent": {nm: {k: summarize(v) for k, v in stats.items()}
                      for nm, stats in per_agent.items()},
    }


#: Headline columns, in the order a report should read them: how much was
#: gained, how large that is relative to the baseline, and how often it was
#: gained at all.
HEADLINE_COLUMNS = ("mean_benefit", "mean_benefit_rate", "positive_rate")


def headline(result: dict) -> dict:
    """The three numbers the research question is answered with."""
    fleet = result["fleet"]["benefit"]
    rate = result["fleet"]["benefit_rate"]
    return {
        "mean_benefit": fleet["mean"],
        "mean_benefit_rate": rate["mean"],
        "positive_rate": fleet["positive_rate"],
        "n_days": result["n_days"],
        "n_days_dropped": len(result["seeds_dropped"]),
    }


def write_csv(result: dict, path: str) -> None:
    """One row per agent plus a fleet row, with the headline columns first."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    base_cols = ["scenario", "ai_arm", "baseline_arm", "agent", "n_days",
                 "n_days_dropped", "mean_benefit", "mean_benefit_rate",
                 "positive_rate"]
    stat_cols = [f"{q}_{s}" for q in ("baseline_profit", "ai_profit",
                                      "benefit", "benefit_rate")
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
                "mean_benefit": stats["benefit"]["mean"],
                "mean_benefit_rate": stats["benefit_rate"]["mean"],
                "positive_rate": stats["benefit"]["positive_rate"],
            }
            for quantity, values in stats.items():
                for stat in STAT_COLUMNS:
                    row[f"{quantity}_{stat}"] = values[stat]
            writer.writerow(row)
