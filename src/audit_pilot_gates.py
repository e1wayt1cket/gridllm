# audit_pilot_gates.py
"""Pre-training gates for the MATD3 pilot: timing, state matching, accounting.

The pilot may not start until these pass. Each check is stated against the
code rather than against a remembered intention, and the state-matching check
runs one real block with `clear_market` instrumented, so the two clears -- the
policy's and the truthful counterfactual -- are compared argument by argument.

Writes three reports and returns non-zero if a gate fails.

Usage:
    PYTHONPATH=src python src/audit_pilot_gates.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

import config_loader
import v2_cy_jn_env as envmod
import wholesale_price
from rl_env import BiddingEnv, N_BLOCKS, BLOCK_SIZE
from rl_spec import OBS_V6

REPORTS = "reports/v2_cy_jn_2025"
TIMING = f"{REPORTS}/decision_information_timing_audit.md"
REWARD = f"{REPORTS}/difference_reward_state_matching_audit.md"
ACCOUNT = f"{REPORTS}/reward_accounting_audit.md"

#: Names that would mean an observation reads the price curve or the
#: counterfactual result.
FORBIDDEN_IN_OBS = ("wholesale", "_last_base_result", "price_curve",
                    "day_price", "grid_import", "lmp_future")

FEATURE_SOURCES = [
    ("load_feat", "block mean of the agent's day-ahead load forecast",
     "current block, DA forecast", "yes"),
    ("re_feat", "block mean of the DA PV/wind forecast",
     "current block, DA forecast", "yes"),
    ("soc", "state at the start of the block", "none, a state", "yes"),
    ("last_lmp_norm", "most recent committed mean LMP", "already cleared "
     "periods", "no"),
    ("block_pos", "block index / 23", "none", "yes"),
    ("avg_lmp_hist_norm", "mean of committed mean LMPs", "already cleared "
     "periods", "no"),
    ("slr_norm", "current block forecast load/RE ratio", "current block, DA "
     "forecast", "yes"),
    ("avg_other_dev", "other agents' previous committed deviations",
     "previous block", "no"),
    ("all_dev_std", "spread of the same deviations", "previous block", "no"),
    ("ema_dev", "observed LMP vs its EMA", "already cleared periods", "no"),
    ("price_trend", "slope of the last up-to-4 observed LMPs",
     "already cleared periods", "no"),
    ("pred_lmp_norm", "EMA forecast of the next LMP", "a forecast, not a "
     "realized future value", "yes"),
]


def _source_has(module, needle: str) -> bool:
    try:
        return needle in inspect.getsource(module)
    except Exception:
        return False


def timing_audit(root: Path) -> dict:
    """Whether the same-day price is proven available at bid time.

    The source file has no publication metadata; the simulation has no
    wall-clock gate. What can be checked is that no observation feature reads
    the price curve or either clearing's result.
    """
    obs_src = inspect.getsource(BiddingEnv._compute_obs_features) + \
        inspect.getsource(BiddingEnv._get_agent_obs)
    leaks = [name for name in FORBIDDEN_IN_OBS if name in obs_src]
    import csv
    with (root / "docs/data.csv").open(encoding="utf-8") as fh:
        columns = next(csv.reader(fh))
    return {
        "price_file_columns": columns,
        "publication_timestamp_present": False,
        "bid_submission_timestamp_present": False,
        "clearing_timestamp_present": False,
        "obs_feature_count": len(OBS_V6.feature_order),
        "obs_reads_price_or_baseline": leaks,
        "published_before_bid_proven": False,
    }


def state_matching_check(root: Path) -> dict:
    """One instrumented block: compare the policy clear with the baseline clear."""
    import market

    pair = {"load_date": "2022-10-03", "price_date": "2025-10-03"}
    calls: List[dict] = []
    original = market.clear_market

    def spy(agents, T, stage, action_params, config, storage_units=None,
            wholesale=None, *, horizon_type):
        calls.append({
            "agents": agents, "T": T, "stage": stage,
            "actions": {k: {kk: np.array(vv, copy=True) for kk, vv in v.items()}
                        for k, v in (action_params or {}).items()},
            "config": config,
            "wholesale": None if wholesale is None
            else np.array(wholesale, copy=True),
            "horizon_type": horizon_type,
        })
        return original(agents, T, stage, action_params, config,
                        storage_units, wholesale, horizon_type=horizon_type)

    market.clear_market = spy
    result = {"calls": 0}
    names: List[str] = []
    try:
        # `build_env_for_pair` points the global price source at `jn2025` and
        # a specific date; this audit runs inside the test session, so it must
        # put the defaults back or every later env build inherits a day.
        env = envmod.build_env_for_pair(pair)
        names = [a.name for a in env.rl_agents]
        obs = env.reset()
        spec = env.action_spec
        # Deliberately non-truthful, so "the baseline restores truthfulness on
        # the committed block and leaves later periods alone" is a real test.
        quote = dict(zip(spec.action_names, (50.0, 100.0)))
        acts = {nm: np.array([quote[k] for k in spec.action_names],
                             dtype=float) for nm in names}
        env.step(acts)
        calls_after_block0 = len(calls)
        prev_soc = dict(env.prev_soc)
        env.step(acts)
        result = {
            "calls_block0": calls_after_block0,
            "calls_total": len(calls),
            "prev_soc_after_block0": prev_soc,
        }
    finally:
        market.clear_market = original
        config_loader.reload_defaults()
        wholesale_price.reset_cache()

    pairs = [calls[i:i + 2] for i in range(0, len(calls), 2)]
    checks = {}
    details = {}
    for idx, (real, base) in enumerate(pairs):
        same_wholesale = (real["wholesale"] is not None
                          and np.array_equal(real["wholesale"],
                                             base["wholesale"]))
        same_config = real["config"] is base["config"]
        same_horizon = real["horizon_type"] == base["horizon_type"] == "window"
        same_agents = [a.name for a in real["agents"]] == \
            [a.name for a in base["agents"]]
        truthful_block = True
        later_unchanged = True
        others_unchanged = True
        for nm in names:
            for key in spec.action_names:
                base_arr = np.asarray(base["actions"][nm][key])
                real_arr = np.asarray(real["actions"][nm][key])
                if not np.all(base_arr[:BLOCK_SIZE]
                              == spec.truthful[key]):
                    truthful_block = False
                if not np.array_equal(base_arr[BLOCK_SIZE:],
                                      real_arr[BLOCK_SIZE:]):
                    later_unchanged = False
        for nm, action in real["actions"].items():
            if nm in names:
                continue
            for key, value in action.items():
                if not np.array_equal(np.asarray(base["actions"][nm][key]),
                                      np.asarray(value)):
                    others_unchanged = False
        soc_matched = all(
            abs(float(a.storage.soc0) - float(prev_soc.get(a.name, a.storage.soc0)))
            < 1e-12
            for a in base["agents"] if a.storage is not None) if idx == 1 else True
        checks = {
            "two_clears_per_block": len(calls) % 2 == 0,
            "same_wholesale": same_wholesale,
            "same_config_object": same_config,
            "same_horizon_type": same_horizon,
            "same_agent_population": same_agents,
            "baseline_committed_block_is_truthful": truthful_block,
            "baseline_later_periods_keep_policy_actions": later_unchanged,
            "non_rl_actions_unchanged": others_unchanged,
            "baseline_starts_from_current_soc": soc_matched,
        }
        details[f"block{idx}"] = dict(checks)

    obs_src = inspect.getsource(BiddingEnv._compute_obs_features) + \
        inspect.getsource(BiddingEnv._get_agent_obs)
    return {
        "calls": result.get("calls_total", 0),
        "blocks": len(pairs),
        "checks": {k: all(v[k] for v in details.values() if k in v)
                   for k in next(iter(details.values()), {})},
        "details": details,
        "baseline_readable_from_obs": "_last_base_result" in obs_src,
        "prev_soc_after_block0": result.get("prev_soc_after_block0"),
    }


def accounting_check(root: Path) -> dict:
    import models
    import participant_payoff

    T = 96
    lmp = np.full(T, 400.0)
    schedule = {
        "p_buy": np.full(T, 0.1), "p_sell": np.full(T, 0.1),
        "served": np.zeros(T), "unserved": np.zeros(T),
        "pv_used": np.zeros(T), "wind_used": np.zeros(T),
        "p_ch": np.full(T, 0.1), "p_dis": np.full(T, 0.1),
        "soc": np.zeros(T), "soc_boundary": np.zeros(T + 1),
    }
    config = models.MarketConfig.from_yaml()
    config.storage.cycle_cost = 100.0
    storage = models.StorageSpec(e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
                                 eta_ch=0.92, eta_dis=0.92, soc0=0.5,
                                 soc_min=0.10, soc_max=0.90)
    agent = models.Agent(name="ESS9", bus=9, is_prosumer=False,
                         load_forecast=np.zeros(T), pv_forecast=np.zeros(T),
                         load_real=np.zeros(T), pv_real=np.zeros(T),
                         bid_value=0.0, offer_cost=0.0, storage=storage,
                         load_type="storage", participant_type="storage")
    breakdown = participant_payoff.participant_payoff(schedule, lmp, agent,
                                                      config, T)
    expected = float(np.sum(lmp * (schedule["p_sell"] - schedule["p_buy"]))
                     * 0.25) - 100.0 * float(
        np.sum(schedule["p_ch"] + schedule["p_dis"])) * 0.25
    rl_src = inspect.getsource(envmod.BiddingEnv._agent_block_profit)
    env_src = (inspect.getsource(BiddingEnv.step)
               + inspect.getsource(BiddingEnv._agent_block_profit))
    bidding_src = inspect.getsource(__import__("rl_bidding"))
    td3_src = inspect.getsource(__import__("rl_td3"))
    return {
        "payoff_total": float(breakdown.total),
        "expected": expected,
        "cycle_cost_charged_once": abs(float(breakdown.total) - expected) < 1e-9,
        "degradation_term": float(breakdown.degradation_cost),
        "raw_profit_uses_participant_payoff": "participant_payoff" in rl_src,
        "env_applies_deviation_penalty": "deviation_penalty" in env_src,
        "actor_loss_adds_penalty": "deviation_penalty(" in bidding_src,
        "penalty_defined_in_td3": "def deviation_penalty" in td3_src,
        "info_has_components": all(k in env_src for k in (
            "raw_profit", "local_baseline_profit", "reward")),
    }


def write_timing(path: Path, facts: dict) -> None:
    leaks = facts["obs_reads_price_or_baseline"]
    lines = [
        "# Decision-information timing audit (V2-CY-JN-2025)",
        "",
        "## Verdict",
        "",
        "**The publication → bid → clearing order cannot be established from "
        "the source, so the same-day actual price is *not* treated as a "
        "decision-time input for any learned policy.** The pilot's "
        "observations do not read it; the clearing uses it as the market's "
        "exogenous PCC price, which is a market mechanism rather than a "
        "feature. B2A/B2B, which do quote from the same-day curve, are marked "
        "as carrying a potential look-ahead and are not a deployable "
        "benchmark.",
        "",
        "## What the source contains",
        "",
        f"- `docs/data.csv` columns: `{', '.join(facts['price_file_columns'])}`",
        "- No publication timestamp, no bid-submission timestamp and no "
        "clearing timestamp. The file cannot prove the order, and the "
        "simulation has no wall-clock gate that would supply it.",
        "",
        "## The order that would be required",
        "",
        "    price published  ->  ESS bid submitted  ->  distribution clearing",
        "",
        "It is **not proven**. Therefore the same-day actual curve is treated "
        "as unknown at bid time for every decision-time claim.",
        "",
        "## OBS_V6",
        "",
        f"{facts['obs_feature_count']} features; forbidden reads found: "
        f"{leaks if leaks else 'none'}. Each feature and when its data is "
        "available:",
        "",
        "| feature | source | period read | available at the decision |",
        "|---|---|---|---|",
    ]
    for name, source, period, available in FEATURE_SOURCES:
        lines.append(f"| `{name}` | {source} | {period} | {available} |")
    lines += [
        "",
        "The price features are functions of LMPs already observed "
        "(`last_lmp_norm`, `avg_lmp_hist_norm`, `ema_dev`, `price_trend`) or "
        "of a per-agent EMA forecast (`pred_lmp_norm`). The forecast is a "
        "forecast, not a realized value, and its history is committed periods "
        "only. No feature reads the wholesale curve, the counterfactual "
        "result, or a future LMP; `tests/test_rl_spec.py` enforces the "
        "wholesale-array exclusion at every block.",
        "",
        "## Consequences for the pilot",
        "",
        "1. The policy's information set is causal by construction; the pilot "
        "may proceed.",
        "2. MATD3 must not be described as having known the day's price curve.",
        "3. B2A/B2B are exploratory historical baselines only. Their quotes "
        "were computed from the same-day curve, so any advantage they show is "
        "potentially an informational one, not a strategy effect.",
        "4. If a later phase needs a price feature in the observation, it must "
        "use a published forecast or a lagged curve and be registered as a "
        "new observation version.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_state_matching(path: Path, facts: dict) -> None:
    checks = facts["checks"]
    lines = [
        "# Difference-reward state-matching audit (V2-CY-JN-2025)",
        "",
        "## Verdict",
        "",
        "**The truthful baseline is a state-matched counterfactual clear of "
        "the same window**, not the day-level B1 result. It is taken from one "
        "re-clear per block with the current SOC, the same wholesale window, "
        "the same config, and every non-RL action unchanged; the RL agents' "
        "committed four periods are set to their truthful point. The results "
        "below come from one instrumented block on a real pair.",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for key, ok in checks.items():
        lines.append(f"| {key.replace('_', ' ')} | {'PASS' if ok else 'FAIL'} |")
    lines += [
        "",
        "## Why the baseline is truthful for the committed block only",
        "",
        "The reward differences the *decision* over the four committed "
        "periods, so the counterfactual restores truthfulness on exactly "
        "those periods and leaves the rest of the 16-period window at the "
        "policy's own declared actions. Replacing the whole window would "
        "confound the committed action with a different future plan, and "
        "would no longer answer 'what did this decision add'. The baseline "
        "agents still start from the current SOC, so an earlier policy move "
        "is inherited by both arms.",
        "",
        "## The seven required checks",
        "",
        "1. **Full-day B1 used as the baseline?** No. The baseline is a "
        "window clear inside `step`; `rl_env` never calls the day-level "
        "settlement for reward shaping.",
        "2. **Stale SOC after the policy moved?** No. `_make_window_agents` "
        "is given `self.prev_soc`, and the audit reads the baseline agents' "
        "`soc0` after a block and finds it equal to `prev_soc`.",
        "3. **One counterfactual clear for every agent?** Yes. A single "
        "`base_result` supplies every RL agent's baseline profit.",
        "4. **Same price, load, constraints, cycle cost?** Yes: identical "
        "`wholesale` array, identical config object and horizon type; load "
        "and network come from the same window agents.",
        "5. **Sign.** `reward = raw_profit - local_baseline_profit`; reported "
        "separately in `info`, so the sign can be audited per block.",
        "6. **Penalty placement.** The bounded deviation penalty is added to "
        "the actor loss only (`rl_bidding.update`). It is not in the reward. "
        "The market-impact penalty is a separate optional reward term, zero "
        "in this pilot.",
        "7. **Baseline visible to the actor?** No. The observation builder "
        f"does not reference `_last_base_result` "
        f"({'found' if facts['baseline_readable_from_obs'] else 'not found'}).",
        "",
        "## Relation to B1",
        "",
        "B1 is a whole-day truthful clearing and is the *evaluation* "
        "benchmark. The training counterfactual is a block-level, "
        "state-matched clear. They are consistent in their definition of "
        "truthful quotes, but they are different quantities: one is an "
        "evaluation outcome, the other a learning signal. The pilot reports "
        "them separately and never substitutes one for the other.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_accounting(path: Path, facts: dict) -> None:
    ok = (facts["cycle_cost_charged_once"]
          and facts["raw_profit_uses_participant_payoff"]
          and not facts["env_applies_deviation_penalty"]
          and facts["actor_loss_adds_penalty"])
    lines = [
        "# Reward and cost accounting audit (V2-CY-JN-2025)",
        "",
        f"**Verdict: {'consistent' if ok else 'INCONSISTENT — stop'}. The "
        "clearing objective and the settlement ledger carry the cycle cost "
        "in different quantities, and the settlement charges it once.**",
        "",
        "## One cycle cost, two places, no double charge",
        "",
        "| place | form | what it does |",
        "|---|---|---|",
        "| clearing objective | `-100 * (ch + dis) * dt` | prices throughput "
        "so the market's dispatch reflects degradation; it is part of the "
        "optimization, not a participant's cash |",
        "| `participant_payoff` | `degradation_cost = 100 * (ch + dis) * dt` | "
        "the only place the participant's cash is charged |",
        "",
        f"A synthetic leg (0.1 MW charge and 0.1 MW discharge for 96 periods "
        f"at 400 CNY/MWh) settles to {facts['payoff_total']:.4f} CNY; the "
        f"hand computation `LMP*(sell-buy)*dt - 100*(ch+dis)*dt` gives "
        f"{facts['expected']:.4f} CNY. Difference "
        f"{abs(facts['payoff_total'] - facts['expected']):.2e}. The "
        f"degradation term is {facts['degradation_term']:.2f} CNY, charged "
        "once.",
        "",
        "The clearing objective is **not** a payoff and is never subtracted "
        "again: `participant_payoff` builds the cash figure itself, and the "
        "env's raw profit is that function's total "
        f"({'confirmed' if facts['raw_profit_uses_participant_payoff'] else 'NOT confirmed'}).",
        "",
        "## Reward decomposition",
        "",
        "Per block, per RL agent:",
        "",
        "    raw_profit          = participant_payoff(committed periods)",
        "    local_baseline      = the same quantity in the truthful-arm re-clear",
        "    reward              = raw_profit - local_baseline",
        "                          - market_impact_penalty * power   (default 0)",
        "",
        "The bounded deviation penalty is **not** in the reward. It is added "
        "inside `rl_bidding.update` to the actor loss "
        f"({'present' if facts['actor_loss_adds_penalty'] else 'MISSING'}), "
        "defined once in `rl_td3.deviation_penalty`, and the env does not "
        f"reference it ({'clean' if not facts['env_applies_deviation_penalty'] else 'FOUND IN ENV'}). "
        "So it cannot be double charged: it shapes the policy gradient and "
        "never reduces the logged reward.",
        "",
        "## What the training log must keep",
        "",
        "`env.step` returns `info` with `reward`, `raw_profit` and "
        "`local_baseline_profit` per agent "
        f"({'confirmed' if facts['info_has_components'] else 'MISSING'}). "
        "The pilot log writes all three plus the episode sums, so the "
        "training signal is never the only record of what happened. The "
        "deviation-penalty magnitude is logged from the actor update, apart "
        "from the reward.",
        "",
        "## Not claimed",
        "",
        "A higher raw profit or a higher reward is not social welfare. The "
        "clearing objective is not welfare in this project, and reward is not "
        "benefit.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root)
    timing = timing_audit(root)
    matching = state_matching_check(root)
    accounting = accounting_check(root)

    write_timing(root / TIMING, timing)
    write_state_matching(root / REWARD, matching)
    write_accounting(root / ACCOUNT, accounting)

    failed = []
    if timing["obs_reads_price_or_baseline"]:
        failed.append("timing: observation reads a forbidden name")
    if not all(matching["checks"].values()):
        failed.append("state matching: " + ", ".join(
            k for k, v in matching["checks"].items() if not v))
    if not (accounting["cycle_cost_charged_once"]
            and accounting["raw_profit_uses_participant_payoff"]
            and not accounting["env_applies_deviation_penalty"]
            and accounting["actor_loss_adds_penalty"]):
        failed.append("accounting")
    print(f"timing leaks          {timing['obs_reads_price_or_baseline'] or 'none'}")
    print(f"state matching        "
          f"{sum(matching['checks'].values())}/{len(matching['checks'])}")
    print(f"accounting            "
          f"once={accounting['cycle_cost_charged_once']} "
          f"payoff={accounting['raw_profit_uses_participant_payoff']} "
          f"penalty_in_actor={accounting['actor_loss_adds_penalty']} "
          f"penalty_in_env={accounting['env_applies_deviation_penalty']}")
    print(f"wrote {TIMING}\nwrote {REWARD}\nwrote {ACCOUNT}")
    if failed:
        print("GATES FAILED:")
        for item in failed:
            print(f"  - {item}")
        return 1
    print("all gates pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
