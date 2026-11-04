"""Per-15-minute winter quote acceptance audit.

For the worst block of each winter day and unit, this script changes only the
target unit's discharge quote, holds every other ESS at the checkpoint's
deterministic action, and runs the real rolling-window clearing. It records
per period: nodal LMP, quote, cycle cost, the price-taking condition, accepted
discharge, SOC, the unit's settlement and the B1 settlement at the same
period. It does not use a block-average price or a merit-order approximation.

The B1 arm is a full truthful day under the same terminal-neutral mechanism;
its per-period series is checked against the stored B1 block payoffs.

Read-only: no training, no checkpoint write, no H2, no legacy test.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import rl_b_c2_gate as mod
import train_b_c2_gate as tb
from diagnose_critic_long_horizon import (DATA, RUN, WINTER, collect_state,
                                          load_trainer, pairs)

CKPTS = (2, 10)
UNITS = ("ESS14", "ESS27")
DT = 0.25
QUOTES = (0.0, 100.0, 150.0, 258.0, 300.0)


def read_worst_blocks():
    rows = list(csv.DictReader(
        (DATA / "winter_block_reconstruction.csv").open(encoding="utf-8")))
    groups = defaultdict(list)
    for row in rows:
        key = (int(row["checkpoint"]), row["pair_id"], row["ess"])
        groups[key].append(row)
    out = []
    for (cp, pid, ess), rs in groups.items():
        worst = min(rs, key=lambda r: float(r["diff_vs_b1"]))
        out.append({
            "checkpoint": cp, "pair_id": pid, "unit": ess,
            "block": int(worst["block"]),
            "recorded_quote": float(worst["quote_discharge"]),
            "recorded_soc": float(worst["soc_start"]),
            "recorded_diff": float(worst["diff_vs_b1"]),
        })
    return sorted(out, key=lambda r: (r["checkpoint"], r["pair_id"],
                                      r["unit"]))


def b1_day(pair, pair_id, b1_blocks, verify_blocks):
    """Truthful full day; return per-period (unit -> list of 96 records)."""
    env = tb.envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    env.reset()
    names = [a.name for a in env.rl_agents]
    truth = {nm: np.zeros(2) for nm in names}
    series = {nm: [] for nm in names}
    block_sums = {nm: [] for nm in names}
    for block in range(24):
        _, _, _, info = env.step(truth)
        result = env._last_result
        n_commit = min(4, env.T - block * 4)
        for nm in names:
            agent = next(a for a in env.all_agents if a.name == nm)
            sched = result["schedules"][nm]
            lmp = np.asarray(result["lmp"][:n_commit, agent.bus], dtype=float)
            p_dis = np.asarray(sched["p_dis"][:n_commit], dtype=float)
            p_ch = np.asarray(sched["p_ch"][:n_commit], dtype=float)
            p_sell = np.asarray(sched["p_sell"][:n_commit], dtype=float)
            p_buy = np.asarray(sched["p_buy"][:n_commit], dtype=float)
            soc = np.asarray(sched["soc"][:n_commit], dtype=float)
            cycle = float(env.config.storage.cycle_cost)
            payoff = DT * (lmp * (p_sell - p_buy) - cycle * (p_dis + p_ch))
            for t in range(n_commit):
                series[nm].append({
                    "period": block * 4 + t, "lmp": float(lmp[t]),
                    "p_dis": float(p_dis[t]), "p_ch": float(p_ch[t]),
                    "soc": float(soc[t]), "payoff": float(payoff[t]),
                })
            block_sums[nm].append(float(np.sum(payoff)))
    worst = 0.0
    for nm in names:
        stored = verify_blocks.get(nm, [])
        diff = max((abs(a - b) for a, b in zip(block_sums[nm], stored)),
                   default=0.0)
        print(f"B1 verify {pair_id} {nm}: max block diff {diff:.3e}",
              flush=True)
        worst = max(worst, diff)
    series["verify_diff"] = worst
    return series


def policy_block(trainer, pair, block, unit, quote, baseline, env):
    actions = {nm: np.asarray(baseline[nm], dtype=float).copy() for nm in
               baseline}
    actions[unit] = actions[unit].copy()
    actions[unit][1] = float(quote)
    env2 = copy.deepcopy(env)
    _, _, _, info = env2.step(actions)
    result = env2._last_result
    agent = next(a for a in env2.all_agents if a.name == unit)
    sched = result["schedules"][unit]
    n_commit = min(4, env2.T - block * 4)
    lmp = np.asarray(result["lmp"][:n_commit, agent.bus], dtype=float)
    wholesale = np.asarray(env2._last_wholesale[:n_commit], dtype=float)
    p_dis = np.asarray(sched["p_dis"][:n_commit], dtype=float)
    p_ch = np.asarray(sched["p_ch"][:n_commit], dtype=float)
    p_sell = np.asarray(sched["p_sell"][:n_commit], dtype=float)
    p_buy = np.asarray(sched["p_buy"][:n_commit], dtype=float)
    soc = np.asarray(sched["soc"][:n_commit], dtype=float)
    soc0 = float(env.prev_soc.get(unit, soc[0]))
    cycle = float(env2.config.storage.cycle_cost)
    payoff = DT * (lmp * (p_sell - p_buy) - cycle * (p_dis + p_ch))
    max_dis = np.array([
        agent.storage.max_discharge_feasible(
            soc0 if t == 0 else float(soc[t - 1]))
        for t in range(n_commit)])
    return {
        "env": env2, "result": result, "info": info, "lmp": lmp,
        "wholesale": wholesale, "p_dis": p_dis, "p_ch": p_ch,
        "p_sell": p_sell, "p_buy": p_buy, "soc": soc, "payoff": payoff,
        "max_dis": max_dis, "cycle": cycle,
        "block_profit": float(info.get("raw_profit", {}).get(unit, 0.0)),
    }


def main(argv=None) -> int:
    pair_map, _ = pairs()
    run_manifest = json.loads((RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    experiment = mod.ExperimentConfig(**run_manifest["experiment"])
    b1_cache = mod.load_b1_cache()
    b1_blocks_json = json.loads(Path(
        DATA / "b1_terminal_neutral_block_payoffs.json").read_text(
            encoding="utf-8"))
    mech = json.loads(Path(mod.B1_MANIFEST).read_text(
        encoding="utf-8"))["mechanism_hash"]
    worth = read_worst_blocks()
    days = sorted({r["pair_id"] for r in worth})
    # B1 per-period series, one truthful day per pair.
    b1_series = {}
    b1_verify = {}
    for pid in days:
        row = next(r for r in csv.DictReader(
            (DATA / "load_price_pairing.csv").open(encoding="utf-8"))
                   if f"{r['load_date']}|{r['price_date']}" == pid)
        b1_series[pid] = b1_day(pair_map[pid], pid,
                                b1_blocks_json.get(pid, {}),
                                b1_blocks_json.get(pid, {}))
        b1_verify[pid] = b1_series[pid]["verify_diff"]

    trainers = {}
    template = tb.envmod.build_env_for_pair(
        pair_map[WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    for cp in CKPTS:
        trainers[cp] = load_trainer(cp, template, experiment)

    rows = []
    summary = {"verification": {}, "states": {}}
    for rec in worth:
        cp, pid, unit, block = rec["checkpoint"], rec["pair_id"], rec["unit"], \
            rec["block"]
        trainer = trainers[cp]
        env, obs, baseline, past, soc, gates, probs, plog, price = \
            collect_state(trainer, pair_map[pid], block)
        actor_quote = float(baseline[unit][1])
        quotes = list(QUOTES) + [actor_quote]
        # Preserve order and drop duplicates.
        seen, candidates = set(), []
        for q in quotes:
            key = round(q, 6)
            if key not in seen:
                seen.add(key)
                candidates.append(q)
        state_key = f"cp{cp}|{pid}|b{block}|{unit}"
        state_out = {
            "checkpoint": cp, "pair_id": pid, "block": block, "unit": unit,
            "actor_quote": actor_quote,
            "actor_actions": {nm: np.asarray(baseline[nm]).tolist()
                              for nm in baseline},
            "gate_probability": gates, "gate_deterministic": probs, "soc": soc,
            "recorded_worst_block": rec,
            "scenarios": {},
        }
        b1 = b1_series[pid]
        for q in candidates:
            out = policy_block(trainer, pair_map[pid], block, unit, q,
                               baseline, env)
            block_rows = []
            for t in range(len(out["lmp"])):
                period = block * 4 + t
                b1_rec = b1[unit][period]
                row = {
                    "checkpoint": cp, "pair_id": pid, "block": block,
                    "unit": unit, "quote": q, "actor_quote": actor_quote,
                    "period": period, "period_in_block": t,
                    "lmp_cny_mwh": float(out["lmp"][t]),
                    "wholesale_cny_mwh": float(out["wholesale"][t]),
                    "cycle_cost_cny_mwh": out["cycle"],
                    "effective_offer_cny_mwh": q + out["cycle"],
                    "price_taking_condition": bool(
                        q + out["cycle"] <= out["lmp"][t] + 1e-9),
                    "max_discharge_mw": float(out["max_dis"][t]),
                    "accepted_discharge_mw": float(out["p_dis"][t]),
                    "accepted_discharge_mwh": float(out["p_dis"][t]) * DT,
                    "charge_mw": float(out["p_ch"][t]),
                    "soc_end": float(out["soc"][t]),
                    "payoff_cny": float(out["payoff"][t]),
                    "b1_discharge_mw": b1_rec["p_dis"],
                    "b1_payoff_cny": b1_rec["payoff"],
                    "diff_vs_b1_cny": float(out["payoff"][t]
                                            - b1_rec["payoff"]),
                    "accepted": bool(out["p_dis"][t] * DT > 1e-6),
                }
                rows.append(row)
                block_rows.append(row)
            state_out["scenarios"][str(q)] = {
                "block_payoff_cny": out["block_profit"],
                "block_period_sum_cny": float(np.sum(out["payoff"])),
                "block_discharge_mwh": float(np.sum(out["p_dis"]) * DT),
                "b1_block_discharge_mwh": float(np.sum(
                    [b1[unit][block * 4 + t]["p_dis"] for t in
                     range(len(out["lmp"]))]) * DT),
                "b1_block_payoff_cny": float(np.sum(
                    [b1[unit][block * 4 + t]["payoff"] for t in
                     range(len(out["lmp"]))])),
                "accepted_periods": int(sum(r["accepted"]
                                            for r in block_rows)),
                "accepted_mwh": float(sum(
                    r["accepted_discharge_mwh"] for r in block_rows)),
                "mean_lmp": float(np.mean(out["lmp"])),
                "min_lmp": float(np.min(out["lmp"])),
                "max_lmp": float(np.max(out["lmp"])),
            }
            state_out["scenarios"][str(q)]["diff_vs_b1_cny"] = (
                state_out["scenarios"][str(q)]["block_period_sum_cny"]
                - state_out["scenarios"][str(q)]["b1_block_payoff_cny"])
            print(f"{state_key} q={q:.1f}: block payoff "
                  f"{out['block_profit']:.2f}, B1 "
                  f"{state_out['scenarios'][str(q)]['b1_block_payoff_cny']:.2f}",
                  flush=True)
        summary["states"][state_key] = state_out
    summary["verification"] = {
        "b1_max_block_diff": {k: v for k, v in b1_verify.items()},
        "note": "B1 per-period payoffs are a full truthful day under the "
                "terminal-neutral rolling mechanism; block sums are compared "
                "with b1_terminal_neutral_block_payoffs.json.",
    }
    out_csv = DATA / "winter_15min_quote_acceptance.csv"
    fieldnames = list(rows[0])
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (DATA / "winter_15min_quote_acceptance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
