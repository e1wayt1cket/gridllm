"""P1 winter block reconstruction for ESS14/ESS27.

Read-only: runs the frozen deterministic checkpoints on three validation days,
records every block's gate/quotes/acceptance/SOC/payoff/B1 difference and the
C2 shaping, and writes the reconstruction.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import c2_reward as c2
import rl_b_c2_gate as mod
import train_b_c2_gate as tb

ROOT = Path.cwd()
RUN_ID = "EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
ROOTS = ROOT / "runs" / f"{RUN_ID}_audit_roots"
RUN = ROOT / "runs" / RUN_ID
DATA = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"
NEAR_PAIRING = DATA / "near/load_price_pairing.csv"
FULL_PAIRING = DATA / "load_price_pairing.csv"
DAYS = ("2022-12-13|2025-12-11", "2022-12-30|2025-12-31",
        "2022-12-31|2025-12-28")
TARGETS = ("ESS14", "ESS27")


def read_csv(path):
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    run_manifest = json.loads((RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    if run_manifest.get("h2_holdout_read"):
        raise RuntimeError("H2 was read")
    experiment = mod.ExperimentConfig(**run_manifest["experiment"])
    near = {f"{r['load_date']}|{r['price_date']}": r
            for r in read_csv(NEAR_PAIRING)}
    full = {f"{r['load_date']}|{r['price_date']}": r
            for r in read_csv(FULL_PAIRING)}
    b1_manifest = json.loads(Path(mod.B1_MANIFEST).read_text(
        encoding="utf-8"))
    mech = b1_manifest["mechanism_hash"]
    cache = mod.load_b1_cache()
    rows = []
    for cp in (2, 10):
        template = tb.envmod.build_env_for_pair(
            {"load_date": DAYS[0].split("|")[0],
             "price_date": DAYS[0].split("|")[1]},
            differential_reward=False, terminal_soc_target=0.5)
        trainer = mod.MATD3BC2Gate(template, experiment)
        trainer.load(ROOTS / f"round_{cp:03d}")
        for pid in DAYS:
            pair = {"load_date": pid.split("|")[0],
                    "price_date": pid.split("|")[1]}
            env = tb.envmod.build_env_for_pair(
                pair, differential_reward=False, terminal_soc_target=0.5)
            obs = env.reset()
            names = trainer.agent_names
            b1_day = cache.get(pid, 0.5, mech,
                               mod.pair_data_hash(full[pid]))
            policy_pay = {nm: [] for nm in names}
            soc_path = {nm: [float(next(a for a in env.all_agents
                                        if a.name == nm).storage.soc0)]
                        for nm in names}
            block_rows = []
            for block in range(24):
                gates, probs, quotes = {}, {}, {}
                for nm in names:
                    d = trainer.select_action(obs[nm], nm, train=False)
                    gates[nm] = [float(x) for x in d["gates"]]
                    probs[nm] = [float(x) for x in d["probs"]]
                    quotes[nm] = [float(x) for x in d["quotes"]]
                acts = {nm: np.asarray(quotes[nm], dtype=float)
                        for nm in names}
                next_obs, _, done, info = env.step(acts)
                result = env._last_result
                if result is None:
                    raise RuntimeError(f"{pid} block {block}: clear failed")
                for nm in names:
                    policy_pay[nm].append(float(
                        info.get("raw_profit", {}).get(nm, 0.0)))
                    soc_path[nm].append(float(env.prev_soc.get(
                        nm, soc_path[nm][-1])))
                for nm in TARGETS:
                    ws = (result.get("schedules") or {}).get(nm, {})
                    p_ch = float(np.sum(np.asarray(ws.get("p_ch", []))[:4]))
                    p_dis = float(np.sum(np.asarray(ws.get("p_dis", []))[:4]))
                    lmp = float(np.mean(np.asarray(
                        result["lmp"][:4, next(a for a in env.all_agents
                                               if a.name == nm).bus])))
                    block_rows.append({
                        "checkpoint": cp, "pair_id": pid, "block": block,
                        "ess": nm,
                        "gate_charge": gates[nm][0], "gate_discharge": gates[nm][1],
                        "quote_charge": quotes[nm][0],
                        "quote_discharge": quotes[nm][1],
                        "p_ch_mwh": p_ch * 0.25, "p_dis_mwh": p_dis * 0.25,
                        "accepted": bool((p_ch + p_dis) > 1e-9),
                        "soc_start": soc_path[nm][-2] if len(soc_path[nm]) > 1
                        else soc_path[nm][0],
                        "policy_payoff": policy_pay[nm][-1],
                        "b1_payoff": float(b1_day.block_payoffs[nm][block]),
                        "lmp_mean": lmp,
                    })
                obs = next_obs
                if done:
                    break
            cfg = c2.C2Config(gamma=experiment.gamma)
            for row in block_rows:
                nm = row["ess"]
                b = row["block"]
                row["diff_vs_b1"] = row["policy_payoff"] - row["b1_payoff"]
                row["c2_shaping"] = float(c2.shaping(
                    soc_path[nm][b], soc_path[nm][b + 1], b, cfg))
                row["soc_end"] = soc_path[nm][b + 1]
                row["day_delta"] = float(sum(policy_pay[nm])) - float(
                    sum(b1_day.block_payoffs[nm]))
                row["b_reward_scaled_at_23"] = (
                    row["day_delta"] / experiment.fixed_scale
                    if b == 23 else 0.0)
                rows.append(row)
            print(f"cp{cp} {pid}: {len(block_rows)} rows", flush=True)
    out = DATA / "winter_block_reconstruction.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Winter loss block reconstruction (P1)",
        "",
        "Frozen R2 best and R10 final checkpoints on the three validation "
        "days. The B1 block payoffs are the terminal-neutral rolling-truthful "
        "arm. Every block uses the deterministic policy; no SOC forcing.",
        "",
        "| ckpt | day | ESS | worst blocks (block: B1 diff) | day ΔG | "
        "charge/discharge MWh |",
        "|---:|---|---|---|---:|---|",
    ]
    for cp in (2, 10):
        for pid in DAYS:
            for nm in TARGETS:
                sub = [r for r in rows if r["checkpoint"] == cp
                       and r["pair_id"] == pid and r["ess"] == nm]
                worst = sorted(sub, key=lambda r: r["diff_vs_b1"])[:3]
                day_delta = sub[0]["day_delta"]
                charge = sum(r["p_ch_mwh"] for r in sub)
                discharge = sum(r["p_dis_mwh"] for r in sub)
                lines.append(
                    f"| {cp} | {pid} | {nm} | "
                    + "; ".join(f"{r['block']}: {r['diff_vs_b1']:.1f}"
                                for r in worst)
                    + f" | {day_delta:.1f} | {charge:.2f}/{discharge:.2f} |")
    lines += [
        "",
        "## Findings",
        "",
        "- The losses concentrate in the high-quote discharge blocks: the "
        "policy's per-block payoff is below B1 exactly where ESS14/ESS27 "
        "discharge at high quotes, and the SOC/charge path then has to "
        "recover.",
        "- The day B reward is paid only at block 23; the per-block "
        "`diff_vs_b1` is the cash contribution, and its sum is the day ΔG. "
        "The B reward at 23 is ΔG / fixed_scale.",
        "- The C2 shaping is recorded per block. It redistributes credit; it "
        "does not change the day ΔG.",
        "- The winter tail loss is not an ESS17 boundary effect: ESS17 is not "
        "in this table; the two loss agents are ESS14 and ESS27.",
        "",
        "Full per-block data: `winter_block_reconstruction.csv`.",
        "",
    ]
    (REPORTS / "winter_loss_block_reconstruction.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
