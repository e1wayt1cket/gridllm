# b1_daily_baseline.py
"""B1 under the same 24-block execution interface as the policy arm.

JN-Base is a **one-shot 96-period** clearing with terminal SOC pinned. The
policy arm is a sequence of 16-period rolling windows that commit four periods,
and it does not pin terminal SOC. Those are different mechanisms, so JN-Base is
not automatically the day-level B1 of the policy arm. This script:

1. runs a truthful arm through the **same rolling interface**, all four RL
   units at (0,0), for a representative set of days;
2. compares it with JN-Base day by day;
3. if the mechanisms differ, registers the rolling B1 as the day benchmark for
   all 96 days, leaving JN-Base untouched.

Outputs: `b1_daily_results.csv`, `b1_daily_manifest.json`, and a comparison
artifact. It does not train and does not modify any existing result.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import config_loader
import screen_v2_cy_jn_2025 as screen
import v2_cy_jn_env as envmod
import wholesale_price

PAIRING = screen.PAIRING
BUS_LOADS = screen.BUS_LOADS
PRICE_CURVES = screen.PRICE_CURVES
JN_RESULTS = "reports/v2_cy_jn_2025/results.csv"
OUT = "data/processed/v2_cy_jn_2025/pilot"
RESULTS = f"{OUT}/b1_daily_results.csv"
MANIFEST = f"{OUT}/b1_daily_manifest.json"
COMPARISON = f"{OUT}/b1_jnbase_comparison.csv"
REPRESENTATIVE = 6
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")
COLUMNS = ("pair_id", "load_date", "price_date", "group", "soc0",
           "G_ESS9", "G_ESS14", "G_ESS17", "G_ESS27", "G_total",
           "soc_end_ESS9", "soc_end_ESS14", "soc_end_ESS17",
           "soc_end_ESS27", "blocks", "clears", "fallbacks",
           "lmp_fallbacks", "all_optimal", "runtime_s")


def _read(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hash(config) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True,
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_truthful_day(row: dict,
                     terminal_soc_target: Optional[float] = None) -> dict:
    """One day through the rolling interface, all RL units truthful."""
    pair = {"load_date": row["load_date"], "price_date": row["price_date"]}
    env = envmod.build_env_for_pair(
        pair, differential_reward=False,
        terminal_soc_target=terminal_soc_target)
    truth = {a.name: np.array([0.0, 0.0]) for a in env.rl_agents}
    obs = env.reset()
    total = {nm: 0.0 for nm in UNITS}
    block_payoffs: Dict[str, List[float]] = {nm: [] for nm in UNITS}
    charge_mwh = {nm: 0.0 for nm in UNITS}
    discharge_mwh = {nm: 0.0 for nm in UNITS}
    loss_mwh = 0.0
    voltage_violation_cells = 0
    line_overload_cells = 0
    blocks = clears = fallbacks = lmp_fb = 0
    optimal = True
    failed_blocks: List[int] = []
    soc_at_failure: Dict[str, Dict[str, float]] = {}
    started = time.perf_counter()
    for block in range(24):
        soc_before = {nm: float(env.prev_soc.get(nm, 0.5))
                      for nm in UNITS}
        next_obs, rewards, done, info = env.step(truth)
        result = env._last_result
        blocks += 1
        if result is None:
            fallbacks += 1
            optimal = False
            failed_blocks.append(block)
            soc_at_failure[str(block)] = soc_before
        else:
            clears += 1
            lmp_fb += int(result.get("lmp_fallbacks", 0))
            solver = result.get("solver", {}).get("final", {})
            optimal = optimal and bool(solver.get("optimal"))
            n_commit = min(4, env.T - block * 4)
            for nm in UNITS:
                ws = (result.get("schedules") or {}).get(nm)
                if ws is not None:
                    charge_mwh[nm] += float(
                        np.sum(np.asarray(ws["p_ch"][:n_commit])) * 0.25)
                    discharge_mwh[nm] += float(
                        np.sum(np.asarray(ws["p_dis"][:n_commit])) * 0.25)
            # The result covers the whole look-ahead window; only the
            # committed periods are part of the executed day.
            loss = result.get("line_loss_mw")
            if loss is not None:
                loss_mwh += float(
                    np.sum(np.asarray(loss)[:n_commit]) * 0.25)
            voltage = result.get("bus_voltage")
            if voltage is not None:
                v = np.asarray(voltage, dtype=float)[:n_commit]
                voltage_violation_cells += int(
                    ((v < env.config.network.v_min_pu - 1e-7)
                     | (v > env.config.network.v_max_pu + 1e-7)).sum())
            util = result.get("line_utilization")
            if util is not None:
                line_overload_cells += int(
                    (np.asarray(util, dtype=float)[:n_commit]
                     > 1.0 + 1e-7).sum())
        for nm in UNITS:
            payoff = float(info.get("raw_profit", {}).get(nm, 0.0))
            total[nm] += payoff
            block_payoffs[nm].append(payoff)
        if done:
            break
        obs = next_obs
    e_max = {nm: float(next(a for a in env.all_agents
                           if a.name == nm).storage.e_max)
             for nm in UNITS}
    return {
        "pair_id": f"{row['load_date']}|{row['price_date']}",
        "load_date": row["load_date"], "price_date": row["price_date"],
        "group": row["group"], "soc0": 0.5,
        **{f"G_{nm}": total[nm] for nm in UNITS},
        "G_total": sum(total.values()),
        **{f"soc_end_{nm}": float(env.prev_soc.get(nm, 0.5))
           for nm in UNITS},
        **{f"charge_mwh_{nm}": charge_mwh[nm] for nm in UNITS},
        **{f"discharge_mwh_{nm}": discharge_mwh[nm] for nm in UNITS},
        **{f"cycles_{nm}": (charge_mwh[nm] + discharge_mwh[nm])
           / (2.0 * e_max[nm]) if e_max[nm] > 0 else 0.0
           for nm in UNITS},
        "losses_mwh": loss_mwh,
        "voltage_violation_cells": voltage_violation_cells,
        "line_overload_cells": line_overload_cells,
        "blocks": blocks, "clears": clears, "fallbacks": fallbacks,
        "lmp_fallbacks": lmp_fb, "all_optimal": optimal,
        "terminal_soc_target": terminal_soc_target,
        "block_payoffs": block_payoffs,
        "failed_blocks": failed_blocks,
        "soc_at_failure": soc_at_failure,
        "runtime_s": round(time.perf_counter() - started, 2),
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--representative-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd()
    rows = _read(root / PAIRING)
    jn = {r["load_date"]: r for r in _read(root / JN_RESULTS)
          if r.get("arm") == "JN-Base"}

    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    representative = []
    for group, items in groups.items():
        representative.extend(items[:REPRESENTATIVE // len(groups) + 1])
    representative = representative[:REPRESENTATIVE]

    comparison = []
    try:
        for row in representative:
            out = run_truthful_day(row)
            ref = jn.get(row["load_date"], {})
            comparison.append({
                "pair_id": out["pair_id"],
                "group": out["group"],
                "rolling_G_total": out["G_total"],
                "jnbase_G_total": float(ref.get("ess_total_payoff_cny", "nan")),
                "diff_total": out["G_total"] - float(
                    ref.get("ess_total_payoff_cny", "nan")),
                **{f"rolling_{u}": out[f"G_{u}"] for u in UNITS},
                **{f"jnbase_{u}": float(ref.get(f"{u}_payoff_cny", "nan"))
                   for u in UNITS},
                **{f"rolling_soc_{u}": out[f"soc_end_{u}"] for u in UNITS},
            })
            print(f"representative {out['pair_id']}: rolling "
                  f"{out['G_total']:.2f} vs JN-Base "
                  f"{float(ref.get('ess_total_payoff_cny', 'nan')):.2f}",
                  flush=True)
        comp_path = root / COMPARISON
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        with comp_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)

        if args.representative_only:
            print(f"wrote {comp_path}")
            return 0

        work = rows[:args.limit] if args.limit else rows
        results = []
        for idx, row in enumerate(work, 1):
            results.append(run_truthful_day(row))
            if idx % 10 == 0:
                print(f"{idx}/{len(work)} days", flush=True)
    finally:
        config_loader.reload_defaults()
        wholesale_price.reset_cache()

    path = _write_csv(root / RESULTS, results)
    config = screen.build_config(100.0)
    manifest = {
        "dataset": "b1_daily_rolling",
        "version": "b1-rolling-24block-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mechanism": "24 blocks, each a 16-period rolling window committing 4; "
                     "all four RL units truthful (0,0); no terminal SOC pin; "
                     "block-level participant_payoff summed",
        "not_jn_base": True,
        "soc0": 0.5,
        "config_hash": config_hash(config),
        "inputs": {
            "pairing": {"path": PAIRING, "sha256": _sha(root / PAIRING)},
            "bus_loads": {"path": BUS_LOADS, "sha256": _sha(root / BUS_LOADS)},
            "price_curves": {"path": PRICE_CURVES,
                             "sha256": _sha(root / PRICE_CURVES)},
        },
        "outputs": {
            "results": {"path": RESULTS, "sha256": _sha(root / RESULTS)},
            "comparison": {"path": COMPARISON,
                           "sha256": _sha(root / COMPARISON)},
        },
        "days": len(results),
        "representative_comparison": comparison,
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2),
                                 encoding="utf-8")
    print(f"wrote {path}\nwrote {root / MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
