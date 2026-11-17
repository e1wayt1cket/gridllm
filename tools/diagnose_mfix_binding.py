"""MFIX review probe: do network constraints actually bind under B1?

NON_FORMAL read-only diagnostic (2026-11-11, reviewer question #2). Clears
two Train days (one summer, one winter) with the full fleet at truthful
quotes through the standard rolling env path and reports, per committed
period: LMP spread across the 33 buses, max line utilization, voltage
extrema, and near-binding counts. No training, no writes outside
reports/algorithm_diagnostics/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import train_da_gate as T
import v2_cy_jn_env as envmod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics" / "mfix_network_binding.json"
PAIRS = ["2022-07-23|2025-07-13", "2022-12-09|2025-12-09"]


def main() -> int:
    t0 = time.time()
    near_rows = T.read_csv(T.NEAR_PAIRING)
    full_rows = T.read_csv(T.FULL_PAIRING)
    nbp = {f"{r['load_date']}|{r['price_date']}": r for r in near_rows}
    fbp = {f"{r['load_date']}|{r['price_date']}": r for r in full_rows}
    report = {"label": "NON_FORMAL_DIAGNOSTIC_PILOT", "days": {}}
    for pid in PAIRS:
        pair = {"load_date": nbp[pid]["load_date"],
                "price_date": fbp[pid]["price_date"],
                "season": nbp[pid].get("season", ""), "pair_id": pid}
        env = envmod.build_env_for_pair(
            pair, differential_reward=False, terminal_soc_target=0.5)
        obs = env.reset()
        names = [a.name for a in env.rl_agents]
        v_min, v_max = env.config.network.v_min_pu, env.config.network.v_max_pu
        lmp_spread, line_max, volt_min, volt_max = [], [], [], []
        line_bind95 = volt_bind = periods = 0
        for block in range(24):
            acts = {nm: np.zeros(2) for nm in names}  # truthful fleet
            obs, _, done, info = env.step(acts)
            res = env._last_result
            if res is None:
                raise RuntimeError(f"{pid} block {block}: clearing failed")
            n = min(4, env.T - block * 4)
            periods += n
            lmp = np.asarray(res["lmp"])[:n]                     # (n, 33)
            lu = np.asarray(res.get("line_utilization", []))[:n]  # (n, lines)
            bv = np.asarray(res.get("bus_voltage", []))[:n]       # (n, 33)
            lmp_spread.append(float(np.max(lmp) - np.min(lmp))
                              if lmp.size else 0.0)
            if lu.size:
                line_max.append(float(np.max(lu)))
                line_bind95 += int(np.sum(lu > 0.95))
            if bv.size:
                volt_min.append(float(np.min(bv)))
                volt_max.append(float(np.max(bv)))
                volt_bind += int(np.sum(
                    (bv < v_min + 5e-3) | (bv > v_max - 5e-3)))
            if done:
                break
        report["days"][pid] = {
            "periods": periods,
            "lmp_spread_cny_mwh": {"mean": float(np.mean(lmp_spread)),
                                   "max": float(np.max(lmp_spread))},
            "line_utilization_max": float(np.max(line_max))
            if line_max else None,
            "line_cells_over_95pct": line_bind95,
            "voltage_min": float(np.min(volt_min)) if volt_min else None,
            "voltage_max": float(np.max(volt_max)) if volt_max else None,
            "voltage_band": [v_min, v_max],
            "voltage_cells_within_5e-3_of_band": volt_bind,
        }
        print(pid, json.dumps(report["days"][pid]), flush=True)
    report["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
