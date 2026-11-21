"""Calibrate the W7 advantage-hinge delta from measured critic noise.

Rule (pre-registered in `b/G1结果与机制分析_2026-11-08.md` §3; never uses
profit to select):

1. From the v4 rankact chunk states, keep the states whose TRUE payoff is
   flat near the truthful quote: payoff range over the three lowest quote
   points below `FLAT_PAYOFF_TOL_CNY`. On those states every quote choice is
   economically equivalent, so any Q variation the critic shows there is
   pure estimation noise.
2. delta is the largest Q variation (max - min of q_min over the same three
   points) across the flat states, rounded UP to the nearest 0.05.

The hinge then refuses any "advantage" smaller than the measured noise
floor. Requires at least one flat state; refuses to guess otherwise.

Writes `reports/algorithm_diagnostics/v4_hinge_calibration.json` and prints
the selected delta.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "reports" / "algorithm_diagnostics" / "rankact_chunks_v4"
OUT = ROOT / "reports" / "algorithm_diagnostics" / "v4_hinge_calibration.json"
FLAT_PAYOFF_TOL_CNY = 0.5
N_NEAR_TRUTHFUL = 3
ROUND_UP = 0.05


def main() -> int:
    states = sorted(CHUNK_DIR.glob("state_*.json"))
    if not states:
        raise SystemExit(f"no rankact chunk states under {CHUNK_DIR}; "
                         "run the rankact sweep first")
    records = []
    for path in states:
        doc = json.loads(path.read_text(encoding="utf-8"))
        pts = sorted(doc["points"], key=lambda p: p["quote"])[:N_NEAR_TRUTHFUL]
        if len(pts) < N_NEAR_TRUTHFUL:
            continue
        payoffs = [float(p["payoff_unit"]) for p in pts]
        qs = [float(p["q_min"]) for p in pts]
        payoff_range = max(payoffs) - min(payoffs)
        q_range = max(qs) - min(qs)
        records.append({
            "state": path.stem,
            "pair_id": doc["pair_id"], "block": doc["block"],
            "unit": doc["unit"], "leg": doc["leg"],
            "quotes_near_truthful": [float(p["quote"]) for p in pts],
            "payoff_range_cny": payoff_range,
            "q_range": q_range,
            "flat_near_truthful": payoff_range < FLAT_PAYOFF_TOL_CNY,
        })
    flat = [r for r in records if r["flat_near_truthful"]]
    if not flat:
        raise SystemExit("no provably flat state found; refusing to guess "
                         "delta (calibration contract)")
    noise = max(r["q_range"] for r in flat)
    delta = math.ceil(noise / ROUND_UP) * ROUND_UP
    payload = {
        "rule": ("delta = round_up_0.05(max Q variation over states whose "
                 "payoff range over the three lowest quotes < "
                 f"{FLAT_PAYOFF_TOL_CNY} CNY); never profit-selected"),
        "flat_payoff_tol_cny": FLAT_PAYOFF_TOL_CNY,
        "n_near_truthful_points": N_NEAR_TRUTHFUL,
        "source": "reports/algorithm_diagnostics/rankact_chunks_v4",
        "states": records,
        "flat_states": [r["state"] for r in flat],
        "measured_q_noise": noise,
        "selected_delta": delta,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_delta": delta,
                      "measured_q_noise": noise,
                      "flat_states": [r["state"] for r in flat]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
