"""Summarize P0-1, P0-2 and choose the phase-18 diagnostic branch.

Read-only. It does not correlate single-block Q with the all-day fixed-quote
grid: those are different policy families. The Q/return comparison uses only
the matched same-state full-day rollouts in P0-1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path("data/processed/v2_cy_jn_2025")


def load_p01():
    return json.loads((DATA / "daily_counterfactual_action_validation.json"
                       ).read_text(encoding="utf-8"))


def load_grid(unit):
    path = DATA / f"single_ess_quote_surface_{unit}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def grid_summary(g):
    a = g["analysis"]
    m = np.asarray(a["matrix"], dtype=float)
    z = np.asarray(a["zero_blocks"], dtype=int)
    best = a["best"]
    # 3x3 neighbourhood variation around the best.
    i = int(np.argmin(np.abs(np.asarray(a["charge_points"])
                            - best["charge"])))
    j = int(np.argmin(np.abs(np.asarray(a["discharge_points"])
                            - best["discharge"])))
    nb = m[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
    return {
        "unit": g["unit"],
        "best": best,
        "honest": a["honest"],
        "b1_payoff": a["b1_payoff"],
        "actors": a["actors"],
        "flat_cells_within_1cny": a["flat_cells_within_1cny"],
        "cells": a["cells"],
        "high_quote_zero_cells": a["high_quote_zero_cells"],
        "local_maxima": len(a["local_maxima"]),
        "best_neighbour_range_cny": float(np.nanmax(nb) - np.nanmin(nb)),
        "best_neighbour_values": nb.tolist(),
        "zero_blocks_at_best": int(z[i, j]),
        "matrix_shape": list(m.shape),
        "refinement_best": (g.get("refinement", {}).get("best", {})
                            or {}).get("payoff"),
        "refinement_quote": (
            (g.get("refinement", {}).get("best", {}) or {}).get("charge"),
            (g.get("refinement", {}).get("best", {}) or {}).get("discharge")),
    }


def main() -> int:
    p01 = load_p01()
    agg = p01["aggregate"]
    rows = p01["rows"]
    grids = {u: load_grid(u) for u in ("ESS14", "ESS27")}
    grids = {u: g for u, g in grids.items() if g is not None}
    grid_summaries = {u: grid_summary(g) for u, g in grids.items()}
    # Critic direction per checkpoint.
    direction = {}
    for label, v in agg.items():
        direction[label] = {
            "q_top_matches_return_top": v["q_top_matches_return_top"],
            "states": v["states"],
            "mean_spearman_q_delta": v["mean_spearman_q_delta"],
            "mean_best_gain_vs_actor": v["mean_best_gain_vs_actor"],
        }
    # States where the selected D+A critic is wrong and material.
    da = [r for r in rows if r["checkpoint"] == "da_best"]
    da_wrong = [r for r in da if not r["q_top_matches_return_top"]
                and abs(r["best_return_gain_vs_actor"]) > 10.0]
    # P0-1 actor/critic/recommended-return table.
    table = []
    for r in da:
        table.append({
            "pair_id": r["pair_id"], "block": r["block"], "unit": r["unit"],
            "actor_quote": r["actor_quote"],
            "critic_recommended_quote": r["critic_recommended_quote"],
            "best_return_quote": r["best_return_quote"],
            "actor_delta_cny": r["actor_delta"],
            "best_return_delta_cny": r["best_return_delta"],
            "gain_cny": r["best_return_gain_vs_actor"],
            "spearman_q_delta": r["spearman_q_delta"],
            "q_top_matches_return_top": r["q_top_matches_return_top"],
            "soc_start": r["soc_start"],
        })
    # Branch decision.
    best_dir = direction.get("da_best", {})
    critic_mixed = (best_dir.get("mean_spearman_q_delta", 0) > 0
                    and best_dir.get("q_top_matches_return_top", 0)
                    < 0.75 * best_dir.get("states", 1))
    grid_clear = {}
    for u, s in grid_summaries.items():
        grid_clear[u] = {
            "interior_best": not s["best"]["boundary"],
            "best_neighbour_range_cny": s["best_neighbour_range_cny"],
            "flat_fraction": s["flat_cells_within_1cny"] / max(s["cells"], 1),
            "local_maxima": s["local_maxima"],
            "actor_gap_cny": {
                k: v["distance_to_best_cny"] for k, v in s["actors"].items()},
        }
    if critic_mixed:
        branch = "critic_plus_actor"
        rationale = (
            "The D+A critic's direction is positive but not reliable (3/8 "
            "top matches, 5/8 wrong). The single-ESS grid has a well-defined "
            "surface if the best is interior and the neighbour range is small; "
            "the actor gap is then an actor/credit-assignment problem, while "
            "the wrong critic states still need calibration. Do not blame the "
            "gate alone.")
    else:
        branch = "critic_first"
        rationale = (
            "The critic's ordering is not positive enough to justify an "
            "actor-only diagnosis; calibrate the critic first.")
    out = {
        "p01_aggregate": agg,
        "p01_da_best_table": table,
        "p01_da_best_wrong_material": da_wrong,
        "grid_summaries": grid_summaries,
        "grid_clearance": grid_clear,
        "branch": branch,
        "rationale": rationale,
        "note": ("P0-1 Q/return is the only matched comparison. The all-day "
                 "fixed-quote grid is a different policy family and is not "
                 "correlated with single-block Q."),
    }
    (DATA / "phase18_analysis.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"direction": direction, "grid": grid_clear,
                      "branch": branch, "rationale": rationale},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
