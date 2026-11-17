"""P20-B: critic gradient versus an offline piecewise-linear value oracle.

Read-only. No training, no checkpoint write, no replay addition, no H2 and no
legacy test.

Inputs are the Phase 18/19 matched counterfactuals:

* ``daily_counterfactual_action_validation.csv``: the five named validation
  states, discharge only, five to six unique quotes per state and checkpoint.
* ``phase19_critic_training_calibration_rows.csv``: three NEAR training days,
  both legs, seven to eight unique quotes per state and checkpoint.

The oracle is a piecewise-linear interpolation of the matched full-day payoff
over the observed quote grid; outside the grid it is constant. It is an
offline proxy, not the environment gradient. The virtual update moves only the
target leg's price logit, with the gate held at its observed value, through the
same v3 map the actor executes.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import da_quote_map as qmap

DATA = Path("data/processed/v2_cy_jn_2025")
TRAIN_ROWS = DATA / "phase19_critic_training_calibration_rows.csv"
VALID_ROWS = DATA / "daily_counterfactual_action_validation.csv"
OUT_JSON = DATA / "phase20_oracle_surrogate.json"
OUT_CSV = DATA / "phase20_oracle_surrogate.csv"
HIGHS = {"charge": 150.0, "discharge": 300.0}
LRS = (1e-4, 1e-3, 1e-2)
MIN_SPREAD_CNY = 1.0
MIN_SPREAD_SCALED = MIN_SPREAD_CNY / 1000.0


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def group_rows(rows: list[dict], checkpoints: set[str]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row.get("checkpoint") not in checkpoints:
            continue
        key = (row["checkpoint"], row["pair_id"], int(row["block"]),
               row["unit"], row["leg"])
        groups[key].append(row)
    out = {}
    for key, values in groups.items():
        values.sort(key=lambda r: float(r["quote"]))
        out[key] = values
    return out


def curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray]:
    quotes = np.asarray([float(r["quote"]) for r in rows], dtype=float)
    returns = np.asarray([float(r["training_return_scaled"]) for r in rows],
                         dtype=float)
    qvals = np.asarray([float(r["q_min"]) for r in rows], dtype=float)
    full = np.asarray([float(r.get("full_delta_cny") or 0.0)
                       for r in rows], dtype=float)
    keep = np.concatenate([[True], np.diff(quotes) > 1e-9])
    return quotes[keep], returns[keep], qvals[keep], full[keep]


def slope_at(quotes: np.ndarray, values: np.ndarray, point: float) -> float:
    if len(quotes) < 2:
        return 0.0
    if point <= quotes[0] or point >= quotes[-1]:
        return 0.0
    knot = np.flatnonzero(np.abs(quotes - point) < 1e-9)
    if knot.size:
        k = int(knot[0])
        slopes = []
        if k > 0:
            slopes.append((values[k] - values[k - 1])
                          / (quotes[k] - quotes[k - 1]))
        if k + 1 < len(quotes):
            slopes.append((values[k + 1] - values[k])
                          / (quotes[k + 1] - quotes[k]))
        return float(np.mean(slopes)) if slopes else 0.0
    i = int(np.searchsorted(quotes, point, side="right") - 1)
    j = min(i + 1, len(quotes) - 1)
    if j == i:
        return 0.0
    return float((values[j] - values[i]) / (quotes[j] - quotes[i]))


def value_at(quotes: np.ndarray, values: np.ndarray, point: float) -> float:
    if point <= quotes[0]:
        return float(values[0])
    if point >= quotes[-1]:
        return float(values[-1])
    i = int(np.searchsorted(quotes, point, side="right") - 1)
    j = min(i + 1, len(quotes) - 1)
    if j == i:
        return float(values[i])
    w = (point - quotes[i]) / (quotes[j] - quotes[i])
    return float(values[i] + w * (values[j] - values[i]))


def inverse_logit(quote: float, high: float) -> float:
    fraction = float(np.clip(quote / high, 1e-9, qmap.Q_HI))
    if fraction >= qmap.Q_HI:
        fraction = qmap.Q_HI - 1e-9
    if fraction <= 0.5 - qmap.SLOPE * qmap.Z_CORE:
        fraction = 0.5 - qmap.SLOPE * qmap.Z_CORE + 1e-9
    if fraction < 0.5 + qmap.SLOPE * qmap.Z_CORE:
        return float((fraction - 0.5) / qmap.SLOPE)
    ratio = math.sqrt(1e-2 / max(1e-12, 1.0 - fraction))
    return float(qmap.Z_CORE + (ratio - 1.0) / qmap.TAIL_B)


def quote_derivative(z: float, high: float) -> float:
    return float(high * qmap.map_derivative_np(
        np.asarray([z], dtype=float))[0])


def actor_row(rows: list[dict]) -> dict:
    flagged = [r for r in rows if r.get("actor_is_quote") == "True"]
    if flagged:
        return flagged[0]
    actor_quote = float(rows[0]["actor_quote"])
    return min(rows, key=lambda r: abs(float(r["quote"]) - actor_quote))


def ranking(quotes: np.ndarray, returns: np.ndarray,
            qvals: np.ndarray) -> dict:
    if len(quotes) < 2 or np.std(returns) < 1e-12 or np.std(qvals) < 1e-12:
        return {"spearman": None, "q_top": None, "return_top": None,
                "top_match": None}
    def ranks(a):
        return np.argsort(np.argsort(a))
    rho = float(np.corrcoef(ranks(qvals), ranks(returns))[0, 1])
    q_top = float(quotes[int(np.argmax(qvals))])
    r_top = float(quotes[int(np.argmax(returns))])
    return {"spearman": rho, "q_top": q_top, "return_top": r_top,
            "top_match": abs(q_top - r_top) < 1e-6}


def analyse_group(key, rows, source: str) -> dict | None:
    checkpoint, pair_id, block, unit, leg = key
    high = HIGHS[leg]
    quotes, returns, qvals, full_delta = curve(rows)
    spread = float(np.max(returns) - np.min(returns)) if len(returns) else 0.0
    row = actor_row(rows)
    actor_quote = float(row["actor_quote"])
    if actor_quote < 0.0 or actor_quote > high:
        actor_quote = float(np.clip(actor_quote, 0.0, high))
    q_slope = slope_at(quotes, qvals, actor_quote)
    oracle_slope = slope_at(quotes, returns, actor_quote)
    z0 = inverse_logit(actor_quote, high)
    dqdz = quote_derivative(z0, high)
    q_dz = q_slope * dqdz
    r_dz = oracle_slope * dqdz
    best_quote = float(quotes[int(np.argmax(returns))]) if len(returns) else 0.0
    best_return = float(np.max(returns)) if len(returns) else 0.0
    full_slope = slope_at(quotes, full_delta, actor_quote)
    full_best_quote = (float(quotes[int(np.argmax(full_delta))])
                       if len(full_delta) else 0.0)
    full_best = float(np.max(full_delta)) if len(full_delta) else 0.0
    full_actor = value_at(quotes, full_delta, actor_quote)
    full_spread = (float(np.max(full_delta) - np.min(full_delta))
                   if len(full_delta) else 0.0)
    full_direction = bool(
        np.sign(q_slope) == np.sign(full_slope)
        and abs(q_slope) > 0 and abs(full_slope) > 0)
    out = {
        "source": source,
        "checkpoint": checkpoint,
        "pair_id": pair_id,
        "block": block,
        "unit": unit,
        "leg": leg,
        "informative": bool(spread >= MIN_SPREAD_SCALED),
        "return_spread_scaled": spread,
        "return_spread_cny_equiv": 1000.0 * spread,
        "grid_points": int(len(quotes)),
        "actor_quote": actor_quote,
        "actor_return_scaled": value_at(quotes, returns, actor_quote),
        "best_quote": best_quote,
        "best_return_scaled": best_return,
        "critic_slope_scaled_per_quote": q_slope,
        "oracle_slope_scaled_per_quote": oracle_slope,
        "critic_slope_cny_equiv_per_quote": 1000.0 * q_slope,
        "oracle_slope_cny_equiv_per_quote": 1000.0 * oracle_slope,
        "quote_derivative_cny_per_logit": dqdz,
        "critic_gradient_scaled_per_logit": q_dz,
        "oracle_gradient_scaled_per_logit": r_dz,
        "direction_match": bool(
            np.sign(q_slope) == np.sign(oracle_slope)
            and abs(q_slope) > 0 and abs(oracle_slope) > 0),
        **ranking(quotes, returns, qvals),
        "full_delta_spread_cny": full_spread,
        "full_delta_actor_cny": full_actor,
        "full_delta_best_cny": full_best,
        "full_delta_best_quote": full_best_quote,
        "full_delta_slope_cny_per_quote": full_slope,
        "full_delta_direction_match": full_direction,
        "q_vs_full_delta_spearman": ranking(
            quotes, full_delta, qvals)["spearman"],
        "updates": [],
    }
    for lr in LRS:
        z_c = z0 + lr * q_dz
        z_o = z0 + lr * r_dz
        q_c = float(np.clip(high * qmap.map_fraction_np(
            np.asarray([z_c]))[0], 0.0, high))
        q_o = float(np.clip(high * qmap.map_fraction_np(
            np.asarray([z_o]))[0], 0.0, high))
        out["updates"].append({
            "logit_step": lr,
            "critic_quote": q_c,
            "oracle_quote": q_o,
            "critic_quote_delta": q_c - actor_quote,
            "oracle_quote_delta": q_o - actor_quote,
            "critic_surrogate_return": value_at(quotes, returns, q_c),
            "oracle_surrogate_return": value_at(quotes, returns, q_o),
            "critic_regret_to_best": best_return
            - value_at(quotes, returns, q_c),
            "oracle_regret_to_best": best_return
            - value_at(quotes, returns, q_o),
        })
    return out


def summary(rows: list[dict]) -> dict:
    out = {}
    for checkpoint in sorted({r["checkpoint"] for r in rows}):
        subset = [r for r in rows if r["checkpoint"] == checkpoint]
        material = [r for r in subset if r["informative"]]
        full_material = [r for r in subset
                         if r["full_delta_spread_cny"] >= MIN_SPREAD_CNY]
        out[checkpoint] = {
            "groups": len(subset),
            "informative_groups": len(material),
            "direction_match": sum(r["direction_match"] for r in material),
            "top_match": sum(bool(r["top_match"]) for r in material
                             if r["top_match"] is not None),
            "median_spearman": float(np.nanmedian(
                [r["spearman"] for r in material
                 if r["spearman"] is not None])) if material else None,
            "median_best_minus_actor_cny": float(1000.0 * np.median(
                [r["best_return_scaled"] - r["actor_return_scaled"]
                 for r in material])) if material else None,
            "full_delta_informative_groups": len(full_material),
            "full_delta_direction_match": sum(
                r["full_delta_direction_match"] for r in full_material),
            "median_q_vs_full_delta_spearman": float(np.nanmedian(
                [r["q_vs_full_delta_spearman"] for r in full_material
                 if r["q_vs_full_delta_spearman"] is not None]))
            if full_material else None,
            "median_full_delta_best_minus_actor_cny": float(np.median(
                [r["full_delta_best_cny"] - r["full_delta_actor_cny"]
                 for r in full_material])) if full_material else None,
        }
    return out


def main() -> int:
    training = read_rows(TRAIN_ROWS)
    validation = read_rows(VALID_ROWS)
    groups = {
        "training": group_rows(training, {"da_best", "da_final"}),
        "validation": group_rows(validation, {"da_best", "da_final"}),
    }
    rows = []
    for source, grouped in groups.items():
        for key, values in sorted(grouped.items()):
            result = analyse_group(key, values, source)
            if result is not None:
                rows.append(result)
    payload = {
        "oracle": ("piecewise-linear matched-rollout proxy over the observed "
                   "quote grid; constant outside the grid"),
        "boundary": ("offline diagnosis only; no training, no checkpoint "
                     "write, no H2 and no legacy test"),
        "training_grid": "both legs, seven to eight unique quotes",
        "validation_grid": "discharge only, five to six unique quotes",
        "min_informative_spread_cny": MIN_SPREAD_CNY,
        "logit_steps": list(LRS),
        "summary": summary(rows),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        fields = ["source", "checkpoint", "pair_id", "block", "unit", "leg",
                  "informative", "return_spread_scaled",
                  "return_spread_cny_equiv", "grid_points", "actor_quote",
                  "actor_return_scaled", "best_quote", "best_return_scaled",
                  "critic_slope_scaled_per_quote",
                  "oracle_slope_scaled_per_quote",
                  "critic_slope_cny_equiv_per_quote",
                  "oracle_slope_cny_equiv_per_quote",
                  "quote_derivative_cny_per_logit",
                  "critic_gradient_scaled_per_logit",
                  "oracle_gradient_scaled_per_logit", "direction_match",
                  "spearman", "q_top", "return_top", "top_match",
                  "full_delta_spread_cny", "full_delta_actor_cny",
                  "full_delta_best_cny", "full_delta_best_quote",
                  "full_delta_slope_cny_per_quote",
                  "full_delta_direction_match",
                  "q_vs_full_delta_spearman"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {OUT_JSON} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
