"""Read-only recheck of the critic's return scale and ranking.

The phase-15 rollouts already compare Q with a discounted training return and
with the actual day settlement. This script makes the definitions explicit:

* Q is in reward units. A reward is the B/C2 block reward divided by
  `fixed_scale` (1000), so Q * 1000 is the CNY-equivalent the critic predicts.
* The same-state, same-action rollout return is
  `gamma ** (23 - block) * full_delta + discounted C2`.
* The undiscounted remaining settlement and the B1 counterfactual are also
  reported, but they are different objects and are never substituted for the
  training return.

No training, no clearing, no checkpoint write.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

DATA = Path("data/processed/v2_cy_jn_2025")
WINTER = ("2022-12-13|2025-12-11", "2022-12-30|2025-12-31",
          "2022-12-31|2025-12-28")
CONTROL = ("2022-07-23|2025-07-13", "2022-08-14|2025-08-17")


def read_rows():
    rows = []
    for path in sorted(glob.glob(str(DATA / "critic_long_horizon_cp*_*.csv"))):
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for key in ("checkpoint", "block"):
                    row[key] = int(row[key])
                for key in ("quote", "actor_quote", "q1", "q2", "q_min",
                            "twin_gap", "future_payoff", "full_delta_cny",
                            "b1_future_payoff", "future_minus_b1",
                            "training_return_scaled", "training_return_cny",
                            "discounted_settlement_cny", "c2_shaping_sum",
                            "p_ch_mwh_at_block", "p_dis_mwh_at_block"):
                    row[key] = float(row[key])
                row["group"] = (f"{row['state_id']}|{row['target']}|"
                                f"{row['leg']}")
                row["day_group"] = ("winter" if row["pair_id"] in WINTER
                                    else "control")
                row["block_band"] = ("early" if row["block"] < 8 else
                                     "mid" if row["block"] < 16 else "late")
                row["q_cny"] = row["q_min"] * 1000.0
                row["err_training_cny"] = (row["q_cny"]
                                           - row["training_return_cny"])
                row["err_settlement_cny"] = (row["q_cny"]
                                             - row["future_payoff"])
                row["err_b1_cny"] = (row["q_cny"]
                                     - row["future_minus_b1"])
                rows.append(row)
    return rows


def safe_corr(a, b, kind):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float((pearsonr if kind == "pearson" else spearmanr)(a, b)[0])


def corr_block(rows, x, y):
    return {
        "n": len(rows),
        "pearson": safe_corr([r[x] for r in rows], [r[y] for r in rows],
                             "pearson"),
        "spearman": safe_corr([r[x] for r in rows], [r[y] for r in rows],
                              "spearman"),
    }


def group_stats(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    out = []
    for key, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["quote"])
        spread = max(r["full_delta_cny"] for r in rs) - \
            min(r["full_delta_cny"] for r in rs)
        qrange = max(r["q_min"] for r in rs) - min(r["q_min"] for r in rs)
        # Pairwise rank agreement between Q and the actual day ΔG.
        pairs = inv = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if rs[i]["full_delta_cny"] == rs[j]["full_delta_cny"]:
                    continue
                pairs += 1
                if (rs[i]["q_min"] - rs[j]["q_min"]) * \
                        (rs[i]["full_delta_cny"]
                         - rs[j]["full_delta_cny"]) < 0:
                    inv += 1
        # Do the twin critics rank the quote the same way?
        twin_inv = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if (rs[i]["q1"] - rs[j]["q1"]) * \
                        (rs[i]["q2"] - rs[j]["q2"]) < 0:
                    twin_inv += 1
        out.append({
            "group": key,
            "checkpoint": rs[0]["checkpoint"],
            "pair_id": rs[0]["pair_id"],
            "block": rs[0]["block"],
            "target": rs[0]["target"],
            "leg": rs[0]["leg"],
            "n": len(rs),
            "actor_quote": rs[0]["actor_quote"],
            "best_quote": max(rs,
                              key=lambda r: r["full_delta_cny"])["quote"],
            "worst_quote": min(rs,
                               key=lambda r: r["full_delta_cny"])["quote"],
            "delta_spread_cny": spread,
            "training_spread_cny": max(
                r["training_return_cny"] for r in rs) - min(
                    r["training_return_cny"] for r in rs),
            "q_range": qrange,
            "spearman_q_delta": safe_corr(
                [r["q_min"] for r in rs],
                [r["full_delta_cny"] for r in rs], "spearman"),
            "spearman_q_training": safe_corr(
                [r["q_min"] for r in rs],
                [r["training_return_scaled"] for r in rs], "spearman"),
            "spearman_q_settlement": safe_corr(
                [r["q_min"] for r in rs],
                [r["future_payoff"] for r in rs], "spearman"),
            "spearman_q_b1": safe_corr(
                [r["q_min"] for r in rs],
                [r["future_minus_b1"] for r in rs], "spearman"),
            "rank_pairs": pairs,
            "rank_inversions": inv,
            "rank_inversion_fraction": inv / pairs if pairs else 0.0,
            "twin_rank_inversions": twin_inv,
            "twin_rank_inversion_fraction": (
                twin_inv / len(rs) / (len(rs) - 1) * 2 if len(rs) > 1 else 0.0),
            "material": spread > 10.0,
            "flat": spread <= 1.0,
        })
    return out


def err_stats(rows, key):
    vals = np.asarray([r[key] for r in rows], dtype=float)
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "mae": float(np.mean(np.abs(vals))),
            "rmse": float(np.sqrt(np.mean(vals ** 2))),
            "p05": float(np.quantile(vals, 0.05)),
            "p95": float(np.quantile(vals, 0.95))}


def main(argv=None) -> int:
    rows = read_rows()
    groups = group_stats(rows)
    material_groups = {g["group"] for g in groups if g["material"]}
    flat_groups = {g["group"] for g in groups if g["flat"]}
    material = [r for r in rows if r["group"] in material_groups]
    flat = [r for r in rows if r["group"] in flat_groups]
    summary = {
        "rows": len(rows),
        "groups": len(groups),
        "material_groups": sum(g["material"] for g in groups),
        "flat_groups": sum(g["flat"] for g in groups),
        "scale": {
            "reward_units": "B/C2 reward divided by fixed_scale",
            "fixed_scale": 1000.0,
            "q_cny": "q_min * 1000",
            "training_return_same_units": (
                "gamma**(23-block) * full_delta + discounted C2, divided by "
                "1000"),
            "settlement": "future_payoff = sum of remaining raw block "
                          "profits, undiscounted, no B1 and no C2",
            "b1_counterfactual": "future_minus_b1 = remaining policy payoff "
                                 "minus remaining B1 payoff",
        },
        "overall": {
            "q_vs_training_scaled": corr_block(
                rows, "q_min", "training_return_scaled"),
            "q_cny_vs_training_cny": corr_block(
                rows, "q_cny", "training_return_cny"),
            "q_cny_vs_settlement": corr_block(
                rows, "q_cny", "future_payoff"),
            "q_cny_vs_b1_counterfactual": corr_block(
                rows, "q_cny", "future_minus_b1"),
            "q_cny_vs_full_day_delta": corr_block(
                rows, "q_cny", "full_delta_cny"),
            "q_cny_vs_discounted_settlement": corr_block(
                rows, "q_cny", "discounted_settlement_cny"),
        },
        "material": {
            "q_vs_training_scaled": corr_block(
                material, "q_min", "training_return_scaled"),
            "q_cny_vs_settlement": corr_block(
                material, "q_cny", "future_payoff"),
            "q_cny_vs_b1_counterfactual": corr_block(
                material, "q_cny", "future_minus_b1"),
            "q_cny_vs_full_day_delta": corr_block(
                material, "q_cny", "full_delta_cny"),
        },
        "flat": {
            "q_vs_training_scaled": corr_block(
                flat, "q_min", "training_return_scaled"),
            "q_cny_vs_full_day_delta": corr_block(
                flat, "q_cny", "full_delta_cny"),
        },
        "errors_cny": {
            "all_training": err_stats(rows, "err_training_cny"),
            "all_settlement": err_stats(rows, "err_settlement_cny"),
            "all_b1": err_stats(rows, "err_b1_cny"),
            "material_training": err_stats(material, "err_training_cny"),
            "material_settlement": err_stats(material, "err_settlement_cny"),
        },
        "twin": {
            "mean_gap": float(np.mean([r["twin_gap"] for r in rows])),
            "median_gap": float(np.median([r["twin_gap"] for r in rows])),
            "p95_gap": float(np.quantile([r["twin_gap"] for r in rows], 0.95)),
            "groups_with_rank_disagreement": int(sum(
                g["twin_rank_inversions"] > 0 for g in groups)),
            "material_groups_with_rank_disagreement": int(sum(
                g["twin_rank_inversions"] > 0
                for g in groups if g["material"])),
        },
        "by_slice": {},
        "material_groups": [g for g in groups if g["material"]],
        "group_spearman_q_delta": {
            "mean": float(np.nanmean(
                [g["spearman_q_delta"] for g in groups])),
            "median": float(np.nanmedian(
                [g["spearman_q_delta"] for g in groups])),
            "positive": int(sum(
                (g["spearman_q_delta"] or 0) > 0 for g in groups)),
            "negative": int(sum(
                (g["spearman_q_delta"] or 0) < 0 for g in groups)),
            "groups": len(groups),
        },
    }
    for name, key in (("checkpoint", "checkpoint"), ("day_group", "day_group"),
                      ("block_band", "block_band"), ("target", "target"),
                      ("leg", "leg")):
        buckets = {}
        for row in rows:
            buckets.setdefault(str(row[key]), []).append(row)
        summary["by_slice"][name] = {
            k: {
                "n": len(v),
                "q_vs_training_scaled": corr_block(
                    v, "q_min", "training_return_scaled"),
                "q_cny_vs_full_day_delta": corr_block(
                    v, "q_cny", "full_delta_cny"),
                "mean_err_training_cny": float(np.mean(
                    [r["err_training_cny"] for r in v])),
            } for k, v in sorted(buckets.items())}

    out_csv = DATA / "critic_return_calibration_rows.csv"
    fieldnames = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (DATA / "critic_return_calibration_groups.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    (DATA / "critic_return_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "rows", "groups", "material_groups", "flat_groups", "overall",
        "material", "twin", "group_spearman_q_delta")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
