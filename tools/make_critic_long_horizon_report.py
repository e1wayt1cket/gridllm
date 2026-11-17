"""Generate the Phase-15 long-horizon calibration report."""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
DATA = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"


def sp(a, b):
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    files = sorted(glob.glob(str(DATA / "critic_long_horizon_*.csv")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    states = {}
    for f in glob.glob(str(DATA / "critic_long_horizon_states_*.json")):
        states.update(json.loads(Path(f).read_text(encoding="utf-8")))
    rows = []
    for (sid, target, leg), g in d.groupby(["state_id", "target", "leg"]):
        q = g["q_min"].values
        delta = g["full_delta_cny"].values
        tr = g["training_return_cny"].values
        rows.append({
            "state_id": sid, "checkpoint": int(g["checkpoint"].iloc[0]),
            "pair": g["pair_id"].iloc[0], "block": int(g["block"].iloc[0]),
            "target": target, "leg": leg, "n": len(g),
            "q_range": float(q.max() - q.min()),
            "delta_range": float(delta.max() - delta.min()),
            "spearman_q_delta": sp(q, delta),
            "spearman_q_train": sp(q, tr),
            "q_scale_mean": float(q.mean() * 1000),
            "train_return_mean": float(tr.mean()),
            "actor_quote": float(g["actor_quote"].iloc[0]),
            "best_quote": float(g.loc[g["full_delta_cny"].idxmax(), "quote"]),
            "worst_quote": float(g.loc[g["full_delta_cny"].idxmin(), "quote"]),
            "best_delta": float(delta.max()),
            "worst_delta": float(delta.min()),
        })
    s = pd.DataFrame(rows)
    material = s[s["delta_range"] > 10].copy()

    lines = [
        "# Critic long-horizon calibration (P0)",
        "",
        f"{len(d)} counterfactual rollouts, {len(s)} state×ESS×leg groups, "
        f"{len(states)} fixed states. Dates: the three winter validation days "
        "and two normal control days. Each rollout changes one ESS/one quote "
        "leg at a fixed block, then continues the frozen deterministic policy "
        "to the day end; the future SOC is not forced. This is an actual "
        "clearing rollout, not a single-block payoff.",
        "",
        "The critic Q is in the training reward units (B+C2 divided by "
        "fixed_scale = 1000). The rollout reports both the discounted training "
        "return in the same units and the actual settlement ΔG in CNY. The "
        "ranking comparison uses the actual full-day ΔG.",
        "",
        "## Aggregate ranking",
        "",
        "| slice | mean Spearman(Q, ΔG) | median | positive groups | groups |",
        "|---|---:|---:|---:|---:|",
    ]

    def add_slice(name, subset):
        vals = subset["spearman_q_delta"].dropna()
        lines.append(f"| {name} | {vals.mean():.3f} | {vals.median():.3f} | "
                     f"{int((vals > 0).sum())}/{len(vals)} | {len(vals)} |")

    add_slice("all", s)
    for cp in (2, 10):
        add_slice(f"checkpoint R{cp}", s[s["checkpoint"] == cp])
    for target, leg in (("ESS14", "discharge"), ("ESS27", "discharge"),
                        ("ESS14", "charge"), ("ESS27", "charge"),
                        ("ESS9", "charge"), ("ESS17", "discharge")):
        add_slice(f"{target}.{leg}",
                  s[(s["target"] == target) & (s["leg"] == leg)])
    lines += [
        "",
        f"- The overall mean is essentially zero; R10 is slightly negative "
        f"({s[s['checkpoint']==10]['spearman_q_delta'].mean():.3f}); R2 is "
        f"slightly positive ({s[s['checkpoint']==2]['spearman_q_delta'].mean():.3f}).",
        f"- **{int((s['delta_range'] < 1).sum())} of {len(s)}** groups have a "
        "less-than-1-CNY spread across the quote grid: for these the market "
        "value is genuinely flat, so a flat Q is not by itself evidence of a "
        "critic defect.",
        f"- **{len(material)} groups** have a material (>10 CNY) spread. Their "
        "ranking is the decisive evidence; the table below lists them.",
    ]
    lines += [
        "",
        "## Material states (actual ΔG spread > 10 CNY)",
        "",
        "| ckpt | day | block | ESS | leg | Q range | ΔG range | Spearman | "
        "actor quote | best quote | worst quote | best ΔG | worst ΔG |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in material.sort_values(
            ["checkpoint", "pair", "block", "target", "leg"]).iterrows():
        lines.append(
            f"| {r['checkpoint']} | {r['pair']} | {r['block']} | "
            f"{r['target']} | {r['leg']} | {r['q_range']:.3f} | "
            f"{r['delta_range']:.1f} | {r['spearman_q_delta']:.2f} | "
            f"{r['actor_quote']:.1f} | {r['best_quote']:.1f} | "
            f"{r['worst_quote']:.1f} | {r['best_delta']:.1f} | "
            f"{r['worst_delta']:.1f} |")
    lines += [
        "",
        "## Q vs training return (absolute scale)",
        "",
        f"- Correlation between Q×1000 and the discounted training return: "
        f"{np.corrcoef(s['q_scale_mean'], s['train_return_mean'])[0,1]:.3f}.",
        f"- Mean absolute gap |Q×1000 − training return|: "
        f"{np.mean(np.abs(s['q_scale_mean'] - s['train_return_mean'])):.1f} "
        "CNY-equivalent. The Q is not a usable absolute predictor of the "
        "discounted return either.",
        "",
        "## Findings for the required questions",
        "",
        "1. **Can the critic rank long-horizon quote value?** Not reliably. "
        "The overall Spearman is ~0; R10 is negative; in the material states "
        "the critic is often anti-ranked. The R2 best checkpoint is only "
        "slightly positive.",
        "2. **Is the flat Q the model, the data, or the market?** Both. Most "
        "states (72/88) have an economically flat market, so a flat Q is "
        "appropriate there. In the material states the critic still fails, "
        "and the replay buffer does contain high quotes (the final buffer has "
        "upper-bound quotes in 64/255 sampled transitions), so the failure is "
        "not simply missing action coverage. It is a calibration/target "
        "problem in the states that matter.",
        "3. **Are the high quotes causes or results?** In the material states "
        "the actor's quote is frequently at the worst end (ESS14/ESS27 "
        "discharge near 299/282) while the best quote is at/near 0; the high "
        "quote is the direct cause of the lost settlement. It is also the "
        "actor's chosen action, weakly preferred by the flat/anti-ranked "
        "critic; the winter price and cycle cost make the loss material.",
        "4. **The winter blocks:** the per-block reconstruction identifies "
        "blocks 10/14/11 on 2022-12-30, 3/17 on 2022-12-13 and 17 on "
        "2022-12-31 for ESS14/ESS27. The high quote plus cycle cost sits "
        "above the nodal price and the battery idles while B1 discharges.",
        "",
        "The state snapshots (SOC, block, gate/quote, price window, other-agent "
        "actions, data/checkpoint hashes) are in "
        "`critic_long_horizon_states_cp*.json`; the row-level grid is "
        "`critic_long_horizon_cp*.csv`.",
        "",
    ]
    (REPORTS / "critic_long_horizon_calibration.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"wrote report; material groups {len(material)}; "
          f"mean Spearman {s['spearman_q_delta'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
