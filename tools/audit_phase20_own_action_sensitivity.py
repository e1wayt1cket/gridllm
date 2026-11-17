"""P20-D: own-action versus cross-action critic sensitivity, best vs final.

Read-only. No training, no checkpoint write, no H2 and no legacy test.

The measurement is local: at a fixed joint state, perturb one unit's quote and
differentiate the target critic's min(Q1, Q2) with respect to that quote. The
absolute sensitivity is scaled Q per CNY/MWh; the box-normalized sensitivity
multiplies it by the leg's (high - low) so a unit is one full box width. The
critic is evaluated without dropout so best and final see the same mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import diagnose_critic_long_horizon as diag
import rl_da_gate as newmod
import train_da_gate as tb

DATA = Path("data/processed/v2_cy_jn_2025")
OUT_JSON = DATA / "phase20_own_action_sensitivity.json"
OUT_CSV = DATA / "phase20_own_action_sensitivity.csv"
RUNS = Path("runs")
NEW_RUN = RUNS / "DEV_DA_GATE_seed11_v2"
NEW_BEST = RUNS / "DEV_DA_GATE_seed11_v2_best"
LEGS = ("charge", "discharge")
HIGHS = {"charge": 150.0, "discharge": 300.0}
UNITS = ("ESS14", "ESS27")
# Initial states: the four Phase 19 training states plus the five named
# validation cases. The validation cases are offline diagnosis only.
INITIAL = (
    ("2022-12-24|2025-12-21", 8, "ESS14", "train_near"),
    ("2022-12-24|2025-12-21", 8, "ESS27", "train_near"),
    ("2022-12-29|2025-12-30", 14, "ESS14", "train_near"),
    ("2022-12-29|2025-12-30", 14, "ESS27", "train_near"),
    ("2022-12-30|2025-12-31", 10, "ESS14", "validation_near"),
    ("2022-12-13|2025-12-11", 12, "ESS27", "validation_near"),
    ("2022-12-30|2025-12-31", 11, "ESS14", "validation_near"),
    ("2022-12-13|2025-12-11", 3, "ESS14", "validation_near"),
    ("2022-12-31|2025-12-28", 17, "ESS14", "validation_near"),
)
# Pre-registered expansion if the initial states do not separate best/final.
EXPAND = (
    ("2022-12-16|2025-12-17", 8, "ESS14", "train_near"),
    ("2022-12-17|2025-12-14", 14, "ESS27", "train_near"),
    ("2022-12-22|2025-12-23", 10, "ESS14", "train_near"),
)


def load_da(template):
    manifest = json.loads(
        (NEW_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    experiment = newmod.DAExperimentConfig(**manifest["experiment"])
    out = {}
    for label, root in (("da_best", NEW_BEST), ("da_final", NEW_RUN)):
        trainer = newmod.DAGateTrainer(template, experiment)
        trainer.load(root)
        out[label] = trainer
    return out


def sensitivity(trainer, obs, baseline, critic_unit, perturb_unit, leg,
                quote):
    names = list(baseline)
    leg_idx = LEGS.index(leg)
    actions = {nm: baseline[nm].copy() for nm in names}
    actions[perturb_unit][leg_idx] = quote
    obs_t = torch.tensor(np.stack([obs[nm] for nm in names]),
                         dtype=torch.float32).reshape(1, -1)
    quote_t = torch.tensor(
        [quote if i == leg_idx else float(baseline[perturb_unit][i])
         for i in range(2)], dtype=torch.float32, requires_grad=True)
    full = torch.tensor(np.stack([actions[nm] for nm in names]),
                        dtype=torch.float32)
    full[names.index(perturb_unit)] = quote_t
    critic = trainer.critics[critic_unit]
    critic.eval()
    critic.zero_grad()
    q1, q2 = critic(obs_t, full.reshape(1, -1))
    torch.min(q1, q2).mean().backward()
    gradient = float(quote_t.grad[leg_idx].detach())
    critic.train()
    return gradient


def state_rows(label, trainer, pair, block, critic_unit, source):
    env, obs, baseline, past, soc, *_ = diag.collect_state(
        trainer, pair, block)
    names = list(baseline)
    own = {}
    cross = {}
    for leg in LEGS:
        quote = float(baseline[critic_unit][LEGS.index(leg)])
        g = sensitivity(trainer, obs, baseline, critic_unit, critic_unit,
                        leg, quote)
        own[leg] = g
        cross[leg] = {}
        for other in names:
            if other == critic_unit:
                continue
            cross[leg][other] = sensitivity(
                trainer, obs, baseline, critic_unit, other, leg,
                float(baseline[other][LEGS.index(leg)]))
    own_abs = sum(own.values())
    own_sq = sum((own[leg] * (HIGHS[leg] - 0.0)) ** 2 for leg in LEGS)
    cross_sq = sum((value * (HIGHS[leg] - 0.0)) ** 2
                   for leg in LEGS for value in cross[leg].values())
    total = own_sq + cross_sq
    return {
        "checkpoint": label,
        "source": source,
        "pair_id": f"{pair['load_date']}|{pair['price_date']}",
        "block": block,
        "critic_unit": critic_unit,
        "charge_own_abs": own["charge"],
        "discharge_own_abs": own["discharge"],
        "charge_own_norm": own["charge"] * HIGHS["charge"],
        "discharge_own_norm": own["discharge"] * HIGHS["discharge"],
        "own_abs_sum": own_abs,
        "own_norm_sq": own_sq,
        "cross_norm_sq": cross_sq,
        "own_share": own_sq / total if total > 0 else 0.0,
        "own_cross_ratio": (float(np.sqrt(own_sq / cross_sq))
                            if cross_sq > 0 else None),
        "cross_charge": {k: v for k, v in cross["charge"].items()},
        "cross_discharge": {k: v for k, v in cross["discharge"].items()},
        "soc": float(soc[critic_unit]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expand", action="store_true",
                        help="add the pre-registered three training states")
    args = parser.parse_args(argv)
    states = INITIAL + (EXPAND if args.expand else ())
    pair_map, _ = diag.pairs()
    template = tb.envmod.build_env_for_pair(
        pair_map[diag.WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    trainers = load_da(template)
    rows = []
    for label, trainer in trainers.items():
        for pid, block, unit, source in states:
            rows.append(state_rows(
                label, trainer, pair_map[pid], block, unit, source))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "boundary": ("read-only local critic sensitivity; no training, no "
                     "checkpoint write, no H2 and no legacy test"),
        "states": [{"pair_id": p, "block": b, "unit": u, "role": s}
                   for p, b, u, s in states],
        "note": ("the four training states use the near/split.json role of "
                 "the D+A run; the named cases are validation there and "
                 "training days under common_split"),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["checkpoint", "source", "pair_id", "block",
                            "critic_unit", "charge_own_abs",
                            "discharge_own_abs", "charge_own_norm",
                            "discharge_own_norm", "own_abs_sum",
                            "own_norm_sq", "cross_norm_sq", "own_share",
                            "own_cross_ratio"])
        writer.writeheader()
        writer.writerows([{k: r[k] for k in writer.fieldnames} for r in rows])
    for label in ("da_best", "da_final"):
        subset = [r for r in rows if r["checkpoint"] == label
                  and r["critic_unit"] in UNITS]
        for unit in UNITS:
            values = [r["own_share"] for r in subset
                      if r["critic_unit"] == unit]
            if values:
                print(label, unit, "median own_share",
                      round(float(np.median(values)), 3),
                      "n", len(values))
    print(f"wrote {OUT_JSON} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
