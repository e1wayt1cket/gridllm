"""Read-only gate initialization and exploration check.

The observed collapse is already present at the first checkpoint, so this
script separates three things:

* the default actor initialization, evaluated on 48 real replay states;
* the Bernoulli probability and its straight-through factor p(1-p) at the
  probabilities the trained actors actually reached;
* what a gate-only temperature or a different initial probability does to
  p(1-p) and to the number of joint modes.

No training and no checkpoint write.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from rl_b_c2_gate import GateActor
from truthful_gate_actor_adapter import fleet_mode

DATA = Path("data/processed/v2_cy_jn_2025")
RUN = Path("runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2")
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")
GATE_NAMES = tuple(f"{u}.{leg}" for u in UNITS
                   for leg in ("charge", "discharge"))


def load_states():
    z = np.load(RUN / "checkpoints/step_00012720/buffer/replay.npz")
    # Two complete training days, 24 blocks each.
    return np.asarray(z["obs"][:48], dtype=np.float32)


def modes_from_gates(gates):
    """(states, 4, 2) hard gates -> distinct fleet modes."""
    out = []
    for row in gates:
        out.append(fleet_mode([int(row[j, 0]) + 2 * int(row[j, 1])
                               for j in range(row.shape[0])]))
    return out


def simulate_modes(probs, draws, seed=0):
    """Bernoulli draws from one 8-gate probability vector, N times."""
    rng = np.random.default_rng(seed)
    p = np.asarray(probs, dtype=float)
    seen = set()
    for _ in range(draws):
        draw = (rng.random(len(p)) < p).astype(int)
        per_unit = []
        for j in range(len(UNITS)):
            per_unit.append(int(draw[2 * j]) + 2 * int(draw[2 * j + 1]))
        seen.add(fleet_mode(per_unit))
    return len(seen)


def main(argv=None) -> int:
    states = load_states()
    n = len(states)
    rows = []
    summary = {"fresh_init": {}, "initial_probability": {}, "temperature": {},
               "observed": {}}

    # 1. Default initialization, 20 seeds.
    det_counts, sample_counts, pvals = [], [], []
    for seed in range(20):
        torch.manual_seed(seed)
        actors = {u: GateActor(obs_dim=states.shape[2]) for u in UNITS}
        with torch.no_grad():
            logits = np.stack([
                np.stack([actors[u](torch.as_tensor(states[i, j]))[:2].numpy()
                          for i in range(n)]) for j, u in enumerate(UNITS)],
                axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        pvals.extend(probs.reshape(-1).tolist())
        det = modes_from_gates((logits > 0).astype(int))
        sample_seen = set()
        rng = np.random.default_rng(seed)
        for i in range(n):
            for _ in range(100):
                draw = (rng.random(probs[i].shape) < probs[i]).astype(int)
                sample_seen.add(fleet_mode([
                    int(draw[j, 0]) + 2 * int(draw[j, 1])
                    for j in range(len(UNITS))]))
        det_counts.append(len(set(det)))
        sample_counts.append(len(sample_seen))
        rows.append({"kind": "fresh_init", "seed": seed,
                     "det_modes": len(set(det)),
                     "sample_modes_100x": len(sample_seen),
                     "p_mean": float(probs.mean()),
                     "p_min": float(probs.min()),
                     "p_max": float(probs.max()),
                     "p1mp_min": float((probs * (1 - probs)).min())})
    summary["fresh_init"] = {
        "seeds": 20, "states": n,
        "det_modes_mean": float(np.mean(det_counts)),
        "det_modes_min": int(np.min(det_counts)),
        "det_modes_max": int(np.max(det_counts)),
        "sample_modes_mean": float(np.mean(sample_counts)),
        "p_mean": float(np.mean(pvals)),
        "p1mp_mean": float(np.mean(np.asarray(pvals)
                                   * (1 - np.asarray(pvals)))),
        "p1mp_min": float(np.min(np.asarray(pvals)
                                 * (1 - np.asarray(pvals)))),
    }

    # 2. Different Bernoulli initial probabilities, one eight-gate vector.
    for p0 in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
        probs = np.full(len(GATE_NAMES), p0)
        modes = simulate_modes(probs, 100, seed=0)
        rows.append({"kind": "initial_probability", "seed": "",
                     "det_modes": "", "sample_modes_100x": modes,
                     "p_mean": p0, "p_min": p0, "p_max": p0,
                     "p1mp_min": p0 * (1 - p0)})
        summary["initial_probability"][str(p0)] = {
            "p1mp": p0 * (1 - p0), "sample_modes_100x": modes,
        }

    # 3. Gate temperature on the probabilities the trained actors reached.
    traj = json.loads((DATA / "gate_trajectory_by_round.json").read_text(
        encoding="utf-8"))
    final = traj["10"]["gates"]
    logits = np.array([final[u][leg]["logit_mean"]
                       for u in UNITS for leg in ("charge", "discharge")])
    for tau in (1.0, 2.0, 4.0):
        probs = 1.0 / (1.0 + np.exp(-logits / tau))
        modes = simulate_modes(probs, 100, seed=0)
        rows.append({"kind": "temperature", "seed": "",
                     "det_modes": "", "sample_modes_100x": modes,
                     "p_mean": float(probs.mean()),
                     "p_min": float(probs.min()),
                     "p_max": float(probs.max()),
                     "p1mp_min": float((probs * (1 - probs)).min())})
        summary["temperature"][str(tau)] = {
            "probabilities": probs.tolist(),
            "p1mp": (probs * (1 - probs)).tolist(),
            "p1mp_min": float((probs * (1 - probs)).min()),
            "p1mp_mean": float(np.mean(probs * (1 - probs))),
            "sample_modes_100x": modes,
        }

    # 4. Recorded deterministic and sampled modes at every round.
    for rnd, rec in traj.items():
        summary["observed"].setdefault("det_modes", {})[rnd] = \
            rec.get("distinct_deterministic_modes")
        summary["observed"].setdefault("sample_modes", {})[rnd] = \
            rec.get("distinct_sampled_modes")

    out_csv = DATA / "gate_initialization_exploration.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (DATA / "gate_initialization_exploration.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    assert summary["fresh_init"]["p1mp_min"] > 0.1
    assert summary["fresh_init"]["det_modes_mean"] > 1.0
    assert summary["temperature"]["4.0"]["p1mp_min"] > \
        summary["temperature"]["1.0"]["p1mp_min"]
    print(json.dumps({k: summary[k] for k in
                      ("fresh_init", "initial_probability", "temperature")},
                     indent=1))
    print(f"wrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
