"""P20-C: twin gap, Bellman residual and target variance on the D+A replay.

Read-only. No training, no checkpoint write, no replay addition, no H2 and no
legacy test.

The Bellman target is reconstructed from the code path actually used by
``MATD3BC2Gate.update``:

    target = reward + gamma * (1 - done) * min(target_Q1, target_Q2)

with the target actions produced by ``DAGateTrainer._target_quotes`` (v3 map
plus the configured target noise). The replay TD residual is a different
quantity from the Phase 19 matched full-day return; both are reported and
labelled separately.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import da_quote_map as qmap
import diagnose_critic_long_horizon as diag
import rl_da_gate as newmod
import train_da_gate as tb

DATA = Path("data/processed/v2_cy_jn_2025")
OUT_JSON = DATA / "phase20_bellman_diagnostics.json"
OUT_CSV = DATA / "phase20_bellman_diagnostics.csv"
CALIBRATION = (
    DATA / "phase19_critic_training_calibration_rows.csv",
    DATA / "daily_counterfactual_action_validation.csv",
)
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
HIGHS = (150.0, 300.0)
BATCH = 2048
SEED = 20261102
RUNS = Path("runs")
NEW_RUN = RUNS / "DEV_DA_GATE_seed11_v2"
NEW_BEST = RUNS / "DEV_DA_GATE_seed11_v2_best"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_da(template):
    manifest = json.loads(
        (NEW_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    experiment = newmod.DAExperimentConfig(**manifest["experiment"])
    out = {}
    for label, root in (("da_best", NEW_BEST),
                        ("da_final", NEW_RUN)):
        trainer = newmod.DAGateTrainer(template, experiment)
        trainer.load(root)
        out[label] = trainer
    return out


def contiguous_batch(trainer, batch: int) -> dict:
    rows = list(trainer.buffer.buffer)
    if not rows:
        raise RuntimeError("empty replay buffer")
    idx = np.linspace(0, len(rows) - 1, min(batch, len(rows))).astype(int)
    picked = [rows[i] for i in idx]
    obs, act, rew, next_obs, done = zip(*picked)
    return {
        "obs": torch.as_tensor(np.asarray(obs), dtype=torch.float32),
        "act": torch.as_tensor(np.asarray(act), dtype=torch.float32),
        "rew": torch.as_tensor(np.asarray(rew), dtype=torch.float32),
        "next_obs": torch.as_tensor(np.asarray(next_obs),
                                    dtype=torch.float32),
        "done": torch.as_tensor(np.asarray(done),
                                dtype=torch.float32),
        "n": len(picked),
    }


def target_with_noise(trainer, next_obs: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [trainer._target_quotes(trainer.actor_targets[nm],
                                next_obs[:, j, :])
         for j, nm in enumerate(trainer.agent_names)], dim=1)


def target_without_noise(trainer, next_obs: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [qmap.target_quotes_v3(trainer.actor_targets[nm](
            next_obs[:, j, :]))
         for j, nm in enumerate(trainer.agent_names)], dim=1)


def band(quote: float, high: float) -> str:
    frac = quote / high
    return "low" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "high")


def replay_diagnostics(trainer, label: str) -> dict:
    batch = contiguous_batch(trainer, BATCH)
    obs, act, rew, next_obs, done = (
        batch["obs"], batch["act"], batch["rew"], batch["next_obs"],
        batch["done"])
    size = batch["n"]
    obs_flat = obs.reshape(size, -1)
    torch.manual_seed(SEED)
    target_actions = target_with_noise(trainer, next_obs)
    torch.manual_seed(SEED)
    target_actions_nn = target_without_noise(trainer, next_obs)
    rows = []
    for i, nm in enumerate(trainer.agent_names):
        target_critic = trainer.critic_targets[nm]
        with torch.no_grad():
            tq1, tq2 = target_critic(
                obs_flat, target_actions.reshape(size, -1))
            tq1_nn, tq2_nn = target_critic(
                obs_flat, target_actions_nn.reshape(size, -1))
            target = rew[:, i:i + 1] + trainer.experiment.gamma \
                * (1.0 - done[:, None]) * torch.min(tq1, tq2)
            target_nn = rew[:, i:i + 1] + trainer.experiment.gamma \
                * (1.0 - done[:, None]) * torch.min(tq1_nn, tq2_nn)
            q1, q2 = trainer.critics[nm](
                obs_flat, act.reshape(size, -1))
        gap = (q1 - q2).abs().flatten()
        residual1 = (q1 - target).flatten()
        residual2 = (q2 - target).flatten()
        smooth = (target - target_nn).flatten()
        nonzero = float(np.std(target.detach().numpy()))
        for leg in (0, 1):
            quotes = act[:, i, leg].numpy()
            for level in ("low", "mid", "high"):
                mask = np.asarray(
                    [band(float(q), HIGHS[leg]) == level for q in quotes])
                if not mask.any():
                    continue
                rows.append({
                    "checkpoint": label,
                    "agent": nm,
                    "leg": "charge" if leg == 0 else "discharge",
                    "band": level,
                    "n": int(mask.sum()),
                    "twin_gap_mean": float(gap.numpy()[mask].mean()),
                    "twin_gap_median": float(np.median(gap.numpy()[mask])),
                    "q1_target_error_mean": float(
                        residual1.numpy()[mask].mean()),
                    "q2_target_error_mean": float(
                        residual2.numpy()[mask].mean()),
                    "abs_residual_median": float(np.median(np.abs(
                        residual1.numpy()[mask]))),
                    "target_mean": float(target.numpy()[mask].mean()),
                    "target_std": float(target.numpy()[mask].std()),
                    "target_std_across_batch": nonzero,
                    "smoothing_abs_delta_mean": float(
                        np.abs(smooth.numpy()[mask]).mean()),
                    "smoothing_spearman": _spearman(
                        target.numpy()[mask].ravel(),
                        target_nn.numpy()[mask].ravel()),
                })
    return {
        "checkpoint": label,
        "buffer_rows": len(trainer.buffer),
        "batch_rows": size,
        "gamma": trainer.experiment.gamma,
        "target_noise": trainer.experiment.target_policy_noise,
        "target_noise_clip": trainer.experiment.target_noise_clip,
        "rows": rows,
    }


def _spearman(a, b) -> float | None:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def matched_return_rows() -> list[dict]:
    rows = []
    for path in CALIBRATION:
        for row in read_rows(path):
            if row.get("checkpoint") not in ("da_best", "da_final"):
                continue
            q = float(row["q_min"])
            r = float(row["training_return_scaled"])
            spread = 0.0
            rows.append({
                "checkpoint": row["checkpoint"],
                "pair_id": row["pair_id"],
                "block": int(row["block"]),
                "unit": row["unit"],
                "leg": row["leg"],
                "quote": float(row["quote"]),
                "q_min": q,
                "training_return_scaled": r,
                "twin_gap": float(row.get("twin_gap") or 0.0),
                "full_delta_cny": float(row.get("full_delta_cny") or 0.0),
                "_spread": spread,
            })
    groups = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["pair_id"], row["block"],
                row["unit"], row["leg"])].append(row)
    out = []
    for key, values in groups.items():
        returns = np.asarray([v["training_return_scaled"] for v in values])
        qvals = np.asarray([v["q_min"] for v in values])
        spread = float(np.max(returns) - np.min(returns))
        out.append({
            "checkpoint": key[0],
            "pair_id": key[1],
            "block": key[2],
            "unit": key[3],
            "leg": key[4],
            "groups": 1,
            "rows": len(values),
            "return_spread_scaled": spread,
            "return_spread_cny_equiv": 1000.0 * spread,
            "q_return_spearman": _spearman(qvals, returns),
            "median_twin_gap": float(np.median(
                [v["twin_gap"] for v in values])),
            "median_abs_q_minus_return_cny": float(1000.0 * np.median(
                np.abs(qvals - returns))),
        })
    return out


def main() -> int:
    pair_map, _ = diag.pairs()
    template = tb.envmod.build_env_for_pair(
        pair_map[diag.WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    trainers = load_da(template)
    replay = [replay_diagnostics(trainers[label], label)
              for label in ("da_best", "da_final")]
    matched = matched_return_rows()
    payload = {
        "boundary": ("read-only replay diagnostics; the TD residual is "
                     "recomputed from the stored reward and next observation, "
                     "not from the matched full-day return"),
        "batch_rule": "evenly spaced transitions over the whole buffer",
        "batch_size": BATCH,
        "seed": SEED,
        "replay": replay,
        "matched_return_groups": matched,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [row for block in replay for row in block["rows"]]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for block in replay:
        gaps = [r["twin_gap_median"] for r in block["rows"]]
        resid = [r["abs_residual_median"] for r in block["rows"]]
        print(block["checkpoint"], "buffer", block["buffer_rows"],
              "median twin gap", round(float(np.median(gaps)), 4),
              "median |residual|", round(float(np.median(resid)), 4))
    print(f"wrote {OUT_JSON} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
