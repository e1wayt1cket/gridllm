"""P2: controlled offline diagnostics for temperature, replay and critic ranking.

Protocol `P2_CONTROLLED_DIAGNOSTICS_V2`. Read-only with respect to every
original artifact:

* no training, no `trainer.save`, no buffer add, no target soft update;
* checkpoints are opened through `DAGateTrainer.load` and stepped only on the
  in-memory actor copies restored from a deep-copied snapshot;
* every state comes from Train pairs; Validation is never fitted, Test and
  Holiday are never touched, H2 stays sealed;
* the only writes are the CSV/JSON files under `reports/algorithm_diagnostics`.

Stages:

  temp    -- what the gate temperature changes in the quote map;
  matrix  -- tau / replay-content / replay-size control matrix (A1-A3);
  a4      -- optional cross-checkpoint replication (seed 23, tau 1.0 / 2.5);
  rank    -- critic ranking against the undiscounted remaining-day payoff on
             winter Train days.

Each stage loads and updates `P2_v4_evidence.json`, so a killed process keeps the
completed stages and cells.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import da_quote_map as qmap
import da_replay as replay
import diagnose_actor_update as P1
import rl_da_gate as mod
import v2_cy_jn_env as envmod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics"
EVIDENCE_PATH = OUT / "P2_v4_evidence.json"
MATRIX_CSV = OUT / "P2_v4_control_matrix.csv"
RANK_CSV = OUT / "P2_v4_winter_critic_ranking.csv"
RANK_ACTIVITY_CSV = OUT / "P2_v4_winter_critic_ranking_activity.csv"

CHECKPOINTS = {
    11: ROOT / "runs" / "V4D_SHORT_seed11_best",
    23: ROOT / "runs" / "V4D_SHORT_seed23_best",
}
NAMES = P1.NAMES
REPS = 5
BATCH = 128
STEPS = 10
TAUS = (1.0, 2.5, 4.0)
CHARGE_GRID = (0.0, 37.5, 75.0, 112.5, 150.0)
DISCHARGE_GRID = (0.0, 75.0, 150.0, 225.0, 300.0)
WINTER_PAIRS = ("2022-12-09|2025-12-09", "2022-12-10|2025-12-07")
RANK_BLOCKS = (4, 10, 16)
RANK_CHARGE_BLOCKS = (4, 16)
RANK_UNITS = ("ESS14", "ESS27")
TIE = 1e-6
# 口径 A (user-approved 2026-11-09, b/W9双分歧铰链_G1重测_2026-11-09.md §5):
# the top-1 vote counts only states whose payoff range clears the hinge
# calibration's flat-payoff tolerance (v4_hinge_calibration.json:
# flat_payoff_tol_cny), and the critic's pick counts as agreement whenever
# its payoff is within that tolerance of the true best quote. Ranking ties
# inside the calibrated noise floor are measurement noise, not failures.
TOP1_FLAT_TOL_CNY = 0.5
ACTIVITY_MWH = 0.05

MATRIX_COLUMNS = (
    "experiment", "checkpoint", "buffer", "tau", "replicate", "seed",
    "pair_id", "season", "G_policy", "delta_vs_arm0",
    *[f"delta_vs_arm0_{nm}" for nm in NAMES],
    "q_before", "q_after", "grad_actor", "grad_gate", "grad_price",
    "batch_size", "batch_unique", "batch_duplicates",
    "batch_indices", "buffer_n", "buffer_sha", "optimizer_sha",
    "updates_applied",
)

RANK_COLUMNS = (
    "state_id", "pair_id", "block", "unit", "leg", "gate_open", "leg_switched",
    "quote", "q1", "q2", "q_min", "twin_gap", "dq_dquote",
    "payoff_remaining_unit", "payoff_remaining_fleet", "accept_mwh",
    "soc_before", "clearing_failed",
)


# ----------------------------------------------------------------------
# Evidence and environment
# ----------------------------------------------------------------------
def load_evidence() -> dict:
    if EVIDENCE_PATH.exists():
        return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    return {}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach())
        return value.detach().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_evidence(evidence: dict, verdict: str | None = None) -> None:
    if verdict is not None:
        evidence["verdict"] = verdict
    EVIDENCE_PATH.write_text(json.dumps(jsonable(evidence), indent=2),
                             encoding="utf-8")


def versions() -> dict:
    import os
    import scipy
    return {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "gurobi": list(__import__("gurobipy").gurobi.version()),
        "cwd": os.getcwd(),
    }


# ----------------------------------------------------------------------
# Borrowed P1 primitives
# ----------------------------------------------------------------------
def triples() -> tuple:
    data = P1.train_pairs()
    ids = P1.select_pairs(data["pairs"])
    return ([(pid, data["pairs"][pid], data["full"][pid]) for pid in ids],
            data["mechanism_hash"])


def load_trainer(ckpt: Path, template_pair: dict):
    env = envmod.build_env_for_pair(
        template_pair, differential_reward=False, terminal_soc_target=0.5)
    trainer = mod.DAGateTrainer(env, mod.DAExperimentConfig("diagnostic"))
    bundle = trainer.load(ckpt)
    return trainer, bundle


def set_temperature(trainer, tau: float) -> None:
    """The actor loss reads `current_gate_temperature`; the eval path does not."""
    trainer.current_gate_temperature = lambda: float(tau)


def optimizer_hash(trainer) -> str:
    h = hashlib.sha256()
    for nm in trainer.agent_names:
        state = trainer.actor_opts[nm].state_dict()
        for key in sorted(state):
            value = state[key]
            if isinstance(value, torch.Tensor):
                h.update(key.encode())
                h.update(value.detach().cpu().numpy().tobytes())
            else:
                h.update(f"{key}={value}".encode())
    return h.hexdigest()


def buffer_fingerprint(buf) -> dict:
    """Content hash, size, uniqueness and schema, without touching the buffer."""
    digest = hashlib.sha256()
    row_digests = set()
    for row in buf.buffer:
        parts = [np.asarray(row[i], dtype=np.float32).tobytes()
                 for i in range(4)]
        parts.append(np.float32(row[4]).tobytes())
        blob = b"".join(parts)
        digest.update(blob)
        row_digests.add(hashlib.sha256(blob).digest())
    for key in sorted(buf.extras):
        for value in buf.extras[key]:
            digest.update(np.asarray(value, dtype=np.float32).tobytes())
    return {
        "n": len(buf.buffer),
        "sha256": digest.hexdigest(),
        "unique_transitions": len(row_digests),
        "extras_keys": sorted(buf.extras),
        "obs_shape": list(np.asarray(buf.buffer[0][0]).shape),
        "action_shape": list(np.asarray(buf.buffer[0][1]).shape),
        "config": dict(buf.config),
        "dates_available": False,
    }


def buffer_compatibility(a: dict, b: dict, manifest_a: dict,
                         manifest_b: dict) -> dict:
    checks = {
        "extras_keys": a["extras_keys"] == b["extras_keys"],
        "obs_shape": a["obs_shape"] == b["obs_shape"],
        "action_shape": a["action_shape"] == b["action_shape"],
        "sampler_config": a["config"] == b["config"],
        "data_hashes": manifest_a.get("data") == manifest_b.get("data"),
        "specs": manifest_a.get("specs") == manifest_b.get("specs"),
    }
    return {"checks": checks, "compatible": all(checks.values())}


def stratified_subset_indices(buf, n: int, seed: int) -> list:
    """Proportional allocation over the buffer's own joint cells."""
    strata = np.asarray(buf.extras["stratum"], dtype=float).reshape(-1)
    strata = np.rint(strata).astype(int)
    if n >= strata.size:
        return list(range(strata.size))
    keys, counts = np.unique(strata, return_counts=True)
    quota = counts.astype(float) * n / strata.size
    alloc = np.floor(quota).astype(int)
    remainder = n - int(alloc.sum())
    order = np.argsort(-(quota - alloc))
    for i in order[:remainder]:
        alloc[i] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for key, k in zip(keys, alloc):
        members = np.flatnonzero(strata == key)
        k = min(int(k), members.size)
        chosen.extend(rng.choice(members, size=k, replace=False).tolist())
    return sorted(int(i) for i in chosen)


def make_subset(buf, indices) -> object:
    out = replay.DAReplayBuffer(
        capacity=len(indices),
        uniform_fraction=buf.config["uniform_fraction"],
        beta=buf.config["beta"],
        max_upweight=buf.config["max_upweight"])
    for i in indices:
        out.buffer.append(buf.buffer[i])
        for key in out.extras:
            out.extras[key].append(buf.extras[key][i])
    return out


# ----------------------------------------------------------------------
# Stage 1: temperature mechanism
# ----------------------------------------------------------------------
GATE_GRID = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
TEMP_OTHER_GATE = 4.0
TEMP_PRICE_LOGITS = (1.5, 0.0)


def _temp_probe(gate_logit: float, tau: float) -> torch.Tensor:
    """One charge-leg logit varied; the other gate fixed open at logit 4."""
    return torch.tensor([[gate_logit, TEMP_OTHER_GATE,
                          TEMP_PRICE_LOGITS[0], TEMP_PRICE_LOGITS[1]]],
                        dtype=torch.float64)


def temp_study() -> dict:
    """What tau changes in `forward_policy_v3`, measured on the real function."""
    high = float(qmap.LEG_HIGHS[0])
    z = torch.tensor([TEMP_PRICE_LOGITS[0]], dtype=torch.float64)
    frac = float(qmap.map_fraction(z))
    dz = float(qmap.map_derivative(z))
    price_quote = high * frac
    seeds = 4000
    h = 1e-5
    reference: dict = {}
    rows = []
    for tau in TAUS:
        probs = torch.sigmoid(torch.tensor(GATE_GRID, dtype=torch.float64)
                              / tau)
        quotes, gates, st_grad, price_grad = {}, {}, {}, {}
        for i, g in enumerate(GATE_GRID):
            q_vals, gate_vals, sg_vals, pg_vals = [], [], [], []
            for seed in range(seeds):
                torch.manual_seed(seed)
                logits = _temp_probe(g, tau).requires_grad_(True)
                out = qmap.forward_policy_v3(logits, train=True,
                                             gate_temperature=tau)
                q_vals.append(float(out["quotes"][0, 0].detach()))
                gate_vals.append(float(out["gates"][0, 0].detach()))
                out["quotes"].sum().backward()
                sg_vals.append(float(logits.grad[0, 0]))
                pg_vals.append(float(logits.grad[0, 2]))
            quotes[g] = np.asarray(q_vals)
            gates[g] = np.asarray(gate_vals)
            st_grad[g] = float(np.mean(sg_vals))
            price_grad[g] = float(np.mean(pg_vals))
        per_point = []
        for i, g in enumerate(GATE_GRID):
            p = float(probs[i])
            analytic_gate = p * (1 - p) / tau * price_quote
            per_point.append({
                "gate_logit": g,
                "probability": p,
                "activation_frequency": float(gates[g].mean()),
                "quote_mean": float(quotes[g].mean()),
                "quote_std": float(quotes[g].std()),
                "quote_min": float(quotes[g].min()),
                "quote_max": float(quotes[g].max()),
                "expected_quote": p * price_quote,
                "st_gate_grad_measured": st_grad[g],
                "st_gate_grad_analytic": analytic_gate,
                "price_grad_measured": price_grad[g],
                "price_grad_expected": p * high * dz,
                "grad_norm": math.hypot(st_grad[g], price_grad[g]),
            })
        if not reference:
            reference = {"quotes": quotes, "price_grad": price_grad,
                         "gates": gates}
            forward_change, price_change = {}, {}
        else:
            forward_change = {str(g): float(np.mean(quotes[g]
                                                    != reference["quotes"][g]))
                              for g in GATE_GRID}
            gate_change = {str(g): float(np.mean(gates[g]
                                                 != reference["gates"][g]))
                           for g in GATE_GRID}
            price_change = {str(g): float(np.mean(
                abs(price_grad[g] - reference["price_grad"][g]) > 1e-12))
                for g in GATE_GRID}
            forward_change = {"quote": forward_change,
                              "gate": gate_change}
        fd_rows = []
        for i, g in enumerate(GATE_GRID):
            p_plus = float(torch.sigmoid(torch.tensor((g + h) / tau,
                                                      dtype=torch.float64)))
            p_minus = float(torch.sigmoid(torch.tensor((g - h) / tau,
                                                       dtype=torch.float64)))
            fd_expected = (p_plus - p_minus) / (2 * h) * price_quote
            torch.manual_seed(12345)
            plus = float(qmap.forward_policy_v3(
                _temp_probe(g + h, tau), train=True,
                gate_temperature=tau)["quotes"][0, 0])
            torch.manual_seed(12345)
            minus = float(qmap.forward_policy_v3(
                _temp_probe(g - h, tau), train=True,
                gate_temperature=tau)["quotes"][0, 0])
            fd_rows.append({
                "gate_logit": g,
                "fd_expected_quote": fd_expected,
                "st_analytic": per_point[i]["st_gate_grad_analytic"],
                "fd_expected_error": abs(fd_expected
                                         - per_point[i]["st_gate_grad_analytic"]),
                "fd_sampled_fixed_seed": (plus - minus) / (2 * h),
            })
        policy = torch.tensor([[0.4, -0.3, 1.5, -1.5],
                               [-0.6, 0.9, 0.2, 2.9]], dtype=torch.float64)
        eval_quotes = qmap.forward_policy_v3(policy, train=False)["quotes"]
        rows.append({
            "tau": tau,
            "probability": {g: float(v) for g, v in zip(GATE_GRID, probs)},
            "per_point": per_point,
            "forward_change_vs_first_tau": forward_change,
            "price_grad_change_vs_first_tau": price_change,
            "fd": fd_rows,
            "eval_quotes": eval_quotes.tolist(),
            "eval_ignores_temperature_argument": bool(torch.equal(
                eval_quotes,
                qmap.forward_policy_v3(policy, train=False,
                                       gate_temperature=tau)["quotes"])),
            "target_equals_eval": bool(torch.equal(
                qmap.target_quotes_v3(policy), eval_quotes)),
            "critic_equals_eval": bool(torch.equal(
                qmap.critic_action_v3(policy), eval_quotes)),
        })

    def varies(key):
        return len({json.dumps([round(p[key], 12) for p in row["per_point"]])
                    for row in rows}) > 1

    p_changes = varies("probability")
    activation_changes = varies("activation_frequency")
    quote_changes = any(v > 0 for row in rows[1:]
                        for v in row["forward_change_vs_first_tau"]["quote"]
                        .values())
    gate_changes = any(v > 0 for row in rows[1:]
                       for v in row["forward_change_vs_first_tau"]["gate"]
                       .values())
    gradient_changes = varies("st_gate_grad_measured")
    price_gradient_changes = varies("price_grad_measured")
    forward_changes = quote_changes and gate_changes
    if forward_changes and (gradient_changes or price_gradient_changes):
        classification = "GATE_FORWARD_AND_BACKWARD"
    elif p_changes and gradient_changes:
        classification = "GATE_PROBABILITY_AND_GRADIENT"
    elif gradient_changes and not p_changes:
        classification = "GATE_GRADIENT_ONLY"
    else:
        classification = "OTHER_EFFECT"
    return {
        "construction": ("charge-leg gate logit grid [-4,-2,-1,0,1,2,4]; "
                         "other gate pinned open at logit 4; price logits "
                         "(1.5, 0.0); 4000 matched torch seeds per point per "
                         "tau; the frozen forward map is called unchanged"),
        "price_quote_open_leg": price_quote,
        "probability_depends_on_tau": p_changes,
        "hard_activation_depends_on_tau": activation_changes,
        "sampled_forward_quote_depends_on_tau": forward_changes,
        "gate_gradient_depends_on_tau": gradient_changes,
        "price_gradient_depends_on_tau": price_gradient_changes,
        "evaluation_independent_of_tau": all(
            r["eval_ignores_temperature_argument"]
            and r["target_equals_eval"] and r["critic_equals_eval"]
            for r in rows),
        "classification": classification,
        "rows": rows,
    }


# ----------------------------------------------------------------------
# Stage 2: control matrix
# ----------------------------------------------------------------------
def arm0_baseline(trainer, triples_, b1_cache, mechanism_hash) -> dict:
    return {pid: P1.rollout_day(trainer, pair, b1_cache,
                                mechanism_hash=mechanism_hash,
                                pair_row=row)
            for pid, pair, row in triples_}


def run_cell(trainer, snap, buf, tau, ckpt_seed, experiment, buffer_label,
             baseline, triples_, b1_cache, mechanism_hash, writer, fh) -> dict:
    set_temperature(trainer, tau)
    opt_sha = optimizer_hash(trainer)
    buffer_info = buffer_fingerprint(buf)
    per_day = {pid: [] for pid, _, _ in triples_}
    per_unit = {nm: {pid: [] for pid, _, _ in triples_} for nm in NAMES}
    q_before, q_after, grads = [], [], []
    seed_used = []
    for rep in range(REPS):
        seed = 10000 * ckpt_seed + 100 * STEPS + rep
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        batch = buf.sample(BATCH)
        indices = np.asarray(batch["indices"])
        steps = []
        stats = P1.run_arm(trainer, snap, batch["obs"], batch["actions"],
                           batch["obs"].reshape(batch["obs"].shape[0], -1),
                           STEPS, steps)
        qb = float(np.mean(list(stats["q_eval_before"].values())))
        qa = float(np.mean(list(stats["q_eval_after"].values())))
        ga = float(np.mean([s["grad_norm_actor"] for s in stats["step_log"]]))
        gg = float(np.mean([s["grad_norm_gate"] for s in stats["step_log"]]))
        gp = float(np.mean([s["grad_norm_price"] for s in stats["step_log"]]))
        q_before.append(qb)
        q_after.append(qa)
        grads.append({"actor": ga, "gate": gg, "price": gp})
        seed_used.append(seed)
        for pid, pair, row in triples_:
            day = P1.rollout_day(trainer, pair, b1_cache,
                                 mechanism_hash=mechanism_hash, pair_row=row)
            d = day["G_policy"]
            d0 = baseline[pid]["G_policy"]
            delta = {nm: float(d[nm] - d0[nm]) for nm in NAMES}
            per_day[pid].append(sum(delta.values()))
            for nm in NAMES:
                per_unit[nm][pid].append(delta[nm])
            writer.writerow({
                "experiment": experiment, "checkpoint": ckpt_seed,
                "buffer": buffer_label, "tau": tau, "replicate": rep,
                "seed": seed, "pair_id": pid, "season": day["season"],
                "G_policy": float(sum(d.values())),
                "delta_vs_arm0": float(sum(delta.values())),
                **{f"delta_vs_arm0_{nm}": delta[nm] for nm in NAMES},
                "q_before": qb, "q_after": qa,
                "grad_actor": ga, "grad_gate": gg, "grad_price": gp,
                "batch_size": int(batch["obs"].shape[0]),
                "batch_unique": int(len(np.unique(indices))),
                "batch_duplicates": int(indices.size
                                        - len(np.unique(indices))),
                "batch_indices": ";".join(str(int(i)) for i in indices),
                "buffer_n": buffer_info["n"],
                "buffer_sha": buffer_info["sha256"],
                "optimizer_sha": opt_sha,
                "updates_applied": STEPS,
            })
            fh.flush()
        P1.restore_state(trainer, snap)
    values = [v for pid in per_day for v in per_day[pid]]
    day_means = {pid: float(np.mean(per_day[pid])) for pid in per_day}
    rng = np.random.default_rng(20261108)
    return {
        "experiment": experiment, "buffer": buffer_label, "tau": tau,
        "checkpoint_seed": ckpt_seed,
        "buffer_info": buffer_info,
        "optimizer_sha": opt_sha,
        "batch_seeds": seed_used,
        "fleet_mean": float(np.mean(values)),
        "fleet_ci_day_clustered": bootstrap(list(day_means.values()), rng),
        "fleet_ci_day_x_replicate": bootstrap(values, rng),
        "day_means": day_means,
        "days_better": int(sum(1 for v in day_means.values() if v > 0)),
        "per_unit_mean": {nm: float(np.mean([v for pid in per_unit[nm]
                                             for v in per_unit[nm][pid]]))
                          for nm in NAMES},
        "q_before_mean": float(np.mean(q_before)),
        "q_after_mean": float(np.mean(q_after)),
        "grad_mean": {k: float(np.mean([g[k] for g in grads]))
                      for k in ("actor", "gate", "price")},
        "n_values": len(values),
    }


def bootstrap(values, rng, resamples=10000, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return [float(arr.mean()), float(arr.mean())]
    draws = rng.choice(arr, size=(resamples, arr.size), replace=True)
    means = draws.mean(axis=1)
    return [float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2)))]


def open_matrix_csv(mat: dict):
    """Drop rows of a cell that never completed, then append."""
    keep = int(mat.get("csv_lines_committed", 0))
    lines = (MATRIX_CSV.read_text(encoding="utf-8").splitlines(keepends=True)
             if MATRIX_CSV.exists() else [])
    if keep == 0:
        lines = [",".join(MATRIX_COLUMNS) + "\n"]
        keep = 1
    elif len(lines) > keep:
        lines = lines[:keep]
    MATRIX_CSV.write_text("".join(lines), encoding="utf-8")
    fh = MATRIX_CSV.open("a", newline="", encoding="utf-8")
    return fh, csv.DictWriter(fh, fieldnames=MATRIX_COLUMNS)


def commit_csv_lines(mat: dict) -> None:
    mat["csv_lines_committed"] = len(
        MATRIX_CSV.read_text(encoding="utf-8").splitlines())


def matrix_study(wanted=None) -> dict:
    """Cells already in the evidence are skipped, so a kill costs one cell."""
    triples_, mechanism_hash = triples()
    b1_cache = mod.load_b1_cache()
    evidence = load_evidence().get("matrix", {})
    evidence.setdefault("triples", [t[0] for t in triples_])

    trainer11, bundle11 = load_trainer(CHECKPOINTS[11], triples_[0][1])
    trainer23, bundle23 = load_trainer(CHECKPOINTS[23], triples_[0][1])

    if "buffer_11" not in evidence:
        evidence["buffer_11"] = buffer_fingerprint(trainer11.buffer)
        evidence["buffer_23"] = buffer_fingerprint(trainer23.buffer)
        evidence["buffer_compatibility"] = buffer_compatibility(
            evidence["buffer_11"], evidence["buffer_23"],
            bundle11["manifest"], bundle23["manifest"])
    if not evidence["buffer_compatibility"]["compatible"]:
        save_evidence({**load_evidence(), "matrix": evidence},
                      "BLOCKED_P2_BUFFER_INCOMPATIBLE")
        raise SystemExit("replay buffers are not proven compatible; stopping")

    if "buffer_23_subset" not in evidence:
        idx = stratified_subset_indices(trainer23.buffer, 1272, seed=2026)
        subset = buffer_fingerprint(make_subset(trainer23.buffer, idx))
        evidence["buffer_23_subset"] = {
            **subset, "selection_seed": 2026, "nested_in_full": True,
            "indices": idx}
    idx = evidence["buffer_23_subset"]["indices"]
    buf23_subset = make_subset(trainer23.buffer, idx)

    snap11 = P1.clone_state(trainer11)
    if "arm0_G_policy" not in evidence:
        base = arm0_baseline(trainer11, triples_, b1_cache, mechanism_hash)
        evidence["arm0_G_policy"] = {pid: day["G_policy"]
                                     for pid, day in base.items()}
        evidence["arm0_fleet"] = {
            pid: float(sum(day["G_policy"].values()))
            for pid, day in base.items()}
        save_evidence({**load_evidence(), "matrix": evidence},
                      "PARTIAL_IN_PROGRESS")
    baseline = {pid: {"G_policy": v}
                for pid, v in evidence["arm0_G_policy"].items()}

    cells = [
        ("A1", "seed11_1272", 1.0, trainer11.buffer),
        ("A1", "seed11_1272", 2.5, trainer11.buffer),
        ("A1", "seed11_1272", 4.0, trainer11.buffer),
        ("A2", "seed23_subset_1272", 2.5, buf23_subset),
        ("A3", "seed23_full_5088", 2.5, trainer23.buffer),
    ]
    done = {(c["experiment"], c["buffer"], c["tau"])
            for c in evidence.get("cells", [])}
    picked = [c for c in cells
              if (wanted is None or c[0] in wanted)
              and (c[0], c[1], c[2]) not in done]
    fh, writer = open_matrix_csv(evidence)
    summaries = evidence.get("cells", [])
    try:
        for experiment, label, tau, buf in picked:
            summary = run_cell(trainer11, snap11, buf, tau, 11, experiment,
                               label, baseline, triples_, b1_cache,
                               mechanism_hash, writer, fh)
            fh.flush()
            summaries.append(summary)
            evidence["cells"] = summaries
            commit_csv_lines(evidence)
            save_evidence({**load_evidence(), "matrix": evidence},
                          "PARTIAL_IN_PROGRESS")
            print(f"{experiment} {label} tau {tau}: "
                  f"{summary['fleet_mean']:+.3f} "
                  f"CI {summary['fleet_ci_day_clustered']}", flush=True)
    finally:
        fh.close()
    return evidence


def a4_study() -> dict:
    triples_, mechanism_hash = triples()
    b1_cache = mod.load_b1_cache()
    trainer23, _ = load_trainer(CHECKPOINTS[23], triples_[0][1])
    snap23 = P1.clone_state(trainer23)
    baseline = arm0_baseline(trainer23, triples_, b1_cache, mechanism_hash)
    mat = load_evidence().get("matrix", {})
    fh, writer = open_matrix_csv(mat)
    summaries = []
    try:
        for tau in (1.0, 2.5):
            summary = run_cell(trainer23, snap23, trainer23.buffer, tau, 23,
                               "A4", "seed23_full_5088", baseline, triples_,
                               b1_cache, mechanism_hash, writer, fh)
            summaries.append(summary)
            fh.flush()
            commit_csv_lines(mat)
            print(f"A4 tau {tau}: {summary['fleet_mean']:+.3f} "
                  f"CI {summary['fleet_ci_day_clustered']}", flush=True)
    finally:
        fh.close()
    return {"arm0": {pid: float(sum(day["G_policy"].values()))
                     for pid, day in baseline.items()},
            "cells": summaries}


# ----------------------------------------------------------------------
# Stage 3: critic ranking
# ----------------------------------------------------------------------
def trace_day(trainer, pair, b1_cache, mechanism_hash, pair_row) -> list:
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    trace = []
    for block in range(24):
        acts, gates, probs = {}, {}, {}
        for nm in names:
            d = trainer.select_action(obs[nm], nm, train=False)
            acts[nm] = np.asarray(d["quotes"], dtype=float)
            gates[nm] = np.asarray(d["gates"], dtype=float)
            probs[nm] = np.asarray(d["probs"], dtype=float)
        trace.append({
            "obs": {k: v.copy() for k, v in obs.items()},
            "acts": {k: v.copy() for k, v in acts.items()},
            "gates": gates, "probs": probs,
            "soc": {k: float(v) for k, v in env.prev_soc.items()},
        })
        obs, _, done, info = env.step(acts)
        result = env._last_result
        if result is None:
            raise RuntimeError(f"clearing failed at block {block}")
        # Committed energy per unit and leg, so a sweep state can be chosen
        # where a quote actually clears; elsewhere the quote has no effect.
        n_commit = min(4, env.T - block * 4)
        commit = {}
        for nm in names:
            sched = (result.get("schedules") or {}).get(nm) or {}
            commit[nm] = tuple(
                float(np.sum(np.asarray(sched.get(k, np.zeros(n_commit)),
                                        dtype=float)[:n_commit]) * 0.25)
                for k in ("p_ch", "p_dis"))
        trace[-1]["commit"] = commit
        if done:
            break
    return trace


def rollout_override(trainer, pair, b1_cache, mechanism_hash, pair_row,
                     block_b, unit, leg, value) -> dict:
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    payoff = {nm: 0.0 for nm in names}
    fleet = 0.0
    soc_before = None
    accepted = 0.0
    failed = False
    for block in range(24):
        acts = {}
        for nm in names:
            d = trainer.select_action(obs[nm], nm, train=False)
            acts[nm] = np.asarray(d["quotes"], dtype=float)
        if block == block_b:
            acts[unit] = acts[unit].copy()
            acts[unit][leg] = float(value)
            soc_before = float(env.prev_soc.get(unit, float("nan")))
        obs, _, done, info = env.step(acts)
        result = env._last_result
        if result is None:
            failed = True
            break
        if block == block_b:
            sched = (result.get("schedules") or {}).get(unit) or {}
            n_commit = min(4, env.T - block * 4)
            key = "p_ch" if leg == 0 else "p_dis"
            if key in sched:
                accepted = float(np.sum(
                    np.asarray(sched[key], dtype=float)[:n_commit]) * 0.25)
        if block >= block_b:
            raw = info.get("raw_profit", {})
            for nm in names:
                v = float(raw.get(nm, 0.0))
                payoff[nm] += v
                fleet += v
        if done:
            break
    return {"payoff": payoff, "fleet": fleet, "soc_before": soc_before,
            "accept_mwh": accepted, "clearing_failed": failed}


def q_at_state(trainer, obs_dict, action_dict, unit, leg) -> dict:
    names = trainer.agent_names
    obs = torch.tensor(np.stack([obs_dict[nm] for nm in names]),
                       dtype=torch.float32).unsqueeze(0)
    act = torch.tensor(np.stack([action_dict[nm] for nm in names])
                       .reshape(1, -1), dtype=torch.float32)
    act.requires_grad_(True)
    q1, q2 = trainer.critics[unit](obs.reshape(1, -1), act)
    q_min = torch.min(q1, q2)
    grad, = torch.autograd.grad(q_min.sum(), act)
    i = names.index(unit)
    j = 2 * i + int(leg)
    return {"q1": float(q1.item()), "q2": float(q2.item()),
            "q_min": float(q_min.item()),
            "twin_gap": abs(float(q1.item()) - float(q2.item())),
            "dq_dquote": float(grad[0, j])}


def spearman(x, y) -> float | None:
    from scipy.stats import spearmanr
    value = spearmanr(x, y).correlation
    return None if value is None or math.isnan(value) else float(value)


def make_state(pair_id, pair, row, trace, block, unit, leg,
               switched=False) -> dict:
    return {
        "pair_id": pair_id, "pair": pair, "row": row,
        "block": block, "unit": unit, "leg": leg,
        "leg_switched": switched,
        "gate_open": bool(trace[block]["gates"][unit][leg] > 0),
        "obs": trace[block]["obs"],
        "acts": trace[block]["acts"],
        "soc": trace[block]["soc"][unit],
    }


def fixed_states(trace, pair_id, pair, row, per_leg=2) -> list:
    states = []
    for block in RANK_BLOCKS:
        if block >= len(trace):
            continue
        for unit in RANK_UNITS:
            leg = 0 if block in RANK_CHARGE_BLOCKS else 1
            switched = False
            if trace[block]["gates"][unit][leg] <= 0:
                other = 1 - leg
                if trace[block]["gates"][unit][other] > 0:
                    leg, switched = other, True
            states.append(make_state(pair_id, pair, row, trace, block, unit,
                                     leg, switched))
    return states


def activity_states(trace, pair_id, pair, row, per_leg=2) -> list:
    """Blocks where the policy's own quote clears energy.

    A quote sweep at a block whose quote never clears leaves the clearing
    unchanged, so the payoff is constant and no ordering can be measured
    there. Selection by realized activity is outcome-based and exploratory;
    it is reported apart from the fixed pre-registered blocks.
    """
    states = []
    for unit in RANK_UNITS:
        for leg in (0, 1):
            ranked = sorted((blk for blk in range(1, min(23, len(trace)))
                             if trace[blk]["commit"][unit][leg] > ACTIVITY_MWH
                             and trace[blk]["gates"][unit][leg] > 0),
                            key=lambda b: -trace[b]["commit"][unit][leg])
            for block in sorted(ranked[:per_leg]):
                states.append(make_state(pair_id, pair, row, trace, block,
                                         unit, leg))
    return states


def sweep_states(trainer, states, csv_path, b1_cache, mechanism_hash) -> dict:
    fh = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=RANK_COLUMNS)
    writer.writeheader()
    fh.flush()
    state_records = []
    try:
        for si, st in enumerate(states):
            grid = CHARGE_GRID if st["leg"] == 0 else DISCHARGE_GRID
            points = []
            for quote in grid:
                acts = {k: v.copy() for k, v in st["acts"].items()}
                acts[st["unit"]][st["leg"]] = float(quote)
                q = q_at_state(trainer, st["obs"], acts, st["unit"],
                               st["leg"])
                roll = rollout_override(
                    trainer, st["pair"], b1_cache, mechanism_hash, st["row"],
                    st["block"], st["unit"], st["leg"], quote)
                points.append({
                    "quote": float(quote), "q1": q["q1"], "q2": q["q2"],
                    "q_min": q["q_min"], "twin_gap": q["twin_gap"],
                    "dq_dquote": q["dq_dquote"],
                    "payoff_unit": roll["payoff"][st["unit"]],
                    "payoff_fleet": roll["fleet"],
                    "accept_mwh": roll["accept_mwh"],
                    "clearing_failed": roll["clearing_failed"],
                    "soc_before": roll["soc_before"],
                })
                writer.writerow({
                    "state_id": si, "pair_id": st["pair_id"],
                    "block": st["block"], "unit": st["unit"],
                    "leg": ("charge" if st["leg"] == 0 else "discharge"),
                    "gate_open": st["gate_open"],
                    "leg_switched": st["leg_switched"],
                    "quote": float(quote),
                    "q1": q["q1"], "q2": q["q2"], "q_min": q["q_min"],
                    "twin_gap": q["twin_gap"], "dq_dquote": q["dq_dquote"],
                    "payoff_remaining_unit": roll["payoff"][st["unit"]],
                    "payoff_remaining_fleet": roll["fleet"],
                    "accept_mwh": roll["accept_mwh"],
                    "soc_before": roll["soc_before"],
                    "clearing_failed": roll["clearing_failed"],
                })
                fh.flush()
            state_records.append({
                **{k: st[k] for k in ("pair_id", "block", "unit",
                                      "leg_switched", "gate_open", "soc")},
                "leg": ("charge" if st["leg"] == 0 else "discharge"),
                "points": points})
            print(f"state {si}/{len(states)} {st['pair_id']} block "
                  f"{st['block']} {st['unit']} leg {st['leg']}", flush=True)
    finally:
        fh.close()
    return summarize_rank(state_records)


def rank_study(selector=fixed_states, csv_path=RANK_CSV,
               per_leg=2) -> dict:
    triples_, mechanism_hash = triples()
    by_id = {pid: (pair, row) for pid, pair, row in triples_}
    b1_cache = mod.load_b1_cache()
    trainer, _ = load_trainer(CHECKPOINTS[23], triples_[0][1])

    states = []
    selected = {}
    for pid in WINTER_PAIRS:
        pair, row = by_id[pid]
        trace = trace_day(trainer, pair, b1_cache, mechanism_hash, row)
        day_states = selector(trace, pid, pair, row, per_leg=per_leg)
        selected[pid] = [[s["block"], s["unit"],
                          "charge" if s["leg"] == 0 else "discharge"]
                         for s in day_states]
        states.extend(day_states)

    summary = sweep_states(trainer, states, csv_path, b1_cache,
                           mechanism_hash)
    summary.update({"checkpoint_seed": 23,
                    "pairs": list(WINTER_PAIRS),
                    "blocks": list(RANK_BLOCKS),
                    "units": list(RANK_UNITS),
                    "selected_states": selected})
    return summary


def summarize_rank(records: list) -> dict:
    open_states = [r for r in records if r["gate_open"]]
    closed = [r for r in records if not r["gate_open"]]
    per_state = []
    pairwise_ok = pairwise_n = 0
    grad_ok = grad_n = 0
    top1_ok = top1_strict_ok = tested = decisive = 0
    flat = 0
    twin_gaps = []
    payoff_range = []
    acceptance = []
    for r in open_states:
        clean = [p for p in r["points"] if not p["clearing_failed"]]
        q = [p["q_min"] for p in clean]
        pay = [p["payoff_unit"] for p in clean]
        rho = spearman(q, pay) if len(clean) >= 3 else None
        if rho is None:
            flat += 1
            per_state.append({"pair_id": r["pair_id"], "block": r["block"],
                              "unit": r["unit"], "leg": r["leg"],
                              "spearman": None, "top1": None,
                              "pairwise_accuracy": None, "n_points": len(clean),
                              "accept_mwh": [p["accept_mwh"] for p in clean]})
            continue
        tested += 1
        top_q = max(range(len(clean)), key=lambda i: q[i])
        top_p = max(range(len(clean)), key=lambda i: pay[i])
        prange = max(pay) - min(pay)
        top1_strict = (clean[top_q]["quote"] == clean[top_p]["quote"])
        top1_strict_ok += int(top1_strict)
        if prange > TOP1_FLAT_TOL_CNY:
            decisive += 1
            top1 = pay[top_q] >= max(pay) - TOP1_FLAT_TOL_CNY
            top1_ok += int(top1)
        else:
            top1 = None
        ok = tot = 0
        for i in range(len(clean)):
            for j in range(i + 1, len(clean)):
                dp = pay[i] - pay[j]
                if abs(dp) <= TIE:
                    continue
                tot += 1
                ok += int((q[i] - q[j]) * dp > 0)
        pairwise_ok += ok
        pairwise_n += tot
        for i in range(len(clean) - 1):
            dp = pay[i + 1] - pay[i]
            if abs(dp) <= TIE:
                continue
            midpoint = 0.5 * (clean[i]["dq_dquote"]
                              + clean[i + 1]["dq_dquote"])
            grad_n += 1
            grad_ok += int(midpoint * dp > 0)
        twin_gaps.extend(p["twin_gap"] for p in clean)
        payoff_range.append(max(pay) - min(pay))
        accept = [p["accept_mwh"] for p in clean]
        acceptance.append({"state": (r["pair_id"], r["block"], r["unit"],
                                     r["leg"]),
                           "accept_mwh": accept,
                           "quote": [p["quote"] for p in clean],
                           "policy_quote_accepts": accept[0] > 0})
        per_state.append({
            "pair_id": r["pair_id"], "block": r["block"], "unit": r["unit"],
            "leg": r["leg"],
            "spearman": rho,
            "top1": (bool(top1) if top1 is not None else None),
            "top1_strict": bool(top1_strict),
            "payoff_range": float(prange),
            "pairwise_accuracy": (ok / tot) if tot else None,
            "q": q, "payoff": pay, "quotes": [p["quote"] for p in clean],
            "accept_mwh": accept, "n_points": len(clean)})
    rho_values = [s["spearman"] for s in per_state if s["spearman"] is not None]
    median_rho = float(np.median(rho_values)) if rho_values else None
    top1_rate = (top1_ok / decisive) if decisive else None
    top1_strict_rate = (top1_strict_ok / tested) if tested else None
    if median_rho is None:
        verdict = "NOT_TESTED"
    elif median_rho >= 0.5 and (top1_rate or 0) >= 0.5:
        verdict = "valid"
    elif median_rho <= 0.1 or (top1_rate or 0) <= 0.25:
        verdict = "invalid"
    else:
        verdict = "ambiguous"
    return {
        "states_total": len(records),
        "states_open": len(open_states),
        "states_closed_gate": len(closed),
        "closed_gate_states": [{k: r[k] for k in ("pair_id", "block",
                                                  "unit", "leg")}
                               for r in closed],
        "states_with_flat_payoff": flat,
        "payoff_range_median": (float(np.median(payoff_range))
                                if payoff_range else None),
        "no_signal": (len(payoff_range) > 0
                      and np.mean([r < 1.0 for r in payoff_range]) >= 0.75),
        "median_spearman": median_rho,
        "top1_agreement": top1_rate,
        "top1_agreement_strict": top1_strict_rate,
        "top1_decisive_states": decisive,
        "top1_flat_tol_cny": TOP1_FLAT_TOL_CNY,
        "pairwise_accuracy": (pairwise_ok / pairwise_n) if pairwise_n else None,
        "twin_gap_mean": float(np.mean(twin_gaps)) if twin_gaps else None,
        "gradient_direction_agreement": ((grad_ok / grad_n)
                                         if grad_n else None),
        "acceptance": acceptance,
        "per_state": per_state,
        "ranking_verdict": verdict,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    evidence = load_evidence()
    evidence.setdefault("protocol", "P2_CONTROLLED_DIAGNOSTICS_V2")
    evidence.setdefault("environment", versions())
    evidence.setdefault("started_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if stage == "temp":
        evidence["temp"] = temp_study()
        save_evidence(evidence, "PARTIAL_IN_PROGRESS")
        print(json.dumps({"classification":
                          evidence["temp"]["classification"]}, indent=2))
    elif stage == "matrix":
        wanted = set(sys.argv[2:]) or None
        evidence["matrix"] = matrix_study(wanted)
        save_evidence(evidence, "PARTIAL_IN_PROGRESS")
    elif stage == "a4":
        evidence["a4"] = a4_study()
        save_evidence(evidence, "PARTIAL_IN_PROGRESS")
    elif stage in ("rank", "rankact"):
        key = "rank" if stage == "rank" else "rank_activity"
        selector = fixed_states if stage == "rank" else activity_states
        path = RANK_CSV if stage == "rank" else RANK_ACTIVITY_CSV
        evidence[key] = rank_study(selector, path)
        save_evidence(evidence, "PARTIAL_IN_PROGRESS")
        print(json.dumps({k: evidence[key][k] for k in
                          ("median_spearman", "top1_agreement",
                           "pairwise_accuracy", "ranking_verdict",
                           "no_signal")},
                         indent=2))
    else:
        print("usage: diagnose_p2.py temp|matrix|a4|rank|rankact", flush=True)
        return 2
    evidence["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_evidence(evidence, "PARTIAL_IN_PROGRESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
