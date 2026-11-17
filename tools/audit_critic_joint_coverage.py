"""Read-only joint state-action coverage audit for the B/C2 gate run.

The question is whether the training replay contains transitions close to the
validation winter states where the critic mis-ranks the long-horizon quote.
Marginal quote histograms are not enough, so the audit measures:

* nearest joint state distance, with explicit per-feature scales;
* nearest joint action distance among the nearest states;
* conditional coverage of the target SOC, block and quote at the same time;
* whether high discharge quotes appear near the material state or only
  elsewhere in the day/state distribution.

No training, no checkpoint write, no H2 and no legacy test.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

import rl_b_c2_gate as mod
import train_b_c2_gate as tb
from diagnose_critic_long_horizon import (CONTROL, DATA, RUN, WINTER,
                                          collect_state, load_trainer, pairs)

CKPTS = (2, 10)
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")
LEGS = ("charge", "discharge")
HIGHS = np.array([150.0, 300.0])

# One block varies by day: union of the phase-15 calibration blocks and the
# winter-loss blocks, so early, middle and late decision times are all present.
TARGET_BLOCKS = {
    "2022-12-13|2025-12-11": (2, 3, 11, 17, 20),
    "2022-12-30|2025-12-31": (10, 11, 14, 20),
    "2022-12-31|2025-12-28": (0, 17, 23),
}
CONTROL_BLOCKS = (11,)
# Both controls are validation days, not training days: 2022-08-14 is in the
# training list and would trivially match itself.
AUDIT_CONTROL = ("2022-07-23|2025-07-13", "2022-10-26|2025-10-24")

# `avg_other_dev`, `all_dev_std` and the three forecaster features are
# computed per agent (the fleet excluding self and a per-agent EMA), so they
# stay in the per-unit block. Only the four realized-price/system features are
# identical for all four agents and are stored once.
AGENT_IDX = (0, 1, 2, 7, 8, 9, 10, 11)
SHARED_IDX = (3, 4, 5, 6)
AGENT_FEATURES = ("load_feat", "re_feat", "soc", "avg_other_dev",
                  "all_dev_std", "ema_dev", "price_trend", "pred_lmp_norm")
SHARED_NAMES = ("last_lmp_norm", "block_pos", "avg_lmp_hist_norm", "slr_norm")
STATE_NAMES = tuple(f"{u}.{f}" for u in UNITS for f in AGENT_FEATURES) \
    + SHARED_NAMES
STATE_DIM = len(AGENT_FEATURES) * len(UNITS) + len(SHARED_NAMES)
PER_UNIT = len(AGENT_FEATURES)

# Fixed, interpretable scales. The observation already normalizes load/RE to
# [0,1], the two opponent features are already normalized deviations and
# every price feature is divided by the frozen reference price. SOC is the
# only raw [0,1] scale here, so one quarter of its range is the unit.
# `block_pos` is normalized over 23 gaps, so 1/24 is one block.
FIXED_SCALES = np.array(
    [1.0, 1.0, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0] * len(UNITS)
    + [1.0, 1.0 / 24.0, 1.0, 1.0]
)


def read_episodes():
    rows = list(csv.DictReader((RUN / "episodes.csv").open(encoding="utf-8")))
    return [r["pair_id"] for r in rows]


def load_buffer(cp):
    step = json.loads((RUN / "checkpoint_index.json").read_text(
        encoding="utf-8"))[str(cp)]["checkpoint_path"]
    path = Path(step) / "buffer" / "replay.npz"
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}, path


def buffer_state(obs):
    """(N,4,12) obs -> (N,36) joint state in the audit's feature order."""
    per_unit = np.asarray(obs[:, :, AGENT_IDX], dtype=np.float64)
    shared = np.asarray(obs[:, 0, SHARED_IDX], dtype=np.float64)
    return np.concatenate(
        [per_unit.reshape(len(obs), -1), shared], axis=1)


def target_state(obs):
    names = list(obs)
    shared = np.asarray(obs[names[0]][list(SHARED_IDX)], dtype=np.float64)
    for nm in names[1:]:
        if not np.allclose(obs[nm][list(SHARED_IDX)], shared, atol=1e-8):
            raise RuntimeError("shared observation features differ by agent")
    per_unit = np.concatenate(
        [np.asarray(obs[nm][list(AGENT_IDX)], dtype=np.float64)
         for nm in names])
    return np.concatenate([per_unit, shared])


def quote_action(actions):
    """(...,4,2) CNY -> (...,8) quotes normalized by the leg upper bound."""
    return np.asarray(actions, dtype=np.float64) / HIGHS


def scaled_distance(a, b, scales):
    diff = (np.asarray(a) - np.asarray(b)) / scales
    return float(np.sqrt(np.mean(diff ** 2)))


def nearest_matrix(query, bank, scales):
    """Euclidean distance in fixed-scale units, normalized to per-dim RMS."""
    q = np.asarray(query, dtype=np.float64)
    b = np.asarray(bank, dtype=np.float64)
    out = np.empty((len(q), len(b)), dtype=np.float64)
    for i, row in enumerate(q):
        d = (b - row) / scales
        out[i] = np.sqrt(np.mean(d ** 2, axis=1))
    return out


def nn_distribution(states, scales, sample=600, seed=0, episodes_at=None):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(states), size=min(sample, len(states)), replace=False)
    q = states[idx]
    d = nearest_matrix(q, states, scales)
    for j, i in enumerate(idx):
        d[j, i] = np.inf
        if episodes_at is not None:
            same_episode = episodes_at == episodes_at[i]
            d[j, same_episode] = np.inf
        if not np.isfinite(d[j]).any():
            d[j] = 0.0
    return np.min(d, axis=1)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {q: float(np.quantile(values, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}


def main(argv=None) -> int:
    pair_map, _ = pairs()
    episode_ids = read_episodes()
    out_rows = []
    states_out = {}
    summary = {"checkpoints": {}, "targets": {}, "normalization": {
        "state_feature_names": list(STATE_NAMES),
        "fixed_scales": FIXED_SCALES.tolist(),
        "distance": "Euclidean, divided by sqrt(dim); one unit = the feature "
                    "scale listed in fixed_scales",
        "z_score": "also reported with the checkpoint buffer's per-feature "
                   "mean and standard deviation",
        "action": "quote / leg upper bound (charge 150, discharge 300); "
                  "gate is 0/1",
    }}

    for cp in CKPTS:
        arrays, path = load_buffer(cp)
        obs = np.asarray(arrays["obs"], dtype=np.float64)
        actions = np.asarray(arrays["actions"], dtype=np.float64)
        gates = (np.asarray(arrays["gates"], dtype=np.float64) > 0).astype(float)
        n = len(obs)
        n_ep = (n + 23) // 24
        eps = np.repeat(np.arange(n_ep), 24)[:n]
        dates = [episode_ids[i] if i < len(episode_ids) else "?"
                 for i in eps]
        states = buffer_state(obs)
        z_mean = states.mean(axis=0)
        z_std = states.std(axis=0)
        z_std[z_std < 1e-9] = 1.0
        z_states = (states - z_mean) / z_std
        # block_pos is the second shared feature, after the per-unit block.
        blocks = np.rint(states[:, PER_UNIT * len(UNITS) + 1]
                         * (24 - 1)).astype(int)
        nn_fixed = nn_distribution(states, FIXED_SCALES,
                                   episodes_at=eps)
        nn_z = nn_distribution(z_states, np.ones(states.shape[1]),
                               episodes_at=eps)
        summary["checkpoints"][str(cp)] = {
            "buffer_path": str(path), "transitions": int(n),
            "episodes": int(n_ep),
            "dates": int(len(set(dates))),
            "nn_fixed_quantiles": quantiles(nn_fixed),
            "nn_z_quantiles": quantiles(nn_z),
        }

        # Rebuild the trainer and walk to each target state.
        template = tb.envmod.build_env_for_pair(
            pair_map[WINTER[0]], differential_reward=False,
            terminal_soc_target=0.5)
        trainer = load_trainer(cp, template, mod.ExperimentConfig(
            **json.loads((RUN / "run_manifest.json").read_text(
                encoding="utf-8"))["experiment"]))

        for pid, blocks_list in (
                [(p, TARGET_BLOCKS[p]) for p in WINTER]
                + [(p, CONTROL_BLOCKS) for p in AUDIT_CONTROL]):
            for block in blocks_list:
                env, tob, baseline, _, soc, gate_p, gate_det, _, _ = \
                    collect_state(trainer, pair_map[pid], block)
                ts = target_state(tob)
                ta = quote_action(np.stack([baseline[nm] for nm in UNITS]))
                tg = np.stack([(np.asarray(gate_det[nm]) > 0).astype(float)
                               for nm in UNITS])
                d_fixed = np.sqrt(np.mean(
                    ((states - ts) / FIXED_SCALES) ** 2, axis=1))
                d_z = np.sqrt(np.mean(
                    ((z_states - (ts - z_mean) / z_std) / 1.0) ** 2, axis=1))
                order = np.argsort(d_fixed)
                nn = int(order[0])
                # Neighbour set excludes the same episode as the nearest
                # transition, so a nearby block of the same day is not counted
                # as independent coverage.
                other = np.where(eps != eps[nn])[0]
                nn_other = int(other[np.argmin(d_fixed[other])])
                k50 = order[:50]
                k200 = order[:200]
                sa = quote_action(actions)
                dq = np.sqrt(np.mean((sa - ta) ** 2, axis=(1, 2)))
                full_buf = np.concatenate(
                    [sa.reshape(len(sa), -1), gates.reshape(len(gates), -1)],
                    axis=1)
                full_tgt = np.concatenate([ta.reshape(-1), tg.reshape(-1)])
                dfull = np.sqrt(np.mean((full_buf - full_tgt) ** 2, axis=1))

                for nm in ("ESS14", "ESS27"):
                    j = UNITS.index(nm)
                    for leg in LEGS:
                        li = LEGS.index(leg)
                        high = HIGHS[li]
                        q_t = float(ta[j, li] * high)
                        soc_t = float(ts[j * PER_UNIT + 2])
                        time_ok = np.abs(blocks - block) <= 1
                        soc_ok = np.abs(states[:, j * PER_UNIT + 2] - soc_t) <= 0.10
                        q_ok = np.abs(sa[:, j, li] * high - q_t) <= 0.10 * high
                        high_ok = sa[:, j, li] * high >= 0.86 * high
                        joint = time_ok & soc_ok & q_ok
                        row = {
                            "checkpoint": cp, "pair_id": pid, "block": block,
                            "unit": nm, "leg": leg,
                            "actor_quote": q_t, "target_soc": soc_t,
                            "state_nn_fixed": float(d_fixed[nn]),
                            "state_nn_fixed_other_episode": float(
                                d_fixed[nn_other]),
                            "state_nn_fixed_percentile": float(
                                (nn_fixed < d_fixed[nn]).mean()),
                            "state_nn_z": float(d_z[nn]),
                            "state_nn_z_percentile": float(
                                (nn_z < d_z[nn]).mean()),
                            "nearest_index": nn,
                            "nearest_date": dates[nn],
                            "nearest_block": int(blocks[nn]),
                            "nearest_other_episode_date": dates[nn_other],
                            "nearest_other_episode_block": int(
                                blocks[nn_other]),
                            "nn_soc_gap": float(
                                states[nn, j * PER_UNIT + 2] - soc_t),
                            "nn_quote_gap": float(
                                sa[nn, j, li] * high - q_t),
                            "action_nn_fixed_all": float(np.min(dq)),
                            "action_nn_fixed_k50": float(np.min(dq[k50])),
                            "action_nn_fixed_k200": float(np.min(dq[k200])),
                            "action_full_nn_k50": float(np.min(dfull[k50])),
                            "count_time": int(time_ok.sum()),
                            "count_time_soc": int((time_ok & soc_ok).sum()),
                            "count_time_soc_quote": int(joint.sum()),
                            "count_high_quote_all": int(high_ok.sum()),
                            "count_high_quote_time": int(
                                (high_ok & time_ok).sum()),
                            "count_high_quote_time_soc": int(
                                (high_ok & time_ok & soc_ok).sum()),
                            "count_high_quote_k50": int(high_ok[k50].sum()),
                            "count_soc_quote_any_time": int(
                                (soc_ok & q_ok).sum()),
                        }
                        out_rows.append(row)
                        states_out[
                            f"cp{cp}|{pid}|b{block}|{nm}|{leg}"] = {
                            "state": ts.tolist(),
                            "action_quotes": (ta * HIGHS).tolist(),
                            "gate_deterministic": tg.tolist(),
                            "soc": soc,
                            **{k: row[k] for k in (
                                "state_nn_fixed", "state_nn_fixed_percentile",
                                "nearest_date", "nearest_block",
                                "nearest_other_episode_date",
                                "nearest_other_episode_block",
                                "action_nn_fixed_k50", "count_time",
                                "count_time_soc", "count_time_soc_quote",
                                "count_high_quote_all",
                                "count_high_quote_time",
                                "count_high_quote_time_soc")},
                        }
                print(f"cp{cp} {pid} b{block}: nn={d_fixed[nn]:.3f} "
                      f"({dates[nn]} b{blocks[nn]})", flush=True)

    # Aggregate: the material winter states are the target rows on the three
    # winter days; the control rows are the comparison.
    rows = [r for r in out_rows if r["pair_id"] in WINTER]
    ctrl = [r for r in out_rows if r["pair_id"] in AUDIT_CONTROL]
    for label, subset in (("winter", rows), ("control", ctrl)):
        summary["targets"][label] = {
            "rows": len(subset),
            "mean_state_nn_fixed": float(np.mean(
                [r["state_nn_fixed"] for r in subset])),
            "median_state_nn_fixed": float(np.median(
                [r["state_nn_fixed"] for r in subset])),
            "rows_within_nn_median": int(sum(
                r["state_nn_fixed"] <= summary["checkpoints"][
                    str(r["checkpoint"])]["nn_fixed_quantiles"][0.5]
                for r in subset)),
            "mean_action_nn_k50": float(np.mean(
                [r["action_nn_fixed_k50"] for r in subset])),
            "rows_with_time_soc_quote": int(sum(
                r["count_time_soc_quote"] > 0 for r in subset)),
            "rows_with_high_quote_near": int(sum(
                r["count_high_quote_k50"] > 0 for r in subset)),
            "total_high_quote_all": int(sum(
                r["count_high_quote_all"] for r in subset)),
            "total_high_quote_time": int(sum(
                r["count_high_quote_time"] for r in subset)),
            "total_high_quote_time_soc": int(sum(
                r["count_high_quote_time_soc"] for r in subset)),
        }
    # Discharge rows only, because the winter loss is on the discharge leg.
    dis = [r for r in rows if r["leg"] == "discharge"]
    summary["winter_discharge"] = {
        "rows": len(dis),
        "mean_state_nn_fixed": float(np.mean(
            [r["state_nn_fixed"] for r in dis])),
        "mean_action_nn_k50": float(np.mean(
            [r["action_nn_fixed_k50"] for r in dis])),
        "rows_with_high_quote_k50": int(sum(
            r["count_high_quote_k50"] > 0 for r in dis)),
        "rows_with_time_soc_quote": int(sum(
            r["count_time_soc_quote"] > 0 for r in dis)),
    }

    out_csv = DATA / "critic_joint_coverage_targets.csv"
    fieldnames = list(out_rows[0])
    for row in out_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    (DATA / "critic_joint_coverage_states.json").write_text(
        json.dumps(states_out, indent=2), encoding="utf-8")
    (DATA / "critic_joint_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_csv} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
