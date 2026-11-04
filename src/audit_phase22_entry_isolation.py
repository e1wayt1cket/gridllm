"""P22: training-entry isolation, replay separation and OBS causality.

Read-only with respect to historical artifacts. One fresh training episode is
written to a temporary run root. The audit hook records every relevant file
open; the formal entry must not open the H2 holdout, the legacy test index or
the old near split. The OBS test feeds two price curves that differ only after
the committed lookahead and requires identical observations, with a positive
control that changes the current block's load forecast.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import grid
import train_da_gate as entry
import v2_cy_jn_env as envmod

DATA = Path("data/processed/v2_cy_jn_2025")
OUT = Path("reports/v2_cy_jn_2025/formal_training_ready/"
           "phase22_entry_isolation.json")
RUN_ID = "PHASE22_ENTRY_ISOLATION"
TRAIN_DAY = "2022-08-14"
OBS_PAIR = "2022-12-27|2025-12-25"
RL_NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")

FORBIDDEN = (
    "holdout_pairs.csv",
    "holdout_manifest.json",
    "new_holdout_manifest.json",
    "holdout_status.json",
    "holdout_exposure_ledger.csv",
    "common_test_index.csv",
    "LEGACY_SEEN_TEST",
)
OPENED: list[str] = []
ACTIVE = False


def _audit_hook(event: str, args) -> None:
    global ACTIVE
    if not ACTIVE or event != "open" or not args:
        return
    target = args[0]
    if not isinstance(target, (str, bytes)):
        return
    path = os.fsdecode(target).replace("\\", "/")
    if any(token in path for token in
           ("data/processed", "runs/", "snapshots/", "policies/",
            "holdout", "split.json")):
        OPENED.append(path)


def forbidden_hits() -> list[str]:
    hits = []
    for path in OPENED:
        lowered = path.lower()
        for token in FORBIDDEN:
            if token.lower() in lowered:
                hits.append(f"{token}: {path}")
        if path.endswith("/near/split.json"):
            hits.append(f"near_split: {path}")
        if path.endswith("/data/processed/v2_cy_jn_2025/split.json"):
            hits.append(f"full96_split: {path}")
    return hits


def run_entry(root: Path) -> dict:
    global ACTIVE
    old_runs, old_snaps = entry.RUNS, entry.SNAPSHOTS
    old_evaluate = entry.evaluate
    replay_checks = []

    def guarded_evaluate(trainer, pairs, *args, **kwargs):
        before = len(trainer.buffer)
        out = old_evaluate(trainer, pairs, *args, **kwargs)
        after = len(trainer.buffer)
        replay_checks.append({
            "before": before, "after": after,
            "no_replay_write": before == after,
            "validation_days": len(pairs),
        })
        return out

    entry.RUNS = root / "runs"
    entry.SNAPSHOTS = root / "snapshots"
    entry.evaluate = guarded_evaluate
    OPENED.clear()
    ACTIVE = True
    try:
        rc = entry.main([
            "--split", "common", "--train-days", TRAIN_DAY,
            "--exposures", "1", "--max-episodes", "1", "--seed", "11",
            "--run-id", RUN_ID, "--early-stop-patience", "0",
        ])
    finally:
        ACTIVE = False
        entry.RUNS, entry.SNAPSHOTS = old_runs, old_snaps
        entry.evaluate = old_evaluate
    manifest = json.loads(
        (root / "runs" / RUN_ID / "run_manifest.json").read_text(
            encoding="utf-8"))
    return {
        "return_code": rc,
        "manifest": manifest,
        "replay_checks": replay_checks,
        "opened_relevant": sorted(set(OPENED)),
        "forbidden_hits": forbidden_hits(),
    }


def _price(pair: dict) -> np.ndarray:
    curves = np.load(DATA / "price_curves.npz")
    return np.asarray(curves[f"south/{pair['price_date']}"], dtype=float)


def obs_causality(pair_map) -> dict:
    pair = pair_map[OBS_PAIR]
    price = _price(pair)
    holder = {"curve": price.copy()}
    original = grid.day_ahead_price_china

    def fake_price(T=96, agents=None, config=None, t_start=0, day_id=None):
        curve = np.asarray(holder["curve"], dtype=float)
        return curve[t_start:t_start + T].copy()

    actions = {nm: np.array([50.0, 100.0]) for nm in RL_NAMES}
    grid.day_ahead_price_china = fake_price
    try:
        env_a = envmod.build_env_for_pair(
            pair, differential_reward=False, terminal_soc_target=0.5)
        obs_a0 = env_a.reset()
        for _ in range(3):
            obs_a, _, _, _ = env_a.step(actions)
        future = price.copy()
        future[24:] += 50.0
        holder["curve"] = future
        env_b = envmod.build_env_for_pair(
            pair, differential_reward=False, terminal_soc_target=0.5)
        env_b.reset()
        for _ in range(3):
            obs_b, _, _, _ = env_b.step(actions)
        same = {
            nm: bool(np.allclose(obs_a[nm], obs_b[nm], atol=0.0))
            for nm in RL_NAMES}
        # Positive control: the current block's load forecast is an input.
        holder["curve"] = price.copy()
        env_c = envmod.build_env_for_pair(
            pair, differential_reward=False, terminal_soc_target=0.5)
        agent = next(a for a in env_c.all_agents
                     if a.participant_type != "storage"
                     and float(np.mean(a.load_forecast[:4])) > 0.0)
        forecast = np.asarray(agent.load_forecast, dtype=float).copy()
        forecast[:4] *= 1.5
        agent.load_forecast = forecast
        obs_c = env_c.reset()
        positive = {
            nm: bool(not np.allclose(obs_a0[nm], obs_c[nm], atol=0.0))
            for nm in RL_NAMES}
        return {
            "pair_id": f"{pair['load_date']}|{pair['price_date']}",
            "future_price_only_identical": same,
            "future_price_all_identical": all(same.values()),
            "positive_control_changed": positive,
            "positive_control_any_changed": any(positive.values()),
            "obs_dim": int(np.asarray(obs_a[RL_NAMES[0]]).size),
        }
    finally:
        grid.day_ahead_price_china = original


def main() -> int:
    sys.addaudithook(_audit_hook)
    from diagnose_critic_long_horizon import pairs
    pair_map, _ = pairs()
    with tempfile.TemporaryDirectory(prefix="makerb_entry_") as tmp:
        entry_result = run_entry(Path(tmp))
    obs = obs_causality(pair_map)
    manifest = entry_result["manifest"]
    checks = {
        "entry_return_zero": entry_result["return_code"] == 0,
        "common_split": manifest.get("split_name") == "common",
        "train_subset_one": len(manifest.get("train_subset", [])) == 1,
        "validation_11": len(manifest.get("validation_pairs", [])) == 11,
        "test_11": len(manifest.get("test_pairs", [])) == 11,
        "holiday_5": len(manifest.get("holiday_pairs", [])) == 5,
        "legacy_16": len(manifest.get("legacy_seen_test_pairs", [])) == 16,
        "no_h2": manifest.get("h2_holdout_read") is False,
        "no_test_eval": manifest.get("test_evaluation") is None,
        "no_legacy_run": manifest.get("legacy_seen_test_run") is False,
        "validation_not_in_replay": bool(
            entry_result["replay_checks"]
            and all(c["no_replay_write"]
                    for c in entry_result["replay_checks"])),
        "no_forbidden_open": not entry_result["forbidden_hits"],
        "no_near_split_open": not any(
            path.endswith("/near/split.json")
            for path in entry_result["opened_relevant"]),
        "obs_future_price_identical": obs["future_price_all_identical"],
        "obs_positive_control": obs["positive_control_any_changed"],
        "obs_dim_12": obs["obs_dim"] == 12,
    }
    payload = {
        "boundary": ("one temporary training episode and one OBS audit; no "
                     "historical run is touched; no H2 or legacy test run"),
        "checks": checks,
        "all_pass": all(checks.values()),
        "entry": {
            "return_code": entry_result["return_code"],
            "replay_checks": entry_result["replay_checks"],
            "opened_relevant": entry_result["opened_relevant"],
            "forbidden_hits": entry_result["forbidden_hits"],
            "manifest_subset": {
                "split": manifest.get("split"),
                "split_name": manifest.get("split_name"),
                "train_subset": manifest.get("train_subset"),
                "validation_pairs": manifest.get("validation_pairs"),
                "test_pairs": manifest.get("test_pairs"),
                "holiday_pairs": manifest.get("holiday_pairs"),
                "legacy_seen_test_pairs":
                    manifest.get("legacy_seen_test_pairs"),
            },
        },
        "obs_causality": obs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"all_pass": payload["all_pass"], "checks": checks},
                     indent=2))
    print(f"wrote {OUT}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
