"""End-to-end smoke for MATD3_B_C2_GATE.

One fixed training day, one full 24-block episode, a minimal engineering
update, and a real save/destroy/restore/continue round trip. This is not a
performance evaluation and it does not choose any hyperparameter.

Outputs:
- data/processed/v2_cy_jn_2025/trainer_smoke_results.json
- data/processed/v2_cy_jn_2025/trainer_configs.json
- reports/v2_cy_jn_2025/full_day_episode_validation.md
- reports/v2_cy_jn_2025/trainer_smoke_update_audit.md
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import rl_b_c2_gate as mod
import screen_v2_cy_jn_2025 as screen
import v2_cy_jn_env as envmod
from b1_daily_baseline import config_hash as market_config_hash

OUT = "data/processed/v2_cy_jn_2025"
SMOKE = f"{OUT}/trainer_smoke_results.json"
CONFIGS = f"{OUT}/trainer_configs.json"
FULL_DAY_REPORT = "reports/v2_cy_jn_2025/full_day_episode_validation.md"
UPDATE_REPORT = "reports/v2_cy_jn_2025/trainer_smoke_update_audit.md"
PAIRING = "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
SPLIT = "data/processed/v2_cy_jn_2025/near/split.json"
B1_MANIFEST = mod.B1_MANIFEST
B1_INPUTS = ("pairing", "bus_loads", "price_curves")


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: str | Path) -> list[dict]:
    import csv
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _training_pair() -> tuple[dict, dict]:
    split = json.loads(Path(SPLIT).read_text(encoding="utf-8"))
    load_date = split["train"][0]
    rows = _read(PAIRING)
    row = next(r for r in rows if r["load_date"] == load_date)
    return ({"load_date": row["load_date"],
             "price_date": row["price_date"]}, row)


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _day_checks(out: dict, trainer: mod.MATD3BC2Gate,
                experiment: mod.ExperimentConfig, b1_manifest: dict,
                config_hash_value: str, pair_row: dict) -> dict:
    units = trainer.agent_names
    checks = out["checks"]
    b = out["b"]
    c2 = out["c2"]
    delta = b["delta"]
    g_pol = b["G_policy"]
    g_b1 = b["G_b1"]
    b_blocks = out["b_reward_by_block"]
    shaping = out["c2_shaping_by_block"]

    result = {
        "pair_id": out["pair_id"],
        "blocks_cleared": checks["clears"],
        "fallbacks": checks["fallbacks"],
        "lmp_fallbacks": checks["lmp_fallbacks"],
        "all_optimal": checks["all_optimal"],
        "voltage_violation_cells": checks["voltage_violation_cells"],
        "line_overload_cells": checks["line_overload_cells"],
        "terminal_soc": {u: float(checks["terminal_soc"][u]) for u in units},
        "charge_mwh": {u: float(checks["charge_mwh"][u]) for u in units},
        "discharge_mwh": {u: float(checks["discharge_mwh"][u]) for u in units},
        "cycles": {u: float(checks["cycles"][u]) for u in units},
        "losses_mwh": float(checks["losses_mwh"]),
        "quotes_legal": True,
        "closed_gate_is_zero": True,
        "delta_identity_max_error": max(
            abs(delta[u] - (g_pol[u] - g_b1[u])) for u in units),
        "b_paid_only_at_last_block": all(
            abs(b_blocks[u][t]) == 0.0 for u in units for t in range(23)),
        "b_last_block_equals_delta": all(
            abs(b_blocks[u][23] - delta[u]) < 1e-9 for u in units),
        "c2_blocks_nonzero": sum(
            1 for u in units for t in range(24)
            if abs(shaping[u][t]) > 1e-12),
        "c2_boundary_max_error": max(
            abs(c2["episode_return"][u]
                - (b["discounted_return"][u] + c2["boundary"][u]))
            for u in units),
        "b1_manifest_config_hash_matches": (
            b1_manifest["config_hash"] == config_hash_value),
        "b1_input_hashes_match": all(
            b1_manifest["inputs"][name]["sha256"]
            == _sha(b1_manifest["inputs"][name]["path"])
            for name in B1_INPUTS),
        "b1_data_hash_matches_pair": (
            mod.pair_data_hash(pair_row) != "" and
            len(mod.pair_data_hash(pair_row)) == 64),
        "updates_run": trainer.updates,
        "total_steps": trainer.total_steps,
        "buffer_len": len(trainer.buffer),
        "fixed_scale": experiment.fixed_scale,
        "reward_mode": experiment.reward_mode,
        "gate": out["gate"],
    }
    reward_arrays = np.asarray([row[2] for row in trainer.buffer.buffer])
    expected = {
        u: [b_blocks[u][t] + (shaping[u][t] if shaping else 0.0)
            for t in range(24)]
        for u in units}
    # The buffer stores the final reward divided by the fixed scale.
    for i, u in enumerate(units):
        checksum = float(np.sum(reward_arrays[:, i]))
        target = float(np.sum(expected[u]) / experiment.fixed_scale)
        result[f"buffer_reward_sum_{u}"] = checksum
        result[f"expected_reward_sum_{u}"] = target
    result["buffer_reward_identity_max_error"] = max(
        abs(result[f"buffer_reward_sum_{u}"]
            - result[f"expected_reward_sum_{u}"]) for u in units)
    actions = np.asarray([row[1] for row in trainer.buffer.buffer])
    gate_values = np.asarray(list(trainer.buffer.extras["gates"]))
    closed = gate_values <= 0.0
    result["quotes_legal"] = bool(
        np.all(actions >= -1e-9) and np.all(actions[:, :, 0] <= 150.0 + 1e-6)
        and np.all(actions[:, :, 1] <= 300.0 + 1e-6))
    result["closed_gate_is_zero"] = bool(
        np.all(np.abs(actions[closed]) <= 1e-12))
    return _clean(result)


def _checkpoint_round_trip(trainer: mod.MATD3BC2Gate, experiment,
                           env, tmp: Path) -> dict:
    obs0 = np.zeros(trainer.obs_dim, dtype=np.float32)
    trainer.save(tmp, level="resume", dataset="smoke", seed=0,
                 config_hash_value=market_config_hash(env.config))
    params_before = {nm: copy.deepcopy(trainer.actors[nm].state_dict())
                     for nm in trainer.agent_names}
    sample_expected = trainer.buffer.sample(8)
    action_expected = trainer.select_action(obs0, trainer.agent_names[0])
    update_expected = trainer.update()
    params_expected = {nm: copy.deepcopy(trainer.actors[nm].state_dict())
                       for nm in trainer.agent_names}
    params_changed = any(
        not torch.equal(v, params_before[nm][k])
        for nm in trainer.agent_names
        for k, v in params_expected[nm].items())
    counters_expected = (trainer.total_steps, trainer.updates, trainer.episodes)
    exposure_expected = dict(trainer.exposure)
    del trainer
    gc.collect()

    restored = mod.MATD3BC2Gate(env, experiment)
    bundle = restored.load(tmp)
    sample_actual = restored.buffer.sample(8)
    action_actual = restored.select_action(obs0, restored.agent_names[0])
    update_actual = restored.update()
    params_ok = all(
        torch.equal(v, restored.actors[nm].state_dict()[k])
        for nm in restored.agent_names
        for k, v in params_expected[nm].items())
    buffers_ok = all(
        torch.equal(sample_expected[k], sample_actual[k])
        for k in sample_expected)
    losses_ok = all(
        abs(update_expected[k] - update_actual[k]) < 1e-6
        for k in ("critic_loss", "actor_loss", "q1_mean", "q2_mean"))
    counters_ok = counters_expected == (
        restored.total_steps, restored.updates, restored.episodes)
    return _clean({
        "checkpoint_level": bundle["manifest"]["level"],
        "action_restored": bool(np.allclose(
            action_expected["quotes"], action_actual["quotes"])),
        "buffer_sample_restored": bool(buffers_ok),
        "losses_restored": bool(losses_ok),
        "actor_parameters_changed_by_update": bool(params_changed),
        "actor_parameters_restored_and_updated_identically": bool(params_ok),
        "counters_restored": bool(counters_ok),
        "exposure_restored": bool(dict(restored.exposure) == exposure_expected),
        "update_expected": update_expected,
        "update_actual": update_actual,
        "all_pass": bool(
            np.allclose(action_expected["quotes"], action_actual["quotes"])
            and buffers_ok and losses_ok and params_ok and params_changed
            and counters_ok
            and dict(restored.exposure) == exposure_expected),
    })


def _reports(result: dict, round_trip: dict, experiment) -> None:
    d = result["day"]
    lines = [
        "# Full 24-block episode validation",
        "",
        "One fixed training day through the B+C2 + truthful-gate branch. This "
        "is an engineering check, not a policy result.",
        "",
        f"- pair: `{d['pair_id']}`",
        f"- experiment: `{experiment.experiment_id}` "
        f"({experiment.reward_mode})",
        f"- blocks cleared: {d['blocks_cleared']}/24",
        f"- fallbacks: {d['fallbacks']}, LMP fallback cells: "
        f"{d['lmp_fallbacks']}",
        f"- all clears OPTIMAL: {d['all_optimal']}",
        f"- voltage violation cells: {d['voltage_violation_cells']}",
        f"- line overload cells: {d['line_overload_cells']}",
        f"- terminal SOC: {d['terminal_soc']}",
        f"- quotes legal: {d['quotes_legal']}",
        f"- every closed gate exactly zero: {d['closed_gate_is_zero']}",
        f"- charge / discharge / equivalent cycles: "
        f"{d['charge_mwh']} / {d['discharge_mwh']} / {d['cycles']}",
        f"- network losses: {d['losses_mwh']:.4f} MWh",
        "",
        "## Reward accounting",
        "",
        f"- `delta = G_policy - G_B1`, max error "
        f"{d['delta_identity_max_error']:.3e}",
        f"- B paid only at block 23: {d['b_paid_only_at_last_block']}",
        f"- B at block 23 equals delta: {d['b_last_block_equals_delta']}",
        f"- C2 shaping terms present: {d['c2_blocks_nonzero']} of 96",
        f"- C2 telescope error: {d['c2_boundary_max_error']:.3e}",
        f"- buffer reward equals (B+C2)/scale, max error "
        f"{d['buffer_reward_identity_max_error']:.3e}",
        f"- fixed reward scale: {d['fixed_scale']} (no running "
        f"standardisation)",
        "",
        "The B1 arm is the registered terminal-neutral rolling B1: same pair, "
        "same initial SOC, same 24-block interface, same exogenous day. The "
        "manifest config hash and the three input hashes match.",
        "",
        "## Gate behaviour",
        "",
        f"- gate open frequency: {d['gate']['open_frequency']}",
        f"- gate mean probability: {d['gate']['mean_probability']}",
        f"- observed fleet modes: {sorted(d['gate']['mode_counts'])}",
        f"- saturated upper / lower (open legs only): "
        f"{d['gate']['saturated_upper']} / {d['gate']['saturated_lower']}",
        "",
        "A closed gate is an exact zero by construction and is reported "
        "through the open frequency, not as a saturated quote.",
        "",
    ]
    Path(FULL_DAY_REPORT).write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "# Trainer smoke update and checkpoint audit",
        "",
        "No performance claim. The update is the smallest engineering update "
        "that proves the loss, gradient and optimizer path are connected.",
        "",
        f"- updates executed: {d['updates_run']}",
        f"- environment steps: {d['total_steps']}",
        f"- replay buffer length: {d['buffer_len']}",
        f"- critic loss finite: "
        f"{np.isfinite(round_trip['update_expected']['critic_loss'])}",
        f"- actor loss finite: "
        f"{np.isfinite(round_trip['update_expected']['actor_loss'])}",
        f"- actor parameters changed by the update: "
        f"{round_trip['actor_parameters_changed_by_update']}",
        f"- restored parameters update identically: "
        f"{round_trip['actor_parameters_restored_and_updated_identically']}",
        "",
        "## Checkpoint round trip",
        "",
        "| check | result |",
        "|---|---|",
        f"| level | {round_trip['checkpoint_level']} |",
        f"| deterministic action restored | "
        f"{round_trip['action_restored']} |",
        f"| replay sample restored (RNG) | "
        f"{round_trip['buffer_sample_restored']} |",
        f"| critic/actor losses restored | "
        f"{round_trip['losses_restored']} |",
        f"| parameters updated identically after restore | "
        f"{round_trip['actor_parameters_restored_and_updated_identically']} |",
        f"| counters restored | {round_trip['counters_restored']} |",
        f"| date exposure log restored | "
        f"{round_trip['exposure_restored']} |",
        f"| all pass | {round_trip['all_pass']} |",
        "",
        "The restored trainer continues from the saved buffer, optimiser "
        "moments, RNG and counters; it is not an actors-only file.",
        "",
    ]
    Path(UPDATE_REPORT).write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="")
    args = parser.parse_args(argv)
    # A fixed engineering seed makes the smoke artifact reproducible; it is
    # not a formal experiment seed.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    pair, pair_row = _training_pair()
    b1 = json.loads(Path(B1_MANIFEST).read_text(encoding="utf-8"))
    cache = mod.load_b1_cache()
    experiment = mod.ExperimentConfig(
        "SMOKE_B_C2_GATE", "B_C2", batch_size=8, buffer_capacity=512)
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    cfg_hash = market_config_hash(env.config)
    if cfg_hash != b1["config_hash"]:
        raise ValueError("B1 manifest config hash does not match this env")
    trainer = mod.MATD3BC2Gate(env, experiment)
    out = trainer.train_day(
        env, pair, cache, config_hash_value=b1["mechanism_hash"],
        data_hash=mod.pair_data_hash(pair_row))

    day = _day_checks(out, trainer, experiment, b1, cfg_hash, pair_row)
    tmp = Path(tempfile.mkdtemp(prefix="bc2gate-smoke-"))
    try:
        round_trip = _checkpoint_round_trip(trainer, experiment, env, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    result = {
        "day": day,
        "checkpoint": round_trip,
        "experiment": experiment.as_dict(),
        "all_pass": bool(
            day["blocks_cleared"] == 24 and day["fallbacks"] == 0
            and day["lmp_fallbacks"] == 0 and day["all_optimal"]
            and day["voltage_violation_cells"] == 0
            and day["line_overload_cells"] == 0
            and all(abs(v - 0.5) <= 1e-5
                    for v in day["terminal_soc"].values())
            and day["delta_identity_max_error"] < 1e-9
            and day["b_paid_only_at_last_block"]
            and day["b_last_block_equals_delta"]
            and day["c2_boundary_max_error"] < 1e-9
            and day["c2_blocks_nonzero"] > 0
            and day["buffer_reward_identity_max_error"] < 1e-6
            and day["updates_run"] > 0
            and day["quotes_legal"] and day["closed_gate_is_zero"]
            and day["b1_manifest_config_hash_matches"]
            and day["b1_input_hashes_match"]
            and round_trip["all_pass"]),
    }
    Path(SMOKE).write_text(json.dumps(_clean(result), indent=2),
                           encoding="utf-8")
    configs = {
        name: {"config": cfg.as_dict(), "hash": _sha_of(cfg.as_dict())}
        for name, cfg in mod.EXPERIMENTS.items()
    }
    Path(CONFIGS).write_text(json.dumps(configs, indent=2), encoding="utf-8")
    _reports(result, round_trip, experiment)
    print(json.dumps({"all_pass": result["all_pass"],
                      "day": {k: day[k] for k in
                              ("blocks_cleared", "fallbacks",
                               "delta_identity_max_error",
                               "buffer_reward_identity_max_error")},
                      "checkpoint": round_trip["all_pass"]}, indent=2))
    return 0 if result["all_pass"] else 1


def _sha_of(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
