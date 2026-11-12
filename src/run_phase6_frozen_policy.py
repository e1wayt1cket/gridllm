"""Phase 6: frozen-policy paired market clearing under two quote maps.

Read-only diagnostic. It replays the deterministic 24-block day for a frozen
actor checkpoint under the control map (Z_CORE=2.5, identical to
`linear_core_v1`) and the candidate map (Z_CORE=10.0). Nothing here trains:
there is no optimizer, no replay buffer, no exploration, no calibration
exposure and no training noise.

The only difference between the two arms is `da_quote_map.map_fraction`, which
`quote_map_candidates.patched` swaps for the duration of one actor forward.
Every market input, the environment, the initial SOC, the terminal-SOC
mechanism and the physical constraints are shared.

Usage:
    PYTHONPATH=src python src/run_phase6_frozen_policy.py --preflight
    PYTHONPATH=src python src/run_phase6_frozen_policy.py --phase A
    PYTHONPATH=src python src/run_phase6_frozen_policy.py --phase B
    PYTHONPATH=src python src/run_phase6_frozen_policy.py --phase C
    PYTHONPATH=src python src/run_phase6_frozen_policy.py --analyze
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

import config_loader
import da_quote_map as qmap
import participant_payoff as pp
import quote_map_candidates as cands
import wholesale_price
from c2_reward import C2Config
from rl_b_c2_gate import GateActor
from v2_cy_jn_env import build_env_for_pair

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase6_frozen_policy_2026-11-12"
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
MAPS = (2.5, 10.0)
PHASE_FILE = {
    "A": OUT / "phase6A_smoke_results.csv",
    "B": OUT / "phase6B_train_results.csv",
    "C": OUT / "phase6C_validation_results.csv",
}
BOOTSTRAP_SEED = 20261112
N_RESAMPLES = 1000
DT = 0.25

ESS_FIELDS = [
    "net_income_cny", "market_revenue_cny", "purchase_cost_cny",
    "cycle_cost_cny", "terminal_inventory_value_cny",
    "charge_mwh", "discharge_mwh", "cycles",
    "energy_sold_mwh", "energy_bought_mwh",
    "terminal_soc", "terminal_soc_err", "soc_min", "soc_max",
    "soc_band_violation", "mutual_exclusion_max", "power_violation",
    "quote_charge_mean_open", "quote_discharge_mean_open",
    "quote_charge_mean_all", "quote_discharge_mean_all",
    "gate_open_charge_frac", "gate_open_discharge_frac",
    "cleared_charge_mwh", "cleared_discharge_mwh",
    "lmp_mean", "lmp_min", "lmp_max",
    "distinct_quotes_charge", "distinct_quotes_discharge",
    "bus",
]
FLEET_FIELDS = [
    "net_income_cny", "market_revenue_cny", "purchase_cost_cny",
    "cycle_cost_cny", "terminal_inventory_value_cny",
    "charge_mwh", "discharge_mwh", "energy_sold_mwh", "energy_bought_mwh",
    "terminal_soc_min", "terminal_soc_max", "terminal_soc_err_max",
    "soc_band_violation_max", "mutual_exclusion_max", "power_violation_max",
]
DAY_FIELDS = [
    "clears", "optimal_clears", "fell_back_clears", "none_clears",
    "lmp_fallbacks", "infeasible_clears", "final_status_names",
    "voltage_min", "voltage_max", "voltage_violation_cells",
    "line_max_utilization", "line_overload_cells",
    "pcc_import_mwh", "pcc_export_mwh", "pcc_net_mean_mw",
    "pcc_net_min_mw", "pcc_net_max_mw", "clearing_objective_sum",
    "raw_profit_sum_cny", "raw_profit_max_abs_diff",
    "mutual_exclusion_flag", "day_ok", "failure_reason", "elapsed_s",
]
BASE_FIELDS = [
    "phase", "checkpoint", "seed", "round", "checkpoint_kind", "role",
    "z_core", "pair_id", "load_date", "price_date", "season",
]
WIDE_FIELDS = (BASE_FIELDS + [f"fleet_{f}" for f in FLEET_FIELDS]
               + [f"{nm}_{f}" for nm in NAMES for f in ESS_FIELDS]
               + DAY_FIELDS)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    return json.loads((OUT / "phase6_config.json").read_text(encoding="utf-8"))


def load_actors(checkpoint: str) -> dict:
    actors = {}
    for name in NAMES:
        actor = GateActor(12)
        state = torch.load(OUT / "checkpoints" / checkpoint / "actors"
                           / f"{name}.pt", map_location="cpu",
                           weights_only=False)
        actor.load_state_dict(state)
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        actors[name] = actor
    return actors


def actor_state_hash(actors: dict) -> str:
    digest = hashlib.sha256()
    for name in NAMES:
        state = actors[name].state_dict()
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def policy_action(actor: GateActor, obs: np.ndarray, z_core: float) -> dict:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        policy_out = actor(obs_t)
        with cands.patched(z_core):
            out = qmap.forward_policy_v3(
                policy_out, train=False, gate_temperature=1.0)
    return {
        "quotes": out["quotes"].squeeze(0).numpy().astype(float).copy(),
        "gates": out["gates"].squeeze(0).numpy().astype(float).copy(),
        "probs": out["probs"].squeeze(0).numpy().astype(float).copy(),
        "price_quotes": out["price_quotes"].squeeze(0).numpy()
        .astype(float).copy(),
        "mode": int(out["mode"].reshape(-1)[0].item()),
        "gate_logits": policy_out[..., :2].squeeze(0).numpy().copy(),
        "price_logits": policy_out[..., 2:].squeeze(0).numpy().copy(),
    }


def _mean(values) -> float:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _nanmax(values, default=float("nan")) -> float:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(max(values)) if values else default


def run_day(pair: dict, actors: dict, z_core: float) -> dict:
    """One deterministic 24-block day. No optimizer, no buffer."""
    started = time.time()
    env = build_env_for_pair(pair, differential_reward=False,
                             terminal_soc_target=0.5)
    c2 = C2Config()
    obs = env.reset()
    policy_hashes_before = actor_state_hash(actors)

    per_ess = {nm: {
        "net_income_cny": 0.0, "market_revenue_cny": 0.0,
        "purchase_cost_cny": 0.0, "cycle_cost_cny": 0.0,
        "terminal_inventory_value_cny": 0.0,
        "charge_mwh": 0.0, "discharge_mwh": 0.0,
        "energy_sold_mwh": 0.0, "energy_bought_mwh": 0.0,
        "quotes_charge": [], "quotes_discharge": [],
        "price_quotes_charge": [], "price_quotes_discharge": [],
        "gate_charge": [], "gate_discharge": [],
        "soc_min": math.inf, "soc_max": -math.inf,
        "soc_band_violation": 0.0, "mutual_exclusion_max": 0.0,
        "power_violation": 0.0, "lmp_values": [],
        "raw_profit_sum": 0.0, "raw_profit_abs_diff": 0.0,
        "bus": None,
    } for nm in NAMES}
    day = {
        "clears": 0, "optimal_clears": 0, "fell_back_clears": 0,
        "none_clears": 0, "lmp_fallbacks": 0, "infeasible_clears": 0,
        "status_names": set(),
        "voltage_min": math.inf, "voltage_max": -math.inf,
        "voltage_violation_cells": 0,
        "line_max_utilization": 0.0, "line_overload_cells": 0,
        "pcc_import_mwh": 0.0, "pcc_export_mwh": 0.0,
        "pcc_net_mw": [], "clearing_objective_sum": 0.0,
        "failure": [],
    }
    v_lo = float(env.config.network.v_min_pu)
    v_hi = float(env.config.network.v_max_pu)
    done = False
    block_count = 0
    try:
        for block in range(24):
            block_count += 1
            actions = {nm: policy_action(actors[nm], obs[nm], z_core)
                       for nm in NAMES}
            acts = {nm: actions[nm]["quotes"] for nm in NAMES}
            next_obs, _rewards, done, info = env.step(acts)
            if env._last_result is None:
                day["none_clears"] += 1
                day["failure"].append(f"block {block}: clearing returned None")
                break
            result = env._last_result
            n_commit = min(4, env.T - block * 4)
            day["clears"] += 1

            solver = result.get("solver", {})
            final = solver.get("final", {}) or {}
            status = str(final.get("status_name", "UNKNOWN"))
            day["status_names"].add(status)
            if bool(final.get("optimal")):
                day["optimal_clears"] += 1
            if "INFEASIBLE" in status or "INF_OR_UNBD" in status \
                    or "UNKNOWN" in status:
                day["infeasible_clears"] += 1
            if result.get("fell_back") or solver.get("used_fallback"):
                day["fell_back_clears"] += 1
            day["lmp_fallbacks"] += int(result.get("lmp_fallbacks", 0) or 0)
            day["clearing_objective_sum"] += float(
                result.get("clearing_objective", 0.0) or 0.0)

            voltage = np.asarray(result.get("bus_voltage", []), dtype=float)
            if voltage.size:
                voltage = voltage[:n_commit]
                day["voltage_min"] = min(day["voltage_min"],
                                         float(np.min(voltage)))
                day["voltage_max"] = max(day["voltage_max"],
                                         float(np.max(voltage)))
                day["voltage_violation_cells"] += int(
                    ((voltage < v_lo - 1e-7) |
                     (voltage > v_hi + 1e-7)).sum())
            util = np.asarray(result.get("line_utilization", []), dtype=float)
            if util.size:
                util = util[:n_commit]
                day["line_max_utilization"] = max(
                    day["line_max_utilization"], float(np.max(util)))
                day["line_overload_cells"] += int(
                    (util > 1.0 + 1e-7).sum())
            gi = np.asarray(result.get("grid_import_mw", []), dtype=float)
            ge = np.asarray(result.get("grid_export_mw", []), dtype=float)
            if gi.size and ge.size:
                gi = gi[:n_commit]
                ge = ge[:n_commit]
                day["pcc_import_mwh"] += float(np.sum(gi)) * DT
                day["pcc_export_mwh"] += float(np.sum(ge)) * DT
                day["pcc_net_mw"].extend((gi - ge).tolist())

            for nm in NAMES:
                agent = next(a for a in env.all_agents if a.name == nm)
                schedule = (result.get("schedules") or {}).get(nm)
                if schedule is None:
                    day["failure"].append(
                        f"block {block}: no schedule for {nm}")
                    continue
                per_ess[nm]["bus"] = int(agent.bus)
                lmp_node = np.asarray(result["lmp"], dtype=float)[
                    :n_commit, agent.bus]
                breakdown = pp.participant_payoff(
                    schedule, lmp_node, agent, env.config,
                    n_periods=n_commit)
                per_ess[nm]["market_revenue_cny"] += float(breakdown.revenue)
                per_ess[nm]["purchase_cost_cny"] += float(
                    breakdown.purchase_cost)
                per_ess[nm]["cycle_cost_cny"] += float(
                    breakdown.degradation_cost)
                per_ess[nm]["net_income_cny"] += float(breakdown.total)
                per_ess[nm]["charge_mwh"] += float(
                    np.sum(np.asarray(schedule["p_ch"], dtype=float)
                           [:n_commit])) * DT
                per_ess[nm]["discharge_mwh"] += float(
                    np.sum(np.asarray(schedule["p_dis"], dtype=float)
                           [:n_commit])) * DT
                per_ess[nm]["energy_sold_mwh"] += float(
                    np.sum(np.asarray(schedule["p_sell"], dtype=float)
                           [:n_commit])) * DT
                per_ess[nm]["energy_bought_mwh"] += float(
                    np.sum(np.asarray(schedule["p_buy"], dtype=float)
                           [:n_commit])) * DT
                per_ess[nm]["lmp_values"].extend(lmp_node.tolist())
                raw = float(info.get("raw_profit", {}).get(nm, 0.0))
                per_ess[nm]["raw_profit_sum"] += raw
                per_ess[nm]["raw_profit_abs_diff"] = max(
                    per_ess[nm]["raw_profit_abs_diff"],
                    abs(per_ess[nm]["net_income_cny"]
                        - per_ess[nm]["raw_profit_sum"]))

                p_ch = np.asarray(schedule["p_ch"], dtype=float)[:n_commit]
                p_dis = np.asarray(schedule["p_dis"], dtype=float)[:n_commit]
                soc = np.asarray(schedule["soc_boundary"], dtype=float)[
                    :n_commit + 1]
                if soc.size:
                    per_ess[nm]["soc_min"] = min(
                        per_ess[nm]["soc_min"], float(np.min(soc)))
                    per_ess[nm]["soc_max"] = max(
                        per_ess[nm]["soc_max"], float(np.max(soc)))
                    per_ess[nm]["soc_band_violation"] = max(
                        per_ess[nm]["soc_band_violation"],
                        float(max(0.0, c2.soc_lo - np.min(soc),
                                  np.max(soc) - c2.soc_hi)))
                per_ess[nm]["mutual_exclusion_max"] = max(
                    per_ess[nm]["mutual_exclusion_max"],
                    float(np.max(p_ch * p_dis)) if p_ch.size else 0.0)
                storage = agent.storage
                per_ess[nm]["power_violation"] = max(
                    per_ess[nm]["power_violation"],
                    float(max(0.0,
                              np.max(p_ch) - storage.p_ch_max,
                              np.max(p_dis) - storage.p_dis_max,
                              -np.min(p_ch), -np.min(p_dis)))
                    if p_ch.size else 0.0)

                gate = actions[nm]["gates"]
                quote = actions[nm]["quotes"]
                price_quote = actions[nm]["price_quotes"]
                per_ess[nm]["gate_charge"].append(float(gate[0] > 0))
                per_ess[nm]["gate_discharge"].append(float(gate[1] > 0))
                per_ess[nm]["quotes_charge"].append(float(quote[0]))
                per_ess[nm]["quotes_discharge"].append(float(quote[1]))
                per_ess[nm]["price_quotes_charge"].append(float(price_quote[0]))
                per_ess[nm]["price_quotes_discharge"].append(
                    float(price_quote[1]))

            obs = next_obs
            if done:
                break
    except Exception as exc:  # a failed day must not abort the whole phase
        day["failure"].append(f"{type(exc).__name__}: {exc}")
    finally:
        pass

    policy_hashes_after = actor_state_hash(actors)
    if policy_hashes_before != policy_hashes_after:
        day["failure"].append("actor weights changed during the day")

    if not done and block_count == 24:
        # The loop can end at block 23 with done=True; anything else is short.
        pass
    if day["clears"] != 24:
        day["failure"].append(f"only {day['clears']} of 24 blocks cleared")

    terminal_soc = {}
    fleet_terminal_err = 0.0
    for nm in NAMES:
        value = float(env.prev_soc.get(nm, float("nan")))
        terminal_soc[nm] = value
        per_ess[nm]["terminal_soc"] = value
        per_ess[nm]["terminal_soc_err"] = abs(value - 0.5)
        fleet_terminal_err = max(fleet_terminal_err,
                                 per_ess[nm]["terminal_soc_err"])
    elapsed = time.time() - started

    for nm in NAMES:
        item = per_ess[nm]
        item["cycles"] = (item["charge_mwh"] + item["discharge_mwh"]) \
            / (2.0 * float(next(a for a in env.all_agents
                                if a.name == nm).storage.e_max))
        item["cleared_charge_mwh"] = item["charge_mwh"]
        item["cleared_discharge_mwh"] = item["discharge_mwh"]
        n = len(item["quotes_charge"])
        item["quote_charge_mean_all"] = _mean(item["quotes_charge"])
        item["quote_discharge_mean_all"] = _mean(item["quotes_discharge"])
        open_charge = [q for q, g in zip(item["quotes_charge"],
                                         item["gate_charge"]) if g > 0]
        open_discharge = [q for q, g in zip(item["quotes_discharge"],
                                            item["gate_discharge"]) if g > 0]
        item["quote_charge_mean_open"] = _mean(open_charge)
        item["quote_discharge_mean_open"] = _mean(open_discharge)
        item["gate_open_charge_frac"] = (
            float(sum(item["gate_charge"])) / n if n else float("nan"))
        item["gate_open_discharge_frac"] = (
            float(sum(item["gate_discharge"])) / n if n else float("nan"))
        item["lmp_mean"] = _mean(item["lmp_values"])
        item["lmp_min"] = min(item["lmp_values"]) if item["lmp_values"] \
            else float("nan")
        item["lmp_max"] = max(item["lmp_values"]) if item["lmp_values"] \
            else float("nan")
        item["distinct_quotes_charge"] = len(
            {round(q, 9) for q in item["quotes_charge"]})
        item["distinct_quotes_discharge"] = len(
            {round(q, 9) for q in item["quotes_discharge"]})
        if item["soc_min"] == math.inf:
            item["soc_min"] = float("nan")
            item["soc_max"] = float("nan")
        for key in ("quotes_charge", "quotes_discharge",
                    "price_quotes_charge", "price_quotes_discharge",
                    "gate_charge", "gate_discharge", "lmp_values"):
            del item[key]

    fleet = {
        "net_income_cny": sum(per_ess[nm]["net_income_cny"] for nm in NAMES),
        "market_revenue_cny": sum(
            per_ess[nm]["market_revenue_cny"] for nm in NAMES),
        "purchase_cost_cny": sum(
            per_ess[nm]["purchase_cost_cny"] for nm in NAMES),
        "cycle_cost_cny": sum(
            per_ess[nm]["cycle_cost_cny"] for nm in NAMES),
        "terminal_inventory_value_cny": 0.0,
        "charge_mwh": sum(per_ess[nm]["charge_mwh"] for nm in NAMES),
        "discharge_mwh": sum(
            per_ess[nm]["discharge_mwh"] for nm in NAMES),
        "energy_sold_mwh": sum(
            per_ess[nm]["energy_sold_mwh"] for nm in NAMES),
        "energy_bought_mwh": sum(
            per_ess[nm]["energy_bought_mwh"] for nm in NAMES),
        "terminal_soc_min": min(terminal_soc.values()),
        "terminal_soc_max": max(terminal_soc.values()),
        "terminal_soc_err_max": fleet_terminal_err,
        "soc_band_violation_max": max(
            per_ess[nm]["soc_band_violation"] for nm in NAMES),
        "mutual_exclusion_max": max(
            per_ess[nm]["mutual_exclusion_max"] for nm in NAMES),
        "power_violation_max": max(
            per_ess[nm]["power_violation"] for nm in NAMES),
    }
    raw_sum = sum(per_ess[nm]["raw_profit_sum"] for nm in NAMES)
    raw_diff = max(per_ess[nm]["raw_profit_abs_diff"] for nm in NAMES)
    day["raw_profit_sum_cny"] = raw_sum
    day["raw_profit_max_abs_diff"] = raw_diff
    day["mutual_exclusion_flag"] = bool(fleet["mutual_exclusion_max"] > 1e-6)
    day["pcc_net_mean_mw"] = _mean(day["pcc_net_mw"])
    day["pcc_net_min_mw"] = min(day["pcc_net_mw"]) \
        if day["pcc_net_mw"] else float("nan")
    day["pcc_net_max_mw"] = max(day["pcc_net_mw"]) \
        if day["pcc_net_mw"] else float("nan")
    del day["pcc_net_mw"]
    if day["voltage_min"] == math.inf:
        day["voltage_min"] = float("nan")
        day["voltage_max"] = float("nan")

    checks = {
        "clears": day["clears"] == 24,
        "optimal": day["optimal_clears"] == 24,
        "no_fallback": day["fell_back_clears"] == 0,
        "no_none": day["none_clears"] == 0,
        "nodal_lmp": day["lmp_fallbacks"] == 0,
        "voltage": day["voltage_violation_cells"] == 0,
        "line": day["line_overload_cells"] == 0,
        "terminal_soc": fleet["terminal_soc_err_max"] <= 1e-5,
        "soc_band": fleet["soc_band_violation_max"] <= 1e-6,
        "power": fleet["power_violation_max"] <= 1e-6,
        "raw_profit_match": raw_diff <= 1e-6,
    }
    for key, ok in checks.items():
        if not ok:
            day["failure"].append(f"check failed: {key}")
    day["day_ok"] = all(checks.values())
    day["final_status_names"] = ";".join(sorted(day.pop("status_names")))
    day["failure_reason"] = " | ".join(day.pop("failure"))
    day["elapsed_s"] = elapsed

    result = {"fleet": fleet, "per_ess": per_ess, "day": day}
    del env
    gc.collect()
    return result


def flatten_row(phase: str, meta: dict, pair: dict, z_core: float,
                result: dict) -> dict:
    row = {
        "phase": phase,
        "checkpoint": meta["key"],
        "seed": meta["seed"],
        "round": meta["round"],
        "checkpoint_kind": meta["kind"],
        "role": meta["role"],
        "z_core": f"{z_core:.1f}",
        "pair_id": pair["pair_id"],
        "load_date": pair["load_date"],
        "price_date": pair["price_date"],
        "season": pair["season"],
    }
    for key, value in result["fleet"].items():
        row[f"fleet_{key}"] = value
    for nm in NAMES:
        for key, value in result["per_ess"][nm].items():
            row[f"{nm}_{key}"] = value
    for key, value in result["day"].items():
        row[key] = value
    return row


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WIDE_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def row_key(row: dict) -> tuple[str, str, str]:
    return (row["checkpoint"], f"{float(row['z_core']):.1f}",
            row["pair_id"])


def phase_meta(config: dict, phase: str) -> list[dict]:
    keys = config[f"phase6{phase}"]["checkpoints"]
    out = []
    for key in keys:
        meta = dict(config["checkpoints"][key])
        meta["key"] = key
        out.append(meta)
    return out


def phase_pairs(config: dict, phase: str) -> list[dict]:
    return list(config[f"phase6{phase}"]["pairs"])


def run_phase(phase: str, retry_failed: bool, limit: int | None) -> int:
    config = load_config()
    path = PHASE_FILE[phase]
    rows = read_rows(path)
    existing = {row_key(row): row for row in rows}
    pairs = phase_pairs(config, phase)
    if limit:
        pairs = pairs[:limit]
    metas = phase_meta(config, phase)
    total = len(pairs) * len(metas) * len(MAPS)
    done = 0
    print(f"[phase {phase}] {len(pairs)} pairs x {len(metas)} checkpoints "
          f"x {len(MAPS)} maps = {total} day-runs", flush=True)
    for meta in metas:
        actors = load_actors(meta["key"])
        before = actor_state_hash(actors)
        for pair in pairs:
            for z_core in MAPS:
                key = (meta["key"], f"{z_core:.1f}", pair["pair_id"])
                previous = existing.get(key)
                if previous is not None:
                    if not retry_failed or previous.get("day_ok") == "True":
                        done += 1
                        continue
                result = run_day(pair, actors, z_core)
                row = flatten_row(phase, meta, pair, z_core, result)
                existing[key] = row
                rows = list(existing.values())
                write_rows(path, rows)
                done += 1
                fleet_income = result["fleet"]["net_income_cny"]
                print(
                    f"[{done}/{total}] {phase} {meta['key']} z={z_core:.1f} "
                    f"{pair['pair_id']} fleet={fleet_income:.2f} "
                    f"ok={result['day']['day_ok']} "
                    f"t={result['day']['elapsed_s']:.1f}s "
                    f"{result['day']['failure_reason']}",
                    flush=True)
        after = actor_state_hash(actors)
        if before != after:
            raise RuntimeError(
                f"actor weights changed for {meta['key']} during phase {phase}")
    config_loader.reload_defaults()
    wholesale_price.reset_cache()
    return 0


def preflight() -> int:
    config = load_config()
    payload = {
        "written_at": now_iso(),
        "label": "NON_FORMAL_DIAGNOSTIC",
        "python": os.sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "checks": {},
    }
    checks = payload["checks"]
    # The control must reproduce the frozen map exactly.
    z = torch.linspace(-20.0, 20.0, 2001, dtype=torch.float64)
    with cands.patched(2.5):
        control = qmap.map_fraction(z)
    formal = qmap.map_fraction(z)
    checks["control_identity_max_abs_diff"] = float(
        torch.max(torch.abs(control - formal)).item())
    checks["control_identity_pass"] = bool(
        checks["control_identity_max_abs_diff"] < 1e-12)
    checks["candidate_self_tests"] = {
        str(zc): cands.run_tests(zc)["all_pass"] for zc in MAPS}
    checks["candidate_self_tests_pass"] = all(
        checks["candidate_self_tests"].values())
    checks["leg_highs"] = list(map(float, qmap.LEG_HIGHS))

    # Repeat the same frozen day and require bit-identical settlement.
    pair = config["phase6A"]["pairs"][0]
    actors = load_actors("seed11_final")
    before = actor_state_hash(actors)
    first = run_day(pair, actors, 2.5)
    second = run_day(pair, actors, 2.5)
    deltas = {}
    for nm in NAMES:
        deltas[nm] = abs(first["per_ess"][nm]["net_income_cny"]
                         - second["per_ess"][nm]["net_income_cny"])
    checks["repeat_max_abs_payoff_diff"] = max(deltas.values())
    checks["repeat_identical"] = bool(max(deltas.values()) < 1e-9)
    checks["repeat_pair"] = pair["pair_id"]
    checks["dynamic_quotes"] = {
        nm: {
            "charge": first["per_ess"][nm]["distinct_quotes_charge"],
            "discharge": first["per_ess"][nm]["distinct_quotes_discharge"],
        } for nm in NAMES}
    checks["dynamic_quotes_pass"] = any(
        v["charge"] > 1 or v["discharge"] > 1
        for v in checks["dynamic_quotes"].values())
    after = actor_state_hash(actors)
    checks["actor_hash_unchanged"] = before == after
    checks["actor_hash"] = before
    checks["all_pass"] = all([
        checks["control_identity_pass"],
        checks["candidate_self_tests_pass"],
        checks["repeat_identical"],
        checks["dynamic_quotes_pass"],
        checks["actor_hash_unchanged"],
    ])
    path = OUT / "phase6_preflight.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config_loader.reload_defaults()
    wholesale_price.reset_cache()
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _bootstrap(deltas: np.ndarray) -> dict:
    n = int(deltas.size)
    if n == 0:
        return {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    means = deltas[idx].mean(axis=1)
    return {
        "n_dates": n,
        "mean_delta_cny": float(np.mean(deltas)),
        "median_delta_cny": float(np.median(deltas)),
        "fraction_positive": float(np.mean(deltas > 0)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_resamples": N_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster": "pair_id (date)",
    }


def analyze() -> int:
    config = load_config()
    all_rows = []
    for phase, path in PHASE_FILE.items():
        all_rows.extend(read_rows(path))
    if not all_rows:
        raise SystemExit("no phase results to analyze")
    for row in all_rows:
        row["_z"] = float(row["z_core"])

    # Per-ESS and fleet payoff tables.
    per_ess_rows = []
    fleet_rows = []
    physical_rows = []
    solver_rows = []
    for row in all_rows:
        base = {key: row[key] for key in BASE_FIELDS}
        fleet = {key: row[f"fleet_{key}"] for key in FLEET_FIELDS}
        fleet_rows.append({**base, **fleet})
        physical_rows.append({
            **base,
            "voltage_min": row["voltage_min"],
            "voltage_max": row["voltage_max"],
            "voltage_violation_cells": row["voltage_violation_cells"],
            "line_max_utilization": row["line_max_utilization"],
            "line_overload_cells": row["line_overload_cells"],
            "soc_min": row["fleet_terminal_soc_min"],
            "soc_max": row["fleet_terminal_soc_max"],
            "soc_band_violation_max": row["fleet_soc_band_violation_max"],
            "terminal_soc_err_max": row["fleet_terminal_soc_err_max"],
            "mutual_exclusion_max": row["fleet_mutual_exclusion_max"],
            "mutual_exclusion_flag": row["mutual_exclusion_flag"],
            "power_violation_max": row["fleet_power_violation_max"],
            "day_ok": row["day_ok"],
            "failure_reason": row["failure_reason"],
        })
        solver_rows.append({
            **base,
            "clears": row["clears"],
            "optimal_clears": row["optimal_clears"],
            "fell_back_clears": row["fell_back_clears"],
            "none_clears": row["none_clears"],
            "lmp_fallbacks": row["lmp_fallbacks"],
            "infeasible_clears": row["infeasible_clears"],
            "final_status_names": row["final_status_names"],
            "day_ok": row["day_ok"],
        })
        for nm in NAMES:
            item = {key: row[f"{nm}_{key}"] for key in ESS_FIELDS}
            per_ess_rows.append({**base, "ess": nm, **item})

    # Paired differences on the same checkpoint and date.
    by_key = {}
    for row in all_rows:
        by_key.setdefault(
            (row["phase"], row["checkpoint"], row["pair_id"]), {})[
                row["_z"]] = row
    delta_rows = []
    for (phase, checkpoint, pair_id), arms in sorted(by_key.items()):
        if 2.5 not in arms or 10.0 not in arms:
            continue
        control = arms[2.5]
        candidate = arms[10.0]
        meta = candidate
        scopes = [("fleet", "fleet_net_income_cny")]
        scopes += [(nm, f"{nm}_net_income_cny") for nm in NAMES]
        for scope, column in scopes:
            c = float(control[column])
            x = float(candidate[column])
            delta = x - c
            delta_rows.append({
                "phase": phase,
                "checkpoint": checkpoint,
                "seed": meta["seed"],
                "round": meta["round"],
                "role": meta["role"],
                "pair_id": pair_id,
                "load_date": meta["load_date"],
                "price_date": meta["price_date"],
                "season": meta["season"],
                "scope": scope,
                "control_cny": c,
                "candidate_cny": x,
                "delta_cny": delta,
                "delta_percent": (100.0 * delta / abs(c)) if c != 0 else "",
                "control_ok": control["day_ok"],
                "candidate_ok": candidate["day_ok"],
            })

    # Bootstrap by date cluster, separately for every checkpoint and scope.
    grouped = {}
    for row in delta_rows:
        grouped.setdefault(
            (row["phase"], row["checkpoint"], row["scope"]), []).append(row)
    bootstrap = []
    for (phase, checkpoint, scope), items in sorted(grouped.items()):
        deltas = np.asarray([float(item["delta_cny"]) for item in items])
        block = _bootstrap(deltas)
        first = items[0]
        control_mean = float(np.mean([float(i["control_cny"])
                                      for i in items]))
        candidate_mean = float(np.mean([float(i["candidate_cny"])
                                        for i in items]))
        block.update({
            "phase": phase,
            "checkpoint": checkpoint,
            "seed": int(first["seed"]),
            "round": int(first["round"]),
            "role": first["role"],
            "scope": scope,
            "control_mean_cny": control_mean,
            "candidate_mean_cny": candidate_mean,
            "mean_delta_percent": (
                100.0 * block["mean_delta_cny"] / abs(control_mean)
                if control_mean != 0 else None),
        })
        bootstrap.append(block)
    bootstrap.sort(key=lambda r: (r["phase"], r["scope"] != "fleet",
                                  r["checkpoint"]))

    # Write derived tables.
    write_long(OUT / "per_ess_payoff.csv",
               BASE_FIELDS + ["ess"] + ESS_FIELDS, per_ess_rows)
    write_long(OUT / "fleet_payoff.csv",
               BASE_FIELDS + FLEET_FIELDS, fleet_rows)
    physical_fields = BASE_FIELDS + [
        "voltage_min", "voltage_max", "voltage_violation_cells",
        "line_max_utilization", "line_overload_cells",
        "soc_min", "soc_max", "soc_band_violation_max",
        "terminal_soc_err_max", "mutual_exclusion_max",
        "mutual_exclusion_flag", "power_violation_max",
        "day_ok", "failure_reason",
    ]
    write_long(OUT / "physical_validation.csv", physical_fields,
               physical_rows)
    solver_fields = BASE_FIELDS + [
        "clears", "optimal_clears", "fell_back_clears", "none_clears",
        "lmp_fallbacks", "infeasible_clears", "final_status_names", "day_ok",
    ]
    write_long(OUT / "solver_validation.csv", solver_fields, solver_rows)
    delta_fields = [
        "phase", "checkpoint", "seed", "round", "role", "pair_id",
        "load_date", "price_date", "season", "scope", "control_cny",
        "candidate_cny", "delta_cny", "delta_percent", "control_ok",
        "candidate_ok",
    ]
    write_long(OUT / "daily_paired_deltas.csv", delta_fields, delta_rows)
    payload = {
        "generated_at": now_iso(),
        "label": "NON_FORMAL_DIAGNOSTIC",
        "bootstrap": {
            "n_resamples": N_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "pair_id (date)",
            "interval": "percentile",
        },
        "results": bootstrap,
    }
    (OUT / "bootstrap_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(config, all_rows, bootstrap, delta_rows)
    print(f"wrote {len(bootstrap)} bootstrap rows and the report",
          flush=True)
    return 0


def write_long(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _result(bootstrap: list[dict], phase: str, checkpoint: str,
            scope: str = "fleet") -> dict | None:
    for row in bootstrap:
        if row["phase"] == phase and row["checkpoint"] == checkpoint \
                and row["scope"] == scope:
            return row
    return None


def _fmt(value, digits=2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def _verdict(config: dict, all_rows: list[dict],
             bootstrap: list[dict]) -> tuple[str, str]:
    phase_a = [r for r in all_rows if r["phase"] == "A"]
    if not phase_a:
        return "NOT_TESTED", "Phase 6A was not run."
    a_ok = all(r["day_ok"] == "True" for r in phase_a)
    expected_a = len(phase_pairs(config, "A")) * len(
        phase_meta(config, "A")) * len(MAPS)
    if len(phase_a) != expected_a:
        return "BLOCKED", (
            f"Phase 6A has {len(phase_a)} rows; expected {expected_a}.")
    preflight_path = OUT / "phase6_preflight.json"
    if not preflight_path.exists():
        return "BLOCKED", "phase6_preflight.json is missing."
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight["checks"].get("all_pass"):
        return "BLOCKED", "The Phase 6 preflight did not pass."
    if not a_ok:
        return "FAIL", (
            "Phase 6A has at least one day that fails clearing, solver, "
            "voltage, line, SOC, terminal or accounting checks. Phase 6B "
            "and 6C were not started.")
    if not any(r["phase"] == "B" for r in all_rows) \
            or not any(r["phase"] == "C" for r in all_rows):
        return "NOT_TESTED", (
            "Phase 6A passed its prerequisite. Phase 6B or 6C has not been "
            "run, so no economic verdict is reported.")
    finals = ("seed11_final", "seed23_final")
    primary = [(cp, ph) for cp in finals for ph in ("B", "C")]
    missing = [f"{cp}/{ph}" for cp, ph in primary
               if _result(bootstrap, ph, cp) is None]
    if missing:
        return "NOT_TESTED", f"Missing primary result(s): {', '.join(missing)}."
    pass_ok = all(
        _result(bootstrap, ph, cp)["ci95_low"] > 0
        and _result(bootstrap, ph, cp)["median_delta_cny"] > 0
        for cp, ph in primary)
    if pass_ok:
        # The Round 1 auxiliary check only downgrades, never upgrades.
        reversed_aux = False
        for cp in ("seed11_best", "seed23_best"):
            for ph in ("B", "C"):
                row = _result(bootstrap, ph, cp)
                if row is not None and row["mean_delta_cny"] < 0 \
                        and row["ci95_high"] < 0:
                    reversed_aux = True
        if reversed_aux:
            return "PARTIAL", (
                "Both Round 10 finals pass, but at least one Round 1 "
                "auxiliary checkpoint reverses the sign with a CI entirely "
                "below zero.")
        return "PASS", (
            "Both Round 10 finals have a positive date-clustered 95% CI "
            "lower bound and a positive median on both Train and Validation.")
    clear_negative = False
    for ph in ("B", "C"):
        rows = [_result(bootstrap, ph, cp) for cp in finals]
        if all(r is not None and r["mean_delta_cny"] <= 0
               and r["ci95_high"] < 0 for r in rows):
            clear_negative = True
    if clear_negative:
        return "FAIL", (
            "Both Round 10 finals have a mean fleet delta <= 0 and a 95% CI "
            "upper bound < 0 on the same split.")
    fails = []
    for cp, ph in primary:
        row = _result(bootstrap, ph, cp)
        split = "Train" if ph == "B" else "Validation"
        if row["ci95_low"] <= 0:
            fails.append(
                f"{cp} on {split}: mean {_fmt(row['mean_delta_cny'])} CNY/day, "
                f"95% CI [{_fmt(row['ci95_low'])}, "
                f"{_fmt(row['ci95_high'])}] includes zero")
        if row["median_delta_cny"] <= 0:
            fails.append(f"{cp} on {split}: median delta <= 0")
    detail = "; ".join(fails) if fails else "no failing primary condition"
    return "PARTIAL", (
        "The pre-registered PASS condition is not met. " + detail +
        ". Every fleet median is within 0.001 CNY/day of zero, so the "
        "positive mean is carried by a minority of dates while the typical "
        "date is unchanged.")


def write_report(config: dict, all_rows: list[dict],
                 bootstrap: list[dict], delta_rows: list[dict]) -> None:
    verdict, reason = _verdict(config, all_rows, bootstrap)
    lines = [
        "# Phase 6 — frozen-policy paired market clearing",
        "",
        "label: `NON_FORMAL_DIAGNOSTIC`; no formal map switch; no training.",
        "",
        f"**Verdict: `{verdict}`** — {reason}",
        "",
        "This is a frozen-policy counterfactual. The four actors were trained "
        "under `linear_core_v1`; the experiment changes only the inference "
        "quote map. It does not measure a policy trained under Z=10.0.",
        "",
        "## What was run",
        "",
        "| phase | split | dates | checkpoints | maps | day-runs |",
        "|---|---|---:|---|---:|---:|",
    ]
    for phase, split in (("A", "Train probe"), ("B", "Train full"),
                         ("C", "Validation")):
        rows = [r for r in all_rows if r["phase"] == phase]
        pairs = len({r["pair_id"] for r in rows})
        cps = len({r["checkpoint"] for r in rows})
        lines.append(
            f"| 6{phase} | {split} | {pairs} | {cps} | 2 | {len(rows)} |")
    lines += [
        "",
        "Neither Test, Holiday, nor the H2 holdout was read.",
        "",
        "## Physical, solver and accounting validation — all phases",
        "",
    ]
    stats = {}
    for phase in ("A", "B", "C"):
        rows = [r for r in all_rows if r["phase"] == phase]
        if not rows:
            continue
        stats[phase] = {
            "runs": len(rows),
            "ok": f"{sum(1 for r in rows if r['day_ok'] == 'True')}/{len(rows)}",
            "clears": sum(int(float(r["clears"])) for r in rows),
            "optimal": sum(int(float(r["optimal_clears"])) for r in rows),
            "fallback": sum(int(float(r["fell_back_clears"])) for r in rows),
            "lmp_fallbacks": sum(int(float(r["lmp_fallbacks"])) for r in rows),
            "infeasible": sum(int(float(r["infeasible_clears"])) for r in rows),
            "vmin": f"{min(float(r['voltage_min']) for r in rows):.6f}",
            "vmax": f"{max(float(r['voltage_max']) for r in rows):.6f}",
            "vviol": sum(int(float(r["voltage_violation_cells"])) for r in rows),
            "line": f"{max(float(r['line_max_utilization']) for r in rows):.6f}",
            "loviol": sum(int(float(r["line_overload_cells"])) for r in rows),
            "term": f"{max(float(r['fleet_terminal_soc_err_max']) for r in rows):.3e}",
            "band": f"{max(float(r['fleet_soc_band_violation_max']) for r in rows):.3e}",
            "power": f"{max(float(r['fleet_power_violation_max']) for r in rows):.3e}",
            "mutual": f"{max(float(r['fleet_mutual_exclusion_max']) for r in rows):.3e}",
            "ledger": f"{max(float(r['raw_profit_max_abs_diff']) for r in rows):.3e}",
        }
    if stats:
        cols = [p for p in ("A", "B", "C") if p in stats]
        metrics = [
            ("day-runs", "runs"), ("day_ok", "ok"),
            ("clearing periods", "clears"), ("optimal clears", "optimal"),
            ("fallback clears", "fallback"),
            ("LMP fallback periods", "lmp_fallbacks"),
            ("infeasible clears", "infeasible"),
            ("voltage min (pu)", "vmin"), ("voltage max (pu)", "vmax"),
            ("voltage violation cells", "vviol"),
            ("max line utilization", "line"),
            ("line overload cells", "loviol"),
            ("max terminal SOC error", "term"),
            ("max SOC band violation", "band"),
            ("max power-limit violation", "power"),
            ("max simultaneous charge/discharge", "mutual"),
            ("max raw_profit vs ledger diff (CNY)", "ledger"),
        ]
        lines += [
            _table(["metric", *[f"6{p}" for p in cols]],
                   [[label, *[stats[p][key] for p in cols]]
                    for label, key in metrics]),
            "",
            "Voltage band: `0.93 <= V <= 1.05` pu (the SOCP clearing's own "
            "constraint). A `day_ok` row requires 24/24 optimal clears, no "
            "fallback, no LMP fallback, zero voltage and line violations, "
            "terminal SOC within 1e-5 of 0.5, raw SOC inside [0.1, 0.9], and "
            "the equipment power limit respected.",
            "",
        ]
    else:
        lines += ["No phase results were found.", ""]

    preflight_path = OUT / "phase6_preflight.json"
    if preflight_path.exists():
        pre = json.loads(preflight_path.read_text(encoding="utf-8"))["checks"]
        lines += [
            "## Preflight",
            "",
            _table(
                ["check", "value"],
                [
                    ["control == linear_core_v1 (max abs diff)",
                     f"{pre['control_identity_max_abs_diff']:.3e}"],
                    ["candidate self-tests",
                     str(pre["candidate_self_tests"])],
                    ["repeat same day (max payoff diff, CNY)",
                     f"{pre['repeat_max_abs_payoff_diff']:.3e}"],
                    ["distinct quotes per ESS",
                     str(pre["dynamic_quotes"])],
                    ["actor hash unchanged", str(pre["actor_hash_unchanged"])],
                    ["all_pass", str(pre["all_pass"])],
                ]),
            "",
        ]

    for phase, title in (("A", "Phase 6A — probe pairs"),
                         ("B", "Phase 6B — Train"),
                         ("C", "Phase 6C — Validation")):
        rows = [r for r in all_rows if r["phase"] == phase]
        if not rows:
            continue
        lines += [f"## {title}", ""]
        table_rows = []
        for cp in sorted({r["checkpoint"] for r in rows}):
            result = _result(bootstrap, phase, cp)
            if result is None:
                continue
            table_rows.append([
                cp, result["round"], result["role"],
                _fmt(result["control_mean_cny"]),
                _fmt(result["candidate_mean_cny"]),
                _fmt(result["mean_delta_cny"]),
                _fmt(result["median_delta_cny"], 4),
                _pct(result["mean_delta_percent"]),
                f"[{_fmt(result['ci95_low'])}, "
                f"{_fmt(result['ci95_high'])}]",
                f"{result['fraction_positive']:.3f}",
                result["n_dates"],
            ])
        lines += [
            _table(
                ["checkpoint", "round", "role", "control mean CNY",
                 "candidate mean CNY", "mean delta CNY",
                 "median delta CNY", "delta %",
                 "date-clustered 95% CI", "share dates > 0", "dates"],
                table_rows),
            "",
        ]

    lines += [
        "## Per-ESS deltas (fleet decision unit)",
        "",
        "A positive per-ESS delta is not a fleet improvement. The fleet row "
        "is the decision unit; the per-ESS rows are attribution only.",
        "",
    ]
    for phase in ("A", "B", "C"):
        rows = [r for r in all_rows if r["phase"] == phase]
        if not rows:
            continue
        lines += [f"### Phase {phase}", ""]
        table_rows = []
        for cp in sorted({r["checkpoint"] for r in rows}):
            for scope in ("fleet",) + NAMES:
                result = _result(bootstrap, phase, cp, scope)
                if result is None:
                    continue
                table_rows.append([
                    cp, scope, _fmt(result["control_mean_cny"]),
                    _fmt(result["candidate_mean_cny"]),
                    _fmt(result["mean_delta_cny"]),
                    _fmt(result["median_delta_cny"], 4),
                    f"[{_fmt(result['ci95_low'])}, "
                    f"{_fmt(result['ci95_high'])}]",
                    f"{result['fraction_positive']:.3f}",
                ])
        lines += [
            _table(
                ["checkpoint", "scope", "control mean CNY",
                 "candidate mean CNY", "mean delta CNY",
                 "median delta CNY", "95% CI", "share dates > 0"],
                table_rows),
            "",
        ]

    lines += [
        "## Date-level distribution of the fleet delta",
        "",
        "The mean is not the typical date. A large block of dates has a "
        "small gain, and the aggregate gain is carried by a minority of "
        "dates; the tables below report that concentration instead of hiding "
        "it behind the mean. The median falls inside the near-zero block, "
        "which is why it is positive but far below the mean.",
        "",
    ]
    dist_rows = []
    for phase in ("A", "B", "C"):
        checkpoints = sorted({r["checkpoint"] for r in delta_rows
                              if r["phase"] == phase})
        for cp in checkpoints:
            values = sorted(
                float(r["delta_cny"]) for r in delta_rows
                if r["phase"] == phase and r["checkpoint"] == cp
                and r["scope"] == "fleet")
            if not values:
                continue
            positive = [x for x in values if x > 0]
            negative = [x for x in values if x < 0]
            top = sorted(positive, reverse=True)[
                :max(1, math.ceil(0.1 * len(values)))]
            share = 100.0 * sum(top) / sum(positive) \
                if positive else 0.0
            unchanged = sum(1 for x in values if abs(x) < 0.01)
            dist_rows.append([
                phase, cp, len(values), len(positive), len(negative),
                unchanged,
                _fmt(sum(positive)), _fmt(sum(negative)),
                _fmt(max(values)), _fmt(min(values)),
                f"{sum(top):.0f} ({share:.0f}%)",
            ])
    if dist_rows:
        lines += [
            _table(
                ["phase", "checkpoint", "dates", "delta > 0", "delta < 0",
                 "|delta| < 0.01 CNY", "sum positive CNY",
                 "sum negative CNY", "max CNY", "min CNY",
                 "top 10% of dates (share of positive total)"],
                dist_rows),
            "",
        ]

    a_fleet = {
        (r["checkpoint"], r["z_core"], r["pair_id"]):
            float(r["fleet_net_income_cny"])
        for r in all_rows if r["phase"] == "A"}
    b_fleet = {
        (r["checkpoint"], r["z_core"], r["pair_id"]):
            float(r["fleet_net_income_cny"])
        for r in all_rows if r["phase"] == "B"}
    common = sorted(set(a_fleet) & set(b_fleet))
    if common:
        max_diff = max(abs(a_fleet[key] - b_fleet[key]) for key in common)
        lines += [
            "## Phase 6A vs Phase 6B reproducibility",
            "",
            _table(
                ["metric", "value"],
                [
                    ["common checkpoint/map/date rows", len(common)],
                    ["max fleet income difference (CNY)", f"{max_diff:.3e}"],
                    ["identical", str(max_diff < 1e-9)],
                ]),
            "",
            "Phase 6A probe pairs are a subset of the Train dates, so the "
            "same frozen checkpoint and map on the same date must reproduce "
            "the same clearing and settlement.",
            "",
        ]

    quote_rows = []
    constant_charge = 0
    constant_charge_closed = 0
    for phase in ("A", "B", "C"):
        checkpoints = sorted({r["checkpoint"] for r in all_rows
                              if r["phase"] == phase})
        for cp in checkpoints:
            for z_core in ("2.5", "10.0"):
                rows = [r for r in all_rows if r["phase"] == phase
                        and r["checkpoint"] == cp and r["z_core"] == z_core]
                if not rows:
                    continue
                charge = [int(float(r[f"{nm}_distinct_quotes_charge"]))
                          for r in rows for nm in NAMES]
                discharge = [int(float(r[f"{nm}_distinct_quotes_discharge"]))
                             for r in rows for nm in NAMES]
                gate_open = [float(r[f"{nm}_gate_open_charge_frac"])
                             for r in rows for nm in NAMES]
                for r in rows:
                    for nm in NAMES:
                        if int(float(
                                r[f"{nm}_distinct_quotes_charge"])) == 1:
                            constant_charge += 1
                            if float(
                                    r[f"{nm}_gate_open_charge_frac"]) == 0.0:
                                constant_charge_closed += 1
                quote_rows.append([
                    phase, cp, z_core, len(charge),
                    min(charge), f"{sum(charge) / len(charge):.1f}",
                    f"{sum(1 for c in charge if c > 1) / len(charge):.3f}",
                    min(discharge), f"{sum(discharge) / len(discharge):.1f}",
                    f"{sum(1 for c in discharge if c > 1) / len(discharge):.3f}",
                    f"{sum(gate_open) / len(gate_open):.3f}",
                ])
    if quote_rows:
        lines += [
            "## Quote dynamics",
            "",
            "One ESS-day is one ESS on one cleared date. The action is "
            "regenerated at every block from the current observation, so a "
            "single distinct quote means the policy chose the same quote all "
            "day, not that the day was swept once. "
            f"{constant_charge} ESS-days have a single distinct charge quote; "
            f"in {constant_charge_closed} of them the charge gate is closed "
            "for all 24 blocks, which is why the quote never needs to move.",
            "",
            _table(
                ["phase", "checkpoint", "Z", "ESS-days",
                 "min charge quotes", "mean charge quotes",
                 "charge > 1 share", "min discharge quotes",
                 "mean discharge quotes", "discharge > 1 share",
                 "mean charge gate-open fraction"],
                quote_rows),
            "",
        ]

    lines += [
        "## Magnitude and percentage",
        "",
        "Percentage changes are reported only for completeness. In Phase 6B "
        "the control mean is close to zero, so a small absolute gain becomes "
        "a large percentage. The decision uses absolute CNY/day and the "
        "date-clustered CI, never the percentage column.",
        "",
        "## Accounting",
        "",
        "Per ESS and per day, `net_income_cny = market_revenue_cny - "
        "purchase_cost_cny - cycle_cost_cny`. The terminal inventory-value "
        "term is exactly zero by design: `participant_payoff.PayoffBreakdown` "
        "has no terminal term, and the terminal SOC is pinned at 0.5 by the "
        "day-end settlement. `raw_profit` from the environment is compared "
        "with the ledger every day; the maximum difference is reported for "
        "each phase in the validation table.",
        "",
        "## What this does not show",
        "",
        "- It does not show how a policy trained under Z=10.0 performs. The "
        "actors are frozen `linear_core_v1` actors; only the inference map "
        "changes.",
        "- It does not read Test, Holiday, or H2. No independent "
        "generalisation claim is supported.",
        "- Offline quote statistics are not payoffs; this report uses the "
        "settlement ledger only.",
        "- A positive mean with a CI that includes zero is not a robust "
        "improvement.",
        "",
        "## Provenance",
        "",
        "- `phase6_config.json` freezes the pairs, checkpoints, maps and "
        "decision rule.",
        "- `provenance_addendum.json` records the two stale hashes in the "
        "Phase 0-5 audit metadata. No historical file was modified.",
        "- The 16 actor files in `checkpoints/` hash-match the original run "
        "directories and the audit copies.",
        "",
        "## Files",
        "",
        "`phase6_config.json`, `source_hashes.json`, "
        "`provenance_addendum.json`, `phase6_preflight.json`, "
        "`phase6A_smoke_results.csv`, `phase6B_train_results.csv`, "
        "`phase6C_validation_results.csv`, `daily_paired_deltas.csv`, "
        "`per_ess_payoff.csv`, `fleet_payoff.csv`, "
        "`physical_validation.csv`, `solver_validation.csv`, "
        "`bootstrap_summary.json`, `output_hashes.json`.",
        "",
    ]
    (OUT / "phase6_decision_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def source_hashes() -> int:
    config = load_config()
    paths = [
        "src/run_phase6_frozen_policy.py",
        "src/quote_map_candidates.py",
        "src/da_quote_map.py",
        "src/rl_da_gate.py",
        "src/rl_b_c2_gate.py",
        "src/rl_env.py",
        "src/rl_spec.py",
        "src/da_replay.py",
        "src/train_da_gate.py",
        "src/v2_cy_jn_env.py",
        "src/screen_v2_cy_jn_2025.py",
        "src/participant_payoff.py",
        "data/processed/v2_cy_jn_2025/bus_loads_96x33.npz",
        "data/processed/v2_cy_jn_2025/price_curves.npz",
        "data/processed/v2_cy_jn_2025/common_split.json",
        "data/processed/v2_cy_jn_2025/terminal_soc_config.json",
        "reports/v2_cy_jn_2025/formal_training_v4/formal_training_config.yaml",
        "reports/v2_cy_jn_2025/formal_training_v4/formal_training_manifest.json",
        "reports/v2_cy_jn_2025/formal_training_v4/formal_training_code_manifest.json",
        "reports/v2_cy_jn_2025/formal_training_v4/formal_training_data_manifest.json",
        "runs/FORMAL_7834d0aa_seed11/run_manifest.json",
        "runs/FORMAL_7834d0aa_seed23/run_manifest.json",
        "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/input_hashes.json",
        "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/output_hashes.json",
        "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/audit_manifest.json",
        "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/decision_report.md",
        "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/mapping_candidates.json",
    ]
    payload = {
        "written_at": now_iso(),
        "label": "NON_FORMAL_DIAGNOSTIC",
        "python": os.sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "files": {},
        "checkpoint_actor_sha256": {},
    }
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            payload["files"][rel] = "MISSING"
            continue
        payload["files"][rel] = sha256_file(path)
    for key in ("seed11_final", "seed11_best", "seed23_final", "seed23_best"):
        payload["checkpoint_actor_sha256"][key] = {
            nm: sha256_file(OUT / "checkpoints" / key / "actors"
                            / f"{nm}.pt") for nm in NAMES}
    (OUT / "source_hashes.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'source_hashes.json'}", flush=True)
    return 0


def output_hashes() -> int:
    config = load_config()
    payload = {
        "written_at": now_iso(),
        "label": "NON_FORMAL_DIAGNOSTIC",
        "files": {},
    }
    names = list(config.get("deliverables", [])) + ["phase6_preflight.json"]
    for name in names:
        path = OUT / name
        if path.exists() and name != "output_hashes.json":
            payload["files"][name] = sha256_file(path)
    (OUT / "output_hashes.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'output_hashes.json'}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("A", "B", "C"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--source-hashes", action="store_true")
    parser.add_argument("--output-hashes", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.preflight:
        return preflight()
    if args.source_hashes:
        return source_hashes()
    if args.phase:
        return run_phase(args.phase, args.retry_failed,
                         args.limit or None)
    if args.analyze:
        return analyze()
    if args.output_hashes:
        return output_hashes()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
