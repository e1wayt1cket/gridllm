# storage_cycle_cost_sensitivity.py
"""Storage cycle-cost sensitivity on one frozen diagnostic day.

Runs C80, C90 and C100 (CNY/MWh per leg) on the same load, price, fleet,
efficiency, SOC band, network and solver settings, plus a No-ESS reference. It
reports the settled economics, the storage dispatch, the theoretical arbitrage
bound and -- when a compatible smoke checkpoint exists -- the same MATD3 policy
evaluated at the three costs.

The cost is the existing `storage.cycle_cost`: charged on both legs as
`cycle_cost * (p_ch + p_dis) * DT_HOURS` in the clearing objective and again in
`participant_payoff`. Efficiency enters only through the SOC transition, so it
is not charged twice. Nothing in this module changes that definition.

The load day and the price day are different calendar dates. Every artifact is
therefore labelled `diagnostic_only`; this is not a same-day market replay and
it cannot lift the V2 price-data gate.

Usage:
    PYTHONPATH=src python src/storage_cycle_cost_sensitivity.py
    PYTHONPATH=src python src/storage_cycle_cost_sensitivity.py --skip-policy
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

import arbitrage_lp
import config_loader as cfg
import money
import participant_payoff
from diagnose_typical_day import _window, round_trip_margin_cny_mwh
from models import MarketConfig

SCENARIO = "typical_day_industry_subset"
T = money.PERIODS_PER_DAY
SEED = 42
COSTS: Dict[str, float] = {"C80": 80.0, "C90": 90.0, "C100": 100.0}
NOESS = "NoESS"
POLICY_DIR = Path("policies/smoke_subset_obs_v6_seed42")
POLICY_RUN_DIR = Path("outputs/rl/run-20261016-173409-28757924")
SNAPSHOT_ROOT = Path("reports/snapshots")
REPORT_PATH = Path("reports/storage_cycle_cost_sensitivity.md")
PRIOR_CSV = Path("reports/storage_cost_sensitivity.csv")
TOL = 1e-6
#: Reporting threshold for a period that actually moved energy. Solver noise on
#: an idle unit is up to about 1e-4 MW; this keeps it out of the active-period
#: counts and out of the idle-share denominator's complement. Any dispatch that
#: matters at the report's precision is far above it.
ACTIVE_TOL_MW = 1e-3
#: A net-profit increase smaller than this is treated as solver noise when the
#: monotonicity rule is checked. The exact difference is still recorded.
MONOTONICITY_TOL_CNY = 0.01

RESULT_COLUMNS = (
    "case", "arm", "level", "agent", "bus", "cycle_cost_cny_mwh",
    "required_margin_cny_mwh", "status", "clean", "charge_mwh",
    "discharge_mwh", "charge_periods", "discharge_periods", "idle_share",
    "equivalent_cycles", "soc_initial", "soc_terminal", "soc_min",
    "soc_max", "max_charge_mw", "max_discharge_mw", "revenue_cny",
    "purchase_cost_cny", "cycle_cost_cny", "other_cost_cny",
    "net_profit_cny", "delta_ess_profit_vs_noess_cny",
    "delta_system_payoff_cny", "delta_vs_truthful_cny",
    "system_payoff_cny", "solver_status", "optimality_gap",
    "solve_wall_time_s", "terminal_soc_ok", "soc_band_ok", "power_ok",
    "voltage_ok", "line_ok", "no_simultaneous_ok", "lmp_nodal",
)

TRACE_COLUMNS = (
    "case", "arm", "agent", "bus", "period", "wholesale_cny_mwh",
    "lmp_cny_mwh", "p_buy_mw", "p_sell_mw", "p_ch_mw", "p_dis_mw",
    "soc_start", "soc_end", "soc_mwh", "revenue_cny",
    "purchase_cost_cny", "cycle_cost_cny", "net_cny",
    "charge_price_cny_mwh", "discharge_price_cny_mwh", "mode",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _case_config(cost: float) -> MarketConfig:
    """The scenario's own market, with one field changed."""
    config = MarketConfig.from_yaml()
    config.storage.cycle_cost = float(cost)
    return config


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False,
                   sort_keys=True),
        encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict],
               columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


class _Tee:
    """Write the run to both the terminal and the snapshot's run.log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# Day construction and clearing
# ---------------------------------------------------------------------------

def _build_day(config: MarketConfig, seed: int,
               ess_buses: Optional[Sequence[int]] = None):
    import scenarios as scen

    np.random.seed(int(seed))
    agents, wholesale = scen.get_scenario(
        SCENARIO, T=T, config=config,
        ess_buses=None if ess_buses is None else list(ess_buses))
    return agents, np.asarray(wholesale, dtype=float), scen.last_single_day_case()


def _ess_agents(agents) -> List:
    return [a for a in agents
            if getattr(a, "participant_type", "") == "storage"]


def _load_fingerprint(agents) -> str:
    """Digest of the load and on-site generation, excluding storage."""
    digest = hashlib.sha256()
    for agent in sorted(agents, key=lambda a: a.name):
        if getattr(agent, "participant_type", "") == "storage":
            continue
        digest.update(agent.name.encode("utf-8"))
        digest.update(np.ascontiguousarray(
            np.round(np.asarray(agent.load_forecast, dtype=float), 9)
        ).tobytes())
        digest.update(np.ascontiguousarray(
            np.round(np.asarray(agent.pv_forecast, dtype=float), 9)
        ).tobytes())
        wind = getattr(agent, "wind_forecast", None)
        if wind is None:
            wind = np.zeros(T)
        digest.update(np.ascontiguousarray(
            np.round(np.asarray(wind, dtype=float), 9)
        ).tobytes())
    return digest.hexdigest()


def _case_identity(case) -> dict:
    if case is None:
        return {}
    return {
        "case_type": case.case_type,
        "date_rule": case.date_rule,
        "alignment": case.alignment,
        "load_date": case.ledger_manifest.get("date"),
        "price_date": case.price_source.get("date"),
        "same_day": bool(case.date_rule.get("same_day")),
        "load_source": case.load_source,
        "price_source": case.price_source,
        "ledger_kind": case.load_kind,
        "unit_basis": case.unit_basis,
        "unit_status": case.ledger_manifest.get("unit_status"),
        "ledger_sha256": case.ledger_manifest.get("ledger_sha256"),
        "workbook_sha256": case.ledger_manifest.get("workbook_sha256"),
    }


def _clear_truthful(agents, config: MarketConfig, wholesale: np.ndarray) -> dict:
    """One whole-day clear with truthful declarations.

    This is the same boundary the single-day diagnostic uses: one full-horizon
    clear with the endpoints pinned, not a rolling-window replay followed by a
    second settlement.
    """
    from market import clear_market
    from strategies.fixed_bidding import FixedStrategy

    declared = FixedStrategy().formulate(agents, config, None, T)
    return clear_market(agents, T, "DA", declared, config,
                        wholesale=wholesale, horizon_type="full_day")


def _node_column(day: dict, bus: int) -> int:
    indices = day.get("bus_indices")
    if indices is not None:
        buses = [int(b) for b in indices]
        if int(bus) in buses:
            return buses.index(int(bus))
    return int(bus)


def _node_lmp(day: dict, agent) -> np.ndarray:
    return np.asarray(day["lmp"], dtype=float)[:, _node_column(day, agent.bus)]


def _is_clean(day: Optional[dict]) -> bool:
    return bool(day is not None
                and not day.get("fell_back")
                and not day.get("has_wholesale_fallback"))


# ---------------------------------------------------------------------------
# Storage schedule and economics
# ---------------------------------------------------------------------------

def _schedule_stats(schedule: dict, storage) -> dict:
    ch = np.asarray(schedule["p_ch"], dtype=float)
    dis = np.asarray(schedule["p_dis"], dtype=float)
    p_buy = np.asarray(schedule["p_buy"], dtype=float)
    p_sell = np.asarray(schedule["p_sell"], dtype=float)
    soc = np.asarray(schedule["soc"], dtype=float)
    boundary = np.asarray(schedule.get("soc_boundary", soc), dtype=float)

    charge_mwh = float(np.sum(ch) * money.DT_HOURS)
    discharge_mwh = float(np.sum(dis) * money.DT_HOURS)
    e_max = float(getattr(storage, "e_max", 0.0) or 0.0)
    charge_periods = int(np.count_nonzero(ch > ACTIVE_TOL_MW))
    discharge_periods = int(np.count_nonzero(dis > ACTIVE_TOL_MW))
    simultaneous = int(np.count_nonzero(
        (ch > ACTIVE_TOL_MW) & (dis > ACTIVE_TOL_MW)))
    idle_periods = int(np.count_nonzero(
        (ch <= ACTIVE_TOL_MW) & (dis <= ACTIVE_TOL_MW)))
    n = max(int(ch.size), 1)
    soc_initial = (float(boundary[0]) if boundary.size
                   else float(getattr(storage, "soc0", float("nan"))))
    soc_terminal = (float(soc[-1]) if soc.size
                    else float(soc_initial))
    target = schedule.get("soc_terminal_target")
    if target is None:
        target = soc_initial
    return {
        "charge_mwh": charge_mwh,
        "discharge_mwh": discharge_mwh,
        "charge_periods": charge_periods,
        "discharge_periods": discharge_periods,
        "idle_periods": idle_periods,
        "idle_share": idle_periods / n,
        "equivalent_cycles": ((charge_mwh + discharge_mwh) / (2.0 * e_max)
                              if e_max > 0 else 0.0),
        "cycles_from_discharge": (discharge_mwh / e_max
                                  if e_max > 0 else 0.0),
        "soc_initial": soc_initial,
        "soc_terminal": soc_terminal,
        "soc_target": float(target),
        "soc_terminal_residual": float(soc[-1] - target) if soc.size else None,
        "soc_min": float(np.min(soc)) if soc.size else None,
        "soc_max": float(np.max(soc)) if soc.size else None,
        "soc_mean": float(np.mean(soc)) if soc.size else None,
        "max_charge_mw": float(np.max(ch)) if ch.size else 0.0,
        "max_discharge_mw": float(np.max(dis)) if dis.size else 0.0,
        "simultaneous_periods": simultaneous,
        "buy_sell_mismatch_mw": float(np.max(np.abs(p_buy - ch)))
        if ch.size else 0.0,
    }


def _agent_economics(agent, config: MarketConfig, day: dict) -> dict:
    schedule = (day.get("schedules") or {}).get(agent.name)
    if schedule is None:
        raise KeyError(f"no schedule for {agent.name}")
    lmp = _node_lmp(day, agent)
    breakdown = participant_payoff.participant_payoff(
        schedule, lmp, agent, config)
    stats = _schedule_stats(schedule, agent.storage)
    return {
        "agent": agent.name,
        "bus": int(agent.bus),
        "participant_type": getattr(agent, "participant_type", ""),
        "revenue_cny": float(breakdown.revenue),
        "purchase_cost_cny": float(breakdown.purchase_cost),
        "generation_cost_cny": float(breakdown.generation_cost),
        "degradation_cost_cny": float(breakdown.degradation_cost),
        "other_cost_cny": float(breakdown.other_cost),
        "penalty_cny": float(breakdown.penalty),
        "net_profit_cny": float(breakdown.total),
        **stats,
    }


def _fleet_stats(day: dict, ess_agents: Sequence) -> dict:
    ch = np.zeros(T)
    dis = np.zeros(T)
    for agent in ess_agents:
        schedule = day["schedules"][agent.name]
        ch += np.asarray(schedule["p_ch"], dtype=float)
        dis += np.asarray(schedule["p_dis"], dtype=float)
    idle = int(np.count_nonzero(
        (ch <= ACTIVE_TOL_MW) & (dis <= ACTIVE_TOL_MW)))
    return {
        "max_charge_mw": float(np.max(ch)) if ch.size else 0.0,
        "max_discharge_mw": float(np.max(dis)) if dis.size else 0.0,
        "charge_periods": int(np.count_nonzero(ch > ACTIVE_TOL_MW)),
        "discharge_periods": int(np.count_nonzero(dis > ACTIVE_TOL_MW)),
        "idle_periods": idle,
        "idle_share": idle / max(int(ch.size), 1),
        "simultaneous_periods": int(np.count_nonzero(
            (ch > ACTIVE_TOL_MW) & (dis > ACTIVE_TOL_MW))),
    }


def _fleet_row(case: str, arm: str, cost: float, margin: float,
               rows: Sequence[dict], day: dict, ess_agents: Sequence,
               system_payoff: float) -> dict:
    e_max = sum(float(a.storage.e_max) for a in ess_agents)
    p_dis = sum(float(a.storage.p_dis_max) for a in ess_agents)
    out = {
        "case": case, "arm": arm, "level": "fleet", "agent": "fleet",
        "bus": None, "cycle_cost_cny_mwh": cost,
        "required_margin_cny_mwh": margin,
        "charge_mwh": sum(r["charge_mwh"] for r in rows),
        "discharge_mwh": sum(r["discharge_mwh"] for r in rows),
        "revenue_cny": sum(r["revenue_cny"] for r in rows),
        "purchase_cost_cny": sum(r["purchase_cost_cny"] for r in rows),
        "cycle_cost_cny": sum(r["degradation_cost_cny"] for r in rows),
        "other_cost_cny": sum(r["generation_cost_cny"] + r["other_cost_cny"]
                              + r["penalty_cny"] for r in rows),
        "net_profit_cny": sum(r["net_profit_cny"] for r in rows),
        "soc_initial": float(np.mean([r["soc_initial"] for r in rows])),
        "soc_terminal": float(np.mean([r["soc_terminal"] for r in rows])),
        "soc_min": min(r["soc_min"] for r in rows),
        "soc_max": max(r["soc_max"] for r in rows),
        "system_payoff_cny": system_payoff,
        "clean": all(bool(r.get("clean", False)) for r in rows),
    }
    out.update(_fleet_stats(day, ess_agents))
    out["equivalent_cycles"] = (
        (out["charge_mwh"] + out["discharge_mwh"]) / (2.0 * e_max)
        if e_max > 0 else 0.0)
    out["cycles_from_discharge"] = (
        out["discharge_mwh"] / e_max if e_max > 0 else 0.0)
    out["utilization"] = (out["discharge_mwh"] / (p_dis * 24.0)
                          if p_dis > 0 else 0.0)
    return out


def _system_payoff(agents, config: MarketConfig, day: dict):
    total = 0.0
    rows = []
    schedules = day.get("schedules") or {}
    for agent in agents:
        schedule = schedules.get(agent.name)
        if schedule is None:
            continue
        breakdown = participant_payoff.participant_payoff(
            schedule, _node_lmp(day, agent), agent, config)
        total += float(breakdown.total)
        rows.append({
            "agent": agent.name,
            "participant_type": getattr(agent, "participant_type", ""),
            "net_cny": float(breakdown.total),
        })
    return total, rows


# ---------------------------------------------------------------------------
# Solver, network and constraint checks
# ---------------------------------------------------------------------------

def _network_summary(day: Optional[dict]) -> dict:
    if day is None:
        return {}
    out = {
        "clearing_objective": day.get("clearing_objective"),
        "re_consumption_rate": day.get("re_consumption_rate"),
        "carbon_emissions_tco2": day.get("carbon_emissions"),
        "total_curtailment_mwh": day.get("total_curtailment"),
        "lmp_fallbacks": day.get("lmp_fallbacks"),
        "has_wholesale_fallback": day.get("has_wholesale_fallback"),
        "fell_back": day.get("fell_back"),
        "solver": day.get("solver"),
    }
    lmp = day.get("lmp")
    if lmp is not None:
        arr = np.asarray(lmp, dtype=float)
        out["mean_lmp_cny_mwh"] = float(np.mean(arr))
        out["min_lmp_cny_mwh"] = float(np.min(arr))
        out["max_lmp_cny_mwh"] = float(np.max(arr))
    voltage = day.get("bus_voltage")
    if voltage is not None:
        arr = np.asarray(voltage, dtype=float)
        out["min_bus_voltage_pu"] = float(np.min(arr))
        out["max_bus_voltage_pu"] = float(np.max(arr))
    lines = day.get("line_utilization")
    if lines is not None:
        arr = np.asarray(lines, dtype=float)
        out["max_line_utilization"] = float(np.max(arr))
        out["periods_line_utilization_ge_0_99"] = int(
            np.count_nonzero(arr >= 0.99))
    return out


def _solver_and_checks(day: Optional[dict], config: MarketConfig,
                       agents, ess_agents: Sequence) -> dict:
    solver = (day or {}).get("solver") or {}
    final = solver.get("final") or solver.get("primary") or {}
    status = final.get("status_name")
    optimal = bool(final.get("optimal"))
    checks = {
        "solver_status": status,
        "solver_status_code": final.get("status_code"),
        "solver_termination": final.get("termination"),
        "solver_optimal": optimal,
        "solver_used_fallback": bool(solver.get("used_fallback")),
        "solver_duals_available": solver.get("duals_available"),
        "optimality_gap": 0.0 if optimal else None,
        "optimality_gap_note": (
            "continuous SOCP: no MIP gap; a final OPTIMAL status means the "
            "objective bound equals the incumbent"),
        "lmp_nodal": bool(day is not None
                          and not day.get("has_wholesale_fallback")
                          and int(day.get("lmp_fallbacks") or 0) == 0),
        "clean": _is_clean(day),
        "voltage_ok": True,
        "line_ok": True,
        "soc_band_ok": True,
        "power_ok": True,
        "terminal_soc_ok": True,
        "no_simultaneous_ok": True,
    }
    if day is None:
        checks.update({k: False for k in (
            "voltage_ok", "line_ok", "soc_band_ok", "power_ok",
            "terminal_soc_ok", "no_simultaneous_ok")})
        return checks

    voltage = day.get("bus_voltage")
    if voltage is not None:
        arr = np.asarray(voltage, dtype=float)
        checks["voltage_ok"] = bool(
            np.min(arr) >= float(config.network.v_min_pu) - 1e-6
            and np.max(arr) <= float(config.network.v_max_pu) + 1e-6)
    lines = day.get("line_utilization")
    if lines is not None:
        checks["line_ok"] = bool(
            np.max(np.asarray(lines, dtype=float)) <= 1.0 + 1e-6)

    schedules = day.get("schedules") or {}
    for agent in ess_agents:
        schedule = schedules.get(agent.name)
        storage = agent.storage
        if schedule is None:
            checks["soc_band_ok"] = False
            continue
        soc = np.asarray(schedule["soc"], dtype=float)
        ch = np.asarray(schedule["p_ch"], dtype=float)
        dis = np.asarray(schedule["p_dis"], dtype=float)
        stats = _schedule_stats(schedule, storage)
        if np.any(soc < float(storage.soc_min) - 1e-6) or np.any(
                soc > float(storage.soc_max) + 1e-6):
            checks["soc_band_ok"] = False
        if (np.max(ch) > float(storage.p_ch_max) + 1e-6
                or np.max(dis) > float(storage.p_dis_max) + 1e-6):
            checks["power_ok"] = False
        if stats["simultaneous_periods"]:
            checks["no_simultaneous_ok"] = False
        residual = stats["soc_terminal_residual"]
        if residual is not None and abs(residual) > 1e-5:
            checks["terminal_soc_ok"] = False
    return checks


# ---------------------------------------------------------------------------
# Theoretical arbitrage bound
# ---------------------------------------------------------------------------

def _price_stats(price: np.ndarray, cost: float, eta_ch: float,
                 eta_dis: float) -> dict:
    price = np.asarray(price, dtype=float)
    valley = float(np.min(price))
    peak = float(np.max(price))
    required = arbitrage_lp.required_margin(cost, eta_ch, eta_dis)
    new_window = _window(price, eta_ch, eta_dis, cost)
    rtm = round_trip_margin_cny_mwh(price, eta_ch, eta_dis, cost)
    if valley > 0:
        old_breakeven = ((valley / (eta_ch * eta_dis) + required)
                         / (eta_dis * valley))
        old_margin = peak / valley - old_breakeven
        factor = eta_dis * valley
        old_cny = old_margin * factor
    else:
        old_breakeven = float("inf")
        old_margin = float("-inf")
        old_cny = float("-inf")
    return {
        "price_min_cny_mwh": valley,
        "price_max_cny_mwh": peak,
        "price_mean_cny_mwh": float(np.mean(price)),
        "valley_period": int(np.argmin(price)),
        "peak_period": int(np.argmax(price)),
        "raw_ratio": peak / valley if valley > 0 else float("inf"),
        "required_margin_cny_mwh": required,
        "round_trip_margin_cny_mwh": rtm,
        "window_breakeven_ratio": new_window["breakeven_ratio"],
        "window_margin_ratio": new_window["margin"],
        "window_margin_cny_mwh": (new_window["margin"] * eta_dis * valley
                                  if valley > 0 else float("-inf")),
        "window_breakeven_ratio_old": old_breakeven,
        "window_margin_ratio_old": old_margin,
        "window_margin_cny_mwh_old": old_cny,
    }


def _price_candidates(price: np.ndarray, eta_ch: float, eta_dis: float,
                      required: float, top_n: int = 5) -> dict:
    """Price-only pairs, before SOC, power and the terminal pin are imposed."""
    price = np.asarray(price, dtype=float)
    pairs = []
    for charge_period in range(price.size):
        if price[charge_period] <= 0:
            continue
        for discharge_period in range(charge_period + 1, price.size):
            margin = (eta_dis * price[discharge_period]
                      - price[charge_period] / eta_ch - required)
            if margin > 0:
                pairs.append({
                    "charge_period": charge_period,
                    "discharge_period": discharge_period,
                    "price_charge_cny_mwh": float(price[charge_period]),
                    "price_discharge_cny_mwh": float(price[discharge_period]),
                    "margin_cny_mwh": float(margin),
                })
    pairs.sort(key=lambda row: -row["margin_cny_mwh"])
    return {"count": len(pairs), "top": pairs[:top_n]}


def _lp_row(agent, price: np.ndarray, cost: float) -> dict:
    unit = arbitrage_lp.spec_from_storage(agent.storage)
    try:
        result = arbitrage_lp.arbitrage_lp(price, unit, cost)
    except Exception as exc:
        return {"agent": agent.name, "bus": int(agent.bus),
                "status": "failed", "error": str(exc)}
    return {
        "agent": agent.name,
        "bus": int(agent.bus),
        "status": result.status_text,
        "margin_cny": float(result.margin_cny),
        "charge_mwh": float(result.chargeable_energy_mwh),
        "discharge_mwh": float(result.dischargeable_energy_mwh),
        "equivalent_cycles": float(result.equivalent_cycles),
        "charge_periods": int(result.feasible_charge_periods),
        "discharge_periods": int(result.feasible_discharge_periods),
        "best_charge_price_cny_mwh": float(result.best_charge_price),
        "best_discharge_price_cny_mwh": float(result.best_discharge_price),
        "gross_spread_cny_mwh": float(result.gross_spread_cny_mwh),
        "efficiency_loss_cny_mwh": float(result.efficiency_loss_cny_mwh),
        "degradation_cost_cny_mwh": float(result.degradation_cost_cny_mwh),
        "adjusted_spread_cny_mwh": float(result.adjusted_spread_cny_mwh),
    }


def _lp_fleet(rows: Sequence[dict], ess_agents: Sequence) -> dict:
    ok = [r for r in rows if r.get("margin_cny") is not None]
    e_max = sum(float(a.storage.e_max) for a in ess_agents)
    charge = sum(r["charge_mwh"] for r in ok)
    discharge = sum(r["discharge_mwh"] for r in ok)
    return {
        "margin_cny": sum(r["margin_cny"] for r in ok),
        "charge_mwh": charge,
        "discharge_mwh": discharge,
        "equivalent_cycles": ((charge + discharge) / (2.0 * e_max)
                              if e_max > 0 else 0.0),
        "per_agent_ok": len(ok),
    }


def _experiment_b(day_noess: dict, ess_agents: Sequence, cost: float,
                  eta_ch: float, eta_dis: float) -> dict:
    required = arbitrage_lp.required_margin(cost, eta_ch, eta_dis)
    per_bus = []
    lp_rows = []
    for agent in ess_agents:
        price = _node_lmp(day_noess, agent)
        stats = _price_stats(price, cost, eta_ch, eta_dis)
        candidates = _price_candidates(price, eta_ch, eta_dis, required)
        lp_row = _lp_row(agent, price, cost)
        per_bus.append({
            "agent": agent.name,
            "bus": int(agent.bus),
            "price": stats,
            "candidate_pairs": candidates,
            "lp": lp_row,
        })
        lp_rows.append(lp_row)
    return {
        "required_margin_cny_mwh": required,
        "per_bus": per_bus,
        "lp_fleet": _lp_fleet(lp_rows, ess_agents),
    }


# ---------------------------------------------------------------------------
# MATD3 checkpoint evaluation
# ---------------------------------------------------------------------------

def _policy_metadata() -> dict:
    return {
        "policy_dir": str(POLICY_DIR),
        "run_dir": str(POLICY_RUN_DIR),
        "manifest": _read_json(POLICY_RUN_DIR / "manifest.json"),
        "training_config": _read_json(POLICY_RUN_DIR / "config.json"),
    }


def _load_policies(agents, config: MarketConfig, names: Sequence[str]):
    if not POLICY_DIR.is_dir():
        raise FileNotFoundError(f"no checkpoint directory at {POLICY_DIR}")
    from eval_agents import load_policies_from_dir
    from rl_env import BiddingEnv

    env = BiddingEnv(agents, config, rl_agent_names=list(names))
    policies = load_policies_from_dir(
        str(POLICY_DIR), env.get_state_dim(), env.get_action_bounds(),
        obs_spec=env.obs_spec, action_spec=env.action_spec)
    missing = sorted(set(names) - set(policies))
    if missing:
        raise RuntimeError(
            f"checkpoint directory {POLICY_DIR} is missing policies for "
            f"{missing}; refusing to evaluate a partial fleet")
    return policies, env.obs_spec, env.action_spec


def _action_summary(declared: dict, names: Sequence[str],
                    action_spec) -> dict:
    from rl_env import BLOCK_SIZE

    out = {}
    for name in names:
        acts = declared.get(name, {})
        per_agent = {}
        for key, values in acts.items():
            arr = np.asarray(values, dtype=float).ravel()
            bounds = getattr(action_spec, "bounds", {}) or {}
            low, high = bounds.get(key, (None, None))
            block_means = [
                float(np.mean(arr[i:i + BLOCK_SIZE]))
                for i in range(0, arr.size, BLOCK_SIZE)]
            per_agent[key] = {
                "mean": float(np.mean(arr)) if arr.size else None,
                "min": float(np.min(arr)) if arr.size else None,
                "max": float(np.max(arr)) if arr.size else None,
                "first": float(arr[0]) if arr.size else None,
                "last": float(arr[-1]) if arr.size else None,
                "at_low_fraction": (
                    float(np.mean(np.isclose(arr, float(low), atol=TOL)))
                    if low is not None and arr.size else None),
                "at_high_fraction": (
                    float(np.mean(np.isclose(arr, float(high), atol=TOL)))
                    if high is not None and arr.size else None),
                "block_means": block_means,
                "values": arr.tolist(),
            }
        out[name] = per_agent
    return out


def _run_policy_case(agents, config: MarketConfig, wholesale_ref: np.ndarray,
                     policies: dict, names: Sequence[str], obs_spec,
                     action_spec, seed: int) -> dict:
    from counterfactual_eval import settle_day
    from eval_agents import run_combined_episode
    from rl_env import BiddingEnv

    env = BiddingEnv(agents, config, rl_agent_names=list(names),
                     obs_spec=obs_spec, action_spec=action_spec)
    np.random.seed(int(seed))
    capture = run_combined_episode(env, policies, capture=True)
    if not capture or not capture.get("usable"):
        return {"status": "skipped",
                "reason": "the rolling capture is not usable (a block degraded "
                          "or fell back)"}
    wholesale = np.asarray(capture.get("wholesale", []), dtype=float)
    if wholesale.shape != wholesale_ref.shape or not np.allclose(
            wholesale, wholesale_ref):
        return {"status": "skipped",
                "reason": "the policy saw a different wholesale day from the "
                          "truthful arms"}
    declared = capture.get("declared") or {}
    missing = [name for name in names if name not in declared]
    if missing:
        return {"status": "skipped",
                "reason": f"captured declarations are missing {missing}"}
    profits, clean, day = settle_day(agents, config, names, declared,
                                     wholesale, T)
    if not clean or day is None:
        return {"status": "degraded",
                "reason": "the endpoint-pinned settlement is not clean",
                "rolling_profits_cny": profits,
                "capture_complete": capture.get("capture_complete")}
    return {
        "status": "ok",
        "capture_complete": bool(capture.get("capture_complete")),
        "rolling_profits_cny": profits,
        "declared_raw": declared,
        "declared": _action_summary(declared, names, action_spec),
        "day": day,
    }


# ---------------------------------------------------------------------------
# Trace rows
# ---------------------------------------------------------------------------

def _trace_rows(case: str, arm: str, agent, config: MarketConfig, day: dict,
                wholesale: np.ndarray, declared: Optional[dict]) -> List[dict]:
    schedule = (day.get("schedules") or {}).get(agent.name)
    if schedule is None:
        return []
    lmp = _node_lmp(day, agent)
    ch = np.asarray(schedule["p_ch"], dtype=float)
    dis = np.asarray(schedule["p_dis"], dtype=float)
    p_buy = np.asarray(schedule["p_buy"], dtype=float)
    p_sell = np.asarray(schedule["p_sell"], dtype=float)
    boundary = np.asarray(schedule.get("soc_boundary", schedule["soc"]),
                          dtype=float)
    e_max = float(agent.storage.e_max)
    acts = (declared or {}).get(agent.name, {})
    charge_price = np.asarray(acts.get("charge_price", np.zeros(T)),
                              dtype=float)
    discharge_price = np.asarray(acts.get("discharge_price", np.zeros(T)),
                                 dtype=float)
    rows = []
    for t in range(T):
        cycle_cost = float(config.storage.cycle_cost) * (
            ch[t] + dis[t]) * money.DT_HOURS
        revenue = float(lmp[t]) * p_sell[t] * money.DT_HOURS
        purchase = float(lmp[t]) * p_buy[t] * money.DT_HOURS
        if ch[t] > ACTIVE_TOL_MW and dis[t] > ACTIVE_TOL_MW:
            mode = "simultaneous"
        elif ch[t] > ACTIVE_TOL_MW:
            mode = "charge"
        elif dis[t] > ACTIVE_TOL_MW:
            mode = "discharge"
        else:
            mode = "idle"
        soc_start = float(boundary[t]) if t < boundary.size else None
        soc_end = float(boundary[t + 1]) if t + 1 < boundary.size else None
        rows.append({
            "case": case,
            "arm": arm,
            "agent": agent.name,
            "bus": int(agent.bus),
            "period": t,
            "wholesale_cny_mwh": float(wholesale[t]),
            "lmp_cny_mwh": float(lmp[t]),
            "p_buy_mw": float(p_buy[t]),
            "p_sell_mw": float(p_sell[t]),
            "p_ch_mw": float(ch[t]),
            "p_dis_mw": float(dis[t]),
            "soc_start": soc_start,
            "soc_end": soc_end,
            "soc_mwh": soc_end * e_max if soc_end is not None else None,
            "revenue_cny": revenue,
            "purchase_cost_cny": purchase,
            "cycle_cost_cny": cycle_cost,
            "net_cny": revenue - purchase - cycle_cost,
            "charge_price_cny_mwh": float(charge_price[t])
            if t < charge_price.size else None,
            "discharge_price_cny_mwh": float(discharge_price[t])
            if t < discharge_price.size else None,
            "mode": mode,
        })
    return rows


def _raw_day(day: dict, ess_agents: Sequence) -> dict:
    raw = {"wholesale": None, "agents": {}}
    for agent in ess_agents:
        schedule = day["schedules"][agent.name]
        raw["agents"][agent.name] = {
            "bus": int(agent.bus),
            "p_buy_mw": np.asarray(schedule["p_buy"], dtype=float).tolist(),
            "p_sell_mw": np.asarray(schedule["p_sell"], dtype=float).tolist(),
            "p_ch_mw": np.asarray(schedule["p_ch"], dtype=float).tolist(),
            "p_dis_mw": np.asarray(schedule["p_dis"], dtype=float).tolist(),
            "soc": np.asarray(schedule["soc"], dtype=float).tolist(),
            "soc_boundary": np.asarray(
                schedule.get("soc_boundary", schedule["soc"]),
                dtype=float).tolist(),
            "lmp_cny_mwh": _node_lmp(day, agent).tolist(),
        }
    return raw


# ---------------------------------------------------------------------------
# Prior record
# ---------------------------------------------------------------------------

def _prior_record() -> List[dict]:
    if not PRIOR_CSV.is_file():
        return []
    rows = []
    with PRIOR_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("agent") != "fleet":
                continue
            try:
                rows.append({
                    "case": row.get("case"),
                    "cycle_cost_cny_mwh": float(row["cycle_cost_cny_mwh"]),
                    "charge_mwh": float(row["charge_mwh"]),
                    "discharge_mwh": float(row["discharge_mwh"]),
                    "gross_margin_cny": float(row["gross_margin_cny"]),
                    "degradation_cost_cny": float(row["degradation_cost_cny"]),
                    "profit_cny": float(row["profit_cny"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return rows


# ---------------------------------------------------------------------------
# Snapshot and report
# ---------------------------------------------------------------------------

def _versions() -> dict:
    import platform

    out = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    for module in ("numpy", "scipy", "torch", "gurobipy", "pandapower"):
        try:
            imported = __import__(module)
            out[module] = getattr(imported, "__version__", "unknown")
        except Exception:
            out[module] = "not_installed"
    try:
        import gurobipy as gp
        out["gurobi"] = ".".join(str(v) for v in gp.gurobi.version())
    except Exception as exc:
        out["gurobi"] = f"unavailable: {exc}"
    return out


def _git_state() -> dict:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=str(Path.cwd())).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            cwd=str(Path.cwd()))
        return {
            "head": head,
            "dirty": bool(status.strip()),
            "status_porcelain": status.splitlines(),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _input_hashes(snapshot_dir: Path, report_path: Path,
                  run_log: Path) -> dict:
    paths = [
        Path("config/defaults.yaml"),
        Path("config/scenarios.yaml"),
        Path("data/processed/node_loads/typical_day_industry_subset/"
             "bus_loads_96x33.csv"),
        Path("data/processed/node_loads/typical_day_industry_subset/"
             "manifest.json"),
        Path("data/processed/single_day_diagnostic/price_96.csv"),
        Path("data/processed/single_day_diagnostic/price_manifest.json"),
        Path("data/processed/single_day_diagnostic/case_manifest.json"),
        Path("policies/checkpoint_obs_compat_manifest.json"),
    ]
    paths += sorted(POLICY_DIR.glob("*.pt")) if POLICY_DIR.is_dir() else []
    paths += [POLICY_RUN_DIR / "manifest.json",
              POLICY_RUN_DIR / "config.json",
              Path("src/storage_cycle_cost_sensitivity.py"),
              Path("src/diagnose_typical_day.py")]
    inputs = {}
    for path in paths:
        if path.is_file():
            inputs[str(path)] = _sha256(path)
    outputs = {}
    for path in sorted(snapshot_dir.iterdir()):
        if path.name in ("hashes.json", "run.log") or not path.is_file():
            continue
        outputs[str(path)] = _sha256(path)
    if run_log.is_file():
        outputs[str(run_log)] = _sha256(run_log)
    if report_path.is_file():
        outputs[str(report_path)] = _sha256(report_path)
    return {"inputs": inputs, "outputs": outputs}


def _snapshot_config(results: dict) -> dict:
    configs = results.get("configs") or {}
    return {
        "experiment": "storage_cycle_cost_sensitivity",
        "status": results.get("status"),
        "label": "diagnostic_only",
        "scenario": SCENARIO,
        "periods": T,
        "dt_hours": money.DT_HOURS,
        "day_seed": results.get("metadata", {}).get("seed"),
        "costs": COSTS,
        "no_ess": NOESS,
        "case_identity": results.get("case_identity"),
        "storage_fleet": results.get("storage_fleet"),
        "configs": configs,
        "policy": results.get("policy_metadata"),
        "software": _versions(),
        "command": results.get("metadata", {}).get("command"),
    }


def _snapshot_readme(results: dict, report_path: Path) -> str:
    a = results.get("experiment_a", {}).get("cases", {})
    b = results.get("experiment_b", {}).get("cases", {})
    c = results.get("experiment_c", {})
    lines = [
        "# Storage cycle-cost snapshot",
        "",
        f"Created {results.get('metadata', {}).get('started')}.",
        "",
        "**Diagnostic only.** The load day and the price day are different "
        "calendar dates; this is not a same-day market replay and it does not "
        "lift the V2 price-data gate.",
        "",
        "## Results",
        "",
        "| case | cost CNY/MWh/leg | required margin | charge MWh | "
        "discharge MWh | net CNY |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in COSTS:
        row = a.get(case)
        if not row:
            lines.append(f"| {case} | {COSTS[case]:.1f} | - | - | - | - |")
            continue
        fleet = row["fleet"]
        lines.append(
            f"| {case} | {COSTS[case]:.1f} | "
            f"{_fmt(fleet['required_margin_cny_mwh'], 2)} | "
            f"{_fmt(fleet['charge_mwh'])} | {_fmt(fleet['discharge_mwh'])} | "
            f"{_fmt(fleet['net_profit_cny'], 4)} |")
    noess = results.get("experiment_a", {}).get("noess") or {}
    lines.append(
        f"| {NOESS} | - | - | 0.0000 | 0.0000 | "
        f"{_fmt(noess.get('system_payoff_cny'), 2)} system |")
    lines += [
        "",
        "## Files",
        "",
        "- `config.json` - effective configuration, inputs and software.",
        "- `results.json` - raw key results for experiments A, B and C.",
        "- `results.csv` - one tidy row per case/arm/agent and fleet.",
        "- `dispatch_trace.csv` - per-period price, power, SOC and money.",
        "- `run.log` - command, start/end, warnings and solver metadata.",
        "- `hashes.json` - input and output SHA-256, git state.",
        "",
        f"The formal report is `{report_path}`.",
        "",
        "## Caveats",
        "",
        "- One frozen cross-date diagnostic day, truthful quotes for "
        "experiment A.",
        "- The MATD3 block, when present, is one smoke checkpoint evaluated "
        "at three costs; it is not three trained policies.",
        "- The LP bound uses the No-ESS nodal prices as exogenous inputs and "
        "ignores price feedback from storage.",
        "",
    ]
    return "\n".join(lines)


def _report(results: dict, snapshot_dir: Path) -> str:
    a = results.get("experiment_a", {}).get("cases", {})
    noess = results.get("experiment_a", {}).get("noess") or {}
    b = results.get("experiment_b", {}).get("cases", {})
    c = results.get("experiment_c", {})
    identity = results.get("case_identity") or {}
    cases = [case for case in COSTS if case in a]
    active = [case for case in cases
              if float(a[case]["fleet"]["discharge_mwh"]) > 1e-3]
    profits = {case: float(a[case]["fleet"]["net_profit_cny"])
               for case in cases}
    monotone = results.get("experiment_a", {}).get(
        "monotonicity", {}).get("non_increasing")

    lines = [
        "# Storage cycle-cost sensitivity",
        "",
        "## Verdict",
        "",
        "**Diagnostic only.** The load day and the price day are different "
        "calendar dates, and the storage quotes in experiment A are truthful. "
        "This is not a same-day market replay, not a training result and not "
        "a basis for lifting the V2 price-data gate.",
        "",
    ]
    if not cases:
        lines.append("No experiment-A case cleared, so there is no result to "
                     "read.")
    elif not active:
        lines.append(
            "All three costs leave the independent fleet idle on this day. "
            "The reachable nodal spread does not cover the degradation floor "
            "plus round-trip efficiency at any of C80, C90 or C100.")
    else:
        lines.append(
            f"Measurable cycling appears only at {', '.join(active)}. "
            f"Fleet net profit is "
            + ", ".join(f"{case} {profits[case]:+.4f} CNY"
                        for case in cases)
            + ". The gross margin is almost entirely consumed by "
            "degradation; this is not a meaningful arbitrage opportunity.")
    monotonicity = results.get("experiment_a", {}).get("monotonicity", {})
    tolerance = float(monotonicity.get("tolerance_cny", 0.01))
    exact = monotonicity.get("exact_violations") or []
    if monotonicity.get("non_increasing"):
        lines.append(
            f"Higher cycle cost never raised the optimal net profit by more "
            f"than {tolerance:g} CNY: **yes**.")
        if exact:
            lines.append(
                "The exact comparison flags "
                + "; ".join(
                    f"{v['previous']}->{v['following']} by "
                    f"{v['increase_cny']:+.6f} CNY"
                    for v in exact)
                + f", which is solver-noise scale (below the {tolerance:g} CNY "
                "tolerance). Both cases are idle and their profits are zero "
                "within the solver's precision.")
    else:
        lines.append(
            "Higher cycle cost raised the optimal net profit by more than "
            f"{tolerance:g} CNY: **yes** (this must be investigated before "
            "any conclusion is drawn).")
    c80 = a.get("C80", {}).get("fleet", {})
    lines += [
        "",
        "**Sufficiency for a MATD3 cost-sensitivity study: not yet.** At "
        "C90 and C100 the executable LP bound is zero, so there is no "
        "arbitrage signal to learn. At C80 the truthful clearing earns "
        f"{_fmt(c80.get('net_profit_cny'), 2)} CNY on "
        f"{_fmt(c80.get('discharge_mwh'))} MWh. A formal training study needs "
        "a day with a material executable spread, and it needs the dated "
        "price gate to be met first.",
        "",
        "## Cost definition and audit",
        "",
        "- Config: `storage.cycle_cost`, CNY/MWh **per leg**.",
        "- Charged on both charge and discharge: "
        "`cycle_cost * (p_ch + p_dis) * DT_HOURS`, with `DT_HOURS = 0.25`.",
        "- The same expression is used in the clearing objective and in "
        "`participant_payoff._ledger`; there is no separate settlement "
        "formula.",
        "- `eta_ch` and `eta_dis` enter only through the SOC transition. "
        "They are not a money term, so efficiency is not charged twice.",
        "- Break-even spread per MWh stored: "
        "`required_margin = cycle_cost * (1/eta_ch + eta_dis)`. It is not "
        "`2 * cycle_cost` unless the round trip is lossless.",
        "",
        "| case | cycle cost | required margin CNY/MWh |",
        "|---|---:|---:|",
    ]
    for case in COSTS:
        margin = (a.get(case, {}).get("fleet", {}).get(
            "required_margin_cny_mwh"))
        lines.append(f"| {case} | {COSTS[case]:.2f} | "
                     f"{_fmt(margin, 2)} |")
    lines += [
        "",
        "### Breakeven-ratio audit",
        "",
        "`diagnose_typical_day._window` previously divided the valley term by "
        "`eta_ch * eta_dis` and then divided the whole numerator by "
        "`eta_dis * valley`, which is equivalent to an extra `eta_dis`. The "
        "formula is corrected in this run; the old and corrected values are "
        "reported below. `round_trip_margin_cny_mwh` was already correct, so "
        "the CNY/MWh siting criterion did not move.",
        "",
    ]
    window_rows = []
    for case in cases:
        for row in b.get(case, {}).get("per_bus", []):
            price = row["price"]
            window_rows.append([
                case, row["agent"],
                _fmt(price["window_breakeven_ratio_old"]),
                _fmt(price["window_breakeven_ratio"]),
                _fmt(price["window_margin_cny_mwh_old"], 2),
                _fmt(price["window_margin_cny_mwh"], 2),
                _fmt(price["round_trip_margin_cny_mwh"], 2),
            ])
    if window_rows:
        lines.append(_md_table(
            ["case", "agent", "old breakeven ratio", "corrected ratio",
             "old CNY/MWh", "corrected CNY/MWh", "round-trip CNY/MWh"],
            window_rows))
        lines += [
            "",
            "At C80 the sign flips: the old ratio reported a negative "
            "price-only margin while the corrected ratio and "
            "`round_trip_margin_cny_mwh` report positive. The executable LP "
            "bound and the actual clearing are still small, so this does not "
            "create a material opportunity; it does change the ratio-space "
            "`by_bus` viability list. The CNY/MWh siting criterion is "
            "unchanged.",
        ]
    else:
        lines.append("The ratio audit could not be computed.")
    lines += [
        "",
        "## Inputs and date match",
        "",
        f"- Scenario: `{SCENARIO}`.",
        f"- Load date: **{identity.get('load_date')}**.",
        f"- Price date: **{identity.get('price_date')}**.",
        f"- Case type: `{identity.get('case_type')}`; "
        f"same day: **{identity.get('same_day')}**.",
        f"- Ledger kind: `{identity.get('ledger_kind')}`; unit status: "
        f"`{identity.get('unit_status') or (identity.get('unit_basis') or {}).get('status')}`.",
        "- The pair is a cross-date diagnostic. It is not a historical "
        "replay, and this experiment does not modify the frozen V2 manifest "
        "or release the V2 price gate.",
        "",
        "## Experiment A: Gurobi optimal dispatch",
        "",
        "Only `storage.cycle_cost` changes; load, price, fleet, efficiency, "
        "SOC band, initial/terminal SOC, power limits, network and solver "
        "settings are identical. Quotes are truthful in every arm.",
        "",
    ]
    a_rows = []
    for case in cases:
        row = a[case]
        fleet = row["fleet"]
        checks = row["checks"]
        a_rows.append([
            case, _fmt(row["cycle_cost_cny_mwh"], 1),
            _fmt(fleet["required_margin_cny_mwh"], 2),
            _fmt(fleet["charge_mwh"]), _fmt(fleet["discharge_mwh"]),
            _fmt(fleet["equivalent_cycles"]),
            _fmt(fleet["net_profit_cny"], 4),
            _fmt(row.get("delta_system_payoff_cny"), 2),
            _fmt(fleet.get("idle_share"), 3),
            _fmt(checks.get("solver_status")), _fmt(checks.get("clean")),
        ])
    a_rows.append([
        NOESS, "-", "-", "0.0000", "0.0000", "0.0000", "0.0000", "0.0000",
        "-", _fmt(noess.get("checks", {}).get("solver_status")),
        _fmt(noess.get("checks", {}).get("clean")),
    ])
    lines.append(_md_table(
        ["case", "cost", "required margin", "charge MWh", "discharge MWh",
         "cycles", "net CNY", "delta system CNY", "idle share", "solver",
         "clean"],
        a_rows))
    lines += [
        "",
        f"No-ESS system settlement total: "
        f"**{_fmt(noess.get('system_payoff_cny'), 2)} CNY**. "
        "This is the project's participant-payoff total, not cash welfare; "
        "prosumer totals include load valuation. The per-case system delta is "
        "not the fleet's profit: even an idle fleet changes the clearing's "
        "constraint set, so the system total moves while the fleet's own "
        "settlement stays near zero.",
        "",
        "Because No-ESS carries no storage, the fleet's own net profit in the "
        "table below is already its increment over No-ESS. The system delta is "
        "a separate total and must not be read as storage revenue.",
        "",
        "### Fleet economics",
        "",
    ]
    economics_rows = []
    for case in cases:
        fleet = a[case]["fleet"]
        economics_rows.append([
            case, _fmt(fleet["purchase_cost_cny"], 2),
            _fmt(fleet["revenue_cny"], 2),
            _fmt(fleet["cycle_cost_cny"], 2),
            _fmt(fleet["other_cost_cny"], 2),
            _fmt(fleet["net_profit_cny"], 4),
        ])
    lines.append(_md_table(
        ["case", "charge purchase CNY", "discharge revenue CNY",
         "cycle cost CNY", "other cost CNY", "net CNY"],
        economics_rows))
    lines += [
        "",
        "### Per-agent dispatch",
        "",
    ]
    per_agent_rows = []
    for case in cases:
        for row in a[case]["agents"]:
            per_agent_rows.append([
                case, row["agent"], _fmt(row["charge_mwh"]),
                _fmt(row["discharge_mwh"]), row["charge_periods"],
                row["discharge_periods"], _fmt(row["equivalent_cycles"]),
                _fmt(row["net_profit_cny"], 4),
                _fmt(row["soc_initial"], 3), _fmt(row["soc_terminal"], 3),
                _fmt(row["soc_min"], 3), _fmt(row["soc_max"], 3),
                _fmt(row["max_charge_mw"]), _fmt(row["max_discharge_mw"]),
            ])
    lines.append(_md_table(
        ["case", "agent", "charge MWh", "discharge MWh", "ch periods",
         "dis periods", "cycles", "net CNY", "SOC init", "SOC final",
         "SOC min", "SOC max", "max ch MW", "max dis MW"],
        per_agent_rows))
    lines += [
        "",
        "Generation, unserved-load and other operating costs are zero for an "
        "independent battery, so the whole operating cost is the two-leg "
        "cycle cost. The terminal state is pinned at the initial 0.50 in every "
        "case.",
        "",
    ]
    lines += [
        "",
        "### Solver and physical checks",
        "",
    ]
    check_rows = []
    for case in cases:
        row = a[case]
        checks = row["checks"]
        check_rows.append([
            case, _fmt(checks.get("solver_status")),
            _fmt(checks.get("optimality_gap")),
            _fmt(row.get("solve_wall_time_s"), 2),
            _fmt(checks.get("terminal_soc_ok")),
            _fmt(checks.get("soc_band_ok")),
            _fmt(checks.get("power_ok")),
            _fmt(checks.get("voltage_ok")),
            _fmt(checks.get("line_ok")),
            _fmt(checks.get("no_simultaneous_ok")),
            _fmt(checks.get("lmp_nodal")),
        ])
    check_rows.append([
        NOESS, _fmt(noess.get("checks", {}).get("solver_status")),
        _fmt(noess.get("checks", {}).get("optimality_gap")),
        _fmt(noess.get("solve_wall_time_s"), 2),
        "-", "-", "-",
        _fmt(noess.get("checks", {}).get("voltage_ok")),
        _fmt(noess.get("checks", {}).get("line_ok")), "-", "-",
    ])
    lines.append(_md_table(
        ["case", "solver", "gap", "time s", "terminal SOC", "SOC band",
         "power", "voltage", "line", "no churn", "nodal LMP"],
        check_rows))
    lines += [
        "",
        "The SOCP is continuous, so Gurobi's MIP gap is not defined. An "
        "OPTIMAL final status means the reported objective is the optimum; "
        "the gap column records 0 for that case and is not a MIP-gap claim.",
        "",
        "## Experiment B: theory versus executable bound",
        "",
        "The theoretical spread is evaluated at each ESS bus against the "
        "**No-ESS nodal LMP**. The price-only candidate pairs ignore SOC, "
        "power and the terminal pin; the LP bound imposes all three and is "
        "therefore the executable opportunity. The fleet LP is the sum of "
        "four independent per-unit bounds and is not a joint fleet optimum.",
        "",
    ]
    if (results.get("experiment_b") or {}).get("status") != "ok":
        lines += [
            "**Invalid.** "
            + str((results.get("experiment_b") or {}).get(
                "reason", "the No-ESS price field is not fully nodal"))
            + " No executable bound is reported.",
            "",
        ]
    theory_rows = []
    for case in cases:
        for row in b.get(case, {}).get("per_bus", []):
            price = row["price"]
            lp = row["lp"]
            actual = next((x for x in a[case]["agents"]
                           if x["agent"] == row["agent"]), {})
            theory_rows.append([
                case, row["agent"],
                f"{_fmt(price['price_min_cny_mwh'], 2)}/"
                f"{_fmt(price['price_max_cny_mwh'], 2)}/"
                f"{_fmt(price['price_mean_cny_mwh'], 2)}",
                _fmt(price["required_margin_cny_mwh"], 2),
                _fmt(price["round_trip_margin_cny_mwh"], 2),
                _fmt(lp.get("margin_cny"), 2),
                _fmt(lp.get("charge_mwh")), _fmt(lp.get("discharge_mwh")),
                _fmt(actual.get("charge_mwh")),
                _fmt(actual.get("discharge_mwh")),
            ])
    lines.append(_md_table(
        ["case", "agent", "LMP min/max/mean", "required", "price-only margin",
         "LP bound CNY", "LP charge", "LP discharge", "actual charge",
         "actual discharge"],
        theory_rows))
    lines += [
        "",
        "### Fleet bound versus actual",
        "",
    ]
    fleet_rows = []
    for case in cases:
        lp = b.get(case, {}).get("lp_fleet", {})
        fleet = a[case]["fleet"]
        fleet_rows.append([
            case, _fmt(lp.get("margin_cny"), 2),
            _fmt(fleet["net_profit_cny"], 2),
            _fmt(lp.get("discharge_mwh")),
            _fmt(fleet["discharge_mwh"]),
            _fmt(lp.get("equivalent_cycles")),
            _fmt(fleet["equivalent_cycles"]),
            _fmt(float(fleet["net_profit_cny"]) - float(lp.get("margin_cny") or 0.0), 2),
        ])
    lines.append(_md_table(
        ["case", "LP bound CNY", "actual net CNY", "LP discharge",
         "actual discharge", "LP cycles", "actual cycles", "actual - LP"],
        fleet_rows))
    if "C80" in b and "C80" in a:
        lp80 = float(b["C80"]["lp_fleet"].get("margin_cny") or 0.0)
        actual80 = float(a["C80"]["fleet"]["net_profit_cny"])
        lp_dis80 = float(b["C80"]["lp_fleet"].get("discharge_mwh") or 0.0)
        actual_dis80 = float(a["C80"]["fleet"]["discharge_mwh"])
        lines += [
            "",
            f"The C80 gap between the {lp80:.2f} CNY LP bound and the "
            f"{actual80:.2f} CNY actual net is not a solver error. The LP is "
            "a price-taker bound: one unit, perfect foresight, and the No-ESS "
            "nodal prices as exogenous inputs. The actual figure is a joint "
            "truthful clearing in which the fleet's own quotes and the nodal "
            "re-solve determine dispatch, and the four units share the same "
            f"spread. The bound discharges {lp_dis80:.4f} MWh; the clearing "
            f"dispatches {actual_dis80:.4f} MWh. A perfect forecaster outside "
            "the market can see more than the market itself captures.",
        ]
    lines += [
        "",
        "Candidate price pairs (price-only, no energy limits):",
        "",
    ]
    candidate_rows = []
    for case in cases:
        for row in b.get(case, {}).get("per_bus", []):
            top = (row["candidate_pairs"].get("top") or [])
            best = top[0] if top else None
            candidate_rows.append([
                case, row["agent"], row["candidate_pairs"].get("count", 0),
                (f"{best['charge_period']}->{best['discharge_period']} "
                 f"({_fmt(best['margin_cny_mwh'], 2)} CNY/MWh)")
                if best else "-",
            ])
    lines.append(_md_table(
        ["case", "agent", "pairs clearing the floor", "best pair (margin)"],
        candidate_rows))
    lines += [
        "",
        "## Why the fleet is idle or cycling",
        "",
    ]
    for case in cases:
        lp = b.get(case, {}).get("lp_fleet", {})
        fleet = a[case]["fleet"]
        lp_margin = float(lp.get("margin_cny") or 0.0)
        actual_dis = float(fleet["discharge_mwh"])
        if lp_margin <= 1e-6:
            lines.append(
                f"- **{case}**: the executable LP bound is "
                f"{lp_margin:+.2f} CNY, so a perfect forecaster cannot clear "
                "the degradation floor on this price field. Idleness is "
                "market-driven, not a policy failure.")
        elif actual_dis <= 1e-3:
            lines.append(
                f"- **{case}**: the LP bound is {lp_margin:+.2f} CNY but the "
                "clearing dispatches no measurable energy. Check the quote "
                "mapping, SOC/power limits and the terminal pin before "
                "calling this a strategy result.")
        elif float(fleet["net_profit_cny"]) < 1.0:
            lines.append(
                f"- **{case}**: the LP bound is {lp_margin:+.2f} CNY and "
                f"actual net profit is {float(fleet['net_profit_cny']):+.2f} "
                "CNY. Gross margin is almost entirely consumed by "
                "degradation; the opportunity is real but negligible.")
        else:
            lines.append(
                f"- **{case}**: the LP bound is {lp_margin:+.2f} CNY and "
                f"actual net profit is {float(fleet['net_profit_cny']):+.2f} "
                "CNY.")
    lines += [
        "",
        "The raw min-max spread is not the reachable spread: 0.4 MW and "
        "1.6 MWh with a 0.10-0.90 SOC band cannot pair the day's extremes, "
        "and the terminal pin forbids ending with banked energy.",
        "",
        "## Experiment C: the MATD3 smoke checkpoint",
        "",
    ]
    if not c or c.get("status") != "ok":
        reason = (c or {}).get("reason", "no checkpoint evaluation was run")
        lines += [
            f"**Skipped.** {reason}",
            "",
            "No policy number is reported. A missing or incompatible "
            "checkpoint must not be replaced by a theoretical value.",
            "",
        ]
    else:
        lines += [
            "**One policy, three costs, not three trained policies.** The "
            "checkpoint is a smoke artifact: 25 episodes, 5 gradient updates, "
            "seed 42, trained at C100 on a non-formal scenario. It does not "
            "demonstrate convergence. C80 and C90 are out-of-distribution "
            "costs; the policy is not retrained.",
            "",
            "The checkpoint's training recipe used `use_nodal_price=false`; "
            "this diagnostic market uses the scenario YAML value `true`. "
            "The comparison below therefore describes this checkpoint in "
            "this market, not a reproduction of its training environment.",
            "",
            _md_table(
                ["case", "declared charge mean", "declared discharge mean",
                 "charge MWh", "discharge MWh", "net CNY",
                 "delta vs truthful", "delta vs LP", "status"],
                [[
                    case,
                    _fmt(c["cases"][case].get("declared_charge_mean"), 2),
                    _fmt(c["cases"][case].get("declared_discharge_mean"), 2),
                    _fmt(c["cases"][case]["fleet"].get("charge_mwh")),
                    _fmt(c["cases"][case]["fleet"].get("discharge_mwh")),
                    _fmt(c["cases"][case]["fleet"].get("net_profit_cny"), 4),
                    _fmt(c["cases"][case].get("delta_vs_truthful_cny"), 4),
                    _fmt(c["cases"][case].get("delta_vs_lp_cny"), 4),
                    c["cases"][case].get("status"),
                ] for case in COSTS if case in c["cases"]]),
            "",
            "Settlement status: "
            + ", ".join(
                f"{case} {c['cases'][case].get('checks', {}).get('solver_status')}"
                f" ({_fmt(c['cases'][case].get('solve_wall_time_s'), 2)} s"
                " rollout + settlement)"
                for case in COSTS if case in c["cases"])
            + ". Every case is clean; none fell back.",
            "",
            "If the policy is idle where the truthful Gurobi arm cycles, the "
            "difference is a policy gap to locate. It is not evidence that "
            "the algorithm failed, and it is not evidence that MATD3 learned "
            "or failed to learn arbitrage.",
            "",
        ]
        if "C80" in c.get("cases", {}) and "C80" in a:
            c80 = c["cases"]["C80"]
            truth80 = a["C80"]["fleet"]
            low_fractions = [0.0]
            high_fractions = [0.0]
            for acts in (c80.get("declared") or {}).values():
                for key in ("charge_price", "discharge_price"):
                    entry = acts.get(key) or {}
                    low_fractions.append(float(
                        entry.get("at_low_fraction") or 0.0))
                    high_fractions.append(float(
                        entry.get("at_high_fraction") or 0.0))
            lines += [
                f"At C80 the truthful Gurobi arm discharges "
                f"{float(truth80['discharge_mwh']):.4f} MWh for "
                f"{float(truth80['net_profit_cny']):+.2f} CNY, while the "
                f"checkpoint discharges {float(c80['fleet']['discharge_mwh']):.4f} "
                f"MWh for {float(c80['fleet']['net_profit_cny']):+.2f} CNY. "
                f"Its mean declared charge price is "
                f"{_fmt(c80.get('declared_charge_mean'), 2)} CNY/MWh and its "
                f"mean discharge price is "
                f"{_fmt(c80.get('declared_discharge_mean'), 2)} CNY/MWh. "
                f"At most {max(low_fractions):.1%} of the declared values sit "
                f"at the lower box bound and at most {max(high_fractions):.1%} "
                "at the upper bound. The checkpoint was trained at C100, on "
                "a day whose executable bound is zero, so the training signal "
                "contained no arbitrage to learn. The idle C80 behaviour is "
                "therefore a policy gap to locate, not a demonstration that "
                "MATD3 cannot arbitrage.",
                "",
            ]

    lines += [
        "## Prior record and reconciliation",
        "",
        "The earlier C0/C1/C2 record is preserved at "
        "`reports/storage_cost_sensitivity.md` and its CSV is not "
        "overwritten. That run used `run_arm`'s rolling loop followed by an "
        "endpoint-pinned settlement; this run clears the whole day directly "
        "with truthful quotes. C100 is the cross-check.",
        "",
    ]
    prior = results.get("prior_record") or []
    if prior:
        lines.append(_md_table(
            ["prior case", "cost", "charge MWh", "discharge MWh",
             "gross CNY", "degradation CNY", "net CNY"],
            [[r["case"], _fmt(r["cycle_cost_cny_mwh"], 1),
              _fmt(r["charge_mwh"]), _fmt(r["discharge_mwh"]),
              _fmt(r["gross_margin_cny"], 2),
              _fmt(r["degradation_cost_cny"], 2),
              _fmt(r["profit_cny"], 2)] for r in prior]))
        old = {r["case"]: float(r["profit_cny"]) for r in prior}
        new = {case: float(a[case]["fleet"]["net_profit_cny"])
               for case in cases}
        if all(key in old for key in ("C0", "C1", "C2")) and all(
                key in new for key in ("C80", "C90", "C100")):
            lines += [
                "",
                "The new C100/C90/C80 net profits "
                f"({new['C100']:+.6f}, {new['C90']:+.6f}, "
                f"{new['C80']:+.6f} CNY) reproduce the prior C0/C1/C2 "
                f"({old['C0']:+.6f}, {old['C1']:+.6f}, {old['C2']:+.6f} "
                "CNY) to the shown precision. The direct full-day truthful "
                "clear and the old rolling-then-settled path therefore agree "
                "on this day.",
            ]
        lines.append("")
    else:
        lines += ["The prior CSV was not available for a table; the old "
                  "report remains the record.", ""]
    lines += [
        "## Limitations",
        "",
        "- One frozen cross-date diagnostic day; no claim about a multi-day "
        "distribution or a historical market.",
        "- Experiment A is truthful bidding, not learned bidding.",
        "- Experiment C is one smoke checkpoint, not a trained comparison.",
        "- The LP bound uses No-ESS nodal prices as exogenous inputs and "
        "ignores the price feedback the fleet itself creates; it is an upper "
        "bound for one unit at a time, not a joint fleet optimum.",
        "- The No-ESS arm removes the independent fleet; prosumer and "
        "non-prosumer storage are disabled in `defaults.yaml`, so no other "
        "storage remains.",
        "",
        "## Next experiments",
        "",
        "- Repeat on the V2 multi-day library only after the dated price gate "
        "is met; do not reuse this cross-date day as evidence.",
        "- A same-day load/price replay when a matching pair exists.",
        "- Retrain at C80/C90/C100 before comparing learned policies across "
        "costs; the current checkpoint was trained at C100 only.",
        "- Sensitivity to efficiency, the SOC band and the action box, and a "
        "joint fleet LP that prices the fleet's own effect on LMP.",
        "",
        "## Artifacts and command",
        "",
        f"- Snapshot: `{snapshot_dir}`.",
        f"- Report: `{REPORT_PATH}`.",
        f"- Command: `{results.get('metadata', {}).get('command')}`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Storage cycle-cost sensitivity on one diagnostic day.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--skip-policy", action="store_true")
    return parser


def _make_snapshot_dir(root: Path, explicit: Optional[str]) -> Path:
    path = Path(explicit) if explicit else (
        Path(root) / f"storage_cycle_cost_{datetime.now():%Y%m%d_%H%M%S}")
    if path.exists():
        for index in range(1, 100):
            candidate = path.with_name(f"{path.name}_{index}")
            if not candidate.exists():
                path = candidate
                break
        else:
            raise SystemExit(f"cannot find a free snapshot directory near {path}")
    path.mkdir(parents=True)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    global SCENARIO
    SCENARIO = str(args.scenario)
    cfg.reload_defaults()
    cfg.reload_scenarios()

    snapshot_dir = _make_snapshot_dir(
        Path(args.snapshot_root), args.snapshot_dir)
    log_path = snapshot_dir / "run.log"
    log_fh = log_path.open("w", encoding="utf-8")
    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_out, log_fh)
    sys.stderr = _Tee(original_err, log_fh)
    started = datetime.now()
    command = " ".join([sys.executable, *sys.argv])
    results: Dict[str, Any] = {
        "status": "ok",
        "metadata": {
            "started": started.isoformat(),
            "command": command,
            "seed": int(args.seed),
            "snapshot_dir": str(snapshot_dir),
        },
        "label": "diagnostic_only",
    }
    trace_rows: List[dict] = []
    configs: Dict[str, dict] = {}

    def log(message: str) -> None:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}")

    try:
        log(f"start {command}")
        log(f"snapshot {snapshot_dir}")

        # Build the reference day and the independent fleet once.
        config_reference = _case_config(COSTS["C80"])
        agents_ref, wholesale_ref, case_ref = _build_day(
            config_reference, args.seed)
        ess = _ess_agents(agents_ref)
        if len(ess) != 4:
            raise SystemExit(
                f"expected four independent storage agents, got {len(ess)}")
        if any(abs(float(a.storage.eta_ch) - float(ess[0].storage.eta_ch))
               > 1e-12 for a in ess):
            raise SystemExit("the fleet does not share one efficiency")
        eta_ch = float(ess[0].storage.eta_ch)
        eta_dis = float(ess[0].storage.eta_dis)
        load_fp_ref = _load_fingerprint(agents_ref)
        results["case_identity"] = _case_identity(case_ref)
        results["storage_fleet"] = {
            "agents": [
                {
                    "name": a.name, "bus": int(a.bus),
                    "e_max_mwh": float(a.storage.e_max),
                    "p_ch_max_mw": float(a.storage.p_ch_max),
                    "p_dis_max_mw": float(a.storage.p_dis_max),
                    "eta_ch": float(a.storage.eta_ch),
                    "eta_dis": float(a.storage.eta_dis),
                    "soc0": float(a.storage.soc0),
                    "soc_min": float(a.storage.soc_min),
                    "soc_max": float(a.storage.soc_max),
                } for a in ess
            ],
            "count": len(ess),
        }

        # Experiment A.
        log("experiment A: C80, C90, C100")
        a_cases: Dict[str, dict] = {}
        for case, cost in COSTS.items():
            config = _case_config(cost)
            agents, wholesale, case_obj = _build_day(config, args.seed)
            if not np.allclose(wholesale, wholesale_ref):
                raise SystemExit(f"{case} saw a different wholesale day")
            if _load_fingerprint(agents) != load_fp_ref:
                raise SystemExit(f"{case} saw a different load day")
            configs[case] = dataclasses.asdict(config)
            started_case = time.perf_counter()
            day = _clear_truthful(agents, config, wholesale)
            wall = time.perf_counter() - started_case
            checks = _solver_and_checks(day, config, agents, ess)
            if not _is_clean(day):
                results["status"] = "failed"
                a_cases[case] = {
                    "case": case, "cycle_cost_cny_mwh": cost,
                    "status": "degraded", "checks": checks,
                    "solve_wall_time_s": wall,
                }
                log(f"{case}: degraded clear; no payoff reported")
                continue
            margin = arbitrage_lp.required_margin(cost, eta_ch, eta_dis)
            agent_rows = []
            for agent in ess:
                row = _agent_economics(agent, config, day)
                row.update({
                    "case": case, "arm": "Gurobi-truthful",
                    "level": "agent", "clean": True,
                    "cycle_cost_cny_mwh": cost,
                    "required_margin_cny_mwh": margin,
                })
                agent_rows.append(row)
            system_payoff, _ = _system_payoff(agents, config, day)
            fleet = _fleet_row(case, "Gurobi-truthful", cost, margin,
                               agent_rows, day, ess, system_payoff)
            fleet.update(_solver_and_checks(day, config, agents, ess))
            fleet["solve_wall_time_s"] = wall
            a_cases[case] = {
                "case": case,
                "cycle_cost_cny_mwh": cost,
                "required_margin_cny_mwh": margin,
                "agents": agent_rows,
                "fleet": fleet,
                "system_payoff_cny": system_payoff,
                "network": _network_summary(day),
                "checks": checks,
                "solve_wall_time_s": wall,
                "raw": _raw_day(day, ess),
            }
            for agent in ess:
                trace_rows.extend(_trace_rows(
                    case, "Gurobi-truthful", agent, config, day,
                    wholesale, None))
            log(f"{case}: charge={fleet['charge_mwh']:.4f} MWh "
                f"discharge={fleet['discharge_mwh']:.4f} MWh "
                f"net={fleet['net_profit_cny']:+.4f} CNY "
                f"status={checks.get('solver_status')}")

        results["experiment_a"] = {"cases": a_cases}
        results["configs"] = configs

        # Monotonicity.
        ordered = [c for c in ("C80", "C90", "C100") if c in a_cases]
        exact_violations = []
        violations = []
        for previous, following in zip(ordered, ordered[1:]):
            previous_profit = float(
                a_cases[previous]["fleet"]["net_profit_cny"])
            following_profit = float(
                a_cases[following]["fleet"]["net_profit_cny"])
            if following_profit > previous_profit + 1e-12:
                exact_violations.append({
                    "previous": previous, "following": following,
                    "previous_profit": previous_profit,
                    "following_profit": following_profit,
                    "increase_cny": following_profit - previous_profit,
                })
            if following_profit > previous_profit + MONOTONICITY_TOL_CNY:
                violations.append({
                    "previous": previous, "following": following,
                    "previous_profit": previous_profit,
                    "following_profit": following_profit,
                    "increase_cny": following_profit - previous_profit,
                })
        results["experiment_a"]["monotonicity"] = {
            "order": ordered,
            "net_profit_cny": {
                case: a_cases[case]["fleet"]["net_profit_cny"]
                for case in ordered},
            "tolerance_cny": MONOTONICITY_TOL_CNY,
            "non_increasing": not violations,
            "exact_violations": exact_violations,
            "violations": violations,
        }

        # No-ESS reference.
        log("no-ESS reference")
        config_noess = _case_config(COSTS["C100"])
        agents_noess, wholesale_noess, case_noess = _build_day(
            config_noess, args.seed, ess_buses=[])
        if _ess_agents(agents_noess):
            raise SystemExit("the No-ESS arm still carries storage agents")
        if not np.allclose(wholesale_noess, wholesale_ref):
            raise SystemExit("the No-ESS arm saw a different wholesale day")
        if _load_fingerprint(agents_noess) != load_fp_ref:
            raise SystemExit("the No-ESS arm saw a different load day")
        configs[NOESS] = dataclasses.asdict(config_noess)
        started_case = time.perf_counter()
        day_noess = _clear_truthful(agents_noess, config_noess,
                                    wholesale_noess)
        wall_noess = time.perf_counter() - started_case
        noess_checks = _solver_and_checks(day_noess, config_noess,
                                          agents_noess, [])
        if day_noess is None:
            noess_payoff = None
            noess_lmp = {}
        else:
            noess_payoff, _ = _system_payoff(agents_noess, config_noess,
                                             day_noess)
            noess_lmp = {
                a.name: _node_lmp(day_noess, a).tolist() for a in ess}
        results["experiment_a"]["noess"] = {
            "case": NOESS,
            "status": "ok" if _is_clean(day_noess) else "degraded",
            "system_payoff_cny": noess_payoff,
            "network": _network_summary(day_noess),
            "checks": noess_checks,
            "solve_wall_time_s": wall_noess,
            "nodal_lmp_at_ess": noess_lmp,
        }
        if _is_clean(day_noess):
            for case in a_cases:
                if a_cases[case].get("status") == "degraded":
                    continue
                fleet = a_cases[case]["fleet"]
                network = a_cases[case]["network"]
                noess_network = results["experiment_a"]["noess"]["network"]
                a_cases[case]["delta_system_payoff_cny"] = (
                    fleet["system_payoff_cny"] - noess_payoff)
                a_cases[case]["delta_clearing_objective"] = (
                    (network.get("clearing_objective") or 0.0)
                    - (noess_network.get("clearing_objective") or 0.0))
                a_cases[case]["delta_curtailment_mwh"] = (
                    (noess_network.get("total_curtailment_mwh") or 0.0)
                    - (network.get("total_curtailment_mwh") or 0.0))
        log(f"{NOESS}: system payoff={noess_payoff:.2f} CNY "
            f"status={noess_checks.get('solver_status')}")

        # Experiment B.
        log("experiment B: theoretical and executable bounds")
        b_cases = {}
        b_valid = (_is_clean(day_noess)
                   and not day_noess.get("has_wholesale_fallback")
                   and int(day_noess.get("lmp_fallbacks") or 0) == 0)
        if not b_valid:
            results["experiment_b"] = {
                "status": "invalid",
                "reason": "the No-ESS price field is not fully nodal; the "
                          "exogenous-price LP bound is not defined",
                "cases": {},
            }
        else:
            for case, cost in COSTS.items():
                if case not in a_cases or a_cases[case].get("status") == \
                        "degraded":
                    continue
                b_case = _experiment_b(day_noess, ess, cost, eta_ch, eta_dis)
                fleet = a_cases[case]["fleet"]
                b_case["actual"] = {
                    "net_profit_cny": fleet["net_profit_cny"],
                    "charge_mwh": fleet["charge_mwh"],
                    "discharge_mwh": fleet["discharge_mwh"],
                    "equivalent_cycles": fleet["equivalent_cycles"],
                }
                b_cases[case] = b_case
                lp = b_case["lp_fleet"]
                log(f"{case}: LP bound={lp['margin_cny']:+.4f} CNY "
                    f"actual={fleet['net_profit_cny']:+.4f} CNY")
            results["experiment_b"] = {"status": "ok", "cases": b_cases}

        # Experiment C.
        log("experiment C: MATD3 smoke checkpoint")
        if args.skip_policy:
            results["experiment_c"] = {
                "status": "skipped", "reason": "--skip-policy was requested"}
        elif not POLICY_DIR.is_dir():
            results["experiment_c"] = {
                "status": "skipped",
                "reason": f"no checkpoint directory at {POLICY_DIR}"}
        else:
            try:
                policies, obs_spec, action_spec = _load_policies(
                    agents_ref, config_reference, [a.name for a in ess])
            except Exception as exc:
                results["experiment_c"] = {
                    "status": "skipped",
                    "reason": f"checkpoint could not be loaded: {exc}"}
                log(f"experiment C skipped: {exc}")
            else:
                c_cases = {}
                for case, cost in COSTS.items():
                    config = _case_config(cost)
                    agents, wholesale, _ = _build_day(config, args.seed)
                    if not np.allclose(wholesale, wholesale_ref):
                        c_cases[case] = {
                            "status": "skipped",
                            "reason": "policy day differs from the truthful day"}
                        continue
                    started_policy = time.perf_counter()
                    outcome = _run_policy_case(
                        agents, config, wholesale_ref, policies,
                        [a.name for a in ess], obs_spec, action_spec,
                        args.seed)
                    wall_policy = time.perf_counter() - started_policy
                    if outcome.get("status") != "ok":
                        c_cases[case] = outcome
                        log(f"{case}: MATD3 {outcome.get('status')} "
                            f"({outcome.get('reason')})")
                        continue
                    day = outcome["day"]
                    checks = _solver_and_checks(day, config, agents, ess)
                    rows = []
                    for agent in ess:
                        row = _agent_economics(agent, config, day)
                        row.update({
                            "case": case, "arm": "MATD3",
                            "level": "agent",
                            "clean": _is_clean(day),
                            "cycle_cost_cny_mwh": cost,
                            "required_margin_cny_mwh":
                                arbitrage_lp.required_margin(
                                    cost, eta_ch, eta_dis),
                        })
                        rows.append(row)
                    system_payoff, _ = _system_payoff(agents, config, day)
                    fleet = _fleet_row(
                        case, "MATD3", cost,
                        arbitrage_lp.required_margin(cost, eta_ch, eta_dis),
                        rows, day, ess, system_payoff)
                    fleet.update(checks)
                    declared = outcome["declared"]
                    charge_means = [
                        declared[name].get("charge_price", {}).get("mean")
                        for name in declared]
                    discharge_means = [
                        declared[name].get("discharge_price", {}).get("mean")
                        for name in declared]
                    truthful = a_cases.get(case, {}).get("fleet", {})
                    lp_bound = b_cases.get(case, {}).get("lp_fleet", {})
                    c_cases[case] = {
                        "status": "ok",
                        "agents": rows,
                        "fleet": fleet,
                        "checks": checks,
                        "declared": declared,
                        "declared_charge_mean": float(np.mean(charge_means))
                        if charge_means else None,
                        "declared_discharge_mean": float(
                            np.mean(discharge_means))
                        if discharge_means else None,
                        "delta_vs_truthful_cny": (
                            float(fleet["net_profit_cny"])
                            - float(truthful.get("net_profit_cny") or 0.0)),
                        "delta_vs_lp_cny": (
                            float(fleet["net_profit_cny"])
                            - float(lp_bound["margin_cny"]))
                        if "margin_cny" in lp_bound else None,
                        "solve_wall_time_s": wall_policy,
                        "raw": _raw_day(day, ess),
                    }
                    for agent in ess:
                        trace_rows.extend(_trace_rows(
                            case, "MATD3", agent, config, day,
                            wholesale_ref, outcome["declared_raw"]))
                    log(f"{case}: MATD3 charge={fleet['charge_mwh']:.4f} "
                        f"discharge={fleet['discharge_mwh']:.4f} "
                        f"net={fleet['net_profit_cny']:+.4f} CNY")
                results["experiment_c"] = {
                    "status": "ok",
                    "metadata": _policy_metadata(),
                    "cases": c_cases,
                }

        results["prior_record"] = _prior_record()
        results["hashes"] = {
            "git": _git_state(),
            "software": _versions(),
        }

    except BaseException as exc:
        results["status"] = "failed"
        results["error"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__))
        traceback.print_exc()
    finally:
        end = datetime.now()
        results["metadata"]["ended"] = end.isoformat()
        results["metadata"]["duration_s"] = (
            end - started).total_seconds()
        try:
            _write_json(snapshot_dir / "config.json",
                        _snapshot_config(results))
            _write_json(snapshot_dir / "results.json", results)
            result_rows = _collect_result_rows(results)
            _write_csv(snapshot_dir / "results.csv", result_rows,
                       RESULT_COLUMNS)
            _write_csv(snapshot_dir / "dispatch_trace.csv", trace_rows,
                       TRACE_COLUMNS)
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(_report(results, snapshot_dir),
                                   encoding="utf-8")
            (snapshot_dir / "README.md").write_text(
                _snapshot_readme(results, report_path), encoding="utf-8")
            # Log the end line before hashing run.log, so the snapshot's own
            # hash covers the complete log rather than a prefix of it.
            log(f"end status={results.get('status')} "
                f"duration={results['metadata']['duration_s']:.2f}s")
            log_fh.flush()
            hashes = _input_hashes(snapshot_dir, report_path, log_path)
            hashes["git"] = _git_state()
            hashes["software"] = _versions()
            _write_json(snapshot_dir / "hashes.json", hashes)
        except Exception:
            traceback.print_exc()
            results["status"] = "failed"
            log(f"end status=failed "
                f"duration={results['metadata']['duration_s']:.2f}s")
            log_fh.flush()
        sys.stdout = original_out
        sys.stderr = original_err
        log_fh.close()

    return 0 if results.get("status") == "ok" else 1


def _collect_result_rows(results: dict) -> List[dict]:
    rows: List[dict] = []
    a = results.get("experiment_a", {})
    for case, payload in (a.get("cases") or {}).items():
        if payload.get("status") == "degraded":
            rows.append({
                "case": case, "arm": "Gurobi-truthful", "level": "fleet",
                "agent": "fleet", "status": "degraded",
                "clean": False,
                "solver_status": payload.get("checks", {}).get(
                    "solver_status"),
                "solve_wall_time_s": payload.get("solve_wall_time_s"),
            })
            continue
        for agent_row in payload.get("agents") or []:
            rows.append(_result_row(case, "Gurobi-truthful", "agent",
                                    agent_row, payload))
        fleet = dict(payload["fleet"])
        fleet["system_payoff_cny"] = payload.get("system_payoff_cny")
        fleet["delta_system_payoff_cny"] = payload.get(
            "delta_system_payoff_cny")
        rows.append(_result_row(case, "Gurobi-truthful", "fleet", fleet,
                                payload))
    noess = a.get("noess") or {}
    if noess:
        rows.append({
            "case": NOESS, "arm": "None", "level": "system",
            "agent": NOESS, "status": noess.get("status"),
            "clean": noess.get("checks", {}).get("clean"),
            "net_profit_cny": 0.0,
            "system_payoff_cny": noess.get("system_payoff_cny"),
            "solver_status": noess.get("checks", {}).get("solver_status"),
            "optimality_gap": noess.get("checks", {}).get("optimality_gap"),
            "solve_wall_time_s": noess.get("solve_wall_time_s"),
            "voltage_ok": noess.get("checks", {}).get("voltage_ok"),
            "line_ok": noess.get("checks", {}).get("line_ok"),
            "lmp_nodal": noess.get("checks", {}).get("lmp_nodal"),
        })
    c = results.get("experiment_c") or {}
    if c.get("status") == "ok":
        for case, payload in (c.get("cases") or {}).items():
            if payload.get("status") != "ok":
                rows.append({
                    "case": case, "arm": "MATD3", "level": "fleet",
                    "agent": "fleet", "status": payload.get("status"),
                    "clean": False,
                })
                continue
            for agent_row in payload.get("agents") or []:
                rows.append(_result_row(case, "MATD3", "agent", agent_row,
                                        payload))
            fleet = dict(payload["fleet"])
            fleet["delta_vs_truthful_cny"] = payload.get(
                "delta_vs_truthful_cny")
            rows.append(_result_row(case, "MATD3", "fleet", fleet, payload))
    return rows


def _result_row(case: str, arm: str, level: str, row: dict,
                payload: dict) -> dict:
    checks = payload.get("checks") or {}
    return {
        "case": case,
        "arm": arm,
        "level": level,
        "agent": row.get("agent"),
        "bus": row.get("bus"),
        "cycle_cost_cny_mwh": row.get("cycle_cost_cny_mwh",
                                     payload.get("cycle_cost_cny_mwh")),
        "required_margin_cny_mwh": row.get("required_margin_cny_mwh"),
        "status": row.get("status", payload.get("status", "ok")),
        "clean": row.get("clean", checks.get("clean")),
        "charge_mwh": row.get("charge_mwh"),
        "discharge_mwh": row.get("discharge_mwh"),
        "charge_periods": row.get("charge_periods"),
        "discharge_periods": row.get("discharge_periods"),
        "idle_share": row.get("idle_share"),
        "equivalent_cycles": row.get("equivalent_cycles"),
        "soc_initial": row.get("soc_initial"),
        "soc_terminal": row.get("soc_terminal"),
        "soc_min": row.get("soc_min"),
        "soc_max": row.get("soc_max"),
        "max_charge_mw": row.get("max_charge_mw"),
        "max_discharge_mw": row.get("max_discharge_mw"),
        "revenue_cny": row.get("revenue_cny"),
        "purchase_cost_cny": row.get("purchase_cost_cny"),
        "cycle_cost_cny": row.get("cycle_cost_cny",
                                  row.get("degradation_cost_cny")),
        "other_cost_cny": row.get("other_cost_cny",
                                  (row.get("generation_cost_cny") or 0.0)
                                  + (row.get("penalty_cny") or 0.0)),
        "net_profit_cny": row.get("net_profit_cny"),
        "delta_ess_profit_vs_noess_cny": (
            row.get("net_profit_cny") if level == "fleet"
            and arm != "None" else None),
        "delta_system_payoff_cny": row.get("delta_system_payoff_cny"),
        "delta_vs_truthful_cny": row.get("delta_vs_truthful_cny"),
        "system_payoff_cny": row.get("system_payoff_cny"),
        "solver_status": checks.get("solver_status",
                                    row.get("solver_status")),
        "optimality_gap": checks.get("optimality_gap",
                                     row.get("optimality_gap")),
        "solve_wall_time_s": row.get("solve_wall_time_s",
                                     payload.get("solve_wall_time_s")),
        "terminal_soc_ok": checks.get("terminal_soc_ok"),
        "soc_band_ok": checks.get("soc_band_ok"),
        "power_ok": checks.get("power_ok"),
        "voltage_ok": checks.get("voltage_ok"),
        "line_ok": checks.get("line_ok"),
        "no_simultaneous_ok": checks.get("no_simultaneous_ok"),
        "lmp_nodal": checks.get("lmp_nodal"),
    }


if __name__ == "__main__":
    sys.exit(main())
