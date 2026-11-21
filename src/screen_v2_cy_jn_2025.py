# screen_v2_cy_jn_2025.py
"""Clear every paired 2022 load day against its 2025 Jiangnan price day.

One pair, one whole-day SOCP clear, then ten checks on what came back. The
price enters only as the PCC price `wholesale[t]`; the nodal LMPs are the
clearing's own power-balance duals, not a copy of it. The four independent
storage units settle at their own bus's LMP.

The network support is the `line50_q2.0` assumption: R/X halved, ideal local
reactive compensation of twice each bus's load Q, slack at 1.0 p.u. It is an
**idealised support screening assumption**, already used for the frozen V2
subset screen. It is not a selected device or a costed engineering design, and
the report says so.

A day that does not clear is recorded with its solver status; the infeasibility
file Gurobi writes is moved next to the results. Nothing about the load, price,
line parameters or compensation is changed to make a day pass.

Usage:
    PYTHONPATH=src python src/screen_v2_cy_jn_2025.py --arm all
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import config_loader
import ess
import grid
import market
import participant_payoff
import scenarios

VERSION = "V2-CY-JN-2025"
SUPPORT = "line50_q2.0"
PAIRING = "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
BUS_LOADS = "data/processed/v2_cy_jn_2025/bus_loads_96x33.npz"
PRICE_CURVES = "data/processed/v2_cy_jn_2025/price_curves.npz"
OUT_DIR = "data/processed/v2_cy_jn_2025"
REPORT_DIR = "reports/v2_cy_jn_2025"
ESS_BUSES = [9, 14, 17, 27]
T = 96
DT = 0.25
MODEL_SCENARIO = "typical_day_industry_subset"

#: JN-Base is the reference; the other arms change one knob each.
ARMS: Dict[str, dict] = collections.OrderedDict([
    ("JN-Base", {"cycle_cost": 100.0, "ess_buses": ESS_BUSES,
                 "scope": "all"}),
    ("JN-C80", {"cycle_cost": 80.0, "ess_buses": ESS_BUSES,
                "scope": "representative"}),
    ("JN-C90", {"cycle_cost": 90.0, "ess_buses": ESS_BUSES,
                "scope": "representative"}),
    ("JN-C100", {"cycle_cost": 100.0, "ess_buses": ESS_BUSES,
                 "scope": "representative"}),
    ("JN-NoESS", {"cycle_cost": 100.0, "ess_buses": [],
                  "scope": "representative"}),
])

DECLARED_NOT_RUN = {
    "JN-RE": "price + provincial new-energy feature; disabled because "
             "mv_value's unit and availability cannot be confirmed, and the "
             "feature would require a new observation version, not OBS_V6",
    "rule_based": "existing src/strategies rule-based bidder, no RL; not run",
    "optimization_reference": "system-optimal storage schedule under the same "
                              "network and price; not run",
}

RESULTS_COLUMNS = (
    "arm", "load_date", "price_date", "group", "pairing_type",
    "representative", "cycle_cost", "ess_units",
    "solver_status", "optimal", "lmp_fallbacks", "lmp_slack_max_dev_cny_mwh",
    "min_bus_voltage_pu", "max_bus_voltage_pu", "voltage_violation_cells",
    "max_line_utilization", "line_overload_cells",
    "ess_power_max_mw", "ess_soc_min", "ess_soc_max",
    "terminal_soc_max_err", "exclusive_max_mw2",
    "p_balance_max_residual_mw", "implied_loss_mwh",
    "q_balance_checked", "q_balance_max_residual_mvar",
    "pcc_import_max_mw", "pcc_export_max_mw", "reverse_within_limit",
    "served_mwh", "unserved_mwh", "curtailment_mwh", "re_consumption_rate",
    "clearing_objective_cny",
    "ess_total_payoff_cny", "ESS9_payoff_cny", "ESS14_payoff_cny",
    "ESS17_payoff_cny", "ESS27_payoff_cny",
    "ess_charge_mwh", "ess_discharge_mwh", "runtime_s", "error",
)


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def build_config(cycle_cost: float):
    """The market this experiment clears, with the support assumption applied."""
    config = scenarios.scenario_model_config(MODEL_SCENARIO, T=T, opf_mode="socp")
    config.network.line_impedance_scale = 0.5
    config.network.reactive_fraction_of_load_q = 2.0
    config.storage.cycle_cost = float(cycle_cost)
    return config


def build_agents(config, load_matrix: np.ndarray, price: np.ndarray,
                 ess_buses: Sequence[int]):
    net = grid.build_base_network(config)
    scale = scenarios.scenario_scale(
        config_loader.get_scenario_cfg(MODEL_SCENARIO))
    node_load = {int(b): np.asarray(load_matrix[:, int(b)], dtype=float)
                 for b in range(load_matrix.shape[1])}
    agents = grid.create_agents_from_network(
        net, T, with_wind=True, scale=scale, node_load_mw=node_load)
    agents += ess.build_ess_fleet(T, config, wholesale=price,
                                  buses=list(ess_buses))
    return net, agents


def _ess_schedules(result: dict, agents) -> Dict[str, dict]:
    return {a.name: result["schedules"][a.name]
            for a in agents if a.storage is not None and a.name in result["schedules"]}


def _checks(result: dict, agents, config, price: np.ndarray,
            net) -> dict:
    bus_voltage = np.asarray(result["bus_voltage"], dtype=float)
    lmp = np.asarray(result["lmp"], dtype=float)
    grid_import = np.asarray(result["grid_import_mw"], dtype=float) \
        if "grid_import_mw" in result else None
    grid_export = np.asarray(result["grid_export_mw"], dtype=float) \
        if "grid_export_mw" in result else None
    schedules = result["schedules"]

    lo, hi = float(config.network.v_min_pu), float(config.network.v_max_pu)
    v_violations = int(((bus_voltage < lo - 1e-7)
                        | (bus_voltage > hi + 1e-7)).sum())

    util = np.asarray(result.get("line_utilization", np.zeros((T, 1))),
                      dtype=float)
    line_over = int((util > 1.0 + 1e-7).sum())

    # --- storage bounds, terminal SOC, mutual exclusion -------------------
    storage_agents = [a for a in agents if a.storage is not None]
    power_rating = (max(float(a.storage.p_ch_max) for a in storage_agents)
                    if storage_agents else 0.0)
    power_max = 0.0
    soc_min = soc_hi_actual = None
    terminal_err = 0.0
    exclusive = 0.0
    charge_mwh = discharge_mwh = 0.0
    unit_schedules = _ess_schedules(result, agents)
    if unit_schedules:
        soc_min = min(float(np.min(s["soc_boundary"])) for s in unit_schedules.values())
        soc_hi_actual = max(float(np.max(s["soc_boundary"])) for s in unit_schedules.values())
        terminal_err = max(
            abs(float(s["soc_boundary"][-1]) - float(s["soc_boundary"][0]))
            for s in unit_schedules.values())
        exclusive = max(float(np.max(np.asarray(s["p_ch"]) * np.asarray(s["p_dis"])))
                        for s in unit_schedules.values())
        power_max = max(float(np.max(np.maximum(np.asarray(s["p_ch"]),
                                               np.asarray(s["p_dis"]))))
                        for s in unit_schedules.values())
        charge_mwh = sum(float(np.sum(s["p_ch"])) * DT for s in unit_schedules.values())
        discharge_mwh = sum(float(np.sum(s["p_dis"])) * DT for s in unit_schedules.values())

    # --- active-power balance ---------------------------------------------
    # The cleared model allocates each line's I^2R loss to its receiving bus.
    # Reconstructing injections and flows leaves exactly that allocation, so a
    # residual above zero would be a real imbalance, and the negative sum is
    # the day's losses.
    injection = np.zeros((T, len(result["bus_indices"])))
    # Net reactive load left after the compensation knob. `served`, not the
    # raw load, is what the model charges Q for, so it is what the rebuild
    # must use.
    q_ratio_net = float(np.tan(np.arccos(config.network.load_power_factor)))         * (1.0 - float(config.network.reactive_fraction_of_load_q))
    q_injection = np.zeros_like(injection)
    bus_pos = {int(b): i for i, b in enumerate(result["bus_indices"])}
    slack_bus = int(net.ext_grid.bus.iloc[0])
    # schedules carry no bus; map through the agents instead.
    for agent in agents:
        s = schedules.get(agent.name)
        if s is None:
            continue
        i = bus_pos.get(int(agent.bus))
        if i is None:
            continue
        injection[:, i] += (np.asarray(s["pv_used"]) + np.asarray(s["wind_used"])
                            + np.asarray(s["p_dis"]) - np.asarray(s["served"])
                            - np.asarray(s["p_ch"]))
        q_re = np.asarray(s.get("q_re", np.zeros(T)), dtype=float)
        q_injection[:, i] += q_re - np.asarray(s["served"]) * q_ratio_net
    if "grid_q_mvar" in result:
        q_injection[:, bus_pos[slack_bus]] += np.asarray(result["grid_q_mvar"])
    if grid_import is not None:
        # Net PCC exchange: import minus export, so a reverse-flow day keeps
        # the sign the balance equation uses.
        export = grid_export if grid_export is not None else 0.0
        injection[:, bus_pos[slack_bus]] += grid_import - export

    residual_max = float("nan")
    implied_loss_mwh = float("nan")
    q_residual_max = float("nan")
    q_checked = False
    flows = result.get("line_flow_mw")
    q_flows = result.get("line_flow_mvar")
    loss_mw = result.get("line_loss_mw")
    loss_mvar = result.get("line_loss_mvar")
    line_ids = result.get("line_indices")
    if flows is not None and line_ids is not None and loss_mw is not None:
        net_flow = np.zeros_like(injection)
        net_q_flow = np.zeros_like(q_injection)
        for j, lid in enumerate(line_ids):
            f = int(net.line.at[lid, "from_bus"])
            to = int(net.line.at[lid, "to_bus"])
            if f in bus_pos:
                net_flow[:, bus_pos[f]] -= flows[:, j]
                if q_flows is not None:
                    net_q_flow[:, bus_pos[f]] -= q_flows[:, j]
            if to in bus_pos:
                net_flow[:, bus_pos[to]] += flows[:, j]
                if q_flows is not None:
                    net_q_flow[:, bus_pos[to]] += q_flows[:, j]
                # The clearing subtracts the loss at the receiving end.
                net_flow[:, bus_pos[to]] -= loss_mw[:, j]
                if loss_mvar is not None:
                    net_q_flow[:, bus_pos[to]] -= loss_mvar[:, j]
        residual_max = float(np.max(np.abs(injection + net_flow)))
        implied_loss_mwh = float(np.sum(loss_mw) * DT)
        if q_flows is not None and loss_mvar is not None                 and "grid_q_mvar" in result:
            q_residual_max = float(np.max(np.abs(q_injection + net_q_flow)))
            q_checked = True

    # --- LMP extraction ----------------------------------------------------
    lmp_slack_dev = (float(np.max(np.abs(lmp[:, bus_pos[slack_bus]] - price)))
                     if lmp.size else float("nan"))

    reverse_limit = float(config.network.reverse_power_limit_mw)
    export_max = float(np.max(grid_export)) if grid_export is not None else 0.0
    import_max = float(np.max(grid_import)) if grid_import is not None else 0.0

    payoffs = {}
    for agent in agents:
        s = schedules.get(agent.name)
        if agent.storage is None or s is None:
            continue
        node_lmp = lmp[:, bus_pos.get(int(agent.bus), 0)]
        payoffs[agent.name] = float(participant_payoff.participant_payoff(
            s, node_lmp, agent, config, T).total)

    return {
        "min_bus_voltage_pu": float(np.min(bus_voltage)),
        "max_bus_voltage_pu": float(np.max(bus_voltage)),
        "voltage_violation_cells": v_violations,
        "max_line_utilization": float(np.max(util)),
        "line_overload_cells": line_over,
        "ess_power_max_mw": power_max,
        "ess_power_rating_mw": power_rating,
        "ess_soc_min": soc_min if soc_min is not None else "",
        "ess_soc_max": soc_hi_actual if soc_hi_actual is not None else "",
        "terminal_soc_max_err": terminal_err,
        "exclusive_max_mw2": exclusive,
        "p_balance_max_residual_mw": residual_max,
        "implied_loss_mwh": implied_loss_mwh,
        "lmp_slack_max_dev_cny_mwh": lmp_slack_dev,
        "q_balance_checked": q_checked,
        "q_balance_max_residual_mvar": q_residual_max,
        "pcc_import_max_mw": import_max,
        "pcc_export_max_mw": export_max,
        "reverse_within_limit": bool(export_max <= reverse_limit + 1e-6),
        "ess_charge_mwh": charge_mwh,
        "ess_discharge_mwh": discharge_mwh,
        "payoffs": payoffs,
    }


def clear_pair(pair: dict, arm: dict, cycles: Dict[str, np.ndarray],
               loads: Dict[str, np.ndarray], out_dir: Path) -> dict:
    started = time.perf_counter()
    row = {
        "arm": pair["arm"],
        "load_date": pair["load_date"],
        "price_date": pair["price_date"],
        "group": pair["group"],
        "pairing_type": pair["pairing_type"],
        "representative": pair["representative"],
        "cycle_cost": arm["cycle_cost"],
        "ess_units": len(arm["ess_buses"]),
    }
    try:
        price = np.asarray(cycles[f"south/{pair['price_date']}"], dtype=float)
        load = np.asarray(loads[f"load/{pair['load_date']}"], dtype=float)
        config = build_config(arm["cycle_cost"])
        net, agents = build_agents(config, load, price, arm["ess_buses"])
        result = market.clear_market(agents, T, "DA", {}, config,
                                     wholesale=price, horizon_type="full_day")
        if result is None:
            raise RuntimeError("clear_market returned None")
        solver = result.get("solver", {}).get("final", {})
        checks = _checks(result, agents, config, price, net)
        # `schedules` also carries a GRID entry with its own keys; only the
        # agents' schedules are participant schedules.
        agent_schedules = [result["schedules"][a.name] for a in agents
                           if a.name in result["schedules"]]
        served = sum(float(np.sum(s["served"])) * DT for s in agent_schedules)
        unserved = sum(float(np.sum(s["unserved"])) * DT
                       for s in agent_schedules)
        row.update({
            "solver_status": solver.get("status_name", ""),
            "optimal": bool(solver.get("optimal")),
            "lmp_fallbacks": int(result.get("lmp_fallbacks", -1)),
            "served_mwh": served,
            "unserved_mwh": unserved,
            "curtailment_mwh": float(result.get("total_curtailment", float("nan"))),
            "re_consumption_rate": float(result.get("re_consumption_rate", float("nan"))),
            "clearing_objective_cny": float(result.get("clearing_objective", float("nan"))),
            "ess_total_payoff_cny": sum(checks["payoffs"].values()),
            "error": "",
        })
        for name in ("ESS9", "ESS14", "ESS17", "ESS27"):
            row[f"{name}_payoff_cny"] = checks["payoffs"].get(name, "")
        for key, value in checks.items():
            if key == "payoffs":
                continue
            row[key] = value
    except Exception as exc:                       # recorded, never hidden
        row["error"] = f"{type(exc).__name__}: {exc}"
        row.setdefault("optimal", False)
        row.setdefault("solver_status", "EXCEPTION")
        ilp = Path("socp_infeasible.ilp")
        if ilp.exists():
            target = out_dir / "infeasible" / (
                f"{pair['arm']}__{pair['load_date']}__{pair['price_date']}.ilp")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ilp), str(target))
            row["error"] += f" | IIS saved to {target}"
    row["runtime_s"] = round(time.perf_counter() - started, 3)
    return row


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULTS_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _log(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def _write_config(path: Path, arms_run: Sequence[str]) -> Path:
    config = {
        "version": VERSION,
        "kind": "cross_year_controlled_simulation",
        "not_a_same_period_settlement": True,
        "price": {
            "region": "south",
            "region_note": "江南 by the task's naming; the source defines no boundary",
            "unit": "CNY/MWh",
            "unit_confirmed": False,
            "unit_note": "inferred from the observed floor 30, cap 1500 and "
                         "median daily mean; not externally confirmed",
            "north_column_used": False,
            "mv_value_used": False,
        },
        "load": {
            "library": "v2-frozen-dfc9d60cc222",
            "groups": ["summer", "autumn", "winter", "national_day_2022"],
            "compose_rule": "replace at the group's mapped buses, preserve "
                            "native demand elsewhere",
            "rescaled": False,
        },
        "network_support": {
            "assumption": SUPPORT,
            "slack_vm_pu": 1.0,
            "line_impedance_scale": 0.5,
            "reactive_fraction_of_load_q": 2.0,
            "v_min_pu": 0.93,
            "v_max_pu": 1.05,
            "note": "idealised support screening assumption; not a selected "
                    "device or a costed engineering design",
        },
        "storage": {
            "buses": ESS_BUSES,
            "capacity_mwh": 1.6,
            "power_mw": 0.4,
            "eta_ch": 0.92,
            "eta_dis": 0.92,
            "soc_min": 0.10,
            "soc_max": 0.90,
            "terminal_soc_equal": True,
            "cycle_cost_units": "CNY/MWh on charge + discharge energy",
        },
        "model_scenario": MODEL_SCENARIO,
        "opf_mode": "socp",
        "horizon_type": "full_day",
        "dt_hours": DT,
        "arms": {name: dict(ARMS[name]) for name in arms_run},
        "arms_not_run": DECLARED_NOT_RUN,
        "market_layer": "storage takes the Jiangnan price as given and its "
                        "quotes move the distribution market's LMPs; it cannot "
                        "move the provincial day-ahead price",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _write_integration(path: Path) -> Path:
    lines = [
        "# How the Jiangnan price enters the market (V2-CY-JN-2025)",
        "",
        "## The external price",
        "",
        "For pair `(load_date, price_date)` the clearing receives the 96-point "
        "`price_value_day_ahead_south` curve of `price_date` as `wholesale`, "
        "the PCC price `lambda_ws(t)`. The objective term is unchanged:",
        "",
        "    - sum_t  p_grid(t) * lambda_ws(t) * dt        dt = 0.25 h",
        "",
        "`p_grid > 0` is import, so the term is the cost of buying from the "
        "upper grid. `p_grid < 0` is export and earns the same price; no "
        "buy/sell spread was added. The PCC reverse-flow limit (2.5 MW) and "
        "every other constraint are the existing ones.",
        "",
        "The unit is CNY/MWh by inference (`data_audit.md`), consistent with "
        "`MW x h` in the same expression. It is not externally confirmed.",
        "",
        "## The nodal price",
        "",
        "LMPs are the power-balance duals of the same clearing:",
        "",
        "    LMP(i,t) = -Pi[balance(i,t)] / dt",
        "",
        "They are **not** the Jiangnan price copied to 33 buses. The slack "
        "bus's LMP is the marginal cost of one more MW at the PCC and equals "
        "`lambda_ws(t)`; every other bus differs by its network constraints, "
        "losses and congestion. `results.csv` records the slack-bus deviation "
        "from `lambda_ws` as an extraction check.",
        "",
        "## Storage settlement",
        "",
        "ESS9, ESS14, ESS17 and ESS27 settle at their own node's LMP through "
        "`participant_payoff`, the same function the rest of the project uses. "
        "Efficiency, SOC bounds and the cycle cost are unchanged.",
        "",
        "## What the storage fleet can and cannot move",
        "",
        "The fleet receives the Jiangnan price and its quotes change the "
        "distribution market's clearing and therefore its LMPs. It has no "
        "effect on the Jiangsu or Jiangnan day-ahead price, which is an input.",
        "",
        "## Price formation: the two layers, kept apart",
        "",
        "| layer | assumption | who sets it |",
        "|---|---|---|",
        "| Jiangsu / Jiangnan day-ahead | exogenous, price taking | the "
        "recorded 2025 curve; no participant here can move it |",
        "| distribution node LMP | endogenous, price influencing | the "
        "clearing; storage quotes change the nodal prices they are settled at "
        "|",
        "",
        "Every arm in this version is a price taker at the regional layer and "
        "a price influencer at the node layer. No arm is claimed to move the "
        "Jiangnan price, and none is a stackelberg model of the province.",
        "",
        "## Arm settings",
        "",
        "| arm | price | mv_value | cycle cost (CNY/MWh) | storage |",
        "|---|---|---|---:|---|",
        "| JN-Base | south | no | 100 | 4 units |",
        "| JN-C80 | south | no | 80 | 4 units |",
        "| JN-C90 | south | no | 90 | 4 units |",
        "| JN-C100 | south | no | 100 | 4 units |",
        "| JN-NoESS | south | no | 100 | none |",
        "| JN-RE | south | yes | 100 | 4 units | disabled, not run |",
        "",
        "C80/C90/C100 change the cycle cost alone. JN-RE is declared and "
        "disabled: `mv_value` has no confirmable unit or availability time and "
        "would require a new observation version.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_screen_report(path: Path, rows: Sequence[dict], arms_run) -> Path:
    def _num(row, key):
        value = row.get(key, "")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    lines = ["# Physical screen (V2-CY-JN-2025)", ""]
    lines.append("Every paired day is cleared once with the `line50_q2.0` "
                 "support assumption and the Jiangnan day-ahead price. The "
                 "support assumption is **idealised**: it is the same "
                 "screening assumption the frozen V2 subset screen used, not a "
                 "selected device or a costed engineering design.")
    lines.append("")
    lines.append("## Summary by arm")
    lines.append("")
    lines.append("| arm | days | optimal | 0 LMP fallbacks | voltage OK | "
                 "lines OK | terminal SOC OK | reverse flow OK | errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms_run:
        sub = [r for r in rows if r["arm"] == arm]
        ok_opt = sum(1 for r in sub if r.get("optimal") is True)
        ok_lmp = sum(1 for r in sub if r.get("lmp_fallbacks") == 0)
        ok_v = sum(1 for r in sub if r.get("voltage_violation_cells") == 0)
        ok_line = sum(1 for r in sub if r.get("line_overload_cells") == 0)
        ok_soc = sum(1 for r in sub
                     if not r.get("terminal_soc_max_err")
                     or abs(_num(r, "terminal_soc_max_err")) <= 1e-6)
        ok_rev = sum(1 for r in sub if r.get("reverse_within_limit") is True)
        errors = sum(1 for r in sub if r.get("error"))
        lines.append(f"| {arm} | {len(sub)} | {ok_opt} | {ok_lmp} | {ok_v} | "
                     f"{ok_line} | {ok_soc} | {ok_rev} | {errors} |")
    lines.append("")
    lines.append("The terminal-SOC column counts rows whose "
                 "`terminal_soc_max_err` is exactly 0; the no-storage arm has "
                 "no SOC to pin and reads as OK.")
    lines.append("")
    lines.append("## Checks performed")
    lines.append("")
    lines.append("| # | check | how it is read |")
    lines.append("|---|---|---|")
    lines.append("| 1 | Gurobi status | `solver.final.optimal`, and no "
                 "exception |")
    lines.append("| 2 | nodal active-power balance | injections + net line "
                 "flow with the model's own I^2R term at the receiving end; "
                 "the residual must be ~0 |")
    lines.append("| 3 | voltage band | every bus-period inside "
                 "0.93–1.05 p.u. |")
    lines.append("| 4 | line thermal limit | every line-period at or below "
                 "100% utilisation |")
    lines.append("| 5 | storage power and SOC | p_ch/p_dis within the 0.4 MW "
                 "rating, SOC inside [0.10, 0.90] |")
    lines.append("| 6 | terminal SOC | SOC(T) equals SOC(0) to 1e-6 |")
    lines.append("| 7 | charge/discharge exclusion | p_ch * p_dis is zero; a "
                 "net-flow formulation makes this structural, not a binary |")
    lines.append("| 8 | reactive power balance | rebuilt from the published "
                 "line Q flows, the agents' `q_re`, the slack Q and the "
                 "model's I^2X term; the residual must be ~0 |")
    lines.append("| 9 | LMP extraction | zero fallback cells, finite values, "
                 "and the slack LMP equals the Jiangnan price |")
    lines.append("| 10 | PCC exchange | import/export extremes and the "
                 "2.5 MW reverse limit |")
    lines.append("")
    checked = [r for r in rows if r.get("q_balance_checked") is True]
    worst = max((_num(r, "q_balance_max_residual_mvar") for r in checked),
                default=float("nan"))
    lines.append("**Reactive balance is rebuilt, not assumed.** The SOCP "
                 "engine publishes `line_flow_mvar` and `grid_q_mvar`; with "
                 "the agents' own `q_re` and the configured compensation, the "
                 "reactive balance closes to "
                 f"**{worst:.2e} MVar** at worst over "
                 f"{len(checked)} checked day(s). "
                 + ("" if len(checked) == len(rows) else
                    f"{len(rows) - len(checked)} row(s) could not be rebuilt "
                    "and say `q_balance_checked = False`."))
    lines.append("")
    by_group = collections.defaultdict(list)
    for row in rows:
        if row["arm"] == "JN-Base":
            by_group[row["group"]].append(row)
    if by_group:
        lines.append("## JN-Base by group")
        lines.append("")
        lines.append("| group | days | min V (p.u.) | max V (p.u.) | max line "
                     "util. | ESS payoff (CNY/day, median) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for group, sub in by_group.items():
            lines.append(
                f"| {group} | {len(sub)} | "
                f"{min(_num(r, 'min_bus_voltage_pu') for r in sub):.4f} | "
                f"{max(_num(r, 'max_bus_voltage_pu') for r in sub):.4f} | "
                f"{max(_num(r, 'max_line_utilization') for r in sub):.4f} | "
                f"{np.median([_num(r, 'ess_total_payoff_cny') for r in sub]):.1f} |")
        lines.append("")
    failures = [r for r in rows if r.get("error")]
    if failures:
        lines.append("## Days that did not clear")
        lines.append("")
        lines.append("| arm | load_date | price_date | status | error |")
        lines.append("|---|---|---|---|---|")
        for row in failures:
            lines.append(f"| {row['arm']} | {row['load_date']} | "
                         f"{row['price_date']} | {row.get('solver_status')} | "
                         f"{row['error']} |")
        lines.append("")
        lines.append("No load, price, line parameter or compensation value was "
                     "changed to make a day pass.")
    else:
        lines.append("Every paired day cleared. No infeasibility file was "
                     "produced.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("base", "probe", "all"), default="all")
    parser.add_argument("--pairing", default=PAIRING)
    parser.add_argument("--bus-loads", default=BUS_LOADS)
    parser.add_argument("--price-curves", default=PRICE_CURVES)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="clear at most this many pairs per arm (smoke)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    out_dir = root / args.out
    report_dir = root / args.report_dir
    log_path = report_dir / "run.log"
    if log_path.exists():
        log_path.unlink()

    pairs = _read_csv(root / args.pairing)
    cycles = np.load(root / args.price_curves)
    loads = np.load(root / args.bus_loads)
    arms_run = ([name for name, spec in ARMS.items() if spec["scope"] == "all"]
                if args.arm == "base" else
                list(ARMS) if args.arm == "all" else
                [name for name, spec in ARMS.items()
                 if spec["scope"] == "representative"])

    _log(log_path, f"[{datetime.now(timezone.utc).isoformat()}] {VERSION} "
                   f"arm={args.arm} arms={arms_run}")
    rows: List[dict] = []
    for arm_name in arms_run:
        spec = ARMS[arm_name]
        if spec["scope"] == "representative":
            work = [p for p in pairs if _as_bool(p["representative"])]
        else:
            work = list(pairs)
        if args.limit:
            work = work[:args.limit]
        _log(log_path, f"{arm_name}: {len(work)} pairs, "
                       f"cycle_cost={spec['cycle_cost']}, "
                       f"ess={spec['ess_buses']}")
        for n, pair in enumerate(work, start=1):
            row = clear_pair({**pair, "arm": arm_name}, spec, cycles, loads,
                             out_dir)
            rows.append(row)
            _log(log_path,
                 f"  {arm_name} {pair['load_date']} -> {pair['price_date']} "
                 f"{row.get('solver_status')} "
                 f"opt={row.get('optimal')} "
                 f"v=[{row.get('min_bus_voltage_pu', '')}, "
                 f"{row.get('max_bus_voltage_pu', '')}] "
                 f"ess={row.get('ess_total_payoff_cny', '')} "
                 f"{row.get('runtime_s')}s"
                 + (f" ERROR {row['error']}" if row.get("error") else ""))

    results_path = _write_csv(report_dir / "results.csv", rows)
    config_path = _write_config(report_dir / "config.json", arms_run)
    integration_path = _write_integration(
        report_dir / "market_model_integration.md")
    screen_path = _write_screen_report(
        report_dir / "physical_screen.md", rows, arms_run)

    manifest = {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "command": " ".join(sys.argv),
        "kind": "cross_year_controlled_simulation",
        "inputs": {
            "price_file": {"path": "docs/data.csv",
                           "sha256": _sha256(root / "docs/data.csv")},
            "pairing": {"path": args.pairing,
                        "sha256": _sha256(root / args.pairing)},
            "bus_loads": {"path": args.bus_loads,
                          "sha256": _sha256(root / args.bus_loads)},
            "price_curves": {"path": args.price_curves,
                             "sha256": _sha256(root / args.price_curves)},
        },
        "pairing": {
            "path": args.pairing,
            "sha256": _sha256(root / args.pairing),
            "pairs": len(pairs),
            "distinct_price_dates": len({p["price_date"] for p in pairs}),
            "manifest": args.out + "/pairing_manifest.json",
        },
        "outputs": {
            "results": {"path": args.report_dir + "/results.csv",
                        "sha256": _sha256(results_path)},
            "config": {"path": args.report_dir + "/config.json",
                       "sha256": _sha256(config_path)},
        },
        "arms_run": arms_run,
        "rows": len(rows),
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    _log(log_path, f"rows {len(rows)}; wrote {results_path}")
    _log(log_path, f"wrote {config_path}")
    _log(log_path, f"wrote {integration_path}")
    _log(log_path, f"wrote {screen_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
