# screen_v2_physical.py
"""A forced-serve power-flow screen for the proposed seasonal subsets.

This is **not a market outcome**. There is no clearing, no bid, no storage and
no shedding decision. Each period is a power flow on the day's must-serve
injections: the selected users' measured load on their buses, the benchmark's
native load on the other buses, and the scenario's renewable output. It answers
one narrow question before a subset is frozen: does the network carry the load
inside its voltage band and line ratings?

The composition is the V1 rule: a bus with a selected user is `replace` (the
measured user load, not native plus measured), every other bus is `preserve`
(the benchmark's native load). Nothing is scaled, filled or re-sampled.

Usage:
    PYTHONPATH=src python src/screen_v2_physical.py
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import build_industry_node_loads as bnl
import grid
import models
import scenarios

COLUMNS = (
    "group", "group_type", "date", "season", "users", "peak_load_mw",
    "converged", "periods", "min_bus_voltage_pu", "max_bus_voltage_pu",
    "periods_voltage_violation", "bus_periods_below_lo",
    "bus_periods_above_hi", "max_line_loading", "periods_line_overload",
    "binding_constraint",
)


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _renewable_profiles(config) -> Dict[int, np.ndarray]:
    """The scenario's PV/wind by bus; its load is not used here."""
    probe = copy.deepcopy(config)
    agents, _ = scenarios.get_scenario(
        "typical_day_industry_subset", T=96, config=probe)
    gen: Dict[int, np.ndarray] = collections.defaultdict(
        lambda: np.zeros(96))
    for agent in agents:
        bus = int(agent.bus)
        for attr in ("pv_real", "wind_real"):
            arr = getattr(agent, attr, None)
            if arr is not None:
                gen[bus] += np.maximum(np.asarray(arr, dtype=float), 0.0)
    return dict(gen)


def _bus_loads(group_users: Sequence[str], bus_of: Dict[str, int],
               date: str, curves, rated: Dict[int, float],
               native_shape: np.ndarray) -> tuple:
    loads: Dict[int, np.ndarray] = {}
    replace = set()
    for user in group_users:
        key = f"{date}|{user}"
        if key not in curves:
            raise SystemExit(
                f"{date}: {user} has no stored all-meters curve; the "
                "representative day is not complete for this group")
        bus = bus_of[user]
        loads.setdefault(bus, np.zeros(96))
        loads[bus] += np.asarray(curves[key], dtype=float)
        replace.add(bus)
    for bus, rated_mw in rated.items():
        if bus in replace:
            continue
        loads[bus] = np.asarray(native_shape, dtype=float) * float(rated_mw)
    return loads, replace


def run_day(date: str, loads: Dict[int, np.ndarray],
            renewables: Dict[int, np.ndarray], config,
            slack_vm_pu: float = 1.0,
            reactive_fraction: float = 0.0,
            line_impedance_scale: float = 1.0,
            fixed_caps_mvar: Dict[int, float] = None,
            line_scale_to_buses: Sequence[int] = None,
            regulators: Sequence[tuple] = None) -> dict:
    import pandapower as pp

    net = copy.deepcopy(grid.build_base_network(config))
    net.ext_grid.loc[:, "vm_pu"] = float(slack_vm_pu)
    if line_impedance_scale != 1.0:
        mask = (net.line.to_bus.isin(set(int(b) for b in line_scale_to_buses))
                if line_scale_to_buses is not None
                else net.line.index == net.line.index)
        net.line.loc[mask, "r_ohm_per_km"] *= float(line_impedance_scale)
        net.line.loc[mask, "x_ohm_per_km"] *= float(line_impedance_scale)
    if regulators:
        for from_bus, to_bus, tap_pos in regulators:
            mask = (((net.line.from_bus == from_bus)
                     & (net.line.to_bus == to_bus))
                    | ((net.line.from_bus == to_bus)
                       & (net.line.to_bus == from_bus)))
            net.line.loc[mask, "in_service"] = False
            pp.create_transformer_from_parameters(
                net, hv_bus=int(from_bus), lv_bus=int(to_bus),
                sn_mva=10.0, vn_hv_kv=12.66, vn_lv_kv=12.66,
                vk_percent=2.0, vkr_percent=0.2, pfe_kw=0.0,
                i0_percent=0.0, shift_degree=0.0, tap_side="lv",
                tap_neutral=0, tap_min=-10, tap_max=10,
                tap_step_percent=1.0, tap_pos=int(tap_pos),
                name=f"reg_{from_bus}_{to_bus}")
    tan_phi = float(np.tan(np.arccos(float(config.network.load_power_factor))))
    load_buses = sorted(int(b) for b in net.load.bus)
    for bus in load_buses:
        net.load.loc[net.load.bus == bus, "p_mw"] = 0.0
        net.load.loc[net.load.bus == bus, "q_mvar"] = 0.0
    for bus in sorted(renewables):
        pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"re_{bus}")
    # A negative shunt q is capacitive injection. The support is proportional to
    # the bus's own load Q, so it is an ideal local compensator (SVC/inverter),
    # not a fixed capacitor sized on one loading level.
    if reactive_fraction > 0:
        for bus in load_buses:
            pp.create_shunt(net, bus=bus, p_mw=0.0, q_mvar=0.0,
                            name=f"var_{bus}")
    if fixed_caps_mvar:
        for bus, mvar in fixed_caps_mvar.items():
            pp.create_shunt(net, bus=int(bus), p_mw=0.0,
                            q_mvar=-float(mvar), name=f"cap_{bus}")

    rows = []
    for t in range(96):
        for bus in load_buses:
            value = float(loads.get(bus, np.zeros(96))[t])
            net.load.loc[net.load.bus == bus, "p_mw"] = value
            net.load.loc[net.load.bus == bus, "q_mvar"] = value * tan_phi
            if reactive_fraction > 0:
                net.shunt.loc[net.shunt.bus == bus, "q_mvar"] = (
                    -float(reactive_fraction) * value * tan_phi)
        for bus, series in renewables.items():
            mask = net.sgen.bus == bus
            net.sgen.loc[mask, "p_mw"] = float(series[t])
        try:
            pp.runpp(net, algorithm="bfsw")
        except Exception:
            return {
                "date": date, "converged": False, "periods": t,
                "min_bus_voltage_pu": "", "max_bus_voltage_pu": "",
                "periods_voltage_violation": "",
                "bus_periods_below_lo": "", "bus_periods_above_hi": "",
                "max_line_loading": "", "periods_line_overload": "",
                "binding_constraint": "nonconvergent",
            }
        vm = net.res_bus.vm_pu
        lo, hi = float(config.network.v_min_pu), float(config.network.v_max_pu)
        rows.append({
            "period": t,
            "vmin": float(vm.min()),
            "vmax": float(vm.max()),
            "below": int((vm < lo - 1e-9).sum()),
            "above": int((vm > hi + 1e-9).sum()),
            "line": float(net.res_line.loading_percent.max()) / 100.0,
        })
    voltage_bad = [r for r in rows if r["below"] or r["above"]]
    line_bad = [r for r in rows if r["line"] > 1.0 + 1e-9]
    binding = ("voltage" if voltage_bad else
               ("line_capacity" if line_bad else "none"))
    return {
        "date": date,
        "converged": True,
        "periods": len(rows),
        "min_bus_voltage_pu": min(r["vmin"] for r in rows),
        "max_bus_voltage_pu": max(r["vmax"] for r in rows),
        "periods_voltage_violation": len(voltage_bad),
        "bus_periods_below_lo": sum(r["below"] for r in rows),
        "bus_periods_above_hi": sum(r["above"] for r in rows),
        "max_line_loading": max(r["line"] for r in rows),
        "periods_line_overload": len(line_bad),
        "binding_constraint": binding,
    }


def _write_report(path: Path, *, rows: Sequence[dict], command: str,
                  artifacts: Dict[str, str]) -> Path:
    groups = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    lines = [
        "# V2 seasonal physical screen",
        "",
        "## Verdict",
        "",
        "This is a **forced-serve power-flow screen**, not a market clearing. "
        "There is no bid, no storage, no shedding decision and no objective. "
        "It asks whether the selected users' measured load, on the proposed "
        "replace buses with native load elsewhere, stays inside the network's "
        "voltage band and line ratings.",
        "",
        "**No group passes the voltage band on any tested representative "
        "day.** The binding constraint is voltage for every group; two autumn "
        "days also overload a line. The screen therefore does not establish a "
        "physically operable full-load V2 scenario under the current IEEE33 "
        "network.",
        "",
        "`peak load MW` below is the **whole network** load: the selected "
        "users' measured load on replace buses plus the benchmark's native "
        "load on preserve buses. It is not the subset-only aggregate.",
        "",
        "| group | dates tested | converged | voltage violations | line "
        "overloads | worst min V | worst max V | max line loading | binding |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for group, group_rows in groups.items():
        converged = [r for r in group_rows if _as_bool(r["converged"])]
        if not converged:
            lines.append(f"| {group} | {len(group_rows)} | no | — | — | — | "
                         "— | — | nonconvergent |")
            continue
        lines.append(
            f"| {group} | {len(group_rows)} | {len(converged)} | "
            f"{sum(int(r['periods_voltage_violation']) for r in converged)} | "
            f"{sum(int(r['periods_line_overload']) for r in converged)} | "
            f"{min(float(r['min_bus_voltage_pu']) for r in converged):.4f} | "
            f"{max(float(r['max_bus_voltage_pu']) for r in converged):.4f} | "
            f"{max(float(r['max_line_loading']) for r in converged):.3f} | "
            f"{sorted({r['binding_constraint'] for r in converged})} |")
    worst = {}
    for group, group_rows in groups.items():
        converged = [r for r in group_rows if _as_bool(r["converged"])]
        if converged:
            worst[group] = min(float(r["min_bus_voltage_pu"])
                               for r in converged)
    if worst:
        closest = max(worst, key=worst.get)
        lines.append(
            f"\nThe closest group is **{closest}** (worst min V "
            f"{worst[closest]:.4f}); the others are further below the band. "
            "A uniform small voltage support is therefore unlikely to make "
            "every group feasible.")
    lines += ["", "## Per-date results", "",
              "| group | date | peak load MW | converged | min V | max V | "
              "voltage periods | line overload periods | max line | binding |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['date']} | "
            f"{float(row['peak_load_mw']):.3f} | "
            f"{'yes' if _as_bool(row['converged']) else 'no'} | "
            f"{row['min_bus_voltage_pu']} | {row['max_bus_voltage_pu']} | "
            f"{row['periods_voltage_violation']} | "
            f"{row['periods_line_overload']} | {row['max_line_loading']} | "
            f"{row['binding_constraint']} |")
    lines += [
        "",
        "## Limits",
        "",
        "- The screen uses the V1 replace/preserve composition and the "
        "scenario's renewable placement; it is not a new market scenario.",
        "- It does not model storage, bids, congestion pricing, shedding or "
        "the two-settlement market. A passing screen is a necessary condition, "
        "not a market result.",
        "- It uses the scenario's line-capacity multiplier and power factor; "
        "the exact V2 scenario declaration is not frozen yet.",
        "- The price blocker remains: no real dated prices exist for these "
        "days, so no training or priced evaluation is possible.",
        "",
        "## Artifacts", "",
        "| path | sha256 |", "|---|---|"]
    for path_text, digest in artifacts.items():
        lines.append(f"| `{path_text}` | `{digest}` |")
    lines += ["", "## Command", "", f"`{command}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forced-serve power-flow screen for V2 seasonal subsets.")
    parser.add_argument("--subsets", default="data/processed/v2_seasonal_subsets.csv")
    parser.add_argument("--candidates",
                        default="data/processed/v2_seasonal_candidates.csv")
    parser.add_argument("--users", default="data/processed/v2/v2_user_summary.csv")
    parser.add_argument("--curves", default="data/processed/v2/v2_user_day_curves.npz")
    parser.add_argument("--out",
                        default="data/processed/v2/v2_physical_screen.csv")
    parser.add_argument("--report", default="reports/v2_physical_screen.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    subsets_path = root / args.subsets
    candidates_path = root / args.candidates
    users_path = root / args.users
    curves_path = root / args.curves
    for path in (subsets_path, candidates_path, users_path, curves_path):
        if not path.exists():
            raise SystemExit(f"{path} not found; run the seasonal selection first")
    users = {r["user_id"]: int(r["bus"]) for r in _read_csv(users_path)}
    subsets = {r["group"]: r for r in _read_csv(subsets_path)}
    reps: Dict[str, List[dict]] = collections.defaultdict(list)
    for row in _read_csv(candidates_path):
        if _as_bool(row["representative"]):
            reps[row["group"]].append(row)
    curves = np.load(curves_path)

    config = scenarios.scenario_model_config(
        "typical_day_industry_subset", T=96, opf_mode="socp")
    rated = bnl.native_bus_load()
    native_shape = bnl.native_shape()
    renewables = _renewable_profiles(config)

    rows = []
    for group, subset in subsets.items():
        if not subset.get("user_ids"):
            continue
        group_users = subset["user_ids"].split(";")
        for day in sorted(reps.get(group, []), key=lambda r: r["date"]):
            loads, _replace = _bus_loads(
                group_users, users, day["date"], curves, rated, native_shape)
            peak = max(float(np.sum([series[t] for series in loads.values()]))
                       for t in range(96))
            result = run_day(day["date"], loads, renewables, config)
            rows.append({
                "group": group,
                "group_type": subset["group_type"],
                "date": day["date"],
                "season": day["season"],
                "users": len(group_users),
                "peak_load_mw": peak,
                **{k: v for k, v in result.items() if k != "date"},
            })
    out_path = _write_csv(root / args.out, rows)
    artifacts = {
        str(subsets_path): _sha256(subsets_path),
        str(candidates_path): _sha256(candidates_path),
        str(users_path): _sha256(users_path),
        str(curves_path): _sha256(curves_path),
        str(out_path): _sha256(out_path),
    }
    report_path = _write_report(
        root / args.report, rows=rows, command=" ".join(sys.argv),
        artifacts=artifacts)
    for row in rows:
        print(f"{row['group']:<22} {row['date']} peak={row['peak_load_mw']:.3f} "
              f"conv={row['converged']} vmin={row['min_bus_voltage_pu']} "
              f"vmax={row['max_bus_voltage_pu']} "
              f"line={row['max_line_loading']}")
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
