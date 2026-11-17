# demo_snapshot.py
"""A self-contained demonstration snapshot of one cleared day.

The day is `typical_day_industry_subset`, cleared once with the fixed truthful
strategy on the SOCP engine. One clearing only: every metric, table and figure
in the snapshot is read from this run's own result, so the report and the
figures cannot describe two different days.

Writes `outputs/demo_snapshot/<run_id>/` and nothing else. The run directory
holds the console log (written by the caller's shell), the resolved scenario,
the input fingerprints, the dates, the solver state, the time series, the
metrics and a Markdown report. Figure generation is a separate step that reads
only these files -- the environment that can clear the day cannot import
matplotlib.

This is a demonstration case, not a formal experiment. The load rests on a
recorded project ruling for its reading unit, and the load day and the price
day are different calendar dates (`cross_date_diagnostic`). The report states
both limitations and no number here is presented as externally certified.

Usage:
    PYTHONPATH=src python src/demo_snapshot.py --out outputs/demo_snapshot/<run_id>
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import money  # noqa: E402
import scenarios  # noqa: E402
import s8_physical_audit as audit  # noqa: E402

SCENARIO = "typical_day_industry_subset"
STRATEGY = "fixed"
#: The baseline the subset was selected against. Read from the scenario's own
#: declaration at run time; this constant only names where to look.
BASE_PEAK_KEY = ("load_model", "subset_selection", "target_peak_mw")

#: The S8 audit is a historical artifact, kept separate from this run.
S8_JSON = Path("outputs/s8_physical_audit.json")
SMOKE_REPORT = Path("reports/s8_matd3_smoke.md")
SMOKE_POLICIES = Path("policies/smoke_subset_obs_v6_seed42")
SMOKE_RUN = Path("outputs/rl/run-20261016-173409-28757924")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    try:
        out = subprocess.check_output(["git", *args], text=True,
                                      stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def git_state() -> dict:
    status = _git("status", "--porcelain=v1") or ""
    diff = _git("diff", "--stat") or ""
    staged = _git("diff", "--cached", "--stat") or ""
    untracked = sorted(line[3:] for line in status.splitlines()
                       if line.startswith("??"))
    return {
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "status_porcelain": status,
        "tracked_diff_stat": diff,
        "staged_diff_stat": staged,
        "untracked": untracked,
        # Any entry -- including an untracked file -- means the tree is not the
        # commit alone. The snapshot is then labelled a worktree snapshot.
        "worktree_snapshot": bool(status.strip()),
    }


def environment() -> dict:
    packages = {}
    for name in ("gurobipy", "pandapower", "numpy", "scipy", "pandas",
                 "matplotlib", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    # The runtime version, where the module imports cleanly in this
    # environment. dist-info and __version__ can disagree.
    imported = {}
    for name in ("numpy", "scipy", "pandas", "pandapower", "torch"):
        try:
            module = __import__(name)
            imported[name] = getattr(module, "__version__", None)
        except Exception as exc:
            imported[name] = f"import failed: {type(exc).__name__}"
    try:
        import gurobipy

        gurobi = ".".join(str(v) for v in gurobipy.gurobi.version())
    except Exception:
        gurobi = None
    return {
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "packages_dist_info": packages,
        "packages_imported": imported,
        "gurobi_version": gurobi,
        "note": ("matplotlib is deliberately not imported in this "
                 "environment; the snapshot's figures are made by a separate "
                 "interpreter that reads only this directory"),
    }


def _json_safe(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def _clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _time_labels(T: int):
    step = int(round(money.DT_HOURS * 60))
    return [(_clock(t * step), _clock((t + 1) * step)) for t in range(T)]


def _total(schedules, agents, key) -> np.ndarray:
    return np.sum([np.asarray(schedules[a.name][key], dtype=float)
                   for a in agents], axis=0)


def _mwh(series) -> float:
    return float(np.sum(np.asarray(series, dtype=float)) * money.DT_HOURS)


def _storage_agents(agents):
    return [a for a in agents if getattr(a, "storage", None) is not None]


def collect(cleared: dict) -> dict:
    """Every number and series this run reports, from the one clearing."""
    config = cleared["config"]
    agents = cleared["agents"]
    result = cleared["result"]
    case = cleared["case"]
    T = int(cleared["T"])
    schedules = result["schedules"]

    asked = np.sum([np.asarray(a.load_real, dtype=float) for a in agents],
                   axis=0)
    served = _total(schedules, agents, "served")
    unserved = _total(schedules, agents, "unserved")
    pv_used = _total(schedules, agents, "pv_used")
    wind_used = _total(schedules, agents, "wind_used")
    p_ch = _total(schedules, agents, "p_ch")
    p_dis = _total(schedules, agents, "p_dis")
    imp = np.asarray(result.get("grid_import_mw", np.zeros(T)), dtype=float)
    exp = np.asarray(result.get("grid_export_mw", np.zeros(T)), dtype=float)
    losses = audit._line_losses(config, result)
    residual = imp - exp + pv_used + wind_used + p_dis - served - p_ch - losses

    v = np.asarray(result["bus_voltage"], dtype=float)
    buses = [int(b) for b in result["bus_indices"]]
    util = np.asarray(result["line_utilization"], dtype=float)
    lines = [int(x) for x in result.get("line_indices") or []]
    band = (float(config.network.v_min_pu), float(config.network.v_max_pu))

    prices = np.asarray(result["price"], dtype=float)
    fallback_periods = [int(t) for t in result.get("lmp_fallback_periods") or []]
    anomalous = sorted(set(fallback_periods)
                       | {t for t in range(T) if not np.isfinite(prices[t])})
    solver = (result.get("solver") or {}).get("final", {}) or {}
    optimal = bool(solver.get("optimal"))
    fell_back = bool(result.get("fell_back"))
    finite = bool(np.all(np.isfinite(prices))
                  and np.all(np.isfinite(v))
                  and np.all(np.isfinite(util)))

    peak_period = int(np.argmax(asked))
    served_share = (float(served.sum() / asked.sum())
                    if asked.sum() > 0 else None)
    base_peak = float(
        scenarios.get_scenario_cfg(cleared["scenario"])["load_model"]
        ["subset_selection"]["target_peak_mw"])

    storage_agents = _storage_agents(agents)
    fleet = {
        "n_units": len(storage_agents),
        "buses": [int(a.bus) for a in storage_agents],
        "p_ch_max_mw_total": float(sum(a.storage.p_ch_max
                                       for a in storage_agents)),
        "e_max_mwh_total": float(sum(a.storage.e_max
                                     for a in storage_agents)),
    }
    renewables = {
        "pv_installed_mw": float(sum(a.pv_capacity for a in agents)),
        "wind_installed_mw": float(sum(a.wind_capacity for a in agents
                                       if getattr(a, "has_wind", False))),
        "pv_buses": [int(a.bus) for a in agents if a.pv_capacity > 0],
        "wind_buses": [int(a.bus) for a in agents
                       if getattr(a, "has_wind", False) and a.wind_capacity > 0],
    }

    storage = []
    for a in storage_agents:
        sched = schedules[a.name]
        soc = np.asarray(sched["soc"], dtype=float)
        storage.append({
            "unit": a.name,
            "bus": int(a.bus),
            "p_ch_max_mw": float(a.storage.p_ch_max),
            "e_max_mwh": float(a.storage.e_max),
            "soc_initial": float(a.storage.soc0),
            "soc_min_bound": float(a.storage.soc_min),
            "soc_max_bound": float(a.storage.soc_max),
            "soc_min": float(soc.min()),
            "soc_max": float(soc.max()),
            "soc_final": float(soc[-1]),
            "charge_mwh": _mwh(sched["p_ch"]),
            "discharge_mwh": _mwh(sched["p_dis"]),
            # The clearing is a barrier-solved QCP: a unit at rest still shows
            # solver-noise flows. The project's own churn tolerance is 1e-4
            # MWh/day, so "idle" is measured against that, not an exact zero.
            "idle": bool(_mwh(sched["p_ch"]) < 1e-4
                         and _mwh(sched["p_dis"]) < 1e-4),
        })

    return {
        "scenario": cleared["scenario"],
        "strategy": STRATEGY,
        "opf_mode": str(config.opf_mode),
        "periods": T,
        "dt_hours": float(money.DT_HOURS),
        "case": case.describe() if case is not None else None,
        "unit_basis": case.unit_basis if case is not None else None,
        "series": {
            "asked": asked, "served": served, "shed": unserved,
            "pv_used": pv_used, "wind_used": wind_used,
            "p_ch": p_ch, "p_dis": p_dis, "import": imp, "export": exp,
            "losses": losses, "price": prices,
            "wholesale": np.asarray(cleared["wholesale"], dtype=float),
            "voltage": v, "utilization": util, "buses": buses, "lines": lines,
        },
        "metrics": {
            "scenario": cleared["scenario"],
            "strategy": STRATEGY,
            "opf_mode": str(config.opf_mode),
            "periods": T,
            "dt_hours": float(money.DT_HOURS),
            "definitions": {
                "asked_mwh": "sum over agents and periods of load_real (MW) "
                             "times dt_hours",
                "served_mwh": "sum over agents and periods of the clearing's "
                              "served schedule (MW) times dt_hours",
                "shed_mwh": "sum over agents and periods of the clearing's "
                            "unserved schedule (MW) times dt_hours",
                "served_share": "served_mwh / asked_mwh",
                "coincident_peak_mw": "max over periods of the agent-summed "
                                      "load_real (MW)",
                "base_peak_mw": "the subset's declared selection target, read "
                                "from config/scenarios.yaml; not a tuned value",
                "clearing_objective": "the clearing model's own optimization "
                                      "objective in CNY; not welfare, not a "
                                      "participant payoff",
                "re_consumption_rate": "100 * renewable energy consumed / "
                                       "renewable energy available, both MWh",
                "carbon_emissions": "grid carbon on the import part of the net "
                                    "exchange times emission_factor_grid",
                "network_losses_mwh": "sum over periods of the model's own "
                                      "r * I^2 line losses (MW) times dt_hours",
                "voltage_min_max": "minimum and maximum bus voltage over all "
                                   "96 periods and 33 buses, pu",
                "line_utilization": "maximum over periods of current / "
                                    "thermal limit, per line",
                "storage_charge_discharge_mwh": "sum over periods of p_ch / "
                                                "p_dis (MW) times dt_hours",
            },
            "system": {
                "asked_mwh": _mwh(asked),
                "served_mwh": _mwh(served),
                "shed_mwh": _mwh(unserved),
                "served_share": served_share,
                "coincident_peak_mw": float(asked.max()),
                "coincident_peak_period": peak_period,
                "base_peak_mw": base_peak,
                "peak_below_base_mw": base_peak - float(asked.max()),
            },
            "energy": {
                "pv_used_mwh": _mwh(pv_used),
                "wind_used_mwh": _mwh(wind_used),
                "re_available_mwh": float(result.get("total_re_available", 0.0)),
                "re_curtailed_mwh": float(result.get("total_curtailment", 0.0)),
                "re_consumption_rate_percent":
                    float(result["re_consumption_rate"]),
                "storage_charged_mwh": _mwh(p_ch),
                "storage_discharged_mwh": _mwh(p_dis),
                "grid_import_mwh": _mwh(imp),
                "grid_export_mwh": _mwh(exp),
                "network_losses_mwh": _mwh(losses),
                "balance_residual_max_mw": float(np.abs(residual).max()),
            },
            "voltage": {
                "band_pu": list(band),
                "min_pu": float(v.min()),
                "min_bus": int(buses[int(np.unravel_index(v.argmin(), v.shape)[1])]),
                "min_period": int(np.unravel_index(v.argmin(), v.shape)[0]),
                "max_pu": float(v.max()),
                "max_bus": int(buses[int(np.unravel_index(v.argmax(), v.shape)[1])]),
                "max_period": int(np.unravel_index(v.argmax(), v.shape)[0]),
                "bus_periods_below": int((v < band[0]).sum()),
                "bus_periods_above": int((v > band[1]).sum()),
                "at_lower_bound": bool(np.isclose(v.min(), band[0], atol=1e-6)),
            },
            "lines": {
                "n_lines": len(lines),
                "max_utilization": float(util.max()) if util.size else None,
                "max_line": (lines[int(np.unravel_index(util.argmax(),
                                                        util.shape)[1])]
                             if util.size else None),
                "max_period": (int(np.unravel_index(util.argmax(), util.shape)[0])
                               if util.size else None),
                "line_periods_at_or_over": int((util >= 1.0 - 1e-6).sum()),
                "lines_at_or_over": int((util.max(axis=0) >= 1.0 - 1e-6).sum())
                                    if util.size else 0,
            },
            "pcc": {
                "import_max_mw": float(imp.max()) if imp.size else None,
                "import_mean_mw": float(imp.mean()) if imp.size else None,
                "export_max_mw": float(exp.max()) if exp.size else None,
                "periods_exporting": int((exp > 1e-9).sum()) if exp.size else 0,
                "net_exchange_min_mw": float((imp - exp).min()) if imp.size else None,
                "reverse_limit_mw": float(config.network.reverse_power_limit_mw),
            },
            "objective": {
                "clearing_objective_cny": float(result["clearing_objective"]),
                "carbon_emissions_tco2": float(result["carbon_emissions"]),
                "carbon_intensity_tco2_per_mwh": float(
                    result.get("carbon_intensity", float("nan"))),
            },
            "fleet": fleet,
            "renewables": renewables,
            "storage": storage,
            "solver": {
                "status": solver.get("status_name"),
                "optimal": optimal,
                "runtime_s": solver.get("runtime_s"),
                "duals_available": (result.get("solver") or {}).get(
                    "duals_available"),
                "fell_back": fell_back,
                "lmp_fallbacks": int(result.get("lmp_fallbacks", 0)),
                "lmp_fallback_periods": fallback_periods,
                "lmp_price_source": result.get("lmp_price_source"),
                "anomalous_periods": anomalous,
                "all_prices_and_network_values_finite": finite,
            },
            "run_success": bool(optimal and not fell_back and finite
                                and not anomalous),
        },
    }


def fingerprints(scenario: str) -> dict:
    """Fresh hashes of the actual inputs, checked against their manifests."""
    import single_day_case as sdc

    cfg = scenarios.get_scenario_cfg(scenario)
    declaration = cfg["single_day_case"]
    ledger = Path(declaration["ledger"])
    manifest_path = Path(declaration["ledger_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    price = sdc.verify_price(str(declaration.get("price_sheet", "261009")))

    actual_ledger = sha256(ledger)
    actual_manifest = sha256(manifest_path)
    workbook = Path(price["workbook"])
    actual_workbook = sha256(workbook)
    price_csv = Path("data/processed/single_day_diagnostic/price_96.csv")
    actual_price_csv = sha256(price_csv) if price_csv.exists() else None

    return {
        "declared_by": "config/scenarios.yaml single_day_case block",
        "ledger": {
            "path": ledger.as_posix(),
            "sha256_actual": actual_ledger,
            "sha256_in_manifest": manifest.get("ledger_sha256"),
            "match": actual_ledger == manifest.get("ledger_sha256"),
        },
        "ledger_manifest": {
            "path": manifest_path.as_posix(),
            "sha256_actual": actual_manifest,
            "date": manifest.get("date"),
            "mapping_version": manifest.get("mapping_version"),
            "reading_unit": manifest.get("reading_unit"),
            "unit_status": manifest.get("unit_status"),
        },
        "price_workbook": {
            "path": str(workbook),
            "sheet": price["sheet"],
            "date": price["date"],
            "sha256_actual": actual_workbook,
            "sha256_read_by_this_run": price["workbook_sha256"],
            "match": actual_workbook == price["workbook_sha256"],
            "spatial_statistic": price["spatial_statistic"],
            "temporal_statistic": price["temporal_statistic"],
            "externally_confirmed": price["externally_confirmed"],
            "min_cny_mwh": price["min_cny_mwh"],
            "max_cny_mwh": price["max_cny_mwh"],
            "mean_cny_mwh": price["mean_cny_mwh"],
        },
        "price_file": {
            "path": price_csv.as_posix(),
            "sha256_actual": actual_price_csv,
        },
        "scenario_files": {
            "config/scenarios.yaml": sha256(Path("config/scenarios.yaml")),
            "config/defaults.yaml": sha256(Path("config/defaults.yaml")),
        },
        "all_manifest_hashes_match": bool(
            actual_ledger == manifest.get("ledger_sha256")
            and actual_workbook == price["workbook_sha256"]),
    }


def write_series(out: Path, collected: dict) -> None:
    series = collected["series"]
    T = collected["periods"]
    labels = _time_labels(T)

    columns = [
        "period", "interval_start", "interval_end",
        "load_asked_mw", "served_mw", "shed_mw",
        "wholesale_cny_mwh", "cleared_system_price_cny_mwh",
        "pv_used_mw", "wind_used_mw",
        "storage_charge_mw", "storage_discharge_mw",
        "grid_import_mw", "grid_export_mw", "network_loss_mw",
    ]
    with (out / "timeseries.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for t in range(T):
            w.writerow([
                t, labels[t][0], labels[t][1],
                series["asked"][t], series["served"][t], series["shed"][t],
                series["wholesale"][t], series["price"][t],
                series["pv_used"][t], series["wind_used"][t],
                series["p_ch"][t], series["p_dis"][t],
                series["import"][t], series["export"][t], series["losses"][t],
            ])

    # Bus and line identifiers are the result's own, not the column positions.
    with (out / "voltage_pu.csv").open("w", newline="",
                                       encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["period", "interval_start"]
                   + [f"bus{b}" for b in series["buses"]])
        for t in range(T):
            w.writerow([t, labels[t][0]]
                       + [f"{series['voltage'][t, j]:.12g}"
                          for j in range(series["voltage"].shape[1])])

    with (out / "line_utilization.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["period", "interval_start"]
                   + [f"line{x}" for x in series["lines"]])
        for t in range(T):
            w.writerow([t, labels[t][0]]
                       + [f"{series['utilization'][t, j]:.12g}"
                          for j in range(series["utilization"].shape[1])])

    with (out / "lmp_cny_mwh.csv").open("w", newline="",
                                        encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["period", "interval_start"]
                   + [f"bus{b}" for b in series["buses"]])
        for t in range(T):
            w.writerow([t, labels[t][0]]
                       + [f"{series['lmp'][t, j]:.12g}"
                          for j in range(len(series["buses"]))])

    with (out / "storage.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "bus", "period", "interval_start",
                    "p_ch_mw", "p_dis_mw", "soc_fraction"])
        for row in collected["storage_series"]:
            for t in range(T):
                w.writerow([row["unit"], row["bus"], t, labels[t][0],
                            f"{row['p_ch'][t]:.12g}",
                            f"{row['p_dis'][t]:.12g}",
                            f"{row['soc'][t]:.12g}"])


def write_report(out: Path, meta: dict, prints: dict, case_doc: dict,
                 metrics: dict, figure_names) -> None:
    system = metrics["system"]
    energy = metrics["energy"]
    voltage = metrics["voltage"]
    lines = metrics["lines"]
    pcc = metrics["pcc"]
    objective = metrics["objective"]
    solver = metrics["solver"]
    basis = prints["unit_basis"] or {}
    rule = (case_doc or {}).get("date_rule", {})
    load_source = (case_doc or {}).get("load_source", {})
    users_per_bus = load_source.get("users_per_bus") or {}
    users_with_data = load_source.get("users_with_data") or []
    bus_ids = sorted(int(b) for b in users_per_bus)
    storage = metrics["storage"]
    fleet = metrics["fleet"]
    renewables = metrics["renewables"]
    worktree = meta["git"]["worktree_snapshot"]

    s8 = None
    if S8_JSON.exists():
        s8 = json.loads(S8_JSON.read_text(encoding="utf-8"))

    md = []
    add = md.append
    add("# Demo snapshot: `typical_day_industry_subset`")
    add("")
    add(f"Run id **`{meta['run_id']}`**, produced {meta['timestamp']} by "
        "`src/demo_snapshot.py`.")
    add("")
    if worktree:
        add("> **Worktree snapshot.** The run was made on working tree "
            f"`{meta['git']['head']}` (branch `{meta['git']['branch']}`) with "
            "uncommitted or untracked changes present. It is not a snapshot of "
            "that commit alone.")
    else:
        add(f"> **Committed-tree snapshot.** HEAD `{meta['git']['head']}`, "
            f"branch `{meta['git']['branch']}`, clean working tree.")
    add("")
    add("## 1. Scenario")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| scenario | `{prints['scenario']}` |")
    add(f"| strategy | `{prints['strategy']}` (every participant at its "
        "truthful declaration) |")
    add(f"| clearing engine | `{prints['opf_mode']}` |")
    add(f"| horizon | {prints['periods']} periods x "
        f"{prints['dt_hours']:.2f} h |")
    add("| topology | IEEE 33-bus radial feeder |")
    add(f"| renewables | PV {renewables['pv_installed_mw']:.2f} MW "
        f"(buses {renewables['pv_buses']}), wind "
        f"{renewables['wind_installed_mw']:.2f} MW "
        f"(buses {renewables['wind_buses']}) |")
    add(f"| storage fleet | {fleet['n_units']} units, "
        f"{fleet['p_ch_max_mw_total']:.2f} MW charging capacity / "
        f"{fleet['e_max_mwh_total']:.2f} MWh at buses {fleet['buses']} |")
    add("")
    add("## 2. Data sources and dates")
    add("")
    add("> **Cross-date diagnostic.** The load day is "
        f"**{rule.get('load_date')}** and the price day is "
        f"**{rule.get('price_date')}**. They are different calendar days "
        "paired by decision (`cross_date_diagnostic`). This is not a "
        "same-day historical replay, and no figure here is what either day "
        "actually met in its own market.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| case type | `{(case_doc or {}).get('case_type')}` |")
    add(f"| load ledger | `{load_source.get('ledger')}` |")
    add(f"| load ledger sha256 | `{prints['fingerprints']['ledger']['sha256_actual']}` |")
    add(f"| load date | {rule.get('load_date')} |")
    add(f"| price sheet | `{prints['fingerprints']['price_workbook']['sheet']}` "
        f"from `{prints['fingerprints']['price_workbook']['path']}` |")
    add(f"| price workbook sha256 | "
        f"`{prints['fingerprints']['price_workbook']['sha256_actual']}` |")
    add(f"| price date | {rule.get('price_date')} |")
    add(f"| price statistic | `{prints['fingerprints']['price_workbook']['spatial_statistic']}` "
        "over an unstated point set; single-day reading by project ruling |")
    add(f"| same day | **{rule.get('same_day')}** |")
    add(f"| reading unit | {basis.get('reading_unit')} (basis "
        f"`{basis.get('basis')}`, evidence `{basis.get('evidence')}`) |")
    add(f"| unit externally verified | **"
        f"{basis.get('is_externally_verified')}** |")
    add("")
    add("The fingerprints above were recomputed from the files on disk. "
        f"Ledger matches its manifest: **"
        f"{prints['fingerprints']['ledger']['match']}**. Price workbook matches "
        f"the hash read by this run: **"
        f"{prints['fingerprints']['price_workbook']['match']}**.")
    add("")
    add("## 3. User subset and load scale")
    add("")
    add(f"- mapping version: `{(case_doc or {}).get('mapping_version')}`")
    add(f"- buses carrying a selected user: {len(bus_ids)} "
        f"({', '.join(str(b) for b in bus_ids)})")
    add(f"- users with data: {len(users_with_data)}")
    add("- reading conversion: a slot value is average active power over its "
        "15 minutes, in kW, times the meter's own 综合倍率, then /1000 to MW")
    add(f"- coincident peak: **{system['coincident_peak_mw']:.4f} MW** at "
        f"period {system['coincident_peak_period']}")
    add(f"- declared subset target: **{system['base_peak_mw']:.3f} MW** "
        f"(the run peaks {system['peak_below_base_mw']:.4f} MW below it)")
    add(f"- asked energy: **{system['asked_mwh']:.4f} MWh**")
    add("")
    add("## 4. Market clearing result")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| solver status | `{solver['status']}` (optimal: "
        f"{solver['optimal']}) |")
    add(f"| runtime | {solver['runtime_s'] if solver['runtime_s'] is not None else 'not reported'} |")
    add(f"| fell back to a single-period engine | {solver['fell_back']} |")
    add(f"| nodal-price fallbacks | {solver['lmp_fallbacks']} cells |")
    add(f"| fallback periods | {solver['lmp_fallback_periods'] or 'none'} |")
    add(f"| anomalous periods | {solver['anomalous_periods'] or 'none'} |")
    add(f"| price source | `{solver['lmp_price_source']}` |")
    add(f"| clearing objective | **{objective['clearing_objective_cny']:,.2f} "
        "CNY** |")
    add(f"| renewable consumption rate | "
        f"**{energy['re_consumption_rate_percent']:.4f} %** |")
    add(f"| carbon emissions | **{objective['carbon_emissions_tco2']:,.4f} "
        "tCO2** |")
    add("")
    add("`clearing_objective` is the clearing model's own optimization "
        "expression, not social welfare and not a participant payoff. The "
        "renewable rate is a ratio of consumed to available renewable energy. "
        "Neither is a training reward.")
    add("")
    add("## 5. Physical results")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| load asked | {system['asked_mwh']:,.4f} MWh |")
    add(f"| load served | **{system['served_mwh']:,.4f} MWh** |")
    add(f"| load shed | **{system['shed_mwh']:.6f} MWh** "
        f"({system['shed_mwh']:.3e} MWh, solver residual) |")
    add(f"| served share | **{system['served_share']:.4%}** |")
    add(f"| PV used | {energy['pv_used_mwh']:,.4f} MWh |")
    add(f"| wind used | {energy['wind_used_mwh']:,.4f} MWh |")
    add(f"| renewable curtailed | {energy['re_curtailed_mwh']:,.4f} MWh |")
    add(f"| storage charged / discharged | "
        f"{energy['storage_charged_mwh']:.6f} / "
        f"{energy['storage_discharged_mwh']:.6f} MWh |")
    add(f"| grid import / export | {energy['grid_import_mwh']:,.4f} / "
        f"{energy['grid_export_mwh']:,.4f} MWh |")
    add(f"| network losses | {energy['network_losses_mwh']:,.4f} MWh |")
    add(f"| active-power identity residual (max) | "
        f"{energy['balance_residual_max_mw']:.3e} MW |")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| voltage band | {voltage['band_pu'][0]:.4f} .. "
        f"{voltage['band_pu'][1]:.4f} pu |")
    add(f"| voltage min | **{voltage['min_pu']:.4f} pu** at bus "
        f"{voltage['min_bus']}, period {voltage['min_period']} |")
    add(f"| voltage max | **{voltage['max_pu']:.4f} pu** at bus "
        f"{voltage['max_bus']}, period {voltage['max_period']} |")
    add(f"| bus-periods below / above the band | "
        f"{voltage['bus_periods_below']} / {voltage['bus_periods_above']} |")
    add(f"| worst line utilization | **{lines['max_utilization']:.4f}** on line "
        f"{lines['max_line']}, period {lines['max_period']} |")
    add(f"| line-periods at or over the limit | "
        f"{lines['line_periods_at_or_over']} |")
    add(f"| PCC import max / mean | {pcc['import_max_mw']:,.4f} / "
        f"{pcc['import_mean_mw']:,.4f} MW (no import cap by construction) |")
    add(f"| PCC export max | {pcc['export_max_mw']:,.4f} MW over "
        f"{pcc['periods_exporting']} periods |")
    add(f"| most negative net exchange | {pcc['net_exchange_min_mw']:,.4f} MW "
        f"against the -{pcc['reverse_limit_mw']:.2f} MW reverse-flow bound |")
    add("")
    add("### Storage")
    add("")
    add("| unit | bus | charge MWh | discharge MWh | SOC initial -> final | "
        "SOC min .. max | idle |")
    add("|---|---|---|---|---|---|---|")
    for row in storage:
        add(f"| {row['unit']} | {row['bus']} | {row['charge_mwh']:.6f} | "
            f"{row['discharge_mwh']:.6f} | {row['soc_initial']:.4f} -> "
            f"{row['soc_final']:.4f} | {row['soc_min']:.4f} .. "
            f"{row['soc_max']:.4f} | {'yes' if row['idle'] else 'no'} |")
    if storage and all(row["idle"] for row in storage):
        add("")
        add("All four units are idle on this day: each charged and discharged "
            "by less than 1e-4 MWh over the whole day, the project's churn "
            "tolerance. The storage figure below therefore shows flat "
            "state-of-charge lines and zero power, which is the result rather "
            "than missing data.")
    add("")
    add("## 6. Figures")
    add("")
    add("Every figure is generated from this run's own snapshot CSVs. The "
        "load day and the price day are printed on each figure.")
    add("")
    for rel, title in figure_names:
        add(f"### {title}")
        add("")
        add(f"![{title}]({rel})")
        add("")
    add("## 7. Known assumptions and limitations")
    add("")
    add("- The reading unit rests on a recorded project ruling, not an "
        "external data dictionary. `is_externally_verified` is "
        f"**{basis.get('is_externally_verified')}**.")
    add("- The load day and the price day differ; this is a cross-date "
        "diagnostic, not a historical replay.")
    add("- The price curve is a spatial mean and does not represent a bus or "
        "zone.")
    add("- Line ampacities are project-assigned, not the published case33bw "
        "ratings; every utilization statement is conditional on that rule.")
    add("- The reactive load is a simulation assumption, "
        "`q = served * tan(acos(0.8503))`, not a meter reading.")
    add("- The shared `data/processed/single_day_diagnostic/case_manifest.json` "
        "was built for the earlier full-set audit ledger. This snapshot takes "
        "the dates and fingerprints of record from the subset ledger manifest "
        "declared by the scenario and verifies the files on disk against it.")
    add("- This is a fixed-strategy clearing. No MATD3 policy was loaded or "
        "evaluated in this run.")
    add("")
    add("## 8. Run status and anomalies")
    add("")
    add(f"**Run success: {metrics['run_success']}.** "
        f"Solver `{solver['status']}`, fell back: {solver['fell_back']}, "
        f"nodal-price fallbacks: {solver['lmp_fallbacks']}, anomalous periods: "
        f"{solver['anomalous_periods'] or 'none'}.")
    add("")
    add("The physical verdict is separate from run success: the network "
        f"served {system['served_share']:.4%} of the asked load and shed "
        f"{system['shed_mwh']:.6f} MWh ({system['shed_mwh']:.3e} MWh). "
        "The shed value is a solver residual, not a market decision to "
        "withhold service.")
    add("")
    add("## 9. Historical S8 comparison (not this run)")
    add("")
    if s8 is None:
        add("`outputs/s8_physical_audit.json` was not found; no historical "
            "comparison is shown.")
    else:
        add("The S8 audit below is a **historical artifact from an earlier "
            "run**, kept separate from this snapshot. It is not reused data "
            "and it is not a second clearing of this run.")
        add("")
        add("| metric | S8 historical | this run |")
        add("|---|---|---|")
        add(f"| served MWh | {s8['energy']['served_mwh']:,.4f} | "
            f"{system['served_mwh']:,.4f} |")
        add(f"| shed MWh | {s8['energy']['shed_mwh']:.6f} | "
            f"{system['shed_mwh']:.6f} |")
        add(f"| voltage min pu | {s8['voltage']['min']:.4f} | "
            f"{voltage['min_pu']:.4f} |")
        add(f"| worst line utilization | "
            f"{s8['lines']['max_utilization']:.4f} | "
            f"{lines['max_utilization']:.4f} |")
        add(f"| LMP fallbacks | {s8['solver']['lmp_fallbacks']} | "
            f"{solver['lmp_fallbacks']} |")
    add("")
    add("## 10. Reinforcement-learning statement")
    add("")
    add("The OBS_V6 smoke test recorded in `reports/s8_matd3_smoke.md` is a "
        "machinery check, not a training result. It verifies checkpoint save, "
        "reload and the evaluation path on a short 25-episode run; that run is "
        "not formal training, and no figure or statement in this snapshot is "
        "attributed to MATD3. This snapshot is a fixed-strategy clearing.")
    add("")
    add("## Reproduce")
    add("")
    add("```bash")
    add("PYTHONPATH=src python src/demo_snapshot.py "
        "--out outputs/demo_snapshot/<new_run_id>")
    add("python src/make_demo_figures.py "
        "--dir outputs/demo_snapshot/<new_run_id>")
    add("```")
    add("")
    (out / "demo_report.md").write_text("\n".join(md), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--out", required=True,
                        help="snapshot directory (one run, never overwritten)")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    # The caller's shell may already have opened run.log in this directory.
    # Refuse to overwrite a snapshot, not an empty directory holding its log.
    if (out / "metrics.json").exists() or (out / "run_metadata.json").exists():
        raise SystemExit(f"{out} already holds a snapshot; a run must not "
                         "overwrite another")
    out.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or out.name
    started = datetime.now().isoformat(timespec="seconds")
    print(f"[snapshot] run_id {run_id}")
    print(f"[snapshot] scenario {args.scenario} strategy {STRATEGY}")
    print("[snapshot] building the scenario and clearing the day once")

    import run_single_day_diagnostic as diag

    cleared = diag.clear_once(args.scenario)
    config = cleared["config"]
    if config.opf_mode != "socp":
        raise SystemExit(f"expected the SOCP engine, got {config.opf_mode!r}")
    print(f"[snapshot] solver returned; collecting the result")

    collected = collect(cleared)
    collected["storage_series"] = [
        {"unit": a.name, "bus": int(a.bus),
         "p_ch": np.asarray(cleared["result"]["schedules"][a.name]["p_ch"],
                            dtype=float),
         "p_dis": np.asarray(cleared["result"]["schedules"][a.name]["p_dis"],
                             dtype=float),
         "soc": np.asarray(cleared["result"]["schedules"][a.name]["soc"],
                           dtype=float)}
        for a in _storage_agents(cleared["agents"])]
    collected["series"]["lmp"] = np.asarray(cleared["result"]["lmp"],
                                            dtype=float)

    prints = fingerprints(args.scenario)
    git = git_state()
    case_doc = collected["case"] or {}

    meta = {
        "run_id": run_id,
        "timestamp": started,
        "command": [sys.executable, *sys.argv],
        "scenario": args.scenario,
        "strategy": STRATEGY,
        "opf_mode": str(config.opf_mode),
        "entry_point": "run_single_day_diagnostic.clear_once",
        "worktree_snapshot": git["worktree_snapshot"],
        "git": git,
        "environment": environment(),
        "rl_smoke_record": {
            "report": SMOKE_REPORT.as_posix(),
            "report_exists": SMOKE_REPORT.exists(),
            "policies_exist": SMOKE_POLICIES.exists(),
            "run_artifact_exists": SMOKE_RUN.exists(),
            "kind": "smoke test only, not formal training",
        },
        "figures_expected_after_separate_step": True,
    }

    _write_json(out / "run_metadata.json", meta)
    _write_json(out / "scenario_config.json", {
        "scenario_cfg": scenarios.get_scenario_cfg(args.scenario),
        "resolved_market_config": config,
        "strategy": STRATEGY,
        "opf_mode": str(config.opf_mode),
        "periods": int(cleared["T"]),
    })
    _write_json(out / "input_fingerprints.json", prints)
    _write_json(out / "case.json", {
        "case": case_doc,
        "unit_basis": collected["unit_basis"],
    })
    _write_json(out / "metrics.json", collected["metrics"])
    write_series(out, collected)

    base_peak = collected["metrics"]["system"]["base_peak_mw"]
    figure_names = [
        ("figures/fig1_system_load.png",
         f"System active load, coincident peak and the {base_peak:.3f} MW "
         "baseline"),
        ("figures/fig2_market_price.png",
         "Wholesale price input and this run's cleared system price"),
        ("figures/fig3_bus_voltage.png",
         "Bus voltage range over the day and the voltage band"),
        ("figures/fig4_line_utilization.png",
         "Maximum line utilization and the thermal limit"),
        ("figures/fig5_served_vs_shed.png",
         "Served load and shed load over the day"),
        ("figures/fig6_storage_soc_power.png",
         "Storage state of charge and charge/discharge power"),
    ]
    write_report(out, meta, {"scenario": args.scenario,
                             "strategy": STRATEGY,
                             "opf_mode": str(config.opf_mode),
                             "periods": int(cleared["T"]),
                             "dt_hours": float(money.DT_HOURS),
                             "unit_basis": collected["unit_basis"],
                             "fingerprints": prints},
                 case_doc, collected["metrics"], figure_names)

    m = collected["metrics"]
    print(f"[snapshot] dates load {case_doc.get('date_rule', {}).get('load_date')}"
          f" price {case_doc.get('date_rule', {}).get('price_date')}")
    print(f"[snapshot] served {m['system']['served_mwh']:.4f} MWh of "
          f"{m['system']['asked_mwh']:.4f} MWh, shed "
          f"{m['system']['shed_mwh']:.6f} MWh")
    print(f"[snapshot] voltage {m['voltage']['min_pu']:.4f} .. "
          f"{m['voltage']['max_pu']:.4f} pu; worst line "
          f"{m['lines']['max_utilization']:.4f}")
    print(f"[snapshot] solver {m['solver']['status']}, fallbacks "
          f"{m['solver']['lmp_fallbacks']}, run_success "
          f"{m['run_success']}")
    print(f"[snapshot] wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
