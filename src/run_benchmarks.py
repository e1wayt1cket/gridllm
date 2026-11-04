# run_benchmarks.py
"""Baseline clears for V2-CY-JN-2025: B0 NoESS, B2 rule policies, B3 LP bound.

B1 is **not run here**: it is JN-Base by construction (empty `action_params`
maps to the truthful point), and `audit_b1_equivalence.py` verifies that rather
than repeating 96 clears.

B0 removes the four storage units. Keeping them and forcing zero flow is the
other possible counterfactual and is *not* used: it leaves the device variables
and their constraints in the model, so it answers a different question.

B2 quotes are deterministic functions of the published day-ahead curve, which
is available before the day-ahead decision. They never read a realized LMP or a
future market outcome. Two families are tuned on the common training split,
selected on the common validation split, frozen, and evaluated on the common
test and holiday blocks. The national-day block never enters tuning.

B3 is the existing `arbitrage_lp` bound against the **external** price: one
unit, perfect foresight of a series it cannot move. It is a price-taker
reference, not a market-clearing result, and the fleet figure (four times the
single-unit bound) ignores the nodal prices the units are actually settled at.

Usage:
    PYTHONPATH=src python src/run_benchmarks.py --segments b0,b3
    PYTHONPATH=src python src/run_benchmarks.py --segments all
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import arbitrage_lp
import market
import screen_v2_cy_jn_2025 as screen

T = screen.T
DT = screen.DT
ESS_BUSES = screen.ESS_BUSES

PAIRING = screen.PAIRING
BUS_LOADS = screen.BUS_LOADS
PRICE_CURVES = screen.PRICE_CURVES
COMMON = "data/processed/v2_cy_jn_2025/common_split.json"
OUT_DIR = "data/processed/v2_cy_jn_2025/benchmark"
RAW = "data/processed/v2_cy_jn_2025/benchmark/benchmark_raw.csv"
PARAMS = "data/processed/v2_cy_jn_2025/benchmark/rule_params.json"
B3_CSV = "data/processed/v2_cy_jn_2025/benchmark/b3_lp_bound.csv"

SEED = 20261101
#: Quote used where a rule does not want the unit to act. Both are the top of
#: the storage action box, so the rule stays inside the action space a policy
#: would search.
CHARGE_SUPPRESS = 150.0
DISCHARGE_SUPPRESS = 300.0

#: Declared grids. Kept small because every candidate costs one clear per
#: training day; the range is stated rather than searched adaptively.
GRID_A: Tuple[Tuple[float, float], ...] = (
    (0.20, 0.65), (0.20, 0.80), (0.35, 0.65), (0.35, 0.80))
GRID_B_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.10, 0.90), (0.20, 0.80))

COLUMNS = (
    "arm", "candidate", "params", "split", "load_date", "price_date",
    "group", "cycle_cost", "ess_units", "optimal", "solver_status",
    "lmp_fallbacks", "min_bus_voltage_pu", "max_bus_voltage_pu",
    "voltage_violation_cells", "max_line_utilization", "line_overload_cells",
    "p_balance_max_residual_mw", "q_balance_max_residual_mvar",
    "q_balance_checked", "lmp_slack_max_dev_cny_mwh",
    "terminal_soc_max_err", "exclusive_max_mw2",
    "pcc_import_max_mw", "pcc_export_max_mw", "reverse_within_limit",
    "served_mwh", "unserved_mwh", "curtailment_mwh", "re_consumption_rate",
    "clearing_objective_cny", "ess_total_payoff_cny",
    "ESS9_payoff_cny", "ESS14_payoff_cny", "ESS17_payoff_cny",
    "ESS27_payoff_cny", "ess_charge_mwh", "ess_discharge_mwh",
    "ess_soc_min", "ess_soc_max", "runtime_s", "error",
)


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _candidate_id(params: dict) -> str:
    """A stable id over the numeric knobs; the family is the arm name."""
    return "|".join(f"{k}={float(params[k]):.2f}" for k in sorted(params)
                    if k != "family")


def threshold_quotes(price: Sequence[float], q_lo: float,
                     q_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Charge only in the low-price band, discharge only in the high one.

    A quote of zero means the unit will act at any non-negative price; the
    suppress value means it will not act unless the market pays its box top.
    The thresholds are quantiles of the published day-ahead curve, so they are
    available at the decision and are scale-free across days.
    """
    series = np.asarray(price, dtype=float)
    lo = float(np.quantile(series, q_lo))
    hi = float(np.quantile(series, q_hi))
    charge = np.where(series <= lo, 0.0, CHARGE_SUPPRESS)
    discharge = np.where(series >= hi, 0.0, DISCHARGE_SUPPRESS)
    return charge, discharge


def soc_band_quotes(price: Sequence[float], q_lo: float, q_hi: float,
                    soc_lo: float, soc_hi: float, e_max: float,
                    p_max: float, eta_ch: float, eta_dis: float,
                    soc0: float, dt: float = DT) -> Tuple[np.ndarray, np.ndarray]:
    """The threshold rule with the SOC band capping how many periods may act.

    The band sets how many periods of full-power flow fit between the starting
    state and each safety edge:
    charge  `(soc_hi - soc0) * E / (eta_ch * P * dt)`,
    discharge `(soc0 - soc_lo) * E * eta_dis / (P * dt)`.
    The threshold picks the eligible periods; this keeps the cheapest eligible
    charge periods and the dearest eligible discharge periods, so the planned
    throughput cannot leave the band.
    """
    charge, discharge = threshold_quotes(price, q_lo, q_hi)
    series = np.asarray(price, dtype=float)
    k_ch = int(math.floor((soc_hi - soc0) * e_max
                          / (eta_ch * p_max * dt)))
    k_dis = int(math.floor((soc0 - soc_lo) * e_max * eta_dis
                           / (p_max * dt)))
    order = np.argsort(series, kind="stable")
    charge_mask = np.zeros(series.size, dtype=bool)
    discharge_mask = np.zeros(series.size, dtype=bool)
    if k_ch > 0:
        charge_mask[order[:k_ch]] = True
    if k_dis > 0:
        discharge_mask[order[-k_dis:]] = True
    charge = np.where(charge_mask, charge, CHARGE_SUPPRESS)
    discharge = np.where(discharge_mask, discharge, DISCHARGE_SUPPRESS)
    return charge, discharge


def quotes_for(params: dict, price: Sequence[float], agents) -> Dict[str, dict]:
    storage = next(a.storage for a in agents if a.storage is not None)
    if params["family"] == "B2A":
        charge, discharge = threshold_quotes(price, params["q_lo"],
                                             params["q_hi"])
    else:
        charge, discharge = soc_band_quotes(
            price, params["q_lo"], params["q_hi"], params["soc_lo"],
            params["soc_hi"], float(storage.e_max), float(storage.p_ch_max),
            float(storage.eta_ch), float(storage.eta_dis),
            float(storage.soc0))
    return {a.name: {"charge_price": charge, "discharge_price": discharge}
            for a in agents if a.storage is not None}


def run_day(row: dict, *, arm: str, candidate: str, params: dict,
            ess_buses: Sequence[int], action_params=None,
            split: str) -> dict:
    started = time.perf_counter()
    out = {
        "arm": arm, "candidate": candidate,
        "params": _candidate_id(params) if params else "",
        "split": split, "load_date": row["load_date"],
        "price_date": row["price_date"], "group": row["group"],
        "cycle_cost": 100.0, "ess_units": len(ess_buses),
    }
    try:
        cycles = np.load(PRICE_CURVES)
        loads = np.load(BUS_LOADS)
        price = np.asarray(cycles[f"south/{row['price_date']}"], dtype=float)
        load = np.asarray(loads[f"load/{row['load_date']}"], dtype=float)
        config = screen.build_config(100.0)
        net, agents = screen.build_agents(config, load, price, ess_buses)
        actions = action_params(price, agents) if callable(action_params) \
            else (action_params or {})
        result = market.clear_market(agents, T, "DA", actions, config,
                                     wholesale=price, horizon_type="full_day")
        if result is None:
            raise RuntimeError("clear_market returned None")
        checks = screen._checks(result, agents, config, price, net)
        solver = result.get("solver", {}).get("final", {})
        agent_schedules = [result["schedules"][a.name] for a in agents
                           if a.name in result["schedules"]]
        out.update({
            "optimal": bool(solver.get("optimal")),
            "solver_status": solver.get("status_name", ""),
            "served_mwh": sum(float(np.sum(s["served"])) * DT
                              for s in agent_schedules),
            "clearing_objective_cny": float(result["clearing_objective"]),
            "ess_total_payoff_cny": sum(checks["payoffs"].values()),
            "error": "",
        })
        for key, value in checks.items():
            if key == "payoffs":
                continue
            out[key] = value
        for name in ("ESS9", "ESS14", "ESS17", "ESS27"):
            out[f"{name}_payoff_cny"] = checks["payoffs"].get(name, "")
    except Exception as exc:
        out.update({"optimal": False, "solver_status": "EXCEPTION",
                    "error": f"{type(exc).__name__}: {exc}"})
    out["runtime_s"] = round(time.perf_counter() - started, 3)
    return out


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS),
                                extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["arm"], r["candidate"], r["load_date"])
            for r in _read_csv(path)}


def _log(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def _split_of(split: dict, load_date: str) -> str:
    common = split["common"]
    for key in ("train", "validation", "test", "holiday_block"):
        if any(i.startswith(load_date + "|") for i in common[key]):
            return key
    return "full96_only"


def candidates() -> List[dict]:
    out = []
    for q_lo, q_hi in GRID_A:
        out.append({"family": "B2A", "q_lo": q_lo, "q_hi": q_hi})
    for q_lo, q_hi in GRID_A:
        for soc_lo, soc_hi in GRID_B_BANDS:
            out.append({"family": "B2B", "q_lo": q_lo, "q_hi": q_hi,
                        "soc_lo": soc_lo, "soc_hi": soc_hi})
    return out


def _run_segment(rows: Sequence[dict], *, arm: str, params: dict,
                 ess_buses, action_params, raw: Path, done: set,
                 log: Path, limit: int = 0) -> None:
    candidate = _candidate_id(params) if params else arm
    for row in (rows[:limit] if limit else rows):
        key = (arm, candidate, row["load_date"])
        if key in done:
            continue
        out = run_day(row, arm=arm, candidate=candidate, params=params,
                      ess_buses=ess_buses, action_params=action_params,
                      split=row["split"])
        _append(raw, out)
        done.add(key)
        _log(log, f"  {arm} {candidate} {row['load_date']} "
                  f"payoff={out.get('ess_total_payoff_cny', '')} "
                  f"opt={out.get('optimal')} {out['runtime_s']}s"
                  + (f" ERROR {out['error']}" if out.get("error") else ""))


def _select(raw: Path, log: Path) -> dict:
    rows = [r for r in _read_csv(raw) if not r.get("error")]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    selection = {"grid": [c for c in candidates()], "seed": SEED,
                 "criterion": "mean daily ESS fleet payoff (CNY/day)",
                 "stages": {}}
    selected = {}
    for family in ("B2A", "B2B"):
        train_rows = [r for r in train if r["arm"] == family]
        means = collections.defaultdict(list)
        for r in train_rows:
            means[r["candidate"]].append(float(r["ess_total_payoff_cny"]))
        ranked = sorted(((float(np.mean(v)), k) for k, v in means.items()),
                        reverse=True)
        top = [k for _m, k in ranked[:2]]
        val_rows = [r for r in val if r["arm"] == family
                    and r["candidate"] in top]
        val_means = collections.defaultdict(list)
        for r in val_rows:
            val_means[r["candidate"]].append(float(r["ess_total_payoff_cny"]))
        val_ranked = sorted(((float(np.mean(v)), k)
                             for k, v in val_means.items()), reverse=True)
        chosen = val_ranked[0][1] if val_ranked else (top[0] if top else None)
        selected[family] = chosen
        selection["stages"][family] = {
            "train_mean": {k: round(m, 3) for m, k in ranked},
            "top2_by_train": top,
            "validation_mean": {k: round(m, 3) for m, k in val_ranked},
            "selected": chosen,
        }
        _log(log, f"{family}: top2 {top} -> selected {chosen}")
    payload = {
        "seed": SEED,
        "grid": {
            "B2A": {"q_lo": [g[0] for g in GRID_A],
                    "q_hi": [g[1] for g in GRID_A]},
            "B2B": {"q_lo": [g[0] for g in GRID_A],
                    "q_hi": [g[1] for g in GRID_A],
                    "soc_band": [list(b) for b in GRID_B_BANDS]},
        },
        "suppress_quotes": {"charge": CHARGE_SUPPRESS,
                            "discharge": DISCHARGE_SUPPRESS},
        "information": "published day-ahead price quantiles of the same day; "
                       "no realized LMP and no future market outcome",
        "selection": selection["stages"],
        "selected": selected,
        "search": "deterministic grid search on the 53 common training days; "
                  "the top two per family are re-scored on the 11 common "
                  "validation days and one is selected",
    }
    return payload


def _max_env(raw: Path) -> None:
    rows = [r for r in _read_csv(raw) if not r.get("error")]
    if not rows:
        return
    def _mx(key):
        vals = [abs(float(r[key])) for r in rows if r.get(key) not in ("", None)]
        return max(vals) if vals else float("nan")
    print(f"rows {len(rows)}  max |p_res| {_mx('p_balance_max_residual_mw'):.2e}"
          f"  max |q_res| {_mx('q_balance_max_residual_mvar'):.2e}"
          f"  max terminal {_mx('terminal_soc_max_err'):.2e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", default="all",
                        help="comma list of b0,b2train,b2val,b2test,b3")
    parser.add_argument("--raw", default=RAW)
    parser.add_argument("--params", default=PARAMS)
    parser.add_argument("--b3", default=B3_CSV)
    parser.add_argument("--common", default=COMMON)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap days per segment (smoke runs)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    raw = root / args.raw
    log = raw.parent / "run.log"
    segments = set(args.segments.split(","))
    if "all" in segments:
        segments = {"b0", "b2train", "b2val", "b2test", "b3"}

    split = json.loads((root / args.common).read_text(encoding="utf-8"))
    pairs = _read_csv(root / PAIRING)
    for row in pairs:
        row["split"] = _split_of(split, row["load_date"])

    done = _done_keys(raw)
    _log(log, f"benchmark start; {len(pairs)} pairs; segments {sorted(segments)}")
    params_path = root / args.params
    by_id = {_candidate_id(c): c for c in candidates()}

    if "b0" in segments:
        _log(log, f"B0 NoESS: {len(pairs)} days")
        _run_segment(pairs, arm="B0", params={}, ess_buses=[],
                     action_params=None, raw=raw, done=done, log=log,
                     limit=args.limit)

    if "b2train" in segments:
        train = [r for r in pairs if r["split"] == "train"]
        for cand in candidates():
            _log(log, f"{cand['family']} {_candidate_id(cand)}: "
                      f"{len(train)} train days")
            _run_segment(train, arm=cand["family"], params=cand,
                         ess_buses=ESS_BUSES,
                         action_params=lambda price, agents, c=cand:
                         quotes_for(c, price, agents),
                         raw=raw, done=done, log=log, limit=args.limit)

    params_payload = None
    if raw.exists() and any(s in segments for s in ("b2train", "b2val", "b2test")):
        params_payload = _select(raw, log)
        params_path.write_text(json.dumps(params_payload, indent=2),
                               encoding="utf-8")

    if "b2val" in segments:
        if not params_payload:
            raise SystemExit("no training rows to select from")
        val = [r for r in pairs if r["split"] == "validation"]
        for family in ("B2A", "B2B"):
            for cand_id in params_payload["selection"][family]["top2_by_train"]:
                _run_segment(val, arm=family, params=by_id[cand_id],
                             ess_buses=ESS_BUSES,
                             action_params=lambda price, agents, c=by_id[cand_id]:
                             quotes_for(c, price, agents),
                             raw=raw, done=done, log=log, limit=args.limit)
        params_payload = _select(raw, log)
        params_path.write_text(json.dumps(params_payload, indent=2),
                               encoding="utf-8")

    if "b2test" in segments:
        if not params_payload:
            raise SystemExit("no selected parameters")
        final = [r for r in pairs if r["split"] in ("test", "holiday_block")]
        for family in ("B2A", "B2B"):
            cand_id = params_payload["selected"][family]
            if cand_id is None:
                raise SystemExit(f"{family}: no parameter selected")
            _run_segment(final, arm=family, params=by_id[cand_id],
                         ess_buses=ESS_BUSES,
                         action_params=lambda price, agents, c=by_id[cand_id]:
                         quotes_for(c, price, agents),
                         raw=raw, done=done, log=log, limit=args.limit)

    if "b3" in segments:
        _b3(root, args.b3, pairs, log)

    _max_env(raw)
    print(f"wrote {raw}")
    if params_payload:
        print(f"wrote {params_path}")
    if "b3" in segments:
        print(f"wrote {root / args.b3}")
    return 0


def _b3(root: Path, out_path: str, pairs: Sequence[dict], log: Path) -> None:
    """The perfect-foresight price-taker bound against the external price."""
    cycles = np.load(root / PRICE_CURVES)
    config = screen.build_config(100.0)
    fleet = screen.ess.build_ess_fleet(T, config, buses=ESS_BUSES)
    spec = arbitrage_lp.spec_from_storage(fleet[0].storage)
    rows = []
    for row in pairs:
        price = np.asarray(cycles[f"south/{row['price_date']}"], dtype=float)
        unit = arbitrage_lp.arbitrage_lp(price, spec, 100.0)
        rows.append({
            "load_date": row["load_date"],
            "price_date": row["price_date"],
            "group": row["group"],
            "split": row["split"],
            "unit_margin_cny": float(unit.margin_cny),
            "fleet4_upper_bound_cny": 4.0 * float(unit.margin_cny),
            "price_series": "external day-ahead (south)",
            "note": "price-taker, perfect foresight, no nodal price and no "
                    "market impact",
        })
    path = root / out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _log(log, f"B3 LP bound: {len(rows)} days, fleet4 mean "
              f"{np.mean([r['fleet4_upper_bound_cny'] for r in rows]):.1f} CNY/day")


if __name__ == "__main__":
    sys.exit(main())
