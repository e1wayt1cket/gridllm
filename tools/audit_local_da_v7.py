"""Audit the v7 training price stack against whole-day clearing.

The training path prices a 20-period window inside a 96-period day; the audit
path clears the whole day at once with the market's declared reactive asset,
the bounded inverter support and an external reactive device. This measures
how far the two price stacks disagree, records the block-to-day gap per
committed period, re-derives the deviation weight, and reads the finished
smoke record. It trains nothing and clears no settlement.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import grid  # noqa: F401  ortools must load before pandas on this platform
import numpy as np

from calibrate_dev_penalty_v7 import calibrate
from da_market_data import DEGRADATION_CNY_MWH, MarketRules
from da_market_model import MarketModel
from da_market_pricing import qualify_prices
from local_da_audit import (SCREEN, compare_stacks, feasibility_selected,
                            rolling_gap)
from local_da_clearing import clear_day
from local_da_contracts import Generator
from local_da_days import STORAGE_BUSES, load_pair, window_day
from local_da_operating import OperatingRules, support_day
from rl_da_env import BLOCK_SIZE, N_BLOCKS, ROLL_HORIZON
from run_local_da import build_day, write_json

PERIODS = BLOCK_SIZE * N_BLOCKS


def audit_day(date):
    """The audit-side construction, recorded so it is never read as the library day."""
    day, _ = build_day(date)
    day = replace(day, terminal_soc_equal=True,
                  storage=tuple(replace(unit, degradation_cny_mwh=DEGRADATION_CNY_MWH)
                                for unit in day.storage))
    day, ratings = support_day(day, OperatingRules())
    rules = MarketRules()
    day = replace(day, generators=day.generators + (
        Generator("EXTERNAL33", rules.external_q_bus, np.zeros(day.periods),
                  reactive_limit_mvar=rules.external_q_capacity_mvar),))
    rules.validate(day)
    return day, ratings, rules


def audit_construction(day, ratings):
    return {
        "periods": day.periods,
        "terminal_soc_equal": day.terminal_soc_equal,
        "degradation_cny_mwh": [unit.degradation_cny_mwh for unit in day.storage],
        "inverter_ratings_mvar": ratings,
        "reactive_devices": [g.name for g in day.generators
                             if g.reactive_limit_mvar and not g.available_mw.any()],
        "external_reactive_bus": 32,
    }


def audit_prices(date):
    """Whole-day dispatch and its qualified price stack, or the refusal."""
    day, ratings, rules = audit_day(date)
    model = MarketModel(day, ratings, rules, exact=True)
    try:
        dispatch = model.solve()
    finally:
        model.close()
    qualified = qualify_prices(day, ratings, rules, dispatch, backend="cutting")
    pricing = qualified.get("pricing") or {}
    return {"construction": audit_construction(day, ratings),
            "dispatch_status": int(dispatch["solver_status"]),
            "qualified_for_settlement": bool(qualified["qualified_for_settlement"]),
            "failure": qualified.get("failure"),
            "lmp": (np.asarray(pricing["lmp_cny_mwh"], dtype=float)
                    if "lmp_cny_mwh" in pricing else None)}


def window_stack(pair):
    """The per-period price the training path sees, block by block."""
    stack = np.full((PERIODS, pair["active"].shape[1]), np.nan)
    blocks, failures = {}, []
    soc = {f"ESS{bus}": .5 for bus in STORAGE_BUSES}
    for block in range(N_BLOCKS):
        t_start = block * BLOCK_SIZE
        t_end = min(t_start + ROLL_HORIZON, PERIODS)
        day = window_day(pair, t_start, t_end, soc=soc, quotes={})
        output = clear_day(day, price_probes=False)
        pricing = output.get("pricing") or {}
        if "lmp_cny_mwh" not in pricing:
            failures.append({"block": block, "failure": output.get("failure")})
            continue
        committed = np.asarray(pricing["lmp_cny_mwh"], dtype=float)[:BLOCK_SIZE]
        blocks[block] = committed
        stack[t_start:t_start + len(committed)] = committed
    return stack, blocks, failures


def acceptance_five(day_lmp, blocks):
    """Block prices against the day's own price for the periods they commit."""
    per_bus = {}
    for bus in STORAGE_BUSES:
        block_prices = np.concatenate([blocks[b][:, bus] for b in sorted(blocks)],
                                      dtype=float)
        per_bus[str(bus)] = rolling_gap(day_lmp[:, bus], block_prices,
                                        BLOCK_SIZE)
    return per_bus


def acceptance_seven(episodes):
    """The smoke record's own flags, read rather than re-run."""
    checks = []
    for episode in episodes:
        checks.append({
            "date": episode["date"],
            "rewards_finite": episode["rewards_finite"],
            "blocks_cleared": episode["clears"] == N_BLOCKS,
            "no_none_clear": episode["none_clears"] == 0,
            "soc_in_band": (.1 - 1e-9 <= episode["soc_min"] and
                            episode["soc_max"] <= .9 + 1e-9),
            "declared_point_deviation_zero": episode["declared_point_deviation"] == 0,
            "inventory_terms_one": episode["inventory_terms"] == 1,
        })
    keys = [k for k in checks[0] if k != "date"] if checks else []
    passed = bool(checks) and all(all(row[k] for k in keys) for row in checks)
    return {"verdict": "PASS" if passed else "FAIL", "episodes": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--floor", type=float, default=1.0)
    args = parser.parse_args()

    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    summary = {"days": smoke["days"], "floor_cny_mwh": args.floor}

    four, five = {}, {}
    for date in smoke["days"]:
        audit = audit_prices(date)
        pair = load_pair(date)
        stack, blocks, failures = window_stack(pair)
        if audit["lmp"] is None:
            four[date] = {"verdict": "BLOCKED", "failure": audit["failure"],
                          "days_blocked": 1}
            five[date] = {"verdict": "BLOCKED", "failure": audit["failure"]}
            continue
        four[date] = dict(
            compare_stacks(stack[:, STORAGE_BUSES], audit["lmp"][:, STORAGE_BUSES],
                           floor=args.floor),
            block_failures=failures)
        five[date] = {"by_bus": acceptance_five(audit["lmp"], blocks),
                      "block_failures": failures}
        summary.setdefault("construction", audit["construction"])
        summary.setdefault("dispatch_status", {})[date] = audit["dispatch_status"]

    def rate_four(rows):
        usable = [r for r in rows if r.get("sign_agreement") is not None]
        if len(usable) < len(rows):
            return "BLOCKED"
        ok = all(r["sign_agreement"] >= .95 and r["median_rel_gap"] <= .10
                 for r in usable)
        return "PASS" if ok else "FAIL"

    summary["acceptance_4"] = {"verdict": rate_four(list(four.values())),
                               "criterion": "sign_agreement >= 0.95 and "
                                            "median_rel_gap <= 0.10",
                               "per_day": four}
    summary["acceptance_5"] = {"per_day": five}

    calibration = calibrate(seed=123)
    summary["acceptance_6"] = {
        "verdict": "PASS" if calibration["gradient_share"] <= .50 else "FAIL",
        "criterion": "gradient_share <= 0.50",
        "gradient_share": calibration["gradient_share"],
        "selected_weight": calibration["selected_weight"],
        "rows": [{k: row[k] for k in
                  ("lambda", "econ_grad_norm", "pen_grad_norm", "grad_ratio",
                   "mean_deviation_after_probe", "corner_fraction")}
                 for row in calibration["rows"]],
    }
    summary["acceptance_7"] = acceptance_seven(smoke["episodes"])
    summary["acceptance_8"] = feasibility_selected(SCREEN)

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", summary)
    for key in ("acceptance_4", "acceptance_6", "acceptance_7"):
        print(key, summary[key]["verdict"])


if __name__ == "__main__":
    main()
