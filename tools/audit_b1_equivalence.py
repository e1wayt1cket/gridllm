# audit_b1_equivalence.py
"""B1 Truthful is JN-Base: verify the equivalence instead of repeating it.

B1 is not an independent experiment. `clear_market(action_params={})` maps
every participant to its own truthful point -- storage to `(0, 0)` and each
load or prosumer to `(bid_mult, offer_adder) = (1, 0)` -- so B1 and JN-Base are
the same call on the same agents, inputs and configuration. This audit checks
that chain for all 96 days and validates the stored JN-Base results. It does
**not** re-solve: a second solve would only show that the engine is
deterministic, not that B1 is a different experiment.

Discrete fields must match exactly. Continuous fields are checked against
declared tolerances imported from the NEAR reuse audit, so the two audits share
one set.

Usage:
    PYTHONPATH=src python src/audit_b1_equivalence.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# `screen` pulls `grid` in first, which loads ortools before pandas. The
# reverse order fails on this platform with WinError 127.
import screen_v2_cy_jn_2025 as screen
import audit_jn_price_source as price_audit
import pair_v2_cy_jn_2025 as pj
from audit_near_clearing_reuse import (
    BAND_SLACK, EXCLUSIVE_MAX_MW2, LINE_SLACK, LMP_SLACK_MAX_CNY_MWH,
    P_RESIDUAL_MAX_MW, Q_RESIDUAL_MAX_MVAR, TERMINAL_SOC_MAX,
)
from dispatch_core import quote_arrays

PAIRING = screen.PAIRING
BUS_LOADS = screen.BUS_LOADS
PRICE_CURVES = screen.PRICE_CURVES
RESULTS = "reports/v2_cy_jn_2025/results.csv"
CONFIG = "reports/v2_cy_jn_2025/config.json"
RUN_MANIFEST = "reports/v2_cy_jn_2025/manifest.json"
OUT = "data/processed/v2_cy_jn_2025/benchmark/b1_equivalence.csv"
REPORT = "reports/v2_cy_jn_2025/b1_truthful_equivalence_audit.md"
ESS_BUSES = [9, 14, 17, 27]

CHECK_KEYS = (
    "agents_ok", "storage_specs_ok", "storage_quotes_truthful",
    "prosumer_quotes_truthful", "load_digest_ok", "price_digest_ok",
    "result_unique", "result_flags_ok", "inputs_ok", "voltage_ok",
    "line_ok", "p_balance_ok", "q_balance_ok", "terminal_soc_ok",
    "exclusive_ok", "soc_band_ok", "lmp_ok", "pcc_ok", "no_error",
)

COLUMNS = (
    "load_date", "price_date", "group", "agents_ok", "storage_specs_ok",
    "storage_quotes_truthful", "prosumer_quotes_truthful",
    "load_digest_ok", "price_digest_ok", "result_unique", "result_flags_ok",
    "inputs_ok", "voltage_ok", "line_ok", "p_balance_ok", "q_balance_ok",
    "terminal_soc_ok", "exclusive_ok", "soc_band_ok", "lmp_ok", "pcc_ok",
    "no_error", "passed",
)


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _num(row: dict, key: str) -> float:
    return float(row[key])


def _config_checks(config) -> Dict[str, bool]:
    net = config.network
    storage = config.storage
    return {
        "opf_is_socp": config.opf_mode == "socp",
        "cycle_cost_is_100": float(storage.cycle_cost) == 100.0,
        "terminal_soc_pinned": bool(storage.terminal_soc_equal),
        "impedance_scale_is_0_5": float(net.line_impedance_scale) == 0.5,
        "reactive_fraction_is_2": float(net.reactive_fraction_of_load_q) == 2.0,
        "band_is_0_93_1_05": (float(net.v_min_pu) == 0.93
                              and float(net.v_max_pu) == 1.05),
        "reverse_limit_is_2_5": float(net.reverse_power_limit_mw) == 2.5,
    }


def _agent_checks(agents) -> Dict[str, bool]:
    storage = [a for a in agents if a.storage is not None]
    names = sorted(a.name for a in agents)
    storage_names = sorted(a.name for a in storage)
    buses = sorted(int(a.bus) for a in storage)
    specs_ok = all(
        abs(float(a.storage.e_max) - 1.6) < 1e-12
        and abs(float(a.storage.p_ch_max) - 0.4) < 1e-12
        and abs(float(a.storage.p_dis_max) - 0.4) < 1e-12
        and abs(float(a.storage.eta_ch) - 0.92) < 1e-12
        and abs(float(a.storage.eta_dis) - 0.92) < 1e-12
        and abs(float(a.storage.soc0) - 0.5) < 1e-12
        and abs(float(a.storage.soc_min) - 0.10) < 1e-12
        and abs(float(a.storage.soc_max) - 0.90) < 1e-12
        for a in storage)
    return {
        "agents_ok": len(agents) == 36 and len(storage) == 4
                     and len(set(names)) == 36,
        "storage_specs_ok": (storage_names
                             == sorted(f"ESS{b}" for b in ESS_BUSES)
                             and buses == ESS_BUSES and specs_ok),
    }


def _quote_checks(agents, config) -> Dict[str, float | bool]:
    max_storage = 0.0
    max_prosumer = 0.0
    for agent in agents:
        bid, offer = quote_arrays(agent, {}, screen.T, config)
        if agent.storage is not None:
            max_storage = max(max_storage,
                              float(np.max(np.abs(bid))),
                              float(np.max(np.abs(offer))))
        else:
            base_bid = np.full(screen.T, float(agent.bid_value))
            base_offer = np.full(screen.T, float(agent.offer_cost))
            max_prosumer = max(
                max_prosumer,
                float(np.max(np.abs(bid - base_bid))),
                float(np.max(np.abs(offer - base_offer))))
    return {
        "storage_quotes_truthful": max_storage == 0.0,
        "prosumer_quotes_truthful": max_prosumer == 0.0,
        "max_storage_quote_dev": max_storage,
        "max_prosumer_quote_dev": max_prosumer,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", default=PAIRING)
    parser.add_argument("--bus-loads", default=BUS_LOADS)
    parser.add_argument("--price-curves", default=PRICE_CURVES)
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--run-manifest", default=RUN_MANIFEST)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--report", default=REPORT)
    args = parser.parse_args(argv)
    root = Path.cwd()

    pairs = _read_csv(root / args.pairing)
    results = [r for r in _read_csv(root / args.results)
               if r.get("arm") == "JN-Base"]
    result_by_id = {f"{r['load_date']}|{r['price_date']}": r for r in results}
    loads = np.load(root / args.bus_loads)
    curves = np.load(root / args.price_curves)
    run_manifest = json.loads((root / args.run_manifest).read_text(encoding="utf-8"))

    config = screen.build_config(100.0)
    config_checks = _config_checks(config)
    input_checks = {}
    for key in ("pairing", "bus_loads", "price_curves"):
        entry = run_manifest["inputs"][key]
        input_checks[f"{key}_hash_matches_manifest"] = (
            _sha256(root / entry["path"]) == entry["sha256"])

    rows: List[dict] = []
    max_errors = {"p_residual_mw": 0.0, "q_residual_mvar": 0.0,
                  "terminal_soc": 0.0, "exclusive_mw2": 0.0,
                  "slack_lmp_dev": 0.0, "storage_quote_dev": 0.0,
                  "prosumer_quote_dev": 0.0}
    for pair in pairs:
        pid = f"{pair['load_date']}|{pair['price_date']}"
        price = np.asarray(curves[f"south/{pair['price_date']}"], dtype=float)
        matrix = np.asarray(loads[f"load/{pair['load_date']}"], dtype=float)
        _net, agents = screen.build_agents(config, matrix, price, ESS_BUSES)

        agent_checks = _agent_checks(agents)
        quote_checks = _quote_checks(agents, config)
        max_errors["storage_quote_dev"] = max(
            max_errors["storage_quote_dev"],
            float(quote_checks["max_storage_quote_dev"]))
        max_errors["prosumer_quote_dev"] = max(
            max_errors["prosumer_quote_dev"],
            float(quote_checks["max_prosumer_quote_dev"]))

        load_ok = (pj.load_curve_digest(pair["load_date"], matrix)
                   == pair["load_curve_sha256"])
        price_ok = (price_audit.curve_digest(pair["price_date"], "south", price)
                    == pair["price_curve_sha256"])

        result = result_by_id.get(pid)
        flags_ok = bool(
            result and result.get("arm") == "JN-Base"
            and _as_bool(result.get("optimal"))
            and result.get("solver_status") == "OPTIMAL"
            and int(float(result["ess_units"])) == 4
            and float(result["cycle_cost"]) == 100.0
            and int(float(result["lmp_fallbacks"])) == 0
            and _as_bool(result.get("q_balance_checked"))
            and _as_bool(result.get("reverse_within_limit"))
            and not result.get("error"))

        if result:
            p_err = abs(_num(result, "p_balance_max_residual_mw"))
            q_err = abs(_num(result, "q_balance_max_residual_mvar"))
            term = abs(_num(result, "terminal_soc_max_err"))
            excl = abs(_num(result, "exclusive_max_mw2"))
            slack = abs(_num(result, "lmp_slack_max_dev_cny_mwh"))
            max_errors["p_residual_mw"] = max(max_errors["p_residual_mw"], p_err)
            max_errors["q_residual_mvar"] = max(max_errors["q_residual_mvar"], q_err)
            max_errors["terminal_soc"] = max(max_errors["terminal_soc"], term)
            max_errors["exclusive_mw2"] = max(max_errors["exclusive_mw2"], excl)
            max_errors["slack_lmp_dev"] = max(max_errors["slack_lmp_dev"], slack)
            vmin = _num(result, "min_bus_voltage_pu")
            vmax = _num(result, "max_bus_voltage_pu")
            checks = {
                "voltage_ok": (int(float(result["voltage_violation_cells"])) == 0
                               and vmin >= 0.93 - BAND_SLACK
                               and vmax <= 1.05 + BAND_SLACK),
                "line_ok": (int(float(result["line_overload_cells"])) == 0
                            and _num(result, "max_line_utilization")
                            <= 1.0 + LINE_SLACK),
                "p_balance_ok": p_err <= P_RESIDUAL_MAX_MW,
                "q_balance_ok": q_err <= Q_RESIDUAL_MAX_MVAR,
                "terminal_soc_ok": term <= TERMINAL_SOC_MAX,
                "exclusive_ok": excl <= EXCLUSIVE_MAX_MW2,
                "soc_band_ok": (_num(result, "ess_soc_min") >= 0.10 - 1e-9
                                and _num(result, "ess_soc_max")
                                <= 0.90 + 1e-9),
                "lmp_ok": slack <= LMP_SLACK_MAX_CNY_MWH,
                "pcc_ok": _as_bool(result.get("reverse_within_limit")),
                "no_error": not result.get("error"),
            }
        else:
            checks = {key: False for key in (
                "voltage_ok", "line_ok", "p_balance_ok", "q_balance_ok",
                "terminal_soc_ok", "exclusive_ok", "soc_band_ok", "lmp_ok",
                "pcc_ok", "no_error")}

        row = {
            "load_date": pair["load_date"],
            "price_date": pair["price_date"],
            "group": pair["group"],
            "agents_ok": agent_checks["agents_ok"],
            "storage_specs_ok": agent_checks["storage_specs_ok"],
            "storage_quotes_truthful": quote_checks["storage_quotes_truthful"],
            "prosumer_quotes_truthful": quote_checks["prosumer_quotes_truthful"],
            "load_digest_ok": load_ok,
            "price_digest_ok": price_ok,
            "result_unique": pid in result_by_id,
            "result_flags_ok": flags_ok,
            "inputs_ok": all(input_checks.values()),
            **checks,
        }
        row["passed"] = all(row[key] for key in CHECK_KEYS)
        rows.append(row)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for r in rows if r["passed"])
    per_check = {key: sum(1 for r in rows if r[key]) for key in CHECK_KEYS}
    lines = [
        "# B1 Truthful equivalence audit (V2-CY-JN-2025)",
        "",
        f"**Result: {passed}/{len(rows)} days pass. B1 is JN-Base, not an "
        "independent repeated experiment.**",
        "",
        "B1 reuses the JN-Base clears; no second solve was run. Equivalence is "
        "established by the code path, the construction and the stored "
        "results, not by re-solving a deterministic engine and comparing it "
        "with itself.",
        "",
        "## The mapping",
        "",
        "`clear_market(action_params={})` sends every participant through "
        "`dispatch_core.quote_arrays`. A storage agent with no declaration "
        "falls back to `ACTION_STORAGE_QUOTE_V2.truthful`, `(charge_price, "
        "discharge_price) = (0, 0)`; a partial declaration raises instead of "
        "being completed. A load or prosumer with no declaration uses "
        "`(bid_mult, offer_adder) = (1, 0)`, so its bid array is its own "
        "`bid_value` and its offer array its own `offer_cost`. This was "
        "checked for every agent on every one of the 96 days: the largest "
        f"storage quote deviation is {max_errors['storage_quote_dev']:.2e} "
        "CNY/MWh and the largest prosumer deviation is "
        f"{max_errors['prosumer_quote_dev']:.2e}.",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for key, count in per_check.items():
        lines.append(f"| {key.replace('_', ' ')} | {count}/{len(rows)} |")
    lines += [
        "",
        "### Configuration and inputs",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for key, ok in {**config_checks, **input_checks}.items():
        lines.append(f"| {key.replace('_', ' ')} | {'PASS' if ok else 'FAIL'} |")
    lines += [
        "",
        "### Numeric margins (maximum over 96 days)",
        "",
        "| quantity | max | tolerance |",
        "|---|---:|---:|",
        f"| active-balance residual (MW) | "
        f"{max_errors['p_residual_mw']:.2e} | {P_RESIDUAL_MAX_MW:g} |",
        f"| reactive-balance residual (MVAr) | "
        f"{max_errors['q_residual_mvar']:.2e} | {Q_RESIDUAL_MAX_MVAR:g} |",
        f"| terminal SOC error | {max_errors['terminal_soc']:.2e} | "
        f"{TERMINAL_SOC_MAX:g} |",
        f"| charge/discharge product (MW^2) | "
        f"{max_errors['exclusive_mw2']:.2e} | {EXCLUSIVE_MAX_MW2:g} |",
        f"| slack-LMP deviation (CNY/MWh) | "
        f"{max_errors['slack_lmp_dev']:.2e} | {LMP_SLACK_MAX_CNY_MWH:g} |",
        "",
        "## Hashes",
        "",
        f"- `results.csv`: `{_sha256(root / args.results)}`",
        f"- `config.json`: `{_sha256(root / args.config)}`",
        f"- run manifest inputs match the files on disk: "
        f"{'yes' if all(input_checks.values()) else 'no'}",
        f"- per-day audit: `{args.out}`",
        "",
        "## What this does not show",
        "",
        "It does not show that a fresh B1 run would reproduce these numbers; "
        "the engine is deterministic, so that would be a weaker check than the "
        "one above. It also does not make B1 an independent sample: B1 and "
        "JN-Base are one clearing described twice, and a benchmark table must "
        "not count them as two arms of evidence.",
    ]
    report = root / args.report
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"days passed           {passed}/{len(rows)}")
    for key, count in per_check.items():
        if count != len(rows):
            print(f"  FAILED {key}: {count}/{len(rows)}")
    print(f"max p_res {max_errors['p_residual_mw']:.2e} "
          f"q_res {max_errors['q_residual_mvar']:.2e} "
          f"terminal {max_errors['terminal_soc']:.2e} "
          f"slack {max_errors['slack_lmp_dev']:.2e}")
    print(f"config checks         {sum(config_checks.values())}/{len(config_checks)}")
    print(f"input checks          {sum(input_checks.values())}/{len(input_checks)}")
    print(f"wrote {out_path}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
