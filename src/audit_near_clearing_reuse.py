# audit_near_clearing_reuse.py
"""Can NEAR reuse the FULL96 JN-Base clears instead of re-running them?

A NEAR row is a copy of a FULL96 row, so its load and price curves are the same
bytes and the same clearing settings apply. Reuse is registered only when every
link in that chain is checked: a unique FULL96 record, identical dates,
identical curve digests, the same SOCP configuration and support assumption,
and an existing JN-Base clear that is OPTIMAL and passes the physical checks.
A record that fails any link is listed, never silently filled.

**Scope is JN-Base only.** The 21-day C80/C90/C100/NoESS probe is not a NEAR
full-sample sensitivity result and is not registered here.

This reads results; it does not clear a market.

Usage:
    PYTHONPATH=src python src/audit_near_clearing_reuse.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

FULL = "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
NEAR = "data/processed/v2_cy_jn_2025/near/load_price_pairing.csv"
RESULTS = "reports/v2_cy_jn_2025/results.csv"
CONFIG = "reports/v2_cy_jn_2025/config.json"
RUN_MANIFEST = "reports/v2_cy_jn_2025/manifest.json"
INDEX = "data/processed/v2_cy_jn_2025/near/clearing_reuse.csv"
REPORT = "reports/v2_cy_jn_2025/near_clearing_reuse_audit.md"
DATASETS = "data/processed/v2_cy_jn_2025/datasets.json"
ARM = "JN-Base"

#: Declared tolerances, not fitted to the data. A genuine clear is far inside
#: every one of them; the numbers only say how far inside counts as the same
#: result rather than a different one.
P_RESIDUAL_MAX_MW = 1e-5
Q_RESIDUAL_MAX_MVAR = 1e-5
TERMINAL_SOC_MAX = 1e-9
EXCLUSIVE_MAX_MW2 = 1e-6
LMP_SLACK_MAX_CNY_MWH = 1e-2
BAND_SLACK = 1e-6
LINE_SLACK = 1e-6


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _num(row: dict, key: str) -> float:
    return float(row[key])


def _label(key: str) -> str:
    """Readable check names; the embedded decimals must not lose their dots."""
    return (key.replace("is_0_5", "= 0.5")
            .replace("is_0_93_1_05", "= 0.93-1.05")
            .replace("_", " "))


def pair_id(row: dict) -> str:
    return f"{row['load_date']}|{row['price_date']}"


def _config_checks(config: dict) -> Dict[str, bool]:
    support = config.get("network_support", {})
    storage = config.get("storage", {})
    return {
        "support_is_line50_q2": support.get("assumption") == "line50_q2.0",
        "impedance_scale_is_0_5": support.get("line_impedance_scale") == 0.5,
        "reactive_fraction_is_2": support.get("reactive_fraction_of_load_q") == 2.0,
        "band_is_0_93_1_05": (support.get("v_min_pu") == 0.93
                              and support.get("v_max_pu") == 1.05),
        "terminal_soc_pinned": storage.get("terminal_soc_equal") is True,
        "storage_rating_matches": (storage.get("capacity_mwh") == 1.6
                                   and storage.get("power_mw") == 0.4),
    }


def _manifest_checks(manifest: dict, root: Path) -> Dict[str, bool]:
    checks = {}
    for key in ("pairing", "bus_loads", "price_curves"):
        entry = manifest.get("inputs", {}).get(key) or {}
        path = entry.get("path")
        checks[f"manifest_{key}_hash"] = bool(
            path and (root / path).exists()
            and _sha256(root / path) == entry.get("sha256"))
    return checks


def _quality_checks(row: dict, config: dict) -> Dict[str, bool]:
    support = config["network_support"]
    return {
        "optimal": _as_bool(row.get("optimal")),
        "solver_optimal": row.get("solver_status") == "OPTIMAL",
        "voltage_ok": (_num(row, "voltage_violation_cells") == 0
                       and _num(row, "min_bus_voltage_pu")
                       >= support["v_min_pu"] - BAND_SLACK
                       and _num(row, "max_bus_voltage_pu")
                       <= support["v_max_pu"] + BAND_SLACK),
        "line_ok": (_num(row, "line_overload_cells") == 0
                    and _num(row, "max_line_utilization")
                    <= 1.0 + LINE_SLACK),
        "p_balance_ok": abs(_num(row, "p_balance_max_residual_mw")) \
            <= P_RESIDUAL_MAX_MW,
        "q_balance_ok": (_as_bool(row.get("q_balance_checked"))
                         and abs(_num(row, "q_balance_max_residual_mvar"))
                         <= Q_RESIDUAL_MAX_MVAR),
        "terminal_soc_ok": abs(_num(row, "terminal_soc_max_err")) \
            <= TERMINAL_SOC_MAX,
        "exclusive_ok": abs(_num(row, "exclusive_max_mw2")) \
            <= EXCLUSIVE_MAX_MW2,
        "lmp_ok": (_num(row, "lmp_fallbacks") == 0
                   and abs(_num(row, "lmp_slack_max_dev_cny_mwh"))
                   <= LMP_SLACK_MAX_CNY_MWH),
        "pcc_ok": _as_bool(row.get("reverse_within_limit")),
        "no_error": not row.get("error"),
    }


COLUMNS = (
    "pair_id", "load_date", "price_date", "group",
    "unique_full_record", "dates_match", "load_hash_match",
    "price_hash_match", "result_unique", "arm_is_jn_base",
    "config_match", "reusable",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", default=FULL)
    parser.add_argument("--near", default=NEAR)
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--run-manifest", default=RUN_MANIFEST)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--datasets", default=DATASETS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    full_rows = _read_csv(root / args.full)
    near_rows = _read_csv(root / args.near)
    results = _read_csv(root / args.results)
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    manifest = json.loads((root / args.run_manifest).read_text(encoding="utf-8"))

    full_by_id = collections.Counter(pair_id(r) for r in full_rows)
    full_row = {pair_id(r): r for r in full_rows}
    base = [r for r in results if r.get("arm") == ARM]
    base_by_id = collections.Counter(pair_id(r) for r in base)
    base_row = {pair_id(r): r for r in base}

    config_checks = _config_checks(config)
    manifest_checks = _manifest_checks(manifest, root)
    shared_checks = {**config_checks, **manifest_checks}

    detail: List[dict] = []
    check_counts: collections.Counter = collections.Counter()
    total_checks = 0
    for row in near_rows:
        pid = pair_id(row)
        full_match = full_by_id.get(pid, 0) == 1
        linked = full_row.get(pid)
        result_match = base_by_id.get(pid, 0) == 1
        result = base_row.get(pid) or {}
        checks = {
            "unique_full_record": full_match,
            "dates_match": bool(
                linked and linked["load_date"] == row["load_date"]
                and linked["price_date"] == row["price_date"]),
            "load_hash_match": bool(
                linked and linked["load_curve_sha256"] == row["load_curve_sha256"]),
            "price_hash_match": bool(
                linked and linked["price_curve_sha256"]
                == row["price_curve_sha256"]),
            "result_unique": result_match,
            "arm_is_jn_base": bool(result and result.get("arm") == ARM
                                   and _num(result, "cycle_cost") == 100.0
                                   and int(float(result["ess_units"])) == 4),
            "config_match": all(shared_checks.values()),
            **(_quality_checks(result, config) if result else
               {"optimal": False, "solver_optimal": False, "voltage_ok": False,
                "line_ok": False, "p_balance_ok": False, "q_balance_ok": False,
                "terminal_soc_ok": False, "exclusive_ok": False,
                "lmp_ok": False, "pcc_ok": False, "no_error": False}),
        }
        checks["reusable"] = all(checks.values())
        total_checks += 1
        for key, ok in checks.items():
            check_counts[key] += bool(ok)
        detail.append({
            "pair_id": pid,
            "load_date": row["load_date"],
            "price_date": row["price_date"],
            "group": row["group"],
            "unique_full_record": checks["unique_full_record"],
            "dates_match": checks["dates_match"],
            "load_hash_match": checks["load_hash_match"],
            "price_hash_match": checks["price_hash_match"],
            "result_unique": checks["result_unique"],
            "arm_is_jn_base": checks["arm_is_jn_base"],
            "config_match": checks["config_match"],
            "reusable": checks["reusable"],
        })

    reusable = sum(1 for r in detail if r["reusable"])
    failures = [r for r in detail if not r["reusable"]]

    index_path = root / args.index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(detail)

    lines = [
        "# NEAR clearing-reuse audit (V2-CY-JN-2025)",
        "",
        f"**Result: {reusable} of {len(detail)} NEAR pairs reuse the FULL96 "
        f"{ARM} clearing.**" + ("" if reusable == len(detail) else
                                f" {len(failures)} pair(s) do not, and are "
                                "listed below; nothing was filled in."),
        "",
        "A NEAR row is a copy of a FULL96 row, so the question is whether every "
        "link holds: a unique source record, the same two dates, the same load "
        "and price digests, the same clearing configuration and support "
        "assumption, and a clear that is OPTIMAL and passes the physical "
        "checks. Reuse is registered only when all of them pass.",
        "",
        "**Scope: JN-Base only.** The 21-day C80/C90/C100/NoESS probe is not a "
        "NEAR full-sample sensitivity result, is not audited here, and must not "
        "be described as one.",
        "",
        "## Per-check pass count",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for key, count in check_counts.items():
        lines.append(f"| {_label(key)} | {count}/{total_checks} |")
    lines += [
        "",
        "## Clearing configuration",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for key, ok in shared_checks.items():
        lines.append(f"| {_label(key)} | {'PASS' if ok else 'FAIL'} |")
    lines += [
        "",
        f"Tolerances: active residual <= {P_RESIDUAL_MAX_MW:g} MW, reactive "
        f"residual <= {Q_RESIDUAL_MAX_MVAR:g} MVAr, terminal SOC <= "
        f"{TERMINAL_SOC_MAX:g}, charge/discharge product <= "
        f"{EXCLUSIVE_MAX_MW2:g}, slack-LMP deviation <= "
        f"{LMP_SLACK_MAX_CNY_MWH:g} CNY/MWh. They are declared, not fitted.",
        "",
    ]
    if failures:
        lines += [
            "## Pairs that cannot be reused",
            "",
            "| pair_id | failed checks |",
            "|---|---|",
        ]
        for row in failures:
            failed = [k.replace("_", " ") for k, v in row.items()
                      if isinstance(v, bool) and not v and k != "reusable"]
            lines.append(f"| {row['pair_id']} | {', '.join(failed)} |")
        lines.append("")
    lines += [
        "## What this does and does not show",
        "",
        "- It shows the FULL96 clear for each NEAR day used the same pair, the "
        "same curves, the same SOCP configuration, the same support assumption, "
        "and returned an OPTIMAL result that passes every physical check.",
        "- It does not re-run the engine. Reuse rests on the inputs and "
        "settings being identical, not on a second solve producing the same "
        "numbers.",
        "- It does not extend to the cost-sensitivity arms: their 21-day "
        "sample is not NEAR.",
        "",
        f"- Per-pair index: `{args.index}`",
    ]
    report_path = root / args.report
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    datasets_path = root / args.datasets
    datasets = json.loads(datasets_path.read_text(encoding="utf-8"))
    datasets["near_jn_base_reuse"] = {
        "audit": args.report,
        "index": args.index,
        "index_sha256": _sha256(index_path),
        "scope": f"{ARM} only",
        "reusable_pairs": reusable,
        "total_near_pairs": len(detail),
        "fully_verified": reusable == len(detail),
        "not_applicable": "C80/C90/C100/NoESS (21-day probe, not NEAR)",
    }
    datasets_path.write_text(json.dumps(datasets, indent=2), encoding="utf-8")

    print(f"NEAR pairs            {len(detail)}")
    print(f"reusable              {reusable}")
    print(f"failed                {len(failures)}")
    print(f"config checks         {sum(shared_checks.values())}/{len(shared_checks)}")
    print(f"wrote {index_path}")
    print(f"wrote {report_path}")
    print(f"updated {datasets_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
