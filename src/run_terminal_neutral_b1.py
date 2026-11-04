"""Terminal-neutral B1 over the 24-block rolling interface.

Every RL storage unit starts and must end the day at SOC 0.5. The constraint is
applied only to the rolling windows whose own end is the day's last period, and
it is absolute rather than window-relative: the final four periods have to be
able to land on 0.5 from wherever the earlier blocks left the battery.

The run doubles as the feasibility audit. A day that cannot clear under the
constraint is recorded with the block that failed and the SOC the units carried
into it; nothing is relaxed and no day is dropped.

Outputs (existing rolling results are left untouched):

- data/processed/v2_cy_jn_2025/b1_terminal_neutral_results.csv
- data/processed/v2_cy_jn_2025/b1_terminal_neutral_manifest.json
- reports/v2_cy_jn_2025/b1_terminal_neutral_audit.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import b1_daily_baseline as b
import screen_v2_cy_jn_2025 as screen

PAIRING = screen.PAIRING
NEAR_PAIRING = "data/processed/v2_cy_jn_2025/near/load_price_pairing.csv"
BUS_LOADS = screen.BUS_LOADS
PRICE_CURVES = screen.PRICE_CURVES
OUT = "data/processed/v2_cy_jn_2025"
RESULTS = f"{OUT}/b1_terminal_neutral_results.csv"
BLOCK_PAYOFFS = f"{OUT}/b1_terminal_neutral_block_payoffs.json"
MANIFEST = f"{OUT}/b1_terminal_neutral_manifest.json"
AUDIT = "reports/v2_cy_jn_2025/b1_terminal_neutral_audit.md"
TARGET = 0.5
COLS = ("pair_id", "load_date", "price_date", "group", "in_near", "soc0",
        *[f"G_{u}" for u in b.UNITS], "G_total",
        *[f"soc_end_{u}" for u in b.UNITS],
        *[f"terminal_err_{u}" for u in b.UNITS],
        *[f"charge_mwh_{u}" for u in b.UNITS],
        *[f"discharge_mwh_{u}" for u in b.UNITS],
        *[f"cycles_{u}" for u in b.UNITS],
        "losses_mwh", "voltage_violation_cells", "line_overload_cells",
        "blocks", "clears", "fallbacks", "lmp_fallbacks", "all_optimal",
        "feasible", "failed_blocks", "runtime_s")


def _pair_id(row: dict) -> str:
    return f"{row['load_date']}|{row['price_date']}"


def _mechanism_hash(config_hash: str) -> str:
    """Bind the B1 cache key to the terminal rule, not only to the market.

    The old rolling B1 has the same market config hash and the same initial
    SOC; without the rule in the key the two are indistinguishable and a
    trainer could silently reuse the no-pin benchmark.
    """
    payload = {
        "mechanism": "terminal_neutral_24block",
        "config_hash": config_hash,
        "terminal_soc_target": TARGET,
        "soc0": TARGET,
        "horizon_rule": "day_end_only",
        "window": "16-period lookahead, commit 4",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _read_pairs(path: str) -> List[dict]:
    return b._read(Path(path))


def _row(result: dict, near_ids: set[str]) -> dict:
    row = dict(result)
    row["in_near"] = row["pair_id"] in near_ids
    row["failed_blocks"] = ";".join(str(x) for x in result["failed_blocks"])
    errors = [abs(float(row[f"soc_end_{u}"]) - TARGET) for u in b.UNITS]
    for u in b.UNITS:
        row[f"terminal_err_{u}"] = abs(float(row[f"soc_end_{u}"]) - TARGET)
    row["soc0"] = TARGET
    row["feasible"] = bool(
        row["clears"] == 24 and row["fallbacks"] == 0
        and max(errors) <= 1e-5
        and int(row["voltage_violation_cells"]) == 0
        and int(row["line_overload_cells"]) == 0)
    return row


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _audit(rows: Sequence[dict], near_check: dict, failures: list) -> str:
    total = len(rows)
    feasible = [r for r in rows if r["feasible"]]
    infeasible = [r for r in rows if not r["feasible"]]
    max_err = max((max(float(r[f"terminal_err_{u}"]) for u in b.UNITS)
                   for r in rows), default=0.0)
    lines = [
        "# Terminal-neutral B1 and feasibility audit",
        "",
        "Every RL storage unit starts and must end the day at SOC 0.5. The "
        "constraint is absolute and is applied only to the rolling windows "
        "whose own end is the day's last period (`horizon_type=day_end`); "
        "earlier windows remain unpinned. The truthful arm uses the same "
        "24-block interface, initial SOC, load, price, network, cycle cost and "
        "block settlement as the policy arm.",
        "",
        "| item | value |",
        "|---|---|",
        f"| terminal target | {TARGET} |",
        f"| days run | {total} |",
        f"| feasible (24 clears, 0 fallbacks, end SOC within 1e-5) | "
        f"{len(feasible)} |",
        f"| infeasible | {len(infeasible)} |",
        f"| max per-unit terminal error | {max_err:.3e} |",
        f"| days with a voltage violation | "
        f"{sum(1 for r in rows if int(r['voltage_violation_cells']) > 0)} |",
        f"| days with a line overload | "
        f"{sum(1 for r in rows if int(r['line_overload_cells']) > 0)} |",
        f"| total network losses (MWh) | "
        f"{sum(float(r['losses_mwh']) for r in rows):.3f} |",
        f"| NEAR records found by pair_id | {near_check['matched']} / "
        f"{near_check['expected']} |",
        f"| NEAR pair ids absent from FULL96 | {near_check['missing']} |",
        "",
        "The earlier rolling B1 (`b1_daily_results.csv`) did not pin terminal "
        "SOC and is kept unchanged as `CASHFLOW_NO_TERMINAL`. It is not the "
        "day-level benchmark for the terminal-neutral experiment.",
        "",
    ]
    if infeasible:
        lines += ["## Infeasible days", "",
                  "No constraint was relaxed and no day was removed. Each "
                  "entry names the first block that failed and the SOC carried "
                  "into it.", ""]
        for r in infeasible:
            lines.append(f"- `{r['pair_id']}`: failed block(s) "
                         f"`{r['failed_blocks']}`; end SOC "
                         + ", ".join(f"{u}={r[f'soc_end_{u}']:.4f}"
                                     for u in b.UNITS) + ".")
        lines += ["", "Diagnostics are in the manifest under `failures`.", ""]
    else:
        lines += ["## Infeasible days", "",
                  "None. Every day completed 24 clears with 0 fallbacks and "
                  "ended each unit on the target.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    root = Path.cwd()
    full = _read_pairs(PAIRING)
    near = _read_pairs(NEAR_PAIRING)
    full_ids = {_pair_id(r) for r in full}
    near_ids = {_pair_id(r) for r in near}
    missing = sorted(near_ids - full_ids)
    near_check = {"expected": len(near_ids), "matched": len(near_ids & full_ids),
                  "missing": missing}

    work = full[:args.limit] if args.limit else full
    rows: List[dict] = []
    failures = []
    block_payoffs: Dict[str, Dict[str, List[float]]] = {}
    for idx, pair in enumerate(work, 1):
        result = b.run_truthful_day(pair, terminal_soc_target=TARGET)
        row = _row(result, near_ids)
        rows.append(row)
        block_payoffs[row["pair_id"]] = result["block_payoffs"]
        if result["failed_blocks"]:
            failures.append({
                "pair_id": row["pair_id"],
                "failed_blocks": result["failed_blocks"],
                "soc_at_failure": result["soc_at_failure"],
            })
        print(f"{idx}/{len(work)} {row['pair_id']} feasible={row['feasible']} "
              f"fallbacks={row['fallbacks']}", flush=True)

    _write_csv(root / RESULTS, rows)
    (root / BLOCK_PAYOFFS).write_text(
        json.dumps(block_payoffs, indent=2), encoding="utf-8")
    config = screen.build_config(100.0)
    manifest = {
        "dataset": "b1_terminal_neutral",
        "version": "b1-terminal-neutral-24block-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mechanism": "24 blocks; each clears a 16-period rolling window and "
                     "commits 4; windows ending at the true day end pin every "
                     "storage unit's absolute SOC to 0.5; all four RL units "
                     "truthful; no intermediate-window pin",
        "not_jn_base": True,
        "not_old_rolling_b1": True,
        "soc0": TARGET,
        "terminal_soc_target": TARGET,
        "config_hash": b.config_hash(config),
        "mechanism_hash": _mechanism_hash(b.config_hash(config)),
        "inputs": {
            "pairing": {"path": PAIRING,
                        "sha256": b._sha(root / PAIRING)},
            "near_pairing": {"path": NEAR_PAIRING,
                             "sha256": b._sha(root / NEAR_PAIRING)},
            "bus_loads": {"path": BUS_LOADS,
                          "sha256": b._sha(root / BUS_LOADS)},
            "price_curves": {"path": PRICE_CURVES,
                             "sha256": b._sha(root / PRICE_CURVES)},
        },
        "outputs": {
            "results": {"path": RESULTS,
                        "sha256": b._sha(root / RESULTS)},
            "block_payoffs": {"path": BLOCK_PAYOFFS,
                              "sha256": b._sha(root / BLOCK_PAYOFFS)},
        },
        "days": len(rows),
        "feasible_days": sum(1 for r in rows if r["feasible"]),
        "near": near_check,
        "failures": failures,
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2),
                                 encoding="utf-8")
    (root / AUDIT).write_text(_audit(rows, near_check, failures),
                              encoding="utf-8")
    print(f"wrote {RESULTS}\nwrote {MANIFEST}\nwrote {AUDIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
