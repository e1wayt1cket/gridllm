"""Phase 7A: post-hoc economic decomposition and Cycle Cost sensitivity.

Read-only over the Phase 6 clearing artifacts. Nothing is re-cleared and no
policy is executed: every figure is recomputed from the per-participant
accounting the Phase 6 runner already settled.

The definitions come from ``participant_payoff.py``. An independent storage
unit carries no load and no on-site generation, so ``IndependentStoragePayoff``
settles exactly three terms:

    revenue       = sum_t lmp[node, t] * p_sell[t] * DT_HOURS
    purchase_cost = sum_t lmp[node, t] * p_buy[t]  * DT_HOURS
    cycle_cost    = rate * (charge_mwh + discharge_mwh)
    net           = revenue - purchase_cost - cycle_cost

``rate`` is ``config.storage.cycle_cost``, applied to charge plus discharge
energy; the term is linear in that rate. Re-estimating at another rate on the
same cleared trajectory is therefore exact arithmetic, not an assumption, and
the script checks the linearity against the stored cycle cost rather than
trusting the reading of the source.

Outputs land in ``reports/phase7a_economic_sensitivity_2026-11-12/``.
"""

from pathlib import Path
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "phase7a_economic_sensitivity_2026-11-12"
PHASE6 = ROOT / "reports" / "phase6_frozen_policy_2026-11-12"
PHASE6D = ROOT / "reports" / "phase6d_profit_attribution_2026-11-12"
SPLIT_FILE = ROOT / "data" / "processed" / "v2_cy_jn_2025" / "common_split.json"
H2_FILE = (ROOT / "reports" / "v2_cy_jn_2025" / "formal_training_v3"
           / "formal_training_data_manifest.json")

LABEL = "NON_FORMAL_DIAGNOSTIC"
ESS_NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
SCOPES = ("fleet",) + ESS_NAMES
SPLIT_LABEL = {"A": "Probe", "B": "Train", "C": "Validation"}
RATES = (0.0, 50.0, 100.0, 150.0, 200.0)
BASE_RATE = 100.0
TOL = 1e-6
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261112
CST = timezone(timedelta(hours=8))

PHASE6_FILES = (
    "phase6A_smoke_results.csv",
    "phase6B_train_results.csv",
    "phase6C_validation_results.csv",
    "fleet_payoff.csv",
    "per_ess_payoff.csv",
    "daily_paired_deltas.csv",
    "bootstrap_summary.json",
    "phase6_config.json",
    "phase6_preflight.json",
    "source_hashes.json",
    "output_hashes.json",
    "phase6_decision_report.md",
)
PHASE6D_FILES = (
    "accounting_reconciliation.csv",
    "return_distribution_summary.csv",
    "concentration_sensitivity.csv",
    "bootstrap_reproduction.json",
    "bootstrap_reproduction.csv",
    "source_integrity_report.json",
    "per_ess_decomposition.csv",
    "robustness_summary.csv",
    "phase6d_decision_report.md",
)
SOURCE_FILES = (
    "src/participant_payoff.py",
    "src/money.py",
    "src/ess.py",
    "src/run_phase6_frozen_policy.py",
    "src/v2_cy_jn_env.py",
)


def now() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def csv_rows(path: Path) -> int:
    with open(path, "rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def write_csv(path: Path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def num(value: float) -> float:
    """Full precision; csv writes the shortest round-trip repr of the float."""
    return float(value)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def fmt_ci(low: float, high: float, digits: int = 2) -> str:
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"


def md_table(headers, rows) -> list:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def stats(deltas) -> dict:
    """Date-clustered bootstrap over the paired deltas, one row per date."""
    arr = np.asarray(deltas, dtype=float)
    n = int(arr.size)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    means = arr[idx].mean(axis=1)
    return {
        "n_dates": n,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "positive_share": float((arr > 0).mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


# --------------------------------------------------------------------------
# Phase 0: inputs
# --------------------------------------------------------------------------

def checkpoint_dirs(config: dict) -> dict:
    """Every checkpoint directory the two maps were replayed from."""
    out = {}
    for name, meta in config["checkpoints"].items():
        out[name] = [
            ROOT / meta["original_dir"],
            PHASE6 / meta["phase6_copy_dir"],
        ]
    return out


def checkpoint_files(dirs: dict) -> list:
    files = []
    for name, paths in dirs.items():
        for base in paths:
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file() and (path.parent.name in ("actors", "buffer")
                                       or path.name.endswith(".pt")
                                       or path.name.endswith(".npz")):
                    files.append((name, path))
    return files


def phase0() -> tuple:
    """Hash every input and refuse to continue on a missing or inconsistent one."""
    problems = []
    inputs = []

    def record(path: Path, kind: str) -> bool:
        if not path.is_file():
            problems.append(f"missing {kind}: {path.relative_to(ROOT)}")
            return False
        inputs.append({
            "kind": kind,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": csv_rows(path) if path.suffix == ".csv" else "",
        })
        return True

    for name in PHASE6_FILES:
        record(PHASE6 / name, "phase6")
    for name in PHASE6D_FILES:
        record(PHASE6D / name, "phase6d")
    for name in SOURCE_FILES:
        record(ROOT / name, "source")
    record(SPLIT_FILE, "data")
    record(H2_FILE, "data")

    config = json.loads((PHASE6 / "phase6_config.json").read_text("utf-8"))
    dirs = checkpoint_dirs(config)
    cp_files = checkpoint_files(dirs)
    cp_hashes = {f"{name}:{path.relative_to(ROOT)}": sha256(path)
                 for name, path in cp_files}

    checks = {}

    # Row counts and pairing.
    ess_rows = read_rows(PHASE6 / "per_ess_payoff.csv")
    fleet_rows = read_rows(PHASE6 / "fleet_payoff.csv")
    checks["ess_row_count"] = (len(ess_rows) == 1280, str(len(ess_rows)))
    checks["fleet_row_count"] = (len(fleet_rows) == 320, str(len(fleet_rows)))

    seen = set()
    duplicated = set()
    for row in ess_rows:
        key = (row["phase"], row["checkpoint"], row["pair_id"], row["z_core"],
               row["ess"])
        if key in seen:
            duplicated.add(key)
        seen.add(key)
    checks["no_duplicate_ess_rows"] = (not duplicated, str(len(duplicated)))

    per_arm = {}
    for row in ess_rows:
        key = (row["phase"], row["checkpoint"], row["pair_id"], row["z_core"])
        per_arm.setdefault(key, set()).add(row["ess"])
    bad_arm = {k: sorted(v) for k, v in per_arm.items()
               if set(v) != set(ESS_NAMES)}
    checks["ess_complete_per_arm"] = (not bad_arm, str(len(bad_arm)))

    per_pair = {}
    for (phase, cp, pair, z) in per_arm:
        per_pair.setdefault((phase, cp, pair), set()).add(z)
    bad_pair = {k: sorted(v) for k, v in per_pair.items()
                if set(v) != {"2.5", "10.0"}}
    checks["both_maps_per_pair"] = (not bad_pair, str(len(bad_pair)))

    bad_date = [row["pair_id"] for row in ess_rows
                if row["pair_id"] != f"{row['load_date']}|{row['price_date']}"]
    checks["pair_id_matches_dates"] = (not bad_date, str(len(bad_date)))

    split = json.loads(SPLIT_FILE.read_text("utf-8"))
    train = set(split["common"]["train"])
    valid = set(split["common"]["validation"])
    test = set(split["common"]["test"])
    holiday = set(split["common"]["holiday_block"])
    b_dates = {r["pair_id"] for r in ess_rows if r["phase"] == "B"}
    c_dates = {r["pair_id"] for r in ess_rows if r["phase"] == "C"}
    checks["B_is_train_split"] = (b_dates <= train, f"{len(b_dates)} dates")
    checks["C_is_validation_split"] = (c_dates <= valid, f"{len(c_dates)} dates")
    leaks = (b_dates | c_dates) & (test | holiday)
    checks["no_test_or_holiday_dates"] = (not leaks, str(len(leaks)))
    h2 = (json.loads(H2_FILE.read_text("utf-8"))
          if H2_FILE.is_file() else {})
    checks["h2_holdout_sealed"] = (h2.get("h2_opened") is False,
                                   f"h2_opened={h2.get('h2_opened')}")

    # The stored cycle cost must be the source formula at the base rate.
    implied, worst_implied = [], 0.0
    for row in ess_rows:
        energy = float(row["charge_mwh"]) + float(row["discharge_mwh"])
        if energy > 1e-9:
            rate = float(row["cycle_cost_cny"]) / energy
            implied.append(rate)
            worst_implied = max(worst_implied, abs(rate - BASE_RATE))
    checks["cycle_cost_is_rate_times_energy"] = (
        bool(implied) and worst_implied <= TOL,
        f"{len(implied)} rows, max |implied rate - 100| = {worst_implied:.3e}")

    # Checkpoint actor files must match the hashes Phase 6 recorded.
    mismatched = []
    for name, meta in config["checkpoints"].items():
        for actor_dir in (ROOT / meta["original_dir"] / "actors",
                          PHASE6 / meta["phase6_copy_dir"] / "actors"):
            for ess, expected in meta["actor_sha256"].items():
                path = actor_dir / f"{ess}.pt"
                if not path.is_file():
                    mismatched.append(f"{path.relative_to(ROOT)} missing")
                elif sha256(path) != expected:
                    mismatched.append(str(path.relative_to(ROOT)))
    checks["checkpoint_actor_hashes"] = (not mismatched, str(len(mismatched)))

    status = "PASS" if not problems and all(ok for ok, _ in checks.values()) \
        else "BLOCKED"
    manifest = {
        "generated_at": now(),
        "label": LABEL,
        "phase": "7A",
        "analysis": "post-hoc accounting re-estimation on frozen cleared "
                    "trajectories; no re-clearing, no training",
        "inputs": inputs,
        "checkpoint_files": cp_hashes,
        "formulas": {
            "market_revenue": "sum_t lmp[node,t] * p_sell[t] * DT_HOURS",
            "purchase_cost": "sum_t lmp[node,t] * p_buy[t] * DT_HOURS",
            "cycle_cost": "rate * (charge_mwh + discharge_mwh), "
                          "rate = config.storage.cycle_cost, linear in rate",
            "net": "revenue - purchase_cost - cycle_cost; generation, other "
                   "and penalty terms are structurally zero for independent "
                   "storage (participant_payoff.IndependentStoragePayoff)",
            "fleet": "sum of the four ESS units",
            "source": ["src/participant_payoff.py::_LedgerPayoff._ledger",
                       "src/participant_payoff.py::IndependentStoragePayoff",
                       "src/money.py::total_money, DT_HOURS"],
        },
        "splits": SPLIT_LABEL,
        "base_rate_cny_per_mwh": BASE_RATE,
        "rates_cny_per_mwh": list(RATES),
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "cluster": "pair_id (date)", "interval": "percentile"},
        "checks": {key: {"status": "PASS" if ok else "FAIL", "detail": detail}
                   for key, (ok, detail) in checks.items()},
        "problems": problems,
        "status": status,
    }
    (OUT / "input_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest, cp_hashes


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_accounts() -> tuple:
    """Per-ESS and fleet settled components, keyed by arm."""
    arms, fleet, meta = {}, {}, {}
    for row in read_rows(PHASE6 / "per_ess_payoff.csv"):
        key = (row["phase"], row["checkpoint"], row["pair_id"],
               float(row["z_core"]))
        arms.setdefault(key, {})[row["ess"]] = {
            "revenue": float(row["market_revenue_cny"]),
            "purchase": float(row["purchase_cost_cny"]),
            "cycle": float(row["cycle_cost_cny"]),
            "energy": float(row["charge_mwh"]) + float(row["discharge_mwh"]),
            "charge": float(row["charge_mwh"]),
            "discharge": float(row["discharge_mwh"]),
            "sold": float(row["energy_sold_mwh"]),
            "bought": float(row["energy_bought_mwh"]),
            "net": float(row["net_income_cny"]),
        }
        meta.setdefault(key[:3], {
            "seed": int(row["seed"]),
            "round": int(row["round"]),
            "role": row["role"],
            "load_date": row["load_date"],
            "price_date": row["price_date"],
            "season": row["season"],
        })
    for row in read_rows(PHASE6 / "fleet_payoff.csv"):
        key = (row["phase"], row["checkpoint"], row["pair_id"],
               float(row["z_core"]))
        fleet[key] = {
            "revenue": float(row["market_revenue_cny"]),
            "purchase": float(row["purchase_cost_cny"]),
            "cycle": float(row["cycle_cost_cny"]),
            "energy": float(row["charge_mwh"]) + float(row["discharge_mwh"]),
            "charge": float(row["charge_mwh"]),
            "discharge": float(row["discharge_mwh"]),
            "sold": float(row["energy_sold_mwh"]),
            "bought": float(row["energy_bought_mwh"]),
            "net": float(row["net_income_cny"]),
        }
    return arms, fleet, meta


def components(arms, fleet, phase, checkpoint, pair, z, scope) -> dict:
    if scope == "fleet":
        return fleet[(phase, checkpoint, pair, z)]
    return arms[(phase, checkpoint, pair, z)][scope]


def groups(arms) -> list:
    """(phase, checkpoint) combos present in the artifacts, sorted."""
    return sorted({(phase, cp) for (phase, cp, _, _) in arms})


def dates_of(arms, phase, checkpoint) -> list:
    """Sorted by pair_id, which is the order the Phase 6 bootstrap clustered."""
    return sorted({pair for (ph, cp, pair, _) in arms
                   if ph == phase and cp == checkpoint})


# --------------------------------------------------------------------------
# Phase 1: full P&L rebuild
# --------------------------------------------------------------------------

ESS_PNL_FIELDS = [
    "split", "phase", "checkpoint", "seed", "round", "role", "mapping",
    "pair_id", "load_date", "price_date", "season", "ess",
    "market_revenue_cny", "purchase_cost_cny", "cycle_cost_cny", "net_cny",
    "net_rebuilt_cny", "identity_residual_cny", "cycle_energy_mwh",
    "charge_mwh", "discharge_mwh", "energy_sold_mwh", "energy_bought_mwh",
    "implied_cycle_rate_cny_per_mwh",
]
FLEET_PNL_FIELDS = [
    "split", "phase", "checkpoint", "seed", "round", "role", "mapping",
    "pair_id", "load_date", "price_date", "season",
    "market_revenue_cny", "purchase_cost_cny", "cycle_cost_cny", "net_cny",
    "net_rebuilt_cny", "identity_residual_cny", "cycle_energy_mwh",
    "charge_mwh", "discharge_mwh", "energy_sold_mwh", "energy_bought_mwh",
    "implied_cycle_rate_cny_per_mwh",
]
DELTA_FIELDS = [
    "split", "phase", "checkpoint", "seed", "round", "role", "pair_id",
    "load_date", "price_date", "season", "scope",
    "control_net_cny", "candidate_net_cny", "delta_net_cny",
    "control_revenue_cny", "candidate_revenue_cny", "delta_revenue_cny",
    "control_purchase_cny", "candidate_purchase_cny", "delta_purchase_cny",
    "control_cycle_cny", "candidate_cycle_cny", "delta_cycle_cny",
    "control_energy_mwh", "candidate_energy_mwh", "delta_energy_mwh",
    "component_residual_cny",
]


def phase1(arms, fleet, meta) -> dict:
    ess_rows, fleet_rows = [], []
    max_identity = 0.0
    max_rate = 0.0

    for (phase, cp, pair, z), per_ess in sorted(arms.items()):
        info = meta[(phase, cp, pair)]
        for ess in ESS_NAMES:
            row = per_ess[ess]
            rebuilt = row["revenue"] - row["purchase"] - row["cycle"]
            residual = row["net"] - rebuilt
            max_identity = max(max_identity, abs(residual))
            if row["energy"] > 1e-9:
                max_rate = max(max_rate,
                               abs(row["cycle"] / row["energy"] - BASE_RATE))
            ess_rows.append({
                "split": SPLIT_LABEL[phase], "phase": phase, "checkpoint": cp,
                "seed": info["seed"], "round": info["round"],
                "role": info["role"], "mapping": z, "pair_id": pair,
                "load_date": info["load_date"], "price_date": info["price_date"],
                "season": info["season"], "ess": ess,
                "market_revenue_cny": num(row["revenue"]),
                "purchase_cost_cny": num(row["purchase"]),
                "cycle_cost_cny": num(row["cycle"]),
                "net_cny": num(row["net"]),
                "net_rebuilt_cny": num(rebuilt),
                "identity_residual_cny": num(residual),
                "cycle_energy_mwh": num(row["energy"]),
                "charge_mwh": num(row["charge"]),
                "discharge_mwh": num(row["discharge"]),
                "energy_sold_mwh": num(row["sold"]),
                "energy_bought_mwh": num(row["bought"]),
                "implied_cycle_rate_cny_per_mwh":
                    num(row["cycle"] / row["energy"]) if row["energy"] > 1e-9
                    else "",
            })

        f = fleet[(phase, cp, pair, z)]
        rebuilt = f["revenue"] - f["purchase"] - f["cycle"]
        residual = f["net"] - rebuilt
        max_identity = max(max_identity, abs(residual))
        fleet_rows.append({
            "split": SPLIT_LABEL[phase], "phase": phase, "checkpoint": cp,
            "seed": info["seed"], "round": info["round"], "role": info["role"],
            "mapping": z, "pair_id": pair, "load_date": info["load_date"],
            "price_date": info["price_date"], "season": info["season"],
            "market_revenue_cny": num(f["revenue"]),
            "purchase_cost_cny": num(f["purchase"]),
            "cycle_cost_cny": num(f["cycle"]), "net_cny": num(f["net"]),
            "net_rebuilt_cny": num(rebuilt),
            "identity_residual_cny": num(residual),
            "cycle_energy_mwh": num(f["energy"]),
            "charge_mwh": num(f["charge"]), "discharge_mwh": num(f["discharge"]),
            "energy_sold_mwh": num(f["sold"]),
            "energy_bought_mwh": num(f["bought"]),
            "implied_cycle_rate_cny_per_mwh":
                num(f["cycle"] / f["energy"]) if f["energy"] > 1e-9 else "",
        })

    # Fleet must be the sum of the four ESS units, term by term.
    max_fleet_sum = 0.0
    for (phase, cp, pair, z), per_ess in arms.items():
        f = fleet[(phase, cp, pair, z)]
        for field in ("revenue", "purchase", "cycle", "net", "energy"):
            total = sum(per_ess[ess][field] for ess in ESS_NAMES)
            max_fleet_sum = max(max_fleet_sum, abs(total - f[field]))

    # Paired deltas and the component identity.
    delta_rows = []
    max_component = 0.0
    for phase, cp in groups(arms):
        info = meta[(phase, cp, dates_of(arms, phase, cp)[0])]
        for pair in dates_of(arms, phase, cp):
            for scope in SCOPES:
                c = components(arms, fleet, phase, cp, pair, 2.5, scope)
                x = components(arms, fleet, phase, cp, pair, 10.0, scope)
                d_rev = x["revenue"] - c["revenue"]
                d_pur = x["purchase"] - c["purchase"]
                d_cyc = x["cycle"] - c["cycle"]
                d_net = x["net"] - c["net"]
                residual = d_net - (d_rev - d_pur - d_cyc)
                max_component = max(max_component, abs(residual))
                delta_rows.append({
                    "split": SPLIT_LABEL[phase], "phase": phase,
                    "checkpoint": cp, "seed": info["seed"],
                    "round": info["round"], "role": info["role"],
                    "pair_id": pair, "load_date": info["load_date"],
                    "price_date": info["price_date"],
                    "season": info["season"], "scope": scope,
                    "control_net_cny": num(c["net"]),
                    "candidate_net_cny": num(x["net"]),
                    "delta_net_cny": num(d_net),
                    "control_revenue_cny": num(c["revenue"]),
                    "candidate_revenue_cny": num(x["revenue"]),
                    "delta_revenue_cny": num(d_rev),
                    "control_purchase_cny": num(c["purchase"]),
                    "candidate_purchase_cny": num(x["purchase"]),
                    "delta_purchase_cny": num(d_pur),
                    "control_cycle_cny": num(c["cycle"]),
                    "candidate_cycle_cny": num(x["cycle"]),
                    "delta_cycle_cny": num(d_cyc),
                    "control_energy_mwh": num(c["energy"]),
                    "candidate_energy_mwh": num(x["energy"]),
                    "delta_energy_mwh": num(x["energy"] - c["energy"]),
                    "component_residual_cny": num(residual),
                })

    # Agreement with the Phase 6D reconciliation of the same arms.
    p6d = read_rows(PHASE6D / "accounting_reconciliation.csv")
    max_p6d = 0.0
    compared = 0
    for row in p6d:
        if row["row_type"] != "arm":
            continue
        key = (row["phase"], row["checkpoint"], row["pair_id"],
               float(row["z_core"]))
        scope = row["scope"]
        ours = components(arms, fleet, key[0], key[1], key[2], key[3], scope)
        for field, name in (("revenue", "market_revenue_cny"),
                            ("purchase", "purchase_cost_cny"),
                            ("cycle", "cycle_cost_cny"),
                            ("net", "net_income_cny")):
            max_p6d = max(max_p6d, abs(ours[field] - float(row[name])))
            compared += 1

    write_csv(OUT / "ess_day_pnl.csv", ESS_PNL_FIELDS, ess_rows)
    write_csv(OUT / "fleet_day_pnl.csv", FLEET_PNL_FIELDS, fleet_rows)
    write_csv(OUT / "paired_mapping_delta.csv", DELTA_FIELDS, delta_rows)

    result = {
        "generated_at": now(),
        "label": LABEL,
        "definition": "G = Revenue - Purchase - Cycle",
        "rows": {"ess_day_pnl": len(ess_rows),
                 "fleet_day_pnl": len(fleet_rows),
                 "paired_mapping_delta": len(delta_rows)},
        "checks": {
            "net_identity": {
                "status": "PASS" if max_identity <= TOL else "FAIL",
                "max_abs_residual_cny": num(max_identity),
                "tolerance_cny": TOL,
                "scope": "every ESS row and every fleet row",
            },
            "fleet_equals_sum_ess": {
                "status": "PASS" if max_fleet_sum <= TOL else "FAIL",
                "max_abs_residual_cny": num(max_fleet_sum),
                "fields": ["revenue", "purchase", "cycle", "net", "energy"],
            },
            "paired_delta_identity": {
                "status": "PASS" if max_component <= TOL else "FAIL",
                "max_abs_residual_cny": num(max_component),
                "scope": "delta_net = delta_revenue - delta_purchase - "
                         "delta_cycle on every pair and scope",
            },
            "phase6d_net_agreement": {
                "status": "PASS" if max_p6d <= TOL else "FAIL",
                "max_abs_residual_cny": num(max_p6d),
                "cells_compared": compared,
                "source": "phase6d/accounting_reconciliation.csv",
            },
            "cycle_cost_formula": {
                "status": "PASS" if max_rate <= TOL else "FAIL",
                "max_abs_deviation_from_100_cny_per_mwh": num(max_rate),
                "detail": "stored cycle cost equals 100 x (charge + discharge) "
                          "on every row with nonzero throughput",
            },
        },
    }
    result["status"] = ("PASS" if all(result["checks"][key]["status"] == "PASS"
                                      for key in result["checks"])
                        else "FAIL")
    (OUT / "pnl_reconciliation.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# Phase 2: Cycle Cost sensitivity on the held-fixed trajectories
# --------------------------------------------------------------------------

SENS_FIELDS = [
    "rate_cny_per_mwh", "primary", "split", "phase", "checkpoint", "seed",
    "round", "role", "scope", "n_dates",
    "control_mean_net_cny", "candidate_mean_net_cny", "delta_mean_net_cny",
    "delta_median_net_cny", "delta_positive_share",
    "ci95_low_cny", "ci95_high_cny",
    "control_mean_cycle_cny", "candidate_mean_cycle_cny",
    "delta_mean_cycle_cny", "delta_mean_energy_mwh",
]


def net_at(components_row: dict, rate: float) -> float:
    """The settled ledger at another cycle-cost rate, same trajectory."""
    return (components_row["revenue"] - components_row["purchase"]
            - rate * components_row["energy"])


def paired_series(arms, fleet, phase, checkpoint, scope) -> dict:
    dates = dates_of(arms, phase, checkpoint)
    ctrl = [components(arms, fleet, phase, checkpoint, d, 2.5, scope)
            for d in dates]
    cand = [components(arms, fleet, phase, checkpoint, d, 10.0, scope)
            for d in dates]
    return {"dates": dates, "control": ctrl, "candidate": cand}


def phase2(arms, fleet, meta) -> tuple:
    rows, summary_groups = [], []
    for phase, cp in groups(arms):
        info = meta[(phase, cp, dates_of(arms, phase, cp)[0])]
        for scope in SCOPES:
            series = paired_series(arms, fleet, phase, cp, scope)
            control_net = {rate: [net_at(c, rate) for c in series["control"]]
                           for rate in RATES}
            candidate_net = {rate: [net_at(c, rate) for c in series["candidate"]]
                             for rate in RATES}
            block = {
                "phase": phase, "split": SPLIT_LABEL[phase],
                "checkpoint": cp, "seed": info["seed"],
                "role": info["role"], "scope": scope,
                "primary": info["role"] == "primary", "by_rate": {},
            }
            for rate in RATES:
                deltas = [x - c for x, c in zip(candidate_net[rate],
                                                control_net[rate])]
                st = stats(deltas)
                c_cycle = float(np.mean([c["cycle"] for c in series["control"]]))
                x_cycle = float(np.mean([c["cycle"]
                                         for c in series["candidate"]]))
                d_energy = float(np.mean([x["energy"] - c["energy"]
                                          for x, c in zip(series["candidate"],
                                                          series["control"])]))
                rows.append({
                    "rate_cny_per_mwh": f"{rate:.1f}",
                    "primary": block["primary"], "split": block["split"],
                    "phase": phase, "checkpoint": cp, "seed": info["seed"],
                    "round": info["round"], "role": info["role"],
                    "scope": scope, "n_dates": st["n_dates"],
                    "control_mean_net_cny": num(float(np.mean(control_net[rate]))),
                    "candidate_mean_net_cny":
                        num(float(np.mean(candidate_net[rate]))),
                    "delta_mean_net_cny": num(st["mean"]),
                    "delta_median_net_cny": num(st["median"]),
                    "delta_positive_share": num(st["positive_share"]),
                    "ci95_low_cny": num(st["ci95_low"]),
                    "ci95_high_cny": num(st["ci95_high"]),
                    "control_mean_cycle_cny": num(c_cycle),
                    "candidate_mean_cycle_cny": num(x_cycle),
                    "delta_mean_cycle_cny": num(x_cycle - c_cycle),
                    "delta_mean_energy_mwh": num(d_energy),
                })
                block["by_rate"][f"{rate:.1f}"] = {
                    "control_mean_net_cny":
                        num(float(np.mean(control_net[rate]))),
                    "candidate_mean_net_cny":
                        num(float(np.mean(candidate_net[rate]))),
                    "delta_mean_net_cny": num(st["mean"]),
                    "delta_median_net_cny": num(st["median"]),
                    "delta_positive_share": num(st["positive_share"]),
                    "ci95_low_cny": num(st["ci95_low"]),
                    "ci95_high_cny": num(st["ci95_high"]),
                    "delta_mean_cycle_cny": num(x_cycle - c_cycle),
                    "delta_mean_energy_mwh": num(d_energy),
                }
            summary_groups.append(block)

    write_csv(OUT / "cycle_cost_sensitivity.csv", SENS_FIELDS, rows)

    # Linearity evidence: moving the rate shifts the paired delta by exactly
    # (base - rate) x the change in throughput, per the source formula.
    max_linear = 0.0
    for block in summary_groups:
        base = block["by_rate"][f"{BASE_RATE:.1f}"]
        for rate in RATES:
            item = block["by_rate"][f"{rate:.1f}"]
            predicted = base["delta_mean_net_cny"] + \
                (BASE_RATE - rate) * base["delta_mean_energy_mwh"]
            max_linear = max(max_linear, abs(predicted - item["delta_mean_net_cny"]))

    # Reproduce the Phase 6 bootstrap at the base rate.
    original = json.loads((PHASE6 / "bootstrap_summary.json").read_text("utf-8"))
    by_key = {(block["phase"], block["checkpoint"], block["scope"]): block
              for block in summary_groups}
    reproduce, max_repro = [], 0.0
    for item in original["results"]:
        block = by_key[(item["phase"], item["checkpoint"], item["scope"])]
        ours = block["by_rate"][f"{BASE_RATE:.1f}"]
        diffs = [abs(ours["delta_mean_net_cny"] - item["mean_delta_cny"]),
                 abs(ours["delta_median_net_cny"] - item["median_delta_cny"]),
                 abs(ours["delta_positive_share"] - item["fraction_positive"]),
                 abs(ours["ci95_low_cny"] - item["ci95_low"]),
                 abs(ours["ci95_high_cny"] - item["ci95_high"]),
                 abs(ours["control_mean_net_cny"] - item["control_mean_cny"]),
                 abs(ours["candidate_mean_net_cny"] - item["candidate_mean_cny"])]
        worst = max(diffs)
        max_repro = max(max_repro, worst)
        reproduce.append({
            "phase": item["phase"], "checkpoint": item["checkpoint"],
            "scope": item["scope"], "n_dates": item["n_dates"],
            "original_mean": item["mean_delta_cny"],
            "reproduced_mean": ours["delta_mean_net_cny"],
            "original_ci95_low": item["ci95_low"],
            "reproduced_ci95_low": ours["ci95_low_cny"],
            "original_ci95_high": item["ci95_high"],
            "reproduced_ci95_high": ours["ci95_high_cny"],
            "max_abs_diff": num(worst),
            "match": worst <= 1e-9,
        })
    write_csv(OUT / "bootstrap_reproduction.csv",
              ["phase", "checkpoint", "scope", "n_dates", "original_mean",
               "reproduced_mean", "original_ci95_low", "reproduced_ci95_low",
               "original_ci95_high", "reproduced_ci95_high", "max_abs_diff",
               "match"], reproduce)

    summary = {
        "generated_at": now(),
        "label": LABEL,
        "statement": "Cycle Cost was varied on the cleared trajectories "
                     "Phase 6 produced. Charge, discharge, prices and "
                     "schedules are held fixed, so these are accounting "
                     "re-estimations of a frozen policy, not the performance "
                     "a policy would obtain if the market were cleared again "
                     "under another cycle cost. Re-clearing would change the "
                     "dispatch, the nodal prices and the settlement.",
        "held_fixed": ["cleared schedules", "nodal prices", "charge_mwh",
                       "discharge_mwh", "gate decisions"],
        "re_estimated": "cycle cost = rate x (charge_mwh + discharge_mwh)",
        "cost_function": "linear in the rate; confirmed against the stored "
                         "cycle cost, which equals 100 x throughput on every "
                         "nonzero row",
        "rates_cny_per_mwh": list(RATES),
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "cluster": "pair_id (date)", "interval": "percentile"},
        "linearity_check": {
            "status": "PASS" if max_linear <= TOL else "FAIL",
            "max_abs_residual_cny": num(max_linear),
            "identity": "delta(r) = delta(100) + (100 - r) * delta_energy",
        },
        "base_rate_reproduction": {
            "status": "PASS" if max_repro <= 1e-9 else "FAIL",
            "groups": len(reproduce),
            "max_abs_diff": num(max_repro),
            "source": "phase6/bootstrap_summary.json",
        },
        "groups": summary_groups,
    }
    (OUT / "cycle_cost_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary, rows


# --------------------------------------------------------------------------
# Phase 3: concentration recheck
# --------------------------------------------------------------------------

CONC_FIELDS = [
    "rate_cny_per_mwh", "primary", "split", "phase", "checkpoint", "seed",
    "scope", "scenario", "n_dates", "n_remaining", "mean_remaining_cny",
    "median_remaining_cny", "removed_dates",
]
CONC_SUMMARY_FIELDS = [
    "rate_cny_per_mwh", "primary", "split", "phase", "checkpoint", "seed",
    "scope", "n_dates", "s_plus_cny", "s_minus_cny", "s_net_cny",
    "cr5_plus", "cr5_minus", "max_pos_date", "max_pos_cny", "max_neg_date",
    "max_neg_cny", "n_positive", "n_negative", "n_zero", "n_abs_lt_0p01",
    "n_abs_lt_0p1", "frac_abs_lt_0p01",
]


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else ""


def phase3(arms, fleet, meta) -> tuple:
    rows, summary_rows = [], []
    for phase, cp in groups(arms):
        info = meta[(phase, cp, dates_of(arms, phase, cp)[0])]
        for scope in SCOPES:
            series = paired_series(arms, fleet, phase, cp, scope)
            for rate in RATES:
                deltas = [
                    net_at(x, rate) - net_at(c, rate)
                    for x, c in zip(series["candidate"], series["control"])]
                labels = list(zip(series["dates"], deltas))
                positive = sorted([item for item in labels if item[1] > 0],
                                  key=lambda item: -item[1])
                negative = sorted([item for item in labels if item[1] < 0],
                                  key=lambda item: item[1])
                s_plus = float(sum(v for _, v in positive))
                s_minus = float(sum(v for _, v in negative))
                arr = np.asarray(deltas, dtype=float)

                for scenario, count in (("full", 0), ("drop_top1_pos", 1),
                                        ("drop_top3_pos", 3),
                                        ("drop_top5_pos", 5)):
                    dropped = positive[:count]
                    removed = {date for date, _ in dropped}
                    kept = [v for date, v in labels if date not in removed]
                    kept_arr = np.asarray(kept, dtype=float)
                    rows.append({
                        "rate_cny_per_mwh": f"{rate:.1f}",
                        "primary": info["role"] == "primary",
                        "split": SPLIT_LABEL[phase], "phase": phase,
                        "checkpoint": cp, "seed": info["seed"],
                        "scope": scope, "scenario": scenario,
                        "n_dates": len(labels), "n_remaining": len(kept),
                        "mean_remaining_cny":
                            num(float(kept_arr.mean())) if kept else "",
                        "median_remaining_cny":
                            num(float(np.median(kept_arr))) if kept else "",
                        "removed_dates": ";".join(date for date, _ in dropped),
                    })

                summary_rows.append({
                    "rate_cny_per_mwh": f"{rate:.1f}",
                    "primary": info["role"] == "primary",
                    "split": SPLIT_LABEL[phase], "phase": phase,
                    "checkpoint": cp, "seed": info["seed"], "scope": scope,
                    "n_dates": len(labels), "s_plus_cny": num(s_plus),
                    "s_minus_cny": num(s_minus), "s_net_cny": num(s_plus + s_minus),
                    "cr5_plus": ratio(float(sum(v for _, v in positive[:5])),
                                      s_plus),
                    "cr5_minus": ratio(float(sum(v for _, v in negative[:5])),
                                       s_minus),
                    "max_pos_date": positive[0][0] if positive else "",
                    "max_pos_cny": num(positive[0][1]) if positive else "",
                    "max_neg_date": negative[0][0] if negative else "",
                    "max_neg_cny": num(negative[0][1]) if negative else "",
                    "n_positive": len(positive), "n_negative": len(negative),
                    "n_zero": int(np.sum(arr == 0)),
                    "n_abs_lt_0p01": int(np.sum(np.abs(arr) < 0.01)),
                    "n_abs_lt_0p1": int(np.sum(np.abs(arr) < 0.1)),
                    "frac_abs_lt_0p01": num(float(np.mean(np.abs(arr) < 0.01))),
                })

    write_csv(OUT / "concentration_sensitivity.csv", CONC_FIELDS, rows)
    write_csv(OUT / "concentration_summary.csv", CONC_SUMMARY_FIELDS,
              summary_rows)
    return rows, summary_rows


# --------------------------------------------------------------------------
# Phase 4: verification
# --------------------------------------------------------------------------

def unique_keys(path: Path, fields) -> tuple:
    rows = read_rows(path)
    seen, duplicates = set(), 0
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return len(rows), duplicates


def phase4(manifest, start_cp_hashes, phase1_result, phase2_summary) -> dict:
    checks = {}

    # 1. Input hashes still match what was locked.
    mismatched = []
    for item in manifest["inputs"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            mismatched.append(item["path"])
    checks["input_hashes_intact"] = {
        "status": "PASS" if not mismatched else "FAIL",
        "detail": f"{len(manifest['inputs'])} inputs, mismatches="
                  f"{len(mismatched)}",
    }

    # 2. No duplicate output rows.
    dups = {}
    for name, fields in (
            ("ess_day_pnl.csv", ("split", "checkpoint", "pair_id", "mapping",
                                 "ess")),
            ("fleet_day_pnl.csv", ("split", "checkpoint", "pair_id", "mapping")),
            ("paired_mapping_delta.csv", ("phase", "checkpoint", "pair_id",
                                          "scope")),
            ("cycle_cost_sensitivity.csv", ("rate_cny_per_mwh", "phase",
                                            "checkpoint", "scope")),
            ("concentration_sensitivity.csv", ("rate_cny_per_mwh", "phase",
                                               "checkpoint", "scope",
                                               "scenario")),
            ("concentration_summary.csv", ("rate_cny_per_mwh", "phase",
                                           "checkpoint", "scope"))):
        count, duplicate = unique_keys(OUT / name, fields)
        dups[name] = {"rows": count, "duplicates": duplicate}
    checks["outputs_unique"] = {
        "status": "PASS" if all(v["duplicates"] == 0 for v in dups.values())
                  else "FAIL",
        "detail": json.dumps(dups),
    }

    # 3. Pairing complete across the two maps.
    delta_rows = read_rows(OUT / "paired_mapping_delta.csv")
    per_key = {}
    for row in delta_rows:
        per_key.setdefault((row["phase"], row["checkpoint"], row["pair_id"]),
                           set()).add(row["scope"])
    incomplete = [k for k, v in per_key.items() if v != set(SCOPES)]
    checks["pairing_complete"] = {
        "status": "PASS" if not incomplete else "FAIL",
        "detail": f"{len(per_key)} checkpoint/date groups, incomplete="
                  f"{len(incomplete)}",
    }

    # 4-6. Accounting identities, re-read from the written CSVs.
    ess_rows = read_rows(OUT / "ess_day_pnl.csv")
    fleet_rows = read_rows(OUT / "fleet_day_pnl.csv")
    residual = max(abs(float(r["identity_residual_cny"])) for r in
                   ess_rows + fleet_rows)
    checks["pnl_identity"] = {
        "status": "PASS" if residual <= TOL else "FAIL",
        "detail": f"max |net - (revenue - purchase - cycle)| = {residual:.3e}",
    }
    sums = {}
    for row in ess_rows:
        key = (row["phase"], row["checkpoint"], row["pair_id"], row["mapping"])
        sums.setdefault(key, {f: 0.0 for f in ("market_revenue_cny",
                                               "purchase_cost_cny",
                                               "cycle_cost_cny", "net_cny")})
        for field in sums[key]:
            sums[key][field] += float(row[field])
    worst = 0.0
    for row in fleet_rows:
        key = (row["phase"], row["checkpoint"], row["pair_id"], row["mapping"])
        for field in sums[key]:
            worst = max(worst, abs(sums[key][field] - float(row[field])))
    checks["ess_fleet_identity"] = {
        "status": "PASS" if worst <= TOL else "FAIL",
        "detail": f"max |fleet - sum(ESS)| = {worst:.3e} over revenue, "
                  f"purchase, cycle and net",
    }
    coupled = max(abs(float(r["component_residual_cny"]))
                  for r in delta_rows)
    checks["paired_delta_identity"] = {
        "status": "PASS" if coupled <= TOL else "FAIL",
        "detail": f"max |delta_net - (delta_rev - delta_pur - delta_cycle)| "
                  f"= {coupled:.3e}",
    }

    # 7. Original net reproduced from the artifacts.
    checks["original_net_reproduced"] = {
        "status": phase1_result["checks"]["phase6d_net_agreement"]["status"],
        "detail": f"{phase1_result['checks']['phase6d_net_agreement']['cells_compared']} "
                  f"cells compared with the Phase 6D reconciliation, max "
                  f"|diff| = "
                  f"{phase1_result['checks']['phase6d_net_agreement']['max_abs_residual_cny']:.3e}",
    }

    # 8-9. Bootstrap reproduction and the base-rate agreement.
    checks["bootstrap_reproduced"] = {
        "status": phase2_summary["base_rate_reproduction"]["status"],
        "detail": f"{phase2_summary['base_rate_reproduction']['groups']} groups, "
                  f"max |diff| = "
                  f"{phase2_summary['base_rate_reproduction']['max_abs_diff']:.3e}",
    }
    checks["cycle_cost_100_matches_existing"] = {
        "status": phase2_summary["base_rate_reproduction"]["status"],
        "detail": "rate 100 rows equal the Phase 6 bootstrap summary on "
                  "control mean, candidate mean, delta mean, median, positive "
                  "share and both CI bounds",
    }
    checks["cycle_cost_linearity"] = {
        "status": phase2_summary["linearity_check"]["status"],
        "detail": f"max residual of delta(r) = delta(100) + (100 - r) x "
                  f"delta_energy: "
                  f"{phase2_summary['linearity_check']['max_abs_residual_cny']:.3e}",
    }

    # 10. No checkpoint, replay buffer or optimizer state changed.
    end_cp_hashes = {f"{name}:{path.relative_to(ROOT)}": sha256(path)
                     for name, path in checkpoint_files(
                         checkpoint_dirs(json.loads(
                             (PHASE6 / "phase6_config.json").read_text("utf-8"))))}
    changed = [k for k in start_cp_hashes
               if end_cp_hashes.get(k) != start_cp_hashes[k]]
    checks["no_checkpoint_or_replay_modification"] = {
        "status": "PASS" if not changed else "FAIL",
        "detail": f"{len(start_cp_hashes)} checkpoint/replay files hashed "
                  f"before and after the analysis, changed={len(changed)}",
    }

    # 11. Test, holiday and H2 data never read.
    split_ok = (manifest["checks"]["B_is_train_split"]["status"] == "PASS"
                and manifest["checks"]["C_is_validation_split"]["status"]
                == "PASS"
                and manifest["checks"]["no_test_or_holiday_dates"]["status"]
                == "PASS")
    checks["no_test_holiday_h2_read"] = {
        "status": "PASS" if split_ok else "FAIL",
        "detail": "every pair_id in the artifacts lies in the Train or "
                  "Validation split of "
                  "data/processed/v2_cy_jn_2025/common_split.json, and both "
                  "are disjoint from its test and holiday blocks; the H2 "
                  "price holdout stays sealed "
                  f"({manifest['checks']['h2_holdout_sealed']['detail']})",
    }

    status = "PASS" if all(v["status"] == "PASS" for v in checks.values()) \
        else "FAIL"
    result = {"generated_at": now(), "label": LABEL, "checks": checks,
              "status": status}
    (OUT / "verification_report.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# Phase 5: report
# --------------------------------------------------------------------------

def primary_groups(summary) -> list:
    """One block per primary checkpoint, fleet scope."""
    return [block for block in summary["groups"]
            if block["primary"] and block["scope"] == "fleet"]


def report(manifest, phase1_result, phase2_summary, phase2_rows,
           conc_rows, conc_summary, ver) -> str:
    groups_ = primary_groups(phase2_summary)

    lines = [
        "# Phase 7A - Economic decomposition and Cycle Cost sensitivity",
        "",
        f"`{LABEL}` | generated {now()} | frozen-policy post-hoc accounting",
        "",
        "## Scope",
        "",
        "This report decomposes the Phase 6 paired-clearing results under the "
        "control quote map (Z_CORE = 2.5) and the admitted candidate "
        "(Z_CORE = 10.0). Nothing is re-cleared and no policy is executed. "
        "Every number is recomputed from the per-participant ledgers the "
        "Phase 6 runner wrote, using the settlement definitions in "
        "`src/participant_payoff.py`.",
        "",
        "The actors were trained under `linear_core_v1`. Phase 6 and this "
        "report change only the inference-time quote map, on frozen weights, "
        "on a fixed set of days. No result here says how a policy trained "
        "under Z_CORE = 10.0 would perform, and no Cycle Cost sensitivity row "
        "describes a market that was cleared again under that cost.",
        "",
        "## Accounting definition",
        "",
        "An independent storage unit (the four ESS) carries no load and no "
        "on-site generation, so `IndependentStoragePayoff` settles three "
        "terms and nothing else:",
        "",
        "```",
        "revenue       = sum_t lmp[node, t] * p_sell[t] * DT_HOURS",
        "purchase_cost = sum_t lmp[node, t] * p_buy[t]  * DT_HOURS",
        "cycle_cost    = rate * (charge_mwh + discharge_mwh)",
        "net           = revenue - purchase_cost - cycle_cost",
        "```",
        "",
        f"`rate` is `config.storage.cycle_cost` = {BASE_RATE:.0f} CNY/MWh on "
        "the frozen artifacts, and the term is linear in it. That linearity "
        "is confirmed on the data rather than assumed: on every row with "
        "nonzero throughput the stored cycle cost equals "
        f"{BASE_RATE:.0f} x (charge + discharge) to within "
        f"{phase1_result['checks']['cycle_cost_formula']['max_abs_deviation_from_100_cny_per_mwh']:.1e} "
        "CNY/MWh.",
        "",
        "## Phase 1 - Full P&L rebuild",
        "",
    ]

    lines += md_table(
        ["check", "status", "evidence"],
        [[name, item["status"], ", ".join(
            f"{k}={v}" for k, v in item.items() if k != "status")]
         for name, item in phase1_result["checks"].items()])
    lines += [
        "",
        f"Rows written: `ess_day_pnl.csv` "
        f"{phase1_result['rows']['ess_day_pnl']}, "
        f"`fleet_day_pnl.csv` {phase1_result['rows']['fleet_day_pnl']}, "
        f"`paired_mapping_delta.csv` "
        f"{phase1_result['rows']['paired_mapping_delta']}.",
        "",
        "### Component deltas at the frozen rate (CNY/day, mean over dates)",
        "",
    ]
    delta_rows = read_rows(OUT / "paired_mapping_delta.csv")
    comp_rows = []
    for block in groups_:
        subset = [r for r in delta_rows
                  if r["phase"] == block["phase"]
                  and r["checkpoint"] == block["checkpoint"]
                  and r["scope"] == "fleet"]
        d_rev = float(np.mean([float(r["delta_revenue_cny"]) for r in subset]))
        d_pur = float(np.mean([float(r["delta_purchase_cny"]) for r in subset]))
        d_cyc = float(np.mean([float(r["delta_cycle_cny"]) for r in subset]))
        d_net = float(np.mean([float(r["delta_net_cny"]) for r in subset]))
        med = float(np.median([float(r["delta_net_cny"]) for r in subset]))
        comp_rows.append([block["split"], block["checkpoint"], len(subset),
                          fmt(d_rev), fmt(d_pur), fmt(d_cyc), fmt(d_net),
                          fmt(med, 3)])
    lines += md_table(
        ["split", "checkpoint", "dates", "dRevenue", "dPurchase", "dCycle",
         "dNet", "median dNet"], comp_rows)

    pooled = [row for row in delta_rows
              if row["split"] == "Train" and row["scope"] == "fleet"]
    if pooled:
        lines += [
            "",
            "Train pooled over both seeds: "
            + ", ".join(f"mean d{name} "
                        f"{fmt(float(np.mean([float(r[key]) for r in pooled])))}"
                        for name, key in (("Revenue", "delta_revenue_cny"),
                                          ("Purchase", "delta_purchase_cny"),
                                          ("Cycle", "delta_cycle_cny"),
                                          ("Net", "delta_net_cny")))
            + ".",
            "",
        ]

    lines += [
        "",
        "## Phase 2 - Cycle Cost sensitivity",
        "",
        "The cleared trajectories are held fixed - the same schedules, nodal "
        "prices, charge and discharge energy, and gate decisions - and only "
        "the cycle-cost rate is changed. A row here answers *what the same "
        "clearing would have paid under another cost assumption*. It does not "
        "answer what the policy would have priced or cleared under that cost, "
        "because a different rate changes the dispatch, the nodal prices and "
        "the settlement.",
        "",
        "### Fleet paired delta by rate (CNY/day)",
        "",
    ]
    sens_table = []
    for block in groups_:
        cells = [block["split"], block["checkpoint"]]
        for rate in RATES:
            item = block["by_rate"][f"{rate:.1f}"]
            cells.append(f"{fmt(item['delta_mean_net_cny'])} "
                         f"{fmt_ci(item['ci95_low_cny'], item['ci95_high_cny'])}")
        cells.append(fmt(block["by_rate"]["100.0"]["delta_mean_energy_mwh"], 4))
        sens_table.append(cells)
    lines += md_table(
        ["split", "checkpoint"] + [f"{r:.0f} CNY/MWh" for r in RATES]
        + ["dEnergy MWh"], sens_table)
    lines += [
        "",
        "Each cell is mean delta with its date-clustered 95% CI. dEnergy is "
        "the mean change in charge-plus-discharge energy per day; the whole "
        "rate dependence follows from it, since "
        "`delta(r) = delta(100) + (100 - r) * dEnergy`.",
        "",
        "### Per-ESS at the frozen rate and at zero cycle cost (CNY/day)",
        "",
    ]
    ess_table = []
    for block in groups_:
        for scope in ESS_NAMES:
            item = next(b for b in phase2_summary["groups"]
                        if b["phase"] == block["phase"]
                        and b["checkpoint"] == block["checkpoint"]
                        and b["scope"] == scope)
            ess_table.append([
                item["split"], item["checkpoint"], scope,
                fmt(item["by_rate"]["0.0"]["delta_mean_net_cny"]),
                fmt(item["by_rate"]["100.0"]["delta_mean_net_cny"]),
                fmt(item["by_rate"]["200.0"]["delta_mean_net_cny"]),
            ])
    lines += md_table(
        ["split", "checkpoint", "scope", "mean dNet at 0", "at 100",
         "at 200"], ess_table)

    lines += [
        "",
        "## Phase 3 - Concentration recheck",
        "",
        "Reported at the frozen rate for the fleet scope; the drop-Top-N rows "
        "are a sensitivity check and do not replace the pre-registered Phase 6 "
        "test.",
        "",
    ]
    conc_table = []
    for block in groups_:
        for scenario in ("full", "drop_top1_pos", "drop_top3_pos",
                         "drop_top5_pos"):
            row = next(r for r in conc_rows
                       if r["phase"] == block["phase"]
                       and r["checkpoint"] == block["checkpoint"]
                       and r["scope"] == "fleet"
                       and r["scenario"] == scenario
                       and float(r["rate_cny_per_mwh"]) == BASE_RATE)
            conc_table.append([block["split"], block["checkpoint"], scenario,
                               row["n_remaining"],
                               fmt(float(row["mean_remaining_cny"])),
                               fmt(float(row["median_remaining_cny"]), 3)])
    lines += md_table(
        ["split", "checkpoint", "scenario", "dates", "mean dNet",
         "median dNet"], conc_table)

    lines += ["", "### Concentration statistics at the frozen rate", ""]
    conc_stat = []
    for block in groups_:
        row = next(r for r in conc_summary
                   if r["phase"] == block["phase"]
                   and r["checkpoint"] == block["checkpoint"]
                   and r["scope"] == "fleet"
                   and float(r["rate_cny_per_mwh"]) == BASE_RATE)
        conc_stat.append([
            block["split"], block["checkpoint"], row["n_dates"],
            fmt(float(row["s_plus_cny"])), fmt(float(row["s_minus_cny"])),
            f"{float(row['cr5_plus']):.3f}" if row["cr5_plus"] != "" else "-",
            f"{float(row['cr5_minus']):.3f}" if row["cr5_minus"] != "" else "-",
            row["max_pos_date"], fmt(float(row["max_pos_cny"])),
            row["max_neg_date"], fmt(float(row["max_neg_cny"])),
            row["n_abs_lt_0p01"]])
    lines += md_table(
        ["split", "checkpoint", "dates", "S+", "S-", "CR5+", "CR5-",
         "max + date", "max +", "max - date", "max -", "|d|<0.01"], conc_stat)

    lines += ["", "## Verification", ""]
    lines += md_table(["check", "status", "detail"],
                      [[name, item["status"], item["detail"]]
                       for name, item in ver["checks"].items()])

    lines += ["", "## Answers", ""]
    lines += answers(phase2_summary, conc_rows, ver)

    lines += [
        "",
        "## Verdict",
        "",
        f"`PHASE7A_ANALYSIS = {ver['status']}` - every accounting identity, "
        "input hash, bootstrap reproduction and base-rate agreement listed "
        "above holds.",
        "",
        "`PHASE6_ORIGINAL_VERDICT = PARTIAL` (unchanged). Phase 7A explains "
        "what that result is made of; it does not upgrade it. The Phase 6 "
        "PASS condition was pre-registered on the re-cleared means and "
        "medians, not on a decomposition of them.",
        "",
        "## What this does not show",
        "",
        "- Not a re-cleared market. Every sensitivity row holds the Phase 6 "
        "dispatch and prices fixed.",
        "- Not a policy trained under Z_CORE = 10.0.",
        "- Not an independent-generalisation claim: the days are the same "
        "Train and Validation pairs Phase 6 used, the price series is the "
        "single 2025 source, and Test, Holiday and the H2 holdout were not "
        "read.",
        "- Block-level quote-to-clearing transmission remains NOT_TESTED: the "
        "Phase 6 runner did not persist the per-block quote, gate and price "
        "series, and re-running the clearing is outside this phase.",
        "",
    ]
    return "\n".join(lines)


def answers(phase2_summary, conc_rows, ver) -> list:
    groups_ = primary_groups(phase2_summary)
    out = []

    # Q1
    delta_rows = read_rows(OUT / "paired_mapping_delta.csv")
    out += ["### 1. Where does the Z=10 gain come from - Revenue, Purchase or Cycle?",
            ""]
    rows = []
    for block in groups_:
        subset = [r for r in delta_rows
                  if r["phase"] == block["phase"]
                  and r["checkpoint"] == block["checkpoint"]
                  and r["scope"] == "fleet"]
        d_rev = float(np.mean([float(r["delta_revenue_cny"]) for r in subset]))
        d_pur = float(np.mean([float(r["delta_purchase_cny"]) for r in subset]))
        d_cyc = float(np.mean([float(r["delta_cycle_cny"]) for r in subset]))
        d_net = float(np.mean([float(r["delta_net_cny"]) for r in subset]))
        rows.append([block["split"], block["checkpoint"], fmt(d_rev),
                     fmt(-d_pur), fmt(-d_cyc), fmt(d_net),
                     f"{100.0 * -d_cyc / d_net:.0f}%" if abs(d_net) > 1e-9
                     else "-"])
    out += md_table(["split", "checkpoint", "dRevenue", "-dPurchase",
                     "-dCycle", "dNet", "cycle term / dNet"], rows)
    out += [
        "",
        "The three columns are the signed contributions that add up to dNet: "
        "`dNet = dRevenue - dPurchase - dCycle`. A negative dPurchase entry "
        "means Z=10 buys less, so it contributes positively.",
        "",
    ]

    # Q2 and Q3
    out += ["### 2. At zero cycle cost, does Z=10 still have an advantage?", ""]
    rows = []
    for block in groups_:
        item = block["by_rate"]["0.0"]
        rows.append([block["split"], block["checkpoint"],
                     fmt(item["control_mean_net_cny"]),
                     fmt(item["candidate_mean_net_cny"]),
                     fmt(item["delta_mean_net_cny"]),
                     fmt(item["delta_median_net_cny"], 3),
                     f"{item['delta_positive_share']:.3f}",
                     fmt_ci(item["ci95_low_cny"], item["ci95_high_cny"])])
    out += md_table(["split", "checkpoint", "Z2.5 net", "Z10 net", "mean d",
                     "median d", "share > 0", "95% CI"], rows)
    eval_groups = [b for b in groups_ if b["split"] in ("Train", "Validation")]
    zero = [b["by_rate"]["0.0"] for b in eval_groups]
    all_negative = all(item["delta_mean_net_cny"] < 0 for item in zero)
    any_positive_ci = any(item["ci95_low_cny"] > 0 for item in zero)
    out += [
        "",
        "On Train and Validation the zero-cost fleet mean delta is "
        + ", ".join(f"{block['split']} {block['checkpoint']} "
                    f"{fmt(item['delta_mean_net_cny'])} "
                    f"{fmt_ci(item['ci95_low_cny'], item['ci95_high_cny'])}"
                    for block, item in zip(eval_groups, zero))
        + f" CNY/day. Negative in every group: {all_negative}. Any CI "
        f"excluding zero from above: {any_positive_ci}. "
        + ("So the gain at the frozen rate is a cost-avoidance effect, not an "
           "earnings improvement: the candidate keeps more because it cycles "
           "less, not because it sells more."
           if all_negative and not any_positive_ci
           else "The zero-cost result is not uniformly negative, so part of "
                "the gain is not attributable to the cycle cost alone."),
        "",
    ]

    out += ["### 3. How does the result move across 0/50/100/150/200 CNY/MWh?", ""]
    rows = []
    for block in groups_:
        cells = [block["split"], block["checkpoint"]]
        for rate in RATES:
            item = block["by_rate"][f"{rate:.1f}"]
            cells.append(fmt(item["delta_mean_net_cny"]))
        rows.append(cells)
    out += md_table(["split", "checkpoint"] + [f"{r:.0f}" for r in RATES], rows)
    first_positive = []
    for block in groups_:
        rate = next((r for r in RATES
                     if block["by_rate"][f"{r:.1f}"]["ci95_low_cny"] > 0), None)
        first_positive.append(
            f"{block['split']} {block['checkpoint']}: "
            + (f"CI excludes zero from {rate:.0f} CNY/MWh"
               if rate is not None else "no rate has a CI above zero"))
    out += [
        "",
        "The rate enters only through `delta(100) + (100 - r) * dEnergy`, so "
        "the whole shape is fixed by the mean change in throughput: a negative "
        "dEnergy means the candidate moves less charge-plus-discharge energy "
        "through the batteries, so it gains more the higher the rate. A "
        "positive dEnergy reverses that ordering.",
        "",
    ] + [f"- {line}" for line in first_positive] + [""]

    # Q4
    out += ["### 4. Do the two seeds agree on Validation?", ""]
    val = [b for b in groups_ if b["split"] == "Validation"]
    rows = []
    for block in val:
        item = block["by_rate"]["100.0"]
        rows.append([block["checkpoint"], fmt(item["delta_mean_net_cny"]),
                     fmt(item["delta_median_net_cny"], 3),
                     f"{item['delta_positive_share']:.3f}",
                     fmt_ci(item["ci95_low_cny"], item["ci95_high_cny"])])
    out += md_table(["checkpoint", "mean d", "median d", "share > 0",
                     "95% CI"], rows)
    if len(val) == 2:
        a = val[0]["by_rate"]["100.0"]
        b = val[1]["by_rate"]["100.0"]
        same_sign = (a["delta_mean_net_cny"] > 0) == (b["delta_mean_net_cny"] > 0)
        both_exclude_zero = a["ci95_low_cny"] > 0 and b["ci95_low_cny"] > 0
        out += ["", f"Same sign: {same_sign}. Both CIs exclude zero: "
                    f"{both_exclude_zero}."]
    out += [""]

    # Q5
    out += ["### 5. Is the gain still positive after dropping the top three "
            "positive dates?", ""]
    rows = []
    for block in groups_:
        cells = [block["split"], block["checkpoint"]]
        for scenario in ("full", "drop_top1_pos", "drop_top3_pos",
                         "drop_top5_pos"):
            row = next(r for r in conc_rows
                       if r["phase"] == block["phase"]
                       and r["checkpoint"] == block["checkpoint"]
                       and r["scope"] == "fleet"
                       and r["scenario"] == scenario
                       and float(r["rate_cny_per_mwh"]) == BASE_RATE)
            cells.append(fmt(float(row["mean_remaining_cny"])))
        rows.append(cells)
    out += md_table(["split", "checkpoint", "full", "drop 1", "drop 3",
                     "drop 5"], rows)
    dropped3, positive_after = [], []
    for block in groups_:
        row = next(r for r in conc_rows
                   if r["phase"] == block["phase"]
                   and r["checkpoint"] == block["checkpoint"]
                   and r["scope"] == "fleet"
                   and r["scenario"] == "drop_top3_pos"
                   and float(r["rate_cny_per_mwh"]) == BASE_RATE)
        value = float(row["mean_remaining_cny"])
        positive_after.append(value > 0)
        dropped3.append(f"{block['split']} {block['checkpoint']} {fmt(value)}")
    out += [
        "",
        "Mean fleet delta after removing the three largest positive dates: "
        + ", ".join(dropped3) + " CNY/day. Still positive: "
        + ", ".join(f"{block['split']} {block['checkpoint']}"
                    for block, ok in zip(groups_, positive_after) if ok)
        + "; negative: "
        + ", ".join(f"{block['split']} {block['checkpoint']}"
                    for block, ok in zip(groups_, positive_after) if not ok)
        + ".",
        "",
    ]

    # Q6
    out += ["### 6. Which ESS move against the fleet result?", ""]
    rows = []
    for block in groups_:
        fleet100 = block["by_rate"]["100.0"]["delta_mean_net_cny"]
        fleet0 = block["by_rate"]["0.0"]["delta_mean_net_cny"]
        for scope in ESS_NAMES:
            item = next(b for b in phase2_summary["groups"]
                        if b["phase"] == block["phase"]
                        and b["checkpoint"] == block["checkpoint"]
                        and b["scope"] == scope)
            d100 = item["by_rate"]["100.0"]["delta_mean_net_cny"]
            d0 = item["by_rate"]["0.0"]["delta_mean_net_cny"]

            def relation(base, value):
                if abs(value) < 1e-9:
                    return "zero"
                return "same" if (base > 0) == (value > 0) else "opposite"

            rows.append([block["split"], block["checkpoint"], scope,
                         fmt(fleet0), fmt(d0), relation(fleet0, d0),
                         fmt(fleet100), fmt(d100), relation(fleet100, d100)])
    out += md_table(["split", "checkpoint", "scope", "fleet @0", "ESS @0",
                     "vs fleet @0", "fleet @100", "ESS @100", "vs fleet @100"],
                    rows)
    opposite_100 = sum(1 for row in rows if row[8] == "opposite")
    opposite_0 = sum(1 for row in rows if row[5] == "opposite")
    out += [
        "",
        f"Units opposite the fleet sign at 0 CNY/MWh: {opposite_0} of "
        f"{len(rows)}. At 100 CNY/MWh: {opposite_100} of {len(rows)}. Where a "
        "unit separates at zero cost it is gaining revenue under Z=10 while "
        "the fleet as a whole loses once the cycle cost is removed.",
        "",
    ]

    # Q7
    out += ["### 7. Any accounting non-closure or missing data?", ""]
    out += md_table(["check", "status", "detail"],
                    [[name, item["status"], item["detail"]]
                     for name, item in ver["checks"].items()])
    failing = [name for name, item in ver["checks"].items()
               if item["status"] != "PASS"]
    out += [
        "",
        f"{len(ver['checks']) - len(failing)} of {len(ver['checks'])} checks "
        "pass. "
        + ("No non-closure and no missing input: every paired row has both "
           "maps and four ESS, and the identities close to "
           f"{ver['checks']['pnl_identity']['detail'].split('=')[-1].strip()}."
           if not failing else f"Failing: {', '.join(failing)}."),
        "",
    ]
    return out


def hash_outputs() -> dict:
    files = {}
    for path in sorted(OUT.glob("*")):
        if path.is_file() and path.name != "output_hashes.json":
            files[path.name] = {"sha256": sha256(path),
                                "bytes": path.stat().st_size}
    payload = {"generated_at": now(), "label": LABEL,
               "files": files, "count": len(files)}
    (OUT / "output_hashes.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return payload


def blocked(problems) -> int:
    (OUT / "input_manifest.json").write_text(json.dumps({
        "generated_at": now(), "label": LABEL, "phase": "7A",
        "status": "BLOCKED", "problems": problems,
    }, indent=1) + "\n", encoding="utf-8")
    (OUT / "decision_report.md").write_text(
        "# Phase 7A - BLOCKED\n\nInputs could not be locked:\n\n"
        + "\n".join(f"- {problem}" for problem in problems) + "\n",
        encoding="utf-8")
    print("BLOCKED:", problems)
    return 1


def main() -> int:
    started = time.time()
    if OUT.exists():
        print(f"refusing to overwrite existing output directory {OUT}")
        return 1
    OUT.mkdir(parents=True)

    manifest, cp_hashes = phase0()
    if manifest["status"] != "PASS":
        failing = [name for name, item in manifest["checks"].items()
                   if item["status"] != "PASS"]
        return blocked(manifest["problems"] + failing)
    print(f"phase 0: {len(manifest['inputs'])} inputs locked, "
          f"{len(cp_hashes)} checkpoint/replay files hashed")

    arms, fleet, meta = load_accounts()
    phase1_result = phase1(arms, fleet, meta)
    print(f"phase 1: {phase1_result['status']}, "
          f"max net residual "
          f"{phase1_result['checks']['net_identity']['max_abs_residual_cny']:.3e}")

    phase2_summary, phase2_rows = phase2(arms, fleet, meta)
    print(f"phase 2: base-rate reproduction "
          f"{phase2_summary['base_rate_reproduction']['status']}, "
          f"max diff "
          f"{phase2_summary['base_rate_reproduction']['max_abs_diff']:.3e}")

    conc_rows, conc_summary = phase3(arms, fleet, meta)
    print(f"phase 3: {len(conc_rows)} sensitivity rows, "
          f"{len(conc_summary)} summary rows")

    ver = phase4(manifest, cp_hashes, phase1_result, phase2_summary)
    print(f"phase 4: {ver['status']} "
          f"({sum(1 for v in ver['checks'].values() if v['status'] == 'PASS')}"
          f"/{len(ver['checks'])} checks)")

    (OUT / "decision_report.md").write_text(
        report(manifest, phase1_result, phase2_summary, phase2_rows,
               conc_rows, conc_summary, ver), encoding="utf-8")

    hashes = hash_outputs()
    print(f"wrote {hashes['count']} files to {OUT.relative_to(ROOT)} "
          f"in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
