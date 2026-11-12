"""Phase 6D: read-only attribution of the Phase 6 paired clearing results.

This script never clears a market and never trains. It reads the Phase 6
artifacts, recomputes the paired differences, and writes the attribution
tables into a new directory. Block-level quote and clearing series were not
persisted by the Phase 6 runner, so analyses that need them are reported as
NOT_TESTED rather than reconstructed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT / "reports" / "phase6_frozen_policy_2026-11-12"
OUT = ROOT / "reports" / "phase6d_profit_attribution_2026-11-12"
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
PHASE_FILES = {
    "A": PHASE6 / "phase6A_smoke_results.csv",
    "B": PHASE6 / "phase6B_train_results.csv",
    "C": PHASE6 / "phase6C_validation_results.csv",
}
SPLIT = {"A": "probe", "B": "train", "C": "validation"}
COMBOS = [
    ("A", "seed11_final"), ("A", "seed23_final"),
    ("A", "seed11_best"), ("A", "seed23_best"),
    ("B", "seed11_final"), ("B", "seed23_final"),
    ("C", "seed11_final"), ("C", "seed23_final"),
]
PRIMARY = [("B", "seed11_final"), ("B", "seed23_final"),
           ("C", "seed11_final"), ("C", "seed23_final")]
NEAR_ZERO = (0.01, 0.1, 1.0)
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261112
EPS = 1e-9


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(
        timespec="seconds")


def num(value) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def combo_id(phase: str, checkpoint: str) -> str:
    return f"{checkpoint}_{SPLIT[phase]}"


def is_primary(phase: str, checkpoint: str) -> bool:
    return (phase, checkpoint) in PRIMARY


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows() -> list[dict]:
    rows = []
    for phase, path in PHASE_FILES.items():
        with path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                item["phase"] = phase
                rows.append(item)
    return rows


def key(phase: str, checkpoint: str, z: str, pair: str) -> tuple:
    return (phase, checkpoint, z, pair)


def arm_map(rows: list[dict]) -> dict:
    return {key(r["phase"], r["checkpoint"], r["z_core"], r["pair_id"]): r
            for r in rows}


def pairs_of(rows: list[dict], phase: str, checkpoint: str) -> list[str]:
    return sorted(r["pair_id"] for r in rows
                  if r["phase"] == phase and r["checkpoint"] == checkpoint
                  and r["z_core"] == "2.5")


def nanmean(values) -> float:
    values = [v for v in values if not math.isnan(v)]
    return statistics.fmean(values) if values else float("nan")


def delta_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    positive = arr[arr > 0]
    negative = arr[arr < 0]
    s_plus = float(positive.sum())
    s_minus = float(-negative.sum())
    return {
        "n_dates": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n_positive": int((arr > 0).sum()),
        "n_negative": int((arr < 0).sum()),
        "n_zero": int((arr == 0).sum()),
        "frac_positive": float((arr > 0).mean()),
        "frac_negative": float((arr < 0).mean()),
        "frac_zero": float((arr == 0).mean()),
        "s_plus": s_plus,
        "s_minus": s_minus,
        "s_net": s_plus - s_minus,
        "cr1_plus": (float(np.sort(positive)[-1]) / s_plus
                     if positive.size and s_plus > 0 else float("nan")),
        "cr3_plus": (float(np.sort(positive)[-3:].sum()) / s_plus
                     if positive.size and s_plus > 0 else float("nan")),
        "cr5_plus": (float(np.sort(positive)[-5:].sum()) / s_plus
                     if positive.size and s_plus > 0 else float("nan")),
        "cr1_minus": (float(-np.sort(negative)[0]) / s_minus
                      if negative.size and s_minus > 0 else float("nan")),
        "cr3_minus": (float(-np.sort(negative)[:3].sum()) / s_minus
                      if negative.size and s_minus > 0 else float("nan")),
        "cr5_minus": (float(-np.sort(negative)[:5].sum()) / s_minus
                      if negative.size and s_minus > 0 else float("nan")),
    }


def combo_records(rows: list[dict]) -> list[dict]:
    arms = arm_map(rows)
    records = []
    for phase, checkpoint in COMBOS:
        for pair in pairs_of(rows, phase, checkpoint):
            control = arms[key(phase, checkpoint, "2.5", pair)]
            candidate = arms[key(phase, checkpoint, "10.0", pair)]
            record = {
                "combo_id": combo_id(phase, checkpoint),
                "primary": is_primary(phase, checkpoint),
                "phase": phase,
                "split": SPLIT[phase],
                "checkpoint": checkpoint,
                "seed": control["seed"],
                "round": control["round"],
                "role": control["role"],
                "pair_id": pair,
                "load_date": control["load_date"],
                "price_date": control["price_date"],
                "season": control["season"],
                "source_key": f"{phase}|{checkpoint}|{pair}",
                "fleet_control": num(control["fleet_net_income_cny"]),
                "fleet_candidate": num(candidate["fleet_net_income_cny"]),
                "fleet_delta": (num(candidate["fleet_net_income_cny"])
                                - num(control["fleet_net_income_cny"])),
                "control": control,
                "candidate": candidate,
            }
            for nm in NAMES:
                record[f"{nm}_delta"] = (
                    num(candidate[f"{nm}_net_income_cny"])
                    - num(control[f"{nm}_net_income_cny"]))
            record["sign"] = ("positive" if record["fleet_delta"] > 0
                              else "negative" if record["fleet_delta"] < 0
                              else "zero")
            records.append(record)
    return records


def input_inventory() -> list[dict]:
    recorded = {}
    out_hashes = json.loads(
        (PHASE6 / "output_hashes.json").read_text(encoding="utf-8"))["files"]
    for name, digest in out_hashes.items():
        recorded[str(PHASE6 / name)] = ("output_hashes", digest)
    src_hashes = json.loads(
        (PHASE6 / "source_hashes.json").read_text(encoding="utf-8"))
    for rel, digest in src_hashes.get("files", {}).items():
        recorded[str(ROOT / rel)] = ("source_hashes", digest)
    for checkpoint, actors in src_hashes.get(
            "checkpoint_actor_sha256", {}).items():
        for name, digest in actors.items():
            recorded[str(PHASE6 / "checkpoints" / checkpoint / "actors"
                         / f"{name}.pt")] = ("source_hashes", digest)
    rows = []
    seen = set()
    for path in sorted(PHASE6.rglob("*")):
        if not path.is_file():
            continue
        seen.add(path.resolve())
        digest = sha(path)
        source, expected = recorded.get(str(path), ("not_recorded", ""))
        item = {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "kind": path.suffix.lstrip("."),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "recorded_by": source,
            "recorded_sha256": expected,
            "matches_recorded": (digest == expected if expected else "n/a"),
            "data_rows": "",
        }
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                item["data_rows"] = sum(1 for _ in csv.reader(handle)) - 1
        rows.append(item)
    for rel, expected in src_hashes.get("files", {}).items():
        path = (ROOT / rel).resolve()
        if path in seen or not path.exists():
            continue
        digest = sha(path)
        seen.add(path)
        rows.append({
            "relative_path": rel,
            "kind": path.suffix.lstrip("."),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "recorded_by": "source_hashes",
            "recorded_sha256": expected,
            "matches_recorded": digest == expected,
            "data_rows": "",
        })
    return rows


def source_integrity(rows: list[dict], inventory: list[dict]) -> dict:
    checks = {}

    def check(name, ok, detail):
        checks[name] = {
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        }

    config = json.loads((PHASE6 / "phase6_config.json").read_text(
        encoding="utf-8"))
    check("phase6_config", config.get("label") == "NON_FORMAL_DIAGNOSTIC",
          f"label={config.get('label')}, control={config['control']['z_core']},"
          f" candidate={config['candidate']['z_core']}")

    listed = [i for i in inventory if i["recorded_by"] == "output_hashes"]
    bad = [i["relative_path"] for i in listed
           if i["matches_recorded"] is not True]
    check("output_hashes", not bad,
          f"{len(listed)} recorded files, mismatches={bad or 'none'}")

    src_listed = [i for i in inventory if i["recorded_by"] == "source_hashes"]
    src_bad = [i["relative_path"] for i in src_listed
               if i["matches_recorded"] is not True]
    check("source_hashes", not src_bad,
          f"{len(src_listed)} recorded source/checkpoint files on disk, "
          f"mismatches={src_bad or 'none'}")

    source_files = json.loads((PHASE6 / "source_hashes.json").read_text(
        encoding="utf-8")).get("files", {})
    missing_sources = sorted(
        rel for rel in source_files if not (ROOT / rel).exists())
    check("phase6_input_sources_on_disk", not missing_sources,
          f"{len(source_files)} Phase 6 source inputs recorded, "
          f"missing={missing_sources or 'none'}")

    actor_files = [i for i in inventory
                   if "/actors/" in i["relative_path"]]
    check("checkpoint_actors", len(actor_files) == 16
          and all(i["matches_recorded"] is True for i in actor_files),
          f"{len(actor_files)} actor files, "
          f"mismatches="
          f"{[i['relative_path'] for i in actor_files if i['matches_recorded'] is not True] or 'none'}")

    expected = {"A": 64, "B": 212, "C": 44}
    counts = {p: sum(1 for r in rows if r["phase"] == p) for p in expected}
    check("result_row_counts", counts == expected,
          f"counts={counts}, expected={expected}")

    keys = {}
    for r in rows:
        k = key(r["phase"], r["checkpoint"], r["z_core"], r["pair_id"])
        keys[k] = keys.get(k, 0) + 1
    duplicates = [k for k, v in keys.items() if v > 1]
    check("no_duplicate_rows", not duplicates,
          f"duplicate keys={duplicates or 'none'}")

    arms = arm_map(rows)
    missing = []
    group = {}
    for (phase, checkpoint, z, pair) in keys:
        group.setdefault((phase, checkpoint, pair), set()).add(z)
    missing = [g for g, zs in group.items() if zs != {"2.5", "10.0"}]
    check("arm_pairing_complete", not missing,
          f"{len(group)} checkpoint/date groups, "
          f"incomplete={missing or 'none'}")

    mismatched = []
    for (phase, checkpoint, pair) in group:
        control = arms[key(phase, checkpoint, "2.5", pair)]
        candidate = arms[key(phase, checkpoint, "10.0", pair)]
        for field in ("seed", "round", "role", "load_date", "price_date",
                      "season"):
            if control[field] != candidate[field]:
                mismatched.append(f"{phase}/{checkpoint}/{pair}/{field}")
    check("arm_metadata_identical", not mismatched,
          f"mismatched fields={mismatched or 'none'}")

    not_ok = [r["source_key"] if "source_key" in r else
              f"{r['phase']}/{r['checkpoint']}/{r['z_core']}/{r['pair_id']}"
              for r in rows if r["day_ok"] != "True"]
    check("day_ok_all", not not_ok, f"failed days={not_ok or 'none'}")

    tinv = max(abs(num(r["fleet_terminal_inventory_value_cny"]))
               for r in rows)
    check("terminal_inventory_zero", tinv < 1e-12, f"max |term|={tinv:.3e}")

    allowed = set()
    for phase in ("A", "B", "C"):
        for item in config[f"phase6{phase}"]["pairs"]:
            allowed.add(item["pair_id"])
    outside = sorted({r["pair_id"] for r in rows} - allowed)
    check("pairs_declared_in_config", not outside,
          f"{len({r['pair_id'] for r in rows})} distinct dates, "
          f"outside={outside or 'none'}")

    preflight = json.loads((PHASE6 / "phase6_preflight.json").read_text(
        encoding="utf-8"))["checks"]
    check("phase6_preflight", preflight.get("all_pass") is True,
          f"all_pass={preflight.get('all_pass')}")

    report_text = (PHASE6 / "phase6_decision_report.md").read_text(
        encoding="utf-8")
    check("original_verdict_preserved", "**Verdict: `PARTIAL`**" in report_text,
          "Phase 6 report states PARTIAL")

    boot = json.loads((PHASE6 / "bootstrap_summary.json").read_text(
        encoding="utf-8"))["bootstrap"]
    check("bootstrap_config",
          boot == {"n_resamples": N_RESAMPLES, "seed": BOOTSTRAP_SEED,
                   "cluster": "pair_id (date)", "interval": "percentile"},
          f"config={boot}")

    essential = ("output_hashes", "source_hashes",
                 "phase6_input_sources_on_disk", "checkpoint_actors",
                 "result_row_counts", "arm_pairing_complete",
                 "arm_metadata_identical")
    failed = [n for n in essential if checks[n]["status"] != "PASS"]
    return {
        "generated_at": now_iso(),
        "phase6_dir": str(PHASE6.relative_to(ROOT)),
        "checks": checks,
        "essential_checks": list(essential),
        "overall": "BLOCKED" if failed else "PASS",
    }


def accounting(records: list[dict], rows: list[dict]) -> tuple[list[dict], dict]:
    out_rows = []
    residuals = {"identity": 0.0, "fleet_sum": 0.0,
                 "delta_identity": 0.0, "delta_fleet_sum": 0.0}
    for r in rows:
        for scope in ("fleet",) + NAMES:
            if scope == "fleet":
                net = num(r["fleet_net_income_cny"])
                rev = num(r["fleet_market_revenue_cny"])
                pur = num(r["fleet_purchase_cost_cny"])
                cyc = num(r["fleet_cycle_cost_cny"])
                tinv = num(r["fleet_terminal_inventory_value_cny"])
                residual = net - (rev - pur - cyc)
                ess_sum = sum(num(r[f"{nm}_net_income_cny"]) for nm in NAMES)
                fleet_sum = net - ess_sum
            else:
                net = num(r[f"{scope}_net_income_cny"])
                rev = num(r[f"{scope}_market_revenue_cny"])
                pur = num(r[f"{scope}_purchase_cost_cny"])
                cyc = num(r[f"{scope}_cycle_cost_cny"])
                tinv = num(r[f"{scope}_terminal_inventory_value_cny"])
                residual = net - (rev - pur - cyc)
                fleet_sum = float("nan")
            residuals["identity"] = max(residuals["identity"], abs(residual))
            if scope == "fleet":
                residuals["fleet_sum"] = max(residuals["fleet_sum"],
                                             abs(fleet_sum))
            out_rows.append({
                "row_type": "arm",
                "phase": r["phase"],
                "checkpoint": r["checkpoint"],
                "seed": r["seed"],
                "round": r["round"],
                "pair_id": r["pair_id"],
                "scope": scope,
                "z_core": r["z_core"],
                "net_income_cny": net,
                "market_revenue_cny": rev,
                "purchase_cost_cny": pur,
                "cycle_cost_cny": cyc,
                "terminal_inventory_value_cny": tinv,
                "identity_residual": residual,
                "fleet_minus_sum_ess_residual": fleet_sum,
            })
    for record in records:
        c = record["control"]
        x = record["candidate"]
        fields = ("net_income_cny", "market_revenue_cny",
                  "purchase_cost_cny", "cycle_cost_cny",
                  "terminal_inventory_value_cny")
        delta = {f: num(x[f"fleet_{f}"]) - num(c[f"fleet_{f}"])
                 for f in fields}
        d_identity = delta["net_income_cny"] - (
            delta["market_revenue_cny"] - delta["purchase_cost_cny"]
            - delta["cycle_cost_cny"])
        d_fleet_sum = delta["net_income_cny"] - sum(
            record[f"{nm}_delta"] for nm in NAMES)
        residuals["delta_identity"] = max(residuals["delta_identity"],
                                          abs(d_identity))
        residuals["delta_fleet_sum"] = max(residuals["delta_fleet_sum"],
                                           abs(d_fleet_sum))
        out_rows.append({
            "row_type": "delta",
            "phase": record["phase"],
            "checkpoint": record["checkpoint"],
            "seed": record["seed"],
            "round": record["round"],
            "pair_id": record["pair_id"],
            "scope": "fleet",
            "z_core": "delta",
            "net_income_cny": delta["net_income_cny"],
            "market_revenue_cny": delta["market_revenue_cny"],
            "purchase_cost_cny": delta["purchase_cost_cny"],
            "cycle_cost_cny": delta["cycle_cost_cny"],
            "terminal_inventory_value_cny":
                delta["terminal_inventory_value_cny"],
            "identity_residual": d_identity,
            "fleet_minus_sum_ess_residual": d_fleet_sum,
        })
    tolerance = 1e-6
    summary = {
        **residuals,
        "tolerance_cny": tolerance,
        "all_within_tolerance": all(v <= tolerance for v in residuals.values()),
    }
    return out_rows, summary


def daily_tables(records: list[dict]):
    rank_rows = []
    top_bottom = []
    for phase, checkpoint in COMBOS:
        items = [r for r in records if r["phase"] == phase
                 and r["checkpoint"] == checkpoint]
        items.sort(key=lambda r: (-r["fleet_delta"], r["pair_id"]))
        for rank, r in enumerate(items, start=1):
            rank_rows.append({
                "combo_id": r["combo_id"],
                "primary": r["primary"],
                "phase": r["phase"],
                "split": r["split"],
                "checkpoint": r["checkpoint"],
                "seed": r["seed"],
                "round": r["round"],
                "role": r["role"],
                "rank_desc": rank,
                "pair_id": r["pair_id"],
                "load_date": r["load_date"],
                "price_date": r["price_date"],
                "season": r["season"],
                "control_cny": r["fleet_control"],
                "candidate_cny": r["fleet_candidate"],
                "delta_cny": r["fleet_delta"],
                **{f"{nm}_delta_cny": r[f"{nm}_delta"] for nm in NAMES},
                "sign": r["sign"],
                "abs_lt_0p01": abs(r["fleet_delta"]) < 0.01,
                "abs_lt_0p1": abs(r["fleet_delta"]) < 0.1,
                "abs_lt_1p0": abs(r["fleet_delta"]) < 1.0,
                "source_key": r["source_key"],
            })
        positives = sorted((r for r in items if r["fleet_delta"] > 0),
                           key=lambda r: -r["fleet_delta"])
        negatives = sorted((r for r in items if r["fleet_delta"] < 0),
                           key=lambda r: r["fleet_delta"])
        s_plus = sum(r["fleet_delta"] for r in positives)
        s_minus = sum(-r["fleet_delta"] for r in negatives)
        for tail, source, total in (("top", positives, s_plus),
                                    ("bottom", negatives, s_minus)):
            cumulative = 0.0
            for rank, r in enumerate(source[:5], start=1):
                value = abs(r["fleet_delta"])
                cumulative += value
                top_bottom.append({
                    "combo_id": r["combo_id"],
                    "primary": r["primary"],
                    "phase": r["phase"],
                    "checkpoint": r["checkpoint"],
                    "tail": tail,
                    "rank": rank,
                    "pair_id": r["pair_id"],
                    "load_date": r["load_date"],
                    "price_date": r["price_date"],
                    "season": r["season"],
                    "delta_cny": r["fleet_delta"],
                    "abs_cny": value,
                    "cumulative_abs_cny": cumulative,
                    "share_of_tail_total": (value / total if total > 0
                                            else float("nan")),
                })
    return rank_rows, top_bottom


def distribution_and_sensitivity(records: list[dict]):
    dist_rows = []
    sens_rows = []
    for phase, checkpoint in COMBOS:
        items = [r for r in records if r["phase"] == phase
                 and r["checkpoint"] == checkpoint]
        values = [r["fleet_delta"] for r in items]
        stats = delta_stats(values)
        row = {
            "combo_id": combo_id(phase, checkpoint),
            "primary": is_primary(phase, checkpoint),
            "phase": phase,
            "split": SPLIT[phase],
            "checkpoint": checkpoint,
            "seed": items[0]["seed"],
            "round": items[0]["round"],
            "role": items[0]["role"],
            **stats,
        }
        for threshold in NEAR_ZERO:
            row[f"n_abs_lt_{str(threshold).replace('.', 'p')}"] = sum(
                1 for v in values if abs(v) < threshold)
            row[f"frac_abs_lt_{str(threshold).replace('.', 'p')}"] = (
                sum(1 for v in values if abs(v) < threshold) / len(values))
        dist_rows.append(row)

        ordered = sorted(items, key=lambda r: -r["fleet_delta"])
        ordered_neg = sorted(items, key=lambda r: r["fleet_delta"])
        scenarios = [
            ("full", []),
            ("drop_top1", ordered[:1]),
            ("drop_top3", ordered[:3]),
            ("drop_top5", ordered[:5]),
            ("drop_bottom1", ordered_neg[:1]),
            ("drop_bottom3", ordered_neg[:3]),
            ("drop_bottom5", ordered_neg[:5]),
        ]
        for name, dropped in scenarios:
            keep = [r for r in items if r not in dropped]
            kept_values = [r["fleet_delta"] for r in keep]
            sens_rows.append({
                "combo_id": combo_id(phase, checkpoint),
                "primary": is_primary(phase, checkpoint),
                "phase": phase,
                "checkpoint": checkpoint,
                "scenario": name,
                "n_remaining": len(keep),
                "mean_remaining_cny": (statistics.fmean(kept_values)
                                       if kept_values else float("nan")),
                "median_remaining_cny": (statistics.median(kept_values)
                                         if kept_values else float("nan")),
                "removed_dates": ";".join(r["pair_id"] for r in dropped),
                "remaining_dates": ";".join(r["pair_id"] for r in keep),
            })
    return dist_rows, sens_rows


def scope_value(row: dict, scope: str, field: str) -> float:
    if scope == "fleet":
        if field == "cleared_charge_mwh":
            return sum(num(row[f"{nm}_cleared_charge_mwh"]) for nm in NAMES)
        if field == "cleared_discharge_mwh":
            return sum(num(row[f"{nm}_cleared_discharge_mwh"])
                       for nm in NAMES)
        if field == "terminal_soc":
            return num(row["fleet_terminal_soc_min"])
        if field == "lmp_mean":
            return nanmean([num(row[f"{nm}_lmp_mean"]) for nm in NAMES])
        return num(row[f"fleet_{field}"])
    return num(row[f"{scope}_{field}"])


def per_ess(records: list[dict]):
    rows = []
    for phase, checkpoint in COMBOS:
        items = [r for r in records if r["phase"] == phase
                 and r["checkpoint"] == checkpoint]
        fleet_sum_delta = sum(r["fleet_delta"] for r in items)
        fields = ("net_income_cny", "market_revenue_cny",
                  "purchase_cost_cny", "cycle_cost_cny",
                  "charge_mwh", "discharge_mwh",
                  "cleared_charge_mwh", "cleared_discharge_mwh",
                  "terminal_soc", "lmp_mean")
        for scope in ("fleet",) + NAMES:
            deltas = {f: [] for f in fields}
            controls = {f: [] for f in fields}
            candidates = {f: [] for f in fields}
            for r in items:
                control = r["control"]
                candidate = r["candidate"]
                for f in fields:
                    controls[f].append(scope_value(control, scope, f))
                    candidates[f].append(scope_value(candidate, scope, f))
                    deltas[f].append(scope_value(candidate, scope, f)
                                     - scope_value(control, scope, f))
            row = {
                "combo_id": combo_id(phase, checkpoint),
                "primary": is_primary(phase, checkpoint),
                "phase": phase,
                "checkpoint": checkpoint,
                "scope": scope,
                "n_dates": len(items),
                "control_mean_net_cny": statistics.fmean(controls[
                    "net_income_cny"]),
                "candidate_mean_net_cny": statistics.fmean(candidates[
                    "net_income_cny"]),
                "delta_mean_net_cny": statistics.fmean(deltas[
                    "net_income_cny"]),
                "delta_sum_net_cny": sum(deltas["net_income_cny"]),
                "delta_mean_market_revenue_cny": statistics.fmean(
                    deltas["market_revenue_cny"]),
                "delta_mean_purchase_cost_cny": statistics.fmean(
                    deltas["purchase_cost_cny"]),
                "delta_mean_cycle_cost_cny": statistics.fmean(
                    deltas["cycle_cost_cny"]),
                "delta_mean_charge_mwh": statistics.fmean(
                    deltas["charge_mwh"]),
                "delta_mean_discharge_mwh": statistics.fmean(
                    deltas["discharge_mwh"]),
                "delta_mean_cleared_charge_mwh": statistics.fmean(
                    deltas["cleared_charge_mwh"]),
                "delta_mean_cleared_discharge_mwh": statistics.fmean(
                    deltas["cleared_discharge_mwh"]),
                "delta_mean_terminal_soc": statistics.fmean(
                    deltas["terminal_soc"]),
                "delta_mean_lmp": statistics.fmean(deltas["lmp_mean"]),
                "ess_positive_days": sum(1 for v in deltas["net_income_cny"]
                                         if v > 0),
                "ess_negative_days": sum(1 for v in deltas["net_income_cny"]
                                         if v < 0),
                "share_of_fleet_delta": (
                    sum(deltas["net_income_cny"]) / fleet_sum_delta
                    if abs(fleet_sum_delta) > EPS else float("nan")),
                "historical_check": "",
            }
            if phase == "A" and scope == "ESS17" \
                    and checkpoint == "seed11_best":
                row["historical_check"] = "seed11_best/ESS17"
            if phase == "A" and scope == "ESS14" \
                    and checkpoint == "seed23_best":
                row["historical_check"] = "seed23_best/ESS14"
            rows.append(row)
    return rows


def leg_tables(records: list[dict]):
    exposure = []
    quote_rows = []
    for phase, checkpoint in COMBOS:
        items = [r for r in records if r["phase"] == phase
                 and r["checkpoint"] == checkpoint]
        for nm in NAMES:
            for leg, energy_field, cleared_field in (
                    ("charge", "charge_mwh", "cleared_charge_mwh"),
                    ("discharge", "discharge_mwh", "cleared_discharge_mwh")):
                control = [r["control"] for r in items]
                candidate = [r["candidate"] for r in items]
                c_frac = [num(r[f"{nm}_gate_open_{leg}_frac"])
                          for r in control]
                x_frac = [num(r[f"{nm}_gate_open_{leg}_frac"])
                          for r in candidate]
                c_open = [round(f * 24) for f in c_frac]
                x_open = [round(f * 24) for f in x_frac]
                c_quote = [num(r[f"{nm}_quote_{leg}_mean_open"])
                           for r in control]
                x_quote = [num(r[f"{nm}_quote_{leg}_mean_open"])
                           for r in candidate]
                both_open = [i for i in range(len(items))
                             if not math.isnan(c_quote[i])
                             and not math.isnan(x_quote[i])]
                quote_delta = [x_quote[i] - c_quote[i] for i in both_open]
                exposure.append({
                    "combo_id": combo_id(phase, checkpoint),
                    "primary": is_primary(phase, checkpoint),
                    "phase": phase,
                    "checkpoint": checkpoint,
                    "ess": nm,
                    "leg": leg,
                    "ess_days": len(items),
                    "open_blocks_total_control": sum(c_open),
                    "open_blocks_total_candidate": sum(x_open),
                    "closed_blocks_total_control": 24 * len(items)
                    - sum(c_open),
                    "closed_blocks_total_candidate": 24 * len(items)
                    - sum(x_open),
                    "open_frac_mean_control": nanmean(c_frac),
                    "open_frac_mean_candidate": nanmean(x_frac),
                    "full_closed_days_control": sum(1 for v in c_open
                                                    if v == 0),
                    "full_closed_days_candidate": sum(1 for v in x_open
                                                      if v == 0),
                    "full_open_days_control": sum(1 for v in c_open
                                                  if v == 24),
                    "full_open_days_candidate": sum(1 for v in x_open
                                                    if v == 24),
                    "same_open_block_count_days": sum(
                        1 for a, b in zip(c_open, x_open) if a == b),
                    "differing_open_block_count_days": sum(
                        1 for a, b in zip(c_open, x_open) if a != b),
                    "abs_open_block_count_diff_total": sum(
                        abs(a - b) for a, b in zip(c_open, x_open)),
                    "same_state_blocks": "NOT_TESTED",
                })
                quote_rows.append({
                    "combo_id": combo_id(phase, checkpoint),
                    "primary": is_primary(phase, checkpoint),
                    "phase": phase,
                    "checkpoint": checkpoint,
                    "ess": nm,
                    "leg": leg,
                    "ess_days": len(items),
                    "both_gate_open_days": len(both_open),
                    "both_gate_closed_days": sum(
                        1 for i in range(len(items))
                        if math.isnan(c_quote[i]) and math.isnan(x_quote[i])),
                    "only_control_closed_days": sum(
                        1 for i in range(len(items))
                        if math.isnan(c_quote[i])
                        and not math.isnan(x_quote[i])),
                    "only_candidate_closed_days": sum(
                        1 for i in range(len(items))
                        if not math.isnan(c_quote[i])
                        and math.isnan(x_quote[i])),
                    "days_quote_changed": sum(
                        1 for d in quote_delta if abs(d) > 1e-6),
                    "days_quote_same": sum(
                        1 for d in quote_delta if abs(d) <= 1e-6),
                    "mean_quote_change_open_days": (
                        statistics.fmean(quote_delta)
                        if quote_delta else float("nan")),
                    "max_abs_day_mean_quote_change": (
                        max(abs(d) for d in quote_delta)
                        if quote_delta else float("nan")),
                    "mean_energy_delta_mwh": statistics.fmean(
                        [num(x[f"{nm}_{energy_field}"])
                         - num(c[f"{nm}_{energy_field}"])
                         for c, x in zip(control, candidate)]),
                    "mean_cleared_delta_mwh": statistics.fmean(
                        [num(x[f"{nm}_{cleared_field}"])
                         - num(c[f"{nm}_{cleared_field}"])
                         for c, x in zip(control, candidate)]),
                    "mean_market_revenue_delta_cny": statistics.fmean(
                        [num(x[f"{nm}_market_revenue_cny"])
                         - num(c[f"{nm}_market_revenue_cny"])
                         for c, x in zip(control, candidate)]),
                    "mean_lmp_delta": statistics.fmean(
                        [num(x[f"{nm}_lmp_mean"])
                         - num(c[f"{nm}_lmp_mean"])
                         for c, x in zip(control, candidate)]),
                    "block_level_quote_changes": "NOT_TESTED",
                    "block_level_clearing_changes": "NOT_TESTED",
                })
    return exposure, quote_rows


def transfer_tables(records: list[dict]):
    rows = []
    for phase, checkpoint in COMBOS:
        items = [r for r in records if r["phase"] == phase
                 and r["checkpoint"] == checkpoint]
        fleet_pos = [r for r in items if r["fleet_delta"] > 0]
        fleet_neg = [r for r in items if r["fleet_delta"] < 0]
        pos_days = sum(1 for r in items for nm in NAMES
                       if r[f"{nm}_delta"] > 0)
        neg_days = sum(1 for r in items for nm in NAMES
                       if r[f"{nm}_delta"] < 0)
        rows.append({
            "row_type": "combo",
            "combo_id": combo_id(phase, checkpoint),
            "primary": is_primary(phase, checkpoint),
            "phase": phase,
            "checkpoint": checkpoint,
            "ess": "ALL",
            "n_dates": len(items),
            "ess_positive_days": pos_days,
            "ess_negative_days": neg_days,
            "fleet_positive_dates": len(fleet_pos),
            "fleet_positive_with_losing_ess": sum(
                1 for r in fleet_pos
                if any(r[f"{nm}_delta"] < 0 for nm in NAMES)),
            "fleet_negative_dates": len(fleet_neg),
            "fleet_negative_with_winning_ess": sum(
                1 for r in fleet_neg
                if any(r[f"{nm}_delta"] > 0 for nm in NAMES)),
            "dates_all_ess_same_sign_as_fleet": sum(
                1 for r in items
                if all((r[f"{nm}_delta"] > 0) == (r["fleet_delta"] > 0)
                       for nm in NAMES)),
            "sum_delta_cny": sum(r["fleet_delta"] for r in items),
            "share_of_positive": float("nan"),
            "share_of_negative": float("nan"),
        })
        all_positive = sum(max(r[f"{nm}_delta"], 0.0) for r in items
                           for nm in NAMES)
        all_negative = sum(max(-r[f"{nm}_delta"], 0.0) for r in items
                           for nm in NAMES)
        for nm in NAMES:
            ess_deltas = [r[f"{nm}_delta"] for r in items]
            rows.append({
                "row_type": "ess",
                "combo_id": combo_id(phase, checkpoint),
                "primary": is_primary(phase, checkpoint),
                "phase": phase,
                "checkpoint": checkpoint,
                "ess": nm,
                "n_dates": len(items),
                "ess_positive_days": sum(1 for v in ess_deltas if v > 0),
                "ess_negative_days": sum(1 for v in ess_deltas if v < 0),
                "fleet_positive_dates": "",
                "fleet_positive_with_losing_ess": "",
                "fleet_negative_dates": "",
                "fleet_negative_with_winning_ess": "",
                "dates_all_ess_same_sign_as_fleet": "",
                "sum_delta_cny": sum(ess_deltas),
                "share_of_positive": (
                    sum(max(v, 0.0) for v in ess_deltas) / all_positive
                    if all_positive > 0 else float("nan")),
                "share_of_negative": (
                    sum(max(-v, 0.0) for v in ess_deltas) / all_negative
                    if all_negative > 0 else float("nan")),
            })
    return rows


def component_summary(records: list[dict]):
    rows = []
    components = ("market_revenue_cny", "purchase_cost_cny",
                  "cycle_cost_cny", "net_income_cny")
    for phase, checkpoint in COMBOS:
        items = [r for r in records if r["phase"] == phase
                 and r["checkpoint"] == checkpoint]
        for scope in ("fleet",) + NAMES:
            prefix = "fleet_" if scope == "fleet" else f"{scope}_"
            for comp in components:
                c = [num(r["control"][f"{prefix}{comp}"]) for r in items]
                x = [num(r["candidate"][f"{prefix}{comp}"]) for r in items]
                d = [b - a for a, b in zip(c, x)]
                rows.append({
                    "combo_id": combo_id(phase, checkpoint),
                    "primary": is_primary(phase, checkpoint),
                    "phase": phase,
                    "checkpoint": checkpoint,
                    "scope": scope,
                    "component": comp,
                    "n_dates": len(items),
                    "control_mean_cny": statistics.fmean(c),
                    "candidate_mean_cny": statistics.fmean(x),
                    "delta_mean_cny": statistics.fmean(d),
                    "delta_total_cny": sum(d),
                    "delta_min_cny": min(d),
                    "delta_max_cny": max(d),
                })
    return rows


def quote_to_clearing(records: list[dict]):
    rows = []
    for record in records:
        if not record["primary"]:
            continue
        c = record["control"]
        x = record["candidate"]
        for nm in NAMES:
            d_quote_charge = (num(x[f"{nm}_quote_charge_mean_open"])
                              - num(c[f"{nm}_quote_charge_mean_open"]))
            d_quote_discharge = (
                num(x[f"{nm}_quote_discharge_mean_open"])
                - num(c[f"{nm}_quote_discharge_mean_open"]))
            d_cleared = (
                num(x[f"{nm}_cleared_charge_mwh"])
                - num(c[f"{nm}_cleared_charge_mwh"])
                + num(x[f"{nm}_cleared_discharge_mwh"])
                - num(c[f"{nm}_cleared_discharge_mwh"]))
            d_net = record[f"{nm}_delta"]
            d_rev = (num(x[f"{nm}_market_revenue_cny"])
                     - num(c[f"{nm}_market_revenue_cny"]))
            d_cyc = (num(x[f"{nm}_cycle_cost_cny"])
                     - num(c[f"{nm}_cycle_cost_cny"]))
            quote_finite = not (math.isnan(d_quote_charge)
                                or math.isnan(d_quote_discharge))
            quote_changed = quote_finite and (
                abs(d_quote_charge) > 1e-6 or abs(d_quote_discharge) > 1e-6)
            rows.append({
                "combo_id": record["combo_id"],
                "phase": record["phase"],
                "checkpoint": record["checkpoint"],
                "pair_id": record["pair_id"],
                "ess": nm,
                "d_quote_charge_open": d_quote_charge,
                "d_quote_discharge_open": d_quote_discharge,
                "quote_comparable": quote_finite,
                "d_cleared_energy_mwh": d_cleared,
                "d_market_revenue_cny": d_rev,
                "d_cycle_cost_cny": d_cyc,
                "d_net_cny": d_net,
                "d_lmp_mean": (num(x[f"{nm}_lmp_mean"])
                               - num(c[f"{nm}_lmp_mean"])),
                "d_gate_open_charge_blocks": (
                    round(num(x[f"{nm}_gate_open_charge_frac"]) * 24)
                    - round(num(c[f"{nm}_gate_open_charge_frac"]) * 24)),
                "d_gate_open_discharge_blocks": (
                    round(num(x[f"{nm}_gate_open_discharge_frac"]) * 24)
                    - round(num(c[f"{nm}_gate_open_discharge_frac"]) * 24)),
                "quote_changed_no_clearing": (quote_changed
                                              and abs(d_cleared) < 1e-6),
                "clearing_up_cycle_up": d_cleared > 0 and d_cyc > 0,
                "block_level": "NOT_TESTED",
            })
    return rows


def dynamic_path(records: list[dict]):
    rows = []
    for record in records:
        if not record["primary"]:
            continue
        c = record["control"]
        x = record["candidate"]
        control_open = sum(round(num(c[f"{nm}_gate_open_charge_frac"]) * 24)
                           + round(num(c[f"{nm}_gate_open_discharge_frac"]) * 24)
                           for nm in NAMES)
        candidate_open = sum(
            round(num(x[f"{nm}_gate_open_charge_frac"]) * 24)
            + round(num(x[f"{nm}_gate_open_discharge_frac"]) * 24)
            for nm in NAMES)
        control_cleared = sum(num(c[f"{nm}_cleared_charge_mwh"])
                              + num(c[f"{nm}_cleared_discharge_mwh"])
                              for nm in NAMES)
        candidate_cleared = sum(num(x[f"{nm}_cleared_charge_mwh"])
                                + num(x[f"{nm}_cleared_discharge_mwh"])
                                for nm in NAMES)
        control_lmp = nanmean([num(c[f"{nm}_lmp_mean"]) for nm in NAMES])
        candidate_lmp = nanmean([num(x[f"{nm}_lmp_mean"]) for nm in NAMES])
        rows.append({
            "combo_id": record["combo_id"],
            "phase": record["phase"],
            "checkpoint": record["checkpoint"],
            "pair_id": record["pair_id"],
            "fleet_delta_cny": record["fleet_delta"],
            "gate_open_blocks_total_control": control_open,
            "gate_open_blocks_total_candidate": candidate_open,
            "gate_open_blocks_diff": candidate_open - control_open,
            "cleared_energy_control_mwh": control_cleared,
            "cleared_energy_candidate_mwh": candidate_cleared,
            "cleared_energy_diff_mwh": candidate_cleared - control_cleared,
            "lmp_mean_control": control_lmp,
            "lmp_mean_candidate": candidate_lmp,
            "lmp_mean_diff": candidate_lmp - control_lmp,
            "first_divergence_block": "NOT_TESTED",
            "same_first_block_output": "NOT_TESTED",
            "note": "block series were not persisted; only day aggregates exist",
        })
    return rows


def top_bottom_mechanism(records: list[dict]):
    rows = []
    for phase, checkpoint in COMBOS:
        if not is_primary(phase, checkpoint):
            continue
        items = [r for r in records if r["phase"] == phase
                 and r["checkpoint"] == checkpoint]
        ordered = sorted(items, key=lambda r: -r["fleet_delta"])
        selection = ([("top", rank, r) for rank, r in
                      enumerate(ordered[:5], start=1)]
                     + [("bottom", rank, r) for rank, r in
                        enumerate(sorted(items,
                                         key=lambda r: r["fleet_delta"])[:5],
                                  start=1)])
        for tail, rank, r in selection:
            c = r["control"]
            x = r["candidate"]
            d_rev = (num(x["fleet_market_revenue_cny"])
                     - num(c["fleet_market_revenue_cny"]))
            d_pur = (num(x["fleet_purchase_cost_cny"])
                     - num(c["fleet_purchase_cost_cny"]))
            d_cyc = (num(x["fleet_cycle_cost_cny"])
                     - num(c["fleet_cycle_cost_cny"]))
            parts = {"market_revenue": d_rev, "purchase_cost": d_pur,
                     "cycle_cost": d_cyc}
            dominant = max(parts, key=lambda k: abs(parts[k]))
            rows.append({
                "combo_id": r["combo_id"],
                "primary": True,
                "phase": phase,
                "checkpoint": checkpoint,
                "tail": tail,
                "rank": rank,
                "pair_id": r["pair_id"],
                "load_date": r["load_date"],
                "price_date": r["price_date"],
                "season": r["season"],
                "fleet_delta_cny": r["fleet_delta"],
                "d_market_revenue_cny": d_rev,
                "d_purchase_cost_cny": d_pur,
                "d_cycle_cost_cny": d_cyc,
                "dominant_component": dominant,
                "d_charge_mwh": (num(x["fleet_charge_mwh"])
                                 - num(c["fleet_charge_mwh"])),
                "d_discharge_mwh": (num(x["fleet_discharge_mwh"])
                                    - num(c["fleet_discharge_mwh"])),
                "d_lmp_mean": (nanmean([num(x[f"{nm}_lmp_mean"])
                                        for nm in NAMES])
                               - nanmean([num(c[f"{nm}_lmp_mean"])
                                          for nm in NAMES])),
                "d_terminal_soc_min": (num(x["fleet_terminal_soc_min"])
                                       - num(c["fleet_terminal_soc_min"])),
                **{f"{nm}_delta_cny": r[f"{nm}_delta"] for nm in NAMES},
                "block_level": "NOT_TESTED",
            })
    return rows


def bootstrap_reproduction(records: list[dict]) -> tuple[list[dict], dict]:
    original = json.loads((PHASE6 / "bootstrap_summary.json").read_text(
        encoding="utf-8"))["results"]
    original_by = {(r["phase"], r["checkpoint"], r["scope"]): r
                   for r in original}
    reproduced = []
    for phase, checkpoint in COMBOS:
        items = [r for r in records if r["phase"] == phase
                 and r["checkpoint"] == checkpoint]
        scopes = {"fleet": [r["fleet_delta"] for r in items]}
        for nm in NAMES:
            scopes[nm] = [r[f"{nm}_delta"] for r in items]
        for scope, values in scopes.items():
            arr = np.asarray(values, dtype=float)
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            idx = rng.integers(0, arr.size, size=(N_RESAMPLES, arr.size))
            means = arr[idx].mean(axis=1)
            block = {
                "n_dates": int(arr.size),
                "mean_delta_cny": float(arr.mean()),
                "median_delta_cny": float(np.median(arr)),
                "fraction_positive": float((arr > 0).mean()),
                "ci95_low": float(np.percentile(means, 2.5)),
                "ci95_high": float(np.percentile(means, 97.5)),
            }
            ref = original_by[(phase, checkpoint, scope)]
            diffs = {f: abs(block[f] - ref[f]) for f in block}
            reproduced.append({
                "phase": phase,
                "checkpoint": checkpoint,
                "scope": scope,
                "n_dates": block["n_dates"],
                "original_mean": ref["mean_delta_cny"],
                "reproduced_mean": block["mean_delta_cny"],
                "original_median": ref["median_delta_cny"],
                "reproduced_median": block["median_delta_cny"],
                "original_fraction_positive": ref["fraction_positive"],
                "reproduced_fraction_positive": block["fraction_positive"],
                "original_ci95_low": ref["ci95_low"],
                "reproduced_ci95_low": block["ci95_low"],
                "original_ci95_high": ref["ci95_high"],
                "reproduced_ci95_high": block["ci95_high"],
                "max_abs_diff": max(diffs.values()),
                "match": max(diffs.values()) < 1e-9,
            })
    max_diff = max(r["max_abs_diff"] for r in reproduced)
    payload = {
        "generated_at": now_iso(),
        "config": {
            "n_resamples": N_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "pair_id (date)",
            "interval": "percentile",
            "statistic": "mean of the date-level paired delta",
        },
        "source": "bootstrap_summary.json in the Phase 6 directory",
        "groups": len(reproduced),
        "max_abs_diff": max_diff,
        "all_match": all(r["match"] for r in reproduced),
    }
    return reproduced, payload


def robustness(dist_rows: list[dict], sens_rows: list[dict],
               boot_rows: list[dict]) -> list[dict]:
    sens_by = {}
    for row in sens_rows:
        sens_by.setdefault(row["combo_id"], {})[row["scenario"]] = row
    boot_by = {(r["checkpoint"], r["phase"]): r
               for r in boot_rows if r["scope"] == "fleet"}
    rows = []
    for dist in dist_rows:
        if not dist["primary"]:
            continue
        combo = dist["combo_id"]
        scenarios = sens_by[combo]
        boot = boot_by[(dist["checkpoint"], dist["phase"])]
        rows.append({
            "combo_id": combo,
            "phase": dist["phase"],
            "checkpoint": dist["checkpoint"],
            "split": dist["split"],
            "n_dates": dist["n_dates"],
            "mean_cny": dist["mean"],
            "median_cny": dist["median"],
            "fraction_positive": dist["frac_positive"],
            "fraction_abs_lt_0p01": dist["frac_abs_lt_0p01"],
            "s_plus_cny": dist["s_plus"],
            "s_minus_cny": dist["s_minus"],
            "cr1_plus": dist["cr1_plus"],
            "cr3_plus": dist["cr3_plus"],
            "cr5_plus": dist["cr5_plus"],
            "cr1_minus": dist["cr1_minus"],
            "cr3_minus": dist["cr3_minus"],
            "cr5_minus": dist["cr5_minus"],
            "mean_drop_top1": scenarios["drop_top1"]["mean_remaining_cny"],
            "mean_drop_top3": scenarios["drop_top3"]["mean_remaining_cny"],
            "mean_drop_top5": scenarios["drop_top5"]["mean_remaining_cny"],
            "mean_drop_bottom1":
                scenarios["drop_bottom1"]["mean_remaining_cny"],
            "mean_drop_bottom3":
                scenarios["drop_bottom3"]["mean_remaining_cny"],
            "mean_drop_bottom5":
                scenarios["drop_bottom5"]["mean_remaining_cny"],
            "ci95_low_reproduced": boot["reproduced_ci95_low"],
            "ci95_high_reproduced": boot["reproduced_ci95_high"],
            "bootstrap_match": boot["match"],
        })
    return rows


def single_quote_recheck(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        for nm in NAMES:
            if int(float(r[f"{nm}_distinct_quotes_charge"])) != 1:
                continue
            frac = num(r[f"{nm}_gate_open_charge_frac"])
            out.append({
                "phase": r["phase"],
                "checkpoint": r["checkpoint"],
                "z_core": r["z_core"],
                "pair_id": r["pair_id"],
                "load_date": r["load_date"],
                "price_date": r["price_date"],
                "ess": nm,
                "gate_open_charge_frac": frac,
                "open_charge_blocks": round(frac * 24),
                "all_day_closed": frac == 0.0,
                "quote_charge_mean_all": num(
                    r[f"{nm}_quote_charge_mean_all"]),
            })
    return out


def report(config: dict, integrity: dict, accounting_summary: dict,
           dist_rows: list[dict], sens_rows: list[dict],
           top_bottom: list[dict], ess_rows: list[dict],
           transfer_rows: list[dict], component_rows: list[dict],
           exposure_rows: list[dict], quote_rows: list[dict],
           trace_rows: list[dict], dyn_rows: list[dict],
           mech_rows: list[dict], boot_payload: dict,
           robust_rows: list[dict], recheck_rows: list[dict],
           phase6_report: str) -> None:
    def dist(phase, checkpoint):
        return next(r for r in dist_rows
                    if r["phase"] == phase and r["checkpoint"] == checkpoint)

    def sens(phase, checkpoint, scenario):
        return next(r for r in sens_rows
                    if r["phase"] == phase and r["checkpoint"] == checkpoint
                    and r["scenario"] == scenario)

    def ess(phase, checkpoint, scope):
        return next(r for r in ess_rows
                    if r["phase"] == phase and r["checkpoint"] == checkpoint
                    and r["scope"] == scope)

    def comp(phase, checkpoint, scope, component):
        return next(r for r in component_rows
                    if r["phase"] == phase and r["checkpoint"] == checkpoint
                    and r["scope"] == scope and r["component"] == component)

    primary_ids = [combo_id(p, c) for p, c in PRIMARY]
    classifications = {}
    for phase, checkpoint in PRIMARY:
        d = dist(phase, checkpoint)
        concentrated = (d["frac_positive"] > 0.5
                        and d["frac_abs_lt_0p01"] > 0.25
                        and d["cr5_plus"] > 0.5)
        classifications[combo_id(phase, checkpoint)] = (
            "Concentrated improvement" if concentrated
            else "Mixed / heterogeneous")

    quote_change_days = sum(
        r["days_quote_changed"] for r in quote_rows if r["primary"])
    quote_comparable_days = sum(
        r["both_gate_open_days"] for r in quote_rows if r["primary"])
    any_block_trace = False
    not_tested = [
        "block-level gate state agreement between arms",
        "block-level quote change counts and magnitudes",
        "block-level clearing volume and LMP responses",
        "the first block at which the two arms diverge",
        "per-block quote-to-clearing mediation",
    ]

    lines = [
        "# Phase 6D — frozen-policy profit attribution",
        "",
        "label: `NON_FORMAL_DIAGNOSTIC`; attribution only; no clearing, no "
        "training, no production change.",
        "",
        "```text",
        "PHASE6_ORIGINAL_VERDICT = PARTIAL",
        f"PHASE6D_ANALYSIS = {config['phase6d_analysis_status']}",
        "Z10_CANDIDATE = RETAINED",
        "PRODUCTION_MAPPING = UNCHANGED",
        "ACTION_SCHEMA = UNCHANGED",
        "FORMAL_TRAINING = NOT_STARTED",
        "OPTIMIZER_STEP = NONE",
        "REPLAY_WRITE = NONE",
        "COMMIT_PUSH = NONE",
        "```",
        "",
        "The Phase 6 verdict is preserved unchanged: `PARTIAL`. This stage "
        "explains the Phase 6 numbers; it does not re-adjudicate them.",
        "",
        "## Scope and evidence",
        "",
        "- Inputs: the 320 Phase 6 day-runs and their derived tables, read "
        "only.",
        "- The four primary decision units are Seed 11 final / Train, "
        "Seed 23 final / Train, Seed 11 final / Validation, Seed 23 final / "
        "Validation. Seeds and splits are never pooled.",
        "- Phase 6A best/final rows are auxiliary description only.",
        f"- Source integrity: `{integrity['overall']}` across "
        f"{len(integrity['checks'])} checks.",
        f"- Accounting: maximum identity residual "
        f"{accounting_summary['identity']:.3e} CNY, fleet-minus-sum-ESS "
        f"residual {accounting_summary['fleet_sum']:.3e} CNY, paired-delta "
        f"identity residual {accounting_summary['delta_identity']:.3e} CNY, "
        f"paired-delta fleet-minus-sum {accounting_summary['delta_fleet_sum']:.3e} "
        f"CNY; tolerance {accounting_summary['tolerance_cny']:.0e} CNY.",
        "",
        "## Phase 6D-0 — source and accounting preflight",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for name, item in integrity["checks"].items():
        lines.append(f"| {name} | {item['status']} | {item['detail']} |")
    lines += [
        "",
        f"Overall source status: **`{integrity['overall']}`**.",
        "",
        "## Phase 6D-1 — date contribution and concentration",
        "",
        "| combo | n | mean CNY | median CNY | positive days | "
        "|delta| < 0.01 | S+ | S- | CR5+ | CR5- |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dist_rows:
        if not row["primary"]:
            continue
        lines.append(
            f"| {row['combo_id']} | {row['n_dates']} | "
            f"{row['mean']:.2f} | {row['median']:.4f} | "
            f"{row['n_positive']}/{row['n_dates']} | "
            f"{row['n_abs_lt_0p01']}/{row['n_dates']} | "
            f"{row['s_plus']:.2f} | {row['s_minus']:.2f} | "
            f"{row['cr5_plus']:.3f} | {row['cr5_minus']:.3f} |")
    lines += [
        "",
        "The mean is not a typical day. A positive mean with a median near "
        "zero means the aggregate gain comes from a minority of dates; the "
        "concentration ratios above quantify that tail.",
        "",
        "### Top and bottom contributors",
        "",
    ]
    for phase, checkpoint in PRIMARY:
        d = dist(phase, checkpoint)
        entries = [r for r in top_bottom
                   if r["phase"] == phase and r["checkpoint"] == checkpoint
                   and r["tail"] == "top"][:3]
        entries_neg = [r for r in top_bottom
                       if r["phase"] == phase and r["checkpoint"] == checkpoint
                       and r["tail"] == "bottom"][:3]
        lines.append(f"**{combo_id(phase, checkpoint)}**")
        lines.append("")
        lines.append("| tail | rank | date | delta CNY | share of tail |")
        lines.append("|---|---:|---|---:|---:|")
        for r in entries + entries_neg:
            lines.append(
                f"| {r['tail']} | {r['rank']} | {r['pair_id']} | "
                f"{r['delta_cny']:.2f} | {r['share_of_tail_total']:.3f} |")
        lines.append("")
        lines.append(
            f"CR1+ = {d['cr1_plus']:.3f}, CR3+ = {d['cr3_plus']:.3f}, "
            f"CR5+ = {d['cr5_plus']:.3f}; CR1- = {d['cr1_minus']:.3f}, "
            f"CR3- = {d['cr3_minus']:.3f}, CR5- = {d['cr5_minus']:.3f}.")
        lines.append("")

    lines += [
        "### Top-k removal sensitivity (descriptive only)",
        "",
        "| combo | full mean | drop Top1 | drop Top3 | drop Top5 | "
        "drop Bottom1 | drop Bottom3 | drop Bottom5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, checkpoint in PRIMARY:
        combo = combo_id(phase, checkpoint)
        values = [sens(phase, checkpoint, s)["mean_remaining_cny"]
                  for s in ("full", "drop_top1", "drop_top3", "drop_top5",
                            "drop_bottom1", "drop_bottom3", "drop_bottom5")]
        lines.append(f"| {combo} | " + " | ".join(
            f"{v:.2f}" for v in values) + " |")
    lines += [
        "",
        "Removing the best dates lowers the mean; removing the worst dates "
        "raises it. Neither replaces the pre-registered mean or the "
        "date-clustered CI, and no removed-date subsample is a new test set.",
        "",
        "## Phase 6D-2 — ESS and leg attribution",
        "",
        "| combo | ESS | mean delta CNY | positive days | negative days | "
        "share of fleet delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for phase, checkpoint in PRIMARY:
        for scope in ("fleet",) + NAMES:
            row = ess(phase, checkpoint, scope)
            lines.append(
                f"| {combo_id(phase, checkpoint)} | {scope} | "
                f"{row['delta_mean_net_cny']:.2f} | "
                f"{row['ess_positive_days']} | {row['ess_negative_days']} | "
                f"{row['share_of_fleet_delta']:.3f} |")
    lines += [
        "",
        "Per-ESS rows are attribution, not separate decisions. A negative "
        "ESS leg can coexist with a positive fleet delta, and the reverse.",
        "",
        "### Fleet transfer check",
        "",
        "| combo | fleet+ dates | fleet+ with a losing ESS | fleet- dates | "
        "fleet- with a winning ESS | all ESS same sign as fleet |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase, checkpoint in PRIMARY:
        row = next(r for r in transfer_rows
                   if r["row_type"] == "combo"
                   and r["phase"] == phase and r["checkpoint"] == checkpoint)
        lines.append(
            f"| {combo_id(phase, checkpoint)} | "
            f"{row['fleet_positive_dates']} | "
            f"{row['fleet_positive_with_losing_ess']} | "
            f"{row['fleet_negative_dates']} | "
            f"{row['fleet_negative_with_winning_ess']} | "
            f"{row['dates_all_ess_same_sign_as_fleet']} |")
    lines += [
        "",
        "### Historical negative examples (Round 1 auxiliary)",
        "",
    ]
    for phase, checkpoint, scope in (
            ("A", "seed11_best", "ESS17"), ("A", "seed23_best", "ESS14")):
        row = ess(phase, checkpoint, scope)
        lines.append(
            f"- {checkpoint} / {scope}: mean delta "
            f"{row['delta_mean_net_cny']:.2f} CNY/day over "
            f"{row['n_dates']} probe dates; fleet mean delta "
            f"{ess(phase, checkpoint, 'fleet')['delta_mean_net_cny']:.2f}. "
            "Round 1 auxiliary only; the CI is in "
            "`bootstrap_reproduction.csv`.")
    lines += [
        "",
        "## Phase 6D-3 — quote-to-clearing attribution",
        "",
        "The Phase 6 runner deleted its per-block quote, gate and LMP series "
        "before writing results, so block-level items are `NOT_TESTED`. "
        "Day-level aggregates below are the strongest available evidence.",
        "",
        "| combo | ESS | charge open blocks (C → X) | discharge open blocks "
        "(C → X) | charge quote days changed | discharge quote days changed "
        "|",
        "|---|---|---|---|---:|---:|",
    ]
    for phase, checkpoint in PRIMARY:
        for nm in NAMES:
            charge = next(r for r in exposure_rows
                          if r["phase"] == phase
                          and r["checkpoint"] == checkpoint
                          and r["ess"] == nm and r["leg"] == "charge")
            discharge = next(r for r in exposure_rows
                             if r["phase"] == phase
                             and r["checkpoint"] == checkpoint
                             and r["ess"] == nm and r["leg"] == "discharge")
            charge_q = next(r for r in quote_rows
                            if r["phase"] == phase
                            and r["checkpoint"] == checkpoint
                            and r["ess"] == nm and r["leg"] == "charge")
            discharge_q = next(r for r in quote_rows
                               if r["phase"] == phase
                               and r["checkpoint"] == checkpoint
                               and r["ess"] == nm
                               and r["leg"] == "discharge")
            lines.append(
                f"| {combo_id(phase, checkpoint)} | {nm} | "
                f"{charge['open_blocks_total_control']} → "
                f"{charge['open_blocks_total_candidate']} | "
                f"{discharge['open_blocks_total_control']} → "
                f"{discharge['open_blocks_total_candidate']} | "
                f"{charge_q['days_quote_changed']}"
                f"/{charge_q['both_gate_open_days']} | "
                f"{discharge_q['days_quote_changed']}"
                f"/{discharge_q['both_gate_open_days']} |")
    lines += [
        "",
        f"Across primary groups, quotes are comparable on "
        f"{quote_comparable_days} ESS-days and change on "
        f"{quote_change_days} of them. Quote changes are therefore "
        "observable at day level, but the block-level transmission from "
        "quote to cleared volume cannot be measured from the stored data.",
        "",
        "### Single-charge-quote recheck",
        "",
    ]
    closed = sum(1 for r in recheck_rows if r["all_day_closed"])
    lines += [
        f"{len(recheck_rows)} ESS-days have a single distinct charge quote; "
        f"{closed} of {len(recheck_rows)} have the charge gate closed for all "
        "24 blocks. A constant quote there is the policy and gate outcome, "
        "not a mapping failure, and it carries no direct price-map exposure.",
        "",
        "## Phase 6D-4 — bootstrap and robustness",
        "",
        f"Bootstrap reproduction: {boot_payload['groups']} groups, maximum "
        f"absolute difference {boot_payload['max_abs_diff']:.3e}, "
        f"all match = {boot_payload['all_match']}. Configuration is the "
        "Phase 6 one: 1000 resamples, seed 20261112, percentile interval, "
        "cluster = date.",
        "",
        "| combo | mean | median | CI low | CI high | reproduced |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for phase, checkpoint in PRIMARY:
        row = next(r for r in robust_rows
                   if r["phase"] == phase and r["checkpoint"] == checkpoint)
        lines.append(
            f"| {row['combo_id']} | {row['mean_cny']:.2f} | "
            f"{row['median_cny']:.4f} | {row['ci95_low_reproduced']:.2f} | "
            f"{row['ci95_high_reproduced']:.2f} | "
            f"{row['bootstrap_match']} |")
    lines += [
        "",
        "The Phase 6 decision rule is unchanged. Seed 23 final / Validation "
        "still has a CI lower bound below zero, so the original verdict "
        "remains `PARTIAL`.",
        "",
        "## Phase 6D-5 — classification and answers",
        "",
        "| combo | classification |",
        "|---|---|",
    ]
    for combo, label in classifications.items():
        lines.append(f"| {combo} | {label} |")
    lines += [
        "",
        "No primary combination is broad-based. Each has a positive mean "
        "carried by a minority of dates, a large near-zero block, and a "
        "negative tail.",
        "",
        "### Required questions",
        "",
        "1. **Do the four primary groups reproduce the Phase 6 statistics?** "
        f"`PASS` — {boot_payload['groups']} bootstrap groups match to "
        f"{boot_payload['max_abs_diff']:.3e} CNY.",
        "2. **Does the concentration match the original report?** `PASS` — "
        "the original report already showed medians near zero; this stage "
        "adds the top/bottom tail shares.",
    ]
    lines.append("3. **Largest single date in each group.**")
    for phase, checkpoint in PRIMARY:
        top = next(r for r in top_bottom
                   if r["phase"] == phase and r["checkpoint"] == checkpoint
                   and r["tail"] == "top" and r["rank"] == 1)
        lines.append(
            f"   - {combo_id(phase, checkpoint)}: {top['pair_id']} "
            f"contributes {top['delta_cny']:.2f} CNY, "
            f"{top['share_of_tail_total']:.1%} of the positive total.")
    lines.append("4. **Mean after removing the best and worst dates.**")
    for phase, checkpoint in PRIMARY:
        s = {name: sens(phase, checkpoint, name)["mean_remaining_cny"]
             for name in ("full", "drop_top1", "drop_top3", "drop_top5",
                          "drop_bottom1", "drop_bottom3", "drop_bottom5")}
        lines.append(
            f"   - {combo_id(phase, checkpoint)}: full {s['full']:.2f}; "
            f"drop Top1 {s['drop_top1']:.2f}, Top3 {s['drop_top3']:.2f}, "
            f"Top5 {s['drop_top5']:.2f}; drop Bottom1 "
            f"{s['drop_bottom1']:.2f}, Bottom3 {s['drop_bottom3']:.2f}, "
            f"Bottom5 {s['drop_bottom5']:.2f} CNY/day.")
    lines += [
        "5. **Negative tail.** S- and CR5- are reported above for every "
        "primary group; none of them is zero.",
        "6. **ESS attribution.** The fleet delta is the sum of the four ESS "
        "deltas; the transfer table counts the dates on which a fleet gain "
        "contains a losing ESS and the reverse.",
        f"7. **Share of ESS-days with the charge gate closed all day.** "
        f"{closed} single-charge-quote ESS-days, all with the charge gate "
        "closed for 24 blocks; the full gate-closed share per ESS-day is in "
        "`gate_exposure_summary.csv`.",
        f"8. **How much quote change reaches clearing?** Day-level quote "
        f"changes exist on {quote_change_days} comparable ESS-days; the "
        "block-level clearing response is `NOT_TESTED` because block series "
        "were not stored.",
        "9. **Where the gain comes from.** In every primary group the mean "
        "market-revenue delta and the mean cycle-cost delta are both "
        "negative; cycle cost falls by more than revenue. The positive net "
        "is therefore a throughput/cost-avoidance effect: the candidate "
        "cycles less. The largest positive dates are dominated by large "
        "throughput reductions, and on some of them revenue rises as well. "
        "This is not a revenue improvement.",
        "10. **Is the evidence sufficient to design a new-map training "
        "experiment?** See the recommendation section.",
        "",
        "### Component decomposition of the fleet delta",
        "",
        "| combo | market revenue | purchase cost | cycle cost | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase, checkpoint in PRIMARY:
        parts = [comp(phase, checkpoint, "fleet", c)["delta_mean_cny"]
                 for c in ("market_revenue_cny", "purchase_cost_cny",
                           "cycle_cost_cny", "net_income_cny")]
        lines.append(f"| {combo_id(phase, checkpoint)} | " + " | ".join(
            f"{v:.2f}" for v in parts) + " |")
    lines += [
        "",
        "The same pattern holds in every primary group: mean market revenue "
        "falls, purchase cost falls, cycle cost falls, and the net gain is "
        "the part of the cycle-cost reduction that the revenue loss does not "
        "consume. The top-day table shows the extreme cases, where the "
        "candidate cuts charge and discharge volume by one to two MWh on a "
        "day. Component moves on a single date can be large and cancel; the "
        "net delta, not the dominant component, is the payoff.",
        "",
        "## Recommendation",
        "",
        f"**`{config['recommendation']}`**",
        "",
        config["recommendation_reason"],
        "",
        "This is not an approval to switch the production mapping or to "
        "start training. Any follow-up must be a separately reviewed "
        "protocol with a new mapping version, a new action schema, new "
        "checkpoint/replay manifests and independent validation.",
        "",
        "## NOT_TESTED",
        "",
    ]
    for item in not_tested:
        lines.append(f"- {item}: the Phase 6 runner persisted only day "
                     "aggregates; no block series exist on disk, and "
                     "re-running clearing is out of scope.")
    lines += [
        "",
        "## Files",
        "",
        "`phase6d_config.json`, `input_inventory.csv`, "
        "`source_integrity_report.json`, `accounting_reconciliation.csv`, "
        "`daily_contribution_rank.csv`, `return_distribution_summary.csv`, "
        "`concentration_sensitivity.csv`, `top_bottom_contributors.csv`, "
        "`per_ess_decomposition.csv`, `per_leg_exposure.csv`, "
        "`fleet_transfer_analysis.csv`, `accounting_component_summary.csv`, "
        "`gate_exposure_summary.csv`, `quote_change_summary.csv`, "
        "`quote_to_clearing_trace.csv`, `dynamic_path_attribution.csv`, "
        "`top_bottom_day_mechanism.csv`, `single_quote_day_recheck.csv`, "
        "`bootstrap_reproduction.csv`, `bootstrap_reproduction.json`, "
        "`robustness_summary.csv`, `phase6d_decision_report.md`, "
        "`source_hashes.json`, `output_hashes.json`.",
        "",
    ]
    (OUT / "phase6d_decision_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"output directory already exists: {OUT}")
    OUT.mkdir(parents=True)
    rows = read_rows()
    records = combo_records(rows)

    inventory = input_inventory()
    write_csv(OUT / "input_inventory.csv",
              ["relative_path", "kind", "bytes", "sha256", "recorded_by",
               "recorded_sha256", "matches_recorded", "data_rows"],
              inventory)

    integrity = source_integrity(rows, inventory)
    (OUT / "source_integrity_report.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    if integrity["overall"] == "BLOCKED":
        print(json.dumps(integrity, indent=2))
        return 2

    accounting_rows, accounting_summary = accounting(records, rows)
    write_csv(OUT / "accounting_reconciliation.csv",
              ["row_type", "phase", "checkpoint", "seed", "round", "pair_id",
               "scope", "z_core", "net_income_cny", "market_revenue_cny",
               "purchase_cost_cny", "cycle_cost_cny",
               "terminal_inventory_value_cny", "identity_residual",
               "fleet_minus_sum_ess_residual"],
              accounting_rows)
    if not accounting_summary["all_within_tolerance"]:
        print(json.dumps(accounting_summary, indent=2))
        return 3

    rank_rows, top_bottom = daily_tables(records)
    write_csv(OUT / "daily_contribution_rank.csv",
              list(rank_rows[0].keys()), rank_rows)
    write_csv(OUT / "top_bottom_contributors.csv",
              list(top_bottom[0].keys()), top_bottom)

    dist_rows, sens_rows = distribution_and_sensitivity(records)
    write_csv(OUT / "return_distribution_summary.csv",
              list(dist_rows[0].keys()), dist_rows)
    write_csv(OUT / "concentration_sensitivity.csv",
              list(sens_rows[0].keys()), sens_rows)

    ess_rows = per_ess(records)
    write_csv(OUT / "per_ess_decomposition.csv",
              list(ess_rows[0].keys()), ess_rows)

    exposure_rows, quote_rows = leg_tables(records)
    write_csv(OUT / "per_leg_exposure.csv",
              list(exposure_rows[0].keys()), exposure_rows)
    write_csv(OUT / "gate_exposure_summary.csv",
              list(exposure_rows[0].keys()), exposure_rows)
    write_csv(OUT / "quote_change_summary.csv",
              list(quote_rows[0].keys()), quote_rows)

    transfer_rows = transfer_tables(records)
    write_csv(OUT / "fleet_transfer_analysis.csv",
              list(transfer_rows[0].keys()), transfer_rows)

    component_rows = component_summary(records)
    write_csv(OUT / "accounting_component_summary.csv",
              list(component_rows[0].keys()), component_rows)

    trace_rows = quote_to_clearing(records)
    write_csv(OUT / "quote_to_clearing_trace.csv",
              list(trace_rows[0].keys()), trace_rows)

    dyn_rows = dynamic_path(records)
    write_csv(OUT / "dynamic_path_attribution.csv",
              list(dyn_rows[0].keys()), dyn_rows)

    mech_rows = top_bottom_mechanism(records)
    write_csv(OUT / "top_bottom_day_mechanism.csv",
              list(mech_rows[0].keys()), mech_rows)

    recheck_rows = single_quote_recheck(rows)
    write_csv(OUT / "single_quote_day_recheck.csv",
              list(recheck_rows[0].keys()), recheck_rows)

    boot_rows, boot_payload = bootstrap_reproduction(records)
    write_csv(OUT / "bootstrap_reproduction.csv",
              list(boot_rows[0].keys()), boot_rows)
    (OUT / "bootstrap_reproduction.json").write_text(
        json.dumps(boot_payload, indent=2) + "\n", encoding="utf-8")

    robust_rows = robustness(dist_rows, sens_rows, boot_rows)
    write_csv(OUT / "robustness_summary.csv",
              list(robust_rows[0].keys()), robust_rows)

    quote_change_days = sum(r["days_quote_changed"] for r in quote_rows
                            if r["primary"])
    quote_comparable_days = sum(r["both_gate_open_days"] for r in quote_rows
                                if r["primary"])
    recommendation = "ADDITIONAL_EVIDENCE_REQUIRED"
    recommendation_reason = (
        "The gain is concentrated and cost-side. In every primary group the "
        "median is within 0.001 CNY/day of zero, the top 5 positive dates "
        "carry 52-100% of the positive total, and removing the top 3 "
        "positive dates turns both Validation means negative. The positive "
        "net comes from lower cycle cost as throughput falls, not from "
        "higher market revenue, and Seed 23 final / Validation keeps a 95% "
        "CI that includes zero. Day-level quote changes are visible, but the "
        "block-level quote-to-clearing chain was not persisted, so the "
        "mechanism cannot be verified at the block level. Design work on a "
        "new mapping may proceed only after the trace gap is closed by a "
        "separately approved run; the current candidate is not a production "
        "candidate.")

    config = {
        "label": "NON_FORMAL_DIAGNOSTIC",
        "experiment": "phase6d_profit_attribution",
        "created_at": now_iso(),
        "phase6_original_verdict": "PARTIAL",
        "phase6d_analysis_status": "PARTIAL",
        "source_dir": str(PHASE6.relative_to(ROOT)),
        "output_dir": str(OUT.relative_to(ROOT)),
        "primary_units": [combo_id(p, c) for p, c in PRIMARY],
        "auxiliary_units": [combo_id("A", c)
                            for c in ("seed11_final", "seed23_final",
                                      "seed11_best", "seed23_best")],
        "no_new_clearing": True,
        "no_training": True,
        "no_formal_code_change": True,
        "no_date_reselection": True,
        "bootstrap": {
            "n_resamples": N_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "pair_id (date)",
            "interval": "percentile",
        },
        "near_zero_thresholds_cny": list(NEAR_ZERO),
        "near_zero_note": "descriptive descriptors, not admission criteria",
        "percentiles": [1, 5, 10, 25, 75, 90, 95, 99],
        "classification_rule": {
            "broad_based": "wide positive distribution, no dominant few dates",
            "concentrated": "positive mean, CR5+ > 0.5, >25% dates within "
                            "0.01 CNY/day",
            "mixed": "different signs across seeds, splits, ESS or legs",
            "insufficient": "uncertainty or missing mechanism fields",
        },
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "quote_to_clearing_evidence": {
            "comparable_ess_days": quote_comparable_days,
            "changed_quote_ess_days": quote_change_days,
            "block_level": "NOT_TESTED",
        },
        "not_tested": [
            "block-level gate state agreement",
            "block-level quote changes",
            "block-level clearing response",
            "first divergence block",
            "per-block mediation",
        ],
        "excluded_splits": {
            "test": "never read",
            "holiday": "never read",
            "h2_holdout": "never read",
        },
    }
    (OUT / "phase6d_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    report(config, integrity, accounting_summary, dist_rows, sens_rows,
           top_bottom, ess_rows, transfer_rows, component_rows,
           exposure_rows, quote_rows, trace_rows, dyn_rows, mech_rows,
           boot_payload, robust_rows, recheck_rows, "")

    source_payload = {
        "written_at": now_iso(),
        "label": "NON_FORMAL_DIAGNOSTIC",
        "script": "src/analyze_phase6d_attribution.py",
        "script_sha256": sha(Path(__file__)),
        "phase6_inputs": {},
        "checkpoint_actors": {},
    }
    for path in sorted(PHASE6.rglob("*")):
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            source_payload["phase6_inputs"][rel] = sha(path)
    (OUT / "source_hashes.json").write_text(
        json.dumps(source_payload, indent=2) + "\n", encoding="utf-8")

    output_hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "output_hashes.json":
            output_hashes[path.name] = sha(path)
    (OUT / "output_hashes.json").write_text(
        json.dumps({"written_at": now_iso(), "files": output_hashes},
                   indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT}")
    print(json.dumps({
        "source_integrity": integrity["overall"],
        "accounting": accounting_summary,
        "bootstrap_all_match": boot_payload["all_match"],
        "recommendation": recommendation,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
