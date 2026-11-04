"""Exposure ledger and prospective-holdout candidate construction.

Reads only frozen artifacts. It does not train, evaluate a policy, or write to
any existing pairing, split or result file.

Two candidate classes are evaluated:

- H1: a joint scenario holdout whose load date and price date are both outside
  the historical/new-algorithm training and validation exposure.
- H2: a price-date holdout built from complete 2025 price dates that never
  entered FULL96, paired with existing load scenarios. The load side is
  historical and is disclosed per pair.

Raw data-quality inspection is recorded separately from policy-result
exposure. A date seen only by the quality audit is not a leaked date.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path.cwd()
OUT = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"

PAIRING = OUT / "load_price_pairing.csv"
NEAR_PAIRING = OUT / "near/load_price_pairing.csv"
PRICE_COVERAGE = OUT / "price_daily_coverage.csv"
PRICE_FILE = ROOT / "docs/data.csv"
BUS_LOADS = OUT / "bus_loads_96x33.npz"
PRICE_CURVES = OUT / "price_curves.npz"
COMMON_SPLIT = OUT / "common_split.json"
FULL_SPLIT = OUT / "split.json"
NEAR_SPLIT = OUT / "near/split.json"
PILOT_SUMMARY = OUT / "pilot/summary.json"
PILOT_EVAL = OUT / "pilot/eval.csv"
BENCHMARK = REPORTS / "results.csv"
B1_RESULTS = OUT / "b1_terminal_neutral_results.csv"
V2_CANDIDATES = ROOT / "data/processed/v2_seasonal_candidates.csv"
MULTIDAY_MANIFEST = ROOT / "reports/multiday_dataset_manifest.csv"

LEDGER = OUT / "holdout_exposure_ledger.csv"
PAIRS = OUT / "holdout_pairs.csv"
MANIFEST = OUT / "holdout_manifest.json"
STATUS = OUT / "holdout_status.json"
AUDIT = REPORTS / "holdout_exposure_audit.md"

SMOKE_PAIR = ("2022-07-22", "2025-07-15")
LEGACY_SPLITS = ("test", "holiday_block")
H2_SEASONS = ("summer", "autumn", "winter")


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_id(load_date: str, price_date: str) -> str:
    return f"{load_date}|{price_date}"


def price_season(date: str) -> str:
    month = int(date[5:7])
    if 6 <= month <= 8:
        return "summer"
    if 9 <= month <= 11:
        return "autumn"
    if month == 12:
        return "winter"
    return "spring"


def is_weekend(date: str) -> bool:
    return dt.date.fromisoformat(date).weekday() >= 5


def month_day_gap(load_date: str, price_date: str) -> int:
    load = dt.date.fromisoformat(load_date)
    mapped = dt.date(2025, load.month, load.day)
    return abs((mapped - dt.date.fromisoformat(price_date)).days)


def split_label(pair: str, mapping: dict) -> str:
    for label in ("train", "validation", "test", "holiday_block"):
        if pair in set(mapping.get(label, [])):
            return label
    return ""


def split_label_load(load_date: str, mapping: dict) -> str:
    for label in ("train", "validation", "test", "holiday_block"):
        if load_date in set(mapping.get(label, [])):
            return label
    return ""


def build_load_labels(common: dict):
    labels = {}
    for label in ("train", "validation", "test", "holiday_block"):
        for pair in common["common"].get(label, []):
            labels[pair] = label
    for pair in common.get("full96_only", []):
        labels.setdefault(pair, "train_full96_only")
    return labels


def main() -> int:
    full_pair_rows = read_csv(PAIRING)
    near_pair_rows = read_csv(NEAR_PAIRING)
    full_pairs = {pair_id(r["load_date"], r["price_date"]): r
                  for r in full_pair_rows}
    near_pairs = {pair_id(r["load_date"], r["price_date"]): r
                  for r in near_pair_rows}
    used_loads = {r["load_date"] for r in full_pair_rows}
    used_prices = {r["price_date"] for r in full_pair_rows}

    common = json.loads(COMMON_SPLIT.read_text(encoding="utf-8"))
    full_split = json.loads(FULL_SPLIT.read_text(encoding="utf-8"))
    near_split = json.loads(NEAR_SPLIT.read_text(encoding="utf-8"))
    historical = build_load_labels(common)
    load_to_pair = {r["load_date"]: pair_id(r["load_date"], r["price_date"])
                    for r in full_pair_rows}
    price_to_pair = {r["price_date"]: pair_id(r["load_date"], r["price_date"])
                     for r in full_pair_rows}

    pilot = json.loads(PILOT_SUMMARY.read_text(encoding="utf-8"))
    visited = Counter()
    for run in pilot:
        for date, count in run["exposure"].items():
            if int(count) > 0:
                visited[date] += 1
    eval_rows = read_csv(PILOT_EVAL)
    eval_loads = {r["load_date"] for r in eval_rows}
    eval_prices = {r["price_date"] for r in eval_rows}
    bench_rows = read_csv(BENCHMARK)
    bench_loads = {r["load_date"] for r in bench_rows}
    bench_prices = {r["price_date"] for r in bench_rows}
    b1_rows = read_csv(B1_RESULTS)
    b1_loads = {r["load_date"] for r in b1_rows}
    b1_prices = {r["price_date"] for r in b1_rows}
    legacy_pairs = set(common["common"].get("test", [])) \
        | set(common["common"].get("holiday_block", []))
    smoke_id = pair_id(*SMOKE_PAIR)

    multiday = {r["date"]: r for r in read_csv(MULTIDAY_MANIFEST)}
    candidates = {r["date"]: r for r in read_csv(V2_CANDIDATES)}
    price_cov = {r["date"]: r for r in read_csv(PRICE_COVERAGE)}

    # ------------------------------------------------------------------
    # Load-side exposure ledger
    # ------------------------------------------------------------------
    ledger = []
    for date in sorted(multiday):
        row = multiday[date]
        v2 = candidates.get(date)
        pair = load_to_pair.get(date)
        hist = historical.get(pair, "") if pair else ""
        new_split = split_label_load(date, near_split) if pair else ""
        proposed = split_label_load(date, full_split) if pair else ""
        labels = {hist, new_split, proposed}
        in_train = bool(labels & {"train", "train_full96_only"})
        in_val = "validation" in labels
        in_test = bool(labels & set(LEGACY_SPLITS))
        hist_train = hist in ("train", "train_full96_only")
        hist_val = hist == "validation"
        hist_test = hist in LEGACY_SPLITS
        reserved = (new_split in ("validation", "test", "holiday_block")
                    or proposed in ("validation", "test", "holiday_block"))
        legacy = pair in legacy_pairs if pair else False
        if pair:
            reason = ("load_in_full96; benchmark and B1 truthful performance "
                      "computed")
        elif v2 is not None:
            reason = ("V2 load candidate not used, but its season has no "
                      "2025 price date (spring/spring_festival)")
        elif str(row.get("load_eligible")).lower() == "true":
            reason = ("complete 17-user load but not in the V2 33-bus load "
                      "library; unused days are autumn/winter only")
        else:
            reason = "not load-eligible"
        ledger.append({
            "date_kind": "load",
            "date": date,
            "season": (v2["season"] if v2 else row.get("season", "")),
            "weekday": row.get("weekday", ""),
            "is_weekend": row.get("is_weekend", ""),
            "complete": bool(
                str(row.get("load_eligible")).lower() == "true" or v2),
            "raw_viewed": True,
            "in_v2_candidates": v2 is not None,
            "in_full96": pair is not None,
            "in_near": pair in near_pairs if pair else False,
            "historical_split": hist,
            "new_algorithm_split": new_split,
            "proposed_full96_split": proposed,
            "in_training_pool": in_train,
            "in_validation": in_val,
            "in_test_or_holiday": in_test,
            "historical_training": hist_train,
            "historical_validation": hist_val,
            "historical_test_or_holiday": hist_test,
            "reserved_for_selection": reserved,
            "pilot_training_visited": visited.get(date, 0) > 0,
            "pilot_training_runs": visited.get(date, 0),
            "pilot_eval": date in eval_loads,
            "benchmark_performance": date in bench_loads,
            "b1_truthful_computed": date in b1_loads,
            "used_for_selection": hist_val or hist_test,
            "legacy_seen_test": legacy,
            "new_algorithm_smoke": pair == smoke_id if pair else False,
            "holdout_h1_eligible": False,
            "holdout_h2_eligible": False,
            "holdout_h2_load_usable": bool(
                v2 is not None and v2["group"] in H2_SEASONS
                and date != SMOKE_PAIR[0]),
            "exclusion_reason": reason,
            "notes": (
                "raw_viewed is the data-quality audit, not policy exposure"
                if not pair else ""),
        })

    # ------------------------------------------------------------------
    # Price-side exposure ledger
    # ------------------------------------------------------------------
    unpaired_prices = sorted(
        date for date, row in price_cov.items()
        if str(row.get("complete")).lower() == "true" and date not in used_prices)
    unpaired_set = set(unpaired_prices)
    for date in sorted(price_cov):
        row = price_cov[date]
        pair = price_to_pair.get(date)
        load = pair.split("|")[0] if pair else ""
        hist = historical.get(pair, "") if pair else ""
        new_split = split_label_load(load, near_split) if pair else ""
        proposed = split_label_load(load, full_split) if pair else ""
        labels = {hist, new_split, proposed}
        in_train = bool(labels & {"train", "train_full96_only"})
        in_val = "validation" in labels
        in_test = bool(labels & set(LEGACY_SPLITS))
        hist_train = hist in ("train", "train_full96_only")
        hist_val = hist == "validation"
        hist_test = hist in LEGACY_SPLITS
        reserved = (new_split in ("validation", "test", "holiday_block")
                    or proposed in ("validation", "test", "holiday_block"))
        legacy = pair in legacy_pairs if pair else False
        complete = str(row.get("complete")).lower() == "true"
        if pair:
            reason = ("price in FULL96; benchmark and B1 truthful performance "
                      "computed")
        elif complete:
            reason = "unpaired complete price date; H2 eligible"
        else:
            reason = "incomplete 96-point day (2025-06-01)"
        ledger.append({
            "date_kind": "price",
            "date": date,
            "season": (price_season(date) if complete else ""),
            "weekday": row.get("weekday", ""),
            "is_weekend": row.get("is_weekend", ""),
            "complete": complete,
            "raw_viewed": True,
            "in_v2_candidates": False,
            "in_full96": pair is not None,
            "in_near": pair in near_pairs if pair else False,
            "historical_split": hist,
            "new_algorithm_split": new_split,
            "proposed_full96_split": proposed,
            "in_training_pool": in_train,
            "in_validation": in_val,
            "in_test_or_holiday": in_test,
            "historical_training": hist_train,
            "historical_validation": hist_val,
            "historical_test_or_holiday": hist_test,
            "reserved_for_selection": reserved,
            "pilot_training_visited": visited.get(load, 0) > 0 if load else False,
            "pilot_training_runs": visited.get(load, 0) if load else 0,
            "pilot_eval": date in eval_prices,
            "benchmark_performance": date in bench_prices,
            "b1_truthful_computed": date in b1_prices,
            "used_for_selection": hist_val or hist_test,
            "legacy_seen_test": legacy,
            "new_algorithm_smoke": pair == smoke_id if pair else False,
            "holdout_h1_eligible": False,
            "holdout_h2_eligible": date in unpaired_set,
            "holdout_h2_load_usable": False,
            "exclusion_reason": reason,
            "notes": (
                "raw_viewed is the data-quality audit, not policy exposure"
                if not pair else ""),
        })

    ledger.sort(key=lambda r: (r["date_kind"], r["date"]))
    with LEDGER.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)

    # ------------------------------------------------------------------
    # H2: price-date holdout with historical load scenarios
    # ------------------------------------------------------------------
    legacy_loads = {pair.split("|")[0] for pair in legacy_pairs}

    def load_tier(date: str) -> int:
        # Lower is less exposed. Benchmark/B1 are performance on every load,
        # so this tier ranks *policy-result* exposure, not raw-data exposure.
        if date in legacy_loads:
            return 3
        if date in eval_loads:
            return 2
        if visited.get(date, 0) > 0:
            return 1
        return 0

    # Only the ordinary seasonal groups. Spring-festival and national-day
    # loads are holiday blocks; their calendar season is not a matching basis.
    load_pool = [
        r for r in candidates.values()
        if r["group"] in H2_SEASONS and r["date"] != SMOKE_PAIR[0]
    ]
    assigned = set()
    h2_rows = []
    for price in unpaired_prices:
        season = price_season(price)
        candidates_season = [r for r in load_pool if r["season"] == season]
        same_type = [r for r in candidates_season
                     if is_weekend(r["date"]) == is_weekend(price)]
        fresh_same = [r for r in same_type if r["date"] not in assigned]
        fresh_any = [r for r in candidates_season
                     if r["date"] not in assigned]
        if fresh_same:
            choose, pairing_type = fresh_same, "same_season_same_day_type"
        elif fresh_any:
            choose, pairing_type = fresh_any, \
                "same_season_day_type_mismatch"
        elif same_type:
            choose, pairing_type = same_type, \
                "same_season_same_day_type_reused"
        else:
            choose, pairing_type = candidates_season, \
                "same_season_day_type_mismatch_reused"
        best = min(choose, key=lambda r: (load_tier(r["date"]),
                                          month_day_gap(r["date"], price),
                                          r["date"]))
        assigned.add(best["date"])
        load = best["date"]
        pid = pair_id(load, price)
        hist_load = historical.get(load_to_pair.get(load), "")
        new_load = split_label_load(load, near_split)
        prop_load = split_label_load(load, full_split)
        hist_selection = hist_load == "validation" or hist_load in LEGACY_SPLITS
        reserved_selection = (
            new_load in ("validation", "test", "holiday_block")
            or prop_load in ("validation", "test", "holiday_block"))
        h2_rows.append({
            "pair_id": pid,
            "load_date": load,
            "price_date": price,
            "season": season,
            "load_day_type": "weekend" if is_weekend(load) else "weekday",
            "price_day_type": "weekend" if is_weekend(price) else "weekday",
            "pairing_type": pairing_type,
            "month_day_gap_days": month_day_gap(load, price),
            "load_historical_split": hist_load,
            "load_new_algorithm_split": new_load,
            "load_proposed_full96_split": prop_load,
            "load_used_for_selection_historically": hist_selection,
            "load_reserved_for_selection": reserved_selection,
            "load_pilot_training_runs": visited.get(load, 0),
            "load_pilot_eval": load in eval_loads,
            "load_benchmark_performance": load in bench_loads,
            "load_b1_truthful_computed": load in b1_loads,
            "load_legacy_seen_test": load in legacy_loads,
            "load_new_algorithm_smoke": load == SMOKE_PAIR[0],
            "load_reused_in_h2": 0,
            "price_historical_exposure": "none (unpaired price date)",
            "h_class": "H2_PRICE_DATE_ONLY",
            "notes": (
                "load side is historical; this set supports price-date "
                "adaptation only, not joint source-load generalisation"),
        })
    counts = Counter(r["load_date"] for r in h2_rows)
    for row in h2_rows:
        row["load_reused_in_h2"] = counts[row["load_date"]] - 1
    h2_rows.sort(key=lambda r: (r["season"], r["price_date"]))
    with PAIRS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(h2_rows[0]))
        writer.writeheader()
        writer.writerows(h2_rows)

    # ------------------------------------------------------------------
    # H1 analysis
    # ------------------------------------------------------------------
    unused_v2 = [r for r in candidates.values() if r["date"] not in used_loads]
    unused_v2_groups = Counter(r["group"] for r in unused_v2)
    complete_17 = [r for r in multiday.values()
                   if str(r.get("load_eligible")).lower() == "true"
                   and r["date"] not in used_loads
                   and r["date"] not in candidates]
    complete_17_seasons = Counter(r.get("season", "") for r in complete_17)
    h1_days = 0
    h1_seasons: List[str] = []
    # Upper bound even if the frozen-library and holiday constraints were
    # relaxed: unused ordinary V2 loads cover no summer/autumn; the seven
    # spring-festival loads are winter holidays; the 28 non-V2 17-user days
    # add 20 autumn and 8 winter. Against 10 autumn and 8 winter unpaired
    # prices, that is at most 18 pairs over two seasons.
    max_joint_pairs = min(10, 20) + min(8, 7 + 8)
    h1_reasons = {
        "unused_v2_load_dates": len(unused_v2),
        "unused_v2_groups": dict(unused_v2_groups),
        "complete_17user_not_in_v2": len(complete_17),
        "complete_17user_seasons": dict(complete_17_seasons),
        "no_2025_spring_price": True,
        "all_v2_ordinary_summer_autumn_winter_loads_used": True,
        "maximum_joint_pairs_even_if_relaxed": max_joint_pairs,
        "maximum_joint_seasons_even_if_relaxed": ["autumn", "winter"],
        "note": ("No load date is both unused in the ordinary V2 library and "
                 "paired with an unused same-season 2025 price date. The 27 "
                 "unused V2 loads are 20 spring and 7 spring-festival; the "
                 "2025 source has no spring price. The 28 complete 17-user "
                 "autumn/winter days are not in the V2 load library. Even "
                 "admitting both would give at most 18 pairs over two "
                 "seasons."),
    }

    h2_seasons = sorted({r["season"] for r in h2_rows})
    legacy_list = sorted(legacy_pairs)
    status_payload = {
        "as_of": "2026-11-02",
        "dataset": "V2-CY-JN-2025",
        "h1_joint_holdout": {
            "days": h1_days,
            "seasons": h1_seasons,
            "meets_30_days_and_3_seasons": False,
            "reason": h1_reasons,
        },
        "h2_price_date_holdout": {
            "days": len(h2_rows),
            "seasons": h2_seasons,
            "meets_30_days_and_3_seasons": bool(
                len(h2_rows) >= 30 and len(h2_seasons) >= 3),
            "load_side": "historical; disclosed per pair",
            "claim_supported": "price-date adaptation only",
            "claim_not_supported": "joint source-load generalisation",
        },
        "legacy_seen_test": {
            "label": "LEGACY_SEEN_TEST",
            "pairs": legacy_list,
            "count": len(legacy_list),
        },
        "new_algorithm_exposure": {
            "smoke_pairs": [smoke_id],
            "note": ("one engineering smoke day used 17 gradient updates; "
                     "excluded from the H2 load pool"),
        },
        "formal_holdout_ready": False,
        "allowed_claim": ("limited price-date adaptation on 40 unpaired 2025 "
                          "price dates with historical load scenarios"),
        "not_allowed_claim": ("a 30-day joint source-load holdout or any "
                              "independent joint generalisation claim"),
    }
    STATUS.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    inputs = {
        "price_file": PRICE_FILE,
        "price_coverage": PRICE_COVERAGE,
        "full96_pairing": PAIRING,
        "near_pairing": NEAR_PAIRING,
        "bus_loads": BUS_LOADS,
        "price_curves": PRICE_CURVES,
        "common_split": COMMON_SPLIT,
        "full96_split": FULL_SPLIT,
        "near_split": NEAR_SPLIT,
        "pilot_summary": PILOT_SUMMARY,
        "pilot_eval": PILOT_EVAL,
        "benchmark": BENCHMARK,
        "b1_terminal_neutral": B1_RESULTS,
        "v2_candidates": V2_CANDIDATES,
        "multiday_manifest": MULTIDAY_MANIFEST,
    }
    manifest = {
        "dataset": "V2-CY-JN-2025",
        "as_of": "2026-11-02",
        "kind": "holdout exposure audit and candidate construction",
        "no_training_or_policy_evaluation": True,
        "pairing_rule": (
            "same season; same weekday/weekend type when a distinct load is "
            "available; otherwise a distinct load of the other type, recorded "
            "as a mismatch; only then reuse a same-type load. Tie-break by "
            "lower historical exposure tier, then nearest month-day, then "
            "load date. All 40 unpaired price dates are used; no price level, "
            "strategy result or expected payoff is used."),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "counts": {
            "load_dates_in_raw_audit": len(multiday),
            "load_eligible_17user": sum(
                1 for r in multiday.values()
                if str(r.get("load_eligible")).lower() == "true"),
            "v2_load_candidates": len(candidates),
            "full96_load_dates": len(used_loads),
            "near_load_dates": len({r["load_date"] for r in near_pair_rows}),
            "price_dates_in_source": len(price_cov),
            "complete_price_dates": sum(
                1 for r in price_cov.values()
                if str(r.get("complete")).lower() == "true"),
            "full96_price_dates": len(used_prices),
            "unpaired_complete_price_dates": len(unpaired_prices),
            "unpaired_price_seasons": dict(Counter(
                price_season(p) for p in unpaired_prices)),
        },
        "h1": status_payload["h1_joint_holdout"],
        "h1_pair_ids": [],
        "h2": status_payload["h2_price_date_holdout"],
        "h2_pair_ids": [r["pair_id"] for r in h2_rows],
        "legacy_seen_test": status_payload["legacy_seen_test"],
        "new_algorithm_exposure": status_payload["new_algorithm_exposure"],
        "outputs": {
            "ledger": {"path": str(LEDGER), "sha256": sha256(LEDGER)},
            "h2_pairs": {"path": str(PAIRS), "sha256": sha256(PAIRS)},
            "status": {"path": str(STATUS), "sha256": sha256(STATUS)},
        },
        "conclusion": (
            "H1 is not available from the current data. H2 has 40 days over "
            "three seasons and supports limited price-date adaptation only."),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _write_audit(ledger, h2_rows, manifest, status_payload)
    print(json.dumps({
        "h1_days": h1_days,
        "h2_days": len(h2_rows),
        "h2_seasons": h2_seasons,
        "legacy_seen_test": len(legacy_list),
        "unpaired_prices": len(unpaired_prices),
        "unused_v2_loads": len(unused_v2),
        "complete_17user_not_in_v2": len(complete_17),
    }, indent=2))
    return 0


def _write_audit(ledger, h2_rows, manifest, status) -> None:
    loads = [r for r in ledger if r["date_kind"] == "load"]
    prices = [r for r in ledger if r["date_kind"] == "price"]
    h1 = status["h1_joint_holdout"]
    h2 = status["h2_price_date_holdout"]
    type_counts = Counter(r["pairing_type"] for r in h2_rows)
    season_counts = Counter(r["season"] for r in h2_rows)
    load_reuse = len(h2_rows) - len({r["load_date"] for r in h2_rows})
    legacy_load_pairs = sum(1 for r in h2_rows
                            if r["load_legacy_seen_test"])
    pilot_train_pairs = sum(1 for r in h2_rows
                            if int(r["load_pilot_training_runs"]) > 0)
    pilot_eval_pairs = sum(1 for r in h2_rows if r["load_pilot_eval"])
    reserved_pairs = sum(1 for r in h2_rows
                         if r["load_reserved_for_selection"])
    historical_selection_pairs = sum(
        1 for r in h2_rows if r["load_used_for_selection_historically"])
    lines = [
        "# Holdout exposure audit (V2-CY-JN-2025)",
        "",
        "No training and no policy evaluation. This separates raw-data "
        "quality inspection from policy-result exposure and asks whether the "
        "existing 2022 load records and 2025 Jiangnan prices can support a "
        "prospective holdout.",
        "",
        "## Definitions",
        "",
        "- **raw_viewed**: the date was read by the data-quality audit. It is "
        "not, by itself, leakage.",
        "- **historical exposure**: the date entered the pilot training pool, "
        "was actually visited, was used in historical validation or model "
        "selection, or had rule-baseline/truthful performance computed.",
        "- **used_for_selection**: historical validation/test/holiday "
        "membership (the pilot used these to compare designs). "
        "**reserved_for_selection** is separate: it marks the frozen NEAR "
        "split or the proposed FULL96 split, which are designated for future "
        "selection but have not been used yet.",
        "- **H1 joint holdout**: load and price both outside the historical "
        "and new-algorithm training/validation exposure, paired in the same "
        "season, with the pair unused before.",
        "- **H2 price-date holdout**: a complete 2025 price date never in "
        "FULL96, paired with an existing load scenario. The load side is "
        "historical and is disclosed per pair.",
        "",
        "## Date inventory",
        "",
        "| item | value |",
        "|---|---:|",
        f"| raw load dates audited | {len(loads)} |",
        f"| 17-user complete load days | "
        f"{manifest['counts']['load_eligible_17user']} |",
        f"| V2 load candidates | {manifest['counts']['v2_load_candidates']} |",
        f"| FULL96 load dates | {manifest['counts']['full96_load_dates']} |",
        f"| NEAR load dates | {manifest['counts']['near_load_dates']} |",
        f"| 2025 price dates in source | {len(prices)} |",
        f"| complete 2025 price dates | "
        f"{manifest['counts']['complete_price_dates']} |",
        f"| FULL96 price dates | {manifest['counts']['full96_price_dates']} |",
        f"| unpaired complete price dates | "
        f"{manifest['counts']['unpaired_complete_price_dates']} |",
        "",
        "The unpaired price dates are "
        f"{manifest['counts']['unpaired_price_seasons']}.",
        "",
        "## Legacy seen test",
        "",
        f"The original common test (11) and national-day block (5) remain "
        f"**LEGACY_SEEN_TEST** ({status['legacy_seen_test']['count']} pairs). "
        "They are not renamed as a blind set.",
        "",
        "## H1 — joint scenario holdout",
        "",
        f"**Days: {h1['days']}.** Seasons: {h1['seasons'] or 'none'}. "
        f"Meets 30 days and 3 seasons: **{h1['meets_30_days_and_3_seasons']}**.",
        "",
        f"- Unused V2 load candidates: {h1['reason']['unused_v2_load_dates']} "
        f"({h1['reason']['unused_v2_groups']}). They are the spring group and "
        "the spring-festival block, not ordinary seasonal days.",
        "- The 2025 price source covers June–December only; it has no spring "
        "price date, so the 20 spring loads cannot be paired.",
        f"- Complete 17-user load days outside the V2 library: "
        f"{h1['reason']['complete_17user_not_in_v2']} "
        f"({h1['reason']['complete_17user_seasons']}). They are autumn/winter "
        "only and use a different user subset; admitting them would change the "
        "load system.",
        "- All ordinary V2 summer, autumn and winter load dates are used.",
        f"- Even if the seven spring-festival winter loads and the 28 non-V2 "
        f"autumn/winter loads were admitted, the upper bound is "
        f"**{h1['reason']['maximum_joint_pairs_even_if_relaxed']} pairs over "
        f"two seasons**, still below 30 days and three seasons. The frozen "
        "library gives **0**.",
        "",
        "H1 is therefore **not available** from the current data. No date is "
        "promoted to H1.",
        "",
        "## H2 — price-date holdout",
        "",
        f"**Days: {len(h2_rows)}.** Seasons: {sorted(season_counts)} "
        f"({dict(season_counts)}). Meets 30 days and 3 seasons: "
        f"**{h2['meets_30_days_and_3_seasons']}**.",
        "",
        "| item | value |",
        "|---|---:|",
        f"| distinct load scenarios | "
        f"{len({r['load_date'] for r in h2_rows})} |",
        f"| load reuses | {load_reuse} |",
        f"| same-season same-type pairs | "
        f"{type_counts.get('same_season_same_day_type', 0)} |",
        f"| same-season type-mismatch pairs | "
        f"{type_counts.get('same_season_day_type_mismatch', 0)} |",
        f"| same-type reused pairs | "
        f"{type_counts.get('same_season_same_day_type_reused', 0)} |",
        f"| pairs whose load was visited in pilot training | "
        f"{pilot_train_pairs} |",
        f"| pairs whose load was in pilot evaluation | {pilot_eval_pairs} |",
        f"| pairs whose load is LEGACY_SEEN_TEST | {legacy_load_pairs} |",
        f"| pairs whose load was used for historical selection | "
        f"{historical_selection_pairs} |",
        f"| pairs whose load is reserved for future selection | "
        f"{reserved_pairs} |",
        "",
        "Every price date in H2 is a complete 2025 date that never entered "
        "FULL96, NEAR, the pilot evaluation or the rule benchmark. The load "
        "side is historical; the per-pair columns in `holdout_pairs.csv` "
        "record exactly which exposure applies.",
        "",
        "## New-algorithm exposure",
        "",
        f"The engineering smoke used one pair, `{SMOKE_PAIR[0]}|"
        f"{SMOKE_PAIR[1]}`, for 17 gradient updates. Its load is excluded "
        "from the H2 load pool. No formal training has run.",
        "",
        "## Verdict",
        "",
        "- H1: **0 days** — the current data cannot support a 30-day, "
        "three-season joint source-load holdout.",
        "- H2: **40 days, 3 seasons** — supports limited price-date "
        "adaptation only.",
        "- A claim of joint source-load generalisation is **not** supported.",
        "- The legacy common test and holiday block remain "
        "`LEGACY_SEEN_TEST`.",
        "",
        "Artifacts: `holdout_exposure_ledger.csv`, `holdout_pairs.csv`, "
        "`holdout_manifest.json`, `holdout_status.json`.",
        "",
    ]
    AUDIT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
