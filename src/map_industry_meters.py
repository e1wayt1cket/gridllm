# map_industry_meters.py
"""Candidate bus assignment for the qualified meters, and what the data covers.

Rule B gives 84 meters with blocks in the shape pool. This module turns each of
those meters into one IEEE33 load bus, states the rule that put it there, and
audits what each (meter, day) cell actually holds.

**Every assignment here is a candidate.** Nothing is frozen, and the script says
which rule produced which row so two candidates can be compared rather than
argued about in the abstract.

The mapping is a simulation mapping: this feeder carries no user-to-node
connection table, and no row here is a claim about where Jiangsu industry is
physically connected. What the rows must be is *checkable* -- one meter to one
bus, that bus a load bus, and that bus assigned the meter's own industry by
`config/scenarios.yaml`.

Why the existing shape mapping cannot be reused unchanged
--------------------------------------------------------

`industry_map.buses` assigns one industry to each load bus, and twenty of those
buses take 商业综合体. That is safe for *shapes*, because a shape is
per-meter-normalized and carries no magnitude. It is not safe for *absolute*
load: putting every 商业综合体 meter on all twenty of those buses would count
each meter's energy twenty times. So the absolute route needs a partition --
each meter on exactly one bus -- and that partition is what this script builds.

Two candidate rules, and what separates them
--------------------------------------------

* `R1_industry_bus` -- a meter takes its industry's bus. Available only where
  the industry is assigned exactly one bus, which is true of the twelve
  industrial classes and false of 商业综合体.
* `R2_user_block` -- meters are grouped by the user that owns them, users are
  ordered by name within their industry, and user *i* takes the *i*-th of that
  industry's buses, wrapping if there are more users than buses. A user's meters
  stay together, so a participant is not split across nodes.
* `R3_meter_spread` -- meters are ordered by id within their industry and
  distributed one per bus. A user with several meters can land on several buses,
  which trades the participant-stays-whole property for using more of the
  feeder: 商业综合体 has 5 users but 20 declared buses, so `R2` would leave 15 of
  those buses carrying nothing at all.

None is chosen here. The three differ in where load concentrates, which changes
the power flow, so the choice is not cosmetic and is not made by this script.

Usage:
    PYTHONPATH=src python src/map_industry_meters.py --out data/processed
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import industry_load as il
import config_loader as cfg

SCENARIO = "typical_day_industry"

#: The rule names written into the candidate table.
R1 = "R1_industry_bus"
R2 = "R2_user_block"
R3 = "R3_meter_spread"

#: The frozen case.
#:
#: `FROZEN_RULE` is the mapping rule the S4.3 ruling adopts. `UNSUPPORTED_BUSES`
#: are the load buses the formal absolute-load case maps no load to: each is the
#: sole bus of an industry whose meters cover too few days to populate it (印刷业
#: 705 days missing, 数据中心 698, 水泥制造 648, of 730). The buses stay in the
#: IEEE33 topology -- the network is untouched -- they simply carry no mapped
#: load, so the case has 29 data-supported load buses rather than 32.
FROZEN_RULE = R3
UNSUPPORTED_BUSES: Tuple[int, ...] = (22, 29, 32)

#: Workbook dates whose readings are the synthesised filler row throughout.
#: Every one of the workbook's 43 `placeholder` records falls on this date, so
#: it is not a measurement and is outside the formal day axis.
FABRICATED_DATES: Tuple[str, ...] = ("2022-05-26",)

#: The period count a complete meter-day must carry.
N_SLOTS = il.N_SLOTS

#: Statuses a (meter, day) cell can take. Ordered by precedence: the first one
#: that applies is the cell's status, and the report never lets a later, milder
#: condition mask an earlier, harder one.
STATUS_PRECEDENCE = (
    "conflicting",      # two records disagree about the same meter-day
    "empty_identity",   # nothing to key the record on
    "other_phase",      # not active power; the unit convention is unset
    "all_zero",         # every slot exactly zero
    "placeholder",      # the workbook's synthesised filler row
    "partial",          # fewer than 96 slots present
    "complete",         # 96 slots present, nothing disqualifying
    "absent",           # no record at all for this meter on this day
)


def load_industry_map() -> Tuple[Dict[str, List[int]], List[str]]:
    """The declared industry -> buses assignment, inverted and sorted.

    The scenario states bus -> industry because that is what a shape needs.
    Meter assignment needs the other direction, so it is inverted here rather
    than duplicated in a second hand-maintained table.
    """
    industry_map = cfg.get_scenario_cfg(SCENARIO).get("industry_map") or {}
    if not industry_map.get("simulation_mapping"):
        raise ValueError(
            f"{SCENARIO}'s industry_map is not marked `simulation_mapping: "
            "true`; this script only builds simulation mappings")
    by_bus = {int(b): str(i) for b, i in (industry_map.get("buses") or {}).items()}
    industry_buses: Dict[str, List[int]] = collections.defaultdict(list)
    for bus, industry in sorted(by_bus.items()):
        industry_buses[industry].append(bus)
    return dict(industry_buses), [str(i) for i in
                                  (industry_map.get("excluded_industries") or [])]


def qualified_meter_facts(records: Sequence[il.MeterDay]
                          ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """One entry per meter that actually contributes to the dataset.

    Two things are deliberately not the same: a meter can pass the
    ratio/capacity gate (87 of them do) and still contribute nothing, because
    every one of its blocks is dropped by the shape pool. Rule B is about the
    intersection, so only the 84 that contribute are returned as the working
    set; the rest come back separately so the report can name them instead of
    letting them vanish.

    `users` is a set because a meter appearing under two accounts is a finding,
    not an assumption -- the caller reports it rather than picking one.
    """
    verdicts = il.qualified_meters(records)
    # Computed once: the membership test below runs for every record, and
    # rebuilding the set per record would make this quadratic.
    in_pool = il.qualified_block_keys(records)
    facts: Dict[str, dict] = {}
    empty: Dict[str, dict] = {}
    for md in records:
        if not verdicts.get(md.meter_key, {}).get("qualified"):
            continue
        if (md.meter_key, md.date) not in in_pool:
            entry = empty.setdefault(md.meter_key, {
                "meter_id": md.meter_key, "industries": set(), "users": set(),
                "days_seen": set(), "flags": collections.Counter(),
            })
            entry["industries"].add(md.industry)
            entry["users"].add(md.user_key)
            entry["days_seen"].add(md.date)
            entry["flags"].update(md.flags)
            continue
        entry = facts.setdefault(md.meter_key, {
            "meter_id": md.meter_key,
            "industries": set(),
            "users": set(),
            "days_in_pool": set(),
        })
        entry["industries"].add(md.industry)
        entry["users"].add(md.user_key)
        entry["days_in_pool"].add(md.date)
    # A meter reaches `empty` on any day it has no pool block, so a meter with
    # a mix of days lands in both. Only the meters that contribute *nothing*
    # are a finding, and those are the ones absent from `facts`.
    empty = {k: v for k, v in empty.items() if k not in facts}
    return facts, empty


def assign_buses(facts: Dict[str, dict],
                 industry_buses: Dict[str, List[int]],
                 excluded: Sequence[str],
                 rule: str) -> List[dict]:
    """Apply one candidate rule to every qualified meter.

    The rule decides *which* of an industry's buses a meter takes; it never
    decides *whether* the meter is qualified, so this function cannot admit a
    meter that rule B left out.
    """
    rows: List[dict] = []
    by_industry: Dict[str, List[dict]] = collections.defaultdict(list)
    for entry in facts.values():
        industry = sorted(entry["industries"])[0]
        by_industry[industry].append(entry)

    for industry in sorted(by_industry):
        members = by_industry[industry]
        buses = industry_buses.get(industry) or []
        for entry in sorted(members, key=lambda e: e["meter_id"]):
            if len(entry["industries"]) > 1:
                rows.append(_row(entry, industry, None, rule,
                                 f"meter carries {len(entry['industries'])} "
                                 f"industries; one bus cannot take them all"))
                continue
            if industry in excluded:
                rows.append(_row(entry, industry, None, rule,
                                 "industry is listed in `excluded_industries` "
                                 "and must not shape a load bus"))
                continue
            if not buses:
                rows.append(_row(entry, industry, None, rule,
                                 "industry is assigned no bus in "
                                 f"`{SCENARIO}.industry_map.buses`"))
                continue

            if rule == R1:
                bus = buses[0] if len(buses) == 1 else None
                reason = (f"{industry} is assigned exactly one bus in "
                          f"`{SCENARIO}.industry_map`" if bus is not None else
                          f"{industry} is assigned {len(buses)} buses, so this "
                          "rule does not choose between them")
            elif rule == R2:
                users = sorted(entry["users"])
                order = sorted({u for m in members for u in m["users"]})
                bus = buses[order.index(users[0]) % len(buses)]
                reason = (f"user is the {order.index(users[0]) + 1}th of "
                          f"{len(order)} users in {industry}, taking bus "
                          f"{bus} of {len(buses)}; all of a user's meters stay "
                          "on one bus")
            elif rule == R3:
                order = sorted(m["meter_id"] for m in members)
                position = order.index(entry["meter_id"])
                bus = buses[position % len(buses)]
                reason = (f"meter is the {position + 1}th of {len(order)} in "
                          f"{industry}, taking bus {bus} of {len(buses)} by "
                          "position; a user's meters may land on several buses")
            else:  # pragma: no cover - guarded by the CLI choices
                raise ValueError(f"unknown rule {rule!r}")
            rows.append(_row(entry, industry, bus, rule, reason))
    return rows


def _row(entry: dict, industry: str, bus: Optional[int], rule: str,
         reason: str) -> dict:
    return {
        "meter_id": entry["meter_id"],
        "user_id": ";".join(sorted(entry["users"])),
        "industry": industry,
        "n_users": len(entry["users"]),
        "destination_bus": "" if bus is None else int(bus),
        "mapping_rule": rule,
        "mapping_reason": reason,
        "candidate_status": "mapped" if bus is not None else "unmapped",
        "simulation_mapping": True,
        "days_in_shape_pool": len(entry["days_in_pool"]),
    }


def collect_cells(records: Sequence[il.MeterDay],
                  facts: Dict[str, dict]) -> Dict[Tuple[str, str], dict]:
    """Every (meter, day) cell, with what the workbook holds for it."""
    cells: Dict[Tuple[str, str], dict] = {}
    for md in records:
        if md.meter_key not in facts:
            continue
        cell = cells.setdefault((md.meter_key, md.date), {
            "records": 0, "present": [], "flags": set(),
        })
        cell["records"] += 1
        cell["flags"].update(md.flags)
        present = sum(1 for v in md.values if v is not None)
        cell["present"].append(present)
    return cells


def classify_cell(cell: Optional[dict]) -> dict:
    """The cell's status, its slot coverage, and whether it may be summed."""
    if cell is None:
        return {"status": "absent", "slots_present": 0, "records": 0,
                "repeats": 0, "conflicts": 0, "usability": "missing",
                "flags": ""}

    flags = cell["flags"]
    repeats = 1 if il.FLAG_DUPLICATE_ROW in flags else 0
    conflicts = 1 if il.FLAG_DUPLICATE_CONFLICT in flags else 0
    slots_present = min(cell["present"])

    if conflicts:
        status = "conflicting"
    elif il.FLAG_EMPTY_IDENTITY in flags:
        status = "empty_identity"
    elif il.FLAG_OTHER_PHASE in flags:
        status = "other_phase"
    elif il.FLAG_ALL_ZERO in flags:
        status = "all_zero"
    elif il.FLAG_PLACEHOLDER in flags:
        status = "placeholder"
    elif il.FLAG_MISSING_SLOTS in flags or slots_present < N_SLOTS:
        status = "partial"
    else:
        status = "complete"

    # A cell is summable only when its magnitude is a measurement. Anything
    # short of a full active-power day is not, and is never zero-filled.
    usable = status == "complete"
    return {
        "status": status,
        "slots_present": slots_present,
        "records": cell["records"],
        "repeats": repeats,
        "conflicts": conflicts,
        "usability": "summable" if usable else "excluded",
        "flags": ";".join(sorted(flags)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Candidate bus assignment and coverage audit.")
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--rules", default=f"{R1},{R2},{R3}",
                        help="comma-separated candidate rules to emit")
    args = parser.parse_args(argv)

    industry_buses, excluded = load_industry_map()
    print("Declared industry -> buses, from "
          f"`{SCENARIO}.industry_map` (simulation mapping)")
    for industry in sorted(industry_buses):
        buses = industry_buses[industry]
        shown = str(buses) if len(buses) <= 12 else \
            f"{buses[:6]} ... {buses[-3:]} ({len(buses)} buses)"
        print(f"  {industry:<12} {shown}")
    print(f"  excluded industries: {excluded}")

    workbook = il.resolve_workbook(args.workbook)
    records = il.flag_records(list(il.iter_meter_days(args.workbook, args.sheet)))
    facts, pool_empty = qualified_meter_facts(records)

    print()
    print("Qualified meters under rule B (meter gate AND a block in the pool)")
    print(f"  meters        {len(facts)}")
    print(f"  users         {len({u for f in facts.values() for u in f['users']})}")
    by_industry = collections.Counter(
        sorted(f["industries"])[0] for f in facts.values())
    for industry in sorted(by_industry):
        buses = industry_buses.get(industry) or []
        excluded_note = "  [EXCLUDED]" if industry in excluded else ""
        print(f"  {industry:<12} meters {by_industry[industry]:<4} "
              f"buses {len(buses)}{excluded_note}")

    print()
    print("Meters that pass the ratio/capacity gate but contribute no block")
    print(f"  meters {len(pool_empty)}, users "
          f"{len({u for f in pool_empty.values() for u in f['users']})}")
    for meter_id in sorted(pool_empty):
        f = pool_empty[meter_id]
        top = ", ".join(f"{flag} x{n}" for flag, n
                        in f["flags"].most_common(3))
        print(f"      {meter_id}  {sorted(f['industries'])[0]:<12} "
              f"days {len(f['days_seen']):<4} {top}")

    # A meter or a user spanning two industries breaks the "one bus per meter,
    # all of a user's meters together" premise, so it is reported, not resolved.
    multi = {k: v for k, v in facts.items() if len(v["industries"]) > 1}
    user_industries: Dict[str, set] = collections.defaultdict(set)
    for f in facts.values():
        for u in f["users"]:
            user_industries[u].update(f["industries"])
    split_users = {u: sorted(i) for u, i in user_industries.items()
                   if len(i) > 1}
    print()
    print(f"  meters carrying >1 industry : {len(multi)}")
    print(f"  users spanning >1 industry  : {len(split_users)}")
    for u, inds in sorted(split_users.items()):
        print(f"      {u} {inds}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: Dict[str, List[dict]] = {}
    for rule in [r.strip() for r in args.rules.split(",") if r.strip()]:
        rows = assign_buses(facts, industry_buses, excluded, rule)
        all_rows[rule] = rows
        mapped = [r for r in rows if r["candidate_status"] == "mapped"]
        unmapped = [r for r in rows if r["candidate_status"] != "mapped"]
        print()
        print(f"Rule {rule}: mapped {len(mapped)}/{len(rows)} meters, "
              f"unmapped {len(unmapped)}")
        reasons = collections.Counter(r["mapping_reason"] for r in unmapped)
        for reason, n in reasons.most_common():
            print(f"    {n:>3}  {reason}")
        bus_counts = collections.Counter(r["destination_bus"] for r in mapped)
        print(f"    buses used: {len(bus_counts)} of 32; "
              f"busiest {bus_counts.most_common(3)}")
        path = out / f"industry_meter_bus_map_candidate_{rule}.csv"
        _write_map(path, rows)
        print(f"    wrote {path}")

    dates = sorted({md.date for md in records if md.date is not None})
    cells = collect_cells(records, facts)
    print()
    print(f"Coverage: {len(facts)} qualified meters x {len(dates)} dates "
          f"= {len(facts) * len(dates):,} cells")

    # A cell's status is a property of the workbook, so it is decided once and
    # shared by every candidate; only which bus a cell lands on depends on the
    # rule. Deciding it per rule would let two candidates disagree about the
    # same measurement.
    verdicts_by_cell = {(m, d): classify_cell(cells.get((m, d)))
                        for m in facts for d in dates}
    status_totals: collections.Counter = collections.Counter(
        v["status"] for v in verdicts_by_cell.values())
    per_date_meters: collections.Counter = collections.Counter()
    for (_meter, date), v in verdicts_by_cell.items():
        if v["usability"] == "summable":
            per_date_meters[date] += 1

    # The matrix labels its bus column with the frozen rule, so the delivered
    # artefact and the frozen case cannot disagree about which bus a meter is
    # on. Among candidates it falls back to the widest one.
    best_rule = FROZEN_RULE if FROZEN_RULE in all_rows else max(
        all_rows, key=lambda r: sum(
            1 for row in all_rows[r] if row["candidate_status"] == "mapped"))
    bus_of = {r["meter_id"]: r["destination_bus"] for r in all_rows[best_rule]}

    matrix_path = out / "industry_meter_day_coverage.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["meter_id", "user_id", "industry", "date", "status",
                         "slots_present", "records", "repeats", "conflicts",
                         "usability", "flags",
                         f"destination_bus[{best_rule}]"])
        for meter_id in sorted(facts):
            fact = facts[meter_id]
            industry = sorted(fact["industries"])[0]
            bus = bus_of.get(meter_id)
            for date in dates:
                info = verdicts_by_cell[(meter_id, date)]
                writer.writerow([meter_id, ";".join(sorted(fact["users"])),
                                 industry, date, info["status"],
                                 info["slots_present"], info["records"],
                                 info["repeats"], info["conflicts"],
                                 info["usability"], info["flags"],
                                 bus if bus else ""])

    total_cells = max(sum(status_totals.values()), 1)
    print()
    print("Cell status, over the whole matrix")
    for status in STATUS_PRECEDENCE:
        n = status_totals.get(status, 0)
        print(f"  {status:<15} {n:>7,}  {n / total_cells:6.2%}")
    print(f"  {'TOTAL':<15} {sum(status_totals.values()):>7,}")

    summable = status_totals.get("complete", 0)
    print()
    print(f"Summable cells (complete 96-slot meter-days): {summable:,} "
          f"= {summable / total_cells:.2%} of the matrix")

    n_meters = len(facts)
    counts = sorted(per_date_meters[d] for d in dates)
    print(f"Complete meters per date: min {counts[0]}, "
          f"median {counts[len(counts) // 2]}, max {counts[-1]} "
          f"(ceiling {n_meters}), over {len(dates):,} dates")
    print("  (dates no qualified meter has a complete day on: "
          f"{len(dates) - len(per_date_meters):,})")
    print()
    print("Dates surviving a completeness threshold (threshold -> dates)")
    for threshold in (n_meters, 80, 70, 60, 50, 40, 35, 30, 20, 1):
        n = sum(1 for d in dates if per_date_meters.get(d, 0) >= threshold)
        print(f"  >= {threshold:>3} complete meters of {n_meters}: "
              f"{n:>4} dates ({n / len(dates):6.1%})")

    # The binding constraint for a full-day absolute load is not "enough meters"
    # in total but "every bus that is supposed to carry load has something on
    # it", and that depends on which buses the rule uses -- so it is scored per
    # candidate rather than once.
    print()
    print("Per-candidate bus coverage")
    print(f"  {'rule':<18}{'mapped':>8}{'buses':>7}{'full days':>11}")
    per_rule: Dict[str, dict] = {}
    for rule, rows in all_rows.items():
        per_bus, used, full, per_date_buses = bus_stats(rows, verdicts_by_cell,
                                                       dates)
        per_rule[rule] = {"per_bus": per_bus, "used": used, "full": full,
                          "rows": rows, "per_date_buses": per_date_buses}
        mapped = sum(1 for r in rows if r["candidate_status"] == "mapped")
        print(f"  {rule:<18}{mapped:>8}{len(used):>7}{len(full):>11}")

    # How many buses can be populated at once is the constraint that decides
    # whether an absolute-load day is buildable at all, and it is a different
    # question from how many meters the rule manages to place.
    print()
    print("Buses carrying a complete meter, on the same date")
    for rule, entry in per_rule.items():
        n_used = len(entry["used"])
        per_date = entry["per_date_buses"]
        counts = sorted(per_date.get(d, 0) for d in dates)
        print(f"  {rule}  ({n_used} buses in use)")
        print(f"      min {counts[0]}, median {counts[len(counts) // 2]}, "
              f"max {counts[-1]}")
        for threshold in (n_used, 30, 25, 20, 15, 10, 5, 1):
            if threshold > n_used:
                continue
            n = sum(1 for d in dates if per_date.get(d, 0) >= threshold)
            print(f"      >= {threshold:>3} buses: {n:>4} dates "
                  f"({n / len(dates):6.1%})")

    print()
    print(f"Per-bus coverage (candidate {best_rule}, only buses in use)")
    print(f"  {'bus':<5}{'meters':>7}{'complete':>10}{'any-day':>9}"
          f"{'dates_ok':>10}  industries")
    for bus in per_rule[best_rule]["used"]:
        row = per_rule[best_rule]["per_bus"][bus]
        industries = sorted({r["industry"] for r in all_rows[best_rule]
                             if r["destination_bus"] == bus})
        print(f"  {bus:<5}{row['meters']:>7}{row['complete_days']:>10}"
              f"{row['any_days']:>9}{row['dates_covered']:>10}  "
              f"{','.join(industries)}")

    # ---- The frozen case ----
    frozen = frozen_case(facts, industry_buses, excluded)
    frozen_bus = {r["meter_id"]: int(r["destination_bus"]) for r in frozen
                  if r["candidate_status"] == "mapped"}
    day_axis = [d for d in dates if str(d) not in FABRICATED_DATES]
    qualified_keys = il.qualified_block_keys(records)
    conflicting = sorted((m, str(d)) for (m, d), v in verdicts_by_cell.items()
                         if v["status"] == "conflicting"
                         and (m, d) in qualified_keys)

    nodes = sorted({b for b in frozen_bus.values()})
    node_days = node_day_coverage(frozen_bus, verdicts_by_cell, day_axis)
    dates_of_node: Dict[int, set] = collections.defaultdict(set)
    for row in node_days:
        if row["covered"]:
            dates_of_node[row["destination_bus"]].add(row["date"])

    print()
    print(f"FROZEN CASE -- {FROZEN_RULE}, buses {UNSUPPORTED_BUSES} carrying no "
          f"mapped load")
    print(f"  meters mapped   {len(frozen_bus)} of {len(facts)}")
    print(f"  nodes in use    {len(nodes)} of 32 (topology unchanged)")
    print(f"  day axis        {len(day_axis)} of {len(dates)} dates "
          f"(excluding {', '.join(FABRICATED_DATES)})")
    unmapped_reasons = collections.Counter(
        r["mapping_reason"] for r in frozen if r["candidate_status"] != "mapped")
    print(f"  unmapped meters {sum(unmapped_reasons.values())}")
    for reason, n in unmapped_reasons.most_common():
        print(f"      {n:>3}  {reason[:100]}")

    print()
    print(f"  {'node':<6}{'meters':>8}{'days covered':>14}{'of days':>9}")
    for node in nodes:
        n_meters = sum(1 for b in frozen_bus.values() if b == node)
        print(f"  {node:<6}{n_meters:>8}{len(dates_of_node[node]):>14}"
              f"{len(day_axis):>9}")

    all_nodes_days = [d for d in day_axis
                      if all(d in dates_of_node[n] for n in nodes)]
    counts = sorted(len([n for n in nodes if d in dates_of_node[n]])
                    for d in day_axis)
    print()
    print(f"  nodes covered per day: min {counts[0]}, "
          f"median {counts[len(counts) // 2]}, max {counts[-1]} of {len(nodes)}")
    print(f"  days with ALL {len(nodes)} nodes covered: {len(all_nodes_days)} "
          f"of {len(day_axis)} ({len(all_nodes_days) / len(day_axis):.1%})")

    frozen_map_path = out / "industry_meter_bus_map_frozen.csv"
    _write_map(frozen_map_path, frozen)
    node_day_path = out / "industry_node_day_coverage.csv"
    with node_day_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(node_days[0]))
        writer.writeheader()
        writer.writerows(node_days)
    excl_path = out / "industry_load_exclusions_s43.csv"
    exclusions = exclusion_rows(frozen, industry_buses, pool_empty, conflicting)
    with excl_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kind", "key", "industry",
                                                "reason"])
        writer.writeheader()
        writer.writerows(exclusions)
    print()
    print(f"  exclusions by kind: "
          f"{dict(collections.Counter(e['kind'] for e in exclusions))}")

    _write_manifest(out, workbook, facts, dates, status_totals, per_rule,
                    best_rule, args.rules, sorted(pool_empty), {
                        "rule": FROZEN_RULE,
                        "unsupported_buses": list(UNSUPPORTED_BUSES),
                        "fabricated_dates": list(FABRICATED_DATES),
                        "meters_mapped": len(frozen_bus),
                        "nodes_in_use": len(nodes),
                        "days": len(day_axis),
                        "days_with_all_nodes_covered": len(all_nodes_days),
                        "reading_unit": il.READING_UNIT,
                        "reading_unit_source": il.READING_UNIT_SOURCE,
                        "mw_path": "refused while READING_UNIT is None",
                    })
    print(f"\nwrote {matrix_path}")
    print(f"wrote {frozen_map_path}")
    print(f"wrote {node_day_path}")
    print(f"wrote {excl_path}")
    print(f"wrote {out / 'industry_meter_mapping_manifest.json'}")
    return 0


def bus_stats(rows: Sequence[dict], verdicts_by_cell: Dict[tuple, dict],
              dates: Sequence) -> Tuple[Dict[int, dict], List[int], List]:
    """Per-bus cell accounting for one candidate, and the fully covered days.

    `complete_days` counts (meter, day) cells; `dates_covered` counts distinct
    days. They differ by how many meters a bus carries, and conflating them is
    how a bus with one well-covered meter looks as covered as one with nine.
    """
    bus_of = {r["meter_id"]: int(r["destination_bus"]) for r in rows
              if r["candidate_status"] == "mapped"}
    per_bus: Dict[int, dict] = collections.defaultdict(
        lambda: {"meters": 0, "complete_days": 0, "any_days": 0,
                 "dates_covered": 0})
    per_bus_dates: Dict[int, set] = collections.defaultdict(set)
    # A set, not a counter: a date is "covered on this bus" once, however many
    # meters that bus carries. Counting cells here would make a bus with
    # fourteen meters look like fourteen buses.
    per_date_buses: Dict[object, set] = collections.defaultdict(set)
    for (meter, date), v in verdicts_by_cell.items():
        bus = bus_of.get(meter)
        if bus is None:
            continue
        if v["usability"] == "summable":
            per_bus[bus]["complete_days"] += 1
            per_bus_dates[bus].add(date)
            per_date_buses[date].add(bus)
        if v["records"]:
            per_bus[bus]["any_days"] += 1
    for bus in bus_of.values():
        per_bus[bus]["meters"] += 1
    used = sorted(per_bus)
    for bus in used:
        per_bus[bus]["dates_covered"] = len(per_bus_dates[bus])
    # `used` empty means nothing is mapped, and `all()` over an empty set is
    # vacuously true -- which would report every date as fully covered for a
    # candidate that covers nothing. A day is fully covered only if there are
    # buses on it to cover.
    full = [d for d in dates
            if used and all(d in per_bus_dates[b] for b in used)]
    counts = {d: len(buses) for d, buses in per_date_buses.items()}
    return per_bus, used, full, counts


def _write_map(path: Path, rows: List[dict]) -> None:
    columns = ["meter_id", "user_id", "industry", "n_users", "destination_bus",
               "mapping_rule", "mapping_reason", "candidate_status",
               "simulation_mapping", "days_in_shape_pool"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["meter_id"]):
            writer.writerow(row)


def frozen_case(facts: Dict[str, dict], industry_buses: Dict[str, List[int]],
                excluded: Sequence[str]) -> List[dict]:
    """The adopted mapping: `FROZEN_RULE`, minus the unsupported buses.

    A meter whose industry resolves only to an unsupported bus is reported
    `unmapped` with that reason. It is not reassigned to another industry's bus
    and it is not dropped silently: withholding a meter is a stated exclusion,
    and the exclusion list is part of the deliverable.
    """
    rows = assign_buses(facts, industry_buses, excluded, FROZEN_RULE)
    for row in rows:
        if row["candidate_status"] != "mapped":
            continue
        if int(row["destination_bus"]) in UNSUPPORTED_BUSES:
            row["candidate_status"] = "unmapped"
            row["mapping_reason"] = (
                f"bus {row['destination_bus']} carries no data-supported load "
                f"in the formal case (S4.3 ruling); {row['industry']} has no "
                "other bus")
            row["destination_bus"] = ""
    return rows


def exclusion_rows(rows: Sequence[dict], industry_buses: Dict[str, List[int]],
                   pool_empty: Dict[str, dict],
                   conflicting: Sequence[tuple]) -> List[dict]:
    """Everything the formal case leaves out, and why. One row per exclusion.

    Four kinds, and each one is stated rather than absorbed: a meter with no
    admissible bus, a bus with no admissible load, a date that is not a
    measurement, and a meter-day whose records disagree.
    """
    out: List[dict] = []
    for row in rows:
        if row["candidate_status"] == "mapped":
            continue
        out.append({
            "kind": "unmapped_meter",
            "key": row["meter_id"],
            "industry": row["industry"],
            "reason": row["mapping_reason"],
        })
    for bus in UNSUPPORTED_BUSES:
        serves = sorted(ind for ind, buses in industry_buses.items()
                        if bus in buses)
        out.append({
            "kind": "unsupported_bus",
            "key": f"bus {bus}",
            "industry": ",".join(serves),
            "reason": ("kept in the IEEE33 topology but carries no mapped load: "
                       "its only feeding industry covers too few days to "
                       "populate it, so the formal case has 29 data-supported "
                       "load buses rather than 32"),
        })
    for date in FABRICATED_DATES:
        out.append({
            "kind": "fabricated_date",
            "key": str(date),
            "industry": "",
            "reason": ("every record on this date carries the workbook's "
                       "synthesised placeholder signature; not a measurement"),
        })
    for meter_id, date in conflicting:
        out.append({
            "kind": "conflicting_meter_day",
            "key": f"{meter_id} {date}",
            "industry": "",
            "reason": ("two records disagree about this meter-day; the block is "
                       "withheld whole, because summing either reading would "
                       "pick one without evidence"),
        })
    for meter_id, entry in sorted(pool_empty.items()):
        out.append({
            "kind": "unresolved_all_zero_meter",
            "key": meter_id,
            "industry": sorted(entry["industries"])[0],
            "reason": ("reads zero on essentially every day in the workbook and "
                       "is therefore outside the qualified set. Whether the "
                       "site is idle or the meter stopped reporting is "
                       "unresolved, so the meter is neither included nor "
                       "declared idle"),
        })
    return out


def node_day_coverage(bus_of: Dict[str, int],
                      verdicts_by_cell: Dict[tuple, dict],
                      dates: Sequence) -> List[dict]:
    """One row per (node, day): how many mapped meters are usable there."""
    meters_of: Dict[int, List[str]] = collections.defaultdict(list)
    for meter, bus in bus_of.items():
        meters_of[bus].append(meter)
    rows = []
    for bus in sorted(meters_of):
        for date in dates:
            cells = [verdicts_by_cell[(m, date)] for m in meters_of[bus]
                     if (m, date) in verdicts_by_cell]
            usable = [c for c in cells if c["usability"] == "summable"]
            rows.append({
                "destination_bus": bus,
                "date": date,
                "meters_mapped": len(meters_of[bus]),
                "meters_with_a_record": sum(1 for c in cells if c["records"]),
                "meters_complete": len(usable),
                "covered": int(bool(usable)),
            })
    return rows


def _write_manifest(out: Path, workbook: Path, facts, dates, status_totals,
                    per_rule: Dict[str, dict], best_rule: str, rules: str,
                    pool_empty: Sequence[str],
                    frozen: Dict[str, object]) -> None:
    manifest = {
        "frozen_case": frozen,
        "workbook_sha256": il.file_sha256(workbook),
        "scenario": SCENARIO,
        "candidate_rules": [r.strip() for r in rules.split(",") if r.strip()],
        "frozen": False,
        "simulation_mapping": True,
        "qualified_meters": len(facts),
        "dates_in_workbook": len(dates),
        "cells": len(facts) * len(dates),
        "status_totals": dict(status_totals),
        "meters_passing_gate_without_a_block": list(pool_empty),
        "candidates": {
            rule: {
                "mapped_meters": sum(1 for r in entry["rows"]
                                     if r["candidate_status"] == "mapped"),
                "buses_used": entry["used"],
                "dates_with_every_used_bus_populated": len(entry["full"]),
                "per_bus": {str(b): dict(v)
                            for b, v in sorted(entry["per_bus"].items())},
            }
            for rule, entry in sorted(per_rule.items())
        },
        "coverage_matrix_column": f"destination_bus[{best_rule}]",
    }
    (out / "industry_meter_mapping_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
