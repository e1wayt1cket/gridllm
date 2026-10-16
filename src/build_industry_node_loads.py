# build_industry_node_loads.py
"""The 33-bus node-load ledger for one scenario on one real date.

Two ledgers can come out of this module and they are not interchangeable.

**The formal ledger** states the network's absolute load in MW. It is refused
while the reading unit is `unit_unverified`, because every physical conclusion
drawn from it would inherit an unconfirmed conversion.

**The audit ledger** is what can be built today: the same composition, every row
carrying the declared unit *and* its status, so the numbers can be examined and
argued with without being published as the network's load. Nothing in it is
scaled, zero-filled or interpolated, and no reading is adjusted toward its
capacity.

Composition is defined once, here:

    preserve   P_final = P_base
    replace    P_final = P_industry
    overlay    P_final = P_base + P_industry

`P_base` is the IEEE33 bus's own rated demand, read from the built network
rather than from a formula, and shaped within the day by the same
`grid._make_load_profile(shape="native")` the agents would otherwise carry.
`P_industry` is the qualified meters' readings converted once and summed by
user, then by bus — never rescaled, never averaged across users, and never
counted on two buses.

All 33 buses are emitted. Bus 0 is the slack: it carries no load, is not a
mapping target, and its row says so rather than being omitted.

Usage:
    PYTHONPATH=src python src/build_industry_node_loads.py --scenario typical_day_industry
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config_loader as cfg
import grid
import industry_load as il
import industry_bus_mapping as ibm
import models
import map_industry_meters as mim

#: The three policies. A bus takes exactly one, and there is no feeder default.
POLICIES = ("preserve", "replace", "overlay")
SLACK_BUS = 0

#: The IEEE33 bus numbers, as the built network has them.
ALL_BUSES = tuple(range(0, 33))
LOAD_BUSES = tuple(range(1, 33))


# ---------------------------------------------------------------------------
# Bus classification
# ---------------------------------------------------------------------------

def bus_classification(map_csv: Path) -> List[dict]:
    """Every one of the 33 buses, and what the mapping file says about it.

    Derived from the mapping artefact rather than from a list maintained by
    hand: a hand-kept list is how a bus ends up described as both carrying a
    user and being empty. `has_user` and `no_user` are complements by
    construction, so they cannot overlap.
    """
    rows = list(csv.DictReader(map_csv.open(encoding="utf-8")))
    by_bus: Dict[int, dict] = collections.defaultdict(
        lambda: {"users": [], "meters": [], "industries": set()})
    for row in rows:
        if row["candidate_status"] != "mapped":
            continue
        bus = int(row["destination_bus"])
        by_bus[bus]["users"].append(row["user_id"])
        by_bus[bus]["industries"].add(row["industry"])
        by_bus[bus]["meters"].extend(
            m for m in row["meter_ids"].split(";") if m)

    out = []
    for bus in ALL_BUSES:
        entry = by_bus.get(bus)
        out.append({
            "bus": bus,
            "role": "slack" if bus == SLACK_BUS else "load",
            "is_mapping_target": bus in LOAD_BUSES,
            "has_user": bool(entry),
            "status": "slack" if bus == SLACK_BUS
                      else ("has_user" if entry else "no_user"),
            "n_users": len(entry["users"]) if entry else 0,
            "n_meters": len(entry["meters"]) if entry else 0,
            "industries": ",".join(sorted(entry["industries"])) if entry else "",
        })
    return out


def assert_classification_complete(classes: Sequence[dict]) -> None:
    """The properties the classification must have, checked not assumed."""
    buses = [c["bus"] for c in classes]
    if sorted(buses) != list(ALL_BUSES):
        raise AssertionError(
            f"classification covers {sorted(buses)}, not the 33 buses "
            f"{list(ALL_BUSES)}")
    has = {c["bus"] for c in classes if c["has_user"]}
    none = {c["bus"] for c in classes if not c["has_user"]}
    if has & none:
        raise AssertionError(f"bus(es) {sorted(has & none)} are both classified "
                             "as carrying a user and as carrying none")
    if SLACK_BUS in has:
        raise AssertionError("the slack bus carries a user; it is not a mapping "
                             "target")


# ---------------------------------------------------------------------------
# The two load terms
# ---------------------------------------------------------------------------

def native_bus_load(load_buses: Sequence[int] = LOAD_BUSES) -> Dict[int, float]:
    """The IEEE33 rated demand of each load bus, in MW, from the built network.

    Read from `net.load`, which is the benchmark's own data, rather than from
    `native_nominal_demand_mw`, which is a sum used as a rescaling target.
    """
    net = grid.build_base_network(models.MarketConfig.from_yaml())
    rated = {int(row.bus): float(row.p_mw) for _i, row in net.load.iterrows()}
    missing = sorted(set(load_buses) - set(rated))
    if missing:
        raise KeyError(f"the network has no rated demand for load bus(es) "
                       f"{missing}")
    return rated


def native_shape(periods: int = il.N_SLOTS) -> np.ndarray:
    """The within-day shape the native load is spread over.

    Reused from `grid` so the ledger's native column is exactly what the agents
    would otherwise carry, instead of a second definition of the same profile.
    """
    load_cfg = cfg.get_default("profiles.load") or {}
    return grid._make_load_profile(np.arange(periods), "industrial", load_cfg,
                                   shape="native")


#: Why a meter contributes nothing to a day. Stated once, because a caller
#: that re-words it is a caller that has re-implemented the guard below.
INCOMPLETE_DAY_REASON = ("not a complete active-power day, or no usable "
                         "multiplier")
UNMAPPED_USER_REASON = "the meter's user is not mapped to any bus"


def per_user_curves(date, records: Sequence[il.MeterDay]
                    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]],
                               List[Tuple[str, str]]]:
    """Every user's 96-point curve on one date, and what would not convert.

    **The aggregation is coincident power.** A slot reading is the average
    active power over its 15 minutes (see `industry_load.reading_to_mw`), so
    every qualified meter-day is converted **once** -- `value x 综合倍率`, in kW,
    then `/1000` to MW -- and the converted values are **summed**. There is no
    `60/15` factor anywhere: multiplying a mean power by four to "make it an
    hour" is the defect S1 removed, and it is what made every implied peak four
    times its true size. Five users on one bus each drawing 10 MW is 50 MW, not
    10, and not 200.

    Summed by user and not averaged across them: that is the difference between
    an absolute load and a shape.

    A cell that is not a complete active-power day contributes nothing and is
    returned as unreadable -- never zero-filled, because a padded day would look
    like a measured one.

    **The date guard lives here and only here.** Both conditions are needed: the
    membership test asks whether the meter has a qualified block on this date,
    and without the date equality beside it the test answers for every date at
    once and the sum becomes a meter's whole history. That defect has been
    introduced twice -- see `reports/s3_node_load_ledger.md` -- so the guard is
    written once and every caller reaches it through this function.

    Returns the curves by user, the meter ids seen per user on the date (in
    record order, so a caller can name what it excluded), and the
    `(meter_id, user_id)` of every record that could not be converted.
    """
    in_pool = il.qualified_block_keys(records)
    per_user: Dict[str, np.ndarray] = {}
    meters: Dict[str, List[str]] = collections.defaultdict(list)
    unreadable: List[Tuple[str, str]] = []
    for md in records:
        if md.date != date or (md.meter_key, md.date) not in in_pool:
            continue
        meters[md.user_key].append(md.meter_key)
        arr = il.reading_to_mw(md)
        if arr is None:
            unreadable.append((md.meter_key, md.user_key))
            continue
        if md.user_key not in per_user:
            per_user[md.user_key] = np.zeros(il.N_SLOTS)
        per_user[md.user_key] = per_user[md.user_key] + arr
    return per_user, dict(meters), unreadable


def industry_curves(date, user_bus: Dict[str, int],
                    records: Sequence[il.MeterDay]
                    ) -> Tuple[Dict[int, np.ndarray], List[dict], Dict[int, List[str]]]:
    """The industry term by bus for one date, the exclusions, and who it came from.

    A thin fold of `per_user_curves` over the bus map, so the date guard above
    has one implementation rather than two that can disagree.

    The third value is the users that actually contributed, by bus: the
    manifest has to be able to name them, not only count the buses.
    """
    per_user, meters, unreadable = per_user_curves(date, records)
    unmapped = {u for u in meters if u not in user_bus}

    exclusions: List[dict] = []
    for meter_id, user in unreadable:
        # An unmapped user is reported once, below, under the reason that
        # actually stopped it; two reasons for one record would be two.
        if user in unmapped:
            continue
        exclusions.append({"meter_id": meter_id, "date": str(date),
                           "bus": user_bus[user],
                           "reason": INCOMPLETE_DAY_REASON})
    for user in sorted(unmapped):
        for meter_id in meters[user]:
            exclusions.append({"meter_id": meter_id, "date": str(date),
                               "bus": "", "reason": UNMAPPED_USER_REASON})

    per_bus: Dict[int, np.ndarray] = collections.defaultdict(
        lambda: np.zeros(il.N_SLOTS))
    bus_users: Dict[int, List[str]] = collections.defaultdict(list)
    for user, arr in per_user.items():
        bus = user_bus.get(user)
        if bus is None:
            continue
        per_bus[bus] = per_bus[bus] + arr
        bus_users[bus].append(user)
    return (dict(per_bus), exclusions,
            {b: sorted(us) for b, us in bus_users.items()})


def compose(native_mw: Optional[float], industry_mw: Optional[float],
            policy: str) -> Tuple[Optional[float], str]:
    """The composition rule, and the status of the industry term.

    A `replace` or `overlay` bus with no admissible industry data for the period
    is **not** written as zero: that would report an unobserved bus as an empty
    one. It is returned as `None` with a status saying why.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown load policy {policy!r}; expected one of "
                         f"{POLICIES}")
    if policy == "preserve":
        return native_mw, "not_applicable"
    if industry_mw is None:
        return None, "excluded_no_data"
    if policy == "replace":
        return industry_mw, "present"
    return native_mw + industry_mw, "present"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def bus_policies(scenario: str, classes: Sequence[dict]) -> Dict[int, str]:
    """The declared policy for every one of the 33 buses.

    Read from the scenario's `bus_load_policy`. A missing bus, or a policy that
    is not one of the three, raises: a feeder-wide default would make the day
    depend on something the scenario never declared.
    """
    declared = (cfg.get_scenario_cfg(scenario).get("load_model") or {})
    policies = declared.get("bus_load_policy") or {}
    if not policies:
        raise ValueError(
            f"{scenario} declares no `load_model.bus_load_policy`; every one of "
            "the 33 buses must be assigned a policy explicitly")
    out: Dict[int, str] = {}
    for bus in ALL_BUSES:
        # YAML reads an unquoted `0:` as an integer key, so accept both forms
        # rather than making the config's quoting decide whether it loads.
        if bus in policies:
            value = str(policies[bus])
        elif str(bus) in policies:
            value = str(policies[str(bus)])
        else:
            raise ValueError(
                f"{scenario}'s bus_load_policy leaves bus {bus} unassigned; "
                "there is no feeder-wide default")
        if value not in POLICIES:
            raise ValueError(
                f"{scenario} gives bus {bus} policy {value!r}; expected one of "
                f"{POLICIES}")
        out[bus] = value
    if out[SLACK_BUS] != "preserve":
        raise ValueError(
            "the slack bus carries no load; its policy is `preserve` so the "
            "per-bus table covers all 33 buses without inventing a load for it")
    return out


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def pick_date(records: Sequence[il.MeterDay], dates: Sequence) -> object:
    """The date with the most complete qualified meters, earliest wins ties.

    **This is the biased rule and it is no longer a default.** More complete
    meters means more load, so the date it returns sits high in the load
    distribution -- on bus 16 the date it picked ranked 694th of 730 days. It
    is kept because the audit of that bias is reproducible through it, and
    because a caller may still ask for it by name; a ledger may not.
    See `reports/s3_date_selection_audit.md`.
    """
    best, best_n = None, -1
    for date in sorted(dates):
        n = sum(1 for md in records
                if md.date == date and il.reading_to_mw(md) is not None)
        if n > best_n:
            best, best_n = date, n
    if best is None:
        raise ValueError("no date in the workbook has a complete qualified "
                         "meter-day")
    return best


#: The rule the single-day diagnostic case uses, named so it can be re-derived
#: and audited rather than being a date somebody picked.
LOAD_DATE_RULE = "earliest_complete_date"


def select_load_date(audit: Sequence[dict]) -> dict:
    """The earliest date, in ascending order, on which the load data is whole.

    "Whole" is `load_complete` from `date_audit`: every mapped bus carries a
    qualified reading for all 96 periods, with no cross-date aggregation, no
    double counting and nothing filled in. Ascending order is the point of the
    rule -- **it is not chosen on any quantity of load**. The rule this replaces
    picked by completeness rank, which is a monotone proxy for how much load
    the day carries, so it selected from the high end of the distribution and
    then described the result as representative. See
    `reports/s3_date_selection_audit.md`.

    The date it returns is a **diagnostic sample**, not a typical day in any
    statistical sense: nothing here estimates a distribution, and a date chosen
    by position on the calendar carries no claim about being average.

    Returns the audit row, so a caller records the evidence and not only the
    date.
    """
    for row in sorted(audit, key=lambda r: str(r["date"])):
        if row.get("load_complete"):
            return row
    raise ValueError(
        "no date in the audit has every mapped bus covered on all 96 periods; "
        "there is no whole day to build the diagnostic case on")


def date_audit(records: Sequence[il.MeterDay], user_bus: Dict[str, int],
               mapped_buses: Sequence[int]) -> List[dict]:
    """One row per real date: what the load data holds on it, and what it does not.

    No date is invented, none is copied from another, and a date with no
    reading is reported as having none rather than being filled. The ranking
    columns are **coverage ranks, not load quantiles**: while the reading unit is
    unverified, a cross-user aggregate may not be read as a physical demand
    percentile.
    """
    in_pool = il.qualified_block_keys(records)
    per_date: Dict[object, dict] = {}
    for md in records:
        if (md.meter_key, md.date) not in in_pool:
            continue
        bus = user_bus.get(md.user_key)
        if bus is None:
            continue
        entry = per_date.setdefault(md.date, {
            "meters": set(), "users": set(), "buses": set(),
            "readings": 0, "unconvertible": 0,
        })
        entry["readings"] += 1
        entry["users"].add(md.user_key)
        entry["buses"].add(bus)
        if il.reading_to_mw(md) is None:
            entry["unconvertible"] += 1
            continue
        entry["meters"].add(md.meter_key)

    wanted = set(int(b) for b in mapped_buses)
    rows = []
    for date in sorted(per_date):
        entry = per_date[date]
        covered = entry["buses"] & wanted
        rows.append({
            "date": str(date),
            "meters_with_a_complete_day": len(entry["meters"]),
            "users_covered": len(entry["users"]),
            "buses_covered": len(covered),
            "buses_mapped": len(wanted),
            "buses_without_data": ";".join(str(b) for b in
                                           sorted(wanted - covered)),
            "readings": entry["readings"],
            "readings_not_convertible": entry["unconvertible"],
            "load_complete": bool(covered == wanted),
            "price_date_available": False,
            "price_note": "no dated real price curve overlaps this load date",
        })
    return rows


def write_date_audit(rows: Sequence[dict], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "date_selection_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def build(scenario: str, date, out_dir: Path, map_csv: Path,
          formal: bool = False, selection: Optional[dict] = None,
          records: Optional[Sequence[il.MeterDay]] = None) -> dict:
    # Reading the 64 MiB workbook takes minutes, so a caller that has already
    # read it hands its flagged records in rather than paying for a second read
    # of the same bytes. Nothing about the ledger depends on who read them.
    if records is None:
        records = il.flag_records(list(il.iter_meter_days()))
    map_rows = list(csv.DictReader(map_csv.open(encoding="utf-8")))
    user_bus = {r["user_id"]: int(r["destination_bus"]) for r in map_rows
                if r["candidate_status"] == "mapped"}

    classes = bus_classification(map_csv)
    assert_classification_complete(classes)
    policies = bus_policies(scenario, classes)

    if formal:
        il.require_unit_basis()
    unit_status = il.unit_status()

    rated = native_bus_load()
    shape = native_shape()
    industry, exclusions, bus_users = industry_curves(date, user_bus, records)

    scenario_dir = out_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    name = "bus_loads_96x33.csv" if formal else "bus_loads_96x33_audit.csv"
    path = scenario_dir / name

    rows = []
    for bus in ALL_BUSES:
        for t in range(il.N_SLOTS):
            policy = policies[bus]
            native_mw = 0.0 if bus == SLACK_BUS else rated[bus] * shape[t]
            industry_mw = None
            if bus in industry:
                industry_mw = float(industry[bus][t])
            elif policy != "preserve":
                industry_mw = None
            final, status = compose(native_mw, industry_mw, policy)
            rows.append({
                "scenario": scenario,
                "date": str(date),
                "bus": bus,
                "time_index": t,
                "policy": policy,
                "native_mw": f"{native_mw:.9f}",
                "industry_mw": "" if industry_mw is None else f"{industry_mw:.9f}",
                "final_mw": "" if final is None else f"{final:.9f}",
                "industry_status": status,
                "unit_status": unit_status,
                "reading_unit": il.READING_UNIT,
                "mapping_version": "industry_bus_map_v2",
                "native_source": "build_base_network:net.load.p_mw x "
                                 "_make_load_profile(shape='native')",
                "exclusion_reason": ("" if final is not None else
                                     "the industry term has no admissible data "
                                     "for this period"),
            })
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    digest = il.file_sha256(path)
    manifest = {
        "scenario": scenario,
        "date": str(date),
        "ledger_kind": "formal" if formal else "audit",
        "unit_status": unit_status,
        "reading_unit": il.READING_UNIT,
        "reading_unit_evidence": il.READING_UNIT_EVIDENCE,
        "reading_unit_source": il.READING_UNIT_SOURCE,
        "mapping_version": "industry_bus_map_v2",
        "mapping_sha256": il.file_sha256(map_csv),
        "workbook_sha256": il.file_sha256(il.resolve_workbook()),
        "buses": list(ALL_BUSES),
        "slots_per_day": il.N_SLOTS,
        "rows": len(rows),
        # How this date was arrived at, when the caller took it by rule rather
        # than naming it. A ledger whose date cannot be re-derived is a ledger
        # with an unexplained constant in it.
        "selection_rule": (selection or {}).get("rule"),
        "selection_evidence": (selection or {}).get("evidence"),
        "policies": {str(b): p for b, p in sorted(policies.items())},
        "buses_with_a_user": sorted(c["bus"] for c in classes if c["has_user"]),
        "buses_with_no_user": sorted(c["bus"] for c in classes
                                     if not c["has_user"]),
        # The users that actually contributed a curve on this date, and where
        # they landed. A bus count without its users is a claim nobody can
        # re-derive, and `buses_with_a_user` comes from the mapping file while
        # these come from the day's own data -- the two can differ, which is
        # exactly what the manifest should make visible.
        "users_with_data": sorted({u for us in bus_users.values() for u in us}),
        "users_per_bus": {str(b): us for b, us in sorted(bus_users.items())},
        "buses_with_data_on_this_date": sorted(bus_users),
        "industry_aggregation": {
            "method": ("per-slot average active power, converted once per "
                       "meter-day and summed by user then by bus"),
            "reading_conversion": "value x 综合倍率 -> kW, / 1000 -> MW",
            "slot_length_factor": None,
            "note": ("a slot reading is a mean power, not an energy, so no "
                     "60/SLOT_MINUTES factor is applied; see S1 and "
                     "reports/industry_load_rebuild.md"),
        },
        "excluded_cells": len(exclusions),
        "compositions_excluded_no_data": sum(
            1 for r in rows if r["industry_status"] == "excluded_no_data"),
        "ledger_sha256": digest,
        "dataset_version": f"node-loads-{'formal' if formal else 'audit'}"
                           f"/{digest[:12]}",
    }
    (scenario_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": rows, "classes": classes, "manifest": manifest,
            "exclusions": exclusions, "path": path}


#: The summary's columns, in order. `scenario` is the key.
SUMMARY_COLUMNS = ("scenario", "date", "ledger_kind", "unit_status", "buses",
                   "buses_with_a_user", "buses_with_no_user", "rows",
                   "n_replace", "n_preserve", "n_overlay",
                   "periods_without_industry_data", "final_energy_mwh",
                   "sum_of_bus_peaks_mw", "mapping_version", "dataset_version")


def upsert_summary_row(path: Path, row: Sequence) -> None:
    """One row per scenario, replaced in place rather than overwritten.

    The file is a summary of every ledger on disk, so rebuilding one scenario
    must not delete the others -- overwriting it with the last build made the
    file a report of the last command run rather than of what exists. Rebuilding
    a scenario replaces its own row and leaves the rest alone.

    A file whose header is not `SUMMARY_COLUMNS` is replaced rather than
    appended to: rows under a header they do not belong to read as data.
    """
    rows: "collections.OrderedDict[str, Sequence]" = collections.OrderedDict()
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            if next(reader, None) == list(SUMMARY_COLUMNS):
                for existing in reader:
                    if existing:
                        rows[existing[0]] = existing
    rows[str(row[0])] = list(row)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(rows.values())


def write_reports(result: dict, reports_dir: Path) -> Tuple[Path, Path]:
    """`bus_load_composition.csv` and `node_loads_summary.csv`.

    Both are derived from the ledger and the mapping file on every run. Neither
    is maintained by hand, which is how a bus came to be described as both
    carrying a user and being empty.

    The two differ in scope: the composition table is one ledger's 33 buses and
    is replaced whole, while the summary is one row per ledger on disk and is
    merged.
    """
    rows = result["rows"]
    classes = {c["bus"]: c for c in result["classes"]}
    per_bus: Dict[int, dict] = collections.defaultdict(
        lambda: {"mwh": 0.0, "peak_mw": 0.0, "blank": 0, "native_mwh": 0.0})
    for row in rows:
        bus = int(row["bus"])
        slot = per_bus[bus]
        slot["native_mwh"] += float(row["native_mw"]) * il.HOURS_PER_SLOT
        if row["final_mw"] == "":
            slot["blank"] += 1
            continue
        value = float(row["final_mw"])
        slot["mwh"] += value * il.HOURS_PER_SLOT
        slot["peak_mw"] = max(slot["peak_mw"], value)

    reports_dir.mkdir(parents=True, exist_ok=True)
    comp = reports_dir / "bus_load_composition.csv"
    with comp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bus", "role", "is_mapping_target", "status", "has_user",
                    "n_users", "n_meters", "industries", "policy",
                    "native_mwh", "final_mwh", "final_peak_mw",
                    "periods_without_industry_data"])
        for bus in ALL_BUSES:
            c, slot = classes[bus], per_bus[bus]
            w.writerow([bus, c["role"], c["is_mapping_target"], c["status"],
                        c["has_user"], c["n_users"], c["n_meters"],
                        c["industries"], result["manifest"]["policies"][str(bus)],
                        f"{slot['native_mwh']:.6f}", f"{slot['mwh']:.6f}",
                        f"{slot['peak_mw']:.6f}", slot["blank"]])

    summary = reports_dir / "node_loads_summary.csv"
    m = result["manifest"]
    policies = collections.Counter(m["policies"].values())
    upsert_summary_row(summary, [
        m["scenario"], m["date"], m["ledger_kind"], m["unit_status"],
        len(m["buses"]), len(m["buses_with_a_user"]),
        len(m["buses_with_no_user"]), m["rows"], policies["replace"],
        policies["preserve"], policies["overlay"],
        m["compositions_excluded_no_data"],
        f"{sum(s['mwh'] for s in per_bus.values()):.4f}",
        f"{sum(s['peak_mw'] for s in per_bus.values()):.4f}",
        m["mapping_version"], m["dataset_version"]])
    return comp, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build the 33-bus node-load ledger for one scenario.")
    p.add_argument("--scenario", default="typical_day_industry")
    p.add_argument("--date", default=None,
                   help="ISO date of the ledger. Required: no date is "
                        "designated by default, because the completeness rule "
                        "that used to pick one is biased toward high-load days "
                        "(see reports/s3_date_selection_audit.md)")
    p.add_argument("--audit-dates", action="store_true",
                   help="write the per-date coverage audit and exit")
    p.add_argument("--date-from-audit", action="store_true",
                   help="take the date from the coverage audit by rule: the "
                        "earliest date with every mapped bus covered on all 96 "
                        f"periods ({LOAD_DATE_RULE}). Reads "
                        "reports/date_selection_audit.csv, which "
                        "--audit-dates writes")
    p.add_argument("--mapping", default="data/processed/"
                                       "industry_meter_bus_map_v2.csv")
    p.add_argument("--out", default="data/processed/node_loads")
    p.add_argument("--formal", action="store_true",
                   help="write the formal MW ledger; refused unless the reading "
                        "unit is verified")
    args = p.parse_args(argv)

    if args.audit_dates:
        map_rows = list(csv.DictReader(Path(args.mapping).open(encoding="utf-8")))
        user_bus = {r["user_id"]: int(r["destination_bus"]) for r in map_rows
                    if r["candidate_status"] == "mapped"}
        records = il.flag_records(list(il.iter_meter_days()))
        audit = date_audit(records, user_bus, sorted(set(user_bus.values())))
        path = write_date_audit(audit, Path("reports"))
        complete = [r for r in audit if r["load_complete"]]
        print(f"dates audited              {len(audit)}")
        print(f"  with every mapped bus covered   {len(complete)}")
        print(f"  with at least one bus missing   {len(audit) - len(complete)}")
        print(f"  dates with a dated real price curve 0 "
              f"(price_dates_available is False on every row)")
        print(f"wrote {path}")
        return 0

    selected = None
    if args.date_from_audit:
        if args.date:
            raise SystemExit("--date and --date-from-audit are alternatives; "
                             "give one")
        audit_path = Path("reports/date_selection_audit.csv")
        if not audit_path.exists():
            raise SystemExit(
                f"{audit_path} not found; run --audit-dates first, so the "
                f"date is taken from a coverage audit rather than assumed")
        audit = list(csv.DictReader(audit_path.open(encoding="utf-8")))
        for row in audit:
            row["load_complete"] = row["load_complete"] == "True"
        selected = select_load_date(audit)
    if not args.date and selected is None:
        raise SystemExit(
            "no --date given. No date is designated by default: the "
            "completeness rule that used to choose one is biased toward "
            "high-load days. Run --audit-dates, then name a date explicitly "
            "or take one by rule with --date-from-audit.")
    date = args.date or selected["date"]

    selection = None
    if selected is not None:
        selection = {
            "rule": LOAD_DATE_RULE,
            "evidence": {
                "audit": "reports/date_selection_audit.csv",
                "audit_sha256": il.file_sha256(
                    Path("reports/date_selection_audit.csv")),
                "date": selected["date"],
                "buses_covered": int(selected["buses_covered"]),
                "buses_mapped": int(selected["buses_mapped"]),
                "readings": int(selected["readings"]),
                "readings_not_convertible":
                    int(selected["readings_not_convertible"]),
                "why": ("ascending order over the calendar; the first date on "
                        "which every mapped bus is covered for all 96 periods. "
                        "It is not chosen on any quantity of load and is not "
                        "described as a typical day."),
            },
        }
    result = build(args.scenario, date, Path(args.out), Path(args.mapping),
                   formal=args.formal, selection=selection)
    manifest = result["manifest"]
    classes = result["classes"]
    comp, summary = write_reports(result, Path("reports"))

    print(f"{args.scenario} on {date}   ledger kind: {manifest['ledger_kind']}")
    print(f"  unit status          {manifest['unit_status']}  "
          f"(evidence: {manifest['reading_unit_evidence']})")
    print(f"  buses                 {len(manifest['buses'])} "
          f"(0..32; bus 0 is the slack)")
    print(f"  buses with a user     {len(manifest['buses_with_a_user'])}")
    print(f"  buses with no user    {len(manifest['buses_with_no_user'])}: "
          f"{manifest['buses_with_no_user']}")
    print(f"  ledger rows           {len(result['rows']):,} (33 x 96)")
    policy_counts = collections.Counter(manifest["policies"].values())
    print(f"  policies              {dict(policy_counts)}")
    print(f"  excluded cells        {manifest['excluded_cells']}")
    print(f"  rows with no industry data on a replace/overlay bus: "
          f"{manifest['compositions_excluded_no_data']}")
    print(f"\nwrote {result['path']}")
    print(f"wrote {result['path'].parent / 'manifest.json'}")
    print(f"wrote {comp}")
    print(f"wrote {summary}")
    if not args.formal:
        print("\nThis is an AUDIT ledger: the industry term carries the "
              "declared unit and its unverified status. It is not published as "
              "the network's formal absolute load.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
