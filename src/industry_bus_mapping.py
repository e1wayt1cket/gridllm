# industry_bus_mapping.py
"""Which bus each qualified **user** is mapped onto.

The unit is the user, not the meter. A user's meters are circuits on one
account and are summed before anything else happens, so a user is never counted
twice and never split across nodes. This supersedes the meter-level
`R3_meter_spread` rule, under which one user's meters could land on several
buses and a single account would appear as several loads.

**This module has no objective.** It does not read a price, a nodal LMP, an
arbitrage margin or an RL return, and it cannot: importing any of them is a
test failure. A mapping chosen to widen price spreads or improve storage revenue
would be a result tuned into the input, and the mapping is an input.

The mapping is also a **simulation mapping**. This feeder carries no
user-to-node connection table, and no row here is a claim about where any user
is physically connected. Every row is checkable — one user, one bus, that bus a
load bus, and that bus assigned the user's own industry by
`config/scenarios.yaml` — and nothing more is asserted.

A bus that receives no user is reported as carrying no mapped load. It is not
given a fabricated user, and it is not removed from the network: whether it
keeps its native IEEE33 load is a per-scenario policy decided in S3, not here.

Usage:
    PYTHONPATH=src python src/industry_bus_mapping.py --out data/processed
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

import industry_load as il
import config_loader as cfg
import map_industry_meters as mim

#: The one rule this module implements. Round robin over each industry's
#: declared buses, in ascending anonymous user-key order.
RULE_USER_ROUND_ROBIN = "U1_user_round_robin"

#: Why a user is not mapped. Each is recorded on its row, never dropped.
REASON_OK = ""
REASON_SPLIT_INDUSTRY = ("the user's meters span more than one industry, so no "
                         "single industry's buses can take the whole account")
REASON_NO_BUS = ("the user's industry is assigned no bus in the scenario's "
                 "industry_map")
REASON_EXCLUDED_INDUSTRY = ("the user's industry is listed in "
                            "`excluded_industries` and must not shape a load "
                            "bus")


def load_declared_load_buses() -> List[int]:
    """Every bus the configuration declares as a load bus, slack excluded.

    Read from `load_types` rather than from a built network so this module does
    not need pandapower to answer an identity question. The slack is named in
    the same configuration; it carries no load and is not a mapping target.
    """
    declared = set()
    for _name, spec in (cfg.get_default("load_types") or {}).items():
        declared.update(int(b) for b in (spec.get("buses") or []))
    declared.discard(0)
    return sorted(declared)


def qualified_users(records: Sequence[il.MeterDay]) -> Dict[str, dict]:
    """One entry per user that contributes at least one qualified block.

    Counted at the user, which is the unit the mapping acts on: a user with
    three meters and two qualified ones is one unit with two meters, not three
    and not two.
    """
    in_pool = il.qualified_block_keys(records)
    users: Dict[str, dict] = {}
    for md in records:
        if (md.meter_key, md.date) not in in_pool:
            continue
        entry = users.setdefault(md.user_key, {
            "user_id": md.user_key,
            "industries": set(),
            "meters": set(),
            "blocks": set(),
        })
        entry["industries"].add(md.industry)
        entry["meters"].add(md.meter_key)
        entry["blocks"].add((md.meter_key, md.date))
    return users


def assign_users(users: Dict[str, dict],
                 industry_buses: Dict[str, List[int]],
                 excluded: Sequence[str],
                 load_buses: Sequence[int]) -> List[dict]:
    """One bus per user, deterministically, with the reason recorded.

    The tie-break is the ascending anonymous `user_key` order and nothing else:
    it does not read a name, a location, a date or a magnitude, so the same
    input gives the same mapping whatever order the records arrive in.
    """
    by_industry: Dict[str, List[dict]] = collections.defaultdict(list)
    for entry in users.values():
        by_industry[sorted(entry["industries"])[0]].append(entry)

    allowed = set(int(b) for b in load_buses)
    rows: List[dict] = []
    for industry in sorted(by_industry):
        members = sorted(by_industry[industry], key=lambda e: e["user_id"])
        buses = sorted(int(b) for b in (industry_buses.get(industry) or []))
        for position, entry in enumerate(members):
            reason = REASON_OK
            bus: Optional[int] = None
            if len(entry["industries"]) > 1:
                reason = REASON_SPLIT_INDUSTRY
            elif industry in set(excluded):
                reason = REASON_EXCLUDED_INDUSTRY
            elif not buses:
                reason = REASON_NO_BUS
            else:
                candidate = buses[position % len(buses)]
                if candidate not in allowed:
                    reason = (f"bus {candidate} is not a declared load bus of "
                              "this network")
                else:
                    bus = candidate
            rows.append({
                "user_id": entry["user_id"],
                "industry": industry,
                "n_meters": len(entry["meters"]),
                "meter_ids": ";".join(sorted(entry["meters"])),
                "destination_bus": bus if bus is not None else "",
                "mapping_rule": RULE_USER_ROUND_ROBIN,
                "mapping_reason": reason,
                "candidate_status": "mapped" if bus is not None else "unmapped",
                "simulation_mapping": True,
                "qualified_blocks": len(entry["blocks"]),
            })
    return sorted(rows, key=lambda r: r["user_id"])


def rule_version() -> str:
    """A digest over what the mapping decision depends on, so a map and the
    rule that produced it cannot drift apart unnoticed."""
    payload = json.dumps({
        "rule": RULE_USER_ROUND_ROBIN,
        "unit": "user",
        "order_key": "user_key ascending",
        "bus_selection": "position modulo the industry's declared buses",
        "scenario": mim.SCENARIO,
        "excluded_industries": sorted(
            mim.load_industry_map()[1]),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assert_invariants(rows: Sequence[dict], load_buses: Sequence[int],
                      users: Dict[str, dict]) -> None:
    """The properties the map must have, checked rather than assumed."""
    mapped = [r for r in rows if r["candidate_status"] == "mapped"]
    ids = [r["user_id"] for r in mapped]
    if len(ids) != len(set(ids)):
        raise AssertionError(
            "a user was mapped more than once; the mapping is not a function")
    if len(rows) != len(users):
        raise AssertionError(
            f"{len(users)} qualified users but {len(rows)} rows; a user was "
            "dropped instead of reported")
    allowed = set(int(b) for b in load_buses)
    for row in mapped:
        if int(row["destination_bus"]) not in allowed:
            raise AssertionError(
                f"{row['user_id']} was mapped to {row['destination_bus']}, "
                "which is not a load bus of this network")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map each qualified user to exactly one load bus.")
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--csv-name", default="industry_meter_bus_map_v2.csv")
    parser.add_argument("--manifest-name",
                        default="industry_meter_bus_map_v2_manifest.json")
    args = parser.parse_args(argv)

    workbook = il.resolve_workbook(args.workbook)
    industry_buses, excluded = mim.load_industry_map()
    load_buses = load_declared_load_buses()

    records = il.flag_records(list(il.iter_meter_days(args.workbook, args.sheet)))
    users = qualified_users(records)
    rows = assign_users(users, industry_buses, excluded, load_buses)
    assert_invariants(rows, load_buses, users)

    mapped = [r for r in rows if r["candidate_status"] == "mapped"]
    buses_used = sorted({int(r["destination_bus"]) for r in mapped})
    ids = {m for r in rows for m in r["meter_ids"].split(";") if m}
    blocks = sum(r["qualified_blocks"] for r in rows)

    print(f"Rule {RULE_USER_ROUND_ROBIN}   version {rule_version()}")
    print(f"  qualified users   {len(users)}")
    print(f"  mapped users      {len(mapped)}")
    print(f"  unmapped users    {len(rows) - len(mapped)}")
    for reason, n in collections.Counter(
            r["mapping_reason"] for r in rows if r["candidate_status"] != "mapped"
    ).most_common():
        print(f"      {n:>3}  {reason}")
    print(f"  meters involved   {len(ids)}")
    print(f"  qualified blocks  {blocks:,}")
    print(f"  load buses        {len(load_buses)}")
    print(f"  buses receiving a user  {len(buses_used)}")
    empty = sorted(set(load_buses) - set(buses_used))
    print(f"  buses with no mapped user ({len(empty)}): {empty}")
    print("      reported, not removed: whether they keep their native load "
          "is a per-scenario policy (S3)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / args.csv_name
    columns = ["user_id", "industry", "n_meters", "meter_ids",
               "destination_bus", "mapping_rule", "mapping_reason",
               "candidate_status", "simulation_mapping", "qualified_blocks"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    digest = il.file_sha256(csv_path)
    manifest = {
        "rule": RULE_USER_ROUND_ROBIN,
        "mapping_rule_version": rule_version(),
        "unit": "user",
        "scenario": mim.SCENARIO,
        "simulation_mapping": True,
        "frozen": True,
        "supersedes": ["industry_meter_bus_map_frozen.csv", "D7 R3_meter_spread"],
        "source_workbook": str(workbook),
        "source_workbook_sha256": il.file_sha256(workbook),
        "source_workbook_bytes": Path(workbook).stat().st_size,
        "mapping_sha256": digest,
        "dataset_version": f"industry-bus-map-v2/{digest[:12]}",
        "counts": {
            "qualified_users": len(users),
            "mapped_users": len(mapped),
            "unmapped_users": len(rows) - len(mapped),
            "meters": len(ids),
            "qualified_blocks": blocks,
            "declared_load_buses": len(load_buses),
            "buses_receiving_a_user": len(buses_used),
            "buses_with_no_user": empty,
        },
        "users_per_bus": dict(collections.Counter(
            int(r["destination_bus"]) for r in mapped)),
    }
    (out / args.manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {csv_path}")
    print(f"wrote {out / args.manifest_name}")
    print(f"  dataset {manifest['dataset_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
