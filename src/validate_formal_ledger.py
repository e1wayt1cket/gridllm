# validate_formal_ledger.py
"""May this ledger be published as the network's formal absolute load?

Nothing here trusts the builder. Every check re-reads the produced artefacts --
the ledger CSV and its manifest -- and re-derives the composition from the two
terms the ledger itself records, so a ledger that was written wrong is caught by
reading it rather than by asking the code that wrote it.

A **formal** ledger must pass every check. An **audit** ledger is exempt: it is
allowed to carry missing values and an unverified unit, because its whole
purpose is to be examined before those things are settled. The distinction is
the point -- an audit ledger may hold a gap, a formal one may not, and nothing
may quietly turn the first into the second.

Usage:
    PYTHONPATH=src python src/validate_formal_ledger.py \
        --ledger data/processed/node_loads/<scenario>/bus_loads_96x33.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import industry_load as il
import single_day_data as sdd
from build_industry_node_loads import ALL_BUSES, POLICIES, SLACK_BUS

#: The date rules this validator knows. Imported rather than restated so a new
#: case type cannot exist in one place and not the other.
CASE_TYPES = sdd.CASE_TYPES

#: Every reason a ledger may not be published as the formal load.
CODES = {
    "unit_not_verified": ("the reading unit rests on no admissible basis, or "
                          "the artefact and the code disagree about which"),
    "bus_coverage": "the ledger does not cover all 33 buses with 96 periods each",
    "missing_value": "a period has no final load value",
    "policy_incomplete": "a bus has no policy, or one that is not recognised",
    "duplicate_row": "a bus and period appears more than once",
    "formula_mismatch": "the final load is not the composition its policy states",
    "multiple_dates": "the ledger spans more than one date",
    "price_not_date_aligned": ("the price curve carries no date to align with, "
                               "and this case type needs one"),
    "case_type_unstated": ("no recognised case type was given, so the date rule "
                           "cannot be chosen; the strict one is applied"),
    "synthetic_source": "a synthetic source is recorded against the ledger",
    "slack_has_load": "the slack bus carries a load",
}


def _violation(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail, "meaning": CODES[code]}


def validate(ledger_csv: Path, manifest_path: Path,
             price_dates_available: bool = False,
             case_type: Optional[str] = None) -> List[dict]:
    """Every reason this ledger may not be the formal absolute load.

    An empty list means the ledger passed. It does not mean the figures are
    right -- it means they are internally consistent, complete and attributable.

    `case_type` decides the date rule, and it is **not** optional in effect:
    whether two different dates are acceptable depends on what the case claims
    to be. A caller that states none, or one this module does not know, gets the
    strict rule *and* a `case_type_unstated` violation naming the omission --
    never the permissive rule by default. A missing date must not be able to
    loosen a check.

    | case type | date rule |
    |---|---|
    | `cross_date_diagnostic` | the two dates may differ; the pairing itself is checked by `single_day_data.load_case` |
    | `same_day_replay` | the load and the price must be the same actual date |
    | anything else | strict, and reported |
    """
    rows = list(csv.DictReader(ledger_csv.open(encoding="utf-8")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: List[dict] = []

    # 1. The unit gate, asked of the artefact rather than of the process.
    #
    # Either admissible basis passes -- an external source or a recorded
    # project ruling -- and the two are not confused: the manifest's status
    # must follow from the manifest's own basis, and that basis must be the one
    # the code is working under. What fails is a basis nothing stands behind, a
    # status that does not follow from the basis beside it, and a stale
    # artefact recording a basis other than the current one.
    reported = manifest.get("reading_unit_evidence")
    recorded_basis = il.canonical_basis(reported)
    code_basis = il.canonical_basis(il.READING_UNIT_EVIDENCE)
    declared = manifest.get("unit_status")
    if recorded_basis is None or code_basis is None:
        out.append(_violation(
            "unit_not_verified",
            f"the basis names nothing admissible: the manifest records "
            f"{reported!r} and the code is working under "
            f"{il.READING_UNIT_EVIDENCE!r}; expected one of {il.UNIT_BASES}"))
    elif declared != il.status_for_basis(recorded_basis):
        out.append(_violation(
            "unit_not_verified",
            f"the manifest's status {declared!r} does not follow from its own "
            f"basis {recorded_basis!r}, which implies "
            f"{il.status_for_basis(recorded_basis)!r}"))
    elif recorded_basis != code_basis:
        out.append(_violation(
            "unit_not_verified",
            f"the ledger records basis {recorded_basis!r} while the code is "
            f"working under {code_basis!r}; this artefact predates the current "
            f"vocabulary and has to be rebuilt rather than reinterpreted"))

    # 2. Coverage: every bus, every period, exactly once.
    seen: Dict[tuple, int] = collections.Counter(
        (int(r["bus"]), int(r["time_index"])) for r in rows)
    duplicated = [k for k, n in seen.items() if n > 1]
    if duplicated:
        out.append(_violation(
            "duplicate_row",
            f"{len(duplicated)} bus/period pairs appear more than once, e.g. "
            f"{sorted(duplicated)[:5]}"))
    per_bus = collections.Counter(int(r["bus"]) for r in rows)
    wrong = {b: n for b, n in per_bus.items() if n != il.N_SLOTS}
    missing_buses = sorted(set(ALL_BUSES) - set(per_bus))
    if wrong or missing_buses:
        out.append(_violation(
            "bus_coverage",
            f"buses with the wrong period count: {wrong}; buses absent: "
            f"{missing_buses}"))

    # 3. No period may be blank, and none may be silently carried.
    blank = [r for r in rows if r["final_mw"].strip() == ""]
    if blank:
        buses = sorted({int(r["bus"]) for r in blank})
        out.append(_violation(
            "missing_value",
            f"{len(blank)} periods have no final load, on bus(es) {buses}. A "
            "formal ledger may not accept a gap: it must be filled from a "
            "source that states it, or the date must be rejected. Zero-filling "
            "or carrying the previous value would invent a measurement"))

    # 4. Every bus carries one recognised policy.
    bad_policy = sorted({r["policy"] for r in rows if r["policy"] not in POLICIES})
    policy_buses = {int(r["bus"]) for r in rows if r["policy"] in POLICIES}
    if bad_policy or set(policy_buses) != set(ALL_BUSES):
        out.append(_violation(
            "policy_incomplete",
            f"unrecognised policies {bad_policy}; buses without a policy "
            f"{sorted(set(ALL_BUSES) - policy_buses)}"))

    # 5. The composition must be the one the policy states.
    mismatched = []
    for row in rows:
        policy = row["policy"]
        native = float(row["native_mw"])
        industry = (float(row["industry_mw"])
                    if row["industry_mw"].strip() else None)
        final = row["final_mw"].strip()
        if policy == "preserve":
            expected = native
        elif industry is None:
            expected = None
        elif policy == "replace":
            expected = industry
        else:
            expected = native + industry
        if expected is None or final == "":
            continue
        if abs(float(final) - expected) > 1e-6:
            mismatched.append((int(row["bus"]), int(row["time_index"])))
    if mismatched:
        out.append(_violation(
            "formula_mismatch",
            f"{len(mismatched)} rows do not equal their policy's formula, e.g. "
            f"{mismatched[:5]}"))

    # 6. One date. A ledger that spans two has summed a history into a day.
    dates = sorted({r["date"] for r in rows})
    if len(dates) != 1:
        out.append(_violation(
            "multiple_dates",
            f"the ledger spans {len(dates)} dates: {dates[:5]}"))

    # 7. The date rule, which follows from the case type rather than from
    # whether a date happened to be present.
    known = case_type in CASE_TYPES
    if not known:
        out.append(_violation(
            "case_type_unstated",
            f"case_type is {case_type!r}; expected one of {CASE_TYPES}. The "
            f"date rule depends on it, so the strict one is applied rather "
            f"than the permissive one by default"))
    same_day_required = (not known) or case_type == "same_day_replay"
    if same_day_required and not price_dates_available:
        out.append(_violation(
            "price_not_date_aligned",
            "the load and the price must belong to the same actual date for "
            "this case type, and no real price curve carries a date to align "
            "to. Pair them as a cross_date_diagnostic if a real load shape "
            "priced against a real price shape is what is wanted -- that case "
            "records both dates and claims no historical replay"))

    # 8. Nothing synthetic may be recorded.
    blob = json.dumps(manifest).lower()
    for marker in ("synthetic", "synthetic_v1", "library"):
        if marker in blob and "supersedes" not in blob.split(marker)[0][-40:]:
            out.append(_violation(
                "synthetic_source",
                f"the manifest mentions {marker!r}"))
            break

    # 9. The slack carries nothing.
    slack = [r for r in rows if int(r["bus"]) == SLACK_BUS
             and float(r["final_mw"] or 0) != 0.0]
    if slack:
        out.append(_violation(
            "slack_has_load", f"{len(slack)} periods give the slack a load"))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Decide whether a ledger may be the formal absolute load.")
    p.add_argument("--ledger", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--price-dates-available", action="store_true")
    p.add_argument("--case-type", default=None,
                   choices=list(CASE_TYPES),
                   help="which date rule this case carries. Required in "
                        "effect: omitting it applies the strict rule and is "
                        "reported as `case_type_unstated`")
    args = p.parse_args(argv)

    ledger = Path(args.ledger)
    manifest = Path(args.manifest) if args.manifest else ledger.parent / "manifest.json"
    if "audit" in ledger.name:
        print(f"{ledger.name} is an audit ledger: it is exempt from this "
              "validator by design, and this validator is for the formal one.")
    violations = validate(ledger, manifest, args.price_dates_available,
                          args.case_type)
    if not violations:
        print("PASS: this ledger may be the formal absolute load.")
        return 0
    print(f"REJECTED: {len(violations)} reason(s).")
    for v in violations:
        print(f"  [{v['code']}] {v['meaning']}")
        print(f"      {v['detail']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
