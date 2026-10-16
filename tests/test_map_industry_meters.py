# test_map_industry_meters.py
"""The candidate mapping and the coverage audit must not overstate coverage.

Two properties carry the whole audit. A cell is summable only when it is a
complete active-power day, so a milder condition must never mask a harder one --
an all-zero cell that happens to be a duplicate is still an all-zero cell. And a
bus is covered on a date once, however many meters it carries, because the
question "how many buses can be populated on one day" is about buses and not
about meter-days wearing a bus's name.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import industry_load as il
import map_industry_meters as mim


def _cell(records=1, present=(96,), flags=()):
    return {"records": records, "present": list(present), "flags": set(flags)}


# ---------------------------------------------------------------------------
# Cell classification
# ---------------------------------------------------------------------------

def test_a_missing_cell_is_absent_and_never_summable():
    """`absent` and `excluded` are different answers: no record at all is not
    the same defect as a record that cannot be summed, and the report separates
    them so a gap in the calendar is not read as a bad reading."""
    info = mim.classify_cell(None)
    assert info["status"] == "absent"
    assert info["usability"] == "missing"
    assert info["records"] == 0 and info["slots_present"] == 0


def test_dimensions_are_only_read_from_a_frozen_capacity_failure():
    assert mim.classify_cell(_cell(present=(0,), flags=["all_zero"]))["status"] \
        == "all_zero"
    assert mim.classify_cell(_cell(present=(0,),
                                   flags=["all_zero", "missing_slots"]))["status"] \
        == "all_zero"          # a milder condition never masks a harder one
    assert mim.classify_cell(_cell(
        flags=["all_zero", "duplicate_conflict"]))["status"] == "conflicting"


def test_no_value_is_ever_zero_filled_or_interpolated():
    """A short day is partial and not summable; it is never padded to 96."""
    info = mim.classify_cell(_cell(present=(95,)))
    assert info["status"] == "partial"
    assert info["slots_present"] == 95
    assert info["usability"] == "excluded"


def test_only_a_full_active_power_day_is_summable():
    info = mim.classify_cell(_cell(present=(96, 96)))
    assert info["status"] == "complete"
    assert info["usability"] == "summable"
    for flags in (["other_phase"], ["placeholder"], ["empty_identity"]):
        assert mim.classify_cell(_cell(flags=flags))["usability"] == "excluded"


def test_the_worst_slot_count_decides_a_blocks_coverage():
    """Two records for one meter-day: the thinner one is the block's coverage,
    so a block cannot be called complete on the strength of its best copy."""
    info = mim.classify_cell(_cell(records=2, present=(96, 40)))
    assert info["slots_present"] == 40
    assert info["status"] == "partial"


# ---------------------------------------------------------------------------
# Candidate rules
# ---------------------------------------------------------------------------

def _facts(pairs):
    out = {}
    for meter, user, industry, days in pairs:
        out[meter] = {"meter_id": meter, "industries": {industry},
                      "users": {user}, "days_in_pool": set(days)}
    return out


def test_every_mapped_meter_gets_exactly_one_bus():
    facts = _facts([("M1", "U1", "化工业", [1]), ("M2", "U2", "商业综合体", [1]),
                    ("M3", "U3", "商业综合体", [1])])
    buses = {"化工业": [23], "商业综合体": [1, 2, 3]}
    for rule in (mim.R1, mim.R2, mim.R3):
        rows = mim.assign_buses(facts, buses, [], rule)
        mapped = [r for r in rows if r["candidate_status"] == "mapped"]
        assert len(mapped) == len({r["meter_id"] for r in mapped}), rule
        for row in mapped:
            assert row["destination_bus"] in sum(buses.values(), []), rule


def test_a_meter_never_lands_on_another_industrys_bus():
    """The bus a meter takes must be one its own industry was assigned."""
    facts = _facts([("M1", "U1", "商业综合体", [1]), ("M2", "U2", "钢铁", [1])])
    buses = {"商业综合体": [1, 2], "钢铁": [16]}
    for rule in (mim.R1, mim.R2, mim.R3):
        for row in mim.assign_buses(facts, buses, [], rule):
            if row["candidate_status"] != "mapped":
                continue
            expected = buses[row["industry"]]
            assert row["destination_bus"] in expected, (rule, row)


def test_a_multi_bus_industry_leaves_R1_unmapped_rather_than_guessing():
    facts = _facts([("M1", "U1", "商业综合体", [1])])
    rows = mim.assign_buses(facts, {"商业综合体": [1, 2, 3]}, [], mim.R1)
    assert rows[0]["candidate_status"] == "unmapped"
    assert "does not choose" in rows[0]["mapping_reason"]


def test_an_excluded_industry_is_never_assigned_a_bus():
    facts = _facts([("M1", "U1", "5G基站", [1])])
    rows = mim.assign_buses(facts, {"5G基站": [4]}, ["5G基站"], mim.R3)
    assert rows[0]["destination_bus"] == ""
    assert rows[0]["candidate_status"] == "unmapped"


def test_a_user_spans_one_bus_under_R2_but_not_under_R3():
    """The property that separates the two candidates, asserted directly."""
    facts = _facts([("M1", "U1", "商业综合体", [1]),
                    ("M2", "U1", "商业综合体", [1]),
                    ("M3", "U2", "商业综合体", [1])])
    buses = {"商业综合体": [1, 2]}
    r2 = {r["meter_id"]: r["destination_bus"]
          for r in mim.assign_buses(facts, buses, [], mim.R2)}
    assert r2["M1"] == r2["M2"]            # one user, one bus
    r3 = {r["meter_id"]: r["destination_bus"]
          for r in mim.assign_buses(facts, buses, [], mim.R3)}
    assert r3["M1"] != r3["M2"]            # adjacent meters, adjacent buses


def test_every_candidate_row_is_marked_a_simulation_mapping():
    facts = _facts([("M1", "U1", "钢铁", [1])])
    for rule in (mim.R1, mim.R2, mim.R3):
        for row in mim.assign_buses(facts, {"钢铁": [16]}, [], rule):
            assert row["simulation_mapping"] is True
            assert row["mapping_rule"] == rule


# ---------------------------------------------------------------------------
# Per-bus coverage
# ---------------------------------------------------------------------------

def test_a_bus_is_counted_once_per_date_however_many_meters_it_carries():
    """The counting unit here is the bus. Adding cells instead would let a bus
    with fourteen meters look like fourteen buses on the same day."""
    facts = _facts([(f"M{i}", f"U{i}", "商业综合体", [1]) for i in range(1, 15)])
    rows = mim.assign_buses(facts, {"商业综合体": [2]}, [], mim.R2)
    cells = {(f"M{i}", "2021-01-01"): mim.classify_cell(_cell())
             for i in range(1, 15)}
    per_bus, used, full, per_date = mim.bus_stats(rows, cells,
                                                  ["2021-01-01"])
    assert used == [2]
    assert per_bus[2]["meters"] == 14
    assert per_bus[2]["complete_days"] == 14    # cells
    assert per_bus[2]["dates_covered"] == 1     # distinct days
    assert per_date["2021-01-01"] == 1          # distinct buses, not 14
    assert full == ["2021-01-01"]


def test_a_bus_with_no_complete_day_is_never_fully_covered():
    facts = _facts([("M1", "U1", "钢铁", [1]), ("M2", "U2", "化工业", [1])])
    rows = mim.assign_buses(facts, {"钢铁": [16], "化工业": [23]}, [], mim.R1)
    cells = {("M1", "d1"): mim.classify_cell(_cell()),
             ("M2", "d1"): mim.classify_cell(_cell(present=(0,),
                                                   flags=["all_zero"]))}
    _per_bus, used, full, per_date = mim.bus_stats(rows, cells, ["d1"])
    assert sorted(used) == [16, 23]
    assert full == []                       # bus 23 has nothing usable
    assert per_date["d1"] == 1


def test_unmapped_meters_contribute_to_no_bus():
    facts = _facts([("M1", "U1", "5G基站", [1])])
    rows = mim.assign_buses(facts, {"5G基站": [4]}, ["5G基站"], mim.R3)
    cells = {("M1", "d1"): mim.classify_cell(_cell())}
    per_bus, used, full, _ = mim.bus_stats(rows, cells, ["d1"])
    assert used == [] and per_bus == {} and full == []


# ---------------------------------------------------------------------------
# The frozen case
# ---------------------------------------------------------------------------

_BUSES = {"商业综合体": [1, 2], "纺织业": [27], "钢铁": [16],
          "水泥制造": [22], "印刷业": [29], "数据中心": [32]}


def _frozen_facts():
    return _facts([("M1", "U1", "商业综合体", [1]), ("M2", "U2", "商业综合体", [1]),
                   ("M3", "U1", "纺织业", [1]), ("M4", "U3", "钢铁", [1]),
                   ("M5", "U4", "水泥制造", [1]), ("M6", "U5", "印刷业", [1]),
                   ("M7", "U6", "数据中心", [1])])


def test_each_meter_maps_to_exactly_one_node_under_the_frozen_rule():
    rows = mim.frozen_case(_frozen_facts(), _BUSES, [])
    mapped = [r for r in rows if r["candidate_status"] == "mapped"]
    assert len(mapped) == len({r["meter_id"] for r in mapped})
    for row in mapped:
        assert row["destination_bus"] in _BUSES[row["industry"]]


def test_the_frozen_case_maps_no_load_to_an_unsupported_bus():
    rows = mim.frozen_case(_frozen_facts(), _BUSES, [])
    for row in rows:
        assert row["destination_bus"] not in mim.UNSUPPORTED_BUSES
    assert {r["destination_bus"] for r in rows
            if r["candidate_status"] == "mapped"} == {1, 2, 27, 16}


def test_a_meter_whose_only_bus_is_unsupported_is_reported_not_reassigned():
    """水泥制造, 印刷业 and 数据中心 have one bus each, and it is out. Their
    meters must be withheld with a reason -- never moved to another industry's
    bus to keep the count up."""
    rows = {r["meter_id"]: r for r in
            mim.frozen_case(_frozen_facts(), _BUSES, [])}
    for meter in ("M5", "M6", "M7"):
        assert rows[meter]["candidate_status"] == "unmapped"
        assert rows[meter]["destination_bus"] == ""
        assert "no data-supported load" in rows[meter]["mapping_reason"]


def test_the_mapping_is_deterministic_whatever_order_it_is_built_in():
    """Same meters, different insertion order, byte-identical assignment.
    A mapping that depends on dict ordering is not reproducible."""
    forward = mim.frozen_case(_frozen_facts(), _BUSES, [])
    shuffled = _frozen_facts()
    items = list(shuffled.items())[::-1]
    backward = mim.frozen_case(dict(items), _BUSES, [])
    key = lambda rows: sorted(  # noqa: E731 - local comparison key
        (r["meter_id"], r["destination_bus"], r["candidate_status"])
        for r in rows)
    assert key(forward) == key(backward)
    assert forward == mim.frozen_case(_frozen_facts(), _BUSES, [])


def test_the_fabricated_day_is_not_on_the_frozen_day_axis():
    """2022-05-26 is the workbook's synthesised filler throughout, so it is
    not a measurement and never enters the formal axis."""
    assert "2022-05-26" in mim.FABRICATED_DATES
    dates = ["2022-05-25", "2022-05-26", "2022-05-27"]
    axis = [d for d in dates if d not in mim.FABRICATED_DATES]
    assert axis == ["2022-05-25", "2022-05-27"]


def test_a_conflicting_meter_day_is_withheld_whole():
    """Both records go, not the inconvenient one: picking a reading would
    invent evidence about which of two disagreeing records is the measurement."""
    cell = mim.classify_cell(_cell(records=2, present=(96, 96),
                                   flags=["duplicate_conflict"]))
    assert cell["status"] == "conflicting"
    assert cell["usability"] != "summable"
    assert cell["records"] == 2

    for date in ("2021-01-26", "2022-05-27"):
        rows = mim.exclusion_rows(
            [], _BUSES, {}, [("Ma65a1fda1669", date)])
        entry = [e for e in rows if e["kind"] == "conflicting_meter_day"][0]
        assert date in entry["key"]
        assert "withheld whole" in entry["reason"]


def test_an_all_zero_day_never_becomes_load_and_is_never_zero_filled():
    """Zero readings are excluded as unusable data, not summed as a zero
    demand: the report must not let a dead meter look like an idle site."""
    cell = mim.classify_cell(_cell(present=(0,) * 96, flags=["all_zero"]))
    assert cell["status"] == "all_zero"
    assert cell["usability"] == "excluded"
    rows = mim.exclusion_rows([], _BUSES,
                              {"M9": {"industries": {"钢铁"}, "users": {"U9"}}},
                              [])
    entry = [e for e in rows
             if e["kind"] == "unresolved_all_zero_meter"][0]
    assert entry["key"] == "M9"
    assert "neither included nor" in entry["reason"]


def test_the_exclusion_list_covers_every_reason_the_case_drops_something():
    rows = mim.frozen_case(_frozen_facts(), _BUSES, [])
    exclusions = mim.exclusion_rows(rows, _BUSES, {}, [])
    kinds = {e["kind"] for e in exclusions}
    assert kinds == {"unmapped_meter", "unsupported_bus", "fabricated_date"}
    assert len([e for e in exclusions if e["kind"] == "unsupported_bus"]) \
        == len(mim.UNSUPPORTED_BUSES)
    assert all(e["reason"] for e in exclusions)


def test_node_day_coverage_reports_one_row_per_node_and_day():
    """Two meters on one node are one row carrying a count of two, and a node
    whose only reading is a zero day is not a covered node."""
    bus_of = {"M1": 1, "M2": 1, "M3": 27}
    cells = {("M1", "d1"): mim.classify_cell(_cell()),
             ("M2", "d1"): mim.classify_cell(_cell(present=(0,),
                                                   flags=["all_zero"])),
             ("M3", "d1"): mim.classify_cell(_cell()),
             ("M1", "d2"): mim.classify_cell(_cell())}
    table = mim.node_day_coverage(bus_of, cells, ["d1", "d2"])
    assert len(table) == 4                    # 2 nodes x 2 days
    by = {(r["destination_bus"], r["date"]): r for r in table}
    assert by[(1, "d1")]["meters_mapped"] == 2
    assert by[(1, "d1")]["meters_complete"] == 1   # M2's zero day is not load
    assert by[(1, "d1")]["covered"] == 1
    assert by[(1, "d2")]["meters_with_a_record"] == 1
    assert by[(27, "d2")]["covered"] == 0          # no record at all that day


# ---------------------------------------------------------------------------
# The conflicting-meter-day rejection, exercised on a supported bus
# ---------------------------------------------------------------------------

_DATE = "2021-01-26"
_SLOTS = tuple(float(t + 1) for t in range(il.N_SLOTS))


def _meter_day(account, meter, industry, values, date=_DATE):
    """One workbook row, in the shape the reader produces."""
    return il.MeterDay(
        account=account, meter=meter, user_name=f"user-{account}",
        industry=industry, location="江苏", excel_row=1, date=date,
        values=tuple(values), ratio=10.0, running_capacity=100000.0,
        phase=il.PHASE_ACTIVE_POWER, flags=())


def _detect_conflicts(*records):
    """The real detection path: flag, collect, classify, then filter."""
    flagged = il.flag_records(list(records))
    cells = mim.collect_cells(flagged, _Facts(flagged))
    verdicts = {(m, d): mim.classify_cell(c) for (m, d), c in cells.items()}
    qualified = il.qualified_block_keys(flagged)
    conflicts = sorted((m, str(d)) for (m, d), v in verdicts.items()
                       if v["status"] == "conflicting" and (m, d) in qualified)
    return flagged, verdicts, conflicts


class _Facts(dict):
    """`collect_cells` asks only whether a meter is known."""

    def __init__(self, records):
        super().__init__({md.meter_key: {"industry": md.industry}
                          for md in records})


def test_a_conflict_on_a_supported_bus_reaches_the_withholding_reason():
    """The rejection the report calls `conflicting_meter_day`, produced the way
    the module produces it -- flagged from two disagreeing rows, classified
    from the cell, and filtered by the qualified-block gate.

    The bus here is 1, which is not one of `UNSUPPORTED_BUSES`, so a rejection
    can only be the conflict and not a bus rule reached first.
    """
    assert 1 not in mim.UNSUPPORTED_BUSES

    clean = tuple(_SLOTS)
    disagreed = tuple(v + 5.0 for v in _SLOTS)
    records = [
        _meter_day("A1", "1", "商业综合体", clean),
        _meter_day("A2", "2", "商业综合体", clean),
        _meter_day("A2", "2", "商业综合体", disagreed),   # same meter-day
    ]
    flagged, verdicts, conflicts = _detect_conflicts(*records)

    conflicted_key = il.anon_id("M", "A2|2")
    assert verdicts[(conflicted_key, _DATE)]["status"] == "conflicting"
    assert verdicts[(conflicted_key, _DATE)]["usability"] != "summable"
    assert (conflicted_key, _DATE) in conflicts

    rows = mim.exclusion_rows([], _BUSES, {}, conflicts)
    withheld = [r for r in rows if r["kind"] == "conflicting_meter_day"]
    assert len(withheld) == 1
    assert conflicted_key in withheld[0]["key"] and _DATE in withheld[0]["key"]
    assert "withheld whole" in withheld[0]["reason"]
    # Not explained away as a bus rule: bus 1 is a supported bus.
    assert not [r for r in rows if r["kind"] == "unsupported_bus"
                and "bus 1" in r["key"]]


def test_a_disagreeing_row_is_withheld_whole_and_its_agreeing_twin_is_not():
    """Both rows of a disagreeing pair go, and a meter-day that merely repeats
    itself byte for byte is untouched."""
    clean = tuple(_SLOTS)
    repeated = tuple(_SLOTS)
    disagreed = tuple(v + 5.0 for v in _SLOTS)
    records = [
        _meter_day("A1", "1", "商业综合体", clean),
        _meter_day("A1", "1", "商业综合体", repeated),   # identical repeat
        _meter_day("A3", "3", "商业综合体", clean),
        _meter_day("A3", "3", "商业综合体", disagreed),  # real disagreement
    ]
    flagged, verdicts, conflicts = _detect_conflicts(*records)

    ok_key = il.anon_id("M", "A1|1")
    bad_key = il.anon_id("M", "A3|3")
    assert verdicts[(ok_key, _DATE)]["status"] == "complete"
    assert verdicts[(ok_key, _DATE)]["usability"] == "summable"
    assert verdicts[(bad_key, _DATE)]["status"] == "conflicting"
    assert [m for m, _ in conflicts] == [bad_key]


def test_a_valid_supported_bus_meter_day_is_never_withheld():
    """The control: nothing about a clean, complete, supported meter-day trips
    the conflict rule."""
    records = [_meter_day("A1", "1", "商业综合体", tuple(_SLOTS)),
               _meter_day("A4", "4", "钢铁", tuple(_SLOTS))]
    _, verdicts, conflicts = _detect_conflicts(*records)
    assert conflicts == []
    for key, info in verdicts.items():
        assert info["status"] == "complete", key
        assert info["usability"] == "summable"
    rows = mim.exclusion_rows([], _BUSES, {}, [])
    assert not [r for r in rows if r["kind"] == "conflicting_meter_day"]
