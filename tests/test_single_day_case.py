# test_single_day_case.py
"""The single-day diagnostic case: how each date is arrived at, and how the two
curves are paired.

Two decisions carry the case and both are checked here rather than described:

* the **load date** is the earliest one on which the load data is whole, taken
  in calendar order. It is not chosen on any quantity of load -- the rule it
  replaced ranked by completeness, which is a monotone proxy for load, and so
  picked from the high end of the distribution;
* the two curves are paired **period index to period index**, which is only
  correct because the load workbook labels its columns at the *start* of each
  interval and the price workbook labels its rows at the *end*. Reading one
  convention as the other rotates every price by one period and nothing
  downstream would notice.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import single_day_case as sdc
from build_industry_node_loads import LOAD_DATE_RULE, select_load_date

_REPO = pathlib.Path(__file__).resolve().parents[1]


def _row(date, complete, readings=0, buses=17, mapped=17):
    return {"date": date, "load_complete": complete, "readings": readings,
            "buses_covered": buses, "buses_mapped": mapped,
            "readings_not_convertible": 0}


# ---------------------------------------------------------------------------
# The load date rule
# ---------------------------------------------------------------------------

def test_the_rule_takes_the_earliest_whole_date_in_calendar_order():
    audit = [_row("2022-12-01", True), _row("2021-06-01", False),
             _row("2021-09-09", True), _row("2023-01-01", True)]
    assert select_load_date(audit)["date"] == "2021-09-09"
    assert LOAD_DATE_RULE == "earliest_complete_date"


def test_the_rule_does_not_depend_on_the_order_it_is_given():
    audit = [_row("2022-12-01", True), _row("2021-09-09", True)]
    assert select_load_date(audit)["date"] == \
        select_load_date(list(reversed(audit)))["date"]


def test_the_rule_is_not_a_load_ranking():
    """The failure the old rule had: completeness ranks with load.

    Here the earliest whole date carries *fewer* readings than a later one, so a
    rule that ranked by completeness -- or by anything monotone in load -- would
    pick the later date. The rule under test does not.
    """
    audit = [_row("2021-09-09", True, readings=40),
             _row("2022-12-01", True, readings=900)]
    selected = select_load_date(audit)
    assert selected["date"] == "2021-09-09"
    assert selected["readings"] == 40


def test_the_rule_refuses_when_no_date_is_whole():
    audit = [_row("2021-09-09", False), _row("2022-12-01", False)]
    with pytest.raises(ValueError, match="no date in the audit"):
        select_load_date(audit)
    with pytest.raises(ValueError, match="no date in the audit"):
        select_load_date([])


def test_the_selected_row_is_returned_so_the_evidence_travels():
    """A date without its coverage numbers is a date nobody can re-derive."""
    selected = select_load_date([_row("2021-09-09", True, readings=40)])
    assert selected["buses_covered"] == selected["buses_mapped"] == 17
    assert selected["readings"] == 40


# ---------------------------------------------------------------------------
# The price curve
# ---------------------------------------------------------------------------

def _price_curve():
    path = sdc.dp.resolve_workbook()
    if not path.exists():                        # pragma: no cover
        pytest.skip("price workbook not present")
    return sdc.verify_price()


def test_the_price_curve_passes_its_own_integrity_checks():
    verified = _price_curve()
    assert verified["n_slots"] == 96
    assert verified["interval_minutes"] == 15
    assert verified["interval_convention"] == "label_is_interval_end"
    assert verified["checks"]["slot_count_is_96"] is True
    for name, outcome in verified["checks"].items():
        if name == "slot_count_is_96":
            continue
        assert "no exception" in outcome, (
            f"{name} is not recorded as an enforced check")


def test_the_price_date_comes_from_the_sheet_label_and_says_so():
    verified = _price_curve()
    assert verified["sheet"] == "261009"
    assert verified["date"] == "2026-10-09"
    assert verified["reference_period_raw"] == "26/10/09"
    # The three things that make it weaker than a timestamp all travel with it.
    assert verified["timezone"] is None
    assert verified["externally_confirmed"] is False
    # The value is a label parsed by hand, not a date field the workbook
    # carries, and it says which it is.
    assert "label" in verified["date_is_parsed_from"].lower()


def test_a_reference_period_that_is_not_a_date_label_is_refused():
    """The parser lives in the entry point now; this checks it is reached."""
    import single_day_data as sdd
    for bad in ("not-a-date", None, "", "2026-10"):
        with pytest.raises(sdd.CaseTypeError, match="YY/MM/DD"):
            sdd.iso_from_reference_period(bad, "test")
    assert sdd.iso_from_reference_period("26/10/09", "261009") == "2026-10-09"


# ---------------------------------------------------------------------------
# The alignment lives in the entry point
# ---------------------------------------------------------------------------

def test_the_case_script_does_not_own_the_alignment_any_more():
    """Two implementations of one rule is one too many, so the rule moved.

    The alignment is asserted where it is implemented, in
    `tests/test_single_day_data.py`; what this file holds is that the case
    script reaches it rather than re-deriving it.
    """
    import single_day_data as sdd
    assert not hasattr(sdc, "period_alignment"), (
        "single_day_case has grown its own alignment again")
    block = sdd.check_alignment(
        sdd.PeriodAxis(96, 15, "interval_start", None),
        sdd.PeriodAxis(96, 15, "interval_end", None))
    assert block["load_axis"]["label_convention"] == "interval_start"
    assert block["price_axis"]["label_convention"] == "interval_end"
    assert block["conversion_applied"] == "none"


# ---------------------------------------------------------------------------
# The case artifacts on disk
# ---------------------------------------------------------------------------

def test_the_case_manifest_records_everything_the_case_turns_on():
    path = sdc.CASE_DIR / "case_manifest.json"
    if not path.exists():                        # pragma: no cover
        pytest.skip("the case has not been built in this checkout")
    case = json.loads(path.read_text(encoding="utf-8"))

    for key in ("load_date", "price_date", "same_day", "load_source",
                "price_source", "unit_status", "mapping_version",
                "load_policies", "period_alignment", "input_fingerprints",
                "integrity_checks"):
        assert key in case, key

    assert case["same_day"] is False, (
        "this case pairs two different days by design; if they ever coincide "
        "the report and the manifest both need revisiting")
    assert case["load_date"] != case["price_date"]
    assert case["load_source"]["selection_rule"] == LOAD_DATE_RULE
    assert case["unit_status"] == "unit_unverified"
    assert set(case["load_policies"].values()) <= {"preserve", "replace",
                                                   "overlay"}
    assert case["not_a_same_day_replay"]


def test_the_price_input_is_ninety_six_rows_of_finite_prices():
    path = sdc.CASE_DIR / "price_96.csv"
    if not path.exists():                        # pragma: no cover
        pytest.skip("the case has not been built in this checkout")
    import csv
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 96
    assert [int(r["time_index"]) for r in rows] == list(range(96))
    values = [float(r["price_cny_mwh"]) for r in rows]
    assert all(v == v and abs(v) != float("inf") for v in values)
    # The periods are labelled at both ends, so the convention that differs
    # from the load workbook's is visible in the file rather than implied.
    assert rows[0]["interval_start"] == "00:00"
    assert rows[0]["interval_end"] == "00:15"
    assert rows[-1]["interval_end"] == "24:00"
