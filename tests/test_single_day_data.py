# test_single_day_data.py
"""The common load-price entry point: what it accepts, what it refuses.

Two things carry this module and both are checked here rather than described:

* the **case type** decides the date rule, and a case that does not state one
  does not get the permissive rule -- a missing field must not be able to loosen
  a check;
* the two series are paired **only** on an index both axes independently agree
  on. A missing, repeated, misordered or differently-spaced period is refused;
  nothing is shifted, interpolated or resampled to make a pair fit.
"""

from __future__ import annotations

import csv
import json
import pathlib

import pytest

import industry_load as il
import single_day_data as sdd

_REPO = pathlib.Path(__file__).resolve().parents[1]
_LEDGER = (_REPO / "data" / "processed" / "node_loads"
           / "typical_day_industry" / "bus_loads_96x33_audit.csv")
_MANIFEST = _LEDGER.parent / "manifest.json"

_N = il.N_SLOTS


def _write_ledger(tmp_path, rows, name="ledger.csv"):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["bus", "time_index", "final_mw",
                                           "policy", "date"])
        w.writeheader()
        w.writerows(rows)
    return path


def _days(rows_per_bus=1, buses=(1, 2)):
    return [{"bus": b, "time_index": t, "final_mw": "1.0",
             "policy": "replace", "date": "2022-11-30"}
            for b in buses for _ in range(rows_per_bus) for t in range(_N)]


# ---------------------------------------------------------------------------
# 1 & 2. The date rule follows from the case type
# ---------------------------------------------------------------------------

def test_different_dates_pass_a_cross_date_diagnostic():
    """The rule this case exists under: two real days, paired and both named."""
    rule = sdd.apply_date_rule("cross_date_diagnostic",
                               "2022-11-30", "2026-10-09")
    assert rule["same_day"] is False
    assert rule["load_date"] == "2022-11-30"
    assert rule["price_date"] == "2026-10-09"
    assert rule["dates_may_differ"] is True
    assert rule["dates_recorded_not_rewritten"] is True


def test_a_same_day_replay_refuses_mismatched_dates():
    with pytest.raises(sdd.CaseTypeError, match="same_day_replay"):
        sdd.apply_date_rule("same_day_replay", "2022-11-30", "2026-10-09")
    # And accepts them when they agree.
    rule = sdd.apply_date_rule("same_day_replay", "2026-10-09", "2026-10-09")
    assert rule["same_day"] is True
    assert rule["dates_may_differ"] is False


def test_an_unstated_case_type_does_not_get_the_permissive_rule():
    """A missing case type is an error, not an invitation to relax."""
    for missing in (None, "", "diagnostic", "cross-date-diagnostic"):
        with pytest.raises(sdd.CaseTypeError, match="case_type"):
            sdd.apply_date_rule(missing, "2022-11-30", "2026-10-09")


def test_a_missing_date_is_refused_rather_than_treated_as_absent():
    for load, price in ((None, "2026-10-09"), ("2022-11-30", None),
                        ("", "")):
        with pytest.raises(sdd.CaseTypeError, match="both dates"):
            sdd.apply_date_rule("cross_date_diagnostic", load, price)


# ---------------------------------------------------------------------------
# 3. Every way a period axis can be wrong
# ---------------------------------------------------------------------------

def test_a_repeated_period_is_refused(tmp_path):
    rows = _days()
    rows.append({"bus": 1, "time_index": 5, "final_mw": "1.0",
                 "policy": "replace", "date": "2022-11-30"})
    path = _write_ledger(tmp_path, rows)
    with pytest.raises(sdd.DataAlignmentError, match="repeated"):
        sdd.load_load_axis(path)


def test_a_missing_period_is_refused(tmp_path):
    rows = [r for r in _days() if not (r["bus"] == 2 and r["time_index"] == 40)]
    path = _write_ledger(tmp_path, rows)
    with pytest.raises(sdd.DataAlignmentError, match="missing or out-of-order"):
        sdd.load_load_axis(path)


def test_a_misordered_period_is_refused_rather_than_sorted(tmp_path):
    """Sorting first would accept a file assembled by concatenating two runs."""
    rows = _days()
    rows[3], rows[7] = rows[7], rows[3]
    path = _write_ledger(tmp_path, rows)
    with pytest.raises(sdd.DataAlignmentError, match="missing or out-of-order"):
        sdd.load_load_axis(path)


def test_a_short_day_is_refused(tmp_path):
    rows = _days()[:-1]
    path = _write_ledger(tmp_path, rows)
    with pytest.raises(sdd.DataAlignmentError):
        sdd.load_load_axis(path)


def test_a_different_sampling_interval_is_refused():
    """A granularity conversion has to be chosen deliberately; none is done."""
    load = sdd.PeriodAxis(96, 15, "interval_start", None)
    price = sdd.PeriodAxis(96, 30, "interval_end", None)
    with pytest.raises(sdd.DataAlignmentError, match="resampling"):
        sdd.check_alignment(load, price)


def test_a_different_period_count_is_refused():
    load = sdd.PeriodAxis(96, 15, "interval_start", None)
    price = sdd.PeriodAxis(48, 15, "interval_end", None)
    with pytest.raises(sdd.DataAlignmentError, match="periods"):
        sdd.check_alignment(load, price)


def test_periods_covering_different_ground_are_refused():
    """Same count, same spacing, different semantics: still not one index."""
    load = sdd.PeriodAxis(96, 15, "interval_start", None)

    class Shifted(sdd.PeriodAxis):
        def interval(self, t):
            start, end = super().interval(t)
            return start + 15, end + 15 if end < 24 * 60 else end

    with pytest.raises(sdd.DataAlignmentError, match="covers"):
        sdd.check_alignment(load, Shifted(96, 15, "interval_end", None))


def test_two_valid_axes_are_one_index():
    """The real pair: opposite label conventions, same intervals."""
    load = sdd.PeriodAxis(96, 15, "interval_start", None)
    price = sdd.PeriodAxis(96, 15, "interval_end", None)
    block = sdd.check_alignment(load, price)
    assert block["conversion_applied"] == "none"
    assert block["interpolation_applied"] == "none"
    assert block["resampling_applied"] == "none"
    for t in (0, 1, 95):
        assert load.interval(t) == price.interval(t)


def test_the_label_convention_does_not_move_the_interval():
    """A start label and an end label describe one interval, not two."""
    start = sdd.PeriodAxis(96, 15, "interval_start", None)
    end = sdd.PeriodAxis(96, 15, "interval_end", None)
    for t in (0, 47, 95):
        assert start.interval(t) == end.interval(t)
    assert start.interval(0) == (0, 15)
    assert start.interval(95) == (1425, 1440)


# ---------------------------------------------------------------------------
# 4. No consumer may bypass the entry point
# ---------------------------------------------------------------------------

def _sources():
    return sorted((_REPO / "src").rglob("*.py"))


def test_the_alignment_rule_has_exactly_one_implementation():
    """An alignment rule with two implementations is one that will disagree
    with itself."""
    defining = {p.name for p in _sources()
                if "def check_alignment" in p.read_text(encoding="utf-8")}
    assert defining == {"single_day_data.py"}, defining
    defining_axis = {p.name for p in _sources()
                     if "class PeriodAxis" in p.read_text(encoding="utf-8")}
    assert defining_axis == {"single_day_data.py"}, defining_axis


def test_every_module_that_pairs_a_ledger_with_a_curve_uses_the_entry_point():
    """The rule this guards: a consumer that reads both a node-load ledger and
    a price curve must not build its own pair."""
    offenders = []
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        reads_ledger = ("bus_loads_96x33" in text
                        or "node_loads" in text)
        reads_curve = ("read_curves" in text or "get_day_ahead_price" in text)
        if reads_ledger and reads_curve and "single_day_data" not in text:
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} pair a ledger with a price curve without going through "
        f"single_day_data; the alignment check is not on their path")


def test_the_case_builder_does_not_reimplement_the_alignment():
    """`single_day_case` writes artifacts; it does not decide whether two
    series may be paired."""
    text = (_REPO / "src" / "single_day_case.py").read_text(encoding="utf-8")
    assert "import single_day_data" in text
    for duplicated in ("def check_alignment", "def apply_date_rule",
                       "class PeriodAxis"):
        assert duplicated not in text, f"{duplicated} is duplicated here"


# ---------------------------------------------------------------------------
# 5. The manifest keeps the sources and both dates
# ---------------------------------------------------------------------------

def test_the_case_manifest_keeps_both_dates_and_the_source_identities():
    path = (_REPO / "data" / "processed" / "single_day_diagnostic"
            / "case_manifest.json")
    if not path.exists():                        # pragma: no cover
        pytest.skip("the case has not been built in this checkout")
    case = json.loads(path.read_text(encoding="utf-8"))
    assert case["load_date"] == "2022-11-30"
    assert case["price_date"] == "2026-10-09"
    assert case["same_day"] is False
    assert case["case_type"] == "cross_date_diagnostic"
    assert case["load_source"]["ledger_sha256"]
    assert case["price_source"]["date_is"]
    assert case["price_source"]["externally_confirmed"] is False


# ---------------------------------------------------------------------------
# 6, 7 & 8. The states, the gaps and the fallbacks
# ---------------------------------------------------------------------------

def _case(tmp_path=None):
    if not (_LEDGER.exists() and _MANIFEST.exists()):  # pragma: no cover
        pytest.skip("ledger not built in this checkout")
    return sdd.load_case("cross_date_diagnostic", _LEDGER, _MANIFEST)


def test_an_audit_ledger_may_not_become_a_formal_network_input():
    """The unit gate, asked of the artifact rather than of the caller."""
    case = _case()
    assert case.load_kind == "audit_ledger"
    assert case.ledger_manifest["unit_status"] == il.UNIT_UNVERIFIED
    with pytest.raises(sdd.CaseTypeError, match="audit_ledger"):
        case.require_formal_load_mw()


def test_the_formal_gate_reads_the_status_rather_than_trusting_the_caller():
    """A ledger that claims `formal` but carries an unverified unit is still
    refused: the two fields have to agree."""
    case = _case()
    case.ledger_manifest = {**case.ledger_manifest, "ledger_kind": "formal"}
    with pytest.raises(sdd.CaseTypeError, match="unit status"):
        case.require_formal_load_mw()


def test_a_blank_final_load_is_carried_as_blank_and_never_zero_filled(tmp_path):
    """The interface hands back what the ledger says. A gap that became a zero
    would be a measurement nobody made."""
    rows = _days()
    for r in rows:
        if r["bus"] == 2 and r["time_index"] == 10:
            r["final_mw"] = ""
    axis, loaded = sdd.load_load_axis(_write_ledger(tmp_path, rows))
    gap = [r for r in loaded if r["bus"] == "2" and r["time_index"] == "10"]
    assert len(gap) == 1
    assert gap[0]["final_mw"] == "", "the gap was filled in by the interface"


def test_there_is_no_synthetic_fallback():
    """A missing or unknown curve is an error, not a reason to generate one."""
    text = (_REPO / "src" / "single_day_data.py").read_text(encoding="utf-8")
    for forbidden in ("Synthetic", "synthetic", "legacy", "library"):
        assert forbidden not in text, (
            f"single_day_data mentions {forbidden!r}; the entry point must "
            f"not be able to reach a generated price")
    if not (_LEDGER.exists() and _MANIFEST.exists()):  # pragma: no cover
        pytest.skip("ledger not built in this checkout")
    with pytest.raises(Exception):
        sdd.load_case("cross_date_diagnostic", _LEDGER, _MANIFEST,
                      price_sheet="no_such_sheet")


def test_a_curve_with_no_date_label_cannot_satisfy_a_date_rule():
    with pytest.raises(sdd.CaseTypeError, match="YY/MM/DD"):
        sdd.iso_from_reference_period(None, "261009")
    with pytest.raises(sdd.CaseTypeError, match="YY/MM/DD"):
        sdd.iso_from_reference_period("not-a-date", "261009")


# ---------------------------------------------------------------------------
# 9. The same input gives the same answer
# ---------------------------------------------------------------------------

def test_the_same_inputs_validate_identically():
    first = _case()
    second = _case()
    assert json.dumps(first.describe(), sort_keys=True) == \
        json.dumps(second.describe(), sort_keys=True)


def test_the_price_accessor_serves_only_the_validated_axis():
    case = _case()
    assert case.price_for().shape == (96,)
    assert case.price_for(95, 1).shape == (1,)
    for bad in ((96, 1), (-1, 1), (0, 97), (90, 7)):
        with pytest.raises(sdd.DataAlignmentError):
            case.price_for(*bad)
