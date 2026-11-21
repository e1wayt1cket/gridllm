# test_screen_industry_blocks.py
"""The block screening's readings must nest, and share the module's verdict.

`classify` is the only decision in `screen_industry_blocks`, and its output is
quoted as evidence about the load input. What it must never do is let a reading
be wider than the one it is built on: A is the pool, B adds the meter verdict,
C adds a block verdict, so A >= B >= C on every row. A row that fails the pool
must also fail the two stricter readings, or the report would claim qualified
blocks the pipeline never admitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import industry_load as il
import screen_industry_blocks as sb

CHECKED_OK = {"M1": {"checkable": True, "qualified": True}}
CHECKED_BAD = {"M1": {"checkable": True, "qualified": False}}
UNCHECKABLE = {"M1": {"checkable": False, "qualified": False}}


def _row(meter="M1", flags=(), in_pool=True):
    return {"meter": meter, "account": "U1", "industry": "化工业",
            "date": "2021-01-01", "flags": set(flags), "in_pool": in_pool}


def test_the_readings_nest_on_every_combination():
    """A >= B >= C for every flag set and meter verdict."""
    cases = [
        (_row(), CHECKED_OK),
        (_row(), UNCHECKABLE),
        (_row(), CHECKED_BAD),
        (_row(), {}),                                    # meter absent entirely
        (_row(flags=["ratio_suspect"]), CHECKED_OK),
        (_row(flags=["capacity_inconsistent"]), CHECKED_OK),
        (_row(in_pool=False, flags=["all_zero"]), CHECKED_OK),
        (_row(in_pool=False, flags=["all_zero"]), UNCHECKABLE),
        (_row(flags=["ratio_suspect"]), CHECKED_BAD),
    ]
    for row, verdicts in cases:
        verdict = sb.classify(row, verdicts)
        assert verdict["qualified_A"] >= verdict["qualified_B"], row
        assert verdict["qualified_B"] >= verdict["qualified_C"], row


def test_only_a_clean_checkable_block_reaches_C():
    assert sb.classify(_row(), CHECKED_OK)["qualified_C"] is True


def test_a_meter_the_check_never_reached_is_not_qualified():
    """The absent verdict must read the same as an explicit failure, or a
    meter the workbook cannot describe would enter by default."""
    for verdicts in (UNCHECKABLE, {}):
        verdict = sb.classify(_row(), verdicts)
        assert verdict["qualified_A"] is True
        assert verdict["qualified_B"] is False
        assert "not checkable" in verdict["reason"]


def test_each_reading_has_a_reason_naming_its_own_cause():
    """The reason column must name the first gate that stops the row, not a
    generic failure -- the report is read to find out why a block is out."""
    dropped = sb.classify(_row(in_pool=False, flags=["all_zero"]), CHECKED_OK)
    assert "all_zero" in dropped["reason"] and not dropped["qualified_A"]

    meter_out = sb.classify(_row(), CHECKED_BAD)
    assert "exceeds" in meter_out["reason"] and not meter_out["qualified_B"]

    advisory = sb.classify(_row(flags=["ratio_suspect"]), CHECKED_OK)
    assert "ratio_suspect" in advisory["reason"]
    assert advisory["qualified_B"] and not advisory["qualified_C"]

    assert sb.classify(_row(), CHECKED_OK)["reason"] == ""


def test_the_reported_tolerances_come_from_the_module():
    """The evidence must quote the tolerance the check applies, not a copy."""
    text = " ".join(t for _, t in sb.rules_in_force())
    assert str(il.CAPACITY_PEAK_TOLERANCE) in text
    assert str(il.RATIO_OUTLIER_FACTOR) in text
    assert str(il.MIN_USERS_FOR_RELIABLE) in text
    assert il.FLAG_CAPACITY_INCONSISTENT in text


def test_the_superseded_pre_s1_digest_cannot_be_reached_again():
    """The pre-S1 rule read a slot value as energy per slot, so it multiplied
    every implied peak by four and rejected meters for a defect they did not
    have. Its digest is recorded here as the thing the current rule must not
    equal: if this fails, the unit correction has been undone somewhere."""
    assert sb.rule_version() != "f647fa60d88c3b29"
    inputs = sb.rule_inputs()
    assert inputs["reading_unit"] == "kW"
    assert inputs["slot_reading_is_average_power"] is True
    assert inputs["mw_per_kw"] == pytest.approx(1e-3)


def test_the_rule_version_moves_when_the_rule_moves(monkeypatch):
    """A manifest is only a tie between a result and a rule if the digest
    actually depends on the rule."""
    before = sb.rule_version()
    monkeypatch.setattr(il, "CAPACITY_PEAK_TOLERANCE", 1.4)
    assert sb.rule_version() != before
    monkeypatch.undo()
    assert sb.rule_version() == before
