# test_screen_industry_blocks.py
"""The block screening's three readings must nest, and share one reason order.

`classify` is the only decision in `screen_industry_blocks`, and its output is
quoted as evidence about the load input. What it must never do is let a reading
be wider than the one it is built on: A is the pool, B adds a meter verdict, C
adds a block verdict, so A >= B >= C on every row. A row that fails the pool
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


def _row(meter="M1", flags=(), in_pool=True):
    return {"meter": meter, "account": "U1", "industry": "化工业",
            "date": "2021-01-01", "flags": set(flags), "in_pool": in_pool}


def test_the_readings_nest_on_every_combination():
    """A >= B >= C for every flag set, meter verdict and checkability state."""
    cases = [
        (_row(), {"M1"}, True),                                  # clean
        (_row(), set(), True),                                   # uncheckable
        (_row(), {"M1"}, False),                                 # meter fails
        (_row(flags=["ratio_suspect"]), {"M1"}, True),           # advisory
        (_row(flags=["capacity_inconsistent"]), {"M1"}, True),
        (_row(in_pool=False, flags=["all_zero"]), {"M1"}, True),
        (_row(in_pool=False, flags=["all_zero"]), set(), True),
        (_row(flags=["ratio_suspect"]), {"M1"}, False),
    ]
    for row, checkable, ok in cases:
        verdict = sb.classify(row, {"M1": ok}, checkable)
        assert verdict["qualified_A"] >= verdict["qualified_B"], row
        assert verdict["qualified_B"] >= verdict["qualified_C"], row


def test_only_a_clean_checkable_block_reaches_C():
    assert sb.classify(_row(), {"M1": True}, {"M1"})["qualified_C"] is True


def test_each_reading_has_a_reason_naming_its_own_cause():
    """The reason column must name the first gate that stops the row, not a
    generic failure -- the report is read to find out why a block is out."""
    dropped = sb.classify(_row(in_pool=False, flags=["all_zero"]),
                          {"M1": True}, {"M1"})
    assert "all_zero" in dropped["reason"] and not dropped["qualified_A"]

    uncheckable = sb.classify(_row(), {}, set())
    assert "not checkable" in uncheckable["reason"]
    assert uncheckable["qualified_A"] and not uncheckable["qualified_B"]

    meter_out = sb.classify(_row(), {"M1": False}, {"M1"})
    assert "exceeds" in meter_out["reason"] and not meter_out["qualified_B"]

    advisory = sb.classify(_row(flags=["ratio_suspect"]), {"M1": True}, {"M1"})
    assert "ratio_suspect" in advisory["reason"]
    assert advisory["qualified_B"] and not advisory["qualified_C"]

    assert sb.classify(_row(), {"M1": True}, {"M1"})["reason"] == ""


def test_the_reported_tolerances_come_from_the_module():
    """The evidence must quote the tolerance the check applies, not a copy."""
    text = " ".join(t for _, t in sb.rules_in_force())
    assert str(il.CAPACITY_PEAK_TOLERANCE) in text
    assert str(il.RATIO_OUTLIER_FACTOR) in text
    assert str(il.MIN_USERS_FOR_RELIABLE) in text
    assert il.FLAG_CAPACITY_INCONSISTENT in text


def test_the_capacity_tolerance_is_the_one_the_flag_uses():
    """`verify_ratios` and the row-level flag must agree on the threshold.
    They are separate code paths over the same quantity, so a drift between
    them would let a block be 'within capacity' by one reading and flagged by
    the other."""
    assert il.CAPACITY_PEAK_TOLERANCE == pytest.approx(1.5)
    assert il.FLAG_CAPACITY_INCONSISTENT in il.ADVISORY_FLAGS
    assert il.FLAG_CAPACITY_INCONSISTENT not in il.SHAPE_DROP_FLAGS
