# test_industry_bus_mapping.py
"""The user-level bus mapping: one user, one bus, and no objective.

Three properties carry this module. A user is the unit, so a user's meters are
summed before anything else and the account never appears twice or on two
buses. The mapping is a function of the input alone, so the same records in a
different order give the same map. And nothing here reads a price or a return —
a mapping chosen to widen price spreads would be a result tuned into an input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import industry_load as il
import industry_bus_mapping as ibm

N = il.N_SLOTS
FLAT = tuple([0.1] * N)


def _md(meter, user, industry, date="2021-01-01", ratio=100.0, running=1000.0,
        values=None):
    """A record that passes both the meter gate and the shape pool."""
    return il.MeterDay(
        account=user, meter=meter, user_name="n", industry=industry,
        location="南京", excel_row=2, date=date,
        values=FLAT if values is None else values, ratio=ratio,
        running_capacity=running, phase=il.PHASE_ACTIVE_POWER, flags=())


BUSES = {"化工业": [23], "商业综合体": [1, 2, 3], "钢铁": [16]}
LOAD = [1, 2, 3, 16, 23]


def _users(records):
    return ibm.qualified_users(il.flag_records(records))


# ---------------------------------------------------------------------------
# The unit is the user
# ---------------------------------------------------------------------------

def test_a_user_with_several_meters_is_one_unit():
    """Two meters on one account are circuits on one account: summed, and
    mapped once. Counting them as two units is how an account gets double
    counted."""
    records = [_md("M1", "U1", "商业综合体"), _md("M2", "U1", "商业综合体")]
    rows = ibm.assign_users(_users(records), BUSES, [], LOAD)
    assert len(rows) == 1
    assert rows[0]["n_meters"] == 2
    # Identities are anonymised, as everywhere else in this chain, so the row
    # names its meters by key rather than by the workbook's own id.
    assert rows[0]["meter_ids"] == ";".join(
        sorted(r.meter_key for r in records))


def test_every_user_is_mapped_exactly_once():
    records = [_md(f"M{i}", f"U{i}", "商业综合体") for i in range(1, 6)]
    rows = ibm.assign_users(_users(records), BUSES, [], LOAD)
    ids = [r["user_id"] for r in rows if r["candidate_status"] == "mapped"]
    assert len(ids) == len(set(ids)) == 5
    ibm.assert_invariants(rows, LOAD, _users(records))


def test_a_user_spanning_two_industries_is_recorded_not_guessed():
    """No single industry's buses can hold the whole account, so the row says
    so rather than picking one of the two."""
    users = _users([_md("M1", "U1", "化工业"), _md("M2", "U1", "钢铁")])
    rows = ibm.assign_users(users, BUSES, [], LOAD)
    assert len(rows) == 1
    assert rows[0]["candidate_status"] == "unmapped"
    assert "more than one industry" in rows[0]["mapping_reason"]


def test_an_excluded_industry_is_never_given_a_bus():
    users = _users([_md("M1", "U1", "5G基站")])
    rows = ibm.assign_users(users, {"5G基站": [4]}, ["5G基站"], LOAD + [4])
    assert rows[0]["destination_bus"] == ""
    assert "excluded_industries" in rows[0]["mapping_reason"]


def test_a_row_is_never_dropped_for_being_unmappable():
    """A user with no admissible bus is a reported row, not a missing one."""
    users = _users([_md("M1", "U1", "化工业"), _md("M2", "U2", "5G基站")])
    rows = ibm.assign_users(users, BUSES, ["5G基站"], LOAD)
    assert len(rows) == len(users) == 2
    ibm.assert_invariants(rows, LOAD, users)


# ---------------------------------------------------------------------------
# Determinism and constraints
# ---------------------------------------------------------------------------

def test_the_mapping_does_not_depend_on_input_order():
    records = [_md(f"M{i}", f"U{i}", "商业综合体") for i in range(1, 7)]
    forward = ibm.assign_users(_users(records), BUSES, [], LOAD)
    backward = ibm.assign_users(_users(records[::-1]), BUSES, [], LOAD)
    assert forward == backward
    assert forward == ibm.assign_users(_users(records), BUSES, [], LOAD)


def test_no_bus_is_forced_to_carry_load():
    """Two users and three declared buses: the third stays empty. Filling it
    would need a user that does not exist."""
    users = _users([_md("M1", "U1", "商业综合体"),
                    _md("M2", "U2", "商业综合体")])
    rows = ibm.assign_users(users, BUSES, [], LOAD)
    used = sorted(int(r["destination_bus"]) for r in rows)
    assert used == [1, 2]
    assert 3 in LOAD and 3 not in used


def test_many_users_share_one_bus_when_the_industry_declares_one():
    users = _users([_md(f"M{i}", f"U{i}", "化工业") for i in range(1, 4)])
    rows = ibm.assign_users(users, BUSES, [], LOAD)
    assert {int(r["destination_bus"]) for r in rows} == {23}


def test_a_bus_the_old_rule_called_unsupported_can_receive_a_user():
    """`UNSUPPORTED_BUSES` was a meter-level blacklist. The user-level mapper
    must not consult it, or a bus could never be given load again."""
    rows = ibm.assign_users(_users([_md("M1", "U1", "水泥制造")]),
                            {"水泥制造": [22]}, [], LOAD + [22])
    assert int(rows[0]["destination_bus"]) == 22
    assert rows[0]["candidate_status"] == "mapped"


def test_a_bus_outside_the_declared_load_buses_is_refused():
    rows = ibm.assign_users(_users([_md("M1", "U1", "化工业")]),
                            {"化工业": [99]}, [], LOAD)
    assert rows[0]["candidate_status"] == "unmapped"
    assert "not a declared load bus" in rows[0]["mapping_reason"]


def test_an_invariant_breach_raises_rather_than_writing_a_bad_map():
    users = _users([_md("M1", "U1", "化工业"), _md("M2", "U2", "化工业")])
    rows = ibm.assign_users(users, BUSES, [], LOAD)
    with pytest.raises(AssertionError, match="more than once"):
        ibm.assert_invariants(rows + rows, LOAD, users)


# ---------------------------------------------------------------------------
# No objective
# ---------------------------------------------------------------------------

def test_the_mapper_imports_no_objective():
    """A mapping that can read a price can be chosen for the price. The rule
    has to be blind to every economic quantity to stay an input."""
    source = Path(ibm.__file__).read_text(encoding="utf-8")
    imports = " ".join(line for line in source.splitlines()
                       if line.startswith(("import ", "from "))).lower()
    for banned in ("lmp", "arbitrage", "wholesale", "counterfactual",
                   "market", "rl_", "surplus", "profit"):
        assert banned not in imports, f"{banned} reached the mapper's imports"


def test_the_rule_version_names_the_rule_not_the_data():
    """The rule digest depends on the rule alone, so it is stable across runs
    and across datasets; a data change moves the map's own digest instead."""
    version = ibm.rule_version()
    assert version == ibm.rule_version()
    assert len(version) == 16 and all(c in "0123456789abcdef" for c in version)
    # Different records, same rule, same digest.
    other = _users([_md("M9", "U9", "钢铁")])
    ibm.assign_users(other, BUSES, [], LOAD)
    assert ibm.rule_version() == version
