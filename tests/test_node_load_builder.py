# test_node_load_builder.py
"""The 33-bus node-load ledger: composition, coverage, and one day at a time.

Four properties carry this module. All 33 buses are classified, and a bus is
either carrying a mapped user or not — never both, never omitted. A bus with no
user keeps the benchmark's own demand, because "no mapped user" is not "delete
this load". The composition is exactly one of the three stated formulas. And a
ledger holds **one date**: the failure this suite pins hardest is a meter's
whole history being summed into a single day, which looks like a plausible large
number and is wrong.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import industry_load as il
import build_industry_node_loads as bnl

N = il.N_SLOTS
FLAT = tuple([0.1] * N)


def _md(meter, user, industry, date="2021-01-01", ratio=100.0, running=1000.0,
        peak=None):
    values = FLAT if peak is None else tuple(
        [0.1] * (N - 1) + [float(peak)])
    return il.MeterDay(
        account=user, meter=meter, user_name="n", industry=industry,
        location="南京", excel_row=2, date=date, values=values, ratio=ratio,
        running_capacity=running, phase=il.PHASE_ACTIVE_POWER, flags=())


def _map_csv(tmp_path, rows):
    path = tmp_path / "map.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["user_id", "industry", "n_meters",
                                           "meter_ids", "destination_bus",
                                           "mapping_rule", "mapping_reason",
                                           "candidate_status",
                                           "simulation_mapping",
                                           "qualified_blocks"])
        w.writeheader()
        for r in rows:
            w.writerow({"user_id": r[0], "industry": r[1], "n_meters": 1,
                        "meter_ids": r[2], "destination_bus": r[3],
                        "mapping_rule": "U1_user_round_robin",
                        "mapping_reason": "", "candidate_status": "mapped",
                        "simulation_mapping": True, "qualified_blocks": 1})
    return path


# ---------------------------------------------------------------------------
# Classification covers all 33 buses, and the two sets are complements
# ---------------------------------------------------------------------------

def test_the_classification_covers_all_33_buses(tmp_path):
    path = _map_csv(tmp_path, [("U1", "化工业", "M1", 23)])
    classes = bnl.bus_classification(path)
    assert [c["bus"] for c in classes] == list(range(33))
    bnl.assert_classification_complete(classes)


def test_a_bus_is_never_both_carrying_a_user_and_carrying_none(tmp_path):
    path = _map_csv(tmp_path, [("U1", "化工业", "M1", 23),
                               ("U2", "钢铁", "M2", 16)])
    classes = bnl.bus_classification(path)
    has = {c["bus"] for c in classes if c["has_user"]}
    none = {c["bus"] for c in classes if not c["has_user"]}
    assert has == {16, 23}
    assert not (has & none)
    assert has | none == set(range(33))


def test_the_bus_count_comes_from_the_mapping_file(tmp_path):
    path = _map_csv(tmp_path, [("U1", "化工业", "M1", 23),
                               ("U2", "化工业", "M2", 23),
                               ("U3", "钢铁", "M3", 16)])
    classes = {c["bus"]: c for c in bnl.bus_classification(path)}
    assert classes[23]["n_users"] == 2      # deduplicated by construction
    assert classes[23]["n_meters"] == 2
    assert classes[16]["has_user"] is True
    assert classes[17]["has_user"] is False


def test_the_slack_bus_is_not_a_mapping_target(tmp_path):
    classes = bnl.bus_classification(_map_csv(tmp_path, []))
    slack = [c for c in classes if c["bus"] == 0][0]
    assert slack["role"] == "slack"
    assert slack["is_mapping_target"] is False
    assert slack["has_user"] is False


def test_a_classification_that_omits_a_bus_is_rejected():
    with pytest.raises(AssertionError, match="33 buses"):
        bnl.assert_classification_complete([{"bus": 1, "has_user": True}])


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_the_three_policies_are_the_stated_formulas():
    assert bnl.compose(3.0, 5.0, "preserve") == (3.0, "not_applicable")
    assert bnl.compose(3.0, 5.0, "replace") == (5.0, "present")
    assert bnl.compose(3.0, 5.0, "overlay") == (8.0, "present")
    # overlay adds, it does not average: two loads on one bus are two loads.
    assert bnl.compose(1.0, 1.0, "overlay")[0] == 2.0


def test_a_replace_bus_with_no_industry_data_is_excluded_not_zero():
    """An unobserved bus must not be reported as an empty one. Writing 0.0
    would say the bus drew nothing, which is a measurement nobody made."""
    final, status = bnl.compose(1.0, None, "replace")
    assert final is None
    assert status == "excluded_no_data"
    assert bnl.compose(1.0, None, "overlay")[0] is None
    # preserve does not need an industry term at all.
    assert bnl.compose(1.0, None, "preserve") == (1.0, "not_applicable")


def test_an_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown load policy"):
        bnl.compose(1.0, 1.0, "scale")


def test_a_missing_bus_policy_raises_rather_than_defaulting(monkeypatch):
    full = {str(b): "preserve" for b in bnl.ALL_BUSES}
    full.pop("17")
    monkeypatch.setattr(bnl.cfg, "get_scenario_cfg",
                        lambda name: {"load_model": {"bus_load_policy": full}})
    with pytest.raises(ValueError, match="leaves bus 17 unassigned"):
        bnl.bus_policies("s", [])
    full["17"] = "overlay"
    monkeypatch.setattr(bnl.cfg, "get_scenario_cfg",
                        lambda name: {"load_model": {"bus_load_policy": full}})
    assert bnl.bus_policies("s", [])[17] == "overlay"


def test_the_real_config_assigns_every_bus_exactly_one_policy():
    policies = bnl.bus_policies("typical_day_industry", [])
    assert sorted(policies) == list(range(33))
    assert set(policies.values()) <= set(bnl.POLICIES)
    assert policies[0] == "preserve"


# ---------------------------------------------------------------------------
# One date at a time
# ---------------------------------------------------------------------------

def test_a_ledger_holds_one_date_and_never_sums_a_meters_history():
    """The bug this pins: filtering on `(meter, date) in the pool` without also
    requiring `md.date == date` admits every record of a meter that has *any*
    qualified block on the date, and the whole history lands in one day. The
    result is a large, plausible-looking number."""
    day_a = [_md("M1", "U1", "化工业", date="2021-01-01", ratio=1000.0)]
    day_b = [_md("M1", "U1", "化工业", date="2021-01-02", ratio=1000.0)]
    # Identities are anonymised, so the bus table is keyed by `user_key`.
    user_bus = {day_a[0].user_key: 23}
    single = bnl.industry_curves("2021-01-01", user_bus, day_a)[0]
    # Both days present in the input: the 1st must be unchanged.
    together = bnl.industry_curves("2021-01-01", user_bus, day_a + day_b)[0]
    assert set(together) == {23}
    assert together[23].sum() == pytest.approx(single[23].sum())
    # 0.1 kW x 1000 / 1000 = 0.1 MW per slot, over 96 slots.
    assert together[23].sum() == pytest.approx(0.1 * N)


def test_the_daily_energy_is_the_power_series_times_the_slot_length():
    """A day is 96 average powers; its energy is their sum times 0.25 h, never
    the sum times four."""
    assert il.HOURS_PER_SLOT == pytest.approx(0.25)


def test_a_user_is_never_counted_on_two_buses():
    """Five users on one bus each at 10 MW is 50 MW on that bus, not 10: the
    absolute path adds users, it does not average them."""
    records = [_md("M1", "U1", "化工业", ratio=1000.0),
               _md("M2", "U2", "化工业", ratio=1000.0),
               _md("M3", "U3", "钢铁", ratio=1000.0)]
    user_bus = {records[0].user_key: 23, records[1].user_key: 23,
                records[2].user_key: 16}
    curves, exclusions, _users = bnl.industry_curves("2021-01-01", user_bus, records)
    assert set(curves) == {16, 23}
    assert curves[23].sum() == pytest.approx(2 * 0.1 * N)
    assert curves[16].sum() == pytest.approx(0.1 * N)
    assert exclusions == []


def test_an_unmapped_user_is_excluded_with_a_reason_not_zero_filled():
    records = [_md("M1", "U1", "5G基站")]
    curves, exclusions, _users = bnl.industry_curves("2021-01-01", {}, records)
    assert curves == {}
    assert len(exclusions) == 1
    assert "not mapped to any bus" in exclusions[0]["reason"]


def test_an_incomplete_day_is_excluded_and_never_padded():
    short = _md("M1", "U1", "化工业")
    short = il.MeterDay(**{**short.__dict__,
                           "values": tuple([0.1] * (N - 1) + [None])})
    curves, exclusions, _users = bnl.industry_curves("2021-01-01",
                                             {short.user_key: 23}, [short])
    assert curves == {}
    assert len(exclusions) == 1


def test_the_date_audit_reports_a_missing_bus_rather_than_a_complete_day():
    """A date is complete only when every mapped bus has data on it. A date
    where one bus is dark is reported as incomplete and the bus is named."""
    records = [_md("M1", "U1", "化工业", date="2021-01-01"),
               _md("M2", "U2", "钢铁", date="2021-01-01"),
               _md("M3", "U3", "化工业", date="2021-01-02")]
    user_bus = {records[0].user_key: 23, records[1].user_key: 16,
                records[2].user_key: 23}
    audit = {r["date"]: r for r in bnl.date_audit(records, user_bus, [16, 23])}
    assert set(audit) == {"2021-01-01", "2021-01-02"}
    assert audit["2021-01-01"]["load_complete"] is True
    assert audit["2021-01-01"]["buses_covered"] == 2
    assert audit["2021-01-02"]["load_complete"] is False
    assert audit["2021-01-02"]["buses_without_data"] == "16"


def test_the_date_audit_never_invents_a_price_date():
    """The real price curves carry no date that overlaps a load day, so every
    row says so. A filled-in price date would be a fabricated alignment."""
    records = [_md("M1", "U1", "化工业", date="2021-01-01")]
    audit = bnl.date_audit(records, {records[0].user_key: 23}, [23])
    assert audit and all(r["price_date_available"] is False for r in audit)
    assert all("no dated real price curve" in r["price_note"] for r in audit)


def test_the_date_audit_does_not_rank_days_by_load():
    """It ranks by coverage. A load ranking would rank an unverified quantity
    and be quoted as a physical percentile."""
    records = [_md("M1", "U1", "化工业", date="2021-01-01")]
    audit = bnl.date_audit(records, {records[0].user_key: 23}, [23])
    assert "final_mwh" not in audit[0] and "energy" not in "".join(audit[0])
    assert "meters_with_a_complete_day" in audit[0]


def test_the_date_pick_is_deterministic():
    records = [_md("M1", "U1", "化工业", date="2021-01-01"),
               _md("M2", "U2", "化工业", date="2021-01-02"),
               _md("M3", "U3", "化工业", date="2021-01-02")]
    dates = ["2021-01-01", "2021-01-02"]
    assert bnl.pick_date(records, dates) == "2021-01-02"
    assert bnl.pick_date(records, dates) == bnl.pick_date(records, dates)


# ---------------------------------------------------------------------------
# The unit gate
# ---------------------------------------------------------------------------

def test_a_unit_with_no_admissible_basis_cannot_carry_a_formal_load(monkeypatch):
    """The gate is on the **basis**, not on external certification.

    A basis may be an external source or a recorded project ruling. A unit
    naming neither is one nothing stands behind, and the audit ledger exists so
    those numbers can still be read.
    """
    monkeypatch.setattr(il, "READING_UNIT_EVIDENCE", "no_basis_recorded")
    assert il.unit_status() == il.UNIT_UNVERIFIED
    with pytest.raises(ValueError, match="not one of"):
        il.require_unit_basis()


def test_a_recorded_project_ruling_admits_the_formal_ledger(monkeypatch):
    """A ruling is a basis. It admits absolute MW -- and it is **not** external
    verification, which is a separate question with its own answer."""
    monkeypatch.setattr(il, "READING_UNIT_EVIDENCE", il.UNIT_BASIS_RULING)
    assert il.unit_status() == il.UNIT_BASIS_RULING_STATUS
    assert il.is_externally_verified() is False
    assert il.require_unit_basis() == il.READING_UNIT_KW


def test_the_formal_ledger_opens_once_an_external_source_states_the_unit(
        monkeypatch):
    monkeypatch.setattr(il, "READING_UNIT_EVIDENCE", il.UNIT_BASIS_EXTERNAL)
    assert il.unit_status() == il.UNIT_VERIFIED
    assert il.is_externally_verified() is True
    assert il.require_unit_basis() == il.READING_UNIT_KW


def test_the_unit_status_travels_with_every_audit_row():
    """A magnitude computed under an unconfirmed unit must not be separable
    from that fact, or it will be quoted as if it were verified."""
    source = Path(bnl.__file__).read_text(encoding="utf-8")
    for column in ("unit_status", "reading_unit", "mapping_version",
                   "native_source"):
        assert f'"{column}"' in source, column


def test_the_slack_bus_is_written_with_no_load(tmp_path):
    """All 33 buses appear, and the slack appears as an explicit zero rather
    than being omitted, so a reader can see it was considered."""
    classes = bnl.bus_classification(_map_csv(tmp_path, []))
    slack = [c for c in classes if c["bus"] == 0][0]
    assert slack["role"] == "slack" and slack["n_meters"] == 0
    assert slack["has_user"] is False


# ---------------------------------------------------------------------------
# The manifest names the users, not only the buses
# ---------------------------------------------------------------------------

def test_the_users_that_contributed_are_returned_by_bus():
    """A bus count without its users is a claim nobody can re-derive."""
    records = [_md("M1", "U1", "化工业", ratio=1000.0),
               _md("M2", "U2", "化工业", ratio=1000.0),
               _md("M3", "U3", "钢铁", ratio=1000.0)]
    user_bus = {records[0].user_key: 23, records[1].user_key: 23,
                records[2].user_key: 16}
    curves, _exclusions, users = bnl.industry_curves("2021-01-01", user_bus,
                                                     records)
    assert set(users) == set(curves) == {16, 23}
    assert sorted(users[23]) == sorted([records[0].user_key,
                                        records[1].user_key])
    assert users[16] == [records[2].user_key]
    # Every user lands on exactly one bus: the partition is what rules out
    # double counting.
    flat = [u for us in users.values() for u in us]
    assert len(flat) == len(set(flat)) == 3


def test_an_excluded_meter_contributes_no_user():
    """The user map is the users that *contributed*, not the ones the mapping
    file mentions."""
    short = _md("M1", "U1", "化工业")
    short = il.MeterDay(**{**short.__dict__,
                           "values": tuple([0.1] * (N - 1) + [None])})
    curves, exclusions, users = bnl.industry_curves("2021-01-01",
                                                    {short.user_key: 23},
                                                    [short])
    assert curves == {} and users == {} and len(exclusions) == 1


def test_the_ledger_manifest_records_the_users_and_the_aggregation():
    """The artifact the case rests on has to say who is in it and how the sum
    was formed, not only how many buses there are."""
    import json
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
            / "node_loads" / "typical_day_industry" / "manifest.json")
    if not path.exists():                        # pragma: no cover
        pytest.skip("ledger not built in this checkout")
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["users_with_data"], "no users recorded"
    per_bus = manifest["users_per_bus"]
    assert sorted(int(b) for b in per_bus) == \
        manifest["buses_with_data_on_this_date"]

    # The users are a partition of the buses: each appears once.
    flat = [u for us in per_bus.values() for u in us]
    assert sorted(flat) == manifest["users_with_data"]
    assert len(flat) == len(set(flat))

    # The aggregation is stated, and states that no slot-length factor is
    # applied -- the S1 defect this ledger must not reintroduce.
    agg = manifest["industry_aggregation"]
    assert agg["slot_length_factor"] is None
    assert "average active power" in agg["method"]
