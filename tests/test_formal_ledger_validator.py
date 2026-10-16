# test_formal_ledger_validator.py
"""What may be published as the network's formal absolute load.

The validator exists to answer one question with reasons rather than with a
convention: is this ledger complete, internally consistent, and attributable to
a source that states its unit? Each check below is exercised by breaking a
ledger that otherwise passes, so a check that silently stops working shows up as
a test that no longer fails.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import industry_load as il
import validate_formal_ledger as vfl
from build_industry_node_loads import ALL_BUSES, SLACK_BUS

COLUMNS = ["scenario", "date", "bus", "time_index", "policy", "native_mw",
           "industry_mw", "final_mw", "industry_status", "unit_status",
           "reading_unit", "mapping_version", "native_source",
           "exclusion_reason"]


def _rows(date="2021-01-01"):
    """A ledger that passes everything: every bus, every period, composed."""
    out = []
    for bus in ALL_BUSES:
        for t in range(il.N_SLOTS):
            if bus == SLACK_BUS:
                policy, native, industry = "preserve", 0.0, None
            elif bus < 16:
                policy, native, industry = "preserve", 1.0, None
            elif bus < 24:
                policy, native, industry = "replace", 1.0, 2.0
            else:
                policy, native, industry = "overlay", 1.0, 3.0
            final = native if policy == "preserve" else (
                industry if policy == "replace" else native + industry)
            out.append({
                "scenario": "s", "date": date, "bus": bus, "time_index": t,
                "policy": policy, "native_mw": f"{native:.6f}",
                "industry_mw": "" if industry is None else f"{industry:.6f}",
                "final_mw": f"{final:.6f}",
                "industry_status": "not_applicable" if industry is None
                                   else "present",
                "unit_status": il.UNIT_VERIFIED, "reading_unit": "kW",
                "mapping_version": "industry_bus_map_v2",
                "native_source": "net.load.p_mw", "exclusion_reason": "",
            })
    return out


def _write(tmp_path, rows, manifest_extra=None):
    ledger = tmp_path / "bus_loads_96x33.csv"
    with ledger.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    manifest = {"unit_status": il.UNIT_VERIFIED,
                "reading_unit_evidence": il.UNIT_BASIS_EXTERNAL}
    manifest.update(manifest_extra or {})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return ledger, path


@pytest.fixture
def verified(monkeypatch):
    monkeypatch.setattr(il, "READING_UNIT_EVIDENCE", il.UNIT_BASIS_EXTERNAL)


# ---------------------------------------------------------------------------
# The ledger that passes
# ---------------------------------------------------------------------------

def test_a_complete_verified_ledger_passes(tmp_path, verified):
    """Passing now requires the case type to be stated as well.

    The date rule follows from it, so a caller that has not said which kind of
    case this is has not finished telling the validator what to check -- and
    that omission is reported rather than resolved in the caller's favour.
    """
    ledger, manifest = _write(tmp_path, _rows())
    assert vfl.validate(ledger, manifest, price_dates_available=True,
                        case_type="same_day_replay") == []


def test_a_verified_ledger_still_has_to_state_its_case_type(tmp_path, verified):
    ledger, manifest = _write(tmp_path, _rows())
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True)}
    assert codes == {"case_type_unstated"}, codes


def test_the_validator_is_reading_the_artefact_not_the_builder(tmp_path, verified):
    """A ledger written by something other than the builder is judged on what
    it contains, which is why this task can be run against any ledger at all.

    Handing the slack bus a load breaks two things at once -- a slack carries
    nothing, and the value is not what its `preserve` policy states -- and both
    are reported rather than the first one masking the second.
    """
    rows = _rows()
    rows[0]["final_mw"] = "999.0"
    ledger, manifest = _write(tmp_path, rows)
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True,
                          case_type="same_day_replay")}
    assert codes == {"slack_has_load", "formula_mismatch"}


# ---------------------------------------------------------------------------
# Each check, broken on purpose
# ---------------------------------------------------------------------------

def test_a_unit_with_no_admissible_basis_is_rejected(tmp_path, monkeypatch):
    """The gate is on the **basis**, not on external certification.

    A basis may be an external source or a recorded project ruling; a string
    that names neither is a unit nothing stands behind, and no ledger may rest
    on it. This is the case the gate still refuses -- what it no longer refuses
    is a ruling, which is a basis.
    """
    monkeypatch.setattr(il, "READING_UNIT_EVIDENCE", "no_basis_recorded")
    ledger, manifest = _write(tmp_path, _rows())
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True)}
    assert "unit_not_verified" in codes


def test_a_recorded_project_ruling_passes_the_unit_gate(tmp_path):
    """The ruling basis admits a formal load, and the artefact says which.

    The two bases are not interchangeable: passing here must not require the
    ledger to claim an external source it does not have.
    """
    ledger, manifest = _write(tmp_path, _rows(), {
        "unit_status": il.UNIT_BASIS_RULING_STATUS,
        "reading_unit_evidence": il.UNIT_BASIS_RULING,
    })
    violations = vfl.validate(ledger, manifest, price_dates_available=True,
                              case_type="cross_date_diagnostic")
    assert "unit_not_verified" not in {v["code"] for v in violations}


def test_the_basis_and_the_status_have_to_agree(tmp_path):
    """A ruling status beside an external basis is a ledger that cannot say
    which of the two it rests on."""
    ledger, manifest = _write(tmp_path, _rows(), {
        "unit_status": il.UNIT_VERIFIED,
        "reading_unit_evidence": il.UNIT_BASIS_RULING,
    })
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True,
                          case_type="cross_date_diagnostic")}
    assert "unit_not_verified" in codes


def test_a_manifest_disagreeing_with_the_module_is_rejected(tmp_path, verified):
    """The ledger's own record of its unit has to match the code that produced
    it, or a stale artefact could claim a status it does not have."""
    ledger, manifest = _write(tmp_path, _rows(),
                              {"unit_status": il.UNIT_UNVERIFIED})
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True)}
    assert "unit_not_verified" in codes


def test_a_gap_is_rejected_and_named(tmp_path, verified):
    """A formal ledger may not silently accept a missing period; this is the
    bus 32 case, where the industry term has no data for the whole day."""
    rows = _rows()
    for row in rows:
        if int(row["bus"]) == 32:
            row["final_mw"] = ""
            row["industry_status"] = "excluded_no_data"
    ledger, manifest = _write(tmp_path, rows)
    violations = vfl.validate(ledger, manifest, price_dates_available=True)
    gap = [v for v in violations if v["code"] == "missing_value"][0]
    assert "96 periods" in gap["detail"] and "bus(es) [32]" in gap["detail"]


def test_a_wrong_composition_is_rejected(tmp_path, verified):
    rows = _rows()
    for row in rows:
        if row["policy"] == "overlay" and int(row["time_index"]) == 0:
            row["final_mw"] = row["native_mw"]      # overlay treated as replace
            break
    ledger, manifest = _write(tmp_path, rows)
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True)}
    assert "formula_mismatch" in codes


def test_a_duplicated_period_is_rejected(tmp_path, verified):
    rows = _rows()
    rows.append(dict(rows[0]))
    ledger, manifest = _write(tmp_path, rows)
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True)}
    assert "duplicate_row" in codes


def test_an_uncovered_bus_is_rejected(tmp_path, verified):
    rows = [r for r in _rows() if int(r["bus"]) != 7]
    ledger, manifest = _write(tmp_path, rows)
    violations = vfl.validate(ledger, manifest, price_dates_available=True)
    gap = [v for v in violations if v["code"] == "bus_coverage"][0]
    assert "[7]" in gap["detail"]


def test_an_unknown_policy_is_rejected(tmp_path, verified):
    rows = _rows()
    for row in rows:
        if int(row["bus"]) == 16:
            row["policy"] = "scale"
    ledger, manifest = _write(tmp_path, rows)
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True)}
    assert "policy_incomplete" in codes


def test_a_ledger_spanning_two_dates_is_rejected(tmp_path, verified):
    """The failure this catches is a meter's history being summed into one day:
    the rows then describe more than one date."""
    rows = _rows()
    rows[0]["date"] = "2021-01-02"
    ledger, manifest = _write(tmp_path, rows)
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True)}
    assert "multiple_dates" in codes


def test_an_unalignable_price_is_rejected(tmp_path, verified):
    """The real price curves carry no date that overlaps a load day, so no
    formal case can be date-aligned to one. The validator says so rather than
    letting the declared scenario pass as if it were the load day's price."""
    ledger, manifest = _write(tmp_path, _rows())
    codes = {v["code"] for v in vfl.validate(ledger, manifest)}
    assert "price_not_date_aligned" in codes


def test_a_synthetic_source_is_rejected(tmp_path, verified):
    ledger, manifest = _write(tmp_path, _rows(),
                              {"wholesale_source": "synthetic_v1"})
    codes = {v["code"] for v in
             vfl.validate(ledger, manifest, price_dates_available=True)}
    assert "synthetic_source" in codes


def test_every_violation_says_what_it_means(tmp_path, verified):
    """A rejection a reader cannot act on is a rejection they will override."""
    ledger, manifest = _write(tmp_path, _rows())
    for v in vfl.validate(ledger, manifest):
        assert v["meaning"] in vfl.CODES.values()
        assert v["detail"]


# ---------------------------------------------------------------------------
# The audit ledger is exempt, and cannot pass itself off as formal
# ---------------------------------------------------------------------------

def test_the_audit_ledger_is_rejected_as_a_formal_one(tmp_path):
    """The ledger the project actually has today is an audit ledger. Run
    through the formal checks it is rejected -- which is the honest answer, and
    is why the two are separate artefacts."""
    writes = Path("data/processed/node_loads/typical_day_industry")
    ledger = writes / "bus_loads_96x33_audit.csv"
    if not ledger.exists():                 # pragma: no cover
        pytest.skip("audit ledger not built")
    codes = {v["code"] for v in vfl.validate(ledger, writes / "manifest.json")}

    # The unit gate is the invariant, and the live blocker: it holds whatever
    # date the ledger is built for.
    assert "unit_not_verified" in codes
    assert codes <= set(vfl.CODES), f"unknown code(s) in {codes}"

    # The rest are properties of the particular ledger, not of the rule. This
    # assertion used to pin `missing_value` too, which fired only because the
    # date the ledger held that day happened to have a gap in it -- pinning the
    # set pinned one day's data rather than the contract.
    assert codes, "an audit ledger must not pass the formal checks"
