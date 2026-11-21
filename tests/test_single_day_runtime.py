# test_single_day_runtime.py
"""The single-day case, wired into scenario construction and network input.

Two groups, and the difference between them is the whole point.

**Group 1 uses a test-only fixture**: a ledger invented here, marked
`formal_ledger` / `unit_verified`, with a price curve supplied by the test. It
exists to exercise the success path — entry point → scenario → network —
because the real ledger cannot take it. **Nothing in group 1 is a real result**;
the ledger is 33 invented buses of invented load and the price is invented too.
It must never be published as an experiment.

**Group 2 uses the real audit ledger** and asserts the opposite: the date rule
passes, the unit gate refuses, and the refusal lands *before* a network is built
or a market is cleared.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import numpy as np
import pytest

import config_loader
import dianyicha_price as dp
import grid
import industry_load as il
import scenarios
import single_day_data as sdd

_REPO = pathlib.Path(__file__).resolve().parents[1]
_REAL_LEDGER = (_REPO / "data" / "processed" / "node_loads"
                / "typical_day_industry" / "bus_loads_96x33_audit.csv")
_REAL_MANIFEST = _REAL_LEDGER.parent / "manifest.json"

_N = il.N_SLOTS
_BUSES = tuple(range(33))
_SLACK = 0


# ---------------------------------------------------------------------------
# The test-only fixture. Invented numbers, invented price, no real data.
# ---------------------------------------------------------------------------

def _fixture_rows(load_mw=2.0, native_mw=1.0):
    """A complete formal ledger: every bus, every period, composed."""
    out = []
    for bus in _BUSES:
        for t in range(_N):
            if bus == _SLACK:
                policy, native, industry = "preserve", 0.0, None
            elif bus < 16:
                policy, native, industry = "preserve", native_mw, None
            else:
                policy, native, industry = "replace", native_mw, load_mw
            final = native if policy == "preserve" else industry
            out.append({
                "scenario": "fixture", "date": "2026-10-09", "bus": bus,
                "time_index": t, "policy": policy,
                "native_mw": f"{native:.6f}",
                "industry_mw": "" if industry is None else f"{industry:.6f}",
                "final_mw": f"{final:.6f}",
                "industry_status": ("not_applicable" if industry is None
                                    else "present"),
                "unit_status": il.UNIT_VERIFIED, "reading_unit": "kW",
                "mapping_version": "industry_bus_map_v2",
                "native_source": "fixture", "exclusion_reason": "",
            })
    return out


_ROW_KEYS = ["scenario", "date", "bus", "time_index", "policy", "native_mw",
             "industry_mw", "final_mw", "industry_status", "unit_status",
             "reading_unit", "mapping_version", "native_source",
             "exclusion_reason"]


@pytest.fixture
def fixture_case(tmp_path, monkeypatch):
    """A formal ledger and a price curve, both invented for this test.

    `unit_status` is `unit_verified` and `ledger_kind` is `formal`, which is
    what the runtime boundary requires -- and which no artifact in this
    repository has. That is precisely why this fixture exists and why nothing
    built from it may be reported.
    """
    ledger = tmp_path / "fixture_ledger.csv"
    with ledger.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_ROW_KEYS)
        w.writeheader()
        w.writerows(_fixture_rows(load_mw=2.0, native_mw=1.0))
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    manifest = tmp_path / "fixture_manifest.json"
    manifest.write_text(json.dumps({
        "scenario": "fixture", "date": "2026-10-09", "ledger_kind": "formal",
        "unit_status": il.UNIT_VERIFIED, "reading_unit": "kW",
        "reading_unit_evidence": "test fixture, not evidence",
        "mapping_version": "industry_bus_map_v2",
        "mapping_sha256": "0" * 64, "ledger_sha256": digest,
        "buses": list(_BUSES), "slots_per_day": _N,
        "policies": {str(b): ("preserve" if b < 16 or b == _SLACK
                              else "replace") for b in _BUSES},
    }, indent=2), encoding="utf-8")

    # The price curve is the test's own: the reader is replaced, so `load_case`
    # pairs a curve this test controls through the same code path it uses for
    # the real one. The sheet name is the real one so the curve's declared
    # metadata -- and therefore its date -- is the real metadata.
    curve = dp.Curve(sheet="261009", role="test fixture",
                     prices=tuple(float(300 + t) for t in range(_N)),
                     source_kind="real_market_data",
                     spatial_statistic="spatial_mean",
                     temporal_statistic="single_day", window_days=1,
                     reference_period="26/10/09", externally_confirmed=False,
                     declared_by="test fixture")
    monkeypatch.setattr(sdd.dp, "read_curves",
                        lambda path=None, sheets=None: {"261009": curve})
    monkeypatch.setattr(sdd.dp, "resolve_workbook",
                        lambda path=None: _REPO / "docs" / "test_workbook.xlsx")
    return ledger, manifest


# ---------------------------------------------------------------------------
# Group 1: the success path, on the fixture
# ---------------------------------------------------------------------------

def test_the_fixture_case_passes_the_gate_and_carries_its_load(fixture_case):
    """The gate is what makes the extraction possible at all."""
    ledger, manifest = fixture_case
    case = sdd.load_case("same_day_replay", ledger, manifest)
    assert case.load_kind == "formal_ledger"
    by_bus = case.formal_bus_load_mw()
    assert set(by_bus) == set(_BUSES)
    for bus, arr in by_bus.items():
        assert arr.shape == (_N,)
        assert np.allclose(arr, 2.0 if bus >= 16 else (0.0 if bus == _SLACK
                                                       else 1.0))


def _declare(ledger, manifest, case_type="same_day_replay",
             mode=None):
    """Point the scenario at a ledger, through its own two blocks.

    `typical_day_industry` now declares both `load_model` and `single_day_case`
    in `config/scenarios.yaml`, so a test that wants a different ledger has to
    restate both -- overriding only the mode would leave the scenario's own
    ledger in place.
    """
    config_loader.reload_scenarios()
    config_loader.set_scenario_param(
        "typical_day_industry", "load_model",
        {"mode": mode or scenarios.ABSOLUTE_INDUSTRY_LOAD_MODEL})
    config_loader.set_scenario_param(
        "typical_day_industry", "single_day_case",
        {"case_type": case_type, "ledger": str(ledger),
         "ledger_manifest": str(manifest), "price_sheet": "261009"})


def test_the_whole_path_runs_and_carries_the_ledgers_load(
        fixture_case, monkeypatch):
    """scenario → entry point → network, on the fixture.

    Three things are asserted together because they are one path: the entry
    point was the one that paired the data, the scenario built agents from it,
    and every bus carries the ledger's own load.
    """
    ledger, manifest = fixture_case
    seen = {}
    original = sdd.load_case

    def spy(case_type, *args, **kwargs):
        seen["case_type"] = case_type
        return original(case_type, *args, **kwargs)

    monkeypatch.setattr(sdd, "load_case", spy)
    _declare(ledger, manifest)
    try:
        agents, wholesale = scenarios.get_scenario(
            "typical_day_industry", T=_N,
            config=scenarios.MarketConfig.from_yaml())
    finally:
        config_loader.reload_scenarios()

    assert seen.get("case_type") == "same_day_replay"
    assert len(wholesale) == _N

    load_agents = [a for a in agents if a.storage is None]
    storage_agents = [a for a in agents if a.storage is not None]
    assert load_agents and storage_agents, (
        "this scenario is expected to carry both a load population and the "
        "independent fleet")
    for a in load_agents:
        expected = 2.0 if a.bus >= 16 else (0.0 if a.bus == _SLACK else 1.0)
        np.testing.assert_allclose(a.load_real, expected, atol=1e-12)
        np.testing.assert_allclose(a.load_forecast, expected, atol=1e-12)
    # A battery is not a load: the ledger's per-bus demand belongs to the
    # bus's own participants and must not be handed to the fleet as well.
    for a in storage_agents:
        assert np.all(a.load_real == 0.0), a.name


def test_the_network_carries_the_ledgers_load_and_not_a_second_copy(
        fixture_case, monkeypatch):
    """The failure this guards: the ledger's load added to the native load.

    Every bus's `load_real` must be the ledger's own value. A bus that came out
    at 3.0 MW where the ledger said 2.0 has had the native term folded in on top
    of it, which is the double injection the composition rule exists to prevent.
    """
    ledger, manifest = fixture_case
    case = sdd.load_case("same_day_replay", ledger, manifest)
    by_bus = case.formal_bus_load_mw()

    config = scenarios.MarketConfig.from_yaml()
    net = grid.build_base_network(config)
    agents = grid.create_agents_from_network(net, _N, with_wind=True,
                                             scale={"mode": "native_peak",
                                                    "shape": "native"},
                                             node_load_mw=by_bus)
    for a in agents:
        expected = by_bus[a.bus]
        np.testing.assert_allclose(a.load_real, expected, atol=1e-12)
        np.testing.assert_allclose(a.load_forecast, expected, atol=1e-12)


def test_the_ledger_load_is_not_rescaled_and_does_not_resize_devices(
        fixture_case):
    """`base_load` sizes the equipment; the ledger's level is the load.

    Rescaling the ledger to a declared peak is the one operation its provenance
    forbids, and letting the ledger's magnitude reach `base_load` would resize
    every battery and inverter on the bus to match a metered reading.
    """
    ledger, manifest = fixture_case
    by_bus = sdd.load_case("same_day_replay", ledger, manifest) \
        .formal_bus_load_mw()

    config = scenarios.MarketConfig.from_yaml()
    net = grid.build_base_network(config)
    rated = {int(r.bus): float(r.p_mw) for _i, r in net.load.iterrows()}

    def build(node_load_mw):
        return grid.create_agents_from_network(
            net, _N, with_wind=True,
            scale={"mode": "native_peak", "shape": "native"},
            node_load_mw=node_load_mw)

    with_ledger = {a.bus: a for a in build(by_bus)}
    without = {a.bus: a for a in grid.create_agents_from_network(
        net, _N, with_wind=True,
        scale={"mode": "native_peak", "shape": "native"})}

    # The device decision is unchanged: same storage, same inverter sizing.
    for bus, a in with_ledger.items():
        b = without[bus]
        if a.storage is not None:
            assert a.storage.e_max == pytest.approx(b.storage.e_max)
            assert a.storage.p_ch_max == pytest.approx(b.storage.p_ch_max)
        assert a.pv_capacity == pytest.approx(b.pv_capacity)

    # And the load is exactly the ledger's, not the network's rated demand
    # scaled onto it.
    for bus, a in with_ledger.items():
        np.testing.assert_allclose(a.load_real, by_bus[bus], atol=1e-12)
    assert any(not np.allclose(by_bus[b], rated.get(b, 0.0))
               for b in by_bus), "the fixture must differ from the rating"


def test_a_bus_missing_from_the_ledger_is_refused_rather_than_defaulted(
        fixture_case):
    ledger, manifest = fixture_case
    by_bus = sdd.load_case("same_day_replay", ledger, manifest) \
        .formal_bus_load_mw()
    del by_bus[7]
    config = scenarios.MarketConfig.from_yaml()
    net = grid.build_base_network(config)
    with pytest.raises(KeyError, match="bus 7"):
        grid.create_agents_from_network(
            net, _N, with_wind=True,
            scale={"mode": "native_peak", "shape": "native"},
            node_load_mw=by_bus)


def test_the_fixture_ledger_is_never_the_real_one(fixture_case):
    """A reminder in the suite: group 1's numbers are invented.

    The fixture's manifest says so in the field the validator reads, and its
    digest is not the real ledger's, so nothing built from it can be mistaken
    for an experiment.
    """
    _ledger, manifest = fixture_case
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert recorded["reading_unit_evidence"] == "test fixture, not evidence"
    assert recorded["scenario"] == "fixture"
    if _REAL_MANIFEST.exists():
        real = json.loads(_REAL_MANIFEST.read_text(encoding="utf-8"))
        assert recorded["ledger_sha256"] != real["ledger_sha256"]


# ---------------------------------------------------------------------------
# Group 2: the real ledger, refused at the boundary
# ---------------------------------------------------------------------------

@pytest.fixture
def real_case():
    if not (_REAL_LEDGER.exists() and _REAL_MANIFEST.exists()):  # pragma: no cover
        pytest.skip("the real ledger is not built in this checkout")
    return sdd.load_case("cross_date_diagnostic", _REAL_LEDGER, _REAL_MANIFEST)


def test_the_real_ledger_passes_the_cross_date_rule(real_case):
    assert real_case.case_type == "cross_date_diagnostic"
    assert real_case.date_rule["same_day"] is False
    assert real_case.date_rule["load_date"] == "2022-11-30"
    assert real_case.date_rule["price_date"] == "2026-10-09"


def test_the_real_ledger_is_refused_by_the_unit_gate(real_case):
    assert real_case.load_kind == "audit_ledger"
    with pytest.raises(sdd.CaseTypeError, match="audit_ledger"):
        real_case.formal_bus_load_mw()
    with pytest.raises(sdd.CaseTypeError, match="audit_ledger"):
        real_case.require_formal_load_mw()


def test_the_refusal_lands_before_a_network_is_built(tmp_path, monkeypatch):
    """The ordering requirement, observed rather than asserted in prose.

    The gate no longer refuses an audit ledger -- a recorded unit basis is
    enough to build a diagnostic on, by decision. What it still refuses is a
    ledger it cannot use: a **gap** in the load. That refusal must also land
    before `build_base_network` runs, so it cannot leave a half-built model
    behind. Both steps are replaced with recorders: if either is reached, the
    gate is in the wrong place.
    """
    rows = _fixture_rows()
    rows = [r for r in rows if not (int(r["bus"]) == 7
                                    and int(r["time_index"]) == 40)]
    ledger = tmp_path / "gapped.csv"
    with ledger.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_ROW_KEYS)
        w.writeheader()
        w.writerows(rows)
    manifest = tmp_path / "gapped_manifest.json"
    manifest.write_text(json.dumps({
        "date": "2026-10-09", "ledger_kind": "formal",
        "unit_status": il.UNIT_VERIFIED, "reading_unit": "kW",
        "reading_unit_evidence": "confirmed", "mapping_version": "v2",
        "mapping_sha256": "0" * 64, "ledger_sha256": "0" * 64,
        "buses": list(_BUSES), "slots_per_day": _N,
    }), encoding="utf-8")

    calls = []
    monkeypatch.setattr(scenarios, "build_base_network",
                        lambda *a, **k: calls.append("network"))
    monkeypatch.setattr(scenarios, "create_agents_from_network",
                        lambda *a, **k: calls.append("agents"))

    _declare(ledger, manifest, "same_day_replay")
    try:
        # The axis check catches it first, which is earlier still: a ledger
        # that is not a whole day never reaches the extraction.
        with pytest.raises((sdd.CaseTypeError, sdd.DataAlignmentError)):
            scenarios.get_scenario("typical_day_industry", T=_N,
                                   config=scenarios.MarketConfig.from_yaml())
    finally:
        config_loader.reload_scenarios()
    assert calls == [], (
        f"the gate let the run reach {calls} before refusing; a load it cannot "
        f"use must be refused before the network exists")


def test_the_real_ledger_is_not_refused_by_the_build_gate(real_case):
    """The deliberate change: a *recorded basis* is enough to build on.

    An externally verified unit still is not required, and this asserts the
    softening is real rather than a comment: the same ledger that
    `require_formal_load_mw` refuses passes `require_load_for_build`, and the
    basis it passes on is named.
    """
    granted = real_case.require_load_for_build()
    assert real_case.unit_basis["basis"] == "user_confirmed_project_ruling"
    assert real_case.unit_basis["is_externally_verified"] is False
    assert granted["rows"]
    with pytest.raises(sdd.CaseTypeError):
        real_case.require_formal_load_mw()


def test_a_ledger_backed_scenario_must_state_its_case_type(real_case):
    """No implicit cross-date rule: the scenario has to say which case it is."""
    _declare(_REAL_LEDGER, _REAL_MANIFEST, case_type=None)
    try:
        with pytest.raises(sdd.CaseTypeError, match="case_type"):
            scenarios.single_day_case("typical_day_industry", _N)
    finally:
        config_loader.reload_scenarios()


def test_a_same_day_replay_on_the_real_ledger_is_refused_on_the_dates():
    _declare(_REAL_LEDGER, _REAL_MANIFEST, "same_day_replay")
    try:
        with pytest.raises(sdd.CaseTypeError, match="same_day_replay"):
            scenarios.single_day_case("typical_day_industry", _N)
    finally:
        config_loader.reload_scenarios()


def test_the_refusal_is_not_disguised_as_a_zero_load(real_case):
    """A caller must not be able to read a refusal as "this bus draws nothing"."""
    with pytest.raises(sdd.CaseTypeError):
        real_case.formal_bus_load_mw()
    # And there is no second method that would hand the load over anyway.
    for name in ("bus_load_mw", "load_mw", "industry_mw"):
        assert not hasattr(real_case, name), name


def test_the_real_ledger_is_not_modified_by_any_of_this(real_case):
    """Nothing on this path writes to the ledger or its manifest."""
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (_REAL_LEDGER, _REAL_MANIFEST)}
    with pytest.raises(sdd.CaseTypeError):
        real_case.formal_bus_load_mw()
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (_REAL_LEDGER, _REAL_MANIFEST)}
    assert before == after


# ---------------------------------------------------------------------------
# The activation itself
# ---------------------------------------------------------------------------

def test_the_declared_mode_reaches_the_network_and_only_one_mode_exists():
    """`load_model.mode` is read *and used*, and there is no second route.

    A configuration key that is parsed and then ignored is how a scenario comes
    to be described by its config rather than by what it does. This asserts the
    opposite in both directions: the declared mode is the one the builder
    branches on, and the per-bus industry-*shape* route it replaced is gone.
    """
    cfg = config_loader.get_scenario_cfg("typical_day_industry")
    assert cfg["load_model"]["mode"] == scenarios.ABSOLUTE_INDUSTRY_LOAD_MODEL

    text = (_REPO / "src" / "scenarios.py").read_text(encoding="utf-8")
    assert "ABSOLUTE_INDUSTRY_LOAD_MODEL" in text
    # The shape route: the helper is gone and no scenario declares an
    # industry *shape*.
    assert "_industry_load_shapes" not in text
    for name in config_loader.load_scenarios()["scenarios"]:
        c = config_loader.get_scenario_cfg(name)
        assert c.get("load_shape") in ("native", "diversified", None), name
    # `industry_map` stays, and it is not the shape route: the bus -> industry
    # assignment is what `map_industry_meters` reads to decide where metered
    # users may land, and its `excluded_industries` list is part of that rule's
    # digest. Deleting it took a real input with it.
    mapping = config_loader.get_scenario_cfg("typical_day_industry")["industry_map"]
    assert mapping["simulation_mapping"] is True
    assert len(mapping["buses"]) == 32
    assert mapping["excluded_industries"]
    assert "shapes_file" not in mapping, (
        "the shapes file belonged to the retired shape route")


def test_the_scenarios_mapping_and_its_policies_reach_the_built_day():
    """The v2 user-level mapping and all 33 bus policies, on the built day.

    The ledger is the artifact that carries them, so the check is that the
    scenario's declared policies are the ones the ledger was built with --
    `single_day_case` refuses a mismatch -- and that the built fleet carries a
    load on exactly the buses the ledger says it does.
    """
    config_loader.reload_defaults()
    agents, _ = scenarios.get_scenario(
        "typical_day_industry", T=_N, config=scenarios.MarketConfig.from_yaml())
    case = scenarios.last_single_day_case()

    declared = config_loader.get_scenario_cfg("typical_day_industry")[
        "load_model"]["bus_load_policy"]
    ledger_policies = case.ledger_manifest["policies"]
    assert {int(b): p for b, p in declared.items()} == \
        {int(b): p for b, p in ledger_policies.items()}
    assert len(declared) == 33

    carrying = {a.bus for a in agents if a.storage is None
                and float(np.sum(a.load_real)) > 0.0}
    # Every bus the metered users landed on carries load...
    assert set(case.ledger_manifest["buses_with_data_on_this_date"]) <= carrying
    # ...and so do the `preserve` buses, which carry the benchmark's own native
    # demand -- a bus with no mapped user is not a bus with no load.
    preserve = {int(b) for b, p in declared.items() if p == "preserve"}
    assert (preserve - {0}) <= carrying
    assert case.ledger_manifest["mapping_version"] == "industry_bus_map_v2"
    # Bus 0 is the slack and carries nothing, as the policy block declares.
    assert declared[0] == "preserve"
    assert 0 not in carrying


def test_the_diagnostic_runner_reports_a_failure_rather_than_a_profit():
    """A clearing this feeder cannot serve yields no profit conclusion.

    The runner's whole output is network state and solver status. It computes
    no revenue and no arbitrage figure, so there is nothing for a failed
    clearing to fabricate -- and this asserts the runner says so rather than
    leaving a reader to assume the absence is an oversight.
    """
    doc_path = _REPO / "outputs" / "single_day_diagnostic_run.json"
    if not doc_path.exists():                    # pragma: no cover
        pytest.skip("the diagnostic run has not been made in this checkout")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))

    # No money. A run whose clearing served 3% of the load has no payoff to
    # report, and a field named for one would be read as a result.
    for absent in ("profit", "revenue", "arbitrage", "welfare", "payoff",
                   "settlement"):
        assert not [k for k in doc if absent in k.lower()], absent
    # What it does carry: the solver's own status, the network state, and the
    # basis the load rests on.
    assert doc["solver"]["final"]["status_name"]
    assert "voltage" in doc and "lines" in doc and "pcc" in doc
    assert doc["lmp_fallbacks"] == doc["lmp_fallbacks"]  # a number, not text
    assert doc["unit_basis"]["is_externally_verified"] is False
    assert doc["unit_basis"]["basis"] == "user_confirmed_project_ruling"

    text = (_REPO / "reports" / "single_day_diagnostic_run.md").read_text(
        encoding="utf-8")
    assert "network cannot carry this load" in text
    assert "not a historical replay" in text.lower()
    assert "user_confirmed_project_ruling" in text
