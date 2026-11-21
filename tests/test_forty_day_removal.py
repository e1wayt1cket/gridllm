# test_forty_day_removal.py
"""The fixed 40-day system is gone, and cannot come back by omission.

Deleting a constant is easy; what this suite pins is that nothing still assumes
the structure it carried. A caller reaching for a dataset length, a default that
quietly restores one, a synthetic provider a formal run can reach by accident --
each of those would put the assumption back without the number ever appearing.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(Path := pathlib.Path))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config_loader as cfg
import market_opportunity as mo
import train_rl
import wholesale_price as wp

SRC = Path(__file__).resolve().parents[1] / "src"

#: Names that carried the fixed dataset, its split, or its generator defaults.
REMOVED_NAMES = ("LIBRARY_DAYS", "SPLIT_SIZES", "ENSEMBLE_SIZE", "MASTER_SEED",
                 "DEFAULT_TRAIN_SCENARIO")


# ---------------------------------------------------------------------------
# No residue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_no_source_file_still_names_a_removed_constant(name):
    """Not just "unused": gone. A constant left in place is one a later caller
    reaches for, and the assumption is back without the number appearing
    anywhere a search would find it."""
    # Word boundaries, so the deliberately-namespaced non-formal constants
    # (`NON_FORMAL_LIBRARY_DAYS`) are not mistaken for the name that carried
    # the project-wide assumption.
    pattern = re.compile(rf"\b{name}\b")
    hits = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) and not line.lstrip().startswith("#"):
                hits.append(f"{path.name}:{number}")
    assert not hits, f"{name} is still live in {hits}"


def test_the_wholesale_config_file_declares_no_dataset_length():
    """Asserted against the file, not the loaded dict: `day_id` is a runtime
    knob `scenarios.py` writes with `set_default`, so after any scenario build
    the key exists in memory whatever the file says. What must not exist is a
    declared value."""
    text = (pathlib.Path(__file__).resolve().parents[1] / "config"
            / "defaults.yaml").read_text(encoding="utf-8")
    for key in ("number_of_days", "master_seed", "library_file", "day_id"):
        declared = re.search(rf"^\s+{key}:", text, re.M)
        assert declared is None, (
            f"defaults.yaml declares {key}; a dataset length or a reference "
            "into one outlives whatever it was written for")


def test_the_scheduler_and_the_audit_hold_no_split_or_size():
    assert not hasattr(mo, "ENSEMBLE_SIZE") and not hasattr(mo, "MASTER_SEED")
    assert not hasattr(train_rl, "DEFAULT_TRAIN_SCENARIO")


def test_the_audit_ensemble_size_is_the_callers():
    """`ensemble_specs` no longer has a size to fall back on, so it cannot
    build an ensemble the caller never asked for."""
    with pytest.raises(TypeError):
        mo.ensemble_specs()                       # missing n and seed
    assert len(mo.ensemble_specs(5, seed=1)) == 5
    assert len(mo.ensemble_specs(17, seed=1)) == 17


def test_a_training_run_must_name_its_scenarios():
    """A default research scenario is a decision made once and then inherited
    by every run that forgets to state one."""
    with pytest.raises(SystemExit, match="no --scenarios"):
        train_rl._resolve_train_scenarios(None)
    # And what it does name must be in the formal set, which today holds one
    # scenario: every other member prices from a generator or the retired map.
    import scenarios as _sc
    formal = _sc.formal_scenarios()
    assert train_rl._resolve_train_scenarios(",".join(formal)) == formal
    with pytest.raises(ValueError, match="not in the formal scenario set"):
        train_rl._resolve_train_scenarios("baseline,high_re")


def test_nothing_in_the_old_split_shape_survives():
    """The 20/10/10 deal was a consequence of a 40-day library. Neither the
    proportions nor the sizes may be declared where the data is not."""
    source = (SRC / "day_rotation.py").read_text(encoding="utf-8")
    for fragment in ("20", "10", "40"):
        assert f"= {fragment}" not in source.replace("0x", ""), fragment


# ---------------------------------------------------------------------------
# A synthetic provider is not reachable from the formal path
# ---------------------------------------------------------------------------

def test_the_shipped_default_closes_the_gate():
    """The suite opens this gate for itself; the module's own default is what a
    formal run gets, so it is asserted from the source rather than from
    whatever the session left behind."""
    source = (SRC / "wholesale_price.py").read_text(encoding="utf-8")
    assert "ALLOW_NON_FORMAL_SOURCES = False" in source


def test_a_synthetic_provider_is_refused_when_the_gate_is_closed(monkeypatch):
    monkeypatch.setattr(wp, "ALLOW_NON_FORMAL_SOURCES", False)
    with pytest.raises(ValueError, match="generates a price series"):
        wp.get_provider("synthetic_v1")
    with pytest.raises(ValueError, match="generates a price series"):
        wp.get_provider("library")


def test_a_synthetic_provider_can_still_be_built_deliberately(monkeypatch):
    """Tests and audits need a market with a known shape. Wanting one is a
    thing a caller states, not a thing a config can leave lying around."""
    monkeypatch.setattr(wp, "ALLOW_NON_FORMAL_SOURCES", True)
    assert wp.get_provider("synthetic_v1").name == "synthetic_v1"


def test_the_library_size_lives_with_the_generator_not_with_the_project():
    """The number is the generator's own; it is not project configuration a
    real dataset would then be measured against."""
    assert "number_of_days" not in cfg.get_default("wholesale_market")
    monkeypatch_none = cfg.get_default("wholesale_market")
    assert monkeypatch_none.get("number_of_days") is None


# ---------------------------------------------------------------------------
# Which scenarios may enter formal evaluation
# ---------------------------------------------------------------------------

def test_a_generated_price_series_is_not_a_formal_source():
    ready, why = wp.formal_data_readiness("synthetic_v1")
    assert ready is False and "does not read a measured market" in why
    ready, why = wp.formal_data_readiness("legacy")
    assert ready is False
    ready, _ = wp.formal_data_readiness("dianyicha")
    assert ready is True


def test_the_scenario_that_prices_from_a_generator_is_marked_not_formal_ready():
    scenario = cfg.get_scenario_cfg("typical_day")
    assert scenario.get("formal_data_ready") is False
    assert "synthetic_v1" in scenario.get("formal_data_reason", "")


def test_no_scenario_priced_by_a_generator_claims_to_be_formal_ready():
    """The rule is over the source, so a scenario added later cannot be
    formal-ready by omission."""
    offenders = []
    for name in cfg.load_scenarios()["scenarios"]:
        scenario = cfg.get_scenario_cfg(name)
        source = str(scenario.get("wholesale_source", "legacy"))
        ready, _ = wp.formal_data_readiness(source)
        if not ready and scenario.get("formal_data_ready") is not False:
            offenders.append((name, source))
    assert not offenders, (
        f"{offenders} price from a generator but are not marked "
        "formal_data_ready: false")


def test_a_diversified_shape_does_not_bypass_the_real_data_requirement():
    """`load_shape: diversified` synthesises the load; the price may still be a
    generated series. Neither bypasses the gate, because the gate is on the
    data, not on the shape."""
    for name in cfg.load_scenarios()["scenarios"]:
        scenario = cfg.get_scenario_cfg(name)
        if scenario.get("load_shape") != "diversified":
            continue
        source = str(scenario.get("wholesale_source", "legacy"))
        ready, _ = wp.formal_data_readiness(source)
        assert ready is False, f"{name} is priced by a measured market after all"
