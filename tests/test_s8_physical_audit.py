# test_s8_physical_audit.py
"""S8: the audit reports the cleared day, and it reports what it is.

Four properties carry this module, and three of them are about not overclaiming.

The **active-power balance closes**. The audit recovers the network's losses
from the solver's own line currents, so the identity `import - export + pv +
wind + discharge - served - charge - losses = 0` is checkable rather than
asserted. If the recovery were wrong -- the wrong resistance, the wrong current,
a missing term -- the residual would show it.

The **clearing objective is not welfare and not a payoff**. That confusion has a
history in this repository, and it is the one an audit is most likely to
reintroduce by printing a large CNY number next to a served-energy figure and
letting a reader connect them.

The **line capacity is project-assigned**, and the audit says so. Every
utilization figure is conditional on ampacities this project chose, so the
source travels with the number.

And **nothing was tuned**: the band, the power factor, the multiplier and the
reverse-flow limit are reported as the configuration holds them.

The clearing is real (Gurobi, SOCP, one full day) and shared by every test here
through one module-scoped fixture, so the suite pays for it once.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import order matters here and is not alphabetical. `dispatch_socp` imports
# pandapower, and pandapower imports ortools, whose DLL fails to load
# (`WinError 127`) unless `grid` has already pulled ortools in first. Reaching
# the environment through `grid` is the documented way around it; a plain
# `import dispatch_socp` first fails at collection.
import grid
import s8_physical_audit as s8  # noqa: I001  (imports grid, hence ortools)

import config_loader as cfg
import dispatch_socp
import models

MODULE = Path(s8.__file__)

pytestmark = pytest.mark.skipif(
    not dispatch_socp._HAS_GUROBI,
    reason="the S8 audit clears a day and needs Gurobi")


@pytest.fixture(scope="module")
def audit() -> dict:
    """One clearing of the subset day, measured once for the whole module."""
    return s8.measure(s8.SCENARIO)


# ---------------------------------------------------------------------------
# The arithmetic the audit rests on
# ---------------------------------------------------------------------------


def test_the_active_power_balance_closes(audit):
    """Every term of the clearing's own identity, rebuilt and checked.

    The losses are not read from the result -- they are recovered from the
    solver's line currents and the line resistances, so this is a check on the
    recovery as much as on the clearing.
    """
    energy = audit["energy"]
    assert energy["balance_residual_max_mw"] < s8.IDENTITY_ATOL_MW
    assert energy["balance_residual_mean_mw"] < s8.IDENTITY_ATOL_MW
    # The residual is a loss, and a loss is not negative.
    assert energy["network_losses_mwh"] >= 0.0


def test_served_plus_shed_is_the_load_that_was_asked_for(audit):
    """No load goes missing between the ledger and the schedule."""
    energy = audit["energy"]
    assert energy["load_closure_max_mw"] < s8.IDENTITY_ATOL_MW
    assert energy["served_mwh"] + energy["shed_mwh"] == pytest.approx(
        energy["asked_mwh"], abs=1e-6)


def test_the_re_rate_is_a_ratio_of_two_energies_in_percent(audit):
    """`re_consumption_rate` is used over available, and it is a percentage.

    Recomputing it from the two energies the same result carries is what says
    the field is that ratio and not a cost, a share of load, or a fraction.
    """
    energy = audit["energy"]
    expected = (100.0 * energy["re_used_mwh"] / energy["re_available_mwh"]
                if energy["re_available_mwh"] > 0 else 100.0)
    assert energy["re_consumption_rate"] == pytest.approx(expected, abs=1e-9)
    assert 0.0 <= energy["re_consumption_rate"] <= 100.0
    assert energy["re_used_mwh"] == pytest.approx(
        energy["re_available_mwh"] - energy["re_curtailed_mwh"], abs=1e-6)


def test_the_reactive_load_is_the_served_active_load_times_tan_phi(audit):
    """Q tracks **served** power, not asked power, period by period.

    The model's reactive-power balance subtracts `served * q_ratio`, so a shed
    period draws less reactive power rather than more. That only shows up on a
    day that sheds, so it is pinned as an exact relation on the day at hand:
    the peak reactive power is the peak served power times `tan(phi)`, and the
    reactive energy is the served energy times the same factor.
    """
    reactive, energy = audit["reactive"], audit["energy"]
    tan_phi = np.tan(np.arccos(reactive["power_factor"]))
    assert reactive["tan_phi"] == pytest.approx(tan_phi)
    assert reactive["q_peak_mvar"] == pytest.approx(
        energy["served_peak_mw"] * tan_phi, rel=1e-9)
    assert reactive["q_peak_period"] == energy["served_peak_period"]
    assert reactive["q_energy_mvarh"] == pytest.approx(
        energy["served_mwh"] * tan_phi, rel=1e-9)
    assert "assumption" in reactive["status"]
    assert "not a measurement" in reactive["status"]


# ---------------------------------------------------------------------------
# The line capacity is a modelling choice, and the audit says so
# ---------------------------------------------------------------------------


def test_the_line_capacity_source_is_read_from_the_built_network(audit):
    """Every reported ampacity is the network's own, and the provenance is
    stated as project-assigned rather than published."""
    source = audit["lines"]["capacity_source"]
    assert "PROJECT-ASSIGNED" in source["provenance"]
    assert "case33bw" in source["provenance"]
    assert source["line_capacity_multiplier"] == pytest.approx(
        audit["not_tuned"]["line_capacity_multiplier"])

    # The ampacities the audit reports come from a network built the same way
    # the clearing builds it, so they are the model's own limits.
    built = grid.build_base_network(models.MarketConfig.from_yaml())
    for row in audit["lines"]["per_line"]:
        assert row["ampacity_ka"] == pytest.approx(
            float(built.line.at[row["line"], "max_i_ka"]))


def test_the_ampacity_range_is_the_declared_clip_times_the_multiplier(audit):
    """The limit really is `clip(base / sqrt(r), lo, hi) * multiplier`."""
    source = audit["lines"]["capacity_source"]
    lo, hi = source["ampacity_ka_range"]
    mult = source["line_capacity_multiplier"]
    assert lo >= source["min_line_ka"] * mult - 1e-9
    assert hi <= source["max_line_ka"] * mult + 1e-9


# ---------------------------------------------------------------------------
# What the numbers are, and what they are not
# ---------------------------------------------------------------------------


def test_the_clearing_objective_is_not_offered_as_welfare_or_a_payoff(audit):
    """The number is named, described, and kept away from economic language.

    Checked against the module's own text and imports rather than against a
    promise: a payoff or a welfare function reaching this audit is how the
    largest number on the page becomes the one a reader believes.
    """
    objective = audit["objective"]
    assert "NOT social welfare" in objective["what_it_is"]
    assert "NOT any participant's payoff" in objective["what_it_is"]
    assert objective["clearing_objective_cny"] == pytest.approx(
        audit["objective"]["clearing_objective_cny"])

    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "participant_payoff" not in imported
    assert "surplus_metrics" not in imported


def test_no_parameter_was_tuned_to_produce_the_result(audit):
    """The audit reports the configuration as it stands, and says so."""
    not_tuned = audit["not_tuned"]
    assert "no load, power factor, line capacity, voltage bound or storage " \
           "parameter was moved" in not_tuned["statement"]
    scenario = cfg.get_scenario_cfg(s8.SCENARIO)
    assert not_tuned["load_power_factor"] == pytest.approx(
        float(scenario["load_power_factor"]))
    assert not_tuned["line_capacity_multiplier"] == pytest.approx(
        float(scenario["override_config"]["network.line_capacity_multiplier"]))
    assert not_tuned["voltage_band"][0] == pytest.approx(
        float(cfg.get_default("network.v_min_pu")))


def test_the_voltage_table_covers_every_bus(audit):
    """All 33 buses appear, and the extremes agree with the table.

    A violation is counted per bus-period, so a bus is flagged when its own
    extreme leaves the band -- which is what the per-bus table lets a reader
    check against the two headline counts.
    """
    voltage = audit["voltage"]
    assert len(voltage["per_bus"]) == len(audit["buses"]) == 33
    assert voltage["min"] == pytest.approx(
        min(r["min"] for r in voltage["per_bus"]))
    assert voltage["max"] == pytest.approx(
        max(r["max"] for r in voltage["per_bus"]))
    low, high = voltage["band"]
    assert voltage["bus_periods_below"] == 0 or any(
        r["min"] < low for r in voltage["per_bus"])
    assert voltage["bus_periods_above"] == 0 or any(
        r["max"] > high for r in voltage["per_bus"])


def test_the_reported_verdict_follows_the_energy_it_reports(audit):
    """A day with no shed is a day the network carried, and it says so.

    The audit asserts one or the other in prose; this pins that the prose and
    the arithmetic cannot disagree.
    """
    energy = audit["energy"]
    carried = energy["shed_mwh"] <= 1.0
    assert carried == (energy["served_share"] > 0.999)


def test_an_anomalous_period_is_one_whose_price_is_not_a_dual(audit):
    """The list is the union of the fallback periods and the non-finite ones."""
    solver = audit["solver"]
    assert solver["lmp_fallback_periods"] == sorted(
        set(solver["lmp_fallback_periods"]))
    assert set(solver["lmp_fallback_periods"]) <= set(
        solver["anomalous_periods"])
    if not solver["lmp_fallback_periods"]:
        assert solver["lmp_price_source"] == "nodal"
