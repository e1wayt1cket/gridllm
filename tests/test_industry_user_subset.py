# test_industry_user_subset.py
"""The subset selection: what it may change, and what it may not.

The properties this suite pins hardest are the negative ones. No reading is
rescaled -- a user's curve is its meters' readings times their own multipliers
and nothing else. No user is split across buses, and no meter is counted twice.
The selection reads no price, no LMP and no learning result: it is a statement
about measured load and the benchmark's own demand, and if a market outcome
could move it, it would stop being one.

Two more carry the arithmetic. The system peak is the largest **sum at one
period**, never the sum of the users' own peaks -- those are different numbers
and the second is not a physical quantity on this feeder. And a bus the subset
empties keeps the benchmark's demand rather than being left at zero.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_industry_node_loads as bnl
import industry_load as il
import select_industry_user_subset as sel

N = il.N_SLOTS
MODULE = Path(sel.__file__)

#: The fixture's day. One period carries the whole native peak, so a user that
#: sits under it there and above it elsewhere moves the peak in a way the test
#: can predict by hand.
PEAK_PERIOD = 76


def _shape():
    return np.array([1.0 if t == PEAK_PERIOD else 0.2 for t in range(N)])


def _curve(own_peak, peak_at, at_native_peak=0.1):
    """A user's 96 periods: low everywhere, at the native peak, and at its own."""
    values = np.full(N, 0.02)
    values[peak_at] = own_peak
    values[PEAK_PERIOD] = at_native_peak
    return values


def _md(meter, user, industry, values_mw, date="2021-01-01", ratio=1.0,
        running=1000.0):
    """One meter-day whose readings are `values_mw` kW-scaled, so ratio 1.0."""
    return il.MeterDay(
        account=user, meter=meter, user_name="n", industry=industry,
        location="南京", excel_row=2, date=date,
        values=tuple(float(v) * 1000.0 for v in values_mw), ratio=ratio,
        running_capacity=running, phase=il.PHASE_ACTIVE_POWER, flags=())


#: Five users, one per bus, each peaking at a period the others do not use. All
#: five together leave the system peak exactly where the benchmark put it.
BUSES = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1}
TARGET = sum(BUSES.values())            # 0.5 MW
TOLERANCE = 0.05


def _fixture():
    curves, user_bus, industries = {}, {}, {}
    for index, bus in enumerate(sorted(BUSES), start=1):
        user = f"U{index}"
        curves[user] = _curve(0.12, 10 * index)
        user_bus[user] = bus
        industries[user] = f"industry{index}"
    return curves, user_bus, industries


def _run(curves, user_bus, industries, tolerance=TOLERANCE):
    return sel.search(curves, user_bus, industries, BUSES, _shape(), TARGET,
                      tolerance)


# ---------------------------------------------------------------------------
# The selection is the largest feasible one, and the peak is a coincident sum
# ---------------------------------------------------------------------------


def test_the_selected_subset_is_the_largest_feasible_one():
    """Every one of the five fits, so all five are taken."""
    curves, user_bus, industries = _fixture()
    result = _run(curves, user_bus, industries)
    assert result["feasible"]
    assert result["selected"] == sorted(user_bus)
    assert result["total_mw"].max() == pytest.approx(TARGET)


def test_the_peak_is_coincident_not_the_sum_of_the_users_own_peaks():
    """The two numbers differ, and the one reported is the coincident one.

    Each user peaks at 0.12 MW in its own period and the five periods are
    distinct, so the sum of their peaks is 0.6 MW -- a figure no bus ever sees
    and no meter ever measures. The system peak is 0.5 MW, the largest sum of
    simultaneous contributions.
    """
    curves, user_bus, industries = _fixture()
    result = _run(curves, user_bus, industries)
    coincident = float(result["total_mw"].max())
    sum_of_peaks = sum(float(curves[u].max()) for u in result["selected"])
    assert coincident == pytest.approx(0.5)
    assert sum_of_peaks == pytest.approx(0.6)
    assert coincident != pytest.approx(sum_of_peaks)
    # And the reported peak is the coincident one, recomputed from the day.
    assert coincident == pytest.approx(result["total_mw"].max())


def test_the_search_is_exhaustive_over_every_admissible_subset():
    """2^5 subsets of the candidate pool, all of them evaluated."""
    curves, user_bus, industries = _fixture()
    result = _run(curves, user_bus, industries)
    assert result["evaluated"] == 2 ** len(result["pool"])
    assert len(result["pool"]) == 5


# ---------------------------------------------------------------------------
# The candidate bound, and the premise it rests on
# ---------------------------------------------------------------------------


def test_a_user_that_cannot_fit_any_feasible_subset_is_pruned_not_considered():
    """A user above the ceiling is proved out, not merely unlikely.

    Its own peak is 0.6 MW against a ceiling of 0.525, so any subset containing
    it has a system peak of at least 0.6 -- outside the tolerance however the
    other four were chosen. Nothing about the search result depends on which of
    its buses it would have sat on.
    """
    curves, user_bus, industries = _fixture()
    curves["Ubig"] = _curve(0.6, 5)
    user_bus["Ubig"] = 1
    industries["Ubig"] = "huge"
    result = _run(curves, user_bus, industries)
    assert "Ubig" not in result["pool"]
    assert result["ceiling"] == pytest.approx(TARGET * (1 + TOLERANCE))
    assert "Ubig" not in result["selected"]


def test_every_eligible_user_left_out_would_break_the_tolerance():
    """The reason written for an eligible user is a claim, and it is checked.

    The selected subset is the largest feasible one, so adding any eligible
    user to it must leave the tolerance. `verify_maximal` recomputes that one
    user at a time rather than trusting the search.
    """
    curves, user_bus, industries = _fixture()
    # A sixth user on bus 1, small enough to be eligible but pushing bus 1 to
    # 0.9 MW at the peak if it is taken alongside the first. So the maximum is
    # five users and one eligible user must be left out.
    curves["U6"] = _curve(0.12, 55, at_native_peak=0.5)
    user_bus["U6"] = 1
    industries["U6"] = "industry6"
    result = _run(curves, user_bus, industries)
    sel.verify_maximal(result, curves, user_bus, BUSES, _shape(), TARGET,
                       TOLERANCE)
    assert len(result["selected"]) == 5
    left_out = [u for u in result["pool"] if u not in result["selected"]]
    assert left_out
    for user in left_out:
        rows = sel.classify_users(curves, user_bus, result["selected"],
                                  result["pool"], result["ceiling"])
        row = next(r for r in rows if r["user_id"] == user)
        assert row["reason"] == sel.REASON_NOT_IN_MAXIMUM


def test_a_negative_reading_refuses_rather_than_pruning_around_it():
    """The bound's premise is non-negativity. Without it, nothing may be cut.

    A negative contribution can lower the system peak, so a user above the
    ceiling might fit after all. Rather than prune on an argument that no
    longer holds -- and return a subset that is not the maximum while calling
    itself one -- the selector stops.
    """
    curves, user_bus, industries = _fixture()
    curves["U1"] = curves["U1"].copy()
    curves["U1"][3] = -0.01
    with pytest.raises(ValueError) as excinfo:
        _run(curves, user_bus, industries)
    assert "negative" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Determinism, and what happens when nothing fits
# ---------------------------------------------------------------------------


def test_repeated_runs_return_the_same_subset_and_the_same_peak():
    curves, user_bus, industries = _fixture()
    first = _run(curves, user_bus, industries)
    second = _run(curves, user_bus, industries)
    assert first["selected"] == second["selected"]
    assert first["total_mw"].max() == second["total_mw"].max()
    assert first["evaluated"] == second["evaluated"]


def _tie_fixture():
    """Three interchangeable users, only one of which can be taken.

    Each displaces 0.45 MW of a 0.5 MW bus while sitting at 0.02 elsewhere, so
    one of them at the peak leaves 1.45 MW (3.3% under the target, inside the
    tolerance), two leave 1.40 (6.7% under, outside it), and the all-preserve
    day leaves exactly the 1.5 MW target. The largest feasible subset therefore
    holds **one** user, and all three of them are the same choice.
    """
    rated = {1: 0.5, 2: 0.5, 3: 0.5}
    curves, user_bus, industries = {}, {}, {}
    for index, bus in enumerate(sorted(rated), start=1):
        user = f"U{chr(96 + index)}"
        curves[user] = _curve(0.45, 40 + index, at_native_peak=0.45)
        user_bus[user] = bus
        industries[user] = f"industry{index}"
    return rated, curves, user_bus, industries


def test_ties_at_the_best_key_are_all_reported_and_resolved_by_user_id():
    """Three subsets attain the same key; the last tie-break is stated."""
    rated, curves, user_bus, industries = _tie_fixture()
    result = sel.search(curves, user_bus, industries, rated, _shape(), 1.5,
                        TOLERANCE)
    assert result["feasible"]
    assert result["n_tied"] == 3
    assert result["n_tied"] == len(result["tied_at_the_best_key"])
    assert result["tied_at_the_best_key"] == sorted(
        result["tied_at_the_best_key"])
    assert tuple(sorted(result["selected"])) == min(
        tuple(sorted(s)) for s in result["tied_at_the_best_key"])
    assert result["selected"] == ["Ua"]


def test_the_largest_subset_wins_even_when_a_smaller_one_is_closer():
    """The rule the task states, on the case that makes it bite.

    The all-preserve day sits exactly on the target and the one-user subsets sit
    3.3% under it. Error-first would take no users at all; the tolerance says
    both are close, so the subset that accounts for a real user wins.
    """
    rated, curves, user_bus, industries = _tie_fixture()
    result = sel.search(curves, user_bus, industries, rated, _shape(), 1.5,
                        TOLERANCE)
    assert len(result["selected"]) == 1
    assert abs(result["total_mw"].max() - 1.5) / 1.5 > 0.0
    assert abs(result["base"].max() - 1.5) / 1.5 == pytest.approx(0.0)


def test_an_unreachable_tolerance_reports_the_closest_and_changes_nothing():
    """No subset fits: the answer is a report, not a wider tolerance.

    The target sits 6.7% above every day this fixture can produce, so nothing
    reaches it. The selector says so and names what came closest instead of
    relaxing the bound or scaling a curve to reach it.
    """
    rated, curves, user_bus, industries = _tie_fixture()
    result = sel.search(curves, user_bus, industries, rated, _shape(), 1.6,
                        TOLERANCE)
    assert result["feasible"] is False
    assert result["selected"] == []
    assert result["closest"] is not None
    assert result["closest"][0] == pytest.approx((1.6 - 1.5) / 1.6)
    # The curves the caller handed in are untouched by the failed search.
    assert float(curves["Ua"].max()) == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Bus policy
# ---------------------------------------------------------------------------


def test_a_bus_with_no_selected_user_keeps_the_native_load():
    curves, user_bus, industries = _fixture()
    result = _run(curves, user_bus, industries)
    # Only buses 1..5 exist in this fixture, so every other bus in the network
    # is a bus with no selected user and must be `preserve`.
    policies = sel.policies_for(result["selected"], user_bus)
    assert set(policies) == set(bnl.ALL_BUSES)
    for bus in bnl.ALL_BUSES:
        if bus not in user_bus.values():
            assert policies[bus] == "preserve"
    native, _ = bnl.compose(0.0605, None, "preserve")
    assert native == pytest.approx(0.0605)


def test_no_replace_bus_is_left_with_nothing_behind_it():
    """`replace` means a selected user is there. An empty bus never says it.

    The composition rule refuses a `replace` bus with no industry term rather
    than zero-filling it, so a policy that claimed `replace` over an emptied bus
    would produce a whole column of blanks.
    """
    curves, user_bus, industries = _fixture()
    result = _run(curves, user_bus, industries)
    policies = sel.policies_for(result["selected"], user_bus)
    carrying = {user_bus[u] for u in result["selected"]}
    for bus, policy in policies.items():
        if policy == "replace":
            assert bus in carrying
        else:
            assert bus not in carrying
    assert bnl.compose(0.1, None, "replace") == (None, "excluded_no_data")


def test_the_slack_bus_is_never_a_mapping_target():
    policies = sel.policies_for([], {})
    assert policies[bnl.SLACK_BUS] == "preserve"
    with pytest.raises(AssertionError):
        bnl.assert_classification_complete([
            {"bus": 0, "has_user": True, "status": "has_user"}])


# ---------------------------------------------------------------------------
# The readings, and only the readings
# ---------------------------------------------------------------------------


def test_the_curve_is_the_readings_times_their_multiplier_and_nothing_else():
    records = il.flag_records([
        _md("M1", "U1", "化工业", _curve(0.12, 10), ratio=4.0),
    ])
    per_user, _meters, unreadable = bnl.per_user_curves("2021-01-01", records)
    assert unreadable == []
    key = records[0].user_key
    expected = np.asarray(records[0].values, dtype=float) * 4.0 * il.MW_PER_KW
    assert np.allclose(per_user[key], expected)


def test_a_users_meters_are_summed_once_and_never_averaged():
    """Two meters at 0.12 MW is 0.24 MW on the user, not 0.12 and not 0.24 twice."""
    records = il.flag_records([
        _md("M1", "U1", "化工业", _curve(0.12, 10)),
        _md("M2", "U1", "化工业", _curve(0.12, 10)),
    ])
    per_user, meters, _unreadable = bnl.per_user_curves("2021-01-01", records)
    key = records[0].user_key
    assert meters[key] == [records[0].meter_key, records[1].meter_key]
    single = np.asarray(records[0].values, dtype=float) * il.MW_PER_KW
    assert np.allclose(per_user[key], 2.0 * single)
    assert not np.allclose(per_user[key], single)


def test_the_day_reconstructs_from_the_readings_and_the_native_term():
    """The reported day is base + readings - native, with nothing else in it.

    Rebuilding it by hand from those three terms is the strongest statement
    available that no scale factor entered anywhere.
    """
    curves, user_bus, industries = _fixture()
    result = _run(curves, user_bus, industries)
    shape = _shape()
    rebuilt = sel.bus_native_load(BUSES, shape)
    for bus in {user_bus[u] for u in result["selected"]}:
        rebuilt = rebuilt - BUSES[bus] * shape
    for user in result["selected"]:
        rebuilt = rebuilt + curves[user]
    assert np.allclose(rebuilt, result["total_mw"])


def test_the_selection_reads_no_price_no_lmp_and_no_learning_result():
    """Checked against the module's own imports, not against a promise.

    A subset chosen on a market outcome would be a subset that changes when the
    market does, which is not a statement about measured load.
    """
    forbidden = {
        "dianyicha_price", "wholesale_price", "market", "market_opportunity",
        "counterfactual_eval", "participant_payoff", "rl_env", "rl_td3",
        "rl_spec", "rl_bidding", "strategies",
    }
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & forbidden), sorted(imported & forbidden)


def test_the_subset_table_records_a_reason_for_every_user(tmp_path):
    curves, user_bus, industries = _fixture()
    result = _run(curves, user_bus, industries)
    rows = sel.classify_users(curves, user_bus, result["selected"],
                              result["pool"], result["ceiling"])
    assert {r["user_id"] for r in rows} == set(user_bus)
    for row in rows:
        assert row["reason"]
        if row["included"]:
            assert row["reason"] == sel.REASON_INCLUDED
        else:
            assert row["reason"] != sel.REASON_INCLUDED
    path = sel.write_subset_csv(tmp_path / "subset.csv", rows)
    text = path.read_text(encoding="utf-8")
    assert "user_id" in text.splitlines()[0]
    assert len(text.splitlines()) == len(rows) + 1


def test_an_unmapped_user_is_excluded_with_its_own_reason():
    """A user the mapping does not place is not a load, however small."""
    curves, user_bus, industries = _fixture()
    curves["Ufree"] = _curve(0.01, 5)
    industries["Ufree"] = "industry1"
    result = _run(curves, user_bus, industries)
    assert "Ufree" not in result["pool"]
    rows = sel.classify_users(curves, user_bus, result["selected"],
                              result["pool"], result["ceiling"])
    row = next(r for r in rows if r["user_id"] == "Ufree")
    assert row["reason"] == sel.REASON_UNMAPPED


# ---------------------------------------------------------------------------
# The derived mapping, and the reactive account
# ---------------------------------------------------------------------------


def test_the_derived_mapping_keeps_only_the_selected_users(tmp_path):
    map_rows = [
        {"user_id": "U1", "industry": "A", "n_meters": "1",
         "destination_bus": "1", "candidate_status": "mapped"},
        {"user_id": "U2", "industry": "B", "n_meters": "1",
         "destination_bus": "2", "candidate_status": "mapped"},
    ]
    path = sel.write_subset_mapping(tmp_path / "subset_map.csv", map_rows,
                                    ["U2"])
    text = path.read_text(encoding="utf-8").splitlines()
    assert len(text) == 2
    assert "U2" in text[1] and "U1" not in text[1]


def test_the_reactive_account_is_an_assumption_and_is_not_rescaled():
    """No meter here reports reactive power, so nothing observes it."""
    account = sel.reactive_account(0.8503, 3.6534, 100.0)
    assert account["rescaled_to_the_benchmark"] is False
    assert "assumption" in account["power_factor_status"].lower()
    assert "NOT a measurement" in account["power_factor_status"]
    assert account["q_peak_mvar"] == pytest.approx(3.6534 * np.tan(
        np.arccos(0.8503)))
    assert account["benchmark_q_mvar"] == 2.3


def test_the_shipped_scenario_declares_the_tolerance_the_rule_uses():
    """The declared tolerance and the one the code applies cannot drift apart."""
    import config_loader as cfg

    declared = cfg.get_scenario_cfg(sel.DEFAULT_SCENARIO)
    block = (declared.get("load_model") or {}).get("subset_selection") or {}
    assert block, "the scenario declares no subset_selection block"
    assert float(block["target_peak_mw"]) == pytest.approx(
        sel.TARGET_PEAK_MW)
    assert float(block["tolerance"]) == pytest.approx(
        sel.TARGET_PEAK_TOLERANCE)
    assert "project choice" in str(block["status"])
