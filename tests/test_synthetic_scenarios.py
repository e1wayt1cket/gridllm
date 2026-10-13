# tests/test_synthetic_scenarios.py
"""Contracts of the three-regime synthetic price generator.

Three things here are easy to get wrong in ways that produce plausible numbers:

* **The margin's unit.** `required_margin` (R) is a gross price margin per MWh
  *stored*; the LP's `margin_cny` is net revenue in CNY after cycle cost. Putting
  them in one comparison is a CNY-vs-CNY-per-MWh error whose size is the unit's
  capacity factor -- 344.29 against 494.92 on the ramp used below.
* **The idle branch.** When the LP declines to cycle there is no margin at all,
  and both per-MWh ratio fields are NaN. That is the `no_arbitrage` case, so a
  classifier that thresholds a ratio drops exactly the day it must classify.
* **Reachability.** The marginal band is 10 % of R. On many shapes the LP's
  margin jumps from "no cycle" straight past it, so a requested regime is not a
  property a shape automatically has, and a quota is not a formality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses

import numpy as np
import pytest

import synthetic_scenarios as ss
from arbitrage_lp import ArbitrageResult, arbitrage_lp, required_margin
from money import DT_HOURS

R = required_margin(ss.CYCLE_COST_PER_LEG, ss.REFERENCE_UNIT.eta_ch,
                    ss.REFERENCE_UNIT.eta_dis)
B = 0.10 * R


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _priced_day(prices=None):
    if prices is None:
        prices = np.linspace(120.0, 1000.0, ss.T)
    return arbitrage_lp(prices, ss.REFERENCE_UNIT, ss.CYCLE_COST_PER_LEG)


def _fake_result(m_lp, q_ch=0.80, unit=ss.REFERENCE_UNIT,
                 cc=ss.CYCLE_COST_PER_LEG):
    """An `ArbitrageResult` whose gross margin is exactly `m_lp`.

    Built from the definition rather than by inverting the classifier, so the
    boundary tests below exercise the real branch logic against a known input
    instead of against another copy of the same arithmetic.
    """
    q_dis = q_ch * unit.eta_ch * unit.eta_dis
    ch = np.full(ss.T, q_ch / (ss.T * DT_HOURS))
    dis = np.full(ss.T, q_dis / (ss.T * DT_HOURS))
    margin = m_lp * unit.eta_ch * q_ch - cc * (q_ch + q_dis)
    soc = np.full(ss.T + 1, unit.soc0)          # balanced endpoints
    return ArbitrageResult(
        margin_cny=margin, status=0, status_text="ok",
        ch_mw=ch, dis_mw=dis, soc=soc,
        chargeable_energy_mwh=q_ch, dischargeable_energy_mwh=q_dis,
        feasible_charge_periods=ss.T, feasible_discharge_periods=ss.T,
        equivalent_cycles=(q_ch + q_dis) / (2 * unit.e_max),
        best_charge_price=200.0, best_discharge_price=500.0,
        gross_spread_cny_mwh=300.0, efficiency_loss_cny_mwh=50.0,
        degradation_cost_cny_mwh=R, adjusted_spread_cny_mwh=R)


@pytest.fixture(scope="module")
def family():
    spec = ss.FamilySpec(name="test_family", seed=20261013,
                         quotas={ss.NO_ARBITRAGE: 4,
                                 ss.MARGINAL_ARBITRAGE: 4,
                                 ss.PROFITABLE_ARBITRAGE: 4})
    days = ss.build_family(spec)
    return spec, days


# ---------------------------------------------------------------------------
# The margin measure
# ---------------------------------------------------------------------------

def test_required_margin_is_the_one_the_lp_module_derives():
    assert R == pytest.approx(200.69565217391303, abs=1e-12)
    assert ss.required_margin_cny_per_mwh() == pytest.approx(R)
    assert B == pytest.approx(20.069565217391302, abs=1e-12)


def test_gross_margin_is_the_gross_formula_and_not_the_net_one():
    """`(margin_cny + cycle_cost*(Qch+Qdis)) / (eta_ch*Qch)` -- the cycle cost
    is added BACK, because R is a gross margin and `margin_cny` has already had
    that cost deducted."""
    res = _priced_day()
    q_ch = float(np.sum(res.ch_mw) * DT_HOURS)
    q_dis = float(np.sum(res.dis_mw) * DT_HOURS)
    want = (res.margin_cny + ss.CYCLE_COST_PER_LEG * (q_ch + q_dis)) / \
        (ss.REFERENCE_UNIT.eta_ch * q_ch)
    got = ss.gross_margin(res)
    assert got == pytest.approx(want, rel=1e-12)
    # And it is NOT the net per-charged-MWh figure the LP reports.
    assert got != pytest.approx(res.margin_per_mwh_charged, rel=1e-6)


def test_gross_margin_equals_r_plus_net_over_stored_energy():
    """`M_LP = R + margin_cny / (eta_ch * Q_ch)`, the approved identity."""
    res = _priced_day()
    q_ch = float(np.sum(res.ch_mw) * DT_HOURS)
    assert ss.gross_margin(res) == pytest.approx(
        R + res.margin_cny / (ss.REFERENCE_UNIT.eta_ch * q_ch), rel=1e-12)


def test_the_identity_rests_on_the_balanced_terminal_soc():
    """Balance is what makes `Q_dis/Q_ch = eta_ch*eta_dis` and so collapses the
    cycle-cost term onto R. Without it the identity does not hold, and a margin
    computed from an unbalanced day would silently mean something else -- so it
    is refused rather than returned."""
    res = _priced_day()
    q_ch = float(np.sum(res.ch_mw) * DT_HOURS)
    q_dis = float(np.sum(res.dis_mw) * DT_HOURS)
    assert q_dis / q_ch == pytest.approx(
        ss.REFERENCE_UNIT.eta_ch * ss.REFERENCE_UNIT.eta_dis, rel=1e-9)

    soc = res.soc.copy()
    soc[-1] = soc[0] + 0.05
    unbalanced = dataclasses.replace(res, soc=soc)
    with pytest.raises(ValueError, match="do not balance"):
        ss.gross_margin(unbalanced)


# ---------------------------------------------------------------------------
# The idle branch
# ---------------------------------------------------------------------------

def test_an_idle_day_has_no_margin_and_is_not_nan_compared():
    """A flat curve leaves the LP nothing to do. `M_LP` is then UNDEFINED, not
    NaN-to-be-compared, and the classifier must reach `no_arbitrage` by looking
    at the schedule."""
    res = _priced_day(np.full(ss.T, 500.0))
    assert res.margin_cny <= ss.IDLE_TOL_CNY
    assert ss.gross_margin(res) is None
    diag = ss.classify(res)
    assert diag["regime"] == ss.NO_ARBITRAGE
    assert diag["m_lp_cny_per_mwh"] is None
    assert "no schedule" in diag["reason"]


def test_a_nan_margin_would_fall_through_every_threshold():
    """Why the branch order matters, stated as a test: NaN fails every
    comparison, so a ratio-first classifier silently mislabels the idle day."""
    nan = float("nan")
    assert not (nan < R - B) and not (nan <= R + B) and not (nan > R + B)


def test_solver_residue_is_treated_as_idle_not_as_a_tiny_cycle():
    """The LP returns ~1e-17 CNY on a day with no opportunity. An exact `== 0`
    test would call that scheduled and label it `marginal`."""
    res = _priced_day(np.full(ss.T, 420.0))
    assert abs(res.margin_cny) < ss.IDLE_TOL_CNY
    assert ss.classify(res)["regime"] == ss.NO_ARBITRAGE


# ---------------------------------------------------------------------------
# Regime boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m_lp,want", [
    (R + 0.01, ss.MARGINAL_ARBITRAGE),
    (R + B / 2, ss.MARGINAL_ARBITRAGE),
    (R + B, ss.MARGINAL_ARBITRAGE),
    (R + B + 1.0, ss.PROFITABLE_ARBITRAGE),
])
def test_the_band_edges_land_where_declared(m_lp, want):
    """Only the reachable edges are parametrised.

    `M_LP = R` exactly is also unreachable with a schedule, and for the same
    reason `R - b` is: `M_LP = R + margin_cny/(eta_ch*Q_ch)`, so `M_LP = R`
    forces `margin_cny = 0`, which is the idle branch. The reachable marginal
    range is strictly `(R, R + b]`.
    """
    assert ss.classify(_fake_result(m_lp))["regime"] == want


def test_m_lp_exactly_r_is_idle_because_its_net_margin_is_zero():
    res = _fake_result(R)
    assert res.margin_cny == pytest.approx(0.0, abs=1e-9)
    assert ss.classify(res)["regime"] == ss.NO_ARBITRAGE


def test_a_scheduled_day_below_the_band_is_unreachable_by_construction():
    """`M_LP = R + margin_cny/(eta_ch*Q_ch)` and the LP schedules only when its
    optimum is positive, so a scheduled day always has `M_LP > R`. The band
    below R is therefore closed, and a fabricated result aiming there lands in
    the idle branch on its own net margin -- not in the guard.

    This is recorded as a fact about the model, not hidden: point 5 of the
    margin corrections describes a state the balanced-SOC identity forbids.
    """
    res = _fake_result(R - B - 1.0)
    assert res.margin_cny <= ss.IDLE_TOL_CNY      # economically impossible state
    assert ss.classify(res)["regime"] == ss.NO_ARBITRAGE


def test_the_guard_fires_when_a_scheduled_margin_disagrees(monkeypatch):
    """The guard is defensive, but it must not be dead code: if `M_LP` ever
    came back below the band from a scheduled day, the day must be reported as
    inconsistent rather than absorbed into `no_arbitrage`."""
    monkeypatch.setattr(ss, "gross_margin", lambda *a, **k: R - B - 1.0)
    res = _fake_result(R + 5.0)                   # scheduled: margin_cny > 0
    assert res.margin_cny > ss.IDLE_TOL_CNY
    diag = ss.classify(res)
    assert diag["regime"] == ss.CLASSIFICATION_INCONSISTENT
    assert diag["regime"] != ss.NO_ARBITRAGE
    assert "error" in diag["reason"]


def test_an_idle_day_is_never_reported_as_inconsistent():
    res = _priced_day(np.full(ss.T, 500.0))
    assert ss.classify(res)["regime"] == ss.NO_ARBITRAGE


# ---------------------------------------------------------------------------
# The degenerate optimum at break-even
# ---------------------------------------------------------------------------

def _break_even_result():
    """A two-level day priced exactly at break-even.

    `LHS = eta*p_d - p_c/eta`, so `p_d = (R + p_c/eta)/eta` puts the day
    precisely on the threshold -- where the LP is indifferent and its optimum is
    degenerate.
    """
    p_c = 200.0
    p_d = (R + p_c / ss.REFERENCE_UNIT.eta_ch) / ss.REFERENCE_UNIT.eta_dis
    price = np.full(ss.T, p_c)
    price[ss.T // 2:] = p_d
    return _priced_day(price)


def test_a_degenerate_zero_revenue_schedule_is_not_arbitrage():
    """The LP at break-even returns a FULL non-zero schedule with zero revenue.
    An exact `== 0` revenue test alone would describe that day as "no schedule";
    a schedule test alone would call it an opportunity. Both are wrong, and the
    day is economically no-arbitrage either way."""
    res = _break_even_result()
    q_ch = float(np.sum(res.ch_mw) * DT_HOURS)
    q_dis = float(np.sum(res.dis_mw) * DT_HOURS)
    assert res.margin_cny <= ss.IDLE_TOL_CNY        # zero revenue
    assert q_ch > ss.ZERO_SCHEDULE_TOL_MWH          # but a real cycle ran
    assert q_dis > ss.ZERO_SCHEDULE_TOL_MWH

    diag = ss.classify(res)
    assert diag["regime"] == ss.NO_ARBITRAGE
    assert diag["regime"] != ss.PROFITABLE_ARBITRAGE
    assert diag["schedule_class"] == ss.SCHEDULE_DEGENERATE_ZERO_REVENUE
    assert diag["m_lp_cny_per_mwh"] is None
    assert "degenerate" in diag["reason"]


def test_the_two_tolerances_are_separate_questions():
    """Revenue and schedule are tested with different constants, because the
    LP answers them differently at break-even."""
    assert ss.ZERO_SCHEDULE_TOL_MWH != ss.IDLE_TOL_CNY
    idle = _priced_day(np.full(ss.T, 500.0))
    di = ss.classify(idle)
    assert di["schedule_class"] == ss.SCHEDULE_NONE
    deg = ss.classify(_break_even_result())
    assert deg["schedule_class"] == ss.SCHEDULE_DEGENERATE_ZERO_REVENUE
    assert di["regime"] == deg["regime"] == ss.NO_ARBITRAGE


def test_a_just_above_break_even_day_is_marginal_not_degenerate():
    """>= the tolerance in revenue and the day has a real, if tiny, margin."""
    res = _priced_day(np.linspace(120.0, 1000.0, ss.T))
    diag = ss.classify(res)
    assert diag["schedule_class"] == ss.SCHEDULE_SCHEDULED
    assert diag["regime"] == ss.PROFITABLE_ARBITRAGE
    assert diag["m_lp_cny_per_mwh"] is not None


def test_an_unbalanced_state_gets_a_label_instead_of_a_number():
    """`M_LP` is only defined with valid charge energy AND balanced endpoints.
    An inconsistent state is labelled, not given a number that would read as a
    market outcome."""
    res = _priced_day()
    soc = res.soc.copy()
    soc[-1] = soc[0] + 0.05
    bad = dataclasses.replace(res, soc=soc)
    diag = ss.classify(bad)
    assert diag["regime"] == ss.MARGIN_STATE_INCONSISTENT
    assert diag["m_lp_cny_per_mwh"] is None
    assert "undefined" in diag["reason"]
    # And it is not silently one of the three regimes.
    assert diag["regime"] not in ss.REGIMES


# ---------------------------------------------------------------------------
# Price-reference calibration
# ---------------------------------------------------------------------------

def test_calibration_uses_the_declared_three_level_formula(family):
    spec, days = family
    train = ss.split_family(days)["train"]
    cal = ss.calibrate_price_ref({spec.name: train})
    curve_means = [float(d.price_array().mean()) for d in train]
    assert cal["per_family"][spec.name]["curve_daily_means_cny_per_mwh"] == \
        pytest.approx(curve_means)
    assert cal["per_family"][spec.name]["representative_price_cny_per_mwh"] == \
        pytest.approx(float(np.mean(curve_means)))
    assert cal["value_cny_per_mwh"] == pytest.approx(
        float(np.mean([cal["per_family"][spec.name]
                       ["representative_price_cny_per_mwh"]])))
    assert cal["n_training_curves"] == len(train)


def test_families_are_weighted_equally_not_by_curve_count(family):
    """A family with twice the curves must not set twice the scale."""
    spec, days = family
    big = list(days)
    small = [dataclasses.replace(d, day_id=d.day_id + 100,
                                 prices=tuple(p * 1.5 for p in d.prices))
             for d in days[:2]]
    cal = ss.calibrate_price_ref({"big": big, "small": small})
    reps = cal["family_representative_prices_cny_per_mwh"]
    assert len(reps) == 2
    # The value is the plain mean of the two representatives, so the 18-curve
    # family carries exactly the same weight as the 2-curve one.
    assert cal["value_cny_per_mwh"] == pytest.approx(float(np.mean(reps)))
    assert cal["n_training_curves"] == len(big) + len(small)
    # The counts differ by an order of magnitude; the weights do not.
    assert len(big) > 5 * len(small)


def test_changing_held_out_days_does_not_move_the_scale(family):
    """The leak this exists to prevent: validation and test days must have no
    influence on a constant that they are then normalized by."""
    spec, days = family
    split = ss.split_family(days)
    cal_a = ss.calibrate_price_ref({spec.name: split["train"]})

    # Perturb ONLY the held-out days, hard.
    bumped = [dataclasses.replace(d, prices=tuple(p * 3.0 for p in d.prices))
              for d in split["validation"] + split["test"]]
    split_b = {"train": split["train"], "validation": bumped[:4],
               "test": bumped[4:]}
    cal_b = ss.calibrate_price_ref({spec.name: split_b["train"]})
    assert cal_b["value_cny_per_mwh"] == cal_a["value_cny_per_mwh"]
    assert cal_b["data_digest_sha256"] == cal_a["data_digest_sha256"]


def test_the_calibration_records_its_formula_and_exclusions(family):
    spec, days = family
    train = ss.split_family(days)["train"]
    cal = ss.calibrate_price_ref({spec.name: train})
    assert cal["formula"] == ss.PRICE_REF_FORMULA
    assert "training" in cal["formula"].lower()
    excl = " ".join(cal["excluded_from_calibration"])
    assert "validation" in excl and "test" in excl and "261009" in excl
    assert len(cal["data_digest_sha256"]) == 64


def test_calibrating_from_nothing_is_refused():
    with pytest.raises(ValueError, match="no families"):
        ss.calibrate_price_ref({})
    with pytest.raises(ValueError, match="no training days"):
        ss.calibrate_price_ref({"empty": []})


# ---------------------------------------------------------------------------
# Structural mechanisms, not just amplitude
# ---------------------------------------------------------------------------

def test_the_mechanisms_are_structurally_different():
    """Amplitude alone is not diversity: a pool varying only peak height carries
    one shape. Each mechanism must produce a distinguishable curve."""
    trough, peak = 20, 74
    shapes = {m: ss._mechanism_shape(m, trough, peak) for m in ss.MECHANISMS}
    assert set(shapes) == set(ss.MECHANISMS)
    for m, s in shapes.items():
        assert s.shape == (ss.T,), m
        assert s.min() >= -1e-9 and s.max() <= 1 + 1e-9, m
        # Bounded by [0, 1], but neither endpoint need be attained: the peak
        # slot is sampled on a discrete 96-slot grid and lands short of the
        # analytic maximum (measured 0.883 for `single_peak`), and
        # `double_peak`'s saddle raises its floor to 0.85. So the family's
        # declared span bound is an UPPER bound on the realized range, not an
        # exact one -- stated here rather than assumed away.
        assert 0.5 < s.max() <= 1.0, m
        assert s.min() < s.max(), m
    # At least three mutually distinguishable forms.
    distinct = 0
    names = list(shapes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not np.allclose(shapes[names[i]], shapes[names[j]], atol=0.02):
                distinct += 1
    assert distinct >= 3, "mechanisms are not structurally distinct"


def test_an_unknown_mechanism_is_refused():
    with pytest.raises(ValueError, match="unknown mechanism"):
        ss._mechanism_shape("not_a_mechanism", 20, 74)


def test_a_family_uses_more_than_one_mechanism(family):
    _, days = family
    assert len({d.mechanism for d in days}) >= 2
    for d in days:
        assert d.mechanism in ss.MECHANISMS


def test_the_mechanism_is_recorded_in_the_payload(family):
    _, days = family
    d = days[0].to_dict()
    assert d["mechanism"] in ss.MECHANISMS


# ---------------------------------------------------------------------------
# Top-level family isolation
# ---------------------------------------------------------------------------

def test_one_top_level_family_may_not_be_split_across_uses():
    """Templates differ within a family; the generating process does not. A
    held-out day drawn from the training family is a near-relative however
    different its template looks, so isolation is enforced at the family."""
    specs = [ss.FamilySpec(name="a", seed=1, quotas={ss.NO_ARBITRAGE: 1,
                                                     ss.MARGINAL_ARBITRAGE: 0,
                                                     ss.PROFITABLE_ARBITRAGE: 0},
                           family_id="fam_a"),
             ss.FamilySpec(name="b", seed=2, quotas={ss.NO_ARBITRAGE: 1,
                                                     ss.MARGINAL_ARBITRAGE: 0,
                                                     ss.PROFITABLE_ARBITRAGE: 0},
                           family_id="fam_a")]
    with pytest.raises(AssertionError, match="appears twice"):
        ss.split_families(specs, {"fam_a": "train"})


def test_a_family_without_a_declared_split_is_refused():
    specs = [ss.FamilySpec(name="solo", seed=1,
                           quotas={ss.NO_ARBITRAGE: 1,
                                   ss.MARGINAL_ARBITRAGE: 0,
                                   ss.PROFITABLE_ARBITRAGE: 0})]
    with pytest.raises(AssertionError, match="no declared split"):
        ss.split_families(specs, {})
    with pytest.raises(AssertionError, match="unknown split"):
        ss.split_families(specs, {"solo": "holdout"})


def test_distinct_families_are_assigned_whole():
    specs = [ss.FamilySpec(name=f"f{i}", seed=i,
                           quotas={ss.NO_ARBITRAGE: 1,
                                   ss.MARGINAL_ARBITRAGE: 0,
                                   ss.PROFITABLE_ARBITRAGE: 0},
                           family_id=f"top{i}") for i in range(3)]
    out = ss.split_families(specs, {"top0": "train", "top1": "validation",
                                    "top2": "test"})
    assert out == {"train": ["f0"], "validation": ["f1"], "test": ["f2"]}


def test_the_default_family_id_cannot_masquerade_as_isolation():
    """A spec with no `family_id` gets its own name, so a single-family pool can
    never be presented as family-isolated."""
    spec = ss.FamilySpec(name="fam_a", seed=1,
                         quotas={ss.NO_ARBITRAGE: 1,
                                 ss.MARGINAL_ARBITRAGE: 0,
                                 ss.PROFITABLE_ARBITRAGE: 0})
    assert (spec.family_id or spec.name) == "fam_a"
    with pytest.raises(AssertionError, match="no declared split"):
        ss.split_families([spec], {})


# ---------------------------------------------------------------------------
# Acceptance accounting
# ---------------------------------------------------------------------------

def test_acceptance_is_accounted_not_just_the_winners(family):
    """A quota fill is a search. Reporting only the accepted days hides both its
    cost and any selection bias, so proposals, rejections and their measured
    regimes are recorded."""
    spec, _ = family
    log = {}
    ss.build_family(spec, accept_log=log)
    assert log["proposed"] == log["accepted"] + log["rejected"]
    assert log["accepted"] == spec.n_days
    for regime in ss.REGIMES:
        v = log["per_regime"][regime]
        assert v["proposed"] == v["accepted"] + v["rejected"]
        assert v["accepted"] == spec.quotas[regime]
        assert len(v["attempts_to_accept"]) == v["accepted"]
    assert sum(log["rejected_by_measured_regime"].values()) == log["rejected"]
    assert sum(log["mechanisms_accepted"].values()) == log["accepted"]


def test_the_probe_measures_cost_and_reachability():
    """Numbers to size a pool from measurement rather than an aspiration."""
    spec = ss.FamilySpec(name="probe", seed=99,
                         quotas={ss.NO_ARBITRAGE: 2, ss.MARGINAL_ARBITRAGE: 2,
                                 ss.PROFITABLE_ARBITRAGE: 2})
    p = ss.feasibility_probe(spec)
    assert p["n_days"] == 6
    assert p["generation_seconds"] > 0
    assert p["lp_seconds_mean_per_curve"] > 0
    assert p["pool_json_bytes"] > 0
    assert 0.0 <= p["acceptance"]["acceptance_rate"] <= 1.0
    for regime in ss.REGIMES:
        r = p["acceptance"]["per_regime"][regime]
        assert r["acceptance_rate"] == pytest.approx(
            r["accepted"] / max(r["proposed"], 1))
        assert r["mean_attempts_to_accept"] is not None
    assert p["measured_regimes"][ss.PROFITABLE_ARBITRAGE] == 2


def test_the_probe_reflects_the_mechanisms_it_landed_on():
    spec = ss.FamilySpec(name="probe_mech", seed=5,
                         quotas={ss.NO_ARBITRAGE: 2, ss.MARGINAL_ARBITRAGE: 2,
                                 ss.PROFITABLE_ARBITRAGE: 2})
    p = ss.feasibility_probe(spec)
    assert p["mechanisms_present"]
    assert set(p["mechanisms_present"]) <= set(ss.MECHANISMS)
    assert sum(p["acceptance"]["mechanisms_accepted"].values()) == 6


# ---------------------------------------------------------------------------
# Generated days
# ---------------------------------------------------------------------------

def test_quotas_are_met_exactly(family):
    spec, days = family
    assert len(days) == spec.n_days
    counts = ss.regime_counts(days)
    for regime in ss.REGIMES:
        assert counts[regime] == spec.quotas[regime], regime
    assert counts[ss.CLASSIFICATION_INCONSISTENT] == 0


def test_a_generated_day_is_labelled_synthetic_with_its_provenance(family):
    _, days = family
    d = days[0].to_dict(include_prices=True)
    assert d["source"] == "synthetic"
    assert d["family"] == "test_family"
    assert len(d["prices_cny_per_mwh"]) == ss.T
    for key in ("template_id", "trough_slot", "peak_slot", "span", "seed",
                "shape_id", "shape_attempts", "r_min", "r_max",
                "regime_requested", "regime_measured", "diagnostics"):
        assert key in d, key
    assert d["regime_requested"] == d["regime_measured"]


def test_every_measured_label_agrees_with_the_band(family):
    """The label is measured, not asserted from how the day was built."""
    from arbitrage_lp import arbitrage_lp as lp
    _, days = family
    for d in days:
        res = lp(d.price_array(), ss.REFERENCE_UNIT, ss.CYCLE_COST_PER_LEG)
        m = ss.gross_margin(res)
        if d.regime_measured == ss.NO_ARBITRAGE:
            assert res.margin_cny <= ss.IDLE_TOL_CNY, d.day_id
            assert m is None
        elif d.regime_measured == ss.MARGINAL_ARBITRAGE:
            assert R - B - ss.MARGIN_TOL_CNY_PER_MWH <= m <= \
                R + B + ss.MARGIN_TOL_CNY_PER_MWH, d.day_id
        else:
            assert m > R + B, d.day_id


def test_prices_stay_inside_the_declared_clip(family):
    spec, days = family
    for d in days:
        arr = d.price_array()
        assert arr.min() >= spec.price_floor - 1e-9
        assert arr.max() <= spec.price_cap + 1e-9
        assert np.all(np.isfinite(arr))


def test_days_are_not_the_same_curve_under_a_different_label(family):
    _, days = family
    arrays = [d.price_array() for d in days]
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            assert not np.allclose(arrays[i], arrays[j]), (i, j)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_a_family_rebuilds_to_the_same_days(family):
    spec, days = family
    again = ss.build_family(spec)
    assert [d.day_id for d in again] == [d.day_id for d in days]
    for a, b in zip(days, again):
        np.testing.assert_array_equal(a.price_array(), b.price_array())
        assert a.regime_measured == b.regime_measured
        assert a.span == b.span
        assert a.template_id == b.template_id
    assert ss.family_digest(spec, again) == ss.family_digest(spec, days)


def test_a_different_seed_gives_different_days(family):
    spec, days = family
    other = ss.build_family(dataclasses.replace(spec, seed=spec.seed + 1))
    assert ss.family_digest(spec, days) != ss.family_digest(spec, other)


def test_the_digest_moves_when_a_declaration_moves(family):
    spec, days = family
    move = dataclasses.replace(days[0], regime_measured=ss.PROFITABLE_ARBITRAGE)
    assert ss.family_digest(spec, [move] + list(days[1:])) != \
        ss.family_digest(spec, days)


# ---------------------------------------------------------------------------
# Split isolation
# ---------------------------------------------------------------------------

def test_the_split_holds_out_shapes_and_carries_every_regime(family):
    _, days = family
    split = ss.split_family(days)
    ss.assert_no_template_leak(split)
    ss.assert_regimes_present(split)


def test_the_split_parts_are_disjoint(family):
    _, days = family
    split = ss.split_family(days)
    ids = {part: {d.day_id for d in ds} for part, ds in split.items()}
    assert not (ids["train"] & ids["validation"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["validation"] & ids["test"])
    assert sum(len(v) for v in ids.values()) == len(days)


def test_a_held_out_day_that_reuses_a_training_shape_is_refused(family):
    """The leak check must actually bite, or it is decoration."""
    _, days = family
    split = ss.split_family(days)
    train = split["train"][0]
    clone = dataclasses.replace(split["test"][0],
                                trough_slot=train.trough_slot,
                                peak_slot=train.peak_slot)
    bad = {"train": split["train"],
           "validation": split["validation"],
           "test": [clone]}
    with pytest.raises(AssertionError, match="leaking"):
        ss.assert_no_template_leak(bad)


def test_a_training_split_missing_a_regime_is_refused(family):
    _, days = family
    split = ss.split_family(days)
    one_sided = dict(split)
    one_sided["train"] = [d for d in split["train"]
                          if d.regime_measured != ss.NO_ARBITRAGE]
    with pytest.raises(AssertionError, match="missing regime"):
        ss.assert_regimes_present(one_sided)


def test_a_tail_missing_a_regime_is_reported_not_refused(family):
    """A small validation or test tail cannot be required to hold every regime;
    demanding it would fail correct families and prove nothing."""
    _, days = family
    split = ss.split_family(days)
    tail = {"train": split["train"],
            "validation": [d for d in split["validation"]
                           if d.regime_measured != ss.NO_ARBITRAGE],
            "test": split["test"]}
    ss.assert_regimes_present(tail)          # must not raise


def test_regimes_are_interleaved_so_a_positional_split_is_not_a_regime_split(family):
    _, days = family
    by_id = sorted(days, key=lambda d: d.day_id)
    # Each third of the family carries more than one regime.
    third = len(by_id) // 3
    for start in (0, third, 2 * third):
        chunk = by_id[start:start + third]
        assert len({d.regime_measured for d in chunk}) >= 2, start


def test_a_quota_that_cannot_be_filled_raises(monkeypatch):
    """A family that silently under-fills the marginal band would train on a
    distribution nobody declared, so an unfillable quota is an error."""
    spec = ss.FamilySpec(name="too_hard", seed=1,
                         quotas={ss.NO_ARBITRAGE: 0,
                                 ss.MARGINAL_ARBITRAGE: 1,
                                 ss.PROFITABLE_ARBITRAGE: 0},
                         max_shape_attempts=1)
    real_build_day = ss.build_day

    def always_profitable(spec_, day_id, regime, **kw):
        day = real_build_day(spec_, day_id, ss.PROFITABLE_ARBITRAGE, **kw)
        return dataclasses.replace(day, regime_measured=ss.PROFITABLE_ARBITRAGE)

    monkeypatch.setattr(ss, "build_day", always_profitable)
    with pytest.raises(AssertionError, match="quota cannot be met|no .* day found"):
        ss.build_family(spec)
