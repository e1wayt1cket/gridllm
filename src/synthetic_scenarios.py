# synthetic_scenarios.py
"""Synthetic day-ahead price scenarios in three declared arbitrage regimes.

The shipped workbook holds one real day and three window means, so it cannot
support a train / validation / test split over days. This module generates days
instead, and labels each one by **what the reference battery can actually
capture on it** -- not by how wide its peak-to-trough range looks. A wide range
that a 0.4 MW / 1.6 MWh unit cannot be on both ends of is not an opportunity,
and the earlier analysis measured that overstatement at 1.11x median and 1.39x
worst.

**A synthetic day is never evidence about the real market.** Every generated
scenario carries `source: "synthetic"`, its family, its template, its generator
parameters and its seed. Nothing here is a measurement, and no result computed
on this module is a statement about Jiangsu day-ahead prices.

The margin measure
------------------

`required_margin` (R) is a **gross price margin per MWh stored**. The LP's
`margin_cny` is **net revenue in CNY, after cycle cost**, so the two must not be
compared directly -- that is a CNY-vs-CNY-per-MWh error whose size is the unit's
capacity factor. The quantity comparable to R is

    Q_ch = sum(ch * dt)        Q_dis = sum(dis * dt)
    M_LP = (margin_cny + cycle_cost * (Q_ch + Q_dis)) / (eta_ch * Q_ch)

which equals `R + margin_cny / (eta_ch * Q_ch)`. That identity is exact **when
the terminal SOC is balanced**: balance makes `Q_dis / Q_ch = eta_ch * eta_dis`,
which is what collapses the cycle-cost term onto R. A day whose endpoints do not
balance therefore has no `M_LP`, and this module refuses rather than guesses.

One structural consequence, stated because it shapes the labels: the LP's
optimum is non-negative, and it schedules only when it is positive. So a
**scheduled** day always has `M_LP > R`. The band below R is unreachable with a
non-empty schedule, and `marginal_arbitrage` therefore means "a positive
capturable margin no larger than R + b" -- barely above break-even, not
straddling it. A scheduled day landing below `R - b` is a generator or
parameterisation error, reported as `classification_inconsistent` rather than
absorbed into `no_arbitrage`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from arbitrage_lp import UnitSpec, arbitrage_lp, required_margin
from money import DT_HOURS

# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

NO_ARBITRAGE = "no_arbitrage"
MARGINAL_ARBITRAGE = "marginal_arbitrage"
PROFITABLE_ARBITRAGE = "profitable_arbitrage"
CLASSIFICATION_INCONSISTENT = "classification_inconsistent"

#: Emitted when `M_LP` cannot be defined at all: the schedule is non-empty but
#: the SOC endpoints do not balance, so the cycle-cost term does not collapse
#: onto R and any number computed would be a different quantity wearing the same
#: name. A diagnostic state, not a market regime.
MARGIN_STATE_INCONSISTENT = "margin_state_inconsistent"

#: How the LP's schedule relates to its revenue. Reported beside the regime, so
#: a zero-revenue day that nonetheless ran a full cycle is distinguishable from
#: one that did nothing.
SCHEDULE_NONE = "none"
SCHEDULE_DEGENERATE_ZERO_REVENUE = "degenerate_zero_revenue"
SCHEDULE_SCHEDULED = "scheduled"

REGIMES: Tuple[str, ...] = (NO_ARBITRAGE, MARGINAL_ARBITRAGE,
                            PROFITABLE_ARBITRAGE)

#: The reference battery the labels are measured on. Fixed, and not varied by
#: any caller: a label is a statement about what THIS unit can capture, so a
#: different unit would make the same day a different regime.
REFERENCE_UNIT = UnitSpec(
    e_max=1.6, p_ch_max=0.4, p_dis_max=0.4,
    eta_ch=0.92, eta_dis=0.92, soc0=0.50, soc_min=0.10, soc_max=0.90)

#: Degradation, per leg, CNY/MWh. Frozen with the rest of the economics.
CYCLE_COST_PER_LEG = 100.0

#: Half-width of the "marginal" band as a fraction of R. Approved at 10 %.
BAND_FRACTION = 0.10

#: Zero-REVENUE tolerance, in CNY. The LP returns float residue (~1e-17) on a
#: day with no opportunity, so an exact `== 0` test would call those scheduled.
IDLE_TOL_CNY = 1e-6

#: Zero-SCHEDULE tolerance, in MWh. Deliberately a separate constant from
#: `IDLE_TOL_CNY`, because the two are different questions and the LP can answer
#: them differently. At exact break-even the optimum is degenerate:
#: `margin_cny = -4e-14` **with a full non-zero schedule** (measured: Q_ch =
#: 0.6957 MWh). Revenue and schedule therefore have to be tested separately --
#: one tolerance for both would report that day as "no schedule" when it ran a
#: whole cycle, and would call a genuine idle day scheduled on any nonzero
#: residue.
ZERO_SCHEDULE_TOL_MWH = 1e-9

#: Tolerance on the band comparisons, CNY/MWh stored.
MARGIN_TOL_CNY_PER_MWH = 1e-6

#: Terminal-SOC balance tolerance, as a fraction of e_max.
SOC_BALANCE_TOL = 1e-6

#: Periods per day.
T = 96


def required_margin_cny_per_mwh(cycle_cost: float = CYCLE_COST_PER_LEG,
                                unit: UnitSpec = REFERENCE_UNIT) -> float:
    """R: the gross price margin a stored MWh must clear, CNY/MWh."""
    return required_margin(cycle_cost, unit.eta_ch, unit.eta_dis)


def marginal_band(cycle_cost: float = CYCLE_COST_PER_LEG,
                  unit: UnitSpec = REFERENCE_UNIT) -> float:
    """b, the half-width of the marginal regime, CNY/MWh."""
    return BAND_FRACTION * required_margin_cny_per_mwh(cycle_cost, unit)


def gross_margin(result, cycle_cost: float = CYCLE_COST_PER_LEG,
                 unit: UnitSpec = REFERENCE_UNIT) -> Optional[float]:
    """`M_LP`, or None when the LP scheduled nothing.

    None is not a failure and not a NaN to be compared: it is the
    `no_arbitrage` case, where the ratio is undefined and the classifier must
    branch on the schedule instead. Raises if the schedule is non-empty but the
    SOC endpoints do not balance, because then `M_LP` has no meaning.
    """
    q_ch = float(np.sum(result.ch_mw) * DT_HOURS)
    q_dis = float(np.sum(result.dis_mw) * DT_HOURS)
    # The idle test is the SAME one `classify` uses. Testing `q_ch <= 0` here
    # instead would let a numerically-scheduled but economically-empty day
    # return a number from this function and be called idle by the classifier,
    # so two functions would disagree about whether the day has a margin at all.
    if q_ch * unit.eta_ch <= 0.0 or result.margin_cny <= IDLE_TOL_CNY:
        return None
    # Balance is what makes the identity hold; without it the cycle-cost term
    # does not collapse onto R and M_LP would silently mean something else.
    if abs(result.soc[-1] - result.soc[0]) > SOC_BALANCE_TOL:
        raise ValueError(
            f"SOC endpoints do not balance ({result.soc[0]!r} -> "
            f"{result.soc[-1]!r}); M_LP is undefined without the terminal pin")
    return (result.margin_cny + cycle_cost * (q_ch + q_dis)) / \
        (unit.eta_ch * q_ch)


def classify(result, cycle_cost: float = CYCLE_COST_PER_LEG,
             unit: UnitSpec = REFERENCE_UNIT,
             band: Optional[float] = None) -> Dict[str, object]:
    """The regime label for one LP result, plus the quantities behind it.

    Branch order matters. The schedule is tested FIRST, because when the LP
    schedules nothing both per-MWh ratio fields are NaN and a threshold on a
    ratio has nothing to compare -- the `no_arbitrage` case is exactly the one a
    ratio-first classifier drops.
    """
    r_required = required_margin_cny_per_mwh(cycle_cost, unit)
    b = marginal_band(cycle_cost, unit) if band is None else float(band)
    q_ch = float(np.sum(result.ch_mw) * DT_HOURS)
    q_dis = float(np.sum(result.dis_mw) * DT_HOURS)
    diag: Dict[str, object] = {
        "margin_cny": float(result.margin_cny),
        "q_charge_mwh": q_ch,
        "q_discharge_mwh": q_dis,
        "chargeable_energy_mwh": float(result.chargeable_energy_mwh),
        "required_margin_cny_per_mwh": r_required,
        "band_cny_per_mwh": b,
        "lp_status": int(result.status),
    }

    # Revenue and schedule are tested SEPARATELY. At exact break-even the LP's
    # optimum is degenerate and it can return a full non-zero schedule with zero
    # revenue; testing only the revenue would describe that day as "no
    # schedule", and testing only the schedule would call any nonzero residue a
    # cycle. An economically no-arbitrage day stays `no_arbitrage` either way --
    # a degenerate zero-revenue schedule is not an opportunity, and calling it
    # one would manufacture arbitrage out of a tie.
    scheduled = max(q_ch, q_dis) > ZERO_SCHEDULE_TOL_MWH
    if result.margin_cny <= IDLE_TOL_CNY:
        diag["regime"] = NO_ARBITRAGE
        diag["m_lp_cny_per_mwh"] = None
        if scheduled:
            diag["schedule_class"] = SCHEDULE_DEGENERATE_ZERO_REVENUE
            diag["reason"] = (
                "zero revenue within tolerance but a non-empty schedule: the "
                "optimum is degenerate at break-even. Economically no arbitrage "
                "-- a tie is not an opportunity -- but the schedule is real and "
                "is reported rather than called 'no schedule'")
        else:
            diag["schedule_class"] = SCHEDULE_NONE
            diag["reason"] = "no schedule: the LP optimum is zero within tolerance"
        return diag

    diag["schedule_class"] = SCHEDULE_SCHEDULED
    try:
        m_lp = gross_margin(result, cycle_cost, unit)
    except ValueError as exc:
        # `M_LP` is only defined on a valid charge energy with balanced
        # endpoints. An inconsistent state gets its own label instead of a
        # number, so it cannot be read as a market outcome.
        diag["regime"] = MARGIN_STATE_INCONSISTENT
        diag["m_lp_cny_per_mwh"] = None
        diag["reason"] = f"M_LP is undefined for this state: {exc}"
        return diag
    diag["m_lp_cny_per_mwh"] = m_lp
    # A scheduled day always has M_LP > R, so this branch is a generator-error
    # guard rather than a regime. It is reported, never folded into
    # `no_arbitrage`, which would hide the error behind a valid-looking label.
    if m_lp < r_required - b - MARGIN_TOL_CNY_PER_MWH:
        diag["regime"] = CLASSIFICATION_INCONSISTENT
        diag["reason"] = (
            "a non-empty schedule produced a gross margin below R - b; the LP "
            "schedules only when profitable, so this indicates a generator or "
            "parameterisation error rather than a market outcome")
        return diag
    if m_lp <= r_required + b + MARGIN_TOL_CNY_PER_MWH:
        diag["regime"] = MARGINAL_ARBITRAGE
        diag["reason"] = "positive capturable margin within the band above R"
        return diag
    diag["regime"] = PROFITABLE_ARBITRAGE
    diag["reason"] = "capturable margin beyond R + b"
    return diag


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

#: Minimum separation, in 15-minute slots, that a held-out day's (trough, peak)
#: pair must keep from every training day's pair. A finite template set large
#: enough to be interesting is smaller than a family's day count, so
#: "disjoint template ids" is unsatisfiable by construction and would either
#: fail always or be relaxed until it meant nothing. Separation is the checkable
#: version of the same intent: a held-out day must not be a near-copy of a
#: training day's shape.
MIN_TEMPLATE_SEPARATION = 6

#: Strides that walk the 96-slot day. Coprime with 96 so successive day ids do
#: not revisit early slots, and chosen so no two days in a family of a few dozen
#: land closer than MIN_TEMPLATE_SEPARATION.
_TROUGH_STRIDE = 13
_PEAK_STRIDE = 11
_PEAK_MIN_AHEAD = 20
_PEAK_MAX_AHEAD = 79


#: Structural price mechanisms. These are different *shapes*, not one cosine at
#: different amplitudes -- a family that varies only the peak height explores one
#: structure, and a policy can learn that structure rather than the task. The set
#: is declared and closed so "which mechanisms were used" is answerable.
#:
#: **This is not a claim about Jiangsu.** These are synthetic structures chosen
#: for mechanical diversity; nothing here is fitted to, or evidence about, the
#: real market's distribution.
MECHANISMS: Tuple[str, ...] = (
    "single_peak",      # one trough, one peak -- the original cosine
    "double_peak",      # a morning and an evening peak, midday saddle
    "broad_plateau",    # a wide flat top, sharp troughs
    "shifted_trough",   # the trough relocated into a different band
)


def _mechanism_shape(mechanism: str, trough_slot: int, peak_slot: int,
                     T_periods: int = T) -> np.ndarray:
    """The intraday [0, 1] shape for one mechanism.

    Each mechanism is a different functional form, so a family built from the
    full set covers structure rather than only amplitude.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; "
                         f"expected one of {MECHANISMS}")
    base = _shape(trough_slot, peak_slot, T_periods)
    if mechanism == "single_peak":
        return base
    if mechanism == "double_peak":
        # Fold in a second, weaker peak half a day away, producing a saddle.
        second = _shape(trough_slot, (peak_slot + T_periods // 2) % T_periods,
                        T_periods)
        return np.clip(np.maximum(base, 0.85 * second), 0.0, 1.0)
    if mechanism == "broad_plateau":
        # Push values near the top towards 1 and leave the trough alone: a wide
        # flat top rather than a spike.
        return np.clip(base ** 0.4, 0.0, 1.0)
    # shifted_trough: relocate the minimum into the second half of the day.
    shifted = (trough_slot + T_periods // 3) % T_periods
    return _shape(shifted, peak_slot, T_periods)


def _day_shape_params(day_id: int, T_periods: int = T) -> Tuple[int, int]:
    """The (trough_slot, peak_slot) pair for one day, from its id alone.

    Distinct per day rather than drawn from a small table, so a family's
    held-out days get shapes no training day used.
    """
    trough = (day_id * _TROUGH_STRIDE) % T_periods
    ahead = _PEAK_MIN_AHEAD + (day_id * _PEAK_STRIDE) % (
        _PEAK_MAX_AHEAD - _PEAK_MIN_AHEAD + 1)
    peak = (trough + ahead) % T_periods
    return trough, peak

#: Target offset of `M_LP` from R, in units of the band, per requested regime.
#: Aiming at the middle of a regime rather than its edge keeps a day from
#: flipping label on a solver-residue-sized change.
TARGET_OFFSETS: Dict[str, float] = {
    # Flat enough that the LP declines to cycle at all.
    NO_ARBITRAGE: -3.0,
    MARGINAL_ARBITRAGE: +0.5,
    PROFITABLE_ARBITRAGE: +3.0,
}


def _shape(trough_slot: int, peak_slot: int, T_periods: int = T) -> np.ndarray:
    """A smooth [0, 1] intraday shape with its minimum at `trough_slot`.

    Cosine interpolation between the peak and the trough, so the curve has no
    corner a numerical solver could latch onto, and both endpoints are declared
    rather than emergent.
    """
    t = np.arange(T_periods, dtype=float)
    # Distance to the peak, measured on the circle so a peak near midnight does
    # not tear the curve.
    d_peak = np.minimum(np.abs(t - peak_slot), T_periods - np.abs(t - peak_slot))
    d_trough = np.minimum(np.abs(t - trough_slot),
                          T_periods - np.abs(t - trough_slot))
    span = np.abs(peak_slot - trough_slot)
    if span == 0:
        raise ValueError("trough and peak must differ")
    # 1 at the peak, 0 at the trough, smooth in between.
    x = (d_peak - d_trough) / span
    x = np.clip(x, -1.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * (1.0 - x)))


@dataclass(frozen=True)
class FamilySpec:
    """One scenario family: the unit of splitting and of normalization."""

    name: str
    seed: int
    #: How many days of each regime the family must emit. Fixed in advance so a
    #: family cannot be dominated by whichever regime its generator finds easy.
    quotas: Dict[str, int]
    #: The **top-level** family this belongs to. Isolation is enforced here and
    #: not at `day_id` or `template_id`: two days of one top-level family share a
    #: generating process, so splitting them across train and test is not
    #: family isolation however different their templates look. Defaults to
    #: `name`, so a single-family spec cannot accidentally look isolated.
    family_id: Optional[str] = None
    #: Search bounds on the stack-position amplitude, in `r` units.
    span_lo: float = 0.02
    span_hi: float = 0.98
    #: Bisection steps. Fixed, so the search is deterministic and reproducible.
    search_steps: int = 44
    #: Declared clip on the generated price, CNY/MWh.
    price_floor: float = 120.0
    price_cap: float = 1000.0
    noise_sigma: float = 15.0
    #: The marginal band is 10 % of R, and on many shapes the LP's margin jumps
    #: from "declines to cycle" to "comfortably profitable" without landing
    #: inside it. A requested regime is therefore not reachable on an arbitrary
    #: shape, and the generator walks candidate shapes until one lands. Both
    #: constants are declared here rather than chosen per run.
    shape_stride: int = 7
    max_shape_attempts: int = 24

    @property
    def n_days(self) -> int:
        return int(sum(self.quotas.values()))

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name, "seed": int(self.seed),
            "quotas": dict(self.quotas), "span_lo": self.span_lo,
            "span_hi": self.span_hi, "search_steps": self.search_steps,
            "price_floor": self.price_floor, "price_cap": self.price_cap,
            "noise_sigma": self.noise_sigma,
            "reference_unit": {
                "e_max": REFERENCE_UNIT.e_max,
                "p_ch_max": REFERENCE_UNIT.p_ch_max,
                "p_dis_max": REFERENCE_UNIT.p_dis_max,
                "eta_ch": REFERENCE_UNIT.eta_ch,
                "eta_dis": REFERENCE_UNIT.eta_dis,
                "soc0": REFERENCE_UNIT.soc0,
                "soc_min": REFERENCE_UNIT.soc_min,
                "soc_max": REFERENCE_UNIT.soc_max,
            },
            "cycle_cost_per_leg": CYCLE_COST_PER_LEG,
            "band_fraction": BAND_FRACTION,
        }


@dataclass(frozen=True)
class ScenarioDay:
    """One generated day, with everything needed to rebuild and audit it."""

    family: str
    day_id: int
    regime_requested: str
    regime_measured: str
    template_id: str
    trough_slot: int
    peak_slot: int
    span: float
    seed: int
    prices: Tuple[float, ...]
    diagnostics: Dict[str, object]
    r_min: float
    r_max: float
    #: Which candidate shape produced this day, and how many were tried before
    #: it landed in the requested regime. Recorded because a regime is not
    #: reachable on an arbitrary shape; a day that took many attempts is a
    #: narrow-band day, not an ordinary one.
    shape_id: int = 0
    shape_attempts: int = 1
    #: Which structural price mechanism this day was built from. Recorded
    #: because amplitude alone is not diversity: a pool that varied only the
    #: peak height would carry one structure.
    mechanism: str = ""

    @property
    def is_synthetic(self) -> bool:
        return True

    def price_array(self) -> np.ndarray:
        return np.asarray(self.prices, dtype=float)

    def to_dict(self, include_prices: bool = False) -> Dict[str, object]:
        d: Dict[str, object] = {
            "source": "synthetic",
            "family": self.family,
            "day_id": int(self.day_id),
            "regime_requested": self.regime_requested,
            "regime_measured": self.regime_measured,
            "template_id": self.template_id,
            "trough_slot": int(self.trough_slot),
            "peak_slot": int(self.peak_slot),
            "span": float(self.span),
            "shape_id": int(self.shape_id),
            "shape_attempts": int(self.shape_attempts),
            "mechanism": self.mechanism,
            "seed": int(self.seed),
            "r_min": float(self.r_min),
            "r_max": float(self.r_max),
            "diagnostics": dict(self.diagnostics),
        }
        if include_prices:
            d["prices_cny_per_mwh"] = [float(p) for p in self.prices]
        return d


def stack_price(r: np.ndarray) -> np.ndarray:
    """The configured supply stack evaluated at stack positions `r` in [0, 1].

    Read through `wholesale_price` rather than restated, so the synthetic price
    model and the shipped one cannot drift into two definitions of what a price
    is.
    """
    import wholesale_price as wp
    from config_loader import get_default
    cfg = get_default(wp.CONFIG_KEY, {}) or {}
    stack = wp.SupplyStack.from_config(cfg["supply_stack"])
    return np.asarray([stack.price(float(x)) for x in np.clip(r, 0.0, 1.0)],
                      dtype=float)


def _family_r_range(spec: FamilySpec) -> Tuple[float, float]:
    """The family's own normalization yardstick, in raw stack units.

    **This is the per-family normalization.** `r` is a day's position measured
    against the range its OWN family spans, not against a whole library's.
    Library-wide normalization makes every price a function of how many days the
    library happens to contain -- the reported consequence of the current
    `synthetic_v1` yardstick -- and the approved per-family rule removes that
    dependence. The accepted cost is that days are not comparable across
    families; nothing needs them to be, because a family is the unit of
    splitting.

    The bounds returned are the **declared** ones, `[0, span_hi]`, and no day's
    realized extreme enters them -- which is the property that matters, because
    a day that scaled itself would make the normalization circular.

    They are an **upper bound** on the realized range rather than an exact one:
    shapes are bounded by `[0, 1]` but sampled on a discrete 96-slot grid, so a
    peak slot lands short of the analytic maximum (measured 0.883 for
    `single_peak`). The effect is that generated days occupy a sub-range of the
    declared span, never more than it. Normalizing each mechanism to reach
    exactly 1 would tighten that, and is deliberately **not** done here: it would
    rescale every generated price and so invalidate the calibration computed
    from the previous pool. It is recorded as an open item for the training-pool
    design instead.
    """
    return 0.0, float(spec.span_hi)


def _day_template(day_id: int) -> Dict[str, object]:
    """The declared shape of one day, with a readable id for the audit trail."""
    trough, peak = _day_shape_params(day_id)
    return {
        "id": f"t{trough:02d}_p{peak:02d}",
        "trough_slot": trough,
        "peak_slot": peak,
    }


def _noise(seed: int, T_periods: int = T) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=T_periods)


def _prices_for_span(spec: FamilySpec, day_id: int, span: float,
                     r_lo: float, r_hi: float,
                     shape_id: Optional[int] = None,
                     mechanism: Optional[str] = None
                     ) -> Tuple[np.ndarray, float, float]:
    sid = day_id if shape_id is None else shape_id
    tpl = _day_template(sid)
    if mechanism is None:
        mechanism = MECHANISMS[sid % len(MECHANISMS)]
    sh = _mechanism_shape(mechanism, int(tpl["trough_slot"]),
                          int(tpl["peak_slot"]))
    raw = span * sh
    width = max(r_hi - r_lo, 1e-12)
    r = (raw - r_lo) / width
    # Noise is keyed to the SHAPE, because that is what identifies a generated
    # day here; `day_id` is reassigned after collection so the regimes
    # interleave, and keying noise to it would make the seed field describe a
    # draw the day did not use.
    price = stack_price(r) + spec.noise_sigma * _noise(
        spec.seed + (day_id if shape_id is None else shape_id))
    return np.clip(price, spec.price_floor, spec.price_cap), float(r.min()), \
        float(r.max())


def _measure(spec: FamilySpec, prices: np.ndarray) -> Dict[str, object]:
    res = arbitrage_lp(prices, REFERENCE_UNIT, CYCLE_COST_PER_LEG)
    return classify(res, CYCLE_COST_PER_LEG, REFERENCE_UNIT)


def _target_margin(regime: str, target_override: Optional[float] = None) -> float:
    r_required = required_margin_cny_per_mwh()
    b = marginal_band()
    if regime == NO_ARBITRAGE:
        # Below R by more than the band: no cycle is worth running, so the LP
        # should decline entirely and the day lands in the idle branch.
        return r_required + TARGET_OFFSETS[NO_ARBITRAGE] * b
    if target_override is not None:
        return float(target_override)
    return r_required + TARGET_OFFSETS[regime] * b


def build_day(spec: FamilySpec, day_id: int, regime: str,
              target_margin: Optional[float] = None,
              shape_id: Optional[int] = None) -> ScenarioDay:
    """Generate one day, searching the span so its measured margin hits target.

    The search is a fixed-length bisection on the single scalar amplitude, using
    the LP as the measuring instrument. Fixed bounds and a fixed step count make
    it reproducible from `(spec, day_id, shape_id, regime)` alone -- no early
    exit on a tolerance, because that would make the result depend on the
    arithmetic path.

    `day_id` names the day (and its noise); `shape_id` selects the (trough,
    peak) pair, defaulting to `day_id`. They are separate because the requested
    regime is not reachable on every shape: the generator keeps the day's
    identity fixed and walks the shape until the requested regime is measured.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    if shape_id is None:
        shape_id = day_id
    r_lo, r_hi = _family_r_range(spec)
    target = _target_margin(regime, target_margin)

    if regime == NO_ARBITRAGE:
        # `no_arbitrage` cannot be reached by aiming at a numeric margin below
        # R: the LP's optimum is non-negative and it declines to cycle rather
        # than cycle at a loss, so a day with no opportunity is IDLE, and an
        # idle day has no M_LP to compare. The search therefore targets the
        # idle branch itself -- the largest amplitude that still leaves the LP
        # with nothing worth doing.
        lo, hi = spec.span_lo, spec.span_hi
        for _ in range(spec.search_steps):
            mid = 0.5 * (lo + hi)
            prices_mid, _, _ = _prices_for_span(spec, day_id, mid, r_lo, r_hi,
                                                shape_id)
            if float(_measure(spec, prices_mid)["margin_cny"]) <= IDLE_TOL_CNY:
                lo = mid              # idle here; a larger swing may still be
            else:
                hi = mid              # scheduled: too large, back off
        best_span = lo
    else:
        lo, hi = spec.span_lo, spec.span_hi
        best_span, best_diag, best_gap = lo, None, float("inf")
        best_prices = None
        for _ in range(spec.search_steps):
            mid = 0.5 * (lo + hi)
            prices_mid, _, _ = _prices_for_span(spec, day_id, mid, r_lo, r_hi,
                                                shape_id)
            diag_mid = _measure(spec, prices_mid)
            m = diag_mid["m_lp_cny_per_mwh"]
            # An idle day has no margin to be near the target; treat it as
            # maximally far so the search is pushed upward instead of stalling.
            gap = float("inf") if m is None else abs(float(m) - target)
            if gap < best_gap:
                best_gap, best_span, best_diag, best_prices = (
                    gap, mid, diag_mid, prices_mid)
            if m is None or float(m) < target:
                lo = mid
            else:
                hi = mid

    prices, d_rlo, d_rhi = _prices_for_span(spec, day_id, best_span, r_lo, r_hi,
                                            shape_id)
    diag = _measure(spec, prices)
    if regime != NO_ARBITRAGE and best_diag is not None \
            and best_diag["regime"] != diag["regime"]:
        diag, prices = best_diag, best_prices
    tpl = _day_template(shape_id)
    return ScenarioDay(
        family=spec.name, day_id=int(day_id), regime_requested=regime,
        regime_measured=str(diag["regime"]), template_id=str(tpl["id"]),
        trough_slot=int(tpl["trough_slot"]), peak_slot=int(tpl["peak_slot"]),
        span=float(best_span), seed=int(spec.seed + shape_id),
        prices=tuple(float(p) for p in prices), diagnostics=diag,
        r_min=d_rlo, r_max=d_rhi, shape_id=int(shape_id),
        mechanism=MECHANISMS[int(shape_id) % len(MECHANISMS)])


def build_family(spec: FamilySpec, seed_offset: int = 0,
                 accept_log: Optional[Dict[str, object]] = None
                 ) -> List[ScenarioDay]:
    """Every day of a family, filled to the declared per-regime quotas.

    Quotas are targets the generator must **meet**, not hints. Because a
    requested regime is not reachable on every shape -- the marginal band is
    10 % of R and the margin often jumps across it -- filling a quota walks
    candidate shapes until one lands. A quota that cannot be filled within
    `max_shape_attempts` raises rather than returning a short family: a family
    that quietly under-fills the marginal band would train on a distribution
    nobody declared, which is the failure the fixed quota exists to prevent.

    Days are numbered **after** collection so the regimes interleave. Numbering
    them as generated would place each regime in its own contiguous block, and a
    split by day id would then be a split by regime.
    """
    collected: Dict[str, List[ScenarioDay]] = {r: [] for r in REGIMES}
    shape_cursor = seed_offset
    # Acceptance accounting. Quota filling is a search, and a search that reports
    # only its winners hides both its cost and its selection bias: a regime
    # reached on the first attempt for every day and one reached after twenty
    # refusals produce pools drawn from different parts of the mechanism space.
    log: Dict[str, object] = {
        "proposed": 0, "solved": 0, "accepted": 0, "rejected": 0,
        "per_regime": {r: {"proposed": 0, "accepted": 0, "rejected": 0,
                           "attempts_to_accept": []} for r in REGIMES},
        "rejected_by_measured_regime": {},
        "mechanisms_accepted": {},
    }
    for regime in REGIMES:
        need = int(spec.quotas.get(regime, 0))
        for _ in range(need):
            landed: Optional[ScenarioDay] = None
            for attempt in range(spec.max_shape_attempts):
                cand = shape_cursor + attempt * spec.shape_stride
                log["proposed"] += 1
                log["per_regime"][regime]["proposed"] += 1
                day = build_day(spec, day_id=0, regime=regime, shape_id=cand)
                log["solved"] += 1
                if day.regime_measured == regime:
                    landed = replace(day, shape_attempts=attempt + 1)
                    shape_cursor = cand + spec.shape_stride
                    log["accepted"] += 1
                    log["per_regime"][regime]["accepted"] += 1
                    log["per_regime"][regime]["attempts_to_accept"].append(
                        attempt + 1)
                    log["mechanisms_accepted"][landed.mechanism] = \
                        log["mechanisms_accepted"].get(landed.mechanism, 0) + 1
                    break
                log["rejected"] += 1
                log["per_regime"][regime]["rejected"] += 1
                key = str(day.regime_measured)
                log["rejected_by_measured_regime"][key] = \
                    log["rejected_by_measured_regime"].get(key, 0) + 1
            if landed is None:
                if accept_log is not None:
                    accept_log.update(log)
                raise AssertionError(
                    f"family {spec.name}: no {regime!r} day found within "
                    f"{spec.max_shape_attempts} candidate shapes; the quota "
                    "cannot be met and the family must not be shipped short")
            collected[regime].append(landed)

    ordered: List[ScenarioDay] = []
    idx = 0
    while any(collected[r] for r in REGIMES):
        for r in REGIMES:
            if collected[r]:
                ordered.append(replace(collected[r].pop(0), day_id=idx))
                idx += 1
    if accept_log is not None:
        accept_log.update(log)
    return ordered


#: Fewer common valid scenarios than this and no ranking is offered.
def split_families(specs: Sequence[FamilySpec],
                   assignment: Dict[str, str]) -> Dict[str, List[str]]:
    """Assign whole **top-level families** to train / validation / test.

    Isolation lives here, not in `split_family`. Splitting one family's days
    across two uses is not family isolation: the two parts share a generating
    process, a mechanism set and a seed lineage, so a held-out day drawn from
    the training family is a near-relative however different its template looks.
    This function refuses any assignment that puts one `family_id` in two uses.

    `fam_a` is a single top-level family, so it can only ever occupy one use --
    which is why its 18 days are generator-validation data and nothing more.
    """
    seen: Dict[str, str] = {}
    for spec in specs:
        fam = spec.family_id or spec.name
        use = assignment.get(fam)
        if use is None:
            raise AssertionError(f"family {fam!r} has no declared split")
        if use not in ("train", "validation", "test"):
            raise AssertionError(f"family {fam!r}: unknown split {use!r}")
        if fam in seen:
            raise AssertionError(
                f"family {fam!r} appears twice; one top-level family may not "
                "be built twice")
        seen[fam] = use
    out: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    for spec in specs:
        out[seen[spec.family_id or spec.name]].append(spec.name)
    return out


def feasibility_probe(spec: FamilySpec, seed_offset: int = 0
                      ) -> Dict[str, object]:
    """Measure what one family actually costs before a size is promised.

    Reports wall-clock generation and LP time, the acceptance rate per regime,
    which mechanisms the search actually landed on, and the serialised size of
    the pool. The point is to size a training pool from measurement rather than
    from an aspiration -- and to expose a regime that is reached only by
    repeated refusal, which is a distributional fact about the pool, not just a
    cost.
    """
    import time
    log: Dict[str, object] = {}
    t0 = time.perf_counter()
    days = build_family(spec, seed_offset=seed_offset, accept_log=log)
    gen_s = time.perf_counter() - t0
    lp_times = []
    for d in days:
        t = time.perf_counter()
        _measure(spec, d.price_array())
        lp_times.append(time.perf_counter() - t)
    payload = json.dumps([d.to_dict(include_prices=True) for d in days])
    per_regime = log["per_regime"]
    return {
        "family": spec.name, "family_id": spec.family_id or spec.name,
        "n_days": len(days),
        "generation_seconds": gen_s,
        "generation_seconds_per_day": gen_s / max(len(days), 1),
        "lp_seconds_total": float(sum(lp_times)),
        "lp_seconds_mean_per_curve": float(np.mean(lp_times)),
        "acceptance": {
            "proposed": log["proposed"], "solved": log["solved"],
            "accepted": log["accepted"], "rejected": log["rejected"],
            "acceptance_rate": log["accepted"] / max(log["proposed"], 1),
            "per_regime": {
                r: {
                    "proposed": v["proposed"], "accepted": v["accepted"],
                    "rejected": v["rejected"],
                    "acceptance_rate": v["accepted"] / max(v["proposed"], 1),
                    "mean_attempts_to_accept": (
                        float(np.mean(v["attempts_to_accept"]))
                        if v["attempts_to_accept"] else None),
                    "max_attempts_to_accept": (
                        max(v["attempts_to_accept"])
                        if v["attempts_to_accept"] else None),
                } for r, v in per_regime.items()},
            "rejected_by_measured_regime": log["rejected_by_measured_regime"],
            "mechanisms_accepted": log["mechanisms_accepted"],
        },
        "measured_regimes": regime_counts(days),
        "mechanisms_present": sorted({d.mechanism for d in days}),
        "pool_json_bytes": len(payload.encode("utf-8")),
        "bytes_per_day": len(payload.encode("utf-8")) / max(len(days), 1),
    }


def family_digest(spec: FamilySpec, days: Sequence[ScenarioDay]) -> str:
    """A hash over the generator configuration and every day's declaration.

    Recorded beside a family so a later run can be checked against the one that
    produced the split, rather than asserted to match it.
    """
    payload = json.dumps({
        "spec": spec.to_dict(),
        "days": [d.to_dict(include_prices=False) for d in days],
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: Where a frozen calibration is recorded.
PRICE_REF_CALIBRATION_FILE = "data/wholesale/price_ref_calibration.json"

#: The calibration formula, kept as a string so the artefact states its own
#: rule and a reader does not have to reconstruct it from the code.
PRICE_REF_FORMULA = (
    "value = mean over families( mean over that family's TRAINING curves("
    "mean_t price[t] ) )  -- each family equally weighted, so a family with more "
    "curves cannot set the scale")


def calibrate_price_ref(days_by_family: Dict[str, Sequence[ScenarioDay]]
                        ) -> Dict[str, object]:
    """The price-normalization scale, computed from **training** days only.

    Three levels, in this order: each training curve's own daily-mean price,
    then a representative price per family, then an **equal-weighted** mean
    across families. The equal weighting is the point -- a plain mean over all
    curves would let whichever family the generator happened to emit more of
    decide the scale, and the scale is then used by validation and test days
    that had no say in it.

    The caller must pass training days only. This function cannot check that;
    `main` enforces it by calibrating on `split["train"]`, and a test asserts
    that changing a validation or test day leaves the frozen value identical.
    Anything derived here is a constant: it is frozen after this call and never
    re-fitted, because a scale re-derived per split is a cross-split leak even
    when no day is shared.
    """
    if not days_by_family:
        raise ValueError("no families supplied; nothing to calibrate from")
    per_family: Dict[str, Dict[str, object]] = {}
    for fam in sorted(days_by_family):
        days = list(days_by_family[fam])
        if not days:
            raise ValueError(f"family {fam!r} has no training days")
        curve_means = [float(d.price_array().mean()) for d in days]
        per_family[fam] = {
            "n_curves": len(curve_means),
            "curve_daily_means_cny_per_mwh": curve_means,
            "representative_price_cny_per_mwh": float(np.mean(curve_means)),
        }
    reps = [v["representative_price_cny_per_mwh"]
            for v in per_family.values()]
    value = float(np.mean(reps))
    body = {
        "formula": PRICE_REF_FORMULA,
        "value_cny_per_mwh": value,
        "n_families": len(per_family),
        "n_training_curves": int(sum(v["n_curves"] for v in per_family.values())),
        "per_family": per_family,
        "family_representative_prices_cny_per_mwh": reps,
        "excluded_from_calibration": [
            "validation split", "test split",
            "261009 (the single real day; held aside, never mixed in)",
        ],
    }
    digest = hashlib.sha256(json.dumps(
        {"formula": PRICE_REF_FORMULA,
         "families": {k: v["curve_daily_means_cny_per_mwh"]
                      for k, v in per_family.items()}},
        sort_keys=True).encode("utf-8")).hexdigest()
    body["data_digest_sha256"] = digest
    return body


def regime_counts(days: Sequence[ScenarioDay]) -> Dict[str, int]:
    counts = {r: 0 for r in REGIMES}
    counts[CLASSIFICATION_INCONSISTENT] = 0
    for d in days:
        counts[d.regime_measured] = counts.get(d.regime_measured, 0) + 1
    return counts


def split_family(days: Sequence[ScenarioDay],
                 fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)
                 ) -> Dict[str, List[ScenarioDay]]:
    """Partition a family into train / validation / test by day id.

    The split is a pure function of the day order, which `build_family` has
    already interleaved by regime -- so no split is a regime split. Templates
    are checked afterwards (`assert_no_template_leak`) rather than assumed
    disjoint.
    """
    n = len(days)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    ordered = sorted(days, key=lambda d: d.day_id)
    return {
        "train": ordered[:n_train],
        "validation": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val:],
    }


def _shape_separation(a: ScenarioDay, b: ScenarioDay) -> int:
    """Slots between two days' (trough, peak) pairs, on the circular day."""
    dt = abs(a.trough_slot - b.trough_slot)
    dt = min(dt, T - dt)
    dp = abs(a.peak_slot - b.peak_slot)
    dp = min(dp, T - dp)
    return dt + dp


def assert_no_template_leak(split: Dict[str, List[ScenarioDay]]) -> None:
    """Every held-out day's shape must be far enough from every training day's.

    Checked as a distance rather than as template-id inequality. A finite
    template table smaller than the day count cannot give disjoint ids, so an
    id-equality check is either always failing or quietly vacuous; what actually
    matters is that a held-out day is not a near-copy of a day the policy
    trained on, which is a separation. `MIN_TEMPLATE_SEPARATION` is the declared
    threshold and is not tuned per family.
    """
    train = split["train"]
    if not train:
        raise AssertionError("no training days; nothing to hold out from")
    for part in ("validation", "test"):
        for d in split[part]:
            closest = min((_shape_separation(d, t) for t in train))
            if closest < MIN_TEMPLATE_SEPARATION:
                worst = min(train, key=lambda t: _shape_separation(d, t))
                raise AssertionError(
                    f"{part} day {d.day_id} ({d.template_id}) is "
                    f"{closest} slots from train day {worst.day_id} "
                    f"({worst.template_id}), below the declared minimum "
                    f"{MIN_TEMPLATE_SEPARATION}; the split is leaking "
                    "near-duplicate shapes")


def assert_regimes_present(split: Dict[str, List[ScenarioDay]]) -> None:
    """The **training** split must carry all three regimes.

    Only training. A small validation or test split cannot be required to hold
    every regime -- with a 60/20/20 split of a dozen interleaved days, a
    two-day validation tail simply does not contain all three, and demanding it
    would make the assertion fail on correct families while proving nothing. The
    requirement that matters is that the policy is trained on all three
    regimes; a tail missing one is a reported fact, not an error.
    """
    train = split.get("train", [])
    seen = {d.regime_measured for d in train}
    missing = [r for r in REGIMES if r not in seen]
    if train and missing:
        raise AssertionError(
            f"the training split is missing regime(s) {missing}; a policy "
            "cannot learn a regime it never sees")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print each family's regime distribution and its split digest."""
    import argparse
    p = argparse.ArgumentParser(
        description="Build synthetic three-regime price scenario families.")
    p.add_argument("--json", default=None, help="write a report to this path")
    p.add_argument("--quotas", default="6,6,6",
                   help="days per regime: no,marginal,profitable")
    p.add_argument("--calibrate-price-ref", action="store_true",
                   help="calibrate the price reference from the TRAIN split only")
    p.add_argument("--train-fraction", type=float, default=0.6,
                   help="fraction of a family that forms the training split")
    args = p.parse_args(argv)

    q = [int(x) for x in args.quotas.split(",")]
    spec = FamilySpec(name="fam_a", seed=20261013,
                      quotas={NO_ARBITRAGE: q[0], MARGINAL_ARBITRAGE: q[1],
                              PROFITABLE_ARBITRAGE: q[2]})
    days = build_family(spec)
    split = split_family(days)
    ok_leak = True
    try:
        assert_no_template_leak(split)
    except AssertionError as exc:
        ok_leak = False
        print(f"template leak: {exc}")
    print(f"family {spec.name}: {spec.n_days} days  "
          f"R={required_margin_cny_per_mwh():.6f}  b={marginal_band():.6f}")
    print(f"  requested : {spec.quotas}")
    print(f"  measured  : {regime_counts(days)}")
    print(f"  templates disjoint across split: {ok_leak}")
    for part, ds in split.items():
        print(f"  {part:11s} n={len(ds):2d}  "
              f"{ {r: sum(1 for d in ds if d.regime_measured == r) for r in REGIMES} }")
    print(f"  digest    : {family_digest(spec, days)}")

    if args.calibrate_price_ref:
        # TRAIN ONLY. The split is taken first and only its train part is
        # passed, so a validation or test day cannot reach the scale.
        split = split_family(days, (args.train_fraction,
                                    1.0 - args.train_fraction, 0.0))
        cal = calibrate_price_ref({spec.name: split["train"]})
        print("\nprice-reference calibration (TRAIN split only)")
        print(f"  formula   : {cal['formula']}")
        print(f"  families  : {cal['n_families']}  "
              f"training curves: {cal['n_training_curves']}")
        for fam, v in cal["per_family"].items():
            print(f"  {fam:10s} n={v['n_curves']:2d}  "
                  f"representative={v['representative_price_cny_per_mwh']:.6f}")
        print(f"  VALUE     : PRICE_REF_CNY_MWH = "
              f"{cal['value_cny_per_mwh']!r}")
        print(f"  data hash : {cal['data_digest_sha256']}")
        path = Path(PRICE_REF_CALIBRATION_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cal, fh, ensure_ascii=False, indent=2)
        print(f"  wrote     : {path}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"spec": spec.to_dict(),
                       "counts": regime_counts(days),
                       "digest": family_digest(spec, days),
                       "days": [d.to_dict(include_prices=True) for d in days]},
                      fh, ensure_ascii=False, indent=2)
        print(f"  wrote     : {args.json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
