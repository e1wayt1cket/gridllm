# market_opportunity.py
"""Does the native-scale market have arbitrage in it, across days rather than one?

The native IEEE33 day is physically clean but its strategic-bidding signal is
under-excited. Two explanations look identical from a single day and call for
opposite responses:

A. The market is *structurally* unable to pay for a battery cycle, so no policy
   will ever have anything to bid over. Stop and diagnose the price and
   degradation specifications; do not restore a larger study scale.
B. The market is viable and the one day is weak, so other native days carry real
   spreads and a multi-day training run would have a signal in it.

This module decides between them. For a fixed ensemble of days built at the
frozen scale it measures, per bus and per day, the **best a single unit with
perfect foresight could earn** against that day's nodal prices, in a clearing
that contains no batteries at all. That figure is an upper bound: no bidding
policy can beat it, so if it cannot clear the degradation floor the market is
case A and no policy work will change that.

Three properties of the measurement are load-bearing, and each is a decision
rather than a detail:

*The prices are exogenous.* The bound is taken against a clearing the audited
unit took no part in. A unit priced against a field it helped set is not
bounded by anything -- its own optimum moves the prices it is optimizing over.
`fleet_absent_day` is therefore not an option but the only usable price source.

*The spread is priced off the schedule the unit can run.* The day's cheapest
period and its dearest are not jointly reachable through a real battery, so the
minimum and maximum overstate the opportunity by roughly a factor of two at this
size. Both are reported; the bound uses the LP's own charge and discharge
prices.

*Nothing here tunes a price.* The ensemble varies the physical day -- weather
realization and demand shape, both at the frozen installed scale -- and never
the price level, the price elasticity, or the degradation cost. A price moved to
create a margin would make the answer a property of the move.

The verdict is reported on the test partition, which is fixed from the days'
*physical* regime labels before any economic quantity exists. Nothing is fitted
here, so the split does no statistical work; it fixes the sample the verdict is
reported on, and the report says so rather than letting a ratio imply otherwise.

Usage:
    PYTHONPATH=src python src/market_opportunity.py
    PYTHONPATH=src python src/market_opportunity.py --days 60 --json out.json --csv out.csv
    PYTHONPATH=src python src/market_opportunity.py --with-fleet --fleet-footprint
    PYTHONPATH=src python src/market_opportunity.py --source series \
        --series data/da_lmp.csv --series-column lmp_cny_mwh

Exit codes: 0 whenever a verdict is emitted, including INSUFFICIENT -- the
verdict is this module's deliverable, not its failure. 1 when the audit could
not run at all, 2 for a usage error.
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import money
from arbitrage_lp import (ArbitrageResult, UnitSpec, arbitrage_lp,
                          spec_from_storage)
from config_loader import (get_default, reload_defaults, reload_scenarios,
                           set_default)
from counterfactual_eval import summarize

# ---------------------------------------------------------------------------
# Declared constants
#
# Every threshold the verdict depends on is fixed here, before any measurement,
# so that it cannot be chosen to suit an answer. The ensemble and the split are
# functions of these and of nothing else.
# ---------------------------------------------------------------------------

#: Ensemble size when the caller does not say. Large enough that the share of
#: How many days an audit ensemble has is **not** declared here. It is the
#: caller's, and the caller is whoever knows which real days exist: a size in
#: this module is a size the data is then measured against, and it outlives
#: whatever it was written for.

#: train / validation / test shares. Proportions, not a day count: the sizes
#: they produce depend on how many days the caller supplies, which is the only
#: way a split can describe data whose length is not known here.
SPLIT_FRACTIONS: Tuple[float, float, float] = (0.5, 0.25, 0.25)
SPLIT_NAMES: Tuple[str, str, str] = ("train", "validation", "test")

#: The margin thresholds the summary reports probabilities for. They are read on
#: the per-MWh adjusted spread, where a ~200 CNY/MWh floor makes them
#: meaningful; the same numbers on the per-day money figure would be
#: unreachable at this unit size and would read as a fact about the market when
#: they are a fact about a 1.6 MWh battery.
MARGIN_THRESHOLDS: Tuple[float, ...] = (0.0, 50.0, 100.0)

#: Most shed load a day may carry and still count as a clean opportunity day.
#: Taken from `diagnose_typical_day.UNSERVED_LIMIT_MWH` so the two audits agree
#: on what a clean day is.
SHED_LIMIT_MWH = 1.1

#: Above this share of dropped days the ensemble has not answered the question
#: and the verdict is UNDETERMINED rather than whatever the survivors say.
MAX_DROP_SHARE = 0.20

#: The share of a unit's own quote box that has to stay tradeable for the bus to
#: be considered to have a strategy space. Taken from `diagnose_typical_day`,
#: which is where the siting rule is defined, so the two audits agree on what a
#: usable site is.
TRADEABLE_BOX_SHARE = 0.25

#: Degradation costs to report the sensitivity at, CNY/MWh per leg. Reported
#: only: no record, margin or verdict is ever computed at these values.
DEGRADATION_SENSITIVITIES: Tuple[float, ...] = (100.0, 60.0, 40.0)

#: The columns the audit's per-day per-bus table carries, in order. Declared as
#: one constant so a column cannot be renamed or dropped without a test failing.
REPORT_COLUMNS: Tuple[str, ...] = (
    "day", "bus", "partition", "stratum",
    "lmp_min_cny_mwh", "lmp_mean_cny_mwh", "lmp_max_cny_mwh", "lmp_range_cny_mwh",
    "feasible_charge_periods", "feasible_discharge_periods",
    "best_charge_price_cny_mwh", "best_discharge_price_cny_mwh",
    "gross_spread_cny_mwh",
    "efficiency_loss_cny_mwh", "degradation_cost_cny_mwh",
    "degradation_adjusted_spread_cny_mwh",
    "charge_mwh", "discharge_mwh", "equivalent_cycles",
    "arbitrage_margin_cny", "physical_margin_cny_mwh", "tradeable_box_share",
    "siting_eligible", "action_space_available", "clean",
    "lmp_price_source", "lmp_fallback_periods",
)

#: The load shapes the ensemble draws from. Named variants rather than sampled
#: parameters, so a reader can tell which days are shapes and which are noise
#: draws. Every one of them re-normalizes to a peak of 1.0, so the coincident
#: demand stays exactly the network's own however the peak is placed.
LOAD_SHAPE_MENU: Dict[str, dict] = {
    "declared": dict(morning_peak_hour=8.0, morning_peak_amplitude=0.40,
                     morning_peak_width=1.5, evening_peak_hour=19.0,
                     evening_peak_amplitude=0.55, evening_peak_width=1.8,
                     night_base=0.50),
    "late_evening": dict(morning_peak_hour=8.0, morning_peak_amplitude=0.40,
                         morning_peak_width=1.5, evening_peak_hour=20.5,
                         evening_peak_amplitude=0.60, evening_peak_width=1.6,
                         night_base=0.50),
    "sharp_evening": dict(morning_peak_hour=8.0, morning_peak_amplitude=0.35,
                          morning_peak_width=1.5, evening_peak_hour=19.0,
                          evening_peak_amplitude=0.70, evening_peak_width=1.2,
                          night_base=0.45),
    "high_night": dict(morning_peak_hour=8.0, morning_peak_amplitude=0.30,
                       morning_peak_width=1.5, evening_peak_hour=19.0,
                       evening_peak_amplitude=0.50, evening_peak_width=2.2,
                       night_base=0.62),
    "morning_led": dict(morning_peak_hour=7.0, morning_peak_amplitude=0.60,
                        morning_peak_width=1.8, evening_peak_hour=18.0,
                        evening_peak_amplitude=0.40, evening_peak_width=2.0,
                        night_base=0.50),
    "flat_day": dict(morning_peak_hour=8.0, morning_peak_amplitude=0.25,
                     morning_peak_width=2.5, evening_peak_hour=19.0,
                     evening_peak_amplitude=0.35, evening_peak_width=3.0,
                     night_base=0.65),
}
SHAPE_NAMES: Tuple[str, ...] = tuple(LOAD_SHAPE_MENU)

#: The keys the ensemble is allowed to move, and the ones it must not. The
#: frozen set is named explicitly because "we did not tune the price" is a claim
#: the report has to be able to substantiate, not an intention.
LEVER_KEYS: Tuple[str, ...] = (
    "profiles.load.morning_peak_hour", "profiles.load.morning_peak_amplitude",
    "profiles.load.morning_peak_width", "profiles.load.evening_peak_hour",
    "profiles.load.evening_peak_amplitude", "profiles.load.evening_peak_width",
    "profiles.load.night_base",
    "profiles.load.forecast_noise_seed", "profiles.load.real_noise_seed",
    "profiles.pv.cloud_seed", "profiles.wind.seed", "price_curve.seed",
)
FROZEN_KEYS: Tuple[str, ...] = (
    "price_curve.base", "price_curve.merit_order_price_elasticity",
    "price_curve.merit_order_min_price", "price_curve.merit_order_max_price",
    "price_curve.noise_sigma", "price_curve.spike_probability",
    "storage.cycle_cost",
)


# ---------------------------------------------------------------------------
# The day ensemble
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadShapeSpec:
    """One demand shape, in the seven parameters `_make_load_profile` reads."""

    morning_peak_hour: float
    morning_peak_amplitude: float
    morning_peak_width: float
    evening_peak_hour: float
    evening_peak_amplitude: float
    evening_peak_width: float
    night_base: float

    @classmethod
    def from_menu(cls, name: str) -> "LoadShapeSpec":
        if name not in LOAD_SHAPE_MENU:
            raise ValueError(f"unknown load shape {name!r}; "
                             f"expected one of {SHAPE_NAMES}")
        return cls(**LOAD_SHAPE_MENU[name])

    def as_defaults(self) -> Dict[str, float]:
        return {
            "profiles.load.morning_peak_hour": self.morning_peak_hour,
            "profiles.load.morning_peak_amplitude": self.morning_peak_amplitude,
            "profiles.load.morning_peak_width": self.morning_peak_width,
            "profiles.load.evening_peak_hour": self.evening_peak_hour,
            "profiles.load.evening_peak_amplitude": self.evening_peak_amplitude,
            "profiles.load.evening_peak_width": self.evening_peak_width,
            "profiles.load.night_base": self.night_base,
        }


@dataclass(frozen=True)
class DaySpec:
    """One day of the ensemble, as the levers that build it.

    A spec is the *definition* of a day and carries no outcome: no price, no
    dispatch, no margin. That is what lets the split be fixed before anything
    economic exists.
    """

    day: int
    source: str                       # "synthetic" | "series"
    shape_name: str
    load_shape: LoadShapeSpec
    pv_cloud_seed: int
    wind_seed: int
    price_seed: int
    load_fcast_seed: int
    load_real_seed: int
    #: Which day of the external wholesale library this day is priced on.
    #: Distinct from `day`, which is the ensemble's own index: once the price
    #: comes from an external market it is no longer a function of this day's
    #: physics, so the price day and the physical day are two different levers.
    day_id: int = 0
    series_label: Optional[str] = None
    series_price: Optional[np.ndarray] = None

    def as_defaults(self) -> Dict[str, object]:
        """The config paths this spec moves, ready for `set_default`."""
        out = dict(self.load_shape.as_defaults())
        out.update({
            "profiles.pv.cloud_seed": self.pv_cloud_seed,
            "profiles.wind.seed": self.wind_seed,
            # Both price levers are written. `price_curve.seed` drives the
            # retired map and `wholesale_market.day_id` drives the external
            # one; whichever is active reads its own and ignores the other.
            "price_curve.seed": self.price_seed,
            "wholesale_market.day_id": self.day_id,
            "profiles.load.forecast_noise_seed": self.load_fcast_seed,
            "profiles.load.real_noise_seed": self.load_real_seed,
        })
        return out


@dataclass(frozen=True)
class RegimeLabel:
    """A day's *physical* regime, computed before any money exists.

    Four axes, all of them properties of the profiles: how much sun, how much
    wind, how much demand, and when the demand peaks. The one thing deliberately
    absent is anything priced -- no LMP, no spread, no margin. A label that
    could see a price would let the split be chosen on the answer.
    """

    pv_energy: str        # low | mid | high
    wind_energy: str      # low | mid | high
    load_energy: str      # low | mid | high
    peak_timing: str      # morning | midday | evening

    @property
    def stratum(self) -> str:
        return (f"{self.pv_energy}/{self.wind_energy}/"
                f"{self.load_energy}/{self.peak_timing}")


@dataclass
class EnsembleDay:
    """A spec, its label, its partition, and (once measured) its prices."""

    spec: DaySpec
    label: RegimeLabel
    partition: str = ""
    physics: dict = field(default_factory=dict)
    lmp: Optional[np.ndarray] = None
    bus_indices: List[int] = field(default_factory=list)
    price_source: str = ""
    lmp_fallbacks: int = 0
    #: Which periods are not nodal, and why -- the count alone says how much of
    #: the day is off the measured curve but not which part of it.
    lmp_fallback_periods: List[int] = field(default_factory=list)
    lmp_fallback_reason: str = ""
    lmp_price_source: str = "nodal"
    shed_mwh: float = 0.0
    clean: bool = False
    notes: List[str] = field(default_factory=list)
    #: Fleet-present nodal prices, filled only when the fleet arm is run.
    lmp_with_fleet: Optional[np.ndarray] = None

    @property
    def day(self) -> int:
        return self.spec.day


def ensemble_specs(n: int, seed: int,
                   shape_names: Optional[Sequence[str]] = None
                   ) -> List[DaySpec]:
    """The declared day ensemble: named shapes crossed with drawn realizations.

    The shapes are cycled rather than sampled, so the ensemble contains every
    variant a reader can look up by name. The seeds are drawn from `seed`, so
    which days exist is reproducible while no single lever is hand-picked.
    """
    shapes = list(shape_names or SHAPE_NAMES)
    # The ensemble index is the day id: there is no separate library whose
    # length this would have to be taken modulo.
    external_days = int(n)
    rng = np.random.RandomState(int(seed))
    specs = []
    for i in range(int(n)):
        shape_name = shapes[i % len(shapes)]
        specs.append(DaySpec(
            day=i,
            source="synthetic",
            shape_name=shape_name,
            load_shape=LoadShapeSpec.from_menu(shape_name),
            # One external-market day per ensemble member. The ensemble's price
            # diversity comes from here when the external source is active; the
            # physical levers below no longer reach the price at all, because
            # the market is external and cannot be moved by this network.
            day_id=i % external_days,
            # Distinct streams, so one draw cannot alias another.
            pv_cloud_seed=int(rng.randint(1, 2 ** 31 - 1)),
            wind_seed=int(rng.randint(1, 2 ** 31 - 1)),
            price_seed=int(rng.randint(1, 2 ** 31 - 1)),
            load_fcast_seed=int(rng.randint(1, 2 ** 31 - 1)),
            load_real_seed=int(rng.randint(1, 2 ** 31 - 1)),
        ))
    return specs


def sweep_specs(lever: str, values: Sequence[object], base: DaySpec
                ) -> List[DaySpec]:
    """One day per value of one lever, everything else held identical.

    The ensemble's job is to answer whether *days* differ. This asks the
    complementary question: whether a named lever does anything on its own,
    which is how a lever that turns out to be inert gets noticed.

    The base day is supplied rather than defaulted: a sweep is a comparison
    against one day, and a day built from a size this module no longer knows is
    not a fixed point to sweep from.
    """
    fields = {
        "pv_cloud_seed": "pv_cloud_seed", "wind_seed": "wind_seed",
        "price_seed": "price_seed", "load_fcast_seed": "load_fcast_seed",
        "load_real_seed": "load_real_seed",
    }
    out = []
    for i, value in enumerate(values):
        if lever == "shape":
            shape_name = str(value)
            out.append(replace(base, day=i, shape_name=shape_name,
                               load_shape=LoadShapeSpec.from_menu(shape_name)))
        elif lever in fields:
            out.append(replace(base, day=i, **{fields[lever]: int(value)}))
        else:
            raise ValueError(
                f"unknown lever {lever!r}; expected 'shape' or one of "
                f"{sorted(fields)}")
    return out


def _apply_spec(spec: DaySpec) -> None:
    """Reset both caches and write this spec's levers into the defaults."""
    reload_defaults()
    reload_scenarios()
    for key, value in spec.as_defaults().items():
        set_default(key, value)


def build_day(spec: DaySpec, T: int = money.PERIODS_PER_DAY) -> EnsembleDay:
    """Build one day's agents and physics. No clearing, therefore no prices.

    This is pass one of the ensemble, and keeping it free of any clearing is
    what guarantees the labels cannot see an economic outcome: at this point
    none exists.
    """
    from scenarios import get_scenario, typical_day_config

    _apply_spec(spec)
    config = typical_day_config(T)
    # No fleet anywhere: the audit prices a unit against a market it is not in.
    agents, wholesale = get_scenario("typical_day", T, config, ess_buses=[])

    demand = np.sum([a.load_forecast for a in agents], axis=0)
    demand_real = np.sum([a.load_real for a in agents], axis=0)
    re_da = np.sum([a.pv_forecast + (a.wind_forecast if a.has_wind
                                     else np.zeros(T)) for a in agents], axis=0)
    pv = np.sum([a.pv_forecast for a in agents], axis=0)
    wind = np.sum([a.wind_forecast if a.has_wind else np.zeros(T)
                   for a in agents], axis=0)

    max_demand = float(np.max(demand)) if float(np.max(demand)) > 0 else 1.0
    net_load = demand - re_da
    dt = money.DT_HOURS

    physics = {
        "load_mwh": float(np.sum(demand_real)) * dt,
        "forecast_peak_mw": float(np.max(demand)),
        "real_peak_mw": float(np.max(demand_real)),
        "pv_mwh": float(np.sum(pv)) * dt,
        "wind_mwh": float(np.sum(wind)) * dt,
        "re_share": float(np.sum(re_da) / np.sum(demand))
                    if np.sum(demand) > 0 else 0.0,
        "net_load_swing_mw": float(np.max(net_load) - np.min(net_load)),
        # The two ratios the merit-order map actually consumes. They are the
        # whole of what the day contributes to the price level.
        "net_load_min_ratio": float(np.min(net_load) / max_demand),
        "net_load_max_ratio": float(np.max(net_load) / max_demand),
        # The realized peak is the network's own demand by construction: the
        # profiles are normalized to it after the noise is applied.
        "installed_pv_mw": float(sum(a.pv_capacity for a in agents)),
        "installed_wind_mw": float(sum(a.wind_capacity for a in agents)),
    }
    return EnsembleDay(spec=spec, label=RegimeLabel("", "", "", ""),
                       physics=physics,
                       notes=[f"wholesale {float(np.min(wholesale)):.1f}"
                              f"..{float(np.max(wholesale)):.1f} CNY/MWh"])


# ---------------------------------------------------------------------------
# Regime labels and the split
# ---------------------------------------------------------------------------

def _tercile(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def label_ensemble(days: Sequence[EnsembleDay],
                   T: int = money.PERIODS_PER_DAY) -> Dict[int, RegimeLabel]:
    """Label every day from its physics alone, by the ensemble's own terciles.

    Terciles rather than fixed cut-offs: the labels have to divide *this*
    ensemble, and an absolute threshold would put every day in one bucket
    whenever the ensemble sits in one climate.
    """
    pv = np.array([d.physics["pv_mwh"] for d in days], dtype=float)
    wind = np.array([d.physics["wind_mwh"] for d in days], dtype=float)
    load = np.array([d.physics["load_mwh"] for d in days], dtype=float)

    def cuts(a):
        return float(np.percentile(a, 33.3)), float(np.percentile(a, 66.7))

    pv_lo, pv_hi = cuts(pv)
    w_lo, w_hi = cuts(wind)
    l_lo, l_hi = cuts(load)

    labels = {}
    for i, d in enumerate(days):
        peak_hour = d.physics.get("peak_hour", None)
        if peak_hour is None:
            # Recovered from the shape's own declaration rather than from a
            # price or a dispatch, which do not exist yet.
            peak_hour = d.spec.load_shape.evening_peak_hour
        if peak_hour < 12.0:
            timing = "morning"
        elif peak_hour < 16.0:
            timing = "midday"
        else:
            timing = "evening"
        labels[d.day] = RegimeLabel(
            pv_energy=_tercile(float(pv[i]), pv_lo, pv_hi),
            wind_energy=_tercile(float(wind[i]), w_lo, w_hi),
            load_energy=_tercile(float(load[i]), l_lo, l_hi),
            peak_timing=timing,
        )
    return labels


def partition_days(labels: Dict[int, RegimeLabel],
                   fractions: Sequence[float] = SPLIT_FRACTIONS
                   ) -> Dict[str, List[int]]:
    """Deal the days into train/validation/test, stratified by label.

    A pure function of the label multiset: no RNG, no ordering dependence, and
    nothing economic in it. Days within a stratum are dealt to whichever
    partition is furthest below its share, ties broken train > validation >
    test, so permuting the input cannot move a day across partitions.
    """
    if len(fractions) != len(SPLIT_NAMES):
        raise ValueError(f"expected {len(SPLIT_NAMES)} fractions; got {fractions}")
    if not math.isclose(float(sum(fractions)), 1.0, abs_tol=1e-9):
        raise ValueError(f"fractions must sum to 1; got {fractions}")

    strata: Dict[str, List[int]] = {}
    for day, label in labels.items():
        strata.setdefault(label.stratum, []).append(day)

    total = len(labels)
    counts = {name: 0 for name in SPLIT_NAMES}
    out: Dict[str, List[int]] = {name: [] for name in SPLIT_NAMES}
    for stratum in sorted(strata):
        for day in sorted(strata[stratum]):
            best, best_key = None, None
            for i, name in enumerate(SPLIT_NAMES):
                deficit = total * float(fractions[i]) - counts[name]
                key = (deficit, -i)
                if best_key is None or key > best_key:
                    best, best_key = name, key
            counts[best] += 1
            out[best].append(day)
    return out


def underfilled_strata(days: Sequence[EnsembleDay],
                       partition: Dict[str, List[int]]) -> List[str]:
    """Strata too large to be represented by one test day but missing from test.

    Reported rather than silently absorbed: a stratum the test set never sees is
    a regime the verdict is not evidence about, and that has to be visible.
    """
    in_test = set(partition.get("test", []))
    by_stratum: Dict[str, List[int]] = {}
    for d in days:
        by_stratum.setdefault(d.label.stratum, []).append(d.day)
    return sorted(s for s, ds in by_stratum.items()
                  if len(ds) >= 4 and not (set(ds) & in_test))


def assign_partitions(days: Sequence[EnsembleDay]) -> None:
    """Label and partition in place, in that order, before any economics."""
    labels = label_ensemble(days)
    for d in days:
        d.label = labels[d.day]
    parts = partition_days(labels)
    for name, ids in parts.items():
        for day in ids:
            days[day].partition = name


# ---------------------------------------------------------------------------
# The audited unit, and one day's prices
# ---------------------------------------------------------------------------

def fleet_buses() -> List[int]:
    return [int(b) for b in get_default("independent_storage.buses", [])]


def audited_unit(bus: int, T: int = money.PERIODS_PER_DAY) -> UnitSpec:
    """The fleet's own unit, sited at `bus`.

    Read from `ess.build_ess_fleet`, which reads the config, so the audited
    battery is the configured one rather than a copy of its numbers that can
    drift. `bus` only decides the placeholder agent's name; the parameters are
    the fleet's, identical at every site by construction.
    """
    from ess import build_ess_fleet
    from models import MarketConfig
    config = MarketConfig.from_yaml(opf_mode="socp")
    fleet = build_ess_fleet(T, config, wholesale=None, buses=[int(bus)])
    if not fleet:
        raise RuntimeError(
            "the independent storage fleet is disabled or has no buses; "
            "there is no unit to audit")
    return spec_from_storage(fleet[0].storage)


def clear_day(spec: DaySpec, with_fleet: bool = False,
              T: int = money.PERIODS_PER_DAY) -> dict:
    """Clear one day and return its nodal prices and the facts about them.

    `with_fleet=False` is the audit's own price source: a market with no
    batteries in it. `with_fleet=True` adds the configured fleet, which is only
    ever used to measure what the fleet's own footprint costs -- never to price
    the bound.
    """
    from market import clear_market
    from scenarios import get_scenario, typical_day_config

    _apply_spec(spec)
    config = typical_day_config(T)
    agents, wholesale = get_scenario("typical_day", T, config,
                                     ess_buses=None if with_fleet else [])
    result = clear_market(agents, T, "DA", {}, config, wholesale=wholesale,
                          horizon_type="full_day"
                          if T == money.PERIODS_PER_DAY else "window")

    shed = sum(float(np.sum(result["schedules"][a.name]["unserved"]))
               for a in agents) * money.DT_HOURS
    return {
        "lmp": np.asarray(result["lmp"], dtype=float),
        "bus_indices": [int(b) for b in result["bus_indices"]],
        "wholesale": np.asarray(wholesale, dtype=float),
        # The clearing's own account of which periods are not nodal, carried
        # through unchanged rather than reduced to a count.
        "lmp_fallbacks": int(result.get("lmp_fallbacks", 0)),
        "lmp_fallback_periods": [int(t) for t in
                                 result.get("lmp_fallback_periods", ())],
        "lmp_fallback_reason": str(result.get("lmp_fallback_reason", "")),
        "lmp_price_source": str(result.get("lmp_price_source", "nodal")),
        "has_wholesale_fallback": bool(
            result.get("has_wholesale_fallback", False)),
        "shed_mwh": shed,
        "curtailment_mwh": float(result.get("total_curtailment", 0.0)),
        "agents": agents,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Per-day, per-bus metrics
# ---------------------------------------------------------------------------

def bus_day_metrics(lmp_node: np.ndarray, unit: UnitSpec, cycle_cost: float,
                    day: int, bus: int, dt: float = money.DT_HOURS,
                    partition: str = "", stratum: str = "",
                    clean: bool = True, lmp_price_source: str = "nodal",
                    lmp_fallback_periods: Sequence[int] = ()) -> dict:
    """Every quantity the audit reports for one bus on one day.

    The spread is quoted on the periods the LP chose to run in, and the day's
    extremes are reported beside it rather than instead of it. Both are needed:
    the extremes say what the price series offered, the schedule prices say what
    a real battery could reach, and the gap between them is a fact about the
    unit's power and energy limits.
    """
    from diagnose_typical_day import action_box, tradeable_box_share

    price = np.asarray(lmp_node, dtype=float)
    res: ArbitrageResult = arbitrage_lp(price, unit, cycle_cost, dt=dt)
    # How much of the quote box the unit can declare in and still cycle. The
    # repo's own siting criterion, and the quantity that decides whether a
    # policy has a strategy space rather than merely a positive profit: a
    # margin that clears zero but leaves 6% of the box tradeable gives a
    # learning run almost nothing to choose between.
    # Two different questions, kept apart. Whether a unit can cycle profitably
    # at all is physics: the spread over the degradation floor, which IS
    # `margin - required_margin` and has no quote box in it. Whether a *policy*
    # has room to choose is the action space, which is a property of the box.
    # Siting on the second made a bus's eligibility move when the box moved --
    # a statement about the action encoding, not about the bus.
    finite = bool(np.isfinite(res.adjusted_spread_cny_mwh))
    physical_margin = (float(res.adjusted_spread_cny_mwh) if finite
                       else float("-inf"))
    share = tradeable_box_share(physical_margin, action_box()) if finite else 0.0
    row = {
        "day": int(day),
        "bus": int(bus),
        "partition": partition,
        "stratum": stratum,
        "lmp_min_cny_mwh": float(np.min(price)),
        "lmp_mean_cny_mwh": float(np.mean(price)),
        "lmp_max_cny_mwh": float(np.max(price)),
        "lmp_range_cny_mwh": float(np.max(price) - np.min(price)),
        "feasible_charge_periods": res.feasible_charge_periods,
        "feasible_discharge_periods": res.feasible_discharge_periods,
        "best_charge_price_cny_mwh": res.best_charge_price,
        "best_discharge_price_cny_mwh": res.best_discharge_price,
        "gross_spread_cny_mwh": res.gross_spread_cny_mwh,
        "efficiency_loss_cny_mwh": res.efficiency_loss_cny_mwh,
        "degradation_cost_cny_mwh": res.degradation_cost_cny_mwh,
        "degradation_adjusted_spread_cny_mwh": res.adjusted_spread_cny_mwh,
        "charge_mwh": res.chargeable_energy_mwh,
        "discharge_mwh": res.dischargeable_energy_mwh,
        "equivalent_cycles": res.equivalent_cycles,
        "arbitrage_margin_cny": res.margin_cny,
        "physical_margin_cny_mwh": physical_margin,
        "tradeable_box_share": share,
        "siting_eligible": bool(physical_margin > 0.0),
        "action_space_available": bool(share >= TRADEABLE_BOX_SHARE),
        "clean": bool(clean),
        # Whether this bus's column of `lmp` is that day's market everywhere.
        # A row whose prices are partly the wholesale curve is not a nodal
        # price series, and every margin on it inherits that.
        "lmp_price_source": lmp_price_source,
        "lmp_fallback_periods": list(lmp_fallback_periods),
    }
    return row


def day_records(day: EnsembleDay, buses: Optional[Sequence[int]] = None,
                T: int = money.PERIODS_PER_DAY) -> List[dict]:
    """One record per bus for a cleared day. `None` means every bus in the day.

    `lmp_node` is passed as a plain series and nothing else is, so no clearing
    result can leak into the LP: the bound is computed against prices the unit
    had no part in.
    """
    if day.lmp is None:
        raise RuntimeError(f"day {day.day} has no prices; clear it first")
    index = {int(b): i for i, b in enumerate(day.bus_indices)}
    wanted = [int(b) for b in (buses if buses is not None else day.bus_indices)]
    cycle_cost = float(get_default("storage.cycle_cost"))
    unit_cache: Dict[int, UnitSpec] = {}
    rows = []
    for bus in wanted:
        if bus not in index:
            continue
        if bus not in unit_cache:
            unit_cache[bus] = audited_unit(bus, T)
        rows.append(bus_day_metrics(
            day.lmp[:, index[bus]], unit_cache[bus], cycle_cost,
            day=day.day, bus=bus, partition=day.partition,
            stratum=day.label.stratum, clean=day.clean,
            lmp_price_source=day.lmp_price_source,
            lmp_fallback_periods=day.lmp_fallback_periods))
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _threshold_stats(values: Sequence[float],
                     thresholds: Sequence[float] = MARGIN_THRESHOLDS) -> dict:
    """Distribution plus the share above each declared threshold.

    Days whose value is not finite are dropped from the denominator rather than
    counted as zero: a day the unit sat out has no spread to report, and
    treating "no trade" as "a spread of zero" would turn an absence of
    opportunity into a measurement of it.
    """
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=float)
    out = dict(summarize(list(arr)))
    out["n_missing"] = int(len(values) - arr.size)
    if arr.size == 0:
        # An empty sample is a count of zero, not an unmeasurable mean. Leaving
        # `n` as NaN would let a partition with no days read as a partition whose
        # statistics were merely unavailable.
        out["n"] = 0
    for th in thresholds:
        label = f"p_gt_{int(th)}" if float(th).is_integer() else f"p_gt_{th}"
        out[label] = (float(np.mean(arr > th)) if arr.size else float("nan"))
    return out


def bus_distribution(records: Sequence[dict], partition: Optional[str] = "test"
                     ) -> List[dict]:
    """Per-bus summary over the days of `partition` (None means every day)."""
    by_bus: Dict[int, List[dict]] = {}
    for r in records:
        if partition is not None and r["partition"] != partition:
            continue
        by_bus.setdefault(int(r["bus"]), []).append(r)

    out = []
    for bus in sorted(by_bus):
        rows = by_bus[bus]
        out.append({
            "bus": bus,
            "n_days": len(rows),
            "n_days_clean": int(sum(1 for r in rows if r["clean"])),
            "adjusted_spread_cny_mwh": _threshold_stats(
                [r["degradation_adjusted_spread_cny_mwh"] for r in rows]),
            "arbitrage_margin_cny": _threshold_stats(
                [r["arbitrage_margin_cny"] for r in rows]),
            "lmp_range_cny_mwh": _threshold_stats(
                [r["lmp_range_cny_mwh"] for r in rows]),
            # The share of the quote box the unit can declare in and still
            # cycle, and how often the bus clears the repo's own eligibility
            # threshold for one. A positive margin with a tiny box is a market
            # a policy cannot express a strategy in.
            "tradeable_box_share": _threshold_stats(
                [r["tradeable_box_share"] for r in rows],
                thresholds=(TRADEABLE_BOX_SHARE,)),
            "siting_eligible_rate": float(
                np.mean([bool(r["siting_eligible"]) for r in rows])),
        })
    return out


def fleet_distribution(records: Sequence[dict], buses: Sequence[int],
                       partition: Optional[str] = "test") -> dict:
    """The fleet's summary, with the honest label on the aggregate.

    `isolated_units_upper_bound_cny` sums the per-bus bounds. It is **not** a
    bound on what the fleet earns: four units share one price field and
    arbitrage each other's spread away, so the sum describes four separate
    markets. It is reported because it is the ceiling the fleet's own footprint
    reduces, and the reduction is the finding. The single best bus is the only
    fleet-adjacent figure that is a true bound on anything.
    """
    bus_set = {int(b) for b in buses}
    by_day: Dict[int, List[dict]] = {}
    for r in records:
        if int(r["bus"]) not in bus_set:
            continue
        if partition is not None and r["partition"] != partition:
            continue
        by_day.setdefault(int(r["day"]), []).append(r)

    per_day_median = []
    per_day_sum = []
    per_day_best = []
    for day in sorted(by_day):
        margins = [r["arbitrage_margin_cny"] for r in by_day[day]]
        per_day_median.append(float(np.median(margins)))
        per_day_sum.append(float(np.sum(margins)))
        per_day_best.append(float(np.max(margins)))

    return {
        "buses": sorted(bus_set),
        "n_days": len(by_day),
        "scope_note": ("isolated_units_upper_bound_cny sums four single-unit "
                       "bounds taken in a market containing none of them; it is "
                       "not a bound on the fleet's profit"),
        "median_bus_margin_cny": _threshold_stats(per_day_median),
        "isolated_units_upper_bound_cny": _threshold_stats(per_day_sum),
        "best_bus_margin_cny": _threshold_stats(per_day_best),
        "per_day_median_cny": per_day_median,
    }


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def _passing_days(per_day_margin: Dict[int, float]) -> List[int]:
    """Days whose median bus margin clears zero. Ties at zero do not pass."""
    return [d for d, v in per_day_margin.items() if v > 0.0]


def verdict(records: Sequence[dict], partition: Optional[str] = "test") -> dict:
    """Whether the native market has arbitrage in it, at the chosen bar.

    The bar: the median achievable margin over buses is above zero on at least
    half the days of the partition. It is a *weak* bar at this scale and the
    report says so -- a day fails only when the price range falls below the
    round-trip floor, which the native day does not. `verdict_attainable`, built
    by the caller from the fleet arm, is the reading that carries the decision;
    this one is reported because it was the declared rule.

    `partition=None` reads every day. Nothing is fitted here, so the partitions
    do no statistical work; reporting the same rule on the whole ensemble beside
    the test one is what keeps the split from implying that they do.
    """
    by_day: Dict[int, List[float]] = {}
    for r in records:
        if partition is not None and r["partition"] != partition:
            continue
        by_day.setdefault(int(r["day"]), []).append(r["arbitrage_margin_cny"])
    per_day = {d: float(np.median(v)) for d, v in by_day.items()}

    n_days = len(per_day)
    passing = _passing_days(per_day)
    n_pass = len(passing)
    sufficient = (n_days > 0 and 2 * n_pass >= n_days)

    if n_days == 0:
        # No days is not evidence of insufficiency. Saying INSUFFICIENT here
        # would turn an ensemble too small to answer into an answer.
        token = "NATIVE_MARKET_OPPORTUNITY_UNDETERMINED"
        summary = (f"no days fell in partition {partition!r}; the ensemble did "
                   f"not answer the question")
    else:
        token = ("NATIVE_MARKET_OPPORTUNITY_SUFFICIENT" if sufficient
                 else "NATIVE_MARKET_OPPORTUNITY_INSUFFICIENT")
        summary = (
            f"{n_pass} of {n_days} days clear a positive median margin against "
            f"prices from a clearing containing no batteries. This is an upper "
            f"bound on what any policy could earn, not an attainable one.")

    return {
        "partition": partition or "all_days",
        "bar": ("median bus margin > 0 on at least half the days"),
        "n_days": n_days,
        "n_days_passing": n_pass,
        "passing_days": sorted(passing),
        "per_day_median_margin_cny": {str(d): per_day[d] for d in sorted(per_day)},
        "verdict": token,
        "reading": "single_unit_fleet_absent_upper_bound",
        "summary": summary,
    }


def strategic_space(records: Sequence[dict], buses: Sequence[int],
                    partition: Optional[str] = "test") -> dict:
    """Whether a bidding policy has room to declare in, by the siting rule.

    A third reading, and the one that speaks to a training run. The margin bar
    above asks whether the unit earns anything; this asks whether there is
    anything to *choose*. They come apart badly here: a margin that clears zero
    while leaving a few percent of the quote box tradeable is a market where the
    optimum is essentially forced, so a policy has almost no action left to
    learn -- which is what "the strategic-bidding signal is under-excited"
    means, stated as a measurement.

    The criterion is not invented here. `tradeable_box_share` and its 25%
    threshold are `diagnose_typical_day`'s siting rule, whose own comment says
    the point is for a unit to have "a strategy space and not merely a positive
    profit".
    """
    rows = [r for r in records
            if partition is None or r["partition"] == partition]
    if not rows:
        return {"available": False}
    bus_set = {int(b) for b in buses}
    fleet_rows = [r for r in rows if int(r["bus"]) in bus_set] or rows
    shares = np.array([r["tradeable_box_share"] for r in rows], dtype=float)
    fleet_shares = np.array([r["tradeable_box_share"] for r in fleet_rows],
                            dtype=float)
    median_fleet = float(np.median(fleet_shares))
    sufficient = bool(median_fleet >= TRADEABLE_BOX_SHARE)
    return {
        "available": True,
        "criterion": f"tradeable box share >= {TRADEABLE_BOX_SHARE}",
        "criterion_source": "diagnose_typical_day.TRADEABLE_BOX_SHARE, the "
                            "siting rule that decides what a usable site is",
        "n_bus_days": len(rows),
        "median_box_share_all_buses": float(np.median(shares)),
        "max_box_share_all_buses": float(np.max(shares)),
        "median_box_share_fleet": median_fleet,
        "max_box_share_fleet": float(np.max(fleet_shares)),
        "eligible_rate_all_buses": float(
            np.mean([bool(r["siting_eligible"]) for r in rows])),
        "eligible_rate_fleet": float(
            np.mean([bool(r["siting_eligible"]) for r in fleet_rows])),
        "verdict_box": (
            "NATIVE_MARKET_OPPORTUNITY_SUFFICIENT" if sufficient
            else "NATIVE_MARKET_OPPORTUNITY_INSUFFICIENT"),
        "summary": (
            f"the fleet's median tradeable box share is {median_fleet:.3f} "
            f"against a {TRADEABLE_BOX_SHARE} threshold "
            f"(highest anywhere {np.max(shares):.3f}); a positive margin here "
            f"leaves a policy almost no action to choose between"),
    }


def attainability(records: Sequence[dict],
                  records_with_fleet: Sequence[dict],
                  partition: str = "test") -> dict:
    """What the fleet keeps of the bound once its own footprint is in the prices.

    The bound above is taken in a market with no batteries. The fleet is four
    batteries, and they move the prices they trade against, so the honest
    second reading is the same bound taken against a field that already
    contains them. The ratio is the share of the opportunity the fleet does not
    destroy by taking it.
    """
    def by_day_bus(rs):
        return {(int(r["day"]), int(r["bus"])): float(r["arbitrage_margin_cny"])
                for r in rs if r["partition"] == partition}

    absent = by_day_bus(records)
    present = by_day_bus(records_with_fleet)
    shared = sorted(set(absent) & set(present))
    if not shared:
        return {"available": False}

    a = np.array([absent[k] for k in shared], dtype=float)
    p = np.array([present[k] for k in shared], dtype=float)

    per_day_absent: Dict[int, List[float]] = {}
    per_day_present: Dict[int, List[float]] = {}
    for (day, _bus), av, pv in zip(shared, a, p):
        per_day_absent.setdefault(day, []).append(av)
        per_day_present.setdefault(day, []).append(pv)

    median_absent = {d: float(np.median(v)) for d, v in per_day_absent.items()}
    median_present = {d: float(np.median(v)) for d, v in per_day_present.items()}
    n_days = len(median_absent)
    n_pass = len(_passing_days(median_present))
    sufficient = (n_days > 0 and 2 * n_pass >= n_days)

    retention = (float(np.sum(p) / np.sum(a)) if float(np.sum(a)) > 0
                 else float("nan"))
    return {
        "available": True,
        "n_days": n_days,
        "n_days_passing": n_pass,
        "retention_ratio": retention,
        "mean_absent_cny": float(np.mean(a)),
        "mean_present_cny": float(np.mean(p)),
        "verdict_attainable": (
            "NATIVE_MARKET_OPPORTUNITY_SUFFICIENT" if sufficient
            else "NATIVE_MARKET_OPPORTUNITY_INSUFFICIENT"),
        "summary": (
            f"with the fleet in the clearing the same bound is met at "
            f"{100.0 * retention:.1f}% of its fleet-absent value; "
            f"{n_pass} of {n_days} days still clear a positive median margin"),
    }


# ---------------------------------------------------------------------------
# Diagnosis: what the numbers say the specification is
#
# Findings, not recommendations. Nothing in this section writes a config value,
# and every entry it produces is labelled as unadopted.
# ---------------------------------------------------------------------------

def price_map_audit(days: Sequence[EnsembleDay]) -> dict:
    """The price map's own shape: reachable range, slope, and dead parameters.

    Reported because whether the map *can* offer a spread is a different
    question from whether this unit can reach it, and both maps answer the
    first one analytically.

    The structural facts come from the provider rather than from a hard-coded
    set of `price_curve` keys, so this audit describes the price it actually
    measured. It used to read the retired map's constants directly, which was
    correct only while that map was the only one -- once the native scenario
    moved to the external market, those keys stopped driving the price and the
    report would have gone on describing a market it no longer priced.
    """
    from wholesale_price import get_provider

    provider = get_provider()
    described = provider.describe()
    per_day = provider.day_structure_for(days)

    floor = float(described["declared_floor_cny_mwh"])
    cap = float(described["declared_cap_cny_mwh"])
    reach_lo = min(p["reachable_price_min_cny_mwh"] for p in per_day)
    reach_hi = max(p["reachable_price_max_cny_mwh"] for p in per_day)
    key = ("slope_cny_mwh_per_mw" if "slope_cny_mwh_per_mw" in per_day[0]
           else "slope_cny_mwh_per_input")
    out = dict(described)
    out.update({
        "reachable_price_min_cny_mwh": reach_lo,
        "reachable_price_max_cny_mwh": reach_hi,
        # The widest peak-to-valley the map can offer across the ensemble. The
        # unit has to clear the round-trip floor against this, not against the
        # spread at one bus on one day.
        "reachable_ratio": (reach_hi / reach_lo if reach_lo > 0
                            else float("inf")),
        # Reachable means the map can produce a price at or beyond the bound,
        # so a bound that never binds is a parameter doing no work -- which is
        # what the retired map's floor and cap both were.
        "floor_reachable": bool(reach_lo <= floor),
        "cap_reachable": bool(reach_hi >= cap),
        "slope_cny_mwh_per_mw_median": float(np.median(
            [float(p[key]) for p in per_day])),
        "per_day": per_day,
        "adopted": False,
    })
    return out


def fleet_footprint(absent_lmp: np.ndarray, present_lmp: np.ndarray,
                    bus_indices: Sequence[int], buses: Sequence[int]) -> dict:
    """How much of the price field the fleet itself moves.

    The mean is not the quantity that matters and is reported to make that
    clear: the LP spends the day's *range*, and the fleet can move the mean by a
    fraction of a CNY/MWh while taking a much larger share off the range.
    """
    index = {int(b): i for i, b in enumerate(bus_indices)}
    rows = []
    for bus in buses:
        bus = int(bus)
        if bus not in index:
            continue
        i = index[bus]
        a = np.asarray(absent_lmp[:, i], dtype=float)
        p = np.asarray(present_lmp[:, i], dtype=float)
        ra = float(a.max() - a.min())
        rp = float(p.max() - p.min())
        rows.append({
            "bus": bus,
            "range_absent_cny_mwh": ra,
            "range_present_cny_mwh": rp,
            "range_retained": (rp / ra if ra > 0 else float("nan")),
            "mean_shift_cny_mwh": float(np.mean(p) - np.mean(a)),
            "mean_abs_shift_cny_mwh": float(np.mean(np.abs(p - a))),
        })
    ranges = [r["range_retained"] for r in rows if np.isfinite(r["range_retained"])]
    return {
        "per_bus": rows,
        "mean_range_retained": float(np.mean(ranges)) if ranges else float("nan"),
        "mean_shift_across_buses_cny_mwh": float(
            np.mean([r["mean_shift_cny_mwh"] for r in rows])) if rows else float("nan"),
        "adopted": False,
    }


def degradation_audit(unit: UnitSpec, cycle_cost: float,
                      sensitivities: Sequence[float] = DEGRADATION_SENSITIVITIES
                      ) -> dict:
    """The round-trip floor under each convention, at each cycle cost.

    Reported as a sensitivity and never adopted. The conventions are stated
    explicitly because they differ by up to a third and the prose elsewhere
    quotes a figure that is one of them: the cost is charged on *both* legs, so
    a per-MWh-of-throughput reading and a per-MWh-stored reading are different
    numbers, and a reader given one of them without its convention cannot tell
    which floor they are looking at.
    """
    def conventions(cc: float) -> dict:
        return {
            "cost_per_leg_cny_mwh": cc,
            # Two legs per stored MWh: the convention the code uses.
            "per_mwh_stored_cny_mwh": 2.0 * cc,
            # A MWh charged puts eta_ch into the battery, which delivers
            # eta_ch*eta_dis; the cost rides on both legs of that MWh.
            "per_mwh_charged_cny_mwh": cc * (1.0 + unit.eta_ch * unit.eta_dis),
            "per_mwh_delivered_cny_mwh": cc * (1.0 / (unit.eta_ch * unit.eta_dis) + 1.0),
        }

    return {
        "declared_cycle_cost_cny_mwh": cycle_cost,
        "eta_ch": unit.eta_ch,
        "eta_dis": unit.eta_dis,
        "charged_on_both_legs": True,
        "at_declared": conventions(cycle_cost),
        "sensitivity": {str(s): conventions(float(s)) for s in sensitivities},
        "adopted": False,
    }


def change_requirements(price: dict, degradation: dict,
                        fleet: Optional[dict]) -> List[dict]:
    """What would have to hold for this scale to carry strategic bidding.

    Each entry is a *finding* with the measured number that supports it, and
    carries `adopted: False`. Nothing here proposes a value: choosing one would
    be tuning prices to a target profit, which is exactly what the audit exists
    not to do.
    """
    out = [{
        "finding": "the day-ahead map is affine in net load over its whole "
                   "reachable range, so the floor and the cap never bind",
        "measured": {
            "reachable_price_min_cny_mwh": price["reachable_price_min_cny_mwh"],
            "reachable_price_max_cny_mwh": price["reachable_price_max_cny_mwh"],
            "declared_floor_cny_mwh": price["declared_floor_cny_mwh"],
            "declared_cap_cny_mwh": price["declared_cap_cny_mwh"],
        },
        "would_change": "a supply stack whose marginal unit changes, so that "
                        "price rises faster than load near the peak",
        "adopted": False,
    }, {
        "finding": "the curve responds to net load at a fixed money-per-MW "
                   "slope, so a fleet sized against the feeder's own peak "
                   "moves the prices it trades on",
        "measured": {
            "slope_cny_mwh_per_mw": price["slope_cny_mwh_per_mw_median"],
            "fleet_power_mw": None,
        },
        "would_change": "a fleet small relative to the market, or a price "
                        "process the fleet cannot move",
        "adopted": False,
    }, {
        "finding": "winning the spread consumes almost the whole quote box, so "
                   "a positive margin leaves a policy almost no action to "
                   "choose between",
        "measured": {"tradeable_box_share_criterion": TRADEABLE_BOX_SHARE},
        "would_change": "a wider spread relative to the declared quote box, or "
                        "a narrower box, which is a market-design question "
                        "rather than an ESS-size one",
        "adopted": False,
    }, {
        "finding": "the round trip is charged on both legs, so the floor is "
                   "twice the declared per-leg cost before conversion losses",
        "measured": degradation["at_declared"],
        "would_change": "a degradation cost stated per MWh of throughput "
                        "rather than per leg, which is a different quantity "
                        "by a factor of two",
        "adopted": False,
    }, {
        "finding": "one valley and one peak per day, so the unit turns its "
                   "usable window over once at most",
        "measured": {"reachable_peak_to_valley_ratio": price["reachable_ratio"],
                     "reachable_price_min_cny_mwh":
                         price["reachable_price_min_cny_mwh"],
                     "reachable_price_max_cny_mwh":
                         price["reachable_price_max_cny_mwh"]},
        "would_change": "a price process with more than one daily cycle, or "
                        "value in a service other than energy arbitrage",
        "adopted": False,
    }]
    if fleet:
        out[1]["measured"]["fleet_power_mw"] = None
        out.append({
            "finding": "the fleet's own footprint takes a measurable share off "
                       "the price range it is trying to capture",
            "measured": {
                "mean_range_retained": fleet.get("mean_range_retained"),
                "mean_shift_across_buses_cny_mwh":
                    fleet.get("mean_shift_across_buses_cny_mwh"),
            },
            "would_change": "fewer or smaller units, which is the sizing "
                            "question rather than a price question",
            "adopted": False,
        })
    return out


# ---------------------------------------------------------------------------
# The series source
# ---------------------------------------------------------------------------

def load_price_series(path: str, T: int = money.PERIODS_PER_DAY,
                      column: Optional[str] = None,
                      resample: Optional[str] = None) -> List[Tuple[str, np.ndarray]]:
    """Read an externally supplied day-ahead price series.

    Returns one `(label, price)` pair per day in the file. A file whose row
    count is not a whole number of days is refused unless resampling is asked
    for, because guessing where a day ends would put a boundary in the data that
    is the reader's invention rather than the source's.
    """
    if not os.path.isfile(path):
        raise ValueError(f"no price series at {path!r}")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"price series {path!r} is empty")

    header = [h.strip() for h in rows[0]]
    numeric = []
    for h in header:
        try:
            float(h)
        except ValueError:
            numeric.append(h)
        else:
            numeric = []
            break
    if not numeric:
        raise ValueError(
            f"price series {path!r} has no header row naming a column; "
            f"got {header!r}")

    if column is None:
        if len(numeric) != 1:
            raise ValueError(
                f"price series {path!r} has columns {numeric!r}; "
                f"name one with --series-column")
        column = numeric[0]
    if column not in header:
        raise ValueError(
            f"price series {path!r} has no column {column!r}; "
            f"got {header!r}")
    col = header.index(column)
    label_col = 0 if header[0] != column else None

    labels, values = [], []
    for lineno, row in enumerate(rows[1:], start=2):
        if not row or all(not c.strip() for c in row):
            continue
        if col >= len(row):
            raise ValueError(
                f"price series {path!r} line {lineno} is short of column "
                f"{column!r}")
        try:
            values.append(float(row[col]))
        except ValueError:
            raise ValueError(
                f"price series {path!r} line {lineno} has a non-numeric value "
                f"{row[col]!r}") from None
        labels.append(row[label_col] if label_col is not None and label_col < len(row)
                      else str(lineno))
    if not values:
        raise ValueError(f"price series {path!r} has no data rows")

    series = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(series)):
        raise ValueError(f"price series {path!r} carries a non-finite value")

    if series.size % T == 0:
        days = series.size // T
        return [(f"{os.path.basename(path)}#{d}",
                 series[d * T:(d + 1) * T]) for d in range(days)]
    if resample != "linear":
        raise ValueError(
            f"price series {path!r} has {series.size} rows, which is not a "
            f"whole number of {T}-period days; pass --series-resample linear "
            f"to interpolate it onto whole days")
    n_days = max(1, int(round(series.size / T)))
    grid = np.linspace(0.0, 1.0, series.size)
    target = np.linspace(0.0, 1.0, n_days * T)
    out = np.interp(target, grid, series)
    return [(f"{os.path.basename(path)}#{d}", out[d * T:(d + 1) * T])
            for d in range(n_days)]


def series_specs(path: str, seed: int, T: int = money.PERIODS_PER_DAY,
                 column: Optional[str] = None,
                 resample: Optional[str] = None) -> List[DaySpec]:
    """One spec per day of an external series, carrying its own prices.

    The shape and seed levers are left at the base day's values: an external
    series is a *price* input, and varying the physics underneath it would make
    the two sources incomparable.
    """
    base = ensemble_specs(1, seed)[0]
    specs = []
    for i, (label, price) in enumerate(load_price_series(path, T, column, resample)):
        specs.append(replace(base, day=i, source="series",
                             series_label=label, series_price=price))
    return specs


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(days: int, seed: int,
                 source: str = "synthetic", partition: str = "test",
                 buses: Optional[Sequence[int]] = None,
                 with_fleet: bool = False, fleet_footprint_day: Optional[int] = 0,
                 series_path: Optional[str] = None,
                 series_column: Optional[str] = None,
                 series_resample: Optional[str] = None,
                 degradation_sensitivity: Sequence[float] = DEGRADATION_SENSITIVITIES,
                 quiet: bool = False,
                 T: int = money.PERIODS_PER_DAY) -> dict:
    """Run the audit and return everything the report and the verdict read.

    The two passes are the design, not an implementation detail: build and
    label first, clear and measure second. Anything computed in pass two can
    only be joined to a partition that already exists.
    """
    frozen_before = {k: get_default(k) for k in FROZEN_KEYS}
    # Imported here rather than at module scope for the same reason the price
    # layer is imported lazily below: it pulls a native library in, and this
    # module is loaded on paths that never price a day.
    from wholesale_price import reset_cache as reset_price_cache
    # A previous run in this process -- one that raised, so its own restore
    # never ran -- may have left a provider built under its levers. Dropping the
    # cache here means this audit cannot be priced by it.
    reset_price_cache()
    report: dict = {
        "source": source,
        "declared": {
            "ensemble_size": days,
            "master_seed": seed,
            "split_fractions": list(SPLIT_FRACTIONS),
            "margin_thresholds": list(MARGIN_THRESHOLDS),
            "shed_limit_mwh": SHED_LIMIT_MWH,
            "frozen_keys": {k: frozen_before[k] for k in FROZEN_KEYS},
        },
    }

    if source == "series":
        if not series_path:
            raise ValueError("--source series needs --series")
        specs = series_specs(series_path, seed, T, series_column,
                             series_resample)
    else:
        specs = ensemble_specs(days, seed)

    # ---- pass one: physics, no clearing --------------------------------
    # The labels and the partition are fixed here and never revisited. Nothing
    # below can move a day between partitions, which is what makes the split
    # independent of the economics it will be reported against.
    ensemble = [build_day(s, T) for s in specs]
    ensemble = [d for d in ensemble if d.spec.source == "synthetic"
                or d.spec.series_price is not None]
    assign_partitions(ensemble)

    # ---- pass two: prices and the bound ---------------------------------
    records: List[dict] = []
    fleet_records: List[dict] = []
    footprint = None
    for d in ensemble:
        if d.spec.source == "series":
            price = np.asarray(d.spec.series_price, dtype=float)
            # One system price mapped onto every bus: the only honest mapping
            # when the source is not nodal. The report says the per-bus table
            # is degenerate rather than letting 33 identical rows read as
            # evidence.
            d.lmp = np.tile(price[:, None], (1, 33))
            d.bus_indices = list(range(33))
            d.price_source = "external_series_uniform"
            d.lmp_fallbacks = 0
            d.lmp_fallback_periods = []
            d.lmp_fallback_reason = ""
            d.lmp_price_source = "external_series_uniform"
            d.shed_mwh = 0.0
            d.clean = True
            d.notes.append("system price mapped uniformly onto every bus; "
                           "per-bus rows are identical by construction")
        else:
            cleared = clear_day(d.spec, with_fleet=False, T=T)
            d.lmp = cleared["lmp"]
            d.bus_indices = cleared["bus_indices"]
            d.price_source = "socp_fleet_absent"
            d.lmp_fallbacks = cleared["lmp_fallbacks"]
            d.lmp_fallback_periods = cleared["lmp_fallback_periods"]
            d.lmp_fallback_reason = cleared["lmp_fallback_reason"]
            d.lmp_price_source = cleared["lmp_price_source"]
            d.shed_mwh = cleared["shed_mwh"]
            d.notes.append(f"curtailment {cleared['curtailment_mwh']:.4f} MWh")
            d.clean = bool(cleared["lmp_fallbacks"] == 0
                           and cleared["shed_mwh"] <= SHED_LIMIT_MWH)
            if not d.clean:
                d.notes.append(
                    f"dropped: fallbacks {cleared['lmp_fallbacks']}, "
                    f"shed {cleared['shed_mwh']:.4f} MWh")
            if cleared["lmp_fallback_periods"]:
                d.notes.append(
                    f"price field is {cleared['lmp_price_source']} over "
                    f"{len(cleared['lmp_fallback_periods'])} periods: "
                    f"{cleared['lmp_fallback_reason']}")
            if with_fleet:
                fc = clear_day(d.spec, with_fleet=True, T=T)
                d.lmp_with_fleet = fc["lmp"]

        # Dropped days are counted, never averaged in: a day that shed or
        # degraded is not a clean opportunity day and quietly including it
        # would let a biased surviving subset stand in for the ensemble.
        if d.clean:
            records.extend(day_records(d, buses, T))
        if with_fleet and d.lmp_with_fleet is not None and d.clean:
            proxy = EnsembleDay(spec=d.spec, label=d.label,
                                partition=d.partition, lmp=d.lmp_with_fleet,
                                bus_indices=d.bus_indices, clean=True)
            fleet_records.extend(day_records(proxy, buses, T))

        if (fleet_footprint_day is not None and d.day == fleet_footprint_day
                and d.clean):
            # Reuse the fleet arm's clear when it ran; otherwise take one here.
            # The two flags ask different questions of the same comparison, and
            # clearing a third time would only make it a different day.
            present = (d.lmp_with_fleet if d.lmp_with_fleet is not None
                       else clear_day(d.spec, with_fleet=True, T=T)["lmp"])
            footprint = fleet_footprint(d.lmp, present, d.bus_indices,
                                        fleet_buses())

    # Assembled after pass two, so a day's price source, its drop reason and
    # its curtailment travel with it. Building this block from the pass-one
    # state would publish every day as clean and unsourced.
    report["ensemble"] = {
        "n_days": len(ensemble),
        "shapes": sorted({d.spec.shape_name for d in ensemble}),
        "partitions": {p: sorted(d.day for d in ensemble if d.partition == p)
                       for p in SPLIT_NAMES},
        "test_underfilled": underfilled_strata(ensemble, {
            p: [d.day for d in ensemble if d.partition == p]
            for p in SPLIT_NAMES}),
        "peak_invariance": {
            "real_peak_mw_min": float(min(d.physics["real_peak_mw"]
                                          for d in ensemble)),
            "real_peak_mw_max": float(max(d.physics["real_peak_mw"]
                                          for d in ensemble)),
            "installed_pv_mw": sorted({round(d.physics["installed_pv_mw"], 9)
                                       for d in ensemble}),
            "installed_wind_mw": sorted({round(d.physics["installed_wind_mw"], 9)
                                         for d in ensemble}),
        },
        # The notes carry the per-day caveats a reader needs to read a day
        # correctly -- a dropped day's reason, an external series mapped
        # uniformly onto every bus. They travel with the day rather than being
        # printed once, because a report is read one row at a time.
        "physics": [{"day": d.day, "partition": d.partition,
                     "stratum": d.label.stratum,
                     "price_source": d.price_source, "clean": d.clean,
                     "shed_mwh": d.shed_mwh,
                     "lmp_fallbacks": d.lmp_fallbacks,
                     "lmp_fallback_periods": list(d.lmp_fallback_periods),
                     "lmp_fallback_reason": d.lmp_fallback_reason,
                     "notes": list(d.notes), **d.physics} for d in ensemble],
    }

    report["records"] = records
    report["n_days_dropped"] = sum(1 for d in ensemble if not d.clean)
    report["drop_share"] = (report["n_days_dropped"] / len(ensemble)
                            if ensemble else 1.0)

    report["per_bus"] = bus_distribution(records, partition)
    report["per_bus_all_days"] = bus_distribution(records, None)
    report["fleet"] = fleet_distribution(records, fleet_buses(), partition)
    report["verdict"] = verdict(records, partition)
    report["verdict_all_days"] = verdict(records, None)
    report["strategic_space"] = strategic_space(records, fleet_buses(), partition)

    if report["drop_share"] > MAX_DROP_SHARE:
        report["verdict"]["verdict"] = "NATIVE_MARKET_OPPORTUNITY_UNDETERMINED"
        report["verdict"]["summary"] = (
            f"{report['n_days_dropped']} of {len(ensemble)} days were dropped "
            f"(>{MAX_DROP_SHARE:.0%}); the ensemble did not answer the question")

    if with_fleet and fleet_records:
        report["attainability"] = attainability(records, fleet_records, partition)
        if report["attainability"].get("available"):
            report["verdict_attainable"] = report["attainability"]["verdict_attainable"]
    if footprint:
        report["fleet_footprint"] = footprint

    unit = audited_unit(fleet_buses()[0], T) if fleet_buses() else None
    report["diagnosis"] = {
        "price_map": price_map_audit(ensemble),
        "degradation": (degradation_audit(unit, float(get_default("storage.cycle_cost")),
                                          degradation_sensitivity)
                        if unit else None),
    }
    report["diagnosis"]["change_requirements"] = change_requirements(
        report["diagnosis"]["price_map"], report["diagnosis"]["degradation"],
        report.get("fleet_footprint"))
    report["diagnosis"]["note"] = (
        "Every entry is a finding with its measured evidence and carries "
        "adopted: False. No price, elasticity or degradation constant is "
        "written by this module.")

    # The levers live in the process-global config cache, so the last day's
    # values would otherwise outlive the audit and silently become the settings
    # the next run in this process starts from. Reset last as well as first, so
    # the audit leaves the caches holding what is on disk.
    reload_defaults()
    reload_scenarios()
    # The providers are cached by the levers they were built under, so restoring
    # the config is not enough on its own: a provider built while this audit had
    # its levers set would go on pricing the next run in this process.
    reset_price_cache()
    assert {k: get_default(k) for k in FROZEN_KEYS} == frozen_before, \
        "the audit must not leave a price or degradation constant changed"
    return report


def write_csv(records: Sequence[dict], path: str) -> None:
    """One row per day and bus, in the declared column order."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        for r in records:
            writer.writerow({c: r.get(c) for c in REPORT_COLUMNS})


def write_json(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def render(report: dict, limit: int = 12) -> None:
    """Print the table the audit exists to produce, then the verdict."""
    records = report["records"]
    print(f"days {report['ensemble']['n_days']}  "
          f"kept {len({r['day'] for r in records})}  "
          f"dropped {report['n_days_dropped']}")
    inv = report["ensemble"]["peak_invariance"]
    print(f"frozen scale: real peak {inv['real_peak_mw_min']:.6f}"
          f"..{inv['real_peak_mw_max']:.6f} MW  "
          f"PV {inv['installed_pv_mw']} MW  wind {inv['installed_wind_mw']} MW")
    print("partitions: " + "  ".join(
        f"{p}={len(v)}" for p, v in report["ensemble"]["partitions"].items()))
    if report["ensemble"]["test_underfilled"]:
        print(f"test set does not cover: {report['ensemble']['test_underfilled']}")

    print()
    print(f"{'day':>4} {'bus':>4} {'LMP range':>17} {'ch#':>4} {'dis#':>4} "
          f"{'gross':>8} {'deg-adj':>8} {'ch MWh':>7} {'dis MWh':>8} "
          f"{'margin':>8}")
    for r in records[:limit]:
        print(f"{r['day']:>4} {r['bus']:>4} "
              f"{r['lmp_min_cny_mwh']:>8.1f}{r['lmp_max_cny_mwh']:>9.1f} "
              f"{r['feasible_charge_periods']:>4} "
              f"{r['feasible_discharge_periods']:>4} "
              f"{r['gross_spread_cny_mwh']:>8.1f} "
              f"{r['degradation_adjusted_spread_cny_mwh']:>8.1f} "
              f"{r['charge_mwh']:>7.4f} {r['discharge_mwh']:>8.4f} "
              f"{r['arbitrage_margin_cny']:>8.2f}")
    if len(records) > limit:
        print(f"... {len(records) - limit} further rows in the CSV")

    print()
    print(f"{'bus':>4} {'med adj':>8} {'P(>0)':>7} {'P(>50)':>7} {'P(>100)':>8} "
          f"{'med CNY':>9} {'P(>0)':>7} {'box':>6} {'elig':>5} {'n':>4}")
    for b in report["per_bus"]:
        s, m = b["adjusted_spread_cny_mwh"], b["arbitrage_margin_cny"]
        box = b["tradeable_box_share"]["median"]
        print(f"{b['bus']:>4} {s['median']:>8.1f} {s['p_gt_0']:>7.2f} "
              f"{s['p_gt_50']:>7.2f} {s['p_gt_100']:>8.2f} "
              f"{m['median']:>9.2f} {m['p_gt_0']:>7.2f} {box:>6.3f} "
              f"{b['siting_eligible_rate']:>5.2f} {b['n_days']:>4}")

    fl = report["fleet"]
    print()
    print(f"fleet buses {fl['buses']} over {fl['n_days']} days")
    print(f"  median bus margin   {fl['median_bus_margin_cny']['median']:.2f} CNY/day")
    print(f"  best bus margin     {fl['best_bus_margin_cny']['median']:.2f} CNY/day")
    print(f"  isolated upper bd   "
          f"{fl['isolated_units_upper_bound_cny']['median']:.2f} CNY/day")
    print(f"  note: {fl['scope_note']}")

    v = report["verdict"]
    print()
    print(f"VERDICT ({v['partition']}, bar: {v['bar']})")
    print(f"  {v['verdict']}")
    print(f"  {v['summary']}")
    if "attainability" in report and report["attainability"].get("available"):
        a = report["attainability"]
        print(f"  attainable: {a['verdict_attainable']}")
        print(f"    {a['summary']}")
        print("    (the same bar against prices the fleet is already in)")
    ss = report.get("strategic_space")
    if ss and ss.get("available"):
        print(f"  strategy space: {ss['verdict_box']}")
        print(f"    {ss['summary']}")
        print(f"    criterion: {ss['criterion']} "
              f"[{ss['criterion_source'].split(',')[0]}]")
    if "fleet_footprint" in report:
        fp = report["fleet_footprint"]
        print(f"  fleet footprint: mean range retained "
              f"{fp['mean_range_retained']:.3f}, "
              f"mean shift {fp['mean_shift_across_buses_cny_mwh']:+.3f} CNY/MWh")


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, as an object a test can inspect.

    Separated from `main` so that "no option can move a frozen constant" is
    checkable against the parser rather than by grepping the source, which would
    pass on an option named something the grep did not anticipate.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="synthetic",
                        choices=("synthetic", "series"))
    parser.add_argument("--days", type=int, required=True,
                        help="how many days the ensemble has; no default, "
                             "because the length belongs to the data")
    parser.add_argument("--ensemble-seed", type=int, required=True,
                        help="fixes which days the ensemble contains")
    parser.add_argument("--partition", default="test", choices=list(SPLIT_NAMES))
    parser.add_argument("--buses", default=None,
                        help="comma-separated bus numbers; default all")
    parser.add_argument("--with-fleet", action="store_true",
                        help="clear each day a second time with the fleet and "
                             "report what it keeps of the bound")
    parser.add_argument("--fleet-footprint", action="store_true",
                        help="measure how much of the price range the fleet "
                             "itself moves, on one reference day")
    parser.add_argument("--series", default=None)
    parser.add_argument("--series-column", default=None)
    parser.add_argument("--series-resample", default=None, choices=(None, "linear"))
    parser.add_argument("--degradation-sensitivity", default=None,
                        help="comma-separated CNY/MWh per leg; reported only")
    parser.add_argument("--json", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    buses = ([int(b) for b in args.buses.split(",") if b.strip()]
             if args.buses else None)
    sens = (tuple(float(s) for s in args.degradation_sensitivity.split(","))
            if args.degradation_sensitivity else DEGRADATION_SENSITIVITIES)

    if args.source == "series" and not args.series:
        parser.error("--source series needs --series PATH")

    report = build_report(
        days=args.days, seed=args.ensemble_seed, source=args.source,
        partition=args.partition, buses=buses, with_fleet=args.with_fleet,
        fleet_footprint_day=0 if args.fleet_footprint else None,
        series_path=args.series, series_column=args.series_column,
        series_resample=args.series_resample,
        degradation_sensitivity=sens, quiet=args.quiet)

    if args.json:
        write_json(report, args.json)
    if args.csv:
        write_csv(report["records"], args.csv)
    if not args.quiet:
        if not report["records"]:
            print(f"no clean days survived in {report['ensemble']['n_days']} "
                  f"days; the ensemble did not answer the question")
        else:
            render(report)

    if not report["records"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
