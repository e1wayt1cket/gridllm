# wholesale_price.py
"""The price at the PCC, from a market the distribution network cannot move.

`wholesale[t]` is the price of energy at the point of common coupling: what the
DSO pays to import and earns to export. It is NOT the nodal price inside the
feeder -- that is `lmp[t, i]`, recovered from the power-balance duals of the
network-constrained clearing. The two are different quantities at different
levels, and conflating them collapses the whole settlement.

What this module replaces is the *source* of the first one. The retired map
built it from the distribution network's own net load, which made the price a
function of the very dispatch it was pricing: a battery could move the number it
was bidding against. It also had no supply-side structure at all -- a single
affine response from the cheapest hour to the dearest, with a declared floor and
cap that no day could ever reach.

Here the price comes from an **external** market with its own demand, its own
renewables and its own supply stack. Nothing in this file reads an Agent, a
network or a clearing result. That is the property that makes it a price rather
than an accounting identity, and it is enforced by test.

    external demand, PV, wind   ->  net load R_t
    R_t against the library's own range  ->  r_t in [0, 1]
    continuous piecewise supply stack  ->  marginal cost
    noise + rare scarcity  ->  lambda_up(t), clipped

`r` is normalized across the WHOLE library rather than per day, so a mild day
sits low on the stack and a severe day climbs it. The consequence, which is real
and is reported rather than hidden: every price is a function of how many days
the library contains, because the yardstick is the library's own net-load range.
The manifest records the range so a library built against a different one can be
refused instead of silently describing different prices.

`get_day_ahead_price` takes a `t_start` as well as a length. A caller clearing a
rolling window needs the window's own periods, and an agent-independent source
cannot infer which periods those are from the agent list the way the retired map
could -- its windows were aligned only because the window's own agents carried
the window's own net load. Without `t_start` every block after the first would be
priced off the first few periods of the day.

Usage:
    PYTHONPATH=src python src/wholesale_price.py --build-library
    PYTHONPATH=src python src/wholesale_price.py --stats
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config_loader import get_default

#: The config block that selects and parameterizes the source.
CONFIG_KEY = "wholesale_market"

#: Sources this module can serve. `legacy` is the retired net-load map, kept
#: reachable so recorded results can still be reproduced; `guangdong` is the
#: phase-two slot and is declared so the interface is fixed before the data
#: arrives.
SOURCES = ("legacy", "synthetic_v1", "library", "guangdong")

#: Periods in a library day. The library is built at 15-minute resolution and a
#: shorter `T` is a slice of it, never a coarser rebuild.
LIBRARY_PERIODS = 96

#: Offset between the stream that draws a day's *parameters* and the stream that
#: draws its noise. Separate streams so that editing one day's parameter draw
#: cannot silently shift its noise, and offset far enough that the two seeds
#: cannot collide for any plausible library.
NOISE_STREAM_OFFSET = 1_000_000


def _deep_get(cfg: dict, path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# The supply stack
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SupplyStack:
    """A continuous piecewise-linear marginal-cost curve over `r` in [0, 1].

    A table of (knot, price) pairs does NOT determine this curve, because the
    five segments have five different slopes. The slopes are carried explicitly
    rather than re-derived from the knots, so that reading this object tells you
    what the curve is without arithmetic, and so that a knot can be moved
    without silently moving every slope below it.

    Continuity is validated on construction. It is the curve's stated property,
    and a typo in one slope leaves a stack that is still monotone and still
    looks plausible -- the continuity check is the only thing that catches it.
    """

    knots: Tuple[float, ...]
    prices: Tuple[float, ...]
    slopes: Tuple[float, ...]

    def __post_init__(self):
        n = len(self.knots)
        if len(self.prices) != n or len(self.slopes) != n - 1:
            raise ValueError(
                f"a stack of {n} knots needs {n} prices and {n - 1} slopes; "
                f"got {len(self.prices)} prices and {len(self.slopes)} slopes")
        if self.knots[0] != 0.0 or self.knots[-1] != 1.0:
            raise ValueError(
                f"knots must span [0, 1]; got "
                f"[{self.knots[0]}, {self.knots[-1]}]")
        for a, b in zip(self.knots, self.knots[1:]):
            if b <= a:
                raise ValueError(f"knots must strictly increase; {a} -> {b}")
        for k, s in enumerate(self.slopes):
            if s <= 0:
                raise ValueError(f"slope {k} must be positive; got {s}")
        for k in range(len(self.slopes) - 1):
            if self.slopes[k + 1] < self.slopes[k]:
                raise ValueError(
                    f"slopes must not decrease: segment {k + 1} is "
                    f"{self.slopes[k + 1]} against {self.slopes[k]}. A supply "
                    f"stack's marginal cost rises as cheap capacity is used up")
        for k in range(len(self.slopes)):
            top = self.prices[k] + self.slopes[k] * (self.knots[k + 1]
                                                     - self.knots[k])
            if abs(top - self.prices[k + 1]) > 1e-9:
                raise ValueError(
                    f"the stack is discontinuous at r={self.knots[k + 1]}: "
                    f"segment {k} ends at {top!r}, segment {k + 1} starts at "
                    f"{self.prices[k + 1]!r}")

    def price(self, r) -> np.ndarray:
        """Marginal cost at each `r`, in CNY/MWh. `r` is clipped to [0, 1]."""
        r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
        out = np.full(r.shape, self.prices[-1], dtype=float)
        for k in range(len(self.slopes) - 1, -1, -1):
            out = np.where(r <= self.knots[k + 1],
                           self.prices[k] + self.slopes[k] * (r - self.knots[k]),
                           out)
        return out

    @property
    def min_price(self) -> float:
        return float(self.prices[0])

    @property
    def max_price(self) -> float:
        return float(self.prices[-1])

    def as_dict(self) -> dict:
        return {"knots": list(self.knots), "prices": list(self.prices),
                "slopes": list(self.slopes)}

    @classmethod
    def from_config(cls, cfg: dict) -> "SupplyStack":
        if not cfg:
            raise ValueError(f"no supply stack in the {CONFIG_KEY} block")
        return cls(knots=tuple(float(v) for v in cfg["knots"]),
                   prices=tuple(float(v) for v in cfg["prices"]),
                   slopes=tuple(float(v) for v in cfg["slopes"]))


# ---------------------------------------------------------------------------
# The external system
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExternalSystem:
    """One day of an external market's own demand and renewables.

    The levels are in arbitrary MW. Only the ratio of demand to PV to wind
    reaches the price, through the normalized net load, so the scale is a free
    choice -- but it is a *declared* one, and it is the only place the external
    market's size enters.

    The profile shapes come from `grid.load_profile` / `pv_profile` /
    `wind_profile`, which are generic and already seeded. Their parameters are
    read from this block and never from `profiles.*`, which belong to the
    distribution network: a price that moved when IEEE33's load moved would be
    a function of the dispatch it is pricing.
    """

    demand_peak_mw: float
    pv_capacity_mw: float
    wind_capacity_mw: float
    demand_amplitude_range: Tuple[float, float]
    pv_intensity_range: Tuple[float, float]
    wind_intensity_range: Tuple[float, float]
    peak_hour_jitter: float
    morning_peak_hour: float
    evening_peak_hour: float
    morning_peak_amplitude: float
    evening_peak_amplitude: float
    morning_peak_width: float
    evening_peak_width: float
    night_base: float
    pv_cloud_prob: float
    wind_ar_coef: float
    wind_noise_std: float
    wind_peak_hour: float

    @classmethod
    def from_config(cls, cfg: dict) -> "ExternalSystem":
        if not cfg:
            raise ValueError(f"no external_system in the {CONFIG_KEY} block")
        def pair(key, default):
            v = cfg.get(key, default)
            return (float(v[0]), float(v[1]))
        return cls(
            demand_peak_mw=float(cfg["demand_peak_mw"]),
            pv_capacity_mw=float(cfg.get("pv_capacity_mw", 0.0)),
            wind_capacity_mw=float(cfg.get("wind_capacity_mw", 0.0)),
            demand_amplitude_range=pair("demand_amplitude_range", (1.0, 1.0)),
            pv_intensity_range=pair("pv_intensity_range", (1.0, 1.0)),
            wind_intensity_range=pair("wind_intensity_range", (1.0, 1.0)),
            peak_hour_jitter=float(cfg.get("peak_hour_jitter", 0.0)),
            morning_peak_hour=float(cfg.get("morning_peak_hour", 8.0)),
            evening_peak_hour=float(cfg.get("evening_peak_hour", 19.0)),
            morning_peak_amplitude=float(cfg.get("morning_peak_amplitude", 0.30)),
            evening_peak_amplitude=float(cfg.get("evening_peak_amplitude", 0.55)),
            morning_peak_width=float(cfg.get("morning_peak_width", 3.0)),
            evening_peak_width=float(cfg.get("evening_peak_width", 4.0)),
            night_base=float(cfg.get("night_base", 0.55)),
            pv_cloud_prob=float(cfg.get("pv_cloud_prob", 0.15)),
            wind_ar_coef=float(cfg.get("wind_ar_coef", 0.90)),
            wind_noise_std=float(cfg.get("wind_noise_std", 0.15)),
            wind_peak_hour=float(cfg.get("wind_peak_hour", 3.0)),
        )

    def net_load(self, day_seed: int, T: int = LIBRARY_PERIODS) -> np.ndarray:
        """`D - PV - wind` for one day, in MW.

        The day's own parameters are drawn from `day_seed`, so two days differ
        in demand amplitude, in when their peaks fall, and in how much sun and
        wind they get -- while the installed capacities stay the external
        market's own. The profile generators are imported here rather than at
        module scope so that this module does not depend on the network stack
        being importable, and so `grid` can import *this* module lazily without
        a cycle.
        """
        from grid import load_profile, pv_profile, wind_profile

        rng = np.random.RandomState(int(day_seed))
        hours = np.arange(T)

        amplitude = self.demand_peak_mw * rng.uniform(*self.demand_amplitude_range)
        jitter = self.peak_hour_jitter
        morning = self.morning_peak_hour + rng.uniform(-jitter, jitter)
        evening = self.evening_peak_hour + rng.uniform(-jitter, jitter)
        demand = amplitude * load_profile(
            hours,
            morning_peak_hour=morning,
            morning_peak_amplitude=self.morning_peak_amplitude,
            morning_peak_width=self.morning_peak_width,
            evening_peak_hour=evening,
            evening_peak_amplitude=self.evening_peak_amplitude,
            evening_peak_width=self.evening_peak_width,
            night_base=self.night_base)

        pv = self.pv_capacity_mw * rng.uniform(*self.pv_intensity_range) * pv_profile(
            hours, cloud_prob=self.pv_cloud_prob,
            cloud_seed=int(rng.randint(1, 2 ** 31 - 1)))
        wind = self.wind_capacity_mw * rng.uniform(*self.wind_intensity_range) * wind_profile(
            hours, seed=int(rng.randint(1, 2 ** 31 - 1)),
            ar_coef=self.wind_ar_coef, noise_std=self.wind_noise_std,
            peak_hour=self.wind_peak_hour)
        return demand - pv - wind


# ---------------------------------------------------------------------------
# The providers
# ---------------------------------------------------------------------------

class WholesalePriceProvider:
    """The market's only view of the PCC price.

    Everything downstream depends on this interface and nothing downstream
    asks which implementation it holds. Adding a real market's data means
    adding a subclass; it must not mean editing a caller.
    """

    #: Short name, also the `source:` value that selects this provider.
    name = "abstract"

    def get_day_ahead_price(self, day_id: int, T: int = LIBRARY_PERIODS,
                            t_start: int = 0,
                            seed: Optional[int] = None) -> np.ndarray:
        """`wholesale` for `T` periods beginning at `t_start` of `day_id`.

        `seed` overrides the day's own draw, for a caller that wants a
        specified realization rather than the library's.
        """
        raise NotImplementedError

    def describe(self) -> dict:
        """What this price IS, structurally, so a report can say so.

        A consumer that needs to describe the price map -- an audit, a figure
        caption -- reads this instead of asking which class it holds.
        """
        raise NotImplementedError

    def day_structure_for(self, days: Sequence) -> List[dict]:
        """Per-day structure of the input the transfer function consumes.

        Each entry reports the day's own input range and the price interval it
        reaches before the clip, so an audit can aggregate across an ensemble
        without knowing how the input was produced. `days` are duck-typed:
        anything with `.day` and `.physics` works.
        """
        raise NotImplementedError


class SyntheticWholesalePriceProvider(WholesalePriceProvider):
    """A simulated external market: own net load, own supply stack.

    Deterministic in `(master_seed, number_of_days)`, so a run never depends on
    a file being present and the committed library is a record of what was
    priced rather than the only copy of it.
    """

    name = "synthetic_v1"

    def __init__(self, cfg: Optional[dict] = None):
        cfg = dict(cfg if cfg is not None else get_default(CONFIG_KEY, {}) or {})
        self.cfg = cfg
        self.master_seed = int(cfg.get("master_seed", 20260920))
        self.number_of_days = int(cfg.get("number_of_days", 40))
        self.floor = float(cfg.get("price_floor", 120.0))
        self.cap = float(cfg.get("price_cap", 1000.0))
        self.noise_sigma = float(cfg.get("noise_sigma", 15.0))
        self.scarcity_probability = float(cfg.get("scarcity_probability", 0.0))
        self.scarcity_min = float(cfg.get("scarcity_min", 150.0))
        self.scarcity_max = float(cfg.get("scarcity_max", 300.0))
        self.stack = SupplyStack.from_config(cfg.get("supply_stack", {}))
        self.external = ExternalSystem.from_config(cfg.get("external_system", {}))
        self._range_cache: Optional[Tuple[float, float]] = None

    # -- the library-wide yardstick ------------------------------------
    def library_range(self) -> Tuple[float, float]:
        """The net-load range of the whole library, in MW.

        Cached, and the reason this provider is library-scoped rather than
        day-scoped: day k's price is measured against every other day's net
        load, so the object has to know the library before it can price any one
        of its days.
        """
        if self._range_cache is None:
            lo = np.inf
            hi = -np.inf
            for day in range(self.number_of_days):
                R = self.external.net_load(self.master_seed + day,
                                           LIBRARY_PERIODS)
                lo = min(lo, float(R.min()))
                hi = max(hi, float(R.max()))
            self._range_cache = (lo, hi)
        return self._range_cache

    def day_net_load(self, day_id: int) -> np.ndarray:
        return self.external.net_load(self.master_seed + int(day_id),
                                      LIBRARY_PERIODS)

    def reduced_input(self, day_id: int) -> np.ndarray:
        """`r` for one day: its net load against the library's range, in [0, 1].

        Clipped, and the clip is a real modelling statement: a day reaching
        outside the range the yardstick was taken from is pinned at the top or
        the bottom of the stack rather than extrapolated past it.
        """
        lo, hi = self.library_range()
        if hi <= lo:
            raise ValueError(
                f"the library's net load spans nothing ({lo!r} to {hi!r}); "
                f"r is undefined")
        R = self.day_net_load(day_id)
        return np.clip((R - lo) / (hi - lo), 0.0, 1.0)

    # -- the price ------------------------------------------------------
    def day_ahead_price(self, day_id: int,
                        seed: Optional[int] = None) -> np.ndarray:
        """One whole library day of prices, length `LIBRARY_PERIODS`."""
        day_id = int(day_id)
        r = self.reduced_input(day_id)
        price = self.stack.price(r)

        # A stream of its own, so editing how a day's parameters are drawn
        # cannot shift its noise.
        stream = self.master_seed if seed is None else int(seed)
        rng = np.random.RandomState(stream + NOISE_STREAM_OFFSET + day_id)
        if self.noise_sigma > 0:
            price = price + rng.normal(0.0, self.noise_sigma, price.shape)
        if self.scarcity_probability > 0:
            # A rare tight period, not a profit generator: the marginal
            # resource switching to something dear. The probability is fixed
            # and must not be raised to make a battery's day look better.
            hit = rng.random(price.shape) < self.scarcity_probability
            if hit.any():
                price = price + hit * rng.uniform(self.scarcity_min,
                                                  self.scarcity_max,
                                                  price.shape)
        return np.clip(price, self.floor, self.cap)

    def get_day_ahead_price(self, day_id: int, T: int = LIBRARY_PERIODS,
                            t_start: int = 0,
                            seed: Optional[int] = None) -> np.ndarray:
        day = self.day_ahead_price(day_id, seed=seed)
        t_start = int(t_start)
        T = int(T)
        if t_start < 0 or T < 1:
            raise ValueError(f"a window of {T} periods from {t_start} is empty")
        if t_start + T > day.size:
            raise ValueError(
                f"a window of {T} periods from {t_start} runs past the "
                f"{day.size}-period day; a window is a slice of the day and "
                f"never wraps")
        return day[t_start:t_start + T].copy()

    # -- self-description ------------------------------------------------
    def describe(self) -> dict:
        d = {
            "source": self.name,
            "kind": "supply_stack",
            "declared_floor_cny_mwh": self.floor,
            "declared_cap_cny_mwh": self.cap,
            "noise_sigma_cny_mwh": self.noise_sigma,
            "scarcity_probability": self.scarcity_probability,
            "spike_structure_present": bool(self.scarcity_probability > 0.0),
            # The interval the stack itself reaches, before noise and before
            # the clip. A floor below this and a cap above it are reachable by
            # the noise alone; one inside it would truncate the stack.
            "stack_min_cny_mwh": self.stack.min_price,
            "stack_max_cny_mwh": self.stack.max_price,
            "supply_stack": self.stack.as_dict(),
            "number_of_days": self.number_of_days,
            "master_seed": self.master_seed,
            "external_system": {
                "demand_peak_mw": self.external.demand_peak_mw,
                "pv_capacity_mw": self.external.pv_capacity_mw,
                "wind_capacity_mw": self.external.wind_capacity_mw,
            },
        }
        lo, hi = self.library_range()
        d["library_net_load_min_mw"] = lo
        d["library_net_load_max_mw"] = hi
        return d

    @staticmethod
    def _library_day(d) -> int:
        """Which library day a member of an ensemble stands for.

        An ensemble day carries its own index and, separately, which library
        day it is priced on -- the two are not the same thing once the price
        comes from an external market rather than from the day's own physics.
        """
        spec = getattr(d, "spec", None)
        day_id = getattr(spec, "day_id", None) if spec is not None else None
        return int(day_id) if day_id is not None else int(getattr(d, "day", d))

    def day_structure_for(self, days: Sequence) -> List[dict]:
        rows = []
        for d in days:
            r = self.reduced_input(self._library_day(d))
            unclipped = self.stack.price(r)
            rows.append({
                "day": int(getattr(d, "day", d)),
                "library_day": self._library_day(d),
                "input_min": float(r.min()),
                "input_max": float(r.max()),
                "reachable_price_min_cny_mwh": float(unclipped.min()),
                "reachable_price_max_cny_mwh": float(unclipped.max()),
                "slope_cny_mwh_per_input": float(np.median(self.stack.slopes)),
                "clipped_at_floor_fraction": float(np.mean(r <= 0.0)),
                "clipped_at_cap_fraction": float(np.mean(r >= 1.0)),
            })
        return rows


class LibraryPriceProvider(SyntheticWholesalePriceProvider):
    """The committed library, read back and checked against the generator.

    Reading the file is what makes a recorded experiment re-runnable from its
    artifact alone. Checking it against the generator is what stops a stale or
    hand-edited library from being priced as though it were the declared one --
    the check is a refusal, not a warning, because a warning leaves the run
    proceeding on the wrong numbers.
    """

    name = "library"

    def __init__(self, cfg: Optional[dict] = None, path: Optional[str] = None):
        super().__init__(cfg)
        self.path = path or self.cfg.get("library_file",
                                         "data/wholesale/synthetic_da_v1.csv")
        self.manifest_path = self.cfg.get("library_manifest_file") or \
            os.path.splitext(self.path)[0] + "_manifest.csv"
        self._prices = self._load()
        self._verify()

    def _load(self) -> Dict[int, np.ndarray]:
        if not os.path.isfile(self.path):
            raise FileNotFoundError(
                f"no price library at {self.path!r}; build one with "
                f"`python src/wholesale_price.py --build-library`")
        by_day: Dict[int, List[float]] = {}
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                by_day.setdefault(int(row["day_id"]), []).append(
                    float(row["price_cny_mwh"]))
        if not by_day:
            raise ValueError(f"price library {self.path!r} has no rows")
        out = {}
        for day, values in by_day.items():
            arr = np.asarray(values, dtype=float)
            if arr.size != LIBRARY_PERIODS:
                raise ValueError(
                    f"library day {day} has {arr.size} periods, expected "
                    f"{LIBRARY_PERIODS}")
            out[day] = arr
        return out

    def _manifest(self) -> dict:
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(
                f"the library at {self.path!r} has no manifest at "
                f"{self.manifest_path!r}; without it the artifact cannot be "
                f"checked against the generator that made it")
        with open(self.manifest_path, "r", encoding="utf-8", newline="") as fh:
            return next(csv.DictReader(fh), {}) or {}

    def _verify(self) -> None:
        man = self._manifest()
        # `seed` is the manifest's name for the master seed, following the
        # declared column list rather than the config key it comes from.
        for key, mine in (("number_of_days", self.number_of_days),
                          ("seed", self.master_seed)):
            if key not in man:
                raise ValueError(
                    f"the library manifest does not record {key!r}; a library "
                    f"whose yardstick cannot be checked cannot be used")
            if int(float(man[key])) != int(mine):
                raise ValueError(
                    f"the library was built with {key}={man[key]} but this run "
                    f"asks for {mine}; rebuilding will change every price, so "
                    f"the mismatch is refused rather than warned about")
        lo, hi = self.library_range()
        for key, mine in (("r_min_mw", lo), ("r_max_mw", hi)):
            if key not in man:
                raise ValueError(
                    f"the library manifest does not record {key!r}; it is the "
                    f"yardstick every price in the file depends on")
            if abs(float(man[key]) - mine) > 1e-6:
                raise ValueError(
                    f"the library's recorded {key} is {man[key]} but this "
                    f"configuration derives {mine}; the file prices a "
                    f"different yardstick and must be rebuilt")

    def get_day_ahead_price(self, day_id: int, T: int = LIBRARY_PERIODS,
                            t_start: int = 0,
                            seed: Optional[int] = None) -> np.ndarray:
        day = self._prices.get(int(day_id))
        if day is None:
            raise KeyError(
                f"the library has no day {day_id}; it holds days "
                f"{min(self._prices)}..{max(self._prices)}")
        t_start, T = int(t_start), int(T)
        if t_start < 0 or T < 1 or t_start + T > day.size:
            raise ValueError(
                f"a window of {T} periods from {t_start} does not fit the "
                f"{day.size}-period day")
        return day[t_start:t_start + T].copy()

    def describe(self) -> dict:
        d = super().describe()
        d["source"] = self.name
        d["library_file"] = self.path
        d["library_days"] = len(self._prices)
        return d


class GuangdongWholesalePriceProvider(WholesalePriceProvider):
    """Phase two: a real Guangdong day-ahead series.

    Declared so the interface is settled before the data arrives, and so that
    the swap is a config change rather than a refactor. It raises until it is
    given a series to read.
    """

    name = "guangdong"

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = dict(cfg if cfg is not None else get_default(CONFIG_KEY, {}) or {})
        raise NotImplementedError(
            "the Guangdong provider has no data source yet; the interface is "
            "fixed so that adding one is a config change "
            f"({CONFIG_KEY}.source: guangdong, plus "
            f"'{CONFIG_KEY}.data_file'), not a change to any caller")


class LegacyMeritOrderProvider(WholesalePriceProvider):
    """The retired net-load map, kept only so recorded results still reproduce.

    It derives the price from the distribution network's own agents, which is
    exactly the endogeneity this module exists to remove. Nothing should select
    it for new work; `config/scenarios.yaml` declares the new source for the
    native scenario and this one for the legacy 10 MW scenarios, which are
    themselves retained only for comparison.
    """

    name = "legacy"

    def get_day_ahead_price(self, day_id: int, T: int = LIBRARY_PERIODS,
                            t_start: int = 0,
                            seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError(
            "the legacy map is computed inside grid.py from the agents it is "
            "given; it is not reachable through this interface on purpose, "
            "because it needs an argument this interface does not carry")

    def describe(self) -> dict:
        price_cfg = get_default("price_curve", {}) or {}
        base = float(price_cfg.get("base", 420.0))
        elasticity = float(price_cfg.get("merit_order_price_elasticity", 1.0))
        return {
            "source": self.name,
            "kind": "affine_net_load",
            "transfer": {"type": "affine", "base_cny_mwh": base,
                         "elasticity": elasticity},
            "declared_floor_cny_mwh": float(
                price_cfg.get("merit_order_min_price", 50.0)),
            "declared_cap_cny_mwh": float(
                price_cfg.get("merit_order_max_price", 900.0)),
            "noise_sigma_cny_mwh": float(price_cfg.get("noise_sigma", 30.0)),
            "scarcity_probability": float(
                price_cfg.get("spike_probability", 0.0)),
            "spike_structure_present": bool(
                float(price_cfg.get("spike_probability", 0.0)) > 0.0),
        }

    def day_structure_for(self, days: Sequence) -> List[dict]:
        price_cfg = get_default("price_curve", {}) or {}
        base = float(price_cfg.get("base", 420.0))
        elasticity = float(price_cfg.get("merit_order_price_elasticity", 1.0))
        rows = []
        for d in days:
            physics = getattr(d, "physics", {}) or {}
            lo = float(physics.get("net_load_min_ratio", 0.0))
            hi = float(physics.get("net_load_max_ratio", 1.0))
            peak = float(physics.get("forecast_peak_mw", 0.0)) or 1.0
            rows.append({
                "day": int(getattr(d, "day", 0)),
                "input_min": lo,
                "input_max": hi,
                "reachable_price_min_cny_mwh": base * (1.0 + elasticity * lo),
                "reachable_price_max_cny_mwh": base * (1.0 + elasticity * hi),
                "slope_cny_mwh_per_input": base * elasticity,
                "forecast_peak_mw": peak,
                "slope_cny_mwh_per_mw": base * elasticity / peak,
            })
        return rows


_PROVIDER_CACHE: Dict[str, WholesalePriceProvider] = {}


def current_source() -> str:
    return str(get_default(f"{CONFIG_KEY}.source", "legacy") or "legacy")


def get_scenario_cfg_source(scenario: str) -> str:
    """Which wholesale market a scenario declares, resolved the same way.

    `_build_scenario` writes the scenario's `wholesale_source` into the price
    layer on every build, falling back to `legacy` when it declares none. This
    asks the same question of the file, without building anything, so a caller
    can decide whether a day id would mean anything here.
    """
    from config_loader import get_scenario_cfg
    return str(get_scenario_cfg(scenario).get("wholesale_source", "legacy"))


def get_provider(source: Optional[str] = None,
                 cfg: Optional[dict] = None) -> WholesalePriceProvider:
    """The provider the configuration names.

    The one place a source name becomes a provider. Nothing downstream repeats
    this decision, so a caller never asks which market it is pricing against.
    """
    source = source or current_source()
    if source not in SOURCES:
        raise ValueError(f"unknown {CONFIG_KEY}.source {source!r}; "
                         f"expected one of {SOURCES}")
    if cfg is None and source in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[source]
    if source in ("synthetic_v1", "library"):
        provider: WholesalePriceProvider = (
            LibraryPriceProvider(cfg) if source == "library"
            else SyntheticWholesalePriceProvider(cfg))
    elif source == "guangdong":
        provider = GuangdongWholesalePriceProvider(cfg)
    else:
        provider = LegacyMeritOrderProvider()
    if cfg is None:
        _PROVIDER_CACHE[source] = provider
    return provider


def reset_cache() -> None:
    """Drop the cached providers, so a config edit is picked up."""
    _PROVIDER_CACHE.clear()


# ---------------------------------------------------------------------------
# The library artifact
# ---------------------------------------------------------------------------

def library_rows(provider: SyntheticWholesalePriceProvider) -> List[dict]:
    """Every period of every library day, in file order."""
    rows = []
    for day in range(provider.number_of_days):
        price = provider.day_ahead_price(day)
        for t in range(price.size):
            minutes = t * 15
            rows.append({
                "day_id": day,
                "time_index": t,
                "time": f"{minutes // 60:02d}:{minutes % 60:02d}",
                "price_cny_mwh": float(price[t]),
            })
    return rows


def build_library(path: Optional[str] = None,
                  manifest_path: Optional[str] = None,
                  cfg: Optional[dict] = None,
                  quiet: bool = False) -> dict:
    """Write the price library and its manifest. Returns the manifest row."""
    cfg = cfg if cfg is not None else (get_default(CONFIG_KEY, {}) or {})
    provider = SyntheticWholesalePriceProvider(cfg)
    path = path or cfg.get("library_file", "data/wholesale/synthetic_da_v1.csv")
    manifest_path = manifest_path or (
        cfg.get("library_manifest_file")
        or os.path.splitext(path)[0] + "_manifest.csv")

    rows = library_rows(provider)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["day_id", "time_index", "time", "price_cny_mwh"])
        writer.writeheader()
        writer.writerows(rows)

    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    lo, hi = provider.library_range()
    manifest = {
        "version": "synthetic_da_v1",
        "seed": provider.master_seed,
        "generator": "wholesale_price.SyntheticWholesalePriceProvider",
        "price_floor": provider.floor,
        "price_cap": provider.cap,
        "noise_sigma": provider.noise_sigma,
        "scarcity_probability": provider.scarcity_probability,
        "supply_stack": json.dumps(provider.stack.as_dict(), sort_keys=True),
        "number_of_days": provider.number_of_days,
        "resolution_minutes": 15,
        # The yardstick every price in the file is measured against. Recorded
        # because changing number_of_days changes it, and with it every price.
        "r_min_mw": lo,
        "r_max_mw": hi,
        "normalization": "minmax_over_library",
        "library_sha256": digest,
        "rows": len(rows),
    }
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)) or ".",
                exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest))
        writer.writeheader()
        writer.writerow(manifest)
    if not quiet:
        print(f"wrote {len(rows)} rows to {path}")
        print(f"wrote manifest to {manifest_path}")
        print(f"  r yardstick {lo:.4f} .. {hi:.4f} MW   sha256 {digest[:16]}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def stats(provider: SyntheticWholesalePriceProvider) -> dict:
    """Distribution of every price in the library."""
    everything = np.concatenate([provider.day_ahead_price(d)
                                 for d in range(provider.number_of_days)])
    r_all = np.concatenate([provider.reduced_input(d)
                            for d in range(provider.number_of_days)])
    occupancy = []
    knots = provider.stack.knots
    for k in range(len(knots) - 1):
        occupancy.append(float(np.mean((r_all >= knots[k])
                                       & (r_all < knots[k + 1]))))
    occupancy[-1] = float(1.0 - sum(occupancy[:-1]))
    return {
        "days": provider.number_of_days,
        "intervals_per_day": int(everything.size // provider.number_of_days),
        "min": float(everything.min()),
        "median": float(np.median(everything)),
        "mean": float(everything.mean()),
        "std": float(everything.std(ddof=1)),
        "max": float(everything.max()),
        "p_at_floor": float(np.mean(everything <= provider.floor + 1e-9)),
        "p_at_cap": float(np.mean(everything >= provider.cap - 1e-9)),
        "p_negative": float(np.mean(everything < 0.0)),
        "p_above_800": float(np.mean(everything > 800.0)),
        "ratio_max_min": (float(everything.max() / everything.min())
                          if everything.min() > 0 else float("inf")),
        "segment_occupancy": occupancy,
        "r_min": float(r_all.min()),
        "r_max": float(r_all.max()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-library", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--path", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.build_library:
        build_library(args.path, args.manifest, quiet=args.quiet)
    if args.stats or not args.build_library:
        provider = SyntheticWholesalePriceProvider()
        s = stats(provider)
        print(f"days {s['days']} x {s['intervals_per_day']} intervals")
        print(f"min {s['min']:.2f}  median {s['median']:.2f}  "
              f"mean {s['mean']:.2f}  std {s['std']:.2f}  max {s['max']:.2f}")
        print(f"P(<0) {s['p_negative']:.4f}  P(>800) {s['p_above_800']:.4f}  "
              f"max/min {s['ratio_max_min']:.2f}")
        print(f"segment occupancy "
              f"{[round(o, 3) for o in s['segment_occupancy']]}")
        print(f"r range {s['r_min']:.4f} .. {s['r_max']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
