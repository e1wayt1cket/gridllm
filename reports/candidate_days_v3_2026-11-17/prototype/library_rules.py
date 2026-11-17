"""Pure rules of the v3 candidate-day library: selection, mapping, split.

No file is read at import time and no rule depends on a clock, a solver
tolerance or a random draw other than the recorded split seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

ANCHOR_MW = 3.715
TOLERANCE = 0.10
PERIODS = 96
BUS_COUNT = 33
LOAD_BUSES = BUS_COUNT - 1
MAX_METERS = LOAD_BUSES
POWER_FACTOR = 0.8503
REACTIVE_RATIO = 0.6190
LOW_MW = ANCHOR_MW * (1.0 - TOLERANCE)
HIGH_MW = ANCHOR_MW * (1.0 + TOLERANCE)
LIBRARY_MONTHS = ("06", "07", "08", "09", "10", "11", "12")


def _array_sha256(array) -> str:
    """Hash the array's little-endian float64 bytes, as v1's pool records them."""
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


array_sha256 = _array_sha256


@dataclass(frozen=True)
class BusAssignment:
    bus_of_source: dict
    target_share: dict
    d_init: float
    d_final: float
    excluded_native_share: float
    target_max_share: float


def native_p0():
    """Native `case33bw` active load per bus, indexed from the slack at 0."""
    import pandapower.networks

    net = pandapower.networks.case33bw()
    p0 = np.zeros(BUS_COUNT, dtype=np.float64)
    for row in net.load.itertuples():
        p0[int(row.bus)] += float(row.p_mw)
    return p0


def assign_buses(pi, p0):
    """Map the day's meters onto the `len(pi)` heaviest native buses.

    One meter per bus, so the assignment is a permutation; the local step
    exchanges the meters on two target buses and only accepts an exchange that
    lowers the squared relative deviation from the target shares.
    """
    sources = sorted(pi, key=lambda k: (-pi[k], k))
    n = len(sources)
    buses = sorted(range(BUS_COUNT), key=lambda b: (-p0[b], b))[:n]
    target_total = float(sum(p0[b] for b in buses))
    share = {b: float(p0[b]) / target_total for b in buses}
    assigned = {bus: float(pi[sources[i]]) for i, bus in enumerate(buses)}

    contrib = {b: (assigned[b] / share[b] - 1.0) ** 2 for b in buses}
    d_init = current = float(sum(contrib.values()))
    while True:
        best = None
        best_value = None
        for i in range(n - 1):
            bi = buses[i]
            for j in range(i + 1, n):
                bj = buses[j]
                value = (current - contrib[bi] - contrib[bj]
                         + (assigned[bj] / share[bi] - 1.0) ** 2
                         + (assigned[bi] / share[bj] - 1.0) ** 2)
                if best is None or value < best_value:
                    best, best_value = (bi, bj), value
        if best is None or current - best_value <= 1e-12:
            break
        bi, bj = best
        assigned[bi], assigned[bj] = assigned[bj], assigned[bi]
        contrib[bi] = (assigned[bi] / share[bi] - 1.0) ** 2
        contrib[bj] = (assigned[bj] / share[bj] - 1.0) ** 2
        current = best_value

    return BusAssignment(
        bus_of_source={sources[i]: bus for i, bus in enumerate(buses)},
        target_share=share,
        d_init=d_init,
        d_final=current,
        excluded_native_share=1.0 - target_total / float(p0.sum()),
        target_max_share=max(share.values()),
    )


def bus_matrix(bus_of_source, curves):
    """Lay each source's curve on its bus; unspecified buses carry no load."""
    matrix = np.zeros((PERIODS, BUS_COUNT), dtype=np.float64)
    for source, bus in bus_of_source.items():
        matrix[:, bus] = curves[source]
    return matrix


def reactive_from_active(active):
    """One declared power factor for every meter; zeros stay zero."""
    return REACTIVE_RATIO * np.asarray(active, dtype=np.float64)


@dataclass(frozen=True)
class SplitPlan:
    blocks: dict
    counts: dict
    pairs: dict
    discarded: dict
    library_size: int
    holiday_shortfall: int


SPLIT_SEED = 20261117
BLOCKS = ("train", "validation", "test", "holiday")
HOLIDAY_MONTH_DAYS = {f"10-0{d}" for d in range(1, 6)}


def draw_split(in_band_dates, price_dates_by_month, seed=SPLIT_SEED):
    """Prepare the month-and-year stratified split, outcome-blind and seeded.

    Every month draws as many load dates as its 2025 price pool can pair with;
    the rest are recorded as discarded rather than silently dropped.
    """
    dates = sorted(set(in_band_dates))
    holiday_window = [d for d in dates if d[5:] in HOLIDAY_MONTH_DAYS]
    holiday = tuple(holiday_window[:10])

    by_month = {}
    for date in dates:
        if date in holiday:
            continue
        by_month.setdefault(int(date[5:7]), []).append(date)

    kept_by_month = {}
    discarded = {}
    holiday_by_month = {}
    for date in holiday:
        month = int(date[5:7])
        holiday_by_month[month] = holiday_by_month.get(month, 0) + 1
    for month in sorted(by_month):
        by_year = {}
        for date in by_month[month]:
            by_year.setdefault(int(date[:4]), []).append(date)
        supply = len(by_month[month])
        # A holiday day consumes one of its month's price dates, so the draw
        # leaves room for it: every library day keeps a unique price date.
        available = len(price_dates_by_month.get(f"{month:02d}", ()))
        keep = min(supply, max(0, available - holiday_by_month.get(month, 0)))
        quota = _largest_remainder({year: len(v) for year, v in by_year.items()}, keep)
        kept = []
        rejected = []
        for year in sorted(by_year):
            candidates = sorted(by_year[year])
            order = np.random.default_rng(seed + 100 * month + year).permutation(len(candidates))
            picked = {candidates[i] for i in order[:quota[year]]}
            kept.extend(sorted(picked))
            rejected.extend(candidates[i] for i in order[quota[year]:])
        kept_by_month[month] = sorted(kept)
        discarded[f"{month:02d}"] = tuple(sorted(rejected))

    groups = {}
    for month in sorted(kept_by_month):
        for date in kept_by_month[month]:
            groups.setdefault((month, int(date[:4])), []).append(date)
    for key in sorted(groups):
        values = sorted(groups[key])
        index = np.random.default_rng(seed + 100 * key[0] + key[1]).permutation(len(values))
        groups[key] = [values[i] for i in index]

    order = []
    keys = sorted(groups)
    for position in range(max((len(v) for v in groups.values()), default=0)):
        for key in keys:
            if position < len(groups[key]):
                order.append(groups[key][position])

    pattern = ["train"] * 5 + ["validation", "test"]
    labels = [pattern[i % len(pattern)] for i in range(len(order))]

    blocks = {label: [] for label in BLOCKS}
    for date, label in zip(order, labels):
        blocks[label].append(date)
    blocks["holiday"] = list(holiday)
    blocks = {label: tuple(sorted(blocks[label])) for label in BLOCKS}

    price_of = _price_of(blocks, price_dates_by_month)
    pairs = {}
    for label in BLOCKS:
        pairs[label] = tuple(f"{date}|{price_of[date]}" for date in blocks[label])

    counts = {label: len(blocks[label]) for label in BLOCKS}
    return SplitPlan(
        blocks=blocks,
        counts=counts,
        pairs=pairs,
        discarded=discarded,
        library_size=len(order) + len(holiday),
        holiday_shortfall=max(0, 10 - len(holiday)),
    )


def _largest_remainder(weights, total):
    """Split `total` over integer keys; ties go to the smaller key."""
    weight_sum = sum(weights.values())
    if weight_sum == 0 or total <= 0:
        return {key: 0 for key in weights}
    share = {key: total * weights[key] / weight_sum for key in weights}
    quota = {key: int(np.floor(share[key])) for key in weights}
    remainder = total - sum(quota.values())
    for key in sorted(weights, key=lambda k: (-(share[k] - quota[k]), k))[:remainder]:
        quota[key] += 1
    return quota


def _price_of(blocks, price_dates_by_month):
    """Give every library day its month's next unused price date, in date order."""
    by_month = {}
    for label in BLOCKS:
        for date in blocks[label]:
            by_month.setdefault(date[5:7], []).append(date)
    mapping = {}
    for month in sorted(by_month):
        prices = sorted(price_dates_by_month.get(month, ()))
        for load, price in zip(sorted(by_month[month]), prices):
            mapping[load] = price
    missing = [date for dates in by_month.values() for date in dates if date not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} library days have no same-month price date")
    return mapping


@dataclass(frozen=True)
class DaySelection:
    date: str
    sources: tuple[str, ...]
    users: tuple[str, ...]
    peak_mw: float
    # Excluded from equality so two runs can be compared without array truth tests.
    total: np.ndarray = field(compare=False)
    status: str
    reason: str
    hit_bus_ceiling: bool


def select_day(date, curves, user_of):
    """Pick the day's meter set; `curves` is already sliced to this date.

    The rule is v1's, plus a ceiling of one meter per load bus and the zero-peak
    exclusion applied before the one-meter-per-user reduction.
    """
    candidates = [k for k in curves if float(np.max(curves[k])) > 0.0]
    peaks = {k: float(np.max(curves[k])) for k in candidates}

    best_of_user = {}
    for k in candidates:
        user = user_of[k]
        if user not in best_of_user or (peaks[k], k) < (
            peaks[best_of_user[user]],
            best_of_user[user],
        ):
            best_of_user[user] = k
    order = sorted((peaks[k], k) for k in best_of_user.values())

    chosen = []
    total = np.zeros(PERIODS, dtype=np.float64)
    peak = 0.0
    hit_bus_ceiling = False
    for _, k in order:
        if len(chosen) >= MAX_METERS:
            hit_bus_ceiling = True
            break
        trial = total + curves[k]
        trial_peak = float(trial.max())
        if trial_peak <= HIGH_MW:
            chosen.append(k)
            total = trial
            peak = trial_peak

    if not chosen:
        total.setflags(write=False)
        return DaySelection(date=date, sources=(), users=(), peak_mw=0.0, total=total,
                            status="OUT_OF_BAND", reason="nothing_selectable",
                            hit_bus_ceiling=False)

    swaps = 0
    initial_peak = peak
    # Every legal swap strictly raises one chosen meter's own peak, so no meter-day
    # can be swapped upward more often than there are candidates; the bound stops a
    # broken rule from cycling rather than describing a reachable state.
    while peak < LOW_MW and swaps < len(candidates):
        best = None
        for index, k in enumerate(chosen):
            user = user_of[k]
            old_peak = peaks[k]
            larger = [other for other in candidates
                      if user_of[other] == user and peaks[other] > old_peak]
            for other in larger:
                trial = total - curves[k] + curves[other]
                trial_peak = float(trial.max())
                if trial_peak > HIGH_MW:
                    continue
                key = (-trial_peak, k, other)
                if best is None or key < best[0]:
                    best = (key, index, other, trial, trial_peak)
        if best is None:
            break
        _, index, other, total, peak = best
        chosen[index] = other
        swaps += 1

    # Report the peak the selected set actually has, not a second recomputation.
    total = np.sum([curves[k] for k in chosen], axis=0)
    total.setflags(write=False)
    peak = float(total.max())
    if initial_peak >= LOW_MW:
        reason = "initial_in_band"
    elif peak >= LOW_MW:
        reason = "raised_in_band"
    else:
        reason = "no_swap_below_band"
    status = "IN_BAND" if LOW_MW <= peak <= HIGH_MW else "OUT_OF_BAND"
    pairs = sorted((k, user_of[k]) for k in chosen)
    return DaySelection(
        date=date,
        sources=tuple(p[0] for p in pairs),
        users=tuple(p[1] for p in pairs),
        peak_mw=peak,
        total=total,
        status=status,
        reason=reason,
        hit_bus_ceiling=hit_bus_ceiling,
    )
