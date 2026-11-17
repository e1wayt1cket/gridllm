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


def _array_sha256(array) -> str:
    """Hash the array's little-endian float64 bytes, as v1's pool records them."""
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


array_sha256 = _array_sha256


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
