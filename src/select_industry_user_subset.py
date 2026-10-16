# select_industry_user_subset.py
"""Pick the real metered users whose day reproduces the IEEE33 benchmark's scale.

The scenario this feeds carries *measured* industry load. Every mapped user
brought in, the fleet's day peaks at 559 MW while the IEEE33 benchmark's own
total active demand is **3.715 MW** -- two orders of magnitude apart, so
"measured industry load on the IEEE33 feeder at the benchmark's scale" was a
description the artifact did not support.

This module chooses a subset of the users that does support it, and it does so
**without rescaling anything**. A user is in or out whole; no curve is
normalised, trimmed, interpolated, resampled or clipped, and no high period is
deleted. The only transform any reading passes through is the unit conversion in
`industry_load.reading_to_mw`, which is where it has always been.

What is being matched, and what is not
--------------------------------------
The target is the benchmark's **total active demand**, not a feeder rating.
This is a load-scale match, not a thermal-capacity verdict, and the two must not
be conflated in anything written about the result.

`TARGET_PEAK_TOLERANCE` is a **project choice**. It is not an IEEE33 parameter
and no source states it; it is written into the scenario configuration
alongside its own status so a reader cannot mistake it for one.

Why the objective is ordered the way it is
------------------------------------------
The peak is a maximum over periods, so leaving every bus on its native load
already reproduces 3.715 MW to machine precision -- the empty subset carries no
error worth the name. An
objective that minimised error first would therefore select nobody, which is
not a selection. The tolerance is what "close" means here: inside it, peak
errors are comparable by construction, so the tie-break that carries information
is how many real users the subset accounts for. Within the tolerance the
selector therefore takes the most users, then the smallest error, then the most
industries, then the lexicographically smallest user set.

The search is exhaustive over the users it can admit, so "most users" is exact
rather than greedy. Which users it can admit is decided by a bound that is
proved, not tuned -- see `candidates`.

Usage:
    PYTHONPATH=src python src/select_industry_user_subset.py --date 2022-11-30
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

import build_industry_node_loads as bnl
import config_loader as cfg
import industry_load as il

#: The IEEE33 benchmark's own total active demand, in MW. The standard
#: `case33bw` load sum, read back from the built network as well as declared
#: here, so the two cannot silently disagree.
TARGET_PEAK_MW = 3.715

#: How far the final peak may sit from the target and still count as a match.
#: **A project ruling, not a benchmark parameter.** Recorded in the scenario
#: configuration and the manifest; no source states it and none is implied.
TARGET_PEAK_TOLERANCE = 0.05

#: The scenario the subset is declared under, and the scenario its non-load
#: settings are inherited from. The base is read, never written.
DEFAULT_SCENARIO = "typical_day_industry_subset"
BASE_SCENARIO = "typical_day_industry"

#: The frozen user-to-bus mapping, and the derived file this module writes for
#: the scenario's own users.
V2_MAPPING = Path("data/processed/industry_meter_bus_map_v2.csv")
SUBSET_MAPPING = Path("data/processed/industry_user_subset_mapping_v2.csv")

#: The two-level search evaluates one half against the other, so the pair count
#: is the runtime and each half is the memory. A larger candidate pool than this
#: is reported rather than silently truncated: a truncated search would return a
#: subset that is not the maximum while claiming to be one.
MAX_PAIR_EVALUATIONS = 1 << 26

#: Combinations evaluated per numpy block. Sized for the widest half this
#: module expects; it only trades peak memory against loop count.
CHUNK = 16


def rule_inputs() -> dict:
    """Everything the selection depends on, in one place.

    The tolerance belongs here beside the target because both decide the answer.
    Recording them makes the manifest move by itself when either changes, which
    is what stops a superseded subset from being read as a current one.
    """
    return {
        "target_peak_mw": TARGET_PEAK_MW,
        "target_peak_tolerance": TARGET_PEAK_TOLERANCE,
        "objective_order": ["n_users_desc", "abs_error_asc",
                            "n_industries_desc", "user_ids_asc"],
        "selection_is_exhaustive_over_candidates": True,
        "reading_unit": il.READING_UNIT,
        "reading_unit_evidence": il.READING_UNIT_EVIDENCE,
        "slot_minutes": il.SLOT_MINUTES,
        "slots_per_day": il.N_SLOTS,
        "mapping_version": "industry_bus_map_v2",
    }


def rule_version() -> str:
    """A digest over everything the selection depends on."""
    payload = json.dumps(rule_inputs(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_mapping(path: Path) -> List[dict]:
    """The frozen mapping's rows, in file order."""
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mapped_users(rows: Sequence[dict]) -> Dict[str, int]:
    """The users the mapping places on a bus. An unmapped user is not a load."""
    return {r["user_id"]: int(r["destination_bus"]) for r in rows
            if r["candidate_status"] == "mapped"}


def industry_of(rows: Sequence[dict]) -> Dict[str, str]:
    return {r["user_id"]: r["industry"] for r in rows}


def meters_of(rows: Sequence[dict]) -> Dict[str, int]:
    return {r["user_id"]: int(r["n_meters"]) for r in rows}


def require_non_negative(curves: Dict[str, np.ndarray],
                         shape: np.ndarray) -> None:
    """The premise the candidate bound rests on, checked rather than assumed.

    The bound below discards a user whose own peak already exceeds the tolerance
    ceiling, on the grounds that the system peak can only be at least that
    user's own contribution. That argument needs every *other* contribution to
    be non-negative at the period in question. A single negative reading breaks
    it, and a silently wrong pruning rule would return a subset that is not the
    maximum while presenting itself as one.
    """
    for user in sorted(curves):
        arr = curves[user]
        if float(arr.min()) < 0.0:
            raise ValueError(
                f"user {user} carries a negative slot ({float(arr.min()):.6g} "
                "MW). The candidate bound assumes every contribution is "
                "non-negative, so no user may be pruned while one is not. "
                "The selector refuses rather than returning a subset it cannot "
                "show to be the maximum.")
    if float(shape.min()) < 0.0:
        raise ValueError(
            "the native diurnal shape is negative in some period; the "
            "candidate bound does not hold for it")


def candidates(curves: Dict[str, np.ndarray], target: float,
               tolerance: float) -> Tuple[List[str], float]:
    """The users that could belong to *some* tolerance-feasible subset.

    A user whose own peak already exceeds `target x (1 + tolerance)` cannot: at
    that user's own peak period the system total is at least the user's
    contribution, and every other bus's contribution is non-negative (checked
    by `require_non_negative`). So the system peak would exceed the ceiling and
    the subset would miss the tolerance however the rest were chosen.

    Nothing here is tuned. The ceiling is the tolerance the caller declared and
    the argument holds for any subset, which is why the exhaustive search below
    is over the *whole* feasible space and not a sample of it.
    """
    ceiling = target * (1.0 + tolerance)
    return (sorted(u for u, arr in curves.items()
                   if float(arr.max()) <= ceiling), ceiling)


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def bus_native_load(rated: Dict[int, float], shape: np.ndarray) -> np.ndarray:
    """The all-`preserve` day: every load bus on the benchmark's own demand."""
    total = np.zeros(il.N_SLOTS)
    for native_mw in rated.values():
        total = total + native_mw * shape
    return total


def bus_options(users: Sequence[str], curves: Dict[str, np.ndarray],
                native_mw: float, shape: np.ndarray
                ) -> List[Tuple[int, np.ndarray, FrozenSet[str]]]:
    """Every subset of one bus's candidates, as a change to the day.

    A bus takes exactly one of these, and the empty one means `preserve`: the
    bus keeps the benchmark's own demand and the day is unchanged by it. Any
    other entry means `replace`, and carries the difference between the selected
    users' summed absolute MW and the native demand it displaces.

    `preserve` is the empty entry rather than a separate policy because that is
    what it is -- a bus whose users are all out is a bus the selection never
    touched.
    """
    options = [(0, np.zeros(il.N_SLOTS), frozenset())]
    for mask in range(1, 1 << len(users)):
        picked = frozenset(u for i, u in enumerate(users) if mask >> i & 1)
        total = np.zeros(il.N_SLOTS)
        for user in picked:
            total = total + curves[user]
        options.append((len(picked), total - native_mw * shape, picked))
    return options


def split_buses(options: Dict[int, List[tuple]]
                ) -> Tuple[List[int], List[int]]:
    """Two groups whose option products are as close as practical.

    Deterministic: the buses are taken in descending option count with the bus
    number breaking ties, so the same input always partitions the same way.
    """
    left: List[int] = []
    right: List[int] = []
    left_size = right_size = 1
    for bus, entries in sorted(options.items(),
                               key=lambda kv: (-len(kv[1]), kv[0])):
        if left_size <= right_size:
            left.append(bus)
            left_size *= len(entries)
        else:
            right.append(bus)
            right_size *= len(entries)
    return left, right


def expand(buses: Sequence[int], options: Dict[int, List[tuple]]
           ) -> Tuple[np.ndarray, np.ndarray, List[FrozenSet[str]]]:
    """One combination per way of choosing an option on each of `buses`."""
    deltas = [np.zeros(il.N_SLOTS)]
    counts = [0]
    users: List[FrozenSet[str]] = [frozenset()]
    for bus in buses:
        next_deltas: List[np.ndarray] = []
        next_counts: List[int] = []
        next_users: List[FrozenSet[str]] = []
        for index, delta in enumerate(deltas):
            for option_count, option_delta, option_users in options[bus]:
                next_deltas.append(delta + option_delta)
                next_counts.append(counts[index] + option_count)
                next_users.append(users[index] | option_users)
        deltas, counts, users = next_deltas, next_counts, next_users
    return np.stack(deltas), np.asarray(counts, dtype=np.int64), users


def _industry_flags(user_sets: Sequence[FrozenSet[str]],
                    index_of: Dict[str, int], width: int) -> np.ndarray:
    """Which industries each combination covers, as a boolean matrix.

    Indexed by user, so a combination's industry set is the union of its users'
    and two users in one industry count once -- which is what "how many
    industries does this subset represent" is asking.

    `width` spans every industry in the mapping, not only those a mapped user
    carries: an industry whose only users are unmapped still holds a column, and
    sizing the matrix from the indices actually used would leave gaps and read
    off the end.
    """
    flags = np.zeros((len(user_sets), width), dtype=bool)
    for row, picked in enumerate(user_sets):
        for user in picked:
            flags[row, index_of[user]] = True
    return flags


def search(curves: Dict[str, np.ndarray], user_bus: Dict[str, int],
           industries: Dict[str, str], rated: Dict[int, float],
           shape: np.ndarray, target: float, tolerance: float) -> dict:
    """The exhaustive search over every subset the bound can admit.

    Returns the winning subset with the evidence for the choice, the closest
    subset overall, and how many subsets were examined -- a count that has to
    travel with the answer, because "the maximum" means nothing without the
    space it is maximal in.
    """
    # First, because everything below prunes on the strength of it.
    require_non_negative(curves, shape)

    pool, ceiling = candidates(curves, target, tolerance)
    # A user the mapping does not place on a bus is not a load anywhere, so it
    # is not a candidate however small its reading: the selection works under
    # the mapping that exists, and invents no placement of its own.
    pool = [u for u in pool if u in user_bus]
    base = bus_native_load(rated, shape)

    options: Dict[int, List[tuple]] = {}
    for bus in sorted({user_bus[u] for u in pool}):
        users = sorted(u for u in pool if user_bus[u] == bus)
        options[bus] = bus_options(users, curves, rated[bus], shape)

    left, right = split_buses(options)
    delta_left, count_left, users_left = expand(left, options)
    delta_right, count_right, users_right = expand(right, options)

    pairs = len(delta_left) * len(delta_right)
    if pairs > MAX_PAIR_EVALUATIONS:
        raise ValueError(
            f"the candidate bound admits {len(pool)} users, which needs "
            f"{pairs:,} subsets to search exhaustively and exceeds the "
            f"{MAX_PAIR_EVALUATIONS:,} this selector will evaluate. Reporting "
            "the count rather than truncating: a partial search would return a "
            "subset that is not the maximum while claiming to be one.")

    names = {name: i for i, name in enumerate(sorted(set(industries.values())))}
    index_of = {user: names[industries[user]] for user in user_bus
                if user in industries}
    flags_left = _industry_flags(users_left, index_of, len(names))
    flags_right = _industry_flags(users_right, index_of, len(names))

    # Pass one: the best key, and the closest subset regardless of tolerance.
    best: Optional[Tuple[int, float, int]] = None
    closest: Optional[Tuple[float, int, int]] = None
    for start in range(0, len(delta_left), CHUNK):
        peak, counts, spreads = _scores(
            base, delta_left[start:start + CHUNK], delta_right,
            count_left[start:start + CHUNK], count_right,
            flags_left[start:start + CHUNK], flags_right)
        error = np.abs(peak - target) / target
        row, col = np.unravel_index(int(np.argmin(error)), error.shape)
        if closest is None or float(error[row, col]) < closest[0]:
            closest = (float(error[row, col]), start + int(row), int(col))
        block = _block_best(counts, error, spreads, error <= tolerance)
        if block is not None and (best is None or _better(block, best)):
            best = block
    if best is None:
        return {"feasible": False, "selected": [], "closest": closest,
                "pool": pool, "ceiling": ceiling, "evaluated": int(pairs),
                "base": base, "total_mw": base, "n_tied": 0,
                "tied_at_the_best_key": []}

    # Pass two: every subset that attains the best key, so the last tie-break
    # is applied to all of them rather than to whichever came first.
    tied: List[FrozenSet[str]] = []
    for start in range(0, len(delta_left), CHUNK):
        peak, counts, spreads = _scores(
            base, delta_left[start:start + CHUNK], delta_right,
            count_left[start:start + CHUNK], count_right,
            flags_left[start:start + CHUNK], flags_right)
        error = np.abs(peak - target) / target
        rows, cols = np.nonzero(
            (counts == best[0]) & (error == best[1]) & (spreads == best[2]))
        for row, col in zip(rows.tolist(), cols.tolist()):
            tied.append(users_left[start + row] | users_right[col])
    if not tied:
        raise AssertionError(
            f"the two passes disagree about the best key {best}; the search "
            "is not reproducible and its result cannot be trusted")

    chosen = min(tied, key=lambda s: tuple(sorted(s)))
    delta = np.zeros(il.N_SLOTS)
    for bus in {user_bus[u] for u in chosen}:
        delta = delta - rated[bus] * shape
    for user in chosen:
        delta = delta + curves[user]
    total = base + delta

    return {
        "feasible": True,
        "selected": sorted(chosen),
        "tied_at_the_best_key": sorted(sorted(s) for s in tied),
        "n_tied": len(tied),
        "pool": pool,
        "ceiling": ceiling,
        "evaluated": int(pairs),
        "base": base,
        "total_mw": total,
        "mean_mw": float(total.mean()),
        "energy_mwh": float(total.sum() * il.HOURS_PER_SLOT),
        "closest": closest,
        "key": best,
    }


def _scores(base, delta_left, delta_right, count_left, count_right,
            flags_left, flags_right):
    """Peak, user count and industry spread for every pair in the block."""
    total = base[None, None, :] + delta_left[:, None, :] + delta_right[None, :, :]
    peak = total.max(axis=2)
    counts = count_left[:, None] + count_right[None, :]
    spreads = (flags_left[:, None, :] | flags_right[None, :, :]).sum(axis=2)
    return peak, counts, spreads


def _block_best(counts: np.ndarray, error: np.ndarray, spreads: np.ndarray,
                feasible: np.ndarray) -> Optional[Tuple[int, float, int]]:
    """This block's best key under (users desc, error asc, industries desc).

    Vectorised rather than looped: the feasible set can run to millions of
    pairs and a Python walk over them would dominate the run.
    """
    if not feasible.any():
        return None
    n_users = counts[feasible].max()
    at_max = feasible & (counts == n_users)
    smallest = error[at_max].min()
    at_error = at_max & (error == smallest)
    return (int(n_users), float(smallest), int(spreads[at_error].max()))


def _better(candidate: Tuple[int, float, int],
            incumbent: Tuple[int, float, int]) -> bool:
    """More users; then smaller error; then more industries."""
    if candidate[0] != incumbent[0]:
        return candidate[0] > incumbent[0]
    if candidate[1] != incumbent[1]:
        return candidate[1] < incumbent[1]
    return candidate[2] > incumbent[2]


# ---------------------------------------------------------------------------
# Why each user is in or out
# ---------------------------------------------------------------------------

#: The reasons a user can carry. Each is a statement the run can show.
REASON_INCLUDED = "included"
REASON_UNMAPPED = "the mapping does not place this user on any bus"
REASON_NO_READING = "no qualified reading on the selected date"
REASON_TOO_LARGE = (
    "own peak exceeds the tolerance ceiling; the system peak is at least this "
    "user's own contribution, so no tolerance-feasible subset contains it")
REASON_NOT_IN_MAXIMUM = (
    "eligible, but the largest tolerance-feasible subset does not contain it: "
    "adding it to the selected subset leaves the tolerance")


def classify_users(curves: Dict[str, np.ndarray], user_bus: Dict[str, int],
                   selected: Sequence[str], pool: Sequence[str],
                   ceiling: float) -> List[dict]:
    """One reason per user, and the evidence for it where there is any.

    The reason for an eligible user that was not taken is a claim, not a
    label: the selected subset is the largest feasible one, so adding any
    eligible user to it must leave the tolerance. `verify_maximal` checks
    exactly that, and the claim is only written after it holds.
    """
    chosen = set(selected)
    pool_set = set(pool)
    rows = []
    for user in sorted(user_bus):
        arr = curves.get(user)
        in_pool = user in pool_set
        if user in chosen:
            reason = REASON_INCLUDED
        elif arr is None:
            reason = REASON_NO_READING
        elif not in_pool:
            reason = REASON_TOO_LARGE
        else:
            reason = REASON_NOT_IN_MAXIMUM
        rows.append({
            "user_id": user,
            "destination_bus": user_bus[user],
            "included": user in chosen,
            "reason": reason,
            "own_peak_mw": None if arr is None else float(arr.max()),
            "own_mean_mw": None if arr is None else float(arr.mean()),
            "own_energy_mwh": (None if arr is None
                               else float(arr.sum() * il.HOURS_PER_SLOT)),
            "eligible": in_pool,
            "vs_ceiling_mw": None if arr is None else float(arr.max() - ceiling),
        })
    for user in sorted(set(curves) - set(user_bus)):
        rows.append({
            "user_id": user, "destination_bus": "", "included": False,
            "reason": REASON_UNMAPPED,
            "own_peak_mw": float(curves[user].max()),
            "own_mean_mw": float(curves[user].mean()),
            "own_energy_mwh": float(curves[user].sum() * il.HOURS_PER_SLOT),
            "eligible": False, "vs_ceiling_mw": None,
        })
    return rows


def verify_maximal(super_set: dict, curves: Dict[str, np.ndarray],
                   user_bus: Dict[str, int], rated: Dict[int, float],
                   shape: np.ndarray, target: float,
                   tolerance: float) -> None:
    """Check the claim `REASON_NOT_IN_MAXIMUM` makes, one user at a time.

    If some other eligible user could be added and the day would still match,
    the selected subset was not the largest and the search is wrong. That is
    worth a failure rather than a wrong reason string.
    """
    base = super_set["base"]
    chosen = list(super_set["selected"])
    for user in super_set["pool"]:
        if user in chosen:
            continue
        trial = chosen + [user]
        delta = np.zeros(il.N_SLOTS)
        for bus in {user_bus[u] for u in trial}:
            delta = delta - rated[bus] * shape
        for u in trial:
            delta = delta + curves[u]
        peak = float((base + delta).max())
        if abs(peak - target) / target <= tolerance:
            raise AssertionError(
                f"the selected subset is not the largest feasible one: "
                f"{user} can be added and the peak is still {peak:.6f} MW. "
                "The search is wrong and the reason written for the users it "
                "left out is false.")


# ---------------------------------------------------------------------------
# Reactive load
# ---------------------------------------------------------------------------


def reactive_account(power_factor: float, peak_mw: float,
                     energy_mwh: float) -> dict:
    """What the final reactive load is, and what standing it has.

    Every load in this network takes its reactive power from the configured
    power factor -- `q = p * tan(phi)` in `dispatch_socp`, `dispatch_ldf` and
    `diagnose_typical_day` -- so the day's reactive load is a function of its
    active load and a declared number, not a measurement of anything.

    The factor is recorded as an **assumption**. It is reported here, never
    adjusted toward the benchmark's 2.3 MVAr: moving it to land on that figure
    would be tuning a knob until a number appears, which is the move this
    project refuses everywhere else. See `reports/industry_user_subset_
    selection.md` for what the calibration still does and does not hold.
    """
    tan_phi = math.tan(math.acos(power_factor))
    return {
        "rule": "q = p * tan(phi), applied per bus and per period",
        "power_factor": power_factor,
        "power_factor_status": (
            "simulation assumption, NOT a measurement. No meter in this "
            "workbook reports reactive power, so no reactive load is observed "
            "anywhere in this study."),
        "tan_phi": tan_phi,
        "q_peak_mvar": peak_mw * tan_phi,
        "q_energy_mvarh": energy_mwh * tan_phi,
        "benchmark_q_mvar": 2.3,
        "applied_in": ["dispatch_socp", "dispatch_ldf", "diagnose_typical_day"],
        "rescaled_to_the_benchmark": False,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_subset_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["user_id", "destination_bus", "included", "eligible", "reason",
               "own_peak_mw", "own_mean_mw", "own_energy_mwh", "vs_ceiling_mw"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([
                row["user_id"], row["destination_bus"],
                "yes" if row["included"] else "no",
                "yes" if row["eligible"] else "no", row["reason"],
                "" if row["own_peak_mw"] is None
                else f"{row['own_peak_mw']:.9f}",
                "" if row["own_mean_mw"] is None
                else f"{row['own_mean_mw']:.9f}",
                "" if row["own_energy_mwh"] is None
                else f"{row['own_energy_mwh']:.9f}",
                "" if row["vs_ceiling_mw"] is None
                else f"{row['vs_ceiling_mw']:.9f}",
            ])
    return path


def write_subset_mapping(path: Path, map_rows: Sequence[dict],
                         selected: Sequence[str]) -> Path:
    """The frozen mapping, restricted to the users this scenario keeps.

    Written rather than filtered at read time so the ledger builder sees one
    mapping artefact and derives `has_user` from it, exactly as it does for the
    full set. A bus the subset empties is then genuinely a bus with no user
    rather than a bus whose policy contradicts its own classification, and no
    `replace` bus can be left with nothing behind it.
    """
    kept = set(selected)
    columns = list(map_rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in map_rows:
            if row["user_id"] in kept:
                writer.writerow(row)
    return path


def policies_for(selected: Sequence[str], user_bus: Dict[str, int]) -> Dict[int, str]:
    """One policy per bus, derived from the users that were actually kept.

    A bus is `replace` only when a selected user landed on it, and `preserve`
    otherwise -- including the slack, which is never a mapping target. Nothing
    is inherited from the full-set scenario: the old policy named buses as
    carrying load that this subset empties.
    """
    carrying = {user_bus[u] for u in selected}
    return {bus: ("replace" if bus in carrying else "preserve")
            for bus in bnl.ALL_BUSES}


def assert_policies_match_ledger(policies: Dict[int, str],
                                 ledger_manifest: dict) -> None:
    """The declared policy and the built ledger must be the same 33 answers."""
    declared = {str(b): p for b, p in policies.items()}
    built = ledger_manifest["policies"]
    if declared != built:
        differing = sorted(b for b in set(declared) | set(built)
                           if declared.get(b) != built.get(b))
        raise AssertionError(
            f"the subset's per-bus policy differs from the ledger it produced "
            f"on bus(es) {differing}")


def report_markdown(result: dict, rows: Sequence[dict], date, scenario: str,
                     target: float, tolerance: float, source: dict) -> str:
    """The selection report."""
    total = result["total_mw"] if result["feasible"] else result["base"]
    peak = float(total.max())
    listed = [r for r in rows if r["included"]]
    industries = sorted({source["industry"][r["user_id"]] for r in listed})

    lines: List[str] = []
    add = lines.append
    add("# IEEE33 真实行业用户子集筛选")
    add("")
    add(f"目标：在不缩放、不修改任何原始负荷曲线的前提下，从实测行业用户中选出一个"
        f"子集，使最终 IEEE33 网络在 **{date}** 上的全网同时有功峰值接近 "
        f"**{target} MW**。")
    add("")
    add("## 结论")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| 场景 | `{scenario}` |")
    add(f"| 负荷日期 | `{date}` |")
    add(f"| 最终系统同时峰值 | **{peak:.4f} MW**（t={int(total.argmax())}） |")
    add(f"| 相对误差 | **{(peak - target) / target:+.3%}** |")
    add(f"| 匹配容差 | ±{tolerance:.0%}（**本项目选择标准，不是 IEEE33 标准参数**） |")
    add(f"| 入选用户 | **{len(listed)}** |")
    add(f"| 覆盖行业 | {len(industries)} — {'、'.join(industries)} |")
    add(f"| 入选母线（`replace`） | "
        f"{sorted({r['destination_bus'] for r in listed})} |")
    add("| 其余母线 | `preserve`，保留原生 IEEE33 负荷 |")
    add(f"| 全日平均功率 | {result['mean_mw']:.6f} MW |")
    add(f"| 全日电量 | {result['energy_mwh']:.4f} MWh |")
    add(f"| 搜索空间 | {result['evaluated']:,} 个子集（候选池穷举） |")
    add("")
    if result["feasible"]:
        tied = result.get("n_tied", 0)
        add(f"取到该结果的最大可行子集有 **{tied}** 个"
            + ("，它们达到同一个（用户数, |误差|, 行业数）三元组，"
               "再由用户 id 字典序取定。" if tied > 1 else "，没有并列。"))
    else:
        add("**没有任何子集落在容差内。** 容差未被放宽，负荷未被缩放，"
            "下面给出的是最接近的方案及其误差。")
    add("")

    add("## 为什么不是最小化误差优先")
    add("")
    add(f"峰值是 96 个时段的**最大值**，因此把所有母线留在原生负荷上就已经复现 "
        f"{target} MW（到机器精度）—— 空集的误差为 0。若把「最小化误差」当作第一"
        "优先级，选出的将是 0 个用户，那不是一次筛选。")
    add("")
    add(f"所以 ±{tolerance:.0%} 容差定义了「误差相近」：容差之内峰值误差按定义可比，"
        "真正携带信息的是子集能解释多少个真实用户。选择次序为：")
    add("")
    add("1. 可行 = 相对误差 ≤ 容差；")
    add("2. 可行集内用户数最多；")
    add("3. 再取 |误差| 最小；")
    add("4. 再取行业数最多；")
    add("5. 再取用户 id 字典序最小。")
    add("")

    add("## 搜索是穷举的，候选池是证明出来的")
    add("")
    add(f"候选池 = 自身峰值 ≤ 目标 ×(1+容差) = **{result['ceiling']:.4f} MW** 的用户，"
        f"共 **{len(result['pool'])}** 个。")
    add("")
    add("依据：若子集含有用户 u，则在 u 自身峰值所处时段，系统总量**至少**是 u 的贡献；"
        "其余各母线的贡献都非负，故系统峰值必然超过上限，该子集不可能落在容差内。"
        "因此非候选者不可入选，穷举候选池即穷举了整个可行空间。")
    add("")
    add("该论证的前提（所有曲线与原生形状逐点非负）由 `require_non_negative` 运行时检查；"
        "一旦出现负值即报错，而不是静默剪枝出一个「并非最大」的子集。")
    add("")

    add("## 每条母线的政策")
    add("")
    add("| 母线 | 政策 | 依据 |")
    add("|---|---|---|")
    for bus in bnl.ALL_BUSES:
        if bus == bnl.SLACK_BUS:
            add("| 0 | `preserve` | slack，不映射任何行业用户 |")
        elif bus in {r["destination_bus"] for r in listed}:
            users = [r["user_id"] for r in listed if r["destination_bus"] == bus]
            add(f"| {bus} | `replace` | 入选用户 {'、'.join(users)} 的绝对 MW 曲线"
                "替换原生负荷 |")
        else:
            add(f"| {bus} | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |")
    add("")
    add("政策由入选用户重新推导，**没有沿用**全量 41 用户场景的旧政策："
        "旧政策把若干母线标为 `replace`，而本子集已把它们清空。"
        "配置、账本与映射三者逐项一致由 `scenarios.single_day_case` 校验 —— "
        "它逐母线比对配置声明的政策与账本 manifest，不一致即 raise。")
    add("")

    add("## 无功负荷")
    add("")
    add("现有代码**按配置的功率因数推导无功**，不是保留原生无功、也不是别的规则："
        "`dispatch_socp`、`dispatch_ldf`、`diagnose_typical_day` 三处都用 "
        "`q = p · tanφ`。")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| 功率因数 | {result['reactive']['power_factor']} |")
    add("| 性质 | **仿真假设，不是实测值** |")
    add(f"| tanφ | {result['reactive']['tan_phi']:.6f} |")
    add(f"| 峰值处无功 | {result['reactive']['q_peak_mvar']:.4f} MVAr |")
    add(f"| 全日无功电量 | {result['reactive']['q_energy_mvarh']:.4f} MVArh |")
    add(f"| 基准无功 | {result['reactive']['benchmark_q_mvar']} MVAr |")
    add("")
    add(f"功率因数 `0.8503` 是按 `3.715 / √(3.715² + 2.3²)` 标定的，"
        f"即**基准有功正好 3.715 MW 时**无功等于基准的 2.3 MVAr。本子集的峰值是 "
        f"{peak:.4f} MW，所以那条标定理由在本场景下只是**近似**成立，"
        f"无功为 {result['reactive']['q_peak_mvar']:.4f} MVAr。")
    add("")
    add("**没有为了凑 2.3 MVAr 去调功率因数。** 那样做的代价是把一个声明调到一个数字上，"
        "与项目其余部分一贯拒绝的做法同类。")
    add("")

    add("## 数据与边界")
    add("")
    add("| | |")
    add("|---|---|")
    for label, value in source["fingerprints"].items():
        add(f"| {label} | `{value}` |")
    add(f"| 读数单位 | `{il.READING_UNIT}`，证据 `{il.READING_UNIT_EVIDENCE}` |")
    add(f"| 单位状态 | **`{il.unit_status()}`** |")
    add(f"| 选择规则版本 | `{rule_version()}` |")
    add("")
    add(f"- 读数单位依据是**已记录的项目裁定**（`{il.READING_UNIT_EVIDENCE}`），"
        f"外部核实**仍未取得**：`is_externally_verified` 恒为 `false`，"
        "任何数字都不得称为外部认证的结果。本报告产出的那一刻账本还是 audit；"
        "此后它已按该裁定重建为 formal 账本，见 `reports/formal_data_blockers.md`。")
    add(f"- **子集是针对 {date} 这一天拟合的。** 换一天，峰值与入选集合都会变；"
        "这不是可泛化的结论，也不描述任何其他日期。")
    add("- **用户→母线是 simulation mapping，不是地理接线。** 本轮只筛选既有映射，"
        "没有改动、重排或新增任何映射关系。")
    add("- 账本 manifest 的 `mapping_version` 写作 `industry_bus_map_v2`，"
        "而实际读取的是**派生**映射文件；精确绑定由 manifest 的 `mapping_sha256` "
        "（派生文件的摘要）承担。")
    add("- 重建会覆盖 `reports/bus_load_composition.csv` 中全量 41 用户的数值；"
        "41 用户账本本体仍在原目录，可用原命令重建。"
        "`reports/node_loads_summary.csv` 按场景累积，两个场景各占一行。")
    add(f"- 账本 manifest 的 `excluded_cells` 是 **{result['excluded_cells']}**，"
        "比全量场景的 1 大得多，但这不是数据质量问题：该字段统计的是"
        "「在**本场景的映射**下未被放置到任何母线的表计」，而派生映射只保留了"
        "入选用户。未被选中的用户的表计由此计入，它们是本场景不使用、"
        "而非有任何缺陷。")
    add("")

    add("## 逐用户")
    add("")
    add("| 用户 | 母线 | 行业 | 入选 | 自身峰值 MW | 理由 |")
    add("|---|---|---|---|---|---|")
    for row in rows:
        industry = source["industry"].get(row["user_id"], "")
        peak_text = ("" if row["own_peak_mw"] is None
                     else f"{row['own_peak_mw']:.6f}")
        add(f"| `{row['user_id']}` | {row['destination_bus'] or '—'} | {industry} | "
            f"{'**是**' if row['included'] else '否'} | {peak_text} | {row['reason']} |")
    add("")

    add("## 复现")
    add("")
    add("```bash")
    add(f"PYTHONPATH=src python src/select_industry_user_subset.py --date {date}")
    add("PYTHONPATH=src python src/build_industry_node_loads.py \\")
    add(f"    --scenario {scenario} --date {date} \\")
    add(f"    --mapping {SUBSET_MAPPING}")
    add("```")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(date: str, scenario: str, target: float, tolerance: float,
        mapping_path: Path, out_dir: Path, reports_dir: Path,
        records: Optional[Sequence[il.MeterDay]] = None) -> dict:
    """Select, then write everything the selection is evidenced by."""
    if records is None:
        records = il.flag_records(list(il.iter_meter_days()))

    map_rows = load_mapping(mapping_path)
    user_bus = mapped_users(map_rows)
    industry = industry_of(map_rows)
    per_user, meters, unreadable = bnl.per_user_curves(date, records)

    rated = bnl.native_bus_load()
    shape = bnl.native_shape()

    result = search(per_user, user_bus, industry, rated, shape, target, tolerance)
    if result["feasible"]:
        verify_maximal(result, per_user, user_bus, rated, shape, target, tolerance)

    rows = classify_users(per_user, user_bus,
                          result.get("selected", []), result["pool"],
                          result["ceiling"])

    power_factor = float(cfg.get_scenario_cfg(BASE_SCENARIO)["load_power_factor"])
    # An infeasible run still reports the day it did evaluate -- the all-preserve
    # baseline -- rather than leaving the figures blank.
    total = result["total_mw"]
    peak = float(total.max())
    energy = float(total.sum() * il.HOURS_PER_SLOT)
    result["peak_mw"] = peak
    result["peak_period"] = int(total.argmax())
    result["mean_mw"] = energy / (il.N_SLOTS * il.HOURS_PER_SLOT)
    result["energy_mwh"] = energy
    result["relative_error"] = (peak - target) / target
    result["reactive"] = reactive_account(power_factor, peak, energy)

    selection_csv = out_dir / "industry_user_subset_ieee33.csv"
    write_subset_csv(selection_csv, rows)
    selected = result["selected"]
    write_subset_mapping(SUBSET_MAPPING, map_rows, selected)
    policies = policies_for(selected, user_bus)

    ledger = bnl.build(scenario, date, out_dir / "node_loads",
                       SUBSET_MAPPING, formal=False, records=records)
    assert_policies_match_ledger(policies, ledger["manifest"])
    result["excluded_cells"] = ledger["manifest"]["excluded_cells"]
    composition, summary = bnl.write_reports(
        ledger, reports_dir)

    closest = result["closest"]
    source = {
        "industry": industry,
        "fingerprints": {
            "source workbook sha256": il.file_sha256(il.resolve_workbook()),
            "frozen mapping (v2) sha256": il.file_sha256(mapping_path),
            "derived subset mapping sha256": il.file_sha256(SUBSET_MAPPING),
            "subset table sha256": il.file_sha256(selection_csv),
            "ledger sha256": ledger["manifest"]["ledger_sha256"],
        },
    }
    manifest = {
        "scenario": scenario,
        "date": str(date),
        "rule_version": rule_version(),
        "rule_inputs": rule_inputs(),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target_peak_mw": target,
        "tolerance": tolerance,
        "tolerance_status": ("a project choice, not an IEEE33 parameter; no "
                             "source states it"),
        "peak_mw": peak,
        "peak_period": result["peak_period"],
        "relative_error": result["relative_error"],
        "mean_mw": result["mean_mw"],
        "energy_mwh": energy,
        "feasible": bool(result["feasible"]),
        "n_selected": len(selected),
        "selected_users": selected,
        "n_tied_at_the_best_key": result.get("n_tied", 0),
        "tied_subsets": result.get("tied_at_the_best_key", []),
        "closest_subset_error": closest[0] if closest else None,
        "n_candidates": len(result["pool"]),
        "candidate_ceiling_mw": result["ceiling"],
        "subsets_evaluated": result["evaluated"],
        "users": rows,
        "buses_with_selected_users": sorted({user_bus[u] for u in selected}),
        "policies": {str(b): p for b, p in sorted(policies.items())},
        "reactive": result["reactive"],
        "unit_status": il.unit_status(),
        "reading_unit": il.READING_UNIT,
        "reading_unit_evidence": il.READING_UNIT_EVIDENCE,
        "mapping_version": "industry_bus_map_v2",
        "derived_mapping": str(SUBSET_MAPPING),
        "ledger": str(ledger["path"]),
        "ledger_dataset_version": ledger["manifest"]["dataset_version"],
        "excluded_cells_in_ledger": ledger["manifest"]["excluded_cells"],
        "fingerprints": source["fingerprints"],
        "selection_algorithm": {
            "method": "exhaustive over the candidate pool, per bus",
            "candidate_bound": ("a user whose own peak exceeds target x "
                                "(1 + tolerance) cannot belong to any "
                                "tolerance-feasible subset"),
            "candidate_bound_premise": ("every contribution is non-negative; "
                                        "checked at runtime by "
                                        "require_non_negative"),
            "order": ["n_users desc", "abs relative error asc",
                      "n_industries desc", "sorted user ids asc"],
            "deterministic": True,
            "uses_price_or_lmp": False,
            "uses_revenue_or_rl": False,
        },
        "mapping_derivation": {
            "from": str(mapping_path),
            "to": str(SUBSET_MAPPING),
            "rule": "rows whose user_id is in the selected subset; no row "
                    "reordered, no bus changed",
        },
    }
    manifest_path = out_dir / "industry_user_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)
                             + "\n", encoding="utf-8")

    markdown = report_markdown(result, rows, date, scenario, target, tolerance,
                               source)
    (reports_dir / "industry_user_subset_selection.md").write_text(
        markdown, encoding="utf-8")

    return {"result": result, "rows": rows, "manifest": manifest,
            "ledger": ledger, "composition": composition, "summary": summary,
            "selection_csv": selection_csv, "manifest_path": manifest_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select the industry users whose day matches the IEEE33 "
                    "benchmark's own load scale.")
    parser.add_argument("--date", required=True,
                        help="the load date, ISO. Required: the subset is a "
                             "statement about one day.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--target-peak-mw", type=float, default=TARGET_PEAK_MW)
    parser.add_argument("--tolerance", type=float,
                        default=TARGET_PEAK_TOLERANCE)
    parser.add_argument("--mapping", default=str(V2_MAPPING))
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    reports_dir = Path(args.reports)
    outcome = run(args.date, args.scenario, args.target_peak_mw,
                  args.tolerance, Path(args.mapping), out_dir, reports_dir)
    result = outcome["result"]
    manifest = outcome["manifest"]

    print(f"date {args.date}   scenario {args.scenario}   "
          f"rule {manifest['rule_version']}")
    print(f"  target peak          {args.target_peak_mw} MW "
          f"(tolerance ±{args.tolerance:.0%}, a project choice)")
    print(f"  candidate pool       {manifest['n_candidates']} users "
          f"(own peak <= {manifest['candidate_ceiling_mw']:.4f} MW)")
    print(f"  subsets evaluated    {manifest['subsets_evaluated']:,}")
    if result["feasible"]:
        print(f"  selected             {manifest['n_selected']} users on "
              f"{len(manifest['buses_with_selected_users'])} buses")
        print(f"  final system peak    {manifest['peak_mw']:.6f} MW "
              f"at period {manifest['peak_period']}")
        print(f"  relative error       {manifest['relative_error']:+.4%}")
        print(f"  tied at that key     {manifest['n_tied_at_the_best_key']}")
    else:
        print("  NO SUBSET FALLS WITHIN THE TOLERANCE. The tolerance was not "
              "widened and no load was scaled.")
        print(f"  closest candidate    {manifest['closest_subset_error']:.4%} "
              "relative error")
    print(f"  mean / energy        {manifest['mean_mw']:.6f} MW / "
          f"{manifest['energy_mwh']:.4f} MWh")
    print(f"  reactive at peak     {manifest['reactive']['q_peak_mvar']:.4f} "
          f"MVAr (pf {manifest['reactive']['power_factor']}, an assumption)")
    print(f"\nwrote {outcome['selection_csv']}")
    print(f"wrote {SUBSET_MAPPING}")
    print(f"wrote {outcome['manifest_path']}")
    print(f"wrote {outcome['ledger']['path']}")
    print(f"wrote {reports_dir / 'industry_user_subset_selection.md'}")
    print(f"wrote {outcome['composition']}")
    print(f"wrote {outcome['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
