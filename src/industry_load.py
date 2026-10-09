# industry_load.py
"""Industry daily load shapes from the metered Jiangsu workbook.

The workbook (`docs/16类行业用户.xlsx`) holds 15-minute meter readings for 73
users across 16 industries and two calendar years. This module turns it into
**normalised 96-point daily shapes, one per industry** — nothing else.

Two products were considered and only one is built here.

*Shape* is what survives the data's defects. Every meter-day is normalised by
its own peak before any pooling, so the result describes the intraday profile
and carries no magnitude. That normalisation is also why the workbook's
unreliable `综合倍率` cannot contaminate it: `x / max|x|` is invariant to any
per-meter scale, so a ratio that is wrong by a factor of a million moves the
shape not at all. The ratio is only consulted where a magnitude is genuinely
being judged, in `verify_ratios`, and there it is reported rather than trusted.

*Absolute MW* is deliberately not built. `综合倍率` reconciles with the declared
运行容量 for some meters and not others (5G基站 implies 2.3–2.6x its rating at a
ratio of 1; several 化工/印刷 meters carry a ratio that implies hundreds of times
theirs), so a global conversion to MW would be a fabricated quantity wearing a
measured one's clothes. `absolute_mw` refuses by default and says why.

Aggregation is hierarchical and equal-weighted at every level, because the
workbook's row counts are not a sample design: 商业综合体2 contributes 15,867
rows and 电解铝 contributes 221, and averaging rows would make the first
thirteen industries' worth of weight. A meter-day is the observation, a user-day
is the mean of its meters, a user is the mean of its days, and an industry is
the mean of its users.

Nothing here imputes, clamps, sign-flips or repairs. Every departure from a
clean complete day is recorded as a flag and carried to a report; the aggregation
drops flagged observations and counts what it dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Contract with the workbook
# ---------------------------------------------------------------------------

N_SLOTS = 96
SLOT_MINUTES = 15

#: Metadata columns, in workbook order, before the 96 slot columns.
META_COLUMNS: Tuple[str, ...] = (
    "序号", "总户号", "地点", "用户名称", "行业", "合同容量", "运行容量",
    "户号", "电表局编号", "综合倍率", "功率相序", "数据日期",
)
COL_ORDER = 0
COL_ACCOUNT = 1
COL_LOCATION = 2
COL_USER_NAME = 3
COL_INDUSTRY = 4
COL_CONTRACT = 5
COL_RUNNING = 6
COL_HOUSEHOLD = 7
COL_METER = 8
COL_RATIO = 9
COL_PHASE = 10
COL_DATE = 11
COL_SLOT0 = 12

PHASE_ACTIVE_POWER = "有功功率"
PV_INDUSTRY = "分布式光伏"

# ---------------------------------------------------------------------------
# Quality flags and the thresholds behind them
# ---------------------------------------------------------------------------

#: A row whose identity cannot be established at all.
FLAG_EMPTY_IDENTITY = "empty_identity"
#: Any slot missing.
FLAG_MISSING_SLOTS = "missing_slots"
#: Every present slot is zero.
FLAG_ALL_ZERO = "all_zero"
#: The synthesised placeholder `[1, 2, 3, 3, 3, 0, ...]`, sum 12.
FLAG_PLACEHOLDER = "placeholder"
#: Negative values on a consumption industry.
FLAG_NEGATIVE_LOAD = "negative_load"
#: Negative values on 分布式光伏, where reverse flow is expected.
FLAG_NEGATIVE_PV = "negative_pv"
#: Power phase is not 有功功率; the unit convention is not established.
FLAG_OTHER_PHASE = "other_phase"
#: `综合倍率` absent or unparseable.
FLAG_RATIO_MISSING = "ratio_missing"
#: `综合倍率` far from its industry's median — reported, never corrected.
FLAG_RATIO_SUSPECT = "ratio_suspect"
#: Implied peak contradicts the declared 运行容量.
FLAG_CAPACITY_INCONSISTENT = "capacity_inconsistent"
#: Row repeats a (account, meter, date) already seen with the same profile.
FLAG_DUPLICATE_ROW = "duplicate_row"
#: Row repeats a (account, meter, date) but disagrees about the profile.
FLAG_DUPLICATE_CONFLICT = "duplicate_conflict"

#: Excluded from the shape pool. A flag here means the observation is dropped.
EXCLUDING_FLAGS: Tuple[str, ...] = (
    FLAG_EMPTY_IDENTITY,
    FLAG_MISSING_SLOTS,
    FLAG_ALL_ZERO,
    FLAG_PLACEHOLDER,
    FLAG_OTHER_PHASE,
)

#: Reported but not by itself disqualifying; the profile may still be a valid
#: shape. Kept separate so a report can distinguish "dropped" from "noted".
ADVISORY_FLAGS: Tuple[str, ...] = (
    FLAG_NEGATIVE_LOAD,
    FLAG_NEGATIVE_PV,
    FLAG_RATIO_MISSING,
    FLAG_RATIO_SUSPECT,
    FLAG_CAPACITY_INCONSISTENT,
    FLAG_DUPLICATE_ROW,
    FLAG_DUPLICATE_CONFLICT,
)

#: Excluded from the shape pool but *kept* in the exclusions report.
#:
#: A redundant repeat of a meter-day is not a bad reading, so it is not an
#: "excluding" flag in the sense the report uses — but it must not be counted
#: twice either. Counting it would quietly re-weight a user by how often their
#: meter happened to be re-exported, which is precisely the distortion the
#: hierarchical aggregation exists to remove. A *conflicting* repeat is not
#: dropped: two different profiles for one meter-day is a real disagreement,
#: and picking one would invent a reading.
SHAPE_DROP_FLAGS: Tuple[str, ...] = EXCLUDING_FLAGS + (FLAG_DUPLICATE_ROW,)

#: The placeholder's leading signature and its exact sum.
PLACEHOLDER_HEAD: Tuple[float, ...] = (1.0, 2.0, 3.0, 3.0, 3.0)
PLACEHOLDER_SUM = 12.0

#: A meter's ratio is called an outlier when it departs from its industry's
#: median by more than this factor. Descriptive only — never applied as a fix.
RATIO_OUTLIER_FACTOR = 100.0
#: Implied peak (kW) above this multiple of 运行容量 is a contradiction.
CAPACITY_PEAK_TOLERANCE = 1.5

#: An industry curve is called reliable only at or above both of these.
MIN_USERS_FOR_RELIABLE = 3
MIN_DAYS_FOR_RELIABLE = 30


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeterDay:
    """One meter's one day, already reduced to a profile plus its verdict."""

    account: str
    meter: str
    user_name: str
    industry: str
    location: str
    excel_row: int
    date: object
    values: Tuple[Optional[float], ...]
    ratio: Optional[float]
    running_capacity: Optional[float]
    phase: str
    flags: Tuple[str, ...]

    @property
    def excluded(self) -> bool:
        return any(f in EXCLUDING_FLAGS for f in self.flags)

    @property
    def user_key(self) -> str:
        """Stable anonymous identifier for the account."""
        return anon_id("U", self.account)

    @property
    def meter_key(self) -> str:
        return anon_id("M", f"{self.account}|{self.meter}")


@dataclass(frozen=True)
class IndustryShape:
    """One industry's standard 96-point shape and how much stands behind it."""

    industry: str
    shape: Tuple[float, ...]
    n_users: int
    n_meters: int
    n_days: int
    n_observations: int
    excluded_flags: Tuple[str, ...]
    method: str
    reliable: bool
    reliability_note: str


@dataclass
class IndustryShapeSet:
    """Every industry shape, plus the provenance of the run that made it."""

    shapes: Dict[str, IndustryShape] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.shapes)

    def industries(self) -> List[str]:
        return sorted(self.shapes)

    def shape_of(self, industry: str) -> np.ndarray:
        return np.asarray(self.shapes[industry].shape, dtype=float)


def anon_id(prefix: str, raw: str) -> str:
    """A stable, non-reversible identifier for a report.

    Reports must locate a record without naming the account holder, so the
    account and meter numbers are replaced by a hash prefix. The Excel row
    number travels alongside for audit.
    """
    digest = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{digest}"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def resolve_workbook(path: Optional[str] = None) -> Path:
    """Locate the workbook, defaulting to the one under `docs/`."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"workbook not found: {p}")
        return p
    root = Path(__file__).resolve().parent.parent
    candidate = root / "docs" / "16类行业用户.xlsx"
    if not candidate.exists():
        raise FileNotFoundError(
            f"workbook not found: {candidate}. Pass an explicit path.")
    return candidate


def detect_sheet(workbook, explicit: Optional[str] = None) -> str:
    """Pick the sheet holding the readings.

    The workbook ships one data sheet, but the name is not a contract: the
    layout is, so the sheet is chosen by finding the header that matches
    `META_COLUMNS` and a 96-column time block.
    """
    if explicit is not None:
        if explicit not in workbook.sheetnames:
            raise KeyError(
                f"sheet {explicit!r} not in {workbook.sheetnames}")
        return explicit
    for name in workbook.sheetnames:
        ws = workbook[name]
        header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True),
                      None)
        if header is None:
            continue
        if tuple(header[:len(META_COLUMNS)]) == META_COLUMNS:
            return name
    raise KeyError(
        f"no sheet carries the expected header {META_COLUMNS}; "
        f"sheets are {workbook.sheetnames}")


def _check_time_header(header: Sequence) -> None:
    """Assert the 96 slot columns are exactly 00:00..23:45 ascending."""
    import datetime as _dt

    slots = header[COL_SLOT0:COL_SLOT0 + N_SLOTS]
    if len(slots) != N_SLOTS:
        raise ValueError(
            f"expected {N_SLOTS} slot columns, found {len(slots)}")
    base = _dt.date(2000, 1, 1)
    for i, cell in enumerate(slots):
        if not isinstance(cell, _dt.time):
            raise ValueError(
                f"slot column {i} header is {type(cell).__name__}, not a time")
        want = (_dt.datetime.combine(base, _dt.time(0, 0))
                + _dt.timedelta(minutes=SLOT_MINUTES * i)).time()
        if cell != want:
            raise ValueError(
                f"slot column {i} is {cell}, expected {want}; the time axis "
                "is not the assumed 00:00..23:45 at 15-minute steps")


def _to_float(raw) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    text = str(raw).strip()
    if text in ("", "/", "-", "—", "NA", "N/A", "nan", "NaN"):
        return None
    try:
        return float(text)
    except ValueError:
        # The workbook stores some readings as Excel-flavoured strings such as
        # '.0047' or '-.0012'; those are valid numbers, anything else is not.
        cleaned = text.replace(" ", "")
        try:
            return float(cleaned)
        except ValueError:
            return None


def iter_meter_days(path: Optional[str] = None,
                    sheet: Optional[str] = None,
                    limit: Optional[int] = None) -> Iterator[MeterDay]:
    """Stream the workbook as `MeterDay` records.

    `openpyxl` is used in read-only, values-only mode: the workbook is 64 MiB
    and must not be loaded into memory whole, and formulas are irrelevant here.
    The source file is never written to.
    """
    import openpyxl

    wb_path = resolve_workbook(path)
    wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
    try:
        name = detect_sheet(wb, sheet)
        ws = wb[name]
        header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
        _check_time_header(header)

        for offset, row in enumerate(ws.iter_rows(min_row=2, values_only=True)):
            if limit is not None and offset >= limit:
                break
            excel_row = offset + 2
            if row is None or all(c is None for c in row):
                continue
            padded = list(row) + [None] * (COL_SLOT0 + N_SLOTS - len(row))
            yield _make_meter_day(padded, excel_row)
    finally:
        wb.close()


def _make_meter_day(row: List, excel_row: int) -> MeterDay:
    values = tuple(_to_float(v) for v in row[COL_SLOT0:COL_SLOT0 + N_SLOTS])
    account = _text(row[COL_ACCOUNT])
    meter = _text(row[COL_METER])
    industry = _text(row[COL_INDUSTRY])
    return MeterDay(
        account=account,
        meter=meter,
        user_name=_text(row[COL_USER_NAME]),
        industry=industry,
        location=_text(row[COL_LOCATION]),
        excel_row=excel_row,
        date=row[COL_DATE],
        values=values,
        ratio=_to_float(row[COL_RATIO]),
        running_capacity=_to_float(row[COL_RUNNING]),
        phase=_text(row[COL_PHASE]),
        flags=(),
    )


def _text(raw) -> str:
    return "" if raw is None else str(raw).strip()


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------


def is_placeholder(values: Sequence[Optional[float]]) -> bool:
    """The synthesised `[1, 2, 3, 3, 3, 0, ...]` row, identified structurally.

    Matched on shape and sum together so that a genuine reading which happens to
    open `1, 2, 3, 3, 3` is not swept up: the placeholder's remaining 91 slots
    are all exactly zero and its total is exactly 12.
    """
    if len(values) != N_SLOTS:
        return False
    head = values[:len(PLACEHOLDER_HEAD)]
    if any(v is None for v in head):
        return False
    if tuple(float(v) for v in head) != PLACEHOLDER_HEAD:
        return False
    tail = values[len(PLACEHOLDER_HEAD):]
    if any(v is None for v in tail):
        return False
    return sum(float(v) for v in tail) == 0.0


def flag_meter_day(md: MeterDay,
                   industry_median_ratio: Optional[float] = None) -> MeterDay:
    """Attach quality flags. Values are inspected, never altered."""
    flags: List[str] = []

    if not md.account or not md.meter or md.date is None:
        flags.append(FLAG_EMPTY_IDENTITY)

    if any(v is None for v in md.values):
        flags.append(FLAG_MISSING_SLOTS)

    present = [v for v in md.values if v is not None]
    if present and all(v == 0.0 for v in present):
        flags.append(FLAG_ALL_ZERO)

    if is_placeholder(md.values):
        flags.append(FLAG_PLACEHOLDER)

    is_pv = md.industry == PV_INDUSTRY
    if any(v < 0.0 for v in present):
        flags.append(FLAG_NEGATIVE_PV if is_pv else FLAG_NEGATIVE_LOAD)

    if md.phase and md.phase != PHASE_ACTIVE_POWER:
        # The unit convention for anything but active power is not established
        # here, so the reading cannot be read as a load.
        flags.append(FLAG_OTHER_PHASE)

    if md.ratio is None or md.ratio <= 0.0:
        flags.append(FLAG_RATIO_MISSING)
    elif (industry_median_ratio is not None and industry_median_ratio > 0.0
          and (md.ratio / industry_median_ratio > RATIO_OUTLIER_FACTOR
               or industry_median_ratio / md.ratio > RATIO_OUTLIER_FACTOR)):
        flags.append(FLAG_RATIO_SUSPECT)

    if _capacity_contradicted(md, present):
        flags.append(FLAG_CAPACITY_INCONSISTENT)

    return MeterDay(**{**md.__dict__, "flags": tuple(flags)})


def _capacity_contradicted(md: MeterDay,
                           present: Sequence[float]) -> bool:
    """Does the implied peak contradict the declared 运行容量?

    Implied peak in kW is `max|reading| * 综合倍率 * (60 / SLOT_MINUTES)`. The
    comparison is only meaningful when both the ratio and the rating parse and
    the rating is positive, so anything else is left unflagged rather than
    guessed at. This is a *flag*: the ratio is never rewritten to make it agree.
    """
    if md.ratio is None or md.ratio <= 0.0:
        return False
    if md.running_capacity is None or md.running_capacity <= 0.0:
        return False
    if not present:
        return False
    peak_kw = max(abs(v) for v in present) * md.ratio * (60.0 / SLOT_MINUTES)
    return peak_kw > CAPACITY_PEAK_TOLERANCE * md.running_capacity


def mark_duplicates(records: Iterable[MeterDay]) -> List[MeterDay]:
    """Collapse repeated `(account, meter, date)` rows.

    The workbook repeats meter-days: 12,974 keys appear more than once and
    13,532 redundant rows follow. Nearly all repeats are byte-identical and are
    simply dropped with a counter. The handful that disagree about the profile
    are real contradictions, so both profiles are kept as separate
    observations and flagged — silently picking one would invent a reading.
    """
    seen: Dict[Tuple[str, str, object], Tuple[float, ...]] = {}
    out: List[MeterDay] = []
    for md in records:
        key = (md.account, md.meter, md.date)
        profile = tuple(md.values)
        if key not in seen:
            seen[key] = profile
            out.append(md)
            continue
        if seen[key] == profile:
            out.append(MeterDay(
                **{**md.__dict__,
                   "flags": tuple(md.flags) + (FLAG_DUPLICATE_ROW,)}))
            continue
        out.append(MeterDay(
            **{**md.__dict__,
               "flags": tuple(md.flags) + (FLAG_DUPLICATE_CONFLICT,)}))
    return out


def flag_records(records: Iterable[MeterDay]) -> List[MeterDay]:
    """Flag every record, using each industry's own median ratio as reference."""
    materialised = list(records)

    ratios: Dict[str, List[float]] = defaultdict(list)
    for md in materialised:
        if md.ratio is not None and md.ratio > 0.0:
            ratios[md.industry].append(md.ratio)
    medians = {ind: float(np.median(v)) for ind, v in ratios.items() if v}

    flagged = [flag_meter_day(md, medians.get(md.industry))
               for md in materialised]
    return mark_duplicates(flagged)


# ---------------------------------------------------------------------------
# Shape aggregation
# ---------------------------------------------------------------------------


def normalise(values: Sequence[float]) -> Optional[np.ndarray]:
    """Scale a profile by its own largest magnitude.

    `max|x|` rather than `max(x)` so that an all-negative day — a PV meter whose
    every slot is reverse flow — scales to a shape instead of being flipped. For
    a profile that is non-negative, as most are, the two are identical. Signs
    are preserved: this scales, it does not rectify.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size != N_SLOTS:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    peak = float(np.max(np.abs(arr)))
    if peak <= 0.0:
        return None
    return arr / peak


def build_shapes(records: Iterable[MeterDay],
                 date_key=None,
                 min_users: int = MIN_USERS_FOR_RELIABLE,
                 min_days: int = MIN_DAYS_FOR_RELIABLE,
                 method: str = "hierarchical_mean") -> IndustryShapeSet:
    """Aggregate flagged records into one standard shape per industry.

    The hierarchy is meter-day, then user-day, then user, then industry, each
    level an equal-weighted mean of the level below. Equal weighting is the
    point: it stops a user with many meters, or many recorded days, from
    outvoting an industry's other users.

    A user-day is the mean of that user's qualified meters *for that date*, so
    a day with three meters recorded still counts once, exactly like a day with
    one.
    """
    if date_key is None:
        date_key = lambda md: md.date  # noqa: E731

    # industry -> user -> date -> list of normalised meter profiles
    nested: Dict[str, Dict[str, Dict[object, List[np.ndarray]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meter_ids: Dict[str, set] = defaultdict(set)
    day_keys: Dict[str, set] = defaultdict(set)
    excl_counts: Dict[str, Counter] = defaultdict(Counter)

    for md in records:
        dropped = [f for f in md.flags if f in SHAPE_DROP_FLAGS]
        if dropped:
            excl_counts[md.industry][dropped[0]] += 1
            continue
        # Grouping is by industry, so 分布式光伏 is never pooled with a
        # consumption industry: it gets its own entry and its own curve. That
        # separation is structural, not a filter that could be forgotten.
        shape = normalise(md.values)
        if shape is None:
            excl_counts[md.industry]["degenerate_profile"] += 1
            continue
        nested[md.industry][md.user_key][date_key(md)].append(shape)
        meter_ids[md.industry].add(md.meter_key)
        day_keys[md.industry].add((md.user_key, date_key(md)))

    shapes: Dict[str, IndustryShape] = {}
    for industry in sorted(nested):
        user_curves = []
        n_obs = 0
        for user, by_date in nested[industry].items():
            day_curves = []
            for _, profiles in sorted(by_date.items(), key=lambda kv: str(kv[0])):
                day_curves.append(np.mean(np.stack(profiles), axis=0))
                n_obs += len(profiles)
            if day_curves:
                user_curves.append(np.mean(np.stack(day_curves), axis=0))
        if not user_curves:
            continue

        industry_curve = np.mean(np.stack(user_curves), axis=0)
        industry_curve = normalise(industry_curve)
        if industry_curve is None:
            excl_counts[industry]["degenerate_industry_curve"] += 1
            continue

        n_users = len(user_curves)
        n_days = len(day_keys[industry])
        reliable = (n_users >= min_users and n_days >= min_days)
        note = _reliability_note(n_users, n_days, min_users, min_days)

        shapes[industry] = IndustryShape(
            industry=industry,
            shape=tuple(float(v) for v in industry_curve),
            n_users=n_users,
            n_meters=len(meter_ids[industry]),
            n_days=n_days,
            n_observations=n_obs,
            excluded_flags=tuple(sorted(excl_counts[industry])),
            method=method,
            reliable=reliable,
            reliability_note=note,
        )

    return IndustryShapeSet(shapes=shapes, meta={
        "method": method,
        "min_users_for_reliable": min_users,
        "min_days_for_reliable": min_days,
        "n_slots": N_SLOTS,
        "excluded_flag_totals": {
            ind: dict(c) for ind, c in sorted(excl_counts.items())},
    })


def _reliability_note(n_users: int, n_days: int,
                      min_users: int, min_days: int) -> str:
    if n_users >= min_users and n_days >= min_days:
        return "ok"
    missing = []
    if n_users < min_users:
        missing.append(f"users {n_users} < {min_users}")
    if n_days < min_days:
        missing.append(f"days {n_days} < {min_days}")
    return "limited: " + "; ".join(missing)


# ---------------------------------------------------------------------------
# Absolute load — refused unless verified
# ---------------------------------------------------------------------------


def verify_ratios(records: Iterable[MeterDay]) -> Dict[str, object]:
    """Check `综合倍率` against 运行容量, per meter.

    Returns the evidence, not a verdict about the whole file: a meter is
    `consistent` when its implied peak lands under its declared rating. Whatever
    this returns, no ratio is modified, because a ratio that disagrees with a
    rating is as likely to mean the rating is nominal as the ratio is wrong.
    """
    per_meter: Dict[Tuple[str, str], Dict[str, object]] = {}
    for md in records:
        if md.ratio is None or md.running_capacity is None:
            continue
        present = [v for v in md.values if v is not None]
        if not present:
            continue
        peak_kw = (max(abs(v) for v in present) * md.ratio
                   * (60.0 / SLOT_MINUTES))
        key = (md.account, md.meter)
        entry = per_meter.setdefault(key, {
            "industry": md.industry,
            "meter": md.meter_key,
            "ratio": md.ratio,
            "running_capacity": md.running_capacity,
            "peak_kw": 0.0,
            "peak_ratio": 0.0,
        })
        if peak_kw > entry["peak_kw"]:
            entry["peak_kw"] = peak_kw
            entry["peak_ratio"] = (peak_kw / md.running_capacity
                                   if md.running_capacity > 0 else float("inf"))

    consistent = sum(1 for e in per_meter.values()
                     if e["peak_ratio"] <= CAPACITY_PEAK_TOLERANCE)
    return {
        "meters_checked": len(per_meter),
        "meters_consistent": consistent,
        "meters_inconsistent": len(per_meter) - consistent,
        "per_meter": per_meter,
    }


def absolute_mw(shapes: IndustryShapeSet,
                bus_map: Dict[int, str],
                bus_ratings_mw: Dict[int, float],
                verified_industries: Sequence[str] = ()) -> Dict[int, np.ndarray]:
    """Convert shapes to MW per bus — **refused** unless every industry is
    explicitly verified.

    This exists so the refusal is a function rather than a convention. The
    workbook's ratio reconciles with declared capacity for some meters and not
    others, so a global MW conversion would produce a number with no measurement
    behind it. Callers that have actually verified an industry pass it in
    `verified_industries`; everything else raises.
    """
    unverified = sorted({ind for ind in bus_map.values()
                         if ind not in set(verified_industries)})
    if unverified:
        raise ValueError(
            "absolute MW refused: unit/ratio/ownership not verified for "
            f"{unverified}. Verify against 运行容量 and pass those industries "
            "in verified_industries, or use the normalised shapes instead.")
    out: Dict[int, np.ndarray] = {}
    for bus, industry in bus_map.items():
        if industry not in shapes.shapes:
            raise KeyError(f"no shape for industry {industry!r} (bus {bus})")
        out[bus] = shapes.shape_of(industry) * float(bus_ratings_mw[bus])
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def shapes_to_frame(shapes: IndustryShapeSet):
    """Long-format rows: one per (industry, slot)."""
    import pandas as pd

    rows = []
    for industry in shapes.industries():
        s = shapes.shapes[industry]
        for t, value in enumerate(s.shape):
            rows.append({
                "industry": industry,
                "time_index": t,
                "time_start": f"{t * SLOT_MINUTES // 60:02d}:"
                              f"{t * SLOT_MINUTES % 60:02d}",
                "shape": value,
                "n_users": s.n_users,
                "n_meters": s.n_meters,
                "n_days": s.n_days,
                "method": s.method,
                "reliable": s.reliable,
                "reliability_note": s.reliability_note,
            })
    return pd.DataFrame(rows)


def exclusions_to_frame(records: Iterable[MeterDay]):
    """Every flagged record, traceable to its Excel row, without identities."""
    import pandas as pd

    rows = []
    for md in records:
        if not md.flags:
            continue
        present = [v for v in md.values if v is not None]
        rows.append({
            "excel_row": md.excel_row,
            "user_id": md.user_key,
            "meter_id": md.meter_key,
            "industry": md.industry,
            "location": md.location,
            "date": str(md.date) if md.date is not None else "",
            "flags": ";".join(md.flags),
            "excluded": md.excluded,
            "n_slots_present": len(present),
            "n_slots_negative": sum(1 for v in present if v < 0),
            "ratio": md.ratio,
            "running_capacity": md.running_capacity,
        })
    return pd.DataFrame(rows)


def summary_to_frame(shapes: IndustryShapeSet):
    import pandas as pd

    rows = []
    for industry in shapes.industries():
        s = shapes.shapes[industry]
        arr = np.asarray(s.shape, dtype=float)
        rows.append({
            "industry": industry,
            "n_users": s.n_users,
            "n_meters": s.n_meters,
            "n_days": s.n_days,
            "n_observations": s.n_observations,
            "n_slots": len(s.shape),
            "shape_min": float(arr.min()),
            "shape_max": float(arr.max()),
            "shape_mean": float(arr.mean()),
            "peak_slot": int(np.argmax(arr)),
            "trough_slot": int(np.argmin(arr)),
            "reliable": s.reliable,
            "reliability_note": s.reliability_note,
            "excluded_flag_kinds": ";".join(s.excluded_flags),
            "method": s.method,
        })
    return pd.DataFrame(rows)


def profiles_to_frame(records: Iterable[MeterDay]):
    """Every qualified meter-day curve, for the parquet artefact."""
    import pandas as pd

    rows = []
    for md in records:
        if any(f in SHAPE_DROP_FLAGS for f in md.flags):
            continue
        shape = normalise(md.values)
        if shape is None:
            continue
        row = {
            "user_id": md.user_key,
            "meter_id": md.meter_key,
            "industry": md.industry,
            "location": md.location,
            "date": str(md.date) if md.date is not None else "",
            "excel_row": md.excel_row,
        }
        row.update({f"t{t:02d}": float(v) for t, v in enumerate(shape)})
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(shapes: IndustryShapeSet,
                  records: Sequence[MeterDay],
                  out_dir: str = "data/processed"):
    """Write the five phase-1 artefacts, creating the directory if absent."""
    import pandas as pd

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written = {}

    typical = directory / "industry_typical_profiles.csv"
    shapes_to_frame(shapes).to_csv(typical, index=False, encoding="utf-8")
    written["typical_profiles"] = typical

    summary = directory / "industry_profile_summary.csv"
    summary_to_frame(shapes).to_csv(summary, index=False, encoding="utf-8")
    written["profile_summary"] = summary

    excl = directory / "industry_exclusions.csv"
    exclusions_to_frame(records).to_csv(excl, index=False, encoding="utf-8")
    written["exclusions"] = excl

    parquet_path = directory / "industry_daily_profiles.parquet"
    try:
        profiles_to_frame(records).to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "parquet output failed; no fallback format is substituted because "
            "that would silently change the artefact contract. "
            f"Underlying error: {exc}") from exc
    written["daily_profiles"] = parquet_path

    return written


#: Long-format columns `load_shapes_csv` requires of the artifact.
_SHAPE_CSV_COLUMNS = ("industry", "time_index", "shape", "n_users", "n_meters",
                      "n_days", "method", "reliable", "reliability_note")


def load_shapes_csv(path: str) -> IndustryShapeSet:
    """Read `industry_typical_profiles.csv` back into shapes.

    A scenario consumes the committed artifact rather than the 67 MiB workbook,
    so this is the reader the scenario path uses. It validates what it reads —
    96 ordered slots per industry, finite values — because an artifact that has
    been edited or truncated must fail here rather than quietly reshape a day.

    `n_observations` is the artifact's own column when present and zero
    otherwise; the shapes artifact does not otherwise carry it, and inventing a
    value from `n_days` would state something the file never measured.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    missing = [c for c in _SHAPE_CSV_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: shapes artifact is missing column(s) {missing}; "
            f"it carries {list(frame.columns)}")

    shapes: Dict[str, IndustryShape] = {}
    for industry, group in frame.groupby("industry", sort=True):
        group = group.sort_values("time_index")
        slots = [int(t) for t in group["time_index"]]
        if slots != list(range(N_SLOTS)):
            raise ValueError(
                f"{path}: industry {industry!r} does not carry exactly the "
                f"{N_SLOTS} slot indices 0..{N_SLOTS - 1}; found {len(slots)}")
        arr = group["shape"].to_numpy(dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{path}: industry {industry!r} has non-finite shape values")

        head = group.iloc[0]
        shapes[str(industry)] = IndustryShape(
            industry=str(industry),
            shape=tuple(float(v) for v in arr),
            n_users=int(head["n_users"]),
            n_meters=int(head["n_meters"]),
            n_days=int(head["n_days"]),
            n_observations=(int(head["n_observations"])
                            if "n_observations" in group.columns else 0),
            excluded_flags=(),
            method=str(head["method"]),
            reliable=bool(head["reliable"]),
            reliability_note=str(head["reliability_note"]),
        )

    return IndustryShapeSet(shapes=shapes, meta={"shapes_file": str(path)})


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def quality_report(shapes: IndustryShapeSet,
                   records: Sequence[MeterDay],
                   workbook: Path,
                   sheet: str,
                   ratio_evidence: Dict[str, object]) -> str:
    """The markdown quality report."""
    flag_totals = Counter()
    per_industry_excl: Dict[str, Counter] = defaultdict(Counter)
    for md in records:
        for f in md.flags:
            flag_totals[f] += 1
            if f in SHAPE_DROP_FLAGS:
                per_industry_excl[md.industry][f] += 1

    n_records = len(records)
    n_excluded = sum(1 for md in records
                     if any(f in SHAPE_DROP_FLAGS for f in md.flags))
    industries_in_data = sorted({md.industry for md in records})

    lines: List[str] = []
    add = lines.append

    add("# Industry Load Quality Report")
    add("")
    add("Produced by `src/industry_load.py`. Records what was measured; no "
        "value in the source workbook was modified, imputed or clamped.")
    add("")

    add("## Source")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| workbook | `{workbook}` |")
    add(f"| sheet | `{sheet}` |")
    add(f"| slots per row | {N_SLOTS} x {SLOT_MINUTES} min (00:00..23:45) |")
    add(f"| data rows read | {n_records:,} |")
    add(f"| industries present | {len(industries_in_data)} |")
    add("")

    add("## Field mapping")
    add("")
    add("| column | index | role |")
    add("|---|---|---|")
    for i, name in enumerate(META_COLUMNS, start=1):
        role = {
            "总户号": "account identity",
            "电表局编号": "meter identity",
            "用户名称": "user label (never published)",
            "行业": "industry class",
            "运行容量": "declared rating, for consistency checks",
            "综合倍率": "meter multiplier, **not trusted**",
            "功率相序": "power phase",
            "数据日期": "day key",
        }.get(name, "-")
        add(f"| `{name}` | {i} | {role} |")
    add(f"| slot columns | {COL_SLOT0 + 1}..{COL_SLOT0 + N_SLOTS} "
        f"| 96 x 15-minute readings |")
    add("")

    add("## Cleaning rules")
    add("")
    add("`drop` means the record is kept out of the shape pool; `note` means it "
        "is counted in the report and the record is still usable as a shape. "
        "No rule alters a reading.")
    add("")
    add("| flag | effect | rule |")
    add("|---|---|---|")
    rules = [
        (FLAG_EMPTY_IDENTITY, "drop",
         "account, meter or date is blank"),
        (FLAG_MISSING_SLOTS, "drop", "at least one of the 96 slots is blank"),
        (FLAG_ALL_ZERO, "drop", "every present slot is exactly 0"),
        (FLAG_PLACEHOLDER, "drop",
         "opens `1,2,3,3,3`, remaining slots all 0, total exactly 12"),
        (FLAG_OTHER_PHASE, "drop",
         f"功率相序 is not {PHASE_ACTIVE_POWER}"),
        (FLAG_NEGATIVE_LOAD, "note",
         f"negative reading on a non-{PV_INDUSTRY} industry"),
        (FLAG_NEGATIVE_PV, "note",
         f"negative reading on {PV_INDUSTRY} (reverse flow expected)"),
        (FLAG_RATIO_MISSING, "note", "综合倍率 blank or not a positive number"),
        (FLAG_RATIO_SUSPECT, "note",
         f"综合倍率 departs from its industry median by more than "
         f"{RATIO_OUTLIER_FACTOR:g}x"),
        (FLAG_CAPACITY_INCONSISTENT, "note",
         f"implied peak exceeds {CAPACITY_PEAK_TOLERANCE:g}x 运行容量"),
        (FLAG_DUPLICATE_ROW, "drop",
         "same (account, meter, date) repeated with an identical profile; "
         "dropped so no observation is counted twice"),
        (FLAG_DUPLICATE_CONFLICT, "note",
         "same (account, meter, date) repeated with a **different** profile; "
         "both are kept, because choosing one would invent a reading"),
    ]
    for flag, effect, rule in rules:
        add(f"| `{flag}` | {effect} | {rule} |")
    add("")

    add("## Flags observed")
    add("")
    add("| flag | records |")
    add("|---|---|")
    for flag, _effect, _rule in rules:
        add(f"| `{flag}` | {flag_totals.get(flag, 0):,} |")
    add(f"| **total records** | **{n_records:,}** |")
    add("")
    add(f"Records kept out of the shape pool: **{n_excluded:,}** "
        f"({100.0 * n_excluded / n_records:.2f}% of rows). "
        "Advisory flags do not remove a record; they are counted here and the "
        "record still contributes its shape.")
    add("")

    add("## Dropped records by industry")
    add("")
    add("| industry | " + " | ".join(f"`{f}`" for f in SHAPE_DROP_FLAGS) + " |")
    add("|---" * (len(SHAPE_DROP_FLAGS) + 1) + "|")
    for ind in industries_in_data:
        cells = [str(per_industry_excl[ind].get(f, 0))
                 for f in SHAPE_DROP_FLAGS]
        add(f"| {ind} | " + " | ".join(cells) + " |")
    add("")

    add("## Ratio and capacity evidence")
    add("")
    add("`综合倍率` is reported, never corrected. A ratio that disagrees with "
        "运行容量 is as likely to mean the rating is nominal as the ratio is "
        "wrong, so the disagreement is recorded as a flag.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| meters checked | {ratio_evidence['meters_checked']:,} |")
    add(f"| meters whose implied peak fits 运行容量 "
        f"(<= {CAPACITY_PEAK_TOLERANCE:g}x) | "
        f"{ratio_evidence['meters_consistent']:,} |")
    add(f"| meters whose implied peak contradicts it | "
        f"{ratio_evidence['meters_inconsistent']:,} |")
    add("")
    inconsistent = sorted(
        (e for e in ratio_evidence["per_meter"].values()
         if e["peak_ratio"] > CAPACITY_PEAK_TOLERANCE),
        key=lambda e: -e["peak_ratio"])[:15]
    if inconsistent:
        add("Worst contradictions (implied peak / 运行容量):")
        add("")
        add("| meter | industry | ratio | 运行容量 | implied peak kW | ratio to rating |")
        add("|---|---|---|---|---|---|")
        for e in inconsistent:
            add(f"| `{e['meter']}` | {e['industry']} | {e['ratio']:,.0f} | "
                f"{e['running_capacity']:,.0f} | {e['peak_kw']:,.0f} | "
                f"{e['peak_ratio']:,.1f}x |")
        add("")

    add("## Aggregation")
    add("")
    add("Method: hierarchical equal-weighted mean (`hierarchical_mean`).")
    add("")
    add("1. each qualified meter-day is scaled by its own `max|x|`, so the "
        "curve carries intraday shape and no magnitude;")
    add("2. a user-day is the mean of that user's qualified meters **for that "
        "date**, so a day with many meters still counts once;")
    add("3. a user curve is the mean of that user's qualified days;")
    add("4. an industry curve is the mean of its users, then rescaled to "
        "`max = 1`.")
    add("")
    add("Because step 1 divides by a per-meter scale, `综合倍率` cancels "
        "exactly and cannot distort any published shape — including the meters "
        "whose ratio is not credible.")
    add("")

    add("## Industry sample statistics")
    add("")
    add("| industry | users | meters | meter-days | observations | peak slot | reliable | note |")
    add("|---|---|---|---|---|---|---|---|")
    for industry in shapes.industries():
        s = shapes.shapes[industry]
        peak = int(np.argmax(np.asarray(s.shape)))
        add(f"| {industry} | {s.n_users} | {s.n_meters} | {s.n_days} | "
            f"{s.n_observations} | {peak:02d} | "
            f"{'yes' if s.reliable else '**no**'} | {s.reliability_note} |")
    add("")
    add(f"Reliability threshold: at least {MIN_USERS_FOR_RELIABLE} users and "
        f"{MIN_DAYS_FOR_RELIABLE} meter-days.")
    add("")
    missing = [i for i in industries_in_data if i not in shapes.shapes]
    if missing:
        add(f"**No shape produced for: {', '.join(missing)}.** These were "
            "present in the workbook but no qualified observation survived. "
            "No curve was interpolated or copied from another industry to fill "
            "the gap.")
        add("")

    add("## Reliability limits")
    add("")
    limited = [i for i in shapes.industries()
               if not shapes.shapes[i].reliable]
    if limited:
        for i in limited:
            add(f"- **{i}** — {shapes.shapes[i].reliability_note}. Treat its "
                "curve as indicative only.")
    else:
        add("- Every industry met the reliability threshold.")
    add("")
    add(f"- `{PV_INDUSTRY}` is a **generator**, not a load. Its shape is "
        "produced outside the load pool and must not be fed to a load bus; the "
        "network already carries its own PV devices.")
    add("- These are normalised shapes. Nothing here states an absolute MW "
        "level; see `absolute_mw`, which refuses until a unit and ratio "
        "verification exists.")
    add("")

    add("## Reproduction")
    add("")
    add("```bash")
    add("PYTHONPATH=src python src/industry_load.py \\")
    add('    --workbook "$(pwd)/docs/16类行业用户.xlsx" \\')
    add("    --out data/processed \\")
    add("    --report reports/industry_load_quality_report.md")
    add("```")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(workbook: Optional[str] = None,
        sheet: Optional[str] = None,
        out_dir: str = "data/processed",
        report_path: Optional[str] = "reports/industry_load_quality_report.md",
        make_figure: bool = True,
        limit: Optional[int] = None) -> IndustryShapeSet:
    """Read, flag, aggregate, and write the phase-1 artefacts."""
    wb_path = resolve_workbook(workbook)
    import openpyxl

    wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
    try:
        sheet_name = detect_sheet(wb, sheet)
    finally:
        wb.close()

    raw = list(iter_meter_days(str(wb_path), sheet_name, limit=limit))
    records = flag_records(raw)
    ratio_evidence = verify_ratios(records)

    shapes = build_shapes(records)
    shapes.meta.update({
        "workbook": str(wb_path),
        "sheet": sheet_name,
        "n_records": len(records),
        "flag_totals": dict(Counter(
            f for md in records for f in md.flags)),
    })

    write_outputs(shapes, records, out_dir)

    if make_figure:
        try:
            figure_path = Path(out_dir) / "industry_shapes.png"
            plot_shapes(shapes, figure_path)
            shapes.meta["figure"] = str(figure_path)
        except Exception as exc:  # pragma: no cover - plotting is optional
            shapes.meta["figure_error"] = str(exc)

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(
            quality_report(shapes, records, wb_path, sheet_name,
                           ratio_evidence),
            encoding="utf-8")
        shapes.meta["report"] = report_path

    return shapes


#: Fonts that can render the industry names, which are Chinese. Without one,
#: every label renders as empty boxes.
_CJK_FONTS = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
              "Source Han Sans SC", "Arial Unicode MS")


def _use_cjk_font(plt) -> Optional[str]:
    """Select an installed font that covers the industry names."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name] + list(
                plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def plot_shapes(shapes: IndustryShapeSet, path) -> None:
    """A 16-panel comparison of the standard curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_cjk_font(plt)
    industries = shapes.industries()
    cols, rows = 4, math.ceil(len(industries) / 4)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.0 * rows),
                             sharex=True)
    hours = np.arange(N_SLOTS) * SLOT_MINUTES / 60.0
    for ax, industry in zip(axes.ravel(), industries):
        s = shapes.shapes[industry]
        ax.plot(hours, s.shape, lw=1.6,
                color="#1f77b4" if s.reliable else "#d62728")
        ax.set_title(f"{industry}  ({s.n_users}u/{s.n_days}d"
                     f"{'' if s.reliable else ', limited'})", fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_ylim(0, 1.05)
    for ax in axes.ravel()[len(industries):]:
        ax.axis("off")
    for ax in axes.ravel()[-cols:]:
        ax.set_xlabel("hour", fontsize=8)
    fig.suptitle("Standard 96-slot industry shapes, normalised to daily peak "
                 "(red = below reliability threshold)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build standard 96-slot industry load shapes.")
    p.add_argument("--workbook", default=None,
                   help="path to the industry workbook")
    p.add_argument("--sheet", default=None, help="sheet name override")
    p.add_argument("--out", default="data/processed",
                   help="output directory for the data artefacts")
    p.add_argument("--report", default="reports/industry_load_quality_report.md",
                   help="path for the markdown quality report")
    p.add_argument("--no-figure", action="store_true",
                   help="skip the comparison figure")
    p.add_argument("--limit", type=int, default=None,
                   help="read only the first N data rows (for smoke checks)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    shapes = run(workbook=args.workbook, sheet=args.sheet, out_dir=args.out,
                 report_path=args.report, make_figure=not args.no_figure,
                 limit=args.limit)
    print(f"industries: {len(shapes)}")
    for industry in shapes.industries():
        s = shapes.shapes[industry]
        flag = "" if s.reliable else "  [LIMITED]"
        print(f"  {industry:<14} users={s.n_users:<3} meters={s.n_meters:<3} "
              f"days={s.n_days:<4} obs={s.n_observations:<6}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
