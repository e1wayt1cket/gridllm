# multiday_dataset.py
"""Calendar labels and real load-to-price pairing, without reading the data.

The multi-day study needs two decisions that must be testable before a 67 MiB
workbook or a Gurobi licence is involved: what kind of day a date is, and which
real price curve may be paired with it. Both live here as pure functions.

**A pairing is a claim about two real dates.** The only admissible price curves
are single-day curves (`temporal_statistic == "single_day"`); a 7/30/180-day
mean is not a day and is refused rather than relabelled. A curve is assigned to
at most one load date, and a load date to at most one curve. When no curve is
left the row says so -- it is not filled by sharing a curve across days.

**The calendar is a declared table, not a library.** The raw load data covers
2021-2022 and the table below covers the public holidays and make-up workdays
in those two years. A date outside it is `ordinary`; extending the study to
later years means extending the table deliberately, not inferring holidays
from weekdays.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: (start, end, anchor, class, name). The anchor is day 0; `holiday_offset` is
#: the signed distance from it. The anchor is the festival itself, so the
#: Spring Festival period starts at offset -1 (New Year's Eve).
HOLIDAY_PERIODS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("2021-01-01", "2021-01-03", "2021-01-01", "new_year", "New Year"),
    ("2021-02-11", "2021-02-17", "2021-02-12", "spring_festival",
     "Spring Festival"),
    ("2021-04-03", "2021-04-05", "2021-04-04", "qingming", "Qingming"),
    ("2021-05-01", "2021-05-05", "2021-05-01", "labour_day", "Labour Day"),
    ("2021-06-12", "2021-06-14", "2021-06-14", "dragon_boat",
     "Dragon Boat Festival"),
    ("2021-09-19", "2021-09-21", "2021-09-21", "mid_autumn",
     "Mid-Autumn Festival"),
    ("2021-10-01", "2021-10-07", "2021-10-01", "national_day", "National Day"),
    ("2022-01-01", "2022-01-03", "2022-01-01", "new_year", "New Year"),
    ("2022-01-31", "2022-02-06", "2022-02-01", "spring_festival",
     "Spring Festival"),
    ("2022-04-03", "2022-04-05", "2022-04-04", "qingming", "Qingming"),
    ("2022-04-30", "2022-05-04", "2022-05-01", "labour_day", "Labour Day"),
    ("2022-06-03", "2022-06-05", "2022-06-03", "dragon_boat",
     "Dragon Boat Festival"),
    ("2022-09-10", "2022-09-12", "2022-09-10", "mid_autumn",
     "Mid-Autumn Festival"),
    ("2022-10-01", "2022-10-07", "2022-10-01", "national_day", "National Day"),
)

#: Weekend days moved into the working week. They are not holidays: the grid
#: is expected to carry a working-day load on them.
MAKEUP_WORKDAYS: Dict[str, Tuple[str, str]] = {
    "2021-02-07": ("spring_festival", "Spring Festival make-up workday"),
    "2021-02-20": ("spring_festival", "Spring Festival make-up workday"),
    "2021-04-25": ("labour_day", "Labour Day make-up workday"),
    "2021-05-08": ("labour_day", "Labour Day make-up workday"),
    "2021-09-18": ("mid_autumn", "Mid-Autumn make-up workday"),
    "2021-09-26": ("national_day", "National Day make-up workday"),
    "2021-10-09": ("national_day", "National Day make-up workday"),
    "2022-01-29": ("spring_festival", "Spring Festival make-up workday"),
    "2022-01-30": ("spring_festival", "Spring Festival make-up workday"),
    "2022-04-02": ("qingming", "Qingming make-up workday"),
    "2022-04-24": ("labour_day", "Labour Day make-up workday"),
    "2022-05-07": ("labour_day", "Labour Day make-up workday"),
    "2022-10-08": ("national_day", "National Day make-up workday"),
    "2022-10-09": ("national_day", "National Day make-up workday"),
}

SEASONS: Dict[int, str] = {
    1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn",
    11: "autumn", 12: "winter",
}

def as_date(value) -> dt.date:
    """A `date` from a `date`, a `datetime`, or an ISO date string."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def _holiday_period(date: dt.date):
    for start, end, anchor, klass, name in HOLIDAY_PERIODS:
        if as_date(start) <= date <= as_date(end):
            return {"holiday_class": klass, "holiday_name": name,
                    "holiday_offset": (date - as_date(anchor)).days}
    return None


def calendar_label(day) -> dict:
    """The season, weekday and holiday facts of one date.

    `day_type` is the dimension a split stratifies on: `holiday` for a public
    holiday, `weekend` for a non-holiday Saturday/Sunday, and `weekday` for a
    working day -- including a make-up workday that falls on a weekend.
    """
    date = as_date(day)
    holiday = _holiday_period(date)
    makeup = MAKEUP_WORKDAYS.get(date.isoformat())
    is_weekend = date.weekday() >= 5

    if holiday is not None:
        holiday_class = holiday["holiday_class"]
        holiday_name = holiday["holiday_name"]
        holiday_offset: Optional[int] = holiday["holiday_offset"]
        is_workday = False
        day_type = "holiday"
    elif makeup is not None:
        holiday_class, holiday_name = makeup
        holiday_offset = None
        is_workday = True
        day_type = "weekday"
    else:
        holiday_class, holiday_name = "ordinary", ""
        holiday_offset = None
        is_workday = not is_weekend
        day_type = "weekend" if is_weekend else "weekday"

    return {
        "season": SEASONS[date.month],
        "weekday": date.strftime("%A"),
        "weekday_index": date.weekday(),
        "is_weekend": is_weekend,
        "is_workday": is_workday,
        "holiday_class": holiday_class,
        "holiday_name": holiday_name,
        "holiday_offset": holiday_offset,
        "day_type": day_type,
    }


@dataclass(frozen=True)
class PriceCurve:
    """A real dated price curve, reduced to what pairing needs.

    The builder constructs these from `dianyicha_price.Curve`. `price_date` is
    the actual date the curve covers; a curve with no such date cannot be
    paired, whatever its label says.
    """
    price_date: dt.date
    curve_id: str
    source: str
    curve_sha256: str
    temporal_statistic: str
    externally_confirmed: bool = False


def _same_month_day(a: dt.date, b: dt.date) -> bool:
    return a.month == b.month and a.day == b.day and a != b


def _candidate(load: dt.date, price: dt.date) -> Optional[Tuple[int, int, str, str]]:
    """(rule index, distance, rule name, pair quality) or None."""
    if load == price:
        return (0, 0, "exact_date", "exact")
    load_label = calendar_label(load)
    price_label = calendar_label(price)
    if _same_month_day(load, price):
        # A make-up workday and an ordinary workday are both `weekday`, but
        # they are not the same day type: the first sits inside a holiday
        # period and the second does not. Compare the holiday class too.
        same_type = (load_label["day_type"] == price_label["day_type"]
                     and load_label["holiday_class"]
                     == price_label["holiday_class"])
        quality = ("same_month_day_same_day_type" if same_type
                   else "same_month_day_day_type_mismatch")
        return (1, abs(load.year - price.year),
                "same_month_day_across_years", quality)
    if (load_label["holiday_class"] == price_label["holiday_class"]
            and load_label["holiday_class"] not in ("ordinary", "makeup_workday")
            and load_label["holiday_offset"] is not None
            and price_label["holiday_offset"] is not None):
        gap = abs(load_label["holiday_offset"] - price_label["holiday_offset"])
        if gap <= 3:
            return (2, gap, "holiday_relative", "holiday_relative")
    gap_days = abs((load - price).days)
    if gap_days <= 7 and load.weekday() == price.weekday():
        return (3, gap_days, "weekday_fallback", "weekday_fallback")
    return None


def pair_prices(load_dates: Iterable, curves: Sequence[PriceCurve]) -> dict:
    """Pair each load date with at most one real single-day price curve.

    Returns `{"pairs": [...], "rejected_curves": [...]}`. Each pair has the
    load date, the price date (or None), the rule, year gap, quality and the
    reason a date is unpaired. A curve is used at most once; when a date has
    only curves that are already taken its reason is
    `price_curve_reuse_conflict`, not a duplicated price.
    """
    loads = sorted({as_date(d) for d in load_dates})
    rejected = [
        {"curve_id": c.curve_id, "price_date": c.price_date.isoformat(),
         "reason": "not_a_single_day_curve"}
        for c in curves if c.temporal_statistic != "single_day"
    ]
    usable = [c for c in curves if c.temporal_statistic == "single_day"]
    used: set = set()
    pairs: List[dict] = []

    for load in loads:
        candidates = []
        for curve in usable:
            if curve.price_date in used:
                continue
            hit = _candidate(load, curve.price_date)
            if hit is None:
                continue
            rule_index, distance, rule, quality = hit
            candidates.append((rule_index, distance, str(curve.curve_id),
                               curve, rule, quality))
        if not candidates:
            any_hit = any(_candidate(load, c.price_date) is not None
                          for c in usable)
            reason = ("price_curve_reuse_conflict" if any_hit
                      else "no_single_day_curve")
            pairs.append({
                "load_date": load.isoformat(),
                "price_date": "",
                "price_pair_rule": "",
                "price_year_gap": "",
                "price_source": "",
                "price_curve_sha256": "",
                "pair_quality": "unpaired",
                "pair_reason": reason,
            })
            continue

        _, _, _, curve, rule, quality = min(candidates, key=lambda row: row[:3])
        used.add(curve.price_date)
        pairs.append({
            "load_date": load.isoformat(),
            "price_date": curve.price_date.isoformat(),
            "price_pair_rule": rule,
            "price_year_gap": abs(load.year - curve.price_date.year),
            "price_source": curve.source,
            "price_curve_sha256": curve.curve_sha256,
            "pair_quality": quality,
            "pair_reason": "",
        })

    return {"pairs": pairs, "rejected_curves": rejected}
