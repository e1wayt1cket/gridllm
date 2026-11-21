# v2_price_matching.py
"""One load day against the real dated price curves that exist for it.

The matching question is narrower than the pairing question: for a given load
date, is there a **same-date**, same-granularity, traceable real single-day
price curve whose periods cover the same intervals? A window mean is not a day,
an adjacent date is not this date, a synthetic curve is not a market
observation, and a curve with an unknown unit or date cannot be matched.

The functions here are pure so the classification can be tested without the
workbook. The audit script builds `PriceDay` objects from the files it finds.

Classes:
* `exact` -- same date, 96 x 15 min, real and traceable, axes align.
* `partial` -- same date and real, but period count/granularity or axis
  alignment is not usable.
* `unverifiable` -- same date but the source kind, units, date or traceability
  cannot establish a legitimate match.
* `missing` -- no same-date real single-day curve at all.

No adjacent-date or window-mean substitution is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import single_day_data as sdd

EXACT = "exact"
PARTIAL = "partial"
MISSING = "missing"
UNVERIFIABLE = "unverifiable"
MATCH_CLASSES = (EXACT, PARTIAL, MISSING, UNVERIFIABLE)


@dataclass(frozen=True)
class PriceDay:
    """One price curve reduced to what a date match needs."""

    source_id: str
    path: str
    sheet: str
    date: Optional[str]
    periods: int
    interval_minutes: int
    label_convention: str
    units: str
    currency: str
    tax_status: str
    kind: str
    source_kind: str
    traceable: bool
    externally_confirmed: bool = False
    spatial_statistic: str = "unspecified"
    provenance: Dict[str, object] = field(default_factory=dict)

    def axis(self) -> sdd.PeriodAxis:
        return sdd.PeriodAxis(
            n_periods=self.periods,
            interval_minutes=self.interval_minutes,
            label_convention=self.label_convention,
            timezone=self.provenance.get("timezone"),
        )


def _load_axis(periods: int, interval_minutes: int,
               label_convention: str) -> sdd.PeriodAxis:
    return sdd.PeriodAxis(
        n_periods=periods,
        interval_minutes=interval_minutes,
        label_convention=label_convention,
        timezone=None,
    )


def _base(price: Optional[PriceDay] = None) -> dict:
    return {
        "match_class": MISSING,
        "reason": "",
        "price_source_id": price.source_id if price else "",
        "price_sheet": price.sheet if price else "",
        "price_date": price.date if price else "",
        "price_periods": price.periods if price else "",
        "price_interval_minutes": price.interval_minutes if price else "",
        "price_label_convention": price.label_convention if price else "",
        "price_units": price.units if price else "",
        "price_currency": price.currency if price else "",
        "price_tax_status": price.tax_status if price else "",
        "price_kind": price.kind if price else "",
        "price_source_kind": price.source_kind if price else "",
        "price_traceable": price.traceable if price else "",
        "price_externally_confirmed": price.externally_confirmed if price else "",
        "same_date": bool(price and price.date is not None),
        "same_granularity": "",
        "interval_alignment": "",
    }


def match_day(load_date: str, price_days: Sequence[PriceDay],
              load_periods: int = 96, load_interval_minutes: int = 15,
              load_label_convention: str = "interval_start") -> dict:
    """Classify one load date against the available price curves.

    The date equality is not optional. A curve for another date is never
    selected, even if it is the only curve and even if its month-day matches.
    """
    same_date = [p for p in price_days if p.date == load_date]
    real = [p for p in same_date
            if p.kind == "single_day"
            and p.source_kind == "real_market_data"
            and p.traceable]
    if not real:
        out = _base()
        if same_date:
            out.update({
                "match_class": UNVERIFIABLE,
                "reason": ("a same-date price artefact exists, but it is not "
                           "a traceable real single-day market curve"),
                "price_source_id": same_date[0].source_id,
                "price_sheet": same_date[0].sheet,
                "price_date": same_date[0].date or "",
                "price_kind": same_date[0].kind,
                "price_source_kind": same_date[0].source_kind,
            })
        else:
            out.update({
                "match_class": MISSING,
                "reason": "no real single-day price curve with this actual date",
            })
        return out

    price = real[0]
    out = _base(price)
    out["same_date"] = True
    out["same_granularity"] = bool(
        price.periods == load_periods
        and price.interval_minutes == load_interval_minutes)
    if price.units != "CNY/MWh" or price.currency != "CNY":
        out.update({
            "match_class": UNVERIFIABLE,
            "reason": (f"price unit/currency is {price.units}/{price.currency}; "
                       "the load study is CNY/MWh"),
        })
        return out
    if not out["same_granularity"]:
        out.update({
            "match_class": PARTIAL,
            "reason": (f"price has {price.periods} periods of "
                       f"{price.interval_minutes} min; the load has "
                       f"{load_periods} periods of {load_interval_minutes} min"),
        })
        return out
    try:
        alignment = sdd.check_alignment(
            _load_axis(load_periods, load_interval_minutes,
                       load_label_convention),
            price.axis())
    except sdd.DataAlignmentError as exc:
        out.update({
            "match_class": PARTIAL,
            "reason": f"period axes do not align: {exc}",
        })
        return out
    out.update({
        "match_class": EXACT,
        "reason": "same actual date, same 96 x 15-min periods, axes align",
        "interval_alignment": alignment["method"],
    })
    return out


def count_classes(rows: Sequence[dict]) -> Dict[str, int]:
    counts = {name: 0 for name in MATCH_CLASSES}
    for row in rows:
        counts[row["match_class"]] += 1
    return counts
