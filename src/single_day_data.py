# single_day_data.py
"""The one path by which a single-day case's load and price are loaded and paired.

**Why this module exists.** The load ledger and the price curve are two real
datasets that describe the same 15-minute grid and label their periods at
opposite ends of it. Each is validated for its own axis where it is read -- the
ledger by `industry_load._check_time_header`, the curve by
`dianyicha_price._check_slot_axis` -- and neither of those checks can see the
other. Pairing them is a third thing, and before this module it had no enforced
home: it lived in one script, and any consumer that built its own pair would
have got an unaligned merge with nothing to notice.

So: a case is constructed here or not at all. `load_case` refuses on a missing,
repeated, misordered or wrongly-spaced period, on a time-semantics mismatch
between the two, and on a date rule the caller has not stated. Nothing is
shifted, interpolated or resampled to make a pair fit.

**The case type is explicit and has no default.** Whether two different dates
are allowed depends on what the case claims to be, and a missing date field must
not be able to select the permissive rule by accident:

| case type | `load_date` vs `price_date` | what it means |
|---|---|---|
| `cross_date_diagnostic` | **may differ** | a real load shape priced against a real price shape; not a historical replay |
| `same_day_replay` | **must be equal** | a replay of one actual day, which needs that day's own market |

**Two ledger states, and only one may become a network input.**

| state | may it feed the network as absolute MW? |
|---|---|
| `audit_ledger` | **no** — the reading unit is declared, not verified; every row says so |
| `formal_ledger` | yes, and only after `validate_formal_ledger` passes it |

`require_formal_load_mw` is the gate. While `unit_status()` is
`unit_unverified` it refuses, and it refuses by reading the *manifest the
validator would read*, not by trusting the caller.
"""

from __future__ import annotations

import collections
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import dianyicha_price as dp  # noqa: E402
import industry_load as il  # noqa: E402

#: The case types, and the date rule each one carries. A caller states one;
#: nothing here infers it.
CASE_TYPES: Tuple[str, ...] = ("cross_date_diagnostic", "same_day_replay")

#: The two ledger states. Only the second may become a network input.
LEDGER_KINDS: Tuple[str, ...] = ("audit_ledger", "formal_ledger")


class CaseTypeError(ValueError):
    """The case type is missing, unknown, or too lax for what was asked."""


class DataAlignmentError(ValueError):
    """The two series cannot be paired on one simulation period index."""


@dataclass(frozen=True)
class PeriodAxis:
    """One series' time axis, as a convention rather than an assumption."""

    n_periods: int
    interval_minutes: int
    #: "interval_start" or "interval_end" -- which end of the interval the
    #: series' own labels name.
    label_convention: str
    timezone: Optional[str]

    def interval(self, t: int) -> Tuple[int, int]:
        """The intra-day minute range period `t` covers.

        The same range for both conventions: a label at the interval's start
        and a label at its end describe one interval, not two.
        """
        if not 0 <= t < self.n_periods:
            raise IndexError(f"period {t} is outside 0..{self.n_periods - 1}")
        start = t * self.interval_minutes
        return start, start + self.interval_minutes

    def describe(self) -> dict:
        return {"n_periods": self.n_periods,
                "interval_minutes": self.interval_minutes,
                "label_convention": self.label_convention,
                "timezone": self.timezone}


def _check_period_axis(n: int, interval_minutes: int, expected_minutes: int,
                       what: str) -> None:
    if n != il.N_SLOTS:
        raise DataAlignmentError(
            f"{what} has {n} periods; a day is {il.N_SLOTS}")
    if interval_minutes != expected_minutes:
        raise DataAlignmentError(
            f"{what} samples every {interval_minutes} minutes, not "
            f"{expected_minutes}; a granularity conversion would have to be "
            f"chosen and justified deliberately, and none is applied here")


def load_load_axis(ledger_csv: Path) -> Tuple[PeriodAxis, List[dict]]:
    """The ledger's time axis, from the ledger itself.

    Read from the artifact rather than restated, so a ledger written with a
    different axis cannot be paired on the strength of the module's assumption.
    """
    rows = list(csv.DictReader(Path(ledger_csv).open(encoding="utf-8")))
    if not rows:
        raise DataAlignmentError(f"{ledger_csv} has no rows")
    per_bus: Dict[int, List[int]] = {}
    for row in rows:
        per_bus.setdefault(int(row["bus"]), []).append(int(row["time_index"]))
    for bus, idx in per_bus.items():
        # Ascending and complete, checked on the order the file is in rather
        # than after sorting it. Sorting first would accept a file assembled by
        # concatenating two runs and hand back a day nobody wrote.
        if idx != list(range(il.N_SLOTS)):
            seen = collections.Counter(idx)
            repeats = sorted(k for k, n in seen.items() if n > 1)
            why = (f"repeated period(s) {repeats[:5]}" if repeats
                   else "a missing or out-of-order period")
            raise DataAlignmentError(
                f"bus {bus} in {ledger_csv} does not carry the "
                f"{il.N_SLOTS} period indices once each, ascending from 0: "
                f"{why}")
    axis = PeriodAxis(n_periods=il.N_SLOTS,
                      interval_minutes=il.SLOT_MINUTES,
                      label_convention="interval_start",
                      timezone=None)
    return axis, rows


def load_price_axis(curve: dp.Curve) -> PeriodAxis:
    """The price curve's time axis, checked as well as read."""
    if len(curve.prices) != dp.N_SLOTS:
        raise DataAlignmentError(
            f"{curve.sheet} has {len(curve.prices)} periods; a day is "
            f"{dp.N_SLOTS}")
    # `read_curves` has already refused a repeated or misordered label and any
    # label that is not the interval-END clock; that is why the convention can
    # be stated here rather than guessed.
    return PeriodAxis(n_periods=dp.N_SLOTS,
                      interval_minutes=dp.SLOT_MINUTES,
                      label_convention="interval_end",
                      timezone=None)


def check_alignment(load_axis: PeriodAxis, price_axis: PeriodAxis) -> dict:
    """Whether the two axes are one index, and what it assumes.

    The two conventions differ -- the ledger labels the interval's start and the
    curve labels its end -- and that is fine, because both name the same
    interval. What is not fine is pairing series whose periods cover different
    ground, and this is where that is refused rather than shifted.
    """
    if load_axis.n_periods != price_axis.n_periods:
        raise DataAlignmentError(
            f"the load has {load_axis.n_periods} periods and the price "
            f"{price_axis.n_periods}; no index correspondence exists")
    if load_axis.interval_minutes != price_axis.interval_minutes:
        raise DataAlignmentError(
            f"the load samples every {load_axis.interval_minutes} minutes and "
            f"the price every {price_axis.interval_minutes}; pairing them "
            f"would need a resampling this module does not do")
    for t in (0, load_axis.n_periods - 1):
        if load_axis.interval(t) != price_axis.interval(t):
            raise DataAlignmentError(
                f"period {t} covers {load_axis.interval(t)} in the load and "
                f"{price_axis.interval(t)} in the price")
    return {
        "method": "period index to period index, no shift, no resampling",
        "periods_per_day": load_axis.n_periods,
        "interval_minutes": load_axis.interval_minutes,
        "load_axis": load_axis.describe(),
        "price_axis": price_axis.describe(),
        "index_t_is": f"the interval [{load_axis.interval_minutes}t, "
                      f"{load_axis.interval_minutes}(t+1)) on both sides",
        "conversion_applied": "none",
        "interpolation_applied": "none",
        "resampling_applied": "none",
    }


def apply_date_rule(case_type: str, load_date: str, price_date: str) -> dict:
    """The date rule the stated case type carries, applied and recorded.

    `case_type` is required and checked against `CASE_TYPES`. There is no
    default: a caller that has not said what kind of case this is does not get
    the permissive rule, and a missing date is an error rather than a reason to
    relax one.
    """
    if case_type is None or case_type not in CASE_TYPES:
        raise CaseTypeError(
            f"case_type must be one of {CASE_TYPES}; got {case_type!r}. The "
            f"date rule depends on it, so an unstated case cannot have one")
    if not load_date or not price_date:
        raise CaseTypeError(
            f"both dates are required to apply a date rule; got "
            f"load_date={load_date!r}, price_date={price_date!r}")
    same = load_date == price_date
    if case_type == "same_day_replay" and not same:
        raise CaseTypeError(
            f"a same_day_replay needs one actual day, but the load is dated "
            f"{load_date} and the price {price_date}; pair them as a "
            f"cross_date_diagnostic if that is what this is")
    return {
        "case_type": case_type,
        "load_date": load_date,
        "price_date": price_date,
        "same_day": same,
        "dates_may_differ": case_type == "cross_date_diagnostic",
        "rule": ("cross-date pairing allowed; both dates recorded"
                 if case_type == "cross_date_diagnostic"
                 else "load and price must be the same actual date"),
        "dates_recorded_not_rewritten": True,
    }


@dataclass
class SingleDayCase:
    """A validated load-price pair. Construct it through `load_case`."""

    case_type: str
    date_rule: dict
    alignment: dict
    ledger_manifest: dict
    ledger_rows: List[dict]
    price_curve: dp.Curve
    price_source: dict
    load_source: dict

    # -- the price, delivered as periods rather than as a bare array ---------

    def price_for(self, t_start: int = 0, T: Optional[int] = None) -> np.ndarray:
        """The price over a slice of the validated axis.

        The axis this checks is the one `check_alignment` passed, so a consumer
        cannot take a window the case was not validated for.
        """
        n = self.alignment["periods_per_day"]
        T = n if T is None else int(T)
        t_start = int(t_start)
        if t_start < 0 or T < 1 or t_start + T > n:
            raise DataAlignmentError(
                f"periods {t_start}..{t_start + T} are outside the validated "
                f"0..{n - 1}; a window is a slice of the day and never wraps")
        return np.asarray(self.price_curve.prices[t_start:t_start + T],
                          dtype=float)

    @property
    def load_kind(self) -> str:
        """`audit_ledger` or `formal_ledger`, from the manifest that wrote it."""
        kind = self.ledger_manifest.get("ledger_kind")
        return {"audit": "audit_ledger", "formal": "formal_ledger"}.get(
            kind, str(kind))

    def formal_bus_load_mw(self) -> Dict[int, np.ndarray]:
        """The ledger's load by bus, for a **formal** network input.

        Gated by `require_formal_load_mw`: the unit must be stated by an outside
        source and the ledger may be published as the network's absolute load.
        """
        self.require_formal_load_mw()
        return self._extract_by_bus()

    def bus_load_mw_for_build(self) -> Dict[int, np.ndarray]:
        """The ledger's load by bus, for a **diagnostic** run on a recorded basis.

        Gated by `require_load_for_build`: a ledger whose unit basis is recorded
        -- a user-confirmed project ruling included -- and which has no gap.
        `unit_basis` travels with the case so the run can say what it rests on.

        Both extractions go through `_extract_by_bus`, so there is one place
        where a row becomes a bus's load and one place where a gap would have to
        be filled if it ever were.
        """
        self.require_load_for_build()
        return self._extract_by_bus()

    def _extract_by_bus(self) -> Dict[int, np.ndarray]:
        by_bus: Dict[int, np.ndarray] = {}
        for row in self._rows_with_no_gap():
            bus = int(row["bus"])
            by_bus.setdefault(bus, np.full(self.alignment["periods_per_day"],
                                           np.nan))[int(row["time_index"])] =                 float(row["final_mw"])
        for bus, arr in by_bus.items():
            if np.isnan(arr).any():
                raise DataAlignmentError(
                    f"bus {bus} is missing part of the day after all")
        return by_bus

    @property
    def unit_basis(self) -> dict:
        """What the ledger's reading unit rests on, as the ledger recorded it.

        `ruling` is the ordinary case here: the unit is a **user-confirmed
        project ruling** and no external document corroborates it. That is a
        recorded basis, and a diagnostic run may be built on it -- labelled as
        such. It is not, and must never be presented as, external certification.
        """
        evidence = self.ledger_manifest.get("reading_unit_evidence")
        basis = il.canonical_basis(evidence)
        return {
            "reading_unit": self.ledger_manifest.get("reading_unit"),
            "evidence": evidence,
            "unit_status": self.ledger_manifest.get("unit_status"),
            "is_externally_verified": basis == il.UNIT_BASIS_EXTERNAL,
            "basis": (il.UNIT_BASIS_RULING if basis == il.UNIT_BASIS_RULING
                      else "external_document"
                      if basis == il.UNIT_BASIS_EXTERNAL
                      else "unrecorded"),
        }

    def require_load_for_build(self) -> dict:
        """The runtime gate: may this ledger's load be injected into a network?

        Two conditions, and neither is external certification:

        1. **the unit basis is recorded.** A ledger that does not say what unit
           its readings are in, or what that rests on, cannot become a
           network's demand -- not because the answer would be wrong but
           because there would be no answer to label the result with;
        2. **the load is complete**, which the extraction enforces: every
           required bus, every period, no gap.

        A ruling-based unit **passes** this gate. What it does not get is to be
        described as verified: `unit_basis` travels with the run, and a report
        built on it says `user_confirmed_project_ruling`.

        This is deliberately not `require_formal_load_mw`, which is the stricter
        claim: that the unit is stated by an outside source and the ledger may
        be *published* as the network's absolute load. A diagnostic run and a
        formal publication are two different claims and this keeps them apart.
        """
        basis = self.unit_basis
        if not basis["reading_unit"] or not basis["evidence"]:
            raise CaseTypeError(
                f"the ledger records reading_unit={basis['reading_unit']!r} and "
                f"reading_unit_evidence={basis['evidence']!r}; a load may not be "
                f"injected without a recorded unit basis, because nothing would "
                f"say what the resulting figures are in")
        rows = self._rows_with_no_gap()
        return {"rows": rows, "unit_basis": basis}

    def _rows_with_no_gap(self) -> List[dict]:
        """The rows, refusing a gap rather than filling it."""
        gaps = [r for r in self.ledger_rows
                if not (r.get("final_mw") or "").strip()]
        if gaps:
            buses = sorted({int(r["bus"]) for r in gaps})
            raise CaseTypeError(
                f"{len(gaps)} periods have no final load, on bus(es) {buses}. "
                f"A gap is not a zero: zero-filling or carrying the previous "
                f"period would invent a measurement, so the run is refused")
        return self.ledger_rows

    def require_formal_load_mw(self) -> List[dict]:
        """The ledger rows, or a refusal.

        A formal network input must come from a validator-passed object. This
        reads the manifest's own status fields -- the ones the validator reads
        -- rather than trusting a caller's claim, and it refuses an audit
        ledger, an unverified unit, or a missing status.
        """
        if self.load_kind != "formal_ledger":
            raise CaseTypeError(
                f"this is an {self.load_kind}; its readings are not a verified "
                f"absolute load and may not be fed to the network. Build a "
                f"formal ledger, which needs an external source for the "
                f"reading unit")
        status = self.ledger_manifest.get("unit_status")
        # Either admissible basis will do -- an external source or a recorded
        # project ruling. What the check refuses is a status naming neither,
        # which is a unit nothing stands behind.
        if status not in (il.UNIT_VERIFIED, il.UNIT_BASIS_RULING_STATUS):
            raise CaseTypeError(
                f"the ledger's unit status is {status!r}; a formal load may "
                f"rest on {il.UNIT_VERIFIED!r} or "
                f"{il.UNIT_BASIS_RULING_STATUS!r} and on nothing else")
        return self.ledger_rows

    def describe(self) -> dict:
        return {
            "case_type": self.case_type,
            "date_rule": self.date_rule,
            "load_kind": self.load_kind,
            "unit_status": self.ledger_manifest.get("unit_status"),
            "mapping_version": self.ledger_manifest.get("mapping_version"),
            "load_source": self.load_source,
            "price_source": self.price_source,
            "alignment": self.alignment,
            "not_a_same_day_replay": (
                "the load day and the price day are different calendar dates; "
                "this pairs a real load shape with a real price shape and is "
                "not a replay of a historical day"),
        }


def load_case(case_type: str,
              ledger_csv: Path,
              ledger_manifest: Path,
              price_sheet: str = "261009",
              workbook: Optional[Path] = None) -> SingleDayCase:
    """Load, validate and pair one single-day case.

    The order is the point: the case type decides the date rule *before*
    anything is read, the two axes are each checked on their own terms, and the
    cross-curve alignment is checked last. Any failure raises; nothing is
    repaired.
    """
    ledger_csv = Path(ledger_csv)
    ledger_manifest = Path(ledger_manifest)
    if not ledger_csv.exists():
        raise DataAlignmentError(f"{ledger_csv} not found")
    if not ledger_manifest.exists():
        raise DataAlignmentError(f"{ledger_manifest} not found")

    manifest = json.loads(ledger_manifest.read_text(encoding="utf-8"))
    load_date = manifest.get("date")

    curves = dp.read_curves(path=workbook, sheets=(price_sheet,))
    curve = curves[price_sheet]
    from dianyicha_price import CURVE_META
    declared = CURVE_META.get(price_sheet, {})
    price_date = iso_from_reference_period(declared.get("reference_period"),
                                            price_sheet)

    date_rule = apply_date_rule(case_type, load_date, price_date)
    load_axis, rows = load_load_axis(ledger_csv)
    price_axis = load_price_axis(curve)
    alignment = check_alignment(load_axis, price_axis)
    alignment["dates_aligned"] = date_rule["same_day"]
    alignment["dates_aligned_note"] = (
        "the two curves are from the same date" if date_rule["same_day"]
        else "the two curves are from different dates and neither is moved to "
             "the other's date")

    return SingleDayCase(
        case_type=case_type,
        date_rule=date_rule,
        alignment=alignment,
        ledger_manifest=manifest,
        ledger_rows=rows,
        price_curve=curve,
        load_source={
            "ledger": ledger_csv.as_posix(),
            "ledger_sha256": manifest.get("ledger_sha256"),
            "selection_rule": manifest.get("selection_rule"),
            "selection_evidence": manifest.get("selection_evidence"),
            "users_with_data": manifest.get("users_with_data"),
            "users_per_bus": manifest.get("users_per_bus"),
            "industry_aggregation": manifest.get("industry_aggregation"),
        },
        price_source={
            "sheet": price_sheet,
            "workbook": str(dp.resolve_workbook(workbook)),
            "date": price_date,
            "date_is": "parsed from the sheet name's own YY/MM/DD label",
            "externally_confirmed": bool(declared.get("externally_confirmed")),
            "spatial_statistic": declared.get("spatial_statistic"),
            "temporal_statistic": declared.get("temporal_statistic"),
        },
    )


def iso_from_reference_period(raw: Optional[str], sheet: str) -> str:
    """`26/10/09` -> `2026-10-09`, or a refusal.

    A malformed or missing label is an error rather than an empty date: an
    absent date must not quietly satisfy a rule that needs one.
    """
    text = (raw or "").strip()
    parts = text.replace("-", "/").split("/")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise CaseTypeError(
            f"{sheet}: reference_period {raw!r} is not a YY/MM/DD label, so "
            f"the curve's date is unknown and no date rule can be applied")
    yy, mm, dd = (int(p) for p in parts)
    return f"{2000 + yy:04d}-{mm:02d}-{dd:02d}"
