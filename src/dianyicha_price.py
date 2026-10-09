# dianyicha_price.py
"""Jiangsu day-ahead prices, read from the 电易电查 workbook.

`docs/日前价格261009.xlsx` carries four 96-point curves and nothing else:

| sheet | what its own column header calls it |
|---|---|
| `261009` | 日前均值（CNY/MWh） |
| `7days` | 日前均价 |
| `30days` | 日前均价 |
| `halfyear` | 日前均价 |

**All four are means.** The header on `261009` says so outright, and the other
three are labelled the same way. None of them is a single day's clearing, so
none may be described as a measured daily price. What separates them is the
averaging window, which is what `role` records; what they have in common is that
each period's value is an average over that window rather than one settlement.
The consequence is stated once here and carried in every `Curve`: a study built
on this file has four *price scenarios*, not four days, and cannot make a
statement about day-to-day variation or about generalisation across days.

**Time labels are interval-END, and the mapping is positional.** A label is the
*close* of the interval it belongs to: `00:15` closes the interval that runs
`00:00`-`00:15`, and `24:00` closes `23:45`-`24:00`. The simulator's period `t`
is `[t*15min, (t+1)*15min)`, so label `i` belongs to period `i` and the mapping
is the identity. Nothing is shifted.

That is asserted rather than assumed. The expected sequence is
`00:15, 00:30, ..., 24:00`, i.e. `15 * (i + 1)` minutes past midnight for slot
`i`, and a file that disagrees raises. A workbook using interval-START labels
(`00:00`-`23:45`) is therefore rejected with a message naming the convention
instead of being silently rotated by one period — a one-period rotation is
exactly the kind of error that survives review, because every number still looks
plausible.

The last label needs care: `24:00` does not exist as a `datetime.time`, so Excel
stores it as a `datetime` at midnight of its epoch date, `1900-01-01 00:00:00`.
It is accepted **only** in the final position; a midnight `datetime` anywhere
else is an error rather than a guess.

The workbook is read with `openpyxl` in read-only, values-only mode and is never
written to. No text or CSV copy is produced: this module is the reader, and the
provider in `wholesale_price.py` calls it directly.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Periods in a day, matching the simulator's horizon.
N_SLOTS = 96
SLOT_MINUTES = 15

#: Sheets, in the order a scenario enumerates them. The order is the `day_id`
#: mapping the provider uses, so it is part of the interface, not a convention.
SHEETS: Tuple[str, ...] = ("261009", "7days", "30days", "halfyear")

#: What each sheet's averaging window is. The role names the window; for the
#: three trailing curves that window is known, and for `261009` it is not.
ROLES: Dict[str, str] = {
    "261009": "same_day_or_same_day_mean",
    "7days": "trailing_7_day_mean",
    "30days": "trailing_30_day_mean",
    "halfyear": "trailing_180_day_mean",
}

#: Per-curve provenance, as far as it can be established from the file.
#:
#: `source_kind` says where the numbers came from; `aggregation` says what was
#: done to them. The two are different questions and the workbook answers them
#: differently for different sheets.
#:
#: **All four curves derive from real historical market data.** What separates
#: them is the aggregation: the three trailing sheets are means over 7, 30 and
#: 180 days of that data.
#:
#: **`261009`'s aggregation is unresolved, and this module does not resolve
#: it.** Two readings are consistent with the file and nothing in it
#: distinguishes them:
#:
#:   1. the 96 actual period prices for 26/10/09, one per settlement period;
#:   2. a *statistical single-day mean* -- one averaged value per period --
#:      which is what its column header says (`日前均值`, "day-ahead mean").
#:
#: The sheet name embeds the date; the header says mean. Neither is decisive,
#: and no cell comment, no extra sheet and no workbook property settles it. The
#: workbook was assembled by hand (its `docProps` record a personal author and a
#: creation date well before the sheet's own date), so it carries no platform
#: export metadata. Settling it needs the provider's own definition of the
#: column. Until then this reads `aggregation: "unresolved"` and every consumer
#: must treat it as *not confirmed* to be a single day's series -- the
#: conservative reading, because assuming (1) would license a cross-period
#: statement the data may not support.
#:
#: `window_start` / `window_end` are `None` for every curve: the workbook does
#: not state the trailing windows' end dates, and inventing them would be
#: worse than leaving them blank. See `reports/final_research_summary.md`.
CURVE_META: Dict[str, Dict[str, object]] = {
    "261009": {
        "source_kind": "real_market",
        "aggregation": "unresolved",
        "aggregation_readings": (
            "the 96 actual period prices for 26/10/09, or a statistical "
            "single-day mean over some set (the column header says 均值)"),
        "mean_window_days": None,
        "reference_period": "26/10/09",
        "window_start": None,
        "window_end": None,
        "header_label": "日前均值（CNY/MWh）",
    },
    "7days": {
        "source_kind": "real_market",
        "aggregation": "trailing_mean",
        "aggregation_readings": "a trailing mean over 7 days of real data",
        "mean_window_days": 7,
        "reference_period": None,
        "window_start": None,
        "window_end": None,
        "header_label": "日前均价",
    },
    "30days": {
        "source_kind": "real_market",
        "aggregation": "trailing_mean",
        "aggregation_readings": "a trailing mean over 30 days of real data",
        "mean_window_days": 30,
        "reference_period": None,
        "window_start": None,
        "window_end": None,
        "header_label": "日前均价",
    },
    "halfyear": {
        "source_kind": "real_market",
        "aggregation": "trailing_mean",
        "aggregation_readings": "a trailing mean over half a year of real data",
        "mean_window_days": 180,
        "reference_period": None,
        "window_start": None,
        "window_end": None,
        "header_label": "日前均价",
    },
}

#: Units, as declared by every sheet's own header.
UNITS = "CNY/MWh"

#: The file name, relative to the repository root.
DEFAULT_WORKBOOK = Path("docs") / "日前价格261009.xlsx"


def resolve_workbook(path: Optional[str] = None) -> Path:
    """Locate the price workbook, defaulting to the one under `docs/`."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"price workbook not found: {p}")
        return p
    root = Path(__file__).resolve().parent.parent
    candidate = root / DEFAULT_WORKBOOK
    if not candidate.exists():
        raise FileNotFoundError(
            f"price workbook not found: {candidate}. Pass an explicit path.")
    return candidate


@dataclass(frozen=True)
class Curve:
    """One 96-point price curve, with what it is and where it came from."""

    sheet: str
    role: str
    prices: Tuple[float, ...]
    #: True only when the curve is **confirmed** to be a single day's measured
    #: series. False both for a mean and for a curve whose aggregation is
    #: unresolved -- two different reasons one flag cannot separate, which is
    #: why `aggregation` sits beside it. Nothing in this workbook is confirmed,
    #: so this is False throughout. It is a field rather than a property so a
    #: future confirmed series has to set it explicitly.
    is_measured: bool
    #: "real_market" for every curve here: all four derive from real historical
    #: market data. The distinction between them is the aggregation, not the
    #: provenance.
    source_kind: str = "unknown"
    #: "trailing_mean" | "unresolved" | "single_day".
    aggregation: str = "unknown"
    #: The trailing window in days, or None when there is no known window.
    mean_window_days: Optional[int] = None
    #: The trailing window's end date, or None when it could not be verified.
    window_end: Optional[str] = None
    #: The reference period the sheet names, where it names one.
    reference_period: Optional[str] = None
    provenance: Dict[str, object] = field(default_factory=dict)

    def array(self) -> np.ndarray:
        return np.asarray(self.prices, dtype=float)

    def stats(self) -> Dict[str, float]:
        arr = self.array()
        return {
            "min_cny_mwh": float(arr.min()),
            "max_cny_mwh": float(arr.max()),
            "mean_cny_mwh": float(arr.mean()),
            "peak_slot": int(np.argmax(arr)),
            "trough_slot": int(np.argmin(arr)),
        }

    def window(self, t_start: int, T: int) -> np.ndarray:
        """`T` periods beginning at `t_start`, the provider's window contract."""
        arr = self.array()
        if t_start < 0 or T <= 0 or t_start + T > N_SLOTS:
            raise ValueError(
                f"window [{t_start}, {t_start + T}) does not fit a "
                f"{N_SLOTS}-period day")
        return arr[t_start:t_start + T]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def label_minutes(cell, index: int) -> int:
    """Minutes past midnight for a slot label; `24:00` returns 1440.

    Excel cannot hold `24:00` in a `time` cell, so the closing label of the day
    arrives as midnight of the epoch date. It is only meaningful in the final
    slot; anywhere else it is ambiguous and raises.
    """
    if isinstance(cell, _dt.datetime):
        if (cell.hour, cell.minute, cell.second) != (0, 0, 0):
            raise ValueError(
                f"slot {index}: datetime label {cell!r} is not midnight; the "
                "only datetime this reader accepts is the end-of-day 24:00")
        if index != N_SLOTS - 1:
            raise ValueError(
                f"slot {index}: a midnight datetime label is only the 24:00 "
                "end-of-day sentinel, which belongs in the last slot")
        return 24 * 60
    if isinstance(cell, _dt.time):
        if cell.second or cell.microsecond:
            raise ValueError(
                f"slot {index}: label {cell!r} has sub-minute precision")
        return cell.hour * 60 + cell.minute
    raise ValueError(
        f"slot {index}: label {cell!r} is {type(cell).__name__}, not a time")


def _check_slot_axis(labels: Sequence) -> None:
    """Assert the labels are exactly the interval-END clock, in order.

    The expected value at slot `i` is `15 * (i + 1)` minutes. Accepting
    interval-START labels here would silently rotate every price by one period,
    so a mismatch is an error naming the convention, not a repair.
    """
    if len(labels) != N_SLOTS:
        raise ValueError(
            f"expected {N_SLOTS} slot labels, found {len(labels)}")
    minutes = [label_minutes(cell, i) for i, cell in enumerate(labels)]
    for i, got in enumerate(minutes):
        want = SLOT_MINUTES * (i + 1)
        if got != want:
            raise ValueError(
                f"slot {i}: label is {got} minutes past midnight, expected "
                f"{want} ({want // 60:02d}:{want % 60:02d}). The labels are not "
                "the interval-END clock the simulator's periods are aligned to; "
                "an interval-START file must be converted deliberately rather "
                "than read as if it were this one.")
    if len(set(minutes)) != N_SLOTS:
        raise ValueError("slot labels repeat")


def _check_price(cell, index: int, sheet: str) -> float:
    if cell is None:
        raise ValueError(f"{sheet} slot {index}: price is blank")
    if isinstance(cell, bool) or not isinstance(cell, (int, float)):
        try:
            cell = float(str(cell).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"{sheet} slot {index}: price {cell!r} is not a number")
    value = float(cell)
    if not np.isfinite(value):
        raise ValueError(f"{sheet} slot {index}: price {cell!r} is not finite")
    return value


def read_curves(path: Optional[str] = None,
                sheets: Sequence[str] = SHEETS) -> Dict[str, Curve]:
    """Read every requested sheet into a `Curve`.

    Raises on a missing sheet, a missing or non-numeric price, a repeated or
    misordered slot, or a label that is not the interval-END clock.
    """
    import openpyxl

    wb_path = resolve_workbook(path)
    digest = hashlib.sha256(wb_path.read_bytes()).hexdigest()
    producer = _producer_properties(wb_path)

    wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
    try:
        missing = [s for s in sheets if s not in wb.sheetnames]
        if missing:
            raise KeyError(
                f"sheet(s) {missing} not in {wb_path.name}; it carries "
                f"{wb.sheetnames}")

        curves: Dict[str, Curve] = {}
        for name in sheets:
            ws = wb[name]
            rows = [r for r in ws.iter_rows(values_only=True)
                    if any(c is not None for c in r)]
            if not rows:
                raise ValueError(f"{name}: sheet is empty")

            header = rows[0]
            body = rows[1:]
            if len(body) != N_SLOTS:
                raise ValueError(
                    f"{name}: expected {N_SLOTS} data rows, found {len(body)}")

            labels = [r[0] for r in body]
            _check_slot_axis(labels)
            prices = tuple(_check_price(r[1] if len(r) > 1 else None, i, name)
                           for i, r in enumerate(body))

            meta = CURVE_META.get(name, {})
            curves[name] = Curve(
                sheet=name,
                role=ROLES.get(name, "unspecified"),
                prices=prices,
                # Confirmed single-day series only. `261009` derives from real
                # market data but its aggregation is unresolved, so it is not
                # promoted here: an unproven single-day reading would license a
                # cross-period statement the file does not support.
                is_measured=False,
                source_kind=str(meta.get("source_kind", "unknown")),
                aggregation=str(meta.get("aggregation", "unknown")),
                mean_window_days=meta.get("mean_window_days"),
                window_end=meta.get("window_end"),
                reference_period=meta.get("reference_period"),
                provenance={
                    "workbook": wb_path.name,
                    "workbook_path": str(wb_path),
                    "workbook_sha256": digest,
                    "sheet": name,
                    "header_label": str(header[1]) if len(header) > 1 else "",
                    "units": UNITS,
                    "interval_convention": "label_is_interval_end",
                    "slot_mapping": "label_index == period_index",
                    "is_measured": False,
                    "role": ROLES.get(name, "unspecified"),
                    # The provenance facts themselves, carried through so a
                    # consumer does not have to re-read this module's constants.
                    "source_kind": meta.get("source_kind"),
                    "aggregation": meta.get("aggregation"),
                    "aggregation_readings": meta.get("aggregation_readings"),
                    "mean_window_days": meta.get("mean_window_days"),
                    "reference_period": meta.get("reference_period"),
                    "window_start": meta.get("window_start"),
                    "window_end": meta.get("window_end"),
                    # The workbook's own document properties, which are the only
                    # producer-side metadata the file carries.
                    "workbook_author": producer.get("author"),
                    "workbook_created": producer.get("created"),
                    "workbook_modified": producer.get("modified"),
                },
            )
        return curves
    finally:
        wb.close()


def _producer_properties(path: Path) -> Dict[str, object]:
    """The workbook's own `docProps/core.xml` fields.

    The only producer-side metadata the file carries: a personal author and a
    creation date. Worth recording because it is what establishes the workbook
    as a hand-assembled extraction rather than a platform export -- which is in
    turn why no window dates are present in it.
    """
    import re
    import zipfile

    out: Dict[str, object] = {"author": None, "created": None, "modified": None}
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("docProps/core.xml").decode("utf-8", "replace")
    except (KeyError, OSError, zipfile.BadZipFile):
        return out
    for key, tag in (("author", "dc:creator"), ("created", "dcterms:created"),
                     ("modified", "dcterms:modified")):
        m = re.search(rf"<{tag}[^>]*>([^<]*)</{tag}>", xml)
        if m:
            out[key] = m.group(1)
    return out


def describe(curves: Dict[str, Curve]) -> Dict[str, object]:
    """A report-ready description of a curve set.

    Says what each curve IS, not just what it is called. Two facts matter for
    every downstream claim:

    * **All four derive from real historical market data.** The difference
      between them is the aggregation, not whether they are real.
    * **No curve is confirmed to be a single day's measured series.**
      `contains_confirmed_daily_series` is therefore False. For the three
      trailing sheets that is because they are means; for `261009` it is
      because its aggregation is unresolved, and this module does not resolve
      it by assumption.
    """
    aggregations = sorted({c.aggregation for c in curves.values()})
    return {
        "source": "dianyicha",
        "units": UNITS,
        "n_slots": N_SLOTS,
        "interval_convention": "label_is_interval_end",
        "source_kind": "real_market",
        "aggregations_present": aggregations,
        "contains_confirmed_daily_series": False,
        # Kept under its original name so existing consumers and reports keep
        # working; the meaning is now stated precisely by the two fields above.
        "contains_measured_daily_series": False,
        "note": (
            "All four curves derive from real historical market data. They "
            "differ in aggregation: 7days, 30days and halfyear are trailing "
            "means over 7, 30 and 180 days, and 261009's aggregation is "
            "UNRESOLVED -- its header says 均值 while its sheet name carries a "
            "date, and the file does not settle which it is. No curve is "
            "confirmed to be a single day's measured series, so results built "
            "on them are price-scenario comparisons, not evidence about "
            "day-to-day behaviour, volatility or generalisation across days."),
        "curves": {
            name: {
                "role": c.role,
                "is_measured": c.is_measured,
                "source_kind": c.source_kind,
                "aggregation": c.aggregation,
                "mean_window_days": c.mean_window_days,
                "reference_period": c.reference_period,
                "window_end": c.window_end,
                "header_label": c.provenance.get("header_label"),
                **c.stats(),
            }
            for name, c in curves.items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Read the 电易电查 day-ahead price workbook.")
    p.add_argument("--workbook", default=None, help="path to the workbook")
    p.add_argument("--sheet", action="append", dest="sheets", default=None,
                   help="sheet to read (repeatable); default is all four")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        curves = read_curves(args.workbook,
                             tuple(args.sheets) if args.sheets else SHEETS)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = describe(curves)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"source: dianyicha   units: {UNITS}   slots: {N_SLOTS}")
    print(payload["note"])
    print()
    print(f"{'sheet':<10}{'role':<24}{'min':>9}{'max':>9}{'mean':>9}"
          f"{'peak':>6}{'trough':>7}  header")
    for name, info in payload["curves"].items():
        print(f"{name:<10}{info['role']:<24}{info['min_cny_mwh']:>9.2f}"
              f"{info['max_cny_mwh']:>9.2f}{info['mean_cny_mwh']:>9.2f}"
              f"{info['peak_slot']:>6}{info['trough_slot']:>7}"
              f"  {info['header_label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
