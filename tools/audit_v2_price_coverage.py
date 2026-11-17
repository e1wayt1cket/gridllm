# audit_v2_price_coverage.py
"""Which V2 candidate load days have a real dated day-ahead price curve.

Only a single-day curve can price a load day. The 7/30/180-day sheets are
window means and are recorded as rejected, never relabelled. A curve is used
for at most one load date; when none is available the row says so. No curve is
copied and no synthetic provider is consulted.

The result is a coverage audit, not a split: until enough distinct priced days
exist, no train/validation/test assignment is written.

Usage:
    PYTHONPATH=src python src/audit_v2_price_coverage.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Sequence

import dianyicha_price as dp
import multiday_dataset as md
import single_day_data as sdd

PAIRING_COLUMNS = (
    "group", "group_type", "load_date", "year", "season", "day_type",
    "holiday_class", "representative", "price_date", "price_pair_rule",
    "price_year_gap", "price_source", "price_curve_sha256", "pair_quality",
    "pair_reason",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _curve_digest(sheet: str, date_text: str, prices: Sequence[float]) -> str:
    payload = json.dumps({
        "sheet": sheet, "date": date_text,
        "prices": [float(p) for p in prices],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_curves(workbook: str | None = None) -> tuple:
    curves = dp.read_curves(workbook)
    accepted = []
    rejected = []
    descriptors = []
    for name, curve in curves.items():
        source = f"{Path(str(curve.provenance.get('workbook', 'price'))).name}#{name}"
        descriptor = {
            "sheet": name,
            "role": curve.role,
            "temporal_statistic": curve.temporal_statistic,
            "window_days": curve.window_days,
            "reference_period": curve.reference_period,
            "source_kind": curve.source_kind,
            "spatial_statistic": curve.spatial_statistic,
            "externally_confirmed": curve.externally_confirmed,
            "source": source,
            "workbook_sha256": curve.provenance.get("workbook_sha256"),
        }
        if curve.source_kind != "real_market_data":
            descriptor.update(accepted=False,
                              reason="not_real_market_data")
            rejected.append(descriptor)
        elif curve.is_single_day:
            date_text = sdd.iso_from_reference_period(
                curve.reference_period, name)
            digest = _curve_digest(name, date_text, curve.prices)
            descriptor.update(accepted=True, price_date=date_text,
                              curve_sha256=digest)
            accepted.append(md.PriceCurve(
                price_date=md.as_date(date_text), curve_id=name,
                source=source, curve_sha256=digest,
                temporal_statistic=curve.temporal_statistic,
                externally_confirmed=curve.externally_confirmed))
        else:
            descriptor.update(accepted=False,
                              reason="not_a_single_day_curve")
            rejected.append(descriptor)
        descriptors.append(descriptor)
    return accepted, rejected, descriptors


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PAIRING_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(path: Path, *, candidates: Sequence[dict],
                  paired: Sequence[dict], descriptors: Sequence[dict],
                  rejected: Sequence[dict], price_sha: str,
                  candidate_source: Path, csv_path: Path,
                  command: str) -> Path:
    reps = [r for r in paired if _as_bool(r["representative"])]
    reps_paired = [r for r in reps if r["price_date"]]
    by_season = collections.Counter(r["season"] for r in paired)
    paired_by_season = collections.Counter(
        r["season"] for r in paired if r["price_date"])
    lines = [
        "# V2 load-price coverage",
        "",
        "## Verdict",
        "",
        f"- Candidate source: `{candidate_source}`",
        f"- Candidate load dates: **{len(paired)}**",
        f"- Representative load dates: **{len(reps)}**",
        f"- Candidate dates with a real single-day price: "
        f"**{sum(1 for r in paired if r['price_date'])}**",
        f"- Representative dates with a real single-day price: "
        f"**{len(reps_paired)}**",
        "- A multi-day priced dataset and a train/validation/test split "
        "remain **blocked**.",
        "",
        "Only a real curve whose own temporal statistic is `single_day` is "
        "admissible. Window means are recorded and rejected; no curve is "
        "copied, interpolated or resampled; no synthetic provider is used.",
        "",
        "## Price inventory",
        "",
        f"- Price workbook sha256: `{price_sha}`",
        "",
        "| sheet | date | temporal | accepted | source |",
        "|---|---|---|---|---|",
    ]
    for d in descriptors:
        lines.append(
            f"| {d['sheet']} | {d.get('price_date', '')} | "
            f"{d['temporal_statistic']} | "
            f"{'yes' if d.get('accepted') else 'no: ' + d.get('reason', '')} | "
            f"{d['source']} |")
    if rejected:
        lines += ["", "| rejected sheet | reason |", "|---|---|"]
        for d in rejected:
            lines.append(f"| {d['sheet']} | {d['reason']} |")
    lines += ["", "## Coverage by season", "",
              "| season | candidates | priced |",
              "|---|---:|---:|"]
    for season in ("spring", "summer", "autumn", "winter"):
        lines.append(f"| {season} | {by_season.get(season, 0)} | "
                     f"{paired_by_season.get(season, 0)} |")
    lines += ["", "## Representative dates", "",
              "| group | load date | season | day type | holiday | price date | "
              "rule | quality | reason |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in sorted(reps, key=lambda r: (r["group"], r["load_date"])):
        lines.append(
            f"| {row['group']} | {row['load_date']} | {row['season']} | "
            f"{row['day_type']} | {row['holiday_class']} | {row['price_date']} | "
            f"{row['price_pair_rule']} | {row['pair_quality']} | "
            f"{row['pair_reason']} |")
    lines += ["", "## Data gaps", "",
              "- Only one real single-day price curve is present. At most one "
              "load date can be paired, and that pair is a cross-year "
              "diagnostic rather than a same-day replay.",
              "- No representative date has a real dated price. This is the "
              "blocker for a multi-day training set.",
              "- The window means (`7days`, `30days`, `halfyear`) are not "
              "daily prices and were not used.",
              "",
              "## Artifacts", "",
              f"- `{csv_path}`",
              "- Pairing rules and curve identity reuse `src/multiday_dataset.py`.",
              "",
              "## Command", "", f"`{command}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit real dated price coverage for V2 candidates.")
    parser.add_argument(
        "--candidates", default="data/processed/v2_typical_day_candidates.csv")
    parser.add_argument("--price-workbook", default=None)
    parser.add_argument("--out", default="reports/v2_load_price_pairing.csv")
    parser.add_argument("--report", default="reports/v2_price_coverage.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    candidates_path = root / args.candidates
    if not candidates_path.exists():
        raise SystemExit(
            f"{candidates_path} not found; select the V2 typical days first")
    candidates = _read_csv(candidates_path)
    load_dates = [r["date"] for r in candidates]
    accepted, rejected, descriptors = load_curves(args.price_workbook)
    if not accepted:
        raise SystemExit("the price workbook carries no real single-day curve")
    pairing = md.pair_prices(load_dates, accepted)
    by_date = {p["load_date"]: p for p in pairing["pairs"]}
    rows = []
    for candidate in candidates:
        pair = by_date[candidate["date"]]
        rows.append({
            "group": candidate.get("group", ""),
            "group_type": candidate.get("group_type", ""),
            "load_date": candidate["date"],
            "year": candidate["year"],
            "season": candidate["season"],
            "day_type": candidate["day_type"],
            "holiday_class": candidate["holiday_class"],
            "representative": candidate["representative"],
            **{k: pair[k] for k in (
                "price_date", "price_pair_rule", "price_year_gap",
                "price_source", "price_curve_sha256", "pair_quality",
                "pair_reason")},
        })
    csv_path = _write_csv(root / args.out, rows)
    price_sha = hashlib.sha256(
        dp.resolve_workbook(args.price_workbook).read_bytes()).hexdigest()
    report_path = _write_report(
        root / args.report, candidates=candidates, paired=rows,
        descriptors=descriptors, rejected=rejected, price_sha=price_sha,
        candidate_source=candidates_path, csv_path=csv_path,
        command=" ".join(sys.argv))
    print(f"candidate load dates    {len(rows)}")
    print(f"priced candidates       {sum(1 for r in rows if r['price_date'])}")
    print(f"representative dates    "
          f"{sum(1 for r in rows if _as_bool(r['representative']))}")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
