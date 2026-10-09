# Phase 2 — 电易电查 Price Adapter

Delivers `src/dianyicha_price.py` (the reader) and
`wholesale_price.DianyichaPriceProvider` (the registered source), plus the
`dianyicha` configuration and the data-source note.

No MATD3 training was started. Phase 1's module and reports are unchanged.

## 1. Files

**Added**

| file | what |
|---|---|
| `src/dianyicha_price.py` | reads the workbook; `Curve`, `read_curves`, `describe` |
| `tests/test_dianyicha_price.py` | 27 tests (26 fast, 1 slow) |
| `reports/phase2_dianyicha_price_report.md` | this report |

**Modified** — four files, each additively:

| file | change |
|---|---|
| `src/wholesale_price.py` | `SOURCES` gains `dianyicha`; new `DianyichaPriceProvider`; one `get_provider` branch; one import of the curve order. All existing providers untouched. |
| `config/defaults.yaml` | two keys under `wholesale_market` (`dianyicha_file`, `dianyicha_curves`) and one comment line. `source: legacy` is unchanged. |
| `tests/test_wholesale_price.py` | the registry dict in `test_every_declared_source_name_has_a_provider` gains `dianyicha`. That test exists to catch exactly this drift. |
| `docs/2026-09-20-external-wholesale-price.md` | a new closing section documenting the source. Existing text untouched (+62 lines, 0 deleted). |

## 2. What was read

| | |
|---|---|
| workbook | `docs/日前价格261009.xlsx`, 20,621 bytes |
| sheets | `261009`, `7days`, `30days`, `halfyear` |
| rows per sheet | 96 data rows + 1 header |
| slots | 96 x 15 minutes |
| units | CNY/MWh, as declared in every sheet's header |
| missing prices | none |

`261009` declares 272 rows but carries only 97 non-empty ones; 175 are trailing
formatted-but-empty rows and are dropped by the non-empty filter. The other
three sheets are exactly 97.

Read with `openpyxl` in read-only, values-only mode. The workbook is not
modified and **no text or CSV copy is produced** — the provider calls the reader
directly, as instructed.

## 3. Slot axis and the mapping

Labels are **interval-END**: the clock runs `00:15` … `24:00`, and label `i`
closes the interval that ends `15 * (i + 1)` minutes past midnight. The
simulator's period `i` is `[15i, 15(i+1))`, so **label `i` belongs to period `i`
and the mapping is the identity**. Nothing is shifted by 15 minutes.

The reader does not assume this. `_check_slot_axis` builds the exact expected
sequence and raises on any departure, so an interval-START file (`00:00` …
`23:45`) is **refused with a message naming the convention** rather than being
rotated by one period. That failure mode is worth the strictness: a one-period
rotation leaves 96 correct-looking numbers in correct-looking units.

The final label is the awkward one. `24:00` has no `datetime.time`
representation, so Excel stores it as `1900-01-01 00:00:00`. It is accepted
**only in slot 95**; a midnight `datetime` anywhere else raises, rather than
being guessed at.

## 4. The four curves

Every one is a **mean**, as the workbook's own column headers state — including
`261009`, whose header reads `日前均值（CNY/MWh）`.

| sheet | header (verbatim) | role | min | max | mean | trough | peak |
|---|---|---|---|---|---|---|---|
| `261009` | 日前均值（CNY/MWh） | `same_day_or_same_day_mean` | 225.90 | 445.00 | 386.83 | 11:15 | 16:00 |
| `7days` | 日前均价 | `trailing_7_day_mean` | 257.50 | 477.87 | 395.55 | 11:30 | 17:45 |
| `30days` | 日前均价 | `trailing_30_day_mean` | 333.08 | 450.86 | 407.82 | 11:45 | 02:15 |
| `halfyear` | 日前均价 | `trailing_180_day_mean` | 284.93 | 393.86 | 346.33 | 09:45 | 02:00 |

All values CNY/MWh, computed by the reader and cross-checked against an
independent reading of the file in Phase 1.

The shapes are the expected Jiangsu pattern: a night plateau near 420–440, a
**midday trough** (249 CNY/MWh at 11:15 on `261009`, 266 on `7days`) as solar
output bites, then recovery to an evening peak. The 7-day mean is the deepest
duck; the half-year mean is the flattest and its daytime dip is the mildest.
`30days` and `halfyear` report their maximum at 02:00 because those curves are
flat enough that the early-hours plateau edges out the evening — a property of
the averaging window, not an error.

## 5. Provider

```
SOURCES = ("legacy", "synthetic_v1", "library", "guangdong", "dianyicha")
```

`day_id` **selects a curve by index** into
`wholesale_market.dianyicha_curves`, default
`["261009", "7days", "30days", "halfyear"]`. It is a price-scenario selector,
not a calendar day, and that is enforced:

- an out-of-range `day_id` **raises** rather than wrapping or falling back to
  the first curve — a silent fallback would make every scenario in a sweep
  report identical prices while appearing to differ;
- `describe()` publishes each curve's `role` and `is_measured: false`, plus
  `contains_measured_daily_series: false`, so a report cannot present a mean as
  an observation without contradicting a recorded field;
- `seed` is accepted for interface compatibility and ignored — a recorded curve
  has no draw to override;
- `window(t_start, T)` slices the day, so a rolling-horizon caller gets its own
  periods rather than the first ones.

The four existing sources still resolve; `synthetic_v1` is untouched.

## 6. Tests

**New:** `tests/test_dianyicha_price.py` — 26 fast + 1 slow.

```
python -m pytest tests/test_dianyicha_price.py -v          -> 26 passed, 1 deselected
python -m pytest tests/test_dianyicha_price.py -v -m slow  -> 1 passed
```

Coverage: positional interval-END mapping asserted label by label; an
interval-START workbook refused; a single shifted label caught; the `24:00`
datetime accepted in the last slot and refused elsewhere; blank and
non-numeric prices refused; wrong row count and missing sheet refused; values
read unchanged with no clipping; mean/min/max statistics; every curve marked a
mean and every `describe()` free of a measured-day claim; the raw header string
preserved in provenance; window slicing equal to a slice of the full day and
refused past the end; provider registration and order; out-of-range and
negative `day_id` refused; seed ignored; existing sources still resolve.

**Regression:** `python -m pytest tests/` -> **640 passed, 51 deselected, 0
failed** (127 s). That is Phase 1's 614 plus the 26 new fast tests.

The slow test asserts the four curves' min/max/mean against independently
measured values, so a future edit to the workbook that silently changes the
data fails here rather than in a conclusion.

## 7. Open items

1. **No measured daily series exists, and this is the binding limitation.**
   Four mean curves are not a basis for a train/validation/test split over real
   prices, for volatility, or for any statement about generalisation across
   days. Results on these curves are **price-scenario comparisons**. This is
   recorded in the module, in `describe()`, in `defaults.yaml`, and in the
   data-source doc — but it is a constraint on what may be claimed, not
   something code can enforce.
2. **The workbook is outside version control.** `.gitignore` excludes `*.xlsx`,
   so `docs/日前价格261009.xlsx` is untracked. The reader records the file's
   SHA-256 in every curve's provenance, which ties a run to the bytes it read,
   but the bytes themselves are not in the repository. If this file is meant to
   be part of the delivered record, it needs an explicit `.gitignore` exception
   or a committed derived artefact. Not changed unasked.
3. **The curves are far narrower than the synthetic band.** 226–478 CNY/MWh
   against `price_floor: 120` / `price_cap: 1000`. Storage arbitrage will be
   thinner on these. That is a finding to report; the floor and cap stay put.
4. **`30days` and `halfyear` peak at 02:00.** Correct for those means, but it
   means "peak slot" is not a useful summary for the flatter curves, and any
   figure caption should say which curve it is describing.
5. **The averaging windows are unverified.** The sheets are named `7days`,
   `30days`, `halfyear`, but nothing in the file states the window's end date,
   and no check has been made that each window ends before `261009`'s date.
   Until that is confirmed against the platform, these cannot be called
   strictly non-anticipative historical information. Carried into Phase 3.

## 8. Difference from the pre-Phase-2 workspace

Recorded before any edit and re-verified after:

```
HEAD     a1151fd2ab52e4a49a610173a1bba04c56bda563   (unchanged)
branch   master                                      (unchanged)
```

Backed up to `/tmp/pre_phase2_worktree.diff`.

Recomputing the diff over **only the 14 files that were already modified** gives
sha256 `f0c79a4f34992721e5b035ffd9d39cda71310bb8977f5f85f1342778e7e03f46` —
identical to the pre-Phase-2 backup. The pre-existing uncommitted work is
byte-for-byte untouched, and none of my Phase-2 edits landed on those files.

Phase 1's deliverables were not modified; only read.

Net change to the working tree: two new untracked files
(`src/dianyicha_price.py`, `tests/test_dianyicha_price.py`), four additive
edits to tracked files, and this report.

## Status

Phase 2 complete. Stopping here for review; Phase 3 is not started.
