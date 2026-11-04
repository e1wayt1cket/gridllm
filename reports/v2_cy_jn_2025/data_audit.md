# 2025 Jiangsu day-ahead price audit (V2-CY-JN-2025)

This audit supports a **cross-year controlled simulation**. A 2022 load day paired with a 2025 price day is not a same-period market settlement, and nothing here may be described as one.

- Command: `src/audit_jn_price_source.py`
- Input: `C:/Projects/00_WorkSpace/01_Projects/MakerB/docs/data.csv`
- SHA-256: `1bf06f8d537a5e06911353e4a7348bec8efcedee8bdc76ba84a4bc4d9de8b00c`
- Rows: 13151; dates: 137; complete: 136; incomplete: 1
- Range: 2025-06-01 to 2025-12-31

## Timestamp semantics

A complete day carries the 96 distinct labels `00:00` to `23:45`, so a label is the **interval start**, with each value the quarter-hour that begins at that label. The first day of the series starts at `00:15`, which is a truncated file, not a different convention; every other day starts at `00:00` and ends at `23:45`.

## Incomplete days

| date | periods | missing labels |
|---|---:|---:|
| 2025-06-01 | 95 | 1 |

These days are excluded from the curve store and from every pairing. They are not filled, extended or interpolated.

## Gaps

| start | end | days |
|---|---|---:|
| 2025-06-01 | 2025-06-01 | 1 |
| 2025-06-16 | 2025-06-30 | 15 |
| 2025-07-16 | 2025-08-16 | 32 |
| 2025-08-31 | 2025-09-14 | 15 |
| 2025-11-16 | 2025-11-30 | 15 |

A gap is a fact about the source. A load day is left unpaired rather than priced from a neighbour or a window mean.

## Values

| statistic | `price_value_day_ahead_south` |
|---|---:|
| min | 30.00 |
| max | 1500.00 |
| daily-mean median | 263.91 |
| daily-mean min / max | 35.25 / 602.17 |
| negative values | 0 |
| values at the observed floor (30) | 1383 |
| values at the observed cap (1500) | 53 |

## Unit and region: inferred, not externally confirmed

The file carries no unit, currency, tax basis, timezone or region definition. The working reading is **CNY/MWh** for `price_value_day_ahead_south`, on three pieces of evidence:

1. A hard floor at 30 and a hard cap at 1500 are the shape of a bid floor and a market cap, not of a cost curve.
2. A median daily mean of 264 sits beside the project's single externally-adjudicated Jiangsu day-ahead sheet (`261009`, 225.9–445.0 CNY/MWh, mean 386.83), which is the same order of magnitude.
3. Reading the same numbers as CNY/kWh would multiply them by 1000 and as CNY/10-MWh would divide them by 10; neither is a plausible Jiangsu day-ahead range.

`price_value_day_ahead_south` is taken as the 江南 (southern Jiangsu) day-ahead price **because the task names it so**. The file itself does not define the boundary, and the north column is retained but not used. No average of the two columns is formed.

## `mv_value`

Observed range 1244.35 to 56618.23, mean 9905.18, over the same 96-slot axis as the price.

**Not used.** Neither its unit nor the time at which it becomes available can be established from the file. A provincial new-energy total is also not a bus-level injection for this feeder, so even a confirmed MW reading would enter as a system-level feature and never as DER at a bus. The field is retained in the artifact and the JN-RE arm stays disabled; enabling it would require a new observation version, not a reinterpretation of OBS_V6.

## Window supply

| window | dates | complete | weekday | weekend |
|---|---|---:|---:|---:|
| summer | 2025-07-01 … 2025-08-31 | 29 | 21 | 8 |
| autumn | 2025-09-01 … 2025-11-30 | 62 | 45 | 17 |
| winter | 2025-12-01 … 2025-12-31 | 31 | 23 | 8 |
| national_day_2022 | 2025-10-03 … 2025-10-07 | 5 | 3 | 2 |

The 2025 file begins on 2025-06-01, so the `spring` and `spring_festival_2022` groups cannot be priced from it at all. They stay in the frozen V2 library and are not substituted with another season's prices.

## Artifacts

- `data/processed/v2_cy_jn_2025/price_daily_coverage.csv` — one row per date
- `data/processed/v2_cy_jn_2025/price_curves.npz` — complete days keyed `south/<date>` and `mv/<date>`
