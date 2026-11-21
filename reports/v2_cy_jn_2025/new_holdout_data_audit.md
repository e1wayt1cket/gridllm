# New holdout data audit (V2-CY-JN-2025)

**Result: not a new independent holdout. G10 is not met.**

The file supplied for this review is `docs/data.csv`. It is the same 2025
Jiangnan source the project already pairs and has inspected; it carries no date
outside the set already assigned or viewed.

## What the file contains

| item | value |
|---|---|
| path | `docs/data.csv` |
| sha256 | `1bf06f8d537a5e06911353e4a7348bec8efcedee8bdc76ba84a4bc4d9de8b00c` |
| encoding | UTF-8 with BOM |
| rows | 13,151 data rows |
| columns | `time`, `price_value_day_ahead_north`, `price_value_day_ahead_south`, `mv_value` |
| date range | 2025-06-01 to 2025-12-31 |
| distinct dates | 137 |
| complete 96-point days | 136 |
| incomplete day | 2025-06-01 (95 rows) |
| duplicate timestamps | 0 |
| south price range | 30.0 to 1500.0 CNY/MWh, no NaN, no negative, no zero |
| north price range | 0.0 to 1500.0 CNY/MWh, one zero |
| `mv_value` range | 1244.3493 to 56618.2349, no NaN, no negative, no zero |

The 96-point days run 00:00 through 23:45; the mapping to the simulator's
periods is positional and was already audited for this source.

## Why it cannot serve as G10

`src/pair_v2_cy_jn_2025.py` already defaults to `--price-file docs/data.csv`,
and `src/audit_jn_price_source.py` audits this exact file. The 2025 days were
assigned to FULL96/NEAR and split into train, validation, test and the
national-day block; the common test and the holiday block have been viewed in
the pilot and in every later report. Re-splitting the same days now is a
post-hoc split, not a blind holdout.

The file also does not supply a new load period, so even a new price file alone
would only support a cross-price-period test, not a joint load-price
generalisation test.

## What would meet G10

A file from a period this project has never used, with:

- complete 96-point days;
- an explicit timestamp convention;
- an explicit unit and geographic region;
- a traceable source file and hash;
- preferably a matching-period load series.

The file must be frozen before any model, reward, potential, gate estimator or
hyperparameter decision, and opened once. The current status artifact
`data/processed/v2_cy_jn_2025/pilot/holdout_status.json` remains the record:
`unseen_2025_holdout_available: false`.

## Consequence for the formal run

G10 fails. Formal multi-seed training must not start under this plan. Any run
on the existing 2025 days is exploratory/development only and cannot be
reported as independent generalisation.
