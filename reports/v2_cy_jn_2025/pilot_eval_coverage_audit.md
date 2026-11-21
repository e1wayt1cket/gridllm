# Pilot evaluation coverage audit (V2-CY-JN-2025)

## Answer

`eval.csv` holds **64 records**, one per `(dataset, seed, load_date)`, with the four storage units as columns in the same record. It is **not** one record per unit. The layout is wide, not long.

| field | unique values |
|---|---|
| datasets | FULL96, NEAR |
| seeds | 11, 22 |
| dates | 16 (11 test + 5 holiday) |
| storage units | 4 as columns: ESS9, ESS14, ESS17, ESS27 |
| splits | {'test': 44, 'holiday_block': 20} |

Primary key `(dataset, seed, load_date)`: 64 unique, 0 duplicates, no missing keys. Every test and holiday day appears once for each of the four runs:

| dataset/seed | test | holiday |
|---|---:|---:|
| FULL96/11 | 11 | 5 |
| FULL96/22 | 11 | 5 |
| NEAR/11 | 11 | 5 |
| NEAR/22 | 11 | 5 |

## Why not 176

A long layout with one row per `(dataset, seed, date, unit)` would have 2 × 2 × 11 × 4 = **176** test rows (and 80 holiday rows). The committed file is wide: 2 × 2 × 16 = 64 rows, each carrying four payoff columns. Both encode the same grid; no unit is missing. `eval.csv` also carries the per-unit B1 payoff, charge/discharge and cycles, so the per-unit facts are present, just not as rows.

## The two aggregation conventions

Both appear in `matd3_pilot_results.md` and must not be conflated:

| convention | definition | days above B1 |
|---|---|---|
| per run `FULL96/11` | each seed's own 11 days | 0/11 |
| per run `FULL96/22` | each seed's own 11 days | 0/11 |
| per run `NEAR/11` | each seed's own 11 days | 0/11 |
| per run `NEAR/22` | each seed's own 11 days | 0/11 |
| seed-averaged `FULL96` | mean of the two seeds per day, then compare with B1 | 0/11 |
| seed-averaged `NEAR` | mean of the two seeds per day, then compare with B1 | 0/11 |

The report's run table uses the first; its paired per-day table uses the second. Both give zero days above B1 for every run and for the seed-averaged comparison, so the conclusion does not depend on the convention here.

## Included

- All 11 common test days for all four runs.
- All 5 national-day days for all four runs, reported separately.
- No training day appears in the evaluation file.
