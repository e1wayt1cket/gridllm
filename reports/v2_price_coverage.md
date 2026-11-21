# V2 load-price coverage

## Verdict

- Candidate source: `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_candidates.csv`
- Candidate load dates: **123**
- Representative load dates: **31**
- Candidate dates with a real single-day price: **0**
- Representative dates with a real single-day price: **0**
- A multi-day priced dataset and a train/validation/test split remain **blocked**.

Only a real curve whose own temporal statistic is `single_day` is admissible. Window means are recorded and rejected; no curve is copied, interpolated or resampled; no synthetic provider is used.

## Price inventory

- Price workbook sha256: `5fc8f9a8461b680fd38a6d2fefb3d2ff99f239c8bd13a6c77df27de3bb9612f3`

| sheet | date | temporal | accepted | source |
|---|---|---|---|---|
| 261009 | 2026-10-09 | single_day | yes | 日前价格261009.xlsx#261009 |
| 7days |  | window_mean | no: not_a_single_day_curve | 日前价格261009.xlsx#7days |
| 30days |  | window_mean | no: not_a_single_day_curve | 日前价格261009.xlsx#30days |
| halfyear |  | window_mean | no: not_a_single_day_curve | 日前价格261009.xlsx#halfyear |

| rejected sheet | reason |
|---|---|
| 7days | not_a_single_day_curve |
| 30days | not_a_single_day_curve |
| halfyear | not_a_single_day_curve |

## Coverage by season

| season | candidates | priced |
|---|---:|---:|
| spring | 20 | 0 |
| summer | 21 | 0 |
| autumn | 52 | 0 |
| winter | 30 | 0 |

## Representative dates

| group | load date | season | day type | holiday | price date | rule | quality | reason |
|---|---|---|---|---|---|---|---|---|
| autumn | 2022-10-15 | autumn | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| autumn | 2022-11-10 | autumn | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| autumn | 2022-11-11 | autumn | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| autumn | 2022-11-13 | autumn | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| autumn | 2022-11-20 | autumn | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| autumn | 2022-11-25 | autumn | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| national_day_2022 | 2022-10-03 | autumn | holiday | national_day |  |  | unpaired | no_single_day_curve |
| national_day_2022 | 2022-10-04 | autumn | holiday | national_day |  |  | unpaired | no_single_day_curve |
| national_day_2022 | 2022-10-07 | autumn | holiday | national_day |  |  | unpaired | no_single_day_curve |
| spring | 2022-04-29 | spring | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| spring | 2022-04-30 | spring | holiday | labour_day |  |  | unpaired | no_single_day_curve |
| spring | 2022-05-21 | spring | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| spring | 2022-05-23 | spring | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| spring | 2022-05-28 | spring | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| spring | 2022-05-29 | spring | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| spring | 2022-05-30 | spring | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| spring_festival_2022 | 2022-01-31 | winter | holiday | spring_festival |  |  | unpaired | no_single_day_curve |
| spring_festival_2022 | 2022-02-01 | winter | holiday | spring_festival |  |  | unpaired | no_single_day_curve |
| spring_festival_2022 | 2022-02-02 | winter | holiday | spring_festival |  |  | unpaired | no_single_day_curve |
| summer | 2022-07-22 | summer | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| summer | 2022-07-23 | summer | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| summer | 2022-08-14 | summer | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| summer | 2022-08-24 | summer | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| summer | 2022-08-25 | summer | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| summer | 2022-08-27 | summer | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| winter | 2022-12-10 | winter | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| winter | 2022-12-11 | winter | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| winter | 2022-12-24 | winter | weekend | ordinary |  |  | unpaired | no_single_day_curve |
| winter | 2022-12-27 | winter | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| winter | 2022-12-28 | winter | weekday | ordinary |  |  | unpaired | no_single_day_curve |
| winter | 2022-12-29 | winter | weekday | ordinary |  |  | unpaired | no_single_day_curve |

## Data gaps

- Only one real single-day price curve is present. At most one load date can be paired, and that pair is a cross-year diagnostic rather than a same-day replay.
- No representative date has a real dated price. This is the blocker for a multi-day training set.
- The window means (`7days`, `30days`, `halfyear`) are not daily prices and were not used.

## Artifacts

- `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\v2_load_price_pairing.csv`
- Pairing rules and curve identity reuse `src/multiday_dataset.py`.

## Command

`src/audit_v2_price_coverage.py --candidates data/processed/v2_seasonal_candidates.csv`
