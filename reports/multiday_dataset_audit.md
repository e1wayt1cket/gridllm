# Multiday load and price audit

## Verdict

- Calendar dates examined: **730**
- Load-eligible dates (17/17 users, 9/9 buses, 96/96 slots): **103**
- Dates with a validated formal ledger: **103**
- Dates with a real single-day price pair: **1**
- Split status: **blocked** (needs at least 30 distinct priced days)

No raw load or price workbook was modified. No missing value was filled, interpolated, copied or resampled.

## Eligibility rule

A date is load-eligible only when every one of the frozen 17 users has at least one qualified meter-day that converts to a complete 96-slot curve, all 9 mapped buses are covered, and no qualified block on the date fails conversion. The user subset and mapping are frozen on 2022-11-30; they were not re-selected per date.

- Frozen mapping: `35e4ef933980fa4d7a3a806d11d933814b83acbea5bce5b910161c8016aab9a2`
- Load workbook: `63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb`
- Price workbook: `5fc8f9a8461b680fd38a6d2fefb3d2ff99f239c8bd13a6c77df27de3bb9612f3`
- Records with no usable date: 81 missing, 0 unparseable
- Missing-date records belong to users: U018990a98eb1, U064f056fac22, U074a148fa4c7, U09aec13324d0, U0abfaa684d80, U11128e1b6964, U12592b9d53aa, U134c5ea1227f, U19e398c01aaa, U1f2d028ac65f (+37 more)

## Eligible dates by season

| season | dates |
|---|---:|
| autumn | 72 |
| winter | 31 |

## Eligible dates by day type

| day type | dates |
|---|---:|
| holiday | 7 |
| weekday | 71 |
| weekend | 25 |

## Eligible dates by month

| month | dates |
|---|---:|
| 2022-09 | 11 |
| 2022-10 | 31 |
| 2022-11 | 30 |
| 2022-12 | 31 |

## Load magnitude

The frozen subset is not scale-stable across dates: it was selected to match 3.715 MW on 2022-11-30, and the same users can be far larger or smaller on another date. The study does not re-select users to make a date fit.

| quantity | min | median | max |
|---|---:|---:|---:|
| network peak (MW) | 2.887 | 17.386 | 25.607 |
| energy (MWh) | 44.734 | 219.949 | 341.168 |

- Dates within 5% of the 3.715 MW benchmark: **21**
- Dates above twice the benchmark: **60**
- Among dates above twice the benchmark, bus 3 is the dominant bus at the system peak on 60 of 60 dates.

| date | peak MW | energy MWh | dominant bus |
|---|---:|---:|---:|
| 2022-10-03 | 25.607 | 341.168 | 3 |
| 2022-10-02 | 24.606 | 330.774 | 3 |
| 2022-10-01 | 24.600 | 324.419 | 3 |
| 2022-09-24 | 20.796 | 273.395 | 3 |
| 2022-10-04 | 20.717 | 267.175 | 3 |

## Rejected dates

| reason | dates |
|---|---:|
| missing_users:1 | 110 |
| missing_users:2 | 165 |
| missing_users:3 | 295 |
| missing_users:4 | 49 |
| missing_users:5 | 7 |
| missing_users:6 | 1 |

## Price curves

| sheet | date | temporal | accepted | source |
|---|---|---|---|---|
| 261009 | 2026-10-09 | single_day | yes | 日前价格261009.xlsx#261009 |
| 7days |  | window_mean | no: not_a_single_day_curve | 日前价格261009.xlsx#7days |
| 30days |  | window_mean | no: not_a_single_day_curve | 日前价格261009.xlsx#30days |
| halfyear |  | window_mean | no: not_a_single_day_curve | 日前价格261009.xlsx#halfyear |

Only single-day curves are admissible. Window means are recorded below and never relabelled as days.

| rejected sheet | reason |
|---|---|
| 7days | not_a_single_day_curve |
| 30days | not_a_single_day_curve |
| halfyear | not_a_single_day_curve |

## Pairing

| load date | price date | rule | quality |
|---|---|---|---|
| 2022-10-09 | 2026-10-09 | same_month_day_across_years | same_month_day_day_type_mismatch |

- 102 load-eligible dates are unpaired: no single-day price curve is available for them.

## Ledger build

- Built and validated: **103** of 103 eligible dates.

Every built ledger passed the formal-ledger validator.

## Data gaps

- The frozen subset is whole only on the dates listed above; the full per-date list is in `reports/multiday_dataset_manifest.csv`. The remaining dates are rejected rather than padded.
- The eligible window contains no Spring Festival, spring or summer dates. The raw load workbook ends on 2022-12-31, so no later Spring Festival can be audited without extending the raw load range.
- Only one real single-day price curve exists; at most one load date can be paired, and that pair is a cross-year diagnostic, not a same-day replay. A multi-day priced dataset and a train/validation/test split are therefore blocked.

## Artifacts

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\multiday_subset_date_audit.csv` | `60ed3641c76c5d390200213f899389741d364ec62d9d2a64738eda634e532802` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\multiday_dataset_manifest.csv` | `909484f2f286212990bf1bc38880d3ec4dc9307d18bd352f533290507c16938e` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\multiday_price_pairing_audit.csv` | `b533a0417609d542899a27aebd1bfda5c94083a7b6290641fc41a0d3e2920f83` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\multiday_split_manifest.json` | `e2e023c8287f76ef2f332a1ddc257d9aca149b1fe3417c1f81484afe1f2d277e` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\node_loads\multiday_industry_subset\dataset_manifest.json` | `e5fd6a99eb7de434a6dbc320e7d613ff680cb4b2a8f96f08c8713e9154df3227` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\node_loads\multiday_industry_subset\raw_input_hashes.json` | `a103c329f3d3129216e53540503861701c410900c601619fd98a3b2eb931628a` |

## Command

`src/build_multiday_manifest.py --build-ledgers`
