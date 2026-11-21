# Load-to-price pairing (V2-CY-JN-2025)

**Cross-year controlled simulation.** Each record carries a 2022 `load_date` and a 2025 `price_date`. They are two independent dates, aligned by quarter-hour index. This is not a same-period settlement and must not be reported as one.

- Command: `src/pair_v2_cy_jn_2025.py`
- Frozen load library: `v2-frozen-dfc9d60cc222` (manifest sha256 `ebed1801c43c32a9…`)
- Price source sha256: `1bf06f8d537a5e06911353e4a7348bec8efcedee8bdc76ba84a4bc4d9de8b00c`
- Pairs: **96**; unpaired load days: **0**
- Distinct `price_date`: **96**; reused: **0**
- Distinct `load_date`: **96**; reused: **0**

## By group

| group | load days | paired | price days supplied | same month-day | same type, nearest | type mismatch |
|---|---:|---:|---:|---:|---:|---:|
| national_day_2022 | 5 | 5 | 5 (3 wd / 2 we) | 5 | 0 | 0 |
| summer | 21 | 21 | 29 (21 wd / 8 we) | 3 | 17 | 1 |
| autumn | 47 | 47 | 62 (45 wd / 17 we) | 14 | 33 | 0 |
| winter | 23 | 23 | 31 (23 wd / 8 we) | 4 | 19 | 0 |
| **total** | **96** | **96** | | **26** | **69** | **1** |

## Distance from the load day's month-day

| group | pairs | same month-day | median gap (days) | max gap | pairs over 14 days |
|---|---:|---:|---:|---:|---:|
| national_day_2022 | 5 | 5 | 0 | 0 | 0 |
| summer | 21 | 4 | 2 | 39 | 3 |
| autumn | 47 | 14 | 3 | 67 | 13 |
| winter | 23 | 4 | 1 | 3 | 0 |

A large gap is a property of the source, not a search failure: the 2025 file has no complete day between 11-16 and 11-30, so late-November load days take the nearest available autumn date, which can fall in September. The gap is recorded per pair in `month_day_gap_days`, and those pairs must not be read as a November price applying to a November load.

## Day-type constraints

1 pair(s) could not keep the weekday/weekend type and are recorded as mismatches, not silently relabelled:

| load_date | price_date | group | reason |
|---|---|---|---|
| 2022-08-20 | 2025-08-20 | summer | same month-day, but its weekday/weekend type differs; no date of the load day's own type was left in the window |

The cause is supply, not a choice:

- summer: 9 weekend load days against 8 complete weekend price days in the window, so at least 1 cannot keep the type.

The mismatch is placed on the pair whose same month-day is available, so the failed type costs no additional date distance. Every other group keeps the type.

Day type is the calendar weekday/weekend flag. Chinese public-holiday makeup workdays are **not** reclassified in this version; that can only under-count same-type matches, and it never changes a date.

## Uniqueness

No `price_date` appears twice, so one price curve cannot enter both training and test through two load days.
No `load_date` appears twice; every load day is used at most once.

## Load curves

Every paired load date assembled a 96x33 matrix: the group's own user curves at their mapped buses (`replace`), the benchmark's native demand on every other bus (`preserve`). No load was rescaled, filled or zeroed to make a day usable.

| group | median group peak (MW) | frozen median (MW) |
|---|---:|---:|
| national_day_2022 | 4.402 | 4.402 |
| summer | 4.656 | 4.656 |
| autumn | 5.553 | 5.553 |
| winter | 5.141 | 5.141 |

The peaks are the group's own metered sum, the quantity the frozen manifest records. The network total is larger because the other buses keep their native demand; both are stored.

## Split

A candidate 70/15/15 split is written beside the pairing, stratified by season, with the national-day block held out whole. It is a proposal for the readiness review, not a training release.

## Artifacts

- `data/processed/v2_cy_jn_2025/load_price_pairing.csv`
- `data/processed/v2_cy_jn_2025/bus_loads_96x33.npz` — `load/<load_date>`, shape (96, 33)
- `data/processed/v2_cy_jn_2025/split.json`
