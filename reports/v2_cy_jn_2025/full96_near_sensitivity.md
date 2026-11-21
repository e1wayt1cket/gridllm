# FULL96 and NEAR sensitivity datasets (V2-CY-JN-2025)

Both datasets are **cross-year controlled simulations**: the load is a 2022 metered day, the price a 2025 Jiangnan day-ahead curve. Neither is a same-period settlement, and the price unit is an inference recorded in `data_audit.md`, not an external confirmation.

## Why not 83 days

The working name assumed only autumn pairs exceeded 14 days. That is not what the committed table contains: the filter removes **16 pairs** — autumn 13, summer 3. The count is computed from the table, not hardcoded, so NEAR has **80 days**, not 83.

## Counts

| group | FULL96 | NEAR | removed |
|---|---:|---:|---:|
| national_day_2022 | 5 | 5 | 0 |
| summer | 21 | 18 | 3 |
| autumn | 47 | 34 | 13 |
| winter | 23 | 23 | 0 |
| **total** | **96** | **80** | **16** |

## Seasonal shift

| group | FULL share | NEAR share | change |
|---|---:|---:|---:|
| national_day_2022 | 5.2% | 6.2% | +1.0% |
| summer | 21.9% | 22.5% | +0.6% |
| autumn | 49.0% | 42.5% | -6.5% |
| winter | 24.0% | 28.7% | +4.8% |

Summer loses 3 of 21 and autumn 13 of 47, so autumn's share falls. The filter is not season-balanced; it is a distance rule and the distance distribution is season-specific.

## Distance

| statistic | FULL96 | NEAR |
|---|---:|---:|
| pairs | 96 | 80 |
| mean (days) | 9.38 | 2.00 |
| median (days) | 2 | 2 |
| max (days) | 67 | 12 |
| over 30 days | 13 | 0 |
| same month-day | 27 | 27 |
| recorded type mismatch | 1 | 1 |

The single recorded type mismatch (2022-08-20 to 2025-08-20) survives the filter: it is a same-month-day pair at distance 0, and dropping it would remove an exact date match to keep a constraint the source cannot satisfy.

The 14-day rule is a **sensitivity filter**, not a validated boundary: the source says nothing about 14 days. It exists so a result can be asked whether it depends on the far pairs; it does not make the NEAR set the correct one.

## Removed pairs

| load_date | price_date | group | gap (days) |
|---|---|---|---:|
| 2022-11-27 | 2025-09-21 | autumn | 67 |
| 2022-11-30 | 2025-09-26 | autumn | 65 |
| 2022-11-29 | 2025-09-29 | autumn | 61 |
| 2022-11-26 | 2025-09-27 | autumn | 60 |
| 2022-11-28 | 2025-09-30 | autumn | 59 |
| 2022-11-25 | 2025-10-01 | autumn | 55 |
| 2022-11-20 | 2025-09-28 | autumn | 53 |
| 2022-11-24 | 2025-10-02 | autumn | 53 |
| 2022-11-23 | 2025-10-08 | autumn | 46 |
| 2022-08-13 | 2025-07-05 | summer | 39 |
| 2022-11-22 | 2025-10-15 | autumn | 38 |
| 2022-11-21 | 2025-10-16 | autumn | 36 |
| 2022-08-12 | 2025-07-11 | summer | 32 |
| 2022-11-18 | 2025-10-22 | autumn | 27 |
| 2022-11-17 | 2025-10-23 | autumn | 25 |
| 2022-07-30 | 2025-07-06 | summer | 24 |

## Status

**No market was cleared for NEAR in this phase.** The dataset and its split are prepared; `results.csv` still describes the 180 FULL96 clears (96 JN-Base and 21 each for C80/C90/C100/NoESS). A NEAR clear is a separate run and has not been performed.

**For any comparison, use `common_split.json`, not these splits.** The per-dataset splits below are each dataset's own record; they are not comparable across datasets. `common_split.json` gives every common record one label in both datasets and is the only split a FULL96-vs-NEAR comparison may use; see `common_split_audit.md`.

## Splits

| split | FULL96 | NEAR |
|---|---:|---:|
| train | 64 | 53 |
| validation | 13 | 11 |
| test | 14 | 11 |
| holiday_block | 5 | 5 |

NEAR uses seed `20261031`, FULL96 uses seed `20261030`. Within each dataset the three training splits and the holiday block are disjoint and cover every day.

## Cross-dataset overlap

Every NEAR day is also a FULL96 day. The rows below count dates shared between a FULL96 split and a NEAR split:

| FULL96 \ NEAR | train | validation | test | holiday |
|---|---:|---:|---:|---:|
| train | 35 | 8 | 7 | 0 |
| validation | 7 | 2 | 4 | 0 |
| test | 11 | 1 | 0 | 0 |
| holiday_block | 0 | 0 | 0 | 5 |

Non-zero off-diagonal cells mean a NEAR training day can be a FULL96 test day, and the reverse. The two splits are independent **assignments over the same days**, not independent samples.

With the committed seeds, NEAR's test split shares 0 days with FULL96's test split and 11 with FULL96's training split. A NEAR test score is therefore out-of-sample within NEAR but in-sample relative to FULL96's training set. **The two datasets cannot cross-validate each other**, and a FULL96 result and a NEAR result on the same day must not be reported as two confirmations.

## Artifacts

- `data/processed/v2_cy_jn_2025/load_price_pairing.csv`, `data/processed/v2_cy_jn_2025/pairing_manifest.json`, `data/processed/v2_cy_jn_2025/split.json` (JN-Base-FULL96)
- `data/processed/v2_cy_jn_2025/near/load_price_pairing.csv`, `data/processed/v2_cy_jn_2025/near/pairing_manifest.json`, `data/processed/v2_cy_jn_2025/near/split.json` (JN-Base-NEAR)
- `data/processed/v2_cy_jn_2025/datasets.json` — both datasets, hashes, and this relationship
