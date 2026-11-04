# Common split audit (V2-CY-JN-2025)

The protocol for any FULL96-vs-NEAR comparison. Every pair record common to the two datasets carries the **same** split label in both, so the test set is the same days for each. The per-dataset splits written earlier remain on disk but are not comparable across datasets; `datasets.json` names this file as the comparison protocol.

- Seed: `20261101`
- Pair key: `load_date|price_date`; load dates and price dates are unique within each dataset
- Common records: 80; FULL96-only: 16

## Counts

| view | train | validation | test | holiday | total |
|---|---:|---:|---:|---:|---:|
| common (both datasets) | 53 | 11 | 11 | 5 | 80 |
| FULL96 | 69 | 11 | 11 | 5 | 96 |
| NEAR | 53 | 11 | 11 | 5 | 80 |

The **common test set has 11 days**, and it is the same list for both datasets: FULL96's test split equals NEAR's. FULL96's validation split is also exactly the common validation split.

## Seasonal composition of the common split

| group | train | validation | test | holiday |
|---|---:|---:|---:|---:|
| national_day_2022 | 0 | 0 | 0 | 5 |
| summer | 13 | 3 | 2 | 0 |
| autumn | 24 | 5 | 5 | 0 |
| winter | 16 | 3 | 4 | 0 |

## FULL96-only records

These are the long-distance pairs NEAR excludes. They are placed in FULL96's training split only, so they cannot enter the common test or validation set.

| group | split | records |
|---|---|---:|
| autumn | train | 13 |
| summer | train | 3 |

## Invariants

| check | result |
|---|---|
| common test equals full test | PASS |
| common test equals near test | PASS |
| common test not in full train or val | PASS |
| common test not in near train or val | PASS |
| common val not in full train | PASS |
| common val not in near train | PASS |
| full partition is disjoint | PASS |
| near partition is disjoint | PASS |
| full unique dates | PASS |
| near unique dates | PASS |

## Use

Any FULL96-vs-NEAR performance comparison must report **per-day paired results on the common test set**, using the same days for both. The two datasets' original test splits may be shown as supplementary context only, and cannot be the main evidence for a filtering effect: they are not the same days and not independent.

- Protocol: `data/processed/v2_cy_jn_2025/common_split.json`
- Per-day test index: `data/processed/v2_cy_jn_2025/common_test_index.csv`
- Command: `src/build_common_split.py`
