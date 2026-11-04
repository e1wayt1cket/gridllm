# Holdout exposure audit (V2-CY-JN-2025)

No training and no policy evaluation. This separates raw-data quality inspection from policy-result exposure and asks whether the existing 2022 load records and 2025 Jiangnan prices can support a prospective holdout.

## Definitions

- **raw_viewed**: the date was read by the data-quality audit. It is not, by itself, leakage.
- **historical exposure**: the date entered the pilot training pool, was actually visited, was used in historical validation or model selection, or had rule-baseline/truthful performance computed.
- **used_for_selection**: historical validation/test/holiday membership (the pilot used these to compare designs). **reserved_for_selection** is separate: it marks the frozen NEAR split or the proposed FULL96 split, which are designated for future selection but have not been used yet.
- **H1 joint holdout**: load and price both outside the historical and new-algorithm training/validation exposure, paired in the same season, with the pair unused before.
- **H2 price-date holdout**: a complete 2025 price date never in FULL96, paired with an existing load scenario. The load side is historical and is disclosed per pair.

## Date inventory

| item | value |
|---|---:|
| raw load dates audited | 730 |
| 17-user complete load days | 103 |
| V2 load candidates | 123 |
| FULL96 load dates | 96 |
| NEAR load dates | 80 |
| 2025 price dates in source | 137 |
| complete 2025 price dates | 136 |
| FULL96 price dates | 96 |
| unpaired complete price dates | 40 |

The unpaired price dates are {'summer': 22, 'autumn': 10, 'winter': 8}.

## Legacy seen test

The original common test (11) and national-day block (5) remain **LEGACY_SEEN_TEST** (16 pairs). They are not renamed as a blind set.

## H1 — joint scenario holdout

**Days: 0.** Seasons: none. Meets 30 days and 3 seasons: **False**.

- Unused V2 load candidates: 27 ({'spring': 20, 'spring_festival_2022': 7}). They are the spring group and the spring-festival block, not ordinary seasonal days.
- The 2025 price source covers June–December only; it has no spring price date, so the 20 spring loads cannot be paired.
- Complete 17-user load days outside the V2 library: 28 ({'autumn': 20, 'winter': 8}). They are autumn/winter only and use a different user subset; admitting them would change the load system.
- All ordinary V2 summer, autumn and winter load dates are used.
- Even if the seven spring-festival winter loads and the 28 non-V2 autumn/winter loads were admitted, the upper bound is **18 pairs over two seasons**, still below 30 days and three seasons. The frozen library gives **0**.

H1 is therefore **not available** from the current data. No date is promoted to H1.

## H2 — price-date holdout

**Days: 40.** Seasons: ['autumn', 'summer', 'winter'] ({'autumn': 10, 'summer': 22, 'winter': 8}). Meets 30 days and 3 seasons: **True**.

| item | value |
|---|---:|
| distinct load scenarios | 38 |
| load reuses | 2 |
| same-season same-type pairs | 33 |
| same-season type-mismatch pairs | 5 |
| same-type reused pairs | 2 |
| pairs whose load was visited in pilot training | 15 |
| pairs whose load was in pilot evaluation | 2 |
| pairs whose load is LEGACY_SEEN_TEST | 2 |
| pairs whose load was used for historical selection | 8 |
| pairs whose load is reserved for future selection | 17 |

Every price date in H2 is a complete 2025 date that never entered FULL96, NEAR, the pilot evaluation or the rule benchmark. The load side is historical; the per-pair columns in `holdout_pairs.csv` record exactly which exposure applies.

## New-algorithm exposure

The engineering smoke used one pair, `2022-07-22|2025-07-15`, for 17 gradient updates. Its load is excluded from the H2 load pool. No formal training has run.

## Verdict

- H1: **0 days** — the current data cannot support a 30-day, three-season joint source-load holdout.
- H2: **40 days, 3 seasons** — supports limited price-date adaptation only.
- A claim of joint source-load generalisation is **not** supported.
- The legacy common test and holiday block remain `LEGACY_SEEN_TEST`.

Artifacts: `holdout_exposure_ledger.csv`, `holdout_pairs.csv`, `holdout_manifest.json`, `holdout_status.json`.
