# V2 load-price matching audit

## Frozen input

- Version: **v2-frozen-dfc9d60cc222**
- Manifest sha256: `ebed1801c43c32a9a4aeb2758171778b8481491f948ea73a1d179bddd251730f`
- Six user sets, 123 candidate dates, 31 representative dates and the `line50_q2.0` support assumption are unchanged.

## Result

| class | candidates (123) | representative (31) |
|---|---:|---:|
| exact | 0 | 0 |
| partial | 0 | 0 |
| missing | 123 | 31 |
| unverifiable | 0 | 0 |

- Exact matches: **0 / 123** candidates, **0 / 31** representatives.
- The existing real single-day curve is dated 2026-10-09; no frozen load date is 2026-10-09. No adjacent-date or window-mean substitution was made.

## Matching rule

- The price must have the **same actual date** as the load day.
- It must be a traceable real single-day market curve, not a window mean, synthetic library or scenario output.
- It must have 96 periods of 15 minutes and an axis that aligns with the load's 96 periods. `single_day_data.check_alignment` is the authority; no shift or resampling is applied.
- Missing is reported as missing; it is never filled with a neighbour or a mean.

## Coverage by season

| season | candidates | representatives | exact | missing |
|---|---:|---:|---:|---:|
| spring | 20 | 7 | 0 | 20 |
| summer | 21 | 6 | 0 | 21 |
| autumn | 52 | 9 | 0 | 52 |
| winter | 30 | 9 | 0 | 30 |

## Coverage by group

| group | candidates | representatives | exact | missing |
|---|---:|---:|---:|---:|
| autumn | 47 | 6 | 0 | 47 |
| national_day_2022 | 5 | 3 | 0 | 5 |
| spring | 20 | 7 | 0 | 20 |
| spring_festival_2022 | 7 | 3 | 0 | 7 |
| summer | 21 | 6 | 0 | 21 |
| winter | 23 | 6 | 0 | 23 |

## Main gaps

- No candidate date has a same-date real daily price. The single real day is four years away from the 2022 load dates.
- Spring Festival and National Day 2022 are represented as load groups, but neither has a real 2022 day-ahead price curve.
- The window means cannot substitute for missing dates; doing so would erase day-to-day variation and leak the same curve across dates.

## Training gate

No train/validation/test split is generated. The minimum to reach the formal training gate is:

1. real 15-minute day-ahead price curves for the **31 representative dates** (all six groups), or preferably all 123 candidate dates. The 31 representative days meet the earlier ≥30 distinct priced-day minimum, but a 70/15/15 split on them is thin (~22/5/4); the full 123-day set is preferred;
2. same market/region/product as the load study (Jiangsu day-ahead spot), with zonal/nodal scope stated;
3. CNY/MWh, tax status, timezone and interval convention stated;
4. a traceable export (source, file sha256, field definitions) for each date;
5. no window means, no synthetic values and no reused/adjacent dates.

Until those exist, the available exact matches are **0** and the candidate library cannot be priced.

## Artifact

- `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_load_price_matching_audit.csv`
- CSV is gitignored by the repository rule (`*.csv`): **True**. The file exists on disk but is not versioned; the rule was not changed.

## Command

`src/audit_v2_price_matching.py`
