# V2 seasonal user subsets

## Verdict

The chosen fallback is **four seasonal subsets plus two 2022 holiday subsets**. They are not one system: a result from spring is not directly comparable to a result from winter, and Spring Festival is not the winter ordinary subset. The user sets are listed below so the difference cannot be hidden.

The subsets are **proposed**, not frozen. No training is started and no physical OPF feasibility is claimed.

## Selection rule

- Candidate users have at least the stated number of all-meters complete dates in the group.
- Users are added smallest-median-peak first while the common complete dates stay at or above the minimum and the aggregate median peak stays within the stated multiple of 3.715 MW.
- No load is scaled, no high-load period is deleted and no multiplier is edited.

## Seasonal subsets

| season | users | buses | industries | common dates | representative days | median peak MW | /3.715 | top bus share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spring | 12 | 8 | 8 | 20 | 7 | 1.852 | 0.50x | 43.0% (bus 4) |
| summer | 17 | 9 | 9 | 21 | 6 | 4.656 | 1.25x | 44.6% (bus 28) |
| autumn | 20 | 10 | 10 | 47 | 6 | 5.553 | 1.49x | 43.7% (bus 28) |
| winter | 20 | 11 | 11 | 23 | 6 | 5.141 | 1.38x | 20.1% (bus 28) |

## 2022 holiday subsets

| group | season | users | buses | industries | common dates | representative days | median peak MW | /3.715 | top bus share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spring_festival_2022 | winter | 11 | 9 | 9 | 7 | 3 | 1.860 | 0.50x | 40.0% (bus 16) |
| national_day_2022 | autumn | 19 | 9 | 9 | 5 | 3 | 4.402 | 1.19x | 52.4% (bus 28) |

## User sets (explicitly different)

### spring

| user | bus | industry |
|---|---:|---|
| U018990a98eb1 | 25 | 通用设备制造业 |
| U1f2d028ac65f | 26 | 电子设备制造业 |
| U39e8a17aba10 | 23 | 化工业 |
| U5765bec968a1 | 26 | 电子设备制造业 |
| U6e0c9c380ad3 | 27 | 纺织业 |
| U6e5b7a2d2097 | 30 | 电制氢 |
| Ub1da4ce43b04 | 24 | 汽车制造业 |
| Ue5f08866ae4e | 31 | 电动汽车（充换电） |
| Ue7a335c557ca | 4 | 商业综合体 |
| Ue971298ed105 | 23 | 化工业 |
| Ue9e499c141dd | 25 | 通用设备制造业 |
| Uf67c750d6c55 | 27 | 纺织业 |

### summer

| user | bus | industry |
|---|---:|---|
| U018990a98eb1 | 25 | 通用设备制造业 |
| U1f2d028ac65f | 26 | 电子设备制造业 |
| U2aa2cf305ab6 | 30 | 电制氢 |
| U39e8a17aba10 | 23 | 化工业 |
| U5765bec968a1 | 26 | 电子设备制造业 |
| U6e0c9c380ad3 | 27 | 纺织业 |
| U6e5b7a2d2097 | 30 | 电制氢 |
| U73fcc2041874 | 27 | 纺织业 |
| U9b80870a7722 | 28 | 机械 |
| Ub1da4ce43b04 | 24 | 汽车制造业 |
| Ucf65d248ce93 | 27 | 纺织业 |
| Ud7aaf2cc7bd7 | 28 | 机械 |
| Ue5f08866ae4e | 31 | 电动汽车（充换电） |
| Ue7a335c557ca | 4 | 商业综合体 |
| Ue971298ed105 | 23 | 化工业 |
| Ue9e499c141dd | 25 | 通用设备制造业 |
| Uf67c750d6c55 | 27 | 纺织业 |

### autumn

| user | bus | industry |
|---|---:|---|
| U018990a98eb1 | 25 | 通用设备制造业 |
| U1f2d028ac65f | 26 | 电子设备制造业 |
| U2aa2cf305ab6 | 30 | 电制氢 |
| U39e8a17aba10 | 23 | 化工业 |
| U3a097bfc3f9e | 25 | 通用设备制造业 |
| U5765bec968a1 | 26 | 电子设备制造业 |
| U6e0c9c380ad3 | 27 | 纺织业 |
| U6e5b7a2d2097 | 30 | 电制氢 |
| U73fcc2041874 | 27 | 纺织业 |
| U9b80870a7722 | 28 | 机械 |
| Ub1da4ce43b04 | 24 | 汽车制造业 |
| Ucf65d248ce93 | 27 | 纺织业 |
| Ud7aaf2cc7bd7 | 28 | 机械 |
| Udfed49bf706b | 31 | 电动汽车（充换电） |
| Ue5f08866ae4e | 31 | 电动汽车（充换电） |
| Ue7a335c557ca | 4 | 商业综合体 |
| Ue971298ed105 | 23 | 化工业 |
| Ue9e499c141dd | 25 | 通用设备制造业 |
| Uf67c750d6c55 | 27 | 纺织业 |
| Ufbfdfb7de5b9 | 29 | 印刷业 |

### winter

| user | bus | industry |
|---|---:|---|
| U018990a98eb1 | 25 | 通用设备制造业 |
| U1f2d028ac65f | 26 | 电子设备制造业 |
| U2aa2cf305ab6 | 30 | 电制氢 |
| U39e8a17aba10 | 23 | 化工业 |
| U3a097bfc3f9e | 25 | 通用设备制造业 |
| U5765bec968a1 | 26 | 电子设备制造业 |
| U6ae415563f9f | 32 | 数据中心 |
| U6e0c9c380ad3 | 27 | 纺织业 |
| U6e5b7a2d2097 | 30 | 电制氢 |
| U73fcc2041874 | 27 | 纺织业 |
| Ub1da4ce43b04 | 24 | 汽车制造业 |
| Ucf65d248ce93 | 27 | 纺织业 |
| Ud7aaf2cc7bd7 | 28 | 机械 |
| Udfed49bf706b | 31 | 电动汽车（充换电） |
| Ue5f08866ae4e | 31 | 电动汽车（充换电） |
| Ue7a335c557ca | 4 | 商业综合体 |
| Ue971298ed105 | 23 | 化工业 |
| Ue9e499c141dd | 25 | 通用设备制造业 |
| Uf67c750d6c55 | 27 | 纺织业 |
| Ufbfdfb7de5b9 | 29 | 印刷业 |

### spring_festival_2022

| user | bus | industry |
|---|---:|---|
| U018990a98eb1 | 25 | 通用设备制造业 |
| U0abfaa684d80 | 16 | 钢铁 |
| U5765bec968a1 | 26 | 电子设备制造业 |
| U6e0c9c380ad3 | 27 | 纺织业 |
| U6e5b7a2d2097 | 30 | 电制氢 |
| Ub1da4ce43b04 | 24 | 汽车制造业 |
| Ue5f08866ae4e | 31 | 电动汽车（充换电） |
| Ue7a335c557ca | 4 | 商业综合体 |
| Ue971298ed105 | 23 | 化工业 |
| Ue9e499c141dd | 25 | 通用设备制造业 |
| Uf67c750d6c55 | 27 | 纺织业 |

### national_day_2022

| user | bus | industry |
|---|---:|---|
| U018990a98eb1 | 25 | 通用设备制造业 |
| U1f2d028ac65f | 26 | 电子设备制造业 |
| U2aa2cf305ab6 | 30 | 电制氢 |
| U39e8a17aba10 | 23 | 化工业 |
| U3a097bfc3f9e | 25 | 通用设备制造业 |
| U5765bec968a1 | 26 | 电子设备制造业 |
| U6e0c9c380ad3 | 27 | 纺织业 |
| U6e5b7a2d2097 | 30 | 电制氢 |
| U73fcc2041874 | 27 | 纺织业 |
| U9b80870a7722 | 28 | 机械 |
| Ub1da4ce43b04 | 24 | 汽车制造业 |
| Ucf65d248ce93 | 27 | 纺织业 |
| Ud7aaf2cc7bd7 | 28 | 机械 |
| Udfed49bf706b | 31 | 电动汽车（充换电） |
| Ue5f08866ae4e | 31 | 电动汽车（充换电） |
| Ue7a335c557ca | 4 | 商业综合体 |
| Ue971298ed105 | 23 | 化工业 |
| Ue9e499c141dd | 25 | 通用设备制造业 |
| Uf67c750d6c55 | 27 | 纺织业 |

## Representative days

### spring

| date | day type | holiday | offset | peak MW | energy MWh | reason |
|---|---|---|---:|---:|---:|---|
| 2022-04-29 | weekday |  |  | 2.055 | 25.224 | closest_to_spring_weekday_median |
| 2022-04-30 | holiday | Labour Day | -1 | 1.648 | 22.823 | holiday_labour_day_2022_offset_-1 |
| 2022-05-21 | weekend |  |  | 1.827 | 25.983 | closest_to_spring_weekend_median |
| 2022-05-23 | weekday |  |  | 2.053 | 27.390 | closest_to_spring_weekday_median |
| 2022-05-28 | weekend |  |  | 1.806 | 28.322 | closest_to_spring_weekend_median |
| 2022-05-29 | weekend |  |  | 1.836 | 27.431 | closest_to_spring_weekend_median |
| 2022-05-30 | weekday |  |  | 2.036 | 27.551 | closest_to_spring_weekday_median |

### summer

| date | day type | holiday | offset | peak MW | energy MWh | reason |
|---|---|---|---:|---:|---:|---|
| 2022-07-22 | weekday |  |  | 4.645 | 76.573 | closest_to_summer_weekday_median |
| 2022-07-23 | weekend |  |  | 4.671 | 75.548 | closest_to_summer_weekend_median |
| 2022-08-14 | weekend |  |  | 4.790 | 76.648 | closest_to_summer_weekend_median |
| 2022-08-24 | weekday |  |  | 4.806 | 76.021 | closest_to_summer_weekday_median |
| 2022-08-25 | weekday |  |  | 4.550 | 75.867 | closest_to_summer_weekday_median |
| 2022-08-27 | weekend |  |  | 4.692 | 75.021 | closest_to_summer_weekend_median |

### autumn

| date | day type | holiday | offset | peak MW | energy MWh | reason |
|---|---|---|---:|---:|---:|---|
| 2022-10-15 | weekend |  |  | 5.316 | 81.599 | closest_to_autumn_weekend_median |
| 2022-11-10 | weekday |  |  | 5.717 | 84.697 | closest_to_autumn_weekday_median |
| 2022-11-11 | weekday |  |  | 5.526 | 86.644 | closest_to_autumn_weekday_median |
| 2022-11-13 | weekend |  |  | 5.169 | 81.255 | closest_to_autumn_weekend_median |
| 2022-11-20 | weekend |  |  | 5.332 | 81.939 | closest_to_autumn_weekend_median |
| 2022-11-25 | weekday |  |  | 5.519 | 84.279 | closest_to_autumn_weekday_median |

### winter

| date | day type | holiday | offset | peak MW | energy MWh | reason |
|---|---|---|---:|---:|---:|---|
| 2022-12-10 | weekend |  |  | 4.687 | 78.037 | closest_to_winter_weekend_median |
| 2022-12-11 | weekend |  |  | 4.397 | 78.037 | closest_to_winter_weekend_median |
| 2022-12-24 | weekend |  |  | 4.784 | 75.086 | closest_to_winter_weekend_median |
| 2022-12-27 | weekday |  |  | 5.451 | 83.702 | closest_to_winter_weekday_median |
| 2022-12-28 | weekday |  |  | 5.576 | 86.010 | closest_to_winter_weekday_median |
| 2022-12-29 | weekday |  |  | 5.395 | 84.902 | closest_to_winter_weekday_median |

### spring_festival_2022

| date | day type | holiday | offset | peak MW | energy MWh | reason |
|---|---|---|---:|---:|---:|---|
| 2022-01-31 | holiday | Spring Festival | -1 | 1.564 | 25.554 | holiday_spring_festival_2022_offset_-1 |
| 2022-02-01 | holiday | Spring Festival | 0 | 1.805 | 29.954 | holiday_spring_festival_2022_offset_0 |
| 2022-02-02 | holiday | Spring Festival | 1 | 1.766 | 29.953 | holiday_spring_festival_2022_offset_1 |

### national_day_2022

| date | day type | holiday | offset | peak MW | energy MWh | reason |
|---|---|---|---:|---:|---:|---|
| 2022-10-03 | holiday | National Day | 2 | 4.402 | 65.272 | holiday_national_day_2022_offset_2 |
| 2022-10-04 | holiday | National Day | 3 | 4.209 | 58.988 | holiday_national_day_2022_offset_3 |
| 2022-10-07 | holiday | National Day | 6 | 4.425 | 68.276 | holiday_national_day_2022_offset_6 |

## Data gaps and limits

- The 2022 Spring Festival and National Day are represented by separate holiday subsets, not by the seasonal ordinary subsets. The seasonal and holiday user sets are not the same system.
- The 2021 Spring Festival and National Day are not covered by these groups; their all-meters common sets are too small.
- Spring and winter have fewer common dates; their representative libraries are smaller than autumn's.
- The price blocker is unchanged: only one real single-day price curve is present, so no priced multi-day training set is possible.
- Physical feasibility is screened in `reports/v2_physical_screen.md` and `reports/v2_voltage_support_screen.md`. Without support every group violates the voltage band; with the `line50_q2.0` assumption (uniform line R/X at 0.5x plus ideal local reactive compensation at 2x load Q) all 31 representative days pass. The network and support rating are not frozen.

## Artifacts

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\v2_user_date_coverage.csv` | `5ffae19bd152bf21553b2ae49c7fef09b997a2901dd8b6963a37857472082964` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_summary.csv` | `8464c14a92e1d0ff777a9b0e7b5ef5b6c2bc962cdbe1b9c40c9f6ae52d0b92de` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_day_curves.npz` | `8ff8cbc00618189d001565068bf8a95c9b9975960baa0ce6343ad508075be472` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_subsets.csv` | `3ea3b009289194aa3b5d8e7b0afdbb148d0da5086423194cfbd1ad6aaec014ca` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_candidates.csv` | `05b63da48f2585044635af745a0c18e76565001a78fead03b122cd285891a369` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_seasonal_subsets.json` | `a3387c8f76057654bea40eb677e2146b36c66084de6b772bc6ae7986e78e0f1b` |

## Command

`src/select_v2_seasonal_subsets.py`
