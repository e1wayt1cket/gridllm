# V2 typical-day selection

## Verdict

- Candidate universe: **149 real dates** on which the provisional **15-user** core is complete at the all-meters standard.
- Representative dates: **36**
- Seasons represented: **autumn, spring, summer, winter**
- No date is scaled, copied, interpolated or synthesised. Holiday positions are taken from the actual holiday period.

## Source

- Provisional core: 15 users, 11 buses, 8 industries.
- Core dates: 149 ({'autumn': 69, 'spring': 25, 'summer': 33, 'winter': 22}).
- Core holiday counts: {'mid_autumn': 3, 'national_day': 6, 'new_year': 1, 'qingming': 3, 'spring_festival': 5}.
- The core is a coverage candidate from Phase A; the final V2 subset is not frozen.

## Candidate coverage

| season | candidates | representative |
|---|---:|---:|
| spring | 25 | 8 |
| summer | 33 | 6 |
| autumn | 69 | 12 |
| winter | 22 | 10 |

| day type | candidates | representative |
|---|---:|---:|
| weekday | 93 | 12 |
| weekend | 42 | 12 |
| holiday | 14 | 12 |

| season | weekday | weekend | holiday |
|---|---:|---:|---:|
| spring | 15 | 8 | 2 |
| summer | 22 | 11 | 0 |
| autumn | 45 | 17 | 7 |
| winter | 11 | 6 | 5 |

## Representative dates

| date | season | day type | holiday | offset | peak MW | energy MWh | load factor | reason |
|---|---|---|---|---:|---:|---:|---:|---|
| 2021-12-19 | winter | weekend |  |  | 59.145 | 1158.634 | 0.816 | closest_to_winter_weekend_median |
| 2021-12-26 | winter | weekend |  |  | 62.901 | 1185.653 | 0.785 | closest_to_winter_weekend_median |
| 2022-01-03 | winter | holiday | New Year | 2 | 59.313 | 1182.708 | 0.831 | holiday_new_year_2022_offset_2 |
| 2022-01-09 | winter | weekend |  |  | 56.281 | 1113.682 | 0.824 | closest_to_winter_weekend_median |
| 2022-01-21 | winter | weekday |  |  | 58.938 | 1156.720 | 0.818 | closest_to_winter_weekday_median |
| 2022-01-24 | winter | weekday |  |  | 58.348 | 1162.434 | 0.830 | closest_to_winter_weekday_median |
| 2022-01-29 | winter | weekday | Spring Festival make-up workday |  | 58.154 | 1157.558 | 0.829 | closest_to_winter_weekday_median |
| 2022-02-02 | winter | holiday | Spring Festival | 1 | 54.644 | 1040.880 | 0.794 | holiday_spring_festival_2022_offset_1 |
| 2022-02-03 | winter | holiday | Spring Festival | 2 | 54.659 | 1071.268 | 0.817 | holiday_spring_festival_2022_offset_2 |
| 2022-02-06 | winter | holiday | Spring Festival | 5 | 54.500 | 1059.801 | 0.810 | holiday_spring_festival_2022_offset_5 |
| 2022-03-01 | spring | weekday |  |  | 64.532 | 1302.920 | 0.841 | closest_to_spring_weekday_median |
| 2022-03-06 | spring | weekend |  |  | 58.849 | 1239.348 | 0.877 | closest_to_spring_weekend_median |
| 2022-03-16 | spring | weekday |  |  | 65.547 | 1225.118 | 0.779 | closest_to_spring_weekday_median |
| 2022-03-26 | spring | weekend |  |  | 58.620 | 1164.733 | 0.828 | closest_to_spring_weekend_median |
| 2022-04-03 | spring | holiday | Qingming | -1 | 48.526 | 835.275 | 0.717 | holiday_qingming_2022_offset_-1 |
| 2022-04-04 | spring | holiday | Qingming | 0 | 32.495 | 653.310 | 0.838 | holiday_qingming_2022_offset_0 |
| 2022-04-10 | spring | weekend |  |  | 60.397 | 1233.820 | 0.851 | closest_to_spring_weekend_median |
| 2022-04-11 | spring | weekday |  |  | 61.657 | 1244.205 | 0.841 | closest_to_spring_weekday_median |
| 2022-07-24 | summer | weekend |  |  | 77.003 | 1497.130 | 0.810 | closest_to_summer_weekend_median |
| 2022-08-13 | summer | weekend |  |  | 77.009 | 1551.920 | 0.840 | closest_to_summer_weekend_median |
| 2022-08-16 | summer | weekday |  |  | 68.860 | 1442.187 | 0.873 | closest_to_summer_weekday_median |
| 2022-08-21 | summer | weekend |  |  | 74.755 | 1516.213 | 0.845 | closest_to_summer_weekend_median |
| 2022-08-22 | summer | weekday |  |  | 67.346 | 1413.053 | 0.874 | closest_to_summer_weekday_median |
| 2022-08-29 | summer | weekday |  |  | 68.244 | 1393.523 | 0.851 | closest_to_summer_weekday_median |
| 2022-09-03 | autumn | weekend |  |  | 74.032 | 1483.554 | 0.835 | closest_to_autumn_weekend_median |
| 2022-09-06 | autumn | weekday |  |  | 73.201 | 1475.688 | 0.840 | closest_to_autumn_weekday_median |
| 2022-09-10 | autumn | holiday | Mid-Autumn Festival | 0 | 57.131 | 1267.124 | 0.924 | holiday_mid_autumn_2022_offset_0 |
| 2022-09-11 | autumn | holiday | Mid-Autumn Festival | 1 | 50.071 | 1049.753 | 0.874 | holiday_mid_autumn_2022_offset_1 |
| 2022-09-12 | autumn | holiday | Mid-Autumn Festival | 2 | 57.338 | 929.565 | 0.676 | holiday_mid_autumn_2022_offset_2 |
| 2022-10-04 | autumn | holiday | National Day | 3 | 81.231 | 1629.416 | 0.836 | holiday_national_day_2022_offset_3 |
| 2022-10-05 | autumn | holiday | National Day | 4 | 78.617 | 1532.925 | 0.812 | holiday_national_day_2022_offset_4 |
| 2022-10-07 | autumn | holiday | National Day | 6 | 78.333 | 1564.403 | 0.832 | holiday_national_day_2022_offset_6 |
| 2022-10-19 | autumn | weekday |  |  | 72.406 | 1496.075 | 0.861 | closest_to_autumn_weekday_median |
| 2022-10-23 | autumn | weekend |  |  | 72.212 | 1498.251 | 0.864 | closest_to_autumn_weekend_median |
| 2022-10-30 | autumn | weekend |  |  | 71.382 | 1481.275 | 0.865 | closest_to_autumn_weekend_median |
| 2022-11-17 | autumn | weekday |  |  | 71.438 | 1454.743 | 0.848 | closest_to_autumn_weekday_median |

## Selection rule

- Ordinary weekdays and weekends: the three candidates closest to the stratum's median coincident peak and median daily energy. This is a centrality rule, not an extreme-value rule.
- Holidays: up to three real positions per holiday instance, preferring the festival day (offset 0), the eve (-1) and the next day (+1), then the earliest and latest available positions.
- Holiday instances are grouped by holiday class and year, so the 2021 and 2022 Spring Festivals are not collapsed into one fixed month-day.

## Data gaps

- A season or day type with no row above has no complete real candidate under the provisional core; it is reported, not filled.
- The candidate loads are the core's measured load. They are not required to match the 3.715 MW IEEE33 reference; that comparison belongs to Phase C.
- Price coverage is not part of this file; it is audited separately and remains the blocking input for training.

## Artifacts

- `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_typical_day_candidates.csv`

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_date_summary.csv` | `c62dc7a5ca9f090582894c05852a727f5a51920cf41a9f4c84d9a02b896c3acc` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_provisional_core.json` | `45653b8b1aedda3cdd84a8f9bc457018e1eb6e91df23e17ceb06a02f13004ad8` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_typical_day_candidates.csv` | `4bf3563617c797bde7fe672495889eb999aae6268e8f02668236244252bb663a` |

## Command

`src/select_v2_typical_days.py`
