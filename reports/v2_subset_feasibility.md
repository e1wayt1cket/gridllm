# V2 fixed-subset feasibility

## Verdict

This is a **load-coverage screen** on the representative dates selected in Phase B. It does not freeze a subset and does not claim that a subset is physically operable; the OPF/network check belongs to the next step.

The key trade-off is coverage versus breadth: a smaller fixed subset survives more seasons and dates; a larger subset adds buses and industries but loses seasons because some users have few complete all-meters days.

## Subsets compared

| subset | users | buses | industries | V1 overlap | all-meters dates | representative days covered | spring | summer | autumn | winter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top03_all_meters | 3 | 2 | 2 | 2 | 713 | 36 | 8 | 6 | 12 | 10 |
| top05_all_meters | 5 | 4 | 4 | 4 | 700 | 36 | 8 | 6 | 12 | 10 |
| top08_all_meters | 8 | 6 | 5 | 5 | 661 | 36 | 8 | 6 | 12 | 10 |
| top10_all_meters | 10 | 7 | 6 | 6 | 605 | 36 | 8 | 6 | 12 | 10 |
| top12_all_meters | 12 | 9 | 7 | 8 | 512 | 36 | 8 | 6 | 12 | 10 |
| top15_provisional_core | 15 | 11 | 8 | 9 | 149 | 36 | 8 | 6 | 12 | 10 |
| top20_all_meters | 20 | 12 | 9 | 11 | 97 | 17 | 0 | 5 | 12 | 0 |
| v1_17 | 17 | 9 | 9 | 17 | 0 | 0 | 0 | 0 | 0 | 0 |

## Load scale on covered representative days

| subset | median peak MW | median peak / 3.715 | peak range MW | median energy MWh | top bus energy share |
|---|---:|---:|---|---:|---:|
| top03_all_meters | 10.471 | 2.82x | 7.828-12.381 | 188.741 | 99.8% (bus 25) |
| top05_all_meters | 10.486 | 2.82x | 7.845-12.468 | 189.309 | 99.5% (bus 25) |
| top08_all_meters | 18.385 | 4.95x | 13.603-23.842 | 317.833 | 58.6% (bus 25) |
| top10_all_meters | 18.913 | 5.09x | 13.842-24.424 | 325.792 | 57.1% (bus 25) |
| top12_all_meters | 27.025 | 7.27x | 20.390-36.357 | 445.486 | 41.8% (bus 25) |
| top15_provisional_core | 62.279 | 16.76x | 32.495-81.232 | 1241.776 | 59.3% (bus 23) |
| top20_all_meters | 94.259 | 25.37x | 70.667-107.194 | 1856.057 | 48.3% (bus 23) |
| v1_17 |  |  |  |  |  |

## Projection of the 3.715 MW reference

3.715 MW is the IEEE33 reference, not a target. The table above is the measured load of each candidate subset; a subset far above the reference may still be a real observation, but it needs a separate physical feasibility check before it is used as a market scenario.

## Provisional core

- Phase A core: **15 users**, 11 buses, 8 industries, 149 all-meters complete dates.
- Season counts: {'autumn': 69, 'spring': 25, 'summer': 33, 'winter': 22}.
- Holiday counts: {'mid_autumn': 3, 'national_day': 6, 'new_year': 1, 'qingming': 3, 'spring_festival': 5}.
- Representative dates selected from this core: 36.

## Provisional core bus distribution

| bus | users | energy MWh | share |
|---:|---:|---:|---:|
| 2 | 1 | 2779.052 | 6.1% |
| 3 | 1 | 4071.214 | 8.9% |
| 4 | 1 | 451.295 | 1.0% |
| 5 | 1 | 4212.956 | 9.2% |
| 23 | 2 | 27213.729 | 59.3% |
| 24 | 1 | 162.787 | 0.4% |
| 25 | 3 | 6639.908 | 14.5% |
| 26 | 1 | 22.937 | 0.1% |
| 27 | 2 | 42.911 | 0.1% |
| 30 | 1 | 0.936 | 0.0% |
| 31 | 1 | 267.140 | 0.6% |

## Recommendation

- No multi-season subset in this screen has a median peak at or below 2x the 3.715 MW reference. A physically reasonable multi-season fixed subset is therefore **not established**; this is a data/scale gap, not a tuning problem.
- Broadest multi-season subset: **top15_provisional_core** (15 users, 11 buses, 8 industries, median peak 62.279 MW, 16.76x).
- The smaller subsets keep more complete dates but cover fewer buses and industries; the larger subsets lose seasons at the all-meters standard.
- The V1 17-user subset is retained as the historical single-day benchmark. At the all-meters standard it covers none of the selected representative dates, so it is not a drop-in V2 multi-season subset.
- Do not freeze a subset until the breadth/coverage/scale trade-off has been reviewed and a physical OPF/network check has been run on the chosen days.

## Artifacts

- `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_subset_feasibility.csv`
- `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_core_bus_distribution.csv`

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\v2_user_date_coverage.csv` | `5ffae19bd152bf21553b2ae49c7fef09b997a2901dd8b6963a37857472082964` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_summary.csv` | `8464c14a92e1d0ff777a9b0e7b5ef5b6c2bc962cdbe1b9c40c9f6ae52d0b92de` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_provisional_core.json` | `45653b8b1aedda3cdd84a8f9bc457018e1eb6e91df23e17ceb06a02f13004ad8` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_typical_day_candidates.csv` | `4bf3563617c797bde7fe672495889eb999aae6268e8f02668236244252bb663a` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_day_curves.npz` | `8ff8cbc00618189d001565068bf8a95c9b9975960baa0ce6343ad508075be472` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_subset_feasibility.csv` | `2ff8daf170c7420e2555abad802563020960aa00b5dd6ade0e165e2b8cb97e7c` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_core_bus_distribution.csv` | `60b7b3c140bc1d2d62a3ec7b9b99f969a64a13190e7a24fbf4abc23fce34e8f9` |

## Command

`src/analyze_v2_subset_feasibility.py`
