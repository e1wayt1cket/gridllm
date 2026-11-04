# V2 scenario library frozen

## Status

- Version: **v2-frozen-dfc9d60cc222**
- Frozen at: `2026-10-21T19:58:57+08:00`
- The library contains **six different fixed user sets**, not one system. A result from spring is not directly comparable to a result from winter or from a holiday subset.
- V1 is preserved as the historical single-day benchmark.
- Formal training is **not** started.

## Groups

| group | type | season | users | buses | industries | common dates | representative days | median peak MW | /3.715 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| spring | season | spring | 12 | 8 | 8 | 20 | 7 | 1.852 | 0.50x |
| summer | season | summer | 17 | 9 | 9 | 21 | 6 | 4.656 | 1.25x |
| autumn | season | autumn | 20 | 10 | 10 | 47 | 6 | 5.553 | 1.49x |
| winter | season | winter | 20 | 11 | 11 | 23 | 6 | 5.141 | 1.38x |
| spring_festival_2022 | holiday | winter | 11 | 9 | 9 | 7 | 3 | 1.860 | 0.50x |
| national_day_2022 | holiday | autumn | 19 | 9 | 9 | 5 | 3 | 4.402 | 1.19x |

## User sets

The frozen user lists are in the manifest and the frozen CSV. They are deliberately different; do not read a cross-season difference as a policy difference without naming the user-set change.

## Network support assumption

- Assumption name: `line50_q2.0`.
- Slack voltage: 1.0 p.u.
- Uniform line R/X scale: 0.5 (all in-service lines).
- Ideal local reactive compensation: 2.0x each bus's load Q.
- This is a screening assumption; no device rating, cost, control dynamics or market clearing is modeled. It is not a cost-optimal network design.
- Representative days screened: 31; all pass: **True**.

| group | worst min V | max V | voltage violations | line overloads |
|---|---:|---:|---:|---:|
| autumn | 0.9448 | 1.0033 | 0 | 0 |
| national_day_2022 | 0.9586 | 1.0074 | 0 | 0 |
| spring | 0.9841 | 1.0155 | 0 | 0 |
| spring_festival_2022 | 0.9804 | 1.0149 | 0 | 0 |
| summer | 0.9551 | 1.0083 | 0 | 0 |
| winter | 0.9468 | 1.0043 | 0 | 0 |

## Price coverage and training gate

- Candidate load days: 123
- Representative days: 31
- Candidate days with a real dated price: 0
- Representative days with a real dated price: 0
- Formal training blocked: **True**.
- Reason: no real dated day-ahead price curve covers the selected candidate or representative days; window means are not daily prices and are not used.

## Artifacts

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_subsets.csv` | `3ea3b009289194aa3b5d8e7b0afdbb148d0da5086423194cfbd1ad6aaec014ca` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_candidates.csv` | `05b63da48f2585044635af745a0c18e76565001a78fead03b122cd285891a369` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_voltage_support_screen.csv` | `89dabb8f9b39ceb9abfa52f58b9c087129f0049cfef99d83f2df9c99865a416f` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\v2_load_price_pairing.csv` | `27d5afc69f497fbf9b18dd9a62f45c5f9bef59fe51a8eddbdff81b71dff43b3d` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_seasonal_subsets_frozen.csv` | `95f3fade38efecfb4d0004a23ccc9b355fcced333a162cecdd7168bb25ad9624` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_scenario_library_manifest.json` | `ebed1801c43c32a9a4aeb2758171778b8481491f948ea73a1d179bddd251730f` |

## Command

`src/freeze_v2_scenario_library.py`
