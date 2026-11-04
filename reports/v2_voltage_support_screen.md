# V2 voltage-support re-screen

## Verdict

This is a **forced-serve power-flow screen with voltage-support assumptions**, not a market clearing. The measured loads are unchanged; the support is a diagnostic assumption on the network, not a frozen V2 design.

The levers are: feeder-head voltage (OLTC), ideal local reactive compensation proportional to each bus's load Q (SVC/inverter), and a uniform line R/X scale (network reinforcement). The line scale is a sensitivity, not a proposal to edit the benchmark data.

Each group is screened on all of its representative days.

| setting | slack | Q fraction | line R/X | group | dates | voltage violations | line overloads | worst min V | max V | passes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| line50_q2.0 | 1.0 | 2.0 | 0.5 | spring | 7 | 0 | 0 | 0.9841 | 1.0155 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | summer | 6 | 0 | 0 | 0.9551 | 1.0083 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | autumn | 6 | 0 | 0 | 0.9448 | 1.0033 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | winter | 6 | 0 | 0 | 0.9468 | 1.0043 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | spring_festival_2022 | 3 | 0 | 0 | 0.9804 | 1.0149 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | national_day_2022 | 3 | 0 | 0 | 0.9586 | 1.0074 | yes |

| line25_q2.0 | 1.0 | 2.0 | 0.25 | spring | 7 | 0 | 0 | 0.9924 | 1.0078 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | summer | 6 | 0 | 0 | 0.9804 | 1.0042 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | autumn | 6 | 0 | 0 | 0.9764 | 1.0017 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | winter | 6 | 0 | 0 | 0.9768 | 1.0022 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | spring_festival_2022 | 3 | 0 | 0 | 0.9907 | 1.0076 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | national_day_2022 | 3 | 0 | 0 | 0.9817 | 1.0037 | yes |

## Conclusion

The first setting in the screened ladder that makes every group pass is **line50_q2.0**. This is not a cost-optimal design: no device rating, cost or control dynamics are modeled. The network, the support rating and the V2 subset still have to be frozen deliberately.

## Limits

- The support is ideal and local; no device rating, cost, dispatch or control dynamics are modelled.
- The screen does not model storage, bids, market clearing, congestion pricing or shedding.
- No load is scaled; the seasonal subset user sets and measured curves are unchanged.
- The price blocker remains: no real dated prices exist for these days, so no training or priced evaluation is possible.

## Artifacts

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_subsets.csv` | `3ea3b009289194aa3b5d8e7b0afdbb148d0da5086423194cfbd1ad6aaec014ca` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_candidates.csv` | `05b63da48f2585044635af745a0c18e76565001a78fead03b122cd285891a369` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_summary.csv` | `8464c14a92e1d0ff777a9b0e7b5ef5b6c2bc962cdbe1b9c40c9f6ae52d0b92de` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_day_curves.npz` | `8ff8cbc00618189d001565068bf8a95c9b9975960baa0ce6343ad508075be472` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_voltage_support_screen.csv` | `89dabb8f9b39ceb9abfa52f58b9c087129f0049cfef99d83f2df9c99865a416f` |

## Command

`src/screen_v2_voltage_support.py --settings line50_q2.0,line25_q2.0`
