# V2 voltage-support re-screen

## Verdict

This is a **forced-serve power-flow screen with voltage-support assumptions**, not a market clearing. The measured loads are unchanged; the support is a diagnostic assumption on the network, not a frozen V2 design.

The levers are: feeder-head voltage (OLTC), ideal local reactive compensation proportional to each bus's load Q (SVC/inverter), and a uniform line R/X scale (network reinforcement). The line scale is a sensitivity, not a proposal to edit the benchmark data.

Each group is screened on its **worst baseline representative day** (the day with the lowest minimum voltage in the baseline screen).

| setting | slack | Q fraction | line R/X | group | dates | voltage violations | line overloads | worst min V | max V | passes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| baseline | 1.0 | 0.0 | 1.0 | spring | 1 | 11 | 0 | 0.9112 | 1.0091 | no |
| baseline | 1.0 | 0.0 | 1.0 | summer | 1 | 89 | 0 | 0.7403 | 1.0021 | no |
| baseline | 1.0 | 0.0 | 1.0 | autumn | 1 | 85 | 1 | 0.6658 | 1.0016 | no |
| baseline | 1.0 | 0.0 | 1.0 | winter | 1 | 87 | 0 | 0.6945 | 1.0000 | no |
| baseline | 1.0 | 0.0 | 1.0 | spring_festival_2022 | 1 | 67 | 0 | 0.8820 | 1.0064 | no |
| baseline | 1.0 | 0.0 | 1.0 | national_day_2022 | 1 | 80 | 0 | 0.7620 | 1.0026 | no |

| slack_1.05 | 1.05 | 0.0 | 1.0 | spring | 1 | 15 | 0 | 0.9661 | 1.0587 | no |
| slack_1.05 | 1.05 | 0.0 | 1.0 | summer | 1 | 38 | 0 | 0.8126 | 1.0520 | no |
| slack_1.05 | 1.05 | 0.0 | 1.0 | autumn | 1 | 67 | 0 | 0.7523 | 1.0516 | no |
| slack_1.05 | 1.05 | 0.0 | 1.0 | winter | 1 | 49 | 0 | 0.7731 | 1.0502 | no |
| slack_1.05 | 1.05 | 0.0 | 1.0 | spring_festival_2022 | 1 | 14 | 0 | 0.9388 | 1.0562 | no |
| slack_1.05 | 1.05 | 0.0 | 1.0 | national_day_2022 | 1 | 41 | 0 | 0.8311 | 1.0525 | no |

| q1.0 | 1.0 | 1.0 | 1.0 | spring | 1 | 0 | 0 | 0.9383 | 1.0199 | yes |
| q1.0 | 1.0 | 1.0 | 1.0 | summer | 1 | 58 | 0 | 0.8113 | 1.0039 | no |
| q1.0 | 1.0 | 1.0 | 1.0 | autumn | 1 | 71 | 0 | 0.7566 | 1.0036 | no |
| q1.0 | 1.0 | 1.0 | 1.0 | winter | 1 | 57 | 0 | 0.7755 | 1.0023 | no |
| q1.0 | 1.0 | 1.0 | 1.0 | spring_festival_2022 | 1 | 9 | 0 | 0.9192 | 1.0174 | no |
| q1.0 | 1.0 | 1.0 | 1.0 | national_day_2022 | 1 | 54 | 0 | 0.8277 | 1.0043 | no |

| q2.5 | 1.0 | 2.5 | 1.0 | spring | 1 | 0 | 0 | 0.9789 | 1.0363 | yes |
| q2.5 | 1.0 | 2.5 | 1.0 | summer | 1 | 1 | 0 | 0.9123 | 1.0109 | no |
| q2.5 | 1.0 | 2.5 | 1.0 | autumn | 1 | 3 | 0 | 0.8810 | 1.0185 | no |
| q2.5 | 1.0 | 2.5 | 1.0 | winter | 1 | 25 | 0 | 0.8921 | 1.0206 | no |
| q2.5 | 1.0 | 2.5 | 1.0 | spring_festival_2022 | 1 | 0 | 0 | 0.9743 | 1.0340 | yes |
| q2.5 | 1.0 | 2.5 | 1.0 | national_day_2022 | 1 | 1 | 0 | 0.9218 | 1.0170 | no |

| line50 | 1.0 | 0.0 | 0.5 | spring | 1 | 0 | 0 | 0.9573 | 1.0048 | yes |
| line50 | 1.0 | 0.0 | 0.5 | summer | 1 | 28 | 0 | 0.8905 | 1.0011 | no |
| line50 | 1.0 | 0.0 | 0.5 | autumn | 1 | 54 | 0 | 0.8691 | 1.0010 | no |
| line50 | 1.0 | 0.0 | 0.5 | winter | 1 | 43 | 0 | 0.8746 | 1.0003 | no |
| line50 | 1.0 | 0.0 | 0.5 | spring_festival_2022 | 1 | 0 | 0 | 0.9441 | 1.0036 | yes |
| line50 | 1.0 | 0.0 | 0.5 | national_day_2022 | 1 | 30 | 0 | 0.8977 | 1.0013 | no |

| line50_q1.0 | 1.0 | 1.0 | 0.5 | spring | 1 | 0 | 0 | 0.9707 | 1.0102 | yes |
| line50_q1.0 | 1.0 | 1.0 | 0.5 | summer | 1 | 1 | 0 | 0.9231 | 1.0021 | no |
| line50_q1.0 | 1.0 | 1.0 | 0.5 | autumn | 1 | 4 | 0 | 0.9070 | 1.0019 | no |
| line50_q1.0 | 1.0 | 1.0 | 0.5 | winter | 1 | 25 | 0 | 0.9106 | 1.0014 | no |
| line50_q1.0 | 1.0 | 1.0 | 0.5 | spring_festival_2022 | 1 | 0 | 0 | 0.9624 | 1.0091 | yes |
| line50_q1.0 | 1.0 | 1.0 | 0.5 | national_day_2022 | 1 | 1 | 0 | 0.9284 | 1.0022 | no |

| line75_q2.0 | 1.0 | 2.0 | 0.75 | spring | 1 | 0 | 0 | 0.9751 | 1.0232 | yes |
| line75_q2.0 | 1.0 | 2.0 | 0.75 | summer | 1 | 1 | 0 | 0.9227 | 1.0044 | no |
| line75_q2.0 | 1.0 | 2.0 | 0.75 | autumn | 1 | 3 | 0 | 0.9022 | 1.0047 | no |
| line75_q2.0 | 1.0 | 2.0 | 0.75 | winter | 1 | 21 | 0 | 0.9074 | 1.0043 | no |
| line75_q2.0 | 1.0 | 2.0 | 0.75 | spring_festival_2022 | 1 | 0 | 0 | 0.9695 | 1.0216 | yes |
| line75_q2.0 | 1.0 | 2.0 | 0.75 | national_day_2022 | 1 | 1 | 0 | 0.9293 | 1.0046 | no |

| line60_q2.0 | 1.0 | 2.0 | 0.6 | spring | 1 | 0 | 0 | 0.9806 | 1.0186 | yes |
| line60_q2.0 | 1.0 | 2.0 | 0.6 | summer | 1 | 0 | 0 | 0.9431 | 1.0035 | yes |
| line60_q2.0 | 1.0 | 2.0 | 0.6 | autumn | 1 | 1 | 0 | 0.9293 | 1.0039 | no |
| line60_q2.0 | 1.0 | 2.0 | 0.6 | winter | 1 | 0 | 0 | 0.9323 | 1.0036 | yes |
| line60_q2.0 | 1.0 | 2.0 | 0.6 | spring_festival_2022 | 1 | 0 | 0 | 0.9764 | 1.0175 | yes |
| line60_q2.0 | 1.0 | 2.0 | 0.6 | national_day_2022 | 1 | 0 | 0 | 0.9477 | 1.0037 | yes |

| line50_q1.5 | 1.0 | 1.5 | 0.5 | spring | 1 | 0 | 0 | 0.9774 | 1.0128 | yes |
| line50_q1.5 | 1.0 | 1.5 | 0.5 | summer | 1 | 0 | 0 | 0.9393 | 1.0025 | yes |
| line50_q1.5 | 1.0 | 1.5 | 0.5 | autumn | 1 | 1 | 0 | 0.9261 | 1.0024 | no |
| line50_q1.5 | 1.0 | 1.5 | 0.5 | winter | 1 | 1 | 0 | 0.9287 | 1.0020 | no |
| line50_q1.5 | 1.0 | 1.5 | 0.5 | spring_festival_2022 | 1 | 0 | 0 | 0.9716 | 1.0119 | yes |
| line50_q1.5 | 1.0 | 1.5 | 0.5 | national_day_2022 | 1 | 0 | 0 | 0.9437 | 1.0027 | yes |

| line60_q1.5 | 1.0 | 1.5 | 0.6 | spring | 1 | 0 | 0 | 0.9726 | 1.0154 | yes |
| line60_q1.5 | 1.0 | 1.5 | 0.6 | summer | 1 | 1 | 0 | 0.9240 | 1.0030 | no |
| line60_q1.5 | 1.0 | 1.5 | 0.6 | autumn | 1 | 3 | 0 | 0.9066 | 1.0028 | no |
| line60_q1.5 | 1.0 | 1.5 | 0.6 | winter | 1 | 22 | 0 | 0.9104 | 1.0023 | no |
| line60_q1.5 | 1.0 | 1.5 | 0.6 | spring_festival_2022 | 1 | 0 | 0 | 0.9654 | 1.0142 | yes |
| line60_q1.5 | 1.0 | 1.5 | 0.6 | national_day_2022 | 1 | 1 | 0 | 0.9297 | 1.0032 | no |

| line50_q2.0 | 1.0 | 2.0 | 0.5 | spring | 1 | 0 | 0 | 0.9841 | 1.0155 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | summer | 1 | 0 | 0 | 0.9551 | 1.0030 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | autumn | 1 | 0 | 0 | 0.9448 | 1.0033 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | winter | 1 | 0 | 0 | 0.9468 | 1.0031 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | spring_festival_2022 | 1 | 0 | 0 | 0.9808 | 1.0146 | yes |
| line50_q2.0 | 1.0 | 2.0 | 0.5 | national_day_2022 | 1 | 0 | 0 | 0.9586 | 1.0031 | yes |

| line25_q2.0 | 1.0 | 2.0 | 0.25 | spring | 1 | 0 | 0 | 0.9924 | 1.0078 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | summer | 1 | 0 | 0 | 0.9804 | 1.0015 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | autumn | 1 | 0 | 0 | 0.9764 | 1.0017 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | winter | 1 | 0 | 0 | 0.9768 | 1.0016 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | spring_festival_2022 | 1 | 0 | 0 | 0.9909 | 1.0074 | yes |
| line25_q2.0 | 1.0 | 2.0 | 0.25 | national_day_2022 | 1 | 0 | 0 | 0.9817 | 1.0016 | yes |

| slack1.05_q1.0 | 1.05 | 1.0 | 1.0 | spring | 1 | 47 | 0 | 0.9944 | 1.0701 | no |
| slack1.05_q1.0 | 1.05 | 1.0 | 1.0 | summer | 1 | 19 | 0 | 0.8849 | 1.0540 | no |
| slack1.05_q1.0 | 1.05 | 1.0 | 1.0 | autumn | 1 | 39 | 0 | 0.8408 | 1.0537 | no |
| slack1.05_q1.0 | 1.05 | 1.0 | 1.0 | winter | 1 | 34 | 0 | 0.8545 | 1.0525 | no |
| slack1.05_q1.0 | 1.05 | 1.0 | 1.0 | spring_festival_2022 | 1 | 47 | 0 | 0.9778 | 1.0678 | no |
| slack1.05_q1.0 | 1.05 | 1.0 | 1.0 | national_day_2022 | 1 | 17 | 0 | 0.8986 | 1.0543 | no |

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
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_voltage_support_ladder.csv` | `1d0258db28181df548528367aadef237a0048395dbca05d2c17f9dc4213ce6ab` |

## Command

`src/screen_v2_voltage_support.py --worst-only --out data/processed/v2/v2_voltage_support_ladder.csv --report reports/v2_voltage_support_ladder.md`
