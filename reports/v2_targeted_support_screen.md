# V2 targeted voltage-support screen

## Verdict

This is a **forced-serve power-flow screen**, not a market clearing. The measured loads are unchanged. The support is a targeted alternative to the uniform `line50_q2.0` setting: scale only the main feeder end (`to_bus` 1..17) and the branch to 32 (`to_bus` 25..32), plus fixed capacitors at buses 17 and 32.

Each group is screened on its worst baseline representative day.

| setting | path R/X | cap bus 17 | cap bus 32 | group | date | min V | max V | voltage periods | line overloads | passes |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| path_line50 | 0.5 | 0.0 | 0.0 | spring | 2022-05-23 | 0.9573116684401093 | 1.0047915150585016 | 0 | 0 | yes |
| path_line50 | 0.5 | 0.0 | 0.0 | summer | 2022-08-27 | 0.8905403136844001 | 1.0036202253279047 | 28 | 0 | no |
| path_line50 | 0.5 | 0.0 | 0.0 | autumn | 2022-11-25 | 0.8690890511302055 | 1.0034534150658996 | 54 | 0 | no |
| path_line50 | 0.5 | 0.0 | 0.0 | winter | 2022-12-28 | 0.8745767474779873 | 1.002829180157332 | 43 | 0 | no |
| path_line50 | 0.5 | 0.0 | 0.0 | spring_festival_2022 | 2022-02-02 | 0.9440833194536539 | 1.004303931469563 | 0 | 0 | yes |
| path_line50 | 0.5 | 0.0 | 0.0 | national_day_2022 | 2022-10-07 | 0.8977279439085499 | 1.003822918392085 | 30 | 0 | no |
| path_line50_cap11 | 0.5 | 1.0 | 1.0 | spring | 2022-05-23 | 0.9771106990419414 | 1.0256861339897652 | 0 | 0 | yes |
| path_line50_cap11 | 0.5 | 1.0 | 1.0 | summer | 2022-08-27 | 0.9069041322702761 | 1.007278907031687 | 5 | 0 | no |
| path_line50_cap11 | 0.5 | 1.0 | 1.0 | autumn | 2022-11-25 | 0.8866398681715368 | 1.0096910579900373 | 26 | 0 | no |
| path_line50_cap11 | 0.5 | 1.0 | 1.0 | winter | 2022-12-28 | 0.8947175311789992 | 1.0078154959953607 | 33 | 0 | no |
| path_line50_cap11 | 0.5 | 1.0 | 1.0 | spring_festival_2022 | 2022-02-02 | 0.9737901265780678 | 1.0246311171199027 | 0 | 0 | yes |
| path_line50_cap11 | 0.5 | 1.0 | 1.0 | national_day_2022 | 2022-10-07 | 0.9139040042233154 | 1.0090858738910267 | 4 | 0 | no |
| path_line50_cap12 | 0.5 | 1.0 | 2.0 | spring | 2022-05-23 | 0.9848543101737955 | 1.0429180204371018 | 0 | 0 | yes |
| path_line50_cap12 | 0.5 | 1.0 | 2.0 | summer | 2022-08-27 | 0.9170545998973993 | 1.0116842932072339 | 1 | 0 | no |
| path_line50_cap12 | 0.5 | 1.0 | 2.0 | autumn | 2022-11-25 | 0.8975867647047329 | 1.0163365858714088 | 14 | 0 | no |
| path_line50_cap12 | 0.5 | 1.0 | 2.0 | winter | 2022-12-28 | 0.90963550875434 | 1.012268278428671 | 18 | 0 | no |
| path_line50_cap12 | 0.5 | 1.0 | 2.0 | spring_festival_2022 | 2022-02-02 | 0.978229573281485 | 1.041828304449281 | 0 | 0 | yes |
| path_line50_cap12 | 0.5 | 1.0 | 2.0 | national_day_2022 | 2022-10-07 | 0.9240455283320486 | 1.0153066456224202 | 1 | 0 | no |
| path_line70_cap11 | 0.7 | 1.0 | 1.0 | spring | 2022-05-23 | 0.9669616044794244 | 1.035951633972591 | 0 | 0 | yes |
| path_line70_cap11 | 0.7 | 1.0 | 1.0 | summer | 2022-08-27 | 0.8613726705411159 | 1.0097679738333938 | 32 | 0 | no |
| path_line70_cap11 | 0.7 | 1.0 | 1.0 | autumn | 2022-11-25 | 0.8284197766704461 | 1.0132665499144307 | 61 | 0 | no |
| path_line70_cap11 | 0.7 | 1.0 | 1.0 | winter | 2022-12-28 | 0.8417451340045002 | 1.0105640762192263 | 43 | 0 | no |
| path_line70_cap11 | 0.7 | 1.0 | 1.0 | spring_festival_2022 | 2022-02-02 | 0.9618275235982596 | 1.0343599892970556 | 0 | 0 | yes |
| path_line70_cap11 | 0.7 | 1.0 | 1.0 | national_day_2022 | 2022-10-07 | 0.8724079638535063 | 1.0123888226828224 | 32 | 0 | no |
| path_line70_cap12 | 0.7 | 1.0 | 2.0 | spring | 2022-05-23 | 0.9779284746563733 | 1.0602862321594175 | 12 | 0 | no |
| path_line70_cap12 | 0.7 | 1.0 | 2.0 | summer | 2022-08-27 | 0.8757214638958447 | 1.0159665082630498 | 26 | 0 | no |
| path_line70_cap12 | 0.7 | 1.0 | 2.0 | autumn | 2022-11-25 | 0.8440144507423565 | 1.0227253722417733 | 33 | 0 | no |
| path_line70_cap12 | 0.7 | 1.0 | 2.0 | winter | 2022-12-28 | 0.8627319518951051 | 1.0168989073580834 | 35 | 0 | no |
| path_line70_cap12 | 0.7 | 1.0 | 2.0 | spring_festival_2022 | 2022-02-02 | 0.9680986034337086 | 1.0586276846229432 | 12 | 0 | no |
| path_line70_cap12 | 0.7 | 1.0 | 2.0 | national_day_2022 | 2022-10-07 | 0.8867051406694149 | 1.021188855363995 | 29 | 0 | no |
| path_line60_cap12 | 0.6 | 1.0 | 2.0 | spring | 2022-05-23 | 0.981458355868964 | 1.0515909988572976 | 2 | 0 | no |
| path_line60_cap12 | 0.6 | 1.0 | 2.0 | summer | 2022-08-27 | 0.8971505081287817 | 1.01385740915657 | 7 | 0 | no |
| path_line60_cap12 | 0.6 | 1.0 | 2.0 | autumn | 2022-11-25 | 0.8720723537318396 | 1.0195445869209485 | 26 | 0 | no |
| path_line60_cap12 | 0.6 | 1.0 | 2.0 | winter | 2022-12-28 | 0.8872150155327977 | 1.0145916675254383 | 33 | 0 | no |
| path_line60_cap12 | 0.6 | 1.0 | 2.0 | spring_festival_2022 | 2022-02-02 | 0.9732765141623683 | 1.0502261031222304 | 2 | 0 | no |
| path_line60_cap12 | 0.6 | 1.0 | 2.0 | national_day_2022 | 2022-10-07 | 0.9060101722880012 | 1.0182686283948532 | 7 | 0 | no |

## Conclusion

No targeted setting passes every group. Targeted line R/X scaling on the two weak paths plus fixed capacitors at buses 17 and 32 is insufficient; the remaining low voltage is driven by the branch's active-power drop and native load. The uniform `line50_q2.0` setting remains the lightest passing option screened. A voltage regulator/OLTC or a stronger reinforcement is required for a lighter design.

## Limits

- One worst representative day per group, not all 31 days.
- Fixed capacitors are static and unrated; no device cost or control dynamics are modeled.
- No load is scaled; no market clearing, storage, shedding or price is modeled.

## Artifacts

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_subsets.csv` | `3ea3b009289194aa3b5d8e7b0afdbb148d0da5086423194cfbd1ad6aaec014ca` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_candidates.csv` | `05b63da48f2585044635af745a0c18e76565001a78fead03b122cd285891a369` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_summary.csv` | `8464c14a92e1d0ff777a9b0e7b5ef5b6c2bc962cdbe1b9c40c9f6ae52d0b92de` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_day_curves.npz` | `8ff8cbc00618189d001565068bf8a95c9b9975960baa0ce6343ad508075be472` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_physical_screen.csv` | `0717996a44cea56e3976fca700d4cf375e1879d017bc108f8c9dba9b29d4fbce` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_targeted_support_screen.csv` | `cb11b6e7532dc0cf3f3271d50a5d23a30b0a8ad64f0f97500e0842442db04351` |

## Command

`src/screen_v2_targeted_support.py`
