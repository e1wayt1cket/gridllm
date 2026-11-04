# V2 user-date coverage audit

## Verdict

- Mapped users audited: **41**
- Mapped buses: **17**
- Calendar dates: **730**
- V1 subset users: **17** (kept as a historical benchmark, not used as the V2 selection rule)
- Provisional fixed core: **15 users** with all-meters data on **149 dates**, at least 20 dates in every season

The V2 completeness standard is **all mapped meters complete**, not one meter per user. Partial user-days are counted and reported but are not eligible for the fixed scenario core. Nothing was scaled, interpolated, resampled, copied or filled.

## Inputs

- Mapping: `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\industry_meter_bus_map_v2.csv` (`a5fae0850b2bd6a4d0f9db7412e71b1aac2c6771a867f11cb1f4c426386eb38b`)
- Load workbook: `C:\Projects\00_WorkSpace\01_Projects\MakerB\docs\16类行业用户.xlsx` (`63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb`)
- Records read: 117271
- Unmapped-user records ignored: 36632
- Unmapped-meter records ignored: 18504
- Records with no usable date: 38

## Seasonal coverage

| season | dates | dates with ≥1 all-meters user | dates with all 41 all-meters | dates with all 37 | dates with core complete | max all-meters users | users with ≥1 all-meters date | buses with ≥1 all-meters date | max user-level complete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spring | 184 | 184 | 0 | 0 | 25 | 23 | 28 | 14 | 30 |
| summer | 184 | 184 | 0 | 0 | 33 | 28 | 31 | 15 | 33 |
| autumn | 182 | 182 | 0 | 0 | 69 | 31 | 35 | 16 | 36 |
| winter | 180 | 180 | 0 | 0 | 22 | 31 | 39 | 17 | 37 |

## Holiday coverage

| holiday | date | all-meters users | core complete |
|---|---|---:|---|
| New Year (offset 0) | 2021-01-01 | 12 | no |
| New Year (offset 1) | 2021-01-02 | 12 | no |
| New Year (offset 2) | 2021-01-03 | 12 | no |
| Spring Festival make-up workday (offset None) | 2021-02-07 | 12 | no |
| Spring Festival (offset -1) | 2021-02-11 | 13 | no |
| Spring Festival (offset 0) | 2021-02-12 | 13 | no |
| Spring Festival (offset 1) | 2021-02-13 | 13 | no |
| Spring Festival (offset 2) | 2021-02-14 | 13 | no |
| Spring Festival (offset 3) | 2021-02-15 | 13 | no |
| Spring Festival (offset 4) | 2021-02-16 | 13 | no |
| Spring Festival (offset 5) | 2021-02-17 | 12 | no |
| Spring Festival make-up workday (offset None) | 2021-02-20 | 13 | no |
| Qingming (offset -1) | 2021-04-03 | 12 | no |
| Qingming (offset 0) | 2021-04-04 | 11 | no |
| Qingming (offset 1) | 2021-04-05 | 11 | no |
| Labour Day make-up workday (offset None) | 2021-04-25 | 12 | no |
| Labour Day (offset 0) | 2021-05-01 | 12 | no |
| Labour Day (offset 1) | 2021-05-02 | 12 | no |
| Labour Day (offset 2) | 2021-05-03 | 12 | no |
| Labour Day (offset 3) | 2021-05-04 | 10 | no |
| Labour Day (offset 4) | 2021-05-05 | 12 | no |
| Labour Day make-up workday (offset None) | 2021-05-08 | 13 | no |
| Dragon Boat Festival (offset -2) | 2021-06-12 | 12 | no |
| Dragon Boat Festival (offset -1) | 2021-06-13 | 12 | no |
| Dragon Boat Festival (offset 0) | 2021-06-14 | 12 | no |
| Mid-Autumn make-up workday (offset None) | 2021-09-18 | 14 | no |
| Mid-Autumn Festival (offset -2) | 2021-09-19 | 14 | no |
| Mid-Autumn Festival (offset -1) | 2021-09-20 | 14 | no |
| Mid-Autumn Festival (offset 0) | 2021-09-21 | 14 | no |
| National Day make-up workday (offset None) | 2021-09-26 | 14 | no |
| National Day (offset 0) | 2021-10-01 | 14 | no |
| National Day (offset 1) | 2021-10-02 | 13 | no |
| National Day (offset 2) | 2021-10-03 | 12 | no |
| National Day (offset 3) | 2021-10-04 | 12 | no |
| National Day (offset 4) | 2021-10-05 | 12 | no |
| National Day (offset 5) | 2021-10-06 | 13 | no |
| National Day (offset 6) | 2021-10-07 | 13 | no |
| National Day make-up workday (offset None) | 2021-10-09 | 11 | no |
| New Year (offset 0) | 2022-01-01 | 14 | no |
| New Year (offset 1) | 2022-01-02 | 12 | no |
| New Year (offset 2) | 2022-01-03 | 15 | yes |
| Spring Festival make-up workday (offset None) | 2022-01-29 | 17 | yes |
| Spring Festival make-up workday (offset None) | 2022-01-30 | 16 | no |
| Spring Festival (offset -1) | 2022-01-31 | 15 | no |
| Spring Festival (offset 0) | 2022-02-01 | 16 | no |
| Spring Festival (offset 1) | 2022-02-02 | 16 | yes |
| Spring Festival (offset 2) | 2022-02-03 | 16 | yes |
| Spring Festival (offset 3) | 2022-02-04 | 15 | no |
| Spring Festival (offset 4) | 2022-02-05 | 16 | yes |
| Spring Festival (offset 5) | 2022-02-06 | 16 | yes |
| Qingming make-up workday (offset None) | 2022-04-02 | 16 | yes |
| Qingming (offset -1) | 2022-04-03 | 16 | yes |
| Qingming (offset 0) | 2022-04-04 | 16 | yes |
| Qingming (offset 1) | 2022-04-05 | 15 | no |
| Labour Day make-up workday (offset None) | 2022-04-24 | 15 | no |
| Labour Day (offset -1) | 2022-04-30 | 17 | no |
| Labour Day (offset 0) | 2022-05-01 | 16 | no |
| Labour Day (offset 1) | 2022-05-02 | 17 | no |
| Labour Day (offset 2) | 2022-05-03 | 18 | no |
| Labour Day (offset 3) | 2022-05-04 | 16 | no |
| Labour Day make-up workday (offset None) | 2022-05-07 | 15 | no |
| Dragon Boat Festival (offset 0) | 2022-06-03 | 23 | no |
| Dragon Boat Festival (offset 1) | 2022-06-04 | 22 | no |
| Dragon Boat Festival (offset 2) | 2022-06-05 | 22 | no |
| Mid-Autumn Festival (offset 0) | 2022-09-10 | 27 | yes |
| Mid-Autumn Festival (offset 1) | 2022-09-11 | 28 | yes |
| Mid-Autumn Festival (offset 2) | 2022-09-12 | 28 | yes |
| National Day (offset 0) | 2022-10-01 | 28 | no |
| National Day (offset 1) | 2022-10-02 | 28 | no |
| National Day (offset 2) | 2022-10-03 | 27 | no |
| National Day (offset 3) | 2022-10-04 | 29 | yes |
| National Day (offset 4) | 2022-10-05 | 29 | yes |
| National Day (offset 5) | 2022-10-06 | 29 | yes |
| National Day (offset 6) | 2022-10-07 | 29 | yes |
| National Day make-up workday (offset None) | 2022-10-08 | 29 | yes |
| National Day make-up workday (offset None) | 2022-10-09 | 29 | yes |

## User summary

| user | bus | industry | V1 | complete days | all-meters days | partial days |
|---|---:|---|---:|---:|---:|---:|
| Ue9e499c141dd | 25 | 通用设备制造业 | yes | 725 | 725 | 0 |
| U579fe8c94614 | 25 | 通用设备制造业 | no | 724 | 724 | 0 |
| U6e0c9c380ad3 | 27 | 纺织业 | yes | 724 | 724 | 0 |
| U6e5b7a2d2097 | 30 | 电制氢 | yes | 724 | 724 | 0 |
| U5765bec968a1 | 26 | 电子设备制造业 | yes | 723 | 723 | 0 |
| Ufe8fb8bfad44 | 5 | 商业综合体 | no | 730 | 718 | 12 |
| Ue7a335c557ca | 4 | 商业综合体 | no | 730 | 717 | 13 |
| U018990a98eb1 | 25 | 通用设备制造业 | yes | 711 | 711 | 0 |
| Uf67c750d6c55 | 27 | 纺织业 | yes | 704 | 704 | 0 |
| Ue5f08866ae4e | 31 | 电动汽车（充换电） | no | 712 | 691 | 21 |
| Ub1da4ce43b04 | 24 | 汽车制造业 | yes | 728 | 690 | 38 |
| U656f86497702 | 3 | 商业综合体 | yes | 730 | 652 | 78 |
| U481b69555b99 | 23 | 化工业 | no | 728 | 443 | 285 |
| Ue971298ed105 | 23 | 化工业 | yes | 390 | 390 | 0 |
| U45f85cc7e9db | 2 | 商业综合体 | no | 729 | 315 | 414 |
| U39e8a17aba10 | 23 | 化工业 | yes | 730 | 255 | 475 |
| U1f2d028ac65f | 26 | 电子设备制造业 | yes | 253 | 253 | 0 |
| U9b80870a7722 | 28 | 机械 | no | 643 | 234 | 409 |
| U1f2af72e4165 | 27 | 纺织业 | no | 719 | 232 | 487 |
| Uac6f64bb506e | 24 | 汽车制造业 | no | 226 | 226 | 0 |
| U074a148fa4c7 | 26 | 电子设备制造业 | no | 721 | 221 | 500 |
| Ucf65d248ce93 | 27 | 纺织业 | yes | 714 | 217 | 497 |
| U73fcc2041874 | 27 | 纺织业 | yes | 717 | 214 | 503 |
| U2aa2cf305ab6 | 30 | 电制氢 | yes | 721 | 213 | 508 |
| U134c5ea1227f | 24 | 汽车制造业 | no | 174 | 174 | 0 |
| U1f95b8d57d1c | 1 | 商业综合体 | no | 164 | 164 | 0 |
| Ud7aaf2cc7bd7 | 28 | 机械 | no | 162 | 162 | 0 |
| Udfed49bf706b | 31 | 电动汽车（充换电） | no | 157 | 157 | 0 |
| U3a097bfc3f9e | 25 | 通用设备制造业 | yes | 113 | 113 | 0 |
| U0abfaa684d80 | 16 | 钢铁 | yes | 718 | 95 | 623 |
| Ufbfdfb7de5b9 | 29 | 印刷业 | no | 82 | 82 | 0 |
| U22609eff325a | 22 | 水泥制造 | no | 70 | 70 | 0 |
| U12592b9d53aa | 32 | 数据中心 | no | 32 | 32 | 0 |
| U6ae415563f9f | 32 | 数据中心 | no | 23 | 23 | 0 |
| U4ea136e58032 | 24 | 汽车制造业 | no | 6 | 6 | 0 |
| U046530842875 | 16 | 钢铁 | no | 725 | 4 | 721 |
| Ub4e7e479aa6f | 29 | 印刷业 | no | 25 | 3 | 22 |
| U11128e1b6964 | 22 | 水泥制造 | no | 2 | 2 | 0 |
| U2de6c0028077 | 23 | 化工业 | no | 730 | 1 | 729 |
| Uc2a51b8e47ff | 22 | 水泥制造 | no | 715 | 1 | 714 |
| U46342f0b3576 | 31 | 电动汽车（充换电） | yes | 723 | 0 | 723 |

## Provisional fixed core

Rule: rank users by all-meters complete days, then take the largest prefix that keeps at least 20 dates in each season. This is a coverage-ranked candidate for Phase C, not a frozen V2 subset.

- Users: **15**
- Buses: **[2, 3, 4, 5, 23, 24, 25, 26, 27, 30, 31]** (11 of 17)
- Industries: **8**
- Dates: **149** ({'spring': 25, 'summer': 33, 'autumn': 69, 'winter': 22})
- Holiday dates: {'national_day': 6, 'mid_autumn': 3, 'spring_festival': 5, 'qingming': 3, 'new_year': 1}

| user | bus | industry | V1 | all-meters days |
|---|---:|---|---:|---:|
| Ue9e499c141dd | 25 | 通用设备制造业 | yes | 725 |
| U579fe8c94614 | 25 | 通用设备制造业 | no | 724 |
| U6e0c9c380ad3 | 27 | 纺织业 | yes | 724 |
| U6e5b7a2d2097 | 30 | 电制氢 | yes | 724 |
| U5765bec968a1 | 26 | 电子设备制造业 | yes | 723 |
| Ufe8fb8bfad44 | 5 | 商业综合体 | no | 718 |
| Ue7a335c557ca | 4 | 商业综合体 | no | 717 |
| U018990a98eb1 | 25 | 通用设备制造业 | yes | 711 |
| Uf67c750d6c55 | 27 | 纺织业 | yes | 704 |
| Ue5f08866ae4e | 31 | 电动汽车（充换电） | no | 691 |
| Ub1da4ce43b04 | 24 | 汽车制造业 | yes | 690 |
| U656f86497702 | 3 | 商业综合体 | yes | 652 |
| U481b69555b99 | 23 | 化工业 | no | 443 |
| Ue971298ed105 | 23 | 化工业 | yes | 390 |
| U45f85cc7e9db | 2 | 商业综合体 | no | 315 |

## Coverage ladder

| prefix k | dates all k complete | season counts |
|---:|---:|---|
| 1 | 725 | {'spring': 182, 'summer': 182, 'autumn': 181, 'winter': 180} |
| 2 | 719 | {'spring': 182, 'summer': 180, 'autumn': 181, 'winter': 176} |
| 3 | 713 | {'spring': 180, 'summer': 178, 'autumn': 179, 'winter': 176} |
| 4 | 707 | {'spring': 180, 'summer': 175, 'autumn': 179, 'winter': 173} |
| 5 | 700 | {'spring': 176, 'summer': 172, 'autumn': 179, 'winter': 173} |
| 6 | 690 | {'spring': 175, 'summer': 168, 'autumn': 177, 'winter': 170} |
| 7 | 677 | {'spring': 169, 'summer': 166, 'autumn': 175, 'winter': 167} |
| 8 | 661 | {'spring': 165, 'summer': 164, 'autumn': 171, 'winter': 161} |
| 9 | 637 | {'spring': 158, 'summer': 149, 'autumn': 170, 'winter': 160} |
| 10 | 605 | {'spring': 139, 'summer': 148, 'autumn': 167, 'winter': 151} |
| 11 | 577 | {'spring': 128, 'summer': 147, 'autumn': 159, 'winter': 143} |
| 12 | 512 | {'spring': 117, 'summer': 142, 'autumn': 143, 'winter': 110} |
| 13 | 310 | {'spring': 46, 'summer': 70, 'autumn': 138, 'winter': 56} |
| 14 | 203 | {'spring': 40, 'summer': 33, 'autumn': 78, 'winter': 52} |
| 15 | 149 | {'spring': 25, 'summer': 33, 'autumn': 69, 'winter': 22} |
| 16 | 102 | {'spring': 2, 'summer': 33, 'autumn': 67, 'winter': 0} |

## Data gaps

- No date carries all 41 mapped users at the all-meters standard; the maximum is 37.
- A user-day with one meter missing is partial and is not admitted to the fixed core. Its load fields are computed only from the meters that converted and are marked partial.
- Users with very few all-meters days cannot be part of a multi-season fixed subset; they remain in the audit but not in the provisional core.
- The provisional core is a coverage choice, not a load-scale or physical-feasibility choice. Phase C must still check the network.

## Artifacts

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\v2_user_date_coverage.csv` | `5ffae19bd152bf21553b2ae49c7fef09b997a2901dd8b6963a37857472082964` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\reports\v2_user_date_completeness_matrix.csv` | `6d791ad0262647a9c5f445e7441e0704d81712b85d36adf73295fc7908e1a1d0` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_date_summary.csv` | `c62dc7a5ca9f090582894c05852a727f5a51920cf41a9f4c84d9a02b896c3acc` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_summary.csv` | `8464c14a92e1d0ff777a9b0e7b5ef5b6c2bc962cdbe1b9c40c9f6ae52d0b92de` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_provisional_core.json` | `45653b8b1aedda3cdd84a8f9bc457018e1eb6e91df23e17ceb06a02f13004ad8` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_day_curves.npz` | `8ff8cbc00618189d001565068bf8a95c9b9975960baa0ce6343ad508075be472` |

## Command

`src/audit_v2_user_date_coverage.py`
