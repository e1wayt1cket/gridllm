# Industry Load Quality Report

Produced by `src/industry_load.py`. Records what was measured; no value in the source workbook was modified, imputed or clamped.

## Source

| | |
|---|---|
| workbook | `C:\Projects\00_WorkSpace\01_Projects\MakerB\docs\16类行业用户.xlsx` |
| workbook sha256 | `63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb` |
| sheet | `Sheet3` |
| slots per row | 96 x 15 min (00:00..23:45) |
| data rows read | 117,271 |
| industries present | 16 |

## Field mapping

| column | index | role |
|---|---|---|
| `序号` | 1 | - |
| `总户号` | 2 | account identity |
| `地点` | 3 | - |
| `用户名称` | 4 | user label (never published) |
| `行业` | 5 | industry class |
| `合同容量` | 6 | - |
| `运行容量` | 7 | declared rating, for consistency checks |
| `户号` | 8 | - |
| `电表局编号` | 9 | meter identity |
| `综合倍率` | 10 | meter multiplier, **not trusted** |
| `功率相序` | 11 | power phase |
| `数据日期` | 12 | day key |
| slot columns | 13..108 | 96 x 15-minute readings |

## Cleaning rules

`drop` means the record is kept out of the shape pool; `note` means it is counted in the report and the record is still usable as a shape. No rule alters a reading.

| flag | effect | rule |
|---|---|---|
| `empty_identity` | drop | account, meter or date is blank |
| `missing_slots` | drop | at least one of the 96 slots is blank |
| `all_zero` | drop | every present slot is exactly 0 |
| `placeholder` | drop | opens `1,2,3,3,3`, remaining slots all 0, total exactly 12 |
| `other_phase` | drop | 功率相序 is not 有功功率 |
| `negative_load` | note | negative reading on a non-分布式光伏 industry |
| `negative_pv` | note | negative reading on 分布式光伏 (reverse flow expected) |
| `ratio_missing` | note | 综合倍率 blank or not a positive number |
| `ratio_suspect` | note | 综合倍率 departs from its industry median by more than 100x |
| `capacity_inconsistent` | note | implied peak exceeds 1.5x 运行容量 |
| `duplicate_row` | drop | same (account, meter, date) repeated with an identical profile; dropped so no observation is counted twice |
| `duplicate_conflict` | note | same (account, meter, date) repeated with a **different** profile; both are kept, because choosing one would invent a reading |

## Flags observed

| flag | records |
|---|---|
| `empty_identity` | 81 |
| `missing_slots` | 4,609 |
| `all_zero` | 15,482 |
| `placeholder` | 43 |
| `other_phase` | 81 |
| `negative_load` | 2,683 |
| `negative_pv` | 7,856 |
| `ratio_missing` | 54 |
| `ratio_suspect` | 9,827 |
| `capacity_inconsistent` | 16,653 |
| `duplicate_row` | 13,528 |
| `duplicate_conflict` | 4 |
| **total records** | **117,271** |

Records kept out of the shape pool: **29,760** (25.38% of rows). Advisory flags do not remove a record; they are counted here and the record still contributes its shape.

## Dropped records by industry

| industry | `empty_identity` | `missing_slots` | `all_zero` | `placeholder` | `other_phase` | `duplicate_row` |
|---|---|---|---|---|---|---|
| 5G基站 | 0 | 1560 | 37 | 0 | 0 | 0 |
| 分布式光伏 | 0 | 251 | 1473 | 0 | 0 | 730 |
| 化工业 | 10 | 397 | 526 | 6 | 10 | 1208 |
| 印刷业 | 4 | 69 | 208 | 8 | 4 | 1 |
| 商业综合体 | 2 | 999 | 1510 | 0 | 2 | 7203 |
| 数据中心 | 2 | 144 | 278 | 0 | 2 | 0 |
| 机械 | 5 | 99 | 31 | 1 | 5 | 385 |
| 水泥制造 | 13 | 208 | 4085 | 5 | 13 | 2400 |
| 汽车制造业 | 7 | 163 | 1391 | 2 | 7 | 2 |
| 电制氢 | 3 | 40 | 286 | 2 | 3 | 0 |
| 电动汽车（充换电） | 7 | 325 | 1029 | 0 | 7 | 161 |
| 电子设备制造业 | 7 | 38 | 731 | 2 | 7 | 0 |
| 电解铝 | 1 | 12 | 0 | 1 | 1 | 0 |
| 纺织业 | 0 | 78 | 21 | 6 | 0 | 0 |
| 通用设备制造业 | 4 | 79 | 1347 | 0 | 4 | 0 |
| 钢铁 | 16 | 147 | 2529 | 10 | 16 | 1438 |

## Ratio and capacity evidence

`综合倍率` is reported, never corrected. A ratio that disagrees with 运行容量 is as likely to mean the rating is nominal as the ratio is wrong, so the disagreement is recorded as a flag.

| | |
|---|---|
| meters checked | 200 |
| meters whose implied peak fits 运行容量 (<= 1.5x) | 87 |
| meters whose implied peak contradicts it | 113 |

Worst contradictions (implied peak / 运行容量):

| meter | industry | ratio | 运行容量 | implied peak kW | ratio to rating |
|---|---|---|---|---|---|
| `M59e598348f4e` | 汽车制造业 | 440,000 | 40,000 | 183,744,000 | 4,593.6x |
| `M1ae3bfbd4249` | 化工业 | 1,320,000 | 240,000 | 542,572,800 | 2,260.7x |
| `M6de3741455e0` | 化工业 | 1,320,000 | 240,000 | 540,196,800 | 2,250.8x |
| `M2c9f3e01a134` | 化工业 | 1,320,000 | 240,000 | 535,920,000 | 2,233.0x |
| `M0f6fdf82497c` | 化工业 | 1,320,000 | 240,000 | 519,182,400 | 2,163.3x |
| `M181c47a33d96` | 化工业 | 2,200,000 | 720,000 | 983,576,000 | 1,366.1x |
| `Mba4eeb12ecb5` | 化工业 | 2,200,000 | 720,000 | 962,016,000 | 1,336.1x |
| `M51f036bd42a3` | 电动汽车（充换电） | 3,000 | 2,500 | 3,264,000 | 1,305.6x |
| `M1fa0bb3d36e3` | 钢铁 | 264,000 | 103,310 | 110,858,880 | 1,073.1x |
| `Mb97972a568cf` | 水泥制造 | 132,000 | 83,580 | 69,648,480 | 833.3x |
| `M505a4765c6ca` | 纺织业 | 10,500 | 6,450 | 4,562,460 | 707.4x |
| `Mcaf9bf10a14c` | 电子设备制造业 | 44,000 | 27,500 | 18,712,320 | 680.4x |
| `Mf32e7531faea` | 电动汽车（充换电） | 2,000 | 2,500 | 1,545,680 | 618.3x |
| `M136ca9f032de` | 纺织业 | 14,000 | 9,700 | 5,957,280 | 614.2x |
| `M30a738d6d057` | 水泥制造 | 132,000 | 83,580 | 49,610,880 | 593.6x |

## Qualified subset

The formal absolute-load input is the intersection of two gates: the shape pool above, and the per-meter ratio/capacity check. The check runs on the **meter**; the pool runs on the **block**, one meter on one date. Both counts are stated because they are different numbers.

| | |
|---|---|
| meters, all | 283 |
| meters the check cannot reach | 83 |
| **meters qualified** | **87** |
| blocks surviving the shape pool | 87,511 |
| **blocks in the qualified subset** | **39,365** |
| meters contributing those blocks | 84 |
| users covered | 36 |
| dates covered | 730 |

A meter the check cannot reach is not qualified: missing information is not a pass, though such a meter is not thereby accused of being wrong, and its records are untouched. A block is never admitted on its own while its meter is out, so the meter gate cannot decay into advice. The ruling and the unit of each figure are recorded in `docs/2026-10-14-data-and-experiment-decisions.md`.

## Aggregation

Method: hierarchical equal-weighted mean (`hierarchical_mean`).

1. each qualified meter-day is scaled by its own `max|x|`, so the curve carries intraday shape and no magnitude;
2. a user-day is the mean of that user's qualified meters **for that date**, so a day with many meters still counts once;
3. a user curve is the mean of that user's qualified days;
4. an industry curve is the mean of its users, then rescaled to `max = 1`.

Because step 1 divides by a per-meter scale, `综合倍率` cancels exactly and cannot distort any published shape — including the meters whose ratio is not credible.

## Industry sample statistics

| industry | users | meters | meter-days | observations | peak slot | reliable | note |
|---|---|---|---|---|---|---|---|
| 5G基站 | 2 | 2 | 626 | 626 | 55 | **no** | limited: users 2 < 3 |
| 分布式光伏 | 4 | 15 | 2407 | 8498 | 50 | yes | ok |
| 化工业 | 6 | 20 | 3514 | 7889 | 33 | yes | ok |
| 印刷业 | 5 | 13 | 1453 | 2806 | 74 | yes | ok |
| 商业综合体 | 5 | 37 | 3083 | 22151 | 77 | yes | ok |
| 数据中心 | 5 | 20 | 2561 | 9435 | 66 | yes | ok |
| 机械 | 5 | 7 | 1554 | 1943 | 02 | yes | ok |
| 水泥制造 | 5 | 13 | 3079 | 4021 | 14 | yes | ok |
| 汽车制造业 | 5 | 15 | 2576 | 5762 | 65 | yes | ok |
| 电制氢 | 3 | 7 | 1575 | 2244 | 47 | yes | ok |
| 电动汽车（充换电） | 5 | 10 | 3016 | 4668 | 92 | yes | ok |
| 电子设备制造业 | 5 | 7 | 2162 | 2596 | 41 | yes | ok |
| 电解铝 | 1 | 1 | 208 | 208 | 41 | **no** | limited: users 1 < 3 |
| 纺织业 | 5 | 15 | 3598 | 6792 | 39 | yes | ok |
| 通用设备制造业 | 4 | 9 | 2902 | 4403 | 65 | yes | ok |
| 钢铁 | 3 | 17 | 1664 | 3469 | 15 | yes | ok |

Reliability threshold: at least 3 users and 30 meter-days.

## Reliability limits

- **5G基站** — limited: users 2 < 3. Treat its curve as indicative only.
- **电解铝** — limited: users 1 < 3. Treat its curve as indicative only.

- `分布式光伏` is a **generator**, not a load. Its shape is produced outside the load pool and must not be fed to a load bus; the network already carries its own PV devices.
- These are normalised shapes. Nothing here states an absolute MW level; see `absolute_mw`, which refuses until a unit and ratio verification exists.

## Reproduction

```bash
PYTHONPATH=src python src/industry_load.py \
    --workbook "$(pwd)/docs/16类行业用户.xlsx" \
    --out data/processed \
    --report reports/industry_load_quality_report.md
```
