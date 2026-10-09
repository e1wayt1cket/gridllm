# Phase 1 — Industry Load Preprocessing

Delivers `src/industry_load.py`: normalised 96-slot daily shapes for the 16
industries in `docs/16类行业用户.xlsx`, plus a quality report and the
exclusions record. No absolute MW output is produced — see section 7.

Nothing else was touched. The price module, the Jiangsu market notes, the SOCP
clearing and the MATD3 stack are all unchanged.

## 1. Files

**Added**

| file | what |
|---|---|
| `src/industry_load.py` | the module |
| `tests/test_industry_load.py` | 37 tests (36 fast, 1 slow) |
| `reports/industry_load_quality_report.md` | generated quality report |
| `reports/phase1_industry_load_report.md` | this report |
| `data/processed/industry_typical_profiles.csv` | 16 x 96 shape table |
| `data/processed/industry_profile_summary.csv` | per-industry sample statistics |
| `data/processed/industry_exclusions.csv` | every flagged record, traceable |
| `data/processed/industry_daily_profiles.parquet` | 87,511 qualified meter-day curves |
| `data/processed/industry_shapes.png` | 16-panel comparison figure |

**Modified:** none. No existing module, config or test was edited.

## 2. What was read

| | |
|---|---|
| workbook | `docs/16类行业用户.xlsx` |
| sheet | `Sheet3`, located by matching the expected 12-column header and a 96-column time block, not by name |
| data rows | 117,271 |
| slot columns | 96, verified as `00:00..23:45` and strictly ascending by 15 minutes |
| industries present | 16 |
| records kept out of the shape pool | 29,760 (25.38 %) |
| qualified meter-days written | 87,511 |

The workbook is opened `read_only=True, data_only=True` and is never written to.

## 3. Industry statistics

`users` = distinct accounts contributing at least one qualified observation.
`days` = distinct (user, date) pairs. Reliability requires >= 3 users and
>= 30 meter-days.

| industry | users | meters | days | observations | reliable |
|---|---|---|---|---|---|
| 5G基站 | 2 | 2 | 626 | 626 | **no** |
| 分布式光伏 | 4 | 15 | 2,407 | 8,498 | yes |
| 化工业 | 6 | 20 | 3,514 | 7,889 | yes |
| 印刷业 | 5 | 13 | 1,453 | 2,806 | yes |
| 商业综合体 | 5 | 37 | 3,083 | 22,151 | yes |
| 数据中心 | 5 | 20 | 2,561 | 9,435 | yes |
| 机械 | 5 | 7 | 1,554 | 1,943 | yes |
| 水泥制造 | 5 | 13 | 3,079 | 4,021 | yes |
| 汽车制造业 | 5 | 15 | 2,576 | 5,762 | yes |
| 电制氢 | 3 | 7 | 1,575 | 2,244 | yes |
| 电动汽车（充换电） | 5 | 10 | 3,016 | 4,668 | yes |
| 电子设备制造业 | 5 | 7 | 2,162 | 2,596 | yes |
| 电解铝 | 1 | 1 | 208 | 208 | **no** |
| 纺织业 | 5 | 15 | 3,598 | 6,792 | yes |
| 通用设备制造业 | 4 | 9 | 2,902 | 4,403 | yes |
| 钢铁 | 3 | 17 | 1,664 | 3,469 | yes |

14 of 16 industries clear the threshold. 5G基站 and 电解铝 do not, and their
curves are published marked `limited` rather than withheld.

## 4. Records flagged and dropped

| flag | records | effect |
|---|---|---|
| `empty_identity` | 81 | drop |
| `missing_slots` | 4,609 | drop |
| `all_zero` | 15,482 | drop |
| `placeholder` | 43 | drop |
| `other_phase` | 81 | drop |
| `duplicate_row` | 13,528 | drop |
| `negative_load` | 2,683 | note |
| `negative_pv` | 7,856 | note |
| `ratio_missing` | 54 | note |
| `ratio_suspect` | 9,827 | note |
| `capacity_inconsistent` | 16,653 | note |
| `duplicate_conflict` | 4 | note |

Drop reasons, in order of size: duplicates (13,528), all-zero days (15,482),
incomplete days (4,609), empty identity (81), wrong phase (81), placeholders
(43).

Two classes deserve naming because they are easy to get wrong:

- **`placeholder`** — the synthesised `[1,2,3,3,3,0,...]` row is matched on
  shape and exact total (12) together. Two lookalike tests confirm that a real
  reading opening `1,2,3,3,3` but totalling something else is *not* swept up.
- **`duplicate_conflict`** — 12,974 (account, meter, date) keys repeat. 12,970
  repeats are byte-identical and are dropped. **4 disagree about the profile.**
  Both readings are kept, because choosing one would invent a measurement.

No record had its values altered by any rule.

## 5. Shapes and figures

Each industry curve is exactly 96 points, finite, normalised to `max = 1`, and
non-negative. Aggregation is `hierarchical_mean`: meter-day (each scaled by its
own `max|x|`) -> user-day (mean over that user's meters for that date) ->
user (mean over days) -> industry (mean over users, then rescaled).

Equal weighting at every level is the point of the design and is pinned by
tests: adding nine copies of a user's meter, or nineteen copies of a user's day,
leaves the industry curve unchanged.

Spot checks against process expectation:

| industry | shape | verdict |
|---|---|---|
| 数据中心 | flat, 0.95–1.00 all day | correct for 24/7 load |
| 钢铁 | flat, 0.92–1.00 | correct for continuous process |
| 分布式光伏 | rises 06:00, peaks 12:00–13:00, falls to night floor | correct generation profile |
| 商业综合体 | trough 0.20 at 05:00, peak 1.00 at 19:00 | plausible retail profile |

`industry_shapes.png` plots all 16; limited industries are drawn in red.

## 6. Tests

**New:** `tests/test_industry_load.py` — 36 fast + 1 slow.

```
python -m pytest tests/test_industry_load.py -v          -> 36 passed, 1 deselected
python -m pytest tests/test_industry_load.py -v -m slow  -> 1 passed, 36 deselected (80.8s)
```

Coverage maps to the required list: slot-axis identification and ordering
(including a rejected 30-minute grid); complete / incomplete / all-zero day
classification; placeholder detection plus two lookalike negatives; negatives
and implausible ratios flagged with values asserted byte-identical afterwards;
single-curve normalisation and scale invariance; multi-meter user-day
aggregation; equal weighting across users and across days; insufficient-sample
marking with no curve fabricated; 96-point finiteness; determinism across runs;
exclusions locating their Excel row without exposing account or meter numbers.

**Regression:** `python -m pytest tests/` -> **614 passed, 50 deselected, 0
failed** (181 s). All 46 pre-existing modules remain green.

Two defects were found by writing the tests and fixed in the module:

1. **Duplicate rows were being double-counted.** `duplicate_row` was advisory,
   so repeats stayed in the shape pool while the report claimed they were
   dropped. A user with 13,718 "meter-days" from 11 meters over 731 days made
   it visible. Fixed with an explicit `SHAPE_DROP_FLAGS`; observation totals
   fell accordingly (商业综合体 28,282 -> 22,151).
2. **`other_phase` could never fire.** The flag was declared but `MeterDay`
   carried no phase field, so nothing ever set it. Now applied, and it flags
   exactly the 81 rows the Phase-0 audit had identified.

## 7. Unresolved data-quality limits

1. **Absolute MW is not produced, and should not be.** `综合倍率` reconciles
   with declared 运行容量 for 87 of 200 checkable meters and contradicts it for
   **113**. The worst case implies 4,594x the rating. `absolute_mw()` therefore
   raises unless a caller explicitly names verified industries. The scenario
   built on this data will run on *IEEE 33-bus peak demand redistributed by
   real Jiangsu industry shapes*, **not** on Jiangsu demand.
2. **`capacity_inconsistent` touches 16,653 records (14 %).** This is reported,
   not resolved. Whether the ratio or the rating is wrong is undecidable from
   the file alone.
3. **`ratio_suspect` fires on 9,827 records.** The rule compares each meter to
   its industry's median, and ratios legitimately span a wide range within an
   industry, so this flag is loose and should be read as "worth a look", not as
   a defect count.
4. **5G基站 and 电解铝 are unreliable.** 5G基站 loses three of five users to
   incomplete days (many meters record only every 30 minutes); 电解铝 rests on
   one user and 208 days. Neither curve should carry weight in a conclusion.
5. **2,683 negatives on non-PV industries are unexplained.** They are flagged
   and preserved, not interpreted. 印刷业3 accounts for 773 of them.
6. **Nothing is imputed.** 4,609 incomplete days are dropped rather than
   filled, so seasonal or weekday coverage is uneven. No check was made that
   the surviving days are representative of the year.
7. **The parquet is 71 MB.** It holds 87,511 x 96 float64 curves and is
   regenerable. If size matters it can be written as float32 (~half) — left
   as-is so no precision is silently discarded.

**Traceability caveat:** all five `data/processed/` artefacts are matched by
`.gitignore` (`*.csv`, `*.parquet`, `*.png`); the only data exception is
`!data/wholesale/*.csv`. They are therefore **not version-controlled**. The
quality report and this report are markdown and will be tracked. If these data
artefacts are meant to be part of the delivered record, `.gitignore` needs an
exception; that was left to you rather than changed unasked.

## 8. Difference from the pre-Phase-1 workspace

Recorded before any edit, and re-verified after:

```
HEAD            a1151fd2ab52e4a49a610173a1bba04c56bda563   (unchanged)
branch          master                                      (unchanged)
git diff --stat 14 files changed, 43 insertions(+), 413 deletions(-)
```

Backed up to `/tmp/phase1_backup/pre_phase1_worktree.diff`
(640 lines, sha256 `f0c79a4f34992721e5b035ffd9d39cda71310bb8977f5f85f1342778e7e03f46`).

The post-run `git diff --stat` is **byte-identical** to the pre-run one, so the
13 modified `src/*.py` files and `TYPICAL_DAY_PHYSICAL_BASELINE.md` are
untouched. `gurobi.env` was not read, copied or listed. No `reset`, `checkout --`
or `clean` was run.

Net change to the working tree: four new untracked files
(`src/industry_load.py`, `tests/test_industry_load.py`, and the two reports),
plus the ignored `data/processed/` artefacts.

## Status

Phase 1 complete. Stopping here for review; Phase 2 is not started.
