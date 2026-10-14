# Industry load screening: what the validators actually decide

Evidence for the load-input ruling, produced by re-running the shipped
validators against the workbook on disk. No reading was edited, imputed or
clamped; nothing here converts to MW.

## How to reproduce

```bash
PYTHONPATH=src python src/industry_load.py --out <scratch> --report <scratch>/quality_report.md
PYTHONPATH=src python src/screen_industry_blocks.py --out <scratch>
python -m pytest tests/test_screen_industry_blocks.py -v
```

## The rules and tolerances in force

Read from `industry_load`, not restated:

| item | value |
|---|---|
| drops a record from the shape pool | `empty_identity`, `missing_slots`, `all_zero`, `placeholder`, `other_phase`, `duplicate_row` |
| advisory only, never drops | `negative_load`, `negative_pv`, `ratio_missing`, `ratio_suspect`, `capacity_inconsistent`, `duplicate_conflict` |
| capacity tolerance | implied peak ≤ **1.5x** 运行容量, peak = `max\|slot\| * 综合倍率 * 60/15` |
| ratio outlier factor | departs from the industry median by more than **100x** — descriptive |
| reliable industry floor | ≥ **3** users and ≥ **30** meter-days |

## The unit 87 belongs to

`verify_ratios` checks one thing per **meter**: whether the peak its 综合倍率
implies fits the 运行容量 the workbook declares for it.

| | |
|---|---|
| meters checked (综合倍率 and 运行容量 both present and usable) | **200** |
| meters whose implied peak fits 运行容量 (≤ 1.5x) | **87** |
| meters whose implied peak contradicts it | **113** |
| meters the check cannot reach at all | 83 of 283 |

So **87 is a meter count**, not a count of data blocks, and not users or days.
It is the third line of this table that the earlier "87 个合格数据块" was
reading.

Worse, the two flags that carry this verdict — `capacity_inconsistent` and
`ratio_suspect` — are **advisory**: a block carrying them still enters the
shape pool. The check that produces 87 is a report, not a gate, and the
pipeline has no notion of a "qualified block" in the ruling's sense.

## Units, counted separately on the same run

| unit | count |
|---|---|
| records (workbook rows) | 117,271 |
| blocks (one meter on one date) | 103,739 |
| meters | 283 |
| users | 73 |
| dates | 731 |

Five different numbers, and the ruling is about one of them. Any figure quoted
from this input without its unit is not reproducible.

## Three readings of "qualified"

Emitted side by side by `src/screen_industry_blocks.py`, because the ruling
does not yet fix one. **None of them normalizes, imputes or edits a reading.**

| reading | blocks | meters | users | dates |
|---|---|---|---|---|
| **A** survives the shape pool (the rule in force today) | 87,511 | 208 | 68 | 730 |
| **B** A + the block's meter passes the ratio/capacity check | 39,365 | 84 | 36 | 730 |
| **C** B + the block itself carries no advisory cap/ratio flag | 35,383 | 74 | 31 | 730 |

B is the faithful reading of the ruling as written: it uses the shipped check,
at its shipped tolerance, and asks a block only to clear the gates that already
exist. The meter count under B is 84 rather than 87 because three of the 87
consistent meters produce no block that survives the pool rule at all.

C is stricter than anything the ruling states — it adds a per-block condition
the code has never applied — and is reported so the cost of that choice is
visible rather than discovered later.

## Reproducibility of the artefacts on disk

Re-running the shipped builder reproduced every committed artefact **byte for
byte**:

| artefact | sha256 (regenerated == committed) |
|---|---|
| `industry_typical_profiles.csv` | `0a19913e…bb8f56` |
| `industry_exclusions.csv` | `3ce33d58…2a45881d` |
| `industry_profile_summary.csv` | `87cb143d…67be262e` |
| `industry_load_quality_report.md` | identical by `diff` |

Source workbook `docs/16类行业用户.xlsx`, sha256
`63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb`,
67,277,368 bytes, 2026-09-17. `docs/16类行业用户 - 副本.xlsx` hashes to the
same value: the two files are the same bytes.

**The provenance gap that remains:** the artefacts were written 2026-10-09 and
carry no workbook hash and no generation stamp of their own — the quality
report records the workbook *path*, never its digest. Their tie to this
workbook is established by re-running, which is an inference a reader cannot
repeat from the files alone. Recording the workbook digest in `shapes.meta`
and in the report would close it.

## What the ruling does not yet decide

1. **Uncheckable meters have no verdict.** 83 of 283 meters lack a usable 综合
   倍率 or 运行容量, so "the capacity check passes" cannot be asked of them.
   Whether such a meter counts as qualified (permissive) or excluded
   (conservative) changes B's block count and is a data-quality decision, not
   an implementation detail.
2. **Whether the check gates blocks or meters.** The check is a meter verdict;
   the ruling speaks of blocks. Applying it per meter disqualifies every block
   that meter ever metered, which is what B does. Applying it per block would
   need a per-block ratio, and a block carries the meter's ratio, not its own.
3. **The pool rule and the ratio/capacity check are independent.** A block can
   survive the pool carrying `capacity_inconsistent`, and a block can be
   dropped by the pool with a perfectly consistent meter. The intersection is
   what B counts.
