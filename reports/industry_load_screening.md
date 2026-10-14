# Industry load screening: what the validators actually decide

Evidence for the load-input ruling, produced by re-running the shipped
validators against the workbook on disk. No reading was edited, imputed or
clamped; nothing here converts to MW.

## How to reproduce

```bash
PYTHONPATH=src python src/industry_load.py                       # shapes + quality report
PYTHONPATH=src python src/screen_industry_blocks.py --out <scratch>
python -m pytest tests/test_industry_load.py tests/test_screen_industry_blocks.py -v
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
The earlier "87 个合格数据块" read it as a block count and is withdrawn.

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

## The frozen rule

**Reading B.** A block is a formal absolute-load block when both hold: it
survives the shape pool, and the meter that metered it passes the
ratio/capacity check. The check runs on the meter, the pool runs on the block,
and both gates must pass — so a meter that fails, or that cannot be checked,
takes all of its blocks out, and no block is picked out of a failing meter to
bypass the meter-level gate. Advisory flags do not disqualify a block; excluding
them is reading C, permitted as a strictness sensitivity check and never as a
substitute for B.

| reading | blocks | meters | users | dates |
|---|---|---|---|---|
| **A** survives the shape pool | 87,511 | 208 | 68 | 730 |
| **B** A + the meter passes ratio/capacity — **frozen** | **39,365** | **84** | **36** | **730** |
| **C** B + the block carries no advisory cap/ratio flag | 35,383 | 74 | 31 | 730 |

A and C bound B and are reported so the cost of loosening or tightening stays
visible. B's meter count is 84 rather than 87 because three of the 87 consistent
meters produce no block that survives the pool.

`industry_block_screening.csv` carries one row per block with its flags, the
meter verdict, all three readings and the reason it falls out of C.

## The trap this table exists to prevent

A block is one meter on one date; a record is one workbook row; a block can hold
more than one record, and the workbook repeats some rows on purpose (a
`duplicate_conflict` is two disagreeing profiles for one meter-day, kept rather
than collapsed). **On this dataset the two counts happen to coincide** —
`qualified_records` and `qualified_block_keys` both return 39,365 — which makes
conflating them look safe here while it is not. The module keeps them as two
functions with separate names for exactly that reason, and a test constructs a
block with two records to pin the difference.

## The gate in code

`industry_load` exposes the frozen rule and nothing weaker:

| function | returns |
|---|---|
| `qualified_meters(records)` | one verdict per **meter**: `checkable`, `qualified`, `peak_ratio`. A meter the check cannot reach is `checkable: False, qualified: False` |
| `qualified_records(records)` | the surviving **records** |
| `qualified_block_keys(records)` | the qualified **blocks** as `(meter_key, date)` keys |

`absolute_mw` still refuses by default. It refuses *for* the qualified set: the
gate is what a magnitude has to pass before one may be built from a block at all.

## Reproducibility and provenance

Re-running the shipped builder reproduced every committed shape artefact **byte
for byte** — `industry_typical_profiles.csv` `0a19913e…bb8f56`,
`industry_exclusions.csv` `3ce33d58…2a45881d`, `industry_profile_summary.csv`
`87cb143d…67be262e` — so the artefacts dated 2026-10-09 are reproducible from
this workbook with these validators.

Source workbook `docs/16类行业用户.xlsx`, sha256
`63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb`,
67,277,368 bytes, 2026-09-17. `docs/16类行业用户 - 副本.xlsx` hashes to the same
value: the two files are the same bytes.

The tie is now recorded rather than remembered:

- the generated `reports/industry_load_quality_report.md` carries the workbook
  digest and a **Qualified subset** section stating both units;
- `industry_block_screening_manifest.json` carries the workbook digest and byte
  count, a `screening_rule_version` digest computed over the rule inputs
  (tolerances, both flag lists, slot width), the generation timestamp, the block
  table's digest and a `dataset_version` derived from it.

A change to any tolerance moves the rule digest, so a manifest cannot describe a
different rule than the one that produced its result. The block table itself is
deterministic across runs; only the manifest's timestamp moves.

## What remains open

The gate and its count are frozen. Not yet decided, and deliberately not decided
here: the 32-load-bus simulation mapping (S4.3), the physical capacity review
against the mapped absolute load (S4.4), and the aggregation rule that turns
qualified blocks into per-bus absolute MW — including what a block with two
disagreeing records contributes.
