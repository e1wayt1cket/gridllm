# Rebuild after the unit correction: old against new

Every figure here was recomputed from the raw workbook
(`docs/16类行业用户.xlsx`, sha256 `63b826d9…abbb`) under the S1 ruling — a slot
value is average active power in kW, so the capacity check no longer multiplies
an implied peak by `60/SLOT_MINUTES`. Nothing was patched onto the old results;
the pre-S1 artefacts are archived under `data/processed/superseded/pre_s1/`.
The raw workbook was not modified.

## The screening figures, old against new

| | pre-S1 | **post-S1** |
|---|---|---|
| screening rule version | `f647fa60d88c3b29` | **`ecb2b3aed37c520e`** |
| dataset version | `industry-blocks-B/310fbfb3ccde` | **`industry-blocks-B/fb0bee11dccd`** |
| records (workbook rows) | 117,271 | 117,271 |
| blocks (meter × date) | 103,739 | 103,739 |
| meters, all | 283 | 283 |
| meters the check can reach | 200 | 200 |
| meters whose peak fits 运行容量 | 87 | **109** |
| meters whose peak contradicts it | 113 | **91** |
| meters the check cannot reach | 83 | 83 |
| meters with ≥1 qualifying block | 84 | **103** |
| **qualified blocks (B)** | 39,365 | **45,736** |
| users covered | 36 | **43** |
| dates covered | 730 | 730 |

The counts that depend only on what parses — records, blocks, meters reachable,
meters unreachable, dates — are **unchanged**, which is what makes the movement
in the others attributable to the unit correction rather than to a different
read of the file.

Removing the four-fold overstatement loosened the capacity gate, so **more**
meters pass (87 → 109) and **fewer** contradict their rating (113 → 91). The
qualified set grew as a consequence: 39,365 → 45,736 blocks, 84 → 103 meters,
36 → 43 users. No figure was steered toward a target.

## Rebuilt data

| | |
|---|---|
| candidate users (all in the workbook) | 73 |
| **qualified users** | **43** |
| excluded users | 30 |
| valid meter-days (qualified blocks) | 45,736 |
| complete 96-slot curves | 45,736 — every qualified block converted, **0 refused** |
| real dates covered | 730, 2021-01-01 to 2022-12-31 |

Excluded users, by reason:

| reason | users |
|---|---|
| every meter of the user fails the ratio/capacity gate or cannot be checked | 27 |
| meters qualify but no block of the user survives the shape pool | 3 |

None is excluded for a defect discovered by the unit change alone; the shape
pool and the meter gate decide them, as before.

## Totals, and the check that matters

| | |
|---|---|
| total industry energy over the qualified set | **1,996,288 MWh** |
| coincident peak (summed meters, worst instant) | **466.37 MW** |
| mean power over the 730 days | 113.9 MW |
| **the feeder's own nominal peak** | **3.715 MW** |
| **ratio of coincident peak to feeder peak** | **125.5x** |

> **Correction.** An earlier version of this table reported a coincident peak of
> 92,024.7 MW and a ratio of ~24,800x. That figure accumulated every day's power
> into the same 96 period buckets, so it summed across roughly 130 days instead
> of taking a coincidence within one instant. A peak is the largest sum of the
> meters at a single (date, period); the sum of the per-meter peaks, 291 MW, is
> its ceiling and 466 MW for the system is consistent with it. The energy and
> the mean never depended on the bucketing and are unchanged.

**Energy conservation holds exactly.** Converting each record and then summing
gives the same energy as converting the aggregate: 1,996,287.5374 MWh by both
routes, a difference of 1e-8 MWh from floating-point summation. No energy is
created or lost by the aggregation, and `reading_to_mw` refused nothing.

Energy by industry (MWh): 钢铁 703,036 · 化工业 510,537 · 商业综合体 233,372 ·
水泥制造 223,990 · 通用设备制造业 132,961 · 汽车制造业 80,822 ·
电子设备制造业 39,942 · 数据中心 28,125 · eight smaller industries.

## The finding this rebuild produces

**The system coincident peak is 466.37 MW against a 3.715 MW feeder — 125.5x.**
A single 化工业 meter carries a declared 运行容量 of 240,000 (kW under this
ruling, i.e. 240 MW), sixty-five times the whole IEEE33 feeder's demand on its
own; the largest bus average, bus 16 at 40.13 MW, is nearly eleven times what
the entire feeder carries at its own peak.

This is a structural mismatch, not a units rounding: the IEEE33 case is a small
distribution feeder, and the workbook's users are metered industrial customers
of a different scale. Mapping them onto the 32 load buses without scaling — as
ruled — puts the system two orders of magnitude above the feeder's rating, so
the clearing will be infeasible.

Per the ruling this is **reported, not corrected**: no load is rescaled, no
rating is raised, no user is dropped to make the case feasible. S3 builds the
ledger from these figures regardless, and S8 reports the infeasibility with its
locations and causes.

## Caveat carried from the gate report

`reports/industry_load_unit_gate.md` records that the cross-industry check
**did not corroborate** the kW ruling: under it the median meter draws 6.9% of
its declared capacity, which is low, while the retired reading makes a tenth of
meters exceed their rating nearly three fold. The totals above inherit that
uncertainty. They are reported as the ruling computes them and are not adjusted
to look more plausible.
