> **Superseded (pre-S1).** The meter set, the coverage counts and the frozen
> `R3_meter_spread` mapping below were all computed from the four-fold capacity
> reading and from the meter-level rule. Both have been replaced: the counts are
> recomputed in `reports/industry_load_rebuild.md`, and the S2 user-level mapper
> supersedes `R3_meter_spread` outright — a user's meters must be aggregated to
> one unit and land on one bus, not be spread across several. The audit method
> described here is still the one in use.

# S4.3: candidate bus mapping and data coverage

What the 84 qualified meters can be attached to, and what the workbook actually
holds for them. **Nothing here is frozen.** Three candidate assignment rules are
generated side by side; none is adopted, and the numbers that would change the
research conclusion are stated as findings rather than resolved.

Produced by `src/map_industry_meters.py`; reproduce with

```bash
PYTHONPATH=src python src/map_industry_meters.py --out data/processed
python -m pytest tests/test_map_industry_meters.py -v
```

## The declared industry assignment

`typical_day_industry.industry_map` is explicitly a simulation mapping
(`simulation_mapping: true`) and states bus -> industry, which is what a shape
needs. It covers all 32 load buses. Inverting it:

| industry | buses |
|---|---|
| 商业综合体 | 20 buses: 1–15, 17–21 |
| 钢铁 · 水泥制造 · 化工业 · 汽车制造业 · 通用设备制造业 · 电子设备制造业 · 纺织业 · 机械 · 印刷业 · 电制氢 · 电动汽车（充换电） · 数据中心 | one bus each: 16, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 |

`excluded_industries: [分布式光伏, 5G基站, 电解铝]`.

**Why this mapping cannot be reused unchanged for absolute load.** Assigning one
industry to each bus is safe for *shapes*, which are per-meter normalized and
carry no magnitude. It is not safe for an absolute magnitude: twenty buses take
商业综合体, so putting every 商业综合体 meter's energy on all twenty would count
each meter twenty times. The absolute route needs a partition — one meter, one
bus — and producing that partition is what this stage adds.

## The qualified set, and who it is

84 meters, 36 users, 730 dates. Meter counts by industry:

| industry | meters | buses available |
|---|---|---|
| 商业综合体 | 35 | 20 |
| 纺织业 | 9 | 1 |
| 化工业 | 6 | 1 |
| 数据中心 | 6 | 1 |
| 钢铁 | 5 | 1 |
| 汽车制造业 | 4 | 1 |
| 电制氢 | 4 | 1 |
| 水泥制造 | 3 | 1 |
| 电动汽车（充换电） | 3 | 1 |
| 印刷业 · 机械 · 通用设备制造业 · 电子设备制造业 | 2 each | 1 each |
| 5G基站 | 1 | **0 — excluded industry** |

No meter carries two industries, and no user spans two industries, so the
"one meter, one bus" and "a user's meters together" premises both hold on this
data.

### Meters that pass the ratio/capacity gate but contribute nothing

Three, and all three because their readings are zero on essentially every day—
they are excluded from the 84 by the shape pool, not by the meter gate:

| meter | industry | days in workbook | dominant flags |
|---|---|---|---|
| `M316fd74d20e9` | 商业综合体 | 719 | `all_zero` x1438, `duplicate_row` x719 |
| `M778133cf9e03` | 通用设备制造业 | 729 | `ratio_suspect` x729, `all_zero` x728 |
| `M85260b4c737e` | 电子设备制造业 | 730 | `all_zero` x730 |

**This one needs a ruling.** An all-zero meter is either a site that genuinely
drew nothing or a meter that stopped reporting, and the two have opposite
consequences: under the first reading, dropping it loses no energy; under the
second, the formal load is biased low by an unknown amount. The workbook cannot
distinguish them, and this report does not guess.

### The one meter that cannot be mapped

`5G基站` has one qualified meter, and its industry is in `excluded_industries`
because the workbook carries only two 5G users. It is listed as `unmapped` with
its reason in every candidate file rather than dropped. Options, none chosen:

1. keep it out, as today (the bus it would need does not exist);
2. give 5G基站 a bus and accept a two-user industry shaping it;
3. fold it into another class — it is a small, flat, always-on load, closest in
   shape to 数据中心, which already has a bus.

## Three candidate rules

| rule | what it does | mapped | buses used | dates with every used bus populated |
|---|---|---|---|---|
| `R1_industry_bus` | meter takes its industry's bus; undefined where the industry has several | 48 / 84 | 12 | 0 |
| `R2_user_block` | a user takes one bus; all of that user's meters follow | 83 / 84 | 17 | 0 |
| `R3_meter_spread` | meters of an industry are spread one per bus by position | 83 / 84 | **32** | 0 |

`R1` leaves 35 商业综合体 meters and the 5G基站 meter unmapped — it is included
to show what the current mapping already supports, not as a contender.

`R2` keeps a participant on one node, which is the more physical rule, but
商业综合体 has only 5 users against 20 declared buses, so **15 of those 20 buses
would carry nothing at all**. Those buses would then hold zero load in every
absolute day, which is a large change to the network's loading pattern.

`R3` uses every bus, at the cost of splitting a user's meters across nodes. It
is the only candidate that exercises the whole feeder, and it scores best on
coverage below.

## Coverage: what the workbook holds

84 meters x 730 dates = 61,320 cells. Every cell is classified; nothing is
zero-filled or interpolated.

| status | cells | share | meaning |
|---|---|---|---|
| `complete` | 39,364 | 64.19% | 96 slots, active power, usable as-is |
| `absent` | 16,532 | 26.96% | no record for that meter on that day |
| `all_zero` | 4,513 | 7.36% | every slot exactly zero |
| `partial` | 904 | 1.47% | fewer than 96 slots (median 86 present) |
| `placeholder` | 5 | 0.01% | the workbook's synthesised filler row |
| `conflicting` | 2 | 0.00% | two records disagree about the same meter-day |

The date axis is 2021-01-01 to 2022-12-31, 730 days. (The screening manifest
reports 731 dates; the extra one is 81 identity-less records whose date is
blank, which that manifest counts as a bucket. 730 is the real axis.)

**`all_zero` is not spread evenly.** By industry: 水泥制造 2,087, 钢铁 847,
汽车制造业 676, 通用设备制造业 617, 印刷业 193, and 92 across five others. Heavy
industry recording zero for long stretches is the dominant missing-data mode,
and it lands on exactly the industrial buses.

**One whole day is fabricated.** All 43 `placeholder` records in the workbook
fall on **2022-05-26** (41 meters, 10 industries). Five of them survive into the
qualified matrix. That date is not a measurement and is a candidate for removal
from the day axis regardless of which mapping rule is chosen.

## The two constraints that decide what is buildable

### 1. No date can populate every bus

Under **every** candidate, the number of dates on which all used buses have at
least one complete meter is **zero**. The chronic blockers, under `R3`:

| bus | industry | dates with no usable meter (of 730) |
|---|---|---|
| 29 | 印刷业 | 705 |
| 32 | 数据中心 | 698 |
| 22 | 水泥制造 | 648 |
| 20 · 17 | 商业综合体 | 312 · 311 |
| 28 | 机械 | 82 |

Dropping buses is therefore the only route to a fully covered day, and the cost
is immediate:

| buses dropped | buses left | dates fully covered |
|---|---|---|
| none | 32 | 0 |
| 22 | 31 | 0 |
| 22, 32 | 30 | 19 |
| **22, 29, 32** | **29** | **313 (42.9%)** |
| 22, 29, 32, 17 | 28 | 318 |
| 22, 29, 32, 17, 20 | 27 | 508 (69.6%) |

### 2. Meter-level completeness is much easier than bus-level

| threshold | dates meeting it |
|---|---|
| ≥ 80 of 84 meters complete | 0 |
| ≥ 70 of 84 | 44 (6.0%) |
| ≥ 60 of 84 | 226 (31.0%) |
| ≥ 50 of 84 | 426 (58.4%) |
| ≥ 40 of 84 | 727 (99.6%) |
| ≥ 35 of 84 | 730 (100%) |

Complete meters per date: min 35, median 54, max 71. The best date in the
workbook carries 71 of 84 meters, so a "near-complete day" rule is arithmetically
impossible — the ceiling is 71.

Buses populated per date, all three candidates: `R1` median 9 of 12, `R2`
median 13 of 17, `R3` **median 28 of 32** (min 21, max 30). `R3` reaches ≥25
buses on 724 of 730 dates (99.2%).

## Aggregation: the logic, specified but not implemented

For a date `d` and a period `t`, with `m` ranging over the meters assigned to
bus `b`:

```
load_MW[b][t] = SUM_m  reading(m, d, t) * ratio(m) * (60 / SLOT_MINUTES) / 1000
```

subject to all of the following, each of which is a guard rather than a
convention:

- only meters whose cell for `d` is `complete` contribute; a meter is never
  counted twice and never on two buses;
- a cell that is `absent`, `all_zero`, `partial`, `placeholder` or `conflicting`
  contributes **nothing** and is not zero-filled, interpolated or carried over;
- the absolute magnitude is preserved: no normalization, no rescaling onto the
  IEEE33 nominal peak;
- `ratio(m)` is the meter's own declared 综合倍率, unmodified.

**The unit of `reading` is inferred, not documented.** The formula above is the
one the existing capacity check uses (`max|slot| * ratio * 60/15` compared
against 运行容量 in kW), which implies the slot values are energy per 15-minute
slot. That inference has not been confirmed against a workbook column definition
and is worth confirming before any MW figure is published.

### The exception list

Exactly **one** of the 39,365 qualified blocks cannot be aggregated:
`Ma65a1fda1669` (水泥制造) on **2021-01-26**, whose two records carry
`all_zero` and `duplicate_conflict`. One record survived the shape pool, so the
block is in the qualified set, but the two records disagree and summing either
would pick a reading. It is withheld and reported.

Two further conflicted meter-days exist in the workbook (`Md41ef75f772d`
2021-07-28, `Mc865e5dd6961` 2021-08-08, both 水泥制造) and are outside the
qualified set because their meters carry `capacity_inconsistent`.

## Candidate day-level handling for gaps

Not chosen. The trade-off each one makes:

| option | what it does | consequence |
|---|---|---|
| **O1** require every mapped bus populated | keep only fully covered dates | **0 dates** — not viable under any candidate |
| **O2** restrict the date axis by a bus-coverage threshold | e.g. `R3` with ≥25 of 32 buses keeps 724 dates (99.2%) | buses below the threshold carry zero load on some days, silently flattening them |
| **O3** drop chronically thin buses from the model | drop 22, 29, 32 for `R3` | 313 dates fully covered; the feeder model loses three industrial buses |
| **O4** keep all buses and let an unobserved bus carry no load | simplest | an unobserved bus is indistinguishable from a bus that genuinely drew nothing — the same ambiguity as the all-zero meters |

`O1` is arithmetically closed, so the real choice is between accepting
unobserved buses (O2/O4) and removing buses (O3). This report recommends nothing
here: it changes which buses the network has, which is not a data-quality
decision.

## What S4.3 did not do

No mapping was frozen; no aggregation was implemented or run; no bus was added
to or removed from the network; the source workbook was not touched; and no
network capacity, storage parameter or price was changed. `R2` and `R3` both
carry an unmapped 5G基站 meter by design.

## The frozen case

The rulings are applied, and `FROZEN_RULE` in `src/map_industry_meters.py`
carries them rather than a reader's memory of them.

| | |
|---|---|
| rule | `R3_meter_spread`, deterministic, one bus per meter |
| meters mapped | **72** of 84 |
| nodes in use | **29** of 32 — the IEEE33 topology is unchanged, buses 22, 29 and 32 simply carry no mapped load |
| day axis | **729** of 730 dates; 2022-05-26 excluded as the fabricated day |
| **days with all 29 nodes covered** | **312 (42.8%)** |
| nodes covered per day | min 21, median 28, max 29 |

Twelve meters are unmapped, each with a stated reason: 6 for 数据中心 (bus 32),
3 for 水泥制造 (bus 22), 2 for 印刷业 (bus 29) — each industry's only bus is
unsupported — and the 1 `5G基站` meter whose industry is excluded. None is
reassigned to another industry's bus to keep the count up.

The exclusion list carries five kinds: `unmapped_meter` (12),
`unsupported_bus` (3), `fabricated_date` (1), `conflicting_meter_day` (1) and
`unresolved_all_zero_meter` (3).

**The one conflicting qualified block turns out to be moot.** It belongs to
`Ma65a1fda1669`, which is a 水泥制造 meter, and 水泥制造's only bus is 22 —
unsupported. So among the 72 mapped meters **no conflicting meter-day remains**,
and the formal case has no whole-block withholding to perform. The rule is still
enforced and still tested: this is a consequence of the bus ruling, not a reason
to weaken the guard.

### Per-node coverage

Every supported node except three is covered on nearly the whole axis. The
exceptions are 17 (418 of 729 days), 20 (417) and 28 (647) — chronically thin but
not empty, so they stay in the case with their coverage stated rather than
silently flattened.

## The blocker

**The reading unit is undeclared, so no MW figure is produced.** This is not a
gap in this report — it is the applied ruling, and it stops the absolute load
and every physical capacity review that consumes one. Full investigation in
`reports/industry_load_unit_gate.md`. The mapping, the coverage matrix and the
exclusion list are unaffected: they classify and count readings and never
convert one.

## Rulings applied

1. **Mapping** — `R3_meter_spread`, deterministic, one bus per meter.
2. **Unsupported buses** — 22, 29 and 32 carry no mapped load; topology kept.
3. **2022-05-26** — excluded from the formal day axis.
4. **The three all-zero meters** — kept unresolved; excluded from the formal
   data, and *not* declared idle.
5. **The `5G基站` meter** — stays unmapped; not folded into another industry.
6. **Conflicting meter-days** — rejected whole, with a standing test.
7. **Units** — no MW output until a source declares the reading unit.

## Rulings still open

1. **What a node with no usable meter contributes on a date.** The frozen case
   reports it; it does not decide whether the node carries zero load or the day
   leaves the axis. Under `O1` (every node populated) only 312 of 729 days
   survive; the case currently keeps all 729 and states the coverage.
2. **A completeness threshold for the day axis**, if one is wanted.
3. **The three unresolved all-zero meters** — idle site or dead meter.
4. **The reading unit**, which is the active blocker (D8).
