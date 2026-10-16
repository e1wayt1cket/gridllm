# The slot reading unit: a declared ruling, and the check that did not confirm it

**Gate state: `unit_unverified` — a unit is declared, and nothing external
corroborates it.** A slot value is taken to be the average active power over
its 15 minutes, in kW, by project ruling. This report records why that is a
ruling rather than a citation, what it changed, and the result of the
cross-industry check the ruling asked for. **That check did not corroborate the
ruling.** It is reported here in full rather than set aside.

## Two gates, not one

| gate | opens on | admits |
|---|---|---|
| `require_reading_unit()` | a unit being **declared**, with a recorded source | a conversion for an **audit** ledger, every row carrying the unit and its status |
| `require_verified_unit()` | an **external source** stating the unit | the network's **formal** absolute MW load |

A declaration is not evidence, so the second gate is closed:
`READING_UNIT_EVIDENCE = "ruling_only"` and `unit_status()` returns
`unit_unverified`. The reading unit and `运行容量`'s unit are audited
**separately** and neither is inferred from their ratio — a comparison whose own
denominator has no stated unit cannot certify a numerator, whatever plausible
load factors it produces.

What the ratio *does* say, reported and not used as evidence: over the 101
mapped meters with a usable rating, 14 peak at 44–90% of it and 58 under 5%,
with `运行容量` spanning 160 to 600,000. The field behaves like a power rating
for a minority of meters and not for the majority.

The consequence is recorded in `reports/s3_node_load_ledger.md`: the 33-bus
composition is built as an **audit** ledger, and the formal one refuses until an
external source states the unit.

## The declaration

| | |
|---|---|
| `industry_load.READING_UNIT` | `"kW"` |
| `industry_load.READING_UNIT_SOURCE` | `project_ruling_2026-10-14: a slot value is the average active power over its 15 minutes, in kW. This is a ruling, not a data dictionary — no field specification for this workbook is in hand.` |

No source in the repository, in the workbook, or reachable from it states the
unit. The header names twelve fields and then ninety-six **times of day**; the
package metadata carries no title, subject or description; the only column that
describes the readings is `功率相序 = 有功功率`.

## The two consequences

1. A 96-slot day is 96 average powers, so a day's energy is their sum times
   `HOURS_PER_SLOT` (0.25 h). **A slot value is never multiplied by four.**
2. The capacity check's `60/SLOT_MINUTES` factor is **removed**. Both sides are
   kW, so the implied peak is `max|slot| * 综合倍率` and nothing else.

`reading_to_mw` is the only sanctioned conversion: it calls
`require_reading_unit()` first, refuses a unit it does not implement, and
returns `None` — never zeros — for a record that is not a complete
active-power day. `absolute_mw` is retired; it multiplied a *normalised shape*
by a rated capacity, which is a magnitude with no measurement behind it.

## The cross-industry check, and what it found

The check compares each meter's implied peak and mean power against its declared
`运行容量`, over **all 198 meters** that have a usable multiplier and rating —
not only the qualified ones, so the selection the capacity gate performs cannot
flatter the result.

| reading hypothesis | peak / 运行容量 p10 | median | p90 | share > 0.5 |
|---|---|---|---|---|
| **average kW (the ruling)** | 0.000 | **0.069** | 0.720 | 18.2% |
| energy per slot (pre-S1) | 0.001 | 0.278 | 2.880 | 41.9% |

Median load factor under the ruling: **0.044**; 48.5% of meters sit above 0.05.

**Neither hypothesis is clean.**

- Under the ruling, the median meter draws **6.9%** of its declared running
  capacity. That is low for a load, though not impossible where 运行容量 is a
  contracted rather than a real maximum. Only 18% of meters reach half their
  rating.
- Under the retired reading, the median meter draws 27.8% — more typical — but a
  **tenth of meters imply 2.88x their declared capacity or more**, which a real
  installation cannot do continuously. That reading is refuted by the top of its
  own distribution.

By industry the ruling splits sharply, which is the same disagreement at finer
resolution:

| plausible under the ruling | implausible under the ruling |
|---|---|
| 水泥制造 0.578, 机械 0.415, 电解铝 0.371, 电子设备制造业 0.348, 5G基站 0.289, 汽车制造业 0.242, 化工业 0.207, 通用设备制造业 0.202 | 纺织业 0.007, 电制氢 0.011, 商业综合体 0.022, 数据中心 0.030, 电动汽车 0.039 |

**Conclusion of the check: the ruling is not corroborated, and it is not
refuted either.** Nine of the ten industries the check was asked to cover fall
into two camps that cannot both be right under one unit. The honest reading is
that this workbook's readings are not consistent with a single unit across all
its meters, or that `运行容量` is not a uniform power rating — and neither can
be settled from the file.

**What this does not change.** The ruling stands as declared, because it is the
owner's decision and no source contradicts it outright. Every figure derived
under it inherits this uncertainty, and the reports that state those figures say
so. No scaling was introduced to make the result look better: the load factors
above are reported at the value they take.

## Residual limits, stated rather than hidden

1. **`运行容量` and `合同容量` have no stated unit either.** The capacity check
   is meaningful only if they are the same power unit as the reading; it is a
   consistency filter, never evidence for the unit.
2. **Nine of ten industries do not behave as one unit predicts**, as above.
3. **An independent check that does not use the capacity relation is still
   missing.** A meter whose interval energy can be reconciled against a
   separately known consumption total would settle the scale without dividing by
   the same unstated denominator.

## What would replace the ruling with a citation

Any one of these, recorded in `READING_UNIT` and `READING_UNIT_SOURCE`:

1. the field specification or data dictionary for the system that exported the
   workbook — naming the unit of the slot values, and of `运行容量` and
   `合同容量` while at it;
2. a written statement from whoever produced or supplied the file;
3. an independent measurement that pins the scale without the capacity relation.

The gate refuses a unit recorded without a source, so replacing this ruling is a
deliberate act that leaves a trace.
