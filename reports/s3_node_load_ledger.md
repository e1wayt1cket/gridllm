# S3: the node-load ledger

The 33-bus composition for one scenario on one real date, with every row
carrying the unit it was computed under and whether that unit is confirmed.

**This is an audit ledger, not the network's formal load.** The reading unit is
`unit_unverified`, so the formal absolute-MW path refuses and a test pins that
it refuses. What is below can be examined and argued with; it may not be cited
as the network's absolute load.

| | |
|---|---|
| scenario | `typical_day_industry` |
| date | 2022-10-14 |
| buses | 33 (0..32) |
| rows | 3,168 (33 x 96) |
| unit status | **`unit_unverified`** (evidence: `ruling_only`) |
| mapping version | `industry_bus_map_v2` |
| dataset version | `node-loads-audit/e340dd25668d` |
| leadger kind | `audit` |

## What the verification pass found

The S3 gate asked for the S2 statistics to be audited before anything was
built on them, and it found a real error.

**The S2 per-bus peaks were wrong.** They accumulated every day's power into the
same 96 period buckets, so the "peak" was a sum over roughly 130 days. A peak is
the largest sum of a bus's meters at a single (date, period); the sum of the
per-meter peaks is its ceiling. Corrected:

| bus | industry | mean MW | peak MW, corrected | as first reported |
|---|---|---|---|---|
| 16 | 钢铁 | 40.13 | **274.35** | ~~36,973~~ |
| 23 | 化工业 | 29.14 | **81.80** | ~~22,686~~ |
| 22 | 水泥制造 | 12.78 | **34.38** | ~~16,711~~ |
| 3 | 商业综合体 | 4.48 | **12.21** | ~~6,031~~ |
| **system coincident peak** | | 113.94 | **466.37** | ~~92,023~~ |

`reports/industry_bus_mapping_v2.md` and `reports/industry_load_rebuild.md` are
corrected, each with a note saying what was wrong. **The means and the energies
were never affected** — they are energy over hours and do not pass through the
period buckets. The magnitude conclusion changes from four orders of magnitude
to two: the system peak is **125.5x** the feeder, not 24,771x.

The same class of error appeared again in the first draft of this module, where
a membership test on `(meter, date)` without a `md.date == date` guard summed a
meter's whole history into one day. It is fixed, the excluded-cell count fell
from 2,028 to 1, and `tests/test_node_load_builder.py` pins it with a fixture
that supplies two dates and asserts the first is unchanged.

## The unit question, asked the way the gate asks it

The reading unit and the `运行容量` unit are audited **separately**, and neither
is inferred from their ratio — a comparison whose own denominator has no stated
unit cannot certify a numerator.

| question | answer |
|---|---|
| Does the workbook state the reading's unit? | **No.** Header names twelve fields and ninety-six times of day; package metadata has no title, subject or description |
| Does any external source state it? | **No** — no data dictionary, no field specification, no platform export record |
| Does the workbook state `运行容量`'s unit? | **No** |
| Is there independent evidence for either? | **No** |

So the reading unit is marked **`unit_unverified`**. `industry_load` now carries
that status separately from the declared unit: `READING_UNIT_EVIDENCE =
"ruling_only"`, and `require_verified_unit()` refuses. A conversion is still
computed for this audit ledger so the figures can be read; it may not be written
as the network's formal load. Nothing was adjusted toward its capacity, and no
record was discarded for being unverifiable.

For the record, what the capacity ratio *does* say — reported, not used as
evidence: over the 101 mapped meters with a usable rating, 14 peak at 44–90% of
it and 58 under 5%. `运行容量` spans 160 to 600,000. The field behaves like a
power rating for a minority of meters and not for the majority.

## The 33-bus classification

Rebuilt from `industry_meter_bus_map_v2.csv` on every run, never from a list
kept by hand — that is how a bus came to be described as both carrying a user
and being empty.

- **The network has 33 buses, numbered 0..32.** Bus 0 is the slack. **There is
  no bus numbered 33**; the 33rd bus is index 32. The 32 load buses are 1..32,
  and the network has 37 line elements of which **32 are in service** (the other
  five are the normally-open ties 20-7, 8-14, 11-21, 17-32, 24-28), which is the
  standard IEEE33 topology.
- Bus 0 is classified `slack`, is not a mapping target, and is written with
  `native = industry = final = 0`.
- 17 buses carry a mapped user; **16 do not** (the 15 load buses plus the slack).
- `has_user` and `no_user` are complements by construction, and
  `assert_classification_complete` rejects a classification that omits a bus,
  double-classifies one, or gives the slack a user.

The earlier report's ambiguity — the **count** "17 buses carry a user" sitting
beside the **bus number** 17 in the list of empty buses — is gone: the count and
the bus number are now different columns of the same table.

## The per-bus policy

Every one of the 33 buses is declared in `config/scenarios.yaml` under
`load_model.bus_load_policy`; a missing bus or an unknown value raises rather
than taking a default.

| policy | buses | why |
|---|---|---|
| `replace` | 17 — the ones with a mapped user | the metered users *are* that bus's load; the IEEE33 rated demand for the same bus is a benchmark number for a benchmark feeder, not a second load standing behind them |
| `preserve` | 16 — including the slack | no mapped user: the bus keeps the benchmark's own demand |

`overlay` is available and unset. Treating the benchmark demand as a background
load the metered users sit on top of is a scenario decision, not a default, and
it is recorded as open in the decision record.

**A bus with no mapped user is never stripped of its load.** The complement is
enumerated in the config rather than left to a default, and a `no_user` bus's
ledger row carries the benchmark demand in `native_mw` and the same value in
`final_mw`.

## The ledger

Composition is applied once, here, and only here:

```
preserve   P_final = P_base
replace    P_final = P_industry
overlay    P_final = P_base + P_industry
```

`P_base` is read from the built network's own `net.load.p_mw` and spread within
the day by the same `grid._make_load_profile(shape="native")` the agents would
otherwise carry — so there is one definition of the native profile, not two.
`P_industry` converts each qualified meter-day once, sums by user, then by bus.
Users are summed, never averaged: five users on one bus at 10 MW each is 50 MW.

Per-bus, on 2022-10-14:

| bus | policy | native MWh | final MWh | peak MW | periods without industry data |
|---|---|---|---|---|---|
| 0 (slack) | preserve | 0.00 | 0.00 | 0.00 | 0 |
| 1 | replace | 1.35 | 67.88 | 5.44 | 0 |
| 3 | replace | 2.69 | 210.36 | 16.28 | 0 |
| **16** | replace | 5.38 | **4,994.73** | **272.63** | 0 |
| 17 | preserve | 1.22 | 1.22 | 0.09 | 0 |
| 23 | replace | 2.70 | 2,651.43 | 114.72 | 0 |
| **32** | replace | 2.15 | **—** | **—** | **96** |
| … | | | | | |

**System: 10,149.59 MWh on the date, sum of per-bus peaks 533.70 MW.**

Bus 32 has a `replace` policy and **no admissible industry data for the whole
day**: its 96 rows carry an empty `final_mw` with
`industry_status = excluded_no_data` and a reason. They are **not** zeros — an
unobserved bus must not be reported as an empty one. This is the one case the
module refuses to compose, and it is visible rather than absorbed.

## The date, and the bias in choosing it

The ledger names a real load date: **2022-10-14**, chosen as the date with the
most complete qualified meter-days, ties broken by earliest. The rule is
deterministic and stated, so two runs on the same workbook build the same
ledger.

**That rule has a bias worth knowing**: more complete meters means more load, so
the chosen date sits high in the 730-day distribution. Bus 16's average that day
is 208 MW against a 730-day average of 40 MW. A median-coverage day would be
more representative; the completeness rule was kept because it maximises what is
actually observed, and the choice is recorded here rather than presented as
typical.

## Price alignment

The load curve is a real dated curve. The price curve is **not** date-matched:
the only dated real price curve is `261009`, which covers 2026-10-09, and the
three others are window means with no dates recorded. There is **no date on
which a real price and a real load day are both in hand**, so the price is used
as a **declared price scenario** — labelled as such, never described as the load
day's real price, and supporting no cross-day generalisation. No price was
copied onto another date and no synthetic price was substituted.

## Tests

`tests/test_node_load_builder.py`, **20 passed**:

| what it pins |
|---|
| all 33 buses classified; `has_user` and `no_user` are complements; the count comes from the mapping file |
| the slack is not a mapping target and is written as an explicit zero |
| `preserve` / `replace` / `overlay` are the stated formulas, and `overlay` adds rather than averages |
| a `replace` or `overlay` bus with no industry data is excluded, never zero-filled |
| an unknown policy, or a missing bus policy, raises |
| **one ledger row set is one date** — a meter's history is not summed into a single day |
| a user is never counted on two buses; users are summed |
| an unmapped user and an incomplete day are excluded with reasons |
| the date pick is deterministic |
| the formal ledger is refused while the unit is unverified, and opens once an external source states it |
| the unit status travels with every audit row |

`tests/test_industry_bus_mapping.py` (13), `tests/test_industry_load.py` and
`tests/test_screen_industry_blocks.py` also pass.

## Deliverables

| artefact | what |
|---|---|
| `config/scenarios.yaml` | `load_model` with `bus_load_policy` for all 33 buses |
| `data/processed/node_loads/typical_day_industry/bus_loads_96x33_audit.csv` | 3,168 rows, one per bus and period |
| `…/manifest.json` | date, policies, unit status and its evidence, mapping digest, per-bus user counts, dataset version |
| `reports/bus_load_composition.csv` | 33 rows: role, status, users, meters, industries, policy, native and final energy, peak, periods without data |
| `reports/node_loads_summary.csv` | one row per scenario ledger |
| `src/build_industry_node_loads.py` | the builder, `--formal` refused |

## What is blocked, and by what

**The formal absolute-MW ledger cannot be frozen yet.** Two gates are open:

1. **The reading unit is `unit_unverified`.** An external source must state it —
   a data dictionary, a field specification, or a written statement from the
   producer. Only then can `--formal` run and the network's absolute load be
   published.
2. **The date selection is biased toward high-coverage days**, as above. A
   median or specified date should be chosen before freezing.

Nothing in this stage scaled a load, discarded a record, adjusted a reading
toward its capacity, modified the workbook, touched the IEEE33 topology, or
substituted synthetic data. S8's network validation waits on both gates.
