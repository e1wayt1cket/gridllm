# Data and experiment decisions

The load, price, network and accounting rulings this project runs on, written
down once so a later reader can tell a decision from a measurement. Verdicts
that supersede an earlier document are listed first, because the superseded
documents are still in the tree and still read as authoritative.

## What this supersedes

| Earlier ruling | Where it lives | Status |
|---|---|---|
| Load peak pinned to the IEEE33 nominal 3.715 MW (`load_scale_mode: native_peak`) | `config/scenarios.yaml` (`typical_day`, `typical_day_industry`), `docs/2026-09-18-scenario-revision.md`, `grid.py` `_coincident_peak` | **Replaced** by the absolute industry load below |
| Absolute MW refused outright, module builds normalized shapes only | `industry_load.py` module docstring, `absolute_mw()` | **Amended**: shapes stay as they are; `absolute_mw` is retired and the absolute path runs through `reading_to_mw` under D8 |
| `OBS_V4`'s `avg_other_bid` / `bid_std` averaged each participant's box position across the fleet | `rl_spec.py` `OBS_V4`, `rl_env._compute_opponent_features` | **Superseded** by `OBS_V6`: the features are the fleet mean and spread of a signed, direction-normalized deviation from truthful. `OBS_V4` stays registered for identification |
| `OBS_V5`'s unsigned `|a - truthful| / (high - low)` | `rl_spec.py` `OBS_V5` | **Superseded** by `OBS_V6` before any policy was trained or exported under it; the sign and the per-direction denominator were restored. `OBS_V5` stays registered so its name cannot be read as V6 |
| A slot reading is energy per slot, so the capacity check multiplies an implied peak by 60/15 | `industry_load.py` `_capacity_contradicted`, `verify_ratios`; the pre-S1 screening figures 87 / 113 / 39,365 / 84 / 72 / 312 | **Superseded** by D8: the four-fold factor is removed, the figures are recomputed, and `data/processed/superseded/pre_s1/` keeps the old ones with a marker |
| Meter-level bus mapping `R3_meter_spread` | D7, `map_industry_meters.py` `FROZEN_RULE`/`frozen_case` | **Superseded** by the S2 user-level mapping; a user's meters are aggregated to one unit and land on one bus |
| Train / validation / test sized at 24 / 6 / 6 days | `b/算例设计与算力预算指导.md` | **Not adopted**: candidate scenarios are generated and inspected before any size is frozen |
| Paper text stating a 3.715 MW / 2.3 MVAr base case | `b/01_SCI_IJEPES_中文稿.md` | Stale; the paper text follows this record, not the reverse |

## D1. Load

- The metered Jiangsu workbook is kept byte-for-byte. Its readings, identities
  and metadata are never rewritten, and no exclusion is applied silently.
- The formal absolute load uses **only the blocks that pass the screening in D5**.
- That load is **not normalized**, and it is **not rescaled onto the IEEE33
  nominal peak**. Whatever magnitude the qualified blocks carry is the magnitude
  the study runs at.
- Qualified blocks are aggregated onto the feeder's load buses by an explicit,
  reproducible rule, frozen in **D7**: 29 data-supported buses of the 32. The
  assignment is a **simulation mapping**: this feeder carries no user-to-node
  connection table, and no result may describe the mapping as real wiring.
- Every block enters at most once. Double counting is a defect, not a tolerance.
- **The magnitude itself is gated** by **D8**: no MW figure is derived while the
  reading's unit is undeclared, so the aggregation above is specified but does
  not run yet.

## D2. Network

- The IEEE 33-bus topology and its line impedances are kept.
- Line thermal ratings, the PCC/upstream rating and the reverse-flow limit are
  **re-reviewed against the mapped absolute load**. A rating is not raised to
  make a clearing feasible; each adjustment records its basis, the value before
  and the value after, and the violation check that motivated it.

## D3. Price

- The formal real price source is the Jiangsu day-ahead workbook
  (`dianyicha_price.py`). It holds **one real day and three window means** —
  `261009`, `7days`, `30days`, `halfyear` — so it is not a series of
  independent days and cannot support a day-level split. It supports a
  descriptive bound analysis under a small sample, and nothing generalizing
  across days.
- `cycle_cost` stays at 100 CNY/MWh/leg and both efficiencies stay at 0.92.
  They are not moved to manufacture an arbitrage spread: a real price set with
  little tradeable margin is an accepted outcome, not a defect to tune away.
- The wholesale curve is an external input. The nodal price inside the feeder
  is an **internal** LMP and is never reported as a Jiangsu settlement price.

## D4. Model and accounting

- Facilities: the independent storage fleet from `independent_storage_config`,
  with the terminal SOC pinned and neutralized by settling both arms over the
  whole day (`counterfactual_eval.settle_day`), never by a money adjustment.
- Action space: `rl_spec.storage_quote_v2` for a battery — `charge_price` in
  [0, 150] and `discharge_price` in [0, 300] CNY/MWh, truthful at (0, 0). A load
  or prosumer takes the shading pair truthful at (1, 0). The spec is chosen by
  participant type, never by whether the participant happens to carry a battery.
- Observation: `OBS_V6`, 12 dimensions. It is causal by contract — every price
  feature is a function of prices already published to the participant, and the
  decision-block load, RE and system ratio come from the day-ahead forecasts in
  **both** stages. A V3 checkpoint is not a valid V4 initialisation and is
  refused rather than remapped. V6 keeps V4's layout and causality and redefines
  the two opponent features as `avg_other_dev` / `all_dev_std`, the fleet mean
  and spread of a **signed, direction-normalized** deviation from truthful
  (`(a - truthful) / (high - truthful)` above, `(a - truthful) / (truthful - low)`
  below, 0 at truthful); no V4 or V5 actor is a valid V6 initialisation.
- Economic conclusion: settled per-participant payoff from `participant_payoff`,
  through `counterfactual_eval`. The training reward is a learning diagnostic
  and is never reported as profit; the clearing objective is neither.

## D5. The qualified-block criterion

### The three units

A **record** is one workbook row. A **block** is one meter on one date, and it
can hold more than one record — one of them being dropped does not drop the
block, and one being kept does not admit the others. A **meter** is the unit the
ratio/capacity check speaks about. Three different things; none is ever
substituted for another, and a count without its unit is not a result.

### The frozen rule (reading B)

A block is a formal absolute-load block when **both** hold:

1. the block survives the shape pool — it carries none of `SHAPE_DROP_FLAGS`;
2. the meter that metered it passes the ratio/capacity check — its implied peak
   fits 运行容量 within `CAPACITY_PEAK_TOLERANCE`.

What that fixes:

- **The check runs on the meter; the pool runs on the block.** A meter that
  fails, or that cannot be checked, takes *all* of its blocks out. No block is
  ever picked out of a failing meter to bypass the meter-level gate.
- **A meter the check cannot reach is not qualified.** 83 of 283 meters lack a
  usable 综合倍率 or 运行容量. Missing information is not a pass — but such a
  meter is not thereby accused of being wrong either. It is out of the formal
  absolute-load input, its records stay untouched, and it can be re-judged the
  day the information arrives.
- **Advisory flags do not remove a block.** `ratio_suspect`,
  `capacity_inconsistent`, a negative reading and a disagreeing repeat are
  reported, never disqualifying. Excluding them is reading C: permitted as a
  separate strictness sensitivity check, never as a substitute for B.
- **The shipped validators and their tolerances are the authority.** Where
  neither `flag_records` nor `verify_ratios` states a rule for something the
  criterion needs, the rule is written down and tested before it is used. A new
  loose threshold invented to admit blocks is not a rule.

### What the rule yields, and in what unit

| | |
|---|---|
| records (workbook rows) | 117,271 |
| blocks (meter × date), all | 103,739 |
| meters, all | 283 |
| meters the check can reach | 200 |
| meters whose implied peak fits 运行容量 | **109** |
| meters whose implied peak contradicts it | 91 |
| meters the check cannot reach | 83 |
| meters with at least one qualifying block | 103 |
| **formal absolute-load blocks (B)** | **45,736** |
| users covered | 43 |
| dates covered | 730 |

**109 is a meter count. 45,736 is a block count.** The earlier "87 qualified
blocks" read the first as the second and is withdrawn. The meter count is 103
rather than 109 because six of the 109 consistent meters produce no block that
survives the pool. Reading A (87,511 blocks) and reading C (41,521) are recorded
as bounds and are not the formal input.

These figures are post-S1 (D8): the capacity check no longer multiplies an
implied peak by `60/SLOT_MINUTES`. The pre-S1 figures — **87 / 113** meters,
**39,365** blocks, **84** contributing meters, **36** users — are superseded and
archived under `data/processed/superseded/pre_s1/`. The rule digest moved from
`f647fa60d88c3b29` to `ecb2b3aed37c520e`, and a test asserts it can never return
to the old value.

### Reproducing it

The screening is reproducible from the source workbook alone, and the output
carries the tie rather than relying on memory of it.
`src/screen_industry_blocks.py` writes
`industry_block_screening_manifest.json` alongside its block table, holding the
source workbook's SHA-256 and byte count, a `screening_rule_version` digest
computed over the rule inputs (tolerances, both flag lists, slot width), the
generation timestamp, the block table's own digest and a `dataset_version`
derived from it. Any change to a tolerance moves the rule digest, so a manifest
cannot describe a different rule than the one that produced its result.

Workbook SHA-256 `63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb`,
67,277,368 bytes. `docs/16类行业用户 - 副本.xlsx` hashes to the same value: the
two files are the same bytes.

## D6. Splitting

- A day is the unit. Splits are drawn by `day_id`, never per episode, and no
  `day_id` appears in more than one split.
- Synthetic days used for method validation are labelled `source: "synthetic"`
  and are reported separately from the real-price layer. A result computed on
  them is never a statement about Jiangsu prices.
- The size and stratification of a split are frozen only after candidate
  scenarios are generated and their regime coverage and variance inspected.

## D7. The meter-to-bus mapping

The 84 qualified meters become load on the 32 IEEE33 load buses under one rule,
frozen here.

- **The rule is `R3_meter_spread`.** Within an industry, meters are ordered by
  id and distributed one per declared bus for that industry. The generation is
  deterministic and each meter takes exactly one bus, whatever order the meters
  are read in. `config/scenarios.yaml` remains the authority on which bus takes
  which industry; the mapping is inverted from it, never restated.
- **Buses 22, 29 and 32 carry no mapped load.** Each is the sole bus of an
  industry whose meters cover too few days to populate it — 印刷业 is missing on
  705 days of 730, 数据中心 on 698, 水泥制造 on 648. Their meters are reported as
  excluded with that reason and are not moved to another industry's bus. **The
  IEEE33 topology is unchanged**: the buses exist, they simply carry no mapped
  load, so the formal case has 29 data-supported load buses.
- **The day axis excludes 2022-05-26.** Every one of the workbook's 43
  `placeholder` records falls on that date, so it is not a measurement.
- **Every mapping is a simulation mapping.** `simulation_mapping: true`; this
  feeder carries no user-to-node connection table and no result may describe the
  assignment as real wiring.
- **Every exclusion is listed**, never absorbed: an unmapped meter, an
  unsupported bus, a fabricated date and a withheld meter-day each appear in the
  exclusion list with its reason.

Frozen outcome: 72 of 84 meters mapped across 29 nodes, over a 729-day axis, of
which **312 days have every one of the 29 nodes covered**. Those figures are
pre-S1 and are superseded: they were computed from the meter set the four-fold
capacity reading produced. The S2 user-level mapping replaces both the rule and
the numbers, and D7 in particular is superseded — a user's meters must not be
split across buses.

## D8. The reading unit: declared as a ruling, and the four-fold error removed

- **A slot reading is the average active power over its 15 minutes, in kW.**
  This is a **declared project ruling, not a data dictionary fact** — no field
  specification for this workbook is in hand, and the ruling says so in
  `READING_UNIT_SOURCE` rather than dressing itself up as a citation.
- **Two consequences, both load-bearing.** A 96-slot day is 96 average powers,
  so a day's energy is their sum times `HOURS_PER_SLOT` (0.25 h) and a slot value
  is **never multiplied by four**. And the capacity check's `60/SLOT_MINUTES`
  factor is **removed**: both sides of that comparison are kW, so the implied
  peak is `max|slot| * 综合倍率` and nothing else.
- **`CAPACITY_PEAK_TOLERANCE` stays at 1.5.** It was ~4x looser in practice
  while the denominator of the comparison was inflated; it is now a genuine
  tolerance. The value is deliberately unchanged, so the unit correction is not
  confused with a threshold decision.
- **The consequence for the data.** Removing the four-fold overstatement raised
  the meter count whose implied peak fits 运行容量 from 87 to **109** and lowered
  the contradictions from 113 to 91. Every downstream figure moved with it. The
  pre-S1 numbers are superseded and archived, and the screening rule digest
  cannot return to its old value.
- **`reading_to_mw` is the only sanctioned conversion.** It calls
  `require_reading_unit()` first, refuses a unit it does not implement, and
  returns `None` — never zeros — for a record that is not a complete
  active-power day. It converts a unit and never rescales a curve.
- **`absolute_mw` is retired.** It multiplied a *normalised shape* by a rated
  capacity, which is a magnitude with no measurement behind it. It has no
  successor.
- **Residual limits, stated rather than hidden.** `运行容量`'s unit is as
  unstated as the reading's, so the capacity check is meaningful only under the
  same assumption; it is a consistency filter, never evidence for the unit. A
  data dictionary, a written statement from the producer, or an independent
  scale measurement would each let the ruling be replaced by a citation. See
  `reports/industry_load_unit_gate.md`.

## Not yet in force

Formal MATD3 training does not start from this record. Observation admission
is one gate among several: the load screening, the mapped magnitude, the reading
unit (D8, **open on a declared ruling**), the network feasibility review, the
settlement semantics, the frozen split and an end-to-end smoke all have to pass
first. Two further blockers are known and recorded in the S0 baseline: the real
price curves and the real load days cover no common date, and the absolute
industry load may exceed the feeder's physical capacity. Until every gate
passes, results quoted from an earlier revision are not evidence about the
current configuration.

## Verifying a claim made here

```bash
# Load screening: regenerates the quality report under reports/
PYTHONPATH=src python src/industry_load.py

# The screening rules, the price provenance and the declared scenario scales
python -m pytest tests/test_industry_load.py tests/test_dianyicha_price.py \
    tests/test_scenario_scale.py -v
```
