# Data and experiment decisions

The load, price, network and accounting rulings this project runs on, written
down once so a later reader can tell a decision from a measurement. Verdicts
that supersede an earlier document are listed first, because the superseded
documents are still in the tree and still read as authoritative.

## What this supersedes

| Earlier ruling | Where it lives | Status |
|---|---|---|
| Load peak pinned to the IEEE33 nominal 3.715 MW (`load_scale_mode: native_peak`) | `config/scenarios.yaml` (`typical_day`, `typical_day_industry`), `docs/2026-09-18-scenario-revision.md`, `grid.py` `_coincident_peak` | **Replaced** by the absolute industry load below |
| Absolute MW refused outright, module builds normalized shapes only | `industry_load.py` module docstring, `absolute_mw()` | **Amended**: shapes stay as they are; the absolute path is now required and must carry the screening record below |
| Train / validation / test sized at 24 / 6 / 6 days | `b/算例设计与算力预算指导.md` | **Not adopted**: candidate scenarios are generated and inspected before any size is frozen |
| Paper text stating a 3.715 MW / 2.3 MVAr base case | `b/01_SCI_IJEPES_中文稿.md` | Stale; the paper text follows this record, not the reverse |

## D1. Load

- The metered Jiangsu workbook is kept byte-for-byte. Its readings, identities
  and metadata are never rewritten, and no exclusion is applied silently.
- The formal absolute load uses **only the blocks that pass the screening in D5**.
- That load is **not normalized**, and it is **not rescaled onto the IEEE33
  nominal peak**. Whatever magnitude the qualified blocks carry is the magnitude
  the study runs at.
- Qualified blocks are aggregated onto the 32 load buses of the feeder by an
  explicit, reproducible rule. The assignment is a **simulation mapping**: this
  feeder carries no user-to-node connection table, and no result may describe
  the mapping as real wiring.
- Every block enters at most once. Double counting is a defect, not a tolerance.

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
- Observation: `OBS_V4`, 12 dimensions. It is causal by contract — every price
  feature is a function of prices already published to the participant, and the
  decision-block load, RE and system ratio come from the day-ahead forecasts in
  **both** stages. A V3 checkpoint is not a valid V4 initialisation and is
  refused rather than remapped.
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
| meters whose implied peak fits 运行容量 | **87** |
| meters whose implied peak contradicts it | 113 |
| meters the check cannot reach | 83 |
| meters with at least one qualifying block | 84 |
| **formal absolute-load blocks (B)** | **39,365** |
| users covered | 36 |
| dates covered | 730 |

**87 is a meter count. 39,365 is a block count.** The earlier "87 qualified
blocks" read the first as the second and is withdrawn. The meter count is 84
rather than 87 because three of the 87 consistent meters produce no block that
survives the pool. Reading A (87,511 blocks) and reading C (35,383) are recorded
as bounds and are not the formal input.

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

## Not yet in force

Formal MATD3 training does not start from this record. Observation admission
is one gate among several: the load screening, the mapped magnitude, the
network feasibility review, the settlement semantics, the frozen split and an
end-to-end smoke all have to pass first. Until then, results quoted from an
earlier revision are not evidence about the current configuration.

## Verifying a claim made here

```bash
# Load screening: regenerates the quality report under reports/
PYTHONPATH=src python src/industry_load.py

# The screening rules, the price provenance and the declared scenario scales
python -m pytest tests/test_industry_load.py tests/test_dianyicha_price.py \
    tests/test_scenario_scale.py -v
```
