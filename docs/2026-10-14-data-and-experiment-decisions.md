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

A block qualifies only if every row below passes. A failure excludes the block
and records the reason.

| Check | Requirement | On failure |
|---|---|---|
| Integrity | The reading and the identity fields the check needs are parseable | Exclude, record why |
| Ratio | Passes the existing `verify_ratios` rules, no unresolved ratio contradiction | Exclude; the ratio is never edited to make it agree |
| Capacity | Passes the existing `flag_records` rules; absolute power is consistent with the recorded meter capacity | Exclude; the capacity threshold is not relaxed |
| Raw values | Readings and metadata preserved as read | Never overwritten or silently corrected |
| Construction | Only fully qualified blocks are used, at their own magnitude | No normalization, no pinning to the IEEE33 peak |

Three riders, each closing a way this criterion could be gamed:

1. **The existing validators and their tolerances are the authority.** Where
   `flag_records` or `verify_ratios` does not state a rule for something the
   criterion needs, the rule is written down and tested before it is used. A
   new loose threshold invented to admit blocks is not a rule.
2. **The qualified count is an outcome, not a target.** 87 is the figure the
   earlier screening produced and is pending re-verification against the
   current validators. If re-running yields a different number, the difference
   and its cause are reported. Blocks are never dropped to reach 87, and
   unqualified blocks are never kept to reach it.
3. **A block is not a meter, a user, or a trading day.** "87 blocks" must not be
   restated as 87 meters, 87 users or 87 independent days.

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
