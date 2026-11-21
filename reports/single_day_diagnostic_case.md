# The single-day diagnostic case

One day of real metered industry load, composed onto the IEEE33 buses, paired
with one real day-ahead price curve. Both inputs are measured data. Nothing is
synthesised, copied, interpolated or rescaled, and the network, the bus mapping
and the facility configuration are the ones the rest of the study uses.

**The load day and the price day are different calendar days, and that is the
decision this case rests on.** So one thing must be said before anything else:

> **This is not a replay of a historical day.** It pairs a real load *shape*
> with a real price *shape*. It is not what 2022-11-30 actually met in its own
> market, and it is not what 2026-10-09's market would have paid for this load.
> No revenue, profit, arbitrage or welfare figure computed from this case
> describes anything that happened, and none may be quoted as though it did.

The convention that would have required the two dates to coincide was withdrawn
on 2026-10-15. What replaces it is stated below, together with what it costs.

**How the pairing is enforced.** This case is declared as the case type
`cross_date_diagnostic`, and every pairing decision — the date rule, each
series' own period axis, and the cross-curve alignment — lives in
`single_day_data.load_case` and is applied there. This script does not
re-implement any of it. That split is deliberate: an alignment rule with two
implementations is a rule that will disagree with itself. See
`reports/single_day_data_entry.md`.

## 1. The load day

**2022-11-30**, selected by the rule `earliest_complete_date`: the earliest date,
in calendar order, on which every mapped bus is covered for all 96 periods.

| | |
|---|---|
| rule | `earliest_complete_date` — `build_industry_node_loads.select_load_date` |
| selected | 2022-11-30 |
| buses covered | **17 of 17** mapped buses |
| **users with data** | **36**, named in the manifest and attributed bus by bus |
| readings | 85, of which **0** were not convertible |
| ledger rows | 3,168 (33 buses × 96 periods) |
| policies | `replace` 17, `preserve` 16 |
| excluded cells | 1 |
| rows with no industry data on a replace/overlay bus | 0 |
| ledger kind | **audit** |

**Every condition this date had to meet, checked against the ledger and not
merely asserted:**

| condition | how it was checked |
|---|---|
| every mapped bus has valid data | `load_complete` — 17/17 covered, from the audit |
| all 96 periods pass completeness on each | the same gate; a date is whole only when the coverage is complete, not when it is nearly so |
| no cross-date aggregation | `industry_curves` requires `md.date == date` **and** the qualified-block membership, so a meter with any block on the date cannot drag its history in |
| no double counting | identities are anonymised and a user's meters are summed into one unit **before** mapping; the manifest's `users_per_bus` is a partition — 36 users, 36 entries, none on two buses |
| no implicit filling | a period with no admissible reading is left empty and its reason recorded; the ledger has **0** empty `final_mw` and 1 excluded cell |
| the corrected coincident-power aggregation | a slot reading is the average active power over its 15 minutes, converted once (`× 综合倍率`, kW → MW) and **summed**; `industry_aggregation.slot_length_factor` is `null`, i.e. no `60/15` factor |
| bus 0 takes no industry user | `assert_classification_complete` raises if the slack carries one; in the ledger bus 0 is `preserve`, `not_applicable`, and all 96 of its `final_mw` are exactly 0.0 |
| one policy per bus, per scenario | 33 buses, each with exactly one of `preserve` / `replace` / `overlay`; no bus carries two |

**One cross-check the manifest now makes possible.** `buses_with_a_user` comes
from the *mapping file*; `buses_with_data_on_this_date` comes from the *day's own
data*. They agree exactly — same 17 buses, neither with a bus the other lacks —
and the 36 users are a partition of them. If they ever diverged, the manifest
would show it rather than the difference being absorbed into a total.

36 users is not a shortfall against the 43 the S1 rebuild counted: that is the
figure across the whole workbook, and a single date need not carry every user.

**Why this rule and not the other one.** The rule it replaces was "the date with
the most complete qualified meters, earliest wins ties". That is deterministic,
which is why it looked safe, and it is **biased**: a date with more reporting
meters carries more load, so the rule selected from the high end of the
distribution and the day was then described as representative. On bus 16 the
date it picked ranked 694th of 730. The new rule takes the earliest whole date
instead, so it is not a function of any quantity of load. A test asserts that
directly — given a whole date with *fewer* readings than a later whole date, it
still returns the earlier one.

**This date is not a typical day.** Nothing here estimates a distribution, and a
date chosen by its position on the calendar carries no claim about being average.
It is a diagnostic sample, and it is called one.

**Re-confirmed rather than taken on trust.** The audit was re-derived from the
workbook in full and is **byte-identical** to the committed
`reports/date_selection_audit.csv` (730 dates, 30 whole, 700 with at least one
bus missing). The date is then taken from that artifact, and the manifest
records the audit's own sha256 so the derivation can be repeated.

**The date it displaced is preserved, not lost.** The ledger path holds one file
per scenario, so building for 2022-11-30 overwrote the 2022-10-14 audit ledger
that S3 had kept as a historical sample. That pair was copied to
`data/processed/superseded/withdrawn_default_date_2022-10-14/` with a marker
before the rebuild; the copy is byte-identical to what was there.

## 2. The price curve

**2026-10-09**, sheet `261009` — the only curve in the workbook that names a
date. Read fresh for this case; every check below is one the reader enforces by
raising, and the record says which of them ran.

| | |
|---|---|
| date | **2026-10-09** |
| where the date comes from | the sheet name's own `26/10/09` label, **parsed by hand** |
| periods | 96 |
| interval | 15 minutes |
| label convention | **interval END** (`00:15` … `24:00`, ascending, no repeats) |
| units | `CNY/MWh` |
| semantics | a price per unit of **energy** for one 15-minute interval — not a power, not an energy |
| range | 225.9 – 445.0, mean 386.83 CNY/MWh |
| spatial statistic | `spatial_mean` — no single bus or zone is represented by it |
| temporal statistic | `single_day`, **by project ruling** |
| externally confirmed | **no** |
| timezone | **not stated in the workbook** |
| missing / duplicate / non-finite | none; a blank, non-numeric, non-finite or repeated value raises at read time |

Three weaknesses travel with the date rather than being flattened into it: it is
a **parsed label** and not a date field, the workbook states **no timezone**, and
the single-day reading is a **project ruling** the workbook cannot confirm
(`externally_confirmed: False`). The curve was checked and used as it is; a
synthetic provider was not substituted and is not an available fallback.

## 3. Time alignment

The two datasets describe the same 15-minute grid and **label their periods at
opposite ends of it**:

| | load | price |
|---|---|---|
| labels | `00:00` … `23:45` | `00:15` … `24:00` |
| convention | interval **start** | interval **end** |
| enforced by | `industry_load._check_time_header` | `dianyicha_price._check_slot_axis` |
| periods/day | 96 | 96 |

So period index `t` is the interval `[15t, 15(t+1))` on **both** sides, and
pairing them index to index is correct **without shifting either curve**. That
equality is asserted in `single_day_data.check_alignment` — the one place the
rule is implemented — and reached only through `load_case`, because reading one
convention as the other rotates every price by one period, by 15 minutes,
silently, and nothing downstream would look wrong.

What was **not** done: no conversion, no interpolation, no resampling, and no
date substitution. Each curve keeps its own date, its own units and its own
magnitudes. `period_alignment` raises if the two sides ever stop agreeing on
96 periods of 15 minutes.

## 4. The unit gate is unchanged

`READING_UNIT_EVIDENCE = "ruling_only"` and `unit_status() == "unit_unverified"`
stand. The date rule changed; the unit rule did not.

Consequences, all enforced in code rather than by convention:

- the ledger is an **audit** ledger. Every row carries the declared unit and its
  status, and the builder prints that it is not published as the network's
  formal absolute load;
- `--formal` is **refused**: `industry_load.require_verified_unit()` raises,
  naming the missing evidence;
- **no magnitude here may be called a verified absolute MW**, and this case may
  not be marked a formal physical network case.

Running the formal-ledger validator against the audit ledger, to see what it
says rather than to pass it:

```
$ PYTHONPATH=src python src/validate_formal_ledger.py \
      --ledger data/processed/node_loads/typical_day_industry/bus_loads_96x33_audit.csv \
      --case-type cross_date_diagnostic
bus_loads_96x33_audit.csv is an audit ledger: it is exempt from this validator
by design, and this validator is for the formal one.
REJECTED: 1 reason(s).
  [unit_not_verified] the reading unit is not stated by an external source
```

**The rejection is preserved**, and the unit reason is now the one that carries
it. The validator's date rule follows the **case type**:

| `--case-type` | reasons returned on this ledger |
|---|---|
| *omitted* | `unit_not_verified`, `case_type_unstated`, `price_not_date_aligned` |
| `cross_date_diagnostic` | `unit_not_verified` |
| `same_day_replay` | `unit_not_verified`, `price_not_date_aligned` |

So `price_not_date_aligned` no longer stands in the way of a cross-date
diagnostic, and there is no path by which leaving the case type out buys the
laxer rule — the omission is itself a violation. Neither the unit requirement
nor the verification status was relaxed. Whether the withdrawal should extend to
a *formal* case is a decision that has **not** been made, and `same_day_replay`
still enforces the strict rule.

## 5. What this case does and does not support

**Supports.** Examining a real load composition on a real network against a real
price shape: the bus-level ledger, the composition policies, the physical
feasibility of the load, and how a real price curve's daily shape interacts with
it. All of it as a **diagnostic**, on an audit ledger.

**Does not support.**

- any claim about 2022-11-30's actual market outcomes, or about what
  2026-10-09's market would have paid this load;
- any day-to-day, volatility or generalisation claim — one load day and one
  price day are not a series;
- any statement that this load day is typical, representative or average;
- a formal physical network case, while the reading unit is unverified.

**Not yet done, and not claimed.** The scenario builder does not read this
ledger: `scenarios.py` and `grid.py` have no reference to
`data/processed/node_loads/`. So the case is a set of verified data artifacts,
and **no market has been cleared on it**. Wiring the ledger into the scenario
build is its own step and is not part of this data decision.

## 6. The artifacts

| file | what it is |
|---|---|
| `data/processed/node_loads/typical_day_industry/bus_loads_96x33_audit.csv` | the 33 × 96 audit ledger |
| `data/processed/node_loads/typical_day_industry/manifest.json` | the ledger's own record, with the selection rule and its evidence |
| `data/processed/single_day_diagnostic/price_96.csv` | the price curve, 96 rows, both interval labels |
| `data/processed/single_day_diagnostic/price_manifest.json` | what the curve is and which checks it passed |
| `data/processed/single_day_diagnostic/case_manifest.json` | both days, both sources, the alignment, the fingerprints |
| `data/processed/superseded/withdrawn_default_date_2022-10-14/` | the ledger the old rule produced, kept |
| `reports/single_day_data_entry.md` | the entry point, the case types, the alignment rule and who is on it |

`case_manifest.json` carries every field this case turns on:

```
load_date            2022-11-30
price_date           2026-10-09
same_day             false
load_source          workbook, ledger, sha256, selection_rule,
                     selection_evidence, users_with_data, users_per_bus,
                     buses_with_data_on_this_date, industry_aggregation
price_source         workbook, sha256, file, sheet, how the date was derived
unit_status          unit_unverified   (reading_unit, reading_unit_evidence)
mapping_version      industry_bus_map_v2
load_policies        per bus, 33 entries
period_alignment     method, both conventions, and the three "none"s
input_fingerprints   five hashes, both workbooks and both derived files
integrity_checks     load coverage, users, exclusions and the aggregation;
                     the price's enforced checks
```

Reproduce with:

```
PYTHONPATH=src python src/build_industry_node_loads.py --audit-dates
PYTHONPATH=src python src/build_industry_node_loads.py --date-from-audit
PYTHONPATH=src python src/single_day_case.py
```

## 7. Cross-references

| subject | report |
|---|---|
| the withdrawn rule and the bias it carried | `reports/s3_date_selection_audit.md` |
| the unit ruling and the check that did not confirm it | `reports/industry_load_unit_gate.md` |
| what the formal case still needs | `reports/formal_data_blockers.md` |
| the composition policies and the ledger schema | `reports/s3_node_load_ledger.md` |
