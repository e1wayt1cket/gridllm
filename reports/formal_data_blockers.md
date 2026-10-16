# What the formal case is still waiting for

The formal absolute-MW ledger, the S8 network validation and MATD3 training are
held by things no amount of code in this repository can settle. Each one is an
**external** material: a document, a dataset or a ruling that has to come from
outside, and until it does the work that depends on it is blocked rather than
approximated.

This report is the list. It is deliberately short on method and long on
evidence: for each blocker, what exists today, what is missing, and where the
gate that enforces it lives.

> **Update, 2026-10-15.** A **single-day diagnostic case** was adopted, and it
> pairs a real load day with a real price curve **from a different date** by
> decision. That resolves blocker 2 *for the diagnostic case* and for nothing
> else: the diagnostic ledger is an audit ledger, it may not be marked a formal
> physical network case, and the formal path is unchanged. Where this report
> says "blocked", read the scope column below. See
> `reports/single_day_diagnostic_case.md`.

> **Update, 2026-10-16.** Two items below that read "not started" were started,
> on `typical_day_industry_subset` — the audit ledger a user subset now carries
> — and **neither closes a blocker**: an **S8 physical audit**
> (`reports/s8_physical_audit.md`) and a **non-formal MATD3 smoke test**
> (`reports/s8_matd3_smoke.md`). The audit measured the subset day and reported
> it as an audit, with `unit_unverified` travelling on every figure; the smoke
> validated the training chain and produced a curve that is not a result.

> **Update, 2026-10-16 (later the same day). Blocker 1 is settled as a
> *basis*, not as a verification.** The project ruling of 2026-10-14 is now
> recorded as the formal modelling basis, so `industry_load` names what a unit
> rests on instead of confusing *a recorded ruling* with *nothing at all*:
>
> | basis | status | admits absolute MW | externally verified |
> |---|---|---|---|
> | `external_source` — a document outside this project | `unit_verified` | yes | **yes** |
> | `user_confirmed_project_ruling` — the project ruled it, and recorded it | `unit_basis_ruling` | yes | **no** |
> | anything else | `unit_unverified` | no | no |
>
> **This is a basis and it is not evidence**, and the report does not pretend
> otherwise: `is_externally_verified` is still `False` and every figure built on
> a ruling says so. What changed is that a decision someone made and wrote down
> is no longer treated as the same thing as a unit nothing stands behind. The
> gate still refuses a unit with no admissible basis, an uncited unit, and an
> artefact whose status does not follow from its own basis; the other nine
> validator checks are untouched.
>
> `typical_day_industry_subset`'s ledger was rebuilt as a **formal** ledger on
> that basis (`data/processed/node_loads/typical_day_industry_subset/
> bus_loads_96x33.csv`, `unit_status: unit_basis_ruling`), `validate_formal_ledger
> --case-type cross_date_diagnostic` now **PASSes** it, and the scenario was
> added to `formal_ensemble.scenarios`. `typical_day_industry` keeps its own
> membership and its own ledger, untouched. See §1 and §5 of this report.
>
> **Blocker 1 remains open in the sense that matters for a publication claim:**
> no external document states the unit, so no figure may be presented as
> externally certified. **The formal network validation now exists** for this
> one scenario, and its verdict is conditional on the project-assigned line
> ampacities — see §3 and `reports/s8_physical_audit.md`.
>
> **One artefact was left stale on purpose.**
> `reports/industry_user_subset_selection.md` was written before this change
> and still says the ledger is an audit ledger that `--formal` refuses. That
> sentence was true when it was written and is false now. It was not
> regenerated because regenerating it means re-running the subset search, which
> this round was told not to re-execute; the module's own text is corrected, so
> the next run of it will read correctly. **This report and
> `reports/s8_physical_audit.md` are the current record.**

## The three blockers

| # | blocker | blocks | status today | what it needs | enforced by |
|---|---|---|---|---|---|
| 1 | the reading unit | **an external certification claim only** — the ruling was adopted as the modelling basis on 2026-10-16 | `unit_basis_ruling` — a basis that is **recorded but not externally corroborated** | a written source stating the unit and the power/energy semantics, if a figure is ever to be called externally certified | `industry_load.require_unit_basis` (refuses no admissible basis), `single_day_data.require_formal_load_mw` (admits either admissible basis) |
| 2 | load and price dates do not overlap | **a `same_day_replay` only** — resolved for a `cross_date_diagnostic` by the 2026-10-15 decision | no date exists on which a real load day and a real price curve are both in hand | real historical day-ahead prices covering the load dates, **or** a decision that extends the withdrawal to same-day replays | `validate_formal_ledger` (`price_not_date_aligned`, when the case type is `same_day_replay` or unstated) |
| 3 | the formal date-selection rule | the formal case | **adopted 2026-10-15**: `earliest_complete_date`, which selects 2022-11-30 | nothing for the diagnostic case; a scope decision on whether a December-only period is a study period | `build_industry_node_loads.select_load_date` |

Blocker 1 blocks the **externally certified** claim and nothing else now. On
2026-10-15 the unit basis was accepted as a **user-confirmed project ruling**;
on 2026-10-16 that basis was made a first-class one, so
`require_formal_load_mw` and `validate_formal_ledger` admit it and a formal
ledger can be written on it. Every figure built on a ruling is labelled
`user_confirmed_project_ruling` with `is_externally_verified: false`, and none
may be presented as externally certified. What the gate still refuses is a unit
naming no admissible basis, a unit recorded without a source, and an artefact
whose status contradicts its own basis. Blocker 3 is answered for the diagnostic
case, which is why that case now exists. Blocker 2 is answered for a
`cross_date_diagnostic` by decision and stays open for a `same_day_replay`.

**What the diagnostic run found.** It was made, and it is reported in
`reports/single_day_diagnostic_run.md`: the clearing **solves** (`OPTIMAL`) and
serves **3.23%** of the load — 354.8 MWh of 10,995.6 — held by the lower
voltage bound (0.9300 pu) and the lines' thermal limit (1.0000). The metered
industry aggregate is 370.9–526.2 MW against a feeder whose own nominal demand
is 3.715 MW. Nothing was scaled to change that.

Which rule applies is not left to a caller's memory: the case type is an
argument, an unstated one is refused rather than defaulted to the lax rule, and
the same distinction is carried by `single_day_data.load_case` and by the
validator. See `reports/single_day_data_entry.md`.

## 1. The reading unit, and the two fields beside it

**What exists.** Every slot reading in `docs/16类行业用户.xlsx` is taken to be an
**average active power over its 15 minutes, in kW**, by project ruling of
2026-10-14. `reading_to_mw` converts under that declaration and every artefact
built from it carries `reading_unit` and `reading_unit_source` so the
assumption travels with the numbers.

**What is missing.** The workbook has **no data dictionary**: the header carries
twelve field names and 96 time columns and no units, and `docProps` has no
title, subject or description. The basis is the project ruling
(`READING_UNIT_EVIDENCE = "user_confirmed_project_ruling"`), so
`unit_status()` returns `unit_basis_ruling` and `is_externally_verified()`
returns `False`. There is still **no external document**, and the rest of this
section explains why a ruling is a basis and not evidence for the unit.

Strictly, three separate things need a written basis, and they are **not**
inferred from one another:

| field | what it is used for | what a source would have to state |
|---|---|---|
| the 96 slot readings | the shape and magnitude of every industry curve | the unit, and whether a value is a **mean power** over the interval or an **energy** accumulated in it |
| `运行容量` | the capacity each reading is tested against | the unit, and whether it is a nameplate, a contracted maximum, or a transformer rating |
| `综合倍率` | the multiplier applied to a reading | its definition, and what it multiplies — a current-transformer ratio, a scaling factor, or something else |

The cross-industry check the ruling asked for was run and **did not corroborate
it**: over the 101 mapped meters with a usable rating, 14 peak at 44–90% of
`运行容量` and 58 peak under 5%, with the field spanning 160 to 600,000. That
pattern is consistent with the field being a power rating for a minority of
meters and something else for the majority, so it cannot certify the unit.
The full result is in `reports/industry_load_unit_gate.md`.

**Asked for.** A meter specification, a utility data dictionary, a provider
response, or any document that states the three definitions above. One document
naming all three would open the gate; three separate ones would too.

## 2. A real price date that overlaps a real load date

**Resolved for the diagnostic case on 2026-10-15, and only there.** The
single-day case pairs **2022-11-30** (load) with **2026-10-09** (price), states
that the two are different days on its face, and forbids reading any figure from
it as what either day actually met. It is an **audit** ledger and may not be
marked a formal physical network case. What follows is the position for the
formal case, which this decision did not change.

**What exists.** Two datasets, and they do not meet.

| | load | price |
|---|---|---|
| source | `docs/16类行业用户.xlsx` | `docs/日前价格261009.xlsx` |
| dates | 2021-01-01 → 2022-12-31 (730 days) | see below |
| resolution | 96 periods/day | 96 periods/day |

The price workbook holds **four curves, and none of them is a series of daily
prices**:

| curve | `temporal_statistic` | window | dated? |
|---|---|---|---|
| `261009` | `single_day` | 1 day | **2026-10-09** — outside the load range |
| `7days` | `window_mean` | 7 days | `window_start`/`window_end` both `None` |
| `30days` | `window_mean` | 30 days | both `None` |
| `halfyear` | `window_mean` | 180 days | both `None` |

So the only curve that names a date names one **2026**, four years after the
last load day. The three others are trailing means whose end dates the workbook
does not state; inventing them would be worse than leaving them blank, which is
why they are.

`261009` is additionally `externally_confirmed: False`: its header reads
「日前均值（CNY/MWh）」 and the project reads that as a **spatial** mean over
buses rather than a temporal one, which the workbook does not settle.

**What is missing.** A series of real day-ahead prices for dates that fall
inside 2021-01-01 → 2022-12-31 — or a load dataset that covers 2026-10-09. With
only mean curves there is no cross-day price variation, so nothing that needs
two different days can be done: no training across days, no model selection, no
generalisation claim.

**Asked for.** Dated day-ahead prices for the load period. Hourly or 15-minute,
CNY/MWh, province or node, with the dates stated — **for the formal case**. The
diagnostic case no longer waits on this; it waits on blocker 1 like everything
else.

## 3. The dates where the load and the coverage actually meet

This is the third blocker's raw material, and it is worth seeing before the
rule that would select from it.

| | |
|---|---|
| dates with any load reading | 730 (2021-01-01 → 2022-12-31) |
| buses carrying a mapped user | 17 |
| **dates on which every one of the 17 has data** | **30 (4.1%)** |
| where those 30 fall | all in December 2022, spanning 2022-11-30 → 2022-12-31 |
| dates with a real *and* dated price curve | **0** |

So the full-coverage set is a single month at the end of the record, and even
that month has no price to go with it. `reports/s3_date_selection_audit.md`
carries the per-bus coverage; `reports/s3_node_load_ledger.md` carries what the
composition looks like on an audit date and why a high-coverage day is not
representative (bus 16 reads 208 MW against a 730-day average of 40 MW).

**What is missing.** Either more load coverage — a wider set of dates on which
all 17 buses report — or a decision to state the study on the 30 dates that
exist while saying plainly that they are one month. Both are legitimate; the
second is a scope decision, not a data one.

**Asked for.** The date list above, confirmed against the provider's own
coverage, and a decision on whether a December-only study period is acceptable.

## 4. A date-selection rule, declared before it is used

**Answered on 2026-10-15.** The rule in force is `earliest_complete_date`: the
earliest date, in calendar order, on which every mapped bus is covered for all
96 periods. It satisfies the three properties this section asked for — its
inputs are coverage counts that exist before any economic result, it is
deterministic given them, and a reader can recompute it from
`reports/date_selection_audit.csv` — and it is **not a function of any quantity
of load**, which was the defect of the rule it replaced.
`build_industry_node_loads.select_load_date` implements it, `--date-from-audit`
applies it, and the selected date's evidence travels in the ledger's manifest.

| | |
|---|---|
| rule | `earliest_complete_date` |
| selects | 2022-11-30 |
| whole dates available | 30, all in December 2022 |
| what it is not | a typical day, a representative day, or a load percentile |

**Still open.** Whether the 30 whole dates — one month at the end of the record
— are a study period anybody wants. The single-day case does not answer that; it
sidesteps it by being one diagnostic day rather than a period. A multi-day
formal study would have to answer it.

The withdrawn rule and the bias behind it are in
`reports/s3_date_selection_audit.md`, and `pick_date` is kept so the audit of
that bias stays reproducible.

## What may not happen until these clear

Each of these is refused by code today, not by convention:

- **a formal absolute-MW ledger may now be frozen on a recorded ruling**, and
  one has been: `typical_day_industry_subset`'s, whose every row carries the
  unit, the basis and `is_externally_verified: false`. `require_unit_basis()`
  still refuses to write one when no admissible basis is recorded.
  `typical_day_industry`'s ledger is unchanged and still an **audit** ledger:
  its manifest records the pre-2026-10-16 spelling, `validate_formal_ledger`
  refuses it on that basis, and it was deliberately not rebuilt — rebuilding it
  would move the digest pinned by the single-day diagnostic case.
- **no figure from the diagnostic run may be called verified.** The run is
  labelled with its basis; a reader that drops the label has turned a declared
  unit into an asserted one.
- **no scenario may be declared formal by omission.** `scenarios.formal_scenarios()`
  checks each member's price source against the readiness rule and raises if the
  set is narrower than the rule allows, so `typical_day_industry` — the one
  scenario whose source is a measured market — is still blocked by blocker 1
  and, for a formal claim, blocker 2.
- **the single-day case is not a formal case.** It is an audit ledger built on a
  declared unit; `--formal` raises, and the ledger carries `unit_unverified` on
  every row.
- **The formal S8 network validation is still not started, and cannot be.** It
  is a statement about a formal load, and there is not one. An **audit** of the
  subset day was run on 2026-10-16 (`reports/s8_physical_audit.md`); it carries
  `unit_unverified` on its face, its line ampacities are project-assigned, and
  it is not the formal validation.
- **No formal MATD3 run has been started.** Four mean price curves cannot
  support cross-day training, model selection or a generalisation claim, and
  repeating a single day to look like many would dress a single-day fit as a
  multi-day result. A **non-formal smoke** was run on 2026-10-16 to validate the
  training chain — 25 episodes, one seed, one day, five gradient steps, one of
  them on a structurally idle fleet — and it is recorded in
  `reports/s8_matd3_smoke.md`. It is not a result. It is also why
  `train_rl.py` now carries an explicit `--allow-non-formal` opt-in: the gate
  that refuses a non-formal `--scenarios` name used to have no way through, and
  a way through that is not written down is a way through nobody can audit.

The conditions are not advisory. They are the reason
`reports/s0_code_and_data_baseline.md` through `reports/s7_defect_repairs.md`
record fixes to the *machinery* while the formal case stays open: the machinery
can be made correct without the data, and has been. The data cannot be made
correct by the machinery.

## Cross-references

| subject | report |
|---|---|
| the unit ruling and the check that did not confirm it | `reports/industry_load_unit_gate.md` |
| the load rebuild and its coverage | `reports/industry_load_rebuild.md` |
| the date audit, the withdrawn rule and the adopted one | `reports/s3_date_selection_audit.md` |
| the single-day diagnostic case, both days and the alignment | `reports/single_day_diagnostic_case.md` |
| the data entry point, the case types and who is on it | `reports/single_day_data_entry.md` |
| the run itself: solver status, network state, PCC, fallbacks | `reports/single_day_diagnostic_run.md` |
| the audit ledger and the bus composition | `reports/s3_node_load_ledger.md` |
| the price curves and their four recorded facts | `reports/phase2_dianyicha_price_report.md` |
| which scenarios may enter a formal evaluation, and why | `reports/s4_40_day_removal_audit.md` |
