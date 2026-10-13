# Final Research Summary

The consolidated account of the storage-bidding study: what was asked, what was
built, what was measured, what the measurements are worth, and what may not be
claimed from them.

This document is the entry point. The phase reports it draws on are listed in
§11.

---

## 1. The question, the approach and the scope

**The question.** Can an independent storage fleet bid better than truthful
declaration in a distribution network, under real Jiangsu day-ahead prices and
real Jiangsu industry load shapes, evaluated as money rather than as a reward?

**The approach.** An agent-based market on the IEEE 33-bus radial distribution
network. 96 periods at 15 minutes. Load comes from metered Jiangsu industry
curves; the external price comes from Jiangsu day-ahead curves; the clearing is
a network-constrained SOCP that recovers nodal prices from its own
power-balance duals, and participants are settled at those nodal prices.

**In scope.** Day-ahead energy only. Four independent storage units comparing
truthful declaration, a rule-based look-ahead policy, and a per-period threshold
rule. One reference bound. Layout, capacity, siting and price-scenario
sensitivity.

**Out of scope, deliberately.** Real-time settlement and deviation charges.
Ancillary services, frequency regulation, reserve and capacity compensation.
Medium- and long-term contracts. A full provincial SCUC/SCED and zonal price
formation. **MATD3 training** — see §7.

---

## 2. Data: sources, statistical basis and quality

### 2.1 Day-ahead price

**File.** `docs/日前价格261009.xlsx`, SHA-256
`5fc8f9a8461b680fd38a6d2fefb3d2ff99f239c8bd13a6c77df27de3bb9612f3`.
Read by `src/dianyicha_price.py` with `openpyxl` in read-only mode. Four sheets,
96 points each, CNY/MWh, interval-END labels (`00:15` … `24:00`) mapping
positionally onto periods with no shift.

**Provenance, as far as it can be established.**

Four questions, answered separately — and they have to be, because one axis
cannot carry them: a curve is routinely both spatially averaged *and* a single
day, which is exactly the case here and exactly what a single "aggregation"
axis could not express.

| sheet | source | spatial | temporal | window | role |
|---|---|---|---|---|---|
| `261009` | real measured market data | `spatial_mean` | **`single_day`** | 1 day | `single_day_spatial_mean` |
| `7days` | real measured market data | `spatial_mean` | `window_mean` | 7 days | `trailing_7_day_mean` |
| `30days` | real measured market data | `spatial_mean` | `window_mean` | 30 days | `trailing_30_day_mean` |
| `halfyear` | real measured market data | `spatial_mean` | `window_mean` | ~180 days | `trailing_180_day_mean` |

Statistics: `261009` 225.90 / 445.00 / 386.83 (min / max / mean); `7days`
257.50 / 477.87 / 395.55; `30days` 333.08 / 450.86 / 407.82; `halfyear`
284.93 / 393.86 / 346.33.

**All four are `spatial_mean` over real measured market data.** Each 96-point
series is an average across locations, so **none of them represents a bus, a
zone or a settlement point**, and no result here may be read as a locational
price. What separates the four curves is the **temporal** statistic alone.

**The spatial extent, the weighting and the number of points behind each
average are not recorded in the workbook** and are left unverified rather than
assumed — `spatial_reference` is `None` on every curve.

**`261009`'s single-day reading is a project ruling, not an external
confirmation.** Its sheet name carries the date 26/10/09; its column header says
`日前均值（CNY/MWh）`, "day-ahead mean". Read as a *temporal* mean the header
would make the sheet a day-mean; the project reads that 均值 as a *spatial* mean
and the named date as the day the curve covers, so `261009` is recorded as
`temporal_statistic: single_day`. The workbook cannot settle it: its `docProps`
record a personal author and a creation date (2023-05-12) well before the
sheet's named date, which establishes it as a **hand-assembled extraction**
rather than a platform export, and is in turn why no window dates are present.
`externally_confirmed` is therefore `False` on `261009` and `declared_by` names
the ruling. Settling it would need the provider's own definition of the column.

Four further readings of the file were taken. They are consistent with the
ruling but **none of them confirms it**, which is why it is recorded as a ruling:

- **Its period column names the date.** `261009`'s column A reads
  `时段（26/10/09）`, "periods (26/10/09)"; the other three read only `时段`.
  Naming a date is what a single day's series does — but the same sheet's
  column B calls itself a mean, so the two halves of one header row disagree.
- **Its values are less precise.** `261009`'s 96 values carry **at most one
  decimal place**; the three trailing means carry up to three. A published
  period price to 0.1 and a window mean to 0.001 is exactly the pattern a
  day-plus-trailing-means export would show — and a mean *rounded* to 0.1
  would look identical, which is why this is suggestive and not decisive.
- **Its sheet is padded, the others are not.** `261009` carries 175 trailing
  blank rows (272 rows against the others' 97). Formatting, not content.
- **It was re-saved later than it was created.** `docProps` record a creation
  date of 2023-05-12 and a modification date of 2026-10-09 — the latter being
  the sheet's own named date, which is consistent with a paste into an older
  template and equally consistent with an unrelated re-save.

**No cell comment, no second sheet, no defined name, no VBA and no platform
export metadata exist to break the tie** — verified against the workbook's own
parts, not inferred from the visible sheets.

**What this does and does not license.** `temporal_statistic` is the **sole
authority** for single-day identity, and `describe()` publishes it per curve
alongside `spatial_statistic`, `window_days`, `externally_confirmed` and
`declared_by`. The retired `is_measured` flag is **deprecated** and now means
"the underlying source is real measured market data" — true for all four curves,
so it must not be read as a single-day claim; its semantics travel beside it in
the payload for one compatibility window.

`contains_single_day_curve` is `True` and `contains_multi_day_series` is
`False`. Those are different questions and the retired single flag could not
hold both: **one day plus three window means is still not a series of daily
prices**, so the binding limitation stands unchanged — no cross-day statistic, no
volatility, no generalisation claim, and no train/validation/test split over
real prices.

**Unverified.** The trailing windows' **end dates**. The workbook does not state
them; they are recorded as `None` rather than invented. Until a platform record
confirms them, these curves may not be called strictly non-anticipative
historical information, and what they are at the platform (province-wide
average / zonal / a re-processed series) is likewise open.

### 2.2 Industry load

**File.** `docs/16类行业用户.xlsx`, 117,271 rows × 108 columns, 73 users, 16
industries, 283 meter pairs, 2021-01-01 → 2022-12-31. Processed by
`src/industry_load.py`.

**What is produced.** Normalised 96-slot shapes per industry, and **nothing
else**. No absolute-MW load is produced or claimed.

**Measured defects**, reproduced by the pipeline rather than hardcoded: 43
placeholder rows (the `[1,2,3,3,3,0,…]` signature, sum exactly 12), 4,609 rows
with missing slots, 81 fully empty rows, 81 rows with a non-active power phase,
15,482 all-zero days, 13,528 duplicate rows (4 conflicting), **2,683 negative
values on non-PV industries**, 7,856 on 分布式光伏. Nothing is imputed,
clamped, sign-flipped or repaired; every departure is flagged and carried to a
report.

**The meter multiplier is unreliable.** Of 200 checkable meters, the implied
peak fits the declared 运行容量 for **87** and contradicts it for **113**; the
worst implies 4,594× the rating. `industry_load.absolute_mw()` therefore
**refuses by default**. The metre-level shape is unaffected — normalisation
divides by each meter's own peak, so the multiplier cancels exactly — which is
why shapes are usable and absolute MW is not.

**Reliability.** 14 of 16 industries clear the sample threshold. `5G基站`
(2 users) and `电解铝` (1 user, 208 days) are marked `limited` and excluded from
the benchmark mapping. `分布式光伏` is a generator, excluded by name so it
cannot re-enter as a load.

**Unverified.** Whether the surviving clean days represent a year; the cause of
the 2,683 non-PV negatives.

---

## 3. The scenario and the facilities

`typical_day_industry` — the IEEE 33-bus network at its **own** demand.

| | |
|---|---|
| buses | 33; slack at 0; 32 load buses (1–32) |
| coincident peak | **3.715 MW**, the network's own nominal demand |
| PV / wind | 0.6 MW (buses 21, 23) / 1.0 MW (bus 30) — unchanged from the benchmark |
| storage fleet | 4 × (0.4 MW / 1.6 MWh), buses **9, 14, 17, 27** |
| efficiency | `eta_ch` = `eta_dis` = 0.92 |
| SOC | band 0.10–0.90, start 0.50, terminal pinned to start |
| `cycle_cost` | 100 CNY/MWh **per leg** |
| voltage band | 0.93 – 1.05 pu (GB/T 12325 ±7%) |

**The load level is the network's own.** The industry data contributes the
*intraday shape* and never the magnitude: shapes are normalised to peak 1.0
before use, and a test multiplies every shape by 137 and asserts the day is
unchanged. So this day is **IEEE 33-bus demand redistributed by Jiangsu
industry shapes** — not measured Jiangsu load.

**The bus ↔ industry assignment is a simulation rule**, declared explicitly in
`config/scenarios.yaml` with `simulation_mapping: true`. This feeder carries no
user-to-node connection table. The rule: every residential and commercial bus
takes 商业综合体 (the workbook has no residential class); each industrial bus
takes one distinct industrial industry. 13 reliable industries are used.

**Siting.** Buses 9/14/17/27 were fixed before this work, selected at the native
scale by a voltage-sensitivity screen whose criterion is the share of the quote
box a unit's round-trip margin leaves tradeable. Phase 9 re-examined siting and
found it does not change the conclusion — see §6.

---

## 4. Computational conventions

**Quoting.** Storage declares two reservation prices in CNY/MWh —
`charge_price ∈ [0, 150]`, `discharge_price ∈ [0, 300]` — truthful at **(0, 0)**.
A unit asks *little* for the leg it wants and *much* for the leg it does not.
Non-storage participants declare a bid multiplier and an offer adder, truthful
at (1, 0). `action_spec_for` selects by `participant_type`, never by whether the
agent carries a battery. **A strategy declares; the clearing decides every
traded quantity.** No strategy writes `p_ch`, `p_dis` or `soc`.

**Clearing.** SOCP over the radial network. Voltage, thermal and
power-balance constraints. Charge and discharge are the parts of one net power
flow, so overlap is unrepresentable. Nodal prices come from the power-balance
duals (`BarHomogeneous=1`; without it the plain barrier returns OPTIMAL while
leaving duals unavailable on some realizations). The result carries a `solver`
block: engine, primary status code and name, termination class, fallback status,
and the **final** solver — never the primary's status after a fallback — with
`duals_available` reported as a question separate from `optimality`.

**State of charge.** `soc[t+1] = soc[t] + (eta_ch·ch[t] − dis[t]/eta_dis)·dt/e_max`,
`dt = 0.25 h`. Results carry `soc_boundary` of length `T+1`, so **all 96
transitions are verifiable**; `soc_terminal_target` and `soc_terminal_actual`
are stored separately so a pin cannot masquerade as a result.

**Settlement.** The research KPI is the **settled day-end net revenue**: the
whole horizon re-cleared at once with both SOC endpoints pinned, priced at the
nodal LMP, from the quotes actually declared (`counterfactual_eval.settle_day`).
The evaluation CLI and the programmatic evaluator call the same function. The
rolling training reward is reported under its own column names and **is not a
payoff** — on the native day it runs 2–5× the settled figure and **ranks the
units in the opposite order**, which is why the two were separated.

**What the money does not include.** `PayoffBreakdown.other_cost` has zero
assignments in the source and is always 0.0; **no cost was invented to fill
it**. The accounting covers revenue, purchase, generation, degradation and
penalty, and **does not** cover 输配电价, 线损, 系统运行费, 政府性基金及附加,
deviation charges, capacity or cost compensation, or imbalance funds. The only
permitted name for a result is **modelled storage day-ahead energy net revenue
within the model's boundary**.

---

## 5. The break-even condition, derived

Taken from the clearing's own objective, per MWh **stored**.

Put one MWh into the battery. At charging efficiency `eta_ch` the grid must
supply `1/eta_ch` MWh, costing `p_c/eta_ch`. Discharging returns `eta_dis` MWh
at `p_d`, earning `eta_dis·p_d`. `cycle_cost` is charged on **both** legs, and
the legs are not equal in energy: `1/eta_ch` in, `eta_dis` out. So

```
profit per MWh stored = eta_dis·p_d − p_c/eta_ch − cycle_cost·(1/eta_ch + eta_dis)
```

and the cycle is worth running only if

```
eta_dis·p_d − p_c/eta_ch  >  cycle_cost · (1/eta_ch + eta_dis)
```

**The right-hand side is `arbitrage_lp.required_margin`. At the configured
values it is 200.6957 CNY/MWh — not 200.**

The familiar `2 · cycle_cost` is the **lossless-round-trip special case**. It
appears throughout the earlier reports and was used as the threshold until it
was corrected; at `eta = 0.92` and `cycle_cost = 100` it is **0.6957 CNY/MWh too
permissive**. Small, but it is the entire margin on a day that sits near the
threshold, so it is not left as an approximation.

**The formula is checked against the optimiser, not asserted.** With the two
prices chosen so the left-hand side equals the derived value exactly, the LP
returns an optimum of zero and is indifferent about charging; half a CNY/MWh
above it, the optimum turns positive. `test_the_required_margin_is_what_the_lp_actually_breaks_even_on`
pins that.

**Three distinct quantities, kept apart throughout.**

| quantity | what it is | contains |
|---|---|---|
| theoretical price bound | `eta_dis·max − min/eta_ch` | the day's extremes; no power or energy limit |
| LP capturable margin | the LP optimum | power limit, energy limit, SOC band, terminal pin, `cycle_cost`, efficiency |
| settled net revenue | what a strategy actually earned | the above **through the clearing**: network constraints, quotes, competitors, nodal settlement |

None of them bounds the next one in general. The theoretical bound ignores the
limits; the LP ignores the market; the settlement is what the market paid.

---

## 6. What was measured

### 6.1 The arbitrage window is shut

| curve | min | max | round-trip margin | required | clearing? |
|---|---|---|---|---|---|
| `261009` | 225.90 | 445.00 | 163.86 | 200.70 | **no** |
| `7days` | 257.50 | 477.87 | 159.75 | 200.70 | **no** |
| `30days` | 333.08 | 450.86 | 52.75 | 200.70 | **no** |
| `halfyear` | 284.93 | 393.86 | 52.64 | 200.70 | **no** |

**The bound is a control-checked zero**: the same LP returns **344.29 CNY** on a
synthetic 120→1000 band and **30.85 CNY** on a flat-500 curve with a 200 trough,
so the zero is economics rather than a broken solver.

**Nodal prices do not open it either.** Clearing the day and measuring every bus
gives a best nodal margin of 173.74 (`261009`), 169.56 (`7days`), 52.75 and
52.64 — **0 of 33 buses clear on any curve**. The deepest-feeder premium is
real (bus 17 reaches 468.07 against a curve maximum of 445.00) and is still not
enough.

### 6.2 The three strategies

| curve | strategy | fleet settled profit | declared intent (chg/dis/neu) | traded |
|---|---|---|---|---|
| `261009` | `fixed` | −8.19e-05 | 0 / 0 / 96 | 1.9e-07 MWh |
| `261009` | `mpc` | −1.05e-04 | 7 / 1 / 88 | 2.4e-07 MWh |
| `261009` | `greedy_threshold` | −2.54e-04 | 21 / 63 / 12 | 7.4e-07 MWh |
| `7days` | `greedy_threshold` | −1.60e-04 | 16 / 64 / 16 | 3.5e-07 MWh |
| `30days` | `greedy_threshold` | −7.90e-05 | 0 / 63 / 33 | 9.5e-08 MWh |
| `halfyear` | `greedy_threshold` | −1.66e-04 | 18 / 0 / 78 | 2.0e-07 MWh |

**Those profit figures are not economic quantities.** Throughput is ~1e-7 MWh —
about **1.6e-8 of one fleet cycle** (6.4 MWh). Every figure is the degradation
cost of solver noise, and the ~1e-4 differences between strategies are float
residue. **No strategy earns anything and no ranking may be read from them.**

The strategies differ in **declared intent**, and that is the real observation:
`fixed` never asks for anything; `mpc`'s look-ahead LP finds almost nothing
(and **agrees** with the independent perfect-foresight LP — two methods, one
conclusion); `greedy_threshold` declares heavily and **is filled in none**. The
threshold rule is not idle; the market is closed to it.

A structural caution: the same fixed 300/400 thresholds mean different things
across curves. On `30days` (floor 333) the rule can only ever want to discharge;
on `halfyear` (ceiling 394) only ever to charge. The thresholds were **not**
re-tuned per curve.

### 6.3 Load shape and configuration

`typical_day` vs `typical_day_industry` — **confounded**, because the two
scenarios differ in load shape *and* price source. Cleanly attributable:
coincident peak 3.7150 MW both (by contract), daily energy 47.04 → 54.79 MWh,
load factor 0.5276 → 0.6145, peak period 18:30 → 16:15. The industry day
carries **more** energy at the same peak, because its shapes are flatter than
the synthesised native profile.

A knife edge worth recording: on the **native** day three of four configured
buses fall within 0.7 CNY/MWh of break-even and the fourth within 2.3, and bus 14
nominally clears by **0.0001** CNY/MWh. The fleet moves 0.0405 MWh there — 5
orders of magnitude more than the industry day — and the LP still returns
exactly zero, because a margin of 0.0001 is not capturable by a battery that
cannot be at the trough and the peak at once. The four margins are read from
`outputs/phase9_sensitivity.json` (`load_shape`); see §5 for the revision that
settled bus 14's figure.

**Capacity and siting cannot open the window.** The theoretical bound does not
contain the capacity at all: it is **identical to six decimal places** from
0.2 MW / 1.6 MWh to 10 MW / 100 MWh. And every one of five placements, including
the two the project previously used and two chosen for the deepest-feeder
premium, leaves its own buses short. What capacity and siting *do* vary is the
physical envelope — and none of the configurations tested violated a constraint
or failed an SOC check.

---

## 7. MATD3

**No MATD3 performance claim of any kind is available, and none is made.**

An earlier version of this section said that no ESS checkpoint existed at all.
**That was wrong**, and the correction changes what the next step is.

**Contract-compatible checkpoints do exist.** Seven of the 45 directories under
`policies/` carry checkpoints for the ESS fleet: `formal_v0_1_seed{1..5}`
(`ESS*_ckpt_{50..450}.pt` plus `best/ESS*.pt`) and `protocol_smoke_v0_1` /
`protocol_smoke_v0_1_lambda010` (final, `best/` and `last/`). Every one loads
with `obs_spec: v3_12d`, `action_spec: storage_quote_v2`, `charge_price
[0, 150]`, `discharge_price [0, 300]` and truthful `(0, 0)` — the **current**
contract, not a retired one. Reading `is_measured` or grepping directory
*names* for "ess" both miss them; the files are named after the agents.

**Contract compatibility is not training completion, and is not evidence of
performance.** `formal_v0_1_seed{1..5}` was launched 2026-09-22 13:11 at 800
episodes per seed (λ = 0.10, `--day-rotation on`, `typical_day`) and **stopped
at episode 450/800**. Its `logs/formal_v0_1_provenance.tsv` carries only a header
row, none of its five run directories was materialised, `formal_v0_1_chain.log`
holds `START` with no `END`, and no seed wrote a final or `last/` policy. The
two `protocol_smoke_v0_1*` runs did complete, but at 40-episode smoke scale, and
the first used the retired λ = 0.25.

**So no completed formal MATD3 run exists** — which is the state this section
should have reported. What the surviving checkpoints are actually worth is an
open measurement requiring a paired settlement against the truthful baseline;
that diagnosis has not been run. They are **not** deleted, overwritten or
resumed, and nothing here asserts they are usable or unusable.

The reason no formal training is appropriate yet is unchanged, and is a data
limit rather than a modelling choice: one real day plus three window means
(§2.1) is not a measured daily series, so there is no basis for a
train / validation / test split over real prices, and repeating one day's
episodes would dress single-day fitting as generalisation.

---

## 8. Experiments run, and not run

**Run.**

| experiment | scope |
|---|---|
| industry load ingestion and quality | full workbook, 117,271 rows |
| price curve reading and provenance | all four sheets |
| arbitrage reference | 4 curves, exogenous and per-bus nodal |
| three-strategy paired comparison | 3 strategies × 4 curves |
| load-shape comparison | `typical_day` vs `typical_day_industry` |
| capacity | 6 configurations, full clears |
| siting | 5 placements, full clears |
| single-day end-to-end acceptance | curves, clearing, SOC, settlement, repeatability |

**Not run, with reasons.**

| not run | reason |
|---|---|
| capacity × siting × curve grid | the bound clears at no bus on any curve for any size, so every cell reproduces an idle clearing |
| full capacity/siting sweeps on `7days`/`30days`/`halfyear` | `261009` has the **widest** spread and highest nodal margin — the most favourable case; the other three can only be worse |
| MATD3 training | §7 |
| real-time / ancillary / capacity markets | declared out of scope |
| absolute-MW load scenarios | the conversion is unverified; `absolute_mw()` refuses by design |

The screen that justifies the small experiment is itself tested: if the bound
ever cleared at some bus, a test fails and names the report to revisit.

---

## 9. Reproduction

```bash
# environment: a Python with requirements.txt and a Gurobi licence.
# `requirements.txt` now carries `openpyxl`, which both workbook readers need
# and which it previously omitted -- without it the two data commands below
# fail and 25 of the price tests error.
export PYTHONPATH=src                     # Git Bash on Windows

# data
python src/industry_load.py                       # industry shapes + quality report
python src/dianyicha_price.py                     # price curves and their provenance

# reference bound and the three-strategy comparison
python src/phase7_strategy_comparison.py

# single-day end-to-end acceptance
python src/accept_typical_day_industry.py --json outputs/phase6_acceptance.json

# sensitivity
python src/phase9_sensitivity.py                  # screen first, then what it leaves open
python src/phase9_sensitivity.py --screen-only    # the screen alone

# The per-bus margin table on the native day (the knife edge in section 6.3)
# comes from the phase-9 run above, under `load_shape` -- not from
# `diagnose_typical_day.py`, whose own defaults clear a different day and
# report different margins.

# tests
python -m pytest tests/ -q
```

Key artefacts: `outputs/phase7_strategy_comparison.json`,
`outputs/phase9_sensitivity.json`, `outputs/phase6_acceptance.json`,
`data/processed/industry_typical_profiles.csv` (version-controlled; the 67 MiB
source workbook is not).

## 10. Tests

**The suite passes, 0 failures: `785 passed, 51 deselected` in 244 s**, run on
the environment §9 names with `python -m pytest tests/ -q` (`pytest.ini`
deselects the `slow` marker). That is a measured count, not a carried-forward
one — the figure this section previously quoted (783) could not be reproduced
as it stood: in the environment found at the time of this revision, `openpyxl`
was absent, so **25 of the price tests errored** and no green suite existed to
confirm it. Whether that environment is the one the 783 was taken on is not
established. **A suite count is only worth what the environment behind it is
worth**, so the missing dependency is now declared in `requirements.txt` and
the count above is the one this revision ran.

The suite is the machine-checkable part of this study. What it holds:

- the load peak is the network's own, and shapes carry no magnitude;
- both price- and load-slot axes are exactly 96 and unshifted;
- the settlement is one function, and both entry points agree;
- **every SOC transition verifies** against the `T+1` boundary array, on a day
  whose fleet actually cycles;
- the arbitrage window is asserted as an **inequality**, so a future series that
  reopened it would fail the test rather than pass against a stale expectation;
- the LP's zero is accompanied by two controls that are not zero;
- the break-even is what the LP breaks even on;
- caches are keyed by configuration, fleet and spec, and do not leak;
- price curves are selected explicitly and do not come from a stale global;
- **no curve is promoted to `is_measured`**, the trailing windows' end dates
  stay `None` rather than being invented, and the header is carried verbatim
  beside its classification, so the price sheet's ambiguity (§2.1) cannot be
  silently resolved into a measured day by a later edit;
- the knife edge in §6.3 is re-derived from each configured bus's own nodal
  prices, so the table cannot drift from the clearing it describes.

---

## 11. Source documents and change record

| document | content |
|---|---|
| `reports/finalization_audit.md` | code inventory; the three accounting defects |
| `reports/industry_load_quality_report.md` | generated data-quality evidence |
| `reports/phase1_industry_load_report.md` | ingestion, flags, shape construction |
| `reports/phase2_dianyicha_price_report.md` | price reading and slot-axis convention |
| `reports/jiangsu_market_mechanism.md` | policy verification, with its own audit record |
| `reports/phase4_industry_scenario_report.md` | the scenario, mapping and scale contract |
| `reports/phase5_storage_economics_audit.md` | settlement, caches, policy provenance |
| `reports/phase6_end_to_end_acceptance.md` | single-day acceptance; solver and SOC schema |
| `reports/phase7_arbitrage_strategy_comparison.md` | the bound and the three strategies |
| `reports/phase8_data_limitations.md` | the claim boundary, tagged by evidence type |
| `reports/phase9_sensitivity_analysis.md` | the screen and the sensitivities |

**Changed in this phase, and committed** (`0bb7c16`). `src/dianyicha_price.py`
(per-curve provenance and classification); `src/arbitrage_lp.py`
(`required_margin`, corrected degradation floor); `src/diagnose_typical_day.py`
(margin and break-even ratio use the derived floor); `src/phase9_sensitivity.py`,
`src/accept_typical_day_industry.py`, `src/audit_synthetic_wholesale_price.py`
(same correction); `tests/` (`test_dianyicha_price.py` gained metadata tests,
`test_arbitrage_lp.py`, `test_siting_criterion.py`,
`test_phase7_strategy_comparison.py`, `test_phase9_sensitivity.py`,
`test_end_to_end_acceptance.py` updated); the phase reports above, synced for
the corrected threshold and the corrected price provenance.

**Changed in the revision that follows it, not yet committed.** That pass left
the tree inconsistent in ways a reader could not see from the reports alone:
the reports were edited by hand to the corrected floor while the generated
artefacts beneath them still held the old one, and the two data-reading
documents disagreed with each other about what the price sheet is.

| file | change |
|---|---|
| `requirements.txt` | `openpyxl` declared; it was required by both workbook readers and listed nowhere |
| `src/dianyicha_price.py` | the module docstring no longer asserts "all four are means" — it now carries the same unresolved-aggregation reading as `CURVE_META`, which it had contradicted |
| `src/dianyicha_price.py` | §2.1's two-axis revision: `source_kind` / `spatial_statistic` / `temporal_statistic` / `window_days` are declared per curve; `temporal_statistic` is the sole authority for single-day identity; `is_measured`, `aggregation`, `mean_window_days` and the two `contains_*_series` flags are demoted to derived / deprecated properties published with their new meanings under `DEPRECATED_FIELDS`. `261009` reads `single_day`/`spatial_mean` by ruling with `externally_confirmed: false`. No price, objective, cost or measured value changed — phases 6, 7 and 9 re-run byte-identically. |
| `src/phase9_sensitivity.py` | `load_shape` now keeps `required_margin` and `nodal_per_bus`, so §6.3's table is a reading of the artefact rather than a transcribed figure |
| `tests/test_phase9_sensitivity.py` | new test re-derives the configured buses' margins from their own prices |
| `reports/phase8_data_limitations.md` | floor synced to the derived one — this report had been missed by the earlier sync |
| `reports/phase9_sensitivity_analysis.md` | knife-edge table corrected; bus 14's shortfall is **+0.000097**, not +0.004 |
| `TYPICAL_DAY_CONSOLIDATION_HANDOFF.md` | correction note added; the round's own recorded numbers are kept as recorded |
| `outputs/phase6_acceptance.json`, `outputs/phase7_strategy_comparison.json`, `outputs/phase9_sensitivity.json` | regenerated; each had still carried the pre-correction `200.0` |

**Why the artefacts mattered.** With the reports hand-edited and the artefacts
stale, nothing on disk corroborated the corrected numbers. Regenerating them
was also the check: `phase7` reproduced its report's table to the digit
(fleet profit −8.19e-05 and ESS9's 0/0/96 intent on `261009`/`fixed`), which is
what shows the hand edits were arithmetic rather than invention — and it is how
the one figure that *was* wrong, bus 14's, was found.

**The correction's effect on recorded numbers.** Every margin and break-even
derived from the degradation floor moves by 0.6957 CNY/MWh. The **siting
selection is unchanged** — the rule turns on a margin of at least 150, and all
four units of the earlier round stay on the same side of that level — and **no
conclusion in any report changes**; the knife-edge observation in §6.3 is
sharpened rather than altered.

Two near-threshold figures do change **sign**, and neither was treated as a
rounding detail:

- the earlier siting round's `ESS31` moves from **+0.7 to −0.002** CNY/MWh, and
- the native day's bus 14 moves from **+0.004 to +0.000097** CNY/MWh.

Both are recorded where they are used — `tests/test_siting_criterion.py` pins
the first and `reports/phase9_sensitivity_analysis.md` the second. Neither
reopens anything: a unit 0.002 CNY/MWh above break-even cannot cycle, which is
the same statement the old sign made less sharply. It is the reason the floor is
carried to four decimals rather than rounded to 200.

---

## 12. Remaining risk and unverified items

**Bound on the conclusions.**

- The price material is four curves, all means or unresolved. **There is no
  measured daily series.** No cross-day variance, positive-day rate or
  generalisation statistic is available, and none is reported.
- The load is the IEEE 33-bus network's own demand, redistributed by normalised
  industry shapes. **It is not Jiangsu load**, and no absolute-MW conversion has
  been performed.
- The internal nodal price is this model's own output. **It is not a Jiangsu
  settlement price**, and this project does not reproduce the provincial
  market's declaration, clearing or settlement.
- One cycle cost, one battery, one network model, one voltage band. **None was
  varied.**

**Unverified, awaiting an external source.**

| item | needs |
|---|---|
| the trailing curves' window start/end dates | the price platform |
| whether `261009` really is one day — the project reads its header's 均值 as a *spatial* mean and records it as `single_day` by ruling, with `externally_confirmed: false` | the provider's column definition |
| how each curve's spatial mean was taken — extent, weighting, number of points | the provider's methodology note |
| whether the spot market has completed its formal-operation switch, and the effective rule version | 江苏电力交易中心 / 江苏能源监管办 |
| storage-specific day-ahead declaration and settlement rules | an official storage document |
| the currently effective price limits | the operative notice's original text |
| whether the modelled units could register and bid | official rules |
| the out-of-model charges listed in §4 | official rules |
| the cause of the 2,683 non-PV negative readings | metering records |
| whether the surviving load days represent a year | more complete data |

**How far this revision's own evidence reaches.**

- **`outputs/` is git-ignored**, so none of the artefacts §6 is read from ships
  with the repository. Every table in this study is reproducible from §9's
  commands and not otherwise; until they are run, a reader has the reports'
  word for the numbers. §9 names the invocation behind each one.
- **The suite's count belongs to the environment that produced it.** The run
  recorded in §10 reports `pandapower 2.14.9`, while `requirements.txt` pins
  `2.14.1`. The gap is a minor version and was not investigated.
- **Two root-level documents are historical records and were deliberately not
  rewritten.** `TYPICAL_DAY_PHYSICAL_BASELINE.md` is marked superseded;
  `TYPICAL_DAY_CONSOLIDATION_HANDOFF.md` now carries a correction note but keeps
  its own measurements as that round recorded them. Neither is a current
  statement of the floor.

**Not claimed.** That the Jiangsu real day-ahead market has no storage
arbitrage; any cross-day statistic; any strategy ranking while every unit is
idle; any MATD3 performance; a complete settlement bill. Nothing was tuned —
`cycle_cost`, efficiency, price limits, thresholds and network constraints are
exactly as configured — and the zero is the measurement.
