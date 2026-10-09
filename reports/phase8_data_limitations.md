# Phase 8 — Data Limitations, Findings and the Boundary of What May Be Claimed

This document is the declaration boundary for the project's final report and
delivery. Every statement below is tagged with how it is known, so a reader can
tell a fact from a finding from a guess:

| tag | meaning |
|---|---|
| **[CONFIRMED]** | Established by code, configuration, a test, or an input file. Re-checkable in this repository. |
| **[LIMITED]** | A conclusion the experiments support, within the conditions they ran under. |
| **[UNVERIFIED]** | Needs an external document or more data. Not to be asserted either way. |
| **[FORBIDDEN]** | An over-claim that must not appear in any report, figure, log or paper. |

No code, configuration, price data or model logic was changed in this phase. No
MATD3 policy was trained, loaded or evaluated. The evidence cited is the Phase
1–7 reports, the test suite, and the input files.

---

## 1. Price data

### What the file is

**[CONFIRMED]** `docs/日前价格261009.xlsx`, SHA-256
`5fc8f9a8461b680fd38a6d2fefb3d2ff99f239c8bd13a6c77df27de3bb9612f3`, read by
`src/dianyicha_price.py` (read-only `openpyxl`).

| sheet | role | its own column header | min | max | mean | `is_measured` |
|---|---|---|---|---|---|---|
| `261009` | `same_day_mean` | **日前均值（CNY/MWh）** | 225.90 | 445.00 | 386.83 | **false** |
| `7days` | `trailing_7_day_mean` | 日前均价 | 257.50 | 477.87 | 395.55 | **false** |
| `30days` | `trailing_30_day_mean` | 日前均价 | 333.08 | 450.86 | 407.82 | **false** |
| `halfyear` | `trailing_180_day_mean` | 日前均价 | 284.93 | 393.86 | 346.33 | **false** |

96 points each, CNY/MWh, interval-END labels (`00:15` … `24:00`) mapping
positionally onto simulation periods with no shift.

**[CONFIRMED]** All four curves are **means**. The `261009` sheet says so in its
own column header. **No measured daily day-ahead series exists anywhere in this
project.**

**[CONFIRMED]** The reader carries this forward: every `Curve` has
`is_measured = False`, and `describe()` publishes
`contains_measured_daily_series: false` with each curve's role. A report cannot
present a mean as an observation without contradicting a recorded field.

### What follows

**[FORBIDDEN]** Calling `261009` — or any of the four — "the real day-ahead
clearing price", "the verified daily price", or "a real day".

**[FORBIDDEN]** Treating the four curves as four independent dates, or
computing cross-day variance, a positive-day rate, or any cross-day
generalisation statistic from them. There is one number per period per window,
not a sample of days.

**[UNVERIFIED]** The **window end dates** of `7days`, `30days` and `halfyear`.
The workbook does not state them and no platform record has been checked. Until
that is confirmed, these curves may **not** be called strictly non-anticipative
historical information available at the decision point; they are, at best, the
platform's own aggregated statistics. This is recorded in
`reports/jiangsu_market_mechanism.md` §6 and is still open.

**[UNVERIFIED]** What these curves actually are at the platform
(province-wide average / zonal average / a re-processed series), which is the
same open item in the same place.

---

## 2. Industry load data

### What the workbook is

**[CONFIRMED]** `docs/16类行业用户.xlsx`, 117,271 rows × 108 columns, one sheet
`Sheet3`, 73 users, 16 industries, 283 meter pairs, 2021-01-01 → 2022-12-31.
Processed by `src/industry_load.py`.

**[CONFIRMED]** Measured defects, reproduced by the pipeline rather than
hardcoded: 43 placeholder rows (`[1,2,3,3,3,0,…]`, sum exactly 12), 4,609 rows
with missing slots, 81 fully empty rows, 81 rows with a non-active power phase,
15,482 all-zero days, 13,528 duplicate rows (4 of them conflicting), **2,683
negative values on non-PV industries**, 7,856 on 分布式光伏. Source:
`reports/industry_load_quality_report.md`.

**[CONFIRMED]** `综合倍率` is unreliable. Of 200 checkable meters, implied peak
fits the declared 运行容量 for **87** and contradicts it for **113**; the worst
case implies 4,594× the rating. `industry_load.absolute_mw()` therefore
**refuses by default** and names the unverified industries.

**[CONFIRMED]** `分布式光伏` is a generator, produced outside the load pool and
excluded by name from the scenario mapping. `5G基站` (2 users) and `电解铝`
(1 user, 208 days) fall below the reliability threshold and are marked
`limited`.

### What follows

**[FORBIDDEN]** Calling the scenario's load "real Jiangsu load", "measured
Jiangsu demand", or "load built from verified Jiangsu absolute power". The day
is **IEEE 33-bus demand redistributed within the day by normalised Jiangsu
industry shapes** — the load level is the network's own 3.715 MW, unchanged.

**[FORBIDDEN]** Presenting the bus → industry assignment as a real geographic
connection map. `config/scenarios.yaml` marks it `simulation_mapping: true`;
there is no user-to-node table for this feeder.

**[FORBIDDEN]** Interpreting the 2,683 non-PV negatives as verified reverse
power flow. They are flagged and preserved, and no cause has been established.

**[FORBIDDEN]** Using the `5G基站` or `电解铝` curves to support a claim about
industry generally. They are marked limited and excluded from the benchmark.

**[UNVERIFIED]** Whether the surviving clean days are representative of a year.
4,609 incomplete days were dropped rather than filled, so seasonal and weekday
coverage is uneven and unchecked.

---

## 3. Market mechanism and the meaning of prices

**[CONFIRMED]** The model uses the 电易电查 mean curves as its **external price
scenario** (`wholesale_market.source: dianyicha`), and stores are settled at
their **nodal LMP** produced by the SOCP distribution-network clearing.

**[CONFIRMED]** The nodal LMP is an output of this project's own model. It is
**not** a reproduction of any Jiangsu provincial settlement price.

**[FORBIDDEN]** Describing the internal LMP as "the Jiangsu nodal price", "the
provincial settlement price", or "the market clearing price".

**[FORBIDDEN]** Claiming the project reproduces the Jiangsu provincial spot
market's declaration, clearing or settlement. It does not: there is no
provincial SCUC/SCED, no full-unit commitment, no zonal price formation, no
market-participant declaration book. See
`reports/jiangsu_market_mechanism.md` §3.2 and §4.

**[UNVERIFIED]** Three policy items recorded in
`reports/jiangsu_market_mechanism.md`: whether the spot market has completed its
formal-operation switch (public sources show it has not, as of 2026-10-09, but
absence of an announcement is not proof of absence); the storage-specific
day-ahead declaration and settlement rules; and the currently effective price
limits. **None may be stated as a confirmed rule.**

**[UNVERIFIED]** Whether a 0.4 MW / 1.6 MWh unit could register and bid in the
real market at all. Registration conditions exist in policy (备案文件, direct
grid connection, independent metering and control, dispatchable, legal-person
status) and this study does not claim the modelled units satisfy them.

---

## 4. Accounting

**[CONFIRMED]** The research KPI is the **settled day-end net revenue** from
`counterfactual_eval.settle_day` — the whole day re-cleared with both SOC
endpoints pinned, priced at the nodal LMP, from the quotes actually declared.
The evaluation CLI's `*_profit` columns and the programmatic evaluator use the
same call (`src/eval_agents.settle_capture`). The rolling training reward is
reported under separate column names and is not a payoff.

**[CONFIRMED]** `PayoffBreakdown.other_cost` has **zero assignments in the
source** and is always `0.0`. No cost was invented to fill it.

**[CONFIRMED]** The accounting covers revenue, purchase cost, generation cost,
degradation cost and penalty, and **does not** cover: 输配电价, 上网环节线损,
系统运行费用, 政府性基金及附加, deviation charges, capacity compensation, cost
compensation, or imbalance funds.

**[FORBIDDEN]** Calling the result a "complete settlement bill", "the storage
project's total revenue", or "the full market settlement". The only permitted
name is **modelled storage day-ahead energy net revenue within the model's
boundary**.

---

## 5. What the experiments found

All from `reports/phase7_arbitrage_strategy_comparison.md` and
`outputs/phase7_strategy_comparison.json`.

**[LIMITED]** On all four mean curves, **no storage unit under any of the three
strategies achieved economically meaningful trade.** Throughput was ~1e-7 MWh —
about 1.6e-8 of one fleet cycle (6.4 MWh).

**[LIMITED]** The perfect-foresight arbitrage LP returned **exactly zero** on
all four curves, both on the exogenous curve and on each unit's own cleared
nodal LMP. Every curve's round-trip margin (52.6–163.9 CNY/MWh) falls short of
the 200 CNY/MWh the clearing charges per cycle.

**[LIMITED]** The zero is not a broken LP: the same LP returns **344.29 CNY**
per unit on a synthetic 120→1000 CNY/MWh band and **30.85 CNY** on a flat-500
curve with a 200 trough. **This control validates the LP, not the price data** —
it says nothing about whether a real daily series would offer arbitrage.

**[LIMITED]** The threshold greedy **declared** directional intent in 84 / 80 /
63 / 18 periods across the four curves and **was filled in none**. It is not an
idle rule; the clearing refused every quote.

**[LIMITED]** MPC, reading 8 periods of look-ahead, found almost nothing to do
(7 / 7 / 3 / 0 charge-intent periods) — agreeing with the independent
perfect-foresight LP. Two different methods, the same conclusion.

**[LIMITED]** The ~1e-4 CNY differences between strategies are **float residue**,
not economics. **No strategy ranking may be derived from them.**

**[LIMITED]** Physical indicators are identical across strategies within each
curve — voltage 0.9697–1.0076 pu inside the 0.93–1.05 band with zero violations,
max line utilisation 0.167 with zero over-limit periods, unserved energy at
solver-noise level, all 96 SOC transitions verified.

**[FORBIDDEN]** Reading that identity as evidence that the three strategies are
equivalent in general. They are identical **because the clearing did not move
the batteries**, which is a statement about this price band, not about the
strategies.

### The arbitrage reference's own boundary

**[CONFIRMED]** The LP's objective is
`Σ price·(dis − ch)·dt − cycle_cost·(ch + dis)·dt`, subject to the SOC
transition, power limits, SOC band and `soc[T] == soc[0]`. It carries the
configured `cycle_cost`, the configured efficiencies and the configured SOC
bounds — and **nothing else**. It contains **no IEEE 33-bus network, no voltage
or line constraints, no declaration or clearing mechanism, no competition, and
no possibility of a quote being refused**.

**[FORBIDDEN]** Calling it "the upper bound on storage revenue in this market",
or claiming any strategy attains it. It is a **storage-side perfect-foresight
arbitrage reference**, and its answer is not automatically a strict upper bound
on what this market's clearing could pay.

---

## 6. The economic statement, in the only form permitted

> Under the four current day-ahead **mean** price scenarios, and given the
> configured storage efficiency, SOC constraints and cycle cost, **no
> economically meaningful storage arbitrage appeared**.

Immediately, and in the same breath:

**[FORBIDDEN]** "The Jiangsu real day-ahead market has no storage arbitrage
opportunity." What was measured is four **mean** curves under **one** cycle
cost, **one** battery and **one** network model. A real day's volatility (which
a mean flattens), a different cycle cost, a larger battery, or different market
rules would change the answer. None was varied.

**[FORBIDDEN]** Any cross-day statistic — variance, positive-day rate, a
distribution of daily returns, a generalisation claim — from four curves that
are not four dates.

**[FORBIDDEN]** Any economic ranking of the three strategies while every unit
is idle.

**[FORBIDDEN]** Any statement about MATD3. **[CONFIRMED]** No MATD3 policy was
trained, loaded or evaluated in Phases 4–8, and none of the 45 directories under
`policies/` carries a checkpoint for `ESS9`/`ESS14`/`ESS17`/`ESS27`. The MATD3
code is retained and unexercised; no performance claim of any kind is available.

**[FORBIDDEN]** Adjusting `cycle_cost`, efficiency, thresholds, price floors or
caps, or network constraints to convert this zero into a positive result. The
zero is the measurement.

---

## 7. Register of unresolved items

| # | item | needs | recorded in |
|---|---|---|---|
| 1 | trailing windows' end dates and whether they precede the evaluation day | the price platform | Phase 3 §6; here §1 |
| 2 | what the curves are at the platform (province / zonal / processed) | the price platform | Phase 3 §6 |
| 3 | current formal-operation status and effective rule version | 江苏电力交易中心 / 江苏能源监管办 | Phase 3 §6 |
| 4 | storage-specific day-ahead declaration and settlement rules | an official storage document | Phase 3 §6 |
| 5 | currently effective price limits | the operative notice's original text | Phase 3 §6 |
| 6 | usable absolute-MW load (ratio ↔ rating contradiction) | metering-site verification | Phase 1 report §7 |
| 7 | cause of the 2,683 non-PV negative values | metering records | Phase 1 report §7 |
| 8 | whether surviving load days are representative of a year | more complete data | Phase 1 report §7 |
| 9 | whether the modelled units could register and bid | official rules | here §3 |
| 10 | out-of-model charges (T&D, losses, funds, deviation, compensation) | official rules | Phase 5 §5; here §4 |

Items 1–5 and 10 are the ones that stand between this work and a claim about
the *real* market. Items 6–8 bound what the *load* can mean. Item 9 bounds the
*assets*.

---

## 8. Evidence index

| claim area | source |
|---|---|
| code inventory, the three accounting defects | `reports/finalization_audit.md` |
| industry defects, exclusion counts, ratio evidence | `reports/industry_load_quality_report.md`, `reports/phase1_industry_load_report.md` |
| price curves, interval convention, provenance | `reports/phase2_dianyicha_price_report.md`, `src/dianyicha_price.py` |
| policy items and their verification status | `reports/jiangsu_market_mechanism.md` |
| scenario mapping, load scale, facilities | `reports/phase4_industry_scenario_report.md` |
| settlement definition, `other_cost`, cache and policy defects | `reports/phase5_storage_economics_audit.md` |
| single-day acceptance, solver provenance, SOC trajectory | `reports/phase6_end_to_end_acceptance.md` |
| the zero-arbitrage finding and the LP controls | `reports/phase7_arbitrage_strategy_comparison.md`, `outputs/phase7_strategy_comparison.json` |
| the invariants behind the numbers | `tests/` — 758 passing, 51 deselected, 0 failed |

The test suite is the machine-checkable part of this document: the load peak,
the 96-slot axes, the settlement identity, the closed arbitrage window
(`margin < 2 · cycle_cost`), the LP controls, the policy-cache provenance and
the curve-selection precedence are all asserted rather than asserted-about.

---

## 9. Workspace

HEAD `a1151fd2ab52e4a49a610173a1bba04c56bda563`, branch `master`, unchanged.
Backed up to
`C:/Users/26036/AppData/Local/Temp/pre_phase8_worktree.diff` before this file
was written. No source file, configuration file or test was modified in this
phase — the working tree differs from the Phase-7 state by this report alone.
No `reset`, `checkout --` or `clean` was run; `gurobi.env` was not read, copied
or printed.

## Status

Phase 8 complete. Stopping here for review; Phase 9 is not started.
