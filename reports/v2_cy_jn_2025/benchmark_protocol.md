# Benchmark protocol (V2-CY-JN-2025)

This protocol fixes what each arm is, what it may know, which days it may be
tuned on, and what must not be compared with what. It is the contract the
benchmark results are read against.

## Arms

| arm | what it is | market mechanism | decision information | storage |
|---|---|---|---|---|
| **B0 NoESS** | the four storage units are **removed** from the fleet; the same 32 load/prosumer agents clear the same network | day-ahead SOCP clearing, whole day | the published DA price curve | none |
| **B1 Truthful** | every participant declares its truthful point: storage `(0,0)`, loads/prosumers `(1,0)` | day-ahead SOCP clearing, whole day | the published DA price curve | 4 units |
| **B2A threshold** | storage quotes from a price-quantile rule | day-ahead SOCP clearing, whole day | DA price quantiles of the same day | 4 units |
| **B2B threshold + SOC band** | B2A restricted to the cheapest/dearest periods a SOC band allows | day-ahead SOCP clearing, whole day | DA price quantiles plus the declared SOC band | 4 units |
| **B3 LP bound** | one unit, perfect foresight of a price series it cannot move | **not a clearing** | the whole external DA curve | price taker |

### B0: removal, not forced zero

B0 removes the devices. The alternative — keep the four units in the model and
force `p_ch = p_dis = 0` — leaves the storage variables, SOC transitions and
terminal pin in the problem, so it answers a different question and is not
used. Both would produce the same power flow because the units carry no load
or generation, but only removal is the no-storage counterfactual.

### B1: equivalence, not repetition

`clear_market(action_params={})` already maps every participant to its truthful
point (`dispatch_core.quote_arrays`). B1 is therefore **JN-Base**: the same
agents, the same `build_config(100)`, the same call. It is registered as an
equivalence (`b1_truthful_equivalence_audit.md`), not as an independent run.
Benchmark tables cite the same 96 rows and must not count B1 and JN-Base as two
arms of evidence.

### B2: rule definitions

Both families quote per period from the published day-ahead curve:

    lo = quantile(price, q_lo),  hi = quantile(price, q_hi)
    charge_price(t)    = 0                if price(t) <= lo, else 150
    discharge_price(t) = 0                if price(t) >= hi, else 300

150 and 300 are the top of the storage action box, so a rule cannot quote
outside the space a policy would search. Zero means "act"; the box top means
"do not act unless the market pays the maximum".

**B2B** additionally keeps only the periods a SOC band can physically use:

    k_charge    = floor((soc_hi - soc0) * E / (eta_ch * P * dt))
    k_discharge = floor((soc0 - soc_lo) * E * eta_dis / (P * dt))

and allows charging only in the `k_charge` cheapest periods and discharging
only in the `k_discharge` dearest. With `E = 1.6`, `P = 0.4`, `eta = 0.92`,
`soc0 = 0.5`: band 0.10–0.90 → 6 charge and 5 discharge periods; band
0.20–0.80 → 5 and 4.

**Information.** The only input is the published day-ahead price curve of the
same day. No realized LMP, no cleared schedule and no later market outcome
enters a quote. The whole-day curve is used, including periods after the
current one, because it is the exogenous input to this day-ahead market and is
published before the clearing; that is stated rather than assumed.

### B3: what the bound is and is not

`arbitrage_lp.arbitrage_lp` maximizes, for one unit,

    sum_t price(t) * (dis(t) - ch(t)) * dt - cycle_cost * (ch(t) + dis(t)) * dt

subject to SOC transitions and limits, power limits and `soc[T] == soc[0]`,
with **perfect foresight** of the whole curve. It is computed against the
**external** day-ahead price, not the nodal LMP, so the unit does not face a
price field it helped set. It is a price-taker upper reference. The fleet
figure is four times the single-unit bound and is therefore loose: it ignores
nodal prices, the units' mutual interaction and any service outside arbitrage.

**Existing MPC, audited not adopted.** `mpc_storage.solve_storage_mpc` is a
rolling-horizon LP over a caller-supplied `price_forecast` of length `H`; it
maximizes arbitrage revenue plus a terminal value (`max` of the forecast window
by default, or the DA mean when asked), does **not** pin terminal SOC, and
executes the plan over the horizon it was given. It is not a benchmark arm
here: its forecast input and terminal treatment differ from the day-ahead
clearing arms, so putting it in the same table would compare mechanisms, not
strategies.

## Dates and splits

The comparison split is `common_split.json` (seed `20261101`):

| set | days | use |
|---|---:|---|
| common train | 53 | grid search for rule parameters |
| common validation | 11 | select the rule hyperparameters |
| common test | 11 | final evaluation, per-day paired |
| national-day block | 5 | held out from tuning, reported separately |
| FULL96-only long pairs | 16 | training coverage / shift analysis only |

Any FULL96-vs-NEAR comparison uses the common split only. The per-dataset
splits are each dataset's own record and are not comparable across datasets.
The 16 long pairs never enter the test set.

## Parameter selection

- Grid: `q_lo ∈ {0.20, 0.35}`, `q_hi ∈ {0.65, 0.80}` for B2A; the same four
  pairs crossed with `soc_band ∈ {(0.10,0.90), (0.20,0.80)}` for B2B.
- Train: all 12 candidates are cleared on the 53 common training days.
- Validation: the top two per family by mean daily fleet payoff are re-scored
  on the 11 common validation days, and one is selected.
- Test and holiday: the selected parameters only; no further tuning.
- Seed `20261101`; the search is a deterministic grid, so the seed only fixes
  the tie-break record. The full grid, means and selection are written to
  `data/processed/v2_cy_jn_2025/benchmark/rule_params.json`.

## Metrics and tolerances

Storage payoff is `participant_payoff` at each unit's own bus LMP, minus the
`100 CNY/MWh` degradation on charge and discharge. The fleet payoff is the sum
of the four units. Served energy, curtailment, charge/discharge energy, minimum
voltage and maximum line utilisation come from the same result dict.

Physical checks use the tolerances declared in `audit_near_clearing_reuse.py`:
active balance ≤ 1e-5 MW, reactive ≤ 1e-5 MVAr, terminal SOC ≤ 1e-9,
charge/discharge product ≤ 1e-6, slack-LMP deviation ≤ 1e-2 CNY/MWh.

## What must not be compared

- **B0 vs B1/B2**: same market, different device set. A difference is the
  value of having storage, not of a strategy.
- **B1 vs B2**: same market and device set; a difference is the value of the
  rule. B2 is not an RL policy and nothing here is a training result.
- **B3 vs anything**: the bound is not a clearing outcome. It has perfect
  foresight, faces the external price, and ignores nodal settlement.
- **FULL96 vs NEAR raw test splits**: not the same days. Use the common test.
- **B1 counted twice**: B1 and JN-Base are one clearing.
