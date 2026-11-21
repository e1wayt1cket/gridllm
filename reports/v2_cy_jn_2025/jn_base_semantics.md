# What JN-Base actually is (V2-CY-JN-2025)

**JN-Base is a day-ahead clearing in which every participant declares its
truthful point.** It is not NoESS, not a rule policy and not an upper bound.
Storage declares `(charge_price, discharge_price) = (0, 0)`; each load or
prosumer declares `(bid_mult, offer_adder) = (1, 0)`. Those are the fallbacks
`dispatch_core.quote_arrays` applies when `action_params` carries no entry, and
the benchmark runner calls `clear_market(..., {})` with no entry at all.

## The chain, checked in code

| question | answer | where |
|---|---|---|
| How are storage quotes generated? | Absent action → `ACTION_STORAGE_QUOTE_V2.truthful`, `(0, 0)` CNY/MWh. A *partial* declaration raises; nothing is completed silently. | `src/dispatch_core.py`, `quote_arrays` |
| Are the four units' quotes identical? | Yes, exactly. They differ only in bus (9/14/17/27), hence in the nodal price they settle at. | `src/ess.py`, `build_ess_fleet` |
| Are quotes fixed, or do they move with SOC or price? | Fixed. `_broadcast` expands the scalar to all 96 periods; no SOC or price enters the quote. | `dispatch_core._broadcast` |
| What is the cycle cost and where is it charged? | `config.storage.cycle_cost = 100` CNY/MWh on **charge + discharge energy**. It appears twice by design: in the clearing objective as `-100·(ch+dis)·dt`, and in the settlement ledger as `degradation_cost = 100·(ch+dis)`. | `dispatch_socp.py` objective; `participant_payoff._ledger` |
| Does storage participate in the clearing? | Yes. The units are agents in `agents_list`, valued at their own declared quotes in the objective, with SOC transitions, power limits, mutual exclusion by net flow, and a terminal SOC pin on a whole day. | `dispatch_socp.solve_socp_opf_batch` |
| How is storage payoff computed? | `participant_payoff` → `IndependentStoragePayoff`: revenue `LMP·p_sell`, purchase cost `LMP·p_buy`, degradation `100·(ch+dis)`. Settlement is at the unit's **own bus LMP**, not the wholesale price. The clearing objective is not a payoff and is not read as one. | `participant_payoff.py` |
| Which days and configuration? | 96 FULL96 pairs; `build_config(100)` with SOCP, `line50_q2.0`, 0.93–1.05 p.u., 4 ESS at 1.6 MWh / 0.4 MW, terminal SOC pinned. | `src/screen_v2_cy_jn_2025.py`; `reports/v2_cy_jn_2025/config.json` |
| Inputs and hashes | Pairing table `9cb980dc…`; `results.csv` and `config.json` hashes are recorded in `manifest.json`; `data/processed/v2_cy_jn_2025/datasets.json` indexes both datasets. | — |

## Consequence: B1 Truthful ≡ JN-Base

B1 is not a second experiment. It uses the same agents, the same
`build_config(100)`, the same empty `action_params` and the same whole-day
clearing, so a "B1 run" would be JN-Base under another name. B1 is therefore
registered as an **equivalence**, not a repetition, and the claim is checked in
`b1_truthful_equivalence_audit.md`:

- the empty-action → truthful mapping was evaluated for every agent on all 96
  days (largest deviation 0.00e+00 CNY/MWh);
- 36 agents, four storage units at buses 9/14/17/27, identical specs every day;
- 7/7 configuration checks and 3/3 input-hash checks pass;
- 96/96 days pass the discrete and numeric checks on the stored results.

**No second solve was run, and none is needed.** The equivalence is a statement
about the code path and the inputs, not about the engine agreeing with itself.

## What JN-Base is not

- **Not NoESS.** The four batteries are present and can cycle; on the 96 days
  the fleet earns a median 86.6 CNY/day under truthful quoting.
- **Not a rule policy.** There is no threshold, SOC rule or forecast in the
  quote; it is the zero-premium declaration.
- **Not an upper bound.** It is a market-clearing outcome with the units'
  own quotes, and it can be below both the rule policies and the price-taker
  LP bound.
- **Not an independent arm from B1.** Benchmark tables list B1 as JN-Base and
  must not count the two as separate evidence.
