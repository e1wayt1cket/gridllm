# Audit remediation — old system, 2026-11-15

Source of the fix list: `reports/full_project_audit_2026-11-15/REPORT.md`.
Plan: `REMEDIATION_PLAN.md` (S1-S6).
Scope: the old engine (`src/dispatch_*`, `src/market.py`, `src/rl_*`), not the
`da_market*` / `local_da*` modules, and not the load scale or the bus mapping.

The frozen patch that this report first described as "prepared, not applied"
was applied in two re-freezes. Live protocol is **v6** (`ac0f78d2`,
`V6_candidate_001`, `reports/v2_cy_jn_2025/formal_training_v6/`); v5
(`6e32398b`) is historical. What each re-freeze changed, and what it moved in
the numbers, is in `REFREEZE_V5.md` and `REFREEZE_V6.md` beside this file. The
baseline recompute that follows from them is in
`reports/baseline_recompute_2026-11-16/RECOMPUTE.md`.

Nothing below is a v6 result unless it says so. The ladder table and the
typical-day acceptance under v6 are in `RECOMPUTE.md`; no training has run.

## Applied outside the pinned sources

| item | file | verdict | evidence |
|---|---|---|---|
| F1 radial tree admitted open tie lines | `src/dispatch_core._select_radial_lines` | `PASS` | before: `[0..5,7,8,10..12,14..25,27..31,32,33,34,36]`; after: `[0..31]`, ties 32/33/34/36 excluded, main segments 6/9/13/26 present; a non-spanning result raises |
| F2 RE above nameplate | `src/grid._noisy_load` | `PASS` | upper clip at the declared capacity, six call sites pass it |
| F3 headline over a partial day set | `src/counterfactual_eval.headline` | `PASS` | raises on `seeds_dropped`; `tests/test_counterfactual_eval.py` |
| F5 per-period discount in the clearing | `src/models.py`, `config/defaults.yaml` | `PASS` | `discount_factor` 0.997 → 1.0 |
| F7 cache key without the case name | `src/grid.build_base_network` | `PASS` | `net_name` is the first element of `cache_key` |
| F8 windows cleared at 00:00 quotes | `src/market._slice_action_params` | `PASS` | both rolling call sites slice to `[t_start:t_end]` |
| 3.5 loss iterate exits on a residual | `src/dispatch_ldf.py` | `PASS` | `loss_allocation_residual` and `LOSS_ALLOCATION_TOL` gate both the single-period and the batch loop; the cap still binds first at the default `n_loss_iters=3` (TD-15) |
| 3.5 loss iterate can be infeasible | `src/dispatch_ldf.py` | `PASS` | single-period and batch both keep the last allocation that solved |
| 3.5 engine column in the summary | `src/data_aggregator.py`, `src/make_training_summary.py` | `PASS` | `run_summary` carries `opf_mode` from the run's own `config.json`; a missing value renders `—` |
| 3.7 two multi-objective methods at once | `src/models.check_market_design` | `PASS` | raises at construction and at `clear_market` entry |
| 3.8 two dead knobs | `src/models.py`, `src/llm.py` | `PASS` | `opf_tolerance` / `opf_max_iter` deleted; neither had a reader |
| 3.8 licence message | `src/dispatch.py` | `PASS` | `_GUROBI_MISSING_MSG` deleted as dead; each engine's own `RuntimeError` names the mode and Gurobi's message names the licence |
| 4.4 duplicate `delta_pi_percent` | `src/baseline_ladder.py` | `PASS` | delegates to `counterfactual_eval.delta_pi_percent` |
| 4.4 reward and payoff in one row | `src/run_artifacts.py` | `PASS` | the three settled-money columns move to `economics.csv`; `tests/test_run_artifacts.py` asserts the header, the absence of the columns from `metrics.csv`, and the load round-trip |
| 5.8 a failed block enters the replay buffer | `src/train_rl.py` | `PASS` | the legacy loop now refuses `_last_result is None` or `fell_back` before `remember()` |
| 8.2 unread `capacity_basis` | `config/independent_da.json` | `PASS` | removed; `grep -rn capacity_basis` returns nothing |
| 9.5.6 NaN inside `positive_rate` | `src/counterfactual_eval.summarize` | `PASS` | non-finite entries dropped before the statistics |
| 8.1 `clarabel` not in the dependency list | `requirements.txt` | `PASS` | `clarabel==0.11.1` |
| 10.3 ignore rule | `.gitignore` | `PASS` (two exceptions) | `!data/processed/industry_meter_bus_map_v2.csv`, `!reports/multi_user_load_2026-11-13/evidence/*.csv`; the global `*.csv` rule stays |
| misnamed coverage flag | `src/accept_typical_day_industry.py` | `PASS` | `served_covers_forecast` asks whether anything was left unserved, not whether the balance closes |

The last three rows of the original list — the two loss-iterate defects and the
coverage flag — were not on the audit list. They surfaced once the corrected
tree was cleared: on `typical_day_industry` the second I2R pass produced losses
that made the third pass infeasible at periods 10, 94 and 95, which turned the
per-period fallback's status into `PARTIAL` and failed
`test_a_fallback_result_names_the_fallback_as_final`. Before the correction the
wrong tree carried less I2R loss, so the iterate always converged.

## Applied inside the pinned sources — v5

Full before/after tables are in `REFREEZE_V5.md`; the one-line attribution is
here.

| item | file | what it changed | measured |
|---|---|---|---|
| F6 terminal value while the endpoint is pinned | `dispatch_socp.py` | the terminal term is zero under an equality | prices and schedules bitwise identical; `clearing_objective` 66432.444952 → 65914.221619, exactly `mean(wholesale) × 3.200 MWh` |
| F4 nodal re-solve without the slack penalties | `dispatch_socp.py` | the re-solve objective carries `penalty_carbon_slack` and `penalty_re_slack` | with caps forced on: `re_consumption_rate` 28.98 → 80.74, LMP mean 168.90 → 421.25, throughput 21.21 → 11.90 MWh; the pre-patch re-solve returned the unconstrained schedule while reporting nonzero slacks |
| F9 leg-averaged anchor, uncalibrated weight | `rl_b_c2_gate.py`, `rl_da_gate.py` | the anchor is a sum over legs, weight 0.01 → 0.10 | `NOT_TESTED` — a training-time quantity with no day-level measurement |
| F10 unrun periods recorded as a zero quote | `rl_env.py` | unrun periods start at the fleet's truthful declaration | 92 of 92 unrun periods read 0.0 → 1.0 |
| 5.3 counterfactual documented as per-agent | `rl_env.py` | comment only; the code was already fleet-wide | n/a |
| 5.10 non-atomic checkpoint | `checkpoint_writer_v2.py` | staging directory and `os.replace`; an existing step refuses | an interrupted save left a complete-looking directory with no manifest; it now leaves none, and a same-step retry raises `FileExistsError` |

`F4` moves the constraint-based method, which is false on the formal day; `F6`
moves every whole-day clear that pins the endpoint, i.e. the formal path, but
only the reported objective and not the prices or the schedules.

## Applied inside the pinned sources — v6

| item | file | what it changed | measured |
|---|---|---|---|
| B1 baseline on the wrong tree | `rl_b_c2_gate.py` | `B1_RESULTS` / `B1_BLOCKS` / `B1_MANIFEST` → `data/processed/v2_cy_jn_2025_v6/` | 96/96 feasible, 0 fallbacks, 0 voltage cells out of band; mean `G_total` 165.3858 → 179.6571 CNY/day, `losses_mwh` 2.8103 → 4.2113, cycles within ±2.2% |
| 5.2 anchor magnitude unobservable | `rl_b_c2_gate.py`, `rl_da_gate.py`, `train_da_gate.py` | `anchor_by_agent` / `anchor_weight` returned from `train_day` and written as two columns | code path `PASS`; no value has been read, `NOT_TESTED` |
| 6.3 non-finite clearing or loss | `dispatch_socp.py`, `rl_b_c2_gate.py` | `_assert_finite` over `lmp` and every schedule series before the result dict; `FloatingPointError` on a non-finite block series or a non-finite loss | `NOT_TESTED` — structural guards, nothing has made them fire |

## Ruled out, with the reason

| item | ruling |
|---|---|
| 3.8 solver-parameter single source | Not done. The six `setParam` calls are not divergent — `Method` 2 (barrier) vs 1 (dual simplex), `QCPDual` and `BarHomogeneous` belong to different engines on purpose. The one measured defect in that row was the two dead knobs, which are gone. |
| 4.2 settlement declaration | Not done. The payload says `two_settlement_ledger` while the formal path settles through `participant_payoff`. Which is right is open (plan D-F), so the declaration was left alone rather than changed to match code that may itself change. |
| 5.4 redundant `ema_dev` | Not done. It is one of `OBS_V6`'s twelve dimensions; deleting it is a frozen-interface change. |
| cycle cost in the clearing objective (D-D) | Kept. Clearing at half the real rate makes the fleet cycle more and earn less (656.77 → 494.41 CNY); moving it out also has to change the quote sign guard, the action spec and the observation anchor. |
| F11 — two settlement ledgers, three cone forms, two pricing rules | Not done. Each is a restructuring with its own regression surface; recorded as a separate item. |
| 82 broad `except Exception` | Not done. Each one needs a decision about whether a silent failure should become a raise. |

## Not done

| item | why |
|---|---|
| 6.2 SOC conversion consolidation | Done only on the clearing path: `market.clear_market` wrote the period length as a literal `0.25` where the engines write `DT_HOURS`, and both call sites now import `DT_HOURS`. The wider `soc * e_max` / `e / e_max` rewrite across nine files did not happen, and a dozen diagnostics still carry a literal `0.25`. Those are readers, not the clearing, so the consolidated form is not what a reported number depends on; a mechanical rewrite of them would only be a transcription risk. |
| 5.9 thread count in the manifest | Not measured. |
| 10.2 `code_revision` and the 12-entry source list | Not measured. |
| 9.2 absolute money tolerances | Not applied. The remaining `assertAlmostEqual` calls on money are all in the new-engine suites (`test_da_market_system`, `test_local_da_market`), which this round does not own; no old-engine money assertion uses a fixed number of decimal places. |
| 3.8 licence error text at the `optimize()` boundary | Not added. Gurobi's own exception names the licence, so a wrapper would restate it. |

## Consequences of F1

`baseline`, `opf_mode="socp"`, T=96:

| quantity | before | after |
|---|---|---|
| price mean (CNY/MWh) | 726.7 | 1544.0 |
| price max | 1260.5 | 1998.8 |
| LMP spatial spread | 45.5 | 700.3 |
| congestion vs baseline spread (LinDistFlow) | 517.3 vs 24.3 | 644.7 vs 582.5 |

- `test_congestion_widens_lmp_spread` asserted a 1.5× ratio that only the
  pre-fix tree produced; it now asserts the mechanism (spread widens) and the
  measured 1.11× is recorded in the docstring. The scenario needs a redesign,
  not a lower threshold.
- The table above is `baseline`, which is `explicit_peak` at 10 MW against a
  3.715 MW feeder. It does not transfer to the training scenario. On all 53
  `near/train` pairs with truthful quotes over the whole day: day-mean LMP
  76.3 / 224.8 / 490.4 (min / median / max), day-mean over the wholesale curve
  1.002 / 1.052 / 1.090, nodal-max over the wholesale day-maximum
  1.04 / 1.14 / 1.20, no shed, no clearing fallback, 198 of 160,908 price cells
  over the `+3.0` clip on 2 days of 53.
- Voltage: 0 of 167,904 bus-period cells outside `0.93 <= V <= 1.05`, 0
  fallbacks. It is met on the edge: the median day's minimum is 0.9303, 26 days
  of 53 touch 0.9300, and 5.0% of cells sit under the strict 0.95 bound.
- One line reaches its limit: pandapower line 4 (bus 4 -> 5, `max_i` 0.414 kA)
  at utilisation 1.000 around 08:30 on 3 days of 53, with voltage at the 0.93
  bound in the same periods.
- `PRICE_REF_CNY_MWH` = 288.19757259048833 was wrongly reported here as
  saturated. The `xfail(strict=True)` that followed from it was a harness
  artifact: `tests/test_rl_spec._make_env` built its env from `baseline`. The
  helper now uses `typical_day` and
  `test_a_blocks_realized_lmp_reaches_only_the_next_observation` passes.
- Every clearing number produced before F1 describes a network that is not
  `case33bw`'s. The B1 baseline, the ladder and the typical-day acceptance have
  been recomputed (`RECOMPUTE.md`); the benchmark table has not, and is
  `BLOCKED` on a producer for its JN-Base rows.

## Verification

```
PYTHONPATH=src python -m pytest tests/ -q                     # split in two halves, see below
PYTHONPATH=src python -m pytest tests/test_frozen_guard.py -q  # 31 passed
PYTHONPATH=src python src/run_formal_training.py --check       # all_pass: true
```

The suite was run as two sequential `pytest` invocations over an alphabetical
split of `tests/test_*.py` (56 files each), because one process for all 112 was
killed for machine memory before it printed a summary. 818 passed / 1 failed
(424.75 s) and 666 passed / 0 failed (450.03 s): **1 failed, 1484 passed,
53 deselected**.

The failure is `tests/test_formal_ensemble.py::test_every_consumer_reads_the_declared_comparison_set`,
and it is the `energy_env` interpreter's broken matplotlib install, item `9.4`
of the audit list. Reproduced alone in 2.57 s:

```
src/make_rot_comparison_figures.py:28: AttributeError: module 'matplotlib' has no attribute 'use'
```

`importlib.util.find_spec("matplotlib")` resolves to
`C:\Users\26036\.conda\envs\energy_env\lib\site-packages\matplotlib` with
`loader=None` and `origin=None` — a namespace package, i.e. no `__init__.py`,
while `pip show matplotlib` reports 3.7.2. Nothing in this round's diff imports
matplotlib or writes to site-packages, and `make_rot_comparison_figures.py` is
not modified, so this is an environment defect and not a regression. It is
`BLOCKED` rather than `FAIL` in the sense that matters here: the check it makes
is about consumers of a comparison declaration, and it never reaches that
assertion. Not verified: that it would pass under an interpreter with a
complete matplotlib, because the other available interpreter fails earlier on
`pandapower` -> `ortools` (`WinError 127`).

`--check` returns, in full: `data_payload_matches true`,
`code_sources_match true`, `code_tests_match true`, `config_hash_matches true`,
`data_manifest_hash_matches true`, `code_manifest_hash_matches true`,
`common_split_counts true`, `all_disjoint true`, `legacy_label_matches true`,
`h2_sealed true`, `budget_contract_ok true`, `all_pass true`, `status: null`,
`accepted_candidate: V6_candidate_001`, `contract_violations: []`.

`status: null` is the honest reading: the tree and its manifest agree, and the
protocol has no validation on its own sources. `run_formal_training.py --seed`
refuses anything but `GO_FORMAL_TRAINING`, so no launch is authorized.
`training_readiness.md` and `pretraining_validation_results.json` describe v4
code and were not carried forward.

## Not tested

| item | why |
|---|---|
| `test_every_consumer_reads_the_declared_comparison_set` passing | `BLOCKED` by the `energy_env` matplotlib install; it fails at the module import, before its assertion |
| F9 leg-summed anchor | a training-time quantity; the only way to observe it is a run under v6, which `status: null` forbids |
| 5.2 anchor columns' values | same; the columns exist and the accumulator runs, but no episode row has been written |
| 6.3 non-finite guards | no test injects a NaN into a solved clearing or a loss vector; the guards compile and nothing has made them fire |
| the benchmark table and the training-readiness headroom | `BLOCKED`; see `RECOMPUTE.md` |
| the three diagnostics on the pre-F1 B1 artifact | `TECHNICAL_DEBT.md` TD-13 |
| an evaluation CSV's clearing engine | `TECHNICAL_DEBT.md` TD-16 |
| the 53 training days' day-mean LMP distribution | measured before `F4`/`F6` and not repeated; TD-12 carries the number and says so |
