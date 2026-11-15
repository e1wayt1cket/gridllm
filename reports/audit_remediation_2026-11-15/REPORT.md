# Audit remediation — old system, 2026-11-15

Source of the fix list: `reports/full_project_audit_2026-11-15/REPORT.md`.
Scope: the old engine (`src/dispatch_*`, `src/market.py`, `src/rl_*`), not the
`da_market*` / `local_da*` modules. The 22 frozen sources are handled by a patch
that has to be applied by hand (`tmp/refreeze_v5/frozen_patches.diff`), because
`.claude/hooks/protect_frozen.py` refuses Edit/Write on them; the user chose
that over adding a re-freeze bypass to the hook.

## Applied and tested in the working tree

| item | file | verdict | evidence |
|---|---|---|---|
| F1 radial tree admitted open tie lines | `src/dispatch_core._select_radial_lines` | `PASS` | before: `[0..5,7,8,10..12,14..25,27..31,32,33,34,36]`; after: `[0..31]`, ties 32/33/34/36 excluded, main segments 6/9/13/26 present; a non-spanning result raises |
| F2 RE above nameplate | `src/grid._noisy_load` | `PASS` | upper clip at the declared capacity, six call sites pass it |
| F3 headline over a partial day set | `src/counterfactual_eval.headline` | `PASS` | raises on `seeds_dropped`; `tests/test_counterfactual_eval.py` (10 passed) |
| F5 per-period discount in the clearing | `src/models.py`, `config/defaults.yaml` | `PASS` | `discount_factor` 0.997 → 1.0 |
| F7 cache key without the case name | `src/grid.build_base_network` | `PASS` | `net_name` is the first element of `cache_key` |
| F8 windows cleared at 00:00 quotes | `src/market._slice_action_params` | `PASS` | both rolling call sites slice to `[t_start:t_end]` |
| 3.7 two multi-objective methods at once | `src/models.check_market_design` | `PASS` | raises at construction and at `clear_market` entry |
| 4.4 duplicate `delta_pi_percent` | `src/baseline_ladder.py` | `PASS` | delegates to `counterfactual_eval.delta_pi_percent` |
| 9.5.6 NaN inside `positive_rate` | `src/counterfactual_eval.summarize` | `PASS` | non-finite entries dropped before the statistics |
| 8.1 `clarabel` not in the dependency list | `requirements.txt` | `PASS` | `clarabel==0.11.1` |
| 3.5 loss iterate can be infeasible | `src/dispatch_ldf.py` | `PASS` | single-period and batch both keep the last allocation that solved; `tests/test_end_to_end_acceptance.py` 39 passed |
| misnamed coverage flag | `src/accept_typical_day_industry.py` | `PASS` | `served_covers_forecast` asks whether anything was left unserved, not whether the balance closes |

The last two were not on the audit list. They surfaced once the corrected tree
was cleared: on `typical_day_industry` the second I2R pass produced losses that
made the third pass infeasible at periods 10, 94 and 95, which turned the
per-period fallback's status into `PARTIAL` and failed
`test_a_fallback_result_names_the_fallback_as_final`. Before the correction the
wrong tree carried less I2R loss, so the iterate always converged.

## Prepared, not applied (frozen sources)

`tmp/refreeze_v5/frozen_patches.diff`; `git apply --check -p1` passes, the
patched copies pass `py_compile`. See `tmp/refreeze_v5/README.md`.

| item | file |
|---|---|
| F4 nodal re-solve without the slack penalties | `src/dispatch_socp.py` |
| F6 terminal value added while the endpoint is pinned | `src/dispatch_socp.py` |
| F9 leg-averaged anchor and an uncalibrated weight | `src/rl_b_c2_gate.py`, `src/rl_da_gate.py` |
| F10 unrun periods recorded as a zero quote | `src/rl_env.py` |
| 5.3 counterfactual documented as per-agent, coded as fleet | `src/rl_env.py` |
| 5.10 non-atomic checkpoint | `src/checkpoint_writer_v2.py` |

## Consequences of F1 that change what may be reported

Measured on `baseline`, `opf_mode="socp"`, T=96:

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
- `PRICE_REF_CNY_MWH` = 288.19757259048833 is below a third of the corrected
  price level, so both price-history features sit on their +3.0 clip and carry
  no level. `tests/test_rl_spec.py::test_a_blocks_realized_lmp_reaches_only_the_next_observation`
  is `xfail(strict=True)` until the reference is re-derived. Recorded as TD-12.
- Every clearing number produced before this change describes a network that is
  not `case33bw`'s: the benchmark tables, `b1_terminal_neutral_results.csv` and
  the training-readiness headroom all have to be recomputed.

## Not addressed

Structural items (F11: two settlement ledgers, three cone forms, two pricing
rules, 82 broad `except Exception`, mtime-based 口径 in
`data_aggregator.py`), the engine column in the training summary (3.5) and the
loss iteration still stopping at a fixed count rather than at a residual
tolerance (3.5), single-sourced solver parameters (3.8),
the Gurobi licence message (3.8), reward and payoff in one metrics row (4.4),
penalty persistence (5.2), the redundant `ema_dev` feature (5.4), the legacy
`train_rl.py` unconditional `remember()` (5.8), the thread count in the
manifest (5.9), duplicated SOC scaling (6.2), NaN/Inf checks inside the clearing
(6.3), absolute money tolerances (9.2), the broken matplotlib install in the
`energy_env` interpreter (9.4), `code_revision` and the 12-entry source list it
shipped with (10.2), and the global `*.csv` ignore rule (10.3). Each is a
separate change with its own regression surface.

## Verification

```
PYTHONPATH=src python -m pytest tests/ -q        # 1484 passed, 53 deselected, 1 xfailed, 698.81s
PYTHONPATH=src python -m pytest tests/test_counterfactual_eval.py tests/test_payoff_kpi.py tests/test_baseline.py tests/test_rl_spec.py -q
git apply --check -p1 tmp/refreeze_v5/frozen_patches.diff
python -m py_compile tmp/refreeze_v5/src/*.py
```

The one `xfail` is `test_a_blocks_realized_lmp_reaches_only_the_next_observation`,
which needs the v5 price reference.

The frozen patch has not been run against any test: it cannot be applied until
the re-freeze is authorised, so F4/F6/F9/F10/5.3/5.10 are `NOT_TESTED`.
