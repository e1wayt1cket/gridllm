# S6: the B1–B6 repairs

Each item, what it was in this repository, what changed, and what holds it. One
defect found while doing B6 is reported and was fixed; two things are reported
and deliberately left alone.

**Baseline**: `HEAD = 9f61c9a`, branch `master`. Nothing was committed or pushed.

The task book's `B1–B6` labels do not appear in the repository; the mapping from
each label to the code below was established in S0 and confirmed against the
task book's own text for this stage.

## B1 — the horizon is stated, never inferred

**What it was.** `pins_terminal_soc(config, T, horizon_type="auto")` guessed the
horizon from the period count:

```python
if horizon_type == "full_day": return True
if horizon_type == "window":   return False
return T == money.PERIODS_PER_DAY     # the guess
```

The guess is wrong in both directions, and each way is silent. A rolling window
that happens to be a whole day long is pinned, which forbids the trajectory the
window was opened to choose. A whole day cleared at any other period length is
not pinned, so a settlement carried a terminal-SOC term it was supposed to
compare away. Every one of this repository's twelve whole-day call sites passed
no `horizon_type` at all and got the answer from the arithmetic.

**What changed.**

| where | what |
|---|---|
| `dispatch_core.pins_terminal_soc` | `horizon_type` is now required, `T` is gone, `"auto"` is gone from `HORIZON_TYPES` |
| `dispatch_socp.solve_socp_opf_batch` | `horizon_type` is keyword-only and required |
| `dispatch_ldf.solve_lindist_opf_batch`, `_CachedOPFModel` | same |
| `market.clear_market` | same |

`T` was dropped rather than left unused: with the inference gone it had no other
reader, and a parameter that cannot change the answer is worse than no
parameter. Removing `"auto"` means a caller cannot buy the wrong semantics by
passing a familiar length — a missing argument is a `TypeError`, not a guess.

Twelve in-repo call sites now say `horizon_type="full_day"`: `run.py` (×4),
`ui/sim_page.py` (×4), `nash.py`, `stackelberg.py`, `compare_methods.py` (×2),
plus the batch call in `tests/test_baseline.py`. The rolling paths already said
`"window"` and are unchanged.

**New tests.** `tests/test_payoff_kpi.py` now asks for both horizons at 96
periods and asserts they disagree, and that `"auto"` is rejected.
`test_end_to_end_acceptance.py::test_a_96_period_window_is_not_the_same_statement_as_a_96_period_day`
clears the same agents over the same 96-period day twice — once as a settled day
and once as a rolling window — and asserts `soc_terminal_target` is set in the
first and `None` in the second. Under the old inference both were pinned and the
second assertion could not have been written.

## B2 — a mixed price field says so

**What it was.** `lmp_fallbacks` was a bare integer, and only the two batch
engines produced it. It reported how many cells came off the wholesale curve and
nothing else: not which periods, not why. Two holes were worse than the
coarseness:

- `market.py`'s per-period path emitted **no** `lmp_fallbacks` key at all, so a
  reader using `.get("lmp_fallbacks", 0)` — which several do — read zero, "every
  price is nodal", on the one path whose prices are carried forward whenever a
  solve fails.
- The two rolling paths discarded the windows' fallback counts entirely, so a
  day assembled from eight windows reported nothing about any of them.

**What changed.** One helper, `dispatch_core.lmp_fallback_report`, emits five
fields together, and every clearing path returns them:

| field | meaning |
|---|---|
| `lmp_fallbacks` | cells that did not come from the period's own clearing (unchanged meaning) |
| `lmp_fallback_periods` | the distinct periods containing at least one such cell |
| `lmp_fallback_reason` | which cause, named |
| `has_wholesale_fallback` | bool |
| `lmp_price_source` | `"nodal"` or `"mixed_wholesale_fallback"` |

The two causes are named separately (`LMP_FALLBACK_NO_DUAL`,
`LMP_FALLBACK_UNSOLVED`) because a period priced off the wholesale curve and a
period whose solve never returned call for different responses, and the price
column cannot tell them apart. The rolling paths collect causes as a set rather
than taking the last window's: different windows can fail differently.

**What a mixed field may be called.** `counterfactual_eval.is_degraded` is now
the one definition of "this is not a settled result", used by both the
block-level reward loop and the day-level settlement. It returns true for a
fallback **and** for a mixed price field. The second was invisible before: a day
settled on a partly-wholesale price field was reported clean, and its payoff
went into the paired comparison. `unified_eval` marks such a record with its own
exclusion reason (`mixed_wholesale_price_fallback`) rather than filing it as a
solver failure — nothing went wrong with the solve — and every record, clean or
not, now carries the price field's state.

**New tests.** `tests/test_lmp_fallback_report.py` (11). The condition the report
exists for is forced at the dual itself — `money.price_from_balance_dual` is made
to raise every third call — so a real fallback travels the whole path from the
solver to the result dict rather than being simulated by handing the engine a
dict. No scenario in the suite falls back naturally, which is why the injection
is needed; a probe over `exclusive_mode` in `binary` / `strategy_only` / `off`
returned `fallbacks=0` and `source=nodal` in all three.

## B3 — a participant's draw repeats across processes

**What it was.** `np.random.RandomState(hash(a.name) % 2**31)` in two places:
`dispatch_core._run_mpc_pass` (the MPC price-forecast noise) and
`strategies/random_bidding.RandomStrategy`. CPython salts `hash()` on a string
per process, so the same configuration drew a different stream in every process.
Inside one process the expression looks perfectly stable, which is why nothing
caught it.

**What changed.** `src/agent_seed.seed_for_agent(name, seed=0)` — a blake2b
digest of the identity — used at both sites. A test asserts the old expression
really does move between two hash salts, so the contrast is measured rather than
assumed.

**Stated limit.** No caller supplies the `seed` argument. Neither call site has
an experiment seed reachable from its parameters: `MarketConfig` carries no seed
field at all, `RandomStrategy.formulate` takes no seed, and inventing a config
knob that nothing sets would claim a control that does not exist. The argument
is there so a caller that owns a seed can vary the draw without renaming anyone;
today the stream is a function of the participant's name alone.

## B4 — one carbon formula

**What it was.** `market.py` charged carbon inline on the per-period path and
again on the RT-rolling path, both with the period length written out as a
literal, neither through `grid_carbon_tco2` — the helper both batch engines use.

**What changed.** Both now accumulate the day's exchange and call the helper
once. A period that failed to solve left its exchange at zero, which is exactly
what the loop used to add for it, so the reported number is unchanged. On today's
constants the two expressions are numerically identical; **what this buys is one
formula, not a different number.** The literal is the part that could have
drifted: the helper reads `DT_HOURS`, and a path with its own `0.25` would not
follow it.

**New tests.** `tests/test_money_units.py` gains two:
`test_carbon_charges_the_import_part_of_the_exchange_only` (positive, reverse and
zero exchange — there was no test of this helper before) and
`test_the_per_period_path_reports_carbon_through_the_shared_rule`, which clears
through `opf_mode="dc"` and compares the reported carbon with the helper applied
to the exchange the result publishes.

**Not changed.** `clear_da_rolling` aggregates each window's carbon weighted by
its commit fraction. It reads the engines' carbon, so the *formula* is the
shared one; the aggregation is its own and re-weighting it is a different
decision from unifying a formula.

## B5 — the provider cache follows the configuration

**What it was.** `market_opportunity._apply_spec` moves the day's levers into the
process-global config and reloads both config caches. `build_report` restores
the config at the end for the same reason. Neither touched
`wholesale_price._PROVIDER_CACHE`, whose entries are built from the configuration
in force at the moment they are created. A provider built while an audit had its
levers set went on answering callers in that process after the config was
restored.

**What changed.** `build_report` resets the provider cache at its end, beside the
two config reloads, and again at its start. The head reset is what covers an
audit that raised before reaching its own restore — the plan asked for a
`finally`, and a reset on the next entry reaches the same observable outcome
without re-indenting a 190-line function that has no other error handling to
move into it.

**Reproduced, then fixed, then verified.** With both resets removed,
`test_the_audit_leaves_no_price_provider_built_under_its_own_levers` fails with
`assert <LegacyMeritOrderProvider object at 0x...> is not <LegacyMeritOrderProvider object at 0x...>` —
the same instance, still serving. With them in place it passes.

**Stated limit.** `_apply_spec` was left alone. No provider reads a lever that
`_apply_spec` writes — none of them reads `wholesale_market.day_id` at
construction, and the day is an argument — so a reset there would rebuild the
provider for every day of an ensemble and buy nothing. For `dianyicha`, whose
workbook read is lazy per instance, that would be one 67 MB re-read per day. If a
provider is ever given a construction-time lever that a spec writes, the reset
has to move there too.

## B6 — the CLI and the API settle the same day the same way

**What it was, verified rather than assumed.** Both entry points already call
`counterfactual_eval.settle_day`; the CLI through `settle_capture`, the API
through `evaluate_policy_dir`. What was not shared was everything the call needs.

**The defect this found.** `eval_agents --output` **crashed**. The per-episode
aggregation names a column per statistic `summarize_payoff` returns
(`baseline_profit_std`, `profit_delta_p25`, …) and appends them to the row, while
the writer was handed the fixed `header` list, which declares none of them.
`csv.DictWriter` refused the rows it had just built:

```
ValueError: dict contains fields not in fieldnames: 'baseline_profit_std',
  'profit_delta_n', 'settled_clean_median', ... (111 columns)
```

Reproduced with an untrained V4 policy directory before the fix; the CLI now
writes its CSV. The header is the declared columns in their declared order,
followed by whatever the aggregation produced — declaring the distribution
columns here would mean declaring them twice and letting the two drift, which is
how this write came to reject its own output.

**New tests.** `tests/test_eval_cli_api_consistency.py` (4) runs both entry
points on the same untrained policies, scenario and seed, and compares
`baseline_profit`, `rl_profit`, `profit_delta` and the four
clearing-objective columns. They agree to `1e-6`. It also asserts that neither
settlement helper reaches for a second path (`clear_market(` appears in neither).

**Stated limit, reported not changed.** The CLI's *default* single-episode path
is unseeded — `paired = (eval_seed is not None) or n_eps > 1`, so with one
episode and no `--eval-seed` the baseline and the fleet clear against
different wholesale days and every delta carries that spread. The API is always
seeded and paired. The test therefore passes `--eval-seed`, which is what "同一
seed 下" means; making the CLI seed by default would move published CLI numbers,
which is a decision about results rather than about a defect.

## Test results

Per-file, from the runs in this session:

```
PYTHONPATH=src pytest -q tests/test_agent_seed.py                     6 passed
PYTHONPATH=src pytest -q tests/test_lmp_fallback_report.py           11 passed
PYTHONPATH=src pytest -q tests/test_eval_cli_api_consistency.py       4 passed  (40.56s)
PYTHONPATH=src pytest -q tests/test_market_opportunity.py            56 passed
PYTHONPATH=src pytest -q tests/test_money_units.py                    7 passed, 5 deselected
PYTHONPATH=src pytest -q tests/test_payoff_kpi.py tests/test_money_units.py
                                                                     18 passed, 5 deselected
PYTHONPATH=src pytest -q tests/test_baseline.py \
    tests/test_clearing_objective_schema.py tests/test_storage_exclusivity.py \
    tests/test_typical_day_physical_baseline.py                      40 passed, 10 deselected
PYTHONPATH=src pytest -q tests/test_end_to_end_acceptance.py \
    -k "96_period_window or terminal"                                 3 passed
PYTHONPATH=src pytest -q tests/test_eval_smoke_24.py                  5 passed  (37.16s)
```

Full suite, exit code 0:

```
PYTHONPATH=src pytest -q
1044 passed, 53 deselected, 17 warnings in 387.31s (0:06:27)
```

1044 against S5's 1019: +6 `test_agent_seed`, +11 `test_lmp_fallback_report`,
+4 `test_eval_cli_api_consistency`, +2 carbon in `test_money_units`, +1 the
provider cache in `test_market_opportunity`, +1 the 96-period window in
`test_end_to_end_acceptance`. One test in `test_payoff_kpi.py` was rewritten
rather than added, so it moves the count by nothing. The 53 deselected are the
`slow` tests, not skips.

## Files

New: `src/agent_seed.py`, `tests/test_agent_seed.py`,
`tests/test_lmp_fallback_report.py`, `tests/test_eval_cli_api_consistency.py`.

Changed: `src/dispatch_core.py`, `src/dispatch_socp.py`, `src/dispatch_ldf.py`,
`src/market.py`, `src/market_opportunity.py`, `src/counterfactual_eval.py`,
`src/unified_eval.py`, `src/eval_agents.py`, `src/run.py`, `src/nash.py`,
`src/stackelberg.py`, `src/compare_methods.py`, `src/ui/sim_page.py`,
`src/strategies/random_bidding.py`, and the call sites in
`tests/test_baseline.py`, `tests/test_clearing_objective_schema.py`,
`tests/test_money_units.py`, `tests/test_storage_exclusivity.py`,
`tests/test_typical_day_physical_baseline.py`, `tests/test_payoff_kpi.py`,
`tests/test_end_to_end_acceptance.py`, `tests/test_market_opportunity.py`.

## Still open

1. **The reading unit is `unit_unverified`** — unchanged, and S6 was not
   permitted to resolve it.
2. **No real price date intersects a real load date** — unchanged.
3. **Only 30 days (all December 2022) cover all 17 mapped buses** — the formal
   date-selection rule is still undecided.
4. **B3's `seed` argument has no supplier**, and **the CLI's default
   single-episode path is unseeded** — both stated above rather than papered
   over.
5. **`make_results_figure.py` still does not import** (`DEFAULT_ACTION`, missing
   from `diagnose_profit`) and **`rl_training.py:80`** still carries a
   single-name scenario default. Both were reported by S5 and are untouched
   here.

## Gate status

Unchanged by this stage: the reading unit is `unit_unverified`, no real price
date intersects a real load date, and the formal date-selection rule is still
undecided. **S8 and formal MATD3 training remain blocked** — on those three, not
on anything S6 touched.
