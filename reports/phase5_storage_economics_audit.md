# Phase 5 — Storage Economics Audit and Three Fixes

The three accounting defects recorded in `reports/finalization_audit.md` §6 are
fixed, each with a regression test that states the failure directly.

No MATD3 training was started. No action space, reward function or market rule
was changed. Phase 6 is untouched.

## 1. Files

| file | change |
|---|---|
| `src/eval_agents.py` | new `settle_capture`; `main()` captures and settles every arm; rolling reward reported under its own columns; `MONEY_COLUMN_MEANINGS` |
| `src/counterfactual_eval.py` | new `baseline_cache_key`, `_config_fingerprint`, `_defaults_fingerprint`, `_scenario_fingerprint`, `_prepare_config`; `validation_kpis` uses the full key and disables the cache if it cannot build one |
| `src/rl_bidding.py` | `_POLICY_PROVENANCE`, `_record_provenance`, `policy_provenance`, `reset_trained_policies`; action-spec check on load; provenance recorded on load and on train |
| `src/strategies/rl_bidding.py` | `_reject_foreign_cached_policies`, applied only to the shared cache |
| `tests/test_storage_economics_audit.py` | 33 new tests |
| `reports/phase5_storage_economics_audit.md` | this report |

## 2. Defect 1 — the evaluation CLI published a learning signal as a payoff

### How it reproduced

`eval_agents.run_episode` accumulates the environment's per-block rewards into
`agent_profits`, and `main()` published `profit_delta` from that dict.
`evaluate_policy_dir` used `counterfactual_eval.settle_day` instead. Both entry
points wrote the same column name for different quantities.

The difference is not a rounding matter. One native day, truthful baseline,
full-day episode:

| unit | rolling reward (old CLI) | settled payoff (research KPI) |
|---|---|---|
| ESS9 | 97.12 | 49.68 |
| ESS14 | 132.97 | 45.01 |
| ESS17 | 205.20 | 40.62 |
| ESS27 | 107.51 | 70.55 |
| **fleet** | **542.80** | **205.86** |

The rolling sum is **2.6×** the settled payoff, and — worse than the scale — it
**ranks the units in exactly the opposite order**: by rolling reward ESS17 is
the best unit and ESS9 the worst; by settled payoff ESS27 is best and ESS17
worst. A study reporting the old column would have drawn the wrong conclusion
about which storage unit was earning.

The reason is structural: the rolling sum is a per-block cash tally with no
endpoint pin, so a unit that finishes the day with an emptier battery collects
more.

### How it is fixed

`settle_capture(agents, config, rl_names, capture)` calls the same
`counterfactual_eval.settle_day` that `evaluate_policy_dir` calls, on the same
captured declarations. `main()` now captures every arm and settles each one:

- `baseline_profit`, `rl_profit`, `profit_delta` → **settled**, identical in
  meaning to the API's columns;
- `baseline_rolling_reward`, `rl_rolling_reward`, `rolling_reward_delta` →
  the learning signal, under its own names;
- `settled_clean` → whether both arms settled without a degraded or
  fallen-back clear. An unclean row reports NaN profits, never zeros.

`MONEY_COLUMN_MEANINGS` records each column's definition in the module, so a
reader of `results/*_eval.csv` need not recover it from source.

### Evidence

```
tests/test_storage_economics_audit.py::test_settle_capture_is_the_shared_settlement
tests/test_storage_economics_audit.py::test_the_settled_payoff_is_not_the_rolling_reward
tests/test_storage_economics_audit.py::test_the_rolling_reward_is_not_the_research_kpi
tests/test_storage_economics_audit.py::test_a_degraded_capture_settles_to_nan_not_zero
tests/test_storage_economics_audit.py::test_both_money_families_are_present_and_separately_named
```

The first asserts the CLI helper returns exactly what `settle_day` returns. The
second fails if the two ever coincide, so it cannot silently stop demonstrating
anything.

The first baseline arm is settled once for the whole matched set: the truthful
arm declares truthfully for every agent, so naming a subset clears identically
and one settlement serves both the per-agent and combined rows.

## 3. Defect 2 — the baseline cache was blind to the fleet

### How it reproduced

`_BASELINE_CACHE` was keyed `(scenario, day_id, T)`. Fleet capacity and siting
come from `defaults.yaml:independent_storage`, **not** from the `MarketConfig`
object, so a sweep changing them passes an identical key and received the
previous fleet's baseline. `assert_paired` could not catch it: it compares the
wholesale arrays, which a fleet change does not move.

### How it is fixed

`baseline_cache_key` now carries ten components:

| component | why |
|---|---|
| scenario | the load |
| scenario declaration digest | scale, load shape, renewables, multipliers, `industry_map` |
| `day_id`, `T` | which day, how long |
| config digest | every bound field at once |
| `defaults.yaml` digest over `profiles`, `independent_storage`, `prosumers`, `load_types`, `non_prosumer_storage`, `price_curve`, `wholesale_market`, `network` | **where the fleet lives** |
| the scenario's declared price source | which market prices the day |
| the agents being settled | which entries the result carries |
| obs/action spec names | the environment the arm is built in |

Two implementation points worth stating:

- **Digests are content-derived.** `dataclasses.asdict` + sorted JSON, hashed —
  never `id()` or `repr()`. A deep copy of an equal config shares the key, which
  a test pins.
- **The price source is read from the scenario's declaration, not the live
  global.** `_build_scenario` rewrites that global on every build, so keying on
  it would make the key depend on which scenario was built last. The same
  applies to `wholesale_market.day_id`, so both are excluded from the defaults
  digest and carried explicitly instead.

If the key cannot be built, the cache is **disabled** for that call rather than
keyed on something partial — a baseline reused across two fleets is a wrong
number that looks like a right one.

### A second bug found while fixing this one

With the new key the cache never hit at all. Cause: `_build_scenario` applies
the scenario's `override_config` to the config object **in place**, so a key
computed before the first build described a different configuration from one
computed after it. `_prepare_config` now applies the scenario's overrides
first, making the key describe the configuration the arm actually runs under;
`apply_scenario_overrides` is idempotent, so the build's later application
changes nothing. This is exactly the class of silent staleness the fix exists
to remove, and it would have made the cache *useless* rather than *wrong* — a
failure mode worth naming because it is the benign one.

### Evidence

```
::test_identical_inputs_produce_an_identical_key
::test_different_config_objects_with_equal_content_share_a_key
::test_a_config_differing_in_content_moves_the_key
::test_a_capacity_change_moves_the_key
::test_a_siting_change_moves_the_key
::test_a_price_source_change_moves_the_key
::test_day_T_and_agent_set_all_move_the_key
::test_the_spec_identity_moves_the_key
::test_the_key_does_not_read_a_mutable_global
::test_a_cached_baseline_is_reused_for_identical_input_and_not_for_a_new_fleet
```

The last is the end-to-end statement: it counts how many times the truthful arm
is actually cleared, asserts two calls on identical inputs clear it **once**
(the second reuses), and that a capacity change makes it clear **again**.

## 4. Defect 3 — the policy cache could not tell one `ESS9` from another

### How it reproduced

`_TRAINED_POLICIES` was a `Dict[str, Actor]` keyed by agent name, `update`d
rather than replaced. Independent-storage names are identical in every scenario,
so a policy trained for one day applied to another with no error. The only guard
was the observation-spec name recorded in the checkpoint.

### How it is fixed

`_POLICY_PROVENANCE` records, per agent: the source (a path, or `train:<config
hash>`), the observation and action spec names **the policy was built with**
(taken from the checkpoint's own `_meta`, not the caller's), and a generation
counter. `reset_trained_policies()` clears the cache and the provenance
together, so a network can never be left with no recorded origin.

`load_policies` now also checks the recorded **action** spec, alongside the
existing observation check, and raises `ValueError` on a mismatch.

`strategies/rl_bidding._reject_foreign_cached_policies` refuses a cached policy
whose observation spec is not the one this environment builds. It is applied
only to the **shared cache** — policies passed explicitly through
`self.policies` are the caller's decision, not a silent reuse. A policy with no
recorded provenance (an older checkpoint) is not refused; the loader has already
checked it against the file's own metadata.

### Evidence

```
::test_the_cache_records_where_each_policy_came_from
::test_a_second_file_wins_for_the_same_agent_name
::test_reset_clears_the_cache_and_the_provenance_together
::test_a_foreign_observation_spec_is_refused
::test_a_policy_with_no_recorded_provenance_is_not_refused
::test_a_matching_spec_is_accepted
::test_an_action_spec_mismatch_on_load_raises
::test_an_observation_spec_mismatch_on_load_raises
```

`test_a_second_file_wins_for_the_same_agent_name` writes two checkpoints
carrying the same agent name, gives the second distinguishable weights, loads
them in order, and asserts both the recorded source **and the cached network's
weights** are the second file's.

## 5. The two distinctions that were deliberately kept

**`PayoffBreakdown.other_cost` is never assigned and is always zero.** Confirmed
by reading the code and pinned by a test. **No cost was invented to fill it**,
and no value was set without a basis. The consequence for reporting is stated
rather than papered over: the breakdown covers revenue, purchase,
generation, degradation and penalty, and **does not** cover 输配电价, 线损,
系统运行费, 政府性基金及附加, deviation charges, capacity or cost compensation,
or imbalance funds — the same limits `reports/jiangsu_market_mechanism.md`
records from the policy side. A result must be called a **modelled storage
day-ahead energy profit**, not a settlement bill.

**The training reward and the settled payoff are different counterfactuals by
design.** Nothing was merged. The reward is an in-window differential against a
local truthful baseline with opponents held at their learned bids; the payoff is
a whole-day, all-truthful, endpoint-pinned settlement. Both are now named
distinctly everywhere: the reward in the training path, the payoff in the
evaluation path, and the two families separated by column name in the
evaluation CSV.

## 6. Accounting invariants pinned

| invariant | test |
|---|---|
| the day's profit comes from the one shared settlement function | `test_settle_capture_is_the_shared_settlement` |
| both arms are priced on the same curve and scenario | `test_a_cached_baseline_...` (via `assert_paired`) |
| `dt = 0.25 h` and one money unit | `test_the_period_is_a_quarter_hour` |
| the breakdown reconciles term by term | `test_the_payoff_breakdown_reconciles` |
| a battery is not charged for load or generation | `test_a_storage_unit_is_not_charged_for_load_or_generation` |
| SOC endpoints pinned identically in both arms | `test_both_arms_settle_with_the_endpoints_pinned` |
| the pin is forced by the settlement, not the config | `test_the_settlement_pins_the_endpoints_regardless_of_the_config` |
| the cache does not leak across fleets | `test_a_cached_baseline_is_reused_...` |
| the same name does not leak across policy dirs | `test_a_second_file_wins_for_the_same_agent_name` |
| phases 1, 2 and 4 interfaces still hold | `test_phase_1_interface_is_intact`, `test_phase_2_interface_is_intact`, `test_phase_4_scenario_is_intact` |

## 7. Test results

```
Python: 3.12.7 (C:\ProgramData\anaconda3\python.exe), Gurobi academic licence

tests/test_storage_economics_audit.py                       -> 33 passed
tests/test_counterfactual_eval.py + participant_payoff + payoff_kpi
  + surplus_metrics + baselines + scenario_scale + rl_spec
  + agent_action_spaces + action_spec_consistency + money_units
                                                            -> 130 passed, 9 deselected
tests/ (whole suite)                                        -> 699 passed, 51 deselected, 0 failed (6m07s)
```

699 = 666 at the end of Phase 4, plus the 33 new.

**Note on the instruction's test list:** `tests/test_rl_bidding.py` **does not
exist** in this repository (the Phase-0 audit's 46-module inventory does not list
it either). Passing it to pytest aborts collection with "collected 0 items", so
the nearest existing modules were run instead: `test_rl_spec.py`,
`test_agent_action_spaces.py`, `test_action_spec_consistency.py`,
`test_money_units.py`. No test was skipped or weakened.

**Defects found during this phase were all in my own test code**, and are fixed:
a comparison against a freshly-read config when the fixture's config had
legitimately been mutated by the scenario build; `ObservationSpec.to_dict()`
round-tripped into a constructor that does not accept its derived keys (fixed
with `dataclasses.replace`); an assertion that was too literal about a
docstring; and `validation_kpis(actors={})` called with no policies. The one
exception is the `_prepare_config` bug in §3, which was a real defect in the fix
itself, found by the test that was written to catch exactly that class of error.

## 8. Remaining limitations

1. **Comparability across price regimes is unaddressed here.** The settled
   payoff is comparable between two arms on the same day. It is not comparable
   across the `dianyicha` mean curves and the synthetic library, which are
   different markets. That is a reporting discipline, not a code guard.
2. **The evaluation CLI's per-agent rows settle one agent at a time; the
   combined row settles the fleet together.** Both are correct for their
   question and they are not the same counterfactual. The `agent` column
   distinguishes them, but a reader who ignores it can still subtract the wrong
   pair.
3. **`_prepare_config` mutates the caller's config**, as `_build_scenario`
   already did. It is idempotent and now happens earlier, but a caller that
   reuses one `MarketConfig` across scenarios still hands over a shared mutable
   object.
4. **The cache key is not free.** It rebuilds three JSON digests per lookup,
   including the whole scenario declaration. Measured cost is small against a
   clear, but it is not zero, and `use_cache=False` remains available.
5. **`other_cost` stays zero.** The breakdown still does not cover the
   out-of-model charges listed in §5, and no attempt was made to estimate them.
6. **The policy provenance is per-process.** It is not persisted with the
   checkpoint beyond the obs/action spec names, so a fresh process loading an
   older file has no source identity beyond the path it was given.

## 9. Difference from the pre-Phase-5 workspace

Backed up to `C:/Users/26036/AppData/Local/Temp/pre_phase5_worktree.diff` before
any edit (1,195 lines). HEAD `a1151fd`, branch `master`, unchanged; no
`reset`, `checkout --` or `clean` was run; `gurobi.env` was not read, copied or
printed.

Four of the files edited this phase (`eval_agents.py`, `rl_bidding.py`,
`scenarios.py`, `strategies/rl_bidding.py`) carried the user's own uncommitted
changes, so a whole-file diff cannot prove nothing was lost. Every pre-existing
hunk was compared line by line instead:

```
src/config_loader.py         rm  14 ->  14   add   0 ->   0   OK
src/data_aggregator.py       rm   5 ->   5   add   1 ->   1   OK
src/diagnose_typical_day.py  rm   0 ->   0   add   1 ->   1   OK
src/dispatch_core.py         rm  17 ->  17   add   0 ->   0   OK
src/llm.py                   rm  23 ->  23   add   1 ->   1   OK
src/mpc_storage.py           rm 229 -> 229   add   0 ->   0   OK
src/price_forecaster.py      rm  98 ->  98   add   7 ->   7   OK
src/rl_bidding.py            rm   3 ->   5   add   0 ->  76   OK
src/rl_env.py                rm   4 ->   4   add   0 ->   0   OK
src/rl_td3.py                rm   5 ->   5   add   0 ->   0   OK
src/scenarios.py             rm  15 ->  15   add 101 -> 101   OK
src/strategies/rl_bidding.py rm   4 ->   5   add   0 ->  39   OK
src/eval_agents.py           rm   1 ->  13   add   1 -> 115   OK
src/wholesale_price.py       rm   2 ->   2   add 108 -> 108   OK
src/grid.py                  rm   5 ->   5   add  42 ->  42   OK
config/scenarios.yaml        rm   0 ->   0   add  98 ->  98   OK
.gitignore                   rm   0 ->   0   add   6 ->   6   OK
```

Every line the user had deleted is still deleted and every line they had added
is still present. The extra removals in `rl_bidding.py`, `strategies/rl_bidding.py`
and `eval_agents.py` are lines **I** replaced while editing.

## Status

Phase 5 complete. Stopping here for review; Phase 6 is not started.
