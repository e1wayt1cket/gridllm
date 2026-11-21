# The single-day case on the runtime chain

How the node-load ledger and the real price curve reach a scenario and a
network, where the gate sits on that path, what the two end-to-end suites prove,
and what happened when it was run for real.

**The headline:** the path is wired, active, and exercised on a real clearing.
The clearing **solves** and the network **cannot carry the load** — 3.23% of it
is served. Both facts are reported rather than repaired.

## The chain, as it now runs

```
config/scenarios.yaml
    typical_day_industry:
      load_model:                      mode: absolute_industry_mw
                                       bus_load_policy: {33 buses}
      single_day_case:                 case_type: cross_date_diagnostic
                                       ledger / ledger_manifest / price_sheet
        │
scenarios.get_scenario(name, T, config)
  └── _build_scenario
        ├── single_day_case(name, T)              ← the runtime boundary
        │     ├── single_day_data.load_case(case_type, ledger, manifest, sheet)
        │     │     ├── apply_date_rule        the case type decides
        │     │     ├── load_load_axis         the ledger's own axis
        │     │     ├── load_price_axis        the curve's own axis
        │     │     └── check_alignment        the two together, or a refusal
        │     ├── case.require_load_for_build()   ← THE GATE
        │     └── the scenario's declared bus policies vs the ledger's
        │
        ├── case.bus_load_mw_for_build()          {bus: array(96)}, gated
        ├── build_base_network(config)
        └── grid.create_agents_from_network(..., node_load_mw=<that mapping>)
```

The gate runs **before `build_base_network`**, so a refusal cannot leave a
half-built network behind.

## The config is load-bearing in both directions

| declaration | what reads it | what would happen if it were wrong |
|---|---|---|
| `load_model.mode: absolute_industry_mw` | `_build_scenario` branches on it | the scenario would build from a shape and this whole path would be inert |
| `load_model.bus_load_policy` (33 buses) | `single_day_case` compares it to the ledger's own per-bus policies | a config that disagreed with the artifact raises, rather than the two drifting |
| `single_day_case.case_type` | `apply_date_rule` | unstated or unknown raises; there is no default and no permissive fallback |
| `load_scale_mode: ledger` | `scenario_scale`, and `create_agents_from_network` skips the target entirely | `native_peak` here would imply the load could be rescaled to a target, which is the one thing the ledger forbids |

## What was removed, because it became a second route

`typical_day_industry` used to carry industry load as a per-bus **shape**
(`load_shape: industry` + `industry_map` + `scenarios._industry_load_shapes` +
`grid`'s `industry` branch + the shapes file). With the ledger active that route
was unreachable and would have been a second way to put industry load into the
network — one that, by construction, cannot express a magnitude.

Deleted: the `industry_map` block and `load_shape: industry` from the scenario,
`_industry_load_shapes`, the `load_shapes` parameter and the `industry` branch in
`_make_load_profile`, and `tests/test_industry_scenario.py` (whose entire subject
was that route). `industry_load.load_shapes_csv` is kept — it reads a general
artifact and `accept_typical_day_industry` still uses it.

## The gate, and what it now refuses

`require_load_for_build()` refuses:

1. **a load with a gap** — the axis check catches it first, and the extraction
   would catch it after: a missing bus or period is not a zero;
2. **a ledger with no recorded unit basis** — nothing would say what the
   resulting figures are in.

It **accepts a ruling-based unit**. That is the deliberate change: the reading
unit is a user-confirmed project ruling, that is a recorded basis, and a
diagnostic run may be built on it — labelled. What it does not get is to be
called verified.

`require_formal_load_mw()` is unchanged and still strict: the unit must be
stated by an outside source before the ledger may be *published* as the
network's absolute load. Two claims, kept apart.

### The refusal, observed

```
scenarios.py, line 591, in get_scenario
    return _build_scenario(name, T, config, ess_buses=ess_buses, ...)
scenarios.py, line 364, in _build_scenario
    node_load_mw = single_day_case(name, T).bus_load_mw_for_build()
scenarios.py, line 305, in single_day_case
    case.require_load_for_build()
single_day_data.py, in require_load_for_build
    raise CaseTypeError(...)
```

A test replaces `build_base_network` and `create_agents_from_network` with
recorders and asserts neither was reached — the ordering is checked, not
described.

## The real run

```
$ PYTHONPATH=src python src/run_single_day_diagnostic.py --json outputs/single_day_diagnostic_run.json
case        cross_date_diagnostic  2022-11-30 / 2026-10-09
unit basis  user_confirmed_project_ruling (externally verified: False)
served      354.8 MWh of 10,995.6 (3.23%)
shed        10,640.8 MWh
voltage     min 0.9300 (at lower bound: True)
worst line  1.0000 (at limit: True)
fallbacks   0   fell_back False
```

| | |
|---|---|
| solver status | `OPTIMAL` — the clearing solves; it is not infeasible |
| served / demanded | **354.8 / 10,995.6 MWh — 3.23%** |
| shed | **10,640.8 MWh** |
| bus voltage | min **0.9300** at bus 24, period 72 — the band's own lower bound |
| line utilization | max **1.0000**, 32 line-periods at or over their limit |
| PCC | import max 30.0 MW; no export periods |
| nodal-price fallbacks | **0**; the price field is the clearing's own duals |
| renewable rate / carbon | reported in the JSON and the run report |

**What is binding**: the lower voltage bound and the lines' thermal limits. The
feeder is held at both, so the shed figure is a statement about the **network's
capacity at this load**, not about the price or the fleet. Nothing was scaled,
widened or relaxed to change it.

**The load is the real aggregate**: 370.9–526.2 MW over the day, mean 458.2 MW,
against a feeder whose own nominal demand is 3.715 MW.

## The two end-to-end suites

`tests/test_single_day_runtime.py` (17).

### Group 1 — a test-only fixture

A ledger invented in the test, marked `formal_ledger` / `unit_verified`, with a
price curve the test supplies through the same code path the real one takes.

| what it proves |
|---|
| `get_scenario` → entry point → network, with a spy confirming `load_case` is what paired the data |
| every load bus carries **exactly** the ledger's value — the failure this guards is the ledger's load *added to* the native load |
| the ESS fleet on the same buses carries **zero** load |
| the device sizing is unchanged from a ledger-free build |
| a bus omitted from the ledger raises rather than defaulting |
| a ledger with a gap is refused, and the refusal precedes the network |

**Nothing in group 1 is a result.** The manifest says `reading_unit_evidence:
"test fixture, not evidence"` and a test asserts its digest is not the real
ledger's.

### Group 2 — the real ledger

| what it proves |
|---|
| `cross_date_diagnostic` passes the date rule (2022-11-30 / 2026-10-09) |
| the build gate **accepts** it, on `user_confirmed_project_ruling` |
| the strict `require_formal_load_mw` still **refuses** it |
| a ledger-backed scenario that omits `case_type` is refused |
| `same_day_replay` on these dates is refused |
| the refusal is not a zero load and there is no second method that hands one over |
| the ledger and its manifest are byte-identical before and after |

### The acceptance list

| # | required | where |
|---|---|---|
| 1 | `absolute_industry_mw` reaches the network | `test_the_declared_mode_reaches_the_network_and_only_one_mode_exists`, `test_the_whole_path_runs_and_carries_the_ledgers_load` |
| 2 | no unused load-mode config, no second route | the same first test, plus the deletions above |
| 3 | all pairing through `load_case` | `test_every_module_that_pairs_a_ledger_with_a_curve_uses_the_entry_point` |
| 4 | dates may differ, semantics must agree | `test_different_dates_pass_a_cross_date_diagnostic`, the alignment suite |
| 5 | 33 bus policies and the v2 mapping effective | `test_the_scenarios_mapping_and_its_policies_reach_the_built_day` |
| 6 | a missing load is never a zero | `test_a_blank_final_load_is_carried_as_blank_and_never_zero_filled` |
| 7 | raw load neither scaled nor double-injected | `test_the_ledger_load_is_not_rescaled_and_does_not_resize_devices`, the fleet-zero test |
| 8 | no synthetic fallback | `test_there_is_no_synthetic_fallback` |
| 9 | a failed clearing fabricates no profit | `test_the_diagnostic_runner_reports_a_failure_rather_than_a_profit` |

## Files

| module | what changed |
|---|---|
| `config/scenarios.yaml` | `typical_day_industry`: `load_model` activated, `single_day_case` block added, `load_scale_mode: ledger`, `load_shape: native`, `industry_map` removed |
| `src/single_day_data.py` | `unit_basis`, `require_load_for_build`, `bus_load_mw_for_build`, `_rows_with_no_gap` |
| `src/scenarios.py` | `ABSOLUTE_INDUSTRY_LOAD_MODEL`, `SCALE_MODES` with `ledger`, `single_day_case` boundary, `last_single_day_case`; `_industry_load_shapes` removed |
| `src/grid.py` | `node_load_mw` path; the `industry` shape route and `load_shapes` removed |
| `src/run_single_day_diagnostic.py` | new — the real run and its report |
| tests | `test_single_day_runtime.py` new (17); `test_industry_scenario.py` deleted; `test_scenario_scale.py`, `test_storage_economics_audit.py`, `test_phase7_strategy_comparison.py`, `test_unified_eval.py` repointed |

### Tests that had to move, and why

`test_phase7_strategy_comparison.py` and `test_unified_eval.py` used
`typical_day_industry` as *a light native day priced from the real curve*. That
combination no longer exists: the scenario now carries the metered load. Both
now run on `typical_day` and set the price source to `dianyicha` per-run, which
reproduces the fixture they were written for. `phase7`'s own `SCENARIO` had to be
patched as well — the module reads its own constant, so setting only the test
file's left it comparing a native day against a metered one.

## Test results

```
PYTHONPATH=src pytest -q tests/test_single_day_runtime.py                  17 passed

PYTHONPATH=src pytest -q tests/test_end_to_end_acceptance.py              44 passed
PYTHONPATH=src pytest -q tests/test_phase9_sensitivity.py                 18 passed
PYTHONPATH=src pytest -q tests/test_phase7_strategy_comparison.py         21 passed
PYTHONPATH=src pytest -q tests/test_scenario_scale.py     tests/test_storage_economics_audit.py                                  63 passed
PYTHONPATH=src pytest -q tests/test_eval_cli_api_consistency.py            4 passed
PYTHONPATH=src pytest -q tests/test_industry_bus_mapping.py     tests/test_map_industry_meters.py tests/test_node_load_builder.py     106 passed

PYTHONPATH=src pytest -q
1128 passed, 53 deselected, 17 warnings in 402.24s (0:06:42)   exit code 0
```

1137 → 1128 is not a loss: `tests/test_industry_scenario.py` (a file whose entire
subject was the retired shape route) was deleted, and the suites it overlapped
with gained the coverage above.

### Two things the path found that are worth keeping

**A frozen default.** `phase9_sensitivity._clear(day_id, scenario=SCENARIO)`
bound its default **at import**, so patching the module constant changed
nothing and the scans kept pricing the old day with no sign of it. It now
resolves the scenario at call time. The same shape is what S4 removed elsewhere.

**A declaration that was load-bearing.** Deleting `industry_map` looked like
removing the shape route; it also removed the bus→industry assignment that
`map_industry_meters` reads to decide where metered users may land, and the
`excluded_industries` list that is part of that rule's digest. Three tests caught
it. The block is back, minus `shapes_file` (which belonged to the shape route).

## Still open

1. **The reading unit is `unit_unverified`.** A diagnostic run may now be built
   on the ruling; a **formal** one may not. `require_formal_load_mw` and
   `validate_formal_ledger` both still refuse.
2. **No real price date overlaps a real load date.** The single-day case pairs
   2022-11-30 with 2026-10-09 by decision, and says so on its face.
3. **The feeder cannot carry this load**, so the case is a diagnostic about
   capacity and not a market result. That is the finding, not a defect to fix by
   scaling.
4. **`typical_day_industry` can no longer serve as a clean fixture.** Seven
   test files used it as "a light day priced from the real curve"; each was
   repointed to `typical_day` with the price source set per-run, or to the
   artifact that now carries the fact. Two assertions moved with it rather than
   being weakened silently: the native day's knife-edge margin (bus 9 now clears
   by ~29.8 CNY/MWh on the real curve) and the `unserved_mwh < 1e-6` check,
   which is a statement about a day the feeder can serve. Both say in place why
   the figure moved.
5. **`accept_typical_day_industry` is now an acceptance run on the metered day.**
   It still runs end to end and 36 of its 39 assertions hold; the three that
   changed are the ones about the native scale, the forecast error (a ledger
   supplies both sides, so there is none by construction) and zero shedding.

See `reports/formal_data_blockers.md`.
