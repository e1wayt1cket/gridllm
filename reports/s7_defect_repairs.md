# S7: the C1–C5 repairs

Each item, what it was in this repository, what changed, what holds it, and what
was deliberately left alone.

**Baseline**: `HEAD = 9f61c9a`, branch `master`, 81 uncommitted paths before this
stage. Nothing was committed or pushed.

The task book's `C1–C5` labels do not appear in the repository; the mapping from
each label to the code below was established in S0 and confirmed against the
task book's own text for this stage.

## C1 — the opponent features carry a signed, direction-normalized deviation

**What it was.** `rl_env._compute_opponent_features` took each participant's
first action, scaled it into that participant's own box, and averaged the
results across the fleet:

```python
scaled = (value - lo) / (hi - lo)
```

A box position has no common meaning between action spaces. The storage
`charge_price` box is `(0, 150)` with truthful at **0**, so 0.0 there means
"declaring truthfully". The prosumer `bid_mult` box is `(0.3, 1.8)` with truthful
at **1.0**, so 0.0 there means "shaded as far as the space allows", and its
truthful point sits 0.467 along. The mean therefore averaged
distance-from-the-low-bound for one convention with distance-from-truthful for
the other. At the truthful profile every participant is making the same
statement, and the feature read 0.0 for the batteries and 0.467 for the loads.

**What changed.** Both features are now the fleet mean and spread of a
**signed** deviation from each participant's own truthful point, normalized by
how far that participant could have gone *in that direction*:

```
d = (a - truthful) / (high - truthful)     when a > truthful   -> [0, +1]
d = (a - truthful) / (truthful - low)      when a < truthful   -> [-1, 0]
d = 0                                      when a == truthful
```

Truthful is zero for every participant, shading down is negative, shading up is
positive, and a participant that has used up its whole room in one direction
reads exactly -1 or +1 there.

The denominator is per-direction on purpose. The storage charge box has no room
below its truthful point at all and 150 CNY/MWh above it, so a `(high - low)`
denominator would put every downward move out of range while a downward move is
exactly what "shade the bid" means. Normalizing each direction by its own reach
is what makes the two directions comparable within one participant and what
makes an asymmetric box expressible at all.

The measured value is clamped into the participant's box before the division, so
a declaration outside its own space reads as the boundary it overran and the
result stays in `[-1, 1]`. After that clamp a zero reach cannot arise; the guard
is kept so a degenerate box cannot divide by zero.

**This is not `rl_td3.normalized_deviation`.** That one is unsigned and divides
by the full box width; it answers "how far from truthful is this policy", which
is what a regularizer needs. These features answer "which way is the fleet
leaning, and by how much of its own room", which is what an observer needs. They
share the idea of normalizing by a distance in the action's own units and are
not the same number.

**Registered as `OBS_V6`.** V3, V4 and V5 all share this layout, so the version
name is the only thing separating four different meanings, and a checkpoint
under any of them must be identifiable rather than reinterpreted. `OBS_V5` holds
an **unsigned magnitude** form (`|a - truthful| / (high - low)`) that was put in
during this same stage and is **not** the approved definition; it is kept
registered precisely so its name can never be read as V6's.

**Was anything trained or exported under the absolute form?** Checked, because
it decides whether V5 has to stay a separate marker. Every checkpoint in
`policies/` and `outputs/` was read: **2072 carry metadata and all record
`obs=v3_12d`; 614 are older files with no metadata block at all. None records
`v5_12d` or `v4_12d`.** Nothing was trained or exported under either form, which
is why V5 could be left as a marker rather than needing a migration.

**New tests.** `tests/test_opponent_features.py` (19), on a population that
carries both ontologies (32 prosumers and 4 batteries). Against the ten required
cases:

| required | where |
|---|---|
| truthful reads zero for all | `test_declaring_truthfully_reads_zero_for_every_ontology` |
| below is negative, above is positive | `test_a_prosumer_below_truthful_is_negative_and_above_is_positive` (4 fractions) |
| same normalized deviation in different boxes | `test_the_same_normalized_deviation_agrees_across_different_boxes` — two environments with different multiplier boxes |
| storage truthful at its low edge; no divide by zero | `test_a_storage_truthful_at_its_low_edge_has_no_downward_half` |
| the two directions do not collapse | `test_the_two_directions_do_not_collapse_to_one_feature` |
| the mean is over signed values | `test_the_mean_is_not_an_average_of_magnitudes` — opposite shadings average to 0 while the mean magnitude is positive |
| missing / non-finite / illegal inputs | three tests, plus `test_a_declaration_outside_its_box_reads_the_boundary_it_overran` |
| dimension and interface agree | `test_the_features_still_land_where_the_policy_expects_them` (range asserted too) |
| an old semantic cannot be loaded as the new one | `test_the_signed_feature_is_not_the_unsigned_one`, `test_every_older_observation_spec_is_still_identifiable` |
| order independence | `test_the_result_does_not_depend_on_the_order_participants_are_visited` |

Also: `test_an_asymmetric_box_is_described_by_each_directions_own_reach` and
`test_a_box_edge_reads_exactly_one_or_minus_one` pin the normalization itself.

`tests/test_rl_spec.py` was updated to the V6 layout, with V4 and V5 added to the
"retired specs stay identifiable" test, and every other test that builds a
policy now declares V6.

**Stated limit.** This changes what a policy sees. No MATD3 training is being run
this round and every recorded checkpoint is V3, so nothing published is
invalidated — but a V6 policy and a V4 or V5 policy are not comparable, and a run
that mixes them fails at load rather than quietly.

## C2 — the ladder is built through the scenario's own config path

**What it was.** `baseline_ladder._config` assembled a `MarketConfig` from the
dataclass defaults, so it read `config/defaults.yaml` for nothing in `network`,
`storage` or `rt`:

```python
config = MarketConfig(opf_mode=opf_mode, verbose=False,
                      market_design=MarketDesignConfig(
                          use_constraint_multi_obj=False))
```

Measured against `typical_day_config()`, the rung differed on **three fields**:

| field | rung | scenario |
|---|---|---|
| `network.load_power_factor` | 0.90 | **0.8503** |
| `network.inverter_smax_multiplier` | 1.0 | **1.10** |
| `network.line_capacity_overrides` | `None` | `{}` |

Both of the material ones are choices about this study's facilities rather than
properties of the network, and `inverter_smax_multiplier` is the field the task
book names. The `MarketDesignConfig(...)` argument was a no-op: the dataclass
default and the file agree that `use_constraint_multi_obj` is `False`.

**What changed.** `scenarios.scenario_model_config(scenario, T, opf_mode)` is
the one construction — `MarketConfig.from_yaml` plus the scenario's own
`override_config` plus its declared power factor — and `typical_day_config`
now delegates to it rather than carrying a second copy. `_config` calls it and
then sets only what is a rung's own: `verbose` and the voltage band.

**The ladder's numbers moved, and one moved qualitatively.**

| rung | band | Vmin | binding | shed MWh | curtailment MWh | util |
|---|---|---|---|---|---|---|
| B1 | primary | 0.949731 → **0.945946** | False → False | 2.05e-07 → 3.18e-07 | 0.0000 | 0.2191 → **0.2325** |
| B1 | strict | 0.950000 → 0.950000 | True → True | 1.28e-03 → **3.17e-02** | 0.0000 | 0.2185 → 0.2243 |
| B2 | primary | 0.958347 → **0.954728** | False → False | 4.53e-09 → 4.53e-08 | 13.405 → 13.381 | 0.1755 → 0.1772 |
| B3 | primary | 0.952631 → **0.930000** | False → **True** | 3.45e-07 → 1.13e-07 | 13.397 → 13.281 | 0.2056 → **0.2731** |
| B3 | strict | 0.952627 → **0.950000** | False → **True** | 4.35e-07 → 4.22e-08 | 13.397 → 13.278 | 0.2056 → **0.2788** |

**B3 now binds its voltage band at both bands** where it previously bound at
neither. B0 is unchanged, as it should be — it is a static power flow that never
reads a `MarketConfig`. `docs/2026-09-18-baseline-ladder.md` carries the new
table and the reason; the stale figure quoted in `test_baseline_ladder.py`'s
shed-tolerance comment was refreshed to the measured value.

**New tests** in `tests/test_baseline_ladder.py` (5): the rung and the scenario
agree on every field except the band; the two facility choices arrive; the two
bands differ only in their bounds; a sweep override reaches `_config` while
`build_rung_agents` deliberately does not inherit it; and re-applying the
scenario onto a rung config three times moves nothing, which is what holds the
applier idempotent. Two of these were written wrong first — they asserted that
an override *should* reach a rung, and the existing
`test_a_rung_does_not_inherit_another_callers_scenario_override` says the
opposite on purpose. The tests were corrected to the design rather than the
design to the tests.

## C3 — the Nash payoff cache key is canonical

**What it was.**

```python
parts.append(tuple(np.round(s.get(n, []), 5)))
```

A strategy component the profile does not declare became `()`, which collides
with every other empty tuple, so a profile that declares nothing caches a payoff
for it. A non-finite value became a key containing NaN, which is not equal to
itself, so the cache missed forever while looking like it was working. And the
tuple held NumPy scalars rather than Python floats.

**What changed.** `_canonical_action` converts one action sequence to a tuple of
Python floats; `_make_cache_key` raises on a declared-action gap and on a
non-finite value, each naming the agent and the action. The key is a tuple of
tuples ordered by agent name and then by each agent's own action order, holding
no object identity, no `repr` and no NumPy type.

`_KEY_DECIMALS = 5` is now stated with its justification rather than appearing
as a literal: the search's own optimizer starts from `_rhobeg_for(spec)` — about
5.0 CNY/MWh on a storage quote and 0.05 on a multiplier — so 1e-5 is three
orders below the smallest step the search will take, and two profiles the key
equates are two profiles it would explore identically.

**New tests** in `tests/test_nash_action_space.py` (8): list, tuple and array
containers key identically; the tuple holds `float`, not `np.float64`; a move
the search can act on is not shared, asserted *against* `_rhobeg_for`; a move
below the tolerance does share a key, stated as the trade the cache makes; the
key is stable across instances and construction orders and contains no address;
a profile equivalent to a cached one is answered from the cache rather than
re-evaluated; and a missing declared action and a non-finite value are both
refused.

## C4 — the conflicting meter-day rejection actually runs

**What it was.** The existing test handed `exclusion_rows` a pre-built
conflicting list:

```python
rows = mim.exclusion_rows([], _BUSES, {}, [("Ma65a1fda1669", date)])
```

That tests how a conflict is *reported*. The path that *decides* a meter-day is
conflicting — `flag_records` writing `FLAG_DUPLICATE_CONFLICT` onto two rows
that disagree, `collect_cells` folding the flag into the cell, `classify_cell`
mapping it to `conflicting`, and `qualified_block_keys` filtering — never ran,
and the meter in the fixture sits on bus 1, which is *not* one of
`UNSUPPORTED_BUSES`, so nothing proved the rejection was the conflict rather
than a bus rule reached first.

**What changed.** Nothing in `src/`; the gap was in the test. Three tests in
`tests/test_map_industry_meters.py` build synthetic `MeterDay` rows for bus 1
and drive the real path end to end. The conflict is confirmed present before the
row is asserted, and the exclusion is asserted *not* to be an
`unsupported_bus` row for bus 1. The control case asserts a clean, complete,
supported meter-day is never withheld, and that a byte-identical repeat is not a
conflict.

The workbook and the real processed data are untouched; every record is built in
the test.

## C5 — one write contract per door into a scenario

**What it was.** Two writers reach a scenario and their key forms are different:
`set_scenario_param` writes a dotted path *inside the scenario block*
(`wholesale_source`, `multipliers.pv`), while `set_scenario_override` writes one
flat `"section.field"` string into `override_config`. Nothing stopped a caller
crossing them. The dangerous crossing is writing a nested path through
`set_scenario_param("…", "override_config.network.v_min_pu", …)`, which builds
`{"network": {"v_min_pu": …}}` — a structure `apply_scenario_overrides` rejects
at apply time, one scenario build away from the sweep that made it.

**What changed.**

| door | now refuses |
|---|---|
| `set_scenario_param` | a path rooted at `override_config`, naming the other door in the message |
| `set_scenario_override` | a key rooted at `override_config`; a mapping value under a *dotted* key |
| both | an empty or malformed key (`""`, `"a."`, `".a"`, `"a..b"`) |

A fourth guard went where the damage happens rather than at one of the writers:
`apply_scenario_overrides` refuses to replace a **section** (a dataclass field)
with a non-dataclass value, so `{"network": {"v_min_pu": …}}` is rejected
however it arrived — a writer, a hand-built dict, or `config/scenarios.yaml`
itself. A key naming no field is still caught by the applier, as before:
`override_config` legitimately carries the builder directive `custom_loads`,
which is not a `MarketConfig` field, so the setters do not police that.

**New tests** in `tests/test_scenario_override_precedence.py` (7), including two
round-trips: a scenario setting written through its own door, read back, and
cleared by `reload_scenarios`; and an override written through its door,
reaching the built model, with the neighbouring field left at the file's value.

## Wrap-up: the four closing items

### 1. Out-of-box actions: projected upstream, refused at the feature

The question was whether the feature's clip hides a value the market never saw.
It does not, and the check is a code fact rather than an assumption:
**`BiddingEnv.step` projects every action into its spec box before anything else
looks at it**

```python
self.current_actions[nm][name][t_start:t_end] = float(np.clip(act[i], lo, hi))
```

and the clearing is called with `window_actions` sliced from that same
`current_actions`, and `_prev_block_actions` — the array the opponent features
read — is snapshotted from it too. So the feature reads the value the market
executed, and the two are the same array rather than two computations that
agree.

What the feature layer was doing was **silently clamping a second time**, which
on the real path could never fire and off it would have turned an overrun into a
boundary reading. That clamp is gone: `signed_deviation` now **refuses** a value
outside its box, naming the caller's mistake. The refusal carries a tolerance of
`1e-9 * box width` for the one legitimate case — a block mean of values pinned
to the top edge can land a few ulps above it — and a test asserts that tolerance
admits that and not a visible overrun.

New tests, in `tests/test_opponent_features.py`:

| what it pins |
|---|
| `test_the_features_are_computed_from_the_value_the_market_executed` — a policy output of `[999, -5]` is projected by `step` to `[150, 0]`; `_prev_block_actions` equals `current_actions` element for element; the feature reports the projected deviation |
| `test_a_declaration_outside_its_box_is_refused_not_clamped` |
| `test_the_block_mean_of_edge_pinned_actions_is_not_refused` — the tolerance is for arithmetic, and does not admit a real overrun |

### 2. The old checkpoints are marked, not migrated

The recorded observation spec is the only thing that says which observation a
checkpoint was trained in. The input width does not, and the count makes the
point: **120 of the untagged files are twelve wide**, the same width as V3, V4,
V5 and V6, which mean different things by it. They used to load unchanged, on
their shape.

`src/check_checkpoint_obs.py` classifies every `.pt` under `policies/` and
`outputs/` and writes `policies/checkpoint_obs_compat_manifest.json`:

| status | count | loadable by the current environment? |
|---|---|---|
| `current` | **0** | — |
| `incompatible_v6_12d` | 2072 | no — the recorded spec names another observation |
| `legacy_obs_unverified` | 614 | no — the file says nothing, so nothing can be established |

The third row is the one that needed making explicit, and it is now refused by
both loaders: `rl_td3.load_policy` and `rl_bidding.load_policies` raise
`legacy_obs_unverified` when a caller supplies a spec and the file has no
metadata. Passing **no** spec still loads the raw weights — that is a caller
stating it takes responsibility, and it is why the check has to live behind a
supplied spec rather than on the file's shape.

`tests/test_checkpoint_obs_compat.py` (8) holds it: every checkpoint on disk is
classified; none is `current`; two files of identical width get opposite
verdicts; and a V3-tagged file and an untagged one are refused with *different*
messages, because "trained against something else" and "nothing is known" are
different findings.

### 3. The old ladder conclusions are kept out of the current record

Three separate things came out of this, because C2 moved more than the ladder's
own table.

**The recorded outputs.** The five pre-C2 files were already marked by
`outputs/ladder_superseded_manifest.json`; they are untouched, and nothing under
`src/` reads them — `data_aggregator`, `make_training_summary`,
`make_results_figure` and `make_rot_comparison_figures` do not reference the
ladder at all, checked by search.

**The calibration.** Four modules build on `baseline_ladder`
(`calibrate_dev_penalty`, `ess_siting`, `smoke_matd3`, `action_box_sensitivity`),
so C2 moved their inputs too — and the dev-penalty weight is one of the task
book's own A6 deliverables. Re-run on the corrected day:

| lambda | \|econ\| | grad ratio | mean dev | quote spread |
|---|---|---|---|---|
| 0.000 | 0.5813 | 0.0000 | 0.5510 | 87.012 |
| **0.100** | 0.5813 | **0.1217** | **0.5241** | **80.100** |
| 0.250 | 0.5813 | 0.3042 | 0.3425 | 57.594 |
| 1.000 | 0.5813 | 1.2167 | 0.0840 | 12.313 |

**The rule still selects 0.10**, so `rl_td3.CALIBRATED_DEV_PENALTY` is unchanged
and no experiment needs re-running on this account. The supporting numbers moved
(`|econ|` 0.7908 → 0.5813, spread at 0.10 80.9 → 80.1) and
`docs/2026-09-18-dev-penalty-calibration.md` now carries the new table with the
old one moved to its superseded section. Worth recording: the test that asserts
the frozen weight is the one the rule selects is marked `slow`, so **it is
deselected from every ordinary run** — `pytest.ini` is `-m "not slow"`. It was
run explicitly here (4 passed) and that is the only reason this was checked at
all.

**The stage report.** S5's A6 table described the pre-C2 day. It is a dated
record of what was measured then, so it was left in place with a supersession
note at its head pointing at the current table, rather than rewritten.

`reports/phase11_training_readiness.md` likewise carried `OBS_V4` as the
selected observation; it now notes what moved and when.

### 4. `reports/formal_data_blockers.md`

Written, and it is a list of **external** material rather than of work: the
written basis for the reading unit and for `运行容量` and `综合倍率`; real dated
day-ahead prices overlapping the load dates; the date list on which all 17
mapped buses report; and a date-selection rule declared in advance. It also
records what is refused while they are open — the formal ledger, the declaration
of any scenario as formal, S8, and MATD3 training — and names the gate that
refuses each.

## Test results

```
PYTHONPATH=src pytest -q tests/test_opponent_features.py             21 passed
PYTHONPATH=src pytest -q tests/test_checkpoint_obs_compat.py          8 passed
PYTHONPATH=src pytest -q tests/test_rl_spec.py tests/test_baselines.py
                                                                     40 passed
PYTHONPATH=src pytest -q tests/test_nash_action_space.py tests/test_regret.py
                                                                     16 passed
PYTHONPATH=src pytest -q tests/test_map_industry_meters.py           26 passed
PYTHONPATH=src pytest -q tests/test_scenario_override_precedence.py \
    tests/test_yaml_config_agreement.py tests/test_scenario_scale.py  53 passed
PYTHONPATH=src pytest -q tests/test_baseline.py tests/test_baseline_ladder.py \
    tests/test_typical_day_physical_baseline.py                      47 passed, 11 deselected
PYTHONPATH=src pytest -q tests/test_dev_penalty_calibration.py -m slow
                                                                       4 passed (115s)
PYTHONPATH=src python src/calibrate_dev_penalty.py   # selected lambda = 0.1
PYTHONPATH=src python src/check_checkpoint_obs.py    # 0 current, 2072 + 614
```

Full suite, exit code 0:

```
PYTHONPATH=src pytest -q
1098 passed, 53 deselected, 17 warnings in 391.45s (0:06:31)
```

1098 against S6's 1044: +8 C3, +7 C5, +5 C2, +3 C4, +21 C1, +8 the checkpoint
compatibility suite, +1 the ladder supersession manifest, +1 the two halves of
the legacy-load contract. The 53 deselected are the `slow` tests, not skips --
which is worth remembering for the four that matter above, run explicitly.

### A pre-existing environment limitation, recorded because it showed up here

`tests/test_baseline_ladder.py::test_b0_is_the_original_ieee33_at_its_own_load`
**fails when that file is the only thing run**, with `OSError: [WinError 127]`
inside ortools loaded from pandapower. It is not this stage's: the same test
fails identically on the stashed pre-S7 tree, and it passes in any run where
something imported `dispatch_core` first, which is what the class docstring
there says the `pywraplp` pre-load exists for. The full suite is unaffected.
`src/validate_industry_network.py` will hit the same thing when S8 runs it as a
script.

## Files

New: `tests/test_opponent_features.py`, `tests/test_checkpoint_obs_compat.py`,
`src/check_checkpoint_obs.py`, `outputs/ladder_superseded_manifest.json`,
`policies/checkpoint_obs_compat_manifest.json`, `reports/formal_data_blockers.md`.

Changed: `src/rl_spec.py`, `src/rl_env.py`, `src/baselines.py`,
`src/strategies/rl_bidding.py`, `src/rl_td3.py`, `src/rl_bidding.py`,
`src/scenarios.py`, `src/config_loader.py`,
`src/baseline_ladder.py`, `src/nash.py`, plus `tests/test_rl_spec.py`,
`tests/test_baselines.py`, `tests/test_eval_smoke_24.py`,
`tests/test_eval_cli_api_consistency.py`,
`tests/test_storage_economics_audit.py`, `tests/test_baseline_ladder.py`,
`tests/test_map_industry_meters.py`, `tests/test_scenario_override_precedence.py`.
Documentation that recorded the old state: `CLAUDE.md`,
`docs/2026-10-14-data-and-experiment-decisions.md` (a supersession row and the
observation line), `docs/2026-09-18-baseline-ladder.md` (the re-measured table).

## Still open

1. **The reading unit is `unit_unverified`** — unchanged, and S7 was not
   permitted to resolve it.
2. **No real price date intersects a real load date** — unchanged.
3. **Only 30 days (all December 2022) cover all 17 mapped buses** — the formal
   date-selection rule is still undecided.
4. **Recorded ladder artifacts predate the config fix, and are now marked.**
   Searching the tree for the old figures found them in exactly five files:
   `outputs/ladder.json`, `outputs/ladder_2.json`,
   `outputs/audit_ladder_current.json`, `outputs/audit_ladder_doc.json` and
   `logs/audit_ladder_current.log`, all untracked and dated September 18–22.
   None was modified or moved. `outputs/ladder_superseded_manifest.json` marks
   them: each file with its sha256 and date, the configuration they record
   (power factor 0.90, inverter 1.0, built from `_config`'s dataclass defaults),
   the configuration in force (0.8503, 1.10, built by
   `scenarios.scenario_model_config`), the cause, and the one qualitative change
   — B3 binds its band now and did not then. A test keeps the marker honest by
   asserting its `configuration_in_force` is what the ladder actually builds
   today, so the note cannot come to describe a model that no longer exists
   either.
5. **`make_results_figure.py` still does not import** (`DEFAULT_ACTION`, missing
   from `diagnose_profit`), **`rl_training.py:80`** still carries a single-name
   scenario default, and **the CLI's default single-episode path is still
   unseeded** — reported by S5 and S6 and untouched here.
6. **No checkpoint in this repository is loadable by the current environment.**
   All 2072 tagged ones record `v3_12d` and the 614 untagged ones cannot be
   verified at all; re-run `src/check_checkpoint_obs.py` after any training so
   the manifest keeps up.
7. **`ess_siting`, `smoke_matd3` and `action_box_sensitivity` also build on
   `baseline_ladder`,** so C2 moved their inputs too. No report in the tree was
   found quoting a number from them, but none has been re-run here either —
   that belongs with whichever stage next needs them.
8. **The four blockers are external material**, listed in
   `reports/formal_data_blockers.md`.
9. **`validate_formal_ledger` gained a `--case-type` argument (2026-10-15).**
   Its date rule now depends on whether the case is a `cross_date_diagnostic` or
   a `same_day_replay`, and an unstated case type is refused rather than
   defaulted to the permissive rule. `unit_not_verified` is unaffected and still
   rejects both. This is a gate change, so it is recorded here as well as in
   `reports/single_day_data_entry.md`.

## Gate status

Unchanged by this stage: the reading unit is `unit_unverified`, no real price
date intersects a real load date, and the formal date-selection rule is still
undecided. **S8 and formal MATD3 training remain blocked** — on those three, not
on anything S7 touched.
