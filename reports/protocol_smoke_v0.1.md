# Protocol Smoke v0.1

> **Superseded by `reports/protocol_smoke_v0.1_lambda010.md`.** This run was
> performed at λ = 0.25, which is no longer the frozen weight — the
> per-agent action-space migration moved what the calibration's selection rule
> picks, and 0.25 no longer passes its tolerances. The run, its artifacts and
> the numbers below are left exactly as measured.

Checks that the frozen protocol executes end to end before committing to the
formal 5-seed run. Records what was measured, not what it implies.

## Environment

| | |
|---|---|
| Python | 3.10.20 (`energy_env`) |
| numpy / pandas | 1.24.3 / 2.3.3 |
| torch | 2.11.0+cpu |
| Gurobi | 13.0.1 |
| git sha | `e98e11609a7d1fcd81a6014ce3ba99346179aca2` (working tree modified) |
| OPF engine | `socp` |
| Scenario | `typical_day` (`synthetic_v1` wholesale) |
| RL agents | `ESS9, ESS14, ESS17, ESS27` (4 storage units) |
| Action box | `storage_quote_v2`, charge `[0,150]`, discharge `[0,300]`, truthful `(0,0)` |
| λ deviation penalty | **0.25** |

Run: `outputs/rl/run-20260921-164045-d13a80cc`
Policy: `policies/protocol_smoke_v0_1/`
Training log: `logs/protocol_smoke_v0_1.log`
Audit output: `results/protocol_smoke_v0_1_audit.json`
Health output: `results/protocol_smoke_v0_1_health.json`

## Scale

| | |
|---|---|
| Seed | 1 |
| Cycles | 2 |
| Train episodes | 40 (20 per cycle) |
| Validation passes | 2 (20 held-out day-pairs) |
| Test days | never touched |
| Wall clock | 649 s training (16.2 s/episode) |

## Test status

```
574 passed, 48 deselected in 149.01s
```

`pytest.ini` excludes 48 `slow`-marked tests from the default run. Both the full
default run and the protocol-relevant slice of the slow set were executed:

| Run | Result |
|---|---|
| `pytest -q` (default) | 574 passed, 0 failed |
| `-m slow tests/test_storage_exclusivity.py` | 8 passed |
| `-m slow` (storage exclusivity, action spec, surplus metrics, money units, dev-penalty calibration) | 18 passed, 1 **failed** |

The single slow-set failure is recorded under *Open findings* below. It is not
a model bug and the default suite does not reach it.

## Checks

### Day rotation — PASS

Recorded per-episode `day_id` (now persisted in `metrics.csv`) against the
declared schedule:

```
cycle 0  [8,30,0,33,16,38,35,4,36,29,15,28,1,19,37,2,11,6,20,22]
cycle 1  [16,36,1,8,28,6,38,37,35,11,4,2,20,15,33,29,30,19,0,22]
```

For both cycles: exactly the 20 training days, each once, no repetition, no
omission, and the order equals `day_rotation.cycle_days(1, cycle)`. No
validation or test day appears in any training episode.

### RNG isolation — PASS

- `cycle_days(1, 0)` is unchanged after 10,000 draws from the global numpy
  stream — the rotation runs on its own tagged stream.
- The same seed gives the same order across calls; a different seed gives a
  different order over the same 20 days (`cycle_days(1,c) != cycle_days(2,c)`,
  but `sorted(cycle_days(1,c)) == sorted(cycle_days(99,c))`).
- The order for cycle *k* does not depend on cycles before it having run.

### Action check — PASS

From the instrumented run (`src/smoke_matd3.py`, seed 1, 24 episodes, λ=0.25).
That script does not train — it never calls `trainer.remember()` — so the rows
below describe the action box and the environment against a seeded actor, not a
trained policy. `reports/protocol_smoke_v0.1_lambda010.md` records this and
replaces the policy-level check with a replay of the saved policies:

| | |
|---|---|
| Actions outside their box | **0** |
| Quotes parked within 2% of truthful corner | **0.369%** |
| Observed quote range | `0.017 … 299.197` |
| Mean box-normalized deviation | 0.517 |
| Fleet quote spread | 24.54 CNY/MWh |

No corner collapse and no degenerate (single identical) quotes. The observed
maximum 299.197 sits just inside the 300 discharge bound; no action crossed it.

### SOC check — PASS

| | |
|---|---|
| SOC out of `[soc_min, soc_max]` | **0** |
| Observed SOC range | `0.1603 … 0.6150` |
| Terminal SOC gap | **0.0** |
| Simultaneous charge/discharge | none — `tests/test_storage_exclusivity.py`, 8 passed |

Both settlement arms pin the day's endpoints, so terminal SOC equality holds by
construction; the instrumented run confirms a gap of exactly 0.0.

### Clearing check — PASS

| | |
|---|---|
| Blocks cleared (instrumented) | 576 |
| Clearing fallbacks | **0** |
| Episodes with `capture_fellbacks > 0` | **0** of 40 |

Every block of every training episode cleared without degrading, so the
day-level economic columns are usable for all 40 episodes.

### Counterfactual check — PASS

Over the 20 held-out day-pairs:

- every row satisfies `delta_pi == pi_ai − pi_baseline` (max deviation < 1e-6);
- every row satisfies `delta_pi_percent == delta_pi / (|pi_baseline| + eps)`;
- each cycle's recorded mean equals the mean of its own ten day rows.

**Arm pairing.** Verified directly on validation day 3 rather than assumed. Both
arms received the same `day_id` (3) and the same `day_seed` (1003); load, PV,
wind, network, wholesale curve and initial SOC (0.5 on all four units) are
identical element for element, both arms settled clean, and `assert_paired`
accepts the pair. Only the bidding policy differs. The reproduced values match
the recorded row exactly:

| | reproduced | `eval.csv` cycle 1, day 3 |
|---|---|---|
| `pi_ai` | 247.685875 | 247.6858746 |
| `pi_baseline` | 265.333580 | 265.3335795 |

### KPI check — PASS

`eval.csv` and `kpi.json` both carry `pi_ai`, `pi_baseline`, `delta_pi`,
`delta_pi_percent`, `p_delta_pi_positive`, per cycle and per day. The baseline
arm is identical across both cycles (`330.9603486452259`) — it is a function of
the day and the configuration, not of the policy, as documented.

No reward is reported or used as a profit anywhere in the KPI path.

### Checkpoint check — PASS

`train → save → load → validation`. The final policies were reloaded from disk
through `load_policies_from_dir` with the manifest's own obs/action specs, and
re-run over the cycle-1 validation days. The reproduced `mean_delta_pi` equals
the recorded value exactly. Under `--eps 40` no mid-run checkpoint is written:
the checkpoint cadence is every 50 episodes and the final save is what the run
produced.

## Key statistics

Held-out days only (20 day-pairs, 2 cycles × 10 validation days):

| Statistic | Value |
|---|---|
| mean `pi_ai` | 286.8674 |
| mean `pi_baseline` | 330.9603 |
| **mean `delta_pi`** | **−44.0929** |
| median `delta_pi` | −15.5268 |
| p(`delta_pi` > 0) | 0.10 (2 of 20) |
| action boundary hit rate (within 2% of truthful) | 0.369% |
| SOC min / max | 0.1603 / 0.6150 |
| clear failure count | 0 |
| NaN / Inf count | 0 |

Per cycle, which is the trend the smoke exists to show:

| Cycle | episode | mean `pi_ai` | mean `delta_pi` | p(>0) |
|---|---|---|---|---|
| 0 | 20 | 250.29 | **−80.67** | 0.0 |
| 1 | 40 | 323.44 | **−7.52** | 0.2 |

The policy improves substantially between cycles but has **not** overtaken the
truthful baseline after 40 episodes. This is expected of a two-cycle smoke and
is not read as a result about the method. One episode-level pattern is worth
carrying forward: in cycle 0 the fleet's quotes collapsed to near-zero
participation on five held-out days (`pi_ai ≈ 0` against baselines of 0.12 to
26.4 CNY), and recovered by cycle 1.

## Deviation from the plan

The plan permits minimal fixes where the protocol cannot execute as written.
Three were made; none changes model logic, reward, box, scenario, split or KPI
definition.

1. **Validation results were not persisted** (`counterfactual_eval`,
   `run_artifacts`, `train_rl`). `validation_kpis` computes the KPI means but
   `record_eval` discarded them, and no per-day detail existed anywhere.
   `validation_kpis` now additionally returns `days`; `eval.csv` carries one row
   per `(cycle, day_id)` and `kpi.json` one block per cycle. Additive.
2. **`day_id` was not persisted** (`run_artifacts`, `train_rl`). The training
   record did not say which day priced each episode, so Phase 3's "each train
   day exactly once per cycle" was unverifiable from the run's own output.
   `metrics.csv` now carries `day_id` and `cycle`. Additive.
3. **`load_evals` would have returned the per-day rows** to the dashboard's
   evaluation chart. It now filters to summary rows. Behaviour for existing
   single-grain files is unchanged.

`src/audit_protocol_smoke.py` was added to run these checks against any run
directory; `--checkpoint-check` performs the reload-and-reproduce test.

## Open findings

**The λ = 0.25 calibration no longer reproduces.** `rl_bidding.py` has
uncommitted changes migrating MATD3 from one environment-wide action spec to
per-agent action spaces, re-anchoring the deviation penalty from logit space to
box-normalized action space. Re-running the documented calibration under the
current code does not reproduce `docs/2026-09-18-dev-penalty-calibration.md`,
and the document's own selection rule now picks **λ = 0.1**:

| λ | mean dev (now / doc) | spread (now / doc) | admissible |
|---|---|---|---|
| 0.000 | 0.561 / 0.597 | 87.6 / 87.9 | reference |
| 0.100 | 0.531 / 0.586 | 80.9 / 82.8 | **largest passing** |
| 0.250 | **0.424** / 0.496 | 58.2 / 65.4 | fails — dev floor is 0.449 |
| 1.000 | 0.107 / 0.083 | 14.0 / 10.9 | rejected |

`|econ|` moved 0.907 → 0.7908 and `grad_ratio` is now `0.903·λ` against a
documented `0.784·λ`, which is why
`test_the_regularizer_pull_grows_with_the_weight` fails. The corner fraction at
λ=0 also moved 0.0% → 12.5%.

λ = 0.25 was used for this smoke as instructed. Its documented justification no
longer holds under the current code, so the choice between keeping 0.25 and
re-freezing at 0.1 needs deciding before the formal run — the value is frozen
into every seed otherwise.

**Coverage note.** Per-block actions and SOC are not recorded by the trainer, so
the Phase 5/6 instrumentation comes from `smoke_matd3.py` on a fixed B3 day
rather than from the 40 rotated episodes themselves. Clearing fallbacks and
NaN/Inf *are* recorded per episode and are clean across all 40.

## Conclusion

```
PROTOCOL_SMOKE_PASS
```

All plan criteria met:

```
[x] full pytest pass                    574 passed, 0 failed
[x] 40 train episodes completed
[x] 2 validation passes completed
[x] train day rotation correct
[x] validation/test isolation correct
[x] RNG streams isolated
[x] no NaN/Inf
[x] no action OOB
[x] no SOC violation
[x] no simultaneous charge/discharge
[x] terminal SOC correct
[x] AI/baseline paired correctly
[x] KPI fields complete
[x] eval.csv written
[x] kpi.json written
[x] checkpoint save/load works
```

The protocol executes. The λ calibration finding above is a parameter decision,
not an execution failure, and is the one item to settle before 5 seeds × 40
cycles.
