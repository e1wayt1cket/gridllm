# Protocol Smoke v0.1 — λ = 0.10

Supersedes `reports/protocol_smoke_v0.1.md`, which was run at the previously
frozen λ = 0.25. That run and its artifacts are kept unchanged; this report
re-runs the same protocol at the newly frozen weight and re-checks it.

## What changed

The deviation-penalty weight was re-frozen from **0.25 to 0.10**. The evidence
that justified 0.25 no longer reproduces: `rl_bidding.py`'s migration from one
environment-wide action spec to per-agent action spaces, with the penalty
re-anchored from logit space to box-normalized action space, changed what the
calibration's own selection rule picks. At λ = 0.25 the mean deviation is 0.424,
below the rule's 0.449 floor, so it is not admissible; 0.10 is the largest
weight that is. Full table and the superseded measurement:
`docs/2026-09-18-dev-penalty-calibration.md`.

The frozen value now lives in one place, `rl_td3.CALIBRATED_DEV_PENALTY`, and
is the default for both `train_rl.py` and `smoke_matd3.py`. **This smoke passes
no `--dev-penalty` flag at all** — the value under test is the default a formal
run would get.

Nothing else moved: scenario, ESS capacity and siting, action box, reward,
baseline, train/validation/test split, day rotation and synthetic_v1 are
untouched, and the two runs differ in exactly one recorded field.

| | λ = 0.25 (kept) | λ = 0.10 (this report) |
|---|---|---|
| run id | `run-20260921-164045-d13a80cc` | `run-20260921-212016-d13a80cc` |
| `cli_args.dev_penalty` | 0.25 | 0.1 |
| episodes / seed / rotation / scenario | 40 / 1 / on / `typical_day` | identical |
| policies | `policies/protocol_smoke_v0_1/` | `policies/protocol_smoke_v0_1_lambda010/` |

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
| RL agents | `ESS9, ESS14, ESS17, ESS27` |
| Action box | `storage_quote_v2`, charge `[0,150]`, discharge `[0,300]`, truthful `(0,0)` |
| λ deviation penalty | **0.10** (`rl_td3.CALIBRATED_DEV_PENALTY`, default) |

Training log: `logs/protocol_smoke_v0_1_lambda010.log`
Audit: `results/protocol_smoke_v0_1_lambda010_audit.json`
Health: `results/protocol_smoke_v0_1_lambda010_health.json`

## Scale

| | |
|---|---|
| Seed | 1 |
| Cycles | 2 |
| Train episodes | 40 (20 per cycle) |
| Validation passes | 2 (20 held-out day-pairs) |
| Test days | never touched |
| Wall clock | 493 s (12.3 s/episode) |

## Test status

| Run | Result |
|---|---|
| `pytest -q` | **574 passed, 0 failed**, 49 deselected |
| `-m slow tests/test_dev_penalty_calibration.py` | 4 passed |
| `-m slow tests/test_storage_exclusivity.py` | 8 passed |

Two tests were changed, both because they encoded the superseded weight:

- `test_the_regularizer_pull_grows_with_the_weight` asserted `grad_ratio ==
  0.784 * lambda`. Rewritten as
  `test_the_regularizer_pull_is_linear_in_its_weight`, which asserts the
  **proportionality** — `grad_ratio / lambda` constant across the grid and
  `grad_ratio(0) == 0` — rather than a fitted constant. The constant belongs to
  the observation batch and the frozen critic; the proportionality is what the
  implementation guarantees. Measured, `grad_ratio / lambda = 0.9030373072`
  for every row (relative spread 1.6e-8), with `econ_grad_norm` bit-identical
  across rows.
- `test_deviation_penalty_defaults` asserted the CLI default was `0.0` ("off by
  default: the coefficient is chosen by calibration, not inherited"). It now
  asserts the default equals `CALIBRATED_DEV_PENALTY`, which is the property
  the freeze establishes.

A new `test_the_frozen_weight_is_the_one_the_rule_selects` re-runs the
selection and fails if `CALIBRATED_DEV_PENALTY` and the rule disagree, so a
changed actor, penalty or box cannot leave the formal runs using a value whose
supporting measurement is gone.

## Checks

All six are from `src/audit_protocol_smoke.py` against this run's artifacts.

### Day rotation — PASS

Both cycles recorded exactly the 20 training days, each once, in the order
`day_rotation.cycle_days(1, cycle)` derives from the run seed:

```
cycle 0  [8,30,0,33,16,38,35,4,36,29,15,28,1,19,37,2,11,6,20,22]
cycle 1  [16,36,1,8,28,6,38,37,35,11,4,2,20,15,33,29,30,19,0,22]
```

### Held-out isolation — PASS

No training episode was priced on a validation or test day; no validation row
names a day outside `[3,5,9,10,13,14,24,26,27,39]`; no test day appears
anywhere in the run.

### Training health — PASS

Zero infinite values; zero clearing fallbacks across all 40 episodes. The
`critic_loss` / `actor_loss` / `q_*` columns are **absent** on exactly episodes
1–20 — with `start_steps=500` and 24 steps per episode, no gradient step runs
before episode 21, so there is no loss to write. The check requires those gaps
to be confined to the warmup and to be all-or-nothing per column; a gap after
the warmup would be a dropped update. `mean_reward`, `clearing_objective` and
`re_rate` are present and finite on all 40.

### Counterfactual — PASS

Across the 20 held-out day-pairs: `delta_pi == pi_ai − pi_baseline` (max
deviation 1.1e-13), `delta_pi_percent == delta_pi / (|pi_baseline| + eps)`, and
each cycle's recorded mean equals the mean of its own ten day rows. The pairing
is enforced in the code path by `counterfactual_eval.assert_paired`, which
compares the two arms' wholesale curves element for element and refuses the
comparison otherwise.

### KPI completeness — PASS

Both cycles carry `mean_pi_ai`, `mean_pi_baseline`, `mean_delta_pi`,
`p_delta_pi_positive` in `kpi.json`, with the per-day rows in `eval.csv`.

### Checkpoint — PASS

The saved policies were reloaded from disk through the manifest's own obs/action
specs and re-run over the cycle-1 validation days. The reproduced
`mean_delta_pi` equals the recorded value exactly.

### Policy behaviour — PASS

New this round. The trainer records no per-block action or SOC, so the saved
policies were replayed over all 10 held-out days (240 blocks) and watched:

| | |
|---|---|
| Declared quotes outside their own box | **0** |
| Simultaneous charge and discharge | **0** |
| SOC outside `[soc_min, soc_max]` | **0** |
| Non-finite actions or schedules | **0** |
| SOC range reached | `0.1000 … 0.8895` |

The fleet reaches its lower SOC bound exactly (0.1000) on the held-out days
without crossing it.

## Key statistics

Held-out days only (20 day-pairs, 2 cycles × 10 validation days):

| Statistic | λ = 0.10 | λ = 0.25 (kept) |
|---|---|---|
| mean `pi_ai` | 275.4956 | 286.8674 |
| mean `pi_baseline` | 330.9603 | 330.9603 |
| **mean `delta_pi`** | **−55.4647** | −44.0929 |
| median `delta_pi` | −37.5059 | −15.5268 |
| p(`delta_pi` > 0) | 0.0000 | 0.1000 |
| action boundary hit rate (within 2% of truthful) | 0.369% | 0.369% |
| SOC min / max | 0.1000 / 0.8895 | — |
| clear failure count | 0 | 0 |
| NaN / Inf count | 0 | 0 |

Per cycle:

| Cycle | episode | mean `pi_ai` | mean `delta_pi` | p(>0) |
|---|---|---|---|---|
| 0 | 20 | 250.29 | −80.67 | 0.0 |
| 1 | 40 | 300.70 | −30.26 | 0.0 |

Neither weight beats the truthful baseline at 40 episodes, and this is not read
as a result about the method — two cycles is a smoke, not an experiment.

**The two runs are not a clean λ comparison, and one part of them is a control
rather than a result.** Because no gradient step runs before episode 21, cycle 0
cannot depend on λ, and it does not: `pi_ai` for every cycle-0 day is
**bit-identical** between the two runs (max difference 0.0), while cycle 1
differs by up to 48.41 CNY. The truthful baseline arm is identical across both
runs *and* across both cycles (difference exactly 0.0), confirming it is a
function of the day and the configuration and not of the policy.

## Open findings

**`smoke_matd3.py` does not train, and its checks are not evidence about a
trained policy.** It never calls `trainer.remember()`, so the replay buffer
stays empty and `MATD3.update()` returns early on every call
(`len(self.buffer) < self.batch_size`). The actor therefore stays at its seeded
initialization for the whole run, no gradient step is ever taken, and
`final_critic_loss` is `None`. This is why running it at λ = 0.25 and at
λ = 0.10 produced **bit-identical** output in every field except `dev_penalty`
itself — the weight only enters the actor loss, which is never computed.

Its seven checks still pass and are still worth running, but they describe the
environment and the box rather than a policy: "no action out of box" is close
to tautological while the actions are drawn by `np.random.uniform` inside the
box, and the SOC bounds are enforced by the clearing. `docs/2026-09-18-matd3-smoke.md`
records a historical run of this script; its wording ("a short training")
overstates what the script does.

The action and SOC checks for the trained policy in this report therefore come
from `audit_protocol_smoke.check_policy_behaviour`, which replays the saved
policies, not from `smoke_matd3.py`. Fixing the instrument — adding the missing
`remember()` call — would change a script whose historical results are already
written down, so it is left as a decision rather than done here.

## Conclusion

```
PROTOCOL_SMOKE_PASS  (lambda = 0.10)
```

```
[x] full pytest pass                    574 passed, 0 failed
[x] 40 train episodes completed
[x] 2 validation passes completed
[x] train day rotation correct
[x] validation/test isolation correct
[x] RNG streams isolated
[x] no NaN/Inf
[x] no action OOB                        0 of 240 replayed blocks
[x] no SOC violation                     0 of 240 replayed blocks
[x] no simultaneous charge/discharge     0 of 240 replayed blocks
[x] terminal SOC correct                 both arms settle clean
[x] AI/baseline paired correctly
[x] KPI fields complete
[x] eval.csv written
[x] kpi.json written
[x] checkpoint save/load works
```

The one open item is the `smoke_matd3.py` instrument described above; it does
not affect any check in this report, all of which are backed by either the
training run's own artifacts or the replay of its saved policies.
