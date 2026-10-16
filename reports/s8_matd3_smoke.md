# S8 — MATD3 OBS_V6 smoke test

> **This is a smoke test, not an experiment.** It exists to show that the
> training chain runs end to end on the subset scenario: environment, reward,
> gradient step, checkpoint, observation metadata, reload, independent
> settlement. **Nothing in it is a result.** The curve, the losses and the
> profit deltas below are machinery measurements — they are not evidence about
> how a policy performs, and none may be quoted in a comparison table.

Read with `reports/s8_physical_audit.md`, whose verdict gates this one.

## 1. The gate that had to be answered first

`train_rl.py:239` routes every `--scenarios` value through
`scenarios.assert_formal_subset`, and the subset scenario is not in
`formal_ensemble`:

```
ValueError: ['typical_day_industry_subset'] are not in the formal scenario set
['typical_day_industry']; a formal run may not add scenarios of its own
```

The gate is deliberate — its docstring says a run naming a scenario priced by a
generator *"is refused here rather than after it has produced a curve: the curve
would be a result about that generator"*. There was no exploratory path, so one
was added with the caller's approval:

| change | what it does |
|---|---|
| `scenarios.assert_formal_subset(names, allow_non_formal=False)` | admits a name outside the set **only** when the caller says so |
| `train_rl.py --allow-non-formal` | the only caller that opts in, and it prints an `EXPLORATORY RUN` warning |

**The set itself did not move.** `formal_scenarios()` still returns
`['typical_day_industry']`, so nothing downstream can read this run as formal.
The flag is left in `vars(args)`, which the run artifacts record — the manifest
below carries `"allow_non_formal": true`. Two tests pin the default refusal and
the announced opt-in, and the refusal message now names the way through.

## 2. Environment

| | |
|---|---|
| interpreter | `C:/Users/26036/.conda/envs/energy_env/python.exe` |
| Python | 3.10.20 |
| torch | 2.11.0+cpu |
| pandapower / gurobipy | 2.14.9 / 13.0.1 |
| tensorboard | 2.14.0 |

**`energy_env` is required and is the environment this was run in.** The base
interpreter (`C:/ProgramData/anaconda3/python.exe`, Python 3.12.7, torch 2.5.1)
has no working `pandapower` import and cannot run a scenario at all.

**Known defect in `energy_env`, pre-existing and not caused by this work:**
`lib/site-packages/matplotlib/` is missing its `__init__.py`, so
`import matplotlib` yields an empty namespace package with no `use` and no
`__version__`. Anything that calls `matplotlib.use("Agg")` at import fails
there — which is why `tests/test_formal_ensemble.py::test_every_consumer_reads_the_declared_comparison_set`
fails under `energy_env` and passes under the base interpreter. The training
chain does not import a figure module, so it is unaffected. The conda
environment was left alone; see §7.

## 3. What the smoke found: a step is a decision block, not a period

The first two attempts produced **no gradient step at all**:

| run | episodes | steps | `critic_loss` / `actor_loss` |
|---|---|---|---|
| `run-20261016-173025-28757924` | 3 | 72 | **empty on every episode** |
| `run-20261016-173216-28757924` | 8 | 192 | **empty on every episode** |

`rl_env.BLOCK_SIZE = 4` and `N_BLOCKS = 24`, and `train_rl` increments
`matd3.total_steps` once per **block**. A day is therefore 24 steps, not 96.
`update()` returns `{"critic_loss": None, ...}` until `total_steps >= start_steps`
(default 500) **and** the buffer holds a full batch of 128 — so a run under ~21
episodes only collects. An "episodes=3 smoke" is not a training smoke.

Both short runs are kept under `outputs/rl/` as the evidence for this, and each
carries the null losses that state it.

## 4. The run

```bash
PYTHONPATH=src $PY src/train_rl.py --algo matd3 --episodes 25 --seed 42 \
    --scenarios typical_day_industry_subset --allow-non-formal \
    --dev-penalty 0.10 --save-dir policies/smoke_subset_obs_v6_seed42 --no-eval
```

| | |
|---|---|
| run id | `run-20261016-173409-28757924` |
| episodes | 25 in 189 s (7.6 s/episode) |
| steps | 600 (25 × 24) — first update at episode 21, step 504 |
| agents | `ESS9`, `ESS14`, `ESS17`, `ESS27` |
| seed | 42 |
| `--dev-penalty` | 0.10, the frozen `rl_td3.CALIBRATED_DEV_PENALTY` |

| episode | critic loss | actor loss | q-gap |
|---|---|---|---|
| 21 | 3.8517 | 1.7680 | 0.8723 |
| 22 | 2.2017 | 1.3947 | 0.4020 |
| 23 | 2.5233 | 1.2790 | 0.3153 |
| 24 | 2.2004 | 1.0789 | 0.2899 |
| 25 | 1.9314 | 1.0113 | 0.2896 |

The losses fall and the twin-critic gap narrows, so the optimizer is stepping.
**That is the whole reading.** Five updates from one seed on one day establish
nothing about learning.

**`mean_reward` is ≈ −0.0017 on every episode, and that is expected here.** The
reward is differential — the agent's own profit minus its truthful-bidding
baseline — and `accept_typical_day_industry.py` already records that on this
price scenario **every storage unit is idle**: the round-trip margin at each
unit's bus is below what the clearing charges it to cycle. An idle unit earns
exactly its baseline, so the differential is ≈ 0. A flat reward curve is
therefore not a defect of the smoke, and it is also not a signal.

## 5. The verification list

| requirement | evidence | verdict |
|---|---|---|
| environment initialises, reward computes | `metrics.csv` per-episode `mean_reward`, `cp_*`, `cs_*`, `lmp_markup_*`, `raw_profit_mean` all populated; no NaN | **pass** |
| training ends, checkpoint saved | 25/25 episodes; `policies/smoke_subset_obs_v6_seed42/{ESS9,ESS14,ESS17,ESS27}.pt`; manifest `policy_paths.final` | **pass** |
| checkpoint carries OBS_V6 metadata | each `.pt` has `_makerb_policy_meta`: `obs_spec.name = v6_12d` (the `OBS_V6` constant's registered name), version 6, `obs_dim` 12, `action_spec.name = storage_quote_v2`, `actor_type = Actor` | **pass** |
| loads and re-evaluates | `eval_agents.py` loaded all four and produced a result | **pass** |
| independent settlement path | the same run reports both the direct settlement and the rolling episode, and they agree (`settled=-0.0`, `rolling=-0.0`) | **pass** |
| no old checkpoint loaded | `check_checkpoint_obs.py`: **4 `current`**, 2072 `incompatible_v6_12d`, 614 `legacy_obs_unverified`. Loading a V3 file raises: *"was trained with observation spec 'v3_12d' (v3) but the environment uses 'v6_12d' (v6); refusing to load"* | **pass** |
| dataset, scenario, seed, model config recorded | `manifest.json` (run id, git sha `9f61c9aa`, `obs_spec`, `action_spec`, full `cli_args`) and `config.json` (the `MarketConfig` snapshot) | **pass** |

The run id's trailing digest is the config+seed hash, and it is **identical**
across the 8- and 25-episode runs (`28757924`) — same configuration, same seed,
as the identity is meant to mean.

## 6. The independent evaluation

```bash
PYTHONPATH=src $PY src/eval_agents.py --policies policies/smoke_subset_obs_v6_seed42 \
    --scenarios typical_day_industry_subset --output results/smoke_subset_eval.csv
```

All four policies load and run. Every per-agent figure is ≈ 0:

| | baseline | RL | delta |
|---|---|---|---|
| clearing objective | 98,918 | 98,919 | +1 |
| settled profit (each agent) | −0.0 | −0.0 | +0.0 |
| RE rate | 85.9% | 85.9% | −0.0 pp |

Again: this is the idle fleet, not a finding about the policy. It says the
settlement path runs and the two settlement routes agree; it says nothing about
whether an RL policy helps.

## 7. What this smoke does not establish, and one thing it leaves open

- It does **not** show that a learned policy improves anything. Five gradient
  steps from one seed, on a day where the fleet is structurally idle, cannot.
- It does **not** make the subset scenario formal, and it does not touch
  `formal_ensemble`. `--allow-non-formal` is recorded on the run, so the curve
  carries the reason it is not a result.
- It does **not** lift the reading-unit blocker. The ledger is still an `audit`
  ledger.
- **It leaves the matplotlib defect in `energy_env` open.** A figure-making
  module cannot be imported in the canonical environment. That does not block
  training or evaluation, but it will block any report that renders a chart,
  and it is an environment repair outside this repository.

## 8. Recommendation — `formal_ensemble` (Phase 3)

> **Superseded later the same day.** The recommendation below was made on the
> day's first finding and is kept as it was written. A subsequent review of the
> unit-basis gate and the PCC constraint — see `reports/formal_data_blockers.md`
> and §4 of `reports/s8_physical_audit.md` — changed the answer: the ruling was
> adopted as the formal modelling basis, the subset's ledger was rebuilt as a
> **formal** ledger and passes `validate_formal_ledger` with
> `--case-type cross_date_diagnostic`, and the scenario **was** added to
> `formal_ensemble.scenarios` on 2026-10-16. `typical_day_industry` was not
> replaced and its ledger was not touched.

**Recommendation at the time: do not add `typical_day_industry_subset` to
`formal_ensemble` yet, and do not replace `typical_day_industry`.** Nothing in
this stage changes the case for or against it:

- S8 is an **audit**, not the formal physical case. The blocker that closes the
  formal path is blocker 1 — the reading unit is still `unit_unverified`, with
  no external document. That gate is untouched by this work.
- The smoke validated machinery, not the scenario's standing as a research case.
- Membership gates the **price source**, and the subset's source was never the
  question. Adding it now would say "a formal run may use this" while the load
  underneath it is still a declared unit.

**What changing it would touch, if that decision is taken later.** Removing the
entry from `formal_ensemble.excluded` and adding the name to
`formal_ensemble.scenarios` would put the subset in `formal_scenarios()` and
therefore in `assert_formal_subset`'s allowed set. Consumers:
`train_rl._resolve_train_scenarios`, `eval_agents` (via its own scenario list),
`unified_eval`, and the dashboard's scenario lists — each would then accept the
subset as a formal-run target. `tests/test_formal_ensemble.py` would need its
exclusion assertions updated, since a name may not be both excluded and formal.
`typical_day_industry` would keep its own membership; nothing replaces it.

## 9. Test evidence, and which environment is the training environment

Full default suite (`pytest.ini` sets `-m "not slow"`, so 53 `@pytest.mark.slow`
tests are deselected in both):

| interpreter | command | result |
|---|---|---|
| **base anaconda** — Python 3.12.7, torch 2.5.1 | `python -m pytest tests/ -q` | **1164 passed**, 0 failed — 9:25 |
| **`energy_env`** — Python 3.10.20, torch 2.11.0 | `$PY -m pytest tests/ -q` | **1163 passed, 1 failed** — 10:02 |

The one `energy_env` failure is
`tests/test_formal_ensemble.py::test_every_consumer_reads_the_declared_comparison_set`,
and it fails at import:

```
src\make_rot_comparison_figures.py:28: AttributeError: module 'matplotlib' has no attribute 'use'
```

That is the broken `matplotlib` install described in §2 — the file it imports
calls `matplotlib.use("Agg")` at module level, and `energy_env`'s
`site-packages/matplotlib/` has no `__init__.py`. The same test **passes** under
the base interpreter, whose matplotlib comes from the user site-packages. **It
is not caused by this work and it is not fixable from inside this repository.**

### The choice, stated

**Training and evaluation must run under `energy_env`, and did.** It is the
only environment with a working `pandapower` import — the base interpreter
cannot build a scenario at all — and it is the environment
`reports/phase11_training_readiness.md` records (its 2026-09-18 numbers name
`energy_env`, Python 3.10.20, `pandapower 2.14.1`, `gurobipy 13.0.1`,
torch 2.11.0). The training chain does not import a figure module, so the
matplotlib defect does not reach it.

The base interpreter remains the one that runs the whole suite green, and is
therefore the one for anything that touches a figure module. **Two environments,
one accepted failure, and the failure is written down rather than worked
around** — the conda environment was deliberately left alone.

## Reproduce

```bash
PY=/c/Users/26036/.conda/envs/energy_env/python.exe
PYTHONPATH=src $PY src/train_rl.py --algo matd3 --episodes 25 --seed 42 \
    --scenarios typical_day_industry_subset --allow-non-formal \
    --dev-penalty 0.10 --save-dir policies/smoke_subset_obs_v6_seed42 --no-eval
PYTHONPATH=src $PY src/eval_agents.py --policies policies/smoke_subset_obs_v6_seed42 \
    --scenarios typical_day_industry_subset --output results/smoke_subset_eval.csv
PYTHONPATH=src $PY src/check_checkpoint_obs.py --check
```
