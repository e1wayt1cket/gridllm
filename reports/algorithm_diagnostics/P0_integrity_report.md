# P0 — integrity report

Protocol `DIAGNOSTIC_PLAN_V1`, phase P0 (strictly read-only).
Written 2026-11-08. Interpreter
`C:\Users\26036\.conda\envs\energy_env\python.exe` (CPython 3.10.20,
conda-forge), torch `2.11.0+cpu`, gurobipy `13.0.1`.

Collector: `src/diagnose_integrity.py` (new, unpinned).
Artifacts: `reports/algorithm_diagnostics/P0_evidence.json`,
`reports/algorithm_diagnostics/P0_tree_hashes.json`.

Command:

```
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" src/diagnose_integrity.py
```

Result: `verdict: PASS_P0_INTEGRITY`, wall clock ~13 s.
Supporting command: `python -m pytest tests/test_frozen_guard.py
tests/test_training_contract.py -q` → **41 passed in 5.33 s**.

No file under `runs/`, `snapshots/`, `src/` (pinned), or
`reports/v2_cy_jn_2025/` was written. Every tree hash below was taken before
and after the collector run in the same process.

## Item table

| # | item | verdict | evidence |
|---|---|---|---|
| 1 | Seed 11 identity conflict | **PASS** | `P0_evidence.json` → `seed11_identity` |
| 2 | Timestamps, checkpoints, manifest | **PASS** | `run_artifacts`, `checkpoints`, `best` |
| 3 | Original OOM evidence | **INFERRED** | `oom` (0 hits); `oom_documented_kills`; `reports/cc03/phase25_launch.md`, `phase25_seed11.md` |
| 4 | Runtime of seeds 23 and 37 | **PASS** (launch) / **INFERRED** (child) | `runtime` |
| 5 | H2 access evidence | **PASS** (artifacts) / **INFERRED** (seal) | `holdout` |
| 6 | Seed 37 partial run | **PASS** | `seed37` |
| 7 | Seed 53 read-only, not resumed | **PASS** | `seed53`, `tree_unchanged` |
| 8 | Frozen integrity | **PASS** | `frozen`, `tree_unchanged` |
| 9 | Launch gate | **PASS** | `gate` |

## 1. Seed 11 identity

The current `runs/FORMAL_1b47e0ed_seed11` is the relaunched run, not the
zero-episode stub.

| field | value |
|---|---|
| `run_id` | `FORMAL_1b47e0ed_seed11` |
| `status` | `completed_early_stop` |
| episodes / block steps | 530 / 12,720 |
| `created_at` | 2026-11-06T02:27:50.512903+00:00 |
| `resumed_at` | 2026-11-06T07:18:57.404588+00:00 |
| `resume_from.checkpoint_step` | 5088 |
| `resume_from.episodes_in_checkpoint` | 212 |
| `resume_from.rounds_completed` | 4 |
| `resume_from.partial_round_resumed` | false |
| best | `_best/checkpoints/step_00001272`, value -35.287932445992034, round 1, source `evaluations` |
| final | `checkpoints/step_00012720` |

`episodes.csv` holds episodes 1…530 contiguous with no duplicate episode, no
duplicate pair, rounds 1–10 and 53 distinct training pairs. The empty-launch
stub is documented in `TECHNICAL_DEBT.md` (TD-09) as removed and its
`eval_history.json` / `train.log` stub hashes (2 B, 0 B) are quoted in
`reports/cc03/phase25_seed11.md`; no stub file survives on disk. The conflict
is historical and closed: the identity recorded in the run manifest is the run
whose artifacts are present.

## 2. Timestamps, checkpoints, manifest

Artifact hashes present in each run manifest:

- seed 11 and seed 23: every recorded artifact hash matches on disk except
  `train.log`, which is the known TD-01 ordering issue (the hash was taken
  before the completion line was appended; the mismatch predates this
  protocol). Recorded as **INFERRED**, not a new defect.
- seed 37: `role_manifest` records no artifact hashes (`None`); a stopped run
  has no completion record to hash against.

Checkpoint step directories, all with `counters` read from `trainer_state`:

| run | step dirs | latest counters |
|---|---|---|
| seed 11 | 10 (1,272 … 12,720) | `total_steps 12720, episode 530, updates 12593, buffer_len 12720` |
| seed 23 | 10 (1,272 … 12,720) | same |
| seed 37 | 3 (1,272 … 3,816) | `total_steps 3816, episode 159, updates 3689, buffer_len 3816` |

Best and final are distinct artifacts for seeds 11 and 23 (`_best` step 1,272 /
5,088 vs final 12,720). At each best checkpoint
`da_gate.actor_loss_map=linear_core_v1`, `quote_map=linear_core_v1`,
`action_schema=storage_quote_v3`, `specs.obs_spec=v6_12d`, and the action spec
carries `quote_map=linear_core_v1`. These are the P1 preconditions.

## 3. OOM

`scan_oom` searched every text artifact of seeds 11/23/37 (19 / 19 / 12 files)
for OOM, `MemoryError`, `OutOfMemory`, `killed`. **Zero hits.** The two harness
kills are therefore **INFERRED** from the phase reports:
`reports/cc03/phase25_launch.md` and `reports/cc03/phase25_seed11.md`, which
record the killed launch process rather than an artifact. No OOM artifact
exists to verify. Seed 53's terminal error is a clearing failure, not OOM.

## 4. Runtime

Seed 11 is recorded in `reports/cc03/phase25_seed11.md` as `energy_env`
3.10.20 / torch 2.11.0: **PASS**. That report also says the run was launched
twice initially under anaconda python, then resumed by hand; history line 2486
shows the original launch as bare `python .\src\run_formal_training.py --seed
11`, and no line records the resume's interpreter. The seed 11 verdict is
therefore PASS for the environment named in the phase report, with the
original launch's interpreter identity **INFERRED** as anaconda python.

Seeds 23, 37 and 53 have no interpreter field in their manifests (TD-10).
PowerShell history at
`C:\Users\26036\AppData\Roaming\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt`
records the launch:

```
line 2518: foreach ($s in 23,37,53,71) {
line 2519:     & C:\Users\26036\.conda\envs\energy_env\python.exe src\run_formal_training.py --seed $s
```

`src/run_formal_training.py:command()` begins with `sys.executable`, so the
launch interpreter is the child interpreter. Verdict: **PASS** for the launch
command, **INFERRED** for the child process's actual runtime. This is weaker
than a recorded `sys.version` and is not reported as measured.

## 5. H2 / Test / Holiday

Artifact-level evidence, all **PASS**:

- every run manifest in seeds 11/23/37: `h2_holdout_read: false`,
  `test_evaluation: null`, `legacy_seen_test_run: false`;
- the live data manifest: `h2_opened: false`;
- `episodes.csv`, `eval_history.json`, `gate_stats.jsonl`,
  `block_decisions.jsonl`: no Test pair id and no Holiday pair id; every id is
  in Train or Validation, 64 distinct pairs = 53 Train + 11 Validation;
- split counts `common` and `near` agree: train 53, validation 11, test 11,
  holiday block 5.

The seal itself is **INFERRED**: no artifact would record a read, so the absence
of a read cannot be demonstrated from the run directory, only that the
recorded flags are false and no Test/Holiday id reached an artifact. Stated as
a known limit, not as an independent observation.

## 6. Seed 37

A stopped run, not formal evidence. `run_manifest.json` status
`exploratory_development`; `episodes_run`, `total_steps`, `best`, `final`,
`finished_at` are all null. `episodes.csv` has 170 rows across rounds 1–4;
`checkpoints/` holds 3 step dirs (1,272 / 2,544 / 3,816). A
`runs/FORMAL_1b47e0ed_seed37_best` directory exists with
`step_00001272` (counters 1,272 / 53 / 1,145 / 1,272), but the run manifest
records no best selection, so it is a copied checkpoint, not a selected best.
Not resumed, not finalized.

## 7. Seed 53

`runs/FORMAL_2148a577_seed53`, status `failed`, 28 episodes / 672 steps,
error `RuntimeError: 2022-10-20|2025-10-20 block 23: clearing failed; training
stops`. Read-only: the tree hash before and after P0 is identical (see item 8).
Not resumed, not rerun.

## 8. Frozen integrity

- 19 live source pins (`formal_training_v3` manifest): all match on disk.
- 6 pinned tests/audits: all match.
- cc01 reference comparison (`reports/cc01/`): `identical: true`, `drift: []`;
  that reference covers only v2 and ready, so it does not cover v3.
- `tree_unchanged`: 13 / 13 trees identical before and after P0 — the three
  v3 `runs/FORMAL_1b47e0ed_seed{11,23,37}` trees, the three `_best` trees, the
  three `snapshots/FORMAL_1b47e0ed_seed{11,23,37}` trees, the v2 seed 53 tree,
  and the three manifest directories (`formal_training_v3`,
  `formal_training_v2`, `formal_training_ready`).

## 9. Launch gate

```
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" src/run_formal_training.py --check
```

→ exit 0; `checks.all_pass: true`, `checks.budget_contract_ok: true`,
`checks.h2_sealed: true`, all other checks true;
`status: GO_FORMAL_TRAINING`;
`config_sha256: 1b47e0ed0e5e469a510a77c6afb9b058aba3e41b90bd98734f40593645ffd2ff`.

This matches `reports/cc03/phase25_results.md`.

## Not tested / inferred

- **INFERRED** — seeds 23/37/53 child interpreter (launch command recorded,
  process environment not).
- **INFERRED** — seed 11's two original launches used bare `python` (anaconda);
  the phase report names `energy_env` for the run and the resume interpreter is
  not separately recorded.
- **INFERRED** — the H2 seal, for the reason in item 5.
- **INFERRED** — the two harness OOM kills, which are cited from the phase
  reports and have no artifact.
- **NOT_TESTED** — seed 37 beyond round 3; the run stopped and was not
  extended.
- **NOT_TESTED** — seed 11's exact file mtimes beyond the manifest's
  `created_at` / `resumed_at`; mtimes are reported in `P0_evidence.json` but
  are not a provenance proof on their own.

No item is **FAIL** or **BLOCKED**.

## Exit

Every P0 item has a named verdict and no original artifact changed.
**`PASS_P0_INTEGRITY`.** The preconditions for P1 (mapping audit, best
checkpoint identity) are confirmed. Stopping here for acceptance; P1 has not
started.
