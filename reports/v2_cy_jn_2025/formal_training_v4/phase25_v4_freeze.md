# Phase 25 — v4 freeze (W7/W8/W9) and G1 gate under 口径 A

Verdict: **PASS**. v4 is the live protocol; v3, v2 and `formal_training_ready`
are historical and read-only.

## Hashes

| set | config hash | prefix | status |
|---|---|---|---|
| v4 (live) | `7834d0aa4191f616211b34d1f0afc28ee25044bab0b123cc2b47172897b70a9b` | `7834d0aa` | `finalized`, `GO_FORMAL_TRAINING`, `protocol_version: 4` |
| v3 (historical) | `1b47e0ed…` | `1b47e0ed` | unchanged, read-only |
| v2 (historical) | `2148a577…` | `2148a577` | unchanged, read-only |
| ready (superseded) | `b21a97c1…` | `b21a97c1` | unchanged, read-only |

Every v4 run id is `FORMAL_7834d0aa_seed<N>`, so no v3/v2 directory is
touched.

## What changed

Mechanisms (each non-profit-selected, each unit-tested in
`tests/test_v4_mechanics.py`):

1. `src/rl_b_c2_gate.py` — W7 advantage hinge: the actor loss is
   `-ReLU(Q(a) - Q(a_truthful) - delta_eff) + lambda * anchor`, with
   `delta = 0.10` from `reports/algorithm_diagnostics/v4_hinge_calibration.json`
   (round-up-0.05 of the max Q variation over states whose payoff range over
   the three lowest quotes is below 0.5 CNY; never profit-selected).
2. `src/rl_b_c2_gate.py` — W9 twin-gap-adaptive floor:
   `delta_eff = clamp(hinge_twin_gap_weight * |Q1 - Q2|, min=delta)`,
   `hinge_twin_gap_weight = 1.0`, read through
   `getattr(experiment, "hinge_twin_gap_weight", 0.0)`.
3. `src/rl_da_gate.py` — W8 calibration exposure: `calibration_interval = 8`;
   every 8th episode resamples open-leg quotes uniformly over the box
   (`exploration_source = 3.0`). Also `hinge_twin_gap_weight = 1.0` on the
   experiment config with a non-negative check, and TD-02's atomic
   `EpisodeClearingError`.
4. `src/train_da_gate.py` — CLI flags `--advantage-hinge-delta` (0.10),
   `--calibration-interval` (8), `--hinge-twin-gap-weight` (1.0), all frozen
   as defaults; `episodes.csv` gains `calibration`, `hinge_active_*`,
   `hinge_twin_gap_*`; TD-01 (log flushed before hashes) and TD-10
   (interpreter recorded).
5. `src/run_formal_training.py` — TD-11: `--resume` re-emits the frozen
   command.
6. `src/diagnose_p2_v4.py` — 口径 A (user-approved 2026-11-09): the top-1
   vote counts only states whose payoff range over the quote grid clears
   `TOP1_FLAT_TOL_CNY = 0.5` (the hinge calibration's own
   `flat_payoff_tol_cny`), and the critic's pick counts as agreement when its
   payoff is within that tolerance of the true best. The strict rate is kept
   as `top1_agreement_strict` and `payoff_range` / `top1_strict` are recorded
   per state.

Freeze plumbing:

7. `src/make_formal_training_manifests.py` — `SOURCE_FILES` adds
   `calibrate_hinge_v4.py`; the config gains a `hinge` block
   (delta, twin_gap_weight, form, calibrated_by, selection_rule, non-profit
   note) and `optimizer.calibration_interval = 8`.
8. `.claude/hooks/protect_frozen.py` — `MANIFEST_DIRS` leads with v4; v3, v2
   and `ready` stay in the list as historical directories and run trees.
9. `tests/test_frozen_guard.py` — v4 manifest, prefix and run-tree deny
   entries alongside the v3/v2/ready ones.
10. `TECHNICAL_DEBT.md` — TD-01/02/11 closed, TD-10 half closed (recording
    done, resume-refusal open). `AGENTS.md` — formal DA-gate stack design
    point.

## The G1 gate that authorised this freeze

| generation | G1b (median / top-1 / verdict) | G1c |
|---|---|---|
| V4A (anchor only) | 0.70 / 0.80 / valid | seed23 CUMULATIVE_DEGRADATION |
| V4B (+W7) | -0.10 / 0.14 / invalid | no degradation |
| V4C (+W8) | -0.30 / 0.29 / invalid | no degradation |
| V4D (+W9) | **0.80 / 0.50 / valid** | **PASS, both seeds** |

V4D's top-1 sits exactly on the 0.50 gate line: 8 of 9 states are
payoff-decisive, 4 of 8 agree (one of them only under the tolerance; the
strict count is 3/9 = 0.33). The misses are the documented correlated-error
false peaks (`twin_gap < delta`, not gated by W9's design) and the autumn day
2022-10-13 at about -30 CNY/day. Full mechanism analysis:
`b/W9双分歧铰链_G1重测_2026-11-09.md`.

## Verification

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | manifest generation + `--check` | PASS | all_pass true, no problems; git_drift is provenance-only |
| 2 | `run_formal_training.py --check` | PASS | all checks true, budget_contract_ok, config `7834d0aa…` |
| 3 | `training_contract.py --check` | PASS | ok, no violations |
| 4 | refusal before finalize | PASS | `--seed 11 --dry-run` refused while status was null |
| 5 | finalize + dry run | PASS | GO_FORMAL_TRAINING; dry run emits `… --seed 11 --exposures 15 --early-stop-patience 3 --min-rounds 10 --max-rounds 15 --max-episodes 795 --run-id FORMAL_7834d0aa_seed11 --trace-blocks` |
| 6 | focused suites | PASS | 96 passed (frozen guard, contract, launch guards, resume, v4 mechanics) |
| 7 | full suite (口径 A code) | PASS | 1452 passed, 54 deselected in 468.19s |
| 8 | full suite (post-freeze tree) | PASS | 1456 passed, 54 deselected in 466.97s; the 4 new v4 guard entries included |

The deselected test is the pre-existing matplotlib environment failure,
unchanged.

## Not tested

- **A live v4 launch.** The dry run prints the frozen command; no seed was
  started at freeze time. G2 follows this freeze.
- **A live v4 resume.** `--resume` is covered by the launcher's argument
  assembly and the trainer's rule-mismatch checks, not by a stopped v4 run.

## Not done

No seed was launched at freeze time. No v3/v2/ready file, run or snapshot was
modified. No commit or push.
