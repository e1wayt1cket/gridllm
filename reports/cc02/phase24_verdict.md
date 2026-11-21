# CC-02 Phase 24 read-only verification

Scope: MASTER_PLAN_V1 §3, "the training loop must carry a minimum-round guard
(min 10, max 15, patience 3, early stop only at or after round 10, resume
restores all counters and state)".

Method: source inspection, a runtime probe of the guard functions, the pinned
hashes, and the completed-run evidence. No resume was executed.

Why resume was not exercised: `train_da_gate.py` resolves `run_dir` to
`ROOT/runs/<run-id>`, and `--resume` truncates `episodes.csv` and the JSONL
files in place (lines 624–627). Any live resume would write inside the frozen
run tree, which CC-02 forbids and which would destroy evidence. Resume findings
below are static-source evidence, marked accordingly.

## Runtime probe

```
PYTHONPATH=src python -c "import train_da_gate as t; ..."
GUARD_CHECKS_PASS
```

Asserted and passed: `should_early_stop(9, ev, 3, 10) is False`;
`should_early_stop(10, ev, 3, 10) is True`;
`should_early_stop(10, ev_with_new_best, 3, 10) is False`;
`episode_budget(53, 15) == 795`; `completed_rounds(795, 53) == 15`;
`completed_rounds(28, 53) == 1`.

Tests: `python -m pytest tests/test_training_contract.py
tests/test_frozen_guard.py tests/test_train_da_gate_guards.py -v` — 43 passed.

Pinned hashes: `train_da_gate.py` `a722b2f9…`, `rl_da_gate.py` `9ed0b833…`,
both matching the live manifest and each run's `source_sha256`.

## Matrix

| §3 requirement | evidence | verdict |
|---|---|---|
| guard runs inside the training loop, not just an external contract | `should_early_stop` at `src/train_da_gate.py:109`, called at lines 892–893 immediately after the round's validation; hash matches the pin; `--check` reports `code_sources_match: true` | `PASS` |
| minimum 10 rounds | floor `round_idx < max(min_rounds, 0)` returns False (line 116); formal launcher passes `--min-rounds 10`; `--seed 71 --dry-run` prints it; seeds 11/23/37 each stopped at round 10 | `PASS` on the formal path; `PARTIAL` globally — the CLI default is 0 (line 422) and `run_formal_training.py` is not in the pin list |
| maximum 15 rounds | `episode_budget(53, 15) == 795`; `completed_rounds(795, 53) == 15`; launcher passes `--exposures 15 --max-episodes 795` | `PARTIAL` — this is episode-budget arithmetic, not a round cap. Any change to `--exposures` or `--max-episodes` moves the ceiling |
| patience = 3 | frozen config `validation.early_stop_patience: 3`; launcher passes 3; run manifests record 3; probe: streak 2 → False, 3 → True | `PASS` |
| early stop only at or after round 10 | floor uses `round_idx = episode // len(order)` (line 755), i.e. completed rounds only; seeds 11/23/37 stopped at 10; `test_train_da_gate_guards.py` covers rounds 3–9 not stopping | `PASS` on the formal path |
| runtime rejects patience < 1 | `patience <= 0` makes the guard return False — early stopping is silently disabled, not refused. The `early_stop_patience >= 1` rule exists only in `training_contract.py` | `FAIL` at runtime; `PASS` in the contract |
| resume restores model, counters, buffer, RNG | `checkpoint_loader_v2.load_checkpoint` returns counters `672/28/545/672` and the RNG state for seed 53; `DAGateTrainer.load` applies them (observed in `seed53_replay.json → restored_state`) | `PASS` for the loader; `INFERRED` for a full resume run |
| resume validates patience and min_rounds | the resume `checks` dict (lines 585–602) covers seed, data hashes, market/b1 hashes, split, exposures, budget, train subset, action schema, quote map and source hashes — not `early_stop_patience` or `min_rounds`; both are re-read from the CLI, whose default for `--min-rounds` is 0 | `FAIL` (static) |
| resume restores partial-round state | `completed_rounds` is a ceiling (line 106), so a 28-episode checkpoint resumes at `resume_episode = 53`. The 25 training days that would have run in episodes 28–52 are never exposed in round 0, leaving them one exposure short of the declared 15 | `FAIL` (static) |
| guard tests cover a real resume | `test_resume_restores_completed_rounds_and_patience` calls `completed_rounds`, `no_improvement_streak` and `should_early_stop` only; it builds no trainer and reads no checkpoint | `NOT_TESTED` |

## Overall

The minimum-round guard is a genuine runtime guard on the formal launch path:
`PASS`, evidenced by the source line, the probe, the pinned hash and three runs
that all stopped at exactly round 10.

The §3 clause "resume restores all counters and state" is `FAIL` on two static
points — patience and `min_rounds` are not validated against the saved manifest,
and a partial round is rounded up so its remaining days are skipped. A resume
launched without `--min-rounds` would run with the floor at 0 and could early
stop at round 1.

Per the CC-02 disposition, these do not block CC-02 but block Phase 25/26 until
the user either commissions a fix (which edits `train_da_gate.py`, a pinned
file, and therefore needs a new candidate id, a new manifest and a re-freeze) or
explicitly accepts the risk. Recorded as TD-05.

## Not tested

- A real resume, full round or partial round, against any run.
- The runtime behaviour when a resume supplies a different `patience` or
  `min_rounds` than the run manifest.
- Whether the guard prevents a stop attempt before round 10 in a live run
  (only the pure function and the three completed runs were observed).
