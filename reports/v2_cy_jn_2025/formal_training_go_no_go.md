# Formal training GO / NO-GO

**Conclusion: NO-GO for formal multi-seed training.**

The machinery is integrated and the terminal-neutral benchmark is complete,
but there is no new independent holdout. G10 fails and the holdout leg of G11
cannot be met. No formal training was started. Any run on the existing 2025
days is exploratory/development only and may not be reported as independent
generalisation.

## Gate table

| # | condition | state | evidence |
|---|---|---|---|
| G1 | main-experiment day-end SOC rule frozen | **PASS** | `terminal_soc_config.json`; all 96 B1 days end at exactly 0.5 per ESS |
| G2 | new B1 under the same 24-block mechanism, 96 days verified | **PASS** | `b1_terminal_neutral_results.csv`, manifest, audit; 96/96 feasible; NEAR 80/80 |
| G3 | B daily differential matches the settlement ledger | **PASS** | B paid only at block 23; `delta = G_pi − G_B1` error 0.0 in the smoke |
| G4 | C2 potential maths and implementation | **PASS** | smoke telescope error 1.2e-15; boundary identity test |
| G5 | gate wired into Actor/Critic/Target/Replay | **PASS** | `test_rl_b_c2_gate.py`; real 24-block smoke |
| G6 | ST gradient works; no illegal quote or out-of-box action | **PASS** | ST-gradient test; quotes legal; closed gate exactly 0; open-leg saturation 0/0 in the smoke |
| G7 | fixed reward scaling; no running standardisation | **PASS** | scale 1000.0; new trainer contains no `reward_std()` use |
| G8 | complete 24-block episode | **PASS** | 24/24 clears, 0 fallbacks, 0 LMP fallbacks, 0 voltage/line violations |
| G9 | full checkpoint save/restore/continue | **PASS** | real save→destroy→restore→update: action, buffer sample, RNG, losses, parameters, counters and date-exposure log identical |
| G10 | new independent holdout sourced and frozen | **FAIL** | supplied `docs/data.csv` is the already-viewed 2025 source; 137 dates, no new period |
| G11 | train/validation/holdout disjoint | **PARTIAL** | NEAR train/validation/test are disjoint by pair and date; no holdout exists, so its leg cannot be met |
| G12 | seeds, budget, hyperparameters and selection rules frozen | **PASS** | `formal_training_protocol.md`, `formal_training_seeds.json`, `trainer_configs.json` |
| G13 | full tests pass; global provider/defaults not polluted | **PASS** | 1353 passed, 0 failed, 53 deselected (`logs/full_test_suite.log`); no `config/defaults.yaml` change in this work |
| G14 | old V1/V2, old MATD3 and pilot artifacts retained | **PASS** | new files use new names; old rolling B1 and pilot directories untouched |

### Note on G13

The first full-suite run exposed test-order pollution introduced by the new
trainer tests: `build_env_for_pair` changes the global price provider and the
tests did not restore it, so later RL tests saw JN2025 prices. An autouse
fixture now reloads the defaults and resets the wholesale cache after every new
test. The recorded run above is the clean rerun; the failure was in the test
harness, not in the trainer or the clearing code.

## Why G10 is not fixable inside this worktree

`docs/data.csv` contains 2025-06-01 to 2025-12-31, 137 dates, 136 complete.
Those are the same dates already paired, split and inspected. A post-hoc
re-split is not a blind holdout. A valid G10 needs an external file from an
unused period, with unit, region, timestamp convention, completeness and
hashes audited before any model decision. None was supplied.

## What is allowed now

- Development/exploratory training on the existing NEAR split, explicitly
  labelled as such.
- Diagnostic runs on the viewed common test, reported as developmental.
- No formal 5-seed matrix; no independent-generalisation claim; no final paper
  result from these days.

## What would flip the conclusion

1. Acquire and freeze a new external period (price and, if possible, load),
   with at least 30 complete paired days across at least three seasons.
2. Update `new_holdout_manifest.json` and `pilot/holdout_status.json` from
   `false`/`not acquired` to the frozen hashes.
3. Re-run this gate table. G10 and the holdout leg of G11 must pass before the
   formal matrix starts.
