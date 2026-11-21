# Frozen-file patch for the v5 re-freeze

`frozen_patches.diff` carries the audit fixes that live in the 22 sources pinned
by `reports/v2_cy_jn_2025/formal_training_v4/formal_training_code_manifest.json`.
`.claude/hooks/protect_frozen.py` refuses Edit/Write on those paths, so the
patch is applied by hand. `git apply --check -p1` on the current tree passes.

## What each hunk does

| file | audit item | change |
|---|---|---|
| `dispatch_socp.py` | F6 | The terminal-SOC value term is zero when the endpoint is pinned by an equality: a constant the optimizer cannot move is not a term it traded off. |
| `dispatch_socp.py` | F4 | The nodal re-solve objective now carries `penalty_carbon_slack` / `penalty_re_slack`, as the first-pass objective does. Without them the second pass could violate a cap for free. |
| `rl_b_c2_gate.py` | F9 | The anchor is the sum of squared box-normalised leg deviations, averaged over the batch — the form `rl_td3.CALIBRATED_DEV_PENALTY` was calibrated for, not the leg-average. |
| `rl_da_gate.py` | F9 | `anchor_weight` default 0.01 → 0.10, the calibrated weight. |
| `rl_env.py` | 5.3 | The differential-reward counterfactual is documented as fleet-truthful. The code clears once with every RL agent's committed quotes truthful, so the comment's "holding the other agents' bids" was wrong. |
| `rl_env.py` | F10 | `DayCapture` starts every unrun period at the fleet's truthful declaration instead of zero. A prosumer's zero multiplier means "pay me nothing"; settlement over 96 periods was pricing the unrun part at zero. |
| `checkpoint_writer_v2.py` | 5.10 | A checkpoint is written into `<step>.tmp/` and renamed when complete; `manifest.json` is written through a `.tmp` and `os.replace`. |

## How to apply

From the repository root:

```powershell
git apply -p1 tmp/refreeze_v5/frozen_patches.diff
```

Then re-freeze into a new version directory rather than editing the live
manifest:

1. Copy `reports/v2_cy_jn_2025/formal_training_v4/` to
   `reports/v2_cy_jn_2025/formal_training_v5/`.
2. Regenerate `formal_training_code_manifest.json`,
   `formal_training_manifest.json` and `formal_training_config.yaml` in v5 with
   `src/make_formal_training_manifests.py`, so the hashes describe the sources
   the patch produced.
3. Point `MANIFEST_DIRS[0]` in `.claude/hooks/protect_frozen.py` at v5 and keep
   v4, v3, v2 and ready in the list.
4. Update `.claude/rules/training_contract.md` and `CLAUDE.md` to name v5 as
   the live manifest.
5. Run `PYTHONPATH=src python src/run_formal_training.py --check` and
   `scripts/run_tests.ps1 -Level Full`.

## Before you apply: two things this patch does not settle

1. **`checkpoint_writer_v2.save_checkpoint` now refuses an existing
   `checkpoints/step_<N>` directory** (`FileExistsError`). If the resume path
   ever re-saves a restored step into the same root, it will refuse. Run
   `tests/test_checkpoint_roundtrip.py` and `tests/test_checkpoint_schema.py`
   plus one resume smoke before trusting it.
2. **`PRICE_REF_CNY_MWH` (288.19757259048833) has to be re-derived.** The
   radial-tree correction changes the cleared price level enough that both
   price-history features saturate at their +3.0 clip. This is not a patch to a
   constant: it is part of the v5 candidate's observation definition, and
   `tests/test_rl_spec.py::test_a_blocks_realized_lmp_reaches_only_the_next_observation`
   is marked `xfail(strict=True)` until it is done.
