# Future holdout protocol (V2-CY-JN-2025)

## Current status: no unseen 2025 day remains

| set | days | status |
|---|---:|---|
| common test (11) | 11 | **viewed** in the pilot and in every later report |
| national-day block (5) | 5 | **viewed** |
| FULL96-only long pairs (16) | 16 | used in FULL96 training and analyzed; not a holdout |
| common train / validation (53 / 11) | 64 | used for training and tuning |

The 2025 file is six months long and its complete days are already assigned
and inspected. Any re-split of these days is **post-hoc** and cannot be called
a blind holdout. Until new data arrives, every result on these days is a
**developmental / exploratory** result, and no claim of independent
generalization may be made.

## A valid future holdout (in preference order)

1. **A new external price period.** Acquire a later Jiangsu day-ahead series
   (e.g. 2026) and pair it with the frozen 2022 load days using the same
   protocol. Before use, audit its unit, region, timestamp convention,
   completeness, floor/cap and publication timing exactly as `data_audit.md`
   did for 2025. Freeze the pairing and its hashes before any model touches it.
2. **A new external load-plus-price dataset** (different year or region),
   frozen as a new library. It must not be used in any prior reward, action or
   hyperparameter decision.
3. **Only if 1–2 are impossible:** a pre-registered temporal block from a file
   never used for selection. With the current 2025 file this option does not
   exist — all its complete days have been viewed.

## Rules for using a holdout

- Freeze the model, reward definition (B/C2), potential, fixed scale, action
  parameterization, gate estimator and hyperparameters **before** the holdout
  is opened.
- Open it once. No iteration, no retuning, no "one more look".
- Report the holdout source, dates, hashes and per-day rows; state whether the
  set is blind or already viewed.
- The 11-day common test remains available as a viewed development benchmark;
  it may be reported, but only with that label.

## Status artifact

`data/processed/v2_cy_jn_2025/pilot/holdout_status.json` records the current
sets, their viewed flag, and the candidate options; it must be updated when a
new holdout is acquired.
