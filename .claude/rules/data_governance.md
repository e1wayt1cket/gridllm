# Data governance

Applies to every dataset build, split, training run, evaluation and report.

## Split

- Frozen block sizes: train 53, validation 11, test 11, holiday block 5.
- The four blocks are disjoint by pair and by date.
- Test carries the legacy label `LEGACY_SEEN_TEST`; the holiday block is
  legacy-seen as well.
- Test and holiday data never enter training, replay, model selection or
  hyperparameter choice.
- The H2 price-date holdout remains sealed (`h2_opened: false`).

## Leakage

- The observation must not contain a future realised price. `avg_lmp_hist_norm`
  is history only; every price feature is normalised by the frozen
  `PRICE_REF_CNY_MWH`.
- `mv_value` is not an observation feature.
- Validation is not written to the replay buffer.

## Known limits that must be stated, not hidden

- Load and price are paired across years. The 2025 price series is the only
  source; there is no independent holdout.
- No result on these days supports an independent-generalisation claim.

## Files

- Data manifests, split files and holdout status are read-only once frozen.
- A new dataset version gets a new directory; existing manifests are never
  edited in place.

## Enforcement

- `src/training_contract.py` checks the split counts, disjointness, legacy
  label, H2 seal and the test/holiday boundaries.
- `src/run_formal_training.py --check` reports `h2_sealed` and the boundary
  flags; a false value is a hard NO_GO.
