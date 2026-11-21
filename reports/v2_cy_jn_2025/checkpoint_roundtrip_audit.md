# Checkpoint round-trip audit (V2-CY-JN-2025)

Writer: `src/checkpoint_writer_v2.py`; loader: `src/checkpoint_loader_v2.py`;
validator: `src/validate_checkpoint_schema.py`; round-trip script:
`src/audit_checkpoint_roundtrip.py`; results:
`checkpoint_roundtrip_results.json`; tests: `tests/test_checkpoint_roundtrip.py`,
`tests/test_checkpoint_schema.py`. No training, no environment.

## What is written

`save_checkpoint` writes, under `runs/<run_id>/`:

- `checkpoints/step_XXXXXXXX/actors/<agent>.pt`
- `.../actor_targets/<agent>.pt`
- `.../critics/<agent>.pt` (the twin critic's state dict)
- `.../critic_targets/<agent>.pt`
- `.../optimizers.pt` (actor and critic optimizers, including Adam moments)
- `.../rng.pt` (Python, NumPy, torch CPU, torch CUDA when present)
- `.../counters.json` (steps, episode, updates, buffer length)
- `.../buffer/replay.npz` (or a declared absence with a reason)
- `manifest.json` (schema v2: identity, specs, networks, optimizers, counters,
  rng, buffer, reward, hyperparameters, data, exposure, provenance)

The manifest is validated against the schema **before** it is written: a
missing critic list, an unsaved buffer without a reason, or a malformed hash
refuses the save.

## Round-trip procedure and result

The script creates tiny actor/critic/target modules and two Adam optimizers,
advances them once so the optimizer state is non-empty, seeds all RNGs, saves,
**deletes the original objects**, creates fresh modules/optimizers, loads, and
compares:

| check | result |
|---|---|
| actor state dicts identical | pass |
| critic state dicts identical | pass |
| actor and critic optimizer states identical | pass |
| same observation → same deterministic action | pass |
| Python/NumPy/torch RNG restore the next draws | pass |
| replay buffer arrays restored (shape and content) | pass |
| counters restored | pass |
| config hash restored | pass |
| every manifest-named file present | pass |
| an incomplete checkpoint (missing critic) is refused | pass |

11/11 checks pass (`checkpoint_roundtrip_results.json`). The round trip is a
save/load test, not smoke training.

## What remains unimplemented

- The trainer does not call this writer yet. The fifth-phase pilot still has
  actors only; that gap is unchanged and documented.
- A resumed run needs the day schedule and exposure log continued, not
  restarted; the schema names them, the trainer must honour them.
- RNG restoration covers the global streams; per-agent generators, if a future
  policy introduces them, must be added to the bundle.
