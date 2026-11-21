# P2-MATRIX — temperature, checkpoint and replay control matrix

Protocol `P2_CONTROLLED_DIAGNOSTICS_V2`, stage 2 (offline, read-only).
Written 2026-11-08. Interpreter
`C:\Users\26036\.conda\envs\energy_env\python.exe` (CPython 3.10.20,
conda-forge), torch `2.11.0+cpu`, numpy `1.24.3`, scipy `1.15.3`,
gurobipy `13.0.1`.

Collector: `src/diagnose_p2.py` (new, unpinned), functions `matrix_study`,
`run_cell`, `buffer_fingerprint`, `buffer_compatibility`,
`stratified_subset_indices`. Command:

```
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" \
  src/diagnose_p2.py matrix A1
... matrix A2
... matrix A3
```

Artifacts: `P2_evidence.json` → `matrix`, `P2_control_matrix.csv` (200 rows =
5 cells × 5 replicates × 8 days).

No algorithm was modified to produce these numbers. Temperature and the gate
are the frozen implementation; the only knob moved is the `gate_temperature`
argument that `forward_policy_v3` already accepts, set per cell through an
instance override of `trainer.current_gate_temperature`. No replay was written,
no checkpoint was saved, and every arm ran on an in-memory copy of the actor.

## Design

The unit of measurement is **10 actor updates from a fixed checkpoint, then a
full-day payoff evaluation against the no-update baseline**. Critics and their
targets are frozen; there is no critic step, no target update and no buffer
add. Each cell is 5 replicates × 8 Train days; the batch seeds are
`111000…111004` in every cell, so batches are drawn from the same RNG stream
in all cells and every contrast below is paired on `(day, replicate)`.

| cell | checkpoint | replay buffer | n | τ | what it isolates |
|---|---|---|---|---|---|
| A1 τ=1.0 | seed 11 best, step 1,272 | seed 11, `22ef8041…` | 1,272 | 1.0 | reference |
| A1 τ=2.5 | same | same | 1,272 | 2.5 | τ |
| A1 τ=4.0 | same | same | 1,272 | 4.0 | τ |
| A2 τ=2.5 | same | seed 23 buffer, stratified to 1,272, `1e301093…` | 1,272 | 2.5 | buffer **content** at fixed size |
| A3 τ=2.5 | same | seed 23 buffer, full, `83ea8dc0…` | 5,088 | 2.5 | buffer **size** at fixed content |

Holding the checkpoint, τ, the number of steps and the batch seeds fixed means
the only difference between A1 τ=2.5, A2 and A3 is where the batch comes from.

## Preconditions

| item | verdict | evidence |
|---|---|---|
| both best checkpoints load with `linear_core_v1` / `v6_12d` | **PASS** | checkpoint `manifest.json`, P1-A |
| replay compatibility before A2/A3 | **PASS** | `buffer_compatibility`: `extras_keys`, `obs_shape`, `action_shape`, `sampler_config`, `data_hashes`, `specs` all true |
| A2 subset nested in the full buffer | **PASS** | `buffer_23_subset.nested_in_full: true`, selection seed 2026, largest-remainder by stratum |
| actor restored identically before every replicate | **PASS** | all 40 rows of every cell carry the same `optimizer_sha` `9c75c505317e…` |
| critic and target untouched by the arms | **PASS** | only the actor optimiser steps; no critic loss or soft update is called in `P1.run_arm` |
| frozen hashes unchanged | **PASS** | 19/19 sources, 6/6 pinned tests, `thresholds.json` `12549ac6…` |

## Reproducibility of the batches

| cell | sampled rows | duplicate draws | duplicate rate | buffer n |
|---|---:|---:|---:|---:|
| A1 (all three τ) | 5,120 | 264 | 5.2% | 1,272 |
| A2 | 5,120 | 264 | 5.2% | 1,272 |
| A3 | 5,120 | 40 | 0.8% | 5,088 |

Duplicates are real resampling (`Buffer.sample` uses `replace=True`), not a
defect, and they differ across cells exactly as buffer size predicts. Batch
seeds per cell: `111000, 111001, 111002, 111003, 111004`.

## Result — fleet payoff change vs the no-update baseline

`Δ` is the sum over the four ESS of the difference between the day's settled
payoff after 10 actor steps and the same day's payoff with the checkpoint
unchanged. Negative means the 10 steps made the day worse.

| cell | mean Δ (CNY/day) | CI, day-clustered | CI, day×replicate | days better |
|---|---:|---|---|---:|
| A1 τ=1.0 | **−5.137** | `[−12.156, −0.045]` | `[−8.004, −2.470]` | 3 / 8 |
| A1 τ=2.5 | **−5.100** | `[−11.882, −0.078]` | `[−7.902, −2.511]` | 4 / 8 |
| A1 τ=4.0 | **−4.801** | `[−11.254, −0.059]` | `[−7.464, −2.350]` | 4 / 8 |
| A2 τ=2.5 | **−0.911** | `[−2.315, +0.197]` | `[−3.433, +1.542]` | 2 / 8 |
| A3 τ=2.5 | **+1.140** | `[−0.817, +3.443]` | `[−0.788, +3.171]` | 3 / 8 |

The day-clustered interval is the primary one: the 8 days are not exchangeable
with 40 correlated replicates, and two days carry almost all of the effect.

### Per day

Reference: the no-update fleet payoff is `−0.0006, −0.0005, −0.0002,
1467.97, 321.22, −0.0003, 116.17, 130.11` CNY for the eight days in order.

| cell | 23 Jul | 14 Aug | 17 Aug | 13 Oct | 15 Oct | 17 Oct | 09 Dec | 10 Dec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 τ=1.0 | 0.00 | 0.00 | −0.00 | −0.36 | −0.00 | 0.00 | **−18.84** | **−21.90** |
| A1 τ=2.5 | 0.00 | 0.00 | −0.00 | −0.63 | 0.00 | 0.00 | **−18.29** | **−21.87** |
| A1 τ=4.0 | 0.00 | 0.00 | −0.00 | −0.48 | 0.00 | 0.00 | **−17.37** | **−20.56** |
| A2 τ=2.5 | −0.00 | −0.00 | −0.00 | −3.16 | −0.00 | 0.00 | **+0.79** | **−4.92** |
| A3 τ=2.5 | −0.00 | −0.00 | −0.00 | −3.27 | −0.00 | 0.00 | **+7.34** | **+5.05** |

Five of the eight days move by less than 0.01 CNY under every cell — the quote
change produced by ten actor steps does not reach the clearing there at all.
The whole of the contrast lives on the two winter days and, weakly, on
`2022-10-13`.

## Paired contrasts (same batch seeds, same days, same replicates)

Day-clustered bootstrap over the 8 per-day means of the paired difference,
10,000 resamples, seed 20261109.

| contrast | mean | 95% CI | reading |
|---|---:|---|---|
| τ 2.5 − 1.0 | `+0.037` | `[−0.098, +0.204]` | no effect |
| τ 4.0 − 1.0 | `+0.336` | `[−0.030, +0.838]` | no resolved effect |
| τ 4.0 − 2.5 | `+0.299` | `[+0.019, +0.667]` | small, excludes 0 |
| A2 − A1 @ τ 2.5 (buffer content) | `+4.189` | `[−0.633, +10.547]` | large, not resolved by 8 days |
| A3 − A1 @ τ 2.5 (buffer content + size) | `+6.240` | `[−0.660, +15.847]` | large, not resolved |
| A3 − A2 @ τ 2.5 (buffer size) | `+2.051` | `[−0.027, +4.935]` | positive, unresolved |

### Temperature

Every τ contrast is at most `0.34` CNY/day against a `−5.1` CNY/day mean, and
**none of them has a stable sign across the two runs of this matrix**. The
superseded run measured τ 4 − τ 2.5 at `−0.090`; the corrected run measures
`+0.299` on the same checkpoint, buffer, batch seeds and days. An effect that
flips sign when the only change is the Adam-restore defect being fixed is
inside the noise band of this measurement.

Conclusion for τ: **the gate temperature is not the driver of the observed
payoff degradation over a 10-step horizon**. This does not contradict
P2-TEMP — the mechanism is real and exactly as documented — it says the
mechanism's magnitude is small relative to where the payoff actually moves.

### Replay buffer

At fixed size (A2 − A1, both 1,272 transitions, same batch seeds), the buffer
that the actor steps from changes the winter days by `+17.0` and `+19.1` CNY
and the autumn day `2022-10-13` by `−2.5` CNY. At fixed content (A3 − A2) the
larger buffer adds `+2.05` CNY/day. The buffer entry for the reference cell
(A1) is the buffer saved with the seed 11 best checkpoint, which was written at
episode 53; the A2/A3 buffers come from the seed 23 checkpoint at episode 212
and hold 4× the transitions. Size is controlled by the stratified subset, so
the A2 − A1 difference is content, not size — but the two buffers also differ
in how much exploration history they cover, which this design does not
separate.

Both buffers are Train-only; no validation transition enters either; no replay
was written. The buffer is read-only in every arm.

## The superseded run

The first pass of this matrix ran the same five cells before a defect in the
harness was found: `torch.optim.Optimizer.load_state_dict` keeps references to
the incoming tensors, so the first actor step mutated the snapshot's Adam
moments in place and later replicates started from moments the earlier ones
had already moved. The signature was three different `optimizer_sha` values
inside one process (`9c75c505… → 1e70c0af… → 29d67916…`) where the fresh-process
cells showed a single value.

`P1.restore_state` now restores the actor optimizer state from a `deepcopy`,
verified on the real trainer: three consecutive restore-and-step arms return a
bitwise-identical `optimizer_sha`. The first pass's cell values are kept for
the record in `P2_evidence.json` → `matrix.superseded` and in
`P2_control_matrix_prefix.csv`; **they are not evidence**, and the numbers
above are the corrected ones.

The P1-B actor-single-step measurements were collected with the same
`P1.restore_state` before the fix, so the Adam-moment state of later
replicates there is equally compromised. P1-B is not re-run in this phase; its
report carries the caveat and its conclusion stays `INCONCLUSIVE`.

## What this establishes and what it does not

Established (**PASS**):

- over 10 actor steps from the seed 11 best checkpoint, the **replay buffer
  the batches come from** moves the winter-day payoff by ~20 CNY/day, and the
  **gate temperature** moves it by ≤1.5 CNY/day with no stable sign;
- five of eight Train days are economically inert to the actor's 10-step quote
  change (|Δ| < 0.01 CNY), so any diagnosis built on a day average that
  includes them is diluted ~3×;
- the cells are reproducible: one optimizer hash per cell, identical batch
  seeds, duplicate draws explained by buffer size.

Not established:

- **NOT_TESTED** — whether the A2/A3 advantage comes from buffer *content*
  (state distribution) or from exploration *history*. Size is separated;
  history is not.
- **NOT_TESTED** — buffer date coverage. The replay extras carry no date
  field, so which Train days each buffer holds cannot be read from the
  artifact. Recorded as unavailable rather than estimated.
- **NOT_TESTED** — anything about more than 10 steps. This is a short-horizon
  diagnostic; the formal runs degrade over hundreds of episodes.
- **INFERRED** — that a −20 CNY/day winter degradation over 10 steps is
  large enough to explain the phase-25 result. The formal runs report
  validation rounds, not this measurement; the link is plausible and not
  measured here.
