# Training readiness

**training_readiness_status = GO_FORMAL_TRAINING**

Formal training was **not** started. This decision authorises the frozen
protocol below and nothing else. It does not authorise an
independent-generalisation claim: the 11-day common test is
`LEGACY_SEEN_TEST`, the holiday block is viewed, and the old G10 NO-GO remains
the boundary for any generalisation claim.

## Scope

| item | value |
|---|---|
| dataset | JN-Base-NEAR / `common_split.json` near view |
| split seed | 20261101 |
| train / validation / test / holiday | 53 / 11 / 11 / 5 |
| test label | `LEGACY_SEEN_TEST` (test + holiday = 16 pairs) |
| H2 price-date holdout | sealed; not opened |
| accepted code candidate | `P21_candidate_001` (actor-loss v3 map) |
| config hash | `b21a97c15d16aae5ed0f8bd599e54a2f8aafb6e687bca2b63b06bd5a8da8e1e7` |

The formal run is a controlled cross-year simulation: 2022 metered load priced
by a 2025 Jiangnan day-ahead curve. It is not same-period settlement and its
test is not blind.

## Frozen protocol

| item | value |
|---|---|
| engine / support | SOCP, `line50_q2.0` |
| reward | B + C2; gamma 0.99; fixed scale 1000; no running standardisation |
| terminal SOC | 0.5 at the start and at the true day end; intermediate windows unpinned |
| cycle cost | 100 CNY/MWh |
| observation | OBS_V6, 12 dimensions, causal |
| action | `storage_quote_v3`, v3 `linear_core_v1` map, ESS9/14/17/27 |
| gate | 8 per-leg gates, straight-through Bernoulli, temperature 4→1 over 106 episodes, evaluation at 1 |
| networks | actor 256-128-4; centralized twin critic 256-256 |
| optimizer | Adam 3e-4; tau 0.005; policy delay 2; batch 128; buffer 100,000 |
| sampler | 50% uniform + stratified joint cells; beta 0.5; cap 5 |
| episode | 24 blocks, 4-period commit, 16-period lookahead |
| seeds | 11, 23, 37, 53, 71 |
| budget | 10–15 exposures/day = 530–795 episodes/seed; 12,720–19,080 block interactions |
| selection | mean fleet ΔG over the 11 validation days, settlement only |
| checkpoints | resume every round, best by validation ΔG, final at the end or early stop |
| early stop | 3 consecutive evaluations without a new best |
| test / holiday | never evaluated |
| H2 | never opened |

## Verification evidence

### Data

- `common_split.json` near view: 53 / 11 / 11 / 5; all four sets pairwise
  disjoint.
- `LEGACY_SEEN_TEST` equals the split test + holiday block, 16 pairs.
- All frozen input hashes match `formal_training_data_manifest.json`.
- H2 is sealed; the training entry's open audit did not open any H2 file, the
  legacy test index, the old near split or the FULL96 split.

### Environment

- The 20-episode smoke cleared 24/24 blocks per episode with 0 fallbacks,
  0 voltage cells, 0 line cells, 0 LMP fallbacks and terminal SOC 0.5.
- The B1 settlement identity `G_policy − G_B1 = delta` closes to 1e-6 in every
  smoke episode.
- The one-episode entry audit evaluated the 11 validation days without writing
  the replay.
- The OBS causality test gave identical observations when only prices after the
  committed lookahead changed, and a positive control changed the observation
  when the current block's load forecast changed.

### Algorithm

- The actor loss and execution share the v3 map; the update-path quote equals
  the execution-path quote under the same gate draw, over gate open/closed,
  both legs, batch input and low/mid/high logits.
- One instrumented actor update calls the v2 map zero times and the v3 map a
  positive number of times; the actor loss is finite.
- The Bellman target audit reconstructs the code's own formula. The twin gap is
  23.8 CNY (best) / 6.3 CNY (final) median; target smoothing changes the target
  by 4.2 / 0.6 CNY median, so smoothing is not the source of the gap.
- The candidate smoke has no persistent action degeneration: high-quote share
  is 0.6–1.9%, zero-quote share 41–53%, no new corner collapse.
- Candidate 001 is accepted under the pre-registered non-inferiority rule:
  best validation −18.5 vs control −25.0, 7/11 vs 6/11 positive days.

### Engineering

- Checkpoint save/destroy/restore/continue: actions, counters, buffer and one
  update's losses all match.
- Pause/resume: 5 episodes paused + 5 resumed = 10 unique episodes; all four
  logs have one row per episode; counters and 240 block interactions match.
- Entry output goes to a new run directory; the entry refuses a populated run
  directory.
- Each episode is traceable to date, block and ESS in `block_decisions.jsonl`
  (96 rows per episode).
- Full default suite: 1,361 core tests + 4 eval-consistency tests passed;
  36 slow tests deselected; 0 failed.

## Residual risks

1. **Critic ranking/calibration remains weak.** On the old checkpoints the
   critic's Q-top equals the matched return-top in 6/15 informative groups
   (best) and 1/15 (final); median within-group Spearman +0.238 / −0.143. The
   twin gap and target-smoothing audit did not support a target change, and the
   ESS14 credit collapse did not reproduce without dropout. No minimal fix was
   justified. The formal run is needed to test whether the larger budget
   changes this; this GO is a readiness decision, not a performance claim.
2. The smoke is a 20-episode diagnostic, not a generalisation result.
3. The load/price pairing is cross-year; the 2025 price is the only source.
4. The test is viewed (`LEGACY_SEEN_TEST`); no blind result is available.
5. The online critic has dropout; the routing and finite-difference checks use
   eval mode, while training-time critic gradients are stochastic.

## Starting the formal run

Only after an explicit start instruction. The prepared entry is:

```bash
# preflight only
PYTHONPATH=src python src/run_formal_training.py --check

# one seed (refuses unless the manifest status is GO_FORMAL_TRAINING)
PYTHONPATH=src python src/run_formal_training.py --seed 11

# all five frozen seeds
PYTHONPATH=src python src/run_formal_training.py --all
```

The entry runs `src/train_da_gate.py --split common --seed <seed>
--exposures 15 --early-stop-patience 3` with a new run id derived from the
config hash. It does not evaluate the test or holiday blocks and does not open
H2.

## Historical protection

No historical checkpoint, report, data file or manifest was modified. The D+A
best/final, R2/R10, Phase 18/19 artifacts, the V2 frozen library and the
original pair table remain read-only. New artifacts use new names and new run
directories. No commit or push was made.

## Decision

All Phase 22 acceptance groups pass: data, environment, algorithm and
engineering, plus the full default test suite. The residual critic risk is
recorded and does not change the readiness decision. **GO_FORMAL_TRAINING.**
