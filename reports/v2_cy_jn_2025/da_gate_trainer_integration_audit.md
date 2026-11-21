# MATD3_DA_GATE_v2 trainer integration audit (P0)

Implemented in `src/rl_da_gate.py`, `src/da_quote_map.py`,
`src/da_replay.py` and `src/train_da_gate.py`. The economic core is inherited
unchanged from `rl_b_c2_gate`: B, C2, gamma=0.99, fixed scale 1000, terminal
SOC 0.5, twin critics and the existing SOCP clearing. Data:
`da_gate_offline_validation.json`, `runs/DEV_DA_GATE_seed11_v2/run_manifest.json`.

## Version and configuration

| item | value |
|---|---|
| experiment | `MATD3_DA_GATE_v2` |
| action schema | `storage_quote_v3` (version 3) |
| quote map | `linear_core_v1` |
| checkpoint schema | `matd3-checkpoint-v3` |
| D1 | linear-core / quadratic-tail map |
| D2 | 50% uniform + 50% joint-cell stratified, beta=0.5, cap=5 |
| D3 | epsilon 0.20→0.05, discharge full, charge half, high-region 50% |
| A1 | gate temperature 4→1 over 106 episodes; evaluation τ=1 |
| reward | B + C2, unchanged |
| gamma / scale / terminal | 0.99 / 1000 / SOC 0.5 |

## Interface checks

| item | result |
|---|---|
| Actor output | four logits: two gate, two price; unchanged dimension 4 |
| Critic input | 4×12 observation + 4×2 executed quotes; unchanged 56-dim |
| Target actor | hard gate, gate τ=1, v3 map, target price noise only |
| Replay action | executed quote after gate and exploration, shape 4×2 |
| Gate branch | hard forward, ST backward `p(1-p)`, per-leg independent |
| Price branch | v3 map; derivative audited; no quote clipping |
| Exploration | training only, bounded, logged as `explored`/source |
| Sampler | per-transition probability, uniform ratio, ESS, cell counts |
| B/C2 | inherited identities; B only at block 23; C2 telescope |
| Checkpoint | actors/targets/critics/critic targets/optimizers/RNG/replay |
| Extra checkpoint state | sampler config+counts, gate schedule, exploration, action schema |
| Data/config hashes | run manifest + checkpoint manifest |
| Exposure log | `exposure.json` |
| Old checkpoints | never loaded; R2/R10 read-only references |

## One-day end-to-end result

The offline script ran a complete day with forced exploration:

- 24/24 clears, all OPTIMAL; 0 voltage and 0 line violation cells.
- terminal SOC max error 0.0.
- B delta identity and C2 discounted identity passed for all four units.
- executed quotes legal in [0,150]/[0,300]; gates binary.
- 75 explored legs in the forced test; 24 transitions stored.
- checkpoint save/restore round-trips an open-gate action; v2 checkpoint is
  refused by the v3 loader.

The full check list (64 checks) is in `da_gate_offline_validation.json`.

## What is not changed

- The B reward is still the whole-day differential paid at block 23.
- C2 is the same state-only SOC potential; no terminal term was added.
- gamma, fixed scale, terminal SOC, action bounds, clearing engine and
  `line50_q2.0` are unchanged.
- OBS_V6 semantics are unchanged; no future LMP feature was added.
- No entropy reward, no REINFORCE, no Gumbel switch.

## Boundary

The trainer is integrated and tested, but the pilot's selected checkpoint is
round 1 and the later rounds degrade; integration is not evidence of policy
quality. The pilot report separates those claims.
