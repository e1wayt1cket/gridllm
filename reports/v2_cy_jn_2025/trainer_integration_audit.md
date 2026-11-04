# Trainer integration audit — MATD3_B_C2_GATE

Branch module: `src/rl_b_c2_gate.py`. The old MATD3 entry points
(`rl_bidding.MATD3`, `rl_td3.TD3`, `run_matd3_pilot.py`) are unchanged and no
pilot artifact is overwritten.

## What is connected

| component | implementation | evidence |
|---|---|---|
| Actor | `GateActor`: obs → 4 logits per ESS | `test_rl_b_c2_gate.py` |
| Truthful gate | `truthful_gate_actor_adapter`; 2 gates per ESS, 8 total | closed-gate and ST-gradient tests |
| Twin critic | `rl_bidding.CentralizedCritic`, action = executed quote | input is the quote, not the gate |
| Target actor | deterministic hard gate | `target_quotes`, smoothing on price logits only |
| Target critics | copied and `.eval()` | checkpoint round trip |
| Replay | `GateReplayBuffer`: quote, gate, probability, gate logit, price logit, Bernoulli draw | buffer round-trip test |
| B reward | `daily_reward_b`, paid at block 23 only | B-only test |
| C2 reward | `c2_reward`, SOC + time potential, γ = 0.99, zero at block 24 | C2 boundary-identity test |
| Fixed scale | `ExperimentConfig.fixed_scale = 1000.0`; no `reward_std()` | source and smoke audit |
| Checkpoint | schema-v2 writer/loader; gate extras and the date-exposure log saved | real save/destroy/restore/update test |
| B1 key | mechanism-bound hash, not only the market config hash | `load_b1_cache` refuses a manifest without it |
| Terminal mechanism | 24-block env with per-ESS SOC0 = SOC24 = 0.5 | `train_day` refuses any other mechanism |

## Reward definitions

**B-only**

\[
R_i = G_i^{\pi} - G_i^{B1},
\]

paid once at block 23. The replay reward is `R_i / fixed_scale`; every other
block is exactly zero.

**B+C2**

\[
r_{i,t} = r^B_{i,t} + \gamma \Phi_i(s_{t+1}, t+1) - \Phi_i(s_t, t),
\]

with \(\Phi_i(soc,t) = (24-t)/24 \cdot \kappa \cdot e_{max} \cdot (soc-soc_0)\),
\(\gamma=0.99\), \(\Phi(\cdot,24)=0\). The per-block reward is divided by the same
fixed scale. The C2 telescope changes the return only by a policy-independent
boundary term; the test checks that identity to 1e-9.

The old block-local differential reward A and the deviation penalty are not
part of either branch. `rl_b_c2_gate` contains no `cycle_cost` subtraction: the
payoff from `participant_payoff` already carries it once.

## Gate contract

- Forward is a hard gate: a closed gate produces exactly `0.0` at any finite
  logit.
- Backward is the biased straight-through surrogate `p(1-p)`; the price branch
  keeps its exact sigmoid derivative.
- Training samples Bernoulli(p); evaluation uses the deterministic threshold.
- The critic consumes the executed quote.
- Target policy smoothing is applied to the price logits only; closed gates
  remain exactly zero.
- Initial gate logits are the default near-zero linear output, so the initial
  open probability is 0.5. Temperature is 1.0.

The gate estimator is frozen as straight-through + Bernoulli. Any change is a
new experiment id and must be frozen before training.

## Frozen trainer configuration

| knob | value |
|---|---|
| gamma | 0.99 per block |
| fixed reward scale | 1000.0 |
| gate temperature | 1.0 |
| price exploration noise (logit) | 0.5 |
| target policy noise (logit) | 0.2 |
| target noise clip | 0.5 |
| actor / critic learning rate | 3e-4 |
| tau | 0.005 |
| policy delay | 2 |
| batch size | 128 |
| buffer capacity | 100,000 |
| start steps | 0 |

These values are written to `data/processed/v2_cy_jn_2025/trainer_configs.json`
with a hash per branch.

## Tests

- `tests/test_rl_b_c2_gate.py`: 6 tests (closed-gate exact zero, ST gradient,
  buffer round trip, B-only payment, C2 identity, real checkpoint round trip).
- `tests/test_terminal_soc_day_end.py`: 2 tests (horizon semantics, absolute
  per-ESS day-end SOC over a full 24-block day).
- Regression run around the changed clearing, checkpoint and reward modules:
  97 passed.
- The tests restore the global price provider after each case; without that,
  the suite-order pollution that the first full run exposed reappears.

## Not done

- No formal or multi-seed training.
- No alternative gate estimator (REINFORCE, Gumbel) was implemented.
- No new independent holdout; G10 is not met.
- The quality of the learned policy is not evaluated here.
