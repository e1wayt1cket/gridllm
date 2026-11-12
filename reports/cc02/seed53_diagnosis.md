# CC-02 seed 53 diagnosis

Classification: **`NOT_REPRODUCED`**, marked `INFERRED` for everything below it.
The run itself failed; the failure could not be made to happen again from the
same weights.

Evidence: `reports/cc02/seed53_replay.json` (full capture), produced by
`src/diagnose_seed53_clearing.py` (read-only against the run directory). The
run tree was hashed before and after and is unchanged
(`guardrail_before.json` / `guardrail_after.json`).

## What failed

`runs/FORMAL_2148a577_seed53` stopped with

```
RuntimeError: 2022-10-20|2025-10-20 block 23: clearing failed; training stops
```

raised by `src/rl_da_gate.py::train_day` when `env._last_result is None`. The
underlying solver exception is swallowed by `src/rl_env.BiddingEnv.step`
(`except Exception: result = None`), so the original error type and message
were never recorded.

It stopped on the 29th training day, after 28 completed episodes and 672 block
steps. Verdict: `NO_GO_RUN_FAILED` (`reports/cc02/seed_audits.md`).

## Preconditions checked

| check | result |
|---|---|
| source hashes against the run manifest | 12 checked, 0 mismatched |
| pair inside the run's train subset | true |
| checkpoint restore | `total_steps 672`, `episode 28`, `updates 545`, `buffer_len 672`, RNG keys `numpy/python/torch_cpu/torch_cuda` |
| truthful-bidding day | 24/24 blocks cleared |
| deterministic policy day (`train=False`) | 24/24 blocks cleared, all `OPTIMAL` |
| exploration sweep, 40 RNG seeds, full day each | 40/40 days completed all 24 blocks, 0 failures |
| terminal SOC reachable at block 23 (deterministic trajectory) | all four ESS reachable |
| stray solver dumps in the repo | none (no `*.ilp` anywhere, `gp.Model.write` disabled) |

## Limitation, stated up front

The failing checkpoint was written by the failure handler inside its `except`
block, so `rng.pt` is the state *after* block 23 had already sampled its
actions, and the original actions were never written to
`block_decisions.jsonl`. **The original stochastic trajectory cannot be
replayed.** The 40-seed sweep is 40 different draws from the same weights, not
the draw that failed. The classification follows from that: the failure is not
attributable to the weights on this day in any way the sweep can observe.

## Why `NOT_REPRODUCED` and not `SOLVER_TRANSIENT`

The four-way rule in the plan named `SOLVER_TRANSIENT` for a same-input
inconsistency. That test was never available: no inputs were captured, because
the deterministic replay did not fail and therefore never recorded a failing
`clear_market` call. `block_stress` is absent from the report for the same
reason. With no repeated-input evidence, `SOLVER_TRANSIENT` would be a guess.

`DATA_INHERENT` is excluded: the truthful day cleared, and the pair is ordinary
training data with no NaN/Inf reported.

`ACTION_POLICY_INHERENT` is not established either: 40 exploration draws from
the same weights on the same day never failed.

## Bounded frequency

0 failures in 40 exploration-days gives an upper 95% bound of about 3/40 ≈ 7.5%
per day by the rule of three. That bound does not transfer to the rest of the
run: the failing policy was 28 episodes old, and the failure may depend on the
policy's state at that point in training. No claim is made about the failure
rate at episode 300 or 500.

## Mechanism hypothesis — `INFERRED`, not proven

The last block's clearing carries the terminal SOC equality. At block 23 the
deterministic trajectory leaves two of the four units with no headroom:

| unit | SOC at block 23 | 4-period reachable range | terminal 0.5 |
|---|---|---|---|
| ESS9 | 0.500 | [0.228, 0.730] | reachable |
| ESS14 | 0.270 | [0.100, 0.500] | reachable, but only at maximum charge every period |
| ESS17 | 0.500 | [0.228, 0.730] | reachable |
| ESS27 | 0.270 | [0.100, 0.500] | reachable, but only at maximum charge every period |

ESS14 and ESS27 can reach 0.5 only by charging at rated power in all four
remaining periods. The charge quote the policy emits for ESS14 at that block is
`150.0`, the top of its box, with the charge gate open. A unit that demands the
maximum charge price only charges if the clearing price clears it. If the
policy's own quote keeps a unit out of the market while its SOC is below the
reachable floor, the terminal SOC equality has no feasible solution and the
clearing returns INFEASIBLE, which is exactly the symptom recorded.

This is consistent with the evidence but not demonstrated by it: the
deterministic run reached the same SOC positions with the same saturated quote
and still cleared `OPTIMAL`, and none of the 40 exploration draws produced the
failing combination. Treat it as the first place to look, not as the cause.

The related design point is that the terminal SOC equality is a hard
constraint while the storage's willingness to charge is an endogenous price
quote. That combination can always be made infeasible by the agent's own
action.

## Disposition options

CC-02 recommends, and does not execute:

1. **Keep seed 53 as failed evidence and leave it alone.** This is the only
   option compatible with the training contract as written ("Seed 53 failed at
   episode 28; a failed run is still evidence"; "never rerun, resume over,
   rename, migrate or overwrite"). Cost: the frozen five-seed set supplies four
   completed seeds at most.
2. **Resume from the failure checkpoint.** Not recommended. The RNG is
   post-failure so the resumed run is not the continuation of the failed one;
   TD-05 means the resumed run would skip one full exposure for 25 training
   days; and `--resume` truncates `episodes.csv` and the JSONL files in place,
   destroying the evidence. It also violates the read-only rule.
3. **Rerun seed 53 under a new run-id.** Requires a user ruling: it conflicts
   with the contract's "never rerun", and the new run would not be the same run
   the contract names. It also needs a decision on whether a rerun-bearing
   directory is acceptable in the frozen set.
4. **Substitute a new seed.** The seed list `11/23/37/53/71` is frozen in
   `formal_training_config.yaml`; changing it is a config change and needs a new
   candidate id, a new manifest and a re-freeze — the same procedure a fix to
   TD-05 would need, so the two decisions are best made together.

Recommendation: option 1 plus launching seed 71 as planned, then deciding
between 3 and 4 at the point where four-vs-five seeds matters for the paper's
statistics. Do not resume.

## Not tested

- The original stochastic trajectory (impossible, see the limitation above).
- Solver behaviour on a repeated identical infeasible input (no captured input).
- Any failure rate for the rest of the training budget.
