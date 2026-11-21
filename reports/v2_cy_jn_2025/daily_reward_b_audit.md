# Daily difference reward B — implementation audit (V2-CY-JN-2025)

Module: `src/daily_reward_b.py`; identities: `reward_b_c2_identities.json`;
tests: `tests/test_daily_reward_b.py`. No training.

## The execution interface, and why JN-Base is not its B1

The policy arm executes **24 blocks**, each clearing a 16-period rolling window
and committing four periods, with no terminal SOC pin. The truthful arm must
use the same interface. JN-Base is a **one-shot 96-period** clearing with
terminal SOC pinned, so the two are not the same mechanism.

The representative check (`b1_jnbase_comparison.csv`) shows the difference is
large, not cosmetic:

| pair | rolling truthful | JN-Base | difference | rolling end SOC |
|---|---:|---:|---:|---:|
| 2022-10-04 → 2025-10-04 | 554.79 | 349.88 | +204.91 | 0.432 |
| 2022-07-22 → 2025-07-15 | 106.91 | 0.00 | +106.91 | 0.432 |
| 2022-07-23 → 2025-07-13 | 31.01 | 0.00 | +31.01 | 0.432 |
| 2022-10-03 → 2025-10-03 | 0.00 | 0.00 | 0.00 | 0.500 |

**JN-Base is not reused as the day-level B1.** A new rolling B1 is registered by
`b1_daily_baseline.py`; JN-Base stays as a separate full-day benchmark and the
two are never mixed.

## Definition

For each day, both arms run the same 24-block execution with the same initial
SOC, load, price, network, storage parameters, cycle cost, block structure and
block settlement:

$$
R_i = G_i^{\text{policy}} - G_i^{B1},
\qquad
G_i = \sum_{t=0}^{23} \text{participant\_payoff}(\text{committed block } t).
$$

Only the quotes differ: the policy's in the first arm, `(0,0)` for all four
units in the second. SOC trajectories evolve independently; no transplant.

The day reward is paid **only at block 23**. Components are kept apart:
`G_policy`, `G_b1`, `delta`, `scaled_delta`, the per-block reward array, and the
discounted return. The implementation refuses a path that is not 24 blocks, a
non-finite payoff, or a policy/B1 unit mismatch.

## Cache

The B1 lookup key is exactly

```
pair_id + initial_soc + environment_config_hash + data_hash
```

and a miss raises `CacheKeyError`. There is no fallback to another day, another
SOC or an approximate value.

## Fixed scaling

`RewardBConfig.fixed_scale` is a declared positive constant. It is applied as
`delta / fixed_scale`; `reward_std` and the replay buffer are not consulted
(checked by a source test). The discounted training return is
`gamma^23 * delta / fixed_scale = 0.7936 * delta / fixed_scale`, a fixed
multiple of the day differential.

## Identity results (`reward_b_c2_identities.json`)

| check | result |
|---|---|
| only the last block pays | pass |
| discounted return equals `gamma^23 * scaled_delta` | pass |
| fixed scale recorded | 1000.0 in the identity test |
| C2 telescope max error | 3.6e-15 |
| C2 boundary max error | 2.5e-15 |
| terminal potential | exactly 0 |
| mid-episode termination rejected | pass |

## What this does not do

- It does not train or evaluate a policy.
- It does not pin terminal SOC; the rolling interface does not, and the B1 is
  the same interface. The differential therefore includes any difference in
  ending SOC. That is stated, not hidden.
- It does not claim the rolling B1 equals JN-Base. They differ by mechanism and
  by terminal treatment, and both are reported.
