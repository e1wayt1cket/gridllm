# Daily return and discount audit (V2-CY-JN-2025)

Numerical facts: `data/processed/v2_cy_jn_2025/pilot/c2_identity_check.json`.
No training.

## The clock

| quantity | value |
|---|---:|
| discount factor | **0.99 per block** |
| block | 4 × 15 min = 1 h |
| blocks per episode | 24 |
| episode length | fixed 24, no early termination in this environment |
| $\sum_{t=0}^{23}\gamma^t$ | 21.4322 |
| first-block weight | 1.0 |
| last-block weight $\gamma^{23}$ | 0.7936 |
| first / last | 1.2601 |
| effective 15-min discount | 0.99749 |
| terminal SOC, training window | **not pinned** |
| terminal SOC, settlement | **pinned** |

## The four return quantities

1. **B raw day difference** $R_{i,\text{day}}=G_i^\pi-G_i^{B1}$, in CNY, no
   discount, full-day settlement with terminal SOC pinned.
2. **Base reward paid at the last block**: $r_{i,23}=R_{i,\text{day}}$, zero
   elsewhere.
3. **Discounted episode return**:
   $\sum_t\gamma^t r_{i,t}=\gamma^{23}R_{i,\text{day}}=0.7936\,R_{i,\text{day}}$.
   Because $T$ is fixed, this is a **fixed positive multiple** of the raw day
   difference: the ordering of policies is unchanged, but the number is not the
   day's money.
4. **C2 shaped return**:
   $\gamma^{23}R_{i,\text{day}}+\gamma^{24}\Phi_i(s_{24})-\Phi_i(s_0)$.
   With $\Phi_i(s_{24})=0$ and a potential measured from $soc_0$, the last
   term becomes $-\Phi_i(s_0)$, which is zero for the relative SOC potential
   and, in every case, independent of the policy. The numerical check gives a
   maximum telescope error of 2.0e-13 across 100 random paths; paths with the
   same boundary but different intermediate states agree to 3.4e-13; the
   terminal-zero variant agrees to 3.6e-14.

## Checks for the hazards listed in the brief

| hazard | state |
|---|---|
| early termination / variable episode length | none: `done` at 24 blocks; no early break in the environment |
| episode termination time varying | none: fixed 24 |
| reward standardization | **present**: `buffer.reward_std()` divides the batch at each update; it is not a fixed scale and changes as the buffer grows |
| different dates with different scaling | the scale is global over the buffer, not per date, but it depends on the day mix; it is not a date-specific scale |
| policy arm vs B1 terminal SOC | **inconsistent in training**: rolling windows do not pin terminal SOC; the settlement and B1 do. B and C2 require the settlement episode, so this must be fixed together with the reward change |

## Consequences

- The discounted base return is **not** the undiscounted day profit. Reporting
  it as such is wrong. Report $G_i^\pi$, $G_i^{B1}$, $\Delta G_i$.
- Under B with a fixed $T$, the $\gamma^{23}$ factor does not change the
  optimal policy; it does change the scale the critic learns.
- Under C2, the terminal potential must be zero (or provably policy-independent)
  and the potential must be bounded; the time-decaying SOC potential satisfies
  both without relying on terminal pinning during training.
- The running reward standardization should be replaced by a **fixed** scale
  for the next version, or the report must state that the return unit drifts.
- The training episode must end with the full-day settlement (terminal pinned),
  or the training return can reward an end-of-day SOC the evaluation removes.
  This is a change to the episode mechanics, not to the clearing engine.
