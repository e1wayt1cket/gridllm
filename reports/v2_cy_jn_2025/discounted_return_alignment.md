# Discounted return and the day objective (V2-CY-JN-2025)

Read-only audit of what the trainer actually optimizes. Facts are in
`data/processed/v2_cy_jn_2025/pilot/reward_design_facts.json`.

## The numbers

| quantity | value | source |
|---|---:|---|
| discount factor | **0.99**, applied **per block** | `MATD3(gamma=0.99)`; one transition per block |
| block | 4 periods × 15 min = 1 h | `BLOCK_SIZE=4` |
| blocks per episode | 24 | `N_BLOCKS=24` |
| $\sum_{t=0}^{23}\gamma^t$ | 21.43 | computed |
| weight of the last block | 0.794 | $\gamma^{23}$ |
| first / last weight | 1.26 | early blocks count 26 % more |
| effective 15-min discount | 0.99749 | $\gamma^{1/4}$ |
| terminal bootstrap | $\gamma(1-d)\min Q$ | `rl_bidding.update` |
| terminal SOC in a training window | **not pinned** | `pins_terminal_soc(...,"window")=False` |
| terminal SOC in the settlement | **pinned** | `pins_terminal_soc(...,"full_day")=True` |
| reward scale | divided by `buffer.reward_std()` before the target | `rl_bidding.update`; std over stored rewards |

## What is being optimized

Per agent $i$, the critic regresses

$$
Q_i \approx \mathbb{E}\Big[\sum_{t=0}^{T-1}\gamma^{t}\,
\frac{r_{i,t}}{\sigma}\Big],
\qquad \gamma=0.99,\ \sigma=\text{std of stored rewards},
$$

where $r_{i,t}$ is the **block-local** differential of Section A. The actor
maximizes $-\min(Q_1,Q_2)$ plus the deviation penalty. So the optimized
object is a **discounted, reward-standardized, block-local differential
return**, not

$$
G_i^\pi-G_i^{B1},
$$

the undiscounted day settlement difference the paper reports.

## Why the CNY unit does not close the gap

Four independent mismatches sit between the two expressions:

1. **Definition.** $r_{i,t}$ is local and state-matched; the day difference
   is a whole-path quantity. This is the dominant gap (sixth phase: local vs
   day correlation −0.32).
2. **Discount.** With $\gamma=0.99$ per hour, the first block is weighted
   1.26× the last. If the paper's objective is the day's undiscounted money,
   the training return is not it. The within-block four periods are summed
   undiscounted, so there is no 15-minute discount either — the model is
   effectively discounted per hour, not per quarter.
3. **Standardization.** Dividing by the std of all stored rewards puts Q in
   units of a **data-dependent scale**. It changes as the buffer grows and
   depends on the day mix; a Q of 1.0 is not one CNY. This is a variance
   device, not a statement about money.
4. **Terminal condition.** Training episodes clear rolling windows and do not
   pin terminal SOC; the settlement pins it. The training return can therefore
   be improved by ending the day at a different SOC, which the evaluation
   neutralizes. The pilot's episodes ended up to 0.40 away from $soc_0$.

A reward denominated in CNY does not make its discounted, standardized sum the
day's CNY income. The unit survives the discount; the **quantity** does not.

## What would align the training object

| option | effect on alignment | cost |
|---|---|---|
| **B: terminal day reward** | exact: the reward is $G_i^\pi-G_i^{B1}$ paid at the end; cross-block discounting is irrelevant (the terminal reward, if discounted, is multiplied by a constant $\gamma^{T-1}$ and the ordering is unchanged) | one full-day settlement per episode; no per-block credit |
| **$\gamma=1$ with A** | removes the discount mismatch only; A's local definition, standardization and terminal gap remain | Q magnitudes grow; keep a **fixed** reward scale, not the running std |
| **$\gamma=1$ with B** | exact on both counts; simplest statement in the paper | same as B |
| **keep $\gamma=0.99$ with A** | none; usable only if the paper never claims the reward is the day objective | — |

If a discount is kept for variance, the honest statement is: "the critic
estimates a discounted, standardized local differential; the reported benefit
is a separate undiscounted day settlement." That is a disclosure, not an
alignment.

## Recommendation

Do not change $\gamma$ in isolation. If the research objective is
$\Delta G_i$, pair **B** with an undiscounted (or terminally discounted)
return and a **fixed** reward scale; if the block signal is kept for credit,
state it as a shaping device and evaluate on the settlement. Whatever is
chosen, the report must say which of the four quantities the number belongs to
(`daily_objective_definition.md`).
