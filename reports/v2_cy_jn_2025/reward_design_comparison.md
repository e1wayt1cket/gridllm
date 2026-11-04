# Reward design comparison (V2-CY-JN-2025)

Four designs, one of which is implemented. Nothing here changes the code.

Notation: $s_t$ is the joint state at block $t$ (all SOCs, committed price
history, block index), $a_t$ the joint quote action, $a^{B1}$ the truthful
quote action, $T=24$ blocks, $\gamma=0.99$ per block.

## A. Current block-local differential (implemented)

$$
r_{i,t}= \underbrace{P_i(s_t,a_t)}_{\text{settled block profit}}
-\underbrace{P_i^{B1}(s_t,a_t^{B1})}_{\text{same window, all RL units truthful}}
$$

with one counterfactual clear per block, the policy's current SOC inherited by
both arms, opponents' actions unchanged, and only the committed four periods
differenced.

- **What it measures:** a local, state-matched, joint-truthful advantage of the
  committed action, at that state.
- **What it is not:** a unilateral marginal contribution (all four units move
  together), and not the day benefit $\Delta G_i$.
- **Target alignment:** low. The counterfactual starts from a state the policy
  produced; the policy is not charged for the SOC it spent, and the day
  terminal condition is absent. The sixth-phase identity
  `day = local + mechanism + state_path` shows the state-path term dominates,
  and the local/day correlation is −0.32.

## B. Whole-day differential return

$$
R_{i,\text{day}}(d)=G_i^\pi(d)-G_i^{B1}(d)=\Delta G_i(d)
$$

paid once at the end of the episode, both arms settled from $soc_0$ with the
terminal SOC pinned.

- **Alignment:** exact — it is the paper's quantity.
- **Credit assignment:** none inside the day. A single number must be assigned
  to 96 decisions and 4 actors. The critic can bootstrap, but the reward itself
  gives no per-block signal; on 24-block episodes with sparse profitable days
  this is the hardest part.
- **Critic:** the reward scale varies by day (0 to ~3400 CNY), so a fixed or
  running standardization is needed; the signal is flat on no-spread days.
- **Implementation:** one full-day settlement per episode plus the B1 value.
  B1 can be precomputed per day and reused, so the marginal cost is one
  whole-day clear per episode — cheaper than A's 24 counterfactual clears.
- **Heterogeneous days:** each episode's reward is in that day's own money;
  comparing across days needs the same normalization as the evaluation.
- **Discounting:** if the reward is paid at the terminal step, γ-discounting
  of earlier blocks is irrelevant; the return is just the terminal reward
  (multiplied by $\gamma^{T-1}$, a constant across actions, so the ordering
  is unchanged). That removes the γ mismatch of Section 4.

## C. Potential-based shaping

Standard form, added to the reward:

$$
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t)
$$

Two distinct variants must not be conflated.

### C1. Potential on top of A

Shaping A preserves **A's** optimal policy, not the day objective's. If A is
misaligned, C1 makes A easier to learn, not more correct. Claims that "SOC
potential fixes the day mismatch" are false under C1.

### C2. Potential on top of B

If the target is $\Delta G_i$, the base reward must be B. With B paid at the
terminal step, adding a potential redistributes the same terminal value across
blocks:

$$
\sum_{t=0}^{T-1} F(s_t,s_{t+1}) = \gamma\Phi(s_T)-\Phi(s_0)
$$

so the episode's shaped return is the B return plus a term that is constant
across policies only when $s_0$ is fixed. **Optimal-policy invariance**
(Ng, Harada & Russell, 1999) holds for either base as long as:

1. $\Phi$ is a function of state only — not of the opponent's actions, the
   action, or time-varying external noise; otherwise invariance fails;
2. the discount in the shaping equals the MDP's $\gamma$ (0.99 per block
   here);
3. the terminal state is handled: set $\Phi(s_T)=0$ at episode end, or add
   the terminal value explicitly; otherwise the telescoping leaves
   $-\Phi(s_0)+\gamma\Phi(s_T)$, which is constant across policies only if
   $s_T$ is absorbing/terminal-fixed;
4. $\Phi$ is bounded (e.g., the value of stored energy at a bounded price) so
   the shaped return does not diverge.

With those conditions the optimal policy is unchanged and the shaping changes
only variance and credit horizon. Without them the shaping can create reward
hacking (e.g., an unbounded potential that pays for hoarding).

**Is an SOC-only potential enough?** It supplies the missing future value of
stored energy (the term A omits), but it cannot price the market interaction,
the terminal condition, or the day's price path. It is a useful **credit
device**, not a substitute for B. On its own it does not turn A into the day
objective.

## D. Long-horizon counterfactual Q / advantage

$$
A_i(s,a)=Q_i(s,a)-Q_i(s,a^{B1})
$$

with $Q_i$ the centralized critic's estimate of the discounted joint return,
and $a^{B1}$ the joint truthful action at the same state.

- **Definition issue:** the counterfactual changes **all** RL units' actions,
  so the difference is a joint advantage, not a per-agent unilateral one. The
  per-agent payoff difference must still be read from settlement, not from Q.
- **Bias:** function-approximation error enters the difference directly; with
  a moving target the difference can be self-consistent yet wrong. The
  sixth-phase gradient check could not inspect the trained critic because it
  was not saved; an untrained critic's Q is not an economic value (it was used
  only as a gradient-scale reference).
- **Stability:** a centralized critic conditioned on all actions is the
  standard CTDE choice, but the difference of two Q-values doubles the
  variance and needs target networks and a freeze/validation schedule.
- **Counterfactual rollout:** not required if Q is trusted, but validating Q
  on a held-out day does require a truthful settlement, which is B1.
- **Verdict:** feasible only after critics are checkpointed and shown to
  predict realized day payoffs on validation days; not now.

## Comparison

| design | matches $\Delta G_i$ | extra counterfactual | credit assignment | critic risk | implemented |
|---|---|---|---|---|---|
| A block-local | no | 1 clear/block (paid) | per block | low | **yes** |
| B day differential | **yes** | 1 settlement/episode | terminal only | scale varies by day | no |
| C1 A + potential | no (A's optimum) | none | per block + potential | potential must be state-only | no |
| C2 B + potential | yes (invariant) | 1 settlement/episode | terminal redistributed | same as B, plus potential design | no |
| D Q advantage | only as good as Q | none (rollout to validate) | bootstrapped | high; needs saved critics | no |

## What the evidence supports

The pilot's negative result is consistent with A's misalignment, not with a
bug. If the research question is $\Delta G_i$, the training base should be
**B**, with **C2** as an optional variance-reduction device under the four
conditions above. A may be kept as a diagnostic signal but must never be
reported as the day benefit. D is a later option, gated on checkpointed
critics and a validation of Q against realized settlements.
