# C2: potential-based shaping on the daily difference reward (V2-CY-JN-2025)

Mathematical review only. Nothing is implemented.

## Setup

- \(T=24\) decision blocks, \(\gamma=0.99\) per block, finite-horizon episode.
- Joint state \(s_t\): the four SOCs, the block index, and the episode's
  exogenous data (load/RE forecasts, the day's price input, the observed
  price history the environment exposes).
- Joint action \(a_t=(a_{1,t},\dots,a_{4,t})\): the four quote pairs.
- \(s_0\) is fixed by the episode: \(soc_i(0)=0.5\), same load/price/network.
- The episode ends with the **settlement** of the policy's 96-period schedule:
  one full-day clear with terminal SOC pinned, exactly as B1.

Base reward B (per agent, from `daily_difference_reward_spec.md`):

$$
r_{i,t}^{B}=0\ \ (t<T-1),\qquad r_{i,T-1}^{B}=R_{i,\text{day}}
=G_i^{\pi}-G_i^{B1}.
$$

C2 adds

$$
F_i(s_t,s_{t+1})=\gamma\Phi_i(s_{t+1})-\Phi_i(s_t),
\qquad r_{i,t}^{C2}=r_{i,t}^{B}+F_i(s_t,s_{t+1}).
$$

## 1. The telescope, exactly

$$
\sum_{t=0}^{23}\gamma^{t}\big[\gamma\Phi_i(s_{t+1})-\Phi_i(s_t)\big]
=
\sum_{t=1}^{24}\gamma^{t}\Phi_i(s_t)-\sum_{t=0}^{23}\gamma^{t}\Phi_i(s_t)
=
\gamma^{24}\Phi_i(s_{24})-\Phi_i(s_0).
$$

Two consequences:

1. The intermediate potentials cancel **for every realized path**, with no
   Markov assumption needed for the identity itself.
2. The only surviving terms are the two boundary states.

## 2. The base reward under the discount

Since B pays only at \(t=23\),

$$
\sum_{t=0}^{23}\gamma^{t}r_{i,t}^{B}=\gamma^{23}R_{i,\text{day}}
=0.7936\,R_{i,\text{day}}.
$$

For a fixed \(T\), \(\gamma^{23}\) is a constant positive factor: it scales every
policy's return identically and does not change the optimal policy. The paper
reports the undiscounted \(R_{i,\text{day}}\); the training return is a fixed
multiple of it, and that must be stated. It is **not** a different objective.

## 3. Conditions for policy invariance

With the telescope, the shaped return equals the base return plus

$$
\gamma^{24}\Phi_i(s_{24})-\Phi_i(s_0).
$$

This is independent of the joint policy if and only if:

1. **State function.** \(\Phi_i\) is a function of the state path (or the
   agent's observation history), never of the agent's action, the opponents'
   actions, or a future realized price/clearing result.
2. **Boundary terms policy-independent.** \(s_0\) is fixed by the episode, and
   either \(\Phi_i(s_{24})=0\) or the terminal state is policy-independent. With
   an SOC potential the clean choice is \(\Phi_i(s)=0\) at the settlement
   endpoint; the settlement pins \(soc_{24}=soc_0\), so a potential measured
   **relative to \(soc_0\)** satisfies both boundary terms exactly.
3. **Same discount.** The shaping uses the MDP's \(\gamma=0.99\)/block. A
   different \(\gamma\) in the shaping breaks the telescope.
4. **Bounded.** \(|\Phi_i|\le \Phi_{\max}\) so the shaped return and the critic
   target stay bounded. Boundedness is not needed for the identity, but it is
   needed for a stable critic and a finite return.

Under 1–4, every policy's return gains the same constant
\(-\Phi_i(s_0)\) (zero if \(\Phi_i(s_0)=0\)), so the **optimal policy is
unchanged**. Shaping then changes only the variance and the credit horizon.

## 4. Multi-agent analysis

Let the joint game have state \(S\), joint action \(A=\prod_i A_i\), transition
\(P\), per-agent rewards \(r_i\), and common discount \(\gamma\). The shaped
game has \(r_i^{C2}=r_i^{B}+F_i\) with \(F_i=\gamma\Phi_i(S')-\Phi_i(S)\).

For any fixed joint policy \(\sigma_{-i}\) of the other three agents, agent
\(i\)'s return under its own policy \(\sigma_i\) is

$$
J_i^{C2}(\sigma_i;\sigma_{-i})
=J_i^{B}(\sigma_i;\sigma_{-i})
+\gamma^{T}\Phi_i(S_T)-\Phi_i(S_0).
$$

If \(\Phi_i\) is state-only and the boundary terms are policy-independent, the
second term is a constant \(c_i\) for **every** \(\sigma_i\) (and every
\(\sigma_{-i}\)). Then

$$
\arg\max_{\sigma_i}J_i^{C2}
=\arg\max_{\sigma_i}J_i^{B},
$$

so best responses are preserved. Because this holds for every agent
simultaneously, the set of Nash equilibria (and of coarse-correlated
equilibria) of the shaped game equals that of the base game. The agents may use
different potentials \(\Phi_i\); the argument is per agent, and the common
discount is the same for all.

**No unilateral assumption is made.** The counterfactual inside B is the
joint-truthful clearing; C2 is a reward transformation, not a change of the
counterfactual.

### Partial observability

The identity and the invariance argument use the **realized state path and its
boundary states**, not the agent's information set. A reward assigned by the
environment on the true state can be shaped without the agent observing the
full state; a constant shift does not change the ordering of the agent's
policies. What partial observability does affect is the agent's ability to
**learn** and the critic's ability to represent \(Q\) and \(\Phi\) — not the
invariance itself.

Two caveats remain:

- If \(\Phi_i\) is computed from a belief that uses **future actual**
  information, or from the opponents' actions, condition 1 fails and the
  boundary terms can become policy-dependent.
- If the terminal does not pin the state and \(\Phi_i(s_T)\ne0\), the boundary
  term is policy-dependent and invariance fails. This is why the episode must
  end with the settlement that pins terminal SOC, and why a potential measured
  relative to \(soc_0\) is the safest choice.

## 5. Reward normalization

Three cases:

1. **No normalization.** The telescope and the fixed \(\gamma^{23}\) factor
   hold exactly.
2. **A fixed constant \(c\).** Dividing all rewards and potentials by \(c\)
   scales the return by \(1/c\): same optimal policy, same fixed relation.
3. **Running standardization by `buffer.reward_std()`.** The critic divides the
   batch by one scalar at each update, so at a fixed update the ordering is
   preserved and the potential/base ratio is unchanged. But the scalar changes
   as the buffer grows and depends on the day mix, so the learned Q is in units
   that drift, and the "fixed multiple of the day profit" is no longer a fixed
   number in learned units. With shaping, the buffer stores shaped rewards, so
   the scale is coupled to the potential.

**Recommendation for the next version:** store the raw components (day
differential and shaping separately) and normalize with a **fixed** scale (or
none) at target time. Keep the reported benefit undiscounted and unnormalized.

## 6. Potential candidates

| candidate | form | bounded | \(s_0\)/\(s_T\) | information needed | verdict |
|---|---|---|---|---|---|
| SOC only | \(\kappa\,e_{\max}(soc_i-soc_0)\) | yes, \(\kappa e_{\max}\cdot0.4\) | both 0 at \(soc_0\); needs terminal pinned | own SOC only | safe, credit only |
| SOC + time | \(\kappa\,e_{\max}(soc_i-soc_0)\,\frac{T-t}{T}\) | yes | 0 at \(t=T\) for any \(s_T\) | own SOC + block index (in the state) | safest terminal handling |
| SOC + time + price | \(\kappa_t\,e_{\max}(soc_i-soc_0)\,\frac{T-t}{T}\), \(\kappa_t\) a **published forecast** | yes, with a declared cap on \(\kappa_t\) | 0 at \(t=T\) | own SOC, time, and a price forecast available at the decision | allowed only with a documented, decision-time-available forecast |

- A price term must **not** use the same-day actual curve: the timing audit
  could not prove its availability at bid time. A lagged or published forecast
  is acceptable if its source and availability are recorded.
- \(\kappa\) should be a declared constant (e.g. the frozen reference price or
  the energy-value constant used in the project), not a fitted per-run number.
- The SOC potential supplies the future-value-of-energy credit that the local
  reward lacked. It **cannot** replace B: it does not price the market
  interaction, the day's price path, or the joint clearing. It is a credit
  device on top of B, never a substitute.

## 7. Verdict

C2 on B is policy-invariant under conditions 1–4. The recommended concrete
form is the **time-decaying SOC potential**, which is zero at the terminal
regardless of the terminal SOC and therefore does not depend on pinning during
training. It requires:

- the whole-day settlement episode (so B is the day differential),
- a fixed reward scale,
- a potential measured from the agent's own SOC and the block index,
- no use of the same-day actual price without a proven publication time.

Under those conditions the optimal policy of the shaped game equals that of B;
the paper still reports \(G_i^\pi\) and \(\Delta G_i\), not the shaped return.
