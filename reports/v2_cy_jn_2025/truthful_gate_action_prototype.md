# Truthful-gate action prototype (V2-CY-JN-2025)

A pure-function prototype, `src/action_gate.py`, with 11 unit tests and an
acceptance artifact (`data/processed/v2_cy_jn_2025/pilot/action_gate_test_results.json`).
It is **not wired** into `rl_spec.py`, `rl_bidding.py` or `rl_td3.py`; no
training uses it.

## The mapping

For each quote leg $k\in\{\text{charge},\text{discharge}\}$:

$$
a_k=\begin{cases}
0, & g_k=0,\\
h_k\,\sigma(z_k), & g_k=1,
\end{cases}
\qquad g_k\in\{0,1\},\quad (h_c,h_d)=(150,300).
$$

The gate selects the **declaration mode** — "declare truthful" versus "declare
a premium" — not a physical charge/discharge switch. The continuous branch is
strictly positive and at most $h_k$: no negative price, no overflow of the
legal box, no clipping. Truth is exact when the gate is closed, at any finite
logit; it does not rely on saturation.

## §4.1 Acceptance results

| # | requirement | result |
|---:|---|---|
| 1 | gate closed → exactly 0 at any logit | pass (`closed_is_exact_zero`) |
| 2 | open gate within [0,150] and [0,300] | pass (`open_is_legal`) |
| 3 | no negative quote for any legal gate/logit | pass (`open_is_legal`) |
| 4 | open branch monotone in the logit | pass (`open_is_monotone`) |
| 5 | extreme finite logits: no NaN/Inf/illegal quote | pass (`extreme_logits_stable`) |
| 6 | no post-hoc clipping; invalid input refused | pass (`no_overshoot`, `bad_bound_rejected`) |
| 7 | charge and discharge gates independent | pass (`gates_are_independent`) |
| 8 | quote semantics unchanged (a price, not a physical switch) | by construction; bounds and truthful point unchanged |
| 9 | invalid shapes, bounds, non-finite inputs refused | pass (`bad_shape_rejected`, `nonfinite_rejected`) |
| 10 | deterministic | pass (`deterministic`) |
| 11 | old tanh vs new gate at truth | old at $z=-5$ is 0.00681 CNY/MWh; new gate is exactly 0 |

## §4.2 The discrete gate's training mechanism

The gate has no analytic derivative. The candidates differ in bias, variance
and complexity, and **a straight-through estimator is biased** — it is not an
unbiased deterministic policy gradient.

| method | bias | variance | exploration | critic input | replay buffer | target actor | complexity |
|---|---|---|---|---|---|---|---|
| **Straight-through** | **biased** (a surrogate for a hard threshold) | low | needs an injected stochastic gate during warmup or noise | the executed quote (gate already applied), or gate + logit | gate, logit, executed quote | threshold the target actor's gate logit | low; minimal change to TD3 |
| **Bernoulli / REINFORCE** | unbiased for a stochastic gate, with a correct baseline | high | natural: sample the gate | executed quote; for the gate gradient, log-prob and gate | gate, log-prob, logit, quote | sample or threshold | high: two policy heads, baseline, higher variance |
| **Gumbel-Softmax** | biased, temperature-dependent | medium | annealed sampling | executed hard action | gate/logit/sample | argmax or straight-through | medium; temperature schedule |

- All three can preserve **exact truth at execution** if the executed action is
  hard (`g=0 → 0`). The approximation lives in the gradient, not the action.
- The critic sees the **executed quote**, not the logit, so the value function
  is defined on the same action space as the market.
- The buffer must store whichever quantities the estimator needs; storing only
  the final quote loses the gate's log-prob and prevents REINFORCE.
- The multi-agent gate is a vector: 8 independent Bernoulli variables (4 units
  × 2 legs) or 4 shared ones. The joint mode space is 256 or 16. A factorised
  gate avoids enumerating them; a joint softmax over 2^8 modes is not
  practical.

**Recommendation (design, not implemented):** keep **per-leg independent
gates** and use a **straight-through** gate as the minimal path for a
deterministic TD3, while logging gate probabilities and the mode frequency. If
an unbiased gate gradient is required, move to a Bernoulli/REINFORCE gate with
a baseline and accept the variance. Do not describe straight-through as
unbiased.

### Gate structure

| structure | dims | joint modes | expressiveness | exploration |
|---|---:|---:|---|---|
| per leg (recommended) | 8 | 256 | truthful on one leg, premium on the other | harder; needs gate monitoring |
| shared per unit | 4 | 16 | both legs truthful or both premium | easier; excludes mixed strategies |

The pilot's observed strategies were **mixed per leg** (e.g. charge at 150
refusing to charge while discharge is near truthful). A shared gate would
exclude exactly those policies, so the per-leg structure is the one consistent
with the evidence.

## §4.3 Tests

`tests/test_action_gate_prototype.py` (11 tests): closed-gate exact zero at
several logits; open-gate legality; monotonicity; per-leg independence; extreme
logits; no overshoot; decode round-trip; invalid shapes/bounds/non-finite
refused; old-vs-new truth difference; the object refuses a non-truthful anchor.

The acceptance script reproduces the same properties in
`action_gate_test_results.json` (`all_pass: true`).

## What this prototype does not decide

- The final gate estimator (ST vs REINFORCE vs Gumbel) and its implementation
  in the trainer.
- How the gate is explored during warmup and how degenerate modes (always
  open/closed) are detected.
- The integration into the action spec and the centralized critic's action
  vector. Those are changes to `rl_spec`/`rl_bidding` and are out of scope
  here.
