# C2 potential shaping — implementation audit (V2-CY-JN-2025)

Module: `src/c2_reward.py`; identities: `reward_b_c2_identities.json`; tests:
`tests/test_c2_reward.py`. No training.

## Definition

$$
r_{i,t}=r^{B}_{i,t}+\gamma\Phi_i(s_{t+1},t+1)-\Phi_i(s_t,t),
\qquad \Phi_i(soc,t)=\frac{T-t}{T}\,\kappa\,e_{\max}\,(soc_i-soc_0),
$$

with \(T=24\), \(\gamma=0.99\), \(e_{\max}=1.6\), \(soc_0=0.5\), and
\(\kappa=288.19757259048833\) — the project's frozen reference price, declared
as a module constant (not fitted per run). The base \(r^B\) is the B reward
from `daily_reward_b.py`; C2 does not import or mix in the old block-local
reward A.

### Properties

- **Terminal zero:** \(\Phi(soc,24)=0\) for any legal SOC, by the factor
  \((24-t)/24\). No dependence on whether the terminal SOC is pinned.
- **Bounded:** \(| \Phi | \le \kappa e_{\max}\max(soc_0-soc_{lo},soc_{hi}-soc_0)
  = 184.446\) CNY. A sampled path's maximum magnitude was 40.24.
- **Information:** own SOC and the block index only. No actual future price,
  no opponent action, no counterfactual result. A price term is deliberately
  absent: the timing audit cannot prove the same-day curve is available at bid
  time.
- **Initial SOC:** a path whose \(soc(0)\) differs from the declared \(soc_0\)
  is refused; a different initial state is a different episode and must be
  registered as such.
- **Termination:** a `done` flag before block 23 raises; C2 has no defined
  mid-episode terminal handling yet.

## Identity results (`reward_b_c2_identities.json`)

| check | result |
|---|---|
| telescope \(\sum\gamma^tF_t=\gamma^{24}\Phi(s_{24})-\Phi(s_0)\) | max error 3.6e-15 |
| B return vs C2 return differs by the boundary term | max error 2.5e-15 |
| terminal potential | exactly 0 |
| same boundary, different intermediate SOC → same total shaping | max error 8.9e-15 |
| double base + double potential → double return | max error 0.0 |
| potential within its declared bound | 40.24 ≤ 184.45 |
| mid-episode termination rejected | pass |
| \(\gamma^{23}\) terminal weight | 0.7936 |

## What this means for the reward

Under C2 the episode return is the B return plus a policy-independent boundary
term (zero here because the potential is relative to \(soc_0\)). The shaping
changes credit assignment, not the optimal policy. It is not a substitute for
B: it does not price the market interaction or the day's price path.

## What is not done

- No integration into a trainer; no training.
- The reward scale remains a declared constant; no running standardization.
- The terminal SOC is not pinned by the rolling interface; the potential's
  terminal zero makes C2 robust to that, but B itself still includes any
  ending-SOC difference between the two arms.
