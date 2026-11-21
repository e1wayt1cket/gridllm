# Action parameterization design review (V2-CY-JN-2025)

Read-only. No mapping is changed here.

## The current mapping

$$
a = a_{\text{mid}} + \tanh(z)\,a_{\text{half}},
\qquad
a_{\text{mid}}=\frac{\text{low}+\text{high}}{2},\quad
a_{\text{half}}=\frac{\text{high}-\text{low}}{2}.
$$

For storage the boxes are charge $0\ldots150$ and discharge $0\ldots300$
CNY/MWh; the midpoints are 75 and 150, and the truthful point $(0,0)$ is the
**lower corner**, i.e. $\tanh(z)=-1$ on both legs.

## The five questions

**1. Can truthful be reached with finite pre-activations?**
No. $\tanh(z)=-1$ only as $z\to-\infty$. At any finite $z$, $a>0$, so
the current actor can express truthful only approximately. A policy that
should declare exactly zero always declares a small positive reservation
price, which is a different market action.

**2. How does $\partial a/\partial z$ behave near truthful?**
$\partial a/\partial z = a_{\text{half}}(1-\tanh^2 z)$. At the truthful edge
this is **0**; at the centre it is $a_{\text{half}}$. The actor's control over
its quote vanishes exactly at the target it is supposed to be able to declare.

**3. Does the deviation penalty vanish near the truthful boundary?**
Yes. The penalty uses $d=(1+\tanh z)/2$ and
$d(d^2)/dz = d(1-\tanh^2 z)$. At the edge this is **0**; it peaks at
$\tanh z=1/3$ ($16/27\cdot w$). Near truth, with $t=-1+\varepsilon$, the
gradient is $O(\varepsilon^2)$: a small deviation from truthful is almost
unpenalised. The penalty also vanishes at the **opposite** edge, where a
saturated refusal quote sits.

**4. Can tanh saturation trap the policy at a wrong edge?**
Saturation sets $\partial a/\partial z=0$ at both edges. The pilot's actors
are saturated on 0–100 % of sampled action dimensions per unit, and units
pinned at the charge box top (150) are **refusing to charge** (ESS14 in
FULL96/11: charge 149.9, discharge 0.6, settled ≈ 0). Leaving such a corner
requires a critic advantage large enough to move a zero-derivative output;
there is no smooth path out. This is consistent with, though not proof of, a
saturation trap.

**5. Is the boundary pattern correlated with payoff degradation?**
The unit pinned at the charge top is the idle one; the near-truthful unit
(ESS17 in FULL96/11, distance 0.01) earns normally. The sample is small and the
correlation is not causal: a policy can choose to refuse charging because its
block reward prefers it. The mapping is a contributing cause, not the only one.

## Design space

Any candidate must satisfy: truthful **exactly** expressible; every quote in
the legal box; no silent clipping (an explicit piecewise boundary is allowed,
a numerical clamp that hides a dead gradient is not); unchanged economic
meaning; testable mapping.

| design | exact truth | legal | gradient at truth | note |
|---|---|---|---|---|
| P0 current tanh / full box | no | yes | 0 | truth only in the limit |
| P1 truthful gate + continuous deviation | **yes** | yes | gate has a discrete step; continuous branch non-zero | straight-through or REINFORCE for the gate |
| P2 two-head act/abstain + price | **yes** | yes | same as P1 | clearest semantics |
| P3 dead-zone piecewise linear | yes on an interval | yes | 0 in the dead zone | kink at the zone edge; policy cannot leave truth by gradient alone |
| P4 one-sided softplus / sigmoid | no | yes | 0 at truth | same asymptotic problem |
| P5 negative-price semantics | yes | **no** | — | would require changing market rules |

The asymmetry is intrinsic: 0 is the market floor, so truthful is a boundary
of the legal set. No smooth reparameterization of the *price* can put it at the
centre **and** keep all quotes legal. A symmetric box around zero would permit
negative quotes, which the market forbids; changing the bounds to make the
design look nicer is not allowed.

### P1/P2: gated exact truth

$$
a = \begin{cases}
a^{\text{truth}}, & g=0,\\
a^{\text{truth}} + (a^{\text{high}}-a^{\text{truth}})\,\sigma(z), & g=1,
\end{cases}
\qquad g\in\{0,1\}.
$$

- Exact truth when the gate is closed; the open branch stays strictly inside
  the legal range and approaches the legal upper bound without exceeding it.
- The gate is a discrete decision. With deterministic TD3 a straight-through
  threshold is the minimal implementation; a REINFORCE head with a baseline is
  an alternative. Either way the gate's gradient is an estimator, and that must
  be stated and tested.
- Semantics: "declare no premium" becomes an explicit action rather than a
  limiting price. That is a semantic improvement, not a market change.

### P3: dead zone

$$
a = a^{\text{truth}} + \max(0,\min(1,z))\,(a^{\text{high}}-a^{\text{truth}}).
$$

Exact truth on $z\le 0$, legal, continuous. Its weakness is the mirror of
P0: the policy cannot leave truth by gradient alone, so exploration must carry
it. The kink is explicit and testable, not a hidden clamp.

## Recommendation (not implemented)

- **Do not keep P0 if "truthful" must be exactly expressible.** It fails at
  finite parameters and its gradient and penalty are both zero at the target.
- **P1/P2 are the only designs that express truth exactly while keeping a
  useful gradient**, at the cost of a discrete gate and its estimator. They
  preserve the economic meaning of the quote and never leave the box.
- **P3 is the simplest exact-truth fallback**, with a dead gradient in the
  truthful region; it needs explicit exploration.
- Any choice must keep the bounds 0/150 and 0/300, be unit-tested for
  `gate closed ⇒ quote == truthful exactly`, `all quotes in bounds`, and
  `no numeric clamp`, and be documented as a change to the action space, not
  just the network output.

The box bounds, the truthful declaration and the market quotes must not be
changed to make the optimization easier.
