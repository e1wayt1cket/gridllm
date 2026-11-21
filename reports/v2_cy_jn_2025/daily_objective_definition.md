# Daily objective definition (V2-CY-JN-2025)

## The economic target

For a complete day scenario $d$ and a policy $\pi$ over the four storage
units:

- $G_i^{\pi}(d)$ — the **full-day settlement payoff** of ESS $i$ under
  $\pi$, for the day's load, Jiangnan price, network, storage parameters and
  initial SOC $soc_0=0.5$, with terminal SOC neutralised by the settlement
  (both arms pinned to the same endpoint).
- $G_i^{B1}(d)$ — the same quantity when **all four RL storage units declare
  truthfully for the whole day** (B1 / JN-Base), from the same initial SOC and
  the same exogenous day.
- $\Delta G_i(d) = G_i^{\pi}(d) - G_i^{B1}(d)$, and the fleet aggregate
  $\Delta G(d) = \sum_i \Delta G_i(d)$.

The paper's question is the distribution of $\Delta G_i$ across days and
agents, not a scalar sum. The fleet aggregate is a reporting convenience; it
does not make the problem single-objective, because each unit's payoff is its
own.

The two arms **share the initial state and the exogenous conditions** and are
then free to diverge: the policy's trajectory and B1's trajectory separate in
SOC because they take different actions. That separation is the object of
interest, not a violation of the comparison. "Same initial conditions" means
same SOC at period 0, same load, price, network, constraints and cycle cost —
not same SOC at every block.

## Four quantities that must not be conflated

| quantity | what it is | where it appears |
|---|---|---|
| **day economic objective** | $G_i^\pi - G_i^{B1}$ over the whole day | the paper's result table |
| **block credit signal** | a local, state-matched block difference | reward shaping / diagnostics |
| **training reward** | what the critic regresses (currently the discounted, standardized block signal) | the training log |
| **reported settlement** | the full-day clear's `participant_payoff` | `eval.csv`, B1 table |

The pilot's reward is the second and third rows. It is **not** the first. A
positive block reward is a statement about one committed block against a
truthful arm placed in the *policy's* state; it is not a day-level advantage,
and the sixth-phase diagnostic found the two to be negatively correlated on
the sampled days (r = −0.32).

## What the daily objective requires of a training signal

For a training signal to optimize $\Delta G_i$ rather than approximate it,
it must, at minimum:

1. be evaluated from the day's initial state, not from a state the policy
   produced and would not have had to pay for;
2. include the terminal SOC condition the settlement imposes (pinned
   endpoints), or neutralize it in both arms;
3. not discount the day's money unless the evaluation does the same;
4. not re-standardize the money by a running reward scale unless the
   comparison does the same;
5. avoid counting an expectation over a critic as a realized payoff.

The current block-local reward fails (1) and (4), and the training arm also
fails (2) and (3) relative to the settlement. Sections 3–5 of the design
review quantify each.

## Multi-objective note

Maximizing each $G_i^\pi$ is a game, not a single scalar: a unit's action
moves the nodal prices at which the others settle. Any training objective that
collapses the four payoffs into one number is making an additional choice
(equal weights, a welfare proxy, or a per-agent decomposition). The project
reports per-agent $\Delta G_i$; it does not claim a scalarized optimum.
