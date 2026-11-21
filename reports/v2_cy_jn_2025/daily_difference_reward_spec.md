# Daily difference reward B — specification (V2-CY-JN-2025)

Definition only; not implemented. This is the proposed base reward for the
next MATD3 version.

## Definition

For a complete day scenario \(d\) and the four RL storage units:

$$
G_i^{\pi}(d)
$$

is the settlement payoff of unit \(i\) after the policy has been executed over
the whole day: the rolling decisions produce a 96-period quote schedule, and
one full-day clearing settles that schedule with terminal SOC pinned, at the
unit's own bus LMP through `participant_payoff`.

$$
G_i^{B1}(d)
$$

is the same quantity when **all four RL units declare truthfully for the whole
day** (B1 / JN-Base), cleared from the same initial SOC and the same day.

$$
R_{i,\text{day}}(d) = G_i^{\pi}(d) - G_i^{B1}(d).
$$

### What the two arms share and what may differ

| shared | may differ |
|---|---|
| day-start SOC (`soc0 = 0.5` for all four units) | the policy's SOC path after period 0 |
| load curves, PV/wind forecasts and output | the quotes and therefore the clearing |
| Jiangnan day-ahead curve | the nodal LMPs the units help set |
| network, line limits, voltage band, PCC rules | — |
| storage ratings, efficiencies, SOC band, `cycle_cost = 100` | — |
| terminal SOC rule (pinned by the full-day settlement) | — |

The policy's trajectory is **not** transplanted into B1, and B1's trajectory is
not transplanted into the policy. Each arm earns its own path; only the
boundary conditions are common. A reward that inherits the other arm's
intermediate SOC is a different quantity and is not B.

### Properties

- It is **not** a unilateral marginal contribution: the four joint quotes clear
  one market, and each unit's payoff reflects that joint clearing.
- Each unit's reward is computed separately from its own node LMP and ledger,
  but the four rewards are coupled through the clearing.
- It is defined **only at the end of a complete day**. Intermediate blocks carry
  no B reward (a shaping term may be added separately; see the C2 proof).
- The paper reports \(G_i^\pi\) and \(\Delta G_i\) separately. The reward is a
  training quantity and is never reported as the benefit.

## Precomputing and reusing B1

B1 per day is independent of any policy and can be computed once. Reuse is
allowed only if **every** condition below matches; otherwise the precomputed
value is refused, not approximated.

| item | check |
|---|---|
| day identity | `(load_date, price_date)` and the `pair_id` |
| initial SOC | all four units at the declared `soc0`, no carried-over state |
| load curve | SHA-256 of the 96×33 matrix matches the pairing table |
| price curve | SHA-256 of the 96-point south curve matches |
| network support | `line50_q2.0` settings (R/X scale, Q fraction, slack, band) |
| storage | capacity, power, efficiencies, SOC bounds, bus placement |
| market | `cycle_cost`, terminal SOC rule, PCC reverse limit, OPF mode |
| config hash | the full `MarketConfig` hash used for the clearing |
| terminal rule | `horizon_type="full_day"` with terminal SOC pinned |
| solver outcome | OPTIMAL, zero LMP fallbacks, no physical violation |

The reuse table should live in a JSON artifact keyed by `pair_id`, carrying the
value, the hashes and the config hash, so a day cannot be silently repledged
under a changed configuration. A mismatch is a hard error, not a re-solve that
overwrites the stored value.

## Implementation requirements (for the next version)

1. Train on complete-day episodes: the policy's 96-period schedule is settled
   with a **full-day** clear, terminal SOC pinned, exactly as B1.
2. Reward each unit with \(R_{i,\text{day}}\) at the terminal step; no
   intermediate base reward.
3. Do not discount the day money in a way the evaluation does not; if the
   terminal reward is discounted by \(\gamma^{23}\), state that the return is a
   fixed multiple of the day profit (see `c2_potential_shaping_proof.md`).
4. Do not standardize the return by a running reward scale unless the same
   scale is used in reporting; a fixed scale is preferred.
5. Save the per-day B1 values, their hashes and the config hash with the run.
6. Report \(G_i^\pi\), \(G_i^{B1}\) and \(\Delta G_i\) per day and per unit;
   never report the reward as the benefit.

## What this does not do

It does not define the credit assignment inside the day. B is sparse: one
number per unit per day. The optional shaping term and its conditions are in
`c2_potential_shaping_proof.md`; it is not part of B.
