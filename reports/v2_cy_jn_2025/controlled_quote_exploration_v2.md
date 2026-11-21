# D3 controlled continuous quote exploration (P0)

Implemented in `DAGateTrainer._policy_action` and stored in the replay as
`explored` / `exploration_source`. Data: `da_gate_exploration_log.csv`,
`da_gate_pilot_summary.json`.

## Frozen rule

- Training episodes only; deterministic validation and the target actor never
  explore.
- The gate is sampled by the existing Bernoulli/ST path. Exploration changes
  only an **open** leg's price, so a closed gate stays exactly 0.
- Discharge exploration probability: `epsilon(t) = 0.20 - 0.15 * t/159`, ending
  at 0.05; charge uses half that probability.
- If a leg is explored, the quote is drawn uniformly from `[0,h]` with
  probability 0.5 and from the high region `[0.8h,h]` with probability 0.5.
  No future LMP, price forecast or realized price enters the draw.
- Every executed quote is clamped by construction to `[0,h]`; the actor's
  pre-exploration quote is recovered as `gate * price_quotes` from the replay.
- The replay records `explored` (per leg) and `exploration_source`
  (`1 = uniform`, `2 = high region`). The observation, reward, SOC, clearing
  and terminal rule are untouched.

## What was logged

| quantity | value |
|---|---:|
| transitions in the final buffer | 3,816 |
| explored legs | 1,960 |
| explored discharge legs | 1,520 |
| explored charge legs | 440 |
| high-region draws (source 2) | 992 |
| uniform draws (source 1) | 968 |
| executed quote minimum / maximum | 0.0 / 299.996 |

Per unit (explored legs / discharge / high share among explored discharge):

| unit | explored | discharge | high ≥0.86h | actor quote mean at explored states | executed mean |
|---|---:|---:|---:|---:|---:|
| ESS9 | 371 | 340 | 42% | 118.0 | 209.2 |
| ESS14 | 539 | 378 | 44% | 136.2 | 210.2 |
| ESS17 | 517 | 415 | 44% | 26.4 | 217.3 |
| ESS27 | 533 | 387 | 45% | 98.4 | 214.5 |

The schedule in `episodes.csv` starts at temperature 4.0 / epsilon 0.20,
crosses 2.53 / 0.151 at episode 53 and 1.03 / 0.101 at episode 106, and ends at
1.0 / 0.051. The exploration count per episode falls from 11–13 legs to 6 at
the end, as designed.

## Coverage effect

The final replay has 705 ESS14 and 252 ESS27 discharge transitions at
≥0.86h, versus 2,307 / 1,276 in the old R10 buffer, which had 3.3× more
transitions. Per transition the shares are 18.5% / 6.6% (new) against
18.1% / 10.0% (old). The exploration therefore **did** put high quotes into
the replay at ESS14/27, but the shorter 159-episode run leaves the absolute
counts lower and the ESS27 share lower. The joint-cell count is 124 (old R10
125), so the raw joint coverage is similar; the new sampler is what reweights
the rare cells.

## Economics

The exploration is off-policy: the actor loss recomputes the policy action,
while the critic and reward use the executed explored quote. B/C2, gamma,
scale, terminal SOC and the clearing engine are unchanged. The one-day
offline check (forced epsilon=1) recorded 75 explored legs and all 24 clears
OPTIMAL, terminal SOC 0.5, zero voltage/line cells.

## Limitation

Exploration is necessary for the replay but not sufficient for the policy. The
pilot's selected round-1 actor is at 117/80 CNY/MWh for ESS14/27 discharge, but
the later rounds drift to 262/300 even with the exploration in the replay. The
exploration supplied examples; the critic and actor did not use them to stay
low. This is reported in the pilot results, not hidden as a success.

## Boundary

No future actual price/LMP is used to choose an exploration action. No
exploration rule enters the observation or the deterministic evaluation. No
H2 and no legacy test.
