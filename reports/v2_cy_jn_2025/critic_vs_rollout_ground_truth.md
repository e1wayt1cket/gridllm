# Critic vs matched rollout ground truth (P0-3)

Data: `daily_counterfactual_action_validation.csv/json`,
`single_ess_quote_surface_ESS14.csv/json`, `phase18_analysis.json`.

The only Q/return correlation reported here is the **matched** one: same
state, same joint action change, same frozen continuation policy, same gamma,
B/C2, reward scale and terminal rule. The all-day fixed-quote grid is a
different policy family and is **not** correlated with single-block Q.

## Pooled matched correlation

Across all audited candidate rows for each checkpoint:

| checkpoint | rows | Q vs B return | Q vs B+C2 return | Q vs full-day ΔG |
|---|---:|---:|---:|---:|
| old R2 | 42 | −0.396 | −0.408 | −0.420 |
| old R10 | 45 | −0.517 | −0.364 | −0.513 |
| **D+A best** | **47** | **+0.595** | **+0.609** | **+0.627** |
| D+A final | 48 | +0.242 | +0.508 | +0.320 |

The D+A best critic is the only one with a positive pooled matched
correlation. The old critics are negative; the final D+A checkpoint loses
part of the direction.

## Within-state ordering

| checkpoint | states | Q-top = return-top | mean Spearman(Q, ΔG) | mean best gain vs actor |
|---|---:|---:|---:|---:|
| old R2 | 8 | 1 | −0.064 | +3.5 CNY |
| old R10 | 8 | 0 | −0.645 | +87.1 CNY |
| **D+A best** | **8** | **3** | **+0.288** | **+38.5 CNY** |
| D+A final | 8 | 0 | −0.171 | +65.5 CNY |

The positive pooled correlation does not make the critic a reliable
within-state selector: it gets the top quote right in only 3/8 states.

## Where D+A best is wrong

| state | critic rec | best return | actor | actor ΔG | best ΔG | gain | note |
|---|---:|---:|---:|---:|---:|---:|---|
| 12-13 b3 ESS14 | 300 | 0 | 123.6 | −14.6 | −10.5 | +4.1 | small material |
| 12-30 b11 ESS14 | 300 | 0 | 136.2 | −163.1 | −142.0 | +21.1 | material |
| 12-31 b17 ESS14 | 150 | 0 | 135.9 | −116.1 | −7.4 | +108.7 | material |
| 12-30 b10 ESS27 | 5 | 83.9 | 83.9 | +3.7 | +3.7 | 0.0 | actor already best; flat |
| 12-30 b11 ESS27 | 0 | 0 | 84.3 | +3.7 | +4.4 | +0.7 | near-flat |
| 12-31 b17 ESS27 | 5 | 75 | 83.5 | −9.6 | −9.6 | 0.0 | flat |

The critic is correct at 12-13 b12 ESS27, 12-30 b10 ESS14 and 12-30 b11
ESS27. At 12-30 b10 ESS14 it recommends q=0, the actor is at 136.8, and the
matched gain is 138.9 CNY; at 12-13 b12 ESS27 it recommends q=0 and the gain
is 34.2 CNY. Those are the strongest "critic right, actor wrong" cases.

## Does the critic recommend a higher actual return?

**Sometimes, not reliably.** The D+A best critic's Q-top is the return-top in
3/8 states; the mean within-state Spearman is +0.288. It captures the largest
material gain correctly at 12-30 b10 ESS14, but misses 108.7 CNY at 12-31 b17
ESS14 and 21.1 CNY at 12-30 b11 ESS14. The old R10 critic is worse (0/8,
−0.645); the old R2 is near zero (1/8, −0.064).

## Single-ESS grid, not correlated with Q

The grid is a fixed all-day quote family with three truthful competitors. Its
value is a conditional optimum, not a rollout from the same state-action
pair, so no Q/return correlation is computed.

| quantity | ESS14 |
|---|---:|
| B1 truthful payoff | 67.73 |
| grid best (charge, discharge) | (50.0, 33.3) |
| best payoff / ΔG vs B1 | 77.05 / +9.32 |
| refined best | (37.5, 46.7), 77.08 |
| D+A best actor mean quotes | (20.8, 110.2) |
| nearest-grid payoff | 66.74 |
| nearest-grid gap to best | 10.31 |
| actor own-policy payoff | −95.41 |
| high discharge ≥166.7 | payoff ≤ 0; 19–24 zero blocks |

The surface is a broad low-discharge basin with several local plateaus, not a
single sharp peak. The actor's discharge mean is far above the grid optimum;
the charge mean is close. The actor's own-policy payoff is much lower because
all four actors are quoting, so the difference is not only the target's quote.

**ESS27 is the control.** Its B1 payoff is 106.62; the grid best is (0, 33.3)
at 112.31, and the D+A best actor's mean (37.9, 77.4) is within 1.75 CNY of
the best, with an own-policy payoff of 110.33. The same actor is near-optimal
on ESS27 and far from optimal on ESS14. The defect is unit-specific, not a
uniform actor/gate failure.

## Simplified problem vs four-agent failure

The single-ESS problem **does** have an interpretable optimum: low discharge
(≈33–47) and charge ≈37–50, with a +9.3 CNY conditional gain over truthful.
The four-agent actor is far from that discharge and its own-policy payoff is
negative. This points to a multi-agent/credit-assignment gap as well as the
critic's residual errors; it does **not** justify blaming the gate alone.

## Decision input

- The critic is directionally positive but wrong in 5/8 matched states, so it
  cannot be declared "basically correct".
- Where the critic is right, the actor still chooses a losing quote
  (12-30 b10 ESS14: critic 0, actor 136.8, gain 138.9).
- The single-ESS grid has a clear conditional optimum, so the actor's failure
  is not because the simplified problem is ill-posed.

The phase-18 branch is therefore **critic-plus-actor**: calibrate the critic
on the wrong material states, and audit the actor gradient / credit assignment
on the states where the critic is already correct but the actor does not move.
Do not run a gate-only diagnosis first.

## Boundary

No H2, no legacy test, no training, no checkpoint write. The grid is a
conditional family and the P0-1 rollouts are same-state counterfactuals, not
the four-agent dynamic equilibrium.
