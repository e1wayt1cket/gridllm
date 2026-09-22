# Calibrating the deviation penalty

The actor loss carries a regularizer that pulls each declared quote toward its
agent's truthful point. This records how its weight was chosen, and what was
deliberately *not* used to choose it.

Reproduce with:

```bash
PYTHONPATH=src python src/calibrate_dev_penalty.py --json outputs/dev_penalty_calibration.json
```

## The definition

Each action's deviation is measured **as a fraction of its own box width**:

```
d_i = |a_i - truthful_i| / (high_i - low_i)        a = mid + tanh(logit) * half
penalty = lambda * sum_i d_i^2                     meaned over the batch
```

`d` lies in `[0, 1]` for any truthful point inside its box. For the storage
fleet the truthful point is `(0, 0)` — the **corner** of `[0,150] x [0,300]`.

**Why normalize.** The discharge box is twice as wide as the charge box in
CNY/MWh. A raw deviation would call a 10 CNY/MWh move on the discharge leg the
same as a 10 CNY/MWh move on the charge leg, when the second is twice as far
from truthful in the only terms the mechanism sees. Normalized, the same
fractional position in either box is the same number, so one coefficient is
legitimate for both legs — which is why the retired per-leg weights
(`--bid-dev-penalty` / `--offer-dev-penalty`) are gone.

**Why this replaced the logit-quadratic form.** The old penalty was
`(logit - reference)^2`, whose gradient `2(logit - reference)` grows without
bound. Far from truthful it therefore outweighed the economic term and held the
policy against the corner. The normalized form's gradient is

```
d(d^2)/dx = (1 + t)(1 - t^2) / 2,   t = tanh(x)
```

which peaks at `t = 1/3` for **16/27 ≈ 0.593** and falls to zero at both box
edges. A pull that weakens exactly where the policy is furthest from truthful
cannot decide for it there.

This also retires the `atanh` machinery: with the penalty measured on the
action rather than the logit, there is no logit of the corner to evaluate, and
so no corner clamp.

## The grid

Fixed B3 day, SOCP, seed 123, a fixed observation batch and critic
initialization — the same inputs for every weight, so a row's difference is the
weight's alone. No learning takes place: the probe holds the critic frozen and
asks where the balance of the two gradients settles the actor.

Measured with the current implementation (per-agent action spaces, penalty on
the box-normalized action):

| lambda | penalty @ d=0.5 | \|econ\| | grad ratio | mean dev | corner | quote spread (CNY/MWh) |
|---|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.7908 | 0.0000 | 0.5608 | 12.5% | 87.6 |
| 0.010 | 0.005 | 0.7908 | 0.0090 | 0.5811 | 12.5% | 89.4 |
| 0.050 | 0.025 | 0.7908 | 0.0452 | 0.5378 | 0.0% | 83.3 |
| **0.100** | **0.050** | 0.7908 | **0.0903** | **0.5313** | 0.0% | **80.9** |
| 0.250 | 0.125 | 0.7908 | 0.2258 | 0.4236 | 0.0% | 58.2 |
| 0.500 | 0.250 | 0.7908 | 0.4515 | 0.2167 | 12.5% | 40.7 |
| 1.000 | 0.500 | 0.7908 | 0.9030 | 0.1070 | 0.0% | 14.0 |

`grad ratio` is exactly proportional to `lambda`: `grad_ratio / lambda =
0.9030373072` for every row (relative spread 1.6e-8, i.e. float rounding). That
is a property of the implementation rather than a fitted number —
`deviation_penalty` multiplies a quantity that does not depend on the weight,
and the probe seeds its actor and critic identically for every row, so the pull
is linear in the weight by construction. The test asserts the proportionality
and not the constant, because the constant belongs to the batch and the frozen
critic and moves when those move.

### Superseded measurement

An earlier version of this document recorded the grid below, measured before
the trainer moved from one environment-wide action spec to **per-agent action
spaces** and the penalty from logit space to box-normalized action space. It is
kept for the record; its selection (`lambda = 0.25`) is no longer supported.

| lambda | \|econ\| | grad ratio | mean dev | corner | quote spread |
|---|---|---|---|---|---|
| 0.000 | 0.907 | 0.000 | 0.597 | 0.0% | 87.9 |
| 0.100 | 0.907 | 0.078 | 0.586 | 0.0% | 82.8 |
| 0.250 | 0.907 | 0.196 | 0.496 | 0.0% | 65.4 |
| 1.000 | 0.907 | 0.784 | 0.083 | 0.0% | 10.9 |

The `0.784` coefficient that appeared here was a measurement of that
implementation, not a guarantee; under the current one it is `0.903`, and a
test that had pinned the number rather than the relationship reported a failure
that said nothing about the code.

## The selection rule

**Nothing here selects on profit.** The calibration never computes a profit, so
it cannot be selected on by accident. The rule is measured against the
*no-penalty* row rather than against constants chosen here — "how much does
turning the penalty on change the policy" rather than "how close to an invented
number is it":

- mean deviation must stay within **20%** of its `lambda = 0` value
- quote spread must stay within **30%** of its `lambda = 0` value

Under those tolerances the largest admissible weight is **`lambda = 0.10`**:
deviation 0.531 against a no-penalty 0.561 (floor 0.449), spread 80.9 CNY/MWh
against 87.6 (floor 61.3). Its pull is 9.0% of the economic term's, and it
costs 0.050 against a normalized economic term of 0.791.

**`lambda = 0.25` no longer passes.** Its mean deviation is 0.424, below the
0.449 floor: the same weight that once sat within tolerance now narrows the
policy more, because the policy it acts on is parameterized differently. This
is the reason the frozen value moved.

`lambda = 0.5` and `1.0` are rejected: they narrow the spread to 40.7 and 14.0
CNY/MWh, i.e. −54% and −84%. At `lambda = 1.0` the fleet's quotes have
effectively collapsed onto one another and there is no bidding left to measure.

## The frozen value

`rl_td3.CALIBRATED_DEV_PENALTY = 0.10`, used as the default by both
`train_rl.py` and `smoke_matd3.py`. A test re-runs the selection and fails if
the frozen constant and the rule disagree, so a changed actor, penalty or box
cannot leave the formal runs using a value whose supporting measurement no
longer exists.

**This is the weight Experiment Baseline v0.1 is frozen at.** The value is a
protocol constant, not a tunable: it was chosen once, against the grid above and
the selection rule rather than against any profit, and the formal runs declare
it rather than search it. A formal result is not a reason to revisit it. If a
future change to the actor, the penalty or the action box moves what the rule
selects, that is a new protocol version, and the test above is what makes the
disagreement visible instead of silent.

The frozen value is recorded in `reports/experiment_baseline_v0.1_freeze.md`
along with the rest of the protocol it belongs to.

## Two things the table is not

**The spread is not monotone across the whole grid, and is not claimed to be.**
Measured, `lambda = 0.01` leaves a slightly *wider* spread than no penalty at
all (89.4 against 87.6 CNY/MWh). At that weight the arbitrary frozen critic does
all the steering and the regularizer rides along. Asserting monotonicity across
the whole grid would hide precisely what the grid exists to show — which
weights actually steer. Monotonicity holds once the pull is a few percent of
the economic term, and is asserted there.

**The corner fraction is not monotone either.** Measured, it is 12.5% at
`lambda = 0` and `0.01`, 0.0% across the middle of the grid, and 12.5% again at
`0.5`. With four storage agents declaring two prices each, 12.5% is one action
of eight. A non-zero corner fraction is therefore not by itself evidence that
the penalty is pinning the fleet; the spread is the signal that shows a policy
being decided for rather than shaped, which is why the rule keys on the spread.

## The instrumentation run

```bash
PYTHONPATH=src python src/smoke_matd3.py --seeds 1,2 --episodes 60
```

An environment and policy instrument, not a training smoke: it walks the B3
facilities through the environment for a fixed number of episodes and looks
directly at what a reward curve cannot show — NaN, quotes outside their own
box, SOC bounds, clearing fallbacks, terminal SOC equality, degenerate or
corner-collapsed quotes. **It does not train.** It never calls
`trainer.remember()`, so the replay buffer stays empty, `MATD3.update()` returns
before its first gradient step, and the actor it samples from is the seeded
initialization rather than a learned policy. Reading its checks as evidence
about a trained policy would be wrong; `docs/2026-09-18-matd3-smoke.md` records
what they do and do not establish.

It runs at `CALIBRATED_DEV_PENALTY` unless overridden. See
`docs/2026-09-18-matd3-smoke.md` for the recorded run, and
`reports/protocol_smoke_v0.1_lambda010.md` for the protocol smoke at this
weight.
