# Phase 20-C: twin gap, Bellman residual and target variance

Read-only. No training, no checkpoint write, no replay addition, no H2 and no
legacy test. Data: `phase20_bellman_diagnostics.{json,csv}`. Code:
`src/audit_phase20_bellman_diagnostics.py`.

## How the target was reconstructed

The target is rebuilt exactly as `MATD3BC2Gate.update` builds it:

```
target_actions = stack(_target_quotes(target_actor, next_obs))
target = reward + gamma * (1 - done) * min(target_Q1, target_Q2)
```

`_target_quotes` is the D+A v3 route with the configured target noise. The
replay TD residual is computed from the stored reward and next observation.
The Phase 19 matched full-day return is a separate Monte Carlo label and is
reported separately; it is not a next-state Q.

## Replay TD diagnostics

The batch is 2048 evenly spaced transitions (the best buffer has 1272, so all
of it is used). Values are in reward units; ×1000 is the CNY-equivalent.

| metric | D+A best | D+A final |
|---|---:|---:|
| buffer rows | 1,272 | 3,816 |
| median twin gap | **0.0238** (23.8 CNY) | **0.0063** (6.3 CNY) |
| median |Q−target| | 0.0206 (20.6) | 0.0094 (9.4) |
| median smoothing effect |target−target_no_noise| | 0.0042 (4.2) | 0.0006 (0.6) |
| median smoothing Spearman | 0.878 | 0.943 |
| median target std | 0.053 (53) | 0.026 (26) |
| median target std across batch | 0.059 | 0.032 |

The twin gap is worst in the high-quote cells: ESS14 charge high 0.0567
(best), ESS14 discharge high 0.0371 (best); ESS14 charge high 0.0154 (final).

## Matched-return comparison

| checkpoint | informative groups | median return spread | median |Q−return| | median twin gap |
|---|---:|---:|---:|---:|
| D+A best | 15 | 15.6 CNY | **254.8 CNY** | 26.1 CNY |
| D+A final | 15 | 18.0 CNY | 44.8 CNY | **8.0 CNY** |

The critic's absolute level is far from the matched return in both checkpoints.
The final checkpoint is better in level but still badly ranked (the oracle
report gives 1/15 top-quote agreement).

## What this answers

1. **Is the twin gap persistently larger than the action-return difference?**
   No. It is larger in the best checkpoint (26.1 against a 15.6 CNY median
   spread) but smaller in the final checkpoint (8.0 against 18.0). The
   min-target is a serious but not a uniformly dominant error.
2. **Does target action smoothing explain the twin divergence?** No. The
   smoothing changes the target by 4.2 CNY (best) and 0.6 CNY (final) at the
   median, an order of magnitude below the twin gap. Its rank correlation with
   the unsmoothed target is 0.88–0.94.
3. **Is the residual localized?** It is largest in the high-quote ESS14 cells
   and is not confined to one unit. Both checkpoints show the same pattern.
4. **Is this a target-arithmetic error?** No. The target closes on the code's
   own formula; the gap is between the two learned critics, not between the
   target and the stored reward.

## Decision consequence

A target-noise or twin-aggregation change is **not** supported by this
evidence: smoothing is too small to explain the gap, and the twin gap does not
persistently dominate the effect. Candidate 002 (target smoothing) is not
opened.

## Boundary

The replay residual is off-policy on the stored buffer; it is not an episode
return. The matched return comparison reuses the Phase 19 calibration rows and
does not add labels. No training, checkpoint write, H2 read or legacy test run
occurred.
