# Actor gradient vs deviation penalty (V2-CY-JN-2025)

## A gap first

**The trained critics were not saved.** `save_policy` writes actors only, so the critic term's gradient at the trained weights cannot be reconstructed without re-running training, which this phase forbids. The `critic gradient` column below is measured with a **freshly initialised** critic at the saved actor's observations; it is an initialisation-scale reference, not a statement about the trained critic. This is a reproducibility gap to close in any future run by checkpointing the critics.

## Penalty gradient (exact at the saved actors)

| unit | penalty grad norm | max per-action logit grad | saturated share | critic grad norm (fresh, reference) |
|---|---:|---:|---:|---:|
| ESS14 | 2.540e-01 | 3.501e-02 | 0.38 | 1.311e+00 |
| ESS17 | 1.932e-01 | 1.881e-02 | 0.62 | 4.245e+00 |
| ESS27 | 1.116e-01 | 1.060e-02 | 0.38 | 2.630e+00 |
| ESS9 | 1.876e-01 | 4.118e-02 | 0.00 | 2.062e+00 |

## What the numbers mean

For a storage unit the truthful point is the lower corner of each box, so with `d = (1 + tanh(x)) / 2` the penalty gradient with respect to one logit is at most `16/27 * weight = 0.0593` at weight 0.10. The actors sit near `tanh = 0`, where the derivative is `0.5 * weight = 0.05` per action, not zero. The penalty is therefore measurable — but small beside the critic term's gradient at initialisation (compare the last two numeric columns).

It is **not** evidence that the penalty is too weak on its own. The actors did travel to saturated corners, but often the wrong corner (a charge quote near 150 refuses to charge); the block reward does not price the SOC that refusal leaves unused. The frozen weight is not retuned here.

## Saturation

Saturation is partial: some action dimensions reach |tanh| > 0.99 (the table's share is over the sampled observations), so on those dimensions the penalty gradient does vanish. The action diagnostic reports the per-block share over twelve days; the actor is not uniformly parked at the centre, but neither has it moved the whole quote pair to the truthful corner.
