# Gate and continuous-quote joint audit (P0, Phase 15)

Read-only. Uses the R2/R10 checkpoints, the fixed-state trajectory audit, the
ST-gradient audit, the validation gate diagnostics and the long-horizon
counterfactual rollouts. No training, test or H2.

## Gate and quote evolution (fixed states)

| round | ESS14.discharge gate prob / quote | ESS27.discharge | ESS17.discharge | ESS14.charge | ESS27.charge | ESS9.charge |
|---|---|---|---|---|---|---|
| R1 | 0.297 / 0.04 | 0.934 / 37.4 | 0.330 / 0.88 | 0.978 / 43.6 | 0.959 / 38.3 | 1.000 / 150.0 |
| R2 | 0.085 / 0.02 | 0.953 / 43.7 | 0.240 / 0.62 | 0.979 / 48.2 | 0.958 / 25.5 | 0.969 / 139.8 |
| R5 | 0.987 / 74.7 | 0.974 / 174.5 | 0.977 / 6.65 | 0.962 / 64.5 | 0.964 / 66.2 | 0.012 / 0.52 |
| R10 | 0.960 / 171.8 | 0.996 / 218.5 | 0.912 / 1.00 | 0.457 / 26.5 | 0.088 / 2.81 | 0.048 / 3.72 |

The first two entries are the gate probability and the open-branch quote mean.
The deterministic validation policy at R10 has ESS17.discharge open with a
1.19 mean quote (95% of open decisions at the lower boundary), ESS14.discharge
open with a 194.2 mean and a 0.49–299.9 range, ESS27.discharge open with a
222.7 mean, ESS14.charge open only 39% of the time, and ESS9/ESS17/ESS27
charge closed.

## The two branch failures

1. **Open gate, near-zero quote.** ESS17 discharge and ESS14 discharge can be
   gate-open while the open-branch quote is at the lower boundary. The gate is
   not the only declaration: the executed quote is `gate × high × sigmoid(price
   logit)`, so an open gate with a very negative price logit is an almost
   truthful declaration. The open gate and the zero quote are not contradictory;
   they say "I declare a price, and it is approximately zero."
2. **Closed gate, no price-branch gradient.** When the gate is closed the
   executed quote is exactly zero and does not depend on the price logit. On
   those samples the price-logit gradient is zero; only the gate logit can
   reopen the branch through the ST surrogate. The batch-mean price gradient is
   nonzero only because other samples are open. A closed gate therefore freezes
   its continuous branch, and the ST gate signal must carry the whole decision.

## High quote vs critic gradient

At the material long-horizon states the critic is nearly flat and sometimes
ranks the quote in the opposite direction to the actual settlement:

| checkpoint | day/block | ESS | actor quote | best quote | Q range | ΔG range | Spearman |
|---|---|---|---:|---:|---:|---:|---:|
| R10 | 2022-12-30 b11 | ESS14 discharge | 299.2 | 0.0 | 0.007 | 56.1 | −0.94 |
| R10 | 2022-12-30 b11 | ESS27 discharge | 282.0 | 0.0 | 0.007 | 62.7 | −0.77 |
| R10 | 2022-12-13 b11 | ESS27 charge | 0.0 | 112.5 | 0.0004 | 17.8 | −0.90 |
| R10 | 2022-12-13 b11 | ESS14 charge | 0.0 | 112.5 | 0.0025 | 36.0 | −0.60 |
| R2 | 2022-12-30 b11 | ESS27 discharge | 51.7 | 0.0 | 0.311 | 17.8 | −0.71 |

The actor's high quote is at the worst end of the actual long-horizon value,
while the critic Q is slightly higher there. The high quote is not a critic
gradient that the actor should follow; it is a region the critic fails to
penalize. The `dq/dquote` at these states is 1e−5–1e−3, against settlement
changes of tens of CNY; the critic's action gradient is too flat to correct it.

## Saturation order

- The gate probabilities are already extreme at R1 for ESS9/14/27 charge
  (0.97–1.00) and ESS27 discharge (0.93). The ESS14/17 discharge gates are
  moderate at R1 (0.30/0.33) but their open quotes are already near zero
  (0.04/0.88).
- By R5 the ESS14/17 discharge gates have saturated open (0.987/0.977) and the
  ESS14/27 discharge quotes have risen (74.7/174.5); by R10 they are 171.8/218.5.
- The pattern is therefore **co-evolution from R1**, not a clean gate-first or
  quote-first sequence: the gate mask is early, the quote branch moves inside
  the mask, and the ESS17 branch stays pinned near the lower bound while the
  ESS14/27 branch drifts high.

## ESS14 and ESS27 common pattern

Both start with an open charge gate and a modest charge quote, then close or
partially close the charge gate (ESS27 0.088, ESS14 0.457 at R10) while the
discharge gate remains open and the discharge quote rises to 171.8/218.5. The
winter loss blocks are exactly the high-discharge-quote blocks (see
`winter_loss_block_reconstruction.md`). The common mode is **charge
suppressed, discharge expensive**, not an ESS17 lower-bound failure.

## Required answer

Gate and quote are jointly degenerate: several gates saturate early; a closed
gate freezes its price branch; an open gate can still declare near-zero or high
quotes. The economic loss is concentrated in the high-discharge ESS14/ESS27
branch, where the critic is flat/anti-ranked. The gate estimator should not be
changed alone.
