# Next gate/critic design decision (Phase 15, no implementation)

Decision evidence:

- Long-horizon calibration: overall Spearman(Q, ΔG) ≈ 0; R2 +0.10, R10
  −0.12. In the 10 material states the critic is often anti-ranked (−0.6 to
  −0.94) while the actor's quote sits at the worst end.
- 72/88 state×ESS×leg groups have a <1 CNY spread: most quote changes do not
  matter economically, so a flat Q is appropriate there. The failure is in the
  material states.
- The replay covers high quotes (64/255 sampled transitions at the upper
  bound), and the closed-gate/executed-quote consistency is exact. The critic
  fits its training batch; it is not calibrated on the validation winter
  states. The state distribution, not the action coverage, is the gap.
- The winter loss is the high ESS14/ESS27 discharge quote plus cycle cost
  against the winter price; the battery idles while B1 captures the spread.
- The gate logit path is connected; ST is biased and its surrogate shrinks at
  saturation. A closed gate freezes its price branch. The deterministic mask
  collapses while Bernoulli sampling covers 29–52 modes.

| option | solves the diagnosed problem? | changes B objective? | gradient bias / variance | new checkpoint version? | complexity | required tests |
|---|---|---|---|---|---|---|
| A. ST + gate init/exploration | partly: keeps probabilities off saturation; does not fix the flat/anti-ranked critic or the high quote | no | ST biased / low | no | low | gate trajectory, sampled modes, same long-horizon calibration |
| B. Bernoulli/REINFORCE + baseline | improves the stochastic gate; does not fix the critic or quote branch | no (expected return; baseline does not move the optimum) | unbiased for the stochastic gate / high | yes: log-prob, baseline, gate sample | medium–high | baseline correctness, variance, long-horizon calibration |
| C. Gumbel-Softmax | exploration only; weak evidence for the diagnosed cause | approximately (relaxed) | biased, temperature-dependent / medium | temperature config | medium | temperature schedule, action legality, long-horizon calibration |
| **D. Critic value learning + continuous quote branch** | **yes: the critic is miscalibrated on the material states and the high-quote branch is the loss source** | no if the quote box/meaning and B/C2 are preserved | gate estimator unchanged; critic/value bias-variance changes | likely yes for the quote/critic architecture | medium | long-horizon Spearman, OOD state calibration, gate/quote joint, reward identity |
| E. Entropy/mode regularisation | reduces mask collapse, not the high quote/critic problem | **yes** (regularised objective) | changes the objective / n/a | reward config | low–medium | objective-change disclosure, no mode-count-only acceptance |

## Recommendation

**D first, with A as the cheap companion.** The confirmed problem is not that
the gate has too few modes; it is that the critic does not rank the material
long-horizon quote values and the continuous branch chooses a high, idle-
inducing quote. D targets exactly that: critic value learning/state coverage
plus the continuous quote branch. A can keep the gate probabilities away from
saturation at low cost and without changing the economic objective.

**Do not switch to B/C/E first.** B changes the estimator but leaves the
critic and the high quote; C is a biased relaxation with no clear advantage;
E changes the B objective and must not be introduced merely to raise the mode
count. Entropy is not a fix for an economically wrong quote.

## Smallest change most likely to work

The smallest candidate set is **D + A**:

1. D: review critic target/value learning on validation states and redesign the
   continuous quote branch so the legal quote region has a usable gradient and
   the high-quote/idle region is valued correctly.
2. A: gate initialization/exploration that keeps the ST probabilities off the
   saturated region; no objective change.

The gate estimator (B) is a follow-up only if A still leaves the mask collapsed
after D. H2 stays untouched until this design is frozen and the five-seed
protocol is ready.

## Not implemented

No actor, critic, gate, B/C2, reward scale, action bound, clearing engine,
checkpoint or data file is changed in this phase. No new test is added; the
diagnostic scripts are standalone and were run in separate processes.
