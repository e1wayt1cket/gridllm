# Gate redesign options (Phase 14, no implementation)

This is a candidate comparison, not a design change. It uses only the
validation/development diagnostics above; the legacy test and H2 were not
read.

## Evidence the choice must answer

- The ST path is connected: gate-logit and gate-weight gradients are nonzero;
  the Adam gate update is comparable to the price branch. The surrogate
  `p(1-p)` is small at saturation, so the gate signal is attenuated.
- The deterministic mask collapses to 1–3 joint modes, while Bernoulli draws
  cover 29–52 modes. The hard threshold, not a completely frozen stochastic
  policy, is the main cause of the evaluated-mode collapse.
- The critic is nearly flat in the quote and weakly misranks high discharge
  quotes (ESS17/ESS14 at the audited state). The tail losses occur in ESS14 and
  ESS27, not in the ESS17 lower-bound branch.
- C2 changes credit assignment, not the day return; there is no C2 scale drift
  or twin-critic divergence. Validation degradation is driven by a few winter
  days, not by a uniform policy collapse.

## Candidate comparison

| option | gradient bias | gradient variance | mode-collapse risk | MATD3 critic compatibility | complexity | changes economic objective? | new data / checkpoint? |
|---|---|---|---|---|---|---|---|
| **A. ST + Bernoulli, init/exploration** | biased (ST) | low | **high**: hard threshold still collapses; probabilities can be kept off saturation | high; no interface change | low | no | no / no |
| **B. Bernoulli/REINFORCE with baseline** | unbiased for the stochastic gate with a correct baseline | high | medium; entropy/exploration can keep modes | high; needs log-prob, baseline and gate sample in replay/checkpoint | medium–high | no (expected return; a baseline does not move the optimum) | no / **replay + checkpoint fields** |
| **C. Gumbel-Softmax** | biased, temperature-dependent | medium | medium; temperature schedule controls exploration | high; action spec unchanged | medium | approximately, via the relaxation | no / temperature config |
| **D. Continuous quote-branch redesign** | unchanged gate estimator | n/a | n/a for the gate; directly addresses quote saturation | high; new quote parameterization | medium | no if the quote box and meaning are preserved | no / **new action spec/network output version** |
| **E. Gate mode/entropy regularization** | changes the objective by the regularizer | n/a | lower | high | low–medium | **yes** (regularized, not the pure B day objective) | no / reward config |

## Recommendation

**D first, with A as the cheap gate companion.** The main economic losses are
the ESS14/ESS27 high discharge quotes and the partial/closed charge gates, not
the ESS17 lower-bound quote; a redesigned continuous quote branch is the
highest-value fix. A gate initialization/exploration change is cheap and can
keep the probabilities off saturation while the branch is redesigned.

**B is the gate-estimator alternative** if the next development explicitly
targets the stochastic gate objective. It removes the ST bias, but it does not
by itself fix the flat critic or the high-quote drift, and the deterministic
evaluation still thresholds. **C** has no clear advantage over B at this
evidence level. **E** risks changing the paper's economic objective and should
not be the first move.

A prerequisite for any option is to review the critic's flat action-value
ranking on the discharge leg and the deterministic evaluation protocol; a
better gate estimator on top of a flat critic may still drift.

## Not implemented

No actor, critic, reward, gate estimator, action mapping, clearing engine,
checkpoint or data file is changed in this phase.
