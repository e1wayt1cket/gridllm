# Gate initialization and exploration design (P1)

Read-only analysis and design. ST, the per-leg gate and the B/C2 objective are
kept. No entropy reward and no REINFORCE are introduced. Data:
`gate_initialization_exploration.csv`,
`gate_initialization_exploration.json`.

## Current initialization is not the cause of the R1 collapse

The gate actor's last layer uses the default PyTorch Linear initialization, so
the four policy logits start near zero. On 48 real replay states and 20
initialization seeds:

| quantity | value |
|---|---:|
| mean gate probability | 0.4988 |
| mean p(1−p) | 0.2488 |
| minimum p(1−p) | 0.2380 |
| deterministic joint modes across 48 states | 2.8 mean (1–5) |
| sampled joint modes, 100 draws per state | 256 mean |

The initialization already sits in the non-saturated region. The collapse is a
training phenomenon: the first checkpoint (53 episodes, 1,145 updates) already
has gates at 0.95–1.00, and the deterministic mask is 1–2 modes at every
round. Initialization alone cannot prevent that.

## What the trained probabilities do to the gate gradient

At the round-10 mean gate logits:

| gate temperature | min p(1−p) | mean p(1−p) | sampled joint modes in 100 draws |
|---:|---:|---:|---:|
| 1.0 (current) | 0.0031 | 0.0753 | 18 |
| 2.0 | 0.0504 | 0.1543 | 40 |
| 4.0 | 0.1550 | 0.2143 | 65 |

The ST factor is exactly p(1−p). At temperature 1 the most saturated gate has
a factor of 0.003, an 80× attenuation against the midpoint 0.25. A gate-only
temperature of 4 restores it to 0.155, a 50× improvement, without changing the
hard gate or the economic reward. Temperature 2 is a smaller, more
conservative step.

## Different Bernoulli initial probabilities

With one eight-gate vector and 100 draws:

| initial p | p(1−p) | sampled joint modes |
|---:|---:|---:|
| 0.1 | 0.090 | 26 |
| 0.2 | 0.160 | 48 |
| 0.3 | 0.210 | 68 |
| 0.5 | 0.250 | 78 |
| 0.7 | 0.210 | 65 |
| 0.8 | 0.160 | 46 |
| 0.9 | 0.090 | 26 |

0.5 is the maximum of p(1−p) and the mode count. Initializing away from 0.5
would lose exploration; it is the wrong lever. The useful lever is what
happens to the probability during training.

## Proposed A

1. **Keep the current initialization**, or set the gate-logit bias to exactly
   zero. Both give p = 0.5; the difference is immaterial.
2. **Gate-only temperature warmup.** During training use
   `tau_g(t) = 1 + (tau0 - 1) * max(0, 1 - t / N)` with `tau0 = 4` and
   `N` = the first two rounds (106 episodes, matching the R2 checkpoint).
   Evaluation and the deterministic targets always use `tau_g = 1`, so the
   economic policy and the final checkpoint semantics are unchanged.
3. **Keep Bernoulli sampling and the per-leg gates.** The draw is independent
   per leg; no joint gate is introduced.
4. **Add a uniform gate draw only if a permanent gate appears.** If a leg is
   open on fewer than 1% of decisions over a full epoch, replace its Bernoulli
   draw with a fair coin for that epoch. This is an exploration fallback, not a
   default. It must be logged with the gate it affected.
5. **Do not add an entropy bonus.** It changes the B objective, and the
   temperature already addresses the ST attenuation without touching the
   reward.

## Evaluation

Judge the design on the *conditional* use of the gate, not on uniform mode
coverage. Report per gate:

- p(1−p) min/mean and the fraction of decisions with p < 0.01 or p > 0.99;
- deterministic joint modes and sampled joint modes on the fixed reference
  states (the phase-14/15 set);
- per-leg open frequency and a permanent-gate flag;
- the deterministic evaluation with `tau_g = 1`, so the temperature cannot
  flatter the reported policy;
- the long-horizon critic calibration and the winter 15-minute acceptance
  audit.

The target is not 256 uniformly visited modes. It is a state-dependent choice
between modes that carries the long-run economic value. A policy that keeps a
leg closed in a genuinely low-price day is correct; a policy that closes it
because the ST factor vanished is not.

## Risks

- Temperature exploration changes the training distribution but not the
  B/C2 objective or the deterministic policy. It can still interact with the
  critic's flat value: more modes are useless if the critic cannot rank them.
  It is therefore paired with the D changes, not used alone.
- The temperature must decay to 1; otherwise the evaluated gate semantics
  drift from the trained one.
- The deterministic mode count is a threshold count and will stay low even at
  p = 0.5 (the init audit shows 2.8 modes). Do not accept or reject the design
  on that number alone.

## Answer to the phase question

**Can gate initialization avoid the R1 probability saturation?** No.
Initialization is already non-saturated; the first round is where the collapse
happens. A gate-only temperature warmup can keep p(1−p) and the ST path alive
through that round (min 0.003 → 0.155 at τ=4 on the round-10 probabilities),
while the deterministic evaluation remains τ=1. That is the recommended A.

## Boundary

No training, no actor/critic change, no entropy, no REINFORCE, no Gumbel, no
checkpoint write. H2 and the legacy test are untouched.
