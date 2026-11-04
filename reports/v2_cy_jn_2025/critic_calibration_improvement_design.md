# Critic calibration improvement design (P1)

Design only. No critic structure, trainer or reward is changed in this phase.

## Evidence this design must answer

| finding | value | source |
|---|---|---|
| pooled Q vs discounted training return | Pearson −0.566, Spearman −0.570 | return recheck |
| within-group Q vs full-day ΔG | mean Spearman −0.008, median +0.057, 47 positive / 38 negative of 88 | return recheck |
| material groups (ΔG spread > 10 CNY) | 10 groups; 7 anti-ranked, 1 flat, 2 correct | return recheck |
| twin-critic ordering disagreement | 85/88 groups, 9/10 material groups | return recheck |
| Q vs B component / C2 component | −0.48 / −0.53 Pearson | this phase, see below |
| winter state coverage | mid-day material states at 0.93–0.99 percentile of buffer NN; 9–75 joint (time+SOC+quote) transitions | coverage audit |
| high-quote coverage | present globally (2,307 / 1,276 transitions at ≥0.86h), thin at the material block+SOC | coverage audit |
| critic failure where coverage is good | 12-13 b2 ESS27: 0.61 percentile, 239 joint, 10 high-quote neighbours, still ρ = −0.77 | coverage audit |
| winter loss mechanism | R10 high quote gives zero accepted discharge; R2 worst blocks include gate/path causes | 15-minute audit |

The target is not only the shaping term: Q anti-correlates with both the
discounted B component (−0.48) and the discounted C2 component (−0.53), and
with their sum more strongly (−0.566). The critic is not merely fitting the
potential; it is mis-ranking the economic part as well.

The material set is also incomplete: the phase-15 calibration grid used blocks
2/11/20, while the winter-loss reconstruction puts the worst R10 blocks at
12-30 b10 and 12-31 b17. The coverage and 15-minute audits add those blocks;
the next calibration must use the winter-loss block set, not only the older
grid.

## Options

| option | what it changes | evidence it targets | data risk | recommendation |
|---|---|---|---|---|
| 1. More joint state-action coverage in the replay | adds transitions near dense high-price/SOC/block combinations | thin 9–75 joint counts at b11 | none if training days only | **yes, via option 2** |
| 2. Controlled quote exploration during training | samples discharge quotes across the legal box in high-price blocks | the actor never visits the middle of the box in those states; 150 can recover part of a block | none; it is the same environment | **yes, first** |
| 3. Independent calibration data from material-state rollouts | auxiliary critic supervision on model-generated counterfactuals | the critic is flat/anti-ranked in the few states where ΔG is material | model-generated labels, not independent market data; must be stored separately and disclosed | **optional, second** |
| 4. Critic loss weighting / sampling | prioritises large |B|, high price spread and large quote deviation | the critic fits low-value states; 72/88 groups are flat | changes the effective training distribution, not the objective | **yes, second** |
| 5. Richer state input | adds features to the observation/critic | the b11 states are OOD in the current 12-feature state | could leak future prices; the causal contract is strict | **no, not first** |

### 1. Joint coverage

The coverage audit shows that adding *marginal* high quotes does not help: the
R10 buffer already has 2,307 ESS14 and 1,276 ESS27 discharge transitions at
≥0.86h. The missing region is the joint (high-price block, mid/high SOC,
high-quote) cell. A blanket replay expansion would add mostly the same
marginal quotes in other states and would not target the b11 cell.

### 2. Controlled quote exploration

During training only, replace a fraction `epsilon` of the actor's executed
discharge quotes with a value drawn from a fixed grid across the legal box
(e.g. 0, 50, 100, 150, 200, 225, 258, 280, 300) **when the block's
day-ahead/realised price is in the upper quantile**. The gate sampling remains
the existing Bernoulli draw; the explored quote is still a legal quote and the
critic sees the executed value. This does not change B, C2, gamma, the reward
scale or the clearing. It is an exploration policy, not an objective.

Why it is first: the 15-minute audit shows that at the same state, q=0/100
recovers the B1 block while q=258–300 loses it, and that q=150 is partially
accepted at ESS27. The actor cannot learn that ordering if the middle of the
box is never visited in those blocks.

### 3. Auxiliary critic calibration data

If options 1–2 still leave the critic flat, generate counterfactual rollouts
**from training states only**, label them `model_generated_counterfactual`, and
store them in a separate dataset (not the real replay). The critic can receive
a small auxiliary regression loss on those labels. The following must be
disclosed with the data:

- the source state (training day, block, SOC, policy action);
- the quote grid and the other agents' fixed actions;
- the same frozen policy, gamma, B/C2 and terminal rule as the main run;
- the model error: the rollout is a clearing model, not an independent market
  observation, and it inherits the same clearing engine it is used to calibrate.

They must not be called real data, must not be mixed into the replay as if they
were environment transitions, and must not come from validation, H2 or the
legacy test.

### 4. Critic weighting / sampling

Two low-risk forms:

1. **Stratified sampling:** 50% uniform plus 50% stratified over the price
   quintile and quote-deviation quintile of the transition. It changes which
   transitions are seen, not the loss definition.
2. **Loss weighting:** weight the critic MSE by `1 + alpha * |B_block|` or by
   the within-block price spread. This emphasises the states where the
   economic difference is material. The weight must be computed from the
   training transition, never from a validation result.

The phase-15 finding that 72/88 groups have a <1 CNY spread means that a
uniform loss is dominated by states where the ordering does not matter; this
is the cheapest way to change that.

### 5. Richer state input

Do not add realised future prices or any feature that breaches the causal
observation contract. If a richer input is considered later, the only legal
candidates are past-only summaries (e.g. the last two blocks' LMP path) or a
state-only SOC history; the current 12-feature observation already includes
SOC, block, load/RE, price history and the per-agent forecast. The failure at
b2 with good coverage suggests the issue is not a missing feature but the
critic's value learning.

## Recommended minimal set

1. **D-a:** controlled quote exploration (option 2) on training days only,
   discharge leg, high-price blocks; log every explored quote and its realised
   ΔG.
2. **D-b:** stratified critic sampling (option 4) with price/quote strata;
   keep the loss function and the Bellman target unchanged.
3. **D-c (conditional):** only if D-a/D-b leave the material states flat,
   add the separate model-generated counterfactual dataset of option 3 with a
   small auxiliary weight and the disclosure above.
4. **Do not** change the critic architecture, the twin structure, gamma, the
   reward scale, the C2 potential or the B definition in this repair.

## Diagnostics and acceptance

- Long-horizon Spearman on **training-only held-out days**: the material
  groups must move from anti-ranked to non-negative; report the group count
  and the material-group mean, not only the pooled correlation.
- Twin ordering disagreement in the material groups must fall.
- The explored-quote distribution and the realised ΔG must be logged; if the
  high-quote cells are still empty in high-price blocks, the exploration did
  not reach the state.
- The winter 15-minute audit must be repeated on the pilot checkpoint; the
  same q=0/100 vs q=258–300 contrast is the economic check.
- No model selection on H2 or the legacy test; the phase-16 protocol uses
  validation only for model selection and the training-only internal split for
  critic diagnostics.

## Boundary

This is a design, not an implementation. No H2 read, no legacy test, no
validation replay addition, no future realised LMP feature, no claim that a
model-generated rollout is independent market data. The existing B/C2,
gamma, fixed scale, terminal SOC and clearing engine are untouched.
