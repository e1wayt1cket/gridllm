# Training readiness (V2-CY-JN-2025)

**No training was started.** This records what the data can support, so the
training budget can be decided from counts rather than from having 96 pairs.

This is a **cross-year controlled simulation**: a 2022 metered load day priced
by a 2025 Jiangnan day-ahead curve. It is not a same-period market settlement
and no result from it may be described as one.

## What is available

| item | value |
|---|---|
| Paired days | 96 |
| Distinct `load_date` | 96 |
| Distinct `price_date` | 96 (none reused) |
| Group split | summer 21, autumn 47, winter 23, national day 5 |
| Complete 2025 price days in source | 136 of 137 dates; the 2025-06-01 day is one period short |
| Price coverage | 2025-06-01 to 2025-12-31 only |
| Price region / unit | `south`, CNY/MWh **inferred**, not externally confirmed |
| `mv_value` | not used, not in any observation |
| Network support | `line50_q2.0` — idealised, not a costed engineering design |

The national-day group is the entire 2022 holiday (5 days) paired to 2025-10-03
to 10-07. It is reported on its own and never used to explain a small effect.

**Distance within the season.** 16 pairs (13 autumn, 3 summer) sit more than
14 days from the load day's own month-day, the furthest 67 days, because the
source has no complete day between 2025-11-16 and 11-30 (and none between
08-31 and 09-14). Those pairs put a September or early-July price beside a
late-November or mid-August load. They are legal under the stated rule (same
season, same weekday/weekend type) and are recorded with their gap, but they
must not be read as a late-November price applying to a late-November load.
The `JN-Base-NEAR` dataset excludes them; see `full96_near_sensitivity.md`.

## Two datasets

| dataset | days | summer | autumn | winter | national day |
|---|---:|---:|---:|---:|---:|
| `JN-Base-FULL96` | 96 | 21 | 47 | 23 | 5 |
| `JN-Base-NEAR` | 80 | 18 | 34 | 23 | 5 |

`JN-Base-NEAR` removes every pair more than 14 days from the load month-day.
The 14-day rule is a **distance sensitivity filter**, not a validated boundary.
NEAR is a subset of FULL96: every NEAR day is also a FULL96 day, so the two
datasets are not independent samples and cannot cross-validate each other. With
the committed seeds, NEAR's test split shares 0 days with FULL96's test split
and 11 with FULL96's training split, so a NEAR test score is in-sample relative
to FULL96's training set.

## Proposed split

Fixed seed `20261030`, stratified by season, the holiday block held out whole.
Written to `data/processed/v2_cy_jn_2025/split.json`.

| split | days | summer | autumn | winter | distinct price dates |
|---|---:|---:|---:|---:|---:|
| train | 64 | 15 | 33 | 16 | 64 |
| validation | 13 | 3 | 7 | 3 | 13 |
| test | 14 | 3 | 7 | 4 | 14 |
| holiday block (held out) | 5 | — | — | — | 5 (national day) |

No load date and no price date crosses a split boundary. The rounding gives
64/13/14 rather than 64/14/13 because each season is rounded on its own; the
difference is one day and is stated rather than adjusted.

`JN-Base-NEAR` has its own split, seed `20261031`, stratified the same way:
train 53, validation 11, test 11, holiday block 5. It is not a relabelling of
the FULL96 assignment; the two overlap matrix is in
`full96_near_sensitivity.md`.

The three splits do **not** have the same price level — daily-mean price runs
35–468 CNY/MWh (median 249) in train, 120–405 (median 291) in validation and
111–342 (median 198) in test. This is a property of the paired dates, not
something to be tuned away by re-pairing; it is a reason to report split-level
results separately.

## What the paired days look like under clearing

From `results.csv`, JN-Base on all 96 pairs. Every day cleared OPTIMAL with
zero non-nodal price cells, no voltage or line violation, exact terminal SOC
and no reverse flow at the PCC.

| statistic | value |
|---|---|
| Served energy, median | 108.8 MWh/day |
| Active losses (the model's own I^2R term), median | 2.75 MWh/day |
| Storage fleet payoff, median | 86.6 CNY/day |
| Storage fleet payoff, mean / max | 259.7 / 3433.3 CNY/day |
| Days with fleet payoff above 50 CNY | 51 of 96 |
| Days where the fleet does nothing | 45 of 96 (daily price spread below cycle cost) |

The cycle-cost probe on the 21 already-frozen representative pairs moves the
median fleet payoff monotonically: 36.4 CNY/day at 80, 26.3 at 90 and 17.6 at
100. Those 21 days are a sensitivity probe, **not** a test set, and the
representative subset is not the same population as the 96.

## Benchmark evidence (no training)

B1 is JN-Base, verified rather than repeated: 96/96 days pass the equivalence
audit, including the empty-action → truthful quote mapping and the input
hashes. The cleared baselines on the common split are:

| arm | common test mean payoff (CNY/day) | median | days above 1 CNY |
|---|---:|---:|---:|
| B0 NoESS | 0.0 | 0.0 | 0 |
| B1 Truthful | 167.4 | 17.6 | 7 |
| B2A price threshold | 160.3 | 17.6 | 7 |
| B2B threshold + SOC band | 84.3 | 17.5 | 7 |
| B3 LP bound (4 units) | 177.9 | — | — |

Paired on the same 11 test days, B2A is **never above** B1 (0 positive, 9
zero, 2 negative) and B2B is never above (0 positive, 6 zero, 5 negative).
The rule policies are not a stronger baseline than truthful; B2B's SOC cap
costs most of the throughput. The truthful clearing is also close to the
price-taker LP bound on these days (167.4 against 177.9), so the arbitrage
headroom a learner could add is small and concentrated in a few days.

The 16 FULL96-only pairs have a higher price level but flatter curves: median
truthful payoff 4.4 CNY/day against 119.9 on the common days. They add
coverage, not arbitrage signal, and stay training-only.

## Pilot outcome (controlled, not formal training)

Two seeds, twenty episodes, two pools that differ only by the 16 FULL96-only
days. All four runs: 480 environment steps, 417 gradient updates, 0 clearing
fallbacks, 0 LMP fallbacks, 0 voltage or line violations, terminal SOC pinned
to 0 in every evaluation clear.

| arm | common test mean (CNY/day) | vs B1 | days above B1 (of 11) |
|---|---:|---:|---:|
| B1 Truthful | 167.4 | — | — |
| B2A threshold | 160.3 | −7.0 | — |
| MATD3-NEAR (2 seeds) | 74.8 | −92.6 | 0 |
| MATD3-FULL96 (2 seeds) | 53.9 | −113.4 | 0 |

- **FULL96 − NEAR** on the common test: mean −20.9 CNY/day, 0 positive days,
  7 negative, 4 zero. With this budget the extra 16 days **did not help**.
- The policies quote well above truthful (charge ~48–127, discharge ~20–82),
  which makes charging dearer and discharge penalised in the clearing; they
  cycle less and settle less than the truthful arm.
- The training differential reward is positive for the NEAR seeds while the
  day-level payoff is below B1. That is not a contradiction: reward is a local
  counterfactual against other agents' contemporaneous bids, not the B1
  benchmark. A reward is not a benefit.
- Holiday block: every run is within 0.04 CNY/day of B1 — on those days there
  is no spread to capture, so policies and truthful coincide.

## Diagnosis of the pilot (read-only)

- **Evaluation files:** `eval.csv` is 64 wide records, one per
  `(dataset, seed, day)` — 4 runs × (11 test + 5 holiday), units as columns.
  No duplicate or missing keys. Both "0/11" conventions (per run and
  seed-averaged) give zero.
- **Exposure:** every run used 20 distinct days once; the FULL96 runs visited
  only 3 and 5 of the 16 extra days and therefore saw 17 and 15 common days
  against NEAR's 20.
- **Reward horizon:** the exact identity
  `day_diff = local + mechanism + state_path` closes to 1e-13, and the summed
  local reward correlates **−0.32** with the day-level difference. The
  state-path term dominates: the local counterfactual inherits the policy's
  SOC, so the policy is credited for a block without paying for the SOC it
  spent. Not a bug; a definition mismatch plus a real cross-period cost.
- **Definition:** the baseline is **joint truthful by all four RL agents** in
  the same window, not a unilateral counterfactual. It must be called a
  joint-truthful, state-matched, block-local differential reward.
- **Actions:** the Actor maps tanh = 0 to the box centre and truthful (0,0)
  to the lower corner. The trained policies move to **corners**, often the
  wrong one (charge ≈ 150 refuses to charge); ESS14 is idle for that reason.
- **Gradients:** the penalty's gradient at the saved actors is 0.11–0.25 and
  the per-action logit gradient 0.01–0.04, against 1.5–3.9 for a **fresh**
  critic (a scale reference only). The trained critics were not saved, so the
  trained critic–penalty ratio cannot be measured; checkpointing critics is a
  required fix for any future run.
- **FULL96 vs NEAR:** the seed spread (64 / 51 CNY/day) exceeds the dataset
  difference (−20.9), and the extra days were visited 3–5 times. The evidence
  cannot separate distribution, exposure and seed noise.

## Seventh-phase design review (no training)

The design documents fix what a next training must satisfy; none of it is
implemented.

- **Objective:** the paper's quantity is the whole-day
  `ΔG_i = G_i^π − G_i^B1`, per agent, with the two arms sharing
  the initial SOC and the exogenous day and diverging afterwards
  (`daily_objective_definition.md`).
- **Reward:** the current block-local differential is not that quantity. A
  whole-day differential (B) is the exact target; potential shaping (C) is
  policy-invariant only if the potential is state-only, the discount matches,
  the terminal potential is zero and the potential is bounded — and C on top of
  A preserves A's optimum, not the day objective (`reward_design_comparison.md`).
- **Discount:** the trainer uses γ=0.99 per one-hour block over 24
  blocks (γ^23 = 0.794, sum 21.43) and divides rewards by the stored
  rewards' std; training windows do not pin terminal SOC while the settlement
  does. The optimized object is a discounted, standardized, local return, not
  the day's money (`discounted_return_alignment.md`).
- **Action space:** truthful (0,0) is the lower box corner at
  tanh = −1, unreachable at finite parameters; both ∂a/∂z
  and the deviation-penalty gradient vanish there, and saturated refusal
  corners are observed. Exact truth needs a gated or dead-zone action
  parameterization, not a new box (`action_parameterization_design.md`).
- **Exposure:** fixed-budget and fixed-common-exposure protocols are defined
  separately, with 3 seeds; the pilot's comparison changed both
  (`dataset_exposure_experiment_design.md`).
- **Checkpoints:** actors alone are insufficient. Critic, targets, optimizers,
  RNG, counters, buffer (or its absence), reward definition and data hashes
  must be saved (`matd3_checkpoint_reproducibility_design.md`).
- **Paper wording:** the current reward is a state-matched, joint-truthful,
  block-local differential; it is not unilateral and not the day benefit
  (`paper_method_definition_review.md`).

## Eighth-phase design status (no training)

- **B (daily difference)** is specified in `daily_difference_reward_spec.md`:
  one whole-day settlement per arm, same initial SOC and exogenous day, no
  trajectory transplant, terminal pinned; B1 may be precomputed only with the
  hash checklist. Not implemented.
- **C2** is proven policy-invariant under four conditions — state-only
  potential, policy-independent boundary terms, matching γ, bounded potential —
  per agent and hence for best responses/equilibria; the recommended concrete
  form is a time-decaying SOC potential, which is zero at the terminal even
  without pinning. Running reward standardization breaks the fixed *scale*
  (not the ordering); a fixed scale is required
  (`c2_potential_shaping_proof.md`).
- **Truthful gate prototype** is implemented as a pure module
  (`src/action_gate.py`) with 11 tests: closed gate is exactly 0, open branch
  is legal and monotone, no clipping, extreme logits stable. Not wired into
  the action spec or any trainer. Open items: gate training mechanism,
  per-leg vs shared gate, mode-exploration checks
  (`truthful_gate_action_prototype.md`).
- **Offline gradient comparison** (`action_mapping_gradient_comparison.md`):
  the old tanh map reaches truthful only by saturating (0.0068 at z=-5, 3.1e-7
  at -10, exactly 0 at -50 with derivative 0); the gate-closed branch is
  exactly 0 at any logit; the open branch's derivative peaks at the midpoint
  (37.5 for charge) and vanishes at both ends. The discrete gate needs an
  estimator and none of them is an analytic derivative.
- **Discount audit** (`daily_return_discount_audit.md`): γ=0.99/block, 24
  blocks, weight sum 21.43, last weight 0.7936, effective 15-min discount
  0.9975; the C2 telescope closes to 2e-13, and the terminal-zero form to
  4e-14; training does not pin terminal SOC while the settlement does; the
  running reward standardization is not a fixed scale.
- **Checkpoint schema v2 + validator** (`matd3_checkpoint_schema_v2.md`): the
  three levels validate, and a missing critic, an unsaved buffer without a
  reason, or a malformed hash are refused. Sample and validation artifacts are
  in `data/processed/v2_cy_jn_2025/pilot/`.
- **Checkpoint interface** is specified (`matd3_checkpoint_schema_v2.md`):
  actors, targets, twin critics, optimizers, RNG, counters, buffer (or its
  declared absence), reward definition, data hashes, exposure; validation by
  identical action and update. Not implemented.

## Ninth-phase implementation status (no training)

- **B1 equivalence:** JN-Base is **not** the 24-block B1 (one-shot 96-period
  with terminal pin vs 24 rolling 16-period windows with no pin). A rolling B1
  is registered for all 96 days (`b1_daily_results.csv`,
  `b1_daily_manifest.json`): 96/96 OPTIMAL, 0 fallbacks, 0 LMP fallbacks;
  terminal SOC differs from 0.5 on 71 days, range 0.10–0.90. NEAR's 80 days map
  by `pair_id`, 80/80.
- **B implemented** (`daily_reward_b.py`): terminal-only payment, components
  kept, strict cache key, fixed scale; identity tests pass.
- **C2 implemented** (`c2_reward.py`): time-decaying SOC potential, exactly
  zero at t=24, telescope error 3.6e-15; mid-episode termination and initial
  SOC mismatch refused.
- **Gate adapter implemented** (`truthful_gate_actor_adapter.py`): 4 logits per
  ESS, hard forward gate, biased straight-through backward, Bernoulli
  exploration, deterministic target/evaluation, critic sees the executed quote.
  15/15 contract checks.
- **Checkpoint writer/loader** (`checkpoint_writer_v2.py`,
  `checkpoint_loader_v2.py`): schema-v2 save/load, RNG, optimizers, buffer;
  round trip 11/11 including a deleted-original-object test and an incomplete
  checkpoint refusal.
- **Holdout:** no unseen 2025 day remains; the 11-day common test and the
  holiday block are viewed. A new external price period is required for any
  generalization claim (`future_holdout_protocol.md`).

## Gates

| gate | state |
|---|---|
| 96 load days have complete 96x33 curves | met |
| Every pair has a distinct, complete 2025 price curve | met |
| Every pair clears under the support assumption | met |
| Price unit externally confirmed | **not met**, inference accepted for this version |
| Price and load from the same period | **not met, by design** — cross-year |
| More than one price year | **not met**, 2025 only, and only Jun–Dec |
| Every pair within 14 days of the load month-day | **not met**, 13 autumn pairs (source gap) |
| NEAR is independent data | **not met** — it is a subset of FULL96 |
| Holiday group large enough to interpret | **not met**, 5 days |
| New-energy feature available | **not met**, `mv_value` unusable |
| B1 equivalence to JN-Base | **met**, 96/96 days |
| Rule baselines established on the common split | **met** |
| A rule beats truthful on the common test | **not met** — never positive on 11 days |
| Arbitrage headroom above truthful | **small**, B3 bound within ~6% on test |
| Pilot machinery stable | **met**, 4/4 runs, 0 fallbacks and 0 violations |
| Pilot beats B1 | **not met**, 0 of 11 days for every run |
| Extra 16 days help | **not met**, FULL96−NEAR mean −20.9, 0 positive days |
| Trained critics available for inspection | **not met**, critics not checkpointed |
| Reward matches the reported day objective | **not met**, local vs day correlation −0.32; design B defined, not implemented |
| Action space can express truthful exactly | **prototype met**, not wired into the action spec |
| B reward implemented | **not met**, spec only |
| Checkpoint writer/loader | **met**, round trip 11/11 |
| B reward implemented | **met**, tests pass |
| C2 reward implemented | **met**, identities pass |
| Gate adapter implemented | **met**, 15/15 contract checks |
| Trainer integration of B/C2/gate/checkpoint | **not met** |
| New independent holdout | **not met**, no unseen 2025 day |
| Offline gradient/discount checks | **met**, scripts + tests |
| C2 telescope exact | **met**, max error 2.0e-13 |
| Test set still an unseen holdout | **no**, the 11-day benchmark has been viewed |

## Sample size

64 training days in FULL96 (53 in NEAR), one episode per day, 24 decision
blocks per episode. The
fleet's payoff is zero on 45 of the 96 days, so nearly half the training set
carries no arbitrage signal at the current cycle cost — that is a property of
the price data and not a defect of the pairing. Whether 64 days is enough for
four interacting actors is the question the budget decision has to answer; this
report does not answer it, and no run was started to find out.

## Recommendation

The pilot has now been run and **does not support scaling to formal
training on this design**. The machinery is verified and stable, but the
policies settle below B1 on every test day, and the extra 16 training days made
the paired result worse rather than better. Two things bound what any
continuation could show: the common test is 11 days, and the truthful baseline
already captures most of the price-taker bound, so quote re-optimisation has
little mean headroom. The diagnosis explains the gap, and the design review now fixes what a next
run must change: the reward base (B) and its discount/terminal treatment, an
action parameterization that expresses truthful exactly, and full
checkpoints. **Do not start another run until those three are implemented and
unit-tested**, and do not use the viewed 11-day benchmark to choose among
designs; it needs a fresh holdout for any new performance claim. Decide the episode count and seed count after reviewing
this page; the split and the arms are already fixed so the pilot cannot move
them.

## Tenth-to-thirteenth phase status (2026-11-02)

The three blockers named in the recommendation above are now implemented and
tested. The formal run is still blocked, for a different reason: there is no
new independent holdout.

- **Terminal rule:** every RL storage unit must finish at SOC 0.5. The
  constraint is absolute and applies only to the rolling windows whose own end
  is the true day end; earlier windows stay unpinned. The old rolling B1 is
  retained as `CASHFLOW_NO_TERMINAL`.
- **New B1:** the terminal-neutral rolling B1 is registered for all 96 days:
  96/96 feasible, 0 fallbacks, 0 LMP fallbacks, per-unit terminal error 0.0.
  NEAR maps 80/80 by `pair_id`. Artifacts and hashes are in
  `b1_terminal_neutral_manifest.json`; the accounting is in
  `terminal_soc_economic_accounting.md`.
- **Trainer:** `rl_b_c2_gate.MATD3BC2Gate` integrates B, C2, 8 truthful gates,
  twin centralized critics, deterministic targets, the gate-aware replay
  buffer, fixed scaling and the schema-v2 checkpoint. Old MATD3 is untouched.
- **End-to-end:** one fixed training day cleared 24/24 blocks with 0 fallbacks,
  0 voltage/line violations and exact terminal SOC; `delta = G_pi - G_B1`
  closed to 0.0; B was paid only at block 23; the C2 telescope closed to
  3.6e-15; 17 engineering updates ran with finite losses; the real
  save/destroy/restore/continue round trip passed every check.
- **Holdout:** the supplied `docs/data.csv` is the existing 2025 source
  (137 dates, all viewed), not a new period. G10 fails.
- **Decision:** NO-GO for formal multi-seed training. Development/exploratory
  runs on the NEAR split are allowed but cannot be reported as independent
  generalisation. See `formal_training_go_no_go.md` and
  `formal_training_protocol.md`.

### Updated gate states

| gate | state |
|---|---|
| Terminal-neutral B1, 96/96 feasible | **met** |
| B/C2/gate/critic/replay/checkpoint integrated | **met** |
| Complete 24-block episode | **met** |
| Full checkpoint restore and continue | **met** |
| New independent holdout | **not met** |
| Formal training GO | **no** |

## Holdout re-audit from existing data (2026-11-02)

No new external data and no training. The question was whether the existing
2022 loads and 2025 Jiangnan prices can supply a prospective holdout that is
independent of the new algorithm's training and tuning.

- **Exposure ledger:** 867 rows (730 load dates, 137 price dates) in
  `data/processed/v2_cy_jn_2025/holdout_exposure_ledger.csv`. Raw-data
  inspection is recorded separately from policy-result exposure: every date
  passed the quality audit, but only some entered training, validation,
  selection or performance evaluation.
- **Legacy seen test:** the original common test (11) and national-day block
  (5) remain `LEGACY_SEEN_TEST` (16 pairs). They are not renamed as a blind
  set.
- **H1 joint holdout: 0 days.** All 96 V2 load dates are used. The 27 unused
  V2 load candidates are 20 spring and 7 spring-festival; the 2025 price
  source has no spring date. The 28 other complete 17-user load days are
  autumn/winter and are not in the V2 33-bus load library. Even admitting both
  would give at most 18 pairs over two seasons, still below 30 days and three
  seasons.
- **H2 price-date holdout: 40 days, three seasons** (summer 22, autumn 10,
  winter 8) in `data/processed/v2_cy_jn_2025/holdout_pairs.csv`. Every price
  date is a complete 2025 date that never entered FULL96, NEAR, the pilot
  evaluation or the rule benchmark. The load side is historical: 15 pairs use
  a load visited in pilot training, 2 in pilot evaluation, 2 in the legacy
  test, 8 in historical selection, 17 in a reserved split; 38 distinct loads,
  2 reuses, 5 same-season type mismatches. H2 supports limited **price-date
  adaptation** only, not joint source-load generalisation.
- **New-algorithm exposure:** the engineering smoke used
  `2022-07-22|2025-07-15` for 17 gradient updates; its load is excluded from
  the H2 pool. No formal training has run.
- **Verdict:** the current data cannot support a 30-day, three-season joint
  holdout. The previous G10 conclusion (no independent joint holdout) stands;
  H2 is a disclosed, limited fallback. See
  `holdout_exposure_audit.md` and `holdout_manifest.json`.

## Fourteenth phase: gate-collapse diagnosis (2026-11-02)

Read-only; no training, no H2, no legacy test, no checkpoint overwrite.

- **Trajectory:** deterministic joint modes are 1–2 across the ten rounds;
  Bernoulli draws from the same actors cover 29–52 modes. The deterministic
  mask is already collapsed at round 1. Several gates are extreme from the
  first checkpoint, but not all probabilities are saturated.
- **ST:** the gate-logit path is connected; gate gradients are nonzero and
  Adam gate updates are comparable to the price branch. The surrogate
  `p(1-p)` is small at saturation, so the gate gradient is attenuated. ST is
  biased; no finite-difference check was used.
- **Critic:** at the audited state the min Q is nearly flat in the quote and
  weakly prefers high discharge quotes while the actual settlement is lower.
  ESS14 and ESS27 carry the winter-tail losses; ESS17 remains positive and its
  lower-bound quote is not the loss source.
- **C2:** the C2−B identity closes at ~3e-15; the boundary is zero; the day
  return equals B. The per-block shaping can dominate a low-B ESS (14.2 vs B23
  0.53), but it does not change the day objective. Q scale and twin gap are
  stable.
- **Validation:** the mean is driven by three winter days; the median is near
  zero. The full 11×10 matrix is `validation_degradation_matrix.csv`.
- **Next-development candidates:** D (continuous quote branch) plus A (gate
  initialization/exploration) is the recommended next step; B
  (Bernoulli/REINFORCE with baseline) is the gate-estimator alternative; C/E
  are not first. No option is implemented.
- **Boundary:** H2 unread, legacy test not run, no checkpoint/data overwrite,
  no training.

## Fifteenth phase: critic long-horizon calibration (2026-11-02)

Read-only; no training, no H2, no legacy test, no checkpoint/data write.

- **Calibration:** 499 counterfactual rollouts over 88 state×ESS×leg groups;
  one ESS/one quote leg changed at a fixed block, then the frozen policy
  continued to day end. Overall Spearman(Q, day ΔG) ≈ 0; R2 +0.10, R10 −0.12.
- **Where it fails:** 72/88 groups have a <1 CNY spread, so the market itself
  is flat there. In the 10 material (>10 CNY) states the critic is often
  anti-ranked (−0.6 to −0.94); ESS14/27 high discharge quotes are the worst
  actions while Q is flat/higher. The actor quote is often at the worst end.
- **Target/replay:** the Bellman target is correct; the closed-gate/executed
  quote consistency is exact; high quotes are in the replay (64/255 sampled
  transitions at the upper bound). The critic fits the training branch but is
  not calibrated on the validation winter states; the state distribution, not
  the action coverage, is the gap.
- **Winter blocks:** R10 losses concentrate at ESS14/27 high-discharge blocks
  10/11/14 (12-30), 3/17 (12-13) and 17 (12-31); the high quote plus cycle
  cost sits at/near the block LMP, the battery idles, and B1 captures 71–128
  CNY/block. ESS17 is not the loss source.
- **C2:** the C2−B identity closes; the day objective is unchanged; the
  per-block shaping is a credit redistribution. No C2/gamma/scale change is
  justified by this diagnosis.
- **Decision:** the next development should target the critic value learning/
  state calibration and the continuous quote branch (D), with gate
  initialization/exploration (A) as the cheap companion. B (REINFORCE+
  baseline) is a follow-up if A leaves the mask collapsed; C/E are not first.
  Entropy must not be added merely to raise the mode count.
- **Boundary:** H2 unread; LEGACY_SEEN_TEST not run; no training, no
  checkpoint/data overwrite, no clearing-engine change.

## Sixteenth phase: critic calibration and continuous quote branch (2026-11-02)

Read-only audits and designs; no training, no H2, no legacy test, no checkpoint
overwrite. Eight reports and their data are under `reports/v2_cy_jn_2025/` and
`data/processed/v2_cy_jn_2025/`; `phase16_artifact_hashes.json` binds them to
the run, data and checkpoint hashes.

- **Joint coverage:** the R10 buffer has 12,720 transitions and the R2 buffer
  2,544. The winter material states are at the 0.93–0.99 percentile of the
  buffer's nearest-neighbour distance at mid-day blocks, with only 9–75
  transitions matching block ±1, SOC ±0.10 and quote ±10% simultaneously.
  High quotes exist globally (2,307/1,276 transitions at ≥0.86h for ESS14/27
  discharge) but are thin at the material block+SOC. Coverage is not the sole
  cause: at 12-13 b2 ESS27 the state is close (0.61 percentile, 239 joint
  transitions) and the critic still anti-ranks the quote.
- **Return scale:** Q is in reward units, so `Q×1000` is the CNY-equivalent of
  the discounted B+C2 return. The comparison is same-scale but not calibrated:
  pooled Pearson −0.566, material −0.557, within-group mean Spearman −0.008,
  7 of 10 material groups anti-ranked, twin ordering disagreement in 9/10
  material groups. The sign reverses by region (winter −0.70, control +0.48).
- **15-minute winter audit:** at R10 12-30 b10 and 12-31 b17, q=0 or q=100
  reproduces the B1 block payoff and q=258–300 produces zero accepted discharge;
  the actor is at 258–299 and loses 98–128 CNY per unit-block. At R2 the worst
  blocks are not all quote losses: 12-13 b17 ESS14 is a closed gate + low-SOC
  path, and 12-30/12-31 b23 are low-price charging blocks. A 50.7 quote can
  also lose a whole block, so the harmful premium is not confined to 258–299.
- **Quote map:** temperature-scaled logistic does not change the high/mid
  gradient ratio; the linear-core map C gives 9.66–58.8 CNY/logit over
  0.86–0.997 against 0.90–36 for the logistic, reaches 299 in 2.57 logits
  against 5.70, needs no clipping, and keeps the gate-closed exact zero. It is
  recommended for a fresh prototype, not a hot swap.
- **Gate:** the default init is already p≈0.5 with p(1−p)≈0.249 and 256 sampled
  modes; the collapse happens in the first round. A gate-only temperature
  warmup 4→1 over two rounds raises the round-10 minimum p(1−p) from 0.003 to
  0.155 while evaluation stays τ=1. Entropy and REINFORCE are not used.
- **Decision:** proceed to a single-seed development pilot with D (quote map,
  stratified critic sampling, controlled exploration) + A (gate warmup), with
  R2/R10 preserved and the stop rule in
  `next_single_seed_pilot_protocol.md`. No budget expansion if the deterministic
  gate mode count stays ≤2 or the material-group Spearman stays negative.
- **Boundary:** H2 unread; LEGACY_SEEN_TEST not run; no training; no
  checkpoint/data overwrite; no clearing-engine change.

## Seventeenth phase: D+A implementation and single-seed pilot (2026-11-02)

Implemented `MATD3_DA_GATE_v2` with `storage_quote_v3` and
`matd3-checkpoint-v3`: D1 linear-core quote map, D2 joint-cell stratified
replay, D3 bounded training-only exploration, A1 gate-temperature warmup.
Offline/end-to-end validation: 64/64 checks (mapping, derivatives, gate
schedule, sampler probabilities/fields, 24/24 OPTIMAL, terminal SOC, B/C2
identities, v3 save/restore, v2 refusal). No R2/R10 actor or critic was loaded.

Pilot `DEV_DA_GATE_seed11_v2`: seed 11, 159 episodes = 3 rounds × 53 days,
validation 11 days, no test/H2. Selected checkpoint is round 1.

- **Economics:** selected validation mean −32.1 (median 0.0, 6/11 positive)
  against old R2 −30.7 and old R10 −77.7; final round 3 −79.3. Winter
  12-13/12-30/12-31: −53.1/−138.5/−108.3 for the selected model, versus R2
  −33.5/−28.0/−42.3 and R10 −224.5/−409.6/−234.8. The selected model is
  R2-level on the mean but worse on the winter days; it is not an improvement
  over R2.
- **Quotes:** selected ESS14/ESS27 discharge means 117.4/79.6; round 3
  261.9/300.0. The later rounds drift back to the high-quote regime despite
  D1/D3. The final replay has 705/252 high-quote (≥0.86h) ESS14/27
  transitions, versus R10 2,307/1,276 (18.5%/6.6% vs 18.1%/10.0% per
  transition).
- **Gate:** deterministic mode count is 1 in every round; the last training
  episode sampled 243 modes; permanent open/closed 7/1, 6/2, 5/3. A1 kept the
  stochastic gate alive but did not prevent the deterministic saturation.
- **Critic:** the selected model's material-state mean Spearman is +0.128
  (median +0.20; 5/15 material groups negative), better than the old R10
  (mean ≈0; 7/10 negative), and the twin gap is smaller. The Q separation is
  still small: at 12-30 b10 ESS14 the Q range is 0.047 against a full-day ΔG
  spread of 138.9 CNY. The critic ranks better but does not separate the
  values strongly enough to move the actor.
- **Winter acceptance:** the selected actor quotes 81–137 CNY/MWh at five of
  the six worst winter blocks; those quotes produce zero accepted discharge,
  while the same-state q=0 reproduces the B1 block. The harmful premium is not
  confined to 258–300. At 12-13 b3 the local block loss (71.6) is mostly
  recovered later (full-day spread 4.1); at 12-30 b10 ESS14 and 12-31 b17
  ESS14 the full-day spreads are 138.9 and 108.7, so those are material.
- **Stop rule:** condition 3 is met (quote-caused zero acceptance with a
  materially better same-state alternative). Conditions 1/2 are not fully met
  (deterministic modes 1 but sampled 243; material mean positive but 5/15
  negative). No physics/SOC/clearing failure and no NaN/checkpoint failure.
  **Do not expand the budget**; the next step is a diagnosis of the actor/gate
  decision to reach the truthful/near-zero quote.
- **Boundary:** H2 unread; LEGACY_SEEN_TEST not run; R2/R10 and the original
  10-round run preserved; no old checkpoint overwritten; pilot is exploratory
  and supports no generalisation claim.

## Eighteenth phase: economic ground truth before actor gradients (2026-11-02)

Read-only. Four checkpoints (D+A best step 1272, D+A final step 3816, old R2
step 2544, old R10 step 12720) were evaluated on eight winter states with
same-state full-day quote interventions; ESS14 and ESS27 were then enumerated
on a 10×10 fixed all-day quote grid on 2022-12-30 with the other three ESS
truthful. No training, no checkpoint change, no H2, no legacy test.

- **P0-1 critic vs matched rollout:** D+A best Q-top matches the return-top in
  3/8 states; mean within-state Spearman +0.288; pooled matched Q vs B+C2
  +0.609. Old R2 is 1/8, −0.064; old R10 0/8, −0.645; D+A final 0/8, −0.171.
  The D+A critic is directionally better but wrong in five states, three
  material (12-13 b3 ESS14, 12-30 b11 ESS14, 12-31 b17 ESS14).
- **Critic-right, actor-wrong:** 12-30 b10 ESS14 critic 0, actor 136.8, gain
  138.9 CNY; 12-13 b12 ESS27 critic 0, actor 81.2, gain 34.2. The actor does
  not follow the critic even where the critic is correct.
- **ESS14 10×10 grid:** B1 67.73; best (charge 50, discharge 33.3) 77.05,
  +9.32 over B1; refined (37.5, 46.7) 77.08; 6/100 cells within 1 CNY; 9 local
  maxima. High discharge ≥166.7 is a clear loss region. The D+A best actor
  mean is (20.8, 110.2), nearest-grid gap 10.31 CNY; its own-policy payoff is
  −95.4 (four-actor family). ESS27 is the control: best (0, 33.3) 112.31 vs
  B1 106.62; the actor mean (37.9, 77.4) is within 1.75 CNY and its
  own-policy payoff is 110.33. The defect is unit-specific.
- **Decision:** critic-plus-actor, not gate-first. The next work is critic
  calibration on the three material wrong states and actor gradient/credit
  assignment on the two critic-right/actor-wrong states; a gate ablation is a
  follow-up only. The all-day grid is not a Q target for single-block states.
- **Boundary:** no training; no checkpoint write; H2 unread; LEGACY_SEEN_TEST
  not run; old checkpoints preserved.
