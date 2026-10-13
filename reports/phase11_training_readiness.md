# Phase 11 — Training Readiness

What must be true before a formal MATD3 run is started, and what is already
true. **No formal training was started, no checkpoint was resumed, none was
deleted or reclassified.** This document plans; it does not run.

Read with `reports/final_research_summary.md` §2.1 (price statistics) and §7
(the checkpoint inventory).

---

## 1. Phase 10A review — the price statistics

**Verdict: the revision is consistent, and it changed no economics.**

The two averaging questions are now carried by separate fields on every price
curve, because one axis cannot hold them: a curve is routinely both spatially
averaged *and* a single day.

| sheet | `source_kind` | `spatial_statistic` | `temporal_statistic` | `window_days` | `externally_confirmed` |
|---|---|---|---|---|---|
| `261009` | `real_market_data` | `spatial_mean` | `single_day` | 1 | **false** (project ruling) |
| `7days` | `real_market_data` | `spatial_mean` | `window_mean` | 7 | true |
| `30days` | `real_market_data` | `spatial_mean` | `window_mean` | 30 | true |
| `halfyear` | `real_market_data` | `spatial_mean` | `window_mean` | 180 | true |

`temporal_statistic` is the **sole authority** for single-day identity. The
retired `is_measured` is a derived property meaning "the source is real
measured market data" and is **true for all four curves** — a caller still
reading it for single-day identity gets the opposite of the intent, so its
semantics are published beside its value through `DEPRECATED_FIELDS`, and a
test asserts the deprecated names are properties rather than stored fields and
so cannot drift from the authority beside them.

**Consistency was checked programmatically, not eyeballed:** for each curve,
the four facts agree across the `Curve` fields, the `provenance` dict and
`describe()`. Two stale statements were found by the audit and fixed — a
`DianyichaPriceProvider` docstring asserting `is_measured: false`, and a test
module docstring asserting "the four curves are means, not days".

**Nothing economic moved.** Phases 6, 7 and 9 were re-run and diffed against
the previously validated artefacts with metadata keys excluded and NaN-aware
comparison: **zero non-metadata differences** across all three. `required_margin`
is still `200.69565217391303`, `261009/fixed` fleet profit is still
`-8.19214836113872e-05`. No price, objective, `cycle_cost`, efficiency or
network parameter was touched.

**Still unverified about the price data** (unchanged by 10A): how each spatial
mean was taken — extent, weighting, number of points; and whether `261009` is
really one day. The workbook cannot settle the second, which is why the reading
is recorded as a ruling with `externally_confirmed: false`.

**Two statements must not harden into facts in any write-up.** Neither blocks
synthetic-scenario development, and both are easy to overstate:

1. **`261009`'s single-day identity is a project-adopted ruling, not a
   provider's proof.** It may be used as the single real-data case; it may not
   be described as an independently verified measured day.
2. **The three trailing curves' window start and end dates remain
   unconfirmed.** They stay `None`. Until a platform record supplies them,
   these curves may not be called strictly non-anticipative historical
   information, and their aggregation window is known only in length
   (7 / 30 / 180 days), not in position.

---

## 1a. Test evidence

Environment: `energy_env`, Python 3.10.20, `pandapower 2.14.1`, `openpyxl 3.1.5`,
`gurobipy 13.0.1`, torch 2.11.0. Command in every case:
`python -m pytest tests/ -q` from the repository root.

| state | result | log |
|---|---|---|
| before 10A (validated reproduction) | 785 passed, 51 deselected | — |
| after 10A (metadata revision) | **798 passed, 51 deselected** | `outputs/_repro_logs_2026-10-12/pytest_10A.log` |
| after the V4 contract change | **805 passed, 51 deselected** | `outputs/_repro_logs_2026-10-12/pytest_v4_final.log` |

No failure at any point in the final state. The one failure seen mid-work —
`test_a_matching_spec_is_accepted` — was a *test* whose premise the change
invalidated: it cached a policy under `OBS_V3` and expected the environment to
accept it. Left as it was, it would have asserted the wrong contract; it now
asserts both halves, and `test_a_v3_policy_is_refused_by_the_v4_environment` and
`test_the_v3_checkpoint_dirs_are_recorded_as_v3` pin the refusal separately.

Three suite runs were taken on this machine; the last took 676 s against the
first run's 253 s because earlier runs were still finishing in parallel. The
**counts** are the evidence; the wall clock is not comparable across rows.

**Not independently re-verified.** The economic zero-drift claim of §1 rests on
the diff described there, which I ran; it has not been re-derived by a second
method. `outputs/` is git-ignored, so the artefacts it compares do not ship.

---

## 2. The future-price leak — root cause, fix, and what is still unverified

**Root cause.** `BiddingEnv` stored the whole 96-period day in `self.wholesale`
and then computed `avg_lmp = np.mean(self.wholesale)` — the mean of the *whole
day's realized external price* — in both `reset()` and `step()`. That single
quantity reached **four of the twelve** observation features:

| feature | how the day-mean reached it |
|---|---|
| `avg_lmp_norm` | it *was* the feature value |
| `last_lmp_norm` | via `price_ref = avg_lmp` in the denominator |
| `pred_lmp_norm` | same denominator |
| `ema_dev` | via `last_lmp = avg_lmp`, the block-0 fallback |

At block 0 an agent bidding for 00:00–01:00 saw a statistic of prices it had not
reached. It was also constant across the episode, which is the tell that it
could not be causal by block: a genuinely per-block quantity would move.

**What was audited and found clean.** `_compute_agent_peaks` normalizes load and
RE by their own **day forecast** peaks, and `_compute_system_load_re_ratio` reads
only the current block's forecast window — both are forecast-based, which a
day-ahead participant legitimately holds. `hist_lmp` is a deque filled by
*previous* blocks' clears, so it is past-only. `load_feat`, `re_feat`, `soc`,
`block_pos`, `avg_other_bid`, `bid_std`, `price_trend` carry no realized future.

**The fix (observation contract V4).**

1. `avg_lmp_norm` → **`avg_lmp_hist_norm`**: the mean of the LMPs observed *so
   far this episode*, renamed rather than kept so that the name states which of
   the two quantities is in the vector.
2. Every price ratio now divides by **`rl_env.PRICE_REF_CNY_MWH`**, a frozen
   module constant, not by the episode's mean. This is what closes the leak in
   the *denominator*, which is where three of the four features were hit.
3. The block-0 fallbacks for `last_lmp` and `ema` use the same constant. No
   whole-day mean remains anywhere in the observation path.
4. `OBS_V3` stays registered — so old policies can be *identified* and
   `load_policy` raises on the name mismatch instead of silently
   reinterpreting an actor built for a different layout. It is **not** the
   default; `BiddingEnv` now selects `OBS_V4`.

**Verification done.** `test_the_observation_ignores_prices_after_the_decision`
changes only periods 60–95 and asserts the pre-decision observations are
identical; `test_the_episode_wholesale_array_does_not_reach_any_feature` checks
it directly at the feature boundary; `test_the_price_reference_is_a_frozen_constant`
asserts the normalizer is module-level and not per-episode state. The regression
test is a **true** regression test: on the V3 formula those two arrays give
`avg_lmp_norm` 0.714 vs 1.250 and `price_ref` 300 vs 525, so it fails there.

**Still unverified.**

- **`PRICE_REF_CNY_MWH` is inherited, not calibrated.** It is the fallback
  denominator the pre-V4 code already carried (`420.0`). It is *not* verified to
  be the training distribution's mean. Re-deriving it from the declared
  **training** split and freezing it is an acceptance item (§5, condition 2).
  Calibrating it on validation or test data would reintroduce a cross-split
  leak by a different route.
- The fix removes *price* leakage. It does not re-derive the forecast-based
  normalizers under a stricter information protocol; §4 records that as a
  declared assumption rather than a proven property.

---

## 3. Checkpoints: disposition

| directory | contents | contract | state |
|---|---|---|---|
| `formal_v0_1_seed{1..5}` | `ESS*_ckpt_{50..450}.pt` + `best/` | `v3_12d` / `storage_quote_v2` | interrupted at 450/800 |
| `protocol_smoke_v0_1` | final + `best/` + `last/` | `v3_12d` / `storage_quote_v2` | complete, 40 episodes, λ=0.25 (retired) |
| `protocol_smoke_v0_1_lambda010` | final + `best/` + `last/` | `v3_12d` / `storage_quote_v2` | complete, 40 episodes, λ=0.10 |

**All are V3 policies and stay on disk, unmodified, recorded as V3.**

**The leak fix makes the resume question almost moot, and this must be stated
plainly.** V4 changes the observation, so a V3 actor cannot be continued into a
V4 run without asserting that an actor trained on a non-causal vector is a
valid starting point for a causal one. It is not a backwards-compatible change.
The options are therefore:

**Ruling: V4 training starts from scratch. The V3 checkpoints are not a V4
initialisation.** The reason is not only the contract version. A V3 actor was
trained on an observation that mixed in whole-day price information, so
continuing it into a causal observation would place two different observation
semantics inside one training trajectory — a result that could not then be
attributed to either. Continuing would be cheaper to run and impossible to
interpret, which is the wrong trade.

Disposition, recorded so the files stay auditable:

- `policies/formal_v0_1_seed{1..5}/best` and the `protocol_smoke_v0_1*`
  directories are **kept unmodified and archived as V3 historical training
  output**, with their recorded `obs_spec: v3_12d` as the identifying fact.
- They are **not** used to initialise any V4 actor.
- Should their historical behaviour ever be wanted, it is a **separate
  historical-study task** evaluated under a V3 environment and labelled as a
  non-causal old-contract result.
- **No paired-settlement diagnosis is required before V4 can start** — that
  question is now answered by ruling rather than by measurement.

**No checkpoint has been loaded, resumed, deleted or reclassified as usable or
unusable, and no MATD3 performance is reported.** `OBS_V3` remains registered so
the archive stays identifiable and `load_policy` raises on the mismatch; tests
`test_a_v3_policy_is_refused_by_the_v4_environment` and
`test_the_v3_checkpoint_dirs_are_recorded_as_v3` pin that refusal.

---

## 4. Synthetic price scenario generator — design

**Purpose.** The shipped workbook supplies one real day and three window means,
so it cannot support a train/validation/test split over days. A generator makes
the split possible; it does **not** make the result evidence about the real
market (§6).

**Regimes are defined on the LP capturable margin, not on the price range.**
This is the load-bearing design choice. §5 of the summary keeps three
quantities apart — the theoretical bound `eta_dis·max − min/eta_ch`, the LP
capturable margin (which respects power, energy, SOC band, terminal pin,
`cycle_cost` and efficiency), and the settled net revenue. A scenario labelled
"profitable" off the raw price range can be unreachable through a 0.4 MW /
1.6 MWh window, which is exactly how the earlier per-day analysis overstated the
attainable spread by 1.11× at the median and 1.39× at worst. So the label is
measured with `arbitrage_lp.margin_cny` against
`required_margin(cycle_cost, eta_ch, eta_dis) = 200.69565217391303`.

### The margin measure — corrections, adopted

The classifier as first drafted compared the wrong quantity. Five corrections
are recorded here and are binding on the implementation:

1. **`margin_cny` is the LP's net revenue after cycle cost, in CNY.** It is
   absolute money and must **never** be compared against
   `required_margin` (CNY/MWh). Measured on the synthetic 120→1000 band:
   `margin_cny = 344.289538` CNY.
2. **`margin_per_mwh_charged = margin_cny / chargeable_energy_mwh` is a
   net-revenue-per-charged-MWh indicator, not the gross price margin that
   `required_margin` corresponds to.** It nets off cycle cost and is taken per
   MWh *charged at the grid*, while `R` is per MWh *stored*. The two are not the
   same quantity and must not be substituted for each other.
3. **The gross price margin is unified to battery-internal stored energy:**

   ```
   Q_ch  = sum(ch  * dt)          Q_dis = sum(dis * dt)
   M_LP  = (margin_cny + cycle_cost * (Q_ch + Q_dis)) / (eta_ch * Q_ch)
   ```

   with terminal SOC balance verified. The identity

   ```
   M_LP = R + margin_cny / (eta_ch * Q_ch)
   ```

   is asserted by test. It is not an approximation: it holds exactly **when the
   terminal SOC is balanced**, because that makes `Q_dis / Q_ch = eta_ch ·
   eta_dis`, which is what collapses the cycle-cost term onto `R =
   cycle_cost·(1/eta_ch + eta_dis)`. The terminal-SOC pin is therefore what
   makes the gross margin well-defined, and a day whose endpoints do not balance
   has no `M_LP` at all.
4. **When the LP schedules nothing, `M_LP` is undefined.** The classifier must
   branch first on the LP's net revenue, the actual scheduled quantities and an
   explicit numeric tolerance — never on a NaN threshold comparison.
   `margin_per_mwh_charged` and `adjusted_spread_cny_per_mwh` are both NaN in
   exactly this case (measured on all four shipped curves: `margin_cny =
   -0.000000` CNY, both per-MWh fields NaN), so a ratio-first classifier falls
   through every branch on the one regime that most needs deciding.
5. **A non-zero schedule whose margin falls below `R − b` produces a
   `classification_inconsistent` diagnostic.** It is not silently folded into
   `no_arbitrage`: that would hide a generator or parameterisation error behind
   a legitimate-looking label.

Approved classification — `R = 200.69565217391303` CNY/MWh,
`b = 0.1·R = 20.0695652…` CNY/MWh, both **fixed before generation**:

| regime | condition | note |
|---|---|---|
| `no_arbitrage` | LP optimum is zero within tolerance, i.e. the unit never charges (`margin_cny ≤ tol`) | ratio fields are NaN here; this branch is decided by the schedule, not by a threshold |
| `marginal_arbitrage` | `R − b ≤ M ≤ R + b` | `M = margin_per_mwh_charged` |
| `profitable_arbitrage` | `M > R + b` | |
| *below band with a non-zero optimum* | `M < R − b` | `no_arbitrage` in economics; kept distinct from the zero case only so a barely-charging unit is visible in the audit |

`M` is measured under **the fixed reference storage configuration** — 0.4 MW /
1.6 MWh, `eta` 0.92/0.92, SOC band 0.10–0.90, `soc0` 0.50, pinned terminal SOC,
`cycle_cost` 100 per leg. Classifying on the raw peak-to-trough range is **not**
acceptable: the day's extremes are not jointly reachable through this window,
and the earlier analysis measured that overstatement at 1.11× median and 1.39×
worst. `margin_per_mwh_charged` and `adjusted_spread_cny_per_mwh` are both
reported per day so the label can be audited.

**`r` is normalized per scenario family**, not across the whole library — so a
family with more curves cannot silently dominate the normalization scale. §4
must state `r`'s definition and its physical meaning before generation. The
earlier library-wide choice made every price a function of how many days the
library contained; per-family removes that dependence at the cost of making days
incomparable *across* families, which is acceptable because a family is the unit
of splitting.

**Sampling quotas, fixed in advance.** The three regimes must each contribute
enough days to train on, and the counts must not be left to whatever the
generator happens to emit. A **pre-declared quota per regime** decides how many
days of each class enter each split, so a family that emits many `no_arbitrage`
days cannot dominate by volume. Validation and test days must not be light
perturbations of training days — asserted, not assumed.

**`261009` is kept aside as the single real-data case.** It is not mixed into
the synthetic statistics, and it is not called a multi-day test set: it is one
day, spatially averaged, held by project ruling.

An open item for step 2: state `r`'s definition and its physical meaning in the
report before any family is generated, since a normalization whose meaning is
unstated is one whose leakage cannot be reasoned about.

**Reproducibility, recorded with every day.** Each generated day is a pure
function of `(generator_config, day_id)` and carries:

- `source: "synthetic"` — a synthetic day may never be presented as a measured one;
- `family` — the scenario-family name;
- `regime` — the label above, **measured after generation and asserted**;
- `generator_params` — supply-stack knots/prices/slopes, external-system
  capacities and shape parameters, floor, cap, noise sigma, scarcity settings;
- `master_seed` and `day_id`; plus a hash over the generator config.

A day whose realized regime disagrees with the regime it was requested under is
an **error**, not a relabelled day — otherwise a mislabelled "profitable" day
would silently enter the profitable split.

**Build on the existing provider, not beside it.** `SyntheticWholesalePriceProvider`
already derives `r` from its own external demand/PV/wind against a library-wide
yardstick and maps it through a continuous supply stack with declared
continuity. The generator extends that with the regime parameterization and the
recorded provenance above; a second, parallel price model would be a second
source of truth for what a price is.

**Recorded for step 2:** `r`'s mathematical definition and its physical meaning
must be written down before any family is generated. The current
implementation derives `r` by normalizing net load against a library-wide
min/max, which is why every price depends on the library's size; the approved
per-family normalization changes that yardstick, so the definition must be
restated rather than inherited.

---

## 5. Train / validation / test isolation

**A scenario family is the unit of splitting**, not a single day. A family is a
named generator configuration plus a declared set of day ids.

**Declared, not derived.** The split is written into a manifest — family, the
generator-config hash, and the explicit train / validation / test day-id lists —
so it is a *declaration* that can be reviewed and hashed, not a function of the
data that a later change could silently move.

**Leakage checks, each as a test.**

1. **Disjointness.** No day id appears in two splits. Asserted on the manifest.
2. **Reproducibility.** Each day's price array is rebuilt from
   `(generator_config, day_id)` alone and compared byte-wise. A day that cannot
   be rebuilt is not a scenario.
3. **Observation causality.** The V4 guarantees of §2 — changing prices after
   the decision period does not move the pre-decision observation.
4. **Frozen normalization.** Every constant used to scale a feature
   (`PRICE_REF_CNY_MWH`, and any regime or threshold constant) is calibrated on
   **train only** and frozen before validation or test is touched. A constant
   re-derived per split is a cross-split leak even when no day is shared.
5. **No label-driven selection.** A policy or hyper-parameter is chosen on
   validation only; the test regime labels are not used to pick anything. The
   `greedy` thresholds stay at the declared 300/400 and are not re-tuned per
   family — the earlier finding that they were not re-tuned per curve is the
   standard to keep.
6. **No repeat-the-same-day.** Episodes must draw distinct days across a
   family. Repeating one day to fill an episode budget is the failure mode the
   finalization round explicitly ruled out, and it is asserted, not trusted.

---

## 6. One evaluation interface

**There is none today, and that is the main structural gap.** Two disjoint paths
exist and meet only at `settle_day`:

- `strategies.adaptive_bidding` drives `fixed` / `mpc` / `greedy_threshold` /
  `rl` and returns **declared quotes only** — no money;
- `counterfactual_eval` returns **settled per-agent profits** but accepts only
  `truthful` / `rule` / `myopic` / `rl` — it cannot drive `fixed`, `mpc` or
  `greedy_threshold`.

**Proposed:** one adapter over `counterfactual_eval.settle_day`, taking a
*declaration source* — either a whole-day strategy name or a per-block policy —
plus a family, a split and a seed list, and emitting **one record schema per
(arm, day, seed)**:

| field | meaning |
|---|---|
| `settled_net_profit_cny` | **primary metric.** Per agent and fleet, from `participant_payoff` through the endpoint-pinned whole-day settlement. Never the clearing objective and never a training reward. |
| `traded_mwh` | per agent `charge`/`discharge`, **and fleet totals**, with the unit count recorded |
| `soc_feasible` | bounds respected and `soc[T] == soc[0]` on the settled day |
| `settle_clean` | `False` when the clearing fell back; such a day is dropped and counted, not zeroed |
| `regime`, `family`, `day_id`, `seed` | provenance of the row |
| `n_seeds`, mean / std / median / positive rate | seed stability, over the same days |

Two things this schema exists to prevent, both observed in the repo: the
`declared_intent` counts are corner-match counts on the declared quotes and are
**not** performance; and the fleet's traded volume is 7.2× the single-unit figure
that `phase7`'s `traded` column quoted, so fleet totals and unit counts are
recorded explicitly rather than inferred by a reader.

`settle_day` stays the single settlement entry point — a second one would be a
second definition of what a participant earns.

---

## 7. Go / no-go conditions for a formal run

Every condition must hold and be backed by a command in the write-up.

1. Observation-contract tests green: no feature reads the episode's realized
   whole-day price; changing later prices leaves earlier observations identical.
2. `PRICE_REF_CNY_MWH` **re-derived from the train split only** and frozen,
   replacing the inherited value, with the derivation recorded.
3. The generator's regime labels **measured**, not asserted from construction:
   each generated day's LP capturable margin is computed and falls in its
   declared band.
4. The split manifest exists, is hashed, and conditions 1–6 of §5 pass.
5. The unified interface emits `settled_net_profit_cny` for all four arms
   (`fixed`, `mpc`, `greedy_threshold`, MATD3) on the same days and seeds.
6. The V3 checkpoints are **archived and marked V3**, and the V4 run's
   configuration and random seeds are fixed. ~~Diagnosis decides resume vs
   scratch~~ — settled by ruling: **from scratch, five independent seeds**.
7. Regime labels pass the **unit check**: `M` is a CNY/MWh quantity
   (`margin_per_mwh_charged`), never `margin_cny`, and the `no_arbitrage`
   branch is decided by the LP's own schedule because both ratio fields are NaN
   there.
8. Per-regime sampling quotas are declared in advance, and validation/test days
   are asserted not to be perturbations of training days.
9. No `cycle_cost`, efficiency, network-constraint or price parameter is moved
   to produce a positive number. The break-even floor stays
   `200.69565217391303`.

## 7a. Approved execution order

1. **Lock the V4 causal contract** — confirm every price feature is free of
   future realized prices, state the block-0 initialisation rule, state which
   forecast inputs are legitimate for the agent, and keep the regression tests
   green. *(Largely done in this round; the "legitimate forecast inputs"
   declaration is the remaining piece.)*
2. **Build and verify the three-regime generator** — unit check first, then the
   approved `R`/`b` classification, per-family `r`, quotas, recorded
   rules/parameters/seeds/provenance, and the split isolation method.
3. **Calibrate and freeze `PRICE_REF_CNY_MWH` from the training split only** —
   compute per-family representative price levels, combine **equally weighted
   across families** so a larger family cannot set the scale, record the
   formula, the data list and its hash, then assert train/validation/test all
   use that one frozen value and that no future price affects an earlier
   observation.
4. **Build the unified evaluation interface** — one settlement path for
   `fixed`, `mpc`, `greedy_threshold` and MATD3, emitting settled net profit,
   fleet charge/discharge, complete traded volume, SOC feasibility,
   `settled_clean`, and per-seed results.

Phase 12 is not entered from here.

---

## 7b. Step 2 — the generator, implemented and verified

**Files.** `src/synthetic_scenarios.py` (new), `tests/test_synthetic_scenarios.py`
(new, 30 tests).

**Suite: 835 passed, 51 deselected** (805 → 835), zero failures. Log:
`outputs/_repro_logs_2026-10-12/pytest_gen.log`.

### What it produces

`fam_a`, 18 days, quotas 6 / 6 / 6 fixed in advance:

| split | n | no_arbitrage | marginal | profitable |
|---|---|---|---|---|
| train | 11 | 4 | 4 | 3 |
| validation | 4 | 1 | 1 | 2 |
| test | 3 | 1 | 1 | 1 |

Measured counts equal the requested quotas exactly — the label is measured by
the LP, not asserted from how the day was built. Shapes disjoint across the
split. Digest `64df374b…2b6109`. Artefact
`outputs/synthetic_scenarios_report.json` (`outputs/` is git-ignored).

### Four findings that changed the design

1. **The band below R is closed, so `marginal` means "barely ABOVE".**
   `M_LP = R + margin_cny/(eta_ch·Q_ch)` and the LP schedules only when its
   optimum is positive. So a scheduled day always has `M_LP > R`; it can never
   sit in `[R − b, R]`. `M_LP = R` exactly forces `margin_cny = 0`, which is the
   idle branch. The reachable marginal range is strictly `(R, R + b]`.
2. **`classification_inconsistent` is a defensive guard, not a regime.**
   Correction 5 describes a state the balanced-SOC identity *forbids*. The
   branch is kept — a scheduled day landing below `R − b` would still be
   reported rather than absorbed — and is tested by driving `gross_margin`
   directly, because no real LP result can reach it. Verified: fabricating
   `M_LP = R − b − 1` yields a *negative* net margin, and the classifier
   correctly calls it `no_arbitrage`.
3. **A requested regime is not reachable on every shape.** The band is 10 % of R
   and on many shapes the margin jumps from "no cycle" straight past it. A fixed
   shape therefore cannot satisfy a quota, and the first implementation measured
   7 / 5 / 6 against a requested 6 / 6 / 6. The generator now walks candidate
   shapes until one lands, and a quota that cannot be filled within
   `max_shape_attempts` **raises** rather than shipping a short family — a
   family that silently under-fills the marginal band would train on a
   distribution nobody declared.
4. **`gross_margin` and `classify` had to share one idle test.** They first
   disagreed: `gross_margin` tested `q_ch ≤ 0` while `classify` tested
   `margin_cny ≤ tol`, so a numerically-scheduled but economically-empty day
   returned a number from one and was called idle by the other. Now both use the
   tolerance.

### `r`, defined

`r_t = (span · shape(t) − r_lo) / (r_hi − r_lo)` with `r_lo, r_hi` the
**family's own** declared range, `[0, span_hi]`. `shape(t)` is a cosine
interpolation between a declared trough and peak slot, spanning exactly [0, 1],
so the family's yardstick is its declared span bounds and **no day's realized
extreme enters the normalization** — a day cannot normalize itself. Physically,
`r` is the day's position on the external supply stack: 0 at the stack's
cheapest segment, 1 at its dearest. Per-family normalization removes the
library-size dependence reported for `synthetic_v1`; the accepted cost is that
days are not comparable across families, which nothing needs.

### Leakage checks

Templates are **continuous per day** (stride-derived trough/peak), not a small
table — a finite table smaller than the day count cannot give disjoint ids, so an
id-equality check is either always failing or vacuous. The check is therefore a
**separation**: every held-out day must be ≥ `MIN_TEMPLATE_SEPARATION` (6 slots)
from every training day, and the test that a reused shape is refused confirms the
check bites. Regimes are interleaved by day id, so a positional split is not a
regime split, and `assert_regimes_present` requires all three regimes in
**training** only — a two-day validation tail cannot hold all three, and
demanding it would fail correct families while proving nothing.

### Not done, per scope

`PRICE_REF_CNY_MWH` is **not** calibrated (step 3). No unified evaluation
interface (step 4). No MATD3 training. No family beyond `fam_a`; no quotas
tuned; no threshold moved.

## 7c. Step 3 — boundary verification, then the price-reference calibration

**Suite: 844 passed, 51 deselected** (835 → 844). Log:
`outputs/_repro_logs_2026-10-12/pytest_step3.log`.

### The boundary check falsified a claim made in §7b

I had written that "a scheduled day always has `M_LP > R`", because the LP's
optimum is non-negative and it schedules only when positive. **That is false at
exact break-even.** Priced exactly on the threshold (`p_d = (R + p_c/eta)/eta`),
the LP returns a **degenerate optimum**:

| day | `margin_cny` | `Q_ch` | `Q_dis` |
|---|---|---|---|
| exact break-even | **−4.263e−14** | **0.695652 MWh** | 0.588800 MWh |
| flat at 500 (genuinely idle) | −0.000e+00 | 0.000000 MWh | 0.000000 MWh |

So a day can run a **full cycle for zero revenue**. Three consequences, all now
implemented:

1. **Revenue and schedule are separate questions and now have separate
   tolerances** — `IDLE_TOL_CNY = 1e-6` CNY and `ZERO_SCHEDULE_TOL_MWH = 1e-9`
   MWh, deliberately not one constant. A single tolerance would report the
   break-even day as "no schedule" when it ran an entire cycle.
2. **A zero-revenue day stays `no_arbitrage` regardless of its schedule.** A tie
   is not an opportunity; classifying it as arbitrage would manufacture
   arbitrage out of indifference. It is labelled
   `schedule_class: degenerate_zero_revenue`, distinct from a genuinely idle
   day's `none`, so the two are never confused -- the regime is the same and the
   diagnostic is not.
3. **`M_LP` is computed only on a valid charge energy with balanced endpoints.**
   An inconsistent state now gets its **own label**,
   `MARGIN_STATE_INCONSISTENT`, instead of a number that would read as a market
   outcome. `gross_margin` still raises for direct callers; `classify` converts
   it to the label.

**Correction to §7b's finding ①:** the reachable marginal range is still
`(R, R + b]` for a day with *positive* revenue, but the claim as stated
over-reached. The correct statement is that `M_LP > R` holds whenever the
revenue is above tolerance, and that at zero revenue the schedule may still be
non-empty. §7b's labels were unaffected; its reasoning was.

### The calibrated scale

```
value = mean over families( mean over that family's TRAINING curves(
        mean_t price[t] ) )
```

Three levels, in that order, with families **equally weighted** so a family with
more curves cannot set the scale.

| | |
|---|---|
| **`PRICE_REF_CNY_MWH`** | **288.19757259048833** |
| families / training curves | 1 / 11 |
| `fam_a` representative price | 288.197573 CNY/MWh |
| data digest (sha256) | `89e1ea4afc2c3288de65dc4bd6ce699a39182b2622aba62a6fb425750f1a3eef` |
| artifact | `data/wholesale/price_ref_calibration.json`, sha256 `400200a36d0a5668f6388070add98dbe28305cb7a88b8e5819974971d89073` |

Computed from the **train split only** (11 of 18 days). The validation split,
the test split and `261009` are recorded in the artifact as excluded, and a test
perturbs the held-out days by 3× and asserts the frozen value and its digest are
**bit-identical** — so a held-out day cannot influence a constant it is then
normalized by. It replaces the inherited `420.0`, which was the pre-V4 fallback
denominator and had never been verified against any distribution. The constant
is frozen in `rl_env` with its provenance; re-calibrating is a deliberate act
that overwrites both the constant and the artifact.

### Scope — the 18 scenarios are generator-validation data

`fam_a`'s 18 days exist to **verify the generator**: that the three regimes are
reachable, that quotas can be met, that shapes are held out, that labels are
measured rather than asserted. They are **not** sufficient evidence that the
scenario coverage for a formal MATD3 run is adequate, and nothing here claims
they are. Before a formal run the protocol still needs a reproducible
**training-set sampling or expansion scheme**, with the validation and test
scenarios fixed and family isolation preserved -- the current 60/20/20 split of
one small family is a mechanism under test, not a training curriculum.

## 7d. Step 4 — the unified evaluation interface

**Location.** `src/unified_eval.py` (new), `tests/test_unified_eval.py` (new,
21 tests). **Suite: 865 passed, 51 deselected** (844 → 865). Log:
`outputs/_repro_logs_2026-10-12/pytest_step4.log`.

### What it is, and what it is not

It is an **adapter**, not a second settlement. Every arm is settled through
`counterfactual_eval.settle_day`, so there is still one definition of what a
participant earns. The interface exists because the two comparison paths did not
meet: `strategies.adaptive_bidding` drives `fixed` / `mpc` / `greedy_threshold` /
`rl` but returns declared quotes only, and `counterfactual_eval` returns settled
profits but accepts only `truthful` / `rule` / `myopic` / `rl`. Neither could
answer "which earned more on the same day".

### Input contract

| input | type | carries |
|---|---|---|
| arm | `ArmRef(label, family, name, policies, version)` | identity, which driver (`whole_day` or `per_block`), loaded actors, checkpoint/version tag |
| scenario | `ScenarioRef(scenario, day_id, split, family, data_digest)` | scenario, which split it belongs to, its family, its data digest |
| config | `MarketConfig` | the frozen economics |
| seed | `int` or `None` | `None` ⇒ the paired `day_seed_for(day_id)`; recorded either way |

`ArmRef` refuses an unknown driver or a name that does not belong to its family,
so a typo cannot silently evaluate a different arm.

### Output contract

Per record (`evaluate_arm`, one arm × one scenario × one seed): arm identity and
version; `scenario_id` / `scenario_family` / `split` / `scenario_data_digest`;
`scenario_freeze` (price-series digest, initial SOC, economics) ;
`economic_regime` (`regime`, `schedule_class`, `m_lp_cny_per_mwh`, `diagnostic`);
`settled_clean`, `solver_status`, `mode`; **`settled_net_profit_cny`** and
**`training_reward`** as separate fields; `per_agent_profit_cny`; `traded` plus
fleet charge/discharge; `soc` (bounds, terminal, feasibility); `price_ref_cny_per_mwh`
and `price_ref_artifact`; and `included` with `exclusion_reason`.

`summarize_records` aggregates over **included** records only, publishing
`n_records` / `n_included` / `n_excluded`, the exclusion reasons, the excluded
scenario ids, and an explicit `denominator` string. An arm with no included
record reports `None`, never `0.0`.

### Three metrics kept apart

Settled net profit is what the participant was paid through the clearing, from
`participant_payoff`. The training reward is the learning signal and is recorded
in its own field — a whole-day arm has `training_reward: None`, not a reward
invented from its profit. The LP reference is what the isolated reference unit
could capture on the same curve; it is a bound, not an outcome, and it is not
substituted for either.

### Failure handling

A failed clear, an SOC violation, or an unbuildable scenario produces a record
with `included: False`, an `exclusion_reason`, and a `failure` string — never a
zero profit, and never a silently dropped day. The scenario list is iterated
identically for every arm and is not filtered per arm, because a list that
shrank when an arm failed would let that arm's average improve by failing.
`assert_comparable` verifies from the recorded price digest, initial SOC and
economics that arms compared on a scenario really saw the same day, and the test
that a tampered digest is refused confirms it bites.

### Evaluation mode

`mode: "evaluation"` is recorded, and the property is tested rather than
asserted: a policy's parameters are bit-identical before and after an
evaluation, and `run_arm` performs inference only — no actor noise, no gradient
step. `PRICE_REF_CNY_MWH` and its calibration artefact are unchanged by
evaluation, checked by hashing the artefact before and after.

### Status — what is and is not established

**Automated tests pass**: 21 new tests, 865 total, zero failures, and no drift in
the phase 6 / 7 / 9 / 10A / 11 baselines.

**No real policy has been formally trained or validated.** Every arm exercised
here is either a rule strategy or an untrained stub actor built for the
no-parameter-change test. Nothing in this step reports a MATD3 result, and the
interface has never run against a trained checkpoint. The step is complete as an
*interface*; it is not evidence of any strategy's performance.

## 7e. Step 4 review — two rigour fixes and the 24-block smoke

**Suite: 879 passed, 51 deselected** (865 → 879). Log:
`outputs/_repro_logs_2026-10-12/pytest_step4b.log`.

### Fix 1 — the comparability check was price-only

`assert_comparable` verified the price digest, initial SOC and economics. Those
three do **not** pin the scenario: two runs can share a price curve, an SOC and
an economics block while differing in load or renewable output, and the profit
difference would then be attributed to the strategy when it belongs to the day.

Added `scenario_input_digest`: a canonical fingerprint of **every exogenous
input at the evaluation boundary**, with each component digested separately so a
mismatch names *which* input moved.

| component | covers |
|---|---|
| `price` | the 96-period price series |
| `load` | 96-period load forecast, per agent |
| `renewables` | 96-period PV and wind forecast, per agent |
| `agents` | bus, participant type, prosumer/wind flags, full storage spec |
| `economics` | `cycle_cost`, `terminal_soc_equal`, OPF mode, line multiplier |
| `network` | the scenario + config fingerprint |
| `combined` | the value `assert_comparable` actually tests |

**Excluded by design: policy actions, parameters and internal state.** Hashing
those would turn a difference *between strategies* into an apparent difference
*between scenarios*, which is the precise confusion the digest exists to prevent.
Verified behaviourally rather than by a source grep: four arms with materially
different declarations on one day produce an identical `combined` digest.

Network topology is covered by the scenario/config fingerprint rather than by
hashing pandapower's element tables — the network is a pure function of the
scenario and the config, both of which are in the digest. The test that a
`load`-only tampering is refused (with price, SOC and economics left identical)
confirms a price-only check would have admitted that pair. The older
price/SOC/economics cross-check is retained beside it: it is independent of the
digest construction, and a disagreement between the two would itself be a defect.

### Fix 2 — the per-arm means are not a ranking

Each single-arm summary uses its own denominator. If arm A fails the low-price
days and arm B the high-price ones, both means carry a selection effect and their
difference is partly an artefact of which days each kept.

Added `paired_summary(records, arm_a, arm_b)`, run on
`S_common = S_included(A) ∩ S_included(B)`:

- `n_included_a` / `n_included_b` / `n_failed_a` / `n_failed_b` / `n_common`
  and the common scenario ids;
- per-scenario `ΔΠ_d = Π_A,d − Π_B,d` and their distribution (mean, median, std,
  min, max, counts positive/negative/zero);
- `denominator: "n_common (scenarios valid for BOTH arms)"`.

Below `MIN_COMMON_SCENARIOS_FOR_RANKING = 3`, the result is
`comparable: False` with a reason and **no delta at all** — falling back to the
unequal-denominator means would present a selection-biased number as a
comparison. Failures are counted *beside* the pairing, so runnability and return
stay separate questions. `evaluate` attaches `paired` automatically for a
two-arm run.

This does not treat a failure as zero profit; it separates "what each arm earned
where it ran" from "where each arm could run at all".

### Fix 3 — the 24-block execution-path smoke (new file)

`tests/test_eval_smoke_24.py`, 6 tests. The `per_block/rl` arm had only been
exercised on a 3-block stub with hand-built actors; nothing had run the
production shape.

- Actors are built from the **real `rl_td3.Actor` class** and go through the real
  `save_policy` / `load_policy` path, so the spec guard, the `_makerb_policy_meta`
  round trip and the actor reconstruction are all exercised. They are
  **untrained** and deliberately **not** V3 checkpoints, which are not a valid V4
  initialisation.
- One fixed scenario, **all 24 blocks**, full `T=96`.
- Checked: observation and action shapes, block boundaries, policy reset, output
  length, every RL agent accounted for with a finite profit, whole-day SOC bounds
  and the terminal pin, and repeatability of actions and settlement under the
  same model/scenario/seed (bit-identical).
- Checked: 24 blocks of inference move **no** parameter.
- Checked: a V3-labelled file is refused by this path.

**This is an execution-path verification, not performance evidence.** The actor
is untrained, so any number it produces describes where a random policy lands.
No claim about strategy quality may be drawn from that file.

### On the reference price of 288.20

Not changed, and it should not be for the sake of resembling a real day's mean.
For a future `261009` evaluation the protocol is: if the question is how a
**trained** policy behaves under real-day prices, keep the training-time frozen
reference and **report the price-distribution shift**; if a different
normalisation is genuinely wanted, it is a **separate declared experiment** with
its own recorded constant, provenance and applicability — never an implicit
re-fit at evaluation time. Otherwise the same policy on different days would
change its observation scale *and* its behaviour at once, and the experiment
could not be interpreted.

## 7f. Step 5 — training-pool design, sampling, and feasibility

**Suite: 890 passed, 51 deselected** (879 → 890). Log:
`outputs/_repro_logs/2026-10-12/pytest_step5.log`. **Design + feasibility only;
no long training started.**

### Implemented and tested

**Top-level family isolation.** The distinction that matters, and the one the
existing 60/20/20 split did **not** provide: a family's days share a generating
process, a mechanism set and a seed lineage, so a held-out day drawn from the
training family is a near-relative *however different its template looks*.
`synthetic_scenarios.split_families(specs, assignment)` assigns whole
`family_id`s and **refuses** any family appearing in two uses, any family with no
declared split, and an unknown split name. `FamilySpec.family_id` defaults to
`name`, so a single-family pool cannot accidentally look isolated — which is
exactly the case for `fam_a`: **its 18 days are generator-validation data, and
the within-family 60/20/20 split must not be described as family isolation.**

**Structural price mechanisms**, replacing amplitude-only variation: the closed
set `MECHANISMS = (single_peak, double_peak, broad_plateau, shifted_trough)`,
each a different *functional form*, not one cosine at different heights. Days
record which mechanism they used. Verified structurally distinct (≥3 pairwise
dissimilar) and that a family lands on more than one.

**Acceptance accounting.** Quota filling is a search, and a search that reports
only its winners hides both its cost and its selection bias. `build_family`
now takes an `accept_log` recording proposals, solves, acceptances, refusals,
**which measured regime each refusal landed in**, attempts-to-accept per regime,
and which mechanisms were accepted.

**`feasibility_probe`** — sized by measurement, not aspiration.

### Measured cost (one family, 4/4/4 = 12 days)

| quantity | measured |
|---|---|
| generation | 2.56 s (0.213 s/day) |
| LP per curve | 2.0 ms (24 ms total) |
| **acceptance rate** | **0.80** overall |
| — `no_arbitrage` | 1.00 (0 refusals, 1.0 attempts) |
| — `marginal_arbitrage` | 0.80 (max 2 attempts) |
| — `profitable_arbitrage` | 0.667 (max 2 attempts) |
| refusals by measured regime | `no_arbitrage` 2, `marginal_arbitrage` 1 |
| mechanisms accepted | all four (3/3/4/2) |
| pool size | 2.7 KB/day of JSON |

The rotation across mechanisms brought attempts-to-accept down to a maximum of
2 — the earlier single-shape generator needed a 24-candidate walk. **The dominant
cost is generation, not the LP**, by two orders of magnitude; a pool is cheap to
size up, and the binding constraint is distributional rather than computational.

### Design recorded, not implemented

**Manifest contract.** Each scenario instance to record: scenario id,
`family_id`, generator version; split use; generation parameters, seed and
generation order; per-input digests (price, load, wind, PV, network/config) —
the `scenario_input_digest` from §7e is the natural carrier; initial state and
economic-parameter identity; LP status, net margin, charge/discharge, measured
regime and diagnostics; and **accept / reject / exclude with its reason**.

**Two distributions, kept apart.** The *proposal* distribution is what the
generator emits given its parameter ranges. The *training sampling* distribution
is what a run actually draws, and may deliberately over-sample rare regimes.
Conflating them would make a balanced training pool look like a property of the
market. Training may use a declared regime-balanced sampler; **independent
evaluation must still report per-scenario results on the fixed evaluation pool**,
with balanced-test and representative-test pools distinguished by purpose.

**Layering.** Round one fixes network topology, storage spec and economics, and
varies only the price mechanism. Load-shape and RE perturbations are added in
later layers so an effect can be attributed to the layer that produced it.

**Versioned re-calibration.** Fix the training family and the sampling
distribution **first**, then compute the calibration from the final training set
by the §7c formula only, write it as a new versioned artifact, and keep the
original value and artifact for audit. Validation, test and `261009` stay out.
Freeze before training; never re-fit during evaluation.

### ⚠ `PRICE_REF_CNY_MWH = 288.19757259048833` is now STALE for the generator

This is a consequence of this step and it must not be missed: the mechanism
extension **changes the days `fam_a` produces** (three mechanisms were previously
all rendered as the single cosine shape), so the pool the constant was calibrated
on is no longer the pool the generator emits. The other three of the five
mechanisms also change the price series of the days whose `shape_id` selects
them.

The constant is therefore **retained as the original for audit and marked
pending re-calibration**, which is precisely what the versioned plan above
provides for. It has **not** been silently updated, and no evaluation may treat it
as current for the extended generator. Re-calibration belongs to the point where
the training family and sampling distribution are fixed — not to this step.

### Also found: the span bound is an upper bound, not exact

Mechanisms are bounded by `[0, 1]` but sampled on a discrete 96-slot grid, so a
peak slot lands short of the analytic maximum (measured 0.883 for `single_peak`),
and `double_peak`'s saddle raises its floor to 0.85. The family's declared span
bound is therefore an **upper bound** on the realized range. Normalizing each
mechanism to reach exactly 1.0 would tighten it, and is deliberately **not** done
here — it would rescale every generated price and so invalidate the existing
calibration by a second route. Recorded as an open item for the pool design.

### Exit-condition status

| # | deliverable | state |
|---|---|---|
| 1 | family definition, distributions, split rules, manifest design | **designed** (above) |
| 2 | reproducible generation and sampling | generation **implemented + tested**; sampling **designed** |
| 3 | feasibility trial and cost estimate | **done** (table above) |
| 4 | regime acceptance rates, refusal reasons, selection bias | **measured**; bias *assessment* still open |
| 5 | train/validation/test top-level isolation tests | **done** |
| 6 | reference-price decision and versioning plan | **plan recorded**; re-calibration pending |
| 7 | updated readiness report and regression results | this section; 890 passed |

**Not a completion criterion: day count.** No claim is made that this pool
represents the real Jiangsu market. The sampling scheme is designed but not
implemented, and the calibration is knowingly stale.

## 8. Remaining risk

**The one that cannot be engineered away.** Synthetic-family training is **not**
a cross-day generalisation result about the real Jiangsu market. The project
holds one real day (`261009`, spatially averaged, by ruling) and three window
means; a generator supplies as many days as asked for and **none of them is a
measurement**. Any result on a synthetic family is a statement about behaviour
under a *declared price process*, and must say so in those words. No cross-day
volatility, positive-day rate or generalisation statistic about the real market
is available from this data, and none may be reported.

**Other open items.**

- `PRICE_REF_CNY_MWH` inherited (condition 2).
- V3 checkpoints are not a valid V4 initialisation; the diagnosis is still
  outstanding and is now a decision, not a measurement, unless run under V3.
- The forecast-based normalizers (`load_feat`, `re_feat`) are a **declared
  modelling assumption** — a day-ahead participant holding forecasts. If the
  protocol is tightened to forbid them, they need the same treatment as
  `avg_lmp_norm` got, and the contract moves again.
- `outputs/` is git-ignored, so no artefact this document refers to ships with
  the repository; every table is reproducible only by running the commands.
- The suite count belongs to the environment that produced it, and the
  `pandapower` pin (`2.14.1` in `requirements.txt`) versus what a given run
  reports must be recorded per run.
- The 300/400 greedy thresholds remain **pre-registered heuristics** against the
  real price band, not values optimised on returns; they must not be revisited
  after seeing any profit figure.
