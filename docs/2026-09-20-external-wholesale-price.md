# The external day-ahead wholesale price

> **Correction (2026-10-12) to the Jiangsu-curve section at the end of this
> document.** That section says "all four are means", reading 均值 as a
> *temporal* mean. The workbook answers two questions, and the section collapses
> them. **Space:** all four curves are `spatial_mean` — an average across
> locations, so no bus or zone is represented by any of them, and the extent and
> weighting are not recorded in the file. **Time:** `261009` is a **single
> day**; `7days` / `30days` / `halfyear` are means over 7 / 30 / 180 days.
> `261009`'s single-day reading is a **project ruling** reading its header's
> 均值 as a *spatial* mean, not an external confirmation.
>
> The section's binding conclusion is unchanged and for the same reason it
> gives: one day plus three window means is not a *series* of daily prices, so
> there is still no measured daily series and no basis for a real-price
> train/validation/test split. Its `Curve.is_measured` statement is now
> deprecated — that field means "the source is real measured data" and is true
> for all four — so single-day identity must be read from
> `Curve.temporal_statistic`. Everything else in this document is unaffected.
> See `reports/final_research_summary.md` §2.1.

## Why

`wholesale[t]` is the price at the point of common coupling -- what the DSO pays
to import and earns to export. It is not the nodal price inside the feeder,
which is `lmp[t, i]` recovered from the power-balance duals. Conflating the two
collapses the settlement.

The retired map built `wholesale` from **IEEE33's own net load**, which made the
price a function of the dispatch it was pricing: a battery bidding into it moved
the number it was settled at. It also had no supply-side structure. The
market-opportunity audit measured both consequences -- the declared floor 50 and
cap 900 were unreachable at any load this network can present, so the whole
reachable range was one straight line at ~121 CNY/MWh per MW, and the fleet's own
1.6 MW charging moved the price it traded against by ~194 CNY/MWh.

The price now comes from an **external market with its own demand, its own
renewables and its own supply stack**. Nothing in `wholesale_price.py` reads an
Agent, a network or a clearing result, and a test asserts that perturbing IEEE33's
load, PV and wind leaves the price bit-identical.

## What it is

```
external demand, PV, wind  ->  R_t                     (100 MW-scale, not IEEE33's)
R_t against the library's range  ->  r_t in [0, 1]
continuous piecewise supply stack  ->  marginal cost
noise + rare scarcity  ->  lambda_up(t), clipped to [120, 1000]
```

| segment | r | price at the knot | slope |
|---|---|---|---|
| 1 | 0.00–0.25 | 160 | 120 |
| 2 | 0.25–0.50 | 190 | 280 |
| 3 | 0.50–0.70 | 260 | 700 |
| 4 | 0.70–0.85 | 400 | 1333.33 |
| 5 | 0.85–1.00 | 600 → 900 | 2000 |

Slopes are non-decreasing, which is what makes it a stack rather than a fitted
line. Continuity is validated at load; a typo in one slope leaves a curve that
is still monotone and still plausible, so the check is the only thing that
catches it. The fourth slope is the exact rational -- writing `1333.0` puts a
0.05 CNY/MWh step at `r = 0.85` in a curve whose stated property is continuity.

`r` is normalized **across the whole library**, so a mild day sits low on the
stack and a severe day climbs it. That is the declared choice, and it has a
consequence that is reported rather than hidden: **every price is a function of
how many days the library contains**, because the yardstick is the library's own
net-load range. The manifest records `r_min_mw` / `r_max_mw` / `number_of_days` /
`library_sha256`, and `library` mode **refuses** a library whose recorded
yardstick disagrees with the configuration rather than warning about it.

## What the library looks like

40 days x 96 intervals, master seed 20260920, committed at
`data/wholesale/synthetic_da_v1.csv`.

| statistic | achieved | design target |
|---|---|---|
| min | 131.96 | 150–200 |
| median | 202.46 | — |
| mean | 238.13 | 350–450 |
| std | 106.09 | — |
| max | 896.93 | 750–900 |
| P(price < 0) | 0.0000 | — |
| P(price > 800) | 0.0049 | — |
| max / min | 6.80 | — |

**The mean sits below its target and nothing was tuned to close it.** The target
was written before `r` was chosen to be normalized across the library; net load
is left-skewed, so most periods sit low on the stack even though every day
reaches its own extremes. Segment occupancy is
`[0.426, 0.404, 0.104, 0.047, 0.019]` -- the top two segments carry 6.6%. Moving
a supply-stack parameter to hit a price target would make the market an answer
key, which is what the retired map was.

Both clips are also effectively decorative under this calibration: measured
P(at floor) = 0.0000 and P(at cap) = 0.0000. They are reachable in principle --
the floor is 40 below the stack's own minimum and the noise sigma is 15 -- but
this library never reaches them. Reported, not corrected.

**Day 0 is the flattest day of the forty** (range 236 CNY/MWh against a median of
434). Leaving `day_id` at 0 would have made every single-day result in the
project -- the baseline ladder, the siting screen, the typical-day diagnosis --
report the mildest day in the library and read as though the market had no
arbitrage in it. The reference day is therefore **declared as day 11**, the
median-range day, chosen on the market's own price range and on nothing
economic. A test asserts it is still the median-range day, so rebuilding the
library fails loudly instead of drifting.

Across the library a perfectly-forecasting unit's round-trip margin has a median
of **175.1 CNY/MWh** (min -7.0, max 441.5) and **37 of 40 days are positive**.

## What it changed

**This is a price-source sensitivity result, not an AI performance result.**
Nothing here was produced by a trained policy: both arms clear a truthful
storage fleet and settle it. The factor below measures how much more
arbitrage the *market* offers, not how well any agent does. Every future
AI-vs-baseline comparison must run on the same `synthetic_v1` library, or
it will be measuring the price source rather than the policy.

Same B3 physics, same network, same storage, same seeds; only the wholesale
array swapped. Arms verified identical in physics on 12 of 12 days.

| | retired map | external market | ratio |
|---|---|---|---|
| price range | — | — | — |
| DLMP range | 345.8 | 495.2 CNY/MWh | 1.43x |
| ESS arbitrage margin | 0.000 | 79.70 CNY/MWh | — |
| **fleet profit** | **2.58** | **337.72 CNY/day** | **130.8x** |
| tradeable box share | 0.000 | 0.154 | — |

The 40-day market-opportunity audit, re-derived:

| | retired map | external market |
|---|---|---|
| days clearing a positive median margin | 9 / 10 | **10 / 10** |
| retention with the fleet present | 61.4% | **83.0%** |
| median bus margin | 19.58 | **81.11 CNY/day** |
| fleet's price range retained | 0.821 | **0.988** |
| price range ratio (typical day) | 1.756 | **3.62** |
| buses with a positive arbitrage window | a handful | **33 of 33** |

**The fleet's own footprint is gone.** It retains 98.8% of the price range
against 82.1% before, and shifts the mean by 0.07 CNY/MWh against 0.43. That is
the structural fix: the external market is 100 MW-scale while the fleet is
1.6 MW, so the slope the fleet trades against fell from ~121 CNY/MWh per MW to
under 12 even in the dearest segment.

**The strategy space improved but is still short.** The fleet's median tradeable
box share went from 0.033 to 0.086 (0.340 at its best bus-day) against the 0.25
threshold the siting rule declares. `verdict_box` remains
`INSUFFICIENT`. The margin is no longer the constraint; the quote box is.

The siting screen re-derived on the new price and selected the **same four
buses** `[9, 14, 17, 27]` -- it is a voltage-sensitivity screen and does not
read the price -- while per-bus degradation-adjusted profit moved from 0.5-3.2
to 65-75 CNY/day and the tradeable share from 0-0.25 to 0.30-0.35.

## What was not done

`dispatch_socp.py` is untouched: `wholesale[t]` still prices the PCC exchange
and `lmp` is still the balance dual. `market.py`'s `price_slice` semantics and
the rolling horizon are unchanged. No generator agents, no P2P, no CVaR, no DLMP
refactor, no change to ESS parameters, MATD3, reward or baselines. The retired
map is kept, marked legacy, reachable via `wholesale_market.source: legacy`, and
is never deleted.

## Two host bugs the re-derivation exposed

Both predate this work and were unreachable until the artifacts were re-run
standalone.

1. **`diagnose_typical_day` read `result["objective"]`**, a name the clearing
   stopped emitting when it became `clearing_objective`. `legacy_schema`
   translates retired names for artifacts *on disk*, but a freshly cleared
   result carries only the canonical one, so the read failed outright. Fixed to
   the canonical name.
2. **`ess_siting` imported pandapower before ortools**, which on Windows fails
   when pandapower's estimation subpackage is the first to load ortools. It
   passed under pytest, where an earlier module had already preloaded ortools,
   and crashed as a script. Fixed with the same preload `dispatch_core` uses.

A third, in `arbitrage_lp`: HiGHS presolve returned status 15 ("Unknown", primal
feasible) on one real bus-day's nodal prices -- library day 20 at bus 17, whose
prices carry a 5e-6 CNY/MWh near-tie -- so the bound simply did not exist for
that bus. Presolve is now disabled; the same instance solves to the same optimum,
and it is faster too (1.8 ms against 2.2 ms). The measured vector is kept as a
regression fixture, because a synthesized near-tie does not reproduce the
failure.

## Renewable curtailment: what actually caused the drop

The renewable consumption rate fell from 98% to **19.7%**. The first
explanation offered here -- "importing at ~205 is cheaper than local bid values
of 480-850" -- **was wrong and is withdrawn.** `bid_value` is a benefit term on
consumption that is identical whether local PV or an import supplies the load,
so it cancels out of the comparison and cannot drive curtailment.

Traced to the objective (`dispatch_socp`, and measured by
`src/audit_renewable_curtailment.py`), an agent without storage has exactly three
terms touching a renewable:

| term | coefficient | enters the decision? |
|---|---|---|
| `+ bid * served` | 350 / 720 | **no** — the term cancels |
| `- offer * (pv + wind + dis)` | 180 / 300 | **yes** |
| `- wholesale[t] * p_grid[t]` | market price | **yes** |

The carbon, RE-incentive and curtailment-penalty terms at lines 578-590 are
gated behind `market_design.enable_multi_objective`, which is **false**. So
`lambda_re: 50` and `lambda_curtail: 200` are declared and **inert** — there is
no curtailment penalty in the objective at all. The whole decision reduces to

```
dispatch  iff  wholesale[t]  >  offer_cost
```

Measured on the reference day:

| agent | bus | offer cost | bid value | share of day with `wholesale > offer` | consumption rate |
|---|---|---|---|---|---|
| Bus21R (PV) | 21 | 180 | 350 | 0.85 | **67.4%** |
| Bus23I (PV) | 23 | 300 | 720 | 0.14 | **0.3%** |
| Bus30I (wind) | 30 | 300 | 720 | 0.14 | **14.5%** |

Curtailment attribution: **13.2577 MWh by the objective's own comparison,
0.0000 MWh by the network**, 0.0696 MWh unexplained — of 13.3273 MWh total. The
voltage band and the line limits force *nothing*; the clearing simply declines to
pay 300 CNY/MWh for energy worth 205.

The rate is a step function of one coefficient per agent: holding prices fixed
and sweeping the offer cost gives 0.3% at 300 and 67.4% at 180. It is not a
statement about renewables being uncompetitive in general.

**No consumption-rate floor is asserted anywhere in the suite**, and the
re-derived 15% floor is **not adopted as a model rule**. A minimum would only be
justified by a documented external market or policy rule — a renewable
obligation, a curtailment penalty, a certificate price — none of which this
model has. Asserting one to restore a number would be fitting the market to the
measurement. The rate is reported by `audit_renewable_curtailment.py` as the
outcome it is, and the baseline test now asserts the *mechanism* instead: that an
agent carrying renewables keeps its own load type's `offer_cost`, rather than the
**999.0** a non-prosumer is given, which would curtail its output outright.

## Readiness: re-checking the tradeable-box criterion

The declared criterion is `diagnose_typical_day.TRADEABLE_BOX_SHARE >= 0.25` on
the quote box charge `[0, 150]` × discharge `[0, 300]`. On `synthetic_v1`, over
the 330 test bus-days:

| | retired map | synthetic_v1 |
|---|---|---|
| fleet median box share | 0.033 | **0.086** |
| fleet max box share | 0.113 | **0.340** |
| eligible bus-days | 0.0% | **10.0%** |
| `verdict_box` | INSUFFICIENT | **INSUFFICIENT** |

The criterion's arithmetic: because the tradeable region is
`charge + discharge < margin` clipped to the box, a share of 0.25 requires an
adjusted margin of **≥ 150 CNY/MWh**. `synthetic_v1` delivers a fleet median of
~80, and a 0.10 share at ~95.

**Not ready.** The gap is now a factor of about two and is quantified, rather
than the open-ended one it was when the margin was zero — and the space is real
on one day in ten rather than never. But `verdict_box` is INSUFFICIENT and no
MATD3 run should start on the strength of the profit figure.

## Instrumented comparison on the new price

`src/smoke_matd3.py --seeds 1,2 --episodes 60`, B3, band [0.93, 1.05],
dev-penalty 0.25. All seven checks pass on both seeds: no NaN, every declared
quote inside its box, every SOC inside its band, 1440 clears with zero fallbacks,
terminal SOC exactly equal, no degenerate quotes, no corner collapse.

This is an **environment and action-box instrument, not evidence about
learning** -- `smoke_matd3.py` never calls `trainer.remember()`, so no gradient
step is ever taken and the recorded quotes come from the seeded actor. See
`docs/2026-09-18-matd3-smoke.md`. The two columns below are a property of the
environment under each price map, not of a policy trained on either.

| | retired map (documented) | synthetic_v1 |
|---|---|---|
| mean deviation | 0.533 / 0.501 | 0.518 / 0.498 |
| corner fraction | 0.1% | 0.1% / 0.08% |
| quote dispersion | 21.09 / 22.79 | 17.76 / 19.91 |
| SOC range | [0.100, 0.500] | **[0.160, 0.615] / [0.160, 0.672]** |
| clears / fallbacks | 1440 / 0 | 1440 / 0 |

Two things are worth reading off this, and one of them is a caution.

**The fleet now charges.** Under the retired map the recorded finding was that
the SOC "never rose above its initial 0.5 -- only discharged net". It now reaches
0.615 and 0.672, so the sampled quotes now let the fleet cycle rather than only
drain. Read this as a property of the environment: the instrument does not
train, so what changed between the two columns is what `synthetic_v1` makes
attractive, not what the policy learned.

**The discharge quote saturates the box.** `quote_max` reaches 299.2 and 299.6
against a discharge box that stops at 300. The policy wants to declare a higher
discharge price than its action space allows, which is the same binding-box
reading the readiness section reaches from the market side.

**The caution:** mean deviation and quote dispersion are almost unchanged from
the legacy run (0.518/0.498 against 0.533/0.501) even though the market changed
completely -- the arbitrage margin went from zero to ~80 CNY/MWh. On 60 episodes
that is equally consistent with "too short to show the change" and with "the
policy's behaviour is governed by the deviation penalty and the action box rather
than by the opportunity". It is a question to investigate, not a conclusion, and
it is the reason no AI-performance claim is made here.

## Action-box sensitivity

Four boxes, everything else frozen: `[0,100]x[0,200]`, the production
`[0,150]x[0,300]`, `[0,200]x[0,400]`, `[0,250]x[0,500]`. The production box was
not changed and no box was selected.

### The criterion measures quote geometry, not strategy

`tradeable_box_share` is the triangle `charge + discharge < margin` divided by
the box's own area. Measured at a fixed margin of 80 CNY/MWh:

| box | box area | share | `m^2/(2WH)` | **absolute tradeable area** | margin needed for 0.25 |
|---|---|---|---|---|---|
| [0,100]x[0,200] | 20,000 | 0.1600 | 0.1600 | **3,200** | 100 |
| [0,150]x[0,300] | 45,000 | 0.0711 | 0.0711 | **3,200** | 150 |
| [0,200]x[0,400] | 80,000 | 0.0400 | 0.0400 | **3,200** | 200 |
| [0,250]x[0,500] | 125,000 | 0.0256 | 0.0256 | **3,200** | 250 |

The **absolute** tradeable area is identical across all four boxes -- the
triangle is bounded by the margin alone. The share is that triangle over the
box's area, so it **falls as the box grows** while nothing about the opportunity
changes. Two consequences:

- the margin needed to reach 0.25 is exactly the box's width
  (100/150/200/250), so the criterion compares the market's margin to the box's
  width and says nothing else about the market;
- it can be **passed by shrinking the box without creating any opportunity**:
  `[0,20]x[0,40]` scores 1.0000 at the same margin with an absolute area of 800
  against 3,200.

**So `tradeable_box_share` does not measure strategic non-degeneracy.** It
measures what fraction of a *declared* box is informative given a margin -- a
real question, but one confounded with the box's own size. The box-invariant
quantities are the margin and the absolute tradeable area; the share is neither.

### Per-box metrics

| box | share med | p90 | max | eligible bus-days | margin med | margin p90 | area med |
|---|---|---|---|---|---|---|---|
| [0,100]x[0,200] | 0.2886 | 0.8801 | 0.9988 | 711 / 1254 | 107.7 | 230.8 | 5772 |
| [0,150]x[0,300] | 0.1289 | 0.5192 | 0.7265 | 290 / 1254 | 107.7 | 230.8 | 5801 |
| [0,200]x[0,400] | 0.0725 | 0.3269 | 0.4824 | 231 / 1254 | 107.7 | 230.8 | 5801 |
| [0,250]x[0,500] | 0.0464 | 0.2130 | 0.3359 | 87 / 1254 | 107.7 | 230.8 | 5801 |

The margin and area columns are identical because they do not depend on the box.
Only the share columns move, and they move the wrong way: **enlarging the box
lowers the share and the eligible count.** The area differs by 0.5% only where
the smallest box's margin exceeds its width and clips the triangle.

### Paired probe

Same seed, same short training (30 episodes), one run per box. A probe, not a
result.

| box | boundary-hit rate | decision quote | cross-unit dispersion | action distance vs production | fleet profit | profit delta |
|---|---|---|---|---|---|---|
| [0,100]x[0,200] | 0.0311 | [72.84, 168.10] | 22.52 | 0.00 | 18.76 | +0.00 |
| [0,150]x[0,300] | 0.0000 | [72.84, 168.10] | 22.52 | 0.00 | 18.76 | +0.00 |
| [0,200]x[0,400] | 0.0000 | [72.84, 168.10] | 22.52 | 0.00 | 18.76 | +0.00 |
| [0,250]x[0,500] | 0.0000 | [72.84, 168.10] | 22.52 | 0.00 | 18.76 | +0.00 |

The override applied -- if it had not, every box would show the same
boundary-hit rate, and `[0,100]x[0,200]` shows 0.0311 against 0.0000 for the
rest. The converged policy declares the **same quotes in all four boxes**,
because that point lies strictly inside even the smallest one. Over a full-day
settlement the fleet's profit, charge and discharge are identical to the cent.

### Is the box a binding constraint?

**No, on this evidence.** The policy's declaration is interior to every box
tested, the boundary is hit only during exploration and only in the smallest
box, and no clearing or profit quantity moves. The criterion fails not because
the box is the wrong size but because it compares the margin to the box width.

**A correction.** An earlier reading of the smoke's `quote_max` of 299.2 against
a discharge ceiling of 300 concluded the box was binding in training. That was
wrong: `quote_max` is the maximum over *all* actions, dominated by the uniform
random warmup draws, and says nothing about the policy. The converged
declaration is 168.1, well inside.

**The caveat.** Thirty episodes and one seed is a short probe, and a converged
policy could in principle settle somewhere else with more training. What the
probe establishes is the mechanism -- the box is not what is stopping the
policy -- not a converged quantity.

### Not auto-selected

No box was chosen. Per the criterion alone the smallest scores best, which the
geometry section shows is the criterion rewarding a smaller denominator rather
than more opportunity. Selecting on it would be fitting the action space to a
metric that moves with the action space.

## The reference day

Day 11 is the library's **median-range day**, used so that single-day reporting
does not land on the flattest of the forty. It is a **reporting reference only**:
it is not claimed to be a statistically typical market day, nothing is fitted to
it, and every conclusion above that matters is drawn across the library rather
than from it. The per-day distribution is in
`outputs/synthetic_wholesale_price_audit.json`.

## Reproducing

```bash
PYTHONPATH=src python src/wholesale_price.py --build-library
PYTHONPATH=src python src/audit_synthetic_wholesale_price.py
PYTHONPATH=src python src/compare_price_sources.py --days 12
PYTHONPATH=src python src/market_opportunity.py --days 40 --with-fleet --fleet-footprint
python -m pytest tests/ -q
```

## The Jiangsu curves: a second source, and what it is not

Everything above describes `synthetic_v1`, a market this project simulates. A
second source now exists — `dianyicha` — which reads real published Jiangsu
prices. It is a different kind of object and the difference matters more than
the prices do.

**What it is.** `docs/日前价格261009.xlsx` carries four 96-point curves, read by
`src/dianyicha_price.py` and served by `wholesale_price.DianyichaPriceProvider`.
`day_id` selects one by index into `wholesale_market.dianyicha_curves`; it is a
*price scenario* selector, not a calendar day, and a day id outside the declared
list raises rather than falling back.

| sheet | its own column header | min / max / mean (CNY/MWh) |
|---|---|---|
| `261009` | 日前均值（CNY/MWh） | 225.90 / 445.00 / 386.83 |
| `7days` | 日前均价 | 257.50 / 477.87 / 395.55 |
| `30days` | 日前均价 | 333.08 / 450.86 / 407.82 |
| `halfyear` | 日前均价 | 284.93 / 393.86 / 346.33 |

**What it is not, and the one thing not to get wrong.** **All four are means.**
The workbook says so in its own headers, including on `261009`. There is
therefore **no measured daily day-ahead series anywhere in this project** — not
one real day, let alone a history. The four curves differ only in the window
they average over.

Consequences that bind the write-up, not just the code:

- A result on these curves is a **price-scenario comparison**, never evidence
  about day-to-day behaviour, volatility, or generalisation across days. With
  no measured daily series there is no basis for a train/validation/test split
  over real prices, and none is claimed.
- `Curve.is_measured` is `False` on every curve and `describe()` publishes each
  role, so a report cannot present a mean as an observation without
  contradicting a recorded field.
- These prices are far **narrower** than `synthetic_v1`'s configured band
  (`price_floor: 120` / `price_cap: 1000`). Storage arbitrage on them is
  correspondingly thinner. That is a finding to report, not a reason to move
  the floor or the cap.

**The slot axis is interval-END, and the mapping is positional.** A label is
the close of its interval, so the clock runs `00:15` to `24:00` and label `i`
belongs to period `i` — the identity, with nothing shifted. `24:00` has no
`datetime.time` representation and arrives as Excel's epoch midnight
(`1900-01-01 00:00:00`); it is accepted only in the final slot. The reader
asserts the exact sequence `15 * (i + 1)` minutes and raises on an
interval-START file rather than rotating it by one period — a rotation that
would leave every number plausible and every result wrong.

**Reproducing**

```bash
PYTHONPATH=src python src/dianyicha_price.py
PYTHONPATH=src python src/dianyicha_price.py --json
python -m pytest tests/test_dianyicha_price.py -v
```

The workbook is read directly with `openpyxl`; no text or CSV copy is produced,
and the reader's `provenance` carries the workbook's SHA-256. Note that
`.gitignore` excludes `*.xlsx`, so the workbook itself is outside version
control — the hash is what ties a run to the bytes it read.
