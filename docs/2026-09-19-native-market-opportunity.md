# The native market's arbitrage opportunity, across days


> **Superseded in part.** The native scenario is now priced by an external
> wholesale market rather than by the retired net-load map, so the ESS and
> price figures below were measured under a price source that no longer
> applies. The siting result is unchanged -- it is a voltage screen and does
> not read the price. See `docs/2026-09-20-external-wholesale-price.md` for
> the re-derived numbers.

## The question

The native IEEE33 single day is physically clean but the strategic-bidding
signal is under-excited. Two explanations call for opposite responses:

- **A, structurally unable.** The market cannot pay for a battery cycle, so no
  policy will ever have anything to bid over. Stop and diagnose the price and
  degradation specs. Do not restore a larger study scale.
- **B, viable but this day is weak.** Other native days carry real spreads, so a
  multi-day training run has a signal in it.

`market_opportunity.py` decides between them. It builds a fixed ensemble of days
at the frozen scale and measures, per bus and per day, the best a **single unit
with perfect foresight** could earn against that day's nodal prices in a
clearing that contains **no batteries at all**. Nothing about bidding enters, so
if a perfect forecaster cannot clear the floor, no policy can.

Frozen throughout, and never written by the audit: load native IEEE33
(3.715 MW / 2.3 MVAr peak-normalized 96-step), PV 0.3 MW @ 21 + 0.3 MW @ 23,
wind 1.0 MW @ 30, ESS 4 × (0.4 MW / 1.6 MWh) @ [9, 14, 17, 27], voltage
[0.93, 1.05] primary.

## The answer

**Case B on the declared bar, but the arbitrage is thin and there is almost no
strategy space to bid in.** The three readings, none of which is the whole
answer:

| reading | question | verdict |
|---|---|---|
| `verdict` | does a perfect forecaster earn anything? | **SUFFICIENT** — 9 of 10 test days, 19 of 20 overall |
| `verdict_attainable` | does it still earn anything with the fleet in the clearing? | **SUFFICIENT** — keeps **61.4%** of the bound, 18.82 → 11.56 CNY/day |
| `verdict_box` | is there an action left to choose? | **INSUFFICIENT** — median tradeable box share **0.033** against a 0.25 threshold, **0 of 330** bus-days eligible |

The market is **not** structurally unable to pay for a cycle, so the 10 MW scale
should not be restored. But "positive margin" and "something to learn" are
different questions and they come apart here: at this size winning the spread
consumes essentially the whole quote box, which is what "under-excited" means
stated as a measurement.

## Method

- **40 days**, built from named demand shapes (`declared`, `late_evening`,
  `sharp_evening`, `high_night`, `morning_led`, `flat_day`) crossed with drawn
  PV/wind/demand-error/price-noise seeds. Both caches are reset per day, so one
  day's levers cannot leak into the next.
- **Nothing economic is varied.** The ensemble moves the physical day only —
  weather realization and demand *shape*, never level, capacity or the price
  map. The real coincident peak measures **3.715000 MW on every day** and the
  installed capacity is **0.6 MW PV / 1.0 MW wind** on every day; both are
  asserted by test rather than promised.
- **The split is fixed before any price exists.** Regime labels come from the
  physics (PV/wind/load energy terciles and the demand peak's timing), and the
  deal into train/validation/test is a pure function of those labels. Nothing is
  fitted, so the split does no statistical work; it fixes the sample the verdict
  is reported on. Train 20 / validation 10 / test 10, no days dropped.
- **The bound is priced against an exogenous field.** Every record is measured
  in a clearing with `ess_buses=[]`, verified to contain **zero** storage
  agents, and the wholesale curve is verified identical in both arms — so the
  two arms are the same day.

The bound is a `scipy`/HiGHS LP (289 variables, 194 equality rows) rather than
the Gurobi clearing path, for a reason that is about the measurement and not
about the licence: a unit priced against a clearing it took part in is not
bounded by anything, because its own optimum moves the prices it is optimizing
over. Keeping the LP a leaf also means the decisive instrument is testable in CI
without a solver licence.

## What the days actually offer

Fleet buses, test partition:

| bus | LMP range (med) | adj spread (med) | P(>0) | P(>50) | P(>100) | margin (med) | P(>0) | box share |
|---|---|---|---|---|---|---|---|---|
| 9 | 408.0 | 57.5 | 1.00 | 0.56 | 0.00 | 18.19 | 0.90 | 0.028 |
| 14 | 412.2 | 61.0 | 1.00 | 0.56 | 0.00 | 19.41 | 0.90 | 0.032 |
| 17 | 415.9 | 64.1 | 1.00 | 0.56 | 0.00 | 20.49 | 0.90 | 0.035 |
| 27 | 419.5 | 70.1 | 1.00 | 0.56 | 0.00 | 22.41 | 0.90 | 0.042 |

Across all 1,320 bus-days the unit **sits out on 69 of them**, so the
probabilities below are quoted over the **1,251 that traded** (the module drops
a sat-out day rather than counting it as a zero spread, which would turn an
absence of opportunity into a measurement of one):

| quantity | over the 1,251 trading bus-days |
|---|---|
| adjusted spread, median | 47.9 CNY/MWh |
| P(adjusted spread > 0) | 1.000 |
| P(> 50) | 0.454 |
| P(> 100) | 0.006 |
| LMP range, median | 390.1 CNY/MWh |
| gross spread, median | 352.2 CNY/MWh |
| efficiency loss, median | 102.7 CNY/MWh |

The full `day | bus | LMP range | feasible charge/discharge | gross spread |
degradation-adjusted margin | charge MWh | discharge MWh` table is
`outputs/market_opportunity.csv` (1,320 rows).

**The spread a real unit can reach is not the day's extremes.** On day 0 at bus
9 the LP charges at 450.0 and discharges at 801.3, against a day range of
446.2–831.7. Pricing the spread off the day's minimum and maximum would report
385.5 CNY/MWh where the attainable figure is 351.2. Across the ensemble that
overstatement runs **1.11× at the median and 1.39× at the worst** — the day's
cheapest period and its dearest are not jointly reachable through a 0.4 MW /
1.6 MWh window. Both are reported; the bound uses the LP's own prices.

**The unit spends a fifth of a cycle a day.** Median equivalent cycles 0.20,
median charge 0.354 MWh against a usable window of 1.392 MWh, in 4 charge
periods and 3 discharge periods. One valley and one peak per day is one fill.

**The fleet destroys a fifth of the range it trades in.** Measured on day 0 by
clearing the same day with and without the fleet:

| bus | range absent | range present | retained | mean shift |
|---|---|---|---|---|
| 9 | 385.5 | 310.1 | 0.804 | −0.443 |
| 14 | 389.8 | 305.6 | 0.784 | −0.523 |
| 17 | 393.6 | 303.4 | 0.771 | −0.596 |
| 27 | 375.6 | 347.2 | 0.924 | −0.160 |

Mean range retained **0.821**, mean shift **−0.430 CNY/MWh**. The mean is not
the quantity that matters and is reported to make that clear: the fleet moves
the average price by well under one CNY/MWh while taking ~18% off the range, and
the range is what the LP spends.

## Diagnosis

Every entry below is a finding with its measured evidence, and every one carries
`adopted: false`. Nothing in this module writes a price, an elasticity or a
degradation constant — enforced by test, not by discipline.

### (a) The price map is affine over its whole reachable range

`price = 420 · (1 + 1.0 · net_load / max_demand)`, then ±30 noise, then a clip.

- Reachable range across the ensemble: **400.46 → 838.48 CNY/MWh**, ratio
  **2.094**.
- Declared floor **50**: **unreachable**. It would need net load below
  −3.27 MW against 1.6 MW of installed renewables.
- Declared cap **900**: **unreachable**. It would need net load above the
  realized peak.
- `spike_probability = 0.0`, so there is **no scarcity or step structure** — no
  mechanism by which price rises faster than load near the peak.
- Slope **121.4 CNY/MWh per MW** (median). The divisor is the *forecast* peak
  (3.485 MW here), not the realized 3.715 MW — quoting the realized peak
  understates the slope by ~6%. At this slope the 1.6 MW fleet's own charging
  moves the price it is bidding into by ~194 CNY/MWh.

Both dead constants are load-bearing: the map has no region in which it stops
responding linearly, so the day's price range is set entirely by the day's
net-load range rather than by any supply-side event.

### (b) The degradation floor, under three conventions

`storage.cycle_cost = 100.0` is charged on **both** legs. The three readings
differ by up to a third, so each is named rather than quoted bare:

| convention | floor at cc = 100 |
|---|---|
| per MWh stored (two legs) | **200.00** CNY/MWh |
| per MWh charged (`cc·(1 + η_ch·η_dis)`) | **184.64** CNY/MWh |
| per MWh delivered (`cc·(1/(η_ch·η_dis) + 1)`) | **218.15** CNY/MWh |

The ~236 CNY/MWh figure in `docs/2026-09-19-native-ess-resize.md` is the
delivered convention with the efficiency loss counted in, which is why it sits
above all three.

**Sensitivity only, never adopted.** Re-running the audit at cc = 60 and cc = 40
changes no record, no margin and no verdict — a test asserts exactly that. The
question the sensitivity raises is whether a degradation cost stated *per leg* is
the right unit at all, since the physically meaningful quantity is cost per MWh
of throughput and the two differ by a factor of two. That is a specification
question for the reader, not a knob to turn.

### (c) What would have to change

Stated as conditions, not recommendations:

1. **The spread must cover the floor with room left over.** Today the gross
   spread is ~350–390 CNY/MWh against a ~200–236 floor, leaving 48–74 CNY/MWh.
   A price process with a supply stack would widen this at the peak without
   touching the level.
2. **The fleet's footprint must be small relative to the spread.** At
   121.4 CNY/MWh per MW, a 1.6 MW fleet moves the price ~194 CNY/MWh per
   direction — comparable to the whole spread. Either a flatter response or a
   smaller fleet.
3. **There must be more than one cycle a day,** or value in a service other
   than energy arbitrage. One valley and one peak means one fill, and
   degradation is charged per MWh through the battery.
4. **The quote box must stay tradeable.** See below.

## What this means for a training run

The repo's own siting rule already answers the training question, and its
comment says the point is for a unit to have "a strategy space and not merely a
positive profit":

> `tradeable_box_share(margin, box) >= 0.25`

At a 44–74 CNY/MWh adjusted margin against a charge `[0, 150]` × discharge
`[0, 300]` box, over the test partition's **330 bus-days** the tradeable share
is **0.027 median across all buses**, **0.033 for the fleet's four buses**, and
**0.113 at its highest anywhere**. **No bus-day is eligible.** A positive margin
with 3% of the box tradeable is a market where the
optimum is essentially forced, so a policy has almost no action left to learn —
which is the under-excited signal, measured.

This is consistent with what the smoke run already showed: post-training quote
spreads of 21–23 CNY/MWh against a calibration that predicted 65.4.

## Conclusion

**Case B: the native market is economically viable and the single day is a weak
draw, not a structural absence of arbitrage.** The 10 MW scale should not be
restored.

**But proceeding straight to a multi-day MATD3 run is not supported by this
evidence.** The margin is thin relative to the degradation floor and the quote
box is almost entirely consumed by winning it, so the strategic-bidding signal
that motivated the exercise is still missing. The blocker is now localised to
three named specification questions — the price map's lack of supply-side
structure, the degradation cost's per-leg unit, and the fleet's size relative to
the price slope it trades against — rather than to the study scale or the day.

## Reproducing

```bash
PYTHONPATH=src python src/market_opportunity.py \
    --days 40 --with-fleet --fleet-footprint \
    --json outputs/market_opportunity.json --csv outputs/market_opportunity.csv
python -m pytest tests/test_market_opportunity.py tests/test_arbitrage_lp.py -v
```

Exit code is 0 whenever a verdict is emitted — including `INSUFFICIENT`, which is
a finding rather than a failure of the audit. `1` means the ensemble produced no
measurable day; `UNDETERMINED` is returned rather than a verdict when more than
20% of days are dropped.
