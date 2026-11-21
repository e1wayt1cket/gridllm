# Gateway import cap: what it changes and what it does not

`local_da_contracts.GATEWAY_IMPORT_LIMIT_MW = 3.0` is the default value of
`DayAheadInput.import_limit_mw`. Every library day and every training window
carries it because neither `run_local_da.build_day` nor
`local_da_days.window_day` passes the field. Decision record: D-012 in
`docs/project_context/10_DECISION_LOG.md`.

This note records the measurement behind that value. It does not re-freeze the
protocol: `import_limit_mw` is declared by a pinned source, so v7 currently
reads `NO_GO` (`code_sources_match: false`, `code_tests_match: false`,
`config_sha256: 0f35fed8`, `V7_candidate_004`) until the re-freeze is run.

## Why a cap at all

Uncapped, the marginal source is always the external purchase, so the fleet's
quote cannot reach any price. Measured on `2021-12-07`, whole-day window, the
four ESS buses over the periods the load fills the feeder (`t = 41..59`):

| quote (CNY/MWh) | corridor LMP, no cap | cap 3.0 | cap 3.5 |
| --- | --- | --- | --- |
| 0 | 38.25 | 327.38 | 312.82 |
| 150 | 38.25 | 203.80 | 193.67 |
| 300 | 38.25 | 76.84 | 70.23 |

and the import peak `pcc_max` over the same day: 4.466 MW uncapped, 3.000 at
the 3.0 cap, 3.500 at the 3.5 cap. Uncapped, the quote moves the price by
nothing; capped, it moves it by 250 CNY/MWh. That is the effect the cap is for.

## Feasibility over the library

`tmp/probe_cap_feasible.py` clears all 136 library days through `build_day` on
the capped contract and reads `dispatch.optimal` and the periods at the cap.

- `NOT_TESTED` for 41 days: the sweep was stopped by the harness after 95 days
  because the machine ran low on memory, so those days have been read and the
  rest have not. The 95 read are all optimal, the cap binds on 91 of them
  (18.3 of 96 periods per day on average, 47 at the most), and `pcc_max` equals
  the cap wherever it binds.
- No day in the 95 read was refused for infeasibility, but a 41-day gap is not
  a proof over the library: the frozen split counts are only protected once all
  136 days have been cleared on the capped contract.

Numbers and per-day rows: `tmp/cap_feasible.json`, `tmp/cap_feasible.log`.

## The wear term, which the cap exposed

`ClearingModel` carried `discharge_price * discharge - charge_price * charge`
and nothing else for the fleet. Wear is charged in settlement at
`degradation_cny_mwh * (charge + discharge)` and was absent from the clearing,
so the market cycled the fleet for a price it never had to pay. Against the
zero-price reference bid of `run_local_da._assemble_day`
(`(0, 0.4, 0, 0.4)`), the fleet discharged for nothing and was then made to
recharge, and under the cap it bought that recharge at the corridor price
(327 CNY/MWh on `2021-12-07` against an external price of 30). Truth-telling
was a settled loss, so a strategic quote had no meaning beyond escaping it.

`ClearingModel` now adds `degradation_cny_mwh` to both legs, so the market's
own cost of a round trip includes what settlement will charge the fleet.
`OperatingModel` and the unified `MarketModel` inherit that objective, so the
change reaches every clearing path, not only `local_da_clearing`. Two analytic
expectations moved with it: in `tests/test_da_market_system.py` a 180 CNY/MWh
spread against a 200 CNY/MWh round trip no longer cycles, so the objective is
`0.25 * (20 + 200) = 55` rather than `42` and the fleet's cash is `0` rather
than `18`.
Measured on five sample days, truthful quotes, fleet cash from the nodal price
less wear:

| day | cap | before the wear term | after |
| --- | --- | --- | --- |
| 2021-06-04 | none | −123 | 581.3 |
| 2021-06-04 | 3.0 | −8 | 582.3 |
| 2021-06-13 | none | 874 | 967.3 |
| 2021-06-13 | 3.0 | 312 | 971.4 |
| 2021-12-07 | none | 1039 | 1272.6 |
| 2021-12-07 | 3.0 | −99 | 701.7 |
| 2022-08-01 | none | −756 | 316.9 |
| 2022-08-01 | 3.0 | −547 | 322.7 |
| 2022-10-30 | none | 266 | 728.1 |
| 2022-10-30 | 3.0 | 0 | 730.7 |

Both columns are the same cash proxy: nodal price times net discharge, less
wear. "Before" is `tmp/cap_options.json`, measured before the wear term
existed, "after" is `tmp/probe_truthful_payoff.py`. Truth-telling is negative
in four of the ten rows before and in none of them after. The cap is not what
made it negative — `2021-06-04` and `2022-08-01` lose money uncapped too — but
it doubled the loss on `2022-08-01` and on `2021-12-07` it turned +1039 into
−99.

## What the cap still does, after the wear term

Shading the discharge ask, same five days, fleet cash at ask 0 / 100 / 200 /
300 CNY/MWh (`tmp/probe_shading_payoff.py`):

| day | cap | 0 | 100 | 200 | 300 |
| --- | --- | --- | --- | --- | --- |
| 2021-06-04 | none | 581.3 | 581.8 | 581.8 | 0.0 |
| 2021-06-04 | 3.0 | 582.3 | 583.2 | 583.2 | 20.8 |
| 2021-06-13 | none | 967.3 | 751.7 | 618.9 | 0.0 |
| 2021-06-13 | 3.0 | 971.4 | 776.0 | 626.9 | 174.5 |
| 2021-12-07 | none | 1272.6 | 1272.6 | 703.8 | 554.1 |
| 2021-12-07 | 3.0 | 701.7 | 939.9 | 703.8 | 554.1 |
| 2022-08-01 | none | 316.9 | 316.9 | 0.0 | 0.0 |
| 2022-08-01 | 3.0 | 322.7 | 322.7 | 200.3 | 300.1 |
| 2022-10-30 | none | 728.1 | 701.8 | 701.8 | 512.1 |
| 2022-10-30 | 3.0 | 730.7 | 742.7 | 735.0 | 735.0 |

Truthful is the best or within 1% of the best on eight of ten rows. The one
real gain is `2021-12-07` under the cap: asking 100 earns 238 CNY/day more
than asking 0, because the cap makes the fleet's own recharge valuation the
corridor price. That is monopsony over a gateway no one else uses, and it is
the only strategic payoff the market currently offers.

The consequence worth stating plainly: this market is close to
truthful-optimal, so a learned policy has little to gain on most days. That is
a property of the mechanism to report, not a defect to patch here.

## Corridor price, after the wear term

`2021-12-07`, whole-day window, the four ESS buses over `t = 41..59`:

| ask (CNY/MWh) | no cap | cap 3.0 | cap 3.5 |
| --- | --- | --- | --- |
| 0 | 38.25 | 146.19 | 137.59 |
| 150 | 33.64 | 33.64 | 33.64 |
| 300 | 33.64 | 33.64 | 33.64 |

The ask no longer moves the corridor price upward; at an ask of 150 the fleet
stops cycling and the corridor empties. The 327 CNY/MWh seen before the wear
term was the fleet pricing its own forced recharge.

## The second tranche, on the same five days

`DayAheadInput` now carries `import_emergency_mw = 1.0` and
`emergency_import_premium_cny_mwh = 250.` (D-013), so physical import can
reach 4.0 MW and every load day stays clearable. Two things change.

**Withholding no longer makes a day infeasible.** Quotes of zero quantity on
all four units, `clear_day(..., price_probes=False)`:

| day | dispatch optimal | physical import peak (MW) |
| --- | --- | --- |
| 2021-06-04 | yes | 3.256 |
| 2021-06-13 | yes | 3.335 |
| 2021-12-07 | yes | 3.012 |
| 2022-08-01 | yes | 3.774 |
| 2022-10-30 | yes | 3.574 |

Before the tranche the same quotes made the capped day refuse.

**The scarcity price is now declared, not dictated by the fleet.** Emergency
MWh and the storage-bus LMP over the periods it is dispatched, discharge ask
swept, fleet cash from nodal price less wear (`tmp/probe_two_tranche.py`):

| day | ask 0 | ask 100 | ask 200 | ask 250 | ask 300 | emergency MWh | scarce LMP | slack ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-06-04 | 582.3 | 583.2 | 583.2 | 466.0 | 20.8 | 0 | — | — |
| 2021-06-13 | 964.5 | 756.2 | 622.8 | 622.8 | 97.2 | 0.27 | 294.7 | 280 |
| 2021-12-07 | 701.7 | 939.9 | 703.8 | 703.8 | 554.0 | 0 | — | — |
| 2022-08-01 | 322.7 | 322.7 | 200.3 | 231.6 | 185.1 | 0.08–0.40 | 350.8–356.8 | 330 |
| 2022-10-30 | 730.7 | 742.7 | 735.0 | 735.0 | 735.0 | 0 | — | — |

The ceiling column is `external price + 250` at the periods emergency runs; the
storage buses price above it by their loss factor. Asking above the local LMP
(294.7 on 2021-06-13) is what collapses the award, which is why ask 300 earns
what ask 200 does not.

Sweeping the charge leg the same way (discharge ask held at 0, charge ask
0/50/100/150) gives 582.3→300.5, 964.5→533.6, 701.7→279.9, 322.7→−264.5 and
730.7→542.7: also monotone down. Settlement pays the nodal LMP on both legs,
so a declared price changes the award and never the unit price; the payoff-
maximizing quote is the reference truthful point itself. The tranche bounds
scarcity (and removes the withholding trap) but does not create room for a
profitable mark-up. The only positive shading in the sample remains the
2021-12-07 monopsony, +238 CNY/day at ask 100.

Two days settle end to end under the tranche with probes on (`2021-06-13` and
`2022-08-01`, truthful quotes): `SMOOTH_PASS`, `qualified_for_settlement:
true`, exact and replay audits passing. `NOT_TESTED` for the rest of the
library.

## The cone tolerance this moved

At `BarQCPConvTol = 1e-8` the exact model left `2021-06-13` at
`cone_tightness_max_mw2 = 1.04e-5`, just past the 1e-5 audit bound; before the
tranche the same day sat at `5.42e-6`. The exact model now runs at `1e-9`
(that day: `8.6e-7`, +1.1 s) and the continuous pricing model keeps `1e-8`
(it returns SUBOPTIMAL at `1e-9`), with all five sampled days passing every
audit. `NOT_TESTED` over the 136-day library.
`tests/test_local_da_pricing_tolerance.py` pinned a shared barrier tolerance and
now pins the split.

## Re-freeze still pending

`local_da_contracts.py`, `local_da_clearing.py` and `tests/test_local_da_market.py`
are pinned, so v7 reads `NO_GO` until the re-freeze runs. Not started:

1. `CANDIDATE` to `V7_candidate_005` in `src/make_formal_training_manifests.py`,
   which moves the config hash.
2. Re-measure the 136-day library with `price_probes` on the new objective, in
   five batches, into a new qualification directory.
3. Rebuild the feasibility screen and repoint `local_da_audit.SCREEN`.
4. Re-run `smoke_local_da_v7` and `tools/audit_local_da_v7.py`, which also
   re-derives the deviation weight; `rl_td3.CALIBRATED_DEV_PENALTY` was chosen
   on the old objective and is not carried over unchecked. The barrier
   tolerances are now model-specific (`1e-9` exact, `1e-8` continuous) and the
   cone bound must be re-measured over the library, not just the five days.
5. Regenerate `pretraining_validation_results.json`, `--finalize`, then
   `--check`.
6. Write the `REFREEZE_V7_*.md` record with the gate proof.
