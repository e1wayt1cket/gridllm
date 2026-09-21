# Native-scale ESS: resize, re-siting, and what the result says

The native-scale B3 gate failed because the fleet was sized for a 10 MW
scenario: 4 MW of battery against a 3.715 MW system peak (108%), with two of
four units doing nothing and ~5 CNY/day of fleet profit. This round resizes it
and re-sites it by a declared electrical rule. **No MATD3 training.**


> **Superseded in part.** The native scenario is now priced by an external
> wholesale market rather than by the retired net-load map, so the ESS and
> price figures below were measured under a price source that no longer
> applies. The siting result is unchanged -- it is a voltage screen and does
> not read the price. See `docs/2026-09-20-external-wholesale-price.md` for
> the re-derived numbers.
## The resize

4 × (0.4 MW, 1.6 MWh), i.e. `capacity_mwh: 4.0 → 1.6`; `power_ratio: 0.25`
already gives 0.4 MW, so it did not change. Everything else is untouched: eta
0.92, SOC [0.10, 0.90], SOC0 0.50, terminal SOC equality, cycle_cost 100,
truthful (0,0).

The fleet is now 1.6 MW against a 3.715 MW peak — 43%, under the system's own
demand, so it cannot decide the day for the network on its own. That is the
property the resize exists to restore, and it is asserted in
`test_the_fleet_is_not_oversized_against_the_feeder`.

## The criterion: B0 finite-difference voltage self-sensitivity

```
S_i = dV_i / dP_i  ~=  [V_i(P_i + eps) - V_i(P_i)] / eps
```

A small real injection at bus `i`, measuring bus `i`'s own voltage response, on
`case33bw` at its own load (3.715 MW / 2.3 MVAr, no DER, no ESS). That is the
standard DG-placement criterion and the one the Enhanced-IEEE33
voltage-sensitivity logic refers to. It contains **no market quantity**.

`eps = 0.01 MW` is declared as a constant and the whole screen is re-run at
`2*eps`: the ranking is **identical** and the largest relative difference is
**0.12%**, so the step size is not doing any work.

**Not to be confused with electrical distance.** `_compute_bus_electrical_distance`
(cumulative line resistance to the slack) is reported as the diagnostic column
`electrical_distance` and takes no part in the screen. The two genuinely
disagree — distance is largest at bus 32, sensitivity at bus 17 — and
`test_electrical_distance_is_a_diagnostic_and_not_the_criterion` holds that.

### Screen result

32 buses measured. `max(S) = 0.0798` at **bus 17** — the network's voltage
minimum (B0 Vmin 0.9131 pu), so the bus with the worst voltage is also the bus
where an injection helps most. Bus 1, beside the slack, scores 0.0006 (0.7% of
max) and falls out on the floor.

Candidate pool = non-slack, non-renewable, `S_i >= 25% of max`:

> **[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 28, 29, 31, 32]** — 16 buses

Bus 30 is excluded as a renewable bus (wind 1.0 MW); 21 and 23 are renewable
and below the floor anyway. **Bus 17 stays in the pool**: it is on the main
trunk, but it is the most voltage-sensitive bus on the feeder and excluding it
on a topological label would discard the network's own weak end.

## The selection rule, declared before measuring

1. Seed = the candidate with the largest sensitivity (bus 17).
2. Each further pick = the candidate whose **minimum pairwise series
   resistance** to those already chosen is largest (max-min dispersion).

Selected: **[9, 14, 17, 27]**.

Note what this is not. It is not the top four by sensitivity (that would be
14/15/16/17, four units stacked at the feeder's end arbitraging the same
spread away), and it is not a profit ranking. Measured, the selected buses are
not the most profitable candidates — 28/29/31/32 all earn more alone — which is
the rule working as declared rather than as would flatter it.

## Screening table (diagnostic only)

Every candidate measured with a single unit there, its own two arms, on the
ESS-free day. Fields split three ways so a thin economic column cannot be read
as a physical verdict:

| bus | S | ch# | dis# | ch MWh | dis MWh | cycles | spread | deg-adj | profit | box share |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 0.3626 | 3 | 1 | 0.1181 | 0.1000 | 0.0682 | 322.8 | 17.0 | 1.77 | 0.32% |
| 9 | 0.3790 | 3 | 1 | 0.1119 | 0.0948 | 0.0646 | 316.9 | 11.1 | 1.07 | 0.14% |
| 14 | 0.7097 | 3 | 2 | 0.1065 | 0.0901 | 0.0614 | 317.4 | 11.2 | 1.02 | 0.14% |
| 17 | 1.0000 | 3 | 2 | 0.0947 | 0.0802 | 0.0546 | 313.4 | 7.2 | 0.54 | 0.06% |
| 27 | 0.3264 | 3 | 2 | 0.1699 | 0.1438 | 0.0980 | 329.5 | 25.1 | 2.65 | 0.70% |
| 28 | 0.4025 | 3 | 2 | 0.1752 | 0.1483 | 0.1011 | 334.1 | 29.6 | 3.18 | 0.97% |
| 32 | 0.5979 | 3 | 2 | 0.1798 | 0.1522 | 0.1038 | 326.9 | 23.2 | 2.44 | 0.60% |

(full 16-row table in `outputs/ess_siting.json`)

**No candidate was eliminated for low profit or low cycles.** Every one of the
16 has at least one charge and one discharge opportunity, so the single hard
feasibility condition is satisfied across the whole pool.

## B3 with the resized, re-sited fleet

Truthful against rule, primary band, seed 123, byte-for-byte reproducible:

| | |
|---|---|
| fleet charge / discharge | 0.3027 / 0.2562 MWh |
| fleet equivalent cycles | 0.1747 |
| active units | **3 of 4** |
| **inactive** | **bus 17** |
| fleet profit | **2.52 CNY/day** |
| truthful / rule | 2.58 / 2.52 CNY (Δ = **−0.06**) |
| LMP min/mean/max | 459.5 / 582.2 / 806.9 (ratio 1.756) |
| Vmin / shed / curtailment | 0.9692 / 0.0000 / 0.2779 MWh |

Per unit: bus 9 (−0.00), bus 14 (−0.00), bus 27 (+2.52), bus 17 (**0.0000**).

### Three readings, kept separate

- **PHYSICAL FEASIBILITY** — bus 17 has the *highest* sensitivity on the
  network (1.0) and, alone, three charge and two discharge opportunities. Its
  physics are fine.
- **MARKET OPPORTUNITY** — in the settled fleet it moves nothing. Units at 9,
  14 and 27 absorb the day's spread; even 9 and 14 end at exactly break-even.
  This is crowding, not a structural defect, and the two must not be conflated.
- **ECONOMIC OUTCOME** — 2.52 CNY/day, and truthful against rule differs by
  0.06 CNY, which is 2.4% of the fleet figure. **The bidding problem is still
  degenerate**: one policy is not meaningfully distinguishable from the other.

## The one alternative size, as instructed

0.5 MW / 2.0 MWh, the larger of the two predefined options:

| | 0.4 / 1.6 | 0.5 / 2.0 |
|---|---|---|
| active units | **3 / 4** | 2 / 4 |
| inactive | 17 | 14, 17 |
| fleet profit | 2.52 | 2.49 |
| truthful vs rule | −0.06 | −0.11 |

**Enlarging makes it worse.** The mechanism is the crowding above: bigger units
take the same daily spread and shut more of their neighbours out.

That also points somewhere, and it is worth stating rather than acting on
unilaterally: activity improved as the fleet shrank (4.0 MWh → 2 of 4 active;
1.6 MWh → 3 of 4; 2.0 MWh → 2 of 4). The trend suggests the smaller predefined
option, 0.25 MW / 1.0 MWh, is the direction that might reach 4 of 4. This round
tested the one alternative it was allowed; a second would be the continuous
search over ESS size that was explicitly ruled out.

## What this round does not do

No MATD3. No numeric pass/fail: per instruction the result is reported and the
decision on a second siting/size sensitivity round is left open. No scenario
parameter was tuned toward a target.

The honest summary of the economic picture is that **the native IEEE33 day has
little for a battery to arbitrage.** The degradation floor is 236 CNY/MWh
round-trip and the system-wide peak-to-valley LMP range is ~350 CNY/MWh, but
that range is spread across different buses and different hours, and the
per-bus degradation-adjusted margin is 6–30 CNY/MWh. The voltage constraint
that used to create deep-feeder price spikes binds in only 3 of 96 periods at
native scale, so the shadow prices that made storage valuable at 10 MW are
largely absent.

## Verification

```bash
PYTHONPATH=src python src/ess_siting.py --json outputs/ess_siting.json
PYTHONPATH=src python src/baseline_ladder.py --rung B3 --bands both --json outputs/ladder_b3.json
python -m pytest tests/ -q
```

417 default tests pass. B3 is byte-identical across repeated runs.
