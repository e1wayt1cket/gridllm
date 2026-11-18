# v3 library clearing screen: results

Date: 2026-11-17. Library: `reports/candidate_days_v3_2026-11-17/evidence`
(136 days, split 90/18/18/10). Rule declared in `evidence/rule.json` before the
first solve:

- **feasible** = the exact complementary dispatch is optimal and every
  bus-period stays inside the network's own band `0.93 <= V <= 1.07`;
- **`qualified_for_settlement`** is a separate and stricter claim (physical
  audits, idle-direction stability, smooth prices) and is reported on its own.

Run command:

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clearing_2026-11-17/evidence
```

## Verdicts

| Claim | Verdict | Evidence |
| --- | --- | --- |
| One clearing per library day, 136 rows | PASS | `evidence/summary.csv` row count 136; dates equal the union of `split.json` `blocks` |
| Feasible days | PASS | 132 of 136; `evidence/summary.json` `counts` |
| Feasible by block: train 87/90, validation 17/18, test 18/18, holiday 10/10 | PASS | `summary.json` `counts` |
| Named exclusions match the screen | PASS | `excluded_days.json` names exactly the four `feasible != True` rows of `evidence_audits_slack1.00/summary.csv`; `tests/test_local_da_v3_inputs.py::ExclusionTests` fails when one entry is removed |
| Library not modified | PASS | `summary.json` `library_unchanged` = true, from comparing all `outputs` hashes before and after |
| Deterministic | PASS | `rerun_check/summary.csv` for 2021-07-18 and 2021-12-24: `optimal`, `feasible`, `price_qualification`, `qualified_for_settlement`, `voltage_min_pu`, `voltage_max_pu` identical to the first run |
| Runtime | PASS | `summary.json` `runtime_seconds_total` 386.4 s; per day median 2.74 s, max 5.52 s |
| `qualified_for_settlement` | PASS (measured) | 0 of 136; `counts` per block |
| Settlement, payoff, or training behaviour of any day | NOT_TESTED | The screen stops at qualification; no ledger was produced |

Nothing in the library was edited or regenerated. The screen writes only under
this directory.

## The four infeasible days

| Date | Block | Solver status | Solutions | Peak MW | Peak rank (1 = highest of 136) |
| --- | --- | --- | --- | --- | --- |
| 2021-12-24 | train | 3 (INFEASIBLE) | 0 | 4.0779 | 6 |
| 2022-08-02 | validation | 3 (INFEASIBLE) | 0 | 4.0333 | 33 |
| 2022-10-11 | train | 3 (INFEASIBLE) | 0 | 4.0718 | 9 |
| 2022-10-14 | train | 3 (INFEASIBLE) | 0 | 4.0847 | 1 |

All four are exact-dispatch infeasibilities, not voltage-band failures: the
solver proves no feasible dispatch exists at all. Three sit in the top 7% of
the library's peaks and the fourth at rank 33, all within 0.06 MW of the
declared band ceiling (4.0865 MW), while the library's median day peaks at
3.91 MW and clears. Feasibility is otherwise uniform across months and load
years (2021: 69/70, 2022: 63/66; the worst month is 2022-10 at 13/15).

### Named exclusions

The library is not edited. The four days are named in
`excluded_days.json`, generated from the screened evidence:

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/name_unservable_days.py
```

The artifact carries the rule that produced it, both declared inputs, the
solver status and failure text of every excluded day, and per-block counts:
train 87/90, validation 17/18, test 18/18, holiday 10/10.
`tests/test_local_da_v3_inputs.py::ExclusionTests` fails if the named set drifts
from the screen.

`docs/superpowers/specs/2026-11-17-local-da-rl-stack-design.md` allows only
train days to be dropped — "every validation and test day in `split.json` is
evaluated" — and its acceptance check 8 requires validation and test counts to
equal `split.json`. Excluding 2022-08-02 breaks that rule by construction. The
artifact records it as a validation exclusion rather than hiding it, and the
design sentence has to change before the acceptance check can pass.

The list describes the zero-capability screen only. Under a declared reactive
capability all four days clear, so a market that declares one needs no drop
list at all; the conflict above disappears with the exclusions rather than
being resolved by them.

## Voltage

| Quantity | Value | Source |
| --- | --- | --- |
| Days whose cleared minimum sits exactly on 0.93 | 119 of 136 | `summary.csv` `voltage_at_lower_bound` |
| Cleared `Vmin` range | 0.9300 - 0.9479 | `summary.csv` `voltage_min_pu` |
| Cleared `Vmax` range | 1.0060 - 1.0654 | `summary.csv` `voltage_max_pu` |
| Days that would fail a `0.93 <= V <= 1.05` band | 41 of 136 | `voltage_max_pu` > 1.05 |

The lower bound is the binding limit on almost every day, as the library design
predicted; the 1.07 upper bound never binds. The band change from 1.05 to 1.07
is what makes 41 of these 136 days feasible: under the old band they would have
failed on the high side.

## Price qualification: two different failures

`qualified_for_settlement` is 0 of 136 for two distinct reasons, and they must
not be merged.

1. **108 days never reached a price.** The conditional price solve returns
   Gurobi status 13 (SUBOPTIMAL) with `optimal: false` while its objective
   still agrees with the exact dispatch objective to 1.3e-4 CNY on an
   18,264 CNY day (`evidence/2021-09-02.json`, `pricing` vs `dispatch`
   `clearing_objective`), so the *status* is what refuses the price, not a
   numerical divergence. The solver's own log says why:
   `Warning: max constraint violation (1.8542e-07) exceeds tolerance`
   (`dry_run/2021-09-02.log`; solver logs are on disk but git-ignored).
   `clear_day` requires `optimal`, so `price_qualification` stays
   `NOT_TESTED`.
2. **24 days reached prices and failed smoothness.** Their pricing solve is
   optimal, `probe_prices` runs, and every one returns
   `NONSMOOTH_OR_FAILED`.

The same pattern predates this library: the v1 screen recorded `NOT_TESTED`
for 102 of 106 rows. It is a property of the price path on this network, not a
consequence of the v3 reactive basis.

## LMP sanity

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/lmp_sanity.py
```

| Claim | Verdict | Evidence |
| --- | --- | --- |
| The slack bus reproduces the exogenous price | PASS | 132 priced days; median `|LMP_slack - external|` 0.0000, max 0.0005 CNY/MWh |
| No negative prices | PASS | 0 of 418,176 bus-periods |
| The ordinary level is anchored | PASS | LMP median 306.51 against an external median of 300; mean bus markup over the slack runs 0.00% - 5.86%, rising with distance from the slack |
| Prices stay inside the declared 0-600 CNY/MWh research interval | FAIL | max 1621.85; 26 days carry a price above 600 and 16 above 800. Ten library days carry an external price above 600 themselves, so this count mixes the model's rent with the data's scarcity price; the markup rows below and the capability section separate them |
| The extreme tail is a voltage rent, not an energy price | PASS | 654 of 12,672 periods carry a bus above 3x the external price (1.94% of bus-periods), and 100% of those periods have a bus exactly on 0.93; 95.6% of them sit where the external price is at or below 60 CNY/MWh |
| The duals are internally consistent | PASS | 41 of 405,504 branch-periods show a price fall above 20 CNY across a branch; the worst is -28.10, a 2% step between two buses both above 1,400 |
| Those prices are certified optimal duals | NOT_TESTED | 108 of the 132 day sets come from a pricing solve with status 13 (SUBOPTIMAL). The above-3x share is the same in the 24 optimal ones (1.92% against 1.94%), so the pattern belongs to the mechanism, not to the status |
| The four unservable days | n/a | no pricing dict exists for 2021-12-24, 2022-08-02, 2022-10-11, 2022-10-14 |

Read together: the price is anchored at the slack, positive everywhere, and
its ordinary variation is the loss markup of a radial feeder. Its extreme tail
is the lower voltage bound. With `generator_q_limits_mvar = 0`, no load
shedding and a storage fleet bidding zero, the only relief from the far-end
0.93 limit is more import, and at the 30 CNY/MWh price floor that relief is
worth 20-40x the energy price. The maxima therefore land on floor-price
periods, i.e. exactly the periods an arbitraging agent wants to charge in.

## Why the lower bound binds

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/voltage_bound_cause.py
```

| Claim | Verdict | Evidence |
| --- | --- | --- |
| The benchmark itself sits below the band with no market at all | PASS | `case33bw` at its own 3.715 MW / 2.300 MVAr gives Vmin 0.9131 pu; the band's lower edge is 0.93 |
| The network carries no reactive source | PASS | 0 shunts, 0 generators, 0 transformers (no tap changer), slack fixed at 1.00 pu; the market declares `generator_q_limits_mvar = 0` |
| The declared reactive basis is not the cause | PASS | Uniform Q = 0.619 P is the benchmark's own aggregate ratio (2.300 / 3.715). At the peak period it differs from the native per-bus basis by at most 0.005 pu, and by 0.0000 on 2021-10-14 |
| The library's own load puts the tail under the band | PASS | Peaks of 3.55 - 4.09 MW give Vmin 0.881 - 0.939 with no storage; the benchmark's own scale already gives 0.9131 |
| Storage charging is the largest term the market controls | PASS | 1.6 MW charging lowers Vmin by 0.04 - 0.05 pu (2022-10-14: 0.9048 to 0.8592); discharging raises it by 0.03 pu |
| The band is not masking an infeasible day | PASS | With the band opened to 0.80 the market's own Vmin is 0.8607 / 0.8749 / 0.9230 / 0.9296 on the four sampled days, so the sag is the network's, not the bound's |

The chain is therefore: the feeder's own load sits below 0.93 nearly always,
nothing in the model can inject reactive power, and the market has no reason to
clear above the band once it is met. So the optimum parks the tail exactly on
0.93, and the dual of that constraint — not the energy price — sets the LMP at
those hours. Storage makes it worse in the direction the market already wants to
go: it charges at the cheapest hours, which deepens the sag exactly when the
constraint is the price-setting one.

### Would generator reactive power fix it?

The capability is already wired: `Generator.reactive_limit_mvar` exists, the
model declares `generator_q` within it and the reactive balance already adds it
at the generator's bus. Only `run_local_da.production_renewable_curves` leaves
the limit at its zero default.

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/reactive_support_probe.py
```

| Day | Q = 0 (now) | PF 0.95 | PF 0.85 |
| --- | --- | --- | --- |
| 2022-10-14 | infeasible | feasible, peak LMP 1088, 38 bus-periods pinned | feasible, peak 799, 32 pinned |
| 2021-09-20 | peak 1284, 6 pinned | peak 998, 2 pinned | peak 924, 2 pinned |
| 2021-06-04 | peak 396, 6 pinned | 396, 3 pinned | 395, 3 pinned |
| 2021-10-14 | peak 414, 3 pinned | 414, 0 pinned, Vmin 0.9331 | 414, 0 pinned, Vmin 0.9333 |

Verdict `PASS` for the mechanism and `FAIL` for sufficiency: enabling it turns
the previously infeasible day into a cleared one, cuts the peak price by about a
quarter and removes the bound from the mild day, but on the stressed days the
tail still sits exactly on 0.93. The reason is size: at PF 0.95 the three units
offer about +/-0.53 MVAr in total (PV 0.3 MW at two buses, wind 1.0 MW at one)
against 2.2 - 2.5 MVAr of load, so the relief is a fifth of what the feeder
needs. A remedy that reaches the corner needs shunt reactive plant of the order
of the load's reactive demand, or a slack voltage the operator can raise.

### What would actually reach the band

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/voltage_remedy_probe.py
```

Power flow on the worst cleared hour (fleet at available output, every battery
charging) for the two hardest days:

| Remedy | 2022-10-14 Vmin | 2021-09-20 Vmin |
| --- | --- | --- |
| none, slack 1.00 pu, no shunt | 0.8592 | 0.8724 |
| slack 1.02 pu | 0.8829 | 0.8956 |
| slack 1.04 pu | 0.9063 | 0.9187 |
| 1.0 MVAr capacitors at buses 17 and 30 | 0.8888 | 0.8963 |
| 2.0 MVAr capacitors at buses 17 and 30 | 0.9068 | 0.9141 |
| slack 1.04 pu plus 1.5 MVAr capacitors | 0.9457 | 0.9523 |

No single passive remedy reaches 0.93 at that hour; the combination does, with
Vmax at 1.041 pu still inside the 1.07 upper bound. The stress hour is not
pessimistic: the market solved with the band opened to 0.80 chose Vmin 0.8607 on
2022-10-14, within 0.0015 pu of this power flow, so the clearing really does
charge into the sag. Whether the clearing would re-optimise to a cheaper
dispatch under either remedy is `NOT_TESTED`; these are power flows, not
cleared days.

### Declared substation voltage

The slack voltage is now an input rather than a constant:
`RadialNetwork.slack_vm_pu` defaults to 1.0, `validate_day` requires it strictly
inside the voltage band, and the clearing holds `v[slack] == slack_vm_pu**2`
instead of a hard-coded 1. Every existing day still declares 1.0, so no archived
evidence changes.

Tests: `tests/test_local_da_market.py` covers the default, a value outside the
band being refused, and the cleared slack following a declaration of 1.04;
`tests/test_local_da_v3_inputs.py` locks the library's own 1.0.

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clearing_2026-11-17/evidence_slack1.02 --slack 1.02 --no-probes
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/compare_screens.py \
    reports/candidate_days_v3_clearing_2026-11-17/evidence{,_slack1.02,_slack1.03}
```

Both screens run with the probes off, so their `price_qualification` is
`NOT_TESTED` by construction and is not comparable with the 1.00 screen.

| Quantity | 1.00 | 1.02 | 1.03 |
| --- | --- | --- | --- |
| Feasible days | 132/136 | 135/136 (see note) | 136/136 |
| The four originally unservable days | 0/4 | 4/4 | 4/4 |
| Days pinned at the lower bound | 119 | 70 | 50 |
| Days pinned at the upper bound | 0 | 46 | 76 |
| Worst day-max LMP / day-max external | 3.17 | 2.38 | 1.74 |
| Days whose day-max ratio exceeds 2x | 9 | 3 | 0 |
| Bus-periods above 3x the external price | 8,100 | 3,659 | 2,233 |
| Bus-periods below half the external price | 0 | 212 | 353 |
| No negative prices | PASS | PASS | PASS |

The lever works: at 1.02 the four unservable days clear and at 1.03 every day
does, and the price tail shrinks by roughly three quarters. The cost is that
the binding constraint moves to the top of the band — 46 days at 1.02 and 76 at
1.03 sit exactly on 1.07 pu — and prices there fall below the wholesale price,
though never negative and only on 0.05% - 0.08% of bus-periods.

Note on 2022-06-22: the 1.02 screen records solver status 9 (time limit) for it.
Re-solving that day with a 600 s budget returns optimal in 2.8 s, so the day is
feasible and the row is a budget artifact. The screen treats a time limit as not
feasible, which is the honest reading; it has no retry pass, so a status 9 row is
`NOT_TESTED` rather than `FAIL`.

### Fixed compensation

The feeder can also declare fixed shunt compensation. `RadialNetwork.shunts`
takes `(bus, injected MVAr)` pairs, the clearing's reactive balance gains that
injection, and both physical checks account for it. The default stays the
uncompensated feeder: `build_day` declares no compensation and
`tests/test_local_da_v3_inputs.py` locks that, so no library day carries an
implicit asset and every earlier screen keeps measuring what it measured. A
screen declares it.

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clearing_2026-11-17/evidence_caps_slack1.00 \
    --capacitors 17:0.5 32:0.5 --no-probes
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clearing_2026-11-17/evidence_audits_slack1.00 --no-probes
```

0.5 MVAr at each of the two lateral ends that carry the sag lifts the benchmark
at its own rated load from 0.9131 to 0.9360 pu (`prototype/capacitor_placement.py`,
a power flow, not a cleared day). `prototype/voltage_levers.py` is the same
measurement for every other input that moves that number.

| Quantity | plain feeder | 1 MVAr compensation |
| --- | --- | --- |
| Days the exact dispatch clears inside the band (SOCP) | 132/136 | 136/136 |
| Days pinned at the lower bound | 119 | 65 |
| Days pinned at the upper bound | 0 | 79 |
| Worst day-max LMP / day-max external | 3.17 | 2.28 |
| Bus-periods above 3x external | 8,100 | 3,290 |
| Days with a negative LMP | 0 | 0 |
| Days the nonlinear replay reproduces | 132/132 | 107/136 |
| Days with a cone gap above 1e-3 MW2 | 0 | 29 |
| Largest cone gap | 2.3e-6 | 31.98 MW2 |

Compensation buys feasibility and halves the price tail, and it breaks the
model where the solution touches the upper bound. On 29 days the cleared
solution sits exactly at 1.07 pu while the nonlinear replay puts the same
dispatch at up to 1.0950 pu. The discrepancy sits in the current variable:
`v * i - p**2 - q**2` reaches 25 to 32 MW2, so the cleared voltage is not the
AC voltage. Those days are `NOT_TESTED` for settlement, not PASS.

The raised-slack screens carry the same defect on the days they pin 1.07.
Measured on two days re-solved with the audits on: 2021-09-30 has a gap of 17.85
MW2 at slack 1.02 and 28.55 at 1.03; 2022-10-07 has 0 at 1.02 and 24.71 at 1.03.
The table in the previous section therefore counts SOCP feasibility, and its
1.03 row overstates what the model verifies.

Separate and smaller: 61 plain days and 36 compensated days fail the cone
tightness gate by 1.9e-6 to 2.3e-6 against a 1e-6 audit tolerance. That is
barrier-convergence noise, not a loose relaxation, and it is why
`clear_day`'s pricing model runs at `BarQCPConvTol = 1e-8`; the audit tolerance
and that parameter need to agree before any day can be qualified.

### Declared reactive capability

The market can declare how much reactive power its generators may supply.
`Generator.reactive_limit_mvar` already bounds the reactive variable in both
the dispatch and the pricing model, the reactive balance already carries it at
the generator's bus, and it has no cost in the objective, so the clearing uses
it whenever it relieves a binding constraint. `run_local_da` leaves the limit
at zero; a screen declares it.

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clearing_2026-11-17/evidence_q0.30_slack1.00 \
    --generator-q 0.3 --no-probes
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/lmp_sanity.py \
    --evidence reports/candidate_days_v3_clearing_2026-11-17/evidence_q0.30_slack1.00
```

| Quantity | plain feeder | 0.3 MVAr per unit |
| --- | --- | --- |
| Days the exact dispatch clears inside the band (SOCP) | 132/136 | 136/136 |
| Bus-periods pinned at 0.93 | 2,326 (0.56%) | 1,807 (0.42%) |
| Periods with at least one bus pinned | 1,582 (12.48%) | 1,393 (10.67%) |
| Pinned-period markup to the external price, median / max | 2.19 / 41.42 | 2.48 / 41.08 |
| Unpinned-period markup, max, and share above 1.5x | 1.15, 0.0% | 1.16, 0.0% |
| Day-max LMP | 1,621.85 | 1,619.99 |
| Days with a price above 600 / 800 | 26 / 16 | 20 / 11 |
| Days the four previously unservable days clear | 0/4 | 4/4 |

Reactive capability fixes feasibility completely: all 136 days clear, no bus
touches 1.07, and the four days that needed a drop list clear without one. It
does not fix the price tail. The generators sit at their declared limit in
51.1% of generator-periods, so every remaining pinned bus-period is limited by
declared capability, and where the constraint binds the price is still the
constraint's dual rather than the energy price. On the same five days the
markup maxima move from 41.1x to 11.9x between 0.3 and 3.0 MVAr per unit, and
at 3.0 the day-max LMP falls to 362.5 - 431.8 CNY/MWh, inside the declared
0-600 interval, with 63 of 15,840 bus-periods pinned and the limit reached in
1-15% of periods.

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Declaring reactive capability makes every library day clear | PASS | `evidence_q0.30_slack1.00/summary.csv`: 136 of 136 `feasible`, 0 touching 1.07 |
| It removes the need for the four-day drop list | PASS | the four named dates clear; `excluded_days.json` describes the zero-capability screen only |
| Capability at nameplate scale fixes the price tail | FAIL | day-max LMP 1,619.99 against 1,621.85 plain; 20 days above 600 |
| The residual tail is capability-limited | PASS | 20,004 of 39,168 generator-periods sit at the declared limit |
| 3.0 MVAr per unit brings the five hardest days inside 0-600 | PASS | day-max LMP 362.5 - 431.8 on the five days ranked by markup |
| The other 131 days at 3.0 MVAr | NOT_TESTED | only five days were screened; no full run exists |

Both capability levels are declared research parameters, not Jiangsu rules and
not equipment data: PV21 and PV23 carry 0.3 MW nameplate and WIND30 1.0 MW, so
0.3 MVAr is one unit of reactive per unit of active for the PVs and 3.0 MVAr is
ten times nameplate. Adopting either needs its own line in the protocol.

### Controllable reactive capability at the binding buses

The generators sit at buses 21, 23 and 30; the sag sits at buses 17 and 31-32.
A reactive-only unit is a generator with no active availability and a reactive
limit, which is the synchronous-condenser or STATCOM structure, and the market
can declare it at any bus.

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clearing_2026-11-17/evidence_qbus_slack1.00_a \
    --reactive-bus 17:1.0 31:0.5 32:1.0 --no-probes --dates <first quarter>
```

All 136 days, in four quarters of the library:

| Quantity | plain feeder | 2.5 MVAr at buses 17/31/32 |
| --- | --- | --- |
| Days the exact dispatch clears inside the band | 132/136 | 136/136 |
| Bus-periods pinned at 0.93 | 2,326 (0.56%) | 1,088 (0.25%) |
| Periods with at least one bus pinned | 1,582 (12.48%) | 1,018 (7.80%) |
| Days whose worst markup exceeds 3x / 10x | 69 / 31 | 7 / 0 |
| Median of the day-worst markup | 3.29 | 1.23 |
| Worst day-worst markup | 41.4 | 9.5 |
| Days touching 1.07 | 0 | 2 |
| Days with a structural cone gap above 1e-3 MW2 | 0 | 0 |

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Capability sited where the sag is collapses the markup tail | PASS | day-worst markup above 10x falls 31 days to 0, worst 41.4x to 9.5x |
| It keeps the convex relaxation exact | PASS | worst cone gap 1.5e-5 MW2 over 136 days; no day above 1e-3 |
| It removes band contact | FAIL | 101 days still touch 0.93 and 2 touch 1.07; 2021-09-30 then fails the cone gate at 8.5e-6 |
| Every price lies inside 0-600 CNY/MWh | FAIL | and 10 library days cannot, whatever the mechanism: their external price itself reaches above 600, one to 1,500 |

The last row corrects a reading of the earlier [LMP sanity](#lmp-sanity) table.
Counting absolute prices above 600 mixes the model's own rent with the external
scarcity price. On 2022-06-12 no bus touches either edge and the day's highest
LMP is 1,595 against an external 1,500 — a 1.06x markup. The markup, not the
absolute price, is what the mechanism contributes.

**Adopted.** This asset is now the market's own declaration, not a screen
variant: `run_local_da.REACTIVE_SUPPORT_MVAR` is
`((17, 1.0), (31, 0.5), (32, 1.0))`, `local_da_contracts.condenser` names the
structure, and both `run_local_da.build_day` and
`local_da_days.window_day` carry it, so a library day and a training window
clear against the same feeder. The library files, the network parameters and
the load curves are unchanged; the legacy/historical reproduction path
(`build_legacy_day`, `build_historical_day`) deliberately does not gain it.
`tests/test_local_da_v3_inputs.py` pins the roster and
`tests/test_local_da_v7_days.py` pins the window. Verification that the
adopted default reproduces this table, not a different case:
`evidence_qbus_adopted_check/` was produced by the command above with no
`--reactive-bus` flag, and its rows for `2022-10-11` and `2021-09-30` give
`clearing_objective`
8,935.059671003673 and 12,601.810703781099, `pcc_peak_mw`
4.868355225508268 and 3.9117163383343385, and `voltage_max_pu`
1.0558306390754162 and 1.07 — identical to the two rows quoted above from
`evidence_qbus_slack1.00_d` and `_a`. `2022-10-11`, one of the four days the
plain feeder cannot serve, now clears.

Three limits of the declaration are stated rather than settled: the size was
chosen by measuring these same 136 days, so it is in-sample; reactive power
carries no price in the declaration objective; and the residual recorded above
(101 days touching 0.93, 2 touching 1.07, worst day-worst markup 9.5x,
`2021-09-30` a cone gap of 8.5e-6 MW2 against the exploratory 1e-6 gate) is
unchanged by adoption. Decision record:
`docs/project_context/10_DECISION_LOG.md` D-011.

### Larger storage capacity

The four units carry 0.4 MW and 1.6 MWh. A screen can declare a larger energy
capacity for the same power, which is the only storage parameter the fleet's
frozen definition leaves open without also changing its charging rate.

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clearing_2026-11-17/tmp_ess6.4 \
    --storage-energy 6.4 --no-probes --dates 2021-09-20 2022-11-11 2022-12-15 \
    2021-12-25 2022-11-27 2022-07-31
```

Six days, three capacities, same power:

| Quantity | 1.6 MWh | 3.2 MWh | 6.4 MWh |
| --- | --- | --- | --- |
| Bus-periods pinned at 0.93 | 242 (1.27%) | 233 (1.23%) | 199 (1.05%) |
| Worst day-max LMP | 1,284.1 | 1,261.5 | 1,228.9 |
| Mean of the six day-max LMPs | 948.1 | 914.3 | 829.2 |
| Total charging over the six days | 29.7 MWh | 32.9 MWh | 32.8 MWh |

| Claim | Verdict | Evidence |
| --- | --- | --- |
| A larger energy capacity reduces lower-bound pinning | PASS | 1.27% to 1.05% of bus-periods over the six days |
| It fixes the price tail | FAIL | the worst day-max LMP falls 4%, to 1,228.9 CNY/MWh |
| The market keeps using the extra energy | FAIL | charging rises 29.7 to 32.9 MWh and then stops at 6.4 MWh |
| A larger charging rate | NOT_TESTED | not screened; the forced-charging measurement below puts 1.6 MW at 0.04-0.05 pu of extra sag, so the expected effect is worse, not better |

Capacity is the wrong lever. The sag depth is set by the charging rate, not by
how long the unit can sustain it, and the extra energy is absorbed by the same
price-shape arbitrage instead of relieved hours: charging saturates between 3.2
and 6.4 MWh. Storage capacity is also frozen by the fleet definition, so
changing it is a protocol decision, not a network parameter.

### Load level or reactive draw?

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/voltage_attribution_probe.py
```

Vmin at the hardest hour, moving one input at a time, generators at available
output:

| Day | base | P and Q -10% | P -10% | Q -50% | Q = 0 |
| --- | --- | --- | --- | --- | --- |
| 2022-10-14 (4.085 MW, 2.528 MVAr) | 0.9048 | 0.9157 | 0.9124 | 0.9210 | 0.9365 |
| 2021-09-20 (3.551 MW, 2.198 MVAr) | 0.9102 | 0.9227 | 0.9188 | 0.9296 | 0.9478 |

The reactive draw costs about 0.013 pu per MVAr; the active draw about 0.019 pu
per MW. Fully removing the reactive load is worth +0.032 and +0.038 pu, which is
roughly the whole gap to 0.93, and it is the only remedy that does not cut
service. Cutting all load by 10% buys +0.011 and +0.013 pu, so a load-only
remedy needs a reduction near 25% on the stressed day.

The decomposition is therefore: the load level against this network is the root
cause (the benchmark violates the band at its own nominal load), and the
uncompensated reactive draw is the decisive lever (it is what a capacitor bank
or a tap changer buys back). The declared Q = 0.619 P is not a mis-setting: it
reproduces the benchmark's own aggregate ratio.

## Decision point

0. **The sited reactive asset is adopted** (D-011), so items 1 and 2 below no
   longer describe the market: all 136 days clear, none is dropped, and
   `excluded_days.json` is the record of the zero-capability screen only. What
   remains open from this list is the price signal itself (items 3, 4 and 6),
   which the asset does not repair.
1. **Train days.** 3 of 90 train days cannot be served under the plain feeder;
   all three are named in
   `excluded_days.json` with their block, price date and solver status. The RL
   design permits dropping train days for feasibility and requires them to be
   named, so this part needs no library edit.
2. **Validation.** 1 of 18 validation days (2022-08-02) also cannot be served
   and is named in the same artifact. Dropping it deviates from the design rule
   that every validation day is evaluated, so acceptance check 8 cannot pass on
   this artifact. The design sentence has to be amended, or the day kept and its
   evaluation recorded as failed; either way an unservable validation day is a
   signal about the mechanism, not just about the day.
3. **Price qualification is empty.** With 0 of 136 days qualified, it cannot
   serve as a training-day admission criterion until the SUBOPTIMAL status and
   the smoothness failures are settled as their own question. Both are market
   model decisions, not library decisions.
4. **Price level.** Cleared LMP reaches 1,621.85 CNY/MWh on voltage-bound
   periods at the price floor, outside the declared 0-600 research interval.
   Any consumer of the price signal, the RL observation included, will read
   those spikes as the cheapest hours of the day.
5. **The network levers are not adoptable; the market lever is.** A higher
   substation voltage and fixed compensation both remove the four unservable
   days and part of the price tail by pushing solutions against the upper band
   edge, where the convex relaxation stops being exact: cone gaps of 18 to 32
   MW2 and nonlinear replay voltages 2.5% above the cleared one. Declared
   generator reactive capability does the same job without touching the network
   and without breaking the relaxation: all 136 days clear and no bus reaches
   1.07. It fixes feasibility, not the tail — at nameplate scale the day-max
   LMP is unchanged, and only a declared capability ten times nameplate brings
   the five hardest days inside 0-600. The open decision is therefore how much
   reactive capability the market declares, and whether reactive power is
   compensated at all; neither is a network parameter.
6. **Audit tolerance.** The cone-tightness gate at 1e-6 rejects 61 plain days
   that miss by 2e-6, while the pricing model runs at a barrier tolerance of
   1e-8. Either the gate or the parameter has to move; today the gate rejects
   solutions the model considers converged.

## Notes on the artifacts

- Each `evidence*/` directory holds `rule.json`, `summary.csv`, `summary.json`,
  one `{date}.json` / `{date}.npz` / `{date}.log` triple per day. They are
  regenerable by the commands above and are not versioned (`*.csv` and `*.npz`
  are ignored repository-wide).
- `evidence/` is the first 1.00 screen, produced before `clear_day` archived its
  audits, so its daily records carry no `physical_audit` or `ac_replay`.
  `evidence_audits_slack1.00/` is the same rule cleared by the current code and
  is the one to read for physical verification; its feasibility counts match
  `evidence/` day for day.
- `evidence_caps_slack1.00/` is the compensated variant; `rule.json` in it names
  both declared inputs (`slack_vm_pu: 1.0`, `shunts_mvar: [[17, 0.5], [32, 0.5]]`).
- `evidence_slack1.02/` and `evidence_slack1.03/` predate the audit fields and
  are read only for the voltage and price columns.
- `dry_run/` holds the three-day timing run, `rerun_check/` the determinism
  rerun.
- `run.log` is the full-run console output.
