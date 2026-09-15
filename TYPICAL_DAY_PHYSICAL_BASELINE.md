# Typical day physical baseline

This is the physical base case every later bidding experiment is measured
against: one day on the IEEE 33-bus radial feeder, cleared jointly over all 96
fifteen-minute periods, with four independent batteries as the only storage.

Nothing here is an RL result. The point of the day is that the physical loop
closes: load, PV and wind enter the network, the clearing prices them nodally,
the batteries are dispatched by those prices, their state of charge returns
where it started, and their cash is settled. Whether a learned bidding policy
can do better than a truthful one is the next question, and it is asked against
this day.

Everything below is measured, not asserted. Reproduce with:

```
PYTHONPATH=src <python> src/diagnose_typical_day.py --json outputs/typical_day.json
PYTHONPATH=src <python> -m pytest tests/ -v
PYTHONPATH=src <python> -m pytest tests/ -v -m slow
```

where `<python>` is an interpreter with `requirements.txt` installed and a
Gurobi licence (`C:/Users/26036/.conda/envs/energy_env/python.exe` on this
machine). The report exits non-zero if a hard gate fails.

## 1. Network and boundary

IEEE 33-bus radial distribution network (`pandapower.case33bw`), 33 buses, 32
radial lines after the five tie switches are dropped, slack at bus 0, 12.66 kV.
No thermal generation exists anywhere in the model: carbon is charged on grid
import alone at 0.58 tCO2/MWh.

The upper grid enters as a **single free exchange** at the slack, positive
meaning import. It has no import limit and a 2.5 MW reverse-flow cap, which for
one variable is the same statement as bounding it below. Import and export are
the positive and negative parts of that one variable, so the network cannot
import and export at the same time — that is an identity of the representation,
not a property the solver happened to find.

## 2. Devices

Four independent storage units, identical by construction, one battery alone on
its bus:

| unit | bus | P_ch / P_dis | E_max | eta_ch / eta_dis | soc0 | soc_min | soc_max | region |
|---|---|---|---|---|---|---|---|---|
| ESS6  |  6 | 1.0 MW | 4.0 MWh | 0.92 / 0.92 | 0.50 | 0.10 | 0.90 | mid-feeder |
| ESS20 | 20 | 1.0 MW | 4.0 MWh | 0.92 / 0.92 | 0.50 | 0.10 | 0.90 | residential branch, beside the PV site |
| ESS24 | 24 | 1.0 MW | 4.0 MWh | 0.92 / 0.92 | 0.50 | 0.10 | 0.90 | beside the largest load |
| ESS31 | 31 | 1.0 MW | 4.0 MWh | 0.92 / 0.92 | 0.50 | 0.10 | 0.90 | end of the longest radial spur |

Fleet: 4.0 MW / 16 MWh. Each unit's state of charge is pinned to its starting
value at the end of the day (`storage.terminal_soc_equal`, applied through
`horizon_type="full_day"`), so no unit can be paid for finishing with banked
energy.

Renewables are the prosumer fleet with the non-wanted sites scaled to zero, so
the installed capacity is exact and spatially split:

| unit | bus | capacity | energy delivered |
|---|---|---|---|
| PV  | 21 | 2.0 MW | 23.71 MWh total PV |
| PV  | 23 | 1.5 MW | (both PV sites) |
| Wind | 30 | 1.5 MW | 18.57 MWh |

PV and wind sit on different buses in different branches, and no renewable bus
is also a battery bus.

## 3. Load level

The load level is stated as the quantity it is meant to be — the **coincident
peak**, the most the whole network draws in any one period — and the scale
factor that produces it is derived from the unscaled profiles at build time:

```yaml
profiles:
  load:
    target_peak_mw: 10.0
    load_scale: null      # null derives the scale from the target
```

The derivation is why the target survives a change to the profile shapes, the
per-bus ratings or the noise seeds. The scale multiplies the load profiles only,
so raising the load level does not resize every device on the bus.

| | value |
|---|---|
| coincident peak | 10.0000 MW (the configured target) |
| peak (sum of per-agent maxima) | 11.2567 MW — a diagnostic, not the headline |
| daily load energy | 179.0 MWh |
| P_ESS / P_peak | 35.5 % |
| E_ESS / load energy | 8.9 % |

The previous baseline quoted 11.0 MW, which was the sum of each agent's own
maximum. That number is larger than anything the network ever draws, because
the buses do not all peak in the same period, so it does not bound the flow the
feeder carries. It is still reported, as a diagnostic.

## 4. Inverters, voltage and line capacity

PV, wind and battery inverters share one apparent-power circle between their
real and reactive output, so every MVar spent holding the feeder inside its
voltage band is a MWh of real output they cannot deliver. The rating is a
multiple of the real-power rating it is sized from:

```
network.inverter_smax_multiplier = 1.10   # set by typical_day_config()
```

At 1.00 the two outputs share the circle exactly. That setting is what produced
the original baseline's 1.64 MWh of curtailment, which is not economic
curtailment: the amount is invariant across load levels and prices, and turning
reactive support off entirely takes it to zero while shedding 11.6 MWh instead.
Measured over the sweep:

| rating | curtailment MWh | shed load MWh |
|---|---|---|
| 1.00 | 1.64 | 1.219 |
| 1.10 | 0.95 | 1.019 |
| 1.40 | 0.11 | 0.539 |
| 1.80 | 0.00 | 0.181 |
| 2.50 | 0.00 | 0.0000 |

Both reach zero only at 2.50 — a 1 MW array behind a 2.5 MVA inverter, which is
not a device anyone installs. 1.10 is a little above the array, which is how
inverters are actually built, and is what the day is defined at.

## 5. The day at a glance

```
peak 10.00 MW, coincident                       valley prices at 11:45
PV and wind peak near midday                    peak prices at 18:15
the evening load ramp lifts the evening price   state of charge returns to 50% for all four
```

## 6. Acceptance report

Reproduced from `outputs/typical_day.json`.

**System**

| quantity | value |
|---|---|
| coincident peak load | 10.0000 MW |
| daily load energy | 179.0 MWh |
| PV installed / delivered | 3.500 MW / 23.71 MWh |
| Wind installed / delivered | 1.500 MW / 18.57 MWh |
| grid exchange, net | 135.8 MWh |
| grid import periods / export periods | 142.1 / 0.0 MWh |
| curtailment | 0.947 MWh |
| unserved energy | 1.044 MWh (0.58 % of load) |
| RE consumption rate | 97.81 % |
| LMP fallbacks | 0 (every price is nodal) |
| clearing objective | 17111 CNY |
| carbon, on net import | 82.44 tCO2 |

**Batteries.** Money comes from `participant_payoff`, the same implementation
the trainer and the evaluator route through.

| unit | bus | charge | discharge | churn | cycles | buy | sell | degradation | net |
|---|---|---|---|---|---|---|---|---|---|
| ESS6  |  6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.00 | 0.00 |
| ESS20 | 20 | 0.5910 | 0.5002 | 0.0000 | 0.1364 | 386.56 | 871.30 | 109.12 | +197.90 |
| ESS24 | 24 | 0.0194 | 0.0164 | 0.0000 | 0.0045 | 10.12 | 18.29 | 3.58 | +1.49 |
| ESS31 | 31 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.00 | 0.00 |
| **fleet** | | 0.6104 | 0.5166 | 0.0000 | 0.141 | 396.68 | 889.59 | 112.70 | **+199.39** |

**This is the weakest part of the baseline and it is reported as it stands: two
of the four units do not move at all.** Every bus has a positive peak-to-valley
*ratio* margin (1.73–2.51 against a 1.56 breakeven), but the clearing costs
charging at the unit's declared bid and discharging at its declared offer, so
what a unit must clear is an absolute number:

```
eta_dis * lmp_peak - lmp_valley / eta_ch  >  bid + offer + 2 * cycle_cost
```

| unit | valley | peak | round trip value | required | shortfall |
|---|---|---|---|---|---|
| ESS6  |  523.7 |  905.8 | 264.0 | 300.0 | **-36.0** |
| ESS20 |  516.8 | 1296.3 | 630.9 | 300.0 | +330.9 |
| ESS24 |  550.7 |  959.7 | 284.4 | 300.0 | **-15.6** |
| ESS31 |  567.1 |  978.8 | 284.1 | 300.0 | **-15.9** |

Measured on the system valley and peak periods; a unit may find a better pair at
its own bus, which is how ESS24 takes a marginal 0.019 MWh. ESS20 sits on a bus
whose price swings far wider than the rest, so it carries the fleet.

**Two things about the quotes are worth stating plainly, because both were
choices.** The anchor is `cycle_cost / 2 = 50`, so the day's `required` figure is
100 from the quotes on top of the 200 the round trip costs. And the action is
multiplicative: a unit declares `bid = anchor * bid_mult` with
`bid_mult in [0.3, 1.8]`, so the *truthful* declaration — a reservation of zero,
pure degradation-cost arbitrage — is **outside the action space**; the lowest
reachable charge bid is 15. If a livelier day is wanted, the lever is this
parameterisation, not the prices. Raising prices until three units clear would
be exactly the tuning this baseline exists to avoid.

**Network** (from the solver's own voltage and current variables)

| quantity | value |
|---|---|
| max line utilization | 0.543 (line 18→19) |
| lines over 0.85 / 0.95 / limit | 0 / 0 / 0 |
| bus voltage | min 0.930 pu, max 1.008 pu |
| most utilized lines | 18→19 (0.543), 1→2 (0.489), 22→23 (0.392) |

Voltage is the binding constraint on every run of this day; the lines are not
close to their thermal limit at any inverter rating tested, including 2.5.

**Funds flow** (`surplus_metrics.reconciliation`): merchandising surplus (rent)
17750 CNY, wholesale bill 91562 CNY, consumer bill 127122 CNY, identity residual
exactly 0, smallest per-period rent 31.34 CNY.

**Gates: 11 of 11 pass.** Nodal prices real · every reported quantity finite ·
state of charge within bounds · no simultaneous import and export ·
charge/discharge exclusive · the fleet has a viable arbitrage opportunity · the
day closes at its starting state of charge · network within limits · coincident
peak on target · installed renewables on target · battery buses distinct. Shed
load is gated at the accepted 1.1 MWh (`--unserved-limit` overrides it).

Properties of this particular day — cycles per unit, fleet size relative to
load, LMP ratio, which units earned — are printed as **diagnostics** and are not
failed on. Pinning them would turn one realization into a contract.

## 7. What is not physically clean

1. **Some load is shed — 1.044 MWh, 0.58 % of the day.** The feeder is
   voltage-limited; `v_min` sits exactly on 0.930 pu in every run. Serving 10 MW
   coincident and shedding nothing are not jointly achievable at a physical
   inverter rating. Zero shedding needs either a 2.50 rating or a coincident
   peak of about 6.1 MW. Both were rejected in favour of accepting roughly one
   megawatt-hour at a 1.10 rating, and the gate is armed at what was accepted.

2. **0.947 MWh of renewable output is spent on voltage support rather than
   energy**, down from 1.64 MWh at parity. This is the mirror image of the
   previous item and is not economic curtailment.

3. **Two of four batteries never discharge.** See section 6 for the arithmetic.
   The opportunity exists at their buses by the ratio measure but not by the
   absolute one the clearing actually applies.

4. **The arbitrage window lands near the round-trip breakeven.** With
   degradation at 100 CNY/MWh per leg and a 0.92/0.92 round trip, ESS6's bus
   delivers 264 CNY/MWh for a full cycle against 200 CNY/MWh of degradation and
   100 CNY/MWh of declared quotes. `cycle_cost` and the anchor are the only
   effective levers and both were held.

5. **Nodal prices at the deep main feeder reach ~2743 CNY/MWh**, roughly 3×
   the slack at the same period, against a wholesale curve that tops out near
   1280. An earlier note attributed this to a per-unit scaling error. That is
   wrong, and the recomputation says what it actually is.

   The per-unit system was checked and is internally consistent at
   `S_base = 1 MVA` across all three engines; `NetworkConfig.base_mva` is never
   read by any solver, which makes it a dead config knob rather than a wrong
   answer. Independently, the marginal loss factor at a bus is
   `2 * sum over the path from the slack of (r_pu * P_pu)`, computable from the
   solved line flows and the line resistances alone. Measured against the
   cleared prices at bus 17:

   | period | load | min |V| | buses at the floor | loss factor | LMP ratio | excess |
   |---|---|---|---|---|---|---|
   | 47 (valley) | 6.82 MW | 0.9446 | 0 | 0.0806 | 0.0916 | +0.011 |
   | 20 | 6.67 MW | 0.9300 | 1 | 0.1154 | 0.1620 | +0.047 |
   | 76 (peak) | 10.00 MW | 0.9300 | 1 | 0.1411 | 2.0451 | **+1.904** |
   | 73 | 9.39 MW | 0.9300 | 3 | 0.1366 | 2.3957 | **+2.259** |

   At the valley, where nothing is at the voltage floor, the loss factor
   accounts for the whole nodal spread (0.011 of disagreement on a small
   number). At the peak, where buses sit exactly on 0.930 pu, it accounts for
   about a fourteenth of it. The rest is the **shadow price of the voltage
   constraint**: the marginal cost of serving one more MW at a bus whose
   voltage cannot be held any lower includes what it costs to keep the feeder
   in band, and that is what the price is charging. So the deep-feeder premium
   is voltage congestion, not mis-scaled losses — and it is the same mechanism
   as the shed load in item 1, seen from the price side.

6. **`config/defaults.yaml` still documents parameters that nothing reads.**
   The `storage:`, `market_design:` and `rt:` blocks are inert; the effective
   values are the dataclass defaults in `models.py`. `network.v_min_pu` is the
   same. `network.inverter_smax_multiplier` is set in code for the same reason.

## 8. What is not clean in the code

1. **`churn_free_quotes` is retired and inert.** It capped a storage unit's
   declared charge bid at its declared discharge offer. Charge and discharge are
   now the parts of one net power flow, so overlap is not representable and no
   quote needs capping. The flag is kept because callers and tests still pass
   it; it changes no dispatch.

2. **The sign of the charge leg was flipped.** Charging is now *costed* at the
   declared bid rather than *rewarded* at it, which is what makes the objective
   concave for any pair of quotes and lets the quotes cross. The consequence for
   anything trained later: a higher `bid_mult` makes a unit **less** willing to
   charge, which is the opposite of what the same quote meant before. No policy
   on disk is affected because none exists for this fleet.

3. **The SOCP clearing needed `BarHomogeneous=1`.** Every price is a
   power-balance dual, and on this model the plain barrier returned OPTIMAL while
   leaving those duals unavailable on some price realizations — two of four
   fixed seeds returned none at all. `_extract_lmp` then falls back to the
   wholesale curve for *every* bus, which looks like a flat nodal price field
   rather than like a failure. The homogeneous self-dual variant returns every
   dual across those seeds for about 0.2 s per clear. This was latent before the
   price curve was seeded; the seed made it reproducible rather than causing it.

4. **LinDistFlow and DC still value prosumer batteries at the wholesale price**
   rather than at declared quotes; only the independent fleet uses the declared
   quote in those engines. The typical day has no prosumer batteries, so this
   does not touch the numbers here, but the engines do not answer the same
   question for that class of participant.

5. **`pseudo_realtime.py` is broken** — it calls `solve_opf_gurobi` with nine
   positional arguments against a signature of eight, so it raises `TypeError`
   today. It is outside this baseline's path.

## 9. What this changes for the research

- **The storage fleet is four units, not fifteen.** The RL agent set is
  `ESS6, ESS20, ESS24, ESS31`; the network carries no other battery.
- **Every checkpoint under `policies/` (39 directories) and every
  `results/*_eval.csv` refer to the retired twelve-unit on-network fleet and
  share no agent with this scenario.** They must be retrained. There is
  currently no `ESS*` policy anywhere on disk.
- **The day is reproducible.** `price_curve.seed: 20260915`; the same
  configuration now clears to the same curve, and the diagnostic writes the same
  JSON byte for byte on repeated runs.
- **The biddable object is a pair of reservation prices**, with the truthful
  point currently outside the reachable set; see section 6.

## 10. Files

| what | where |
|---|---|
| the day's configuration | `config/defaults.yaml` (`profiles.load.target_peak_mw`, `price_curve.*`, `prosumers.*`, `independent_storage`, `network.base_ampacity_ka`) |
| canonical entry point | `scenarios.typical_day_config()` — pins the engine, the horizon, the terminal SOC, the inverter rating |
| load level and device switches | `src/grid.py` (`create_agents_from_network`, `_coincident_peak`) |
| independent fleet | `src/ess.py` (`build_ess_fleet`) |
| storage quotes and the net power flow | `src/dispatch_core.py` (`add_storage_net_power`, `check_storage_quote_signs`, `StorageConfig.quote_anchor` in `src/models.py`) |
| grid exchange and carbon | `src/dispatch_core.py` (`add_grid_exchange`, `grid_import_export`, `grid_carbon_tco2`) |
| horizon semantics | `src/dispatch_core.py` (`pins_terminal_soc`) |
| diagnostic and gates | `src/diagnose_typical_day.py` |
| physical contract | `tests/test_typical_day_physical_baseline.py` |
| storage mechanism contract | `tests/test_storage_exclusivity.py` |
