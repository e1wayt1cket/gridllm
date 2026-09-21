# The B0-B3 physical baseline ladder

Four rungs. Each adds one thing to the one before it, on a fixed spatial load
structure that is frozen for this round. Entry point `src/baseline_ladder.py`;
every rung is written as one JSON record and reproduces byte for byte at the
same configuration, seed and scenario.

```bash
PYTHONPATH=src python src/baseline_ladder.py --rung all --bands both --json outputs/ladder.json
```

## B0 is a benchmark reference; B1-B3 are the project scenario

This is the one distinction that must not be lost. **B0 is not a baseline for
B1.** They are different networks at different loads, and the step between them
is a change of question, not a change of parameter.

| | B0 | B1 |
|---|---|---|
| what it is | the published IEEE 33-bus | this study's operating scenario |
| load | **3.715 MW / 2.3 MVAr**, the native per-bus distribution | **10 MW coincident peak, 179 MWh/day** |
| time | one snapshot, no horizon | 96 periods at 15 minutes |
| market | none | day-ahead clearing at the SOCP engine |
| shedding | no mechanism exists | 0.42 MWh at the primary band |
| answers | what does the untuned network do at its own load? | what does this study's facility set do on a real day? |

The rungs are therefore **not required to share a load level**, and B0's
`energy_load` is `null` with `metric_not_applicable: true` — a snapshot has no
horizon, and reporting 0.0 MWh would read as a day that consumed nothing.

B0's `shed` is reported as `0.0` with `shed_mechanism: "absent"`. It is not
evidence the network is feasible: B0 cannot shed, so it cannot demonstrate
anything by not shedding.

## The rungs

| rung | built from | has DER | has ESS |
|---|---|---|---|
| B0 | `pandapower.networks.case33bw()` + `runpp` | no | no |
| B1 | `typical_day` with the DER zeroed | no | no |
| B2 | `typical_day` | PV 25.4 MWh + wind 19.5 MWh | no |
| B3 | `typical_day` | yes | `[2, 12, 13, 19]`, rule policy |

No new physics was added for any rung. B1's absent DER is the scenario's own
fleet zeroed on the agents, and B2's absent battery is `ess_buses=[]`, which
`ess.build_ess_fleet` already turns into an empty fleet.

## The two voltage bands

```python
PRIMARY_BAND = (0.93, 1.07)   # operational; GB/T 12325 +/-7%
STRICT_BAND  = (0.95, 1.05)   # sensitivity; the enhanced-IEEE33 convention
```

Every rung runs at both. **Neither band is reached by tuning**, and no rung's
scenario parameters differ between them — the difference is the band's own.

The reason this has to be stated so plainly is measured:

- The original IEEE 33-bus at its **own** nominal load has Vmin `0.9131` pu at
  bus 17 — **21 of 33 buses below 0.95 and 14 below 0.93**. B0 therefore
  violates both bands without anything being wrong with it.
- At this study's 10 MW level the unconstrained minimum is `0.7209` pu.

So a voltage band is a *constraint the market clears against*, not a property
of the network. Enhanced-IEEE33 papers that use 0.95-1.05 pair it with DG,
reactive compensation, reconfigurable lines and time-series DER; taking the
band alone and calling the bare network "enhanced" is not a reading this
project supports.

The project's operational band stays at 0.93. `defaults.yaml` keeps
`v_max_pu: 1.05`; the primary band's `1.07` is supplied by the ladder as an
explicit run parameter, so the project's default model is unchanged. Measured,
`v_max` is only *weakly* binding: raising it 1.05 → 1.07 moves the objective by
2.9e-8 relative and the LMPs by 3.6e-4, while repeated runs at one setting are
bit-identical. It is not the constraint that matters — `v_min` is, and it pins
exactly on the band in every market rung.

## What the ladder measures

At seed 123, primary band:

| rung | Vmin | shed MWh | curtail MWh | fleet Pi |
|---|---|---|---|---|
| B0 | 0.9131 | — (absent) | — | — |
| B1 | 0.9300 | 32.94 | 0.00 | — |
| B2 | 0.9300 | 2.37 | 1.64 | — |
| B3 | 0.9300 | 0.42 | 1.64 | 613.29 |

At the strict band, shed rises at every market rung: B1 32.94 → 54.62, B2 2.37
→ 10.24, B3 0.42 → 4.62 MWh. That cost is the band's own, and it is reported
rather than optimised away.

Two things follow that are worth carrying forward:

- **Renewables and storage are principally voltage support here.** Adding the
  DER cuts shed by 14× (32.94 → 2.37 MWh) and adding the battery cuts it again
  by 5.7× (2.37 → 0.42). `v_min` pins on the band in every market rung, so
  voltage is the only binding constraint at any of them.
- **B3's rule policy loses money against truthful** (`pi_ai` 613.29 vs
  `pi_baseline` 664.60, `delta_pi` −51.32). That is a legitimate result, not a
  defect: the deterministic rule is a *baseline* the AI policy has to beat, and
  on this day doing nothing special pays better. It is the reason the research
  question is worth asking.

## Shed is a voltage result

Two diagnostics separate the possible causes of shed. Both are diagnostic only
and neither replaces the economic market run.

```bash
PYTHONPATH=src python src/diagnose_typical_day.py --min-shed
PYTHONPATH=src python src/diagnose_typical_day.py --no-shed-voltage
```

**`--min-shed` (MIN_SHED_RUN)** clears the same day with total shed ahead of
every economic term (`penalty_unserved` at 1e7, well above every price and bid
in the study) and every constraint untouched. It answers how much of the shed
the *market declined* to serve:

| | MWh |
|---|---|
| economic clearing | 0.1709 |
| minimum under identical physics | 0.1286 |
| **declined by the market** | **0.042** (0.024% of a 179 MWh day) |

**`--no-shed-voltage` (NO_SHED_VOLTAGE_DIAGNOSTIC)** forces every unit served
and every renewable at its available output, then runs a power flow per period
with no market at all. The result is unambiguous:

- **all 96 periods** violate the voltage band
- **1968 bus-periods** below the bound; worst sag `0.7209` pu at bus 17
- **zero periods** overload a line; the worst line loading is 0.914
- `binding_constraint: "voltage"`

So shed is a voltage-constraint phenomenon. It is not an energy shortage, not a
line-capacity limit, and not an artifact of the clearing's objective.

## Reproducibility

`build_ladder` writes JSON with sorted keys and a fixed newline, so the same
configuration, seed and scenario produce a byte-identical file. Verified across
all eight runs (four rungs × two bands) with `cmp`. Repeated runs at one
configuration are also bit-identical at the solver level, so the ladder's
determinism is not merely the serializer's.

## Not done in this round

- **No α scenario family was added.** The pre-stated trigger was "only if B0-B3
  all look too healthy"; the measured state is the opposite, so the spatial
  load structure stays frozen and `congestion`'s weights are untouched. Nothing
  may be retuned toward `v_min = 0.95`.
- **B3 runs one deterministic policy** (`rule`, from `baselines.py`). No RL is
  involved in this ladder, and no MATD3 training was started.
