# The B0-B3 physical baseline ladder

Four rungs. Each adds one thing to the one before it, on a fixed spatial load
structure that is frozen for this round. Entry point `src/baseline_ladder.py`;
every rung is written as one JSON record and reproduces byte for byte at the
same configuration, seed and scenario.

```bash
PYTHONPATH=src python src/baseline_ladder.py --rung all --bands both --json outputs/ladder.json
```

Every number below is a measurement at seed 123 on the configuration in
`config/`, not a target. The record is regenerated from the code, so a number
here that the code no longer produces is a documentation defect and not a
result.

## B0 is a benchmark reference; B1-B3 are the project scenario

This is the one distinction that must not be lost. **B0 is not a baseline for
B1.** They are different networks at different loads, and the step between them
is a change of question, not a change of parameter.

| | B0 | B1-B3 |
|---|---|---|
| what it is | the published IEEE 33-bus | this study's operating scenario |
| load | **3.715 MW / 2.3 MVAr**, the native per-bus distribution | `typical_day`'s own declared scale |
| time | one snapshot, no horizon | 96 periods at 15 minutes |
| market | none | day-ahead clearing at the SOCP engine |
| shedding | no mechanism exists | reported per rung |

The rungs are therefore **not required to share a load level**, and B0's
`energy_load` is `null` with `metric_not_applicable: true` — a snapshot has no
horizon, and reporting 0.0 MWh would read as a day that consumed nothing.

B0's `shed` is reported as `0.0` with `shed_mechanism: "absent"`. It is not
evidence the network is feasible: B0 cannot shed, so it cannot demonstrate
anything by not shedding.

### How stressed B1-B3 are is a property of the scenario's scale

`typical_day` declares `load_scale_mode: native_peak` and `load_shape: native`,
so one shared peak-normalized `lambda_t` gives `P_i(t) = P_i_base * lambda_t`
and the day's real coincident peak is the network's own nominal demand. Its
renewables are declared per bus in MW and its fleet is sized against that
scale.

Every stress quantity the ladder reports — how deep the voltage sags, how much
load is shed, how far the clearing pushes the band — follows from that
declaration and moves with it. A scenario that declares a larger peak is a
harder day and sheds more; nothing about the mechanism changes. Reading the
ladder's absolute numbers as properties of the *network* rather than of the
scenario that was declared for it is the mistake this section exists to
prevent.

## The rungs

| rung | built from | has DER | has ESS |
|---|---|---|---|
| B0 | `pandapower.networks.case33bw()` + `runpp` | no | no |
| B1 | `typical_day` with the DER zeroed | no | no |
| B2 | `typical_day` | PV + wind | no |
| B3 | `typical_day` | yes | the configured fleet, rule policy |

No new physics was added for any rung. B1's absent DER is the scenario's own
fleet zeroed on the agents, and B2's absent battery is `ess_buses=[]`, which
`ess.build_ess_fleet` already turns into an empty fleet.

## The two voltage bands

```python
PRIMARY_BAND = (0.93, 1.05)   # operational; GB/T 12325, the lower bound is the binding one
STRICT_BAND  = (0.95, 1.05)   # sensitivity
```

Every rung runs at both. **Neither band is reached by tuning**, and no rung's
scenario parameters differ between them — the difference is the band's own.

A band is a constraint the market clears against. The voltage variables are
created with hard bounds and **carry no slack**, so a cleared day cannot sit
outside its band at any bus or period; the only slack variables in the model
are the carbon and renewable-share ones. That is the invariant worth gating.

**`voltage_binding` is the clearing's own active-constraint report.** It is
true when the cleared extremum sits numerically on a band edge
(`|vmin - lo| <= 1e-6` or `|vmax - hi| <= 1e-6`), which is how a solver's active
set is read off a solution. Two consequences follow, and both matter:

- It holds only for a rung stressed enough to *need* the bound. A rung that
  meets its band with room to spare is not binding, and reporting it as such
  would be wrong.
- It is not the forced-service diagnostic's answer. `--no-shed-voltage` asks
  which limit breaks first if nothing may be shed; `voltage_binding` asks
  whether this clearing pushed against a bound. They are different questions
  asked of different objects, and a day can have one without the other.

## What the ladder measures

At seed 123, current configuration. `Vmin` is the cleared minimum over all
buses and periods; `binding` is `voltage_binding`; `util` is the worst line
utilization over the day.

| rung | band | Vmin | binding | shed MWh | curtailment MWh | util |
|---|---|---|---|---|---|---|
| B0 | both | 0.913090 | False | 0.00e+00 | — (absent) | 0.5256 |
| B1 | primary | 0.945946 | False | 3.18e-07 | 0.0000 | 0.2325 |
| B1 | strict | 0.950000 | **True** | 3.17e-02 | 0.0000 | 0.2243 |
| B2 | primary | 0.954728 | False | 4.53e-08 | 13.3808 | 0.1772 |
| B2 | strict | 0.954728 | False | 9.76e-08 | 13.3810 | 0.1772 |
| B3 | primary | 0.930000 | **True** | 1.13e-07 | 13.2813 | 0.2731 |
| B3 | strict | 0.950000 | **True** | 4.22e-08 | 13.2779 | 0.2788 |

These numbers were re-measured after the rung configuration was routed through
`scenarios.scenario_model_config`. Every row moved, and one moved
qualitatively: **B3 now binds its voltage band at both bands** where it
previously bound at neither, and its worst line utilization rose from 0.2056 to
0.2731 (primary). The cause is that `_config` had been assembled from the
dataclass defaults rather than from `config/defaults.yaml` and the scenario's
own overrides, so a rung cleared at a load power factor of 0.90 against the
scenario's 0.8503 and with inverters rated at 1.0 against the scenario's 1.10.
Both are choices about this study's facilities; a rung that did not carry them
was not a rung of the scenario it claimed to be. See
`reports/s7_defect_repairs.md`.

Earlier runs of the same ladder are still on disk under `outputs/ladder.json`,
`outputs/ladder_2.json`, `outputs/audit_ladder_current.json`,
`outputs/audit_ladder_doc.json` and `logs/audit_ladder_current.log`. They record
what those runs produced against the pre-fix configuration and were left
untouched; the table above is the current one.

Shed below ~1e-5 MWh is the solver's residue on the served/unserved split
rather than a decision to shed; the tests state that line once as
`SHED_TOL_MWH`. The one row that sheds above it is B1 at the strict band.

B3's fleet over the primary-band day:

| | |
|---|---|
| active units | 4 of 4 |
| charge / discharge | 1.4284 / 1.2090 MWh |
| equivalent cycles | 0.8242 |
| fleet profit (rule policy) | 104.12 CNY |
| truthful arm | 205.86 CNY |
| `delta_pi` | **−101.73** (−49.4%) |

**B3's rule policy loses money against truthful.** That is a legitimate result,
not a defect: the deterministic rule is a *baseline* the AI policy has to beat,
and on this day doing nothing special pays better. It is the reason the
research question is worth asking.

At the strict band, only B1's shed moves materially (2.05e-07 → 1.283e-03 MWh).
B2 and B3 hold the tighter bound with their own devices and their differences
sit at the floor. A tighter band is a strictly smaller feasible set, so no rung
can shed *less* under it — that direction is the invariant; the magnitude at
B2 and B3 is not.

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
the *market declined* to serve. On this configuration both arms sit at the
floor:

| | MWh |
|---|---|
| economic clearing | 8.73e-08 |
| minimum under identical physics | 2.00e-10 |
| **declined by the market** | **~0** |

The invariant this establishes is directional: the minimum can never serve less
than the economic run did, because the constraints are identical and only the
objective's ordering changed. Its *magnitude* is zero here because the day is
served in full — see the `stress case` below for where the attribution question
is actually exercised.

**`--no-shed-voltage` (NO_SHED_VOLTAGE_DIAGNOSTIC)** is a different object and
has no clearing in it at all. It forces every unit served and every renewable
at its available output, fixes shed at zero by construction, and runs a power
flow per period. Whichever limit is reached first is what has to give if shed
is to be zero:

| | |
|---|---|
| periods violating the band | **3** of 96 |
| bus-periods below the bound | 11 |
| worst sag | **0.918157** pu at bus 17 |
| periods overloading a line | **0** |
| worst line loading | 0.3099 |
| `binding_constraint` | **voltage** |

So shed is a voltage-constraint phenomenon: not an energy shortage, not a
line-capacity limit, and not an artifact of the clearing's objective. How many
periods violate is a property of the declared scale, which is why it is
reported rather than gated.

**The stress case.** B1 carries neither DER nor a battery, so a bound it does
not naturally satisfy can only be met by dropping load — and a bound met that
way is met exactly. At the strict band that is what happens: B1 sheds
1.283e-03 MWh, two orders above the floor, and its cleared `vmin` pins on
0.950000. That is where the ladder is required to show the voltage constraint
actually binding, since most rungs at the native scale do not need to.

## Reproducibility

`build_ladder` writes JSON with sorted keys and a fixed newline, so the same
configuration, seed and scenario produce a byte-identical file. Verified across
all eight runs (four rungs × two bands) with `cmp`. Repeated runs at one
configuration are also bit-identical at the solver level, so the ladder's
determinism is not merely the serializer's. The profiles carry fixed noise
seeds (`forecast_noise_seed` / `real_noise_seed`), so a repeat is a repeat and
not a redraw.

## Not done in this round

- **No α scenario family was added.** The spatial load structure stays frozen
  and `congestion`'s weights are untouched. Nothing may be retuned toward a
  chosen `v_min`.
- **B3 runs one deterministic policy** (`rule`, from `baselines.py`). No RL is
  involved in this ladder, and no MATD3 training was started.
