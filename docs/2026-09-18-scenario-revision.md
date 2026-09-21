# Scenario revision: the IEEE33 native scale, and who declares what

The research scenario moved from a 10 MW study scale back to the IEEE 33-bus's
own nominal demand, with renewable capacities that stand on their own rather
than being derived from the load. The larger outcome is a config refactor: four
quantities that used to be one number in `defaults.yaml` are now separately
declared and separately traceable.

## Why the 10 MW scale went

It was not a property of anything. `profiles.load.target_peak_mw: 10.0` sat in
`defaults.yaml` and simultaneously meant the benchmark's physical demand, this
study's operating scale, and the basis the renewables were sized against. No
reader could tell which of the three was in force, and no reviewer could be
told where 10 MW came from.

It also turned out to be unnecessary. The original IEEE 33-bus **already
violates the 0.93 band at its own load**: Vmin 0.9131 pu at bus 17, 21 of 33
buses below 0.95 and 14 below 0.93. The network's voltage pressure is its own;
10 MW amplified it, it did not cause it.

## A / B / C / D

| | meaning | where it lives |
|---|---|---|
| **A** benchmark physical base | IEEE33 native 3.715 MW / 2.3 MVAr, original nodal distribution | `config/defaults.yaml` |
| **B** scenario operating scale | declared per scenario | `config/scenarios.yaml` |
| **C** renewable installed capacity | declared per scenario, per bus, in MW | `config/scenarios.yaml` |
| **D** temporal profile | shape only, never level | `profiles.*` |

`defaults.yaml` no longer contains `target_peak_mw`, `load_multiplier` as a
scale, `pv_capacity_factor`, `wind_capacity_factor`, or any `*_capacity_scale`.
Renewable capacity is an input, not something computed from the local load — a
capacity that moves whenever the load does is a property of the load, not a
choice the study can defend.

## The native day

```
lambda_t         = raw_profile_t / max(raw_profile)      peak-normalized, shared
P_i(t)           = P_i_base * lambda_t                   P_i_base = case33bw p_mw
Q_i(t)           = Q_i_base * lambda_t                   at load_power_factor 0.8503
max_t sum P(t)   = 3.715 MW                              exactly
```

Three things were needed to make that hold bus by bus, not just in aggregate:

- **`load_shape: native`** — one shared shape for every bus. The legacy
  per-load-type phase shifts and amplitude blends give each type its own
  multiplier, so no single `lambda_t` describes the fleet.
- **System-wide noise, not per-bus.** `_noisy_load` gives each bus its own
  realization, which makes every bus's shape its own. Under `native` the error
  factor is drawn once per period and shared, so the nodal *shape* survives
  while forecast and realized day still differ.
- **The realized peak is what is pinned.** `native_peak` normalizes
  `load_real`, because `P` is the load actually drawn; `explicit_peak` keeps
  normalizing the forecast, which is what its recorded numbers were built on.

`load_power_factor: 0.8503` = 3.715/√(3.715² + 2.3²) is the value that makes
the day's reactive load the benchmark's own 2.3 MVAr. It is declared per
scenario, not globally: applying it everywhere would silently move every legacy
scenario's reactive load.

Measured for `typical_day`: real peak **3.715 MW**, forecast peak 3.485 MW,
daily energy **47.04 MWh**, load factor 0.57, PV 0.6 MW, wind 1.0 MW.

## Renewable capacity and placement

PV 0.3 MW at bus 21 + 0.3 MW at bus 23; wind 1.0 MW at bus 30.

These are a **deliberate scenario choice informed by** the literature, not a
benchmark value. An existing modified IEEE33 source-load-storage study in the
supplied literature uses 1 MW wind and 0.6/0.3 MW PV; the Enhanced IEEE33
literature supports integrating renewable DG and recommends voltage/loss
sensitivity for placement and capacity. This document does not claim 0.6/1.0 MW
are standard IEEE33 figures, because they are not a standard anything.

The placement rule is topological and pre-declared: injection goes on load buses
of the lateral spurs rather than on the main trunk, and never on a bus carrying
an independent battery. Buses 21 and 23 sit on the residential and industrial
laterals and bus 30 at the end of the second-longest spur — which is where
placement by voltage sensitivity puts them, since the laterals are the buses
whose voltage responds most to a local injection. The rule was fixed before the
ladder ran and the buses are frozen.

## The legacy scenarios

`baseline`, `high_re`, `peak_load`, `congestion`, `no_congestion`,
`tight_bottleneck` and the two `re_ramp_*` entries are unchanged at their own
10 MW scale, tagged `scenario_scale_source: legacy_explicit` so their scale
cannot be mistaken for the benchmark's.

Verified rather than assumed: **56 load and shape fields across all eight
scenarios are bit-identical** before and after the revision — peak, mean, daily
energy, forecast peak, agent counts, ESS counts and power factor. The only
change anywhere is the *declared* renewable capacity, which moved from the
derived `3.499999734` / `1.5000002325` to exactly `3.5` / `1.5`: the old value
was float rounding from `base_load × factor × scale`, and declaring the
capacity makes it exact.

## The band

Primary `(0.93, 1.05)`, strict `(0.95, 1.05)`. They differ in the lower bound
alone, which is the only one this network binds on; the 1.07 the primary band
used to carry separated the two bands on a constraint that never fires.

## Verification

```bash
PYTHONPATH=src python src/baseline_ladder.py --rung all --bands both --json outputs/ladder.json
PYTHONPATH=src python src/baseline_ladder.py --inventory
python -m pytest tests/ -q
```
