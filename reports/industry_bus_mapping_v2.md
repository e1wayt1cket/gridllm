# S2: the user-level bus mapping

Each qualified user lands on exactly one load bus, deterministically, with no
economic quantity anywhere in the decision. This report is the evidence for
that map and for the load it implies on each bus.

| | |
|---|---|
| rule | `U1_user_round_robin` |
| rule version | `77199f7b8244e188` |
| dataset version | `industry-bus-map-v2/a5fae0850b2b` |
| unit | **user** — a user's meters are summed before anything else |
| `simulation_mapping` | true |
| supersedes | `industry_meter_bus_map_frozen.csv`, **D7 `R3_meter_spread`** |

Per-user round robin: within each industry, users are ordered by ascending
anonymous `user_key` and user *i* takes the *i*-th of that industry's declared
buses. The order key reads no name, no location, no date and no magnitude, so
the same records in a different order give the same map — asserted by a test
that shuffles the input.

## Why the unit changed

`R3_meter_spread` mapped **meters**. One user's meters could therefore land on
several buses, and a single account would appear as several separate loads on
several nodes. A user's meters are circuits on one account: they are summed
first and the account is mapped once. This is the substance of the supersession,
not a preference about ordering.

## What the map contains

| | |
|---|---|
| qualified users | 43 |
| **mapped users** | **41** |
| unmapped users | 2 |
| meters involved | 103 |
| qualified blocks | 45,736 |
| declared load buses | 32 |
| buses receiving a user | **17** |
| buses with no mapped user | **15** |

Unmapped users: **2**, both for the same reason — their industry (`5G基站`) is
listed in `excluded_industries` and must not shape a load bus. They are rows in
the map with that reason written down, not omissions.

## Per industry

| industry | users mapped | buses declared |
|---|---|---|
| 商业综合体 | 5 | 20 |
| 纺织业 | 5 | 1 |
| 化工业 | 4 | 1 |
| 汽车制造业 | 4 | 1 |
| 通用设备制造业 | 4 | 1 |
| 水泥制造 | 3 | 1 |
| 电动汽车（充换电） | 3 | 1 |
| 电子设备制造业 | 3 | 1 |
| 印刷业 · 机械 · 电制氢 · 数据中心 · 钢铁 | 2 each | 1 each |

## Per bus: what lands where, and how much

Peak and mean are the absolute industry load in MW, computed with
`reading_to_mw` over the qualified blocks and summed per bus over the 730-day
axis (17,520 hours).

A peak is a **coincidence within one instant**: the largest sum of the bus's
meters at a single (date, period). Summing each meter's own peak instead would
report a moment that never happened; accumulating across dates would report a
week's energy as a power. Both errors are easy to make here and the first draft
of this table made the second — see the correction note below.

| bus | users | meters | industry | mean MW | peak MW | peak vs the feeder's 3.715 MW |
|---|---|---|---|---|---|---|
| 16 | 2 | 7 | 钢铁 | 40.13 | **274.35** | **73.8x** |
| 23 | 4 | 10 | 化工业 | 29.14 | 81.80 | 22.0x |
| 32 | 2 | 6 | 数据中心 | 1.61 | 61.53 | 16.6x |
| 24 | 4 | 5 | 汽车制造业 | 4.61 | 38.19 | 10.3x |
| 22 | 3 | 4 | 水泥制造 | 12.78 | 34.38 | 9.3x |
| 5 | 1 | 6 | 商业综合体 | 4.75 | 12.80 | 3.4x |
| 25 | 4 | 4 | 通用设备制造业 | 7.59 | 12.78 | 3.4x |
| 3 | 1 | 11 | 商业综合体 | 4.48 | 12.21 | 3.3x |
| 2 | 1 | 14 | 商业综合体 | 2.71 | 10.38 | 2.8x |
| 1 | 1 | 2 | 商业综合体 | 0.83 | 10.26 | 2.8x |
| 26 | 3 | 4 | 电子设备制造业 | 2.28 | 9.37 | 2.5x |
| 27 | 5 | 10 | 纺织业 | 1.32 | 6.34 | 1.7x |
| 28 | 2 | 3 | 机械 | 0.44 | 5.05 | 1.4x |
| 31 | 3 | 6 | 电动汽车（充换电） | 0.60 | 3.36 | 0.9x |
| 4 | 1 | 2 | 商业综合体 | 0.55 | 1.58 | 0.4x |
| 29 | 2 | 3 | 印刷业 | 0.07 | 1.54 | 0.4x |
| 30 | 2 | 4 | 电制氢 | 0.05 | 0.26 | 0.1x |
| **system** | **41** | **103** | — | **113.94** | **466.37** | **125.5x** |

> **Correction.** The first version of this table reported bus peaks of 36,973 /
> 22,686 / 16,711 MW and a system peak of 92,023 MW. Those were not peaks: the
> accumulation added every day's power into the same 96 period buckets, so the
> "peak" was a sum over roughly 130 days. The per-meter peaks sum to 291.3 MW on
> bus 16, which is the ceiling a correct coincidence can reach, and the table
> above respects it. The means and the energies were unaffected — they are
> energy over hours and never depended on the bucketing.

## Buses with no mapped user

Fifteen, all of them commercial or residential buses the old meter-level rule
had spread 商业综合体 meters across:

`6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21`

Four of the five commercial-complex **users** land on buses 1-5; the industry
declares twenty buses and has only five users, so fifteen stay empty.

**They are reported, not removed.** Whether such a bus keeps its native IEEE33
load, or carries nothing, is a per-scenario policy decided in S3 — it is not
this module's decision, and "no industry user mapped here" must not be read as
"delete the bus's load".

## The mapper has no objective

`industry_bus_mapping` imports `industry_load`, `config_loader` and
`map_industry_meters` and nothing else. A test reads the module's own import
lines and fails if any of `lmp`, `arbitrage`, `wholesale`, `counterfactual`,
`market`, `rl_`, `surplus` or `profit` appears. A mapping chosen to widen price
spreads or improve storage revenue would be a result tuned into an input, and
the mapping is an input.

It also never consults `UNSUPPORTED_BUSES`, `FROZEN_RULE` or `FABRICATED_DATES`
from `map_industry_meters`: those are the superseded rule's blacklists, and a
bus must be able to receive load again. A test maps a user onto bus 22 to prove
it.

## What this map implies

**The system coincident peak is 466.37 MW against a 3.715 MW network — 125.5x.**
On the busiest bus (16, 钢铁) the coincident peak is 274.35 MW, 73.8x the whole
feeder's demand, and its mean is 40.13 MW, 10.8x.

The mapped users are metered industrial customers far larger than a small
distribution feeder's whole demand — bus 16 alone averages eleven times what the
entire IEEE33 case carries — while four buses (4, 29, 30 and below) sit under
the feeder's own peak. The mismatch is therefore not uniform: it is concentrated
on the industrial buses.

**It is still a structural mismatch, not a rounding.** The kW ruling and the
reading it replaced differ by four; both leave the system peak two orders of
magnitude above the feeder. Per the ruling the mapping is **not** adjusted to
close the gap: no user is dropped, no bus is stretched, no load is rescaled. S3
builds the node ledger from exactly these figures, and S8 reports the resulting
infeasibility with its locations and causes.

### What the capacity field says about the unit

Over the 101 mapped meters with a usable `运行容量`, the ratio of implied peak to
declared capacity is:

| peak / 运行容量 | meters | share |
|---|---|---|
| below 0.05 | 58 | 57.4% |
| 0.05 – 0.20 | 21 | 20.8% |
| 0.20 – 0.50 | 8 | 7.9% |
| 0.50 – 1.00 | 13 | 12.9% |
| above 1.00 | 1 | 1.0% |

`运行容量` spans 160 to 600,000 across these meters. **Fourteen meters behave as
if the field is a power rating in the same unit as `raw x 综合倍率`** — they peak
at 44–90% of it, which is what a real installation does. **Fifty-eight do not**:
their implied peak is under 5% of the declared capacity.

So the field's unit cannot be confirmed from the workbook, and the reading it
agrees with cannot be either. This is the same heterogeneity the unit-gate
report records from the other direction, now located per meter: a minority of
meters are consistent with a single power unit and the majority are not.

## Reproducing it

```bash
PYTHONPATH=src python src/industry_bus_mapping.py --out data/processed
python -m pytest tests/test_industry_bus_mapping.py -v
```

Outputs: `data/processed/industry_meter_bus_map_v2.csv` (one row per user),
`data/processed/industry_meter_bus_map_v2_manifest.json` (workbook digest, rule
version, dataset version derived from the map's own digest, per-bus user counts,
and the `supersedes` list).
