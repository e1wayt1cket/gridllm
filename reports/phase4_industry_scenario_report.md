# Phase 4 — IEEE 33-bus Industry-Shape Scenario

Delivers the `typical_day_industry` scenario: the IEEE 33-bus at its own
declared scale, with each bus's **intraday load shape** taken from the metered
Jiangsu industry curves instead of from the synthesised profile.

No revenue-accounting repair, no market-rule change, no MATD3 training. Phase 5
is untouched.

## 1. Files

**Added**

| file | what |
|---|---|
| `tests/test_industry_scenario.py` | 26 tests |
| `reports/phase4_industry_scenario_report.md` | this report |

**Modified** — five files, all additively except where noted

| file | change |
|---|---|
| `src/grid.py` | `create_agents_from_network` gains an optional `load_shapes=None`; `_make_load_profile` gains `bus_shape=None` and the `industry` branch; two conditions widened from `== "native"` to `in ("native", "industry")` |
| `src/scenarios.py` | new `_industry_load_shapes`; `_build_scenario` gains `day_id`; the price call now takes the declared curve index. **The user's uncommitted deletions in this file are preserved** (verified, §6) |
| `config/scenarios.yaml` | the `typical_day_industry` scenario, including the explicit `industry_map` |
| `src/industry_load.py` | new `load_shapes_csv`, reading the shapes artifact back |
| `tests/test_scenario_scale.py` | two policy assertions widened to admit `industry` and the second native-scale scenario |
| `.gitignore` | one exception, `!data/processed/industry_typical_profiles.csv` — see §5 |

## 2. The industry map

An **explicit** bus → industry assignment, written into
`config/scenarios.yaml` and marked `simulation_mapping: true`.

**This is a simulation rule, not a geographic one.** The feeder carries no
user-to-node connection table, and nothing here is a claim about where Jiangsu
industry is actually connected. The rule is stated in the file so it can be
checked and replaced:

- every **residential and commercial** bus takes 商业综合体 — the workbook has
  no residential or general-retail class, and the commercial-complex curve
  (night trough, evening peak) is the closest available shape;
- each **industrial** bus takes one distinct industrial industry, so the
  feeder's industrial load is not one curve wearing twelve names.

Bus 0 is absent: it is the slack and carries no load, though `load_types` names
it. Load buses are 1–32.

| industry | buses | peak MW | % peak | energy MWh | % energy |
|---|---|---|---|---|---|
| 商业综合体 | 1–15, 17–21 (20) | 1.8144 | 47.9 % | 22.03 | 40.2 % |
| 汽车制造业 | 24 | 0.4575 | 12.1 % | 7.82 | 14.3 % |
| 化工业 | 23 | 0.4245 | 11.2 % | 7.54 | 13.8 % |
| 印刷业 | 29 | 0.2097 | 5.5 % | 3.46 | 6.3 % |
| 电动汽车（充换电） | 31 | 0.2203 | 5.8 % | 3.09 | 5.6 % |
| 电制氢 | 30 | 0.1432 | 3.8 % | 2.16 | 4.0 % |
| 机械 | 28 | 0.1130 | 3.0 % | 1.81 | 3.3 % |
| 水泥制造 | 22 | 0.0870 | 2.3 % | 1.47 | 2.7 % |
| 数据中心 | 32 | 0.0654 | 1.7 % | 1.16 | 2.1 % |
| 钢铁 | 16 | 0.0649 | 1.7 % | 1.14 | 2.1 % |
| 通用设备制造业 | 25 | 0.0654 | 1.7 % | 1.05 | 1.9 % |
| 电子设备制造业 | 26 | 0.0625 | 1.6 % | 1.06 | 1.9 % |
| 纺织业 | 27 | 0.0612 | 1.6 % | 1.01 | 1.8 % |

13 industries, all of them **reliable** by the Phase-1 threshold. The three
excluded ones are declared and enforced: 分布式光伏 (a generator), 5G基站 and
电解铝 (below the sample threshold). They remain available for exploratory runs
marked limited, and `_industry_load_shapes` **raises** if any of them is used.

`peak MW` is the sum of each bus's own realized maximum — 3.7890 MW, larger
than the coincident peak, because the buses do not peak together. It is
reported as a diagnostic, not as the day's load.

## 3. Load scale

The scenario changes **the intraday shape only**. The level derivation is
unchanged and is the reason the two research days are comparable:

```
target_peak_mw = native_nominal_demand_mw(net)            # 3.715 MW, read off the network
raw_peak       = max_t sum_i P_i_real(t)                  # unscaled, realized
load_scale     = target_peak_mw / raw_peak
P_i(t)         = shape_i(t) * P_i_base * common_noise(t) * load_scale
```

Three properties make this safe, each pinned by a test:

1. **Shapes are normalised to peak 1.0** before use, so the artifact supplies
   form and never level. `test_the_shape_does_not_carry_its_own_magnitude`
   multiplies every shape by 137 and asserts the day is unchanged.
2. **The scale is derived, not declared** — it is recomputed from the day's own
   unscaled peak, so it survives a change of shapes.
3. **The peak is pinned to the network's own rating**:
   `test_the_declared_scale_reaches_the_agents` asserts the realized coincident
   peak equals `native_nominal_demand_mw(net)` to 1e-6, and 3.715 MW exactly.

Measured, with `typical_day` alongside as the reference:

| | peak MW | energy MWh | load factor | peak slot |
|---|---|---|---|---|
| `typical_day` | 3.7150 | 47.04 | 0.5276 | 74 (18:30) |
| `typical_day_industry` | 3.7150 | **54.79** | **0.6145** | 65 (16:15) |

**The industry day carries more energy at the same peak, not less** — the
opposite of what the plan anticipated. The reason is worth stating: the
synthesised `native` shape has a deep night trough (`night_base: 0.60`), while
several industry curves are near-flat (钢铁 0.92–1.00, 数据中心 0.95–1.00), so
mixing them in raises the fleet's load factor and therefore the daily energy at
a fixed peak. The peak also moves earlier, from 18:30 to 16:15.

This is a measured property of the two shape sets, not a defect. It is recorded
here rather than smoothed over, and the direction is stated because the plan
predicted the other one.

## 4. What the scenario does *not* claim

Restating the boundaries, because the scenario is easy to over-describe:

- **Not measured Jiangsu load.** The workbook carries normalised shapes only;
  no absolute-MW conversion has been verified (`综合倍率` reconciles with the
  declared rating for 87 of 200 checkable meters and contradicts it for 113).
  The day is *IEEE 33-bus demand redistributed by Jiangsu industry shapes*.
- **Not a geographic mapping.** `industry_map` is a simulation rule.
- **Not a confirmed real daily price.** `wholesale_curve_index: 0` selects the
  `261009` curve. All four derive from real historical market data, but none is
  confirmed to be a single day's series: the three trailing sheets are means,
  and `261009`'s aggregation is unresolved. `describe()` reports
  `is_measured: false` throughout and the reader carries the raw header.
- **Not a facility change.** PV 0.6 MW and wind 1.0 MW are unchanged, the fleet
  is still `ESS9/14/17/27`, and the workbook's 分布式光伏 is **excluded by
  name** so its generation cannot re-enter as a device.

## 5. Price selection and the `day_id` hazard

`wholesale_curve_index` is an index into `wholesale_market.dianyicha_curves` — a
**price scenario**, not a date. The scenario declares `0` (`261009`).

It is passed to `day_ahead_price_china(day_id=...)` rather than written into the
config cache, and that was deliberate. `day_id` is a *sticky caller knob* by
existing design — `get_scenario(day_id=...)` writes it, and `train_rl`'s day
rotation depends on that. A scenario writing it directly would silently reprice
every scenario built afterwards **in the same process**, including `typical_day`.
Passing it keeps the caller's explicit `day_id` winning and leaves the global
untouched. `test_other_shapes_ignore_load_shapes_entirely` and the legacy-scale
test guard the neighbouring property.

### The `.gitignore` exception

`data/processed/*` is ignored (matching `*.csv`). The scenario **cannot build
without** `industry_typical_profiles.csv`, and `baseline_ladder.scenario_inventory()`
builds every scenario in the list — a test asserts no row carries an error. Left
ignored, the new scenario would fail that test on any fresh clone.

So exactly one file is now excepted:

```
!data/processed/industry_typical_profiles.csv
```

The 71 MB `industry_daily_profiles.parquet`, the exclusions CSV and the figure
remain ignored. This is the first change to `.gitignore` in this project's
finalization work; it is flagged here rather than made quietly.

## 6. Tests

**New:** `tests/test_industry_scenario.py`, 26 tests.

```
python -m pytest tests/test_industry_scenario.py -v   -> 26 passed
```

Covered: the scenario declares its own scale and a mapping marked as a
simulation mapping; every load bus assigned exactly once; unassigned bus,
non-load bus, excluded industry and unknown industry each raise; a missing
artifact names the command that builds it; the declared scale reaches the
agents and equals the network's own demand; both research days peak at the same
number; a shape scaled by 137 leaves the day unchanged; every bus carries its
assigned industry's shape; every profile is 96 finite non-negative slots; buses
sharing an industry carry the same shape and different industries do not
collapse; renewables unchanged at 0.6/1.0; the fleet unchanged; the declared
curve prices the day and is not measured; other shapes ignore `load_shapes`
entirely; an unknown shape name still raises; `typical_day` still carries one
shared shape; legacy scenarios keep their own scale.

The per-bus shape check runs through `_industry_load_shapes` (un-noised), because
the realized profiles carry one shared multiplicative error factor per period
that does not cancel under max-normalisation. The realized day is checked
through **proportionality**, which that shared factor does preserve exactly.

**Regression:** `python -m pytest tests/` -> **666 passed, 51 deselected, 0
failed** (134 s). 640 previously, plus the 26 new.

**All three defects found were in my own test or wiring, and each is fixed:**

1. `Dict` was missing from `scenarios.py`'s typing import — a runtime
   `NameError` the syntax check could not see.
2. The first mapping draft assigned **bus 0**, which is the slack and carries no
   load. The validation caught it immediately with a named bus.
3. Two tests of mine used `read_curves()[0]` — that dict is keyed by sheet
   *name*, not index — and one depended on process-global state another test
   resets.

## 7. Open items

1. **The industry day is not comparable to the synthetic-price days.** It is
   priced on a real Jiangsu mean curve, so any cross-scenario revenue
   comparison must state that the price differs as well as the load shape.
2. **The mapping is one choice among many.** 20 of 32 buses share one curve, so
   the day's overall shape is dominated by 商业综合体. A different defensible
   assignment would give a different day. The mapping is declared, not derived,
   and no sensitivity has been run on it — a candidate for Phase 9.
3. **The peak moved 18:30 → 16:15 and the energy rose 16 %.** Both follow from
   the shape sets. Neither is a target, and neither should be tuned.
4. **No clearing has been run on this scenario yet.** Phase 4 only builds the
   day; whether it clears inside the voltage band, and whether the fleet finds
   arbitrage on the narrower real price curve, is Phase 6/7 work.
5. **`load_shapes` is accepted and ignored by non-industry shapes**, which the
   test pins. It is not an error, but it means a mis-declared `load_shape` would
   silently ignore a supplied mapping rather than complain.

## 8. Difference from the pre-Phase-4 workspace

Backed up to `/tmp/pre_phase4_worktree.diff` before any edit. HEAD `a1151fd`,
branch `master`, unchanged; no destructive git command was run.

`src/scenarios.py` was **already modified** by the user before this phase, so a
whole-file diff cannot prove nothing was lost. Instead every pre-existing hunk
was compared line by line:

```
src/scenarios.py        removed 10 -> 15   added   0 -> 101   OK
src/config_loader.py    removed 14 -> 14   added   0 ->   0   OK
src/dispatch_core.py    removed 17 -> 17   added   0 ->   0   OK
src/mpc_storage.py      removed 229 -> 229 added   0 ->   0   OK
src/price_forecaster.py removed 98 -> 98   added   7 ->   7   OK
src/rl_td3.py           removed  5 ->  5   added   0 ->   0   OK
src/llm.py              removed 23 -> 23   added   1 ->   1   OK
src/data_aggregator.py  removed  5 ->  5   added   1 ->   1   OK
src/eval_agents.py      removed  1 ->  1   added   1 ->   1   OK
src/rl_bidding.py       removed  3 ->  3   added   0 ->   0   OK
src/rl_env.py           removed  4 ->  4   added   0 ->   0   OK
src/diagnose_typical_day.py removed 0 -> 0 added   1 ->   1   OK
src/strategies/rl_bidding.py removed 4 -> 4 added  0 ->   0   OK
```

Every line the user had deleted is still deleted, and every line they had added
is still present. `scenarios.py`'s five extra removals are lines **I** replaced
while editing the price call. `TYPICAL_DAY_PHYSICAL_BASELINE.md` was not touched
(it was outside the backup's path scope; it appears unchanged in `git diff`).
`gurobi.env` was not read, copied or listed.

## Status

Phase 4 complete. Stopping here for review; Phase 5 is not started.
