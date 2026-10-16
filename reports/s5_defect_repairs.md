# S5: the A2–A6 repairs

Each item, what it actually was in this repository, what changed, and what
holds it. Two things found along the way are reported rather than fixed, because
they are outside this stage and guessing at them would be worse than saying so.

The task book's `A2–A6` labels do not appear anywhere in the repository; the
mapping from each label to the code below was established by inspection in S0.

## A2 — eligibility decided on physics, not on the quote box

**What it was.** Three sites decided whether a bus could host a battery from
`tradeable_box_share(...) >= 0.25` — a share of the *action box*:

| site | line |
|---|---|
| `market_opportunity.bus_day_metrics` | `siting_eligible` |
| `diagnose_typical_day` per-unit price report | `siting_eligible` |
| `diagnose_typical_day.siting_sweep` | `eligible` |

The share is a property of the box a policy declares in. Deciding siting on it
made a bus's eligibility move when the box moved, which is a statement about the
action encoding rather than about the bus.

**What changed.** Eligibility is now the **physical margin** — the spread over
the degradation floor, which is `margin - required_margin` and has no box in it.
The quantity already existed: `arbitrage_lp`'s `adjusted_spread_cny_mwh` is
`gross_spread - efficiency_loss - degradation_cost`, and `degradation_cost` is
exactly `required_margin`. So nothing new is computed; the eligibility test was
pointed at the term that means what it says.

The box share is **kept, reported beside it**, under a name that says what it
is: `action_space_available`. Both facts matter and neither stands for the
other — a bus can clear the degradation floor and still leave a policy almost
nothing to choose.

**New test.** `tests/test_siting_criterion.py`:
`test_a_bus_is_eligible_on_its_margin_not_on_the_size_of_its_quote_box` runs
`bus_day_metrics` on the same price day with the project's own box and with a
fifth of it, and asserts the physical margin and `siting_eligible` are identical
while `tradeable_box_share` is not. A companion test asserts the two answers are
reported separately.

The first draft of that test used a smooth sinusoidal day and failed: the
achieved spread is priced off the schedule the unit can run, and at 0.4 MW /
1.6 MWh a gentle curve leaves the LP with no profitable cycle at all, so every
box share was zero and the comparison was empty. It now uses a day with a
reached spread of ~313 CNY/MWh, well over the 200.7 CNY/MWh floor.

## A3 — the three buses are no longer special-cased

Handled in S3: buses 22, 29 and 32 are no longer excluded by a constant in the
mapping module. Every one of the 33 carries a policy the config states, and
`tests/test_formal_ensemble.py` asserts it for those three by name. "No mapped
user" is nowhere read as "delete this bus's load".

## A4 — one declaration of the scenario sets

**What it was.** Four modules kept their own list of the same four scenario
names, and `train_rl` kept a fifth, `DEFAULT_TRAIN_SCENARIO`, naming a scenario
none of the others listed.

**What changed.** `config/scenarios.yaml` now carries one block:

```yaml
formal_ensemble:
  scenarios: [typical_day_industry]
  legacy_comparison: [baseline, high_re, peak_load, congestion]
```

`scenarios.formal_scenarios()` returns the formal list **after checking it**:
every member's price source must read a measured market, and a scenario whose
source qualifies but is left out raises too, so the set cannot be narrower than
the rule without saying so. `scenarios.assert_formal_subset()` is how a module
declares that it needs only part of the set.

Consumers now read the declaration: `run.EXPORT_SCENARIOS`,
`make_rot_comparison_figures.SCENARIOS`, `ui/sim_page.RL_SCENARIOS`,
`make_training_summary._SCENARIO_ORDER`, and `make_results_figure.SCEN` (which
appends its own `no_congestion` sensitivity variant on top). `train_rl` requires
its `--scenarios` to be a subset of the formal set.

**The formal set is one scenario, and that is the rule's answer.** The task book
suggested `baseline, high_re, peak_load, congestion` as an initial candidate.
Applying the readiness rule to the registry gives none of them: all four price
from the retired net-load map, which is an accounting identity over this
network's own dispatch, not a measured market. `typical_day_industry` is the
only scenario whose source reads one — and it is still blocked by the unit gate
and the price-date gap from S3.

**New test.** `tests/test_formal_ensemble.py` (12 tests): every consumer agrees
with the declaration; membership follows the rule in **both** directions; an
empty set is reported rather than defaulted; a non-qualifying scenario cannot be
declared formal; a subset may be declared but not extended; the two sets do not
overlap; no source file declares a set inline.

That last check is worth a note. Its first version flagged any line containing a
scenario name and found six more modules; inspecting them separated two genuine
duplicates (`make_results_figure.SCEN`, `make_training_summary._SCENARIO_ORDER`
— both now point at the declaration) from four that are not scenario sets at all:
an LLM prompt example, two dropdown defaults, and a single-name eval default. The
check now looks for a literal list holding **two or more** names, which is what a
set looks like.

## A5 — the 40-day rotation dependency

Done in S4. Nothing further here.

## A6 — the deviation-penalty weight, re-measured

> **Superseded measurement, 2026-10-15.** The B3 day this was measured on is
> built by `baseline_ladder`, whose rung configuration was later found to be
> assembled from dataclass defaults rather than from `config/defaults.yaml` and
> the scenario's own overrides. Every number in the table below therefore
> describes a model the ladder no longer builds. **The selection rule's answer
> did not move** — re-running it under the corrected configuration still picks
> `lambda = 0.10`, so `CALIBRATED_DEV_PENALTY` is unchanged — but the numbers
> supporting it are not these. See `reports/s7_defect_repairs.md` and
> `docs/2026-09-18-dev-penalty-calibration.md`, which carries the current table.

`calibrate_dev_penalty.py` was re-run on the fixed B3 day with no learning:

```
 lambda   pen@mid    |econ|  grad ratio  mean dev   corner    spread
  0.000    0.0000    0.6423      0.0000    0.5542    12.5%    88.386
  0.010    0.0050    0.6423      0.0110    0.5530    12.5%    88.754
  0.050    0.0250    0.6423      0.0551    0.5427     0.0%    82.474
  0.100    0.0500    0.6423      0.1102    0.5058     0.0%    70.073
  0.250    0.1250    0.6423      0.2754    0.3137     0.0%    53.152
  0.500    0.2500    0.6423      0.5508    0.1776     0.0%    29.285
  1.000    0.5000    0.6423      1.1015    0.0845    12.5%    12.638

selected lambda = 0.1
```

The selection rule picks **0.10**, which is the value already frozen in
`rl_td3.CALIBRATED_DEV_PENALTY`. **No constant changes and no experiment needs
re-running.** At 0.10 the regularizer claims 0.110 of the economic term's
gradient and costs 0.050 against a normalized economic term of 0.642; the quote
spread falls from 88.4 to 70.1 CNY/MWh against a floor of 61.9, so the fleet can
still move off its truthful corner.

## What the gate revealed about the non-formal tools

Adding the synthetic-source gate in S4 blocked more than the formal path.
`calibrate_dev_penalty`, `baseline_ladder`, `diagnose_typical_day`,
`accept_typical_day_industry` and `smoke_matd3` all build `typical_day`, which is
priced by `synthetic_v1` — so the calibration failed until it declared itself.

That is the gate working, but the declaration belongs in one place.
`wholesale_price.enable_non_formal_market()` is now called from those five
entry points, with the set greppable in one search. A library does not open the
gate for its caller; only an entry point says "this is not a formal run".

## Found, reported, not fixed

**`make_results_figure.py` does not import.** Line 24 asks `diagnose_profit` for
`DEFAULT_ACTION`, which that module does not define:

```
ImportError: cannot import name 'DEFAULT_ACTION' from 'diagnose_profit'
```

`git diff --stat` shows `diagnose_profit.py` unmodified by this stage, so this is
pre-existing. **No test imports `make_results_figure`**, which is why it went
unnoticed. My edit to that file (pointing its scenario list at the declaration)
is therefore syntactically correct but not importable end-to-end. The fix needs
someone who knows what `DEFAULT_ACTION` was meant to be — line 84 uses it as a
constant `action_provider`, and `diagnose_profit.run_day` takes one as an
argument — so it is reported rather than guessed.

**`rl_training.py:80`** still carries a single-name default,
`eval_scenarios ... else ["baseline"]`. It is not a scenario *set* and the check
above deliberately does not flag it, but it is the same shape of implicit
default that S4 removed elsewhere. Worth a look when that trainer is next
touched.

## Test results

```
PYTHONPATH=src pytest -q tests/test_siting_criterion.py
18 passed

PYTHONPATH=src pytest -q tests/test_formal_ensemble.py
12 passed

# affected set: dev penalty, td3 penalty, siting, market opportunity,
# formal ensemble, baseline ladder, typical-day baseline
135 passed, 13 deselected, 17 warnings in 61.32s
```

Full-suite result recorded below from the actual run.

```
PYTHONPATH=src pytest -q            # full suite, exit code 0
1019 passed, 53 deselected, 17 warnings in 347.42s (0:05:47)
```

1019 against S4's 1005: +12 from `test_formal_ensemble.py`, +2 from the siting
box-independence tests. The 53 deselected are the `slow` tests, not skips.

## Gate status

Unchanged by this stage: the reading unit is `unit_unverified`, no real price
date intersects a real load date, and the formal date-selection rule is still
undecided. **S8 and formal MATD3 training remain blocked** — on those three, not
on anything S5 touched.
