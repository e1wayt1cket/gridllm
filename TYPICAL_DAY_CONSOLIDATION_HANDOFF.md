# Typical-day consolidation round — handoff

The plan this round executes is `C:\Users\26036\.claude\plans\claude-code-cc-matd3-wondrous-dahl.md`.
It has ten steps, S0–S10. **S0–S6 are done. S6 escalated rather than
succeeded — see §4d — so `[2,12,13,19]` at headroom 1.10 with 0.0556 MWh of
shed load is the standing baseline, pending your ruling on §4d's remaining
levers. S7–S10 have not been started.** Nothing in this round has been
committed.

The goal of the round: turn the typical day from "runs" into "usable as the
formal MATD3 training baseline" — correct quote semantics, four batteries with
a real strategy space, and no shed load. MATD3 itself is not touched and no
training happens in this round.

---

## 1. Status

| step | state | evidence |
|---|---|---|
| S0 freeze the pre-change baseline | **done** | `outputs/typical_day_pre.json`; diagnostic exits 1 under `--unserved-limit 0` |
| S1 YAML as single source of truth | **done** | typical-day JSON byte-identical to `outputs/typical_day_pre.json` |
| S2 retire the anchor, measure the (0,0) fleet | **done** | four of four units now cycle; fleet net +199.39 → **+542.88** CNY/day |
| S3 new quote semantics in the solvers | **done** | `outputs/typical_day_s3.json` byte-identical to `s2.json` |
| S4 RL / strategy plumbing | **done** | `outputs/typical_day_s4.json` byte-identical to `s3.json` and `s2.json`; 261 default / 35 slow |
| S5 ESS placement measurement | **done — layout moved** | fleet `[6,20,24,31]` → **`[2,12,13,19]`**; 4/4 units now have a strategy space (was 1/4). See §4c |
| S6 zero-shedding search | **done — escalated** | not reachable inside the 1.25 envelope (needs 1.80); no DER candidate helps. Pareto table in §4d |
| S7 observation V4 | not started | |
| S8 day-level validation logging | not started | |
| S9 `price_threshold` baseline | not started | |
| S10 `TYPICAL_DAY_FINAL_BASELINE.md` | not started | |

**Test state**: default suite **300 passed** (244 before S4; S4 added 3 to
`test_rl_td3_penalty`, 2 to `test_rl_spec`, 7 to `test_action_spec_consistency`
and 5 to `test_nash_action_space`; S5 added 16 to `test_siting_criterion`; S6
added 20 to `test_der_relocation` and 3 to
`test_scenario_override_precedence`).
Slow suite **35 passed / 1 skipped in 705 s**, all green, including the two
`stackelberg` tests whose box change is what made them slow (see §6).

Interpreter: `C:/Users/26036/.conda/envs/energy_env/python.exe`, always with
`PYTHONPATH=src` (`pytest.ini` already sets `pythonpath = src` for pytest).

---

## 2. The three things that must not be lost

### 2.1 truth = `(0, 0)`, and it is what makes the fleet work

`cycle_cost` (=100 CNY/MWh per leg) is charged separately in the objective as
`-cycle_cost*(ch+dis)`, so degradation must **not** also enter the quote — that
would double-count it. At truthful `(0,0)` the round-trip a unit must clear is
exactly `2*cycle_cost = 200` CNY/MWh, and the measured per-bus round-trip value
at the system valley/peak is 264 / 631 / 284 / 284 for ESS6/20/24/31.

The measured consequence (S2): before, two units were exactly idle and one
barely moved; after, **all four both charge and discharge**.

### 2.2 The action space is a corner, and that is the real obstacle (R2)

The margin above the 200 floor is thin for three of the four:

| unit | net profit | margin over 200 | fraction of the action box that trades |
|---|---|---|---|
| ESS6 | +3.30 | **+17.4** | ~0.34 % |
| ESS20 | +537.51 | +416.5 | ~98 % |
| ESS24 | +2.08 | **+9.4** | ~0.10 % |
| ESS31 | −0.00 | **+0.7** | ~0.0005 % |

The box is `charge_price ∈ [0,150] × discharge_price ∈ [0,300]`; a unit trades
only while `charge_price + discharge_price < margin`. So ESS31 trades almost
only when both quotes are essentially zero.

This changes S5's selection rule. Re-siting is no longer about "making the four
profitable" — they already are — but about **giving each one a tradeable quote
space**. The rule must therefore require the margin to be *large enough*, not
merely positive. See §5.

### 2.3 Two silent errors found; both fixed in S4

- **Fixed in S4**: `baselines.rule_policy` / `myopic_policy` were written before
  the charge-leg sign flip and had described the opposite of what they computed
  ever since. `test_baselines.py` only asserted orderings, so the inversion
  passed. Both are rewritten with the correct convention — *a unit charges when
  it asks little to charge and declines to sell* — and the tests now assert that
  semantically (`_likes_to(act, "charge")`).
- **Fixed in S4**: the deviation penalty anchored at the logit zero, while
  `Actor.forward` maps `action = mid + tanh(logit)*half`. For the new box the
  midpoint is `(75, 150)`, so the penalty pulled the policy toward charging 75
  and discharging 150 — far from truthful, and squarely inside the region where
  ESS6/24/31 cannot trade at all. It now references the logit that reproduces
  the spec's truthful point, and the weights that had been tuned for the
  midpoint meaning are flagged as needing revision before training (see §4b,
  which also records that the plan's `atanh` formula is degenerate here and why).

---

## 3. What changed, by file

**Configuration**
- `config/defaults.yaml` — every field of `NetworkConfig` / `StorageConfig` /
  `MarketDesignConfig` / `RTConfig` now has an entry. `market_design
  .enable_multi_objective` corrected from `true` to `false` (the value that was
  actually in effect). `storage.bid_anchor` removed.
- `config/scenarios.yaml` — `typical_day` gains
  `network.inverter_smax_multiplier: 1.10` **through the per-scenario override**,
  which is what keeps the other scenarios bit-identical.

**Model and construction**
- `src/models.py` — `YAML_FIELD_MAP`, `KNOWN_INERT`, `YAML_KEYS_READ_ELSEWHERE`,
  `MarketConfig.from_yaml()`. `StorageConfig.quote_anchor()` and `bid_anchor`
  deleted. `bid_value`/`offer_cost` are now documented as settlement-side prices
  a battery alone has no use for.
- `src/scenarios.py` — `apply_scenario_overrides()` extracted, with a strict
  guard: an override naming no field raises instead of becoming a stray
  attribute. `typical_day_config()` is three statements and assigns nothing.
  `_build_scenario` builds from YAML on both paths.
- `src/ess.py` — the fleet is built with `bid_value = offer_cost = 0.0`.

**Action space and quotes**
- `src/rl_spec.py` — `ActionSpec` gains a `truthful` field;
  `ACTION_STORAGE_QUOTE_V2 = (charge_price, discharge_price)` with bounds
  `[0,150] × [0,300]` and truthful `(0,0)`; `ACTION_SPECS` registry;
  `action_spec_for(agent)`; an import-time assertion that every spec's truthful
  point is inside its own bounds.
- `src/dispatch_core.py` — `quote_arrays(agent, action_params, T, config)`, the
  single quote transform all three engines now delegate to. A storage agent
  given the retired `bid_mult`/`offer_adder` keys raises; an absent declaration
  means truthful.
- `src/dispatch_socp.py`, `src/dispatch_ldf.py` — delegate to it; LDF gains the
  sign check it was missing.

**Producers and consumers migrated so far**
- `src/rl_env.py` — action writing, `_default_actions`, `DayCapture.add_declared`,
  the differential-reward baseline arm and `_compute_opponent_features` are all
  driven by the agent's spec. Default action spec is now the storage one.
- `src/strategies/fixed_bidding.py`, `rl_bidding.py`, `random_bidding.py` — spec-driven.
- `src/mpc_storage.py` — declares corner quotes by direction.
- `src/stackelberg.py` — searches whatever its leader's spec is.
- `src/baselines.py` — `truthful_policy` returns the spec's truthful point;
  `rule_policy` / `myopic_policy` rewritten with the correct direction; bounds
  read from the spec instead of restated.

**Tests added**
- `tests/test_yaml_config_agreement.py` (15) — coverage in both directions, the
  live-value check that would have caught `enable_multi_objective`, override
  rejection, `custom_loads` never reaching the config.
- `tests/test_storage_quote_semantics.py` (15) — the two ontologies, the
  refusals, the exactness floor, spec selection.
- `tests/test_baselines.py` — rewardritten from ordering to semantic assertions.
- `tests/test_storage_exclusivity.py` — `_clear` now takes `charge_price`; the
  crossing-quote test replaces the retired no-crossing rule test.

---

## 4. Current day (S3/S4 state)

`outputs/typical_day_s4.json`, byte-identical to `s3.json` and `s2.json`.

| | |
|---|---|
| coincident peak | 10.0000 MW |
| load energy | 179.00 MWh |
| unserved | 1.0440 MWh (0.58 %) |
| curtailment | 0.9469 MWh |
| RE rate | 97.81 % |
| carbon (net import) | 82.44 tCO2 |
| LMP | 503.9 / 782.8 / 2743.4, ratio 2.406 |
| fleet net | +542.88 CNY/day |
| gates | 11/11, diagnostic exit 0 |
| reproducibility | two runs byte-identical |

Per-unit: ESS6 ch 0.25 / dis 0.21 / +3.30; ESS20 1.87 / 1.58 / +537.51;
ESS24 0.52 / 0.44 / +2.08; ESS31 0.07 / 0.06 / −0.00.

---

## 4b. What S4 did, and what it found

**S4 — done.** What it changed, and the three things it found that the plan
did not anticipate:

*Plumbing.* `nash.py`'s best-response search (COBYLA and the sampling fallback)
and its payoff-cache key are now driven by each agent's spec instead of
`agent.is_prosumer` + `market_design.bid_mult_range`; `counterfactual_eval`
and `diagnose_profit` key their recorded declarations off the spec;
`eval_agents` builds the policy-load box and the regret profile from the spec;
`rl_profit_diagnostics.valuation_artifact` dispatches on `participant_type` the
way `quote_arrays` does. `rl_env._spec_for` was renamed `spec_for` (public) —
two of those callers need it.

1. **`rl_bidding.MATD3` — not `rl_td3` — is the algorithm this round is for,
   and it carried the same defect.** The plan named `rl_td3.py:302–307`. But
   `MATD3` (in `rl_bidding.py`) is what `--algo matd3`, the default, actually
   runs, and it had the identical `logits**2` penalty *and* its own hardcoded
   `act_dim = 2`. Fixing only the plan's named file would have left the named
   prerequisite broken in the file that matters. Both are fixed; both now take
   `act_dim` from `env.action_spec.dim`.
2. **The plan's deviation-penalty formula is degenerate.** §2.3 says the
   reference must be `atanh((truthful − mid) / half)`. For
   `ACTION_STORAGE_QUOTE_V2` the truthful point `(0, 0)` **is the box's lower
   corner**, so that expression is `atanh(−1) = −∞` on both legs. Implemented
   as intended with a documented clamp one part in a thousand short of the edge
   (`rl_td3.truthful_logits`), which stands for an action within 0.2 CNY/MWh of
   truthful. **Consequence to carry into training**: the penalty no longer
   regularizes toward the middle of the box, it pins the policy into the
   corner — which for ESS6/24/31 (§2.2) is the region where they can barely
   trade. The weights `--bid-dev-penalty 5.0` / `--offer-dev-penalty 0.5` were
   tuned for the old midpoint meaning and must be revisited before a training
   run, not inherited.
3. **Three archived arms still carry the unrepaired penalty** — `rl_diffusion`,
   `rl_masac`, `rl_quantile` each have `logits**2` and `act_dim = 2`. They are
   reachable via `--algo mad3pg|qmatd3|masac`. **Deliberately not fixed**: they
   are archived (see the memory notes for MAD3PG/qmatd3/MASAC) and are not on
   this round's path, but re-running one would silently reproduce the old
   midpoint anchor. One-line change each when wanted.

4. **The checkpoint guard the rename was justified by did not exist.**
   `rl_spec`'s docstring and this handoff both said renaming the actions makes
   a stale checkpoint "fail loudly", because `save_policy`/`load_policy` check
   the saved spec name. They check the **observation** spec only —
   `load_policy` accepted an `action_spec` argument, `save_policy` wrote it into
   the file, and nothing ever compared it (`test_load_policy_spec_mismatch_raises`
   is an `obs_spec` test). Since the storage actions have the same arity as the
   retired pair, a `bid_offer_v1` checkpoint loads into a storage actor with no
   symptom and is read as CNY/MWh. The comparison is now implemented, and the
   three call sites that had it switched off (`diagnose_profit`,
   `make_results_figure`) now pass their specs.

Also two smaller consequences of the same migration:

- `eval_agents.py` lost `--bid-mult-low/high`. They existed only to assemble
  the action box for `load_policy`, which the spec now supplies, so they had
  become a knob that selected nothing.
- That hardcoded box was also **wrong for the current spec**: `main()` built
  `[[0.3, 0.0], [1.8, 50.0]]` while a storage policy is trained in
  `[0, 150] × [0, 300]`, so every evaluated action was rescaled by 150× on the
  charge leg and 6× on the discharge leg. The library entry point
  (`evaluate_policy_dir`) read the box off the env and was correct; only the CLI
  path was affected.

## 4c. What S5 did, and what it overturned

**The fleet moved: `[6,20,24,31]` → `[2,12,13,19]`.** All four steps of S5 are in
place — `ess_buses` threaded through `get_scenario` → `_build_scenario` →
`run_day`, `siting_sweep` (M2) and `layout_search` behind `--sweep-ess-buses`
and `--search-ess-buses`. `outputs/typical_day_s5.json` is the new day, two runs
byte-identical, gates 11/11.

**The threshold was fixed before measuring**, as §2.2 required: eligibility is
the share of the action box a unit's round-trip margin leaves tradeable, and
`TRADEABLE_BOX_SHARE = 0.25` was committed to the code before the sweep ran.
The equivalence was stated first and then confirmed: **25 % of the box ⟺ margin
≥ 150.0 CNY/MWh.**

**The headline finding: M2, the plan's siting metric, does not measure what a
placement decision is about.** The plan's §3.2 says a single-unit clear at each
bus "is the siting metric (value before introduction, no competitors)". It is
not — four units arbitrage the same daily spread, so the price field a
candidate bus shows alone is one the other three have already flattened.

| layout | how chosen | unserved MWh | fleet CNY/day | units eligible |
|---|---|---|---|---|
| `[6,20,24,31]` | the original placement | 1.0126 | **542.88** | **1 of 4** |
| `[10,15,19,26]` | M2 solo scan (max-min spacing) | 0.0034 | 89.64 | **0 of 4** |
| `[2,12,13,19]` | joint hill-climb | 0.1709 | 526.64 | **4 of 4** |

The middle row is the trap: the layout the plan's own metric selects leaves
*none* of the four with a usable action space, and costs 83 % of the fleet's
profit. The same bus makes the point directly — bus 6 leaves a lone unit
**186.3** CNY/MWh and **17.4** once the fleet bids.

So the selection metric is now the **joint** tradeable share, read off the day
the four units actually produce together (`price_report`'s `siting_eligible`),
and `layout_search` hill-climbs it with a lexicographic objective: units
clearing the bar first, then the worst-placed unit's share, then the total. The
floor comes first on purpose — maximising the total gives one unit almost
everything, which is the old layout's failure mode. `siting_sweep` is retained
as a per-bus price landscape and its docstring now says plainly that it is not
the selection metric.

**The two margin readings disagree sharply, and neither is the answer alone.**
Under the committed layout, bus 17 shows **1405 CNY/MWh** read off the joint
day and **45.8** with a single unit cleared there. Both are real measurements of
different markets. The joint reading is inflated by a voltage-constraint shadow
price on the deep feeder — a price that a battery sited there would itself
relieve, so it is not a sum anyone could collect. The isolated reading has the
mirror-image flaw: it is the value a lone unit finds *before* the other three
flatten it. Neither can select on its own, which is why the pool the search runs
over is now a pure constraint (not the slack, not on a renewable) with no margin
filter at all: a search bound that excludes the optimum cannot be repaired by
the objective. `by_bus` and `siting_sweep` both stay in the artifact as the two
readings, and only the realized joint share decides.

**Two criteria in the plan's §3.2 rule are unusable as written**, both recorded
in the artifact rather than silently dropped:

- **Spacing** (pairwise series resistance ≥ the network median, 3.6168 Ω) admits
  **no** 4-subset of the eligible buses. The best achievable min-pairwise is
  3.4928. It was replaced by a max-min objective, which has no free parameter.
- **"One per primary feeder zone"** is degenerate on this network: the IEEE
  33-bus feeder is a single trunk whose slack has one neighbour, so every bus is
  in one zone.

**Two consequences to carry forward.**

1. **S2's numbers are superseded, not wrong.** The +542.88 CNY/day head-line is
   from the old placement; the new one is **+526.64**, spread across four units
   instead of concentrated in one. The `diff s3.json s2.json` chain is intact up
   to S4 and breaks at S5 — which is correct, S5 is the first step that changes
   the day on purpose.
2. **S5 may have partly done S6's job.** Shed load fell from **1.0126 → 0.1709
   MWh** purely from re-siting, with no change to the inverter multiplier. §3.3
   predicted this coupling ("远端放电本身就是电压支撑") and required it be
   measured rather than assumed; it is measured, and it is large. S6's remaining
   problem is therefore much smaller than the 1.044 MWh it was written against,
   and its §7-vs-§8 conflict needs re-checking at the new layout before any
   inverter multiplier is raised. **Do not carry the old headroom frontier
   forward** — §3.3 already says the 1.00/1.10/1.25/… table was measured on the
   old layout and may only be used as a contrast.

---

## 4d. What S6 did, and why it stops short

**S6 escalated rather than succeeded.** Zero shedding is not reachable inside
the 1.25 inverter envelope, and no candidate renewable layout helps. The plan's
§3.3 escalation point is therefore reached, and this section is the Pareto table
it asked for. Nothing was tuned past the envelope.

### First: the headroom knob was dead, and had been since S1

`--sweep-headroom` produced **identical rows at every multiplier** — the tell
that a knob is not connected. It was not. `run_day` set
`config.network.inverter_smax_multiplier`, then `get_scenario` re-applied
`typical_day`'s `override_config` on top of that same config and put the file's
1.10 back. So the diagnostic only ever cleared one model, and every sweep of
this knob — including the one this round inherited — measured nothing.

The regression is S1's. Before S1 the multiplier was assigned in
`typical_day_config()`; S1 moved it into `config/scenarios.yaml` so the file
would be the single source of truth, and that put it in the very dict
`_build_scenario` re-applies. **S1's gate could not see it**, because the gate
was "the default day is byte-identical" and the default is 1.10 either way.

Fixed by expressing a per-run override where both readers agree — the scenario
cache (`config_loader.set_scenario_override`) — and by clearing that cache in
`run_day`, since it is process-global and otherwise leaks a swept value into the
next run. `tests/test_scenario_override_precedence.py` holds both. The lesson
generalises: **a knob that is swept must be shown to move the model**, and a
byte-identical gate on the default value cannot show that.

### The frontier, at the committed layout `[2,12,13,19]`

§7 is zero shedding; §8 is curtailment below 0.5 % of available renewable
energy, which is **0.2162 MWh** of the day's 43.23 MWh.

| headroom | unserved MWh | curtailed MWh | RE % | v_min | coincident peak | §7 | §8 |
|---|---|---|---|---|---|---|---|
| 1.10 (configured) | 0.1709 | 0.9469 | 97.81 | 0.9300 | 10.0000 | ✗ | ✗ |
| 1.15 | 0.1237 | 0.6728 | 98.44 | 0.9300 | 10.0000 | ✗ | ✗ |
| 1.20 | 0.0853 | 0.4862 | 98.88 | 0.9300 | 10.0000 | ✗ | ✗ |
| **1.25 (envelope limit)** | **0.0556** | **0.3488** | 99.19 | 0.9300 | 10.0000 | ✗ | ✗ |
| 1.35 | 0.0281 | 0.1597 | 99.63 | 0.9300 | 10.0000 | ✗ | ✓ |
| 1.40 | 0.0170 | 0.1073 | 99.75 | 0.9300 | 10.0000 | ✗ | ✓ |
| 1.50 | 0.0077 | 0.0573 | 99.87 | 0.9300 | 10.0000 | ✗ | ✓ |
| **1.80** | **0.0000** | **0.0000** | 100.00 | 0.9300 | 10.0000 | ✓ | ✓ |

At 1.80 the residual is 5.66e-07 MWh — a solver tolerance, not load dropped.
The two goals cross at different multipliers: **§8 needs ≈1.32, §7 needs 1.80**,
so they are not in conflict with each other but both sit outside 1.25. The
coincident peak stays at the configured 10.0000 MW throughout, and `v_min` is
pinned at 0.9300 at every row — voltage remains the only binding constraint,
exactly as §3.3 says.

### The candidate layouts all make it worse

Each candidate is a *move* of an existing site, so installed MW and the energy
under them are conserved by construction (`scenarios.relocate_renewables`, held
by `tests/test_der_relocation.py`). At headroom 1.25, against the incumbent's
0.0556 unserved / 0.3488 curtailed:

| layout | what it moves | unserved MWh | curtailed MWh |
|---|---|---|---|
| **none (incumbent)** | — | **0.0556** | **0.3488** |
| S-PV1 | 2.0 MW PV, bus 21 → 17 (deep main feeder) | 0.5701 | 13.3522 |
| S-PV2 | 1.5 MW PV, bus 23 → 24 (beside the largest load) | 0.2805 | 10.3833 |
| S-WIND | 1.5 MW wind, bus 30 → 32 (end of the spur) | 2.5864 | 4.8225 |
| S-SPLIT | the 2.0 MW site halved onto buses 17 and 2 | 0.6639 | 6.4862 |
| S-COMB | all three moves together | 6.3166 | 14.8785 |

**This falsifies §3.3's stated hypothesis.** The plan predicted that moving
injection toward the voltage-limited end of the feeder would buy shedding
("假设是「远端放电本身就是电压支撑」"). S-PV1 does the opposite: shedding rises
4× and curtailment rises 14×, because the deep feeder cannot export what is
injected there and the array is curtailed instead. The configured layout is
already the best of the candidates tested, and by a wide margin.

### The remaining levers, stated in full

Per §3.3, the search stops here. What is left is exactly:

1. **inverter multiplier above 1.25**, up to 1.80 for zero shedding — a 1 MW
   array behind a 1.8 MVA inverter;
2. **peak below 10 MW** (you fixed 10 MW; the peak is not currently binding —
   the coincident peak is on target at every row above);
3. **relaxing `v_min_pu`** — forbidden by the plan, and it is the binding
   constraint;
4. **spatial redistribution of load** (the existing `custom_loads` lever) —
   which changes what the day *is*, not only where the equipment sits.

**No documented decision is overturned.** `TYPICAL_DAY_PHYSICAL_BASELINE.md`
§7.1 rejected the zero-shedding routes as unphysical; that rejection now stands
on a measured frontier instead of an assumed one, and the envelope's limit is
0.0556 MWh of shed load — 0.031 % of the day's 179 MWh.

---

## 5. What is left, in the order it should be done

**S5 placement.** Plumbing first: `ess_buses` through
`get_scenario` → `_build_scenario` → `run_day`, plus a `siting_sweep` and
`--sweep-ess-buses`. Note `build_ess_fleet(buses=...)` **already exists and has
no caller**. The selection rule must use the §2.2 criterion: eligibility by a
margin large enough to leave a tradeable fraction of the action box (the plan
proposes 25 %; that threshold should be fixed before measuring, so the outcome
cannot be reverse-engineered). Then measure M1 (joint clear at `(0,0)`) and M2
(single-unit clear at each of the 32 buses) and decide whether to move.

**S6 zero-shedding.** Order is layout → inverter multiplier (stop at **1.25**) →
peak (report only; the user fixed 10 MW). The user's §7 wants zero shedding at
1.10–1.25 and their §8 wants curtailment < 0.5 % of available RE (≈0.211 MWh,
which needs ≈1.35); the two cannot both hold in those ranges **on the current
layout**, which is exactly why the layout search runs first. If zero shedding is
not reached within the envelope, stop and escalate with a Pareto table — do not
tune past 1.25. Note that achieving it would **overturn a documented decision**
(`TYPICAL_DAY_PHYSICAL_BASELINE.md` §7.1 rejected both zero-shedding routes as
unphysical), and that overturning must be recorded explicitly.

**S7 observation V4** — 19 dims (10 unique + 9 shared), the seven local network
features, `dispatch_core.radial_parent_map` / `radial_children` promoted out of
`dispatch_ldf`, staleness documented (five of the seven describe the *previous*
committed block).

**S8 day-level logging**, **S9 `price_threshold` baseline**, **S10 the final
baseline document** — see the plan for each.

S9 carries one thing S4 could not finish: S4's verification names four arms
(`truthful`/`rule`/`myopic`/`price_threshold`) run a day with no fallback, and
only the first three exist yet. `BASELINE_NAMES` is
`("truthful", "rule", "myopic")`; adding the fourth there is enough for
`tests/test_action_spec_consistency.py` to pick it up in both the fast and the
slow half, since both are parametrized over that tuple.

---

## 6. Open items and traps

- **The two `stackelberg` tests dominate the slow suite** — 1330 s for all 36,
  of which they are the bulk. The leader's search went from one parameter to two
  and the box from `[0.3,1.8]×[0,50]` to `[0,150]×[0,300]`, so differential
  evolution needs more iterations. Real, measured, unattended; budget for it
  before running the slow suite in a step's verification.
- **`enable_multi_objective`** is now `false` in YAML *and* in effect. Setting it
  `true` would change the `baseline` / `high_re` objectives and is out of scope.
- **Adding YAML keys turned dead knobs into live ones.** `v_min_pu`, `base_kv`,
  `load_power_factor`, `n_loss_iters`, `reactive_support` and the rest were
  uneditable-in-effect before; editing `defaults.yaml` now changes the model.
  That is the intent, and S1's byte-identical gate is the proof it landed
  behaviour-neutral.
- **`network.base_mva` is still dead** and stays dead — it is named in
  `KNOWN_INERT` with the reason (wiring it means reworking the per-unit base in
  all three engines at once).
- **`churn_free_quotes` is inert.** Charge and discharge are the parts of one net
  flow, so overlap is not representable and no quote needs capping. The field is
  kept only because callers still pass it.
- **`pseudo_realtime.py` is broken** (calls `solve_opf_gurobi` with nine
  positional arguments against eight). Outside this round's path.
- **`llm.LLMAdvisor.apply_llm_config_to_agents` is dead and was already
  broken** — it builds `MarketConfig(penalty_unserved=..., lambda_re=..., ...)`
  with the pre-refactor *flat* keyword API, which now raises `TypeError` on the
  first keyword, and it has no caller anywhere in `src/` or `tests/`. The
  handoff previously listed it as an S4 consumer; that was wrong. It is not a
  retired-action-key problem and rewriting it would mean guessing at intent on
  a dead path — note in particular that its `enable_multi_objective` default is
  `true`, which contradicts the S1 decision. Left alone deliberately.
- **LinDistFlow and DC still price prosumer batteries at wholesale**, not at
  declared quotes; only the independent fleet uses declared quotes in those
  engines. There are no prosumer batteries in the typical day, so this does not
  touch the current numbers, but the engines do not answer the same question for
  that class.
- **Every policy under `policies/` (39 directories) and every
  `results/*_eval.csv` belongs to the retired twelve-unit fleet** and shares no
  agent with this scenario. They must be retrained; there is no `ESS*` policy on
  disk.

---

## 7. Verification

```bash
PY=/c/Users/26036/.conda/envs/energy_env/python.exe
export PYTHONPATH=src

$PY -m pytest tests/ -q                 # 300 passed
$PY -m pytest tests/ -q -m slow         # 35 passed / 1 skipped, 705 s

$PY src/diagnose_typical_day.py --json outputs/typical_day.json --unserved-limit 1.1
$PY src/diagnose_typical_day.py --quiet --json outputs/typical_day_b.json --unserved-limit 1.1
# the two JSONs must be byte-identical — the reproducibility claim S1 and S3 rely on

$PY src/diagnose_typical_day.py --unserved-limit 0    # exits 1 today; S6 is what changes that
```

Regression guards worth re-running after every step:
`diff outputs/typical_day_s3.json outputs/typical_day_<step>.json` should be
empty for any step that claims to change only the *representation*
(S3 and S4 are the worked examples), and will differ for any step that changes
the day.

After any step that touches a producer or consumer of actions, also run
`$PY -m pytest tests/test_action_spec_consistency.py -q -m slow` — three
baselines driving the fleet for a whole day with no fallback. A producer that
regressed to the retired keys is refused by `quote_arrays`, which the
non-slow half of that file cannot see.

Siting (S5). The placement in `config/defaults.yaml` is an output, so these
are how it is re-derived rather than re-guessed:

```bash
$PY src/diagnose_typical_day.py --sweep-ess-buses all   # per-bus landscape
$PY src/diagnose_typical_day.py --search-ess-buses all  # re-choose placement
$PY src/diagnose_typical_day.py --ess-buses 6,20,24,31  # a layout, for contrast
```

The search converges to the committed `[2,12,13,19]` from more than one
starting layout, which is the evidence that the result is not an artifact of
where it started. It is a local search: a different answer is possible in
principle and would need the same treatment — measure it, do not assume the
committed one is best.
