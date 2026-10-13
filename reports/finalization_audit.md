# Finalization Audit

> **Correction (2026-10-12) — the price rows conflate two averaging axes.**
> Where this audit records the four curves as means and `261009` as an
> unresolved aggregation, it is answering two questions as one. All four are
> **spatial** means (`spatial_statistic: spatial_mean`); on the **temporal**
> axis `261009` is a **single day** and the other three are means over 7 / 30 /
> 180 days. `261009`'s single-day reading is a **project ruling**, not an
> external confirmation, and the audit's own finding stands: the workbook
> carries no platform export metadata, so the file alone cannot settle it.
> The accounting and coverage findings below are untouched. Current reading:
> `reports/final_research_summary.md` §2.1.

Phase-0 reconnaissance for the IEEE 33-bus storage-bidding finalization. This
records what the code and data actually are, and where they disagree with the
planning documents. Nothing here is a proposed change; the change list is
section 8.

Read-only. No core module was modified to produce this report.

## 1. Environment

| | |
|---|---|
| Repo | `C:\Projects\00_WorkSpace\01_Projects\MakerB` |
| Interpreter | `C:/Users/26036/.conda/envs/energy_env/python.exe` (Gurobi 13.0.1) |
| pandas / openpyxl | 2.3.3 / 3.1.5 |
| Test config | `pytest.ini`: `pythonpath = src`, `addopts = -m "not slow"` |
| Test modules | 46 under `tests/` |

`officecli` (v1.0.140) is installed but its XML path is broken on this machine —
`view`/`get` raise `FileNotFoundException: System.Private.Xml, Version=10.0.0.0`.
All workbook inspection below was done with `openpyxl`
(`read_only=True, data_only=True`).

## 2. Repository state

Branch `master`. Last commit `a1151fd`.

**Uncommitted work — do not overwrite.**

```
 M TYPICAL_DAY_PHYSICAL_BASELINE.md
 M src/config_loader.py, data_aggregator.py, diagnose_typical_day.py,
   dispatch_core.py, eval_agents.py, llm.py, mpc_storage.py,
   price_forecaster.py, rl_bidding.py, rl_env.py, rl_td3.py,
   scenarios.py, strategies/rl_bidding.py
?? gurobi.env            (contents: `Threads 6`)
```

`git diff --stat` = 14 files, +43 / −413. This is a **deletion-dominated
dead-code cleanup**, not a feature branch: `mpc_storage` loses
`solve_storage_mpc_batch`/`mpc_rt_rolling` (−229), `price_forecaster` loses its
algorithm registry and four forecasters (−105), `llm` loses its config-cache
mutators (−24), `config_loader` loses `save_defaults`/`save_scenarios` (−14).
Nothing touches the price-source path or the dispatch/clearing math.
`TYPICAL_DAY_PHYSICAL_BASELINE.md` gains a `SUPERSEDED — HISTORICAL RECORD`
banner at the top.

## 3. What the code actually is

Confirmed by reading the modules, not the docs.

| Concern | Symbol | Location |
|---|---|---|
| Canonical research entry | `typical_day_config(T)` (pins SOCP, T=96, terminal SOC) | `scenarios.py` |
| Scenario build | `_build_scenario` via `get_scenario(name, T, config, ess_buses, der_moves, day_id)` | `scenarios.py` |
| Declared scale | `scenario_scale(cfg)` — raises unless `load_scale_mode` + `load_shape` + `renewable` are all present | `scenarios.py` |
| Network + agents | `build_base_network`, `create_agents_from_network`, `_make_load_profile`, `_coincident_peak` | `grid.py` |
| Price routing | `day_ahead_price_china(T, agents, config, t_start, day_id)` | `grid.py` |
| Price providers | `SOURCES = ("legacy","synthetic_v1","library","guangdong")`, `get_provider` | `wholesale_price.py` |
| Storage fleet | `independent_storage_config()`, `build_ess_fleet(T, config, wholesale, buses, scale)` | `ess.py` |
| Primary clearing | `solve_socp_opf_batch` | `dispatch_socp.py` |
| Auxiliary clearing | `solve_lindist_opf_batch` / `solve_dc_opf_gurobi` | `dispatch_ldf.py` / `dispatch_dc.py` |
| Storage net power, quotes, grid exchange, terminal SOC | `add_storage_net_power`, `check_storage_quote_signs`, `add_grid_exchange`, `pins_terminal_soc` | `dispatch_core.py` |
| Market | `clear_market`, `two_settlement`, `clear_rt_rolling_mpc` | `market.py` |
| Payoff | `participant_payoff`, `PayoffBreakdown`, `IndependentStoragePayoff` | `participant_payoff.py` |
| Paired counterfactual | `settle_day`, `delta_pi`, `delta_pi_percent`, `p_delta_pi_positive`, `assert_paired` | `counterfactual_eval.py` |
| Action box | `ACTION_STORAGE_QUOTE_V2`, `ACTION_BID_OFFER_V1`, `action_spec_for`, `OBS_V3` | `rl_spec.py` |
| MATD3 | `MATD3`, `agent_action_spaces`, `remember` | `rl_bidding.py` |
| Single-agent TD3 | `TD3`, `deviation_penalty`, `CALIBRATED_DEV_PENALTY = 0.10` | `rl_td3.py` |
| Day split | `SPLIT`, `SPLIT_SIZES`, `LIBRARY_DAYS = 40` | `day_rotation.py` |
| Strategies | `"fixed"`, `"random"`, `"mpc"`, `"rl"` | `strategies/` |

**Fleet, verified against the plan's baseline table — it already matches.**

`config/defaults.yaml` → `independent_storage`: `buses: [9, 14, 17, 27]`,
`capacity_mwh: 1.6`, `power_ratio: 0.25` ⇒ **4 units at 0.4 MW / 1.6 MWh each**,
`eta_ch`/`eta_dis` 0.92, `soc0` 0.50, `soc_min`/`soc_max` 0.10/0.90. Names are
`f"ESS{bus}"`. Independently corroborated by `reports/protocol_smoke_v0.1.md`,
which lists its agents as `ESS9, ESS14, ESS17, ESS27`.

Nothing in `ess.py` hardcodes 0.4 or 1.6; both are derived from config.

## 4. Where documents disagree with the code

The plan says to treat the code as authoritative and record the differences.
These are the differences.

| Claim | Actual |
|---|---|
| Fleet buses `[2, 12, 13, 19]` (prior handoff note) | `[9, 14, 17, 27]` — none of the four claimed buses appears anywhere in the config; the YAML records earlier placements `6/20/24/31` and `10/15/19/26` |
| Storage action is multiplicative, `bid = anchor * bid_mult`, truthful point unreachable (`TYPICAL_DAY_PHYSICAL_BASELINE.md` §6/§9, now superseded) | `ACTION_STORAGE_QUOTE_V2`: absolute reservation prices, `charge_price ∈ [0,150]`, `discharge_price ∈ [0,300]`, truthful `(0,0)` **inside** the box. `rl_spec._check_truthful_is_reachable()` asserts reachability at import |
| Clearing lives in `dispatch.py` | `dispatch.py` is a registry shim; the engines are `dispatch_socp.py`, `dispatch_ldf.py`, `dispatch_dc.py`, with shared primitives in `dispatch_core.py` |
| `solve_lind_opf_batch` | `solve_lindist_opf_batch` |
| `random_actions`, `best_response_bidding`, `clear_rt_rolling` exist in `market.py` (CLAUDE.md, AGENTS.md, README.md) | None exist. Real: `RandomStrategy`/`"random"`, `adaptive_bidding` (in `strategies/__init__.py`), `clear_rt_rolling_mpc` |
| Six legacy scenarios | **Eight**: `baseline`, `high_re`, `peak_load`, `congestion`, `no_congestion`, `tight_bottleneck`, `re_ramp_drop`, `re_ramp_surge`, plus `typical_day` |
| MATD3 lives in `rl_td3.py` | `rl_td3.py` is single-agent TD3 only; MATD3/CTDE is in `rl_bidding.py` |
| `rl_env.py` observation is "V2" (module docstring) | `OBS_V3`, `version=3`, 12-dim = 3 unique + 9 shared. The docstring is stale; the code is right |
| A greedy baseline exists | **Does not exist.** `strategies/` holds only `fixed_bidding.py`, `mpc_bidding.py`, `random_bidding.py`, `rl_bidding.py` |
| A `dianyicha` price source exists | `SOURCES` has no `dianyicha`. The real-data slot is `guangdong`, whose `__init__` raises `NotImplementedError` |

Also: `TYPICAL_DAY_PHYSICAL_BASELINE.md` names
`reports/experiment_baseline_v0.1_freeze.md` as authoritative. **That file does
not exist** — `reports/` holds only `protocol_smoke_v0.1.md` and
`protocol_smoke_v0.1_lambda010.md`.

## 5. Data files

### 5.1 `docs/16类行业用户.xlsx`

67,277,368 bytes (~64 MiB), mtime Sep 17. An identical copy sits beside it as
`16类行业用户 - 副本.xlsx`. **No `~$` lock files exist.** Both are gitignored
(`.gitignore` `*.xlsx`), so neither is in the repository.

One sheet, `Sheet3`, **117,271 data rows × 108 columns**:
columns 1–12 are metadata (`序号, 总户号, 地点, 用户名称, 行业, 合同容量, 运行容量,
户号, 电表局编号, 综合倍率, 功率相序, 数据日期`); columns 13–108 are **96 slots,
00:00 → 23:45 at 15-min** — the same count and grid as the simulator's T=96.

73 users · 16 industries · 283 `(总户号, 电表局编号)` pairs · 13 Jiangsu cities ·
2021-01-01 → 2022-12-31 (731 days).

**Coverage is sparse and uneven.** 283 × 731 = 206,873 theoretical
meter-days against 117,271 rows ≈ 57 % filled. Per-user coverage runs from
**1 day** (`钢铁1`, `钢铁4`) to 730 days.

**Measured defects** (to be reproduced by the Phase-1 pipeline, not hardcoded):

| Defect | Count | Note |
|---|---|---|
| Placeholder rows `[1,2,3,3,3,0,…]` | 43 | sum exactly 12; spread over 21 users; structurally detectable |
| Rows with missing slots | 4,609 | 1–15+ slots each |
| Fully empty rows, `功率相序='其他'` | 81 | meter id and date both null; all under 电动汽车 users |
| Rows containing negatives | 10,539 | legitimate backflow for 分布式光伏 (3,517 of 3,650 rows for 分布式光伏1); suspicious for pure-load users (773 rows for 印刷业3) |
| Unreliable `综合倍率` | — | `1320000` recurs across several 化工/印刷 meters; `4000` on 印刷业2/4/5; `2200000` on 化工5. Implied peak reaches hundreds to thousands × declared 运行容量 |
| 5G基站 unit mismatch | 4 users | ratio 1, implied peak 2.3–2.6 × 运行容量 |

**Unit relation.** `value × 综合倍率` = kWh per 15-min slot, so
`value × 综合倍率 × 4` = kW. Verified against 运行容量 for a subset: many
meters land at peak/run 0.3–1.0 and load factor 0.2–0.5, which is physically
sane. It is **not** reliable globally — see the ratio row above. The relation is
therefore usable for shape and for *local* plausibility checks, not as a
wholesale conversion to MW.

**Normalised shapes are sound.** The ratio cancels under normalisation
(`x / max(x)` is scale-invariant), so per-industry shapes can be built without
trusting `综合倍率`. Filtering to complete clean days gives physically sensible
profiles: 数据中心 flat (0.96–1.04); 商业综合体 a commercial peak (0.29 at
04:00 → 1.66 at 13:00); 分布式光伏 a midday bell peaking 1.92 at 11:00;
钢铁/水泥/化工 near-constant base load. 电解铝 is unreliable — 208 clean days
from a single user, shape contrary to process physics.

### 5.2 `docs/日前价格261009.xlsx`

20,621 bytes, mtime 2026-10-09, gitignored, no lock files.

Four sheets, **96 points each, no missing values**, units **CNY/MWh**:

| Sheet | Header | min | max | mean |
|---|---|---|---|---|
| `261009` | `日前均值（CNY/MWh）` | 225.90 | 445.00 | 386.83 |
| `7days` | `日前均价` | 257.50 | 477.87 | 395.55 |
| `30days` | `日前均价` | 333.08 | 450.86 | 407.82 |
| `halfyear` | `日前均价` | 284.93 | 393.86 | 346.33 |

**All four derive from real historical market data, and none is confirmed to be
a single day's series.** The three trailing sheets are means over a named
window. `261009` is the unresolved one: its column header says 均值, "mean",
while its period column names a date, and nothing in the workbook settles which
it is — see `reports/final_research_summary.md` §2.1, which records both
readings rather than choosing. There is therefore **no confirmed real daily
day-ahead price series anywhere in the project**, and any claim of "real daily
price" would be unsupported.

**Time labels are interval-END.** `261009` runs `00:15` → `24:00`, where the
last label arrives as a `datetime` (`1900-01-01 00:00:00`, Excel's rendering of
24:00) rather than a `time`. Since the simulator's period `t` is
`[t·15min, (t+1)·15min)` and the label is that interval's end, the mapping is
**positional identity**: `label[i] → slot i`. No 15-minute subtraction is
needed; the only trap is the datatype of the final label.

**Real prices are far narrower than the synthetic curve.** `defaults.yaml`
sets `price_floor: 120.0` / `price_cap: 1000.0` for `synthetic_v1`; the real
curves span 225.9–477.9. Storage arbitrage is correspondingly thinner.

## 6. Accounting defects that must be resolved before results are reported

Three, all verified by reading the code. Each is carried by the plan into
Phase 5.

**6.1 `eval_agents.main()` bypasses `settle_day`.** `run_episode` accumulates
`agent_profits[nm] += rewards[nm]` from the environment's rolling per-block
rewards and returns it as `"profits"`. `main()` reports
`profit_delta = rl_profit - base_profit` from that dict, with no terminal-SOC
pinning. `settle_day` is imported at the top of the module but is called only
inside `evaluate_policy_dir`. The two entry points publish the same column name
(`profit_delta`) with different definitions — one a block-summed learning
signal, one the endpoint-pinned settled payoff. **The CLI's `results/*_eval.csv`
is therefore not the research KPI.**

**6.2 `counterfactual_eval._BASELINE_CACHE` is keyed by `(scenario, day_id, T)`
only.** Config, price source, observation/action spec and the fleet are all
absent from the key. `assert_paired` compares only the `wholesale` array, which
does not move when fleet capacity or siting changes. A capacity or siting sweep
will silently pair a fresh arm against a stale baseline.

**6.3 `rl_bidding._TRAINED_POLICIES` is keyed by agent name only**, and is
`update`d rather than replaced. Fleet names are `f"ESS{bus}"` — identical across
every scenario — so a policy trained under one scenario is applied to another
with no error. Only the obs/action spec names are checked at load time.

Also noted, not code defects: `participant_payoff.PayoffBreakdown.other_cost` is
never assigned (always 0.0), and the training reward is deliberately a
*different* counterfactual from the settled payoff (documented in both
`rl_env.step` and `counterfactual_eval`). Both are acknowledged design
positions, not bugs.

## 7. Test coverage

46 modules. There is **no** `test_grid.py`, `test_dispatch.py`, `test_market.py`
or `test_scenarios.py`; those modules are covered indirectly through
`test_baseline.py`, `test_clearing_registry.py`, `test_scenario_scale.py`,
`test_typical_day_physical_baseline.py` and others. RL has 13 dedicated
`test_rl_*.py` modules.

Two existing tests encode policy that the new work deliberately extends, and
will need explicit review rather than a silent edit:
`test_scenario_scale.py` asserts `load_shape ∈ ("native","diversified")` and that
`native == ["typical_day"]`.

Process-global caches that tests must reset: `wholesale_price._PROVIDER_CACHE`,
`config_loader._defaults_cache`/`_scenarios_cache`, `grid._net_cache`.
`test_day_rotation._arm` already demonstrates the reset pattern.

## 8. What this changes

Change list, in review order. Each step is independently verifiable and stops at
a review gate.

| # | Change | Depends on |
|---|---|---|
| 0 | This report | — |
| 1 | `src/industry_load.py` + `tests/test_industry_load.py` | — |
| 2 | `src/dianyicha_price.py` + `dianyicha` provider in `wholesale_price.py` + tests | — (independent of 1) |
| 3 | `reports/jiangsu_market_mechanism.md` | — |
| 4 | `load_shapes` keyword on `create_agents_from_network`; `typical_day_industry` scenario; `industry_map`; two test widenings | 1, 2 |
| 5 | Fix 6.1 / 6.2 / 6.3 + `tests/test_storage_economics_audit.py` | — |
| 6 | Single-day end-to-end acceptance | 4, 5 |
| 7 | Perfect-foresight arbitrage bound; three-strategy paired comparison | 6 |
| 8 | Written record of the price-data limitation; **no MATD3 training** | 7 |
| 9 | Load / capacity / siting / price-scenario sensitivity | 7 |

Two constraints bound all of the above and are not negotiable in this round:

- **No MATD3 training.** Four mean curves are not a basis for cross-day
  training, model selection or generalisation claims, and repeating one day's
  episodes would dress single-day fitting as generalisation. The existing MATD3
  code stays; it is neither deleted nor credited with a verified benefit.
- **No tuning toward a favourable result.** `price_floor`/`price_cap` stay at
  120/1000, `cycle_cost` stays at its configured value, and the greedy
  thresholds stay at their pre-declared 300/400 CNY/MWh. A thin or negative
  arbitrage result is a legitimate finding.

The absolute-MW product is expected to be **unusable globally** — `综合倍率` is
only locally plausible — so it defaults to refusing. The scenario therefore runs
on *IEEE 33-bus peak demand redistributed by real Jiangsu industry shapes*, not
on *Jiangsu demand*. That distinction is the reason this audit exists.

## 9. Reproducing the measurements

```bash
PYTHONPATH=src <python> -m pytest tests/ -v
PYTHONPATH=src <python> src/diagnose_typical_day.py --json outputs/typical_day.json
git status --porcelain=v1 --untracked-files=all
git diff --stat
```

Workbook facts in section 5 were read with `openpyxl` in `read_only=True,
data_only=True` mode over `docs/16类行业用户.xlsx` and
`docs/日前价格261009.xlsx`; the quality counts are reproducible from the file
by counting, not by any cached artefact.
