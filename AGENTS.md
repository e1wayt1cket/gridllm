# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Development commands

All modules live flat in `src/` and import each other by bare name, so scripts
run with `src/` on `PYTHONPATH` (or from inside `src/`). `pytest.ini` sets
`pythonpath = src`, so pytest works from the repository root.

```bash
# Run a single scenario (CLI)
PYTHONPATH=src python src/run.py --scenario baseline --strategy adaptive --opf-mode lindistflow
PYTHONPATH=src python src/run.py --scenario high_re --nash --nash-iters 10

# Compare weighted-sum vs constraint-based multi-objective methods
PYTHONPATH=src python src/compare_methods.py

# Launch interactive dashboard (port 8056)
PYTHONPATH=src python src/dashboard.py

# Train / evaluate a policy (see docs/2026-08-27-rl-handoff.md)
PYTHONPATH=src python src/train_rl.py --algo matd3 --episodes 200 --save-dir policies/my_run
PYTHONPATH=src python src/eval_agents.py --policies policies/my_run --scenarios all --consumer-metrics

# Consolidate every run and evaluation on disk into summary tables
PYTHONPATH=src python src/make_training_summary.py

# Run all tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_baseline.py::test_baseline_da_clearing_objective -v
```

No linter or formatter is configured for this project. The repository `venv/`
is not usable (it was created under WSL); use a Python with the dependencies in
`requirements.txt` installed and `PYTHONPATH=src`.

## Architecture

GridLLM is an agent-based electricity market simulator on the IEEE 33-bus radial distribution network. It models 96 periods (24h at 15-min resolution) with two-settlement clearing (day-ahead financial + real-time imbalance).

### Data flow

```
scenarios.py ──→ grid.py ──→ models.py     (config → network → Agent objects)
market.py    ──→ dispatch.py ──→ models.py  (bidding → OPF → Gurobi MILP)
nash.py      ──→ market.py                  (fictitious play ↔ market clearing)
llm.py       ──→ Ollama API                 (NL config + <200-word insights)
```

### Core modules

- **`models.py`**: `MarketConfig` (all market/OPF/multi-objective params), `StorageSpec` (battery parameters, SOC feasibility checks), `Agent` (load/PV/wind forecasts, storage reference)
- **`grid.py`**: Builds the IEEE 33-bus pandapower network; creates agent population (residential/commercial/industrial prosumers) with synthetic load/PV/wind profiles; generates China day-ahead price curves. Network is cached by line capacity multiplier. All tunable parameters (load multipliers, storage specs, wind/PV params, bid values, price curve) are read from `config/defaults.yaml` via `config_loader.py`.
- **`config_loader.py`**: Loads `config/defaults.yaml` and `config/scenarios.yaml`; provides `get_default(key_path)`, `get_scenario_cfg(name)`, `get_prosumer_cfg(load_type)`, `get_load_type_cfg(load_type)` for dotted-key access to configuration.
- **`config/defaults.yaml`**: Externalized defaults for network topology, load types, prosumer specs, storage parameters, profile generation, and price curve.
- **`config/scenarios.yaml`**: Scenario-specific multipliers and descriptors for all six built-in scenarios.
- **`dispatch.py`**: Two OPF engines using Gurobi — DC-OPF (lossless linear) and `solve_lind_opf_batch` (LinDistFlow for radial networks). Both solve multi-period joint optimization with storage constraints (SOC transition, charge/discharge efficiency, ramp limits). `StorageConstraints` provides feasibility checks and post-dispatch SOC updates. Multi-objective supports weighted-sum (λ terms in objective) and constraint-based (hard caps on carbon and RE rate with shadow prices).
- **`market.py`**: Market clearing orchestration — `clear_market` builds the OPF problem and calls dispatch; bidding strategies (`random_actions`, `best_response_bidding`); `adaptive_bidding` dispatches by strategy name; `two_settlement` computes DA+RT settlement payments; `clear_rt_rolling` implements MPC-style rolling horizon real-time market.
- **`scenarios.py`**: Six registered scenarios (`baseline`, `high_re`, `peak_load`, `congestion`, `re_ramp_drop`, `re_ramp_surge`) accessed via `get_scenario(name, T)`.
- **`nash.py`**: Nash equilibrium testing via fictitious play with parallel multiprocessing (`Pool`). Tests unilateral deviation incentives; iterates to approximate equilibrium.
- **`llm.py`**: `LLMAdvisor` calls local Ollama (`qwen2.5:7b`) for natural language → scenario config parsing and post-simulation insight generation (rule-based fallback if unavailable).
- **`pseudo_realtime.py`**: `PseudoRealTimeSimulator` — step-by-step RT execution with incremental storage state updates and configurable wall-clock speed.
- **`dashboard.py`**: Thin entry point (port 8056) that builds the multi-page app and serves it. Pages live under `src/ui/`: `shell.py` (one Dash app, navbar, `dcc.Location` router), `theme.py` (shared colours and the injected stylesheet), `sim_page.py` (the original simulation dashboard: scenario/OPF/strategy selectors, Nash trigger, LMP heatmap, KPI cards, settlement tables, AI insights), `train_page.py` (training monitor), `user_page.py` (consumer payment/surplus), `exp_page.py` (experiment index).
- **`data_aggregator.py`**: Read layer over everything on disk — `outputs/rl/<run>/` artifacts, TensorBoard logs, `results/*_eval.csv` and `policies/`. Every UI page reads through it rather than parsing files itself. Imports pandas lazily (Windows ortools/pandas load order).
- **`surplus_metrics.py`**: Three-layer accounting (consumer payment/surplus, LMP markup, market-power split) plus `DAY_ECON_COLUMNS`/`day_econ_metrics`, shared verbatim by evaluation and the per-episode training log.
- **`make_training_summary.py`**: Consolidates all runs and evaluations into `results/training_summary.md` plus two CSVs.
- **`compare_methods.py`**: Sweeps weighted-sum vs constraint-based multi-objective across varying carbon caps and RE rate targets, printing a clearing-objective/emission/shadow-price comparison table.

- **`baseline_ladder.py`**: The B0-B3 physical ladder, run at two voltage bands.
  B0 is the original IEEE 33-bus at its own nominal load (3.715 MW, a static
  power flow — no market, no shedding mechanism); B1-B3 are the project
  scenario with DER and the ESS fleet added one at a time. B0 is a benchmark
  reference and B1 the project scenario, so they are **not** required to share a
  load level. See `docs/2026-09-18-baseline-ladder.md`.

### Key design points

- **Gurobi is required** — the `_HAS_GUROBI` flag guards the import but all OPF paths depend on it. There is no open-source solver fallback.
- **`MarketConfig` controls everything**: OPF mode (`dc` / `lindistflow`), multi-objective method (`use_constraint_multi_obj` with `carbon_cap_tco2` and `re_min_rate`, or weighted-sum with `lambda_re`/`lambda_curtail`/`lambda_carbon`), line capacity multiplier, storage thresholds, RT horizon/step.
- **T=96 is the standard horizon** (24h at 15-min intervals, dt=0.25h). All modules assume this unless explicitly overridden.
- **Result dict schema**: `clear_market` returns `{price, lmp (96×33), schedules (per-agent dicts with p_buy/p_sell/served/unserved/pv_used/wind_used/p_ch/p_dis/soc), clearing_objective, re_consumption_rate, total_re_available, carbon_emissions, carbon_intensity, total_curtailment, shadow_prices}`, plus `bus_voltage` / `line_utilization` / `grid_import_mw` under `socp`. `clearing_objective` is the clearing model's own objective, **not** a participant payoff and not social welfare, and it is emitted under that one name only; `legacy_schema` translates the retired `welfare` spelling when reading older artifacts.
- **Action space is per participant** (`rl_spec.action_spec_for`, never by whether an agent carries a battery). A battery alone declares `charge_price` / `discharge_price` in CNY/MWh, truthful at **(0, 0)**; a load or prosumer declares `bid_mult` / `offer_adder` shading its own `bid_value` / `offer_cost`, truthful at (1, 0). Every trainer builds each policy in its own agent's box and anchors that agent's deviation penalty to its own truthful point — see `rl_bidding.agent_action_spaces`.
- **The `raw/` directory** contains an earlier standalone trading agent implementation (`agent_trading.py`) that is independent of the main simulation modules.
- **RL bidding (CTDE MATD3 / TD3)**: trains on a single fixed scenario by default (pass `--scenarios` a list to rotate); 12-dim compact observation (OBS_V3: 3 unique + 9 shared); differential reward (per-block profit minus the truthful-bidding baseline, each agent's own) with a box-normalized L2 deviation penalty in the actor loss (`rl_td3.deviation_penalty`), whose single dimensionless weight is frozen at `rl_td3.CALIBRATED_DEV_PENALTY`.
- **Voltage bands**: primary `0.93 <= V <= 1.05` (GB/T 12325 ±7%), strict
  `0.95 <= V <= 1.05` (sensitivity). They differ in the lower bound alone,
  which is the only one this network binds on. A band is a constraint the
  market clears against, never a property to tune the scenarios toward.
- **Scenario scale is declared, never inherited** (`config/scenarios.yaml`).
  `defaults.yaml` holds the IEEE33 benchmark and no study scale. `typical_day`
  is `load_scale_mode: native_peak` with `load_shape: native`, so one shared
  peak-normalized `lambda_t` gives `P_i(t) = P_i_base * lambda_t` and a real
  coincident peak of exactly 3.715 MW. The legacy scenarios are
  `explicit_peak` at 10 MW, tagged `scenario_scale_source: legacy_explicit`.
  Renewable capacity is declared per bus in MW, never derived from
  `base_load`. See `docs/2026-09-18-scenario-revision.md`.
- **Dispatch engines**: `socp` is the primary engine and every reported result
  is cleared with it. `lindistflow` and `dc` are auxiliary and implement
  different markets — DC has no voltage variables, ignores losses, and does not
  model the independent storage fleet at all. See
  `docs/2026-09-18-dispatch-engine-semantics.md`.
- **Economic KPIs**: the research question is answered by `pi_ai`,
  `pi_baseline`, `delta_pi`, `delta_pi_percent` and `p_delta_pi_positive`
  (`counterfactual_eval`), defined on participant payoff from
  `participant_payoff`, never on the clearing objective and never on a training
  reward. A reward is not a payoff.
- **Shed diagnostics** (diagnostic only, not market outcomes):
  `diagnose_typical_day.py --min-shed` separates the shed the market declined
  to serve (0.042 MWh) from the shed the network cannot serve (0.1286 MWh);
  `--no-shed-voltage` forces every unit served and reports that the **voltage**
  band breaks first, in all 96 periods, with no line ever reaching its limit.
- **`calibrate_dev_penalty.py`**: Chooses the deviation-penalty weight on a
  fixed B3 + SOCP day with no learning, by how much of the actor gradient the
  regularizer claims and how far it narrows the fleet's quotes. It never
  computes a profit, so the weight cannot be selected on one. See
  `docs/2026-09-18-dev-penalty-calibration.md`.
- **`smoke_matd3.py`**: A short MATD3 run instrumented for what a reward curve
  cannot show — NaN, out-of-box actions, SOC bounds, clearing fallbacks,
  terminal SOC equality, degenerate or corner-collapsed quotes.
- **Deviation penalty**: one dimensionless weight `--dev-penalty` on the sum of
  squared **box-normalized** deviations, `d = |a - truthful| / (high - low)`,
  differentiated through the tanh so its pull is bounded and vanishes at the box
  edges. It replaced a per-leg logit-quadratic penalty whose gradient grew
  without bound and pinned the storage fleet at its truthful corner. The frozen
  value is `rl_td3.CALIBRATED_DEV_PENALTY` (`0.10`), the default for both
  `train_rl.py` and `smoke_matd3.py`; a test fails if it drifts from what the
  calibration's selection rule picks. See the calibration doc for the table, the
  rule, and the earlier measurement it superseded.
- **Terminal SOC** is neutralized by the *settlement*, not by a money
  adjustment: both arms are cleared over the whole day with `terminal_soc_equal`
  and the endpoints pinned (`counterfactual_eval.settle_day`). `PayoffBreakdown`
  carries no terminal term.
- **UI language**: Dashboard labels are in Chinese; physical quantities and abbreviations use English.
