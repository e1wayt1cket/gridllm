# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

```bash
# Run a single scenario (CLI)
python run.py --scenario baseline --strategy adaptive --opf-mode lindistflow
python run.py --scenario high_re --nash --nash-iters 10

# Compare weighted-sum vs constraint-based multi-objective methods
python compare_methods.py

# Launch interactive dashboard (port 8050)
python dashboard.py

# Run all tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_baseline.py::test_baseline_da_welfare -v
```

No linter or formatter is configured for this project. Virtual environment is at `venv/`.

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
- **`llm.py`**: `LLMAdvisor` calls local Ollama (`gemma4:e2b`) for natural language → scenario config parsing and post-simulation insight generation (rule-based fallback if unavailable).
- **`pseudo_realtime.py`**: `PseudoRealTimeSimulator` — step-by-step RT execution with incremental storage state updates and configurable wall-clock speed.
- **`dashboard.py`**: Plotly Dash app (port 8050) with scenario/OPF/strategy selectors, static analysis, pseudo-real-time controls, Nash trigger, LMP heatmap on bus topology, time-series charts, KPI cards, settlement tables, and AI insight panel.
- **`compare_methods.py`**: Sweeps weighted-sum vs constraint-based multi-objective across varying carbon caps and RE rate targets, printing welfare/emission/shadow-price comparison table.

### Key design points

- **Gurobi is required** — the `_HAS_GUROBI` flag guards the import but all OPF paths depend on it. There is no open-source solver fallback.
- **`MarketConfig` controls everything**: OPF mode (`dc` / `lindistflow`), multi-objective method (`use_constraint_multi_obj` with `carbon_cap_tco2` and `re_min_rate`, or weighted-sum with `lambda_re`/`lambda_curtail`/`lambda_carbon`), line capacity multiplier, storage thresholds, RT horizon/step.
- **T=96 is the standard horizon** (24h at 15-min intervals, dt=0.25h). All modules assume this unless explicitly overridden.
- **Result dict schema**: `clear_market` returns `{price, lmp (96×33), schedules (per-agent dicts with p_buy/p_sell/served/unserved/pv_used/wind_used/p_ch/p_dis/soc), welfare, re_consumption_rate, total_re_available, carbon_emissions, carbon_intensity, total_curtailment, shadow_prices}`.
- **Bidding strategies** operate on `bid_mult` (scales willingness-to-pay) and `offer_adder` (added to marginal cost for prosumers). `best_response_bidding` adapts based on prior LMP signals.
- **The `agent/` directory** contains an earlier standalone trading agent implementation (`agent_trading.py`) that is independent of the main simulation modules.
- **UI language**: Dashboard labels are in Chinese; physical quantities and abbreviations use English.
