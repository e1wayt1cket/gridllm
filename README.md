# GridLLM — Distribution Network Electricity Market Simulator

Agent-based simulation of day-ahead (DA) and real-time (RT) electricity markets on the IEEE 33-bus distribution network. Prosumers with solar PV, wind, and battery storage bid into two-settlement markets cleared via optimal power flow (OPF), with LLM-powered natural language configuration and insight generation.

## Architecture

```
run.py / dashboard.py / batch_export.py
  ├─ config/defaults.yaml + config/scenarios.yaml  (externalized config)
  ├─ config_loader.py                              (YAML → typed access)
  ├─ scenarios.py → grid.py → models.py            (config → network → agents)
  ├─ market.py → dispatch.py / dispatch_*.py → models.py  (clearing → OPF → constraints)
  ├─ nash.py → market.py → dispatch.py             (game theory → clearing)
  ├─ outputs.py                                     (CSV + PNG chart export)
  └─ llm.py → Ollama API                           (AI advisor)
```

## Features

- **Three OPF modes**: DC-OPF (lossless linear), LinDistFlow (branch-flow model for radial networks), and SOCP-OPF (second-order cone relaxation), all solved via Gurobi MILP/QCQP
- **Multi-objective optimization**: weighted-sum and constraint-based methods covering carbon emissions, renewable consumption rate, and curtailment
- **Bidding strategies**: random exploration, adaptive best-response based on LMP signals, and MATD3-based reinforcement learning bidding
- **Two-settlement system**: day-ahead financial settlement + real-time imbalance settlement, settled at nodal LMP per agent
- **Rolling real-time market**: MPC-style rolling horizon clearing with configurable forecast modes (perfect, DA-as-forecast, noisy-DA)
- **Storage self-scheduling**: MPC pre-computed storage charge/discharge plans used as fixed injections during OPF, eliminating LMP spikes from storage intertemporal arbitrage
- **DA rolling horizon**: limits storage price foresight for more realistic market behavior
- **Eight built-in scenarios**: baseline, high renewable (2x PV and wind), peak load (1.5x), network congestion (line capacity halved), no-congestion verification (5x), tight bottleneck (selected lines at 20%), renewable sudden drop (to 10%), renewable surge (10% to full)
- **Nash equilibrium analysis**: diagonalization (Gauss-Seidel), Jacobi, and fictitious play with configurable block-level strategy parameters and parallel multiprocessing
- **Batch export**: `--export-all` runs all default scenarios with Nash testing and outputs per-scenario CSV + PNG chart files
- **LLM advisor**: natural language scenario configuration and <200-word simulation insights via Ollama, with rule-based fallback when unavailable
- **Interactive dashboard**: Plotly Dash on port 8050 with IEEE 33-bus topology visualization, LMP heatmap, time-series curves, storage SOC, KPI cards, settlement tables, and LLM insight panel
- **Pseudo-real-time simulation**: 96-period step-by-step execution with incremental storage state and configurable wall-clock speed
- **Stackelberg game**: leader-follower model with supplier as leader and prosumers as followers
- **Reinforcement learning bidding**: MATD3 (multi-agent twin-delayed DDPG) bidding strategy training with centralized critics and decentralized actors, gym-style environment, and L2-regularized optimizers

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# CLI: run a single scenario
python run.py --scenario baseline --strategy best_response --opf-mode lindistflow

# CLI: run with Nash equilibrium test
python run.py --scenario high_re --nash --nash-method diagonalization --nash-iters 5

# CLI: batch export all scenarios (CSV + PNG charts + Nash)
python run.py --export-all

# Or use the standalone batch exporter
python batch_export.py

# Launch interactive dashboard
python dashboard.py

# Compare multi-objective optimization methods
python compare_methods.py

# Train MATD3 bidding agents (200 episodes, checkpoints every 50)
python train_rl.py

# Run all tests
python -m pytest tests/ -v
```

## Requirements

- Python 3.10+
- Gurobi (license required; all OPF paths depend on Gurobi; DC-OPF has HiGHS fallback)
- Ollama (optional, for LLM advisor features)

Key dependencies: `gurobipy`, `pandapower`, `dash`, `plotly`, `numpy`, `scipy`, `pandas`, `matplotlib`, `ortools`, `pyyaml`

## Module Overview

| Module | Purpose |
|--------|---------|
| `models.py` | Data classes: `MarketConfig` (all market/OPF/multi-objective params), `StorageSpec` (storage params and SOC feasibility checks), `Agent` (load/PV/wind forecasts and actuals, storage reference) |
| `grid.py` | IEEE 33-bus network builder; agent population (residential/commercial/industrial prosumers) with synthetic load/PV/wind/storage profiles; China day-ahead price curve generation. Network cached by line capacity multiplier. All tunable parameters read from `config/defaults.yaml` via `config_loader.py` |
| `config_loader.py` | YAML config loader for `defaults.yaml` and `scenarios.yaml` with dotted-key access |
| `config/defaults.yaml` | Externalized defaults: network topology, load types, prosumer specs, storage parameters, profile generation, price curve |
| `config/scenarios.yaml` | Scenario-specific multipliers and descriptors; new scenarios only need YAML entries |
| `dispatch.py` | Unified OPF entry point, routes to DC/LDF/SOCP engines |
| `dispatch_core.py` | `StorageConstraints`: storage feasibility checks and post-clearing SOC updates |
| `dispatch_dc.py` | DC-OPF engine: lossless linear OPF (Gurobi; HiGHS fallback available) |
| `dispatch_ldf.py` | LinDistFlow engine: branch-flow model with multi-period joint optimization, storage constraints, multi-objective (weighted-sum and constraint-based) |
| `dispatch_socp.py` | SOCP-OPF engine: second-order cone relaxation with line capacity diamond constraint and loss iteration |
| `market.py` | Market clearing orchestration: builds OPF problem and calls dispatch; bidding strategy dispatch (`random_actions`, `best_response_bidding`); two-settlement calculation; MPC-style rolling RT clearing |
| `nash.py` | Nash equilibrium tester: diagonalization, Jacobi, fictitious play with parallel multiprocessing (`Pool`) and block-level strategy parameter configuration |
| `scenarios.py` | Scenario registry, YAML-driven: `get_scenario(name, T)` builds agents and price curves |
| `outputs.py` | Clearing result export: 10 curve CSVs + 6 PNG chart types matching dashboard visual style |
| `llm.py` | Ollama LLM integration: NL → scenario config parsing + post-simulation insight generation (<200 words), rule-based fallback when unavailable |
| `pseudo_realtime.py` | Pseudo-real-time simulator: step-by-step RT clearing with incremental storage state updates and configurable wall-clock speed |
| `dashboard.py` | Interactive Plotly Dash dashboard (port 8050): topology visualization, LMP heatmap, time-series curves, storage SOC, KPI cards, settlement tables, AI insight panel, pseudo-real-time controls, Nash trigger |
| `run.py` | CLI entry point: single scenario, batch export, Nash testing, multi-scale MPC |
| `batch_export.py` | Standalone batch runner: 4 scenarios with fast Nash testing |
| `compare_methods.py` | Multi-objective method comparison: sweeps carbon caps and RE rate targets, outputs welfare/emission/shadow-price comparison table |
| `stackelberg.py` | Supplier-prosumer leader-follower game model |
| `rl_env.py` | Reinforcement learning environment: Gym-style interface for bidding strategy training (103-dim observation, 24 decision blocks/day) |
| `rl_bidding.py` | MATD3 bidding strategy training: centralized critics, twin delayed Q-learning, L2 regularization |
| `price_forecaster.py` | Price forecasting: synthetic sinusoidal and supply-stack merit-order methods |
| `export_analysis.py` | Data quality analysis: agent energy balance, SOC boundaries, anomaly detection |
| `mpc_storage.py` | MPC storage self-scheduling: rolling-horizon optimization of storage charge/discharge plans |
| `export_curves.py` | Load curve and price curve visualization export |
| `export_charts_only.py` | Standalone chart export from saved clearing results |
| `plot_diagrams.py` | System architecture diagrams, load curves, parameter table visualization |
| `topology_data.py` | IEEE 33-bus topology coordinate data |

## Reinforcement Learning Bidding

- **Algorithm**: MATD3 with CTDE — one Actor (local observation → bid) and one centralized twin-Q Critic per agent
- **Action space**: `(bid_mult ∈ [0.3, 1.8], offer_adder ∈ [0, 50])` per decision block; 24 blocks/day (15-min periods, 1-hour decisions)
- **Observation**: 103-dim vector — 24-period lookahead load/RE generation, LMP history, price forecast, SOC, congestion index, opponent bid statistics
- **Training**: `python train_rl.py` (200 episodes, ~7s/episode); TensorBoard logs under `runs/`

## Scenarios

| Name | Description |
|------|-------------|
| `baseline` | Default wind/solar/storage configuration |
| `high_re` | High renewable: 2x PV and wind capacity |
| `peak_load` | Peak load: 1.5x all loads and storage |
| `congestion` | Network congestion: line thermal limits halved |
| `no_congestion` | No congestion: 5x line capacity for algorithm validation |
| `tight_bottleneck` | Tight bottleneck: selected lines (11→12, 15→16) capped at 20% for LMP congestion study |
| `re_ramp_drop` | Renewable sudden drop: PV/wind output falls to 10% after midpoint |
| `re_ramp_surge` | Renewable surge: output rises from 10% to full after midpoint |

## Dashboard

Launch with `python dashboard.py` and open `http://localhost:8050`. Features:

- Scenario quick-select with Chinese labels and OPF mode toggle (DC / LinDistFlow / SOCP)
- Bidding strategy selection (random / rl / best_response)
- Static market clearing and pseudo-real-time simulation controls
- Nash equilibrium test trigger with configurable method and iterations
- Natural language input for LLM-driven configuration
- IEEE 33-bus topology with LMP-based node coloring and prosumer star markers
- LMP time-series curves with IQR envelope and representative bus traces
- Aggregate trade (buy/sell), RE generation (PV/wind), and load (served/unserved) curves
- Storage SOC and charge/discharge dual-panel profiles
- KPI summary cards (welfare, RE rate, carbon, curtailment) with settlement table
- Auto-generated AI insights after each simulation run

## Batch Export

Run `python run.py --export-all` or `python batch_export.py` to produce per-scenario output:

```
exports/<timestamp>/
  baseline/      (PNG charts + CSV data files)
  high_re/
  peak_load/
  congestion/
```

PNG charts replicate dashboard visuals: LMP curves, trade curves, storage SOC, RE generation, load profile, IEEE 33-bus topology, and Nash test summary.

## Key Design Points

- **Gurobi is required**: the `_HAS_GUROBI` flag guards imports but all OPF paths depend on Gurobi; DC-OPF has a HiGHS fallback, LinDistFlow and SOCP have no open-source alternative
- **T=96 is the standard horizon**: 24h × 15-min resolution (dt=0.25h); all modules assume this value
- **`MarketConfig` centralizes control**: OPF mode, multi-objective method (`use_constraint_multi_obj` with `carbon_cap_tco2` / `re_min_rate`, or weighted-sum with `lambda_re` / `lambda_curtail` / `lambda_carbon`), line capacity, storage thresholds, RT horizon/step
- **Result dict schema**: `clear_market` returns `{price, lmp (96×33), schedules, welfare, re_consumption_rate, carbon_emissions, carbon_intensity, total_curtailment, shadow_prices}`
- **Bidding strategies** operate on `bid_mult` (scales willingness-to-pay) and `offer_adder` (added to marginal cost); `best_response_bidding` adapts based on prior LMP signals
- **UI language**: dashboard labels use Chinese; physical quantities and abbreviations keep English
