# GridLLM — Distribution Network Electricity Market Simulator

Agent-based simulation of day-ahead (DA) and real-time (RT) electricity markets on the IEEE 33-bus distribution network. Prosumers with solar PV, wind, and battery storage bid into two-settlement markets cleared via optimal power flow (OPF), with LLM-powered natural language configuration and insight generation.

## Architecture

```
run.py / dashboard.py / batch_export.py
  ├─ config/defaults.yaml + config/scenarios.yaml  (externalized config)
  ├─ config_loader.py                              (YAML → typed access)
  ├─ scenarios.py → grid.py → models.py            (config → network → agents)
  ├─ market.py → dispatch.py → models.py           (clearing → OPF → constraints)
  ├─ nash.py → market.py → dispatch.py             (game theory → clearing)
  ├─ outputs.py                                     (CSV + PNG chart export)
  └─ llm.py → Ollama API                           (AI advisor)
```

## Features

- **Two OPF modes**: DC-OPF (lossless linear) and LinDistFlow (branch-flow model for radial distribution networks), both solved via Gurobi MILP
- **Multi-objective optimization**: weighted-sum and constraint-based methods for carbon emissions, renewable consumption rate, and curtailment
- **Bidding strategies**: random exploration and adaptive best-response bidding based on locational marginal price (LMP) signals
- **Two-settlement system**: day-ahead financial settlement + real-time imbalance settlement, settled at nodal LMP per agent
- **Rolling real-time market**: MPC-style rolling horizon clearing with configurable forecast modes (perfect, DA-as-forecast, noisy-DA)
- **Six built-in scenarios**: baseline, high renewable (2x), peak load (1.8x), network congestion (line capacity halved), renewable sudden drop (to 10%), renewable surge (10% to full)
- **Nash equilibrium analysis**: diagonalization (Gauss-Seidel), Jacobi, and fictitious play with configurable block-level strategy parameters
- **Batch export**: `--export-all` mode runs all 4 default scenarios with Nash testing and outputs per-scenario CSV + PNG chart files
- **LLM advisor**: natural language scenario configuration and <200-word simulation insights via Ollama, with rule-based fallback when unavailable
- **Interactive dashboard**: Plotly Dash on port 8050 with IEEE 33-bus topology visualization, LMP heatmap, time-series curves, storage SOC, KPI cards, settlement tables, and LLM insight panel
- **Pseudo-real-time simulation**: 96-period step-by-step execution with incremental storage state and configurable wall-clock speed

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
```

## Requirements

- Python 3.10+
- Gurobi (license required for the optimization solver)
- Ollama (optional, for LLM advisor features)

Key dependencies: `gurobipy`, `pandapower`, `dash`, `plotly`, `numpy`, `scipy`, `pandas`, `matplotlib`

## Module Overview

| Module | Purpose |
|--------|---------|
| `models.py` | Data classes: `MarketConfig` (all market/OPF/multi-objective params), `StorageSpec`, `Agent` |
| `grid.py` | IEEE 33-bus network builder; agent population with synthetic load/PV/wind/storage profiles |
| `config_loader.py` | YAML config loader for `defaults.yaml` and `scenarios.yaml` with dotted-key access |
| `dispatch.py` | OPF engines via Gurobi MILP: DC-OPF and LinDistFlow with storage constraints |
| `market.py` | Market clearing, bidding strategies, two-settlement, MPC rolling RT |
| `nash.py` | Nash equilibrium tester: block-parameterized best response via random sampling or COBYLA |
| `scenarios.py` | Scenario registry with YAML-driven multipliers |
| `outputs.py` | CSV + PNG chart export: 10 curve CSVs, 6 chart types matching dashboard visual style |
| `llm.py` | Ollama LLM integration for NL scenario config and simulation insight generation |
| `pseudo_realtime.py` | Step-by-step pseudo-real-time simulator with incremental storage updates |
| `dashboard.py` | Interactive Plotly Dash dashboard (port 8050) |
| `run.py` | CLI entry point with `--export-all`, `--nash`, `--multi-scale`, and other flags |
| `batch_export.py` | Standalone batch runner: 4 scenarios with fast Nash testing |
| `export_analysis.py` | Data quality analysis: agent energy balance, SOC boundaries, anomaly detection |

## Scenarios

| Name | Description |
|------|-------------|
| `baseline` | Default wind/solar/storage configuration |
| `high_re` | 2x renewable capacity |
| `peak_load` | 1.8x load |
| `congestion` | Line thermal limits halved |
| `re_ramp_drop` | Renewable output drops to 10% |
| `re_ramp_surge` | Renewable output surges from 10% to full |

## Dashboard

Launch with `python dashboard.py` and open `http://localhost:8050`. Features:

- Scenario quick-select with Chinese labels and OPF mode toggle
- Bidding strategy selection (random / learning / best_response)
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
  baseline/      (7 PNG charts + 11 CSV files)
  high_re/
  peak_load/
  congestion/
```

PNG charts replicate dashboard visuals: LMP curves, trade curves, storage SOC, RE generation, load profile, IEEE 33-bus topology, and Nash test summary. Nash equilibrium testing uses random-sampling best response for practical speed.
