# GridLLM — Distribution Network Electricity Market Simulator

Agent-based simulation of day-ahead (DA) and real-time (RT) electricity markets on the IEEE 33-bus distribution network. Prosumers with solar PV, wind, and battery storage bid into two-settlement markets cleared via optimal power flow (OPF), with LLM-powered natural language configuration and insight generation.

## Architecture

```
run.py / dashboard.py
  ├─ scenarios.py → grid.py → models.py    (config → network → agents)
  ├─ market.py → dispatch.py → models.py   (clearing → OPF → constraints)
  ├─ nash.py → market.py → dispatch.py     (game theory → clearing)
  └─ llm.py → Ollama API                   (AI advisor)
```

## Features

- **Two OPF modes**: DC-OPF (lossless linear) and LinDistFlow (branch-flow model for radial distribution networks), both solved via Gurobi MILP
- **Multi-period joint optimization**: storage constraints including SOC transition, charge/discharge efficiency, ramp limits, and minimum run durations — solved across coupled time periods
- **Bidding strategies**: random exploration and adaptive best-response bidding based on locational marginal price (LMP) signals
- **Two-settlement system**: day-ahead financial settlement + real-time imbalance settlement
- **Rolling real-time market**: MPC-style rolling horizon clearing with incremental storage state updates
- **Six built-in scenarios**: baseline, high renewable (2x), peak load (1.8x), network congestion (line capacity halved), renewable sudden drop (to 10%), renewable surge (10% to full)
- **Nash equilibrium analysis**: fictitious play with parallel multi-processing to test unilateral deviation incentives
- **LLM advisor**: natural language scenario configuration and <200-word simulation insights via Ollama (local `gemma4:e2b`), with rule-based fallback when unavailable
- **Interactive dashboard**: Plotly Dash on port 8050 with IEEE 33-bus topology visualization, LMP heatmap, generation/consumption power curves, storage SOC and charge/discharge plots, KPI cards, and settlement tables
- **Pseudo-real-time simulation**: 96-period step-by-step execution with configurable wall-clock speed

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# CLI: run a single scenario
python run.py --scenario baseline --strategy adaptive --opf-mode lindistflow

# CLI: run with Nash equilibrium test
python run.py --scenario high_re --nash --nash-iters 10

# Launch interactive dashboard
python dashboard.py
```

## Requirements

- Python 3.10+
- Gurobi (license required for the optimization solver)
- Ollama (optional, for LLM advisor features)

Key dependencies: `gurobipy`, `pandapower`, `dash`, `plotly`, `numpy`, `scipy`, `pandas`

## Module Overview

| Module | Purpose |
|--------|---------|
| `models.py` | Data classes: `MarketConfig`, `StorageSpec`, `Agent` (load/PV/wind forecasts, bids, storage state) |
| `grid.py` | IEEE 33-bus network builder using pandapower; agent placement (residential/commercial/industrial prosumers) |
| `dispatch.py` | Core OPF engines: DC-OPF and LinDistFlow via Gurobi MILP; storage constraint formulation |
| `market.py` | Market clearing orchestration; bidding strategies (`random_actions`, `best_response_bidding`); rolling RT market |
| `nash.py` | Nash equilibrium solver with fictitious play and parallel multi-processing |
| `scenarios.py` | Scenario registry: `baseline`, `high_re`, `peak_load`, `congestion`, `re_ramp_drop`, `re_ramp_surge` |
| `llm.py` | Ollama LLM integration for natural language → config parsing and simulation insight generation |
| `pseudo_realtime.py` | Step-by-step pseudo-real-time simulator with incremental storage updates |
| `dashboard.py` | Interactive Plotly Dash dashboard (port 8050) |
| `run.py` | CLI entry point with argparse |

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

- Scenario and OPF mode dropdown selectors
- Bidding strategy selection
- Static analysis and pseudo-real-time simulation controls
- Nash equilibrium test trigger
- Natural language input for LLM-driven configuration
- Node LMP heatmap on IEEE 33-bus topology
- Time-series LMP curves
- Generation and consumption power breakdown
- Storage SOC and charge/discharge profiles
- KPI summary cards and settlement tables
- Auto-generated AI insights after each static run
