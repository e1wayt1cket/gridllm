"""Export 4 scenarios with charts, no Nash equilibrium testing."""
import os, warnings
from ortools.linear_solver import pywraplp  # preload before numpy to avoid DLL order conflict
import numpy as np, pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")
np.random.seed(1)
from models import MarketConfig
from market import clear_market, adaptive_bidding
from scenarios import get_scenario
from outputs import export_curve_csvs

SCENARIOS = ["baseline", "high_re", "peak_load", "congestion"]
config = MarketConfig(opf_mode="socp", verbose=False, enable_multi_objective=True)
strategy = "rl"
T = 96

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_dir = os.path.join("exports", ts)
os.makedirs(base_dir, exist_ok=True)

all_kpis = []

for scenario in SCENARIOS:
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario}")
    print(f"{'='*60}")

    agents, _ = get_scenario(scenario, T=T, config=config)
    da_actions = adaptive_bidding(agents, config, strategy=strategy)
    da_results = clear_market(agents, T, "DA", da_actions, config)

    print(f"  DA welfare: {da_results['welfare']:,.1f}, "
          f"RE rate: {da_results['re_consumption_rate']:.1f}%, "
          f"CO2: {da_results.get('carbon_emissions',0):.1f}t, "
          f"curt: {da_results.get('total_curtailment',0):.1f}MWh")

    sc_dir = os.path.join(base_dir, scenario)
    kpi = export_curve_csvs(da_results, None, agents, config, scenario, strategy, sc_dir)
    all_kpis.append(kpi)

if all_kpis:
    df_all = pd.DataFrame(all_kpis)
    df_all.to_csv(os.path.join(base_dir, "all_scenarios_comparison.csv"),
                  index=False, float_format="%.4f")

print(f"\n{'='*60}")
print(f"  Export complete - {len(SCENARIOS)} scenarios")
print(f"  Output: {base_dir}")
print(f"{'='*60}")
