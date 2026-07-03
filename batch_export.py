"""Batch export: 4 scenarios with charts + Nash equilibrium tests."""
import os, sys, warnings, time
from ortools.linear_solver import pywraplp  # preload before numpy to avoid DLL order conflict
import numpy as np
warnings.filterwarnings("ignore")
np.random.seed(1)
from models import MarketConfig
from market import clear_market, adaptive_bidding
from scenarios import get_scenario
from outputs import export_curve_csvs
from nash import NashEquilibriumTester, plot_nash_results
from datetime import datetime

SCENARIOS = ["baseline", "high_re", "peak_load", "congestion"]
T = 96
config = MarketConfig(opf_mode="lindistflow", verbose=False)
strategy = "rl"

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_dir = os.path.join("exports", ts)
os.makedirs(base_dir, exist_ok=True)

for scenario in SCENARIOS:
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario}")
    print(f"{'='*60}")

    # ---- Market clearing ----
    agents, _ = get_scenario(scenario, T=T, config=config)
    t0 = time.time()
    da_actions = adaptive_bidding(agents, config, strategy=strategy)
    da_results = clear_market(agents, T, "DA", da_actions, config)
    print(f"  DA welfare: {da_results['welfare']:,.1f}, RE rate: {da_results['re_consumption_rate']:.1f}%, "
          f"CO2: {da_results.get('carbon_emissions',0):.1f}t, curt: {da_results.get('total_curtailment',0):.1f}MWh "
          f"({time.time()-t0:.1f}s)")

    # ---- Export CSVs + charts ----
    sc_dir = os.path.join(base_dir, f"{scenario}")
    export_curve_csvs(da_results, None, agents, config, scenario, strategy, sc_dir)

    # ---- Nash equilibrium test (quick) ----
    print(f"  Nash equilibrium test...")
    t0 = time.time()
    tester = NashEquilibriumTester(agents, config, T, stage="DA", use_optimization=False)
    is_nash, improvements = tester.test_nash_equilibrium(
        da_actions, num_variations=3)
    n_prof = sum(1 for v in improvements.values() if v.get("profitable"))
    print(f"  Nash={'Y' if is_nash else 'N'} (profitable deviations: {n_prof}) ({time.time()-t0:.1f}s)")

    # Save Nash chart + CSV
    if improvements:
        plot_nash_results(improvements, is_nash,
                          save_path=os.path.join(sc_dir, "nash_chart.png"),
                          title=f"Nash Eq. Test — {scenario}")
        import pandas as pd
        nash_rows = []
        for name, imp in improvements.items():
            nash_rows.append({
                "agent": name, "is_prosumer": imp.get("is_prosumer", False),
                "base_payoff": imp.get("base_payoff", np.nan),
                "best_payoff": imp.get("best_payoff", np.nan),
                "gain_cny": imp.get("gain", np.nan),
                "rel_gain": imp.get("rel_gain", np.nan),
                "profitable_deviation": imp.get("profitable", False),
            })
        pd.DataFrame(nash_rows).to_csv(
            os.path.join(sc_dir, "nash_test.csv"), index=False, float_format="%.4f")

print(f"\n{'='*60}")
print(f"  Batch complete — {len(SCENARIOS)} scenarios")
print(f"  Output: {base_dir}")
print(f"{'='*60}")
