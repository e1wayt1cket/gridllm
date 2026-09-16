"""Run 4 scenarios without Nash and export all curve PNGs + CSVs."""
import os, sys, warnings
from ortools.linear_solver import pywraplp  # preload to avoid DLL order conflict
import numpy as np
from datetime import datetime

from models import MarketConfig
from market import clear_market, adaptive_bidding, clear_rt_rolling_mpc
from scenarios import get_scenario
from outputs import export_curve_csvs


SCENARIOS = ["baseline", "high_re", "peak_load", "congestion"]


def run_one(scenario_name, config, output_dir):
    print(f"\n{'='*50}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*50}")

    T = 96
    agents, _ = get_scenario(scenario_name, T=T, config=config)

    da_actions = adaptive_bidding(agents, config, strategy="rl")
    da_results = clear_market(agents, T, "DA", da_actions, config)
    rt_results = clear_rt_rolling_mpc(agents, T, da_actions, config)

    sc_dir = os.path.join(output_dir, scenario_name)
    kpi = export_curve_csvs(da_results, rt_results, agents, config,
                            scenario_name, "rl", sc_dir)

    print(f"  welfare={da_results['welfare']:.1f}  "
          f"RE={da_results['re_consumption_rate']:.1f}%  "
          f"CO2={da_results.get('carbon_emissions',0):.1f}t  "
          f"curtail={da_results.get('total_curtailment',0):.1f}MWh")
    return kpi


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(1)

    config = MarketConfig(opf_mode="socp", verbose=False)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("output", ts)
    os.makedirs(output_dir, exist_ok=True)

    for sc in SCENARIOS:
        run_one(sc, config, output_dir)

    print(f"\n{'='*50}")
    print(f"  Export complete — 4 scenarios")
    print(f"  Output: {output_dir}")
    for sc in SCENARIOS:
        d = os.path.join(output_dir, sc)
        files = sorted(os.listdir(d))
        print(f"    {sc}/: {', '.join(files)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
