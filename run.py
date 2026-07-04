# run.py
"""Full-feature test entry: LinDistFlow / rolling RT / parallel Nash.

Single scenario:
  python run.py --scenario high_re --multi-scale --rt-forecast-mode noisy_da
  python run.py --scenario high_re --nash --nash-method fictitious_play --nash-iters 10

Batch export (4 scenarios + Nash + curve CSVs):
  python run.py --export-all
"""
import os, sys, warnings
from ortools.linear_solver import pywraplp  # preload before numpy to avoid DLL order conflict
import numpy as np
import pandas as pd
from datetime import datetime
from models import MarketConfig
from market import clear_market, adaptive_bidding, two_settlement, clear_rt_rolling_mpc
from nash import NashEquilibriumTester, plot_nash_results
from scenarios import get_scenario
from outputs import save_run_results, export_curve_csvs


# Default export scenarios
EXPORT_SCENARIOS = ["baseline", "high_re", "peak_load", "congestion"]


def _run_single_scenario(scenario_name, strategy_name, config, run_nash=False,
                         nash_method="diagonalization", nash_iters=5,
                         nash_variations=8, multi_scale=False):
    """Run one scenario end-to-end. Returns (agents, da_results, rt_results,
    da_actions, is_nash, improvements)."""
    T = 96
    agents, _ = get_scenario(scenario_name, T=T, config=config)

    da_actions = adaptive_bidding(agents, config, strategy=strategy_name)
    da_results = clear_market(agents, T, "DA", da_actions, config)

    if multi_scale:
        rt_results = clear_rt_rolling_mpc(agents, T, da_actions, config)
    else:
        rt_actions = adaptive_bidding(agents, config, strategy=strategy_name)
        rt_results = clear_market(agents, T, "RT", rt_actions, config)

    is_nash = None
    improvements = None

    if run_nash:
        method_map = {
            "diagonalization": "diagonalization",
            "jacobi": "jacobi",
            "fp": "iter_fictitious_play",
            "fictitious_play": "iter_fictitious_play",
        }
        method_func = method_map.get(nash_method, "diagonalization")

        tester = NashEquilibriumTester(agents, config, T, stage="DA",
                                         use_optimization=False)
        is_nash, improvements = tester.test_nash_equilibrium(
            da_actions, num_variations=nash_variations)

        if not is_nash:
            solver = getattr(tester, method_func)
            nash_strat, iters = solver(da_actions, max_iter=nash_iters, num_variations=nash_variations)
            final_nash, final_improvements = tester.test_nash_equilibrium(
                nash_strat, num_variations=nash_variations)
            is_nash = final_nash
            improvements = final_improvements

    return agents, da_results, rt_results, da_actions, is_nash, improvements


def _print_scenario_summary(scenario, da_results, rt_results, config, is_nash, improvements):
    """Print one-line KPI summary for a scenario."""
    n_prof = sum(1 for v in (improvements or {}).values() if v.get("profitable"))
    nash_str = f"Nash={'Y' if is_nash else 'N'}(prof={n_prof})" if is_nash is not None else "Nash=skip"
    print(f"  {scenario:16s} | W={da_results['welfare']:>10,.1f} | "
          f"RE={da_results['re_consumption_rate']:>5.1f}% | "
          f"CO2={da_results.get('carbon_emissions',0):>6.1f}t | "
          f"Curt={da_results.get('total_curtailment',0):>6.1f}MWh | "
          f"{nash_str}")


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(1)
    args = sys.argv[1:]
    scenario_name = "baseline"
    strategy_name = "rl"
    config = MarketConfig(opf_mode="socp", verbose=False)
    run_nash = False
    nash_iters = 5
    nash_method = "diagonalization"
    nash_variations = 150
    multi_scale = False
    export_all = False

    for i, arg in enumerate(args):
        if arg in ("--help", "-h"):
            print(__doc__)
            print("Options:")
            print("  --scenario <name>        baseline | high_re | peak_load | congestion | re_ramp_drop | re_ramp_surge")
            print("  --strategy <name>        rl | stackelberg (default: rl)")
            print("  --opf-mode <mode>        dc | lindistflow")
            print("  --nash                   enable Nash equilibrium search")
            print("  --nash-method <name>     diagonalization | jacobi | fictitious_play")
            print("  --nash-iters <N>         max Nash iterations (default 5)")
            print("  --nash-variations <N>    variations per agent for Nash test (default 8)")
            print("  --multi-scale            use MPC rolling RT with multi-period OPF")
            print("  --rt-forecast-mode <m>   perfect | da_as_forecast | noisy_da")
            print("  --rt-forecast-noise <p>  noise std as %% of DA price (default 10)")
            print("  --da-rolling             enable rolling-horizon DA (limits storage foresight)")
            print("  --da-window-len <N>      DA window length in periods (default 24)")
            print("  --da-window-step <N>     DA window step in periods (default 8)")
            print("  --da-forecast-noise <p>  noise std as % of DA price (default 0)")
            print("  --storage-self-schedule   enable MPC self-scheduling (default: on)")
            print("  --no-storage-self-schedule  disable, use batch OPF for storage")
            print("  --export-all             batch export 4 scenarios + Nash + curve CSVs")
            return
        elif arg == "--scenario" and i+1 < len(args):
            scenario_name = args[i+1]
        elif arg == "--strategy" and i+1 < len(args):
            strategy_name = args[i+1]
        elif arg == "--opf-mode" and i+1 < len(args):
            config.opf_mode = args[i+1]
        elif arg == "--nash":
            run_nash = True
        elif arg == "--nash-method" and i+1 < len(args):
            nash_method = args[i+1]
        elif arg == "--nash-iters" and i+1 < len(args):
            nash_iters = int(args[i+1])
        elif arg == "--nash-variations" and i+1 < len(args):
            nash_variations = int(args[i+1])
        elif arg == "--multi-scale":
            multi_scale = True
        elif arg == "--rt-forecast-mode" and i+1 < len(args):
            config.rt_forecast_mode = args[i+1]
        elif arg == "--rt-forecast-noise" and i+1 < len(args):
            config.rt_forecast_noise_pct = float(args[i+1])
        elif arg == "--da-rolling":
            config.da_rolling_enabled = True
        elif arg == "--da-window-len" and i+1 < len(args):
            config.da_window_length = int(args[i+1])
        elif arg == "--da-window-step" and i+1 < len(args):
            config.da_window_step = int(args[i+1])
        elif arg == "--da-forecast-noise" and i+1 < len(args):
            config.da_forecast_noise_pct = float(args[i+1])
        elif arg == "--storage-self-schedule":
            config.storage_self_schedule = True
        elif arg == "--no-storage-self-schedule":
            config.storage_self_schedule = False
        elif arg == "--export-all":
            export_all = True

    valid_nash_methods = {"diagonalization", "jacobi", "fictitious_play", "fp"}
    if nash_method not in valid_nash_methods:
        print(f"Invalid --nash-method '{nash_method}'. Must be one of: {', '.join(sorted(valid_nash_methods))}")
        return

    # ---- batch export mode ----
    if export_all:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = os.path.join("exports", ts)
        os.makedirs(base_dir, exist_ok=True)

        all_kpis = []

        for scenario in EXPORT_SCENARIOS:
            print(f"\n{'='*50}")
            print(f"  Scenario: {scenario}")
            print(f"{'='*50}")

            agents, da_results, rt_results, da_actions, is_nash, improvements = \
                _run_single_scenario(scenario, strategy_name, config,
                                     run_nash=True, nash_method=nash_method,
                                     nash_iters=nash_iters,
                                     nash_variations=nash_variations,
                                     multi_scale=multi_scale)

            _print_scenario_summary(scenario, da_results, rt_results, config,
                                    is_nash, improvements)

            # Export per-scenario curve CSVs
            sc_dir = os.path.join(base_dir, scenario)
            kpi = export_curve_csvs(da_results, rt_results, agents, config,
                                    scenario, strategy_name, sc_dir,
                                    nash_improvements=improvements, is_nash=is_nash)
            all_kpis.append(kpi)

            # Nash chart
            if improvements:
                plot_nash_results(improvements, is_nash,
                                  save_path=os.path.join(sc_dir, "nash_chart.png"),
                                  title=f"Nash Eq. Test — {scenario}")

        # ---- cross-scenario comparison CSV ----
        if all_kpis:
            df_all = pd.DataFrame(all_kpis)
            df_all.to_csv(os.path.join(base_dir, "all_scenarios_comparison.csv"),
                          index=False, float_format="%.4f")

        print(f"\n{'='*50}")
        print(f"  Batch export complete — {len(EXPORT_SCENARIOS)} scenarios")
        print(f"  Output: {base_dir}")
        print(f"{'='*50}")
        return

    # ---- single-scenario mode ----
    T = 96
    agents, _ = get_scenario(scenario_name, T=T, config=config)

    da_actions = adaptive_bidding(agents, config, strategy=strategy_name)
    da_results = clear_market(agents, T, "DA", da_actions, config)

    if multi_scale:
        rt_results = clear_rt_rolling_mpc(agents, T, da_actions, config)
    else:
        rt_actions = adaptive_bidding(agents, config, strategy=strategy_name)
        rt_results = clear_market(agents, T, "RT", rt_actions, config)

    payment, _ = two_settlement(agents, da_results, rt_results)

    carbon_em = da_results.get('carbon_emissions', 0)
    carbon_int = da_results.get('carbon_intensity', 0)
    curtail = da_results.get('total_curtailment', 0)
    print(f"OPF mode: {config.opf_mode}")
    if config.da_rolling_enabled:
        print(f"DA rolling: window={config.da_window_length}, step={config.da_window_step}, noise={config.da_forecast_noise_pct}%")
    if config.storage_self_schedule:
        print(f"Storage: self-scheduled via MPC (horizon={config.storage_mpc_horizon})")
    if multi_scale:
        print(f"RT forecast mode: {config.rt_forecast_mode}, noise: {config.rt_forecast_noise_pct}%")
    print(f"DA welfare: {da_results['welfare']:.2f}, RE rate: {da_results['re_consumption_rate']:.1f}%")
    print(f"carbon: {carbon_em:.1f} tCO2, intensity: {carbon_int:.3f} tCO2/MWh, curtailment: {curtail:.1f} MWh")
    if rt_results:
        print(f"RT welfare: {rt_results['welfare']:.2f}, RE rate: {rt_results['re_consumption_rate']:.1f}%, "
              f"carbon: {rt_results.get('carbon_emissions', 0):.1f} tCO2")

    is_nash = None
    improvements = None

    if run_nash:
        method_map = {
            "diagonalization": "diagonalization",
            "jacobi": "jacobi",
            "fp": "iter_fictitious_play",
            "fictitious_play": "iter_fictitious_play",
        }
        method_func = method_map.get(nash_method, "diagonalization")
        print(f"\nNash method: {method_func}")

        tester = NashEquilibriumTester(agents, config, T, stage="DA",
                                         use_optimization=False)
        is_nash, improvements = tester.test_nash_equilibrium(
            da_actions, num_variations=nash_variations)
        plot_nash_results(improvements, is_nash, save_path="nash_test_initial.png",
                          title=f"Nash Eq. Test — Initial ({strategy_name})")
        if not is_nash:
            solver = getattr(tester, method_func)
            nash_strat, iters = solver(da_actions, max_iter=nash_iters, num_variations=nash_variations)
            final_nash, final_improvements = tester.test_nash_equilibrium(
                nash_strat, num_variations=nash_variations)
            plot_nash_results(final_improvements, final_nash,
                              save_path="nash_test_final.png",
                              title=f"Nash Eq. Test — After {nash_method} ({iters} iters)")
            print(f"Nash equilibrium: {final_nash}, iterations: {iters}")
            is_nash = final_nash
            improvements = final_improvements

    save_run_results(da_results, rt_results, config, scenario_name, strategy_name,
                     nash_improvements=improvements, is_nash=is_nash)

if __name__ == "__main__":
    main()