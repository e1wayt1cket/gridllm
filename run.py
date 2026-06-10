# run.py
"""Full-feature test entry: LinDistFlow / rolling RT / parallel Nash.

Nash methods:
  --nash                  enable Nash equilibrium search
  --nash-method <name>    diagonalization (default) | jacobi | fictitious_play
  --nash-iters <N>        max iterations (default 5)

Example:
  python run.py --scenario high_re --nash --nash-method fictitious_play --nash-iters 10
"""
import sys, warnings, numpy as np
from models import MarketConfig
from market import clear_market, adaptive_bidding, two_settlement, clear_rt_rolling
from nash import NashEquilibriumTester
from scenarios import get_scenario

def main():
    warnings.filterwarnings("ignore")
    np.random.seed(1)
    args = sys.argv[1:]
    scenario_name = "baseline"
    strategy_name = "random"
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    run_nash = False
    nash_iters = 5
    nash_method = "diagonalization"
    multi_scale = False

    for i, arg in enumerate(args):
        if arg == "--scenario" and i+1 < len(args):
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
        elif arg == "--multi-scale":
            multi_scale = True

    T = 96
    agents, _ = get_scenario(scenario_name, T=T)

    da_actions = adaptive_bidding(agents, config, strategy=strategy_name)
    da_results = clear_market(agents, T, "DA", da_actions, config)

    if multi_scale:
        rt_results = clear_rt_rolling(agents, T, da_actions, config)
    else:
        rt_actions = adaptive_bidding(agents, config, strategy=strategy_name)
        rt_results = clear_market(agents, T, "RT", rt_actions, config)

    payment = two_settlement(agents, da_results, rt_results)

    carbon_em = da_results.get('carbon_emissions', 0)
    carbon_int = da_results.get('carbon_intensity', 0)
    curtail = da_results.get('total_curtailment', 0)
    mode_str = "constraint" if config.use_constraint_multi_obj else "weighted-sum"
    print(f"multi-obj method: {mode_str}, OPF mode: {config.opf_mode}")
    print(f"DA welfare: {da_results['welfare']:.2f}, RE rate: {da_results['re_consumption_rate']:.1f}%")
    print(f"carbon: {carbon_em:.1f} tCO2, intensity: {carbon_int:.3f} tCO2/MWh, curtailment: {curtail:.1f} MWh")
    if config.use_constraint_multi_obj:
        print(f"constraints: carbon <= {config.carbon_cap_tco2} tCO2, RE rate >= {config.re_min_rate}%")
        sp = da_results.get('shadow_prices', {})
        if sp:
            for k, v in sp.items():
                print(f"  shadow price {k}: {v:.2f}")

    if run_nash:
        method_map = {
            "diagonalization": "diagonalization",
            "jacobi": "jacobi",
            "fp": "iter_fictitious_play",
            "fictitious_play": "iter_fictitious_play",
        }
        method_func = method_map.get(nash_method, "diagonalization")
        print(f"\nNash method: {method_func}")

        tester = NashEquilibriumTester(agents, config, T, stage="DA")
        is_nash, _ = tester.test_nash_equilibrium(da_actions, threshold=30.0, num_variations=20)
        if not is_nash:
            solver = getattr(tester, method_func)
            nash_strat, iters = solver(da_actions, max_iter=nash_iters, num_variations=20)
            final_nash, _ = tester.test_nash_equilibrium(nash_strat, threshold=30.0)
            print(f"Nash equilibrium: {final_nash}, iterations: {iters}")

if __name__ == "__main__":
    main()