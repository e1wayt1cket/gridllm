# run.py
"""全功能测试入口，支持 LinDistFlow / 滚动 RT / 并行纳什"""
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
    print(f"OPF模式: {config.opf_mode}, 日前社会福利: {da_results['welfare']:.2f}, 可再生消纳率: {da_results['re_consumption_rate']:.1f}%")
    print(f"碳排放: {carbon_em:.1f} tCO2, 碳强度: {carbon_int:.3f} tCO2/MWh, 弃电量: {curtail:.1f} MWh")

    if run_nash:
        tester = NashEquilibriumTester(agents, config, T, stage="DA")
        is_nash, _ = tester.test_nash_equilibrium(da_actions, threshold=30.0, num_variations=20)
        if not is_nash:
            nash_strat, iters = tester.iter_fictitious_play(da_actions, max_iter=nash_iters, num_variations=20)
            final_nash, _ = tester.test_nash_equilibrium(nash_strat, threshold=30.0)
            print(f"纳什均衡: {final_nash}, 迭代次数: {iters}")

if __name__ == "__main__":
    main()