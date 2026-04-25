# run.py
"""
全功能测试入口：构建 IEEE 33 智能体 → 日前出清 → 实时出清 → 两阶段结算 → 结果摘要
支持命令行参数：
  --scenario NAME   指定场景名称 (默认: baseline)
  --strategy NAME   指定报价策略 (默认: random)
  --nash            执行纳什均衡检验
  --nash-iters N    纳什均衡迭代次数 (默认: 3)
  --wind           包含风电工业产消者
  --dc             强制使用 DC-OPF（默认）
  --green          开启可再生能源激励
  --verbose        打印详细 OPF 日志
  --summary        仅打印摘要（默认）
  --list-scenes    列出所有可用场景
"""
import sys
import warnings
import numpy as np

from models import MarketConfig
from grid import build_base_network, create_agents_from_network
from market import clear_market, adaptive_bidding  # 假设已从 agent_trading 迁移到 market 或独立
from market import two_settlement  # 假设已从 agent_trading 迁移到 market 或独立
from nash import NashEquilibriumTester  # 导入纳什均衡检验器
from scenarios import scenario_baseline, scenario_no_pv, scenario_high_re, scenario_peak_load, scenario_unbalanced, get_scenario, print_scenario_info, list_scenarios  # 假设已定义多个场景函数

def two_settlement(agents, da, rt):
    T = len(da["price"])
    payments = {}
    for a in agents:
        da_import = da["schedules"][a.name]["p_buy"] - da["schedules"][a.name]["p_sell"]
        rt_import = rt["schedules"][a.name]["p_buy"] - rt["schedules"][a.name]["p_sell"]
        da_cost = np.sum(da["price"] * da_import)
        rt_cost = np.sum(rt["price"] * (rt_import - da_import))
        payments[a.name] = float(da_cost + rt_cost)
    return payments


def calc_load_satisfaction(result, agents, stage):
    total_load = sum(np.sum(a.load_forecast if stage == "DA" else a.load_real) for a in agents)
    total_served = sum(np.sum(result["schedules"][a.name]["served"]) for a in agents)
    return (total_served / total_load * 100) if total_load > 0 else 100.0


def print_summary(da, rt, payment, agents):
    print("\n" + "=" * 70)
    print("市场出清结果")
    print("=" * 70)
    print(f"{'指标':<20} {'日前(DA)':>15} {'实时(RT)':>15}")
    print("-" * 50)
    print(f"{'社会福利(¥)':<20} {da['welfare']:>15.2f} {rt['welfare']:>15.2f}")
    print(f"{'可再生消纳率(%)':<20} {da['re_consumption_rate']:>15.1f} {rt['re_consumption_rate']:>15.1f}")
    da_load = calc_load_satisfaction(da, agents, 'DA')
    rt_load = calc_load_satisfaction(rt, agents, 'RT')
    print(f"{'负荷满足率(%)':<20} {da_load:>15.1f} {rt_load:>15.1f}")

    print("\n结算结果 (正数=成本, 负数=收益):")
    print("-" * 50)
    total = sum(payment.values())
    for name, val in payment.items():
        ptype = "成本" if val > 0 else "收益" if val < 0 else "平衡"
        print(f"{name:14s}  {val:10.2f} ¥ ({ptype})")
    print(f"{'总计':14s}  {total:10.2f} ¥")


def print_storage_snapshots(da, rt, agents):
    """打印储能 SOC 与充放电摘要"""
    print("\n储能状态摘要 (日前首/末时段 SOC 及总充放电):")
    print("-" * 70)
    for a in agents:
        if a.storage:
            s_da = da['schedules'][a.name]
            s_rt = rt['schedules'][a.name]
            da_soc0, da_soc_end = s_da['soc'][0], s_da['soc'][-1]
            rt_soc0, rt_soc_end = s_rt['soc'][0], s_rt['soc'][-1]
            total_ch = np.sum(s_da['p_ch'])
            total_dis = np.sum(s_da['p_dis'])
            print(f"🔋 {a.name} @ Bus {a.bus}: "
                  f"DA SOC {da_soc0*100:.1f}% → {da_soc_end*100:.1f}% | "
                  f"充电 {total_ch:.3f} MWh, 放电 {total_dis:.3f} MWh")
            total_ch_rt = np.sum(s_rt['p_ch'])
            total_dis_rt = np.sum(s_rt['p_dis'])
            print(f"   RT SOC {rt_soc0*100:.1f}% → {rt_soc_end*100:.1f}% | "
                  f"充电 {total_ch_rt:.3f} MWh, 放电 {total_dis_rt:.3f} MWh")


def print_snapshots(agents, da, rt):
    print("\n各智能体运行快照 (显示前 8 个):")
    print("-" * 70)
    for a in agents[:8]:
        s_da, s_rt = da["schedules"][a.name], rt["schedules"][a.name]
        print(f"\n🔹 {a.name} @ Bus {a.bus} ({a.load_type})")
        if a.is_prosumer:
            print(f"  PV 总发电: DA {np.sum(s_da['pv_used']):.2f} MWh")
        print(f"  净购电: DA {np.sum(s_da['p_buy'])-np.sum(s_da['p_sell']):.2f} MWh")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*MessageStream size changed.*")
    np.random.seed(1)

    args = sys.argv[1:]
    
    if "--list-scenes" in args:
        print_scenario_info()
        sys.exit(0)
    
    # 检查是否需要执行纳什均衡检验
    run_nash = "--nash" in args
    
    # 获取纳什迭代次数，默认为3
    nash_iterations = 3
    for i, arg in enumerate(args):
        if arg == "--nash-iters" and i + 1 < len(args):
            try:
                nash_iterations = int(args[i + 1])
            except ValueError:
                print(f"⚠️ 无效的纳什迭代次数: {args[i + 1]}, 使用默认值 3")
    
    # 获取场景名称，默认为baseline
    scenario_name = "baseline"
    for i, arg in enumerate(args):
        if arg == "--scenario" and i + 1 < len(args):
            scenario_name = args[i + 1]
            break
    
    # 获取报价策略，默认为random
    strategy_name = "random"
    for i, arg in enumerate(args):
        if arg == "--strategy" and i + 1 < len(args):
            strategy_name = args[i + 1]
            break

    config = MarketConfig(         # 若使用 Gurobi 则忽略
        verbose="--verbose" in args,
        use_ac_opf=False,
        lambda_re=50.0,    # 每消纳 1MWh 可再生奖励 50 元
    # w_re_consume 可弃用或置 0
    )
    T = 24 * 4   # 96 时段

    print("=" * 70)
    print(f"多能配网电力市场仿真 · 全功能测试 ({scenario_name})")
    print(f"策略: {strategy_name} | 纳什检验: {run_nash} | 纳什迭代: {nash_iterations} | 时段数: {T} | AC-OPF: {config.use_ac_opf} | 可再生激励: {config.lambda_re} | 详细日志: {config.verbose}")
    print("=" * 70)

    # 1. 从指定场景获取网络与智能体
    try:
        agents, wholesale = get_scenario(scenario_name, T=T)
        print(f"智能体总数: {len(agents)} | 场景: {scenario_name}")
    except ValueError as e:
        print(f"错误: {e}")
        print("\n使用 '--list-scenes' 参数查看所有可用场景。")
        sys.exit(1)

    # 2. 报价策略（使用指定策略）
    da_actions = adaptive_bidding(agents, config, strategy=strategy_name)
    rt_actions = adaptive_bidding(agents, config, strategy=strategy_name)

    # 3. 市场出清
    print("\n执行日前市场出清...")
    da_results = clear_market(agents, T, "DA", da_actions, config)
    print("执行实时市场出清...")
    rt_results = clear_market(agents, T, "RT", rt_actions, config)

    # 4. 两阶段结算
    payment = two_settlement(agents, da_results, rt_results)

    # 5. 纳什均衡检验（如果需要）
    if run_nash:
        print(f"\n执行纳什均衡检验 (最多 {nash_iterations} 次迭代)...")
        nash_tester = NashEquilibriumTester(agents, config, T, stage="DA")
        
        # 使用日前市场的策略进行纳什均衡检验
        print("正在检验初始策略是否为纳什均衡...")
        is_nash, improvements = nash_tester.test_nash_equilibrium(
            da_actions, 
            threshold=30.0, 
            num_vars=6
        )
        
        if not is_nash:
            print(f"\n初始策略不是纳什均衡，开始迭代寻找近似纳什均衡 (最多 {nash_iterations} 次)...")

            # 保存原始策略用于比较
            original_strategies = {name: strat.copy() for name, strat in da_actions.items()}
            
            nash_strat, iterations = nash_tester.iter_approx_nash(
                da_actions,
                max_iter=nash_iterations,
                threshold=30.0,
                num_vars=6
            )
            
            print(f"\n迭代完成，共执行 {iterations} 次迭代")
            
            # 对纳什均衡策略进行最终检验
            print("\n对找到的近似纳什均衡进行最终检验...")
            final_is_nash, _ = nash_tester.test_nash_equilibrium(
                nash_strat,
                threshold=30.0,
                num_vars=6
            )
            
            if final_is_nash:
                print("✅ 找到了近似的纳什均衡策略！")
                
                # 计算纳什均衡前后社会福利的变化
                print("\n正在评估纳什均衡策略的影响...")
                nash_results = clear_market(agents, T, "DA", nash_strat, config)
                print(f"初始策略社会福利: {da_results['welfare']:.2f} ¥")
                print(f"纳什均衡策略社会福利: {nash_results['welfare']:.2f} ¥")
                welfare_diff = nash_results['welfare'] - da_results['welfare']
                print(f"社会福利变化: {welfare_diff:+.2f} ¥ ({welfare_diff/da_results['welfare']*100:+.2f}%)")
            else:
                print("⚠️ 未能找到完全的纳什均衡，但找到了改进策略。")
        else:
            print("✅ 初始策略已经是纳什均衡！")

    # 6. 结果输出
    print_summary(da_results, rt_results, payment, agents)
    print_storage_snapshots(da_results, rt_results, agents)
    if "--verbose" in args:
        print_snapshots(agents, da_results, rt_results)

    print("\n✅ 全功能测试完成。")