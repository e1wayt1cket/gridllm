import numpy as np
import copy
import sys
from typing import Dict, List, Tuple

# 修正导入路径：使用实际存在的模块
from models import Agent, MarketConfig
from market import clear_market
from grid import build_base_network, create_agents_from_network
from market import adaptive_bidding  # 导入策略函数调度器


class NashEquilibriumTester:
    """纳什均衡检验器：验证智能体策略是否满足纳什均衡"""

    def __init__(
        self,
        agents: List[Agent],
        config: MarketConfig,
        T: int = 96,
        stage: str = "DA"
    ):
        self.agents = agents
        self.config = config
        self.T = T
        self.stage = stage

    def _calculate_agent_payoff(
        self,
        action_params: Dict[str, Dict],
        target_agent: str = None #type: ignore
    ) -> Dict[str, float]:
        """计算智能体收益：售电收入 - 购电成本 - 缺电惩罚"""
        try:
            result = clear_market(
                agents=self.agents,
                T=self.T,
                stage=self.stage,
                action_params=action_params,
                config=self.config
            )
        except Exception as e:
            print(f"⚠️ 市场出清失败，跳过该策略：{str(e)[:80]}")
            sys.stdout.flush()
            return {}

        payoffs = {}
        for agent in self.agents:
            if target_agent and agent.name != target_agent:
                continue

            sch = result["schedules"][agent.name]
            price = result["price"]

            sell = np.sum(sch["p_sell"] * price)
            buy = np.sum(sch["p_buy"] * price)
            penalty = np.sum(sch["unserved"] * self.config.penalty_unserved)

            payoffs[agent.name] = sell - buy - penalty

        return payoffs

    def _generate_strategy_variations(
        self,
        base_strategy: Dict[str, Dict],  # 这里是 base_strategy，不是 base_strat
        agent_name: str,
        num_variations: int = 8
    ) -> List[Dict[str, Dict]]:
        """为单个智能体生成策略扰动变体"""
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if not agent:
            return []

        variations = []
        for _ in range(num_variations):
            new_strat = copy.deepcopy(base_strategy)

            if agent.is_prosumer:
                bid_mult = np.clip(
                    np.random.normal(base_strategy[agent_name]["bid_mult"], 0.02),
                    *self.config.bid_mult_range
                )
                offer_adder = np.clip(
                    np.random.normal(base_strategy[agent_name]["offer_adder"], 1.0),
                    *self.config.offer_adder_range
                )
                new_strat[agent_name] = {
                    "bid_mult": float(bid_mult),
                    "offer_adder": float(offer_adder)
                }
            else:
                bid_mult = np.clip(
                    np.random.normal(base_strategy[agent_name]["bid_mult"], 0.02),
                    *self.config.bid_mult_range
                )
                new_strat[agent_name] = {
                    "bid_mult": float(bid_mult)
                }

            variations.append(new_strat)

        return variations

    def test_nash_equilibrium(
        self,
        base_strategy: Dict[str, Dict],
        improvement_threshold: float = 30.0,
        num_variations: int = 8
    ) -> Tuple[bool, Dict[str, Dict]]:
        """检验是否为纳什均衡：无人能通过单方面改策略显著提升收益"""
        print("\n" + "="*70)
        print("          纳什均衡检验开始")
        print("="*70)
        sys.stdout.flush()

        base_payoffs = self._calculate_agent_payoff(base_strategy)
        if not base_payoffs:
            print("❌ 基准策略计算失败，无法继续检验")
            sys.stdout.flush()
            return False, {}

        best_improvements = {}
        is_nash = True

        for agent in self.agents:
            name = agent.name
            base = base_payoffs.get(name, -99999.0)

            print(f"\n🔍 检验智能体：{name}")
            print(f"   基准收益：{base:.2f} 元")
            sys.stdout.flush()

            vars_list = self._generate_strategy_variations(
                base_strategy, name, num_variations
            )
            best_pay = base
            best_strat = None

            for i, var_strat in enumerate(vars_list):
                print(f"   测试变体 {i+1}/{len(vars_list)}...")
                sys.stdout.flush()

                p = self._calculate_agent_payoff(var_strat, name).get(name, base)
                if p > best_pay + improvement_threshold:
                    best_pay = p
                    best_strat = var_strat[name]

            if best_strat is not None:
                gain = best_pay - base
                print(f"   ❌ 可单方面提升：+{gain:.2f} 元")
                best_improvements[name] = {
                    "strategy": best_strat,
                    "gain": gain,
                    "base_pay": base,
                    "best_pay": best_pay
                }
                is_nash = False
            else:
                print(f"   ✅ 无改进空间")

        print("\n" + "="*70)
        if is_nash:
            print("✅ 当前策略 满足纳什均衡")
        else:
            print("❌ 当前策略 不满足纳什均衡")
            print("\n可改进智能体：")
            for n, v in best_improvements.items():
                print(f"  {n}：可提升 {v['gain']:.2f} 元")
        print("="*70)
        sys.stdout.flush()

        return is_nash, best_improvements

    def iter_approx_nash(
        self,
        init_strategy: Dict[str, Dict],
        max_iter: int = 4,
        threshold: float = 30.0,
        num_vars: int = 6
    ) -> Tuple[Dict[str, Dict], int]:
        """迭代逼近纳什均衡"""
        current = copy.deepcopy(init_strategy)

        for it in range(max_iter):
            print(f"\n===== 迭代逼近纳什均衡 {it+1}/{max_iter} =====")
            sys.stdout.flush()

            is_nash, impr = self.test_nash_equilibrium(current, threshold, num_vars)

            if is_nash:
                print(f"\n🎉 第 {it+1} 轮达到近似纳什均衡")
                sys.stdout.flush()
                return current, it+1

            if not impr:
                break

            best_agent = max(impr.keys(), key=lambda k: impr[k]["gain"])
            current[best_agent] = impr[best_agent]["strategy"]
            print(f"🔁 更新最优改进：{best_agent}")
            sys.stdout.flush()

        print("\n⚠️ 达到最大迭代次数，返回当前最优策略")
        sys.stdout.flush()
        return current, max_iter

# ========================= 可直接运行的测试入口 =========================
if __name__ == "__main__":
    np.random.seed(1)
    config = MarketConfig(verbose=False, use_ac_opf=False)  # 先关AC-OPF，用DC-OPF跑通
    T = 24 * 4
    
    # 使用正确的模块导入
    net = build_base_network(config)
    agents = create_agents_from_network(net, T, with_wind=True)

    # 使用 market 模块中的策略函数
    init_strat = adaptive_bidding(agents, config, strategy="random")
    print("已使用随机策略作为初始策略")
    sys.stdout.flush()

    tester = NashEquilibriumTester(agents, config, T)

    print("\n[1] 检验初始策略")
    sys.stdout.flush()
    is_nash, _ = tester.test_nash_equilibrium(init_strat, improvement_threshold=30)

    if not is_nash:
        print("\n[2] 迭代逼近纳什均衡")
        sys.stdout.flush()
        nash_strat, it = tester.iter_approx_nash(init_strat, max_iter=3)

        print("\n[3] 检验最终策略")
        sys.stdout.flush()
        final_ok, _ = tester.test_nash_equilibrium(nash_strat)

        print(f"\n✅ 完成 | 迭代次数：{it} | 最终是否纳什：{final_ok}")
        sys.stdout.flush()