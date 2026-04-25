import numpy as np
import copy
import sys
from typing import Dict, List, Tuple
from models import Agent, MarketConfig
from market import clear_market, adaptive_bidding, extract_lmp_history


class NashEquilibriumTester:
    """纳什均衡检验器（支持节点LMP & 分时段策略）"""

    def __init__(self, agents: List[Agent], config: MarketConfig, T: int = 96, stage: str = "DA"):
        self.agents = agents
        self.config = config
        self.T = T
        self.stage = stage
        self.bus_of_agent = {a.name: a.bus for a in agents}

    def _calculate_agent_payoff(self, action_params: Dict, target_agent: str = None) -> Dict[str, float]: #type: ignore
        """使用节点LMP计算收益"""
        try:
            result = clear_market(self.agents, self.T, self.stage, action_params, self.config)
        except Exception as e:
            print(f"⚠️ 出清失败: {e}")
            return {}
        lmp = result["lmp"]   # (T, n_buses)
        payoffs = {}
        for a in self.agents:
            if target_agent and a.name != target_agent:
                continue
            sch = result["schedules"][a.name]
            node_price = lmp[:, a.bus]
            sell = np.sum(sch["p_sell"] * node_price)
            buy = np.sum(sch["p_buy"] * node_price)
            penalty = np.sum(sch["unserved"] * self.config.penalty_unserved)
            payoffs[a.name] = sell - buy - penalty
        return payoffs

    def _generate_strategy_variations(self, base_strategy: Dict, agent_name: str, num_variations: int = 6) -> List[Dict]:
        """分时段扰动（数组形式）"""
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if not agent:
            return []
        variations = []
        for _ in range(num_variations):
            new_strat = copy.deepcopy(base_strategy)
            if agent.is_prosumer:
                bm = np.clip(np.random.normal(base_strategy[agent_name]["bid_mult"], 0.02),
                             *self.config.bid_mult_range)
                oa = np.clip(np.random.normal(base_strategy[agent_name]["offer_adder"], 1.0),
                             *self.config.offer_adder_range)
                new_strat[agent_name] = {"bid_mult": bm, "offer_adder": oa}
            else:
                bm = np.clip(np.random.normal(base_strategy[agent_name]["bid_mult"], 0.02),
                             *self.config.bid_mult_range)
                new_strat[agent_name] = {"bid_mult": bm}
            variations.append(new_strat)
        return variations

    def test_nash_equilibrium(self, base_strategy: Dict, threshold: float = 30.0, num_vars: int = 6) -> Tuple[bool, Dict]:
        print("\n" + "="*60)
        print("       纳什均衡检验")
        print("="*60)
        base_payoffs = self._calculate_agent_payoff(base_strategy)
        if not base_payoffs:
            return False, {}
        is_nash = True
        improvements = {}
        for a in self.agents:
            name = a.name
            base = base_payoffs.get(name, -99999)
            print(f"\n🔍 {name}  基准收益: {base:.2f}")
            vars_list = self._generate_strategy_variations(base_strategy, name, num_vars)
            best_pay = base
            best_strat = None
            for i, vs in enumerate(vars_list):
                p = self._calculate_agent_payoff(vs, name).get(name, base)
                if p > best_pay + threshold:
                    best_pay = p
                    best_strat = vs[name]
            if best_strat:
                gain = best_pay - base
                print(f"   ❌ 可提升: +{gain:.2f}")
                improvements[name] = {"strategy": best_strat, "gain": gain, "base_pay": base, "best_pay": best_pay}
                is_nash = False
            else:
                print(f"   ✅ 无显著改进")
        print("\n" + "="*60)
        print("✅ 达到纳什均衡" if is_nash else "❌ 未达纳什均衡")
        return is_nash, improvements

    def iter_approx_nash(self, init_strategy: Dict, max_iter: int = 4, threshold: float = 30.0, num_vars: int = 6) -> Tuple[Dict, int]:
        current = copy.deepcopy(init_strategy)
        for it in range(max_iter):
            print(f"\n========== 迭代 {it+1}/{max_iter} ==========")
            is_nash, impr = self.test_nash_equilibrium(current, threshold, num_vars)
            if is_nash:
                print(f"🎉 第{it+1}轮已达近似均衡")
                return current, it+1
            if not impr:
                break
            best = max(impr.keys(), key=lambda k: impr[k]["gain"])
            current[best] = impr[best]["strategy"]
            print(f"🔁 更新 {best}")
        return current, max_iter


# ========================= 测试入口 =========================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(1)

    from grid import build_base_network, create_agents_from_network

    config = MarketConfig(verbose=False, use_ac_opf=False, lambda_re=0.0, w_re_consume=0.0)
    T = 24 * 4

    net = build_base_network(config)
    agents = create_agents_from_network(net, T, with_wind=True)

    # 初始分时段策略 + 节点电价反馈
    history = None
    init_strat = adaptive_bidding(agents, config, "best_response", market_history=history, T=T)
    print("初始策略：best_response (无历史电价)")

    tester = NashEquilibriumTester(agents, config, T)

    # 第一轮检验
    is_nash, _ = tester.test_nash_equilibrium(init_strat, threshold=30)

    if not is_nash:
        print("\n===== 开始迭代逼近均衡 =====")
        # 第一轮出清后采集历史 LMP
        result = clear_market(agents, T, "DA", init_strat, config)
        history = {'lmp_per_agent': extract_lmp_history(result, agents)}

        # 使用历史电价重新生成策略（节点反馈）
        updated_strat = adaptive_bidding(agents, config, "best_response", market_history=history, T=T)
        nash_strat, it = tester.iter_approx_nash(updated_strat, max_iter=3)

        print("\n===== 最终检验 =====")
        final_ok, _ = tester.test_nash_equilibrium(nash_strat)
        print(f"\n迭代次数: {it} | 最终是否纳什: {final_ok}")