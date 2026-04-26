# nash.py
"""纳什均衡求解器（并行版）"""
import numpy as np
import copy
from multiprocessing import Pool
from typing import Dict, List, Tuple
from models import Agent, MarketConfig
from market import clear_market

def _evaluate_payoffs(args):
    """多进程目标函数：评估单个策略变体的收益"""
    agents_copy, config, T, stage, action_variant, target_name = args
    try:
        result = clear_market(agents_copy, T, stage, action_variant, config)
    except Exception:
        return target_name, -1e12
    sched = result["schedules"][target_name]
    lmp = result["lmp"]
    bus = next(a.bus for a in agents_copy if a.name == target_name)
    node_price = lmp[:, bus]
    sell = np.sum(sched["p_sell"] * node_price)
    buy = np.sum(sched["p_buy"] * node_price)
    penalty = np.sum(sched["unserved"] * config.penalty_unserved)
    payoff = sell - buy - penalty
    return target_name, float(payoff)

class NashEquilibriumTester:
    def __init__(self, agents, config, T=96, stage="DA"):
        self.agents = agents
        self.config = config
        self.T = T
        self.stage = stage

    def _generate_strategy_variations(self, base_strategy, agent_name, num_variations=5):
        agent = next(a for a in self.agents if a.name == agent_name)
        variants = []
        base_bid = np.array(base_strategy[agent_name]["bid_mult"])
        if agent.is_prosumer:
            base_offer = np.array(base_strategy[agent_name]["offer_adder"])
        for _ in range(num_variations):
            new_strat = copy.deepcopy(base_strategy)
            if agent.is_prosumer:
                bid_mult = np.clip(np.random.normal(base_bid, 0.03), *self.config.bid_mult_range)
                offer_adder = np.clip(np.random.normal(base_offer, 2.0), *self.config.offer_adder_range)
                new_strat[agent_name] = {"bid_mult": bid_mult, "offer_adder": offer_adder}
            else:
                bid_mult = np.clip(np.random.normal(base_bid, 0.03), *self.config.bid_mult_range)
                new_strat[agent_name] = {"bid_mult": bid_mult}
            variants.append(new_strat)
        return variants

    def _find_best_response(self, base_strategy, agent_name, num_variations=5):
        variants = self._generate_strategy_variations(base_strategy, agent_name, num_variations)
        # 多进程评估
        tasks = [(copy.deepcopy(self.agents), self.config, self.T, self.stage, var, agent_name) for var in variants]
        with Pool(processes=2) as pool:
            results = pool.map(_evaluate_payoffs, tasks)
        best_pay = -1e12
        best_strat = copy.deepcopy(base_strategy[agent_name])
        for (name, payoff), var in zip(results, variants):
            if name == agent_name and payoff > best_pay:
                best_pay = payoff
                best_strat = var[agent_name]
        base_pay = next(payoff for name, payoff in results if name == agent_name)  # 粗略基准
        return best_strat, best_pay

    def iter_fictitious_play(self, init_strategy, max_iter=10, num_variations=5, alpha=0.3, tol_relative=0.001):
        current = copy.deepcopy(init_strategy)
        prev_avg_pay = None
        for it in range(max_iter):
            print(f"\n========== 虚构博弈迭代 {it+1}/{max_iter} ==========")
            payoffs = {}
            for a in self.agents:
                _, pay = self._find_best_response(current, a.name, num_variations)
                payoffs[a.name] = pay
            avg_pay = np.mean(list(payoffs.values()))
            print(f"当前平均收益: {avg_pay:.2f}")
            # 同步更新策略
            for a in self.agents:
                name = a.name
                br_strat, _ = self._find_best_response(current, name, num_variations)
                if a.is_prosumer:
                    current[name]["bid_mult"] = (1-alpha)*current[name]["bid_mult"] + alpha*br_strat["bid_mult"]
                    current[name]["offer_adder"] = (1-alpha)*current[name]["offer_adder"] + alpha*br_strat["offer_adder"]
                else:
                    current[name]["bid_mult"] = (1-alpha)*current[name]["bid_mult"] + alpha*br_strat["bid_mult"]
            if prev_avg_pay and abs(avg_pay-prev_avg_pay)/(abs(prev_avg_pay)+1e-6) < tol_relative:
                print("收敛")
                break
            prev_avg_pay = avg_pay
        return current, it+1

    def test_nash_equilibrium(self, base_strategy, threshold=30.0, num_variations=5):
        print("\n纳什检验...")
        is_nash = True
        improvements = {}
        for a in self.agents:
            br, best_pay = self._find_best_response(base_strategy, a.name, num_variations)
            # 简单比较（并行已评估过）
            base_pay_res = _evaluate_payoffs((copy.deepcopy(self.agents), self.config, self.T, self.stage, base_strategy, a.name))
            base_pay = base_pay_res[1] if base_pay_res[0]==a.name else -1e6
            if best_pay > base_pay + threshold:
                improvements[a.name] = {"gain": best_pay - base_pay}
                is_nash = False
        return is_nash, improvements