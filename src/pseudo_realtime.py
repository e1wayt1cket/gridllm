# pseudo_realtime.py
"""
Pseudo-real-time simulation: solves RT market step by step with rolling storage SOC.
Each simulation period maps to configurable real seconds (default 1s), with wall-clock sync.
"""
import sys
import time
import warnings
import numpy as np
from typing import Dict

from models import MarketConfig, Agent
from grid import build_base_network, day_ahead_price_china
from dispatch import solve_opf_gurobi, StorageConstraints
from strategies.random_bidding import RandomStrategy
from scenarios import get_scenario

warnings.filterwarnings("ignore")
np.random.seed(1)


class PseudoRealTimeSimulator:
    def __init__(self, scenario_name: str, config: MarketConfig, step_sec: float = 1.0):
        """
        step_sec: 每个仿真时段对应的真实秒数（建议 0.5 ~ 2.0）
        """
        self.config = config
        self.step_sec = step_sec
        self.scenario_name = scenario_name
        self.T = 96                      # 总时段数
        self.agents, _ = get_scenario(scenario_name, T=self.T, config=config)
        self.net = build_base_network(config)
        self.wholesale = day_ahead_price_china(self.T, agents=self.agents, config=config)
        self.action_params = RandomStrategy().formulate(self.agents, config, T=self.T)

        # 储能状态容器
        self.prev_soc: Dict[str, float] = {}
        self.prev_power: Dict[str, tuple] = {}

        # 结果存储
        self.results = {
            'lmp': np.zeros((self.T, len(self.net.bus))),
            'welfare': 0.0,
            'agent_log': []
        }

    def run(self):
        print(f"🚀 伪实时仿真启动，共 {self.T} 时段，每时段 {self.step_sec} 秒")
        print(f"OPF 模式: {self.config.opf_mode} | 场景: {self.scenario_name}")
        print("-" * 60)

        for t in range(self.T):
            tic = time.time()

            # 1. 单时段 OPF 求解（按 RT 阶段，使用真实负荷数据）
            success, lmp_t, welfare_t, agent_res, p_grid = solve_opf_gurobi(
                self.net, self.agents, t, "RT", self.prev_soc,
                self.wholesale[t], self.action_params, self.config, self.prev_power
            )

            if not success:
                print(f"⚠️ 时段 {t} 求解失败，沿用上一时段结果")
                if t > 0:
                    self.results['lmp'][t] = self.results['lmp'][t-1]
                # 跳过储能更新，继续
            else:
                self.results['lmp'][t] = lmp_t
                self.results['welfare'] += welfare_t

                # 更新储能 SOC 及功率记录
                for a in self.agents:
                    if a.storage is None:
                        continue
                    name = a.name
                    res = agent_res[name]
                    soc0 = self.prev_soc.get(name, a.storage.soc0)
                    ch, dis, new_soc = StorageConstraints.execute_dispatch(
                        a.storage, soc0, res['p_ch'], res['p_dis'], dt=0.25
                    )
                    self.prev_soc[name] = new_soc
                    self.prev_power[name] = (ch, dis)

            # 2. 打印当前时段摘要
            avg_lmp = np.mean(self.results['lmp'][t]) if success else (np.mean(self.results['lmp'][t-1]) if t>0 else 0)
            total_load = sum(a.load_real[t] for a in self.agents)
            total_re = sum(agent_res[a.name]['pv_used'] + agent_res[a.name]['wind_used']
                           for a in self.agents if success) if success else 0
            print(f"⏱ 时段 {t:02d} | 平均LMP: {avg_lmp:7.2f} ¥/MWh | "
                  f"负荷: {total_load:5.2f} MW | 可再生: {total_re:5.2f} MW | "
                  f"耗时: {time.time()-tic:.3f}s")

            # 3. 墙钟等待（伪实时）
            elapsed = time.time() - tic
            wait = self.step_sec - elapsed
            if wait > 0:
                time.sleep(wait)

        print("-" * 60)
        print(f"✅ 仿真完成，累计社会福利: {self.results['welfare']:.2f} ¥")

    def get_final_results(self):
        """返回与 clear_market 类似结构的字典，便于后续分析"""
        return {
            "lmp": self.results['lmp'],
            "welfare": self.results['welfare'],
            "prev_soc": self.prev_soc,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="配电网电力市场伪实时仿真")
    parser.add_argument("--scenario", type=str, default="baseline", help="场景名称")
    parser.add_argument("--opf-mode", type=str, default="lindistflow", choices=["dc", "lindistflow"], help="潮流模型")
    parser.add_argument("--step-sec", type=float, default=1.0, help="每时段真实秒数 (默认1秒)")
    parser.add_argument("--verbose", action="store_true", help="打印 Gurobi 日志")
    args = parser.parse_args()

    config = MarketConfig(
        opf_mode=args.opf_mode,
        verbose=args.verbose
    )

    sim = PseudoRealTimeSimulator(args.scenario, config, step_sec=args.step_sec)
    sim.run()