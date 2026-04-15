
"""
Defines 3 types of load agents and batch scenario testing routines
"""
import random
from pathlib import Path
import importlib.util
import sys

file_path = Path(__file__).parent / "agent_trading.py"
spec = importlib.util.spec_from_file_location("agent_trading", file_path)
agent_trading = importlib.util.module_from_spec(spec)
sys.modules["agent_trading"] = agent_trading
spec.loader.exec_module(agent_trading)

# 定义三类负荷


class LoadA:
    pass


class LoadB:
    pass


class LoadC:
    pass


AGENT_TYPES = [LoadA, LoadB, LoadC]


def create_agents(n_each=3):
    agents = []
    for idx, T in enumerate(AGENT_TYPES):
        for i in range(n_each):
            ag = T()
            ag.id = f"{T.__name__}_{i}"
            agents.append(ag)
    return agents


def batch_scenarios(num_cases=3):
    results = []
    for case in range(num_cases):
        agents = create_agents()
        result = agent_trading.run_market(agents) if hasattr(
            agent_trading, 'run_market') else None
        results.append({"case": case, "data": result})
    return results


if __name__ == "__main__":
    print(batch_scenarios())
