"""Tests for best-response regret evaluation (nash.py + eval_agents)."""

import pytest

from nash import NashEquilibriumTester, compute_regret_summary


def test_compute_regret_summary_numeric():
    improvements = {
        "Bus5R": {"gain": 100.0, "regret": 100.0, "rel_gain": 0.1,
                  "base_payoff": 1000.0, "profitable": True},
        "Bus6R": {"gain": -5.0, "regret": -5.0, "rel_gain": -0.01,
                  "base_payoff": 500.0, "profitable": False},
    }
    s = compute_regret_summary(improvements)
    assert s["total_regret"] == pytest.approx(95.0)
    assert s["mean_regret"] == pytest.approx(47.5)
    assert s["max_regret"] == pytest.approx(100.0)
    assert s["n_profitable"] == 1
    assert s["regret_share"] == pytest.approx(95.0 / 1500.0)


def test_compute_regret_summary_empty():
    s = compute_regret_summary({})
    assert s["total_regret"] == 0.0
    assert s["n_profitable"] == 0


class _DummyAgent:
    def __init__(self, name, is_prosumer=False):
        self.name = name
        self.is_prosumer = is_prosumer


def test_nash_improvements_include_regret(monkeypatch):
    """test_nash_equilibrium emits regret fields without running a real OPF."""
    agents = [_DummyAgent("Bus5R"), _DummyAgent("Bus6R", is_prosumer=True)]
    tester = NashEquilibriumTester(agents, None, T=96, stage="DA")

    def fake_base(strategy):
        return {"Bus5R": 1000.0, "Bus6R": 500.0}

    def fake_best(agent, base_strategy, num_variations=150):
        best = {"Bus5R": 1100.0, "Bus6R": 800.0}[agent.name]
        return agent.name, best, {}

    monkeypatch.setattr(tester, "compute_base_payoffs", fake_base)
    monkeypatch.setattr(tester, "_best_response", fake_best)

    base = {"Bus5R": {"bid_mult": [1.0] * 96},
            "Bus6R": {"bid_mult": [1.0] * 96, "offer_adder": [0.0] * 96}}
    is_nash, improvements = tester.test_nash_equilibrium(base)
    assert is_nash is False
    assert improvements["Bus5R"]["regret"] == pytest.approx(100.0)
    assert improvements["Bus5R"]["relative_regret"] == pytest.approx(0.1)
    assert improvements["Bus6R"]["regret"] == pytest.approx(300.0)
    s = compute_regret_summary(improvements)
    assert s["total_regret"] == pytest.approx(400.0)
