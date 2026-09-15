"""A rolling window slices an agent's profiles, not the participant.

Installed capacity and participant type are read inside the clearing solver and
by the settlement ledger, so an agent that loses them when a window is cut is
not a shorter version of itself: it is a different participant with an inverter
that cannot supply reactive power and, potentially, someone else's books.
"""

import numpy as np
import pytest

from market import _make_window_agents
from models import MarketConfig
from scenarios import get_scenario


@pytest.fixture(scope="module")
def agents():
    agents, _ = get_scenario("typical_day", 96, MarketConfig(opf_mode="socp"))
    return agents


def test_installed_capacity_survives_the_slice(agents):
    window = _make_window_agents(agents, 8, 24, {}, {})
    for original, sliced in zip(agents, window):
        assert sliced.pv_capacity == original.pv_capacity, \
            f"{sliced.name} lost its PV capacity in the window"
        assert sliced.wind_capacity == original.wind_capacity, \
            f"{sliced.name} lost its wind capacity in the window"

    # The inverter limit is the sum of the two, so it has to be non-zero for
    # the agents that had one; zero here is what lets a window curtail output
    # that the whole-day clear dispatches.
    original_max = [a.pv_capacity + a.wind_capacity for a in agents]
    window_max = [a.pv_capacity + a.wind_capacity for a in window]
    assert window_max == original_max
    assert any(v > 0 for v in window_max), \
        "no windowed agent has an inverter rating, so the check is vacuous"


def test_participant_type_survives_the_slice(agents):
    window = _make_window_agents(agents, 0, 16, {}, {})
    fleet = [a for a in window if a.participant_type == "storage"]
    assert {a.name for a in fleet} == {"ESS6", "ESS20", "ESS24", "ESS31"}, \
        "the storage fleet lost its participant type in the window"


def test_an_agent_without_a_declared_type_settles_as_a_prosumer():
    """The fallback is what the payoff registry already assumes for an agent
    that predates the field, so slicing must not turn it into something else."""
    from models import Agent

    legacy = Agent(
        name="LEGACY", bus=3, is_prosumer=True,
        load_forecast=np.ones(24), pv_forecast=np.zeros(24),
        load_real=np.ones(24), pv_real=np.zeros(24),
        bid_value=400.0, offer_cost=180.0,
    )
    del legacy.participant_type

    window = _make_window_agents([legacy], 0, 8, {}, {})
    assert window[0].participant_type == "prosumer"
    assert window[0].load_forecast.shape == (8,)
