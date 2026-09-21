"""Every trainer builds each policy in its own agent's action space.

The action space is a property of the participant, not of the environment: an
independent battery declares two reservation prices in CNY/MWh and a load or
prosumer declares the multiplier/adder pair its own `bid_value`/`offer_cost`
live in. The two differ in bounds and in meaning, and training one in the
other's box produces no error -- only a policy whose actions are read as a
different quantity.

These tests pin the three things that has to hold for every trainer: the
width, the box, and the point the deviation penalty is anchored to.
"""

import numpy as np
import pytest
import torch

from rl_spec import (ACTION_BID_OFFER_V1, ACTION_STORAGE_QUOTE_V2,
                     action_spec_for)


class _Agent:
    """An agent as far as action-space selection is concerned."""

    def __init__(self, name, participant_type="prosumer"):
        self.name = name
        self.participant_type = participant_type


class _StubEnv:
    """The BiddingEnv surface every trainer constructor touches.

    Delegates spec selection and box construction to the same functions the
    real environment uses, so these tests exercise the production rule.
    """

    def __init__(self, agents, obs_dim=12):
        from types import SimpleNamespace

        self.rl_agents = list(agents)
        self.all_agents = list(agents)
        self._obs_dim = obs_dim
        self.obs_spec = SimpleNamespace(total_dim=obs_dim)
        self.action_spec = ACTION_STORAGE_QUOTE_V2
        self._legacy_action_spec = ACTION_BID_OFFER_V1
        self.unique_obs_dim = 3

    def get_state_dim(self):
        return self._obs_dim

    def spec_for(self, agent):
        return action_spec_for(agent, storage_spec=self.action_spec,
                               other_spec=self._legacy_action_spec)

    def action_bounds_for(self, agent):
        from rl_env import bounds_tensor
        return bounds_tensor(self.spec_for(agent))

    def get_action_bounds(self):
        from rl_env import bounds_tensor
        return bounds_tensor(self.action_spec)

    def reset(self):
        return {a.name: np.zeros(self._obs_dim, dtype=np.float32)
                for a in self.rl_agents}

    def step(self, actions):
        """One block that immediately ends the episode.

        The diffusion trainer calibrates its reward scale by rolling the
        environment out at construction, so a stub has to be steppable; ending
        after one block keeps that to a single call.
        """
        rewards = {a.name: 1.0 for a in self.rl_agents}
        return self.reset(), rewards, True, {"clearing_objective": 0.0, "re_rate": 0.0}


def _storage_agents(n=2):
    return [_Agent(f"ESS{i}", "storage") for i in range(n)]


def _trainers(env):
    """One constructed trainer per registered algorithm, keyed by name."""
    from rl_bidding import MATD3
    from rl_diffusion import MAD3PG
    from rl_masac import MASAC
    from rl_quantile import QuantileMATD3

    return {
        "matd3": MATD3(env, batch_size=4),
        "masac": MASAC(env, batch_size=4),
        "mad3pg": MAD3PG(env, batch_size=4, diff_steps=2),
        "qmatd3": QuantileMATD3(env, batch_size=4),
    }


def test_storage_truthful_is_zero_zero():
    """A battery declares no premium as (0, 0), in CNY/MWh."""
    assert ACTION_STORAGE_QUOTE_V2.action_names == ("charge_price",
                                                    "discharge_price")
    assert np.allclose(ACTION_STORAGE_QUOTE_V2.truthful_array(), [0.0, 0.0])
    assert action_spec_for(_Agent("ESS2", "storage")) is \
        ACTION_STORAGE_QUOTE_V2


def test_a_prosumer_keeps_the_declared_price_pair():
    """Carrying a battery must not move a prosumer into the storage space.

    A prosumer's prices are its own load's value and its own generation's
    cost, which its multiplier pair shades; it has no reservation prices to
    declare. Only a battery alone does.
    """
    spec = action_spec_for(_Agent("Bus5R", "prosumer"))
    assert spec is ACTION_BID_OFFER_V1
    assert np.allclose(spec.truthful_array(), [1.0, 0.0])


@pytest.mark.parametrize("trainer", ["matd3", "masac", "mad3pg", "qmatd3"])
def test_every_trainer_reads_the_width_off_the_spec(trainer):
    env = _StubEnv(_storage_agents(3))
    t = _trainers(env)[trainer]
    assert t.act_dim == ACTION_STORAGE_QUOTE_V2.dim
    assert len(t.agent_names) == 3


@pytest.mark.parametrize("trainer", ["matd3", "masac", "mad3pg", "qmatd3"])
def test_every_trainer_builds_actors_in_the_agents_own_box(trainer):
    from rl_env import bounds_tensor

    env = _StubEnv(_storage_agents(2))
    t = _trainers(env)[trainer]
    expected = bounds_tensor(ACTION_STORAGE_QUOTE_V2)
    for nm in t.agent_names:
        assert torch.allclose(t.bounds_by_agent[nm], expected)
        actor = t.actors[nm]
        low, high = actor.action_low, actor.action_high
        assert torch.allclose(torch.stack([low, high]), expected)


@pytest.mark.parametrize("trainer", ["matd3", "masac", "mad3pg", "qmatd3"])
def test_the_deviation_reference_is_the_agents_own_truthful_point(trainer):
    """Each trainer measures deviation from that agent's spec truthful point.

    Held as the action, not as a logit: the truthful point of the storage box
    is a corner, where `atanh` is infinite, and the penalty is defined on the
    normalized action deviation anyway.
    """
    env = _StubEnv(_storage_agents(2))
    t = _trainers(env)[trainer]
    spec = ACTION_STORAGE_QUOTE_V2
    expected = torch.as_tensor(spec.truthful_array(), dtype=torch.float32)
    for nm in t.agent_names:
        assert torch.allclose(t.truthful_action[nm], expected)


@pytest.mark.parametrize("trainer", ["matd3", "masac", "mad3pg", "qmatd3"])
def test_a_mixed_population_gets_one_box_per_agent(trainer):
    """Each agent's reference comes from its own spec, not a shared default.

    A storage unit and a prosumer have the same action width here, so a
    trainer that took the reference from one environment-wide spec would load
    without complaint -- and pull the prosumer toward the battery's (0, 0).
    """
    agents = [_Agent("ESS2", "storage"), _Agent("Bus5R", "prosumer")]
    env = _StubEnv(agents)
    t = _trainers(env)[trainer]

    assert t.spec_by_agent["ESS2"] is ACTION_STORAGE_QUOTE_V2
    assert t.spec_by_agent["Bus5R"] is ACTION_BID_OFFER_V1

    storage_ref = t.truthful_action["ESS2"]
    prosumer_ref = t.truthful_action["Bus5R"]
    assert not torch.allclose(storage_ref, prosumer_ref), \
        "one shared reference would make these identical"
    assert torch.allclose(storage_ref, torch.zeros(2))
    assert torch.allclose(prosumer_ref, torch.tensor([1.0, 0.0]))

    # The prosumer's box is the multiplier/adder box, not the price box.
    from rl_env import bounds_tensor
    assert torch.allclose(t.bounds_by_agent["Bus5R"],
                          bounds_tensor(ACTION_BID_OFFER_V1))
    assert not torch.allclose(t.bounds_by_agent["Bus5R"],
                              bounds_tensor(ACTION_STORAGE_QUOTE_V2))


def test_independent_storage_charging_is_costed_in_the_objective():
    """A battery's charge leg is a cost, in whichever pass builds the objective.

    The first pass and the nodal re-solve used to write this unit's terms
    separately and disagreed on the charge sign. That is silent: the re-solve
    overwrites the objective, so the dispatch followed one sign while the
    prices were read off the other. Both passes now build it here, and this
    pins the direction.
    """
    gp = pytest.importorskip("gurobipy")
    from types import SimpleNamespace

    from dispatch_socp import _add_independent_storage

    m = gp.Model()
    m.Params.OutputFlag = 0
    T = 2
    ch = {t: m.addVar(name=f"ch{t}") for t in range(T)}
    dis = {t: m.addVar(name=f"dis{t}") for t in range(T)}
    soc_end = m.addVar(name="soc_end")

    su = SimpleNamespace(name="ESS2", bid_value=10.0, offer_cost=20.0,
                         storage=SimpleNamespace(e_max=4.0))
    obj = gp.LinExpr()
    _add_independent_storage(obj, su, {"ESS2": ch}, {"ESS2": dis},
                             {"ESS2": {T: soc_end}}, T,
                             dt_scale=0.25, disc_T=0.9,
                             terminal_value=100.0, cycle_cost=5.0)
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.update()

    expr = m.getObjective()
    coeffs = {expr.getVar(i).VarName: expr.getCoeff(i)
              for i in range(expr.size())}
    # Both legs negative: the quote is what the unit asks to be paid for the
    # direction it moves in, so charging at a declared price is a purchase.
    # Each leg carries its own quote plus the round trip's cycle cost.
    assert coeffs["ch0"] == pytest.approx(-(10.0 + 5.0) * 0.25)
    assert coeffs["dis0"] == pytest.approx(-(20.0 + 5.0) * 0.25)
    # Terminal SOC is an energy times a price, so it is not scaled by the period.
    assert coeffs["soc_end"] == pytest.approx(0.9 * 100.0 * 4.0)


def test_a_mixed_width_population_is_rejected():
    """The centralized critic concatenates actions, so widths must agree.

    There is no correct way to build the critic for a population whose actions
    are not all the same length, so this fails here rather than reshaping one
    agent's action into another's box.
    """
    from rl_bidding import agent_action_spaces, MATD3

    wide = ACTION_STORAGE_QUOTE_V2
    narrow = type(wide)(
        name="one_action_v0", version=0, action_names=("only",),
        bounds={"only": (0.0, 1.0)}, truthful={"only": 0.0})

    env = _StubEnv(_storage_agents(2))
    env.spec_for = lambda agent: (wide if agent.name == "ESS0" else narrow)

    with pytest.raises(ValueError, match="share one action width"):
        agent_action_spaces(env)
    with pytest.raises(ValueError, match="share one action width"):
        MATD3(env, batch_size=4)
