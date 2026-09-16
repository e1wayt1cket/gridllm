# tests/test_action_spec_consistency.py
"""Every action producer emits exactly what its agent's spec declares.

The storage action space was renamed rather than reinterpreted, so a producer
still handing a battery `bid_mult`/`offer_adder` fails loudly in the clearing
instead of silently shading a price that no longer exists. These tests move
that failure to the producer, where it can be attributed: an arm that emits a
key its spec does not declare, or a value outside its spec's bounds, is wrong
before it ever reaches a solver.

The full-day arm test is slow (one rolling clear per arm), so it is marked and
excluded from the default run. The structural tests above it are cheap and
cover the same contract at the producer's own boundary.
"""

import numpy as np
import pytest

import baselines
from models import MarketConfig
from rl_env import BiddingEnv, DayCapture
from rl_spec import ACTION_STORAGE_QUOTE_V2, action_spec_for
from scenarios import get_scenario, typical_day_config
from strategies import STRATEGY_REGISTRY


def _typical_day(T=96):
    """The fleet and config every producer below is held to."""
    config = typical_day_config(T)
    agents, wholesale = get_scenario("typical_day", T, config)
    return agents, wholesale, config


def _storage_names(agents):
    return [a.name for a in agents
            if getattr(a, "participant_type", "") == "storage"]


def _assert_declaration(agent, actions, T, source):
    """`actions` declares exactly the spec's actions, in bounds, length T."""
    spec = action_spec_for(agent)
    declared = set(actions)
    assert declared == set(spec.action_names), (
        f"{source} declared {sorted(declared)} for {agent.name}, whose spec "
        f"{spec.name!r} declares {sorted(spec.action_names)}")
    for name in spec.action_names:
        arr = np.asarray(actions[name], dtype=float)
        assert arr.shape == (T,), (
            f"{source} gave {agent.name}.{name} shape {arr.shape}, expected "
            f"({T},)")
        lo, hi = spec.bounds[name]
        assert arr.min() >= lo - 1e-9 and arr.max() <= hi + 1e-9, (
            f"{source} gave {agent.name}.{name} a value in "
            f"[{arr.min()}, {arr.max()}], outside its spec bounds [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# Producers of whole-fleet declarations
# ---------------------------------------------------------------------------

def test_fleet_default_actions_match_every_agents_spec():
    """The truthful reference is the spec's, for every participant."""
    agents, _, config = _typical_day()
    env = BiddingEnv(agents, config)
    defaults = env._default_actions()

    assert set(defaults) == {a.name for a in agents}
    for a in agents:
        spec = action_spec_for(a)
        assert set(defaults[a.name]) == set(spec.action_names)
        for name in spec.action_names:
            assert np.allclose(defaults[a.name][name], spec.truthful[name])


def test_horizon_strategies_declare_their_own_specs_actions():
    """`run.py`'s offline strategies, held to the same contract."""
    agents, _, config = _typical_day()
    for name in ("fixed", "random"):
        actions = STRATEGY_REGISTRY[name]().formulate(agents, config, T=96)
        assert set(actions) == {a.name for a in agents}
        for a in agents:
            _assert_declaration(a, actions[a.name], 96, f"strategy {name!r}")


def test_storage_fleet_gets_the_storage_spec_and_nobody_else_does():
    """The dispatch rule is `participant_type`, not "carries a battery"."""
    from scenarios import get_scenario as _get

    for scenario in ("typical_day", "baseline"):
        config = (typical_day_config() if scenario == "typical_day"
                  else MarketConfig(opf_mode="socp"))
        agents, _ = _get(scenario, 96, config)
        for a in agents:
            spec = action_spec_for(a)
            is_storage = getattr(a, "participant_type", "prosumer") == "storage"
            assert (spec is ACTION_STORAGE_QUOTE_V2) == is_storage, (
                f"{a.name} in {scenario} is "
                f"{getattr(a, 'participant_type', 'prosumer')} but got "
                f"{spec.name!r}")


# ---------------------------------------------------------------------------
# Producers of one agent's action
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", baselines.BASELINE_NAMES)
def test_baseline_arms_emit_a_vector_of_their_specs_dimension(arm):
    """A baseline returns one vector per observation, in the spec's order."""
    policy = baselines.make_baseline_policy(arm)
    obs = np.zeros(12, dtype=np.float32)

    action = np.asarray(policy(obs), dtype=float)

    spec = ACTION_STORAGE_QUOTE_V2
    assert action.shape == (spec.dim,)
    for i, name in enumerate(spec.action_names):
        lo, hi = spec.bounds[name]
        assert lo - 1e-9 <= action[i] <= hi + 1e-9


def test_step_refuses_an_action_of_the_wrong_length():
    """A vector from the retired ontology is two long, but for a different
    two: length alone does not identify it, so the clearing checks the keys it
    is handed. This pins the shape half of that check."""
    agents, _, config = _typical_day()
    env = BiddingEnv(agents, config, rl_agent_names=_storage_names(agents))
    env.reset()

    with pytest.raises(ValueError, match="shape"):
        env.step({_storage_names(agents)[0]: np.zeros(3, dtype=np.float32)})


# ---------------------------------------------------------------------------
# End to end: each arm clears a whole day without degrading
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("arm", baselines.BASELINE_NAMES)
def test_each_arm_runs_a_full_day_without_falling_back(arm):
    """The verification the rename exists for.

    Every arm drives the fleet for a whole rolling day. A producer still using
    the retired keys would be refused by `quote_arrays`, the block would be
    recorded as a fallback, and the arm's profit would be built on zero-filled
    periods -- which is exactly the silent failure the refusal replaced.
    """
    T = 96
    agents, wholesale, config = _typical_day(T)
    rl_names = _storage_names(agents)
    policy = baselines.make_baseline_policy(arm)

    np.random.seed(0)
    env = BiddingEnv(agents, config, rl_agent_names=rl_names)
    cap = DayCapture.from_env(env)
    obs = env.reset()
    for block in range(24):
        t_start = block * 4
        obs, _rewards, done, info = env.step(
            {nm: np.asarray(policy(obs[nm]), dtype=np.float32)
             for nm in rl_names})
        # `declared` is keyed from the spec, so a missing action name would
        # raise here rather than record a zero.
        cap.add_block(env._last_result, t_start, env._last_wholesale)
        cap.add_declared(env.current_actions, t_start)
        assert env._last_result is not None
        assert not env._last_result.get("fell_back"), (
            f"arm {arm!r} fell back at block {block}")
        if done:
            break

    day = cap.finish()
    assert day["complete"], f"arm {arm!r} left the day incomplete"
    assert day["fell_backs"] == 0
    for nm in rl_names:
        _assert_declaration(
            next(a for a in agents if a.name == nm),
            {k: v for k, v in day["declared"][nm].items()}, T, f"arm {arm!r}")
