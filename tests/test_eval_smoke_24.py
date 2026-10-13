# tests/test_eval_smoke_24.py
"""Full-horizon execution-path smoke test for the per-block RL arm.

**What this is.** An execution-path verification: the real `Actor` class, the
real save/load path, the real rolling environment, all **24** blocks, on one
fixed scenario. It exercises the production shape of the `per_block/rl` arm,
which the unified-eval suite only covered with a 3-block stub.

**What this is not.** It is not performance evidence. The actor is freshly
constructed and **untrained** -- deliberately not a V3 checkpoint, which is not
a valid V4 initialisation -- so any number here describes where a random policy
happens to land, not what MATD3 can do. No claim about strategy quality may be
drawn from this file.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

import numpy as np
import pytest
import torch

import unified_eval as ue
from models import MarketConfig
from rl_spec import OBS_V4, ACTION_STORAGE_QUOTE_V2

SCENARIO = "typical_day_industry"
RL_NAMES = ["ESS9", "ESS14", "ESS17", "ESS27"]
T = 96
N_BLOCKS = 24                      # the real horizon, not a compressed one


def _bounds():
    return torch.tensor([[-1.0, -1.0], [1.0, 1.0]])


def _untrained_policies(tmp_path):
    """Fresh V4 actors, written and read back through the real serialisation.

    Going through `save_policy`/`load_policy` is the point: it is the path a
    trained checkpoint would take, and it validates the spec guard, the
    `_makerb_policy_meta` round trip and the actor reconstruction.
    """
    import rl_td3

    policies = {}
    for name in RL_NAMES:
        actor = rl_td3.Actor(OBS_V4.total_dim, ACTION_STORAGE_QUOTE_V2.dim,
                             _bounds())
        path = tmp_path / f"{name}.pt"
        rl_td3.save_policy(actor, str(path), obs_spec=OBS_V4,
                           action_spec=ACTION_STORAGE_QUOTE_V2)
        policies[name] = rl_td3.load_policy(
            str(path), OBS_V4.total_dim, _bounds(),
            obs_spec=OBS_V4, action_spec=ACTION_STORAGE_QUOTE_V2)
    return policies


@pytest.fixture(scope="module")
def config():
    return MarketConfig.from_yaml()


def _arm(policies, seed):
    return ue.ArmRef(label="rl", family="per_block", name="rl",
                     policies=policies, version=f"untrained-smoke-seed{seed}")


def _scen():
    return ue.ScenarioRef(scenario=SCENARIO, day_id=0, split="test",
                          family="fam_a")


def test_a_full_24_block_episode_runs_the_real_path(config, tmp_path):
    """Observation/action shapes, block boundaries, policy reset and output
    length must all be right on the production horizon."""
    policies = _untrained_policies(tmp_path)
    rec = ue.evaluate_arm(_arm(policies, 1), _scen(), config, seed=1,
                          rl_names=RL_NAMES, n_blocks=N_BLOCKS)
    assert rec["mode"] == "evaluation"
    assert rec["T"] == T
    assert rec["settled_clean"] is True, rec.get("failure")
    # Every RL agent is accounted for, with a finite settled profit.
    assert set(rec["per_agent_profit_cny"]) == set(RL_NAMES)
    assert all(np.isfinite(v) for v in rec["per_agent_profit_cny"].values())
    # Traded quantities cover the whole day, per unit.
    for name in RL_NAMES:
        assert name in rec["traded"]
        assert rec["traded"][name]["charge_mwh"] >= 0.0
        assert rec["traded"][name]["discharge_mwh"] >= 0.0
    # The reward channel was populated for all 24 blocks' worth of agents.
    if rec["training_reward"] is not None:
        assert set(rec["training_reward"]) == set(RL_NAMES)


def test_soc_bounds_and_the_terminal_pin_hold_over_the_whole_day(config,
                                                                 tmp_path):
    policies = _untrained_policies(tmp_path)
    rec = ue.evaluate_arm(_arm(policies, 1), _scen(), config, seed=1,
                          rl_names=RL_NAMES, n_blocks=N_BLOCKS)
    soc = rec["soc"]
    assert soc["all_feasible"] is True
    for name in RL_NAMES:
        u = soc["per_unit"][name]
        assert u["present"] is True
        assert u["within_bounds"] is True, name
        assert u["terminal_equals_initial"] is True, name
        lo, hi = u["soc_bounds"]
        assert lo <= u["soc_min_observed"] and u["soc_max_observed"] <= hi


def test_the_same_untrained_model_repeats_over_24_blocks(config, tmp_path):
    """Same model, scenario and seed must give the same actions and the same
    settlement. The clearing is deterministic and the actor is run without
    exploration noise, so this is bit-identical."""
    policies = _untrained_policies(tmp_path)
    a = ue.evaluate_arm(_arm(policies, 3), _scen(), config, seed=3,
                        rl_names=RL_NAMES, n_blocks=N_BLOCKS)
    b = ue.evaluate_arm(_arm(policies, 3), _scen(), config, seed=3,
                        rl_names=RL_NAMES, n_blocks=N_BLOCKS)
    assert a["settled_net_profit_cny"] == b["settled_net_profit_cny"]
    assert a["per_agent_profit_cny"] == b["per_agent_profit_cny"]
    assert a["traded"] == b["traded"]
    assert a["scenario_input_digest"]["combined"] == \
        b["scenario_input_digest"]["combined"]


def test_a_full_day_evaluation_does_not_move_the_model(config, tmp_path):
    """24 blocks of inference, no gradient step anywhere."""
    policies = _untrained_policies(tmp_path)
    before = {k: [p.detach().clone() for p in v.parameters()]
              for k, v in policies.items()}
    ue.evaluate_arm(_arm(policies, 1), _scen(), config, seed=1,
                    rl_names=RL_NAMES, n_blocks=N_BLOCKS)
    for k, v in policies.items():
        after = [p.detach() for p in v.parameters()]
        assert len(after) == len(before[k]), k
        for p_before, p_after in zip(before[k], after):
            assert torch.equal(p_before, p_after), k


def test_the_loaded_actors_carry_the_v4_contract(config, tmp_path):
    """The serialisation round trip is what a real checkpoint would go
    through, so the spec recorded in the file must be V4, not V3."""
    policies = _untrained_policies(tmp_path)
    for actor in policies.values():
        assert isinstance(actor, torch.nn.Module)
    # A V3-labelled file must not load into this path.
    import rl_td3
    from rl_spec import OBS_V3
    path = tmp_path / "v3.pt"
    actor = rl_td3.Actor(OBS_V3.total_dim, ACTION_STORAGE_QUOTE_V2.dim,
                         _bounds())
    rl_td3.save_policy(actor, str(path), obs_spec=OBS_V3,
                       action_spec=ACTION_STORAGE_QUOTE_V2)
    with pytest.raises(ValueError, match="observation spec"):
        rl_td3.load_policy(str(path), OBS_V4.total_dim, _bounds(),
                           obs_spec=OBS_V4,
                           action_spec=ACTION_STORAGE_QUOTE_V2)
