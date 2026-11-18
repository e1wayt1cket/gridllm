"""The v7 trainer: the four-dimension quote, its anchor and its checkpoint."""

import numpy as np
import torch
import torch.nn as nn

from checkpoint_writer_v2 import save_checkpoint
from da_quote_map_v5 import ACTION_BOX_HIGH, QUOTE_DIM, TRUTHFUL
from local_da_days import load_pair
from rl_da_env import LocalDABiddingEnv
from rl_da_gate_v7 import DAGateTrainerV7, DAV7ExperimentConfig, GateActorV5
from rl_spec_v7 import ACTION_STORAGE_QUOTE_V5, OBS_V7

HEX64 = "a" * 64


def trainer():
    env = LocalDABiddingEnv()
    return DAGateTrainerV7(env, DAV7ExperimentConfig()), env


def test_the_actor_emits_six_logits_with_a_four_wide_box():
    env = LocalDABiddingEnv()
    trainer = DAGateTrainerV7(env, DAV7ExperimentConfig())
    assert trainer.act_dim == QUOTE_DIM == 4
    assert list(trainer.actors["ESS9"](torch.zeros(1, 12)).shape) == [1, 6]
    assert isinstance(trainer.actors["ESS9"], GateActorV5)
    assert trainer.action_spec.name == "storage_quote_v5"
    assert trainer.obs_spec.name == "v7_12d"


def test_the_anchor_is_zero_at_the_declared_point_and_one_at_the_far_corner():
    t, _ = trainer()
    truth = torch.tensor([list(TRUTHFUL)], dtype=torch.float32)
    corner = torch.tensor([[ACTION_BOX_HIGH[0], 0., ACTION_BOX_HIGH[2], 0.]])
    assert t._anchor_tensor(truth).item() == 0.
    assert t._anchor_tensor(corner).item() == 4.


def test_one_day_writes_every_block_to_the_buffer():
    # A deterministic near-truthful day: the exact clearing of a window with
    # four units freely quoted on both legs can consume its whole solve
    # budget, which would measure the solver rather than the trainer. Gates
    # below the map's bound with the draw seeded keep the quotes at the
    # truthful declaration.
    torch.manual_seed(0)
    np.random.seed(0)
    env = LocalDABiddingEnv()
    t = DAGateTrainerV7(env, DAV7ExperimentConfig(
        exploration_epsilon_start=0.0, exploration_epsilon_end=0.0,
        price_noise_std=0.0, gate_temperature_start=1.0,
        gate_temperature_end=1.0))
    for actor in list(t.actors.values()) + list(t.actor_targets.values()):
        nn.init.constant_(actor.net[-1].weight, 0.0)
        nn.init.constant_(actor.net[-1].bias, -100.0)
    record = t.train_day(env, load_pair("2021-06-04"), blocks=2)
    assert len(t.buffer) == 2
    assert len(record["per_block"]["ESS9"]) == 2
    assert t.episodes == 1


class _FixedPolicy(nn.Module):
    """A policy output fixed by the test, so the map sees an extreme logit."""

    def __init__(self, row):
        super().__init__()
        self.row = torch.as_tensor([row], dtype=torch.float32)

    def forward(self, obs):
        return self.row.expand(obs.shape[0], -1)


def test_the_train_mode_map_keeps_the_quotes_in_the_box_and_the_gate_hard():
    t, _ = trainer()
    # Gate logits far beyond the map's bound: a raw sigmoid would saturate to
    # exactly 1 and 0, and the four quote logits decide price and quantity.
    policy = _FixedPolicy([100., -100., 0.5, -0.5, 0.5, -0.5])
    diag = t._policy_action(policy, np.zeros(12, dtype=np.float32),
                            train=True)
    quotes = np.asarray(diag["quotes"], dtype=float)
    assert quotes.shape == (QUOTE_DIM,)
    assert np.all(quotes >= 0.0)
    # The map runs in float32, where 0.4 is not exact; the box is compared at
    # the precision the quote is evaluated in.
    assert np.all(quotes <= np.asarray(ACTION_BOX_HIGH, dtype=float) + 1e-6)
    assert set(np.unique(diag["gates"])) <= {0.0, 1.0}
    probs = np.asarray(diag["probs"], dtype=float)
    assert np.all((probs > 0.0) & (probs < 1.0))


def test_a_v4_checkpoint_is_refused(tmp_path):
    t, _ = trainer()
    save_checkpoint(
        tmp_path, level="inference", run_id="v4", dataset="d", seed=1,
        code_revision="rev", config_hash=HEX64,
        specs={"obs_spec": {"name": "v6_12d", "version": 6},
               "action_spec": {"name": "storage_quote_v4", "version": 4}},
        actors=t.actors, actor_targets=t.actor_targets, critics=t.critics,
        critic_targets=t.critic_targets,
        optimizers={}, counters={"total_steps": 0, "episode": 0, "updates": 0},
        reward={"definition": "B_C2", "gamma": 0.99, "scale": 100.})
    try:
        t.load(tmp_path)
    except ValueError as error:
        assert "storage_quote_v5" in str(error)
    else:
        raise AssertionError("a v4 checkpoint must not load into a v7 trainer")
