"""v4 mechanics: critic regularisation, anchor, bounded gates, config.

Covers the changes of `b/v4修复方案_2026-11-08.md` (W1/W2/W3a/W4/W5): the
critic carries no dropout and no weight decay, the actor loss is anchored
toward the truthful quote, gate logits are tanh-bounded so a closed gate
keeps a reopen probability and a straight-through gradient, and the v4
config defaults / schema guard hold.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import da_quote_map
import rl_da_gate as da
from rl_bidding import CentralizedCritic

NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")


def _env():
    return SimpleNamespace(
        rl_agents=[SimpleNamespace(name=n) for n in NAMES],
        get_state_dim=lambda: 12,
    )


def _trainer(**overrides):
    return da.DAGateTrainer(
        _env(), da.DAExperimentConfig("test", **overrides),
        agent_names=list(NAMES))


def _fill_buffer(trainer, size=128):
    obs = np.zeros((4, 12), dtype=np.float32)
    action = np.array([[75.0, 150.0]] * 4, dtype=np.float32)
    reward = np.zeros(4, dtype=np.float32)
    extras = {
        "gates": np.ones((4, 2), dtype=np.float32),
        "probs": np.full((4, 2), 0.5, dtype=np.float32),
        "gate_logits": np.zeros((4, 2), dtype=np.float32),
        "price_logits": np.zeros((4, 2), dtype=np.float32),
        "price_quotes": action.copy(),
        "sampled": np.ones((4, 2), dtype=np.float32),
        "explored": np.zeros((4, 2), dtype=np.float32),
        "exploration_source": np.zeros((4, 2), dtype=np.float32),
    }
    for _ in range(size):
        trainer.buffer.add(obs, action, reward, obs, 0.0, extras)


# ----------------------------------------------------------------------
# W1 — critic regularisation
# ----------------------------------------------------------------------
def test_critic_dropout_defaults_to_zero():
    critic = CentralizedCritic(n_agents=4, obs_dim=12, act_dim=2)
    drops = [m for m in critic.modules() if isinstance(m, torch.nn.Dropout)]
    assert drops, "architecture changed; this test needs updating"
    assert all(m.p == 0.0 for m in drops)


def test_critic_weight_decay_is_zero_actor_keeps_1e_5():
    trainer = _trainer()
    for nm in NAMES:
        assert trainer.critic_opts[nm].param_groups[0]["weight_decay"] == 0.0
        assert trainer.actor_opts[nm].param_groups[0]["weight_decay"] == 1e-5


# ----------------------------------------------------------------------
# W4 / W5 / schema — config defaults and the v4 guard
# ----------------------------------------------------------------------
def test_v4_config_defaults():
    cfg = da.DAExperimentConfig("test")
    assert cfg.fixed_scale == 100.0
    assert cfg.buffer_capacity == 6_000
    assert cfg.start_steps == 480
    # The calibrated weight for the sum-form anchor. v4 froze 0.01 against a
    # leg-averaged anchor, which halved a two-leg box and so meant something
    # else; the v4 record keeps that value and the v5 manifest keeps this one.
    assert cfg.anchor_weight == 0.10
    assert cfg.gate_logit_bound == da_quote_map.GATE_LOGIT_BOUND == 4.0
    assert cfg.action_schema == "storage_quote_v4"


def test_v3_schema_is_refused():
    with pytest.raises(ValueError, match="storage_quote_v4"):
        da.DAExperimentConfig("test", action_schema="storage_quote_v3")


def test_gate_bound_config_must_match_the_map():
    with pytest.raises(ValueError, match="GATE_LOGIT_BOUND"):
        da.DAExperimentConfig("test", gate_logit_bound=2.0)


def test_negative_anchor_weight_is_refused():
    with pytest.raises(ValueError, match="anchor_weight"):
        da.DAExperimentConfig("test", anchor_weight=-0.1)


# ----------------------------------------------------------------------
# W3a — bounded gates
# ----------------------------------------------------------------------
def test_gate_probability_floor_and_cap():
    bound = da_quote_map.GATE_LOGIT_BOUND
    lo = float(torch.sigmoid(torch.tensor(-bound)))
    hi = float(torch.sigmoid(torch.tensor(bound)))
    policy = torch.tensor([[-100.0, 100.0, 0.0, 0.0]])
    out = da_quote_map.forward_policy_v3(policy, train=False)
    probs = out["probs"][0]
    assert float(probs[0]) >= lo - 1e-7
    assert float(probs[1]) <= hi + 1e-7
    # The hard threshold at 0 is untouched: eval gates follow the raw sign.
    assert float(out["gates"][0, 0]) == 0.0
    assert float(out["gates"][0, 1]) == 1.0
    # Closed gate is still exactly zero quote.
    assert float(out["quotes"][0, 0]) == 0.0


def test_closed_gate_keeps_reopen_gradient():
    policy = torch.tensor([[-20.0, 0.0, 2.5, 2.5]], requires_grad=True)
    torch.manual_seed(0)
    quotes = da_quote_map.forward_policy_v3(policy, train=True)["quotes"]
    grad, = torch.autograd.grad(quotes.sum(), policy)
    # v3 would give exactly 0 here (sigmoid(-20) saturates); the tanh bound
    # keeps the straight-through path alive.
    assert float(grad[0, 0].abs()) > 0.0


def test_sampling_can_reopen_a_closed_gate():
    bound = da_quote_map.GATE_LOGIT_BOUND
    lo = float(torch.sigmoid(torch.tensor(-bound)))
    policy = torch.tensor([[-100.0, -100.0, 0.0, 0.0]])
    torch.manual_seed(2026)
    draws = [da_quote_map.forward_policy_v3(policy, train=True)
             ["gates"][0, 0].item() for _ in range(2000)]
    reopen = float(np.mean(draws))
    # Bernoulli(lo): 2000 draws at lo ~ 0.018 land within +-2 sigma of 36.
    assert reopen > 0.0


# ----------------------------------------------------------------------
# W2 — anchor
# ----------------------------------------------------------------------
def test_anchor_is_zero_at_truthful_and_positive_elsewhere():
    highs = torch.as_tensor(da_quote_map.LEG_HIGHS, dtype=torch.float32)
    closed = da_quote_map.forward_policy_v3(
        torch.tensor([[-100.0, -100.0, 0.0, 0.0]]), train=False)["quotes"]
    anchor = ((closed / highs) ** 2).mean()
    assert float(anchor) == 0.0
    open_q = da_quote_map.forward_policy_v3(
        torch.tensor([[100.0, 100.0, 2.5, 2.5]]), train=False)["quotes"]
    anchor = ((open_q / highs) ** 2).mean()
    assert float(anchor) > 0.0


def test_anchor_gradient_pulls_quotes_down():
    highs = torch.as_tensor(da_quote_map.LEG_HIGHS, dtype=torch.float32)
    policy = torch.tensor([[10.0, 10.0, 1.0, 1.0]], requires_grad=True)
    torch.manual_seed(0)
    quotes = da_quote_map.forward_policy_v3(policy, train=True)["quotes"]
    anchor = ((quotes / highs) ** 2).mean()
    grad, = torch.autograd.grad(anchor, policy)
    # Descending the anchor lowers every logit that raises the quote.
    assert float(grad[0, 2]) > 0.0
    assert float(grad[0, 3]) > 0.0


def test_actor_update_applies_anchor():
    def one_update(anchor_weight):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        trainer = _trainer(anchor_weight=anchor_weight)
        _fill_buffer(trainer)
        trainer.total_steps = 2  # policy_delay divides it
        trainer.update()
        return {nm: trainer.actors[nm].state_dict() for nm in NAMES}

    plain = one_update(0.0)
    anchored = one_update(0.1)
    diffs = [float(torch.max(torch.abs(plain[nm][k] - anchored[nm][k])))
             for nm in NAMES for k in plain[nm]]
    assert max(diffs) > 0.0, "anchor_weight had no effect on the actor step"


# ----------------------------------------------------------------------
# W7 — advantage hinge
# ----------------------------------------------------------------------
def _one_actor_update(**overrides):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    trainer = _trainer(**overrides)
    _fill_buffer(trainer)
    trainer.total_steps = 2  # policy_delay divides it
    trainer.update()
    return trainer


def _actor_params(trainer):
    return {nm: {k: v.detach().clone()
                 for k, v in trainer.actors[nm].state_dict().items()}
            for nm in NAMES}


def test_hinge_default_delta_and_validation():
    assert da.DAExperimentConfig("test").advantage_hinge_delta == 0.10
    with pytest.raises(ValueError, match="advantage_hinge_delta"):
        da.DAExperimentConfig("test", advantage_hinge_delta=-0.01)


def test_hinge_suppresses_econ_gradient_under_delta():
    # No advantage can exceed 1e6, so the econ term is identically zero and
    # with no anchor the actor must not move at all. Weight decay is zeroed
    # for this measurement: Adam normalisation would otherwise turn the
    # 1e-5 decay into a full-lr step and mask the gradient check.
    trainer = _trainer(advantage_hinge_delta=1e6, anchor_weight=0.0)
    _fill_buffer(trainer)
    trainer.total_steps = 2
    for nm in NAMES:
        trainer.actor_opts[nm].param_groups[0]["weight_decay"] = 0.0
    before = _actor_params(trainer)
    trainer.update()
    after = _actor_params(trainer)
    diffs = [float(torch.max(torch.abs(before[nm][k] - after[nm][k])))
             for nm in NAMES for k in before[nm]]
    assert max(diffs) == 0.0, \
        "sub-delta advantage still produced an actor gradient"


def test_hinge_gates_the_econ_term():
    # Same seed, same buffer: the only difference is the hinge, so any
    # divergence proves the econ term flows when the hinge is disabled.
    suppressed = _actor_params(_one_actor_update(
        advantage_hinge_delta=1e6, anchor_weight=0.0))
    free = _actor_params(_one_actor_update(
        advantage_hinge_delta=0.0, anchor_weight=0.0))
    diffs = [float(torch.max(torch.abs(suppressed[nm][k] - free[nm][k])))
             for nm in NAMES for k in suppressed[nm]]
    assert max(diffs) > 0.0, "hinge made no difference to the actor step"


def test_hinge_truthful_forward_zeroes_only_own_quotes():
    trainer = _trainer(advantage_hinge_delta=0.10, anchor_weight=0.0)
    _fill_buffer(trainer)
    trainer.total_steps = 2
    seen = []
    for i, nm in enumerate(NAMES):
        orig = trainer.critics[nm].forward

        def make_spy(orig_fn, idx):
            def spy(obs, act):
                seen.append((idx, act.detach().clone()))
                return orig_fn(obs, act)
            return spy

        trainer.critics[nm].forward = make_spy(orig, i)
    trainer.update()
    # The batch actions are 75/150 on every leg; a truthful forward zeroes
    # exactly that agent's two columns and keeps the rest.
    found = 0
    for idx, act in seen:
        own = act[:, 2 * idx:2 * idx + 2]
        others = torch.cat([act[:, :2 * idx], act[:, 2 * idx + 2:]], dim=1)
        if torch.all(own == 0.0) and torch.all(others != 0.0):
            found += 1
    assert found == len(NAMES), \
        f"expected one zeroed-own-quotes critic call per agent, got {found}"


def test_update_logs_hinge_metrics():
    trainer = _trainer(advantage_hinge_delta=0.10, anchor_weight=0.01)
    _fill_buffer(trainer)
    trainer.total_steps = 2
    out = trainer.update()
    assert out["hinge_delta"] == 0.10
    assert set(out["hinge_advantage_by_agent"]) == set(NAMES)
    for nm in NAMES:
        frac = out["hinge_active_fraction_by_agent"][nm]
        assert 0.0 <= frac <= 1.0


# ----------------------------------------------------------------------
# W8 — calibration exposures
# ----------------------------------------------------------------------
def test_calibration_interval_default_and_validation():
    assert da.DAExperimentConfig("test").calibration_interval == 8
    with pytest.raises(ValueError, match="calibration_interval"):
        da.DAExperimentConfig("test", calibration_interval=-1)


def test_calibration_redraws_open_gate_quotes_over_the_box():
    trainer = _trainer(calibration_interval=1)
    trainer._calibrating = True
    np.random.seed(3)
    torch.manual_seed(3)
    obs = np.zeros(12, dtype=np.float32)
    opened = 0
    for _ in range(200):
        d = trainer._policy_action(trainer.actors["ESS9"], obs, train=True)
        for leg in (0, 1):
            if d["gates"][leg] > 0.0:
                opened += 1
                quote = float(d["quotes"][leg])
                assert 0.0 <= quote <= da_quote_map.LEG_HIGHS[leg]
                assert d["explored"][leg] == 1.0
                assert d["exploration_source"][leg] == 3.0
            else:
                # A closed gate keeps its exact zero quote; calibration must
                # never reopen a leg the policy closed.
                assert float(d["quotes"][leg]) == 0.0
                assert d["exploration_source"][leg] != 3.0
    assert opened > 0, "no open gate in 200 draws; the test is vacuous"


def test_calibration_flag_off_keeps_policy_quotes():
    trainer = _trainer(calibration_interval=8)
    assert trainer._calibrating is False
    np.random.seed(3)
    torch.manual_seed(3)
    obs = np.zeros(12, dtype=np.float32)
    d = trainer._policy_action(trainer.actors["ESS9"], obs, train=True)
    assert 3.0 not in d["exploration_source"]


# ----------------------------------------------------------------------
# W9 — twin-gap-adaptive hinge floor
# ----------------------------------------------------------------------
def _disagreeing_critics(trainer, gap):
    for nm in NAMES:
        orig = trainer.critics[nm].forward

        def make(orig_fn):
            def wrapped(obs, act):
                q1, q2 = orig_fn(obs, act)
                return q1, q2 + gap
            return wrapped

        trainer.critics[nm].forward = make(orig)


def test_hinge_twin_gap_weight_default_and_validation():
    assert da.DAExperimentConfig("test").hinge_twin_gap_weight == 1.0
    with pytest.raises(ValueError, match="hinge_twin_gap_weight"):
        da.DAExperimentConfig("test", hinge_twin_gap_weight=-0.1)


def test_twin_gap_floor_suppresses_gradient_where_critics_disagree():
    # A 1e6 disagreement lifts delta_eff to 1e6 everywhere, so no advantage
    # can clear the floor; with no anchor and no weight decay the actor must
    # not move at all.
    trainer = _trainer(advantage_hinge_delta=0.10, anchor_weight=0.0,
                       hinge_twin_gap_weight=1.0)
    _fill_buffer(trainer)
    trainer.total_steps = 2
    for nm in NAMES:
        trainer.actor_opts[nm].param_groups[0]["weight_decay"] = 0.0
    _disagreeing_critics(trainer, 1e6)
    before = _actor_params(trainer)
    out = trainer.update()
    after = _actor_params(trainer)
    diffs = [float(torch.max(torch.abs(before[nm][k] - after[nm][k])))
             for nm in NAMES for k in before[nm]]
    assert max(diffs) == 0.0, \
        "advantage below the twin-gap floor still produced an actor gradient"
    for nm in NAMES:
        assert out["hinge_twin_gap_by_agent"][nm] == pytest.approx(1e6,
                                                                   rel=1e-3)


def test_zero_twin_gap_weight_restores_the_constant_floor():
    # Same seeds, same buffer, same forced disagreement: only the weight
    # differs. With weight 0 the floor stays at delta and the econ term
    # moves the actor, so the two runs must diverge.
    def one_update(weight):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        trainer = _trainer(advantage_hinge_delta=0.10, anchor_weight=0.0,
                           hinge_twin_gap_weight=weight)
        _fill_buffer(trainer)
        trainer.total_steps = 2
        _disagreeing_critics(trainer, 1e6)
        trainer.update()
        return _actor_params(trainer)

    gated = one_update(1.0)
    free = one_update(0.0)
    diffs = [float(torch.max(torch.abs(gated[nm][k] - free[nm][k])))
             for nm in NAMES for k in gated[nm]]
    assert max(diffs) > 0.0, \
        "twin-gap weight made no difference to the actor step"
