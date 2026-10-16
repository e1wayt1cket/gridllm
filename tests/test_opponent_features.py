# test_opponent_features.py
"""What the two opponent features mean, and what they may not be read as.

The features are `avg_other_dev` / `all_dev_std`, the mean and spread of a
**signed, direction-normalized** deviation from each participant's own truthful
point:

    d = (a - truthful) / (high - truthful)   when a > truthful   -> [0, +1]
    d = (a - truthful) / (truthful - low)    when a < truthful   -> [-1, 0]
    d = 0                                    when a == truthful

Truthful is zero, down is negative, up is positive, and a participant that has
used up its whole room in one direction reads exactly -1 or +1 there. The
denominator is the reach of the direction the action went, not the box width.

The pair this replaced scaled each action into its own box and averaged the
results, which read as one fact about the fleet while being
distance-from-the-low-bound for a storage unit and distance-from-truthful for a
prosumer. The unsigned magnitude form that briefly stood in for it is a third
quantity again, and `OBS_V5` marks it.
"""

from __future__ import annotations

import numpy as np
import pytest

import rl_env
from models import MarketConfig
from rl_env import BiddingEnv, signed_deviation
from rl_spec import (ACTION_BID_OFFER_V1, ACTION_STORAGE_QUOTE_V2, OBS_V3,
                     OBS_V4, OBS_V5, OBS_V6, get_obs_spec)
from scenarios import get_scenario

T = 96
BS = rl_env.BLOCK_SIZE

# The two boxes, read off the specs rather than restated.
PROSUMER = ACTION_BID_OFFER_V1
P_LO, P_HI = PROSUMER.bounds["bid_mult"]
P_TRUTH = PROSUMER.truthful["bid_mult"]
STORAGE = ACTION_STORAGE_QUOTE_V2
S_LO, S_HI = STORAGE.bounds["charge_price"]
S_TRUTH = STORAGE.truthful["charge_price"]


def _build(bid_mult_low=None, bid_mult_high=None):
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=T, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    kwargs = {}
    if bid_mult_low is not None:
        kwargs["bid_mult_low"] = bid_mult_low
    if bid_mult_high is not None:
        kwargs["bid_mult_high"] = bid_mult_high
    return BiddingEnv(agents, config, rl_agent_names=rl_names, **kwargs)


@pytest.fixture(scope="module")
def env():
    return _build()


def _split(env):
    """(storage names, prosumer names) among the participants.

    Compared by spec *name*: the environment builds its own non-storage spec
    from `bid_mult_low/high` rather than reusing the registered one, so the two
    are equal in value and not identical as objects.
    """
    storage = [a.name for a in env.all_agents
               if env.spec_for(a).name == ACTION_STORAGE_QUOTE_V2.name]
    others = [a.name for a in env.all_agents
              if env.spec_for(a).name == ACTION_BID_OFFER_V1.name]
    assert storage and others, "the fixture must carry both ontologies"
    return storage, others


def _truthful(env):
    return {a.name: env.spec_for(a).truthful[env.spec_for(a).action_names[0]]
            for a in env.all_agents}


def _actions(env, first_value):
    """A whole profile, every participant's first action set to `first_value`.

    A float applies to everyone; a name -> float mapping moves one participant
    without the others following.
    """
    out = {}
    for a in env.all_agents:
        spec = env.spec_for(a)
        value = (first_value if not isinstance(first_value, dict)
                 else first_value[a.name])
        out[a.name] = {n: np.full(T, spec.truthful[n], dtype=float)
                       for n in spec.action_names}
        out[a.name][spec.action_names[0]] = np.full(T, float(value),
                                                    dtype=float)
    return out


def _features(env, actions, observer):
    """The two features as the environment computes them for one observer."""
    env._prev_block_actions = None
    env._prev_block_index = None
    env.current_actions = actions
    return env._compute_opponent_features(observer)


def _deviations(env, actions):
    """Each participant's own signed deviation, re-derived from the definition."""
    out = {}
    for a in env.all_agents:
        spec = env.spec_for(a)
        if a.name not in actions or not actions[a.name]:
            continue
        name = spec.action_names[0]
        lo, hi = spec.bounds[name]
        if not hi > lo:
            continue
        value = float(np.mean(actions[a.name][name][0:BS]))
        if not np.isfinite(value):
            continue
        out[a.name] = signed_deviation(value, spec.truthful[name], lo, hi)
    return out


def _expected(env, actions, observer):
    devs = _deviations(env, actions)
    others = [v for k, v in devs.items() if k != observer]
    everyone = list(devs.values())
    return (float(np.mean(others)) if others else 0.0,
            float(np.std(everyone)) if len(everyone) >= 2 else 0.0)


# ---------------------------------------------------------------------------
# 1. Truthful is zero for everyone
# ---------------------------------------------------------------------------

def test_declaring_truthfully_reads_zero_for_every_ontology(env):
    storage, others = _split(env)
    actions = _actions(env, _truthful(env))

    for observer in (storage[0], others[0]):
        avg, std = _features(env, actions, observer)
        assert avg == pytest.approx(0.0), observer
        assert std == pytest.approx(0.0), observer

    # The box-position quantity this replaced would not have been zero for the
    # prosumers, who declare truthfully 47% along their box.
    assert (P_TRUTH - P_LO) / (P_HI - P_LO) > 0.4


# ---------------------------------------------------------------------------
# 2 & 5. The sign is kept, and the two directions do not collapse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [-0.5, -0.25, 0.25, 0.75])
def test_a_prosumer_below_truthful_is_negative_and_above_is_positive(
        env, fraction):
    """Half the reach down is -0.5; half the reach up is +0.5. Equal
    magnitudes, and the feature must still tell them apart."""
    _, others = _split(env)
    observer = others[0]
    mover = others[1] if len(others) > 1 else others[0]
    value = (P_TRUTH + fraction * (P_TRUTH - P_LO) if fraction < 0
             else P_TRUTH + fraction * (P_HI - P_TRUTH))

    values = _truthful(env)
    values[mover] = value
    avg, _ = _features(env, _actions(env, values), observer)

    n_others = len([a for a in env.all_agents if a.name != observer])
    assert avg == pytest.approx(fraction / n_others)


def test_the_two_directions_do_not_collapse_to_one_feature(env):
    """An equal move either side of truthful must not read the same."""
    _, others = _split(env)
    observer = others[0]
    mover = others[1] if len(others) > 1 else others[0]
    base = _truthful(env)

    down, up = dict(base), dict(base)
    down[mover] = P_TRUTH - 0.4 * (P_TRUTH - P_LO)
    up[mover] = P_TRUTH + 0.4 * (P_HI - P_TRUTH)

    a_down, _ = _features(env, _actions(env, down), observer)
    a_up, _ = _features(env, _actions(env, up), observer)
    assert a_down < 0.0 < a_up
    assert a_down != pytest.approx(a_up)


# ---------------------------------------------------------------------------
# 3. The same normalized deviation is the same number in different boxes
# ---------------------------------------------------------------------------

def test_the_same_normalized_deviation_agrees_across_different_boxes():
    """A fifth of the way up is a fifth of the way up, in any box.

    Two environments with different multiplier boxes, the same normalized move
    in each: the prosumers' deviation feature must be identical.
    """
    narrow = _build(bid_mult_low=0.5, bid_mult_high=1.5)
    wide = _build(bid_mult_low=0.2, bid_mult_high=2.0)

    def profile(env, fraction):
        values = _truthful(env)
        for a in env.all_agents:
            spec = env.spec_for(a)
            if spec.name != ACTION_BID_OFFER_V1.name:
                continue
            lo, hi = spec.bounds["bid_mult"]
            values[a.name] = spec.truthful["bid_mult"] + fraction * (
                hi - spec.truthful["bid_mult"])
        return _actions(env, values)

    _, others_n = _split(narrow)
    observer = others_n[0]
    storage_n, _ = _split(narrow)
    fraction = 0.2
    a_narrow, _ = _features(narrow, profile(narrow, fraction), observer)
    a_wide, _ = _features(wide, profile(wide, fraction), observer)
    assert a_wide == pytest.approx(a_narrow)

    # The fleet is mixed and only the prosumers moved, so the mean is their
    # share of it -- which is what makes the equality above a statement about
    # the normalization rather than about two numbers that happened to match.
    n_others = len([a for a in narrow.all_agents if a.name != observer])
    moved = len([a for a in narrow.all_agents
                 if a.name != observer and a.name not in storage_n])
    assert a_narrow == pytest.approx(fraction * moved / n_others)

    # The raw moves behind them are different sizes.
    assert (1.5 - 1.0) * fraction != (2.0 - 1.0) * fraction


# ---------------------------------------------------------------------------
# 4. A truthful point on a boundary, and the direction that does not exist
# ---------------------------------------------------------------------------

def test_a_storage_truthful_at_its_low_edge_has_no_downward_half(env):
    """The storage charge box is (0, 150) with truthful at 0: the whole box is
    upward, and the safe answer is that zero and a clamped undershoot read the
    same rather than dividing by a zero reach."""
    assert S_TRUTH == S_LO, "this test states the storage box's convention"
    assert signed_deviation(S_TRUTH, S_TRUTH, S_LO, S_HI) == 0.0
    assert signed_deviation(S_LO + 0.4 * (S_HI - S_LO), S_TRUTH, S_LO, S_HI) \
        == pytest.approx(0.4)
    assert signed_deviation(S_HI, S_TRUTH, S_LO, S_HI) == pytest.approx(1.0)
    # A downward deviation does not exist in this box. The only value the
    # mechanism could express there is the edge itself, which is truthful; an
    # action below it is not a downward deviation, it is undeclarable, and it
    # is refused rather than divided by a zero reach.
    assert S_TRUTH == S_LO
    with pytest.raises(ValueError, match="outside its own box"):
        signed_deviation(S_LO - 5.0, S_TRUTH, S_LO, S_HI)

    storage, _ = _split(env)
    observer = storage[0]
    values = _truthful(env)
    for nm in storage:
        values[nm] = 0.4 * S_HI
    avg, _ = _features(env, _actions(env, values), observer)
    moved = len([nm for nm in storage if nm != observer])
    n_others = len([a for a in env.all_agents if a.name != observer])
    assert avg == pytest.approx(0.4 * moved / n_others)


def test_an_asymmetric_box_is_described_by_each_directions_own_reach():
    """The prosumer box is asymmetric about truthful (0.7 down, 0.8 up), so the
    same *absolute* move either side is a different normalized deviation."""
    down_reach = P_TRUTH - P_LO
    up_reach = P_HI - P_TRUTH
    assert down_reach != up_reach
    move = 0.5
    assert signed_deviation(P_TRUTH - move, P_TRUTH, P_LO, P_HI) == \
        pytest.approx(-move / down_reach)
    assert signed_deviation(P_TRUTH + move, P_TRUTH, P_LO, P_HI) == \
        pytest.approx(move / up_reach)


def test_a_box_edge_reads_exactly_one_or_minus_one():
    assert signed_deviation(P_LO, P_TRUTH, P_LO, P_HI) == pytest.approx(-1.0)
    assert signed_deviation(P_HI, P_TRUTH, P_LO, P_HI) == pytest.approx(1.0)


def test_a_declaration_outside_its_box_is_refused_not_clamped():
    """An action the mechanism cannot express has no deviation to measure.

    Folding it onto the nearest edge would report a participant sitting at the
    boundary when it had in fact overrun it, and would hide the caller's
    mistake. The result also stays in [-1, 1] without needing a clamp.
    """
    with pytest.raises(ValueError, match="outside its own box"):
        signed_deviation(999.0, P_TRUTH, P_LO, P_HI)
    with pytest.raises(ValueError, match="outside its own box"):
        signed_deviation(-999.0, P_TRUTH, P_LO, P_HI)
    # The edges themselves are legal and read exactly +/-1.
    assert signed_deviation(P_LO, P_TRUTH, P_LO, P_HI) == pytest.approx(-1.0)
    assert signed_deviation(P_HI, P_TRUTH, P_LO, P_HI) == pytest.approx(1.0)
    # A box with no width has nothing in it to deviate.
    assert signed_deviation(5.0, 5.0, 5.0, 5.0) == 0.0


def test_the_block_mean_of_edge_pinned_actions_is_not_refused():
    """The tolerance is for arithmetic, not for real overruns.

    A block-mean built from values pinned to the top edge can land a few ulps
    above it; that is the only thing the tolerance admits.
    """
    mean = float(np.mean([P_HI] * BS))
    assert signed_deviation(mean, P_TRUTH, P_LO, P_HI) == pytest.approx(1.0)
    # ... and it does not admit a visible overrun.
    with pytest.raises(ValueError):
        signed_deviation(P_HI + 1e-3, P_TRUTH, P_LO, P_HI)


def test_the_features_are_computed_from_the_value_the_market_executed(env):
    """The upstream projection, checked end to end rather than assumed.

    `BiddingEnv.step` projects every action into its spec box before it writes
    `current_actions`, and the clearing reads that same array. So an
    out-of-box policy output reaches the market as its projection, and the
    opponent features must carry that projected value -- not the raw one, and
    not a clamp applied at the feature layer.
    """
    storage, _ = _split(env)
    target = storage[0]

    env.reset(wholesale=np.full(T, 300.0, dtype=float))
    raw = np.array([999.0, -5.0])          # both components out of the box
    spec = env.spec_for(next(a for a in env.all_agents if a.name == target))
    for i, name in enumerate(spec.action_names):
        lo, hi = spec.bounds[name]
        assert not (lo <= raw[i] <= hi), name

    env.step({target: raw})

    # What the market executed is the projection, and it is in the box.
    for name in spec.action_names:
        lo, hi = spec.bounds[name]
        written = env.current_actions[target][name]
        assert written.min() >= lo and written.max() <= hi, name
    assert env.current_actions[target]["charge_price"][0] == pytest.approx(150.0)
    assert env.current_actions[target]["discharge_price"][0] == pytest.approx(0.0)

    # The snapshot the features read is that same projected array...
    for name in spec.action_names:
        np.testing.assert_array_equal(
            env._prev_block_actions[target][name],
            env.current_actions[target][name])

    # ...so the feature reports the projected deviation. The target is the only
    # storage unit away from truthful, and it sits on its top edge.
    observer = storage[1]
    avg, _ = env._compute_opponent_features(observer)
    others = [a for a in env.all_agents if a.name != observer]
    # Exactly one of them is at +1, except the observer is not the target.
    assert avg == pytest.approx(1.0 / len(others))

    # And no exception was raised, which is the whole point: the caller did not
    # have to know about the projection, and the feature did not have to clamp.
    env.reset(wholesale=np.full(T, 300.0, dtype=float))


# ---------------------------------------------------------------------------
# 6. The mean is over signed deviations
# ---------------------------------------------------------------------------

def test_the_mean_is_not_an_average_of_magnitudes(env):
    """Two participants shading opposite ways by the same amount.

    The signed mean is zero; the mean of the magnitudes is not. The feature
    must report the first -- the fleet is cancelling out, not leaning.
    """
    _, others = _split(env)
    observer = others[0]
    m1, m2 = others[1], others[2]
    values = _truthful(env)
    values[m1] = P_TRUTH - 0.4 * (P_TRUTH - P_LO)
    values[m2] = P_TRUTH + 0.4 * (P_HI - P_TRUTH)
    actions = _actions(env, values)

    avg, _ = _features(env, actions, observer)
    devs = _deviations(env, actions)
    magnitude_mean = float(np.mean([abs(v) for k, v in devs.items()
                                    if k != observer]))
    assert avg == pytest.approx(0.0, abs=1e-12)
    assert magnitude_mean > 0.0, "the contrast is empty"


# ---------------------------------------------------------------------------
# 7. Missing, non-finite and illegal inputs
# ---------------------------------------------------------------------------

def test_a_non_finite_quote_is_left_out_rather_than_read_as_truthful(env):
    storage, others = _split(env)
    observer = storage[0]
    clean = _truthful(env)
    baseline, _ = _features(env, _actions(env, clean), observer)
    assert baseline == pytest.approx(0.0)

    polluted = dict(clean)
    polluted[others[0]] = float("nan")
    actions = _actions(env, polluted)
    avg, std = _features(env, actions, observer)
    assert np.isfinite(avg) and np.isfinite(std)
    assert avg == pytest.approx(0.0)
    e_avg, e_std = _expected(env, actions, observer)
    assert (avg, std) == (pytest.approx(e_avg), pytest.approx(e_std))


def test_a_participant_with_no_quotes_at_all_is_left_out(env):
    storage, _ = _split(env)
    observer = storage[0]
    actions = _actions(env, _truthful(env))
    actions[storage[1]] = {}
    avg, std = _features(env, actions, observer)
    assert avg == pytest.approx(0.0) and std == pytest.approx(0.0)


def test_an_empty_action_set_reads_no_deviation(env):
    storage, _ = _split(env)
    env._prev_block_actions = None
    env._prev_block_index = None
    env.current_actions = {}
    assert env._compute_opponent_features(storage[0]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 8 & 10. The interface, and independence from visit order
# ---------------------------------------------------------------------------

def test_the_features_still_land_where_the_policy_expects_them(env):
    """Same dimension, same positions -- a meaning change, not a layout one."""
    assert env.obs_spec is OBS_V6
    assert env.obs_spec is get_obs_spec("v6_12d")
    assert env.get_state_dim() == 12

    obs = env.reset(wholesale=np.full(T, 300.0, dtype=float))
    order = list(env.obs_spec.feature_order)
    i_dev, i_std = order.index("avg_other_dev"), order.index("all_dev_std")
    assert i_std == i_dev + 1
    assert i_dev == order.index("slr_norm") + 1
    for name, vec in obs.items():
        assert vec.shape == (12,), name
        assert np.all(np.isfinite(vec)), name
        assert -1.0 <= vec[i_dev] <= 1.0, name
        assert vec[i_std] >= 0.0, name


def test_the_result_does_not_depend_on_the_order_participants_are_visited(env):
    """The mean is over a set, so the order it is built in cannot matter."""
    storage, others = _split(env)
    observer = storage[0]
    values = _truthful(env)
    values[others[0]] = P_TRUTH - 0.3 * (P_TRUTH - P_LO)
    values[others[1]] = P_TRUTH + 0.5 * (P_HI - P_TRUTH)
    for nm in storage[1:]:
        values[nm] = 0.6 * S_HI

    forward = _features(env, _actions(env, values), observer)
    original = list(env.all_agents)
    try:
        env.all_agents = list(reversed(original))
        backward = _features(env, _actions(env, values), observer)
    finally:
        env.all_agents = original
    assert forward == (pytest.approx(backward[0]), pytest.approx(backward[1]))


# ---------------------------------------------------------------------------
# 9. The old semantics cannot be loaded as the new one
# ---------------------------------------------------------------------------

def test_the_signed_feature_is_not_the_unsigned_one(env):
    """The magnitude form that briefly stood in for this is a different number
    on the same input. If the two agreed on a downward deviation the version
    bump would be cosmetic; the sign is the difference."""
    _, others = _split(env)
    observer = others[0]
    mover = others[1]
    values = _truthful(env)
    values[mover] = P_TRUTH - 0.4 * (P_TRUTH - P_LO)
    actions = _actions(env, values)
    signed, _ = _features(env, actions, observer)

    unsigned_by_name = {}
    for a in env.all_agents:
        if a.name == observer:
            continue
        spec = env.spec_for(a)
        name = spec.action_names[0]
        lo, hi = spec.bounds[name]
        unsigned_by_name[a.name] = abs(
            values[a.name] - spec.truthful[name]) / (hi - lo)

    assert signed < 0.0
    assert signed != pytest.approx(float(np.mean(list(unsigned_by_name.values()))))


def test_every_older_observation_spec_is_still_identifiable():
    """V3 through V5 share the layout and differ in meaning, so a checkpoint
    under any of them must be identifiable rather than reinterpreted."""
    for old in (OBS_V3, OBS_V4, OBS_V5):
        assert old.name != OBS_V6.name
        assert old.total_dim == OBS_V6.total_dim
    assert OBS_V5.name == "v5_12d"
    # V5 held the unsigned magnitude form; its feature names are V6's, so the
    # version is the only thing that separates them.
    assert OBS_V5.feature_order == OBS_V6.feature_order
    assert OBS_V5 != OBS_V6
