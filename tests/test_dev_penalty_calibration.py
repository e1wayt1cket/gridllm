"""Calibrating the deviation weight, and the sanity properties it rests on.

The weight is chosen from a fixed B3 + SOCP day with a fixed seed and no
learning: the same observations, the same critic initialization, the same batch
for every weight, so what differs between rows is only the weight. Selection is
by how much the regularizer claims of the gradient and how far it narrows the
policy -- never by profit, which is why nothing in the calibration reports it.
"""

import numpy as np
import pytest

from calibrate_dev_penalty import LAMBDA_GRID, calibrate, select
from rl_td3 import deviation_penalty, normalized_deviation
from rl_spec import ACTION_STORAGE_QUOTE_V2
import torch

_STORAGE_BOUNDS = torch.tensor([[0.0, 0.0], [150.0, 300.0]],
                              dtype=torch.float32)
_STORAGE_TRUTHFUL = np.array([0.0, 0.0], dtype=np.float32)


def _fake_doc(rows):
    return {"rows": rows, "econ_grad_norm_reference": 1.0}


# ---------------------------------------------------------------------------
# The sanity properties the calibration relies on
# ---------------------------------------------------------------------------

def test_the_deviation_is_monotone_in_every_action_at_once():
    """No action becomes less deviant while another becomes more."""
    logits = torch.linspace(-15.0, 15.0, 301).unsqueeze(1).repeat(1, 2)
    d = normalized_deviation(logits, _STORAGE_TRUTHFUL, _STORAGE_BOUNDS)
    assert torch.all(d[1:] >= d[:-1] - 1e-7)
    # Both legs move together, so a sum over actions is monotone too.
    assert torch.allclose(d[:, 0], d[:, 1], atol=1e-6)


def test_the_deviation_has_no_unit_dependence():
    """Normalizing by the box width removes the box's units.

    Calibrated on the storage box as configured, the same weight means the same
    thing if the box is later rescaled -- which matters because the bound on
    any action is a modelling choice, not a market fact.
    """
    x = float(torch.atanh(torch.tensor(0.25)))
    logits = torch.full((1, 2), x)
    base = normalized_deviation(logits, _STORAGE_TRUTHFUL, _STORAGE_BOUNDS)
    for factor in (0.5, 2.0, 10.0):
        scaled = _STORAGE_BOUNDS * factor
        assert torch.allclose(
            base, normalized_deviation(logits, _STORAGE_TRUTHFUL, scaled),
            atol=1e-6), factor


def test_the_penalty_is_proportional_to_the_weight():
    """Doubling the weight doubles the penalty, at any deviation.

    This is what makes the grid meaningful: a row's penalty column is the whole
    story of that row's strength, not a coincidence of where the probe sat.
    """
    logits = torch.randn(16, 2) * 2.0
    one = deviation_penalty(logits, _STORAGE_TRUTHFUL, _STORAGE_BOUNDS, 0.1)
    two = deviation_penalty(logits, _STORAGE_TRUTHFUL, _STORAGE_BOUNDS, 0.2)
    assert float(two) == pytest.approx(2.0 * float(one), rel=1e-6)


def test_the_storage_truthful_point_is_a_box_corner():
    """The property that makes the corner question real for this fleet.

    A penalty measured from the midpoint would be measuring from somewhere the
    fleet has no reason to sit; this one is measured from the corner it would
    declare if it had nothing to say.
    """
    spec = ACTION_STORAGE_QUOTE_V2
    for name, value in spec.truthful.items():
        assert value == spec.bounds[name][0], name


# ---------------------------------------------------------------------------
# The grid and the selection rule
# ---------------------------------------------------------------------------

def test_the_selection_rule_is_anchored_to_the_no_penalty_row():
    """Tolerances are relative, so the rule cannot drift with the scale of the
    probe's own numbers."""
    doc = _fake_doc([
        {"lambda": 0.0, "mean_deviation_after_probe": 0.60,
         "action_spread": 100.0, "grad_ratio": 0.0, "penalty_at_mid": 0.0,
         "corner_fraction": 0.0},
        # Within both tolerances.
        {"lambda": 0.1, "mean_deviation_after_probe": 0.55,
         "action_spread": 90.0, "grad_ratio": 0.08, "penalty_at_mid": 0.05,
         "corner_fraction": 0.0},
        # Narrows the policy well past the 20% / 30% tolerances.
        {"lambda": 1.0, "mean_deviation_after_probe": 0.10,
         "action_spread": 10.0, "grad_ratio": 0.80, "penalty_at_mid": 0.5,
         "corner_fraction": 1.0},
    ])
    out = select(doc)
    assert out["selected_lambda"] == 0.1
    assert out["deviation_floor"] == pytest.approx(0.48)
    assert out["spread_floor"] == pytest.approx(70.0)


def test_the_selection_never_takes_the_largest_weight_by_default():
    """The strongest weight in the grid is exactly what must not be chosen.

    A penalty that pins the fleet at truthful removes the bidding the research
    measures, and it does so while looking like a clean, well-regularized run.
    """
    doc = _fake_doc([
        {"lambda": 0.0, "mean_deviation_after_probe": 0.5,
         "action_spread": 100.0, "grad_ratio": 0.0, "penalty_at_mid": 0.0,
         "corner_fraction": 0.0},
        {"lambda": 1.0, "mean_deviation_after_probe": 0.01,
         "action_spread": 1.0, "grad_ratio": 0.9, "penalty_at_mid": 0.5,
         "corner_fraction": 1.0},
    ])
    out = select(doc)
    assert out["selected_lambda"] == 0.0, out["reason"]


def test_the_selection_falls_back_to_no_penalty_when_everything_is_admissible():
    """If even the smallest weight narrows the policy, nothing is supported."""
    doc = _fake_doc([
        {"lambda": 0.0, "mean_deviation_after_probe": 0.5,
         "action_spread": 100.0, "grad_ratio": 0.0, "penalty_at_mid": 0.0,
         "corner_fraction": 0.0},
        {"lambda": 0.01, "mean_deviation_after_probe": 0.01,
         "action_spread": 1.0, "grad_ratio": 0.01, "penalty_at_mid": 0.005,
         "corner_fraction": 1.0},
    ])
    assert select(doc)["selected_lambda"] == 0.0


def test_the_grid_spans_negligible_to_as_large_as_the_economic_term():
    assert LAMBDA_GRID[0] == 0.0
    assert 1.0 in LAMBDA_GRID
    assert list(LAMBDA_GRID) == sorted(LAMBDA_GRID)


# ---------------------------------------------------------------------------
# On the real day
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_calibration_is_deterministic_on_the_fixed_day():
    """Same seed, same day, same batch -> the same table."""
    a = calibrate(seed=123)
    b = calibrate(seed=123)
    for ra, rb in zip(a["rows"], b["rows"]):
        assert ra["lambda"] == rb["lambda"]
        for key in ("grad_ratio", "mean_deviation_after_probe",
                    "action_spread", "corner_fraction"):
            assert ra[key] == pytest.approx(rb[key], rel=1e-9), (key, ra[key])


@pytest.mark.slow
def test_the_regularizer_pull_is_linear_in_its_weight():
    """The weight alone sets the pull, and the grid holds no interior surprise.

    `deviation_penalty` multiplies a quantity that does not depend on the
    weight, and the probe seeds its actor and critic identically for every row,
    so the gradient the regularizer claims is exactly proportional to the
    weight it is given. That is what lets the selection rule read the grid as
    one measurement repeated at different weights.

    Asserted as a ratio rather than against a fitted constant. The constant
    depends on the observation batch and the frozen critic, so it moves when
    those move; the proportionality is the part the implementation guarantees.
    Pinning the measured value is how this test came to encode `0.784`, which
    stopped being true once the penalty moved from logit space to
    box-normalized action space -- a failure that said nothing about the code.
    """
    doc = calibrate(seed=123)
    rows = sorted(doc["rows"], key=lambda r: r["lambda"])

    # The premise: every row was measured against the same critic and batch,
    # so a difference between rows is the weight's alone.
    econs = [r["econ_grad_norm"] for r in rows]
    assert max(econs) - min(econs) < 1e-9, \
        "the probe must hold the critic and the batch fixed across weights"

    ratios = {r["lambda"]: r["grad_ratio"] for r in rows}
    assert ratios[0.0] == pytest.approx(0.0, abs=1e-12)
    per_unit = [ratios[lam] / lam for lam in ratios if lam > 0]
    assert per_unit[0] > 0
    assert per_unit == pytest.approx([per_unit[0]] * len(per_unit), rel=1e-6)


@pytest.mark.slow
def test_the_frozen_weight_is_the_one_the_rule_selects():
    """The weight the experiments run at must be the one the rule picks.

    `CALIBRATED_DEV_PENALTY` is what every formal run uses and `select` is what
    justifies it. If the two drift apart -- a changed actor, a changed penalty,
    a changed action box -- the runs keep using a number whose supporting
    measurement no longer exists, and nothing else in the suite would notice.
    """
    from rl_td3 import CALIBRATED_DEV_PENALTY

    selected = select(calibrate(seed=123))["selected_lambda"]
    assert selected == CALIBRATED_DEV_PENALTY, (
        f"the calibration selects {selected} but the frozen weight is "
        f"{CALIBRATED_DEV_PENALTY}; the formal runs would use an unjustified "
        f"value")


@pytest.mark.slow
def test_a_strong_weight_narrows_the_policy_and_a_weak_one_does_not():
    """Where the penalty bites, the quotes converge; where it does not, it
    does not.

    The spread is deliberately **not** asserted monotone across the whole grid.
    At the low end it is not: measured, `lambda = 0.01` leaves a slightly wider
    spread than no penalty at all, because at that weight the arbitrary frozen
    critic is doing all the steering and the regularizer is along for the ride.
    Calling that monotone would hide the very thing the grid exists to show --
    which weights are actually steering the policy.
    """
    doc = calibrate(seed=123)
    rows = sorted(doc["rows"], key=lambda r: r["lambda"])
    ref = rows[0]
    weakest, strongest = rows[1], rows[-1]

    # The weakest weight is not steering: its spread is not below the reference.
    assert weakest["action_spread"] >= ref["action_spread"] * 0.98
    # The strongest is: the fleet's quotes have collapsed onto each other.
    assert strongest["action_spread"] < ref["action_spread"] * 0.2

    # Between those, once the pull is a few percent of the economic term, the
    # spread falls away consistently.
    steering = [r for r in rows if r["grad_ratio"] >= 0.03]
    spreads = [r["action_spread"] for r in steering]
    assert all(b <= a + 1e-6 for a, b in zip(spreads, spreads[1:])), spreads

    sel = select(doc)
    assert sel["selected_lambda"] in LAMBDA_GRID
    # The strongest weight must not be the answer, or the grid is too short.
    assert sel["selected_lambda"] < max(LAMBDA_GRID)
    assert sel["grad_ratio"] < 0.5
