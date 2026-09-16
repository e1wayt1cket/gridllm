# tests/test_siting_criterion.py
"""The rule that decides where a battery goes, and the arithmetic under it.

The placement in `config/defaults.yaml` is the output of this criterion, so if
the criterion drifts the placement stops having a reason. These are the cheap
parts of it: the margin a price window leaves, and the share of the action box
that margin keeps tradeable.

The expensive parts -- clearing a day per candidate -- are exercised by
`diagnose_typical_day --search-ess-buses`; nothing here clears a market.
"""

import numpy as np
import pytest

from diagnose_typical_day import (
    TRADEABLE_BOX_SHARE,
    action_box,
    round_trip_margin_cny_mwh,
    tradeable_box_share,
)
from rl_spec import ACTION_STORAGE_QUOTE_V2

ETA = 0.92
CYCLE = 100.0


def test_action_box_is_the_specs_own_bounds():
    """The rule is about the space a policy declares in, so it reads it."""
    box = action_box()
    assert box == (ACTION_STORAGE_QUOTE_V2.bounds["charge_price"],
                   ACTION_STORAGE_QUOTE_V2.bounds["discharge_price"])


def test_margin_reproduces_the_published_per_unit_numbers():
    """The four margins the siting round is argued from.

    Recomputed from the valley and peak each unit's bus cleared at, so the
    reported table and the rule cannot disagree about what the margin is.
    """
    published = {  # name: (valley, peak, margin CNY/MWh)
        "ESS6": (547.277532388794, 882.8668782709076, 17.4),
        "ESS20": (518.4151340251901, 1282.6000183930653, 416.5),
        "ESS24": (586.7928237830914, 920.9037381268764, 9.4),
        "ESS31": (604.9732745272252, 932.9054842815647, 0.7),
    }
    for name, (valley, peak, expected) in published.items():
        got = round_trip_margin_cny_mwh(np.array([valley, peak]), ETA, ETA, CYCLE)
        assert got == pytest.approx(expected, abs=0.05), name


def test_margin_is_the_round_trip_less_the_degradation_floor():
    """One MWh stored costs valley/eta to put in and returns eta*peak."""
    valley, peak = 500.0, 1000.0
    got = round_trip_margin_cny_mwh(np.array([valley, peak]), ETA, ETA, CYCLE)
    assert got == pytest.approx(ETA * peak - valley / ETA - 2 * CYCLE)
    assert got == pytest.approx(920.0 - 543.47826087 - 200.0, abs=1e-6)


def test_a_flat_price_leaves_no_margin():
    """Efficiency losses plus degradation make a flat day unprofitable."""
    flat = np.full(96, 700.0)
    assert round_trip_margin_cny_mwh(flat, ETA, ETA, CYCLE) < 0


BOX_AREA = 150.0 * 300.0


@pytest.mark.parametrize("margin,area", [
    # Below 150 the region is a triangle of area m^2 / 2.
    (0.7, 0.7 ** 2 / 2),
    (9.4, 9.4 ** 2 / 2),
    (17.4, 17.4 ** 2 / 2),
    (150.0, 150.0 ** 2 / 2),
    # Above it the slice is capped by the discharge side: a trapezoid.
    (300.0, 300.0 * 150.0 - 150.0 ** 2 / 2),
    (416.5, 300.0 * 116.5 + 416.5 * 33.5 - (150.0 ** 2 - 116.5 ** 2) / 2),
    # Past 450 the box is entirely inside the region.
    (450.0, BOX_AREA),
    (600.0, BOX_AREA),
])
def test_tradeable_share_matches_the_hand_computed_area(margin, area):
    """`charge + discharge < margin` clipped to `[0,150] x [0,300]`.

    The expected areas are written as the triangles and trapezoids they are,
    not as decimals, so they can be checked by hand against the figure.
    """
    assert tradeable_box_share(margin, action_box()) == pytest.approx(
        area / BOX_AREA, abs=1e-12)


def test_tradeable_share_pieces_together_at_the_shorter_side():
    """The box is 150 by 300, so the two regimes meet at 150, not at 0.

    Below the shorter side the tradeable region is a triangle and the share is
    quadratic; above it the slice is capped by the box and the growth turns
    linear. Checked against the areas computed by hand at the seam.
    """
    box = action_box()
    area = 150.0 * 300.0
    assert tradeable_box_share(75.0, box) == pytest.approx(
        (0.5 * 75.0 * 75.0) / area)
    assert tradeable_box_share(150.0, box) == pytest.approx((0.5 * 150.0 ** 2) / area)
    # 225: every slice is uncapped, so the area is a trapezoid of height 225
    # falling to 75 across the full 150-wide charge axis.
    assert tradeable_box_share(225.0, box) == pytest.approx(
        (225.0 * 150.0 - 0.5 * 150.0 ** 2) / area)
    assert tradeable_box_share(225.0, box) == pytest.approx(0.5)


def test_tradeable_share_is_monotone_and_never_exceeds_the_box():
    box = action_box()
    shares = [tradeable_box_share(m, box) for m in range(0, 600, 25)]
    assert shares == sorted(shares)
    assert max(shares) <= 1.0 + 1e-12


def test_a_margin_at_or_below_zero_makes_the_whole_box_untradeable():
    box = action_box()
    assert tradeable_box_share(0.0, box) == 0.0
    assert tradeable_box_share(-50.0, box) == 0.0


def test_the_threshold_and_its_margin_are_the_same_statement():
    """The number the sweep was judged against, stated both ways.

    `TRADEABLE_BOX_SHARE` was fixed before any bus was measured; this pins the
    margin it corresponds to, so a later change to either is visible as a
    change to the criterion rather than as a quietly different result.
    """
    from scipy.optimize import brentq
    box = action_box()
    margin = brentq(lambda m: tradeable_box_share(m, box) - TRADEABLE_BOX_SHARE,
                    -1.0, sum(hi - lo for lo, hi in box))
    assert margin == pytest.approx(150.0, abs=1e-6)
    assert TRADEABLE_BOX_SHARE == 0.25
