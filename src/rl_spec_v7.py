"""Observation and action specs for the local day-ahead stack.

The storage action is a price **and** a quantity per leg: a unit that declares
a high discharge price but withholds most of its energy is expressing something
the price-only box could not, and the cleared quantity is what the settlement
pays for. The v5 box therefore has four entries, ordered
charge_price, charge_quantity, discharge_price, discharge_quantity.

The observation keeps the twelve dimensions of V6 but moves the two price
features that describe *this* unit's own bus into the per-agent block, so the
fleet-level mean and spread no longer stand in for a local price.
"""

from __future__ import annotations

from rl_spec import (ActionSpec, FeatureSpec, ObservationSpec,
                     register_action_spec, register_obs_spec)


#: The price-and-quantity storage box. The prices carry the same meaning and
#: bounds as `storage_quote_v4`; the quantities are capped at the unit's rated
#: power per period, so a quote above the cap would be clipped by the clearing
#: anyway and is not a decision. Truthful is "no premium, full availability":
#: price 0 on both legs and the whole rated power offered.
ACTION_STORAGE_QUOTE_V5 = ActionSpec(
    name="storage_quote_v5",
    version=5,
    action_names=("charge_price", "charge_quantity",
                  "discharge_price", "discharge_quantity"),
    bounds={"charge_price": (0.0, 150.0),
            "charge_quantity": (0.0, 0.4),
            "discharge_price": (0.0, 300.0),
            "discharge_quantity": (0.0, 0.4)},
    truthful={"charge_price": 0.0, "charge_quantity": 0.4,
              "discharge_price": 0.0, "discharge_quantity": 0.4},
)
register_action_spec(ACTION_STORAGE_QUOTE_V5)


#: 5 unique + 7 shared = 12 dims.
#:
#: `lmp_bus_norm` and `avg_lmp_bus_hist_norm` are the last committed LMP at
#: this unit's own bus and the mean over the committed periods at that bus, both
#: normalised by the frozen `rl_env.PRICE_REF_CNY_MWH`. They are per-agent
#: because a radial network prices nodes differently, and V6's fleet-mean
#: `last_lmp_norm` / `avg_lmp_hist_norm` told a unit at bus 27 about a price it
#: would never be paid. Both remain history only: nothing here reads a period
#: the day has not reached.
#:
#: The shared block keeps V6's opponent features and timing/forecast features;
#: with the two price features moved, it no longer needs a fleet-average price
#: at all.
OBS_V7 = ObservationSpec(
    name="v7_12d",
    version=7,
    unique_features=(
        FeatureSpec("load_feat"),
        FeatureSpec("re_feat"),
        FeatureSpec("soc"),
        FeatureSpec("lmp_bus_norm"),
        FeatureSpec("avg_lmp_bus_hist_norm"),
    ),
    shared_features=(
        FeatureSpec("block_pos"),
        FeatureSpec("slr_norm"),
        FeatureSpec("avg_other_dev"),
        FeatureSpec("all_dev_std"),
        FeatureSpec("ema_dev"),
        FeatureSpec("price_trend"),
        FeatureSpec("pred_lmp_norm"),
    ),
)
register_obs_spec(OBS_V7)
