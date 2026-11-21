"""The v5 storage-quote action box and the v7 observation layout."""

import numpy as np
import pytest

import local_da_days
import rl_spec_v7
from rl_spec import get_action_spec, get_obs_spec


def test_the_v5_box_and_its_declared_point():
    spec = get_action_spec("storage_quote_v5")
    assert spec.dim == 4
    assert spec.version == 5
    # truthful_array() is float32, where 0.4 is not exact.
    assert spec.truthful_array().tolist() == pytest.approx([0., 0.4, 0., 0.4])
    assert spec.bounds == {"charge_price": (0., 150.), "charge_quantity": (0., 0.4),
                           "discharge_price": (0., 300.), "discharge_quantity": (0., 0.4)}


def test_the_v7_observation_is_five_unique_and_seven_shared():
    spec = get_obs_spec("v7_12d")
    assert spec.version == 7
    assert spec.total_dim == 12 and spec.unique_dim == 5
    assert spec.feature_order[:5] == ("load_feat", "re_feat", "soc",
                                      "lmp_bus_norm", "avg_lmp_bus_hist_norm")
    assert spec.feature_order[5:] == ("block_pos", "slr_norm", "avg_other_dev",
                                      "all_dev_std", "ema_dev", "price_trend",
                                      "pred_lmp_norm")


def test_the_day_reader_and_the_spec_declare_one_truthful_quote():
    assert np.allclose(local_da_days.TRUTHFUL_QUOTE,
                       rl_spec_v7.ACTION_STORAGE_QUOTE_V5.truthful_array())
