from smoke_local_da_v7 import summarize


def record(**over):
    base = {"rewards_finite": True, "clears": 24, "none_clears": 0,
            "soc_min": 0.32, "soc_max": 0.71, "declared_point_deviation": 0.0,
            "inventory_terms": 1}
    base.update(over)
    return base


def test_a_clean_day_passes_every_acceptance_flag():
    out = summarize(record())
    assert out == {"rewards_finite": True, "no_none_clears": True,
                   "soc_in_band": True, "one_inventory_term": True,
                   "declared_point_deviation": 0.0}


def test_a_missing_clear_and_a_second_inventory_term_fail_their_flags():
    out = summarize(record(none_clears=1, inventory_terms=2, soc_max=0.95))
    assert out["no_none_clears"] is False
    assert out["one_inventory_term"] is False
    assert out["soc_in_band"] is False
