import copy

import training_contract_v7 as tc


def contract():
    formal = {
        "terminal_soc": {"mode": "free", "initial": 0.5},
        "wall_clock": {"hours_per_seed_min": 12.0, "hours_per_seed_max": 18.0},
        "episode": {"blocks": 24},
        "validation": {"early_stop_patience": 3},
        "budget": {"seeds": [11, 23], "minimum_rounds": 10,
                   "exposures_per_day_min": 10, "exposures_per_day_max": 15,
                   "episodes_per_seed_min": 900,
                   "episodes_per_seed_max": 1350,
                   "block_steps_per_seed_min": 21600,
                   "block_steps_per_seed_max": 32400},
        "boundaries": {"test_evaluated": False, "holiday_evaluated": False,
                       "h2_opened": True},
    }
    data = {"counts": {"train": 90, "validation": 18, "test": 18,
                       "holiday": 10},
            "all_disjoint": True, "legacy_label": "LEGACY_SEEN_TEST",
            "h2_opened": True, "h2_price_dates_used": 40,
            "independent_generalisation_claim": False}
    return {"formal_training": formal}, data


def codes(config, data):
    return {item["code"] for item in tc.validate_contract(config, data)}


def test_a_consistent_v7_config_has_no_violation():
    config, data = contract()
    assert tc.validate_contract(config, data) == []


def test_a_pinned_endpoint_is_refused():
    config, data = contract()
    config["formal_training"]["terminal_soc"]["final"] = 0.5
    assert "terminal_soc_final_declared" in codes(config, data)


def test_a_non_free_mode_is_refused():
    config, data = contract()
    config["formal_training"]["terminal_soc"]["mode"] = "pinned"
    assert "terminal_soc_mode_not_free" in codes(config, data)


def test_the_v3_library_counts_are_the_expected_ones():
    config, data = contract()
    data["counts"]["train"] = 53
    assert "split_counts_mismatch" in codes(config, data)


def test_a_missing_wall_clock_projection_is_named_but_unordered_is_too():
    config, data = contract()
    config["formal_training"]["wall_clock"] = {}
    assert "wall_clock_not_recorded" in codes(config, data)
    config["formal_training"]["wall_clock"] = {"hours_per_seed_min": 20.0,
                                               "hours_per_seed_max": 10.0}
    assert "wall_clock_inverted" in codes(config, data)


def test_the_price_dates_drawn_on_must_be_declared():
    config, data = contract()
    data["h2_price_dates_used"] = 0
    assert "h2_use_undeclared" in codes(config, data)


def test_a_joint_generalisation_claim_is_refused():
    config, data = contract()
    data["independent_generalisation_claim"] = True
    assert "h2_use_undeclared" in codes(config, data)


def test_the_legacy_label_is_required_on_the_v3_split():
    config, data = contract()
    data["legacy_label"] = "SEALED_TEST"
    assert "legacy_label_undeclared" in codes(config, data)


def test_the_v6_arithmetic_still_gates_the_v7_budget():
    config, data = contract()
    broken = copy.deepcopy(config)
    broken["formal_training"]["budget"]["episodes_per_seed_min"] = 899
    assert "min_episodes_mismatch" in codes(broken, data)


def test_a_superseded_protocol_is_checked_against_the_common_split():
    # `--config-dir` against v5 or older still has to answer; those
    # directories declare the common split, so the table falls back to it
    # instead of raising on a protocol it does not list.
    import run_formal_training

    data = {"counts": {"train": 53, "validation": 11, "test": 11,
                       "holiday_block": 5}, "all_disjoint": True,
            "legacy_label_matches_declared": True, "h2_opened": False}
    for protocol in (2, 3, 4, 5):
        checks = run_formal_training.data_checks(protocol, data)
        assert checks["common_split_counts"] is True
        assert checks["h2_sealed"] is True
