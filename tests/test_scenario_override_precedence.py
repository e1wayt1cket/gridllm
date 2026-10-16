# tests/test_scenario_override_precedence.py
"""A per-run scenario override must survive the scenario builder.

`get_scenario` re-applies a scenario's `override_config` onto any config it is
handed, so that a caller which did not build its config from the file still
gets the scenario's settings. That is deliberate. The consequence is a
precedence rule that has to hold: an override written for one run must be
visible to *both* readers -- `typical_day_config`, which builds the config, and
`_build_scenario`, which re-applies the scenario on top of it -- or the second
one silently undoes the first.

This went wrong once. Moving `inverter_smax_multiplier` from code into
`config/scenarios.yaml` made the diagnostic's `--sweep-headroom` a no-op: the
sweep set the value on the config object and `_build_scenario` put the file's
1.10 back one line later, so every swept value produced an identical day. The
gate that step used could not see it, because that gate only ever ran at the
default value.
"""

import pytest

from config_loader import (get_scenario_cfg, reload_scenarios,
                           set_scenario_override, set_scenario_param)
from scenarios import get_scenario, typical_day_config


@pytest.fixture(autouse=True)
def clean_scenario_cache():
    """The scenarios cache is process-global, so a test that writes to it must
    not be able to affect the next one."""
    reload_scenarios()
    yield
    reload_scenarios()


def _set_headroom(value):
    """The per-run override the headroom sweep writes."""
    set_scenario_override("typical_day",
                          "network.inverter_smax_multiplier", value)


def test_a_per_run_override_reaches_the_built_scenario():
    _set_headroom(1.90)

    config = typical_day_config()
    assert config.network.inverter_smax_multiplier == pytest.approx(1.90)

    get_scenario("typical_day", 96, config)
    assert config.network.inverter_smax_multiplier == pytest.approx(1.90), \
        "the scenario builder put the file's value back over the override"


def test_an_override_does_not_outlive_the_run_that_set_it():
    """Without clearing the cache, a swept value leaks into the next build."""
    _set_headroom(1.90)
    assert typical_day_config().network.inverter_smax_multiplier == \
        pytest.approx(1.90)

    reload_scenarios()
    assert typical_day_config().network.inverter_smax_multiplier == \
        pytest.approx(1.10), "the file's value did not come back"


def test_the_written_key_is_the_flat_dotted_form():
    """`override_config`'s keys are flat "network.field" strings.

    Setting a nested dict instead builds a structure `apply_scenario_overrides`
    rejects, so the shape of this dict is load-bearing rather than cosmetic.
    """
    _set_headroom(1.90)
    override = get_scenario_cfg("typical_day")["override_config"]
    assert all(isinstance(k, str) for k in override)
    assert any("." in k for k in override)
    assert not any(isinstance(v, dict) for k, v in override.items()
                   if k != "custom_loads")


# ---------------------------------------------------------------------------
# One contract per write path
# ---------------------------------------------------------------------------

def test_the_scenario_param_door_refuses_a_config_field_path():
    """The nested writer builds `{"network": {...}}`, which the applier then
    rejects -- one scenario build away from the sweep that made it, so the
    mistake is refused at the line that made it instead."""
    with pytest.raises(ValueError, match="set_scenario_override"):
        set_scenario_param("typical_day", "override_config.network.v_min_pu",
                           0.95)


def test_the_override_door_refuses_a_scenario_setting():
    """And the crossing is refused in the other direction too."""
    with pytest.raises(ValueError, match="set_scenario_param"):
        set_scenario_override("typical_day", "override_config.anything", 1)
    with pytest.raises(ValueError, match="set_scenario_param"):
        set_scenario_override("typical_day", "override_config", {})


def test_neither_door_accepts_a_key_with_an_empty_section():
    """An empty part is not a naming style: `_deep_set` would create a `""` key
    that nobody can reach and no report mentions."""
    for bad in ("", "network.", ".network", "a..b"):
        with pytest.raises(ValueError, match="empty section|non-empty string"):
            set_scenario_param("typical_day", bad, 1)
    for bad in ("", "network.", ".network"):
        with pytest.raises(ValueError, match="empty section|non-empty string"):
            set_scenario_override("typical_day", bad, 1)


def test_a_dotted_override_key_takes_a_value_not_a_mapping():
    """A dotted key addresses one field; a dict there is the nested form the
    flat one exists to avoid."""
    with pytest.raises(ValueError, match="not a nested mapping"):
        set_scenario_override("typical_day", "network.v_min_pu",
                              {"V_min_pu": 0.95})


def test_a_bare_key_does_not_replace_a_section_with_a_mapping():
    """`apply_scenario_overrides` refuses it wherever the key came from -- a
    writer, a hand-built dict, or `config/scenarios.yaml` itself."""
    from models import MarketConfig
    from scenarios import apply_scenario_overrides

    config = MarketConfig()
    with pytest.raises(ValueError, match="replace the NetworkConfig section"):
        apply_scenario_overrides(config, {"network": {"v_min_pu": 0.95}})
    # The section is untouched, and the dotted form still works.
    assert config.network.v_min_pu != 0.95
    apply_scenario_overrides(config, {"network.v_min_pu": 0.95})
    assert config.network.v_min_pu == pytest.approx(0.95)


def test_a_scenario_setting_round_trips_through_its_own_door():
    """Write, read back, build: the value arrives, and nothing else moved."""
    from config_loader import get_scenario_cfg, set_scenario_param

    before = dict(get_scenario_cfg("typical_day"))
    set_scenario_param("typical_day", "wholesale_source", "dianyicha")
    assert get_scenario_cfg("typical_day")["wholesale_source"] == "dianyicha"
    # A sibling key is untouched: the write is to one path, not to the block.
    assert get_scenario_cfg("typical_day").get("load_scale_mode") == \
        before.get("load_scale_mode")

    reload_scenarios()
    assert get_scenario_cfg("typical_day").get("wholesale_source") == \
        before.get("wholesale_source"), "the write outlived its run"


def test_an_override_round_trips_through_the_builder():
    """Write, read back, and reach the model the scenario actually clears."""
    _set_headroom(1.65)
    written = get_scenario_cfg("typical_day")["override_config"]
    assert written["network.inverter_smax_multiplier"] == pytest.approx(1.65)
    assert isinstance(written, dict)

    config = typical_day_config()
    assert config.network.inverter_smax_multiplier == pytest.approx(1.65)
    # And not somewhere else on the way: the neighbouring field is the file's.
    assert config.network.line_capacity_multiplier == pytest.approx(1.5)
