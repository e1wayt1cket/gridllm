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
                           set_scenario_override)
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
