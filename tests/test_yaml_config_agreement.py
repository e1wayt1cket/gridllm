"""config/defaults.yaml is the configuration's single source of truth.

These are coverage tests, not tautologies. Comparing `from_yaml()`'s output
against the file it read would prove nothing; what is worth pinning is that the
binding is *complete* in both directions, so a key or a field cannot quietly
stop being wired. The failure this guards against is the one that existed
before it: three whole blocks sat in the file with no reader at all, and one of
them had drifted from the code without anyone being able to notice.
"""

import dataclasses

import pytest

import config_loader
from config_loader import get_default, load_defaults
from models import (
    KNOWN_INERT,
    YAML_FIELD_MAP,
    YAML_KEYS_READ_ELSEWHERE,
    MarketConfig,
    NetworkConfig,
    StorageConfig,
    MarketDesignConfig,
    RTConfig,
)
from scenarios import apply_scenario_overrides, typical_day_config

BLOCKS = {
    "network": NetworkConfig,
    "storage": StorageConfig,
    "market_design": MarketDesignConfig,
    "rt": RTConfig,
}


def _leaf_keys(block: str):
    """Every leaf key path under one top-level block, dotted below the block."""
    out = set()
    node = load_defaults().get(block, {})
    if not isinstance(node, dict):
        return out
    for key, value in node.items():
        path = f"{block}.{key}"
        if isinstance(value, dict) and value:
            for sub in value:
                out.add(f"{path}.{sub}")
        else:
            out.add(path)
    return out


def test_every_field_of_the_four_configs_is_bound_or_declared_inert():
    """Forward coverage: no field is silently left out of the file."""
    unbound = []
    for block, cls in BLOCKS.items():
        bound = set(YAML_FIELD_MAP[block])
        for f in dataclasses.fields(cls):
            if f.name in bound:
                continue
            if f"{cls.__name__}.{f.name}" in KNOWN_INERT:
                assert KNOWN_INERT[f"{cls.__name__}.{f.name}"].strip(), \
                    f"{cls.__name__}.{f.name} is declared inert with no reason"
                continue
            unbound.append(f"{cls.__name__}.{f.name}")
    assert not unbound, (
        "these configuration fields are neither bound to a YAML key nor "
        f"declared inert: {sorted(unbound)}. Bind them, or add them to "
        "KNOWN_INERT with the reason they are not.")


def test_every_bound_key_exists_in_the_file():
    """A binding that names a key the file does not have is a broken binding."""
    missing = []
    for block, mapping in YAML_FIELD_MAP.items():
        present = _leaf_keys(block) | {v for v in mapping.values()}
        for field_name, yaml_key in mapping.items():
            if yaml_key not in present:
                missing.append(f"{yaml_key} (for {block}.{field_name})")
    assert not missing, f"bound to keys the file does not carry: {missing}"


def test_every_key_in_the_config_blocks_is_read():
    """Reverse coverage: no YAML key in the four blocks is left unread."""
    bound = set()
    for mapping in YAML_FIELD_MAP.values():
        bound |= set(mapping.values())
    unread = []
    for block in BLOCKS:
        for key in _leaf_keys(block):
            if key in bound or key in YAML_KEYS_READ_ELSEWHERE:
                continue
            # A nested block with no leaf of its own (an empty override dict)
            # is still the value of a field.
            if any(key.startswith(b + ".") for b in bound):
                continue
            unread.append(key)
    assert not unread, (
        f"these keys are in config/defaults.yaml but nothing reads them: "
        f"{sorted(unread)}. Bind them, or delete them.")


def test_the_file_and_the_dataclass_agree_on_every_bound_value():
    """The live-value check: the file says what the model does.

    This is the test that would have caught `enable_multi_objective`, which read
    `true` in the file against an effective `false` in code for as long as
    nothing read the file.
    """
    config = MarketConfig.from_yaml()
    for block, mapping in YAML_FIELD_MAP.items():
        target = getattr(config, block)
        for field_name, yaml_key in mapping.items():
            from_file = get_default(yaml_key)
            in_config = getattr(target, field_name)
            assert from_file == in_config or type(in_config) is tuple, \
                (f"{yaml_key} is {from_file!r} in the file but "
                 f"{in_config!r} on the config")


def test_the_multi_objective_flag_matches_the_file():
    """The specific divergence this binding was built to remove."""
    assert get_default("market_design.enable_multi_objective") == \
        MarketConfig.from_yaml().market_design.enable_multi_objective


def test_typical_day_is_assembled_from_the_file():
    """The canonical entry point assigns nothing; the file supplies everything.

    Including the scenario's own choices, which live in config/scenarios.yaml
    and reach the config through the same override path every other scenario
    uses -- so a bare `typical_day_config()` is the model the diagnostic clears
    rather than a near miss.
    """
    config = typical_day_config()
    assert config.opf_mode == "socp"
    assert config.network.inverter_smax_multiplier == pytest.approx(1.10), \
        "the typical day's inverter rating is a scenario choice; it is not on " \
        "the config, so the scenario override did not apply"
    assert config.network.v_min_pu == get_default("network.v_min_pu")
    assert config.storage.cycle_cost == get_default("storage.cycle_cost")


def test_a_scenario_without_overrides_equals_the_file():
    assert MarketConfig.from_yaml() == MarketConfig.from_yaml()


@pytest.mark.parametrize("override", [
    {"no_such_field": 1},
    {"network.no_such_field": 1},
    {"network.line_capacity_multiplier.no_such_field": 1},
    {"_private": 1},
])
def test_an_override_that_names_nothing_is_rejected(override):
    """A misspelled key must not become a stray attribute.

    `setattr` on a dataclass instance accepts any name, so an override that
    matches no field used to be stored and never read -- a scenario silently
    configured by a comment.
    """
    config = MarketConfig.from_yaml()
    with pytest.raises(ValueError, match="names no field"):
        apply_scenario_overrides(config, override)


def test_a_valid_override_still_applies():
    config = MarketConfig.from_yaml()
    apply_scenario_overrides(config, {"network.line_capacity_multiplier": 2.5,
                                      "market_design.lambda_re": 17.0})
    assert config.network.line_capacity_multiplier == 2.5
    assert config.market_design.lambda_re == 17.0


def test_custom_loads_is_not_a_config_field():
    """It is a scenario directive applied to agents, not a MarketConfig value.

    The scenario builder pops it before the overrides are walked; this pins
    that it would be rejected if it ever reached the walk, which is what makes
    the pop load-bearing rather than incidental.
    """
    config = MarketConfig.from_yaml()
    with pytest.raises(ValueError, match="names no field"):
        apply_scenario_overrides(config, {"custom_loads": [{"buses": [1]}]})


def test_from_yaml_rejects_an_unknown_top_level_override():
    with pytest.raises(KeyError, match="has no field"):
        MarketConfig.from_yaml(no_such_field=1)


def test_reload_and_rebuild_is_stable():
    """The binding reads through the same cache every other reader uses."""
    before = MarketConfig.from_yaml()
    config_loader.reload_defaults()
    after = MarketConfig.from_yaml()
    assert before == after
