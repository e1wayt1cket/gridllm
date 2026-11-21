# config_loader.py
"""Load YAML configuration files and provide nested-key access."""
import os
import yaml
from typing import Any, List, Optional


def _find_config_dir() -> str:
    """Return the nearest ancestor of this module that holds config/defaults.yaml
    and config/scenarios.yaml.

    Scripts run from src/ while the YAML files may live in a sibling repo-root
    config/ directory, so a path fixed relative to __file__ would break.
    Walking upward keeps both layouts working regardless of the current cwd.
    """
    start = os.path.dirname(os.path.abspath(__file__))
    d = start
    while True:
        if (os.path.isfile(os.path.join(d, "config", "defaults.yaml"))
                and os.path.isfile(os.path.join(d, "config", "scenarios.yaml"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return start


_CONFIG_DIR = _find_config_dir()
_DEFAULTS_PATH = os.path.join(_CONFIG_DIR, "config", "defaults.yaml")
_SCENARIOS_PATH = os.path.join(_CONFIG_DIR, "config", "scenarios.yaml")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_get(d: dict, dotted_key: str, default: Any = None) -> Any:
    """Access nested dict via dotted key, e.g. 'profiles.pv.amplitude'."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


_defaults_cache: Optional[dict] = None
_scenarios_cache: Optional[dict] = None


def load_defaults() -> dict:
    global _defaults_cache
    if _defaults_cache is None:
        _defaults_cache = _load_yaml(_DEFAULTS_PATH)
    return _defaults_cache


def load_scenarios() -> dict:
    global _scenarios_cache
    if _scenarios_cache is None:
        _scenarios_cache = _load_yaml(_SCENARIOS_PATH)
    return _scenarios_cache


def get_default(key_path: str, default: Any = None) -> Any:
    return _deep_get(load_defaults(), key_path, default)


def get_scenario_cfg(name: str) -> dict:
    """Return the config dict for a named scenario, or {} if not found."""
    return _deep_get(load_scenarios(), f"scenarios.{name}", {})


def get_prosumer_cfg(load_type: str) -> dict:
    return get_default(f"prosumers.{load_type}", {})


def get_load_type_cfg(load_type: str) -> dict:
    return get_default(f"load_types.{load_type}", {})



# ---------------------------------------------------------------------------
# Write / persist methods
# ---------------------------------------------------------------------------

def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set nested dict value via dotted key, creating intermediate dicts."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def set_default(key_path: str, value: Any) -> None:
    """Write a value into the in-memory defaults cache."""
    _deep_set(load_defaults(), key_path, value)


def reload_defaults() -> None:
    """Clear cache so the next load re-reads from disk."""
    global _defaults_cache
    _defaults_cache = None


def reload_scenarios() -> None:
    """Clear the scenarios cache, undoing any `set_scenario_param` writes.

    Separate from `reload_defaults` because the two caches are read by
    different callers and a sweep usually overrides only one of them. Without
    this, a per-run override written with `set_scenario_param` outlives the run
    that asked for it: the cache is process-global and nothing else clears it,
    so the next scenario built in the same process silently inherits the last
    value a sweep set.
    """
    global _scenarios_cache
    _scenarios_cache = None


#: The block of a scenario that carries configuration-field overrides. Two
#: write paths reach a scenario, and this is the one that names them.
OVERRIDE_BLOCK = "override_config"


def _dotted_parts(key_path: str, what: str) -> List[str]:
    """A dotted key's parts, refusing one that is empty or malformed.

    An empty part is not a naming style: `_deep_set` would create a `""` key,
    and the value would then be reachable by nobody and reported by nothing.
    """
    if not isinstance(key_path, str) or not key_path:
        raise ValueError(f"{what} must be a non-empty string; got {key_path!r}")
    parts = key_path.split(".")
    if any(not part for part in parts):
        raise ValueError(
            f"{what} {key_path!r} has an empty section; a dotted path names "
            f"one field per part")
    return parts


def set_scenario_param(scenario_name: str, key_path: str, value: Any) -> None:
    """Write a nested parameter into the in-memory scenarios cache.

    `key_path` is relative to the scenario dict: `'wholesale_source'`,
    `'multipliers.pv'`. This is the door for **the scenario's own settings** --
    the keys that live directly in the scenario block.

    It is not the door for a configuration field. Those live in the scenario's
    `override_config` block under flat `"section.field"` keys, and
    `set_scenario_override` is the only way to write one. Passing a nested path
    like `'override_config.network.v_min_pu'` here builds
    `{"network": {"v_min_pu": ...}}`, which `apply_scenario_overrides` then
    rejects as naming no field of a dict -- an error that surfaces at apply
    time, one scenario build away from the sweep that made it. It is refused
    here instead.
    """
    parts = _dotted_parts(key_path, "scenario parameter path")
    if parts[0] == OVERRIDE_BLOCK:
        rest = ".".join(parts[1:]) or "<section>.<field>"
        raise ValueError(
            f"{key_path!r} addresses {OVERRIDE_BLOCK!r}, whose entries are "
            f"flat 'section.field' strings rather than a nested structure; "
            f"use set_scenario_override({scenario_name!r}, {rest!r}, ...) "
            f"instead")
    _deep_set(get_scenario_cfg(scenario_name), key_path, value)


def set_scenario_override(scenario_name: str, key: str, value: Any) -> None:
    """Write one `override_config` entry for a scenario, in the cache.

    A sibling of `set_scenario_param` rather than a use of it, because
    `override_config`'s keys are flat "section.field" strings rather than a
    nested structure: `set_scenario_param` would build
    `{"network": {"v_min_pu": ...}}`, which `apply_scenario_overrides` then
    rejects as naming no field of a dict.

    Writing here rather than onto a config object is what makes a per-run
    override stick. `get_scenario` re-applies this block onto whatever config
    it is handed, so a value set on the config alone is silently replaced by
    the file's -- and a sweep built on it reads as flat.

    A key that names the scenario's own settings -- `'wholesale_source'`,
    `'multipliers'` -- belongs to `set_scenario_param`; it is refused here so
    the two paths cannot be crossed in either direction.
    """
    parts = _dotted_parts(key, "scenario override key")
    if parts[0] == OVERRIDE_BLOCK:
        raise ValueError(
            f"scenario override key {key!r} names the override block itself; "
            f"a scenario's own settings are written with "
            f"set_scenario_param({scenario_name!r}, {key!r}, ...)")
    if len(parts) > 1 and isinstance(value, dict):
        # A dotted key addresses one field of a sub-config, and replacing one
        # of those with a dict replaces the dataclass that holds the rest of
        # the section.
        raise ValueError(
            f"scenario override {key!r} takes a value, not a nested mapping; "
            f"write each field it contains as its own 'section.field' key")
    cfg = get_scenario_cfg(scenario_name)
    cfg.setdefault(OVERRIDE_BLOCK, {})[key] = value


