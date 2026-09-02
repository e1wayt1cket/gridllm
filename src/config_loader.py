# config_loader.py
"""Load YAML configuration files and provide nested-key access."""
import os
import yaml
from typing import Any, Optional


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


def save_defaults(path: Optional[str] = None) -> None:
    """Persist the current defaults cache to YAML."""
    target = path or _DEFAULTS_PATH
    with open(target, "w", encoding="utf-8") as f:
        yaml.dump(load_defaults(), f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)


def reload_defaults() -> None:
    """Clear cache so the next load re-reads from disk."""
    global _defaults_cache
    _defaults_cache = None


def set_scenario_param(scenario_name: str, key_path: str, value: Any) -> None:
    """Write a nested parameter into the in-memory scenarios cache.
    key_path is relative to the scenario dict, e.g. 'multipliers.pv'.
    """
    _deep_set(get_scenario_cfg(scenario_name), key_path, value)


def save_scenarios(path: Optional[str] = None) -> None:
    """Persist the current scenarios cache to YAML."""
    target = path or _SCENARIOS_PATH
    with open(target, "w", encoding="utf-8") as f:
        yaml.dump(load_scenarios(), f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)
