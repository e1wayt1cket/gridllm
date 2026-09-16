# rl_spec.py
"""Pluggable observation and action space definitions for the RL pipeline.

Observation/action specs decouple the BiddingEnv's feature layout from the
policies that consume it. When the feature set changes (e.g. a new compact
observation design), the change is isolated behind a new spec name/version
instead of silently breaking every previously trained `.pt` checkpoint.

- FeatureSpec: one named feature with a dimensionality (default 1).
- ObservationSpec: ordered unique (per-agent, at vector start) and shared
  (system-wide) feature groups. MATD3's centralized critic slices the first
  `unique_dim` entries of each other agent's observation to build its global
  state, so per-agent features must stay contiguous at position 0.
- ActionSpec: ordered continuous action names and their bounds.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dim: int = 1


@dataclass(frozen=True)
class ObservationSpec:
    name: str
    version: int
    unique_features: Tuple[FeatureSpec, ...]
    shared_features: Tuple[FeatureSpec, ...]

    @property
    def unique_dim(self) -> int:
        return sum(f.dim for f in self.unique_features)

    @property
    def shared_dim(self) -> int:
        return sum(f.dim for f in self.shared_features)

    @property
    def total_dim(self) -> int:
        return self.unique_dim + self.shared_dim

    @property
    def feature_order(self) -> Tuple[str, ...]:
        return tuple(f.name for f in (*self.unique_features,
                                      *self.shared_features))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "unique_dim": self.unique_dim,
            "shared_dim": self.shared_dim,
            "total_dim": self.total_dim,
            "unique_features": [f.name for f in self.unique_features],
            "shared_features": [f.name for f in self.shared_features],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObservationSpec":
        return cls(
            name=d["name"],
            version=d["version"],
            unique_features=tuple(FeatureSpec(n) for n in d["unique_features"]),
            shared_features=tuple(FeatureSpec(n) for n in d["shared_features"]),
        )


@dataclass(frozen=True)
class ActionSpec:
    """The action space, and the point in it that means "declare no premium".

    `truthful` is part of the spec rather than a convention held by each
    producer, because a baseline that declares something other than its own
    reservation is not a baseline and every caller has to agree on which point
    that is. Holding it here makes the agreement checkable at import.
    """
    name: str
    version: int
    action_names: Tuple[str, ...]
    bounds: Dict[str, Tuple[float, float]]
    truthful: Dict[str, float]

    @property
    def dim(self) -> int:
        return len(self.action_names)

    def truthful_array(self) -> np.ndarray:
        """The truthful action, in `action_names` order."""
        return np.array([self.truthful[n] for n in self.action_names],
                        dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "action_names": list(self.action_names),
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "truthful": dict(self.truthful),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSpec":
        return cls(
            name=d["name"],
            version=d["version"],
            action_names=tuple(d["action_names"]),
            bounds={k: tuple(v) for k, v in d["bounds"].items()},
            truthful=dict(d.get("truthful", {})),
        )


OBS_SPECS: Dict[str, ObservationSpec] = {}
ACTION_SPECS: Dict[str, ActionSpec] = {}


def register_action_spec(spec: ActionSpec) -> None:
    if spec.name in ACTION_SPECS:
        raise ValueError(f"Action spec '{spec.name}' already registered")
    ACTION_SPECS[spec.name] = spec


def get_action_spec(name: str) -> ActionSpec:
    if name not in ACTION_SPECS:
        raise ValueError(f"Unknown action spec '{name}'. "
                         f"Available: {list(ACTION_SPECS)}")
    return ACTION_SPECS[name]


def register_obs_spec(spec: ObservationSpec) -> None:
    if spec.name in OBS_SPECS:
        raise ValueError(f"Observation spec '{spec.name}' already registered")
    OBS_SPECS[spec.name] = spec


def get_obs_spec(name: str) -> ObservationSpec:
    if name not in OBS_SPECS:
        raise ValueError(f"Unknown observation spec '{name}'. "
                         f"Available: {list(OBS_SPECS)}")
    return OBS_SPECS[name]


# Current compact observation (V3): 3 unique + 9 shared = 12 dims.
OBS_V3 = ObservationSpec(
    name="v3_12d",
    version=3,
    unique_features=(
        FeatureSpec("load_feat"),
        FeatureSpec("re_feat"),
        FeatureSpec("soc"),
    ),
    shared_features=(
        FeatureSpec("last_lmp_norm"),
        FeatureSpec("block_pos"),
        FeatureSpec("avg_lmp_norm"),
        FeatureSpec("slr_norm"),
        FeatureSpec("avg_other_bid"),
        FeatureSpec("bid_std"),
        FeatureSpec("ema_dev"),
        FeatureSpec("price_trend"),
        FeatureSpec("pred_lmp_norm"),
    ),
)
register_obs_spec(OBS_V3)

# The non-storage action space: a multiplier on the agent's own declared price
# and an adder to its own offer. Retained for load and prosumer participants,
# whose `bid_value` and `offer_cost` are meaningful settlement prices that the
# action shades around.
ACTION_BID_OFFER_V1 = ActionSpec(
    name="bid_offer_v1",
    version=1,
    action_names=("bid_mult", "offer_adder"),
    bounds={"bid_mult": (0.3, 1.8), "offer_adder": (0.0, 50.0)},
    truthful={"bid_mult": 1.0, "offer_adder": 0.0},
)
register_action_spec(ACTION_BID_OFFER_V1)


# The storage action space: the two reservation prices themselves, in CNY/MWh.
#
# This replaced `anchor * bid_mult` because the multiplicative form could not
# reach the truthful declaration. With the anchor at 50 and `bid_mult` floored
# at 0.3, the lowest charge price a unit could declare was 15 CNY/MWh, so
# "declare no premium" -- the point every baseline and every comparison is
# measured from -- was outside the space the policy could express.
#
# Both bounds are non-negative, and that is a property of the mechanism rather
# than a modelling choice: the clearing costs charging at the charge price on
# top of the degradation charge, and the decomposition of charge and discharge
# into the parts of one net flow is exact only while that combined coefficient
# stays negative. A quote below `-cycle_cost` would flip it. See
# `dispatch_core.check_storage_quote_signs`.
ACTION_STORAGE_QUOTE_V2 = ActionSpec(
    name="storage_quote_v2",
    version=2,
    action_names=("charge_price", "discharge_price"),
    bounds={"charge_price": (0.0, 150.0), "discharge_price": (0.0, 300.0)},
    truthful={"charge_price": 0.0, "discharge_price": 0.0},
)
register_action_spec(ACTION_STORAGE_QUOTE_V2)


def action_spec_for(agent, storage_spec: ActionSpec = None,
                    other_spec: ActionSpec = None) -> ActionSpec:
    """Which action space this participant's quotes live in.

    Chosen by `participant_type`, never by whether the agent carries a battery:
    a prosumer with storage keeps the declared-price pair its load and
    generation give it, and only a battery alone declares reservation prices.
    Every producer of actions goes through this, so the rule cannot drift
    between them.
    """
    if getattr(agent, "participant_type", "prosumer") == "storage":
        return storage_spec or ACTION_STORAGE_QUOTE_V2
    return other_spec or ACTION_BID_OFFER_V1


def _check_truthful_is_reachable() -> None:
    """A declared truthful point that the space cannot express is a bug.

    Run at import, so the invariant cannot be lost by editing a spec without
    running the tests that use it.
    """
    for spec in ACTION_SPECS.values():
        assert set(spec.truthful) == set(spec.action_names), (
            f"action spec {spec.name!r} declares a truthful point for "
            f"{sorted(spec.truthful)} against actions "
            f"{sorted(spec.action_names)}")
        for name in spec.action_names:
            lo, hi = spec.bounds[name]
            value = spec.truthful[name]
            assert lo <= value <= hi, (
                f"action spec {spec.name!r}: the truthful {name} is {value}, "
                f"outside its own bounds [{lo}, {hi}]")


_check_truthful_is_reachable()
