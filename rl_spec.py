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
    name: str
    version: int
    action_names: Tuple[str, ...]
    bounds: Dict[str, Tuple[float, float]]

    @property
    def dim(self) -> int:
        return len(self.action_names)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "action_names": list(self.action_names),
            "bounds": {k: list(v) for k, v in self.bounds.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSpec":
        return cls(
            name=d["name"],
            version=d["version"],
            action_names=tuple(d["action_names"]),
            bounds={k: tuple(v) for k, v in d["bounds"].items()},
        )


OBS_SPECS: Dict[str, ObservationSpec] = {}


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

# Current action space: bid_mult and offer_adder.
ACTION_BID_OFFER_V1 = ActionSpec(
    name="bid_offer_v1",
    version=1,
    action_names=("bid_mult", "offer_adder"),
    bounds={"bid_mult": (0.3, 1.8), "offer_adder": (0.0, 50.0)},
)
