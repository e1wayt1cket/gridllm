# validate_checkpoint_schema.py
"""A JSON schema for the next MATD3 checkpoint and a pure validator.

No torch, no env, no training. Validates a bundle dictionary against the three
levels: inference, resume, bundle.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-f]{64}$")
LEVELS = ("inference", "resume", "bundle")

SCHEMA_V2 = {
    "required": ("schema_version", "level", "run_id", "dataset", "seed",
                 "code_revision", "specs", "config_hash"),
    "resume_required": ("networks", "optimizers", "counters", "rng",
                        "buffer", "reward"),
    "bundle_required": ("data", "exposure_log", "eval_provenance"),
    "specs": ("obs_spec", "action_spec"),
    "network_agents": ("actors", "actor_targets", "critics", "critic_targets"),
    "counter_keys": ("total_steps", "episode", "updates", "buffer_len"),
    "rng_keys": ("python", "numpy", "torch_cpu"),
    "hash_keys": ("pairing", "bus_loads", "price_curves", "common_split"),
    "reward_keys": ("definition", "gamma", "scale", "potential",
                    "normalization", "terminal_rule", "b1_source"),
}


def _err(errors, path, message):
    errors.append(f"{path}: {message}")


def validate(bundle: dict) -> list:
    errors = []
    if not isinstance(bundle, dict):
        return ["bundle: not an object"]
    for key in SCHEMA_V2["required"]:
        if key not in bundle:
            _err(errors, key, "missing")
    level = bundle.get("level")
    if level not in LEVELS:
        _err(errors, "level", f"must be one of {LEVELS}")
    for key in ("run_id", "dataset", "code_revision"):
        if key in bundle and not isinstance(bundle[key], str):
            _err(errors, key, "must be a string")
    if "seed" in bundle and not isinstance(bundle["seed"], int):
        _err(errors, "seed", "must be an int")
    config_hash = bundle.get("config_hash")
    if config_hash is not None and not (isinstance(config_hash, str)
                                        and HEX64.match(config_hash)):
        _err(errors, "config_hash", "must be a 64-hex sha256")
    specs = bundle.get("specs", {})
    for key in SCHEMA_V2["specs"]:
        spec = specs.get(key)
        if not isinstance(spec, dict) or "name" not in spec \
                or "version" not in spec:
            _err(errors, f"specs.{key}", "must carry name and version")

    if level in ("resume", "bundle"):
        for key in SCHEMA_V2["resume_required"]:
            if key not in bundle:
                _err(errors, key, "missing for a resumable checkpoint")
        networks = bundle.get("networks", {})
        for key in SCHEMA_V2["network_agents"]:
            if not networks.get(key):
                _err(errors, f"networks.{key}", "must list every agent")
        for key in SCHEMA_V2["counter_keys"]:
            value = bundle.get("counters", {}).get(key)
            if not isinstance(value, int) or value < 0:
                _err(errors, f"counters.{key}", "must be a non-negative int")
        for key in SCHEMA_V2["rng_keys"]:
            if key not in bundle.get("rng", {}):
                _err(errors, f"rng.{key}", "missing")
        buffer = bundle.get("buffer", {})
        if buffer.get("saved") is True:
            if not buffer.get("path"):
                _err(errors, "buffer.path", "required when saved")
            if not isinstance(buffer.get("capacity"), int) \
                    or buffer["capacity"] <= 0:
                _err(errors, "buffer.capacity", "must be positive")
        elif buffer.get("saved") is False:
            if not buffer.get("reason"):
                _err(errors, "buffer.reason",
                     "a non-saved buffer must state why it is not resumable")
        else:
            _err(errors, "buffer.saved", "must be true or false")
        reward = bundle.get("reward", {})
        for key in SCHEMA_V2["reward_keys"]:
            if key not in reward:
                _err(errors, f"reward.{key}", "missing")
        if reward.get("gamma") is not None and not isinstance(
                reward["gamma"], (int, float)):
            _err(errors, "reward.gamma", "must be numeric")

    if level == "bundle":
        for key in SCHEMA_V2["bundle_required"]:
            if key not in bundle:
                _err(errors, key, "missing for a bundle")
        for key in SCHEMA_V2["hash_keys"]:
            value = bundle.get("data", {}).get(key)
            if not isinstance(value, str) or not HEX64.match(value):
                _err(errors, f"data.{key}", "must be a 64-hex sha256")
        for key in ("exposure_log", "eval_provenance"):
            value = bundle.get(key)
            if not isinstance(value, str) or not value:
                _err(errors, key, "a bundle must name a non-empty path")
    return errors


def _sample(level: str) -> dict:
    h = "a" * 64
    bundle = {
        "schema_version": "matd3-checkpoint-v2",
        "level": level,
        "run_id": "NEAR_11_8th",
        "dataset": "NEAR",
        "seed": 11,
        "code_revision": "rev",
        "config_hash": h,
        "specs": {"obs_spec": {"name": "v6_12d", "version": 6},
                  "action_spec": {"name": "storage_quote_v2", "version": 2}},
    }
    if level in ("resume", "bundle"):
        bundle.update({
            "networks": {"actors": ["ESS9", "ESS14", "ESS17", "ESS27"],
                         "actor_targets": ["ESS9", "ESS14", "ESS17", "ESS27"],
                         "critics": ["ESS9", "ESS14", "ESS17", "ESS27"],
                         "critic_targets": ["ESS9", "ESS14", "ESS17", "ESS27"]},
            "optimizers": {"actor": "adam", "critic": "adam"},
            "counters": {"total_steps": 480, "episode": 20, "updates": 417,
                         "buffer_len": 480},
            "rng": {"python": 1, "numpy": 2, "torch_cpu": "state",
                    "torch_cuda": None, "env": None},
            "buffer": {"saved": True, "path": "buffer/replay.npz",
                       "capacity": 20000, "index": 480, "count": 480},
            "reward": {"definition": "C2_potential_soc_time",
                       "gamma": 0.99, "scale": "fixed",
                       "potential": {"form": "soc_time", "kappa": 288.2,
                                     "bound": 184.4, "terminal": 0.0},
                       "normalization": "none",
                       "terminal_rule": "settlement_pin",
                       "b1_source": "b1_table.json"},
        })
    if level == "bundle":
        bundle.update({
            "data": {"pairing": h, "bus_loads": h, "price_curves": h,
                     "common_split": h},
            "exposure_log": "exposure.csv",
            "eval_provenance": "eval/provenance.json",
        })
    return bundle


def main(argv=None) -> int:
    root = Path.cwd()
    out = root / "data/processed/v2_cy_jn_2025/pilot"
    out.mkdir(parents=True, exist_ok=True)
    samples = {level: _sample(level) for level in LEVELS}
    (out / "checkpoint_schema_v2.sample.json").write_text(
        json.dumps(samples["bundle"], indent=2), encoding="utf-8")

    checks = {}
    for level, bundle in samples.items():
        checks[f"{level}_valid"] = validate(bundle) == []
    broken = json.loads(json.dumps(samples["resume"]))
    broken["networks"].pop("critics")
    checks["missing_critic_detected"] = bool(validate(broken))
    broken = json.loads(json.dumps(samples["resume"]))
    broken["buffer"] = {"saved": False}
    checks["buffer_without_reason_detected"] = bool(validate(broken))
    broken = json.loads(json.dumps(samples["bundle"]))
    broken["data"]["pairing"] = "not-a-hash"
    checks["bad_hash_detected"] = bool(validate(broken))
    checks["inference_does_not_require_buffer"] = validate(
        samples["inference"]) == []
    (out / "checkpoint_schema_validation.json").write_text(
        json.dumps({"checks": checks, "errors": {
            name: validate(bundle) for name, bundle in {
                "bundle": samples["bundle"],
                "inference": samples["inference"],
                "resume": samples["resume"],
            }.items()}}, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"wrote {out / 'checkpoint_schema_v2.sample.json'}")
    print(f"wrote {out / 'checkpoint_schema_validation.json'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
