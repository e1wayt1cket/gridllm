"""Tests for structured RL run artifacts (run_artifacts)."""

import json
import os

from models import MarketConfig
from run_artifacts import RunArtifacts, _json_safe
from rl_spec import OBS_V3, ACTION_BID_OFFER_V1


def _make_artifacts(base_dir):
    config = MarketConfig(opf_mode="socp", verbose=False)
    cli_args = {"algo": "matd3", "episodes": 8, "seed": 42}
    art = RunArtifacts.create(
        config, cli_args, OBS_V3.to_dict(), ACTION_BID_OFFER_V1.to_dict(),
        base_dir=base_dir)
    art.record_episode(0, -480.0, 7695.0, 56.0, 1.2, None, "baseline")
    art.record_episode(1, 5857.0, 3370.0, 59.8, 0.9, -0.5, "baseline")
    art.record_eval(2, {"mean_reward": 50.0, "welfare": 1000.0,
                        "re_rate": 50.0}, is_best=True)
    art.add_policy_paths("final", ["policies/x/Bus5R.pt"])
    art.add_policy_paths("best", ["policies/x/best/Bus5R.pt"])
    art.kpi = {"final_mean_reward": 5857.0, "best_episode": 2}
    return art


def test_create_save_load_round_trip(tmp_path):
    art = _make_artifacts(str(tmp_path))
    root = art.save()
    loaded = RunArtifacts.load(root)
    assert loaded.run_id == art.run_id
    assert loaded.cli_args["algo"] == "matd3"
    assert loaded.obs_spec["name"] == "v3_12d"
    assert loaded.action_spec["name"] == "bid_offer_v1"
    assert loaded.metrics[1]["mean_reward"] == 5857.0
    assert loaded.evals[0]["is_best"] is True
    assert loaded.policy_paths["final"] == ["policies/x/Bus5R.pt"]
    assert loaded.kpi["best_episode"] == 2


def test_save_writes_expected_files(tmp_path):
    art = _make_artifacts(str(tmp_path))
    root = art.save()
    for name in ("manifest.json", "config.json", "metrics.csv",
                 "eval.csv", "kpi.json"):
        assert os.path.exists(os.path.join(root, name)), name
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["obs_spec"]["name"] == "v3_12d"
    assert manifest["cli_args"]["algo"] == "matd3"
    # pandas is a lazy import in run_artifacts; read the CSV without it.
    with open(os.path.join(root, "metrics.csv")) as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == 3  # header + 2 rows


def test_json_safe_handles_config_with_tuples():
    config = MarketConfig(opf_mode="socp", verbose=False)
    safe = _json_safe(config)
    assert isinstance(safe, dict)
    # The market-design bid/offer ranges are stored as tuples in the dataclass.
    ranges = safe["market_design"]["bid_mult_range"]
    assert isinstance(ranges, list)
    assert ranges == [0.3, 1.8]
    # Nested dataclasses were fully recursed.
    assert isinstance(safe["network"], dict)
    # Round-trips through json.dumps without error.
    json.dumps(safe, default=str)


def test_json_safe_numpy_values():
    import numpy as np
    assert _json_safe(np.float32(1.5)) == 1.5
    assert _json_safe(np.array([1, 2, 3])) == [1, 2, 3]
    assert _json_safe((1, "a")) == [1, "a"]
