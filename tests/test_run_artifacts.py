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


def test_record_episode_carries_per_agent_q_and_econ_columns(tmp_path):
    art = _make_artifacts(str(tmp_path))
    art.record_episode(
        2, 5857.0, 3370.0, 59.8, 0.9, -0.5, "baseline",
        per_agent_reward={"Bus5R": 100.0, "Bus6R": 200.0},
        q_stats={"q1_mean": 1.5, "q2_mean": 1.4, "q_gap": 0.1},
        econ={"cs_delta": 3.0, "cp_delta": -3.0, "lmp_markup_delta": 0.01},
        capture_fellbacks=0)
    root = art.save()
    with open(os.path.join(root, "metrics.csv")) as f:
        header, *rows = [ln.strip() for ln in f if ln.strip()]
    cols = header.split(",")
    assert "reward__Bus5R" in cols
    assert "reward__Bus6R" in cols
    assert "q1_mean" in cols and "q2_mean" in cols and "q_gap" in cols
    assert "cs_delta" in cols and "cp_delta" in cols
    assert "capture_fellbacks" in cols
    # Existing columns keep their names and positions.
    assert cols[:7] == ["episode", "mean_reward", "welfare", "re_rate",
                        "critic_loss", "actor_loss", "scenario"]
    assert len(rows) == 3


def test_metrics_columns_are_stable_across_rows_with_ragged_extras(tmp_path):
    # Episode 1 has no econ (e.g. differential reward off) while episode 2
    # does; the file still needs one header and NaN for the missing cells.
    art = _make_artifacts(str(tmp_path))
    art.record_episode(2, 5857.0, 3370.0, 59.8,
                       per_agent_reward={"Bus5R": 100.0},
                       econ={"cs_delta": 3.0}, capture_fellbacks=0)
    root = art.save()
    with open(os.path.join(root, "metrics.csv")) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header = lines[0].split(",")
    # Every row has the same field count as the header.
    for ln in lines[1:]:
        assert len(ln.split(",")) == len(header), ln


def test_ordered_columns_puts_extras_after_the_base_columns():
    from run_artifacts import _ordered_columns
    rows = [{"episode": 1, "mean_reward": 1.0, "scenario": "baseline",
             "cs_delta": 2.0, "reward__Bus6R": 5.0, "reward__Bus5R": 4.0,
             "q_gap": 0.1, "capture_fellbacks": 0}]
    cols = _ordered_columns(rows)
    assert cols[:3] == ["episode", "mean_reward", "scenario"]
    # Per-agent reward columns are sorted so cross-run diffs stay readable.
    assert cols.index("reward__Bus5R") < cols.index("reward__Bus6R")
    assert cols.index("q_gap") < cols.index("cs_delta")
    assert cols[-1] == "capture_fellbacks"


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
