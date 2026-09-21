"""Tests for structured RL run artifacts (run_artifacts)."""

import json
import os

import pytest

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
    art.record_eval(2, {"mean_reward": 50.0, "clearing_objective": 1000.0,
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
    assert cols[:7] == ["episode", "mean_reward", "clearing_objective", "re_rate",
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


def test_episode_log_reports_profit_alongside_the_reward(tmp_path):
    """A run must report the money, not only the learning signal.

    `mean_reward` is the differential signal the actor learns from; raw profit
    and its in-window truthful counterfactual are what that signal is built out
    of. A falling actor loss with a flat raw profit is a failed run, and that is
    only visible if the profit is a column. Neither column is the paper's
    benefit, which uses a whole-day all-truthful baseline instead.
    """
    art = _make_artifacts(str(tmp_path))
    art.record_episode(2, 12.0, 1000.0, 50.0, None, None, "baseline",
                       raw_profit=-151.8, local_baseline_profit=-20.0,
                       profit_delta_local=-131.8)
    row = art.metrics[-1]
    assert row["raw_profit_mean"] == -151.8
    assert row["local_baseline_profit_mean"] == -20.0
    assert row["profit_delta_local_mean"] == -131.8
    # The signal is the difference of the two, recorded rather than inferred.
    assert row["profit_delta_local_mean"] == (
        row["raw_profit_mean"] - row["local_baseline_profit_mean"])


def test_episode_log_still_accepts_rows_without_profit_columns(tmp_path):
    """An episode recorded without them keeps its old shape and stays readable."""
    art = _make_artifacts(str(tmp_path))
    art.record_episode(3, 1.0, 2.0, 3.0, None, None, "baseline")
    assert "raw_profit_mean" not in art.metrics[-1]
    loaded = RunArtifacts.load(art.save())
    assert loaded.metrics[0]["mean_reward"] == -480.0


# ---------------------------------------------------------------------------
# Validation persistence: the counterfactual has to survive the run that made it
# ---------------------------------------------------------------------------

#: The fields a reader needs to check a cycle's mean against the days it was
#: taken over. Written out rather than derived, because this list *is* the
#: requirement the protocol states.
VALIDATION_DAY_FIELDS = ("cycle", "day_id", "pi_ai", "pi_baseline",
                         "delta_pi", "delta_pi_percent")


def _record_one_validation(art, cycle=0, episode=20):
    """One cycle's validation pass: a summary row plus its per-day rows."""
    days = [{"day_id": d, "pi_ai": -100.0 + d, "pi_baseline": -110.0 + d,
             "delta_pi": 10.0, "delta_pi_percent": 10.0 / (110.0 - d)}
            for d in (3, 5, 9)]
    metrics = {"mean_reward": 5.0, "clearing_objective": 900.0, "re_rate": 40.0,
               "mean_pi_ai": -96.0, "mean_pi_baseline": -106.0,
               "mean_delta_pi": 10.0, "p_delta_pi_positive": 1.0,
               "n_days": 3, "n_days_dropped": 0, "days": days}
    art.record_eval(episode, metrics, is_best=True, cycle=cycle)
    art.record_validation_cycle(cycle, metrics, episode=episode)
    for day in days:
        art.record_validation_day(cycle, day, episode=episode)
    return metrics


def test_validation_day_rows_reach_eval_csv(tmp_path):
    """Phase 9: each cycle's per-day counterfactual is written, not just its mean."""
    art = _make_artifacts(str(tmp_path))
    _record_one_validation(art)
    root = art.save()
    import pandas as pd
    frame = pd.read_csv(os.path.join(root, "eval.csv"))
    for field in VALIDATION_DAY_FIELDS:
        assert field in frame.columns, field
    day_rows = frame[frame["day_id"].notna()]
    assert len(day_rows) == 3
    assert sorted(day_rows["day_id"]) == [3, 5, 9]
    # The row's own arithmetic, not a value carried in from elsewhere.
    for _, row in day_rows.iterrows():
        assert row["delta_pi"] == pytest.approx(row["pi_ai"] - row["pi_baseline"])


def test_validation_summary_row_carries_the_cycle_kpis(tmp_path):
    """The mean the cycle is selected on travels with the row it was measured on."""
    art = _make_artifacts(str(tmp_path))
    _record_one_validation(art)
    # Located by the field under test rather than by position: the per-day rows
    # share the file, and a future ordering change must not fake a pass.
    summary, = [r for r in art.evals if "mean_delta_pi" in r]
    assert summary["cycle"] == 0
    assert summary["mean_delta_pi"] == 10.0
    assert summary["mean_pi_ai"] == -96.0
    assert summary["mean_pi_baseline"] == -106.0
    assert summary["p_delta_pi_positive"] == 1.0
    assert summary["n_days"] == 3


def test_kpi_json_carries_one_entry_per_validation_cycle(tmp_path):
    art = _make_artifacts(str(tmp_path))
    _record_one_validation(art, cycle=0, episode=20)
    _record_one_validation(art, cycle=1, episode=40)
    root = art.save()
    with open(os.path.join(root, "kpi.json")) as f:
        kpi = json.load(f)
    assert [c["cycle"] for c in kpi["cycles"]] == [0, 1]
    assert all(c["days"] == 3 for c in kpi["cycles"])
    # Round-trips out of kpi.json rather than staying in the manifest.
    loaded = RunArtifacts.load(root)
    assert [c["cycle"] for c in loaded.validation] == [0, 1]
    assert "cycles" not in loaded.kpi


def test_a_run_without_validation_writes_no_cycles_block(tmp_path):
    """No validation must read as no validation, not as an empty pass."""
    art = _make_artifacts(str(tmp_path))
    root = art.save()
    with open(os.path.join(root, "kpi.json")) as f:
        assert "cycles" not in json.load(f)
    assert RunArtifacts.load(root).validation == []


def test_an_evaluation_without_kpis_leaves_the_cells_blank(tmp_path):
    """A missing measurement is absent, never a zero that reads as no gain."""
    art = _make_artifacts(str(tmp_path))
    art.record_eval(25, {"mean_reward": 1.0}, cycle=0)
    row = art.evals[0]
    for field in RunArtifacts._EVAL_KPI_FIELDS:
        assert field not in row, field


def test_episode_log_records_which_day_priced_it(tmp_path):
    """A cycle's coverage is checkable from the file the run wrote.

    Without the day on the row, a run that rotated through its twenty training
    days and a run stuck on one day produce the same forty shapeless rows.
    """
    art = _make_artifacts(str(tmp_path))
    art.record_episode(20, 1.0, 2.0, 3.0, None, None, "typical_day",
                       day_id=38, cycle=0)
    row = art.metrics[-1]
    assert row["day_id"] == 38
    assert row["cycle"] == 0
    # An unrotated run records neither rather than a misleading zero.
    art.record_episode(21, 1.0, 2.0, 3.0, None, None, "baseline")
    assert "day_id" not in art.metrics[-1]
    assert "cycle" not in art.metrics[-1]
