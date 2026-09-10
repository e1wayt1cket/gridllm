"""Unit tests for the unified data layer (src/data_aggregator.py).

Every test builds its own artifact tree under tmp_path: no Gurobi, no torch,
no reads of the real outputs/. The aggregator is the only place that knows how
run artifacts, TensorBoard logs, evaluation CSVs and policy directories are
laid out, so these tests pin that knowledge.
"""

import json
import os

import numpy as np
import pytest

import data_aggregator as da


# --------------------------------------------------------------------------
# Fixtures: a synthetic artifacts tree
# --------------------------------------------------------------------------
def _write_run(root, run_id, timestamp, metrics_rows=None, metrics_header=None,
               manifest_extra=None, log_dir=None):
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_sha": "a" * 40,
        "cli_args": {"algo": "matd3", "episodes": 8, "seed": 42,
                     "save_dir": "policies/run_x"},
        "obs_spec": {"name": "v3_12d"},
        "action_spec": {"name": "bid_offer_v1"},
        "policy_paths": {},
        "kpi": {"best_episode": 4, "n_episodes": 8},
    }
    if log_dir is not None:
        manifest["log_dir"] = log_dir
    manifest.update(manifest_extra or {})
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"opf_mode": "socp"}, f)
    with open(os.path.join(run_dir, "kpi.json"), "w") as f:
        json.dump(manifest["kpi"], f)
    if metrics_header is not None:
        with open(os.path.join(run_dir, "metrics.csv"), "w") as f:
            f.write(",".join(metrics_header) + "\n")
            for row in (metrics_rows or []):
                f.write(",".join(str(row.get(c, "")) for c in metrics_header)
                        + "\n")
    return run_dir


_LEGACY_HEADER = ["episode", "mean_reward", "welfare", "re_rate",
                  "critic_loss", "actor_loss", "scenario"]
_NEW_HEADER = _LEGACY_HEADER + ["reward__A", "reward__B",
                                "q1_mean", "q2_mean", "q_gap",
                                "cs_delta", "cp_delta", "capture_fellbacks"]


def test_list_runs_reports_legacy_and_extended_runs(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-20260910-120000-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_rows=[{"episode": 1, "mean_reward": -100.0,
                              "scenario": "baseline"}],
               metrics_header=_LEGACY_HEADER)
    _write_run(root, "run-20260910-130000-bbbbbbbb", "2026-09-10T13:00:00",
               metrics_rows=[{"episode": 1, "mean_reward": -100.0,
                              "scenario": "baseline", "cs_delta": 5.0,
                              "capture_fellbacks": 0}],
               metrics_header=_NEW_HEADER)

    df = da.list_runs(root)
    assert len(df) == 2
    by_id = {r["run_id"]: r for r in df.to_dict("records")}
    legacy = by_id["run-20260910-120000-aaaaaaaa"]
    extended = by_id["run-20260910-130000-bbbbbbbb"]

    assert legacy["has_econ"] is False
    assert extended["has_econ"] is True
    assert legacy["algo"] == "matd3"
    assert legacy["seed"] == 42
    assert legacy["save_dir"] == "policies/run_x"
    assert legacy["best_episode"] == 4
    # Discovery reads only the CSV header, so it stays cheap on a long run.
    assert "cs_delta" in extended["metrics_columns"]


def test_list_runs_skips_a_corrupt_manifest_without_raising(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-good-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_header=_LEGACY_HEADER)
    bad = os.path.join(root, "run-bad-bbbbbbbb")
    os.makedirs(bad)
    with open(os.path.join(bad, "manifest.json"), "w") as f:
        f.write("{not valid json")

    df = da.list_runs(root)
    assert df["run_id"].tolist() == ["run-good-aaaaaaaa"]
    # The skip is reported rather than silent.
    assert any("run-bad-bbbbbbbb" in w for w in da.warnings())


def test_load_metrics_backfills_canonical_columns_for_a_legacy_file(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-legacy-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_rows=[{"episode": 1, "mean_reward": -100.0,
                              "welfare": 8000.0, "re_rate": 52.0,
                              "critic_loss": 1.0, "actor_loss": 0.5,
                              "scenario": "baseline"}],
               metrics_header=_LEGACY_HEADER)

    df = da.load_metrics("run-legacy-aaaaaaaa", root)
    # Existing columns are intact and the reader does not raise.
    assert df["mean_reward"].tolist() == [-100.0]
    assert df["welfare"].tolist() == [8000.0]
    # Every canonical column is addressable; the ones a legacy file lacks are
    # present but empty.
    for col in da.CANONICAL_METRIC_COLUMNS:
        assert col in df.columns, col
    assert df["cs_delta"].isna().all()
    assert df["q_gap"].isna().all()


def test_load_metrics_returns_empty_frame_when_the_csv_is_absent(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-nometrics-aaaaaaaa", "2026-09-10T12:00:00")
    df = da.load_metrics("run-nometrics-aaaaaaaa", root)
    assert len(df) == 0
    for col in da.CANONICAL_METRIC_COLUMNS:
        assert col in df.columns, col


# --------------------------------------------------------------------------
# Per-agent reward: metrics.csv preferred, TensorBoard as fallback
# --------------------------------------------------------------------------
def _write_tb_log(path, agents, episodes=3):
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(path)
    for ep in range(episodes):
        for i, nm in enumerate(agents):
            w.add_scalar(f"Reward/{nm}", 100.0 * (i + 1) + ep, ep)
        w.add_scalar("Reward/mean", 50.0 + ep, ep)
    w.close()


def test_agent_reward_frame_prefers_metrics_columns(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-wide-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_rows=[
                   {"episode": 1, "reward__A": 10.0, "reward__B": 20.0},
                   {"episode": 2, "reward__A": 11.0, "reward__B": 21.0}],
               metrics_header=_NEW_HEADER)

    df = da.agent_reward_frame("run-wide-aaaaaaaa", root)
    assert set(df["source"]) == {"metrics"}
    got = {(r["episode"], r["agent"]): r["reward"]
           for r in df.to_dict("records")}
    assert got[(1, "A")] == 10.0
    assert got[(2, "B")] == 21.0


def test_agent_reward_frame_falls_back_to_tensorboard(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    tb_dir = str(tmp_path / "runs" / "train-multi-20260910-120000")
    _write_tb_log(tb_dir, ["A", "B"])
    _write_run(root, "run-tb-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_rows=[{"episode": 1}], metrics_header=_LEGACY_HEADER,
               log_dir=tb_dir)

    df = da.agent_reward_frame("run-tb-aaaaaaaa", root)
    assert set(df["source"]) == {"tensorboard"}
    got = {(r["episode"], r["agent"]): r["reward"]
           for r in df.to_dict("records")}
    assert got[(0, "A")] == 100.0
    assert got[(0, "B")] == 200.0
    assert got[(2, "A")] == 102.0
    # The fleet mean is not an agent.
    assert "mean" not in set(df["agent"])


def test_agent_reward_frame_prefers_a_complete_wide_block_only(tmp_path):
    # One agent has a metrics column and the other does not: a partial wide
    # block would silently drop an agent, so the frame falls back for both.
    root = str(tmp_path / "outputs" / "rl")
    tb_dir = str(tmp_path / "runs" / "train-multi-20260910-120000")
    _write_tb_log(tb_dir, ["A", "B"])
    _write_run(root, "run-partial-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_rows=[{"episode": 1, "reward__A": 10.0}],
               metrics_header=_NEW_HEADER, log_dir=tb_dir)

    df = da.agent_reward_frame("run-partial-aaaaaaaa", root)
    assert set(df["source"]) == {"tensorboard"}


# --------------------------------------------------------------------------
# TensorBoard log linking
# --------------------------------------------------------------------------
def test_link_log_dir_is_exact_when_the_manifest_records_it(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    tb_dir = str(tmp_path / "runs" / "train-multi-20260910-120000")
    _write_tb_log(tb_dir, ["A"])
    _write_run(root, "run-exact-aaaaaaaa", "2026-09-10T12:00:00",
               log_dir=tb_dir)
    path, mode = da.link_log_dir("run-exact-aaaaaaaa", root)
    assert mode == "exact"
    assert path == tb_dir


def test_link_log_dir_guesses_by_timestamp_within_the_window(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    runs_root = str(tmp_path / "runs")
    tb_dir = os.path.join(runs_root, "train-multi-20260910-120002")
    _write_tb_log(tb_dir, ["A"])
    _write_run(root, "run-guess-aaaaaaaa", "2026-09-10T12:00:00")

    path, mode = da.link_log_dir("run-guess-aaaaaaaa", root,
                                 runs_dir=runs_root)
    assert mode == "nearest"
    assert path == tb_dir


def test_link_log_dir_refuses_to_guess_between_two_candidates(tmp_path):
    # Two logs starting in the window means the guess could plot another run's
    # curve; reporting that is better than picking one.
    root = str(tmp_path / "outputs" / "rl")
    runs_root = str(tmp_path / "runs")
    _write_tb_log(os.path.join(runs_root, "train-multi-20260910-120001"), ["A"])
    _write_tb_log(os.path.join(runs_root, "train-multi-20260910-120005"), ["A"])
    _write_run(root, "run-ambig-aaaaaaaa", "2026-09-10T12:00:00")

    path, mode = da.link_log_dir("run-ambig-aaaaaaaa", root,
                                 runs_dir=runs_root)
    assert mode == "ambiguous"
    assert path is None


def test_link_log_dir_reports_none_when_nothing_is_close(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    runs_root = str(tmp_path / "runs")
    _write_tb_log(os.path.join(runs_root, "train-multi-20200101-000000"), ["A"])
    _write_run(root, "run-nomatch-aaaaaaaa", "2026-09-10T12:00:00")

    path, mode = da.link_log_dir("run-nomatch-aaaaaaaa", root,
                                 runs_dir=runs_root)
    assert mode == "none"
    assert path is None


# --------------------------------------------------------------------------
# Policy directories and evaluation results
# --------------------------------------------------------------------------
def test_list_policy_dirs_reports_only_complete_checkpoints(tmp_path):
    root = str(tmp_path / "policies")
    d = os.path.join(root, "run_x")
    os.makedirs(os.path.join(d, "best"))
    os.makedirs(os.path.join(d, "last"))
    for nm in ("A", "B"):
        open(os.path.join(d, f"{nm}.pt"), "w").close()
        open(os.path.join(d, "best", f"{nm}.pt"), "w").close()
        open(os.path.join(d, "last", f"{nm}.pt"), "w").close()
        for ck in (50, 100):
            open(os.path.join(d, f"{nm}_ckpt_{ck}.pt"), "w").close()
    # A half-written checkpoint: only one of the two agents has it.
    open(os.path.join(d, "A_ckpt_150.pt"), "w").close()

    df = da.list_policy_dirs(root)
    assert len(df) == 1
    row = df.to_dict("records")[0]
    assert row["name"] == "run_x"
    assert row["checkpoints"] == [50, 100]      # 150 dropped as incomplete
    assert set(row["roles"]) == {"final", "best", "last"}
    assert row["n_agents"] == 2


def test_list_eval_results_tolerates_files_without_consumer_columns(tmp_path):
    root = str(tmp_path / "results")
    os.makedirs(root)
    with open(os.path.join(root, "a_eval.csv"), "w") as f:
        f.write("scenario,agent,profit_delta,welfare_delta\n")
        f.write("baseline,ALL,1509.0,765.0\n")
    with open(os.path.join(root, "b_eval.csv"), "w") as f:
        f.write("scenario,agent,profit_delta,cs_delta,cp_delta\n")
        f.write("congestion,ALL,1803.0,7.67,-7.68\n")

    df = da.list_eval_results(root)
    assert len(df) == 2
    by_file = {r["source_file"]: r for r in df.to_dict("records")}
    assert by_file["a_eval.csv"]["has_consumer_metrics"] is False
    assert by_file["b_eval.csv"]["has_consumer_metrics"] is True
    assert by_file["b_eval.csv"]["cs_delta"] == 7.67
    # The missing column is addressable rather than absent.
    assert np.isnan(by_file["a_eval.csv"]["cs_delta"])


def test_list_policy_dirs_maps_back_to_the_run_that_wrote_it(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-mapped-aaaaaaaa", "2026-09-10T12:00:00",
               manifest_extra={"cli_args": {"algo": "matd3", "seed": 42,
                                            "save_dir": "policies\\run_x"}})
    pol = str(tmp_path / "policies")
    os.makedirs(os.path.join(pol, "run_x"))
    open(os.path.join(pol, "run_x", "A.pt"), "w").close()

    df = da.list_policy_dirs(pol)
    row = df.to_dict("records")[0]
    # The manifest's save_dir uses a backslash; the link must survive it.
    assert da.policy_dir_run_id(row["name"], root) \
        == "run-mapped-aaaaaaaa"


def test_run_detail_gathers_the_page_payload(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-detail-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_rows=[{"episode": 1, "mean_reward": -100.0,
                              "reward__A": 10.0, "reward__B": 20.0,
                              "cs_delta": 5.0}],
               metrics_header=_NEW_HEADER)
    d = da.run_detail("run-detail-aaaaaaaa", root)
    assert d["run_id"] == "run-detail-aaaaaaaa"
    assert d["kpi"]["best_episode"] == 4
    assert len(d["metrics"]) == 1
    assert d["config"]["opf_mode"] == "socp"
    assert set(d["per_agent_rewards"]["agent"]) == {"A", "B"}


def test_repo_root_functions_are_consistent():
    # Every root derives from the single repo_root() definition, so a caller
    # never has to know the layout twice.
    root = da.repo_root()
    assert os.path.isdir(root)
    assert os.path.dirname(da.artifacts_root()) == os.path.join(root,
                                                                "outputs")
    for fn in (da.results_root, da.policies_root, da.runs_root):
        assert os.path.dirname(fn()) == root, fn.__name__
        assert os.path.basename(fn()) in ("results", "policies", "runs")


# --------------------------------------------------------------------------
# Fleet rows: one row per (evaluation file, scenario) for the fleet as a whole
# --------------------------------------------------------------------------
def _write_eval_file(root, name, rows):
    os.makedirs(root, exist_ok=True)
    cols = ["scenario", "agent", "profit_delta", "genuine_welfare_delta",
            "cs_delta", "cp_delta", "lmp_markup_delta", "market_power_power"]
    with open(os.path.join(root, name), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def test_fleet_rows_keeps_only_the_aggregate_row(tmp_path):
    # Per-agent rows repeat the day-level consumer columns as blanks; only the
    # ALL row carries the fleet accounting, so mixing them would double-count.
    root = str(tmp_path / "results")
    _write_eval_file(root, "exp_a_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1509.0,
         "genuine_welfare_delta": 765.0, "cs_delta": 193.4, "cp_delta": -193.4,
         "lmp_markup_delta": -0.01, "market_power_power": 153.5},
        {"scenario": "baseline", "agent": "Bus5R", "profit_delta": 100.0},
    ])
    df = da.fleet_rows(root)
    assert len(df) == 1
    assert df.iloc[0]["agent"] == "ALL"
    assert df.iloc[0]["cs_delta"] == 193.4


def test_fleet_rows_labels_each_file_with_its_policy(tmp_path):
    root = str(tmp_path / "results")
    _write_eval_file(root, "e8_matd3_10_eval.csv", [
        {"scenario": "congestion", "agent": "ALL", "cs_delta": 8.4}])
    _write_eval_file(root, "e9_matd3_p03_eval.csv", [
        {"scenario": "congestion", "agent": "ALL", "cs_delta": 8.05}])
    df = da.fleet_rows(root)
    labels = set(df["policy_label"])
    assert labels == {"e8_matd3_10", "e9_matd3_p03"}
    # The source file stays addressable for drill-down.
    assert set(df["source_file"]) == {"e8_matd3_10_eval.csv",
                                      "e9_matd3_p03_eval.csv"}


def test_fleet_rows_reports_missing_consumer_columns_as_empty(tmp_path):
    # Older evaluation files predate the three-layer columns.
    root = str(tmp_path / "results")
    _write_eval_file(root, "old_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1509.0}])
    df = da.fleet_rows(root)
    assert len(df) == 1
    assert np.isnan(df.iloc[0]["cs_delta"])
    assert df.iloc[0]["profit_delta"] == 1509.0


def test_fleet_rows_returns_empty_frame_without_files(tmp_path):
    df = da.fleet_rows(str(tmp_path / "nothing"))
    assert len(df) == 0
    assert "policy_label" in df.columns
    assert "cs_delta" in df.columns


# --------------------------------------------------------------------------
# Accounting convention: the 2026-09-09 consumer-surplus fix flipped the sign
# of cs/cp, so mixing files across it inverts the conclusion.
# --------------------------------------------------------------------------
def test_fleet_rows_tags_the_new_consumer_surplus_semantics(tmp_path):
    # A file whose consumer surplus is positive under the corrected
    # convention must be marked as such, and one from before the p_dis fix as
    # legacy, so a chart never pools the two. The generations are separated by
    # when the file was written.
    import os
    import time
    root = str(tmp_path / "results")
    _write_eval_file(root, "quickval_new_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "cs_delta": 428.7}])
    _write_eval_file(root, "three_layer_old_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "cs_delta": -5205.4}])
    # Backdate the second file to before the 2026-09-09 fix.
    old = os.path.join(root, "three_layer_old_eval.csv")
    stamp = time.mktime((2026, 9, 8, 17, 0, 0, 0, 0, -1))
    os.utime(old, (stamp, stamp))


def test_fleet_rows_dates_the_fix_by_time_not_by_day(tmp_path):
    # The corrected convention first appeared at 2026-09-09 12:23, but files
    # from that same morning still carry the old semantics; a date-only cutoff
    # would misfile them.
    import os
    import time
    root = str(tmp_path / "results")
    _write_eval_file(root, "same_day_before_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "cs_delta": -5127.7}])
    _write_eval_file(root, "same_day_after_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "cs_delta": 428.7}])
    before = os.path.join(root, "same_day_before_eval.csv")
    after = os.path.join(root, "same_day_after_eval.csv")
    t_before = time.mktime((2026, 9, 9, 0, 54, 0, 0, 0, -1))
    t_after = time.mktime((2026, 9, 9, 12, 23, 0, 0, 0, -1))
    os.utime(before, times=(t_before, t_before))
    os.utime(after, times=(t_after, t_after))

    by = {r["policy_label"]: r["cs_convention"]
          for r in da.fleet_rows(root).to_dict("records")}
    assert by["same_day_after"] == "p_dis_excluded"
    assert by["same_day_before"] == "p_dis_offset_legacy"


def test_fleet_rows_marks_a_file_without_consumer_columns_as_unknown(tmp_path):
    root = str(tmp_path / "results")
    _write_eval_file(root, "old_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1509.0}])
    df = da.fleet_rows(root)
    assert df.iloc[0]["cs_convention"] == "unknown"


def test_fleet_rows_can_restrict_to_one_convention(tmp_path):
    import os
    import time
    root = str(tmp_path / "results")
    _write_eval_file(root, "a_new_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "cs_delta": 428.7}])
    _write_eval_file(root, "b_old_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "cs_delta": -5205.4}])
    old = os.path.join(root, "b_old_eval.csv")
    stamp = time.mktime((2026, 9, 8, 17, 0, 0, 0, 0, -1))
    os.utime(old, (stamp, stamp))

    assert list(da.fleet_rows(root, convention="p_dis_excluded")
                ["policy_label"]) == ["a_new"]
    assert list(da.fleet_rows(root, convention="p_dis_offset_legacy")
                ["policy_label"]) == ["b_old"]
    assert da.CONSUMER_CONVENTIONS == ("p_dis_excluded",
                                       "p_dis_offset_legacy", "unknown")


# --------------------------------------------------------------------------
# Profit / consumer-surplus trade-off pairing
# --------------------------------------------------------------------------
def test_pareto_pairs_joins_profit_and_surplus_per_policy_scenario(tmp_path):
    root = str(tmp_path / "results")
    _write_eval_file(root, "p_a_eval.csv", [
        {"scenario": "peak_load", "agent": "ALL", "profit_delta": 14509.2,
         "cs_delta": 3640.6, "genuine_welfare_delta": -16410.7},
        {"scenario": "peak_load", "agent": "Bus5R", "profit_delta": 100.0}])
    df = da.pareto_pairs(root)
    assert len(df) == 1                       # per-agent row excluded
    row = df.iloc[0]
    assert row["scenario"] == "peak_load"
    assert row["profit_delta"] == 14509.2
    assert row["cs_delta"] == 3640.6
    # A policy that profits while consumer surplus also rises is the
    # non-conflicting quadrant the trade-off analysis looks for.
    assert row["cs_rises"] == True and row["profit_rises"] == True


def test_pareto_pairs_flags_the_conflicting_quadrant(tmp_path):
    root = str(tmp_path / "results")
    _write_eval_file(root, "p_b_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 15000.0,
         "cs_delta": -5000.0}])
    row = da.pareto_pairs(root).iloc[0]
    assert row["profit_rises"] is True or row["profit_rises"] == True
    assert row["cs_rises"] == False
    assert row["conflict"] == True


def test_pareto_pairs_excludes_rows_without_consumer_accounting(tmp_path):
    root = str(tmp_path / "results")
    _write_eval_file(root, "p_c_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1500.0}])
    assert len(da.pareto_pairs(root)) == 0


def test_pareto_pairs_carries_the_convention_for_axis_labelling(tmp_path):
    import os
    import time
    root = str(tmp_path / "results")
    _write_eval_file(root, "legacy_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1509.0,
         "cs_delta": -5205.0}])
    p = os.path.join(root, "legacy_eval.csv")
    t = time.mktime((2026, 9, 8, 17, 0, 0, 0, 0, -1))
    os.utime(p, times=(t, t))
    # The default keeps only the corrected generation, so a legacy point has
    # to be asked for explicitly — and then it is labelled as such.
    assert len(da.pareto_pairs(root)) == 0
    row = da.pareto_pairs(root, convention=None).iloc[0]
    assert row["cs_convention"] == "p_dis_offset_legacy"


# --------------------------------------------------------------------------
# Experiment grouping: what a result file's name says about its settings
# --------------------------------------------------------------------------
def test_experiment_family_groups_related_results():
    assert da.experiment_family("e8_matd3_10") == "E8 网络耦合"
    assert da.experiment_family("e8_phys_08") == "E8 网络耦合"
    assert da.experiment_family("e9_matd3_p03") == "E9 市场力惩罚"
    assert da.experiment_family("qmatd3_rot_seed42_three_layer") == "qmatd3 对照"
    assert da.experiment_family("quickval_matd3_cc_ref") == "快速验证"
    assert da.experiment_family("masac_pilot_masac_bl") == "MASAC pilot"
    assert da.experiment_family("matd3_cc_rot_seed42_three_layer") == "MATD3 主基线"
    assert da.experiment_family("something_unknown") == "其他"


def test_experiment_variant_extracts_the_manipulated_setting():
    # The filename encodes what was varied, which is the column a reader needs
    # to interpret a row; a bare policy_label does not carry it.
    assert da.experiment_variant("e8_matd3_10") == "capacity=1.0"
    assert da.experiment_variant("e8_matd3_15") == "capacity=1.5"
    assert da.experiment_variant("e8_phys_08") == "capacity=0.8"
    assert da.experiment_variant("e9_matd3_p03") == "λ=0.3"
    assert da.experiment_variant("qmatd3_rot_seed42_three_layer") == "seed=42"
    assert da.experiment_variant("matd3_cc_rot_seed7_three_layer") == "seed=7"


def test_experiment_variant_is_empty_when_nothing_is_encoded():
    assert da.experiment_variant("quickval_matd3_cc_ref") == ""


def test_experiment_table_annotates_every_fleet_row(tmp_path):
    import os
    root = str(tmp_path / "results")
    _write_eval_file(root, "e8_matd3_08_eval.csv", [
        {"scenario": "congestion", "agent": "ALL", "profit_delta": 18400.0,
         "genuine_welfare_delta": 1440.0, "cs_delta": 8560.0,
         "cp_delta": -8570.0, "lmp_markup_delta": -0.036,
         "market_power_power": 4150.0}])
    df = da.experiment_table(root, convention=None)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["family"] == "E8 网络耦合"
    assert row["variant"] == "capacity=0.8"
    assert row["scenario"] == "congestion"
    assert row["cs_delta"] == 8560.0


def test_experiment_table_lists_families_in_a_stable_order(tmp_path):
    import os
    root = str(tmp_path / "results")
    _write_eval_file(root, "e9_matd3_p03_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1.0}])
    _write_eval_file(root, "e8_matd3_10_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1.0}])
    df = da.experiment_table(root, convention=None)
    assert list(df["family"]) == ["E8 网络耦合", "E9 市场力惩罚"]


def test_experiment_table_accepts_a_family_filter(tmp_path):
    import os
    root = str(tmp_path / "results")
    _write_eval_file(root, "e9_matd3_p03_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1.0}])
    _write_eval_file(root, "e8_matd3_10_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 1.0}])
    df = da.experiment_table(root, convention=None,
                             families=["E9 市场力惩罚"])
    assert list(df["policy_label"]) == ["e9_matd3_p03"]


# --------------------------------------------------------------------------
# Consolidated summaries
# --------------------------------------------------------------------------
def test_run_summary_reports_training_health_per_run(tmp_path):
    # One row per run: what it trained, how it ended, and whether the curve
    # stayed bounded. The critic-loss spread is the cheap divergence check.
    root = str(tmp_path / "outputs" / "rl")
    rows = [{"episode": i, "mean_reward": -1000.0 + 100.0 * i,
             "critic_loss": 2.0 - 0.01 * i, "actor_loss": -0.5,
             "scenario": "baseline"} for i in range(1, 9)]
    _write_run(root, "run-20260910-120000-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_rows=rows, metrics_header=_LEGACY_HEADER,
               manifest_extra={"kpi": {"best_episode": 4,
                                       "n_episodes": 8,
                                       "final_mean_reward": -300.0}})

    df = da.run_summary(root)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["algo"] == "matd3"
    assert r["seed"] == 42
    assert r["episodes"] == 8
    assert r["best_episode"] == 4
    assert r["critic_first"] == pytest.approx(2.0 - 0.01 * 1.5, abs=0.05)
    assert r["critic_last"] == pytest.approx(2.0 - 0.01 * 6.5, abs=0.05)
    assert r["critic_max"] == pytest.approx(1.99, abs=0.01)
    assert r["critic_trend"] == "falling"


def test_run_summary_flags_a_diverging_curve(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    rows = [{"episode": i, "mean_reward": float(i),
             "critic_loss": 1.0 * (2.0 ** i), "scenario": "baseline"}
            for i in range(1, 7)]
    _write_run(root, "run-20260910-130000-bbbbbbbb", "2026-09-10T13:00:00",
               metrics_rows=rows, metrics_header=_LEGACY_HEADER)
    r = da.run_summary(root).iloc[0]
    assert r["critic_trend"] == "rising"
    # Losses are 2^i for i in 1..6, so the peak is 2**6.
    assert r["critic_max"] == pytest.approx(64.0)


def test_run_summary_handles_a_run_with_no_loss_recorded(tmp_path):
    # Early episodes are random exploration, so a short run can have none.
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-20260910-140000-cccccccc", "2026-09-10T14:00:00",
               metrics_rows=[{"episode": 1, "mean_reward": -100.0,
                              "scenario": "baseline"}],
               metrics_header=_LEGACY_HEADER)
    r = da.run_summary(root).iloc[0]
    assert r["critic_max"] is None or np.isnan(r["critic_max"])
    assert r["critic_trend"] == "unknown"


def test_run_summary_marks_whether_economics_were_recorded(tmp_path):
    root = str(tmp_path / "outputs" / "rl")
    _write_run(root, "run-legacy-aaaaaaaa", "2026-09-10T12:00:00",
               metrics_rows=[{"episode": 1}], metrics_header=_LEGACY_HEADER)
    _write_run(root, "run-new-bbbbbbbb", "2026-09-10T13:00:00",
               metrics_rows=[{"episode": 1, "cs_delta": 5.0,
                              "capture_fellbacks": 0}],
               metrics_header=_NEW_HEADER)
    df = da.run_summary(root).set_index("run_id")
    assert bool(df.loc["run-legacy-aaaaaaaa", "has_econ"]) is False
    assert bool(df.loc["run-new-bbbbbbbb", "has_econ"]) is True


def test_eval_summary_averages_scenarios_per_convention(tmp_path):
    # The two consumer-accounting generations must never be averaged together.
    import os
    import time
    root = str(tmp_path / "results")
    _write_eval_file(root, "n1_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 100.0,
         "cs_delta": 10.0},
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 200.0,
         "cs_delta": 20.0}])
    _write_eval_file(root, "o1_eval.csv", [
        {"scenario": "baseline", "agent": "ALL", "profit_delta": 300.0,
         "cs_delta": -30.0}])
    old = os.path.join(root, "o1_eval.csv")
    t = time.mktime((2026, 9, 8, 17, 0, 0, 0, 0, -1))
    os.utime(old, times=(t, t))

    s = da.eval_summary(root)
    by = {(r["cs_convention"], r["scenario"]): r
          for r in s.to_dict("records")}
    assert by[("p_dis_excluded", "baseline")]["n_points"] == 2
    assert by[("p_dis_excluded", "baseline")]["cs_delta"] == pytest.approx(15.0)
    assert by[("p_dis_offset_legacy", "baseline")]["n_points"] == 1
    assert by[("p_dis_offset_legacy", "baseline")]["cs_delta"] \
        == pytest.approx(-30.0)


def test_eval_summary_returns_empty_frame_without_results(tmp_path):
    s = da.eval_summary(str(tmp_path / "nothing"))
    assert len(s) == 0
    assert "cs_convention" in s.columns and "scenario" in s.columns
