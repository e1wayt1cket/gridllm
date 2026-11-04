"""Contracts for the D+A training entry's run guards and fixed-subset mode."""

from __future__ import annotations

from pathlib import Path

import pytest

import train_da_gate as t


def test_no_improvement_streak_counts_trailing_non_best():
    evaluations = [
        {"mean_fleet_delta": 1.0},
        {"mean_fleet_delta": 2.0},
        {"mean_fleet_delta": 1.5},
        {"mean_fleet_delta": 0.5},
    ]
    assert t.no_improvement_streak(evaluations) == 2
    assert t.no_improvement_streak([]) == 0
    assert t.no_improvement_streak([{"mean_fleet_delta": 1.0}]) == 0


def test_artifact_hashes_covers_present_outputs(tmp_path: Path):
    (tmp_path / "episodes.csv").write_text("episode\n1\n", encoding="utf-8")
    (tmp_path / "eval_history.json").write_text("[]", encoding="utf-8")
    hashes = t.artifact_hashes(tmp_path)
    assert set(hashes) == {"episodes.csv", "eval_history.json"}
    assert all(len(value) == 64 for value in hashes.values())


def test_fresh_run_refuses_non_empty_run_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(t, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(t, "SNAPSHOTS", tmp_path / "snapshots")
    run_dir = tmp_path / "runs" / "occupied"
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("old run", encoding="utf-8")

    with pytest.raises(FileExistsError):
        t.main(["--run-id", "occupied", "--split", "common", "--skip-eval",
                "--exposures", "1", "--max-episodes", "1"])


def test_train_days_must_belong_to_selected_split(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(t, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(t, "SNAPSHOTS", tmp_path / "snapshots")

    with pytest.raises(ValueError, match="not in common train split"):
        t.main(["--run-id", "bad_subset", "--split", "common",
                "--train-days", "1999-01-01", "--skip-eval",
                "--exposures", "1", "--max-episodes", "1"])


def _evals(scores):
    return [{"mean_fleet_delta": score} for score in scores]


def test_no_early_stop_rounds_3_to_6_without_improvement():
    evaluations = _evals([1.0, 0.5, 0.4, 0.3, 0.2, 0.1])
    for round_idx in range(3, 7):
        assert not t.should_early_stop(round_idx, evaluations[:round_idx], 3, 10)


def test_no_early_stop_rounds_7_to_9_without_improvement():
    evaluations = _evals([1.0] + [0.5] * 8)
    for round_idx in range(7, 10):
        assert not t.should_early_stop(round_idx, evaluations[:round_idx], 3, 10)


def test_early_stop_allowed_at_round_ten_when_patience_met():
    evaluations = _evals([1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3])
    assert t.should_early_stop(10, evaluations, 3, 10)


def test_round_ten_new_best_does_not_stop():
    evaluations = _evals([1.0] + [0.0] * 8 + [2.0])
    assert t.no_improvement_streak(evaluations) == 0
    assert not t.should_early_stop(10, evaluations, 3, 10)


def test_three_non_improving_rounds_after_ten_stop_normally():
    evaluations = _evals([0.0] * 9 + [1.0, 0.0, 0.0, 0.0])
    assert t.no_improvement_streak(evaluations[:10]) == 0
    assert not t.should_early_stop(10, evaluations[:10], 3, 10)
    assert not t.should_early_stop(12, evaluations[:12], 3, 10)
    assert t.should_early_stop(13, evaluations, 3, 10)


def test_no_early_stop_at_or_before_round_nine():
    evaluations = _evals([1.0] + [-1.0] * 9)
    for round_idx in range(1, 10):
        assert not t.should_early_stop(
            round_idx, evaluations[:round_idx], 3, 10)


def test_budget_minimum_and_maximum():
    assert t.episode_budget(53, 10) == 530
    assert t.episode_budget(53, 15) == 795
    assert t.episode_budget(53, 15, max_episodes=800) == 795
    assert t.episode_budget(53, 15, max_episodes=100) == 100


def test_without_early_stop_runs_to_max_round_fifteen():
    evaluations = []
    for round_idx in range(1, 16):
        evaluations.append({"mean_fleet_delta": float(round_idx)})
        assert not t.should_early_stop(round_idx, evaluations, 3, 10)
    assert len(evaluations) == t.episode_budget(53, 15) // 53 == 15


def test_resume_restores_completed_rounds_and_patience():
    assert t.completed_rounds(530, 53) == 10
    assert t.completed_rounds(560, 53) == 11
    evaluations = _evals([0.0] * 7 + [1.0, 0.5, 0.4])
    assert t.completed_rounds(530, 53) == 10
    assert t.no_improvement_streak(evaluations) == 2
    assert not t.should_early_stop(10, evaluations, 3, 10)
    evaluations = evaluations + _evals([0.2])
    assert t.no_improvement_streak(evaluations) == 3
    assert t.should_early_stop(11, evaluations, 3, 10)
