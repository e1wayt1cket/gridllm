"""Resume semantics for the D+A training entry.

The training loop takes its training pair from `pair_index`, its restart point
from `resume_position`, its history from `filter_evaluations` and its best from
`restore_best`. Driving a live trainer would test a stand-in for those
functions rather than the real ones, so this file pins the functions the loop
calls and asserts the pair identity directly.
"""

from __future__ import annotations

from pathlib import Path

import train_da_gate as t

ORDER = 53


def _pairs_seen(start: int, stop: int) -> list[int]:
    return [t.pair_index(episode, ORDER) for episode in range(start, stop)]


def test_partial_round_resume_starts_at_pair_28():
    position = t.resume_position(28, ORDER)
    assert position["resume_episode"] == 28
    assert _pairs_seen(28, 29) == [28]
    # episodes 28..52 finish round 1 with the remaining 25 pairs, in order.
    assert _pairs_seen(28, 53) == list(range(28, 53))


def test_partial_round_resume_completes_one_exposure_of_every_pair():
    # 28 done before the checkpoint plus 25 after it covers all 53 pairs once.
    assert sorted(_pairs_seen(0, 28) + _pairs_seen(28, 53)) == list(range(53))


def test_full_round_resume_starts_at_the_next_round():
    position = t.resume_position(53, ORDER)
    assert position["resume_episode"] == 53
    assert position["partial_round"] is False
    assert _pairs_seen(53, 54) == [0]


def test_round_start_matches_the_loop_arithmetic():
    for episodes in (0, 1, 28, 52, 53, 54, 105, 106):
        position = t.resume_position(episodes, ORDER)
        round_idx = position["rounds_completed"] + 1
        round_start = (round_idx - 1) * ORDER
        assert episodes - round_start == t.pair_index(episodes, ORDER)


def test_resume_rule_mismatch_names_the_key():
    saved = {"early_stop_patience": 3, "min_rounds": 10, "max_rounds": 15}
    bad = [key for key, ok
           in t.resume_rule_checks(saved, 3, 9, 15).items() if not ok]
    assert bad == ["min_rounds"]


def test_missing_rule_key_is_a_mismatch_not_a_default():
    saved = {"early_stop_patience": 3, "min_rounds": 10}
    bad = [key for key, ok
           in t.resume_rule_checks(saved, 3, 10, 15).items() if not ok]
    assert bad == ["max_rounds"]


def test_stale_evaluations_after_the_checkpoint_are_dropped():
    rows = [{"round": 1, "mean_fleet_delta": 1.0},
            {"round": 2, "mean_fleet_delta": 2.0},
            {"round": 3, "mean_fleet_delta": 3.0}]
    assert t.filter_evaluations(rows, 1) == rows[:1]
    assert t.filter_evaluations(rows, 3) == rows


def test_restored_best_survives_a_resume(tmp_path: Path):
    saved_best = {"value": 7.5, "round": 2, "checkpoint": "ckpt/best"}
    evaluations = [{"round": 1, "mean_fleet_delta": 9.0},
                   {"round": 2, "mean_fleet_delta": 7.5}]
    restored = t.restore_best(saved_best, evaluations, tmp_path)
    assert restored["round"] == 2
    assert restored["value"] == 7.5
    assert restored["source"] == "manifest"
