# test_pilot_diagnostics.py
"""The diagnostic artifacts: coverage, exposure, identity, mapping, gradients."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "processed" / "v2_cy_jn_2025" / "pilot"
pytestmark = pytest.mark.skipif(
    not (PILOT / "reward_horizon_days.csv").exists(),
    reason="the pilot diagnostics have not run")


def _read(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_eval_coverage_is_the_wide_grid():
    rows = _read(PILOT / "eval.csv")
    keys = {(r["dataset"], r["seed"], r["load_date"]) for r in rows}
    assert len(rows) == len(keys) == 64
    assert sum(1 for r in rows if r["split"] == "test") == 44
    assert sum(1 for r in rows if r["split"] == "holiday_block") == 20
    for unit in ("ESS9", "ESS14", "ESS17", "ESS27"):
        assert f"settled_{unit}" in rows[0]


def test_exposure_is_twenty_distinct_days_per_run():
    summary = json.loads((PILOT / "summary.json").read_text(encoding="utf-8"))
    for entry in summary:
        visited = [d for d, n in entry["exposure"].items() if n > 0]
        assert len(visited) == 20
        assert max(entry["exposure"].values()) == 1


def test_the_reward_identity_holds_on_every_sampled_day():
    rows = _read(PILOT / "reward_horizon_days.csv")
    assert len(rows) == 48
    assert {r["split"] for r in rows} == {"train", "validation"}
    for r in rows:
        total = (float(r["sum_local_diff"]) + float(r["mechanism_term"])
                 + float(r["state_path_term"]))
        assert abs(float(r["day_diff_vs_b1"]) - total) < 1e-6


def test_quotes_stay_inside_the_action_box():
    rows = _read(PILOT / "action_samples.csv")
    charge = [float(r["charge_quote"]) for r in rows]
    discharge = [float(r["discharge_quote"]) for r in rows]
    assert min(charge) >= -1e-9 and max(charge) <= 150.0 + 1e-9
    assert min(discharge) >= -1e-9 and max(discharge) <= 300.0 + 1e-9


def test_the_truthful_point_is_the_lower_corner_of_the_box():
    # a = mid + tanh * half; tanh = -1 gives low, the truthful storage point.
    for low, high in ((0.0, 150.0), (0.0, 300.0)):
        mid, half = (low + high) / 2.0, (high - low) / 2.0
        assert mid + (-1.0) * half == pytest.approx(low)
        assert mid + 1.0 * half == pytest.approx(high)


def test_gradient_samples_are_positive_and_bounded():
    rows = _read(PILOT / "actor_gradients.csv")
    assert rows
    assert all(float(r["penalty_grad_norm"]) > 0 for r in rows)
    assert all(0.0 <= float(r["saturated_share"]) <= 1.0 for r in rows)
