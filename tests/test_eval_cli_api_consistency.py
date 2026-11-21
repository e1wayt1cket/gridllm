# test_eval_cli_api_consistency.py
"""The evaluation CLI and the evaluation API report the same settlement.

Both arms of every comparison are settled by `counterfactual_eval.settle_day`,
so the *function* was already shared. What was not shared was everything the
call needs -- the configuration, the agent set, the seed, the horizon -- and a
disagreement there produces two plausible numbers for the same day rather than
an error. These tests run both entry points and compare what they report.
"""

from __future__ import annotations

import csv
import sys

import pytest
import torch

import eval_agents as ev
import rl_td3
from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V6

#: The CLI/API settlement contract, not the industry load. 
#: `typical_day_industry` now carries the metered load, on which the
#: settlement is degraded (every price cell falls back) and neither entry
#: point reports a payoff -- which would make this a test of the shedding.
SCENARIO = "typical_day"
RL_NAMES = ["ESS9", "ESS14", "ESS17", "ESS27"]
SEED = 42


def _bounds():
    return torch.tensor([[-1.0, -1.0], [1.0, 1.0]])


@pytest.fixture(scope="module")
def policy_dir(tmp_path_factory):
    """Fresh V4 actors, written and read back through the real serialisation.

    Untrained on purpose: this file is about where the numbers come from, not
    how good a policy is, and an untrained actor is reproducible without a
    checkpoint on disk.
    """
    out = tmp_path_factory.mktemp("policies")
    for name in RL_NAMES:
        actor = rl_td3.Actor(OBS_V6.total_dim, ACTION_STORAGE_QUOTE_V2.dim,
                             _bounds())
        rl_td3.save_policy(actor, str(out / f"{name}.pt"), obs_spec=OBS_V6,
                           action_spec=ACTION_STORAGE_QUOTE_V2)
    return out


@pytest.fixture(scope="module")
def cli_and_api(policy_dir, tmp_path_factory):
    """One CLI run and one API run, on the same policies, scenario and seed."""
    csv_path = tmp_path_factory.mktemp("cli") / "eval.csv"
    argv = sys.argv
    sys.argv = ["eval_agents.py", "--policies", str(policy_dir),
                "--scenarios", SCENARIO, "--seed", str(SEED),
                "--eval-seed", str(SEED), "--output", str(csv_path)]
    try:
        ev.main()
    finally:
        sys.argv = argv
    with open(csv_path, newline="") as fh:
        cli_rows = list(csv.DictReader(fh))

    api = ev.evaluate_policy_dir(str(policy_dir), [SCENARIO], seed=SEED)
    return cli_rows, api


def _cli_all(rows):
    match = [r for r in rows if r["scenario"] == SCENARIO
             and r["agent"] == "ALL"]
    assert match, f"the CLI wrote no ALL row for {SCENARIO}"
    return match[0]


def test_the_cli_writes_its_csv_at_all(cli_and_api):
    """The write used to raise on its own output.

    The per-episode aggregation names a column per statistic it computes, and
    those columns were not in the header the writer was given, so
    `csv.DictWriter` refused the rows it had just built.
    """
    cli_rows, _ = cli_and_api
    assert cli_rows
    assert "baseline_profit" in cli_rows[0]


def test_the_cli_and_the_api_report_the_same_settled_payoff(cli_and_api):
    cli_rows, api = cli_and_api
    row = _cli_all(cli_rows)
    api_row = api["scenarios"][0]

    # Averaged as a numeric column, so a clean run writes 1.0 rather than True.
    assert float(row["settled_clean"]) == 1.0
    assert api_row["settled_clean"] is True

    for key in ("baseline_profit", "rl_profit", "profit_delta"):
        assert float(row[key]) == pytest.approx(api_row[key], abs=1e-6), (
            f"{key}: the CLI reported {row[key]} and the API reported "
            f"{api_row[key]} for the same policies on the same seeded day")


def test_the_cli_and_the_api_agree_on_what_was_cleared(cli_and_api):
    """The same day, not merely the same arithmetic on two different days."""
    cli_rows, api = cli_and_api
    row = _cli_all(cli_rows)
    api_row = api["scenarios"][0]
    for key in ("clearing_objective_baseline", "clearing_objective_rl",
                "clearing_objective_delta", "valuation_artifact"):
        assert float(row[key]) == pytest.approx(api_row[key], abs=1e-6), key


def test_the_api_settles_through_the_cli_s_settlement_function():
    """The one call both entry points make, asserted rather than assumed."""
    import inspect
    import eval_agents

    api_src = inspect.getsource(eval_agents.evaluate_policy_dir)
    cli_src = inspect.getsource(eval_agents.settle_capture)
    assert "settle_day(" in api_src
    assert "settle_day(" in cli_src
    # Neither may reach for a second settlement path.
    assert "clear_market(" not in cli_src
    assert "clear_market(" not in api_src
