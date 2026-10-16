"""One name for the clearing objective, and old artifacts still readable.

`welfare` was that quantity's name and was never right for it: it is the
clearing model's own objective value, priced at each unit's *declared* quotes,
while the unit is settled at the nodal price. Only the settlement figure is
money a participant keeps. The rename has to hold at both ends -- new output
carries one name, and files written under the old one still read.
"""

import os
import pathlib
import warnings

import numpy as np
import pytest

from legacy_schema import LEGACY_ALIASES, canonical_name, clearing_objective, \
    upgrade_columns, upgrade_row

_REPO = pathlib.Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"


# ---------------------------------------------------------------------------
# The alias map itself
# ---------------------------------------------------------------------------

def test_welfare_is_an_alias_and_clearing_objective_is_canonical():
    assert canonical_name("welfare") == "clearing_objective"
    assert canonical_name("clearing_objective") == "clearing_objective"
    # Every canonical target must itself be canonical, or a chain of renames
    # would leave a reader holding an intermediate name.
    for canonical in LEGACY_ALIASES.values():
        assert canonical_name(canonical) == canonical, canonical


def test_a_legacy_only_row_reads_under_the_canonical_name():
    """A JSON artifact written before the rename."""
    row = upgrade_row({"welfare": 7695.0, "re_rate": 56.0})
    assert row["clearing_objective"] == 7695.0
    assert "welfare" not in row
    assert clearing_objective({"welfare": 7695.0}) == 7695.0


def test_a_row_carrying_both_names_agreeing_reads_normally():
    row = upgrade_row({"welfare": 100.0, "clearing_objective": 100.0})
    assert row["clearing_objective"] == 100.0
    assert clearing_objective(row) == 100.0


def test_a_row_carrying_both_names_disagreeing_warns():
    """The file contradicts itself, and that is reported rather than resolved.

    Picking a winner silently would turn a self-contradicting file into one
    that looks consistent -- exactly the failure the translation exists to
    avoid.
    """
    with pytest.warns(UserWarning, match="disagrees with itself"):
        row = upgrade_row({"welfare": 1.0, "clearing_objective": 2.0})
    # The canonical spelling keeps the canonical value.
    assert row["clearing_objective"] == 2.0


def test_column_lists_upgrade():
    assert upgrade_columns(["episode", "welfare", "re_rate"]) == \
        ["episode", "clearing_objective", "re_rate"]
    assert upgrade_columns(["clearing_objective"]) == ["clearing_objective"]


# ---------------------------------------------------------------------------
# New output carries exactly one name
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def day():
    from models import MarketConfig, MarketDesignConfig
    from market import clear_market, adaptive_bidding
    from scenarios import get_scenario

    T = 96
    md = MarketDesignConfig(use_constraint_multi_obj=False)
    config = MarketConfig(opf_mode="socp", verbose=False, market_design=md)
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, wholesale = get_scenario("baseline", T=T, config=config)
    actions = adaptive_bidding(agents, config, strategy="rl")
    return clear_market(agents, T, "DA", actions, config,
                        wholesale=wholesale, horizon_type="full_day"), agents, config


def test_a_freshly_cleared_day_carries_only_the_canonical_name(day):
    result, _, _ = day
    assert "clearing_objective" in result
    assert "welfare" not in result
    assert "objective" not in result
    assert np.isfinite(result["clearing_objective"])


def test_the_forest_engine_agrees_on_the_schema(day):
    """Auxiliary engines emit the same key, not a second spelling of it."""
    from models import MarketConfig, MarketDesignConfig
    from market import clear_market, adaptive_bidding
    from scenarios import get_scenario

    T = 96
    md = MarketDesignConfig(use_constraint_multi_obj=False)
    config = MarketConfig(opf_mode="lindistflow", verbose=False,
                          market_design=md)
    agents, wholesale = get_scenario("baseline", T=T, config=config)
    actions = adaptive_bidding(agents, config, strategy="rl")
    result = clear_market(agents, T, "DA", actions, config,
                          wholesale=wholesale, horizon_type="full_day")
    assert "clearing_objective" in result
    assert "welfare" not in result
    assert "objective" not in result


# ---------------------------------------------------------------------------
# The read layer accepts both generations
# ---------------------------------------------------------------------------

def _write_run(root, run_id, columns: dict, kpi: dict = None):
    """A minimal artifact directory as `data_aggregator` expects to find it.

    `root` is the `outputs/rl` directory itself, and discovery only picks up
    directories whose name starts with `run-`.
    """
    import json

    import pandas as pd

    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "cli_args": {}, "kpi": kpi or {}}),
        encoding="utf-8")
    if kpi is not None:
        (run_dir / "kpi.json").write_text(json.dumps(kpi), encoding="utf-8")
    return run_dir


def test_aggregator_reads_a_legacy_metrics_csv(tmp_path):
    """A metrics.csv with the old column name loads and addresses canonically."""
    import data_aggregator as agg

    root = tmp_path / "rl"
    _write_run(root, "run-legacy", {
        "episode": [0, 1],
        "mean_reward": [1.0, 2.0],
        "welfare": [100.0, 200.0],   # legacy spelling
        "re_rate": [50.0, 51.0],
    })

    frame = agg.load_metrics("run-legacy", root=str(root))
    assert frame["clearing_objective"].tolist() == [100.0, 200.0]
    assert "welfare" not in frame.columns


def test_aggregator_reads_a_new_metrics_csv(tmp_path):
    import data_aggregator as agg

    root = tmp_path / "rl"
    _write_run(root, "run-new", {
        "episode": [0, 1],
        "mean_reward": [1.0, 2.0],
        "clearing_objective": [100.0, 200.0],
        "re_rate": [50.0, 51.0],
    })

    frame = agg.load_metrics("run-new", root=str(root))
    assert frame["clearing_objective"].tolist() == [100.0, 200.0]
    assert "welfare" not in frame.columns


def test_aggregator_reads_a_legacy_kpi_manifest(tmp_path):
    """A run manifest written before the rename."""
    import data_aggregator as agg

    root = tmp_path / "rl"
    _write_run(root, "run-kpi", {"episode": [0]}, kpi={
        "best_episode": 7, "best_mean_reward": 12.0,
        "best_welfare": 555.0,          # legacy spelling
        "final_mean_reward": 3.0, "n_episodes": 10,
    })

    runs = agg.list_runs(root=str(root))
    row = runs[runs["run_id"] == "run-kpi"].iloc[0]
    assert float(row["best_clearing_objective"]) == 555.0


def test_export_round_trip_preserves_the_value(tmp_path):
    """Writing a row out and reading it back must not move the figure."""
    import json

    import run_artifacts as ra

    row = {"episode": 0, "mean_reward": 1.0, "clearing_objective": 4242.0,
           "re_rate": 50.0}
    path = tmp_path / "row.json"
    path.write_text(json.dumps(ra._json_safe(row)), encoding="utf-8")
    back = json.loads(path.read_text(encoding="utf-8"))
    assert back["clearing_objective"] == 4242.0
    assert "welfare" not in back
    assert clearing_objective(back) == 4242.0


def test_a_saved_run_round_trips_through_the_artifact_container(tmp_path):
    """The whole export/import path, not just the JSON encoder."""
    import run_artifacts as ra

    art = ra.RunArtifacts.create(
        config={"opf_mode": "socp"}, cli_args={"episodes": 1},
        obs_spec={"name": "v3_12d", "version": 3}, action_spec={
            "name": "storage_quote_v2", "version": 2},
        base_dir=str(tmp_path))
    art.record_episode(0, mean_reward=-480.0, clearing_objective=7695.0,
                       re_rate=56.0)
    # An eval row is recorded as well: `RunArtifacts.load` cannot read back an
    # eval.csv that has no columns, so a run with no evaluations would fail to
    # reload for a reason that has nothing to do with this schema.
    art.record_eval(0, {"mean_reward": -480.0, "clearing_objective": 7695.0,
                        "re_rate": 56.0})
    art.save()

    back = ra.RunArtifacts.load(art.root_dir)
    row = back.metrics[0]
    assert row["clearing_objective"] == 7695.0
    assert "welfare" not in row
    assert clearing_objective(row) == 7695.0
    assert back.evals[0]["clearing_objective"] == 7695.0


# ---------------------------------------------------------------------------
# The dashboard reads the new name
# ---------------------------------------------------------------------------

def test_no_ui_module_still_reads_the_retired_key():
    """A dashboard reading a key that no longer exists shows a blank card.

    Checked across the whole package rather than one page, because each page
    reads its own fields and a partial rename leaves the others silently empty.
    """
    offenders = []
    for path in sorted((_SRC / "ui").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in ('"welfare"', "'welfare'", '"objective"', "welfare_"):
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, f"retired key(s) still read by the UI: {offenders}"


def test_kpi_cards_show_the_clearing_objective():
    """The card builder reads the canonical key and renders its value."""
    from ui.sim_page import create_kpi_cards
    from types import SimpleNamespace

    result = {
        "clearing_objective": 1234.0,
        "re_consumption_rate": 97.5,
        "carbon_emissions": 82.4,
        "carbon_intensity": 0.5,
        "total_curtailment": 0.9,
        "schedules": {"A": {"served": np.zeros(4)}},
    }
    agent = SimpleNamespace(name="A", load_forecast=np.ones(4))
    cards = create_kpi_cards(result, result, {"A": 1.0}, [agent])
    rendered = str(cards)
    assert "1,234" in rendered
    assert "出清目标" in rendered
    assert "福利" not in rendered
