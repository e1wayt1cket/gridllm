# test_formal_ensemble.py
"""One declaration of the formal scenario set, and one of the comparison set.

A list repeated in four modules is four lists that can disagree, and nothing
notices: each keeps working, each is internally consistent, and only a reader
comparing them sees the drift. These tests hold every consumer to the single
declaration in `config/scenarios.yaml` and check that membership of the formal
set follows the readiness rule rather than a preference.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import config_loader as cfg
import scenarios
import wholesale_price as wp

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------
# One source
# ---------------------------------------------------------------------------

def test_every_consumer_reads_the_declared_comparison_set():
    """The three modules that used to keep their own copy of the legacy four
    now read the declaration, so they cannot drift apart."""
    import make_rot_comparison_figures as figures
    import run
    from ui import sim_page

    declared = scenarios.legacy_comparison_scenarios()
    assert declared, "the declaration is empty"
    assert run.EXPORT_SCENARIOS == declared
    assert figures.SCENARIOS == declared
    assert [s["value"] for s in sim_page.RL_SCENARIOS] == declared


def test_no_source_file_declares_a_scenario_list_of_its_own():
    """A literal list holding two or more scenario names is a scenario *set*,
    and a set belongs in one place.

    Two or more, because a single name is usually a default or an example -- an
    LLM prompt, a dropdown's initial value -- and forbidding those would be
    forbidding something else. The drifted duplicates this replaced all held
    several names.
    """
    names = {f'"{n}"' for n in cfg.load_scenarios()["scenarios"]}
    offenders = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path) or path.name == "scenarios.py":
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#") or "[" not in line:
                continue
            if sum(1 for n in names if n in line) >= 2:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, (
        f"{offenders} declare a scenario set inline; the sets live in "
        "config/scenarios.yaml")


# ---------------------------------------------------------------------------
# Membership follows the rule, not a preference
# ---------------------------------------------------------------------------

def test_the_formal_set_contains_only_measured_market_sources():
    for name in scenarios.formal_scenarios():
        source = str(cfg.get_scenario_cfg(name).get("wholesale_source",
                                                    "legacy"))
        ready, why = wp.formal_data_readiness(source)
        assert ready, f"{name}: {why}"


def _declared_exclusions() -> dict:
    """The scenarios left out of the formal set, each with its reason."""
    return (cfg.load_scenarios().get("formal_ensemble") or {}).get(
        "excluded") or {}


def test_a_qualifying_scenario_is_either_formal_or_excluded_with_a_reason():
    """The converse, so the set cannot be *narrower* than the rule either.

    A scenario that reads a measured market and is named in neither block is one
    a formal run cannot reach with nothing on record saying why -- which is the
    failure this refuses. An exception is available and it costs a sentence:
    the scenario goes under `formal_ensemble.excluded` with a reason. That
    keeps the original meaning exactly, since the narrowing it was written
    against was the *unstated* kind.
    """
    declared = set(scenarios.formal_scenarios())
    excluded = _declared_exclusions()
    missing = []
    for name in cfg.load_scenarios()["scenarios"]:
        source = str(cfg.get_scenario_cfg(name).get("wholesale_source",
                                                    "legacy"))
        ready, _ = wp.formal_data_readiness(source)
        if ready and name not in declared and name not in excluded:
            missing.append(name)
    assert not missing, (
        f"{missing} read a measured market but are in neither the formal set "
        "nor its declared exclusions")


def _exclusion_problems(exclusions, known, declared):
    """The two ways an exclusion can be wrong. Returned, so both the live
    declaration and a synthetic one can be held to the same rule."""
    problems = []
    for name, reason in (exclusions or {}).items():
        if name not in known:
            problems.append(f"{name} is excluded but does not exist")
        if not str(reason).strip():
            problems.append(f"{name} is excluded with no reason")
        if name in declared:
            problems.append(f"{name} is both formal and excluded")
    return problems


def test_an_exclusion_is_a_statement_with_a_reason_and_a_subject(monkeypatch):
    """An exclusion is a declaration, so it can be wrong in two ways: it can
    give no reason, or it can name a scenario that does not exist.

    The block is empty today -- the one scenario that ever sat in it was
    admitted on 2026-10-16 -- and an empty block would make a loop over it
    assert nothing. So the rule is checked twice: once on the live declaration,
    and once on a synthetic one that actually exercises it.
    """
    known = set(cfg.load_scenarios()["scenarios"])
    declared = set(scenarios.formal_scenarios())
    assert _exclusion_problems(_declared_exclusions(), known, declared) == []

    non_formal = scenarios.legacy_comparison_scenarios()[0]
    assert non_formal in known and non_formal not in declared
    assert _exclusion_problems({non_formal: "a stated reason"}, known,
                               declared) == []
    assert _exclusion_problems({non_formal: "   "}, known, declared) == [
        f"{non_formal} is excluded with no reason"]
    assert _exclusion_problems({"not_a_scenario": "why"}, known,
                               declared) == [
        "not_a_scenario is excluded but does not exist"]
    assert _exclusion_problems({non_formal: "why"},
                               known, known) == [
        f"{non_formal} is both formal and excluded"]


def test_an_empty_formal_set_is_reported_rather_than_defaulted(monkeypatch):
    """If nothing qualifies, the answer is that nothing qualifies -- not a
    fallback list that happens to be at hand."""
    monkeypatch.setattr(scenarios, "_declared_sets",
                        lambda: {"scenarios": []})
    with pytest.raises(ValueError, match="no scenarios to draw on"):
        scenarios.formal_scenarios()


def test_a_scenario_that_does_not_qualify_cannot_be_declared_formal(monkeypatch):
    """The declaration is a claim and this is where it is tested: naming a
    generated-price scenario in the formal set raises instead of returning."""
    monkeypatch.setattr(scenarios, "_declared_sets",
                        lambda: {"scenarios": ["typical_day"]})
    with pytest.raises(ValueError, match="not formal sources"):
        scenarios.formal_scenarios()


# ---------------------------------------------------------------------------
# Subsets are declared and checked
# ---------------------------------------------------------------------------

def test_a_module_may_declare_a_subset_of_the_formal_set():
    formal = scenarios.formal_scenarios()
    assert scenarios.assert_formal_subset(formal) == formal
    assert scenarios.assert_formal_subset(formal[:1]) == formal[:1]


def test_a_subset_may_not_add_a_scenario_of_its_own():
    with pytest.raises(ValueError, match="may not add scenarios of its own"):
        scenarios.assert_formal_subset(["baseline"])


def test_a_non_formal_name_needs_an_explicit_opt_in():
    """The gate is off by default, and the way past it is stated.

    An exploratory run -- a smoke test that validates the environment and the
    checkpoint path rather than producing a result -- may name a scenario
    outside the formal set, but only by saying so, and the refusal names the
    way. Opting in does not widen the set: `formal_scenarios()` still holds
    one name, so nothing downstream can read a smoke run as a formal one.
    """
    non_formal = scenarios.legacy_comparison_scenarios()[0]
    with pytest.raises(ValueError, match="allow_non_formal=True"):
        scenarios.assert_formal_subset([non_formal])
    assert scenarios.assert_formal_subset(
        [non_formal], allow_non_formal=True) == [non_formal]
    assert non_formal not in scenarios.formal_scenarios()


def test_the_training_cli_opt_in_reaches_the_gate_and_announces_itself(capsys):
    """`--allow-non-formal` is the only way in, and the run says so out loud.

    A curve produced this way has to carry the reason it is not a result, and
    the flag is left in `vars(args)`, which the run artifacts record.
    """
    import train_rl

    non_formal = scenarios.legacy_comparison_scenarios()[0]
    with pytest.raises(ValueError):
        train_rl._resolve_train_scenarios(non_formal)
    assert train_rl._resolve_train_scenarios(non_formal, True) == [non_formal]
    assert "EXPLORATORY RUN" in capsys.readouterr().out


def test_the_two_declared_sets_do_not_overlap():
    """A scenario cannot be both formal and a legacy comparison: the two say
    different things about the same name's price source."""
    formal = set(scenarios.formal_scenarios())
    legacy = set(scenarios.legacy_comparison_scenarios())
    assert not (formal & legacy)


# ---------------------------------------------------------------------------
# The buses that used to be handled implicitly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bus", [22, 29, 32])
def test_the_three_previously_special_buses_have_an_explicit_policy(bus):
    """These three were excluded by a constant in the mapping module, which is
    how their load came to be dropped without a scenario saying so. Every one
    now carries a policy the config states."""
    from build_industry_node_loads import bus_policies
    policies = bus_policies("typical_day_industry", [])
    assert bus in policies
    assert policies[bus] in ("preserve", "replace", "overlay")
