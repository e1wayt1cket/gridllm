# test_agent_seed.py
"""A participant's draw must repeat in a fresh process.

These tests exist because the previous seeding expression was
`hash(agent.name) % 2**31`, and CPython salts `hash()` on a string per process.
The draw therefore differed between two runs of the same configuration. The
cross-process test below is the only way to observe that: inside one process
the old expression looked perfectly stable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_seed import seed_for_agent

_SRC = Path(__file__).resolve().parents[1] / "src"

_NAMES = ["Bus_1_storage", "Bus_12_storage", "Bus_17_storage",
          "Bus_27_storage", "prosumer_3", "GRID"]


def _run_in_fresh_process(code: str, hash_seed: int) -> str:
    """Run `code` with `src` importable and CPython's hash salt pinned."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = str(hash_seed)
    done = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, check=True)
    return done.stdout.strip()


def test_the_seed_repeats_in_a_fresh_process_with_a_different_hash_salt():
    code = ("from agent_seed import seed_for_agent\n"
            f"print(','.join(str(seed_for_agent(n)) for n in {_NAMES!r}))\n")
    first = _run_in_fresh_process(code, hash_seed=0)
    second = _run_in_fresh_process(code, hash_seed=1)
    assert first == second, "the seed moved when the process hash salt changed"


def test_the_old_expression_really_did_move_between_processes():
    """The contrast that justifies the fix, measured rather than assumed.

    If this ever stops holding the environment cannot demonstrate the defect,
    which is a fact about the environment and not a pass.
    """
    code = (f"print(','.join(str(hash(n) % 2**31) for n in {_NAMES!r}))\n")
    first = _run_in_fresh_process(code, hash_seed=0)
    second = _run_in_fresh_process(code, hash_seed=1)
    assert first != second, (
        "CPython's string hash did not move between these two salts, so this "
        "environment cannot show why the seed had to change")


def test_the_seed_follows_the_name_and_not_the_visit_order():
    forward = [seed_for_agent(n) for n in _NAMES]
    backward = [seed_for_agent(n) for n in reversed(_NAMES)]
    assert forward == list(reversed(backward))
    assert len(set(forward)) == len(_NAMES), "two participants share a stream"


def test_an_experiment_seed_moves_the_draw_without_renaming_anyone():
    plain = [seed_for_agent(n) for n in _NAMES]
    seeded = [seed_for_agent(n, seed=7) for n in _NAMES]
    assert plain != seeded
    assert seed_for_agent(_NAMES[0], seed=7) == seeded[0], "not a pure function"


@pytest.mark.parametrize("module", ["dispatch_core.py",
                                    "strategies/random_bidding.py"])
def test_no_drawing_site_seeds_from_pythons_hash(module):
    """The call sites, held to the helper rather than to a stable-looking hash."""
    source = (_SRC / module).read_text(encoding="utf-8")
    assert "hash(" not in source, (
        f"{module} seeds from `hash()`, which is salted per process")
    assert "seed_for_agent(" in source
