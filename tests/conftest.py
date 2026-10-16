# conftest.py
"""Shared fixtures for the test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session", autouse=True)
def _non_formal_market_session():
    """Open the gate that keeps generated price series out of formal runs.

    The test suite exercises the whole control path -- clearing, settlement,
    the ladder, the audit -- against a market whose shape is known, and that
    market is generated. Opening the gate here is the suite stating that, once,
    rather than every module remembering to.

    The gate's shipped default is closed and is asserted separately
    (`test_forty_day_removal.py`), so a formal run reaching a synthetic market
    by accident still fails; what this fixture does is keep the *tests* from
    having to re-declare the obvious. Scope is the session because the
    module-scoped fixtures below build days, and a function-scoped flag would
    be set after they had already run.
    """
    # Imported here rather than at module scope: this file is imported during
    # collection, and pulling the price layer in that early loads a native
    # library before the ones pandas needs, which on Windows fails the process
    # with a DLL entry-point error rather than a test failure.
    import wholesale_price as wp

    previous = wp.ALLOW_NON_FORMAL_SOURCES
    wp.ALLOW_NON_FORMAL_SOURCES = True
    yield
    wp.ALLOW_NON_FORMAL_SOURCES = previous
