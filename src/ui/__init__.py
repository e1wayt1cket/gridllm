# ui/__init__.py
"""Multi-page shell for the research dashboard.

Pages register themselves into PAGE_REGISTRY, mirroring the strategy registry
in src/strategies/. Registration order is navigation order.

Each page owns a distinct id prefix: Dash ids share one global namespace across
pages, so two pages must never emit the same id. The simulation page keeps its
original unprefixed ids (kpi-cards, topology-graph, rl-*) because they predate
the shell and are already unique; new pages prefix theirs.
"""

from dataclasses import dataclass
from typing import Any, Callable

from dash import Dash


@dataclass(frozen=True)
class Page:
    """One navigable page.

    key                  URL path segment, e.g. "sim" -> /sim
    label_cn             navbar label
    id_prefix            prefix every component id on this page uses
    layout_fn            () -> Dash component tree
    register_callbacks   (app) -> None, attaching this page's callbacks
    """

    key: str
    label_cn: str
    id_prefix: str
    layout_fn: Callable[[], Any]
    register_callbacks: Callable[[Dash], None]


# Insertion order is the navigation order; dict preserves it.
PAGE_REGISTRY: dict = {}


def register_page(page: Page) -> None:
    """Register a page. Re-registering the same key replaces it."""
    PAGE_REGISTRY[page.key] = page


def nav_items() -> list:
    """Navbar entries in registration order."""
    return [{"label": p.label_cn, "href": f"/{p.key}"}
            for p in PAGE_REGISTRY.values()]


def default_key() -> str:
    """The page the root URL lands on."""
    return next(iter(PAGE_REGISTRY))
