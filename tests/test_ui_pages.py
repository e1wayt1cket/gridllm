"""Tests for the multi-page shell and its page registry.

Importing the page modules builds the pandapower network, so these run under
the slow marker.
"""

import pytest

pytestmark = pytest.mark.slow


def _app():
    from ui.shell import create_app
    import ui
    return create_app(), ui


def test_every_registered_page_is_well_formed():
    app, ui = _app()
    assert ui.PAGE_REGISTRY, "no pages registered"
    keys, prefixes = [], []
    for page in ui.PAGE_REGISTRY.values():
        assert callable(page.layout_fn)
        assert callable(page.register_callbacks)
        assert page.label_cn
        keys.append(page.key)
        prefixes.append(page.id_prefix)
        # The layout must build, and its root must be renderable.
        assert page.layout_fn() is not None
    assert len(keys) == len(set(keys)), "duplicate page key"
    # Two pages sharing an id prefix would collide in Dash's global id space.
    assert len(prefixes) == len(set(prefixes)), "duplicate id prefix"


def test_nav_order_matches_registration_order():
    app, ui = _app()
    nav = ui.nav_items()
    assert [p["label"] for p in nav] == \
        [p.label_cn for p in ui.PAGE_REGISTRY.values()]
    assert [p["href"] for p in nav] == \
        [f"/{p.key}" for p in ui.PAGE_REGISTRY.values()]


def test_shell_suppresses_callback_exceptions():
    # Page layouts are mounted by the router, so their ids do not exist when
    # the app is built; without this Dash rejects the callbacks at startup.
    app, _ = _app()
    assert app.config.suppress_callback_exceptions is True


def test_sim_page_contributes_exactly_the_original_callbacks():
    # The extraction into ui/sim_page.py must not change the simulation page's
    # callback surface.
    import dash
    import dash_bootstrap_components as dbc
    from dash import dcc, html
    from ui import sim_page
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                    suppress_callback_exceptions=True)
    app.layout = html.Div([dcc.Location(id="url"),
                           html.Div(id="page-content")])
    sim_page.register_callbacks(app)
    assert len(app.callback_map) == 4


def test_pages_do_not_share_component_ids():
    # Component ids share one global namespace across pages, so a collision
    # would make two pages fight over the same callback targets.
    app, ui = _app()

    def collect(component, acc):
        if isinstance(component, (list, tuple)):
            for c in component:
                collect(c, acc)
        else:
            cid = getattr(component, "id", None)
            if isinstance(cid, str):
                acc.append(cid)
            for child in getattr(component, "children", []) or []:
                collect(child, acc)
        return acc

    seen = {}
    for page in ui.PAGE_REGISTRY.values():
        for cid in collect(page.layout_fn(), []):
            if cid in seen:
                assert seen[cid] == page.key, \
                    f"id '{cid}' used by both '{seen[cid]}' and '{page.key}'"
            seen[cid] = page.key
    assert seen, "no component ids found; the walker is wrong"


def test_router_resolves_each_page_and_unknown_paths():
    app, ui = _app()
    route = app.callback_map["page-content.children"]["callback"]
    raw = getattr(route, "__wrapped__", route)
    for page in ui.PAGE_REGISTRY.values():
        body = str(raw(f"/{page.key}"))
        assert page.layout_fn().to_plotly_json() is not None
        # An unknown path yields a short not-found card, not a page.
        assert len(body) > 500, page.key
    # The root path lands on the first registered page.
    assert len(str(raw("/"))) > 500
    fallback = str(raw("/does-not-exist"))
    assert len(fallback) < 500
    assert "does-not-exist" in fallback


# --- training-monitor page: panels that read data -------------------------
def test_ckpt_table_renders_labels_not_missing_markers():
    # The checkpoint role and scenario name are text; a formatter that only
    # understands numbers would show "—" for both, hiding which row is which.
    from ui.train_page import _ckpt_table
    rows = [{"checkpoint": "best", "scenario": "baseline",
             "profit_delta": 4875.31, "genuine_welfare_delta": 1665.19,
             "cs_delta": 193.42, "cp_delta": -193.42,
             "lmp_markup_delta": -0.01, "market_power_power": 153.5}]
    table = _ckpt_table(rows)
    body = table.children[1].children[0]
    cells = [td.children for td in body.children]
    assert cells[0] == "best"
    assert cells[1] == "baseline"
    assert cells[2] == "4,875.31"
    assert cells[5] == "-193.42"


def test_ckpt_table_marks_only_genuinely_absent_values():
    from ui.train_page import _ckpt_table
    rows = [{"checkpoint": "ckpt_50", "scenario": "peak_load",
             "profit_delta": None, "genuine_welfare_delta": float("nan"),
             "cs_delta": -5.0, "cp_delta": 5.0, "lmp_markup_delta": 0.0,
             "market_power_power": 1.0}]
    table = _ckpt_table(rows)
    body = table.children[1].children[0]
    cells = [td.children for td in body.children]
    assert cells[0] == "ckpt_50"          # label survives
    assert cells[1] == "peak_load"
    assert cells[2] == "—"                # None is missing
    assert cells[3] == "—"                # NaN is missing
    assert cells[4] == "-5.00"


def test_format_cell_colours_signed_metrics_only():
    from ui.train_page import _format_cell
    from ui.theme import C_SUCCESS, C_DANGER, C_MUTED, C_TEXT
    assert _format_cell("profit_delta", 10.0)[1] == C_SUCCESS
    assert _format_cell("profit_delta", -10.0)[1] == C_DANGER
    assert _format_cell("profit_delta", 0.0)[1] == C_MUTED
    # A scenario label is text, coloured as text.
    assert _format_cell("scenario", "baseline")[1] == C_TEXT


def test_available_checkpoints_resolves_each_role_to_its_own_path(tmp_path):
    # best/last live in subdirectories of their own; evaluating them from the
    # run directory would re-score the final policies under another label.
    import os
    from ui import train_page as tp
    import data_aggregator as da

    root = str(tmp_path / "policies")
    d = os.path.join(root, "run_x")
    for sub in ("best", "last"):
        os.makedirs(os.path.join(d, sub))
        open(os.path.join(d, sub, "A.pt"), "w").close()
    open(os.path.join(d, "A.pt"), "w").close()
    for ck in (50, 100):
        open(os.path.join(d, f"A_ckpt_{ck}.pt"), "w").close()

    original = da.policies_root
    da.policies_root = lambda: root
    try:
        got = dict(tp._available_checkpoints("run_x"))
    finally:
        da.policies_root = original

    assert got["ckpt_50"] == d
    assert got["ckpt_100"] == d
    assert got["final"] == d
    assert got["best"] == os.path.join(d, "best")
    assert got["last"] == os.path.join(d, "last")


# --- user impact page -----------------------------------------------------
def test_user_page_registered_with_its_own_id_prefix():
    app, ui = _app()
    page = ui.PAGE_REGISTRY["user"]
    assert page.id_prefix == "ui-"
    assert page.label_cn == "用户影响"


def test_user_page_filters_empty_selection_to_nothing():
    # Clearing every option must yield no rows, not silently fall back to all:
    # showing everything while the controls read empty contradicts the UI.
    pytest.importorskip("pandas")
    from ui import user_page
    assert len(user_page._filtered([], ["baseline"])) == 0
    assert len(user_page._filtered(["e8_matd3_15"], [])) == 0


def test_user_page_only_uses_the_corrected_consumer_convention():
    # The pre-2026-09-09 generation reports the opposite sign, so the page's
    # data source must not pool it in.
    pytest.importorskip("pandas")
    import data_aggregator as da

    def pairs():
        return da.pareto_pairs()

    df = pairs()
    if len(df):
        assert set(df["cs_convention"]) == {"p_dis_excluded"}


# --- experiment manager page ----------------------------------------------
def test_exp_page_registered_with_its_own_id_prefix():
    app, ui = _app()
    page = ui.PAGE_REGISTRY["experiments"]
    assert page.id_prefix == "ex-"
    assert page.label_cn == "实验管理"


def test_exp_page_index_table_shows_the_manipulated_variant():
    # A row without its variant is uninterpretable: the file name alone does
    # not say which capacity or lambda produced the numbers.
    pytest.importorskip("pandas")
    from ui import exp_page

    handle = exp_page._fmt("variant", "capacity=1.0")
    assert handle[0] == "capacity=1.0"


def test_exp_page_formatter_marks_only_absent_values():
    pytest.importorskip("pandas")
    from ui import exp_page
    from ui.theme import C_SUCCESS, C_DANGER, C_MUTED, C_TEXT

    assert exp_page._fmt("profit_delta", 1509.0)[0] == "1,509"
    assert exp_page._fmt("profit_delta", 1509.0)[1] == C_SUCCESS
    assert exp_page._fmt("profit_delta", -1509.0)[1] == C_DANGER
    assert exp_page._fmt("lmp_markup_delta", -0.0029)[0] == "-0.0029"
    # A missing metric is a dash; a label is text.
    assert exp_page._fmt("cs_delta", None)[0] == "—"
    assert exp_page._fmt("cs_delta", float("nan"))[0] == "—"
    assert exp_page._fmt("policy_label", "e8_matd3_10")[1] == C_TEXT
    assert exp_page._fmt("variant", "")[0] == "—"


def test_exp_page_rows_filters_by_family_scenario_and_convention():
    pytest.importorskip("pandas")
    from ui import exp_page

    assert len(exp_page._rows([], ["baseline"], "all")) == 0
    t = exp_page._rows(["E8 网络耦合"], ["congestion"], "all")
    assert set(t["family"]) == {"E8 网络耦合"}
    assert set(t["scenario"]) == {"congestion"}
