# ui/shell.py
"""Application shell: the single Dash app, its navbar and its router.

One app instance serves every page. Two Dash apps in one process would mean two
callback registries, two injected stylesheets and two ports.

Page layouts are mounted by the router rather than laid out up front, so their
component ids do not exist when the app is constructed; callbacks referring to
them must be exempt from Dash's startup id validation, hence
suppress_callback_exceptions.
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from ui import PAGE_REGISTRY, default_key, nav_items, register_page
from ui import theme


def _navbar() -> dbc.Navbar:
    """Top navigation. Lives in the shell layout so it survives page swaps."""
    return dbc.Navbar(
        dbc.Container([
            html.Span("RL 电力市场研究平台",
                      style={"fontWeight": "700", "fontSize": "17px",
                             "color": theme.C_TEXT}),
            dbc.Nav([
                dbc.NavLink(item["label"], href=item["href"], active="exact",
                            style={"fontWeight": "600", "fontSize": "14px"})
                for item in nav_items()
            ], navbar=True, className="ms-auto"),
        ], fluid=True),
        color="white", dark=False, className="mb-3",
        style={"borderBottom": f"1px solid {theme.C_BORDER}"},
    )


def _not_found(pathname: str):
    return dbc.Container(dbc.Alert(
        [html.Strong("页面不存在  "), html.Code(pathname)],
        color="warning", className="mt-4"), fluid=True)


def create_app() -> dash.Dash:
    """Build the app with every registered page attached."""
    # Importing the page modules registers them; there is no package
    # auto-discovery in this flat layout, so the order is explicit.
    from ui import sim_page, train_page, user_page, exp_page      # noqa: F401

    app = dash.Dash(
        __name__,
        title="RL 电力市场研究平台",
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
    )
    app.index_string = theme.INDEX_STRING
    app.layout = html.Div([
        dcc.Location(id="url"),
        _navbar(),
        html.Div(id="page-content"),
    ])

    @app.callback(Output("page-content", "children"),
                  Input("url", "pathname"))
    def _route(pathname):
        key = (pathname or "/").lstrip("/").split("/")[0] or default_key()
        page = PAGE_REGISTRY.get(key)
        if page is None:
            return _not_found(pathname or "/")
        return page.layout_fn()

    for page in PAGE_REGISTRY.values():
        page.register_callbacks(app)
    return app
