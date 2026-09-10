# ui/user_page.py
"""User impact page: what end users pay and receive under each bidding policy.

Reads results/*_eval.csv through data_aggregator, so every figure is the same
number the evaluation CSV reports. All panels load from files; nothing here
clears a market.

Consumer surplus is defined as bid_value * served - consumer payment, with the
payment covering only the load energy a site imports (its own PV/wind comes
first and storage discharge does not offset the bill). Files written before the
2026-09-09 correction report the opposite sign on three of the four scenarios,
so the corrected generation is the default and the page says which one it is
showing.
"""

import numpy as np
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import data_aggregator as da
from ui import Page, register_page
from ui.theme import (C_SURFACE, C_PRIMARY, C_TEXT, C_MUTED, C_SUCCESS,
                      C_WARNING, C_DANGER, C_ACCENT, _placeholder_fig,
                      _placeholder_text)

_CHART_CONFIG = {"displayModeBar": "hover"}

_CONVENTION_LABELS = {
    "p_dis_excluded": "现行口径（2026-09-09 起）：储能放电不抵扣消费者账单",
    "p_dis_offset_legacy": "旧口径（2026-09-09 前）：储能放电抵消购电，符号相反",
    "unknown": "无消费者账目",
}


def _card(title, children, subtitle=None):
    body = [html.H6(title, className="fw-bold mb-1",
                    style={"color": C_TEXT, "fontSize": "15px"})]
    if subtitle:
        body.append(html.Div(subtitle, style={
            "color": C_MUTED, "fontSize": "12px", "marginBottom": "8px"}))
    body.append(children)
    return dbc.Card(dbc.CardBody(body), className="mb-3 chart-card")


def _kpi(label, value, color=None, hint=None):
    kids = [html.Div(label, style={"color": C_MUTED, "fontSize": "12px",
                                   "marginBottom": "4px"}),
            html.Div(value, style={"color": color or C_TEXT,
                                   "fontSize": "18px", "fontWeight": "700"})]
    if hint:
        kids.append(html.Div(hint, style={"color": C_MUTED,
                                          "fontSize": "11px"}))
    return dbc.Col(dbc.Card(dbc.CardBody(kids)), md=3, className="mb-2")


def _base_layout(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor=C_SURFACE,
                      plot_bgcolor="#f8fafc", height=400,
                      margin=dict(l=20, r=20, t=20, b=20),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def layout():
    pairs = da.pareto_pairs()
    if not len(pairs):
        return dbc.Container(
            _card("用户影响", _placeholder_text(
                "暂无可用的消费者账目结果。运行 "
                "eval_agents.py --consumer-metrics 生成。")),
            fluid=True, className="mt-3")

    policies = sorted(pairs["policy_label"].unique())
    scenarios = sorted(pairs["scenario"].unique())
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("策略（评估结果文件）", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": C_MUTED, "marginBottom": "6px"}),
                dcc.Dropdown(id="ui-policies", options=policies,
                             value=policies, multi=True,
                             style={"fontSize": "13px"}),
            ], md=6),
            dbc.Col([
                html.Label("场景", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": C_MUTED, "marginBottom": "6px"}),
                dcc.Dropdown(id="ui-scenarios", options=scenarios,
                             value=scenarios, multi=True,
                             style={"fontSize": "13px"}),
            ], md=6),
        ], className="mb-3"),

        html.Div(id="ui-convention-note", className="mb-2"),
        html.Div(id="ui-kpi-row", className="mb-3"),

        _card("消费者支付与剩余", dcc.Graph(
            id="ui-payment", figure=_placeholder_fig("加载中"),
            config=_CHART_CONFIG),
            subtitle="按策略×场景：绿色为消费者获益（CS 上升），红色为受损"),

        _card("利润–消费者剩余权衡", dcc.Graph(
            id="ui-pareto", figure=_placeholder_fig("加载中"),
            config=_CHART_CONFIG),
            subtitle="每个点是一个策略在某场景下的出清日；左上为非冲突区"
                     "（储能与用户同时获益），右下为权衡区"),

        _card("LMP 溢价与市场力", dcc.Graph(
            id="ui-markup", figure=_placeholder_fig("加载中"),
            config=_CHART_CONFIG),
            subtitle="markup 为负荷支付加权的 (LMP−wholesale)/wholesale；"
                     "市场力为价格影响项（相对真实出清）"),
    ], fluid=True, className="mt-3 pb-5")


def _filtered(policies, scenarios):
    """Rows matching the selection.

    An empty list means the user cleared every option, which must yield no
    rows rather than silently falling back to "all" — showing everything when
    nothing is selected would contradict the controls on screen.
    """
    df = da.pareto_pairs()
    df = df[df["policy_label"].isin(policies or [])]
    df = df[df["scenario"].isin(scenarios or [])]
    return df


def _scenario_short(name):
    return {"baseline": "基准", "high_re": "高可再生",
            "peak_load": "高峰负荷", "congestion": "空间阻塞"}.get(name, name)


def register_callbacks(app):
    @app.callback(
        Output("ui-convention-note", "children"),
        Input("ui-policies", "value"))
    def _note(_policies):
        """State which accounting generation the page is showing."""
        return dbc.Alert(
            [html.Strong("消费者账目口径："),
             _CONVENTION_LABELS["p_dis_excluded"],
             html.Br(),
             html.Span(
                 "results/ 中 2026-09-09 之前的结果使用旧口径（CS/CP 符号相反），"
                 "默认不参与本页对比，以免混口径得出相反结论。",
                 style={"fontSize": "12px"})],
            color="light", style={"fontSize": "13px",
                                  "borderLeft": f"4px solid {C_PRIMARY}"})

    @app.callback(
        Output("ui-kpi-row", "children"),
        Input("ui-policies", "value"),
        Input("ui-scenarios", "value"))
    def _kpis(policies, scenarios):
        df = _filtered(policies, scenarios)
        if not len(df):
            return _placeholder_text("当前筛选无数据")
        n_conflict = int(df["conflict"].sum())
        worst = df.loc[df["cs_delta"].idxmin()]
        best = df.loc[df["cs_delta"].idxmax()]
        return dbc.Row([
            _kpi("评估点数", f"{len(df)}",
                 hint=f"{df['policy_label'].nunique()} 个策略"),
            _kpi("平均利润增量", f"{df['profit_delta'].mean():,.0f}",
                 color=C_SUCCESS if df["profit_delta"].mean() > 0 else C_DANGER,
                 hint="CNY/日（相对真实报价）"),
            _kpi("平均消费者剩余增量", f"{df['cs_delta'].mean():,.0f}",
                 color=C_SUCCESS if df["cs_delta"].mean() > 0 else C_DANGER,
                 hint="CNY/日"),
            _kpi("利益冲突点数", f"{n_conflict}",
                 color=C_DANGER if n_conflict else C_SUCCESS,
                 hint="储能获利但用户受损"
                      + (f"；CS 最低 {worst['policy_label']}"
                         f"@{_scenario_short(worst['scenario'])}"
                         if n_conflict else "")),
        ])

    @app.callback(
        Output("ui-payment", "figure"),
        Input("ui-policies", "value"),
        Input("ui-scenarios", "value"))
    def _payment(policies, scenarios):
        df = _filtered(policies, scenarios)
        if not len(df):
            return _placeholder_fig("当前筛选无数据")
        df = df.copy()
        df["label"] = (df["policy_label"].str.replace("_eval", "", regex=False)
                       + " · " + df["scenario"].map(_scenario_short))
        df = df.sort_values("cs_delta")
        colors = [C_SUCCESS if v > 0 else C_DANGER for v in df["cs_delta"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["cs_delta"], y=df["label"],
                             orientation="h", marker_color=colors,
                             name="消费者剩余增量",
                             hovertemplate="%{y}<br>CS %{x:,.0f} CNY<extra></extra>"))
        # Payment moves opposite to surplus: the same money, other direction.
        fig.add_trace(go.Scatter(x=df["cp_delta"], y=df["label"],
                                 mode="markers", name="消费者支付增量",
                                 marker=dict(color=C_ACCENT, size=9,
                                             symbol="diamond"),
                                 hovertemplate="%{y}<br>CP %{x:,.0f} CNY<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=C_MUTED, width=1, dash="dot"))
        fig.update_layout(xaxis_title="CNY / 日（相对真实报价出清）",
                          yaxis_title="")
        return _base_layout(fig)

    @app.callback(
        Output("ui-pareto", "figure"),
        Input("ui-policies", "value"),
        Input("ui-scenarios", "value"))
    def _pareto(policies, scenarios):
        df = _filtered(policies, scenarios)
        if not len(df):
            return _placeholder_fig("当前筛选无数据")
        fig = go.Figure()
        symbols = {"baseline": "circle", "high_re": "square",
                   "peak_load": "triangle-up", "congestion": "diamond"}
        for pol, sub in df.groupby("policy_label"):
            fig.add_trace(go.Scatter(
                x=sub["profit_delta"], y=sub["cs_delta"], name=pol,
                mode="markers",
                marker=dict(size=12,
                            symbol=[symbols.get(s, "circle")
                                    for s in sub["scenario"]],
                            line=dict(width=1, color="white")),
                text=[_scenario_short(s) for s in sub["scenario"]],
                hovertemplate="%{text}<br>利润 %{x:,.0f}<br>"
                              "CS %{y:,.0f}<extra>" + pol + "</extra>"))
        fig.add_hline(y=0, line=dict(color=C_MUTED, width=1, dash="dot"))
        fig.add_vline(x=0, line=dict(color=C_MUTED, width=1, dash="dot"))
        # Label the two quadrants that matter for the mechanism argument.
        fig.add_annotation(x=0.02, y=0.97, xref="paper", yref="paper",
                           text="非冲突区<br>储能与用户同获益", showarrow=False,
                           align="left", font=dict(color=C_SUCCESS, size=11),
                           bgcolor="rgba(16,185,129,0.08)")
        fig.add_annotation(x=0.98, y=0.03, xref="paper", yref="paper",
                           text="权衡区<br>储能获益、用户受损", showarrow=False,
                           align="right", font=dict(color=C_DANGER, size=11),
                           bgcolor="rgba(239,68,68,0.08)")
        fig.update_layout(xaxis_title="储能利润增量 (CNY/日)",
                          yaxis_title="消费者剩余增量 (CNY/日)")
        return _base_layout(fig)

    @app.callback(
        Output("ui-markup", "figure"),
        Input("ui-policies", "value"),
        Input("ui-scenarios", "value"))
    def _markup(policies, scenarios):
        df = _filtered(policies, scenarios)
        if not len(df) or df["lmp_markup_delta"].isna().all():
            return _placeholder_fig("当前筛选无 markup / 市场力数据")
        fig = go.Figure()
        symbols = {"baseline": "circle", "high_re": "square",
                   "peak_load": "triangle-up", "congestion": "diamond"}
        for pol, sub in df.groupby("policy_label"):
            fig.add_trace(go.Scatter(
                x=sub["lmp_markup_delta"], y=sub["market_power_power"],
                name=pol, mode="markers",
                marker=dict(size=12,
                            symbol=[symbols.get(s, "circle")
                                    for s in sub["scenario"]],
                            line=dict(width=1, color="white")),
                text=[_scenario_short(s) for s in sub["scenario"]],
                hovertemplate="%{text}<br>markup %{x:.4f}<br>"
                              "市场力 %{y:,.0f}<extra>" + pol + "</extra>"))
        fig.add_vline(x=0, line=dict(color=C_MUTED, width=1, dash="dot"))
        fig.update_layout(
            xaxis_title="LMP markup 增量（负荷支付加权溢价）",
            yaxis_title="市场力：价格影响项 (CNY/日)",
            yaxis2=None)
        return _base_layout(fig)


register_page(Page(
    key="user",
    label_cn="用户影响",
    id_prefix="ui-",
    layout_fn=layout,
    register_callbacks=register_callbacks,
))
