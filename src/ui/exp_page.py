# ui/exp_page.py
"""Experiment manager: every evaluated result on disk, grouped by experiment.

Reads results/*_eval.csv through data_aggregator. Nothing here clears a
market or retrains; it is an index of what has already been measured, with the
manipulated setting decoded from each file name so a row can be interpreted
without knowing the naming convention.

Each row carries the consumer-accounting generation it was written under. The
E8/E9 comparison spans both generations of consumer columns in principle, so
the table marks them rather than silently pooling.
"""

import os

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import data_aggregator as da
from ui import Page, register_page
from ui.theme import (C_SURFACE, C_PRIMARY, C_TEXT, C_MUTED, C_SUCCESS,
                      C_WARNING, C_DANGER, _placeholder_fig,
                      _placeholder_text)

_CHART_CONFIG = {"displayModeBar": "hover"}

# Columns the index table shows, in order: what was varied, then the outcome
# quantities, ending with the market-power term that separates scenarios.
_TABLE_COLUMNS = [
    ("policy_label", "结果文件"),
    ("variant", "变体"),
    ("scenario", "场景"),
    ("profit_delta", "利润增量", "{:,.0f}"),
    ("genuine_welfare_delta", "真实福利增量", "{:,.0f}"),
    ("cs_delta", "CS 增量", "{:,.0f}"),
    ("cp_delta", "CP 增量", "{:,.0f}"),
    ("lmp_markup_delta", "markup 增量", "{:+.4f}"),
    ("market_power_arb", "套利", "{:,.0f}"),
    ("market_power_power", "市场力", "{:,.0f}"),
]
_SIGNED_COLUMNS = {"profit_delta", "genuine_welfare_delta", "cs_delta",
                   "cp_delta"}

_CONVENTION_MARK = {
    "p_dis_excluded": "",
    "p_dis_offset_legacy": " ⚠旧口径",
    "unknown": " —",
}


def _card(title, children, subtitle=None):
    body = [html.H6(title, className="fw-bold mb-1",
                    style={"color": C_TEXT, "fontSize": "15px"})]
    if subtitle:
        body.append(html.Div(subtitle, style={
            "color": C_MUTED, "fontSize": "12px", "marginBottom": "8px"}))
    body.append(children)
    return dbc.Card(dbc.CardBody(body), className="mb-3 chart-card")


def _fmt(key, v):
    """(text, colour) for one cell.

    Labels pass through as text; a blank or absent value reads as a dash, so
    "this run did not vary anything" is distinguishable from a real label.
    """
    spec = {c[0]: (c[2] if len(c) > 2 else None) for c in _TABLE_COLUMNS}
    pattern = spec.get(key)
    if isinstance(v, (int, float)) and v is not None and not _isnan(v):
        color = C_MUTED
        if key in _SIGNED_COLUMNS and v != 0:
            color = C_SUCCESS if v > 0 else C_DANGER
        return (pattern.format(v) if pattern else str(v)), color
    if isinstance(v, str):
        return (v, C_TEXT) if v else ("—", C_MUTED)
    return "—", C_MUTED


def _isnan(v):
    return isinstance(v, float) and np.isnan(v)


def layout():
    table = da.experiment_table()
    if not len(table):
        return dbc.Container(
            _card("实验管理", _placeholder_text(
                "results/ 下暂无评估结果。")),
            fluid=True, className="mt-3")

    families = sorted(table["family"].unique(),
                      key=lambda f: list(table["family"]).index(f))
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("实验族", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": C_MUTED, "marginBottom": "6px"}),
                dcc.Dropdown(id="ex-families", options=families,
                             value=families, multi=True,
                             style={"fontSize": "13px"}),
            ], md=5),
            dbc.Col([
                html.Label("场景", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": C_MUTED, "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="ex-scenarios",
                    options=sorted(table["scenario"].unique()),
                    value=sorted(table["scenario"].unique()), multi=True,
                    style={"fontSize": "13px"}),
            ], md=4),
            dbc.Col([
                html.Label("口径", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": C_MUTED, "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="ex-convention",
                    options=[{"label": "全部（标注口径）", "value": "all"},
                             {"label": "仅现行口径", "value": "p_dis_excluded"}],
                    value="all", clearable=False,
                    style={"fontSize": "13px"}),
            ], md=3),
        ], className="mb-3"),

        html.Div(id="ex-summary", className="mb-2"),

        _card("E8 网络耦合：容量 → 市场力 → 消费者", dcc.Graph(
            id="ex-e8", figure=_placeholder_fig("加载中"),
            config=_CHART_CONFIG),
            subtitle="同一四场景形状下仅热容量裕度变化；沿 capacity 递减方向"
                     "看市场力与消费者侧的走向"),

        _card("E9 市场力惩罚对照", dcc.Graph(
            id="ex-e9", figure=_placeholder_fig("加载中"),
            config=_CHART_CONFIG),
            subtitle="λ=0 与 λ=0.3 的实测对比；惩罚是否真的压住价格影响项"),

        _card("结果索引", html.Div(id="ex-table"),
              subtitle="每个结果文件的 fleet（ALL）行；变体列是文件名里编码的"
                       "被操纵设置"),
    ], fluid=True, className="mt-3 pb-5")


def _rows(families, scenarios, convention):
    t = da.experiment_table()
    t = t[t["family"].isin(families or [])]
    t = t[t["scenario"].isin(scenarios or [])]
    if convention == "p_dis_excluded":
        t = t[t["cs_convention"] == "p_dis_excluded"]
    return t


def _scenario_short(n):
    return {"baseline": "基准", "high_re": "高可再生", "peak_load": "高峰负荷",
            "congestion": "空间阻塞"}.get(n, n)


def register_callbacks(app):
    @app.callback(
        Output("ex-summary", "children"),
        Input("ex-families", "value"),
        Input("ex-scenarios", "value"),
        Input("ex-convention", "value"))
    def _summary(families, scenarios, convention):
        t = _rows(families, scenarios, convention)
        if not len(t):
            return _placeholder_text("当前筛选无数据")
        legacy = int((t["cs_convention"] == "p_dis_offset_legacy").sum())
        items = [
            html.Span(f"{len(t)} 行 · {t['policy_label'].nunique()} 个结果文件"
                      f" · {t['family'].nunique()} 个实验族",
                      style={"fontSize": "13px", "color": C_TEXT,
                             "fontWeight": "600"}),
        ]
        if legacy:
            items.append(html.Span(
                f"；其中 {legacy} 行使用 2026-09-09 前的旧消费者口径"
                f"（CS/CP 符号相反）", style={"fontSize": "13px",
                                              "color": C_WARNING}))
        return html.Div(items)

    @app.callback(
        Output("ex-e8", "figure"),
        Input("ex-families", "value"),
        Input("ex-scenarios", "value"),
        Input("ex-convention", "value"))
    def _e8(families, scenarios, convention):
        t = _rows(families, scenarios, convention)
        t = t[t["family"] == "E8 网络耦合"].copy()
        if not len(t):
            return _placeholder_fig("未选择 E8 实验族")
        # Order by capacity so the x axis reads as increasing coupling slack.
        t["cap"] = t["variant"].str.extract(r"capacity=([\d.]+)").astype(float)
        t = t.dropna(subset=["cap"]).sort_values("cap")
        if not len(t):
            return _placeholder_fig("E8 行缺少可解析的 capacity 变体")
        fig = go.Figure()
        for metric, name, color, dash in (
                ("market_power_power", "市场力（价格影响项）", C_DANGER, None),
                ("cs_delta", "消费者剩余增量", C_SUCCESS, None),
                ("profit_delta", "储能利润增量", C_PRIMARY, "dot")):
            for sc, sub in t.groupby("scenario"):
                fig.add_trace(go.Scatter(
                    x=sub["cap"], y=sub[metric],
                    name=f"{name} · {_scenario_short(sc)}",
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=8),
                    legendgroup=name, showlegend=False,
                    hovertemplate="capacity=%{x}<br>%{y:,.0f} CNY<extra>"
                                  + f"{name} · {_scenario_short(sc)}</extra>"))
        # One legend entry per metric rather than per scenario.
        for metric, name, color, dash in (
                ("market_power_power", "市场力（价格影响项）", C_DANGER, None),
                ("cs_delta", "消费者剩余增量", C_SUCCESS, None),
                ("profit_delta", "储能利润增量", C_PRIMARY, "dot")):
            fig.add_trace(go.Scatter(x=[None], y=[None], name=name,
                                     mode="lines",
                                     line=dict(color=color, dash=dash,
                                               width=2)))
        fig.update_layout(xaxis_title="线路容量倍数 capacity（越左耦合越紧）",
                          yaxis_title="CNY / 日（相对真实出清）")
        fig.update_layout(template="plotly_white", paper_bgcolor=C_SURFACE,
                          plot_bgcolor="#f8fafc", height=430,
                          margin=dict(l=20, r=20, t=20, b=20),
                          legend=dict(orientation="h", y=-0.25))
        return fig

    @app.callback(
        Output("ex-e9", "figure"),
        Input("ex-families", "value"),
        Input("ex-scenarios", "value"),
        Input("ex-convention", "value"))
    def _e9(families, scenarios, convention):
        t = _rows(families, scenarios, convention)
        penalised = t[t["family"] == "E9 市场力惩罚"].copy()
        if not len(penalised):
            return _placeholder_fig("未选择 E9 实验族")
        # The control is the same training without the penalty: the E8 run at
        # capacity 1.5 is the λ=0 reference for the same code path.
        control = t[t["policy_label"] == "e8_matd3_15"].copy()
        if not len(control):
            return _placeholder_fig(
                "缺少 λ=0 对照（e8_matd3_15）。取消勾选限制或先跑该评估。")
        penalised["arm"] = "λ=0.3（惩罚）"
        control["arm"] = "λ=0（对照）"
        both = pd.concat([control, penalised], ignore_index=True)

        fig = go.Figure()
        metrics = [("market_power_power", "市场力（价格影响项）", C_DANGER),
                   ("cs_delta", "消费者剩余增量", C_SUCCESS),
                   ("profit_delta", "储能利润增量", C_PRIMARY)]
        for arm, sub in both.groupby("arm"):
            sub = sub.set_index("scenario")
            fig.add_trace(go.Bar(
                x=[_scenario_short(s) for s in sub.index],
                y=sub["market_power_power"], name=f"市场力 · {arm}",
                marker_color=C_DANGER if "惩罚" in arm else C_MUTED,
                opacity=0.95 if "惩罚" in arm else 0.55,
                hovertemplate="%{x}<br>%{y:,.0f} CNY<extra>" + arm
                              + "</extra>"))
        fig.update_layout(barmode="group",
                          xaxis_title="场景",
                          yaxis_title="市场力：价格影响项 (CNY/日)")
        # The consumer and profit arms go on a second axis: same units, but a
        # different magnitude scale than the market-power term.
        for arm, sub in both.groupby("arm"):
            fig.add_trace(go.Scatter(
                x=[_scenario_short(s) for s in sub["scenario"]],
                y=sub["cs_delta"], name=f"CS 增量 · {arm}",
                mode="markers+lines", yaxis="y2",
                marker=dict(size=10, symbol="diamond"),
                line=dict(color=C_SUCCESS, dash="dot")))
        fig.update_layout(template="plotly_white", paper_bgcolor=C_SURFACE,
                          plot_bgcolor="#f8fafc", height=430,
                          margin=dict(l=20, r=20, t=20, b=20),
                          yaxis2=dict(title="CS 增量 (CNY/日)",
                                      overlaying="y", side="right",
                                      showgrid=False),
                          legend=dict(orientation="h", y=-0.25))
        return fig

    @app.callback(
        Output("ex-table", "children"),
        Input("ex-families", "value"),
        Input("ex-scenarios", "value"),
        Input("ex-convention", "value"))
    def _table(families, scenarios, convention):
        t = _rows(families, scenarios, convention)
        if not len(t):
            return _placeholder_text("当前筛选无数据")
        header = [html.Th(name, style={"fontSize": "12px"})
                  for _, name, *_ in _TABLE_COLUMNS]
        header.append(html.Th("口径", style={"fontSize": "12px"}))
        body = []
        for r in t.to_dict("records"):
            tds = []
            for spec in _TABLE_COLUMNS:
                key, _name = spec[0], spec[1]
                txt, color = _fmt(key, r.get(key))
                tds.append(html.Td(txt, style={"fontSize": "12px",
                                               "color": color,
                                               "whiteSpace": "nowrap"}))
            conv = r.get("cs_convention", "unknown")
            tds.append(html.Td(conv, style={
                "fontSize": "11px",
                "color": C_WARNING if conv == "p_dis_offset_legacy"
                else C_MUTED}))
            body.append(html.Tr(tds))
        return dbc.Table([html.Thead(html.Tr(header)), html.Tbody(body)],
                         bordered=False, hover=True, size="sm",
                         responsive=True, className="mb-0")


register_page(Page(
    key="experiments",
    label_cn="实验管理",
    id_prefix="ex-",
    layout_fn=layout,
    register_callbacks=register_callbacks,
))
