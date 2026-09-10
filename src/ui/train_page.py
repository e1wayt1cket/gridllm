# ui/train_page.py
"""Training monitor page: run selection, learning curves, losses, day
economics and checkpoint comparison.

Panels that read files render on page load. Panels that clear the market run
only when asked, since each one costs Gurobi solves.
"""

import os

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import data_aggregator as da
from ui import Page, register_page
from ui.theme import (C_SURFACE, C_PRIMARY, C_TEXT, C_MUTED,
                      C_SUCCESS, C_WARNING, C_DANGER, _placeholder_fig,
                      _placeholder_text)

# Training curves share one episode axis; the loss curves get their own.
_CHART_CONFIG = {"displayModeBar": "hover"}

# Panels that clear the market: one truthful baseline episode plus one
# combined-fleet episode per scenario, each n_blocks clears.
CLEARS_PER_BLOCK = 2

RL_SCENARIO_OPTIONS = [
    {"label": "基准 · 典型日", "value": "baseline"},
    {"label": "高可再生 · PV/风出清 ×2", "value": "high_re"},
    {"label": "高峰负荷 · 负荷×1.5", "value": "peak_load"},
    {"label": "空间阻塞 · 空间负荷重分布", "value": "congestion"},
]
NBLOCK_OPTIONS = [
    {"label": "4 块（4 小时）", "value": 4},
    {"label": "8 块（8 小时）", "value": 8},
    {"label": "12 块（12 小时）", "value": 12},
    {"label": "24 块（完整一天）", "value": 24},
]


def _card(title, children, subtitle=None):
    body = [html.H6(title, className="fw-bold mb-1",
                    style={"color": C_TEXT, "fontSize": "15px"})]
    if subtitle:
        body.append(html.Div(subtitle, style={
            "color": C_MUTED, "fontSize": "12px", "marginBottom": "8px"}))
    body.append(children)
    return dbc.Card(dbc.CardBody(body), className="mb-3 chart-card")


def _kpi(label, value, color=None):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, style={"color": C_MUTED, "fontSize": "12px",
                               "marginBottom": "4px"}),
        html.Div(value, style={"color": color or C_TEXT, "fontSize": "18px",
                               "fontWeight": "700"}),
    ])), md=3, className="mb-2")


def _run_options():
    runs = da.list_runs()
    if not len(runs):
        return []
    opts = []
    for r in runs.to_dict("records"):
        econ = "·三层" if r["has_econ"] else ""
        label = (f"{r['run_id'][4:20]}  {r['algo'] or '?'}"
                 f"·seed{r['seed']}{econ}  best@{r['best_episode']}")
        opts.append({"label": label, "value": r["run_id"]})
    return opts


def layout():
    runs = _run_options()
    if not runs:
        return dbc.Container(
            _card("训练监控", _placeholder_text(
                "outputs/rl/ 下暂无训练产物，先运行 train_rl.py")),
            fluid=True, className="mt-3")

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("训练运行", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": C_MUTED, "marginBottom": "6px"}),
                dcc.Dropdown(id="tm-run", options=runs, value=runs[0]["value"],
                             clearable=False,
                             style={"fontSize": "13px"}),
            ], md=8),
            dbc.Col([
                html.Label(" " * 1, style={"display": "block"}),
                dbc.Button("刷新产物", id="tm-refresh", n_clicks=0,
                           color="secondary", outline=True,
                           style={"width": "100%", "fontWeight": "600"}),
            ], md=4),
        ], className="mb-3"),

        html.Div(id="tm-kpi-row", className="mb-2"),
        html.Div(id="tm-banner"),

        _card("训练曲线", dcc.Loading(dcc.Graph(
            id="tm-curves", figure=_placeholder_fig("加载中"),
            config=_CHART_CONFIG)),
            subtitle="reward / 逐代理 reward / 可再生消纳；虚线为训练内评估"),

        _card("损失与评论家分歧", dcc.Loading(dcc.Graph(
            id="tm-loss", figure=_placeholder_fig("加载中"),
            config=_CHART_CONFIG)),
            subtitle="critic / actor 损失；Q 分歧为归一化回报单位，不跨 run 比较"),

        _card("训练期用户代价", dcc.Loading(dcc.Graph(
            id="tm-econ", figure=_placeholder_fig("加载中"),
            config=_CHART_CONFIG)),
            subtitle="每集 RL 相对真实报价基准的消费者剩余/支付增量"),

        _card("Checkpoint 对比", html.Div([
            dbc.Row([
                dbc.Col([html.Label("场景", style={"fontSize": "12px",
                                                 "color": C_MUTED}),
                         dcc.Dropdown(id="tm-ckpt-scenarios",
                                      options=RL_SCENARIO_OPTIONS,
                                      value=["baseline"], multi=True,
                                      style={"fontSize": "13px"})], md=4),
                dbc.Col([html.Label("块数", style={"fontSize": "12px",
                                                 "color": C_MUTED}),
                         dcc.Dropdown(id="tm-ckpt-nblocks",
                                      options=NBLOCK_OPTIONS,
                                      value=24, clearable=False,
                                      style={"fontSize": "13px"})], md=4),
                dbc.Col([html.Label("策略目录", style={"fontSize": "12px",
                                                    "color": C_MUTED}),
                         dcc.Dropdown(id="tm-ckpt-dir", options=[],
                                      value=None,
                                      style={"fontSize": "13px"})], md=4),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Button("开始评估（0 次出清）", id="tm-ckpt-run",
                                   n_clicks=0, color="primary",
                                   style={"fontWeight": "600"}), md=6),
                dbc.Col(html.Div(id="tm-ckpt-status",
                                 style={"color": C_MUTED,
                                        "fontSize": "12px"}), md=6),
            ]),
            html.Div(id="tm-ckpt-table", className="mt-3"),
        ]),
            subtitle="每个 checkpoint 按配对种子重评估；出清次数显示在按钮上，"
                     "单 episode 为单一价格日，差异小于价格日噪声时不可读"),
    ], fluid=True, className="mt-3 pb-5")


def _empty(msg):
    return _placeholder_fig(msg)


def _available_checkpoints(policy_dir):
    """[(label, path)] for every artifact a policy directory offers.

    Numeric checkpoints load {name}_ckpt_{N}.pt from the run directory; the
    best/last roles are separate subdirectories of policies and must be
    evaluated from their own path, or they would silently re-score the final
    policies under a different label.
    """
    for r in da.list_policy_dirs().to_dict("records"):
        if r["name"] != policy_dir:
            continue
        out = [(f"ckpt_{n}", r["path"]) for n in r["checkpoints"]]
        for role in r["roles"]:
            sub = os.path.join(r["path"], role) if role != "final" else None
            out.append((role, sub or r["path"]))
        return out
    return [("final", os.path.join(da.policies_root(), policy_dir))]


# Table columns: (result key, Chinese header).
_CKPT_COLUMNS = [
    ("checkpoint", "Checkpoint"), ("scenario", "场景"),
    ("profit_delta", "利润增量"),
    ("genuine_welfare_delta", "真实福利增量"),
    ("cs_delta", "CS 增量"), ("cp_delta", "CP 增量"),
    ("lmp_markup_delta", "markup 增量"),
    ("market_power_power", "市场力"),
]
# Columns where a sign is meaningful and worth colouring.
_SIGNED_COLUMNS = {"profit_delta", "genuine_welfare_delta", "cs_delta",
                   "cp_delta"}


def _format_cell(key, v):
    """(text, colour) for one table cell.

    Labels (checkpoint role, scenario name) are text rather than missing
    values, so they must not fall through to the missing-value marker.
    """
    if isinstance(v, (int, float)) and v is not None and not _isnan(v):
        color = C_MUTED
        if key in _SIGNED_COLUMNS and v != 0:
            color = C_SUCCESS if v > 0 else C_DANGER
        return f"{v:,.2f}", color
    if v is not None and not _isnan(v):
        return str(v), C_TEXT
    return "—", C_MUTED


def _isnan(v):
    return isinstance(v, float) and np.isnan(v)


def _ckpt_table(rows):
    header = [html.Th(name, style={"textAlign": "right", "fontSize": "12px"})
              for _, name in _CKPT_COLUMNS]
    body = []
    for r in rows:
        tds = []
        for key, _ in _CKPT_COLUMNS:
            txt, color = _format_cell(key, r.get(key))
            tds.append(html.Td(txt, style={"textAlign": "right",
                                           "color": color,
                                           "fontSize": "12px"}))
        body.append(html.Tr(tds))
    return dbc.Table([html.Thead(html.Tr(header)), html.Tbody(body)],
                     bordered=False, hover=True, size="sm", className="mb-0")


def register_callbacks(app):
    @app.callback(
        Output("tm-run", "options"),
        Output("tm-run", "value"),
        Input("tm-refresh", "n_clicks"),
        State("tm-run", "value"),
        prevent_initial_call=True)
    def _refresh(n_clicks, current):
        """Re-scan outputs/rl/ so a run finished while the page is open shows
        up without restarting the server."""
        options = _run_options()
        values = {o["value"] for o in options}
        return options, (current if current in values
                         else (options[0]["value"] if options else None))

    @app.callback(
        Output("tm-kpi-row", "children"),
        Output("tm-banner", "children"),
        Input("tm-run", "value"))
    def _on_run(run_id):
        if not run_id:
            return _placeholder_text("未选择运行"), ""
        kpi = da.load_kpi(run_id)
        manifest = da.load_manifest(run_id)
        args = manifest.get("cli_args") or {}
        row = dbc.Row([
            _kpi("最佳集", str(kpi.get("best_episode", "—"))),
            _kpi("最佳回报", f"{kpi.get('best_mean_reward', 0):,.0f}"
                if kpi.get("best_mean_reward") is not None else "—"),
            _kpi("总集数", str(kpi.get("n_episodes", "—"))),
            _kpi("算法 / 种子",
                 f"{args.get('algo', '?')} · {args.get('seed', '?')}"),
        ])
        return row, _banner(run_id)

    def _banner(run_id):
        """Surface capture integrity and how the log directory was found."""
        metrics = da.load_metrics(run_id)
        items = []
        if "capture_fellbacks" in metrics.columns \
                and metrics["capture_fellbacks"].notna().any():
            fb = float(np.nansum(metrics["capture_fellbacks"].values))
            if fb > 0:
                items.append(dbc.Alert(
                    f"本 run 有 {fb:.0f} 次降级出清，三层指标列不可用",
                    color="danger", style={"fontSize": "13px"}))
        elif not da.load_manifest(run_id).get("log_dir"):
            items.append(dbc.Alert(
                "该 run 早于三层指标日志，训练期用户代价不可用；"
                "逐代理曲线取自 TensorBoard", color="info",
                style={"fontSize": "13px"}))
        mode = da.link_log_dir(run_id)[1]
        if mode == "ambiguous":
            items.append(dbc.Alert(
                "TensorBoard 日志时间戳歧义，逐代理曲线不可用",
                color="warning", style={"fontSize": "13px"}))
        return items

    @app.callback(
        Output("tm-curves", "figure"),
        Input("tm-run", "value"))
    def _curves(run_id):
        if not run_id:
            return _empty("未选择运行")
        metrics = da.load_metrics(run_id)
        if not len(metrics):
            return _empty("该 run 无 metrics.csv")
        fig = go.Figure()
        ep = metrics["episode"]
        fig.add_trace(go.Scatter(x=ep, y=metrics["mean_reward"],
                                 name="平均回报", mode="lines",
                                 line=dict(color=C_PRIMARY, width=2)))
        per_agent = da.agent_reward_frame(run_id)
        if len(per_agent):
            for agent in sorted(per_agent["agent"].unique()):
                sub = per_agent[per_agent["agent"] == agent]
                fig.add_trace(go.Scatter(
                    x=sub["episode"], y=sub["reward"], name=agent,
                    mode="lines", line=dict(width=1), opacity=0.45,
                    legendgroup="agents", showlegend=False))
        evals = da.load_evals(run_id)
        if len(evals):
            fig.add_trace(go.Scatter(
                x=evals["episode"], y=evals["mean_reward"], name="训练内评估",
                mode="lines+markers",
                line=dict(color=C_SUCCESS, width=2, dash="dash")))
        fig.update_layout(
            template="plotly_white", paper_bgcolor=C_SURFACE,
            plot_bgcolor="#f8fafc", height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Episode", yaxis_title="Reward",
            legend=dict(orientation="h", y=-0.18))
        return fig

    @app.callback(
        Output("tm-loss", "figure"),
        Input("tm-run", "value"))
    def _loss(run_id):
        if not run_id:
            return _empty("未选择运行")
        metrics = da.load_metrics(run_id)
        if not len(metrics) or metrics["critic_loss"].isna().all():
            return _empty("该 run 未记录损失（前 N 集为随机探索）")
        fig = go.Figure()
        ep = metrics["episode"]
        for col, name, color in (("critic_loss", "Critic loss", C_PRIMARY),
                                 ("actor_loss", "Actor loss", C_WARNING)):
            if col in metrics.columns:
                fig.add_trace(go.Scatter(
                    x=ep, y=metrics[col], name=name, mode="lines",
                    line=dict(color=color, width=2)))
        if "q_gap" in metrics.columns and metrics["q_gap"].notna().any():
            fig.add_trace(go.Scatter(
                x=ep, y=metrics["q_gap"], name="双评论家分歧（归一化）",
                mode="lines", yaxis="y2",
                line=dict(color=C_MUTED, width=1.5, dash="dot")))
            fig.update_layout(yaxis2=dict(
                title="Q 分歧（归一化）", overlaying="y", side="right",
                showgrid=False))
        fig.update_layout(
            template="plotly_white", paper_bgcolor=C_SURFACE,
            plot_bgcolor="#f8fafc", height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Episode", yaxis_title="Loss",
            legend=dict(orientation="h", y=-0.2))
        return fig

    @app.callback(
        Output("tm-econ", "figure"),
        Input("tm-run", "value"))
    def _econ(run_id):
        if not run_id:
            return _empty("未选择运行")
        metrics = da.load_metrics(run_id)
        if not len(metrics) or "cs_delta" not in metrics.columns \
                or metrics["cs_delta"].isna().all():
            return _empty("该 run 未记录训练期经济指标（需差分奖励 + 完整采集）")
        fig = go.Figure()
        ep = metrics["episode"]
        for col, name, color in (("cs_delta", "消费者剩余增量 (CS)",
                                  C_SUCCESS),
                                 ("cp_delta", "消费者支付增量 (CP)",
                                  C_DANGER),
                                 ("lmp_markup_delta", "LMP markup 增量",
                                  C_WARNING)):
            if col in metrics.columns and metrics[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=ep, y=metrics[col], name=name, mode="lines",
                    line=dict(color=color, width=2)))
        fig.update_layout(
            template="plotly_white", paper_bgcolor=C_SURFACE,
            plot_bgcolor="#f8fafc", height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Episode", yaxis_title="CNY（相对真实报价基准）",
            legend=dict(orientation="h", y=-0.2))
        return fig

    @app.callback(
        Output("tm-ckpt-dir", "options"),
        Input("tm-run", "value"))
    def _ckpt_dirs(run_id):
        manifest = da.load_manifest(run_id) if run_id else {}
        save_dir = (manifest.get("cli_args") or {}).get("save_dir")
        out = []
        for r in da.list_policy_dirs().to_dict("records"):
            if not r["checkpoints"] and not r["roles"]:
                continue
            mark = " ←本 run" if save_dir and \
                os.path.normpath(r["name"]) == \
                os.path.normpath(os.path.basename(save_dir)) else ""
            out.append({"label": f"{r['name']}{mark}", "value": r["name"]})
        return out

    @app.callback(
        Output("tm-ckpt-run", "children"),
        Input("tm-ckpt-scenarios", "value"),
        Input("tm-ckpt-nblocks", "value"),
        Input("tm-ckpt-dir", "value"))
    def _estimate(scenarios, n_blocks, policy_dir):
        """Put the clearing cost on the button: it is the one panel that
        spends solver time, so the count must be visible before clicking."""
        n_ckpt = _checkpoint_count(policy_dir)
        clears = CLEARS_PER_BLOCK * int(n_blocks or 0) \
            * len(scenarios or []) * max(1, n_ckpt)
        return f"开始评估（约 {clears} 次出清）"

    def _checkpoint_count(policy_dir):
        if not policy_dir:
            return 0
        for r in da.list_policy_dirs().to_dict("records"):
            if r["name"] == policy_dir:
                return len(r["checkpoints"]) + len(r["roles"])
        return 1

    @app.callback(
        Output("tm-ckpt-table", "children"),
        Output("tm-ckpt-status", "children"),
        Input("tm-ckpt-run", "n_clicks"),
        State("tm-ckpt-dir", "value"),
        State("tm-ckpt-scenarios", "value"),
        State("tm-ckpt-nblocks", "value"),
        prevent_initial_call=True)
    def _run_ckpt_eval(n_clicks, policy_dir, scenarios, n_blocks):
        if not policy_dir or not scenarios:
            return _placeholder_text("请选择策略目录与场景"), ""
        from eval_agents import evaluate_policy_dir
        rows, done, skipped = [], [], []
        for ck, path in _available_checkpoints(policy_dir):
            try:
                res = evaluate_policy_dir(
                    path, list(scenarios), n_blocks=int(n_blocks),
                    checkpoint=ck if isinstance(ck, int) else None,
                    consumer_metrics=True)
            except Exception as exc:                # noqa: BLE001
                skipped.append(f"{ck}: {exc}")
                continue
            for s in res["scenarios"]:
                # The role subdirectories (best/last) load their own policies,
                # so this label is what distinguishes the rows.
                rows.append({**s, "checkpoint": ck})
            done.append(str(ck))
        if not rows:
            return _placeholder_text(
                "评估失败：" + ("; ".join(skipped) or "无可用 checkpoint")), ""
        return _ckpt_table(rows), f"完成 {len(done)} 个 checkpoint"


register_page(Page(
    key="train",
    label_cn="训练监控",
    id_prefix="tm-",
    layout_fn=layout,
    register_callbacks=register_callbacks,
))
