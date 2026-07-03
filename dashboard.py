# dashboard.py
"""
配电网电力市场仿真仪表板
集成：场景切换 / OPF模式 / 策略切换 / 纳什检验 / 伪实时仿真 /
      自然语言解析 / 自动 AI 分析
运行: python dashboard.py
"""

import dash
from dash import dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

from models import MarketConfig
from grid import build_base_network, day_ahead_price_china
from dispatch import solve_opf_gurobi, StorageConstraints
from market import adaptive_bidding, two_settlement, clear_market
from scenarios import get_scenario
from llm import LLMAdvisor
from nash import NashEquilibriumTester

# ------------------------------
# 场景名映射
# ------------------------------
SCENARIO_NAMES_CN = {
    "baseline":        "基准--风光储",
    "high_re":         "高可再生渗透",
    "peak_load":       "高峰负荷",
    "congestion":      "线路阻塞",
    "re_ramp_drop":    "新能源骤降",
    "re_ramp_surge":   "新能源骤升",
}
CN_TO_EN = {v: k for k, v in SCENARIO_NAMES_CN.items()}

# Representative nodes for LMP display: (bus_index, label, color)
REPRESENTATIVE_BUSES = [
    (0,  "Bus1 (主网)",     "#e74c3c"),
    (5,  "Bus6 (居民产消)", "#3b82f6"),
    (12, "Bus13 (商业负荷)", "#10b981"),
    (17, "Bus18 (商业末端)", "#f59e0b"),
    (21, "Bus22 (居民产消)", "#8b5cf6"),
    (24, "Bus25 (工业产消)", "#ec4899"),
    (32, "Bus33 (工业末端)", "#6366f1"),
]

# High-contrast palette for SOC curves — maximum visual distinction
SOC_COLORS = [
    '#e74c3c',  # red
    '#3b82f6',  # blue
    '#10b981',  # green
    '#f59e0b',  # amber
    '#8b5cf6',  # violet
    '#ec4899',  # pink
    '#6366f1',  # indigo
    '#14b8a6',  # teal
    '#f97316',  # orange
    '#06b6d4',  # cyan
]

# ------------------------------
# 全局伪实时状态
# ------------------------------
# ------------------------------
# 时间轴格式化
# ------------------------------

def _make_datetime_x(T):
    """Return list of datetime objects for 15-min periods starting at 2024-01-01."""
    base = datetime(2024, 1, 1)
    return [base + timedelta(minutes=15 * i) for i in range(T)]

def _make_time_labels(T):
    """Generate H:MM labels for 15-min periods: 0:00, 0:15, ..., 23:45."""
    return [f"{i // 4}:{i % 4 * 15:02d}" for i in range(T)]

def _time_axis_config(T, step=16):
    """Return xaxis config dict for datetime axis."""
    return dict(
        tickformat='%H:%M',
        hoverformat='%-H:%M',
        dtick=step * 15 * 60 * 1000,
    )

# ------------------------------
# 图表构建函数
# ------------------------------

def create_topology_figure(net, agents_info=None, lmp_arr=None):
    """IEEE 33 节点拓扑图 (正交布局)"""

    NODE_COORDS = {
        0: (0, 0), 1: (2, 0), 2: (4, 0), 3: (6, 0), 4: (8, 0),
        5: (10, 0), 6: (12, 0), 7: (14, 0), 8: (16, 0), 9: (18, 0),
        10: (20, 0), 11: (22, 0), 12: (24, 0), 13: (26, 0), 14: (28, 0),
        15: (30, 0), 16: (32, 0), 17: (34, 0),
        18: (2, 3), 19: (4, 3), 20: (6, 3), 21: (8, 3),
        22: (4, -3), 23: (6, -3), 24: (8, -3),
        25: (10, -3), 26: (12, -3), 27: (14, -3), 28: (16, -3),
        29: (18, -3), 30: (20, -3), 31: (22, -3), 32: (24, -3),
    }

    LINES = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10),
        (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), (16,17),
        (1,18), (18,19), (19,20), (20,21),
        (2,22), (22,23), (23,24),
        (5,25), (25,26), (26,27), (27,28), (28,29), (29,30), (30,31), (31,32),
    ]

    C_TEXT = "#1e293b"
    C_CARD = "#ffffff"
    C_BORDER = "#e2e8f0"
    C_NODE = "#4f46e5"

    bus_agents = {b: [] for b in range(33)}
    if agents_info:
        for info in agents_info:
            bus_agents[info["bus"]].append(info["name"])

    fig = go.Figure()

    for i, j in LINES:
        x0, y0 = NODE_COORDS[i]
        x1, y1 = NODE_COORDS[j]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=C_BORDER, width=2.5),
            hoverinfo="skip", showlegend=False
        ))

    node_x, node_y, hover_text, labels = [], [], [], []
    for b in range(33):
        x, y = NODE_COORDS[b]
        node_x.append(x)
        node_y.append(y)
        labels.append(f"Bus{b+1}")

        txt = f"<b>Bus{b+1}</b>"
        agents_on = bus_agents[b]
        if agents_on:
            txt += f"<br>Agent: {', '.join(agents_on[:2])}"
            if len(agents_on) > 2:
                txt += f" +{len(agents_on)}"
        hover_text.append(txt)

    # LMP-based node coloring — use numeric values so colorscale + colorbar work
    lmp_values = None
    lmp_cmin, lmp_cmax = 0.0, 1.0
    if lmp_arr is not None and not np.all(lmp_arr == 0):
        lmp_avg = lmp_arr.mean(axis=0)
        # Clip color range at 5th/95th percentile so outliers don't wash out contrast
        lmp_cmin = float(np.percentile(lmp_avg, 5))
        lmp_cmax = float(np.percentile(lmp_avg, 95))
        if lmp_cmax - lmp_cmin < 1.0:
            lmp_cmax = lmp_cmin + 1.0
        lmp_values = lmp_avg.tolist()
        for i, b in enumerate(range(33)):
            hover_text[i] += f"<br>LMP avg: {lmp_avg[b]:.1f} CNY/MWh"

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=18,
            color=lmp_values if lmp_values is not None else [C_NODE] * 33,
            cmin=lmp_cmin if lmp_values is not None else None,
            cmax=lmp_cmax if lmp_values is not None else None,
            colorscale='RdYlBu_r',
            line=dict(width=2, color='#334155'),
            showscale=lmp_values is not None,
            colorbar=dict(
                title="Avg LMP (CNY/MWh)", x=1.02, len=0.8,
                tickformat='.0f',
            ) if lmp_values is not None else None,
        ),
        text=labels,
        textposition="top center",
        textfont=dict(size=10, color='#1e293b', family='Arial Black'),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
        showlegend=False,
        name="bus"
    ))

    if agents_info:
        prosumer_buses = list({info["bus"] for info in agents_info if info.get("is_prosumer")})
        if prosumer_buses:
            px = [NODE_COORDS[b][0] for b in prosumer_buses]
            py = [NODE_COORDS[b][1] for b in prosumer_buses]
            fig.add_trace(go.Scatter(
                x=px, y=py,
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=14,
                    color="#f59e0b",
                    line=dict(width=2, color='#d0d7e3')
                ),
                name="产消者",
                text=[f"Bus{b+1}" for b in prosumer_buses],
                hovertemplate="产消者 Bus %{text}<extra></extra>"
            ))

    fig.update_layout(
        title=dict(
            text="IEEE 33 Bus 配电系统拓扑 (星标 = 产消者)",
            font=dict(size=15, color='#2c3e50')
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        template="plotly_white",
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_CARD,
        margin=dict(l=20, r=20, t=60, b=20),
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.9)", font=dict(color=C_TEXT, size=12)),
        font=dict(color=C_TEXT),
    )

    return fig


def create_lmp_figure(lmp_matrix, title="节点边际电价"):
    if lmp_matrix is None or np.all(lmp_matrix == 0):
        fig = go.Figure()
        fig.add_annotation(text="求解失败或数据异常 (LMP全为零)", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#e74c3c", size=16))
        fig.update_layout(title=title, template="plotly_white",
                          paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc')
        return fig
    T, n = lmp_matrix.shape
    x_dt = _make_datetime_x(T)
    time_labels = _make_time_labels(T)

    # IQR envelope (25th-75th percentile) — robust against outlier buses
    p25 = np.percentile(lmp_matrix, 25, axis=1)
    p75 = np.percentile(lmp_matrix, 75, axis=1)
    p_min = lmp_matrix.min(axis=1)
    p_max = lmp_matrix.max(axis=1)
    p_mean = lmp_matrix.mean(axis=1)

    fig = go.Figure()
    # Full range as thin reference lines
    fig.add_trace(go.Scatter(x=x_dt, y=p_max, mode='lines',
                             line=dict(color="#cbd5e1", width=0.5, dash='dot'),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_dt, y=p_min, mode='lines',
                             line=dict(color="#cbd5e1", width=0.5, dash='dot'),
                             showlegend=False, hoverinfo='skip',
                             fill='tonexty', fillcolor='rgba(203,213,225,0.15)',
                             name='全节点范围'))
    # IQR envelope as main shaded band
    fig.add_trace(go.Scatter(x=x_dt, y=p75, mode='lines',
                             line=dict(color="#94a3b8", width=0.5),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_dt, y=p25, mode='lines',
                             line=dict(color="#94a3b8", width=0.5),
                             fill='tonexty', fillcolor='rgba(148,163,184,0.25)',
                             name='IQR (25%-75%)', hoverinfo='skip'))
    # Mean reference line
    fig.add_trace(go.Scatter(
        x=x_dt, y=p_mean, mode='lines',
        name='全网均值', line=dict(color='#1e293b', width=2.2, dash='dash'),
        customdata=time_labels,
        hovertemplate='%{customdata}<br>LMP=%{y:.1f} CNY/MWh<extra>全网均值</extra>',
    ))

    # Representative node curves
    for bus_idx, label, color in REPRESENTATIVE_BUSES:
        if bus_idx < n:
            fig.add_trace(go.Scatter(
                x=x_dt, y=lmp_matrix[:, bus_idx], mode='lines',
                name=label, line=dict(color=color, width=1.8),
                customdata=time_labels,
                hovertemplate='%{customdata}<br>LMP=%{y:.1f} CNY/MWh<extra>%{fullData.name}</extra>',
            ))

    fig.update_layout(height=630, title=title,
                      xaxis=dict(title="时间", **_time_axis_config(T)),
                      yaxis_title="LMP (CNY/MWh)",
                      template="plotly_white", legend=dict(orientation='h', y=1.12),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor="#fafbfc")
    return fig

def create_trade_figure(da_results, agents):
    T = len(next(iter(da_results["schedules"].values()))["p_buy"])
    x_dt = _make_datetime_x(T)
    time_labels = _make_time_labels(T)
    buy = np.zeros(T); sell = np.zeros(T)
    for a in agents:
        s = da_results["schedules"][a.name]
        buy += s["p_buy"]; sell += s["p_sell"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x_dt, y=buy, mode='lines', name='总购电量 (MW)',
                             line=dict(color='#ef4444', width=2),
                             customdata=time_labels,
                             hovertemplate='%{customdata}<br>购电=%{y:.2f} MW<extra></extra>'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=x_dt, y=sell, mode='lines', name='总售电量 (MW)',
                             line=dict(color='#10b981', width=2),
                             customdata=time_labels,
                             hovertemplate='%{customdata}<br>售电=%{y:.2f} MW<extra></extra>'),
                  secondary_y=False)
    fig.update_layout(height=525,
                      xaxis=dict(title="时间", **_time_axis_config(T)),
                      yaxis_title="功率 (MW)",
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation='h', y=1.1),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc')
    return fig

def create_soc_figure(da_results, agents):
    storage_agents = [a for a in agents if a.storage is not None]
    if not storage_agents:
        return go.Figure().update_layout(title="无储能设备", template="plotly_white")
    T = len(da_results["schedules"][storage_agents[0].name]["soc"])
    x_dt = _make_datetime_x(T)
    time_labels = _make_time_labels(T)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("储能 SOC", "充放电功率"),
                        row_heights=[0.52, 0.48])
    for i, a in enumerate(storage_agents):
        color = SOC_COLORS[i % len(SOC_COLORS)]
        s = da_results["schedules"][a.name]
        fig.add_trace(go.Scatter(x=x_dt, y=s["soc"]*100, mode='lines',
                                 name=a.name, line=dict(width=2.5, color=color),
                                 legendgroup=f'g{i}',
                                 customdata=time_labels,
                                 hovertemplate='%{customdata}<br>SOC=%{y:.1f}%<extra></extra>'),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=x_dt, y=s["p_ch"], name=f"{a.name} 充电",
                             marker_color=color, opacity=0.65,
                             legendgroup=f'g{i}', showlegend=False,
                             customdata=time_labels,
                             hovertemplate='%{customdata}<br>充电=%{y:.3f} MW<extra></extra>'),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=x_dt, y=-s["p_dis"], name=f"{a.name} 放电",
                             marker_color=color, opacity=0.35,
                             marker_pattern_shape='/',
                             legendgroup=f'g{i}', showlegend=False,
                             customdata=time_labels,
                             hovertemplate='%{customdata}<br>放电=%{y:.3f} MW<extra></extra>'),
                      row=2, col=1)
    fig.update_layout(height=1050, barmode='overlay', template="plotly_white",
                      hovermode="x unified",
                      legend=dict(orientation='v', yanchor='top', y=0.99,
                                  xanchor='left', x=1.01, bgcolor='rgba(255,255,255,0.9)'),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc')
    fig.update_xaxes(**_time_axis_config(T), row=2, col=1)
    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="SOC (%)", row=1, col=1)
    fig.update_yaxes(title_text="功率 (MW)", row=2, col=1)
    return fig

def create_re_gen_figure(da_results, agents):
    """Renewable generation: PV and wind actual output over time."""
    T = len(next(iter(da_results["schedules"].values()))["served"])
    x_dt = _make_datetime_x(T)
    time_labels = _make_time_labels(T)
    pv_total = np.zeros(T)
    wind_total = np.zeros(T)
    for a in agents:
        s = da_results["schedules"][a.name]
        pv_total += s.get("pv_used", np.zeros(T))
        wind_total += s.get("wind_used", np.zeros(T))

    has_pv = pv_total.max() > 0.001
    has_wind = wind_total.max() > 0.001

    fig = go.Figure()
    if has_pv:
        fig.add_trace(go.Scatter(
            x=x_dt, y=pv_total, mode='lines', name='光伏 (PV)',
            line=dict(color='#f59e0b', width=2.5),
            fill='tozeroy', fillcolor='rgba(245,158,11,0.25)',
            customdata=time_labels,
            hovertemplate='%{customdata}<br>PV=%{y:.3f} MW<extra></extra>',
        ))
    if has_wind:
        fig.add_trace(go.Scatter(
            x=x_dt, y=wind_total, mode='lines', name='风电 (Wind)',
            line=dict(color='#3b82f6', width=2.5),
            fill='tozeroy', fillcolor='rgba(59,130,246,0.25)',
            customdata=time_labels,
            hovertemplate='%{customdata}<br>Wind=%{y:.3f} MW<extra></extra>',
        ))
    fig.add_trace(go.Scatter(
        x=x_dt, y=pv_total + wind_total, mode='lines',
        name='可再生总计', line=dict(color='#10b981', width=2, dash='dot'),
        customdata=time_labels,
        hovertemplate='%{customdata}<br>RE=%{y:.3f} MW<extra></extra>',
    ))
    fig.update_layout(height=480, title="可再生能源实际出力",
                      xaxis=dict(title="时间", **_time_axis_config(T, step=4)),
                      yaxis_title="功率 (MW)",
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation='h', y=1.12),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc')
    return fig

def create_load_figure(da_results, agents):
    """Total load profile: served, unserved, and total forecast."""
    T = len(next(iter(da_results["schedules"].values()))["served"])
    x_dt = _make_datetime_x(T)
    time_labels = _make_time_labels(T)
    served = np.zeros(T)
    unserved = np.zeros(T)
    for a in agents:
        s = da_results["schedules"][a.name]
        served += s.get("served", np.zeros(T))
        unserved += s.get("unserved", np.zeros(T))

    total_load = served + unserved
    has_unserved = unserved.max() > 0.01

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_dt, y=served, mode='lines', name='已满足负荷',
        line=dict(color='#3b82f6', width=2.5),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.15)',
        customdata=time_labels,
        hovertemplate='%{customdata}<br>已满足=%{y:.2f} MW<extra></extra>',
    ))
    if has_unserved:
        fig.add_trace(go.Scatter(
            x=x_dt, y=total_load, mode='lines', name='总负荷',
            line=dict(color='#ef4444', width=2.5),
            fill='tonexty', fillcolor='rgba(239,68,68,0.2)',
            customdata=time_labels,
            hovertemplate='%{customdata}<br>总负荷=%{y:.2f} MW<extra></extra>',
        ))

    fig.update_layout(height=480, title="负荷曲线",
                      xaxis=dict(title="时间", **_time_axis_config(T)),
                      yaxis_title="功率 (MW)",
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation='h', y=1.12),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc')
    return fig

def create_kpi_cards(da_results, rt_results, payment, agents, config=None):
    load_total = sum(np.sum(a.load_forecast) for a in agents)
    served = sum(np.sum(da_results["schedules"][a.name]["served"]) for a in agents)
    satisfaction = (served / load_total * 100) if load_total > 0 else 100.0
    total_cost = sum(payment.values())

    carbon_emissions = da_results.get('carbon_emissions', 0)
    carbon_intensity = da_results.get('carbon_intensity', 0)
    curtailment = da_results.get('total_curtailment', 0)

    items = [
        ("日前社会福利", f"{da_results['welfare']:,.0f} CNY",   "#4f46e5"),
        ("实时社会福利", f"{rt_results['welfare']:,.0f} CNY",   "#6366f1"),
        ("可再生消纳率", f"{da_results['re_consumption_rate']:.1f}%", "#10b981"),
        ("负荷满足率",   f"{satisfaction:.1f}%",                "#3b82f6"),
        ("总市场成本",   f"{total_cost:,.0f} CNY",              "#f59e0b"),
        ("碳排放总量",   f"{carbon_emissions:.1f} tCO2",        "#ef4444"),
        ("碳强度",       f"{carbon_intensity:.3f} tCO2/MWh",    "#f97316"),
        ("弃电量",       f"{curtailment:.1f} MWh",              "#8b5cf6"),
    ]

    cards = []
    for label, value, accent in items:
        cards.append(
            dbc.Col(
                dbc.Card([
                    html.Div(label, style={
                        'fontSize': '12px', 'fontWeight': '600', 'color': '#64748b',
                        'letterSpacing': '0.5px', 'marginBottom': '6px'
                    }),
                    html.Div(value, style={
                        'fontSize': '17px', 'fontWeight': '700', 'color': '#1e293b'
                    }),
                ], body=True, style={
                    'borderTop': f'3px solid {accent}',
                    'borderRadius': '14px',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
                    'height': '100%',
                }), xs=6, sm=4, md=3,
            )
        )

    return dbc.Row(cards, className='g-3')

def create_payment_table(payment, agents):
    rows = []
    for a in agents:
        val = payment[a.name]
        color = "#ef4444" if val > 0 else "#10b981" if val < 0 else "#64748b"
        rows.append(html.Tr([
            html.Td(a.name, style={'fontWeight': '600', 'color': '#1e293b'}),
            html.Td(dbc.Badge(a.load_type, color='light', text_color='#475569',
                              className='border', style={'fontSize': '11px'})),
            html.Td(f"{val:,.2f}", style={
                'color': color, 'fontWeight': '700',
                'fontFamily': '"JetBrains Mono", monospace'
            }),
        ]))
    total_pay = sum(payment.values())
    rows.append(html.Tr([
        html.Td("总计", style={'fontWeight': '700', 'color': '#1e293b'}),
        html.Td(""),
        html.Td(f"{total_pay:,.2f}", style={
            'fontWeight': '700', 'fontFamily': '"JetBrains Mono", monospace',
            'color': '#1e293b'
        }),
    ], style={'background': '#fafbfc'}))
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("智能体"), html.Th("类型"), html.Th("结算金额 (CNY)")
        ])),
        html.Tbody(rows),
    ], bordered=True, hover=True, responsive=True, size='sm',
       style={'fontSize': '13px'})

# ------------------------------
# AI 分析曲线特征提取
# ------------------------------
def build_simulation_outputs(scenario_en, strategy="rl", opf_mode="socp", nl_msg_prefix=""):
    """Run simulation for a scenario and build all chart/KPI/table outputs.

    Returns the 8-tuple expected by the Dash output callbacks:
    (kpi, topo_fig, lmp_fig, trade_fig, soc_fig, pay_tab, nl_msg, ai_output)
    """
    from config_loader import get_scenario_cfg
    sc_cfg = get_scenario_cfg(scenario_en) or {}
    config = MarketConfig(opf_mode=opf_mode, verbose=False)
    T = 96
    agents, _ = get_scenario(scenario_en, T=T, config=config)
    da_actions = adaptive_bidding(agents, config, strategy=strategy, T=T)
    da_results = clear_market(agents, T, "DA", da_actions, config)
    rt_actions = adaptive_bidding(agents, config, strategy=strategy, T=T)
    rt_results = clear_market(agents, T, "RT", rt_actions, config)
    payment = two_settlement(agents, da_results, rt_results)

    net = build_base_network(config)
    kpi = create_kpi_cards(da_results, rt_results, payment, agents, config)
    agents_info = [{"bus": a.bus, "name": a.name, "is_prosumer": a.is_prosumer} for a in agents]
    topo_fig = create_topology_figure(net, agents_info=agents_info, lmp_arr=da_results['lmp'])
    lmp_fig = create_lmp_figure(da_results['lmp'], "日前节点边际电价 (LMP)")
    trade_fig = create_trade_figure(da_results, agents)
    soc_fig = create_soc_figure(da_results, agents)
    re_gen_fig = create_re_gen_figure(da_results, agents)
    load_fig = create_load_figure(da_results, agents)
    pay_tab = create_payment_table(payment, agents)

    scenario_cn = SCENARIO_NAMES_CN.get(scenario_en, scenario_en)
    advisor = LLMAdvisor()
    summary = build_insight_summary(da_results, agents, scenario_cn)
    ai_output = _render_ai_sections(advisor.get_insight(summary))

    nl_msg = (f"{nl_msg_prefix}场景: {scenario_cn} | "
              f"策略: {strategy} | OPF: {opf_mode}") if nl_msg_prefix else \
             (f"场景: {scenario_cn} | 策略: {strategy} | OPF: {opf_mode}")

    sim_state = {
        "scenario": scenario_en, "strategy": strategy,
        "opf_mode": opf_mode, "T": T,
    }

    return (kpi, topo_fig, lmp_fig, trade_fig, soc_fig, re_gen_fig, load_fig, pay_tab, nl_msg, ai_output, sim_state)


def build_insight_summary(da_results, agents, scenario_cn):
    lmp_matrix = da_results['lmp']
    avg_lmp = lmp_matrix.mean()
    min_lmp = lmp_matrix.min()
    max_lmp = lmp_matrix.max()

    soc_values = []
    total_ch = 0.0
    total_dis = 0.0
    active = False
    for a in agents:
        sched = da_results['schedules'][a.name]
        if a.storage:
            soc_arr = sched['soc']
            soc_values.extend(soc_arr)
            total_ch += np.sum(sched['p_ch'])
            total_dis += np.sum(sched['p_dis'])
            active = active or (np.any(sched['p_ch'] > 0.01) or np.any(sched['p_dis'] > 0.01))
    avg_soc = np.mean(soc_values) * 100 if soc_values else 0.0
    min_soc = np.min(soc_values) * 100 if soc_values else 0.0
    max_soc = np.max(soc_values) * 100 if soc_values else 0.0

    load_total = sum(np.sum(a.load_forecast) for a in agents)
    served = sum(np.sum(da_results['schedules'][a.name]['served']) for a in agents)
    satisfaction = (served / load_total * 100) if load_total > 0 else 100.0

    total_buy = sum(np.sum(da_results['schedules'][a.name]['p_buy']) for a in agents)
    total_sell = sum(np.sum(da_results['schedules'][a.name]['p_sell']) for a in agents)

    return {
        'scenario': scenario_cn,
        'welfare_da': f"{da_results['welfare']:.0f}",
        're_rate': f"{da_results['re_consumption_rate']:.1f}",
        'load_sat': f"{satisfaction:.1f}",
        'avg_lmp': f"{avg_lmp:.1f}",
        'min_lmp': f"{min_lmp:.1f}",
        'max_lmp': f"{max_lmp:.1f}",
        'avg_soc': f"{avg_soc:.1f}",
        'min_soc': f"{min_soc:.1f}",
        'max_soc': f"{max_soc:.1f}",
        'total_ch': f"{total_ch:.2f}",
        'total_dis': f"{total_dis:.2f}",
        'total_buy': f"{total_buy:.2f}",
        'total_sell': f"{total_sell:.2f}",
        'storage_active': '是' if active else '否',
        'carbon_emissions': f"{da_results.get('carbon_emissions', 0):.1f}",
        'carbon_intensity': f"{da_results.get('carbon_intensity', 0):.3f}",
        'curtailment': f"{da_results.get('total_curtailment', 0):.1f}",
    }

# ------------------------------
# Dash 应用
# ------------------------------
app = dash.Dash(__name__, title="配电网电力市场仿真仪表板",
                external_stylesheets=[dbc.themes.FLATLY])

# -- 颜色面板 (保留原有设计) --
C_BG       = "#f0f2f5"
C_SURFACE  = "#ffffff"
C_PRIMARY  = "#4f46e5"
C_ACCENT   = "#7c3aed"
C_TEXT     = "#1e293b"
C_MUTED    = "#64748b"
C_BORDER   = "#e2e8f0"
C_SUCCESS  = "#10b981"
C_WARNING  = "#f59e0b"
C_DANGER   = "#ef4444"
C_INFO     = "#3b82f6"

# -- 占位辅助函数 --
def _placeholder_fig(text="等待仿真结果..."):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white", paper_bgcolor=C_SURFACE,
        plot_bgcolor='#f8fafc', height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.add_annotation(text=text, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=C_MUTED, size=15))
    return fig

def _placeholder_kpi():
    return dbc.Alert(
        "输入场景描述并点击「运行仿真」查看结果",
        color='light', className='text-center',
        style={'color': C_MUTED, 'borderRadius': '14px'}
    )

def _placeholder_text(text="暂无数据"):
    return html.Div(text, style={'color': C_MUTED, 'textAlign': 'center',
                                  'padding': '24px', 'fontSize': '14px'})

def _render_ai_sections(ai_text: str):
    """Split LLM output into numbered sections and render as styled cards."""
    if not ai_text or not ai_text.strip():
        return _placeholder_text("AI 分析暂不可用")

    import re
    # Split on numbered section headers: "1. xxx", "2. xxx", etc.
    parts = re.split(r'\n(?=\d+\.\s)', ai_text.strip())
    if len(parts) <= 1:
        # Fallback: try splitting on double newlines
        parts = [p.strip() for p in ai_text.strip().split('\n\n') if p.strip()]
        if len(parts) <= 1:
            return html.Div(ai_text, style={'whiteSpace': 'pre-wrap', 'lineHeight': '1.8',
                                            'color': C_TEXT, 'fontSize': '14px'})

    section_colors = ['#4f46e5', '#10b981', '#f59e0b', '#ef4444']
    section_icons = ['📊', '👥', '⚠️', '💡']
    cards = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        lines = part.split('\n', 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ''
        accent = section_colors[min(i, len(section_colors) - 1)]
        icon = section_icons[min(i, len(section_icons) - 1)]
        cards.append(
            dbc.Card(
                dbc.CardBody([
                    html.Div(f"{icon} {header}", style={
                        'fontSize': '14px', 'fontWeight': '700', 'color': accent,
                        'marginBottom': '10px', 'letterSpacing': '0.3px',
                    }),
                    html.Div(body, style={
                        'whiteSpace': 'pre-wrap', 'lineHeight': '1.7',
                        'color': '#334155', 'fontSize': '13px',
                    }),
                ]),
                style={
                    'borderLeft': f'4px solid {accent}',
                    'borderRadius': '10px',
                    'marginBottom': '12px',
                    'boxShadow': '0 1px 2px rgba(0,0,0,0.04)',
                    'backgroundColor': '#ffffff',
                }
            )
        )
    return html.Div(cards)

app.layout = dbc.Container([
    # -- header --
    dbc.Row(dbc.Col(
        html.H1("配电网电力市场仿真仪表板",
                style={'fontSize': '26px', 'fontWeight': '700', 'color': C_TEXT}),
        className='text-center py-3'
    )),

    # -- 场景快速选择 --
    dbc.Row([
        dbc.Col([
            html.Label("场景快速选择", style={
                'fontSize': '13px', 'fontWeight': '600', 'color': C_MUTED,
                'marginBottom': '10px', 'display': 'block'
            }),
            dbc.ButtonGroup([
                dbc.Button("基准", id='scenario-btn-baseline', n_clicks=0,
                           color='primary', outline=True, className='scenario-chip',
                           style={'fontWeight': '600', 'fontSize': '14px',
                                  'padding': '8px 18px', 'borderRadius': '10px 0 0 10px',
                                  'borderRight': '1px solid #c7d2fe'}),
                dbc.Button("高可再生", id='scenario-btn-high_re', n_clicks=0,
                           color='success', outline=True, className='scenario-chip',
                           style={'fontWeight': '600', 'fontSize': '14px',
                                  'padding': '8px 16px', 'borderRadius': '0',
                                  'borderRight': '1px solid #a7f3d0'}),
                dbc.Button("高峰负荷", id='scenario-btn-peak_load', n_clicks=0,
                           color='warning', outline=True, className='scenario-chip',
                           style={'fontWeight': '600', 'fontSize': '14px',
                                  'padding': '8px 16px', 'borderRadius': '0',
                                  'borderRight': '1px solid #fde68a'}),
                dbc.Button("线路阻塞", id='scenario-btn-congestion', n_clicks=0,
                           color='danger', outline=True, className='scenario-chip',
                           style={'fontWeight': '600', 'fontSize': '14px',
                                  'padding': '8px 18px', 'borderRadius': '0 10px 10px 0'}),
            ], style={'boxShadow': '0 1px 3px rgba(0,0,0,0.06)'}),
        ], md=12, className='text-center'),
    ], className='mb-4'),

    # -- NL 输入卡片 --
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("场景描述", style={
                        'fontSize': '13px', 'fontWeight': '600', 'color': C_MUTED,
                        'marginBottom': '8px'
                    }),
                    dcc.Textarea(id='nl-input',
                        placeholder='用自然语言描述你想仿真的场景，例如：在bus 20新增5MW光伏，bus 10-15负荷翻倍...',
                        value='',
                        style={'width': '100%', 'height': '80px', 'padding': '14px 18px',
                               'borderRadius': '10px', 'border': f'1px solid {C_BORDER}',
                               'backgroundColor': C_BG, 'color': C_TEXT, 'fontSize': '14px',
                               'boxSizing': 'border-box', 'outline': 'none',
                               'resize': 'vertical', 'fontFamily': 'inherit'}),
                ], md=10, className='pe-md-3'),
                dbc.Col([
                    html.Label(" ", style={'display': 'block', 'marginBottom': '8px'}),
                    dbc.Button("运行仿真", id='parse-btn', n_clicks=0, color='primary',
                               style={'width': '100%', 'height': '38px', 'fontWeight': '600'}),
                    dbc.Button("纳什检验", id='nash-btn', n_clicks=0, color='warning',
                               style={'width': '100%', 'height': '38px', 'fontWeight': '600',
                                      'marginTop': '8px'}),
                ], md=2),
            ]),
        ]),
    ], className='mb-4', style={'borderRadius': '16px', 'border': 'none',
                                'boxShadow': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)'}),

    html.Div(id='nl-result', children="就绪",
             style={'marginBottom': '18px', 'color': C_MUTED, 'textAlign': 'center',
                    'fontSize': '13px', 'fontWeight': '500'}),

    # -- KPI --
    dcc.Loading(
        id="loading-kpi",
        type="default",
        children=html.Div(id='kpi-cards', children=_placeholder_kpi(), className='mb-4'),
    ),

    # -- 图表区 --
    dbc.Card(dbc.CardBody([
        html.H6("网络拓扑", className='fw-bold text-secondary mb-2'),
        dcc.Graph(id='topology-graph', figure=_placeholder_fig("拓扑图 -- 等待仿真"),
                  config={'displayModeBar': 'hover'}),
    ]), className='mb-4 chart-card'),

    dbc.Card(dbc.CardBody([
        html.H6("节点边际电价 (LMP)", className='fw-bold text-secondary mb-2'),
        dcc.Graph(id='lmp-graph', figure=_placeholder_fig("节点电价曲线 -- 等待仿真"),
                  config={'displayModeBar': 'hover'}),
    ]), className='mb-4 chart-card'),

    dbc.Card(dbc.CardBody([
        html.H6("可再生能源出力", className='fw-bold text-secondary mb-2'),
        dcc.Graph(id='re-gen-graph', figure=_placeholder_fig("可再生出力 -- 等待仿真"),
                  config={'displayModeBar': 'hover'}),
    ]), className='mb-4 chart-card'),

    dbc.Card(dbc.CardBody([
        html.H6("负荷曲线", className='fw-bold text-secondary mb-2'),
        dcc.Graph(id='load-graph', figure=_placeholder_fig("负荷曲线 -- 等待仿真"),
                  config={'displayModeBar': 'hover'}),
    ]), className='mb-4 chart-card'),

    dbc.Card(dbc.CardBody([
        html.H6("购售电功率", className='fw-bold text-secondary mb-2'),
        dcc.Graph(id='trade-graph', figure=_placeholder_fig("购售电曲线 -- 等待仿真"),
                  config={'displayModeBar': 'hover'}),
    ]), className='mb-4 chart-card'),

    dbc.Card(dbc.CardBody([
        html.H6("储能 SOC", className='fw-bold text-secondary mb-2'),
        dcc.Graph(id='soc-graph', figure=_placeholder_fig("储能 SOC -- 等待仿真"),
                  config={'displayModeBar': 'hover'}),
    ]), className='mb-4 chart-card'),

    # -- AI 分析卡片 --
    dbc.Card(dbc.CardBody([
        html.H6("AI 分析", className='fw-bold mb-2', style={'color': C_PRIMARY}),
        html.Div(id='ai-output', children=_placeholder_text("仿真完成后自动生成分析")),
    ]), className='mb-4 chart-card'),

    # -- 纳什均衡检验卡片 --
    dbc.Card(dbc.CardBody([
        html.H6("纳什均衡检验", className='fw-bold mb-2', style={'color': C_WARNING}),
        dcc.Loading(
            id="loading-nash",
            type="default",
            children=html.Div(id='nash-output', children=_placeholder_text("运行仿真后可点击「纳什检验」")),
        ),
    ]), className='mb-4 chart-card'),

    # -- 结算表格卡片 --
    dbc.Card(dbc.CardBody([
        html.H6("各节点结算结果 (CNY)", className='fw-bold text-secondary mb-3'),
        html.Div(id='payment-table', children=_placeholder_text("暂无结算数据")),
    ]), className='mb-4 chart-card'),

    # -- 结果区 --
    dcc.Store(id='sim-state', storage_type='memory'),


], fluid=True, style={'backgroundColor': C_BG, 'minHeight': '100vh', 'padding': '24px',
                       'fontFamily': '"Inter", "Segoe UI", "PingFang SC", Arial, sans-serif'})

# ------------------------------
# 场景快速选择回调
# ------------------------------
SCENARIO_BUTTON_MAP = {
    'scenario-btn-baseline': 'baseline',
    'scenario-btn-high_re': 'high_re',
    'scenario-btn-peak_load': 'peak_load',
    'scenario-btn-congestion': 'congestion',
}

@app.callback(
    [Output('kpi-cards', 'children', allow_duplicate=True),
     Output('topology-graph', 'figure', allow_duplicate=True),
     Output('lmp-graph', 'figure', allow_duplicate=True),
     Output('trade-graph', 'figure', allow_duplicate=True),
     Output('soc-graph', 'figure', allow_duplicate=True),
     Output('re-gen-graph', 'figure', allow_duplicate=True),
     Output('load-graph', 'figure', allow_duplicate=True),
     Output('payment-table', 'children', allow_duplicate=True),
     Output('nl-result', 'children', allow_duplicate=True),
     Output('ai-output', 'children', allow_duplicate=True),
     Output('sim-state', 'data', allow_duplicate=True)],
    [Input('scenario-btn-baseline', 'n_clicks'),
     Input('scenario-btn-high_re', 'n_clicks'),
     Input('scenario-btn-peak_load', 'n_clicks'),
     Input('scenario-btn-congestion', 'n_clicks')],
    prevent_initial_call=True
)
def scenario_quick_select_callback(b_n, hr_n, pl_n, cg_n):
    triggered = ctx.triggered_id
    if triggered is None or triggered not in SCENARIO_BUTTON_MAP:
        raise PreventUpdate
    scenario_en = SCENARIO_BUTTON_MAP[triggered]
    print(f"快速选择场景: {scenario_en}")
    return build_simulation_outputs(scenario_en, opf_mode="socp")


# ------------------------------
# 主回调 (NL 输入)
# ------------------------------
@app.callback(
    [Output('kpi-cards', 'children', allow_duplicate=True),
     Output('topology-graph', 'figure', allow_duplicate=True),
     Output('lmp-graph', 'figure', allow_duplicate=True),
     Output('trade-graph', 'figure', allow_duplicate=True),
     Output('soc-graph', 'figure', allow_duplicate=True),
     Output('re-gen-graph', 'figure', allow_duplicate=True),
     Output('load-graph', 'figure', allow_duplicate=True),
     Output('payment-table', 'children', allow_duplicate=True),
     Output('nl-result', 'children', allow_duplicate=True),
     Output('ai-output', 'children', allow_duplicate=True),
     Output('sim-state', 'data', allow_duplicate=True)],
    [Input('parse-btn', 'n_clicks')],
    [State('nl-input', 'value')],
    prevent_initial_call=True
)
def main_callback(parse_clicks, nl_text):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white", paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc', height=200,
    )
    empty_fig.add_annotation(text="等待输入场景描述...", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="#94a3b8", size=16))
    empty_table = html.Div("暂无结算数据", style={'color': '#94a3b8', 'textAlign': 'center',
                                                   'padding': '32px', 'fontSize': '14px'})
    empty_kpi = dbc.Alert(
        "输入自然语言描述的场景并点击「运行仿真」",
        color='light', className='text-center',
        style={'color': '#94a3b8', 'borderRadius': '14px'}
    )

    if not nl_text or not nl_text.strip():
        return (empty_kpi, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_table,
                "请输入场景描述后点击运行仿真", "", None)

    advisor = LLMAdvisor()
    parsed = advisor.parse_natural_language_to_config(nl_text.strip())
    base_scenario = parsed.get("base_scenario", "baseline")
    gp = parsed.get("global_params", {})
    T = int(gp.get("T", 96))
    strategy_nl = "rl"
    # Count all changes from user input
    agent_mod_count = len(parsed.get("agent_modifications", []))
    defaults_ov_count = len(parsed.get("defaults_overrides", {}))
    scenario_ov_count = len(parsed.get("scenario_overrides", {}))
    default_gp = {"T": 96, "load_factor": 1.0, "line_capacity_factor": 1.0,
                  "penalty_unserved": 5000.0,
                  "pv_factor": 1.0, "wind_factor": 1.0}
    param_changes = sum(1 for k, v in gp.items() if k in default_gp and v != default_gp[k])
    total_changes = param_changes + agent_mod_count + defaults_ov_count + scenario_ov_count
    nl_msg = (f"场景: {base_scenario} | "
              f"参数变更: {param_changes}项 | "
              f"Agent修改: {agent_mod_count}项 | "
              f"合计: {total_changes}项变更")
    print(nl_msg)

    try:
        base_agents, _ = get_scenario(base_scenario, T=T)
    except ValueError:
        base_agents, _ = get_scenario("baseline", T=T)
        nl_msg += " (未知场景，回退到基线)"

    agents, config = advisor.apply_llm_config_to_agents(parsed, base_agents, T)

    da_actions = adaptive_bidding(agents, config, strategy=strategy_nl, T=T)
    da_results = clear_market(agents, T, "DA", da_actions, config)
    rt_actions = adaptive_bidding(agents, config, strategy=strategy_nl, T=T)
    rt_results = clear_market(agents, T, "RT", rt_actions, config)
    payment = two_settlement(agents, da_results, rt_results)

    net = build_base_network(config)
    kpi = create_kpi_cards(da_results, rt_results, payment, agents, config)
    agents_info = [{"bus": a.bus, "name": a.name, "is_prosumer": a.is_prosumer} for a in agents]
    topo_fig = create_topology_figure(net, agents_info=agents_info, lmp_arr=da_results['lmp'])
    lmp_fig = create_lmp_figure(da_results['lmp'], "日前节点边际电价 (LMP)")
    trade_fig = create_trade_figure(da_results, agents)
    soc_fig = create_soc_figure(da_results, agents)
    re_gen_fig = create_re_gen_figure(da_results, agents)
    load_fig = create_load_figure(da_results, agents)
    pay_tab = create_payment_table(payment, agents)

    summary = build_insight_summary(da_results, agents, base_scenario)
    ai_output = _render_ai_sections(advisor.get_insight(summary))

    sim_state = {
        "scenario": base_scenario,
        "strategy": strategy_nl,
        "opf_mode": config.opf_mode,
        "T": T,
    }

    return (kpi, topo_fig, lmp_fig, trade_fig, soc_fig, re_gen_fig, load_fig, pay_tab, nl_msg, ai_output, sim_state)


# ------------------------------
# Nash equilibrium test callback
# ------------------------------
@app.callback(
    Output('nash-output', 'children'),
    Input('nash-btn', 'n_clicks'),
    State('sim-state', 'data'),
    prevent_initial_call=True
)
def nash_callback(n_clicks, sim_state):
    if not n_clicks:
        raise PreventUpdate

    if not sim_state:
        return dbc.Alert("请先运行仿真，然后再点击纳什检验", color='warning',
                         style={'borderRadius': '12px'})

    scenario_en = sim_state["scenario"]
    strategy = sim_state["strategy"]
    opf_mode = sim_state["opf_mode"]
    T = sim_state.get("T", 96)

    from models import MarketConfig
    config = MarketConfig(opf_mode=opf_mode, verbose=False)
    agents, _ = get_scenario(scenario_en, T=T, config=config)
    actions = adaptive_bidding(agents, config, strategy=strategy)

    tester = NashEquilibriumTester(agents, config, T=T, parallel=False,
                                   use_optimization=False)
    is_nash, improvements = tester.test_nash_equilibrium(actions)

    rows = []
    for name, imp in improvements.items():
        gain = imp["gain"]
        profitable = imp["profitable"]
        badge_color = "danger" if profitable else "success"
        status_text = "可获利" if profitable else "均衡"
        status_color = "#ef4444" if profitable else "#10b981"

        rows.append(html.Tr([
            html.Td(name, style={'fontWeight': '600', 'color': '#1e293b'}),
            html.Td(f"{imp['base_payoff']:,.0f}", style={
                'fontFamily': '"JetBrains Mono", monospace', 'textAlign': 'right'}),
            html.Td(f"{imp['best_payoff']:,.0f}", style={
                'fontFamily': '"JetBrains Mono", monospace', 'textAlign': 'right'}),
            html.Td(dbc.Badge(
                f"{gain:+,.0f} ({imp['rel_gain']:+.3f})",
                color=badge_color,
                className='border',
                style={'fontSize': '12px', 'fontFamily': '"JetBrains Mono", monospace'}
            )),
            html.Td(status_text, style={'color': status_color, 'fontWeight': '600'}),
        ]))

    # Summary bar
    n_total = len(improvements)
    n_profitable = sum(1 for imp in improvements.values() if imp["profitable"])
    status_color = "#10b981" if is_nash else "#ef4444"
    status_text = "纳什均衡已达成" if is_nash else f"未达均衡（{n_profitable}/{n_total} Agent 存在获利偏离）"
    status_icon = "✔" if is_nash else "✖"

    result = html.Div([
        html.Div([
            html.Span(status_icon, style={
                'fontSize': '22px', 'marginRight': '10px',
                'color': status_color, 'fontWeight': '700'
            }),
            html.Span(status_text, style={
                'color': status_color, 'fontWeight': '600', 'fontSize': '15px'
            }),
        ], style={'marginBottom': '16px'}),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Agent"), html.Th("当前收益", style={'textAlign': 'right'}),
                html.Th("最优收益", style={'textAlign': 'right'}),
                html.Th("偏离增益 (相对)"), html.Th("判定"),
            ])),
            html.Tbody(rows),
        ], bordered=True, hover=True, responsive=True, size='sm',
           style={'fontSize': '13px'}),
    ])

    return result


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
        <style>
            *, *::before, *::after { box-sizing: border-box; }
            body {
                background-color: #f0f2f5; margin: 0;
                font-family: "Inter", "Segoe UI", "PingFang SC", Arial, sans-serif;
                -webkit-font-smoothing: antialiased;
            }

            .chart-card {
                transition: box-shadow 0.2s ease;
            }
            .chart-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04) !important;
            }

            .kpi-card {
                background: #ffffff;
                padding: 18px 20px;
                border-radius: 14px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
                text-align: center;
                min-width: 140px;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .kpi-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.10);
            }

            #nl-input:focus, #nl-input:focus-within {
                border-color: #4f46e5 !important;
                box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
            }

            #parse-btn:hover {
                background-color: #4338ca !important;
                box-shadow: 0 4px 12px rgba(79,70,229,0.35);
            }
            #parse-btn:active {
                transform: scale(0.97);
            }

            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

            .modebar { opacity: 0.3; transition: opacity 0.2s; }
            .chart-card:hover .modebar { opacity: 1; }
        </style>
    </head>
    <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=8056)
