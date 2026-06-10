# dashboard.py
"""
配电网电力市场仿真仪表板
集成：场景切换 / OPF模式 / 策略切换 / 纳什检验 / 伪实时仿真 /
      自然语言解析 / 自动 AI 分析
运行: python dashboard.py
"""

import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import time
import threading
from collections import deque

from models import MarketConfig
from grid import build_base_network, day_ahead_price_china
from dispatch import solve_opf_gurobi, StorageConstraints
from market import adaptive_bidding, two_settlement, clear_market
from scenarios import get_scenario
from llm import LLMAdvisor

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

# ------------------------------
# 全局伪实时状态
# ------------------------------
class RealtimeState:
    def __init__(self):
        self.running = False
        self.t = 0
        self.agents = None
        self.config = None
        self.net = None
        self.wholesale = None
        self.action_params = None
        self.prev_soc = {}
        self.prev_power = {}
        self.lmp_history = []
        self.welfare_acc = 0.0
        self.lock = threading.Lock()
        self.total_T = 96

realtime_state = RealtimeState()

# ------------------------------
# 静态分析函数
# ------------------------------
def run_static_analysis(scenario_en, strategy="random", opf_mode="lindistflow"):
    config = MarketConfig(opf_mode=opf_mode, verbose=False)
    T = 96
    agents, _ = get_scenario(scenario_en, T=T)
    da_actions = adaptive_bidding(agents, config, strategy=strategy)
    da_results = clear_market(agents, T, "DA", da_actions, config)
    rt_actions = adaptive_bidding(agents, config, strategy=strategy)
    rt_results = clear_market(agents, T, "RT", rt_actions, config)
    payment = two_settlement(agents, da_results, rt_results)
    return agents, config, da_results, rt_results, payment, da_actions

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

    bus_lmp = {}
    if lmp_arr is not None and agents_info is not None:
        for b in range(33):
            bus_lmp[b] = float(np.mean(lmp_arr[:, b])) if b < lmp_arr.shape[1] else 0.0

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

    node_x, node_y, node_color, node_size, hover_text, labels = [], [], [], [], [], []
    for b in range(33):
        x, y = NODE_COORDS[b]
        node_x.append(x)
        node_y.append(y)
        labels.append(str(b + 1))

        if bus_lmp:
            val = bus_lmp[b]
            node_color.append(val)
            node_size.append(18 + min(32, val / 15))
        else:
            node_color.append("#6366f1")
            node_size.append(16)

        agents_on = bus_agents[b]
        txt = f"<b>节点 {b+1}</b>"
        if bus_lmp:
            txt += f"<br>LMP: {bus_lmp[b]:.2f} CNY/MWh"
        if agents_on:
            txt += f"<br>Agent: {', '.join(agents_on[:2])}"
            if len(agents_on) > 2:
                txt += f" +{len(agents_on)}"
        hover_text.append(txt)

    marker_dict = dict(
        size=node_size,
        line=dict(width=2, color='#d0d7e3'),
    )
    if bus_lmp:
        marker_dict = {**marker_dict,
            "color": node_color,
            "colorscale": "Plasma",
            "colorbar": dict(
                title=dict(text="LMP<br>(CNY/MWh)", side="right"),
                x=1.02,
                thickness=14,
                tickfont=dict(color=C_TEXT),
            ),
            "cmin": 200,
            "cmax": 800,
        }
    else:
        marker_dict["color"] = node_color

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=marker_dict,
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
                text=[f"{b+1}" for b in prosumer_buses],
                hovertemplate="产消者节点 %{text}<extra></extra>"
            ))

    fig.update_layout(
        title=dict(
            text="IEEE 33 节点配电系统拓扑 (颜色 = LMP, 星标 = 产消者)",
            font=dict(size=15, color='#2c3e50')
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        template="plotly_white",
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_CARD,
        margin=dict(l=20, r=90, t=60, b=20),
        height=400,
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
    hours = np.arange(T) * 0.25
    mean_lmp = lmp_matrix.mean(axis=1)
    fig = go.Figure()
    for b in range(n):
        fig.add_trace(go.Scatter(x=hours, y=lmp_matrix[:, b], mode='lines',
                                 line=dict(color="#e2e8f0", width=0.5), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=hours, y=mean_lmp, mode='lines', name='节点均价',
                             line=dict(color='#4f46e5', width=2.5)))
    fig.update_layout(title=title, xaxis_title="时间 (h)", yaxis_title="LMP (CNY/MWh)",
                      template="plotly_white", legend=dict(orientation='h', y=1.1),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor="#fafbfc")
    return fig

def create_trade_figure(da_results, agents):
    T = len(next(iter(da_results["schedules"].values()))["p_buy"])
    hours = np.arange(T) * 0.25
    buy = np.zeros(T); sell = np.zeros(T)
    for a in agents:
        s = da_results["schedules"][a.name]
        buy += s["p_buy"]; sell += s["p_sell"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=hours, y=buy, mode='lines', name='总购电量 (MW)',
                             line=dict(color='#ef4444', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=hours, y=sell, mode='lines', name='总售电量 (MW)',
                             line=dict(color='#10b981', width=2)), secondary_y=False)
    fig.update_layout(title="日前市场总购售电功率", xaxis_title="时间 (h)", yaxis_title="功率 (MW)",
                      template="plotly_white", hovermode="x unified", legend=dict(orientation='h', y=1.1),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc')
    return fig

def create_soc_figure(da_results, agents):
    storage_agents = [a for a in agents if a.storage is not None]
    if not storage_agents:
        return go.Figure().update_layout(title="无储能设备", template="plotly_white")
    T = len(da_results["schedules"][storage_agents[0].name]["soc"])
    hours = np.arange(T) * 0.25
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("储能 SOC", "充放电功率"))
    for a in storage_agents:
        s = da_results["schedules"][a.name]
        fig.add_trace(go.Scatter(x=hours, y=s["soc"]*100, mode='lines',
                                 name=f"{a.name} SOC", line=dict(width=2)), row=1, col=1)
        fig.add_trace(go.Bar(x=hours, y=s["p_ch"], name=f"{a.name} 充电",
                             marker_color='#3b82f6', opacity=0.75), row=2, col=1)
        fig.add_trace(go.Bar(x=hours, y=-s["p_dis"], name=f"{a.name} 放电",
                             marker_color='#f59e0b', opacity=0.75), row=2, col=1)
    fig.update_layout(barmode='overlay', template="plotly_white", hovermode="x unified",
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc')
    fig.update_yaxes(title_text="SOC (%)", row=1, col=1)
    fig.update_yaxes(title_text="功率 (MW)", row=2, col=1)
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

    if config is not None and config.use_constraint_multi_obj:
        sp = da_results.get('shadow_prices', {})
        sc = sp.get('carbon_cap', None)
        sr = sp.get('re_min_rate', None)
        constraint_items = [
            ("碳上限", f"<= {config.carbon_cap_tco2:.0f} tCO2", "#ef4444",
             f"影子价格: {sc:.1f} CNY/tCO2" if sc is not None and abs(sc) > 1e-6 else "未绑定"),
            ("可再生下限", f">= {config.re_min_rate:.0f}%", "#10b981",
             f"影子价格: {abs(sr):.1f} CNY/%" if sr is not None and abs(sr) > 1e-6 else "未绑定"),
        ]
        for label, value, accent, extra in constraint_items:
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
                        html.Div(extra, style={
                            'fontSize': '11px', 'color': '#94a3b8', 'marginTop': '4px'
                        }),
                    ], body=True, style={
                        'backgroundColor': '#f8fafc', 'borderRadius': '14px',
                        'boxShadow': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
                        'borderTop': f'3px solid {accent}',
                        'border': '1px dashed #e2e8f0',
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
# 伪实时线程
# ------------------------------
def run_realtime_thread(scenario_en, opf_mode, strategy, step_sec=0.5):
    global realtime_state
    with realtime_state.lock:
        if realtime_state.running: return
        realtime_state.running = True
        realtime_state.t = 0
        realtime_state.lmp_history = []
        realtime_state.welfare_acc = 0.0
        realtime_state.prev_soc = {}
        realtime_state.prev_power = {}

    try:
        config = MarketConfig(opf_mode=opf_mode, verbose=False)
        agents, _ = get_scenario(scenario_en, T=96)
        net = build_base_network(config)
        wholesale = day_ahead_price_china(96)
        action_params = adaptive_bidding(agents, config, strategy=strategy)

        with realtime_state.lock:
            realtime_state.agents = agents
            realtime_state.config = config
            realtime_state.net = net
            realtime_state.wholesale = wholesale
            realtime_state.action_params = action_params
            realtime_state.total_T = 96

        fallback_lmp = np.zeros(len(net.bus))
        for t in range(96):
            with realtime_state.lock:
                if not realtime_state.running: break

            try:
                success, lmp_t, welfare_t, agent_res, p_grid = solve_opf_gurobi(
                    net, agents, t, "RT", realtime_state.prev_soc,
                    wholesale[t], action_params, config, realtime_state.prev_power)
                if not success:
                    if realtime_state.lmp_history:
                        lmp_t = realtime_state.lmp_history[-1]
                    else:
                        lmp_t = fallback_lmp
                else:
                    fallback_lmp = lmp_t.copy()
                    realtime_state.welfare_acc += welfare_t
                    for a in agents:
                        if a.storage is None: continue
                        name = a.name
                        res = agent_res[name]
                        soc0 = realtime_state.prev_soc.get(name, a.storage.soc0)
                        ch, dis, new_soc = StorageConstraints.execute_dispatch(a.storage, soc0, res['p_ch'], res['p_dis'])
                        realtime_state.prev_soc[name] = new_soc
                        realtime_state.prev_power[name] = (ch, dis)

                if not isinstance(lmp_t, np.ndarray) or lmp_t.ndim != 1 or len(lmp_t) != len(net.bus):
                    if realtime_state.lmp_history:
                        lmp_t = realtime_state.lmp_history[-1]
                    else:
                        lmp_t = fallback_lmp
                realtime_state.lmp_history.append(lmp_t.copy())
            except Exception as e:
                print(f"伪实时时段 {t} 出错: {e}")
                if realtime_state.lmp_history:
                    lmp_t = realtime_state.lmp_history[-1]
                else:
                    lmp_t = fallback_lmp
                realtime_state.lmp_history.append(lmp_t.copy())
            time.sleep(step_sec)
    except Exception as e:
        print(f"伪实时线程严重错误: {e}")
    finally:
        with realtime_state.lock:
            realtime_state.running = False

# ------------------------------
# AI 分析曲线特征提取
# ------------------------------
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
        plot_bgcolor='#f8fafc', height=280,
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

app.layout = dbc.Container([
    # -- header --
    dbc.Row(dbc.Col(
        html.H1("配电网电力市场仿真仪表板",
                style={'fontSize': '26px', 'fontWeight': '700', 'color': C_TEXT}),
        className='text-center py-3'
    )),

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

    # -- 结算表格卡片 --
    dbc.Card(dbc.CardBody([
        html.H6("各节点结算结果 (CNY)", className='fw-bold text-secondary mb-3'),
        html.Div(id='payment-table', children=_placeholder_text("暂无结算数据")),
    ]), className='mb-4 chart-card'),

    # -- 结果区 --


], fluid=True, style={'backgroundColor': C_BG, 'minHeight': '100vh', 'padding': '24px',
                       'fontFamily': '"Inter", "Segoe UI", "PingFang SC", Arial, sans-serif'})

# ------------------------------
# 主回调
# ------------------------------
@app.callback(
    [Output('kpi-cards', 'children'),
     Output('topology-graph', 'figure'),
     Output('lmp-graph', 'figure'),
     Output('trade-graph', 'figure'),
     Output('soc-graph', 'figure'),
     Output('payment-table', 'children'),
     Output('nl-result', 'children'),
     Output('ai-output', 'children')],
    [Input('parse-btn', 'n_clicks')],
    [State('nl-input', 'value')]
)
def main_callback(parse_clicks, nl_text):
    if parse_clicks is None or parse_clicks == 0:
        raise PreventUpdate

    nl_msg = ""
    ai_output = ""

    if not nl_text or not nl_text.strip():
        nl_msg = "请输入场景描述后点击运行仿真"
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
        return (empty_kpi, empty_fig, empty_fig, empty_fig, empty_fig, empty_table, nl_msg, "")

    advisor = LLMAdvisor()
    parsed = advisor.parse_natural_language_to_config(nl_text.strip())
    base_scenario = parsed.get("base_scenario", "baseline")
    description = parsed.get("description", "")
    gp = parsed.get("global_params", {})
    T = int(gp.get("T", 96))
    strategy_nl = "mpc"
    # Count all changes from user input
    agent_mod_count = len(parsed.get("agent_modifications", []))
    defaults_ov_count = len(parsed.get("defaults_overrides", {}))
    scenario_ov_count = len(parsed.get("scenario_overrides", {}))
    # Count non-default global_params
    default_gp = {"T": 96, "load_factor": 1.0, "line_capacity_factor": 1.0,
                  "penalty_unserved": 800.0, "carbon_cap_tco2": 200.0,
                  "re_min_rate": 95.0, "lambda_carbon": 50.0, "lambda_re": 50.0,
                  "lambda_curtail": 15.0, "pv_factor": 1.0, "wind_factor": 1.0}
    param_changes = sum(1 for k, v in gp.items() if k in default_gp and v != default_gp[k])
    total_changes = param_changes + agent_mod_count + defaults_ov_count + scenario_ov_count
    scenario_label = base_scenario if base_scenario != "baseline" else "基线"
    nl_msg = (f"场景: {scenario_label} | "
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

    da_actions = adaptive_bidding(agents, config, strategy=strategy_nl)
    da_results = clear_market(agents, T, "DA", da_actions, config)
    rt_actions = adaptive_bidding(agents, config, strategy=strategy_nl)
    rt_results = clear_market(agents, T, "RT", rt_actions, config)
    payment = two_settlement(agents, da_results, rt_results)

    # 构建图表
    net = build_base_network(config)
    kpi = create_kpi_cards(da_results, rt_results, payment, agents, config)
    agents_info = [{"bus": a.bus, "name": a.name, "is_prosumer": a.is_prosumer} for a in agents]
    topo_fig = create_topology_figure(net, agents_info=agents_info, lmp_arr=da_results['lmp'])
    lmp_fig = create_lmp_figure(da_results['lmp'], "日前节点边际电价 (LMP)")
    trade_fig = create_trade_figure(da_results, agents)
    soc_fig = create_soc_figure(da_results, agents)
    pay_tab = create_payment_table(payment, agents)

    # AI 分析
    advisor2 = LLMAdvisor()
    summary = build_insight_summary(da_results, agents, base_scenario)
    ai_output = advisor2.get_insight(summary)

    return (kpi, topo_fig, lmp_fig, trade_fig, soc_fig, pay_tab, nl_msg, ai_output)


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
    app.run(debug=True, port=8050)
