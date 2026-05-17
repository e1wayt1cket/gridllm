# dashboard.py
"""
配电网电力市场仿真仪表板 (深色主题 · 每个图单占一行)
集成：场景切换 / OPF模式 / 策略切换 / 纳什检验 / 伪实时仿真 /
       自然语言解析 / 自动 AI 分析（曲线原因与建议 ≤200字）
运行: python dashboard.py
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import time
import threading
from collections import deque
from collections import deque
# 导入项目模块
from models import MarketConfig
from grid import build_base_network, day_ahead_price_china
from dispatch import solve_opf_gurobi, StorageConstraints
from market import random_actions, adaptive_bidding, two_settlement, clear_market
from nash import NashEquilibriumTester
from scenarios import get_scenario
from llm import LLMAdvisor

# ------------------------------
# 中文场景名映射（精简版）
# ------------------------------
SCENARIO_NAMES_CN = {
    "baseline":        "基准–风光储",
    "high_re":         "高可再生渗透",
    "peak_load":       "高峰负荷",
    "congestion":      "线路阻塞",
    "re_ramp_drop":    "新能源骤降",
    "re_ramp_surge":   "新能源骤升",
}
CN_TO_EN = {v: k for k, v in SCENARIO_NAMES_CN.items()}

# 策略名称映射
STRATEGY_MAP = {
    "随机": "random",
    "最佳响应": "best_response",
}

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
# 静态分析函数（使用指定策略）
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
# 绘图辅助函数
# ------------------------------
from collections import deque

def create_topology_figure(net, agents_info=None, lmp_arr=None):
    """绘制 IEEE 33 节点拓扑 (严格水平/竖直，符合标准结构)"""
    
    # 标准IEEE 33节点坐标 (1-based → 0-based)
    # 主馈线: 1-2-3-...-18 (水平)
    # 分支1: 2-19-20-21-22 (从节点2向上)
    # 分支2: 3-23-24-25 (从节点3向下)
    # 分支3: 6-26-27-...-33 (从节点6向下)
    NODE_COORDS = {
        0: (0, 0),     # Bus 1 (根节点/Grid)
        1: (2, 0),     # Bus 2
        2: (4, 0),     # Bus 3
        3: (6, 0),     # Bus 4
        4: (8, 0),     # Bus 5
        5: (10, 0),    # Bus 6
        6: (12, 0),    # Bus 7
        7: (14, 0),    # Bus 8
        8: (16, 0),    # Bus 9
        9: (18, 0),    # Bus 10
        10: (20, 0),   # Bus 11
        11: (22, 0),   # Bus 12
        12: (24, 0),   # Bus 13
        13: (26, 0),   # Bus 14
        14: (28, 0),   # Bus 15
        15: (30, 0),   # Bus 16
        16: (32, 0),   # Bus 17
        17: (34, 0),   # Bus 18
        # 分支1: 从Bus 2向上 (y=3)
        18: (2, 3),    # Bus 19
        19: (4, 3),    # Bus 20
        20: (6, 3),    # Bus 21
        21: (8, 3),    # Bus 22
        # 分支2: 从Bus 3向下 (y=-3)
        22: (4, -3),   # Bus 23
        23: (6, -3),   # Bus 24
        24: (8, -3),   # Bus 25
        # 分支3: 从Bus 6向下 (y=-3)
        25: (10, -3),  # Bus 26
        26: (12, -3),  # Bus 27
        27: (14, -3),  # Bus 28
        28: (16, -3),  # Bus 29
        29: (18, -3),  # Bus 30
        30: (20, -3),  # Bus 31
        31: (22, -3),  # Bus 32
        32: (24, -3),  # Bus 33
    }
    
    # 标准连线 (0-based)
    LINES = [
        # 主馈线
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10),
        (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), (16,17),
        # 分支1 (从节点2=索引1)
        (1,18), (18,19), (19,20), (20,21),
        # 分支2 (从节点3=索引2)
        (2,22), (22,23), (23,24),
        # 分支3 (从节点6=索引5)
        (5,25), (25,26), (26,27), (27,28), (28,29), (29,30), (30,31), (31,32),
    ]
    
    # 配色
    C_TEXT = "#2c3e50"
    C_CARD = "#ffffff"
    C_BORDER = "#d0d7e3"
    
    # 计算LMP颜色 (如果提供)
    bus_lmp = {}
    if lmp_arr is not None and agents_info is not None:
        for b in range(33):
            bus_lmp[b] = float(np.mean(lmp_arr[:, b])) if b < lmp_arr.shape[1] else 0.0
    
    # 收集每个bus的agent
    bus_agents = {b: [] for b in range(33)}
    if agents_info:
        for info in agents_info:
            bus_agents[info["bus"]].append(info["name"])
    
    fig = go.Figure()
    
    # 绘制连线 (严格正交)
    for i, j in LINES:
        x0, y0 = NODE_COORDS[i]
        x1, y1 = NODE_COORDS[j]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=C_BORDER, width=2.5),
            hoverinfo="skip", showlegend=False
        ))
    
    # 绘制节点
    node_x, node_y, node_color, node_size, hover_text, labels = [], [], [], [], [], []
    for b in range(33):
        x, y = NODE_COORDS[b]
        node_x.append(x)
        node_y.append(y)
        labels.append(str(b + 1))
        
        # 节点颜色: 基于LMP或默认蓝色
        if bus_lmp:
            val = bus_lmp[b]
            node_color.append(val)
            node_size.append(18 + min(32, val / 15))
        else:
            node_color.append("#3498db")
            node_size.append(16)
        
        # 悬停文本
        agents_on = bus_agents[b]
        txt = f"<b>节点 {b+1}</b>"
        if bus_lmp:
            txt += f"<br>LMP: {bus_lmp[b]:.2f} ¥/MWh"
        if agents_on:
            txt += f"<br>Agent: {', '.join(agents_on[:2])}"
            if len(agents_on) > 2:
                txt += f" 等{len(agents_on)}个"
        hover_text.append(txt)
    
    # 节点散点
    marker_dict = dict(
        size=node_size,
        line=dict(width=2, color='#d0d7e3'),
    )
    if bus_lmp:
        # 使用字典合并的方式替代update方法，避免类型检查错误
        marker_dict = {**marker_dict, 
            "color": node_color,
            "colorscale": "Plasma",
            "colorbar": dict(
                title="LMP<br>(¥/MWh)",
                x=1.02,
                thickness=14,
                titleside="right",
                tickfont=dict(color=C_TEXT),
                titlefont=dict(color=C_TEXT)
            ),
            "cmin": 200,   #type: ignore
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
        textfont=dict(size=10, color='#2c3e50', family='Arial Black'),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
        showlegend=False,
        name="节点"
    ))
    
    # 产消者星标
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
                    color="#f39c12",
                    line=dict(width=2, color='#d0d7e3')
                ),
                name="产消者",
                text=[f"{b+1}" for b in prosumer_buses],
                hovertemplate="产消者节点 %{text}<extra></extra>"
            ))
    
    fig.update_layout(
        title=dict(
            text="IEEE 33 节点配电系统拓扑（颜色=节点边际电价，★=产消者）",
            font=dict(size=15, color='#2c3e50')
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
            scaleanchor="x",
            scaleratio=1
        ),
        template="plotly_white",
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_CARD,
        margin=dict(l=20, r=90, t=60, b=20),
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(color=C_TEXT, size=12)
        ),
        font=dict(color=C_TEXT),
    )
    
    return fig
def create_lmp_figure(lmp_matrix, title="节点电价"):
    if lmp_matrix is None or np.all(lmp_matrix == 0):
        fig = go.Figure()
        fig.add_annotation(text="求解失败或数据异常（LMP全零）", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#e74c3c", size=16))
        fig.update_layout(title=title, template="plotly_white",
                          paper_bgcolor='#ffffff', plot_bgcolor='#fafbfc')
        return fig
    T, n = lmp_matrix.shape
    hours = np.arange(T) * 0.25
    mean_lmp = lmp_matrix.mean(axis=1)
    fig = go.Figure()
    for b in range(n):
        fig.add_trace(go.Scatter(x=hours, y=lmp_matrix[:, b], mode='lines',
                                 line=dict(color="#bdc3c7", width=0.6), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=hours, y=mean_lmp, mode='lines', name='节点均价',
                             line=dict(color='#e67e22', width=3)))
    fig.update_layout(title=title, xaxis_title="时间 (h)", yaxis_title="电价 (¥/MWh)",
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
                             line=dict(color='#e74c3c', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=hours, y=sell, mode='lines', name='总售电量 (MW)',
                             line=dict(color='#27ae60', width=2)), secondary_y=False)
    fig.update_layout(title="日前市场总买卖功率", xaxis_title="时间 (h)", yaxis_title="功率 (MW)",
                      template="plotly_white", hovermode="x unified", legend=dict(orientation='h', y=1.1),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor='#fafbfc')
    return fig

def create_soc_figure(da_results, agents):
    storage_agents = [a for a in agents if a.storage is not None]
    if not storage_agents:
        return go.Figure().update_layout(title="无储能设备", template="plotly_white")
    T = len(da_results["schedules"][storage_agents[0].name]["soc"])
    hours = np.arange(T) * 0.25
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("储能SOC", "充放电功率"))
    for a in storage_agents:
        s = da_results["schedules"][a.name]
        fig.add_trace(go.Scatter(x=hours, y=s["soc"]*100, mode='lines',
                                 name=f"{a.name} SOC", line=dict(width=2)), row=1, col=1)
        fig.add_trace(go.Bar(x=hours, y=s["p_ch"], name=f"{a.name} 充电",
                             marker_color='#3498db', opacity=0.7), row=2, col=1)
        fig.add_trace(go.Bar(x=hours, y=-s["p_dis"], name=f"{a.name} 放电",
                             marker_color='#e67e22', opacity=0.7), row=2, col=1)
    fig.update_layout(barmode='overlay', template="plotly_white", hovermode="x unified",
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor='#ffffff', plot_bgcolor='#fafbfc')
    fig.update_yaxes(title_text="SOC (%)", row=1, col=1)
    fig.update_yaxes(title_text="功率 (MW)", row=2, col=1)
    return fig

def create_kpi_cards(da_results, rt_results, payment, agents):
    load_total = sum(np.sum(a.load_forecast) for a in agents)
    served = sum(np.sum(da_results["schedules"][a.name]["served"]) for a in agents)
    satisfaction = (served / load_total * 100) if load_total > 0 else 100.0
    total_cost = sum(payment.values())

    def card(title, value_main):
        return html.Div([
            html.H3(title, style={'color': '#5a6c7d'}),
            html.Div(value_main, style={'fontSize': '20px', 'color': '#2c3e50'})
        ], className='kpi-card')

    return html.Div([
        card("日前社会福利", f"{da_results['welfare']:,.0f} ¥"),
        card("实时社会福利", f"{rt_results['welfare']:,.0f} ¥"),
        card("可再生消纳率", f"{da_results['re_consumption_rate']:.1f}%"),
        card("负荷满足率", f"{satisfaction:.1f}%"),
        card("总市场成本", f"{total_cost:,.0f} ¥"),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'})

def create_payment_table(payment, agents):
    rows = []
    for a in agents:
        val = payment[a.name]
        status = "成本" if val > 0 else "收益" if val < 0 else "平衡"
        color = "#e74c3c" if val > 0 else "#27ae60" if val < 0 else "#5a6c7d"
        rows.append(html.Tr([
            html.Td(a.name), html.Td(a.load_type),
            html.Td(f"{val:.2f}", style={'color': color, 'fontWeight': 'bold'}), html.Td(status)
        ]))
    total_pay = sum(payment.values())
    rows.append(html.Tr([html.Td("总计", style={'fontWeight': 'bold'}), html.Td(""),
                         html.Td(f"{total_pay:.2f}", style={'fontWeight': 'bold'}), html.Td("")]))
    table = html.Table([html.Thead(html.Tr([html.Th("智能体"), html.Th("类型"), html.Th("结算金额 (¥)"), html.Th("状态")])),
                        html.Tbody(rows)], style={'width': '100%', 'borderCollapse': 'collapse', 'color': '#2c3e50'})
    return table

# ------------------------------
# 伪实时线程（增强鲁棒性）
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
            realtime_state.agents = agents #type:ignore
            realtime_state.config = config      #type:ignore
            realtime_state.net = net        #type:ignore    
            realtime_state.wholesale = wholesale                #type:ignore
            realtime_state.action_params = action_params            #type:ignore
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
                    fallback_lmp = lmp_t.copy() # type:ignore
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
# 曲线特征提取（供 AI 分析使用）
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
        'storage_active': '是' if active else '否'
    }

# ------------------------------
# Dash 应用
# ------------------------------
app = dash.Dash(__name__, title="配电网电力市场仿真仪表板")
app.layout = html.Div(
    style={'backgroundColor': '#f5f7fa', 'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'minHeight': '100vh'},
    children=[
        html.H1("配电网电力市场实时仿真仪表盘", style={'textAlign': 'center', 'color': '#3498db'}),

        # 自然语言输入区
        html.Div([
            html.Label("💬 自然语言指令（示例：高光伏低负荷，阻塞严重）", style={'color': '#2c3e50'}),
            dcc.Input(id='nl-input', type='text', placeholder='输入自然语言描述...',
                      value='', style={'width': '60%', 'marginRight': '10px', 'padding': '8px',
                                       'borderRadius': '6px', 'border': '1px solid #d0d7e3',
                                       'backgroundColor': '#ffffff', 'color': '#2c3e50'}),
            html.Button("解析并运行", id='parse-btn', n_clicks=0,
                        style={'backgroundColor': '#9b59b6', 'color': 'white', 'border': 'none',
                               'borderRadius': '6px', 'padding': '8px 20px'}),
        ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': '#ffffff', 'borderRadius': '8px'}),

        # 控制面板
        html.Div([
            html.Div([
                html.Label("场景选择", style={'color': '#2c3e50'}),
                dcc.Dropdown(id='scenario-dropdown',
                             options=[{'label': v, 'value': v} for v in SCENARIO_NAMES_CN.values()],
                             value="基准–风光储", clearable=False,
                             style={'color': '#2c3e50', 'width': '220px'})
            ], style={'marginRight': '20px'}),
            html.Div([
                html.Label("OPF 模式", style={'color': '#2c3e50'}),
                dcc.Dropdown(id='opf-dropdown',
                             options=[{'label': 'DC-OPF', 'value': 'dc'}, {'label': 'LinDistFlow', 'value': 'lindistflow'}],
                             value='lindistflow', clearable=False,
                             style={'color': '#2c3e50', 'width': '140px'})
            ], style={'marginRight': '20px'}),
            html.Div([
                html.Label("报价策略", style={'color': '#2c3e50'}),
                dcc.Dropdown(id='strategy-dropdown',
                             options=[{'label': '随机', 'value': '随机'},
                                      {'label': '最佳响应', 'value': '最佳响应'}],
                             value='随机', clearable=False,
                             style={'color': '#2c3e50', 'width': '140px'})
            ], style={'marginRight': '20px'}),
            html.Div([
                html.Button("静态分析", id='static-btn', n_clicks=0,
                            style={'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                                   'borderRadius': '6px', 'padding': '8px 20px', 'marginRight': '10px'}),
                html.Button("启动伪实时", id='realtime-btn', n_clicks=0,
                            style={'backgroundColor': '#e67e22', 'color': 'white', 'border': 'none',
                                   'borderRadius': '6px', 'padding': '8px 20px', 'marginRight': '10px'}),
                html.Button("停止伪实时", id='stop-btn', n_clicks=0,
                            style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                   'borderRadius': '6px', 'padding': '8px 20px', 'marginRight': '10px'}),
                html.Button("纳什检验", id='nash-btn', n_clicks=0,
                            style={'backgroundColor': '#9b59b6', 'color': 'white', 'border': 'none',
                                   'borderRadius': '6px', 'padding': '8px 20px'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'marginBottom': '20px', 'padding': '10px',
                  'backgroundColor': '#ffffff', 'borderRadius': '8px'}),

        dcc.Interval(id='realtime-interval', interval=500, disabled=True),
        dcc.Store(id='static-data-store'),

        html.Div(id='kpi-cards'),

        dcc.Graph(id='topology-graph', style={'width': '100%', 'marginBottom': '20px'}),
        dcc.Graph(id='lmp-graph', style={'width': '100%', 'marginBottom': '20px'}),
        dcc.Graph(id='trade-graph', style={'width': '100%', 'marginBottom': '20px'}),
        dcc.Graph(id='soc-graph', style={'width': '100%', 'marginBottom': '20px'}),
        dcc.Graph(id='lmp-realtime', style={'width': '100%', 'marginBottom': '20px'}),

        html.Div(id='nash-output', style={'marginTop': '10px', 'color': '#2c3e50'}),
        # AI 自动分析（无需按钮）
        html.Div(id='ai-output', style={'marginTop': '15px', 'color': '#2c3e50',
                                        'backgroundColor': '#ffffff', 'padding': '12px',
                                        'borderRadius': '8px'}),

        html.Div([html.H3("各智能体结算结果 (¥)", style={'color': '#2c3e50'}),
                  html.Div(id='payment-table')],
                 style={'backgroundColor': '#ffffff', 'padding': '15px', 'borderRadius': '8px', 'marginTop': '20px'}),

        html.Div(id='nl-result', style={'marginTop': '10px', 'color': '#5a6c7d'})
    ]
)

# ------------------------------
# 统一主回调（含自动 AI 分析）
# ------------------------------
@app.callback(
    [Output('static-data-store', 'data'),
     Output('kpi-cards', 'children'),
     Output('topology-graph', 'figure'),
     Output('lmp-graph', 'figure'),
     Output('trade-graph', 'figure'),
     Output('soc-graph', 'figure'),
     Output('payment-table', 'children'),
     Output('realtime-interval', 'disabled'),
     Output('lmp-realtime', 'figure'),
     Output('nl-result', 'children'),
     Output('ai-output', 'children')],   # 新增 AI 输出
    [Input('static-btn', 'n_clicks'),
     Input('realtime-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('realtime-interval', 'n_intervals'),
     Input('parse-btn', 'n_clicks')],
    [State('scenario-dropdown', 'value'),
     State('opf-dropdown', 'value'),
     State('strategy-dropdown', 'value'),
     State('nl-input', 'value')]
)
def main_callback(static_clicks, realtime_clicks, stop_clicks, n_intervals,
                  parse_clicks, scenario_cn, opf_mode, strategy_cn, nl_text):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    strategy_en = STRATEGY_MAP.get(strategy_cn, "random")

    realtime_fig = go.Figure().update_layout(title="伪实时电价 (点击启动)", template="plotly_white",
                                             paper_bgcolor='#ffffff', plot_bgcolor='#fafbfc')
    nl_msg = ""
    ai_output = dash.no_update  # 默认不更新

    # 定时器刷新伪实时
    if trigger_id == 'realtime-interval':
        with realtime_state.lock:
            if not realtime_state.running or len(realtime_state.lmp_history) == 0:
                return (dash.no_update,) * 10 + (dash.no_update,)  # 11个输出
            try:
                lmp_array = np.array(realtime_state.lmp_history)
                if lmp_array.ndim != 2:
                    raise ValueError("数据形状异常")
                T, n = lmp_array.shape
            except Exception as e:
                print(f"LMP数据转换错误: {e}")
                return (dash.no_update,) * 10 + (dash.no_update,)

            hours = np.arange(T) * 0.25
            mean_lmp = lmp_array.mean(axis=1) if n > 0 else np.zeros(T)
            realtime_fig = go.Figure()
            for b in range(n):
                realtime_fig.add_trace(go.Scatter(
                    x=hours, y=lmp_array[:, b],
                    mode='lines', line=dict(color='#bdc3c7', width=0.6),
                    showlegend=False, hoverinfo='skip'))
            realtime_fig.add_trace(go.Scatter(
                x=hours, y=mean_lmp, mode='lines', name='节点均价',
                line=dict(color='#e67e22', width=3)))
            realtime_fig.update_layout(
                title=f"伪实时电价 (已仿真 {T}/{realtime_state.total_T} 时段)",
                xaxis_title="时间 (h)", yaxis_title="电价 (¥/MWh)",
                template="plotly_white", legend=dict(orientation='h', y=1.1),
                margin=dict(l=40, r=20, t=60, b=40),
                paper_bgcolor='#ffffff', plot_bgcolor='#fafbfc')
            return (dash.no_update,) * 8 + (realtime_fig, dash.no_update, dash.no_update)

    # 停止伪实时
    if trigger_id == 'stop-btn':
        with realtime_state.lock:
            realtime_state.running = False
        return (dash.no_update,) * 7 + (True, dash.no_update, dash.no_update, dash.no_update)

    # 启动伪实时
    if trigger_id == 'realtime-btn':
        with realtime_state.lock:
            if not realtime_state.running:
                thread = threading.Thread(target=run_realtime_thread,
                                          args=(CN_TO_EN.get(scenario_cn, "baseline"), opf_mode, strategy_en, 0.5))
                thread.daemon = True
                thread.start()
        realtime_fig = go.Figure().update_layout(title="伪实时已启动，等待数据...", template="plotly_white",
                                                 paper_bgcolor='#ffffff', plot_bgcolor='#fafbfc')
        return (dash.no_update,) * 7 + (False, realtime_fig, dash.no_update, dash.no_update)

    # 停止仿真（静态分析 / 自然语言解析）
    with realtime_state.lock:
        realtime_state.running = False

    # 准备静态分析
    if trigger_id in ['static-btn', 'parse-btn', 'scenario-dropdown']:
        if trigger_id == 'parse-btn' and nl_text and nl_text.strip():
            advisor = LLMAdvisor()
            parsed = advisor.parse_natural_language_to_config(nl_text.strip())
            nl_msg = f"✅ 解析结果: 场景={parsed['scenario_type']}, 参数={parsed['parameters']}"
            print(nl_msg)

            # 提取参数
            scenario_type = parsed.get("scenario_type", "baseline")
            params = parsed.get("parameters", {})
            T = int(params.get("T", 96))
            load_factor = float(params.get("load_factor", 1.0))
            re_factor = float(params.get("re_factor", 1.0))
            line_cap_factor = float(params.get("line_capacity_factor", 1.0))
            strategy_nl = params.get("strategy", "random")
            if strategy_nl not in ["random", "best_response"]:
                strategy_nl = "random"

            config = MarketConfig(opf_mode='lindistflow', verbose=False)
            config.line_capacity_multiplier = 3.0 * line_cap_factor

            try:
                agents, _ = get_scenario(scenario_type, T=T)
            except ValueError:
                agents, _ = get_scenario("baseline", T=T)
                nl_msg += " (警告: 未知场景，已替换为基准)"

            # 缩放可再生和负荷
            for a in agents:
                a.load_forecast *= load_factor
                a.load_real *= load_factor
                if a.is_prosumer:
                    a.pv_forecast *= re_factor
                    a.pv_real *= re_factor
                if a.has_wind:
                    a.wind_forecast *= re_factor #type: ignore
                    a.wind_real *= re_factor   #type: ignore

            da_actions = adaptive_bidding(agents, config, strategy=strategy_nl)
            da_results = clear_market(agents, T, "DA", da_actions, config)
            rt_actions = adaptive_bidding(agents, config, strategy=strategy_nl)
            rt_results = clear_market(agents, T, "RT", rt_actions, config)
            payment = two_settlement(agents, da_results, rt_results)

            # 静态数据
            static_data = {'agents_names': [a.name for a in agents], 'da_actions': {}}
            for a in agents:
                static_data['da_actions'][a.name] = {
                    'bid_mult': da_actions[a.name].get('bid_mult', np.ones(T)).tolist()
                    if isinstance(da_actions[a.name].get('bid_mult'), np.ndarray)
                    else da_actions[a.name].get('bid_mult', 1.0),
                    'offer_adder': da_actions[a.name].get('offer_adder', np.zeros(T)).tolist()
                    if isinstance(da_actions[a.name].get('offer_adder'), np.ndarray)
                    else da_actions[a.name].get('offer_adder', 0.0)
                }
        else:
            # 普通静态分析
            scenario_en = CN_TO_EN.get(scenario_cn, "baseline")
            agents, config, da_results, rt_results, payment, da_actions = run_static_analysis(
                scenario_en, strategy=strategy_en, opf_mode=opf_mode
            )
            static_data = {'agents_names': [a.name for a in agents], 'da_actions': {}}
            for a in agents:
                static_data['da_actions'][a.name] = {
                    'bid_mult': da_actions[a.name].get('bid_mult', np.ones(96)).tolist()
                    if isinstance(da_actions[a.name].get('bid_mult'), np.ndarray)
                    else da_actions[a.name].get('bid_mult', 1.0),
                    'offer_adder': da_actions[a.name].get('offer_adder', np.zeros(96)).tolist()
                    if isinstance(da_actions[a.name].get('offer_adder'), np.ndarray)
                    else da_actions[a.name].get('offer_adder', 0.0)
                }

        # 构建图表
        net = build_base_network(config)
        kpi = create_kpi_cards(da_results, rt_results, payment, agents)
        topo_fig = create_topology_figure(net)
        lmp_fig = create_lmp_figure(da_results['lmp'], "日前节点边际电价")
        trade_fig = create_trade_figure(da_results, agents)
        soc_fig = create_soc_figure(da_results, agents)
        pay_tab = create_payment_table(payment, agents)

        # 自动 AI 分析（≤200字）
        advisor = LLMAdvisor()
        summary = build_insight_summary(da_results, agents, scenario_cn)
        ai_output = advisor.get_insight(summary)

        return (static_data, kpi, topo_fig, lmp_fig, trade_fig, soc_fig, pay_tab,
                True, realtime_fig, nl_msg, ai_output)

    raise PreventUpdate

# ------------------------------
# 纳什检验回调（保持不变）
# ------------------------------
@app.callback(
    Output('nash-output', 'children'),
    Input('nash-btn', 'n_clicks'),
    State('static-data-store', 'data'),
    State('scenario-dropdown', 'value'),
    State('opf-dropdown', 'value')
)
def run_nash_check(n_clicks, static_data, scenario_cn, opf_mode):
    if n_clicks is None or static_data is None:
        return "请先运行静态分析"
    scenario_en = CN_TO_EN.get(scenario_cn, "baseline")
    config = MarketConfig(opf_mode=opf_mode, verbose=False)
    agents, _ = get_scenario(scenario_en, T=96)
    da_actions = {}
    for a in agents:
        if a.name in static_data['agents_names']:
            da_actions[a.name] = {
                'bid_mult': np.array(static_data['da_actions'][a.name]['bid_mult']),
                'offer_adder': np.array(static_data['da_actions'][a.name]['offer_adder'])
            }
        else:
            da_actions[a.name] = {'bid_mult': np.ones(96), 'offer_adder': np.zeros(96)}
    tester = NashEquilibriumTester(agents, config, T=96, stage="DA")
    is_nash, improvements = tester.test_nash_equilibrium(da_actions, threshold=30.0)
    if is_nash:
        return html.Span("✅ 当前策略接近纳什均衡", style={'color': '#27ae60'})
    else:
        try:
            nash_strat, iters = tester.iter_fictitious_play(da_actions, max_iter=5, num_variations=10, alpha=0.3)
            final_nash, _ = tester.test_nash_equilibrium(nash_strat, threshold=30.0)
            if final_nash:
                return html.Span(f"✅ 找到近似纳什均衡 (迭代 {iters} 次)", style={'color': '#27ae60'})
            else:
                return html.Span(f"⚠️ 未达均衡，策略已优化 (迭代 {iters} 次)", style={'color': '#e67e22'})
        except Exception as e:
            return html.Span(f"纳什检验异常: {e}", style={'color': '#e74c3c'})

# ------------------------------
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
        <style>
            body { background-color: #f5f7fa; margin: 0; }
            .kpi-card { background-color: #ffffff; padding: 15px 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); text-align: center; min-width: 140px; border-left: 6px solid #3498db; }
            .kpi-card h3 { margin-top: 0; font-size: 14px; font-weight: 600; color: #5a6c7d; }
            table { width: 100%; border-collapse: collapse; font-size: 14px; }
            th { background-color: #ebf5fb; color: #2c3e50; padding: 12px; text-align: left; }
            td { padding: 10px 12px; border-bottom: 1px solid #d0d7e3; }
            tr:hover { background-color: #ebf5fb; }
        </style>
    </head>
    <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=8050)