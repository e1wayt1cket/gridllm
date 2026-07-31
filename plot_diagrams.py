"""Generate all project diagrams: RL mode, load curves, architecture,
RL concept, and IEEE 33-bus topology."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle
import numpy as np

from topology_data import (
    NODE_COORDS, LINES, TIE_LINES,
    RESIDENTIAL_BUSES, COMMERCIAL_BUSES, INDUSTRIAL_BUSES,
    PROSUMER_RESIDENTIAL_BUSES, PROSUMER_INDUSTRIAL_BUSES,
    PROSUMER_BUSES, STANDALONE_STORAGE_BUSES,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Shared colour palette
# ---------------------------------------------------------------------------
C_BG     = "#F8FAFC"
C_BOX    = "#FFFFFF"
C_EDGE   = "#CBD5E1"
C_STATE  = "#DBEAFE"
C_ACTION = "#FEF3C7"
C_REWARD = "#DCFCE7"
C_ENV    = "#F3E8FF"
C_POLICY = "#FFEDD5"
C_ARROW  = "#64748B"
C_TEXT   = "#1E293B"
C_TITLE  = "#0F172A"
C_SUB    = "#64748B"


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def box(ax, x, y, w, h, text, color, title="", fontsize=9, title_fs=10,
        edgewidth=1.5, zorder=2, round_pad=0.15):
    """Draw a rounded box with optional title and body text."""
    rect = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={round_pad}",
                          facecolor=color, edgecolor=C_EDGE, linewidth=edgewidth, zorder=zorder)
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 0.25, title, fontsize=title_fs, fontweight="bold",
                ha="center", va="top", color=C_TEXT, zorder=zorder + 1)
    ax.text(x + w/2, y + h/2 - 0.1, text, fontsize=fontsize, ha="center", va="center",
            color=C_TEXT, zorder=zorder + 1, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, style="->", color=C_ARROW, lw=1.8, zorder=1, rad=0.0):
    """Draw a styled arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=zorder)


def label_arrow_text(ax, x, y, text, color=C_ARROW, fs=10):
    """Add a label near an arrow."""
    ax.text(x, y, text, fontsize=fs, fontweight="bold", color=color, ha="center",
            bbox=dict(facecolor="white", edgecolor=C_EDGE, pad=3, boxstyle="round"))


# ============================================================================
# Diagram 1: RL mode (Independent PPO)
# ============================================================================
def draw_rl_diagram():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    # Title
    ax.text(9, 9.5, u"强化学习模式 — 独立 PPO 多智能体竞价框架",
            fontsize=16, fontweight="bold", ha="center", color=C_TITLE)

    # Environment box (large, center-bottom)
    box(ax, 5.5, 1.0, 7.0, 3.0,
        u"日前市场出清 (DC-OPF / LinDistFlow)\n"
        u"实时滚动 MPC 出清\n"
        u"两结算制度 (DA+RT)\n"
        u"碳排放 / 弃电 / 阻塞 约束",
        C_ENV, u"电力市场仿真环境 (Market + Dispatch)", 9, 11)

    # Agent boxes (left column)
    for i, (label, ypos) in enumerate([
        (u"智能体 i\n(Bus 5, 居民产消者)\nPV + 储能", 3.0),
        (u"智能体 j\n(Bus 16, 工业产消者)\n风电 + 储能", 5.5),
        (u"智能体 k\n(Bus 23, 工业产消者)\n风光 + 储能", 8.0),
    ]):
        box(ax, 0.3, ypos, 3.2, 2.0, label, C_BOX,
            f"Agent {label.split(chr(10))[0]}", 8, 9)

    # Policy box (center-top)
    box(ax, 5.5, 6.0, 7.0, 2.5,
        u"输入: 20维观测状态\n"
        u"  · 负荷预测 (4块)   · PV+风电预测 (4块)\n"
        u"  · 历史LMP (4块)    · Nodal Price预测 (4块)\n"
        u"  · SOC / 平均LMP / 系统负荷比 / 阻塞指数 (各1)\n"
        u"输出: 24个离散动作 → (bid_mult × offer_adder)",
        C_POLICY, u"策略网络 (Independent PPO Actor-Critic)", 8, 10)

    # State box
    box(ax, 13.5, 6.5, 3.8, 1.8,
        u"20维观测向量\n负荷 / 可再生 / 价格\nSOC / 阻塞",
        C_STATE, u"状态 (Observation)", 8.5, 10)

    # Action box
    box(ax, 13.5, 3.5, 3.8, 1.8,
        "BID_MULTS = [0.3, 0.6, 0.9,\n  1.2, 1.5, 1.8]\n"
        "OFFER_ADDERS = [0, 15, 30, 45]\n"
        u"→ 6×4 = 24 个离散动作",
        C_ACTION, u"动作 (Action)", 8, 10)

    # Reward formula
    box(ax, 0.3, 1.0, 4.5, 2.5,
        u"R_i = 消费价值(bid × served)\n"
        u"   − 发电成本(offer × gen)\n"
        u"   + 市场收支(p_sell·LMP − p_buy·LMP)\n"
        u"   − 未满足惩罚(penalty × unserved)",
        C_REWARD, u"奖励函数 (Reward)", 8.5, 10)

    # Arrows
    arrow(ax, 1.9, 4.0, 5.5, 7.2, color="#2563EB", lw=2)
    arrow(ax, 1.9, 6.5, 5.5, 7.2, color="#2563EB", lw=2)
    arrow(ax, 1.9, 9.0, 5.5, 7.2, color="#2563EB", lw=2)
    arrow(ax, 12.5, 7.2, 13.5, 5.3, color="#D97706", lw=2)
    arrow(ax, 15.4, 3.5, 12.5, 2.5, color="#D97706", lw=2)
    arrow(ax, 8.5, 1.0, 2.5, 2.25, color="#059669", lw=2)
    arrow(ax, 12.5, 2.5, 15.4, 6.5, color="#2563EB", lw=2)

    # Arrow labels
    ax.text(3.7, 7.6, u"状态", fontsize=9, color="#2563EB", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=C_EDGE, pad=2))
    ax.text(13.0, 6.0, u"动作", fontsize=9, color="#D97706", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=C_EDGE, pad=2))
    ax.text(7.5, 1.3, u"奖励反馈", fontsize=9, color="#059669", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=C_EDGE, pad=2))

    # Bottom annotation
    ax.text(9, 0.3,
            u"每 Episode = 1天 (96时段, 15min)  |  每 4 时段决策一次 (24步)  |  独立训练，联合出清",
            fontsize=9, ha="center", color=C_ARROW, fontstyle="italic")

    fig.tight_layout(pad=0.5)
    fig.savefig("diagram_rl_mode.png", dpi=200, bbox_inches="tight", facecolor=C_BG)
    fig.savefig("diagram_rl_mode.svg", bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("Saved: diagram_rl_mode.png / .svg")


# ============================================================================
# Diagram 2: Three-type load characteristics
# ============================================================================
def draw_load_curves():
    from grid import load_profile
    from config_loader import load_defaults

    cfg = load_defaults()
    lc = cfg["profiles"]["load"]
    to = lc["type_overrides"]

    T = 96
    hours = np.arange(T)
    h = hours * 0.25

    bp = {k: lc[k] for k in ["morning_peak_hour","morning_peak_amplitude",
           "morning_peak_width","evening_peak_hour","evening_peak_amplitude",
           "evening_peak_width","night_base"]}

    base  = load_profile(hours, **bp, phase_shift=0, amplitude_scale=1.0)

    # Add multiplicative noise to each profile (seeded for reproducibility)
    rng = np.random.RandomState(42)
    noise_sigma = lc.get("forecast_noise_sigma", 0.06)

    def _noisy(prof, seed):
        return np.clip(prof * (1 + np.random.RandomState(seed).normal(0, noise_sigma, size=prof.shape)), 0, None)

    res   = _noisy(load_profile(hours, **bp, phase_shift=to["residential"]["phase_shift_hours"],
                                amplitude_scale=to["residential"]["amplitude_scale"]), 1)
    com   = _noisy(load_profile(hours, **bp, phase_shift=to["commercial"]["phase_shift_hours"],
                                amplitude_scale=to["commercial"]["amplitude_scale"]), 2)
    ind   = _noisy(load_profile(hours, **bp, phase_shift=to["industrial"]["phase_shift_hours"],
                                amplitude_scale=to["industrial"]["amplitude_scale"]), 3)

    # Weighted composite
    w_res = 11 * 0.9
    w_com = 10 * 1.0
    w_ind = 12 * 1.1
    w_total = w_res + w_com + w_ind
    total = (res * w_res + com * w_com + ind * w_ind) / w_total

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 9),
                                    gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor("#F8FAFC")

    # Left: Individual curves
    ax1.fill_between(h, 0, res, alpha=0.12, color="#2563EB")
    ax1.fill_between(h, 0, com, alpha=0.18, color="#10B981")
    ax1.fill_between(h, 0, ind, alpha=0.12, color="#F59E0B")
    ax1.plot(h, res, color="#2563EB", linewidth=2.5,
             label=u"居民负荷 (11节点, +0.5h, ×1.10)")
    ax1.plot(h, com, color="#10B981", linewidth=2.5,
             label=u"商业负荷 (10节点, −1.0h, ×0.95)")
    ax1.plot(h, ind, color="#F59E0B", linewidth=2.5,
             label=u"工业负荷 (12节点, 0h, ×0.15)")
    ax1.plot(h, total, color="#374151", linewidth=3.0, linestyle="--",
             label=u"系统总负荷 (加权平均)", zorder=5)

    # Peak annotations
    for prof, c, label, off in [
        (res, "#2563EB", u"居民晚峰\n19:30", (1, 18)),
        (com, "#10B981", u"商业早峰\n7:00", (-15, 10)),
    ]:
        idx = np.argmax(prof)
        ax1.annotate(label, (h[idx], prof[idx]),
                     textcoords="offset points", xytext=off,
                     fontsize=13, ha="center", color=c, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=c, lw=1.5))

    ax1.axhline(1.0, color="#94A3B8", linewidth=0.8, linestyle=":")
    ax1.text(0.5, 1.02, u"均值=1.0 pu", fontsize=12, color="#94A3B8",
             transform=ax1.get_yaxis_transform())

    ax1.set_xlabel(u"时间 (小时)", fontsize=16, color="#475569")
    ax1.set_ylabel(u"标幺值 (pu, 均值=1.0)", fontsize=16, color="#475569")
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0.2, 2.0)
    ax1.set_xticks(range(0, 25, 2))
    ax1.tick_params(labelsize=13)
    ax1.legend(loc="upper right", fontsize=13, framealpha=0.95, ncol=2,
               edgecolor="#CBD5E1")
    ax1.set_title(u"三类负荷特性曲线 — 双峰高斯模型", fontsize=20, fontweight="bold",
                  color="#0F172A", pad=12)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Right: Parameter comparison bar chart
    categories = [u"相位偏移\n(小时)", u"振幅缩放\n(pu)", u"早峰时刻\n(实际)",
                  u"晚峰时刻\n(实际)", u"峰谷差\n(pu)", u"节点数"]
    res_vals = [0.5, 1.10, 8.5, 19.5, 1.02, 11]
    com_vals = [-1.0, 0.95, 7.0, 18.0, 0.89, 10]
    ind_vals = [0.0, 0.15, 8.0, 19.0, 0.14, 12]

    x = np.arange(len(categories))
    w_bar = 0.25
    bars1 = ax2.bar(x - w_bar, res_vals, w_bar, color="#2563EB", alpha=0.85,
                    label=u"居民", edgecolor="white", linewidth=0.5)
    bars2 = ax2.bar(x, com_vals, w_bar, color="#10B981", alpha=0.85,
                    label=u"商业", edgecolor="white", linewidth=0.5)
    bars3 = ax2.bar(x + w_bar, ind_vals, w_bar, color="#F59E0B", alpha=0.85,
                    label=u"工业", edgecolor="white", linewidth=0.5)

    for bars, vals in [(bars1, res_vals), (bars2, com_vals), (bars3, ind_vals)]:
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                     str(val), ha="center", fontsize=12, color="#475569", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=13)
    ax2.set_ylabel(u"参数值", fontsize=16, color="#475569")
    ax2.legend(loc="upper right", fontsize=13, framealpha=0.9, edgecolor="#CBD5E1")
    ax2.set_title(u"负荷参数对比", fontsize=20, fontweight="bold", color="#0F172A", pad=12)
    ax2.set_ylim(-1.5, 20)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax2.tick_params(labelsize=13)

    fig.tight_layout(pad=2.0)
    fig.savefig("diagram_load_curves.png", dpi=200, bbox_inches="tight", facecolor="#F8FAFC")
    fig.savefig("diagram_load_curves.svg", bbox_inches="tight", facecolor="#F8FAFC")
    plt.close()
    print("Saved: diagram_load_curves.png / .svg")


# ============================================================================
# Diagram 3: Platform architecture
# ============================================================================
def draw_architecture():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    # Layer colors
    C_UI    = "#DBEAFE"
    C_LLM   = "#F3E8FF"
    C_ALGO  = "#FFEDD5"
    C_MKT   = "#DCFCE7"
    C_DISP  = "#FCE7F3"
    C_DATA  = "#E2E8F0"
    C_EXT   = "#FEF3C7"

    def layer_box(ax, x, y, w, h, title, color, subtitle="", fontsize=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                               facecolor=color, edgecolor=C_EDGE, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.3, title, fontsize=fontsize+1, fontweight="bold",
                ha="center", va="top", color=C_TEXT, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 0.8, subtitle, fontsize=fontsize-1, ha="center",
                    va="top", color=C_SUB, zorder=3)

    def sub_box(ax, x, y, w, h, text, color=C_BOX, fs=8.5):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor=C_EDGE, linewidth=1, zorder=4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, fontsize=fs, ha="center", va="center",
                color=C_TEXT, zorder=5, linespacing=1.3)

    def v_arrow(ax, x, y1, y2, color=C_SUB, lw=1.5):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=1)

    def h_arrow(ax, x1, x2, y, color=C_SUB, lw=1.5):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=1)

    # Title
    ax.text(10, 11.55, "GridLLM — 基于智能体的配电网电力市场仿真平台架构",
            fontsize=17, fontweight="bold", ha="center", color="#0F172A")
    ax.text(10, 11.1,
            "IEEE 33-Bus  |  12.66 kV  |  T=96 (15-min)  |  Gurobi MILP/SOCP  |  Ollama LLM",
            fontsize=9.5, ha="center", color=C_SUB)

    # Layer 1: UI
    layer_box(ax, 0.3, 9.8, 19.4, 1.0, u"交互层 (UI & Interaction)", C_UI,
              "Plotly Dash  |  自然语言输入  |  CLI (run.py)")

    sub_box(ax, 0.6, 9.9, 4.2, 0.75,
            "Dashboard (port 8056)\n拓扑图 · LMP· 负荷 · 储能 · KPI")
    sub_box(ax, 5.1, 9.9, 3.5, 0.75,
            u"自然语言配置\n\"Bus20增加5MW光伏\"")
    sub_box(ax, 8.9, 9.9, 3.5, 0.75,
            u"场景快速切换\n6种内置场景 + YAML")
    sub_box(ax, 12.7, 9.9, 3.2, 0.75,
            u"批量导出\nCSV + PNG 图表")
    sub_box(ax, 16.2, 9.9, 3.2, 0.75,
            u"伪实时仿真\n逐时段可视化")

    # Layer 2: LLM Advisor
    layer_box(ax, 0.3, 8.3, 19.4, 1.0, u"智能顾问层 (LLM Advisor)", C_LLM,
              "Ollama 本地部署 (qwen2.5:7b)  |  NL→配置解析  |  结果→策略洞察")

    sub_box(ax, 0.6, 8.4, 7.5, 0.75,
            u"NL → 场景配置: 解析自然语言 → MarketConfig + Agent 列表\n"
            u"关键字/正则回退: carbon_cap / load_factor / re_multiplier / add_storage")
    sub_box(ax, 8.4, 8.4, 5.8, 0.75,
            "配置覆盖: apply_llm_config_to_agents()\n"
            u"修改负荷 / PV / Wind / 储能 / 线路容量")
    sub_box(ax, 14.5, 8.4, 4.9, 0.75,
            u"结果洞察: 4段分析\n市场概况 · 参与表现\n风险评估 · 策略建议")

    # Layer 3: Algorithm & Bidding
    layer_box(ax, 0.3, 5.8, 19.4, 2.0, u"算法与竞价层 (Algorithm & Bidding)", C_ALGO,
              u"独立 PPO 强化学习  |  纳什均衡检验  |  自适应竞价策略")

    sub_box(ax, 0.6, 6.0, 6.2, 1.6,
            u"RL 竞价 (rl_env.py)\n"
            u"· 独立 PPO Actor-Critic\n"
            u"· 状态: 20维 (负荷+RE+价格+SOC)\n"
            u"· 动作: 24 离散 (bid_mult × offer_adder)\n"
            u"· 奖励: 消费价值−发电成本+市场收支")
    sub_box(ax, 7.1, 6.0, 5.2, 1.6,
            u"纳什均衡 (nash.py)\n"
            u"· 对角化 (Gauss-Seidel, α=0.7)\n"
            u"· 雅可比 (并行最优响应, α=0.6)\n"
            u"· 虚拟博弈 (Fictitious Play)\n"
            u"· COBYLA / 随机采样 最优响应")
    sub_box(ax, 12.6, 6.0, 3.5, 1.6,
            u"价格预测\n"
            u"· Merit-Order 基本面\n"
            u"· EMA 滑动平滑\n"
            u"· 残差负荷定价\n"
            u"· 外部数据注入")
    sub_box(ax, 16.4, 6.0, 3.0, 1.6,
            u"自适应竞价\n"
            u"· RL 策略\n"
            u"· Stackelberg\n"
            u"· MPC 自调度\n"
            u"· 价格预测驱动")

    # Layer 4: Market Clearing
    layer_box(ax, 0.3, 3.3, 19.4, 2.0, u"市场出清层 (Market Clearing)", C_MKT,
              u"两结算制度 (DA+RT)  |  滚动窗口 MPC  |  多目标优化 (加权/约束)")

    sub_box(ax, 0.6, 3.5, 5.5, 1.6,
            u"日前市场 (DA)\n"
            "· day_ahead_price_china()\n"
            "· clear_market(agents, \"DA\")\n"
            u"· 滚动窗口 (da_rolling_enabled)\n"
            u"· 竞价: adaptive_bidding()")
    sub_box(ax, 6.4, 3.5, 5.5, 1.6,
            u"实时市场 (RT)\n"
            "· clear_market(agents, \"RT\")\n"
            u"· MPC 滚动出清 (rt_horizon, rt_step)\n"
            "· PriceForecaster (noisy_da)\n"
            u"· 储能实时调度")
    sub_box(ax, 12.2, 3.5, 3.8, 1.6,
            u"多目标优化\n"
            u"· 加权法: λ_re, λ_carbon\n"
            u"  λ_curtail\n"
            u"· 约束法: carbon_cap\n"
            u"  re_min_rate\n"
            u"· 影子价格")
    sub_box(ax, 16.3, 3.5, 3.1, 1.6,
            u"结算\n"
            u"· 两结算 (DA+RT)\n"
            u"· 节点LMP定价\n"
            u"· 支付分解\n"
            u"· 社会福利计算")

    # Layer 5: Dispatch (OPF)
    layer_box(ax, 0.3, 0.8, 19.4, 2.0, u"物理调度层 (Dispatch / OPF)", C_DISP,
              "Gurobi MILP / SOCP  |  DC-OPF  |  LinDistFlow  |  储能约束  |  网络安全")

    sub_box(ax, 0.6, 1.0, 4.8, 1.6,
            "DC-OPF (dispatch_dc.py)\n"
            u"· 无损线性潮流\n"
            "· Gurobi / HiGHS\n"
            u"· 单时段求解\n"
            u"· 无电压约束")
    sub_box(ax, 5.7, 1.0, 5.2, 1.6,
            "LinDistFlow (dispatch_ldf.py)\n"
            u"· 辐射状配电网\n"
            u"· 电压约束 (±7%)\n"
            u"· I²R 损耗迭代\n"
            u"· 反向功率限制")
    sub_box(ax, 11.2, 1.0, 5.0, 1.6,
            "SOCP OPF (dispatch_socp.py)\n"
            u"· 二阶锥松弛\n"
            u"· 多时段联合优化\n"
            u"· 储能 MPC 自调度\n"
            u"· 节点电价计算")
    sub_box(ax, 16.5, 1.0, 2.9, 1.6,
            u"储能约束\n"
            u"· SOC 转移\n"
            u"· 充放电效率\n"
            u"· 循环老化\n"
            u"· 终端价值")

    # Data & Config Layer
    layer_box(ax, 0.3, 0.0, 19.4, 0.55, "", C_DATA)
    ax.text(10, 0.27,
            u"数据与配置层: IEEE 33-Bus (pandapower)  |  config/defaults.yaml  |  "
            u"config/scenarios.yaml  |  负荷/光伏/风电 合成曲线  |  outputs/ CSV+PNG",
            fontsize=9, ha="center", color=C_TEXT)

    # External integration
    sub_box(ax, 17.0, 8.9, 2.4, 0.65, "Ollama\n(qwen2.5:7b)", C_EXT, fs=8)
    sub_box(ax, 17.0, 4.5, 2.4, 0.65, "Gurobi\nOptimizer", C_EXT, fs=8)
    sub_box(ax, 17.0, 2.3, 2.4, 0.65, "pandapower\nIEEE 33-Bus", C_EXT, fs=8)

    # Vertical flow arrows
    for y_src, y_dst in [(9.8, 9.3), (8.3, 7.8), (5.8, 5.3), (3.3, 2.8), (0.8, 0.55)]:
        v_arrow(ax, 10, y_src, y_dst)

    # External arrows
    ax.annotate("", xy=(17.0, 9.55), xytext=(17.0, 9.35),
                arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=1.5))
    ax.annotate("", xy=(17.0, 5.15), xytext=(17.0, 4.85),
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5))

    fig.tight_layout(pad=0.3)
    fig.savefig("diagram_architecture.png", dpi=200, bbox_inches="tight", facecolor=C_BG)
    fig.savefig("diagram_architecture.svg", bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("Saved: diagram_architecture.png / .svg")


# ============================================================================
# Diagram 4: RL conceptual diagram (Agent-Environment loop)
# ============================================================================
def draw_rl_concept():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    C_AGENT_BLUE  = "#DBEAFE"
    C_ENV_PURPLE  = "#F3E8FF"
    C_POLICY_ORANGE = "#FFEDD5"
    C_STATE_GREEN = "#DCFCE7"
    C_ACTION_YELLOW = "#FEF3C7"
    C_REWARD_PINK = "#FCE7F3"

    # Title
    ax.text(10, 11.55, u"强化学习核心概念 — Agent 与环境交互闭环",
            fontsize=18, fontweight="bold", ha="center", color="#0F172A")
    ax.text(10, 11.1, "Independent PPO  |  Multi-Agent  |  电力市场竞价",
            fontsize=9.5, ha="center", color=C_SUB)

    # Top row
    box(ax, 0.4, 8.0, 4.6, 2.8,
        u"策略 π (Policy)\n"
        u"π_θ(a | s) = Softmax(MLP(s))\n"
        u"输入: 20维观测向量\n"
        u"输出: 24个动作的概率分布\n"
        u"参数: ≈3,600 (2层MLP)",
        C_POLICY_ORANGE, u"策略 π (Policy)", 8.5, 11, round_pad=0.18)

    box(ax, 5.4, 8.0, 4.6, 2.8,
        u"奖励函数 (Reward)\n"
        u"R_i = bid×served − offer×gen\n"
        u"    + p_sell·LMP − p_buy·LMP\n"
        u"    − penalty×unserved\n"
        u"γ = 0.99 折扣因子",
        C_REWARD_PINK, u"奖励函数 (Reward)", 8.5, 11, round_pad=0.18)

    box(ax, 10.4, 8.0, 4.6, 2.8,
        u"价值函数 (Critic)\n"
        u"V(s) ≈ E[ sum gamma^t * r_t ]\n"
        u"GAE advantage:\n"
        u"  A_t = sum (gamma*lambda)^k * delta_{t+k}\n"
        u"δ_t = r_t + γV(s_{t+1}) − V(s_t)",
        C_STATE_GREEN, u"价值函数 (Critic)", 8.5, 11, round_pad=0.18)

    box(ax, 15.4, 8.0, 4.2, 2.8,
        "PPO Clipped Objective\n"
        "L = E[ min( r_t·A_t,\n"
        "  clip(r_t, 0.8, 1.2)·A_t ) ]\n"
        u"ε = 0.2  |  10 epochs\n"
        "Entropy bonus = 0.01",
        C_AGENT_BLUE, "PPO Clipped Objective", 8.5, 11, round_pad=0.18)

    # Center: Agent - Environment loop
    box(ax, 1.5, 2.5, 6.5, 4.6,
        u"Agent (产消者)\n"
        u"• 观测状态 s_t\n"
        u"• 从 π 采样动作 a_t\n"
        u"• 收到奖励 r_t\n"
        u"• 更新 π 使未来累积奖励最大化\n"
        u"• 每个产消者独立训练, 不共享参数",
        C_AGENT_BLUE, u"Agent (产消者)", 9.5, 13, round_pad=0.18)

    box(ax, 11.5, 2.5, 7.0, 4.6,
        u"Environment (市场仿真)\n"
        u"• clear_market() — Gurobi OPF\n"
        u"• 日前+实时两结算\n"
        u"• 33节点配电网 (IEEE 33-Bus)\n"
        u"• 储能SOC跨时段转移\n"
        u"• 碳排放/阻塞/电压约束\n"
        u"• 返回: LMP, schedules, welfare",
        C_ENV_PURPLE, u"Environment (市场仿真)", 9.5, 13, round_pad=0.18)

    # Loop arrows
    arrow(ax, 8.0, 6.5, 11.5, 6.5, color="#D97706", lw=2.5)
    label_arrow_text(ax, 9.75, 6.85, u"动作 a (bid_mult, offer_adder)", "#D97706", fs=9.5)

    arrow(ax, 11.5, 3.5, 8.0, 3.5, color="#2563EB", lw=2.5)
    label_arrow_text(ax, 9.75, 3.85, u"状态 s (20维观测)", "#2563EB", fs=9.5)

    arrow(ax, 11.5, 5.5, 8.0, 5.5, color="#059669", lw=2.5)
    label_arrow_text(ax, 9.75, 5.85, u"奖励 r (利润)", "#059669", fs=9.5)

    # Bottom row
    box(ax, 0.4, 0.3, 6.2, 1.6,
        u"经验收集 (RolloutBuffer)\n"
        u"每个Episode = 1天 = 24步 | 缓存 (s, a, logπ, r, V, done)\n"
        u"Episode结束后: 整条轨迹 GAE → returns → PPO update",
        "#F1F5F9", u"经验收集 (RolloutBuffer)", 8.5, 11, round_pad=0.18)

    box(ax, 7.0, 0.3, 6.0, 1.6,
        u"策略更新 (PPO Update)\n"
        "mini-batch = 64 | epochs = 10\n"
        u"对 Actor + Critic 分别做梯度下降\n"
        "numpy手动反向传播 (无深度学习框架)",
        "#F1F5F9", u"策略更新 (PPO Update)", 8.5, 11, round_pad=0.18)

    box(ax, 13.4, 0.3, 6.2, 1.6,
        u"多智能体维度\n"
        u"6个产消者各自维护独立PPO策略\n"
        u"共享环境, 不通信, 不共享观测\n"
        u"对手行为 = 环境的非平稳噪声",
        "#F1F5F9", u"多智能体维度", 8.5, 11, round_pad=0.18)

    # Arrows from loop to bottom
    arrow(ax, 4.75, 2.5, 3.5, 1.9, lw=1.8)
    arrow(ax, 14.5, 2.5, 14.5, 1.9, lw=1.8)
    arrow(ax, 10.0, 2.5, 10.0, 1.9, lw=1.8)

    # Time structure annotation
    ax.text(10, 11.4, "Episode = 1天 | Step = 4时段 (1h) | 24决策步/Episode",
            fontsize=9, ha="center", color=C_SUB, fontstyle="italic")

    fig.tight_layout(pad=0.5)
    fig.savefig("diagram_rl_concept.png", dpi=200, bbox_inches="tight", facecolor=C_BG)
    fig.savefig("diagram_rl_concept.svg", bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("Saved: diagram_rl_concept.png / .svg")


# ============================================================================
# Diagram 5: IEEE 33-bus topology
# ============================================================================
def draw_topology():
    from config_loader import load_defaults

    cfg = load_defaults()
    lt_cfg = cfg.get("load_types", {})

    # Build bus→load_type map
    bus_lt: dict = {}
    for lt_name, lt in lt_cfg.items():
        for b in lt.get("buses", []):
            bus_lt[b] = lt_name

    # Bus type colors
    C_SUBSTATION    = "#374151"
    C_RES           = "#93C5FD"
    C_COM           = "#86EFAC"
    C_IND           = "#FDBA74"
    C_PROS_RES      = "#2563EB"
    C_PROS_IND      = "#EA580C"
    C_STORAGE       = "#7C3AED"
    C_STORAGE_RING  = "#4C1D95"
    C_LINE          = "#CBD5E1"
    C_TIE           = "#94A3B8"
    C_TEXT_DARK     = "#1E293B"
    C_SUBTLE        = "#64748B"

    def bus_style(b):
        if b == 0:
            return C_SUBSTATION, "#1F2937", 2.5, 200, "s"
        if b in PROSUMER_RESIDENTIAL_BUSES:
            return C_PROS_RES, "#1E3A5F", 3.0, 180, "s"
        if b in PROSUMER_INDUSTRIAL_BUSES:
            return C_PROS_IND, "#7C2D12", 3.0, 180, "s"
        if b in STANDALONE_STORAGE_BUSES:
            return C_STORAGE, C_STORAGE_RING, 2.5, 160, "o"
        if b in RESIDENTIAL_BUSES:
            return C_RES, "#94A3B8", 1.5, 120, "s"
        if b in COMMERCIAL_BUSES:
            return C_COM, "#94A3B8", 1.5, 120, "s"
        if b in INDUSTRIAL_BUSES:
            return C_IND, "#94A3B8", 1.5, 120, "s"
        return "#E2E8F0", "#94A3B8", 1.0, 100, "s"

    def _role_text(b):
        """Short role label."""
        if b == 0:
            return u"变电站"
        if b in PROSUMER_RESIDENTIAL_BUSES:
            return u"居民产消者"
        if b in PROSUMER_INDUSTRIAL_BUSES:
            return u"工业产消者"
        if b in STANDALONE_STORAGE_BUSES:
            return u"独立储能"
        lt = bus_lt.get(b, "")
        return {"residential": u"居民用户", "commercial": u"商业用户", "industrial": u"工业用户"}.get(lt, "")

    fig, ax = plt.subplots(figsize=(30, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 40)
    ax.set_ylim(-6.5, 7.0)

    # Tie lines
    for u, v in TIE_LINES:
        pu, pv = NODE_COORDS[u], NODE_COORDS[v]
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], linestyle=(0, (4, 3)), color=C_TIE,
                linewidth=1.2, zorder=1, alpha=0.7)

    # Distribution lines
    for u, v in LINES:
        pu, pv = NODE_COORDS[u], NODE_COORDS[v]
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=C_LINE, linewidth=3.0, zorder=1,
                solid_capstyle="round")

    # Nodes
    for bus, (x, y) in NODE_COORDS.items():
        fc, ec, ew, sz, mkr = bus_style(bus)
        ax.scatter(x, y, s=sz, c=fc, edgecolors=ec, linewidths=ew, zorder=5,
                   marker=mkr, clip_on=False)

        # Bus label — offset direction adapts to trunk / branch
        if y == 0:
            # Trunk — alternate above / below to reduce crowding
            if bus % 4 in (0, 3):
                bus_xy = (0, 16)
                bus_va = "bottom"
            else:
                bus_xy = (0, -16)
                bus_va = "top"
        elif y > 0:
            bus_xy = (-14, 6)
            bus_va = "bottom"
        else:
            bus_xy = (-14, -6)
            bus_va = "top"

        ax.annotate(f"Bus{bus}", (x, y), textcoords="offset points", xytext=bus_xy,
                    fontsize=9, ha="center", va=bus_va, fontweight="bold",
                    color=C_TEXT_DARK, zorder=10,
                    bbox=dict(facecolor="white", edgecolor=C_EDGE, pad=1.5, boxstyle="round,pad=0.15"))

        # Role label
        role = _role_text(bus)
        if role:
            if y > 0:
                voff = -18
            elif y < 0:
                voff = 18
            else:
                if bus % 4 in (0, 3):
                    voff = -18
                else:
                    voff = 18
            ax.annotate(role, (x, y), textcoords="offset points",
                        xytext=(0, voff), fontsize=9, ha="center",
                        va="top" if voff < 0 else "bottom",
                        color=C_TEXT_DARK, fontweight="bold", zorder=10)

    # Region labels
    region_style = dict(fontsize=13, ha="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC",
                                  edgecolor="#CBD5E1", alpha=0.9))
    ax.annotate(u"主干线 (Bus 0–17, 混合负荷)", xy=(17, 1.8), color="#475569", **region_style)
    ax.annotate(u"分支A — 居民区\nBus 18–21 (接自 Bus 1)", xy=(5, 5.2),
                color="#1E40AF", **region_style)
    ax.annotate(u"分支B — 工业区\nBus 22–24 (接自 Bus 2)", xy=(6, -5.5),
                color="#9A3412", **region_style)
    ax.annotate(u"分支C — 工业区\nBus 25–32 (接自 Bus 5)", xy=(17, -5.5),
                color="#9A3412", **region_style)

    # Legend panel
    lx = 36.5
    ly = 5.5
    eh = 0.72
    hdr = dict(fontsize=14, fontweight="bold", color=C_TEXT_DARK, ha="left", va="center")

    ax.text(lx, ly, u"图例", **hdr)

    entries = [
        (C_SUBSTATION, "#1F2937", "s",
         u"变电站  |  Bus 0  |  12.66 kV  |  平衡节点"),
        (C_RES, "#94A3B8", "s",
         u"居民用户  |  Bus 1–4,6,18–20"),
        (C_COM, "#94A3B8", "s",
         u"商业用户  |  Bus 7–15,17"),
        (C_IND, "#94A3B8", "s",
         u"工业用户  |  Bus 22,24,26–28,32"),
        (C_PROS_RES, "#1E3A5F", "s",
         u"居民产消者 (P)  |  Bus 5, 21  |  光伏 + 储能"),
        (C_PROS_IND, "#7C2D12", "s",
         u"工业产消者 (P)  |  Bus 16, 23, 25, 30  |  风电/光伏 + 储能"),
        (C_STORAGE, C_STORAGE_RING, "o",
         u"独立储能 (S)  |  Bus 6, 13, 15, 17, 29, 31"),
    ]

    for i, (fc, ec, mkr, desc) in enumerate(entries):
        y = ly - (i + 1) * eh
        ax.scatter(lx - 0.1, y, s=55, c=fc, edgecolors=ec, linewidths=1.5,
                   marker=mkr, zorder=10, clip_on=False)
        ax.text(lx + 0.4, y, desc, fontsize=11, color=C_TEXT_DARK, va="center")

    # Line type legend
    ly2 = ly - (len(entries) + 1.2) * eh
    ax.text(lx, ly2, u"线路", **hdr)

    ax.plot([lx - 0.1, lx + 0.7], [ly2 - 0.55, ly2 - 0.55],
            color=C_LINE, linewidth=3.0, solid_capstyle="round", clip_on=False)
    ax.text(lx + 1.0, ly2 - 0.55, u"配电线路 (常闭, 32条)", fontsize=12, color=C_TEXT_DARK, va="center")

    ax.plot([lx - 0.1, lx + 0.7], [ly2 - 1.1, ly2 - 1.1],
            linestyle=(0, (4, 3)), color=C_TIE, linewidth=1.2, clip_on=False)
    ax.text(lx + 1.0, ly2 - 1.1, u"联络线 (常开, 5条, 用于重构)", fontsize=12, color=C_TEXT_DARK, va="center")

    # Stats
    ly3 = ly2 - 2.8
    ax.text(lx, ly3, u"统计", **hdr)
    stats = [
        "33节点 | 32负荷 | 1变电站",
        "37线路 (32常闭 + 5联络)",
        u"10产消者 (6工业 + 2居民 + 2PV)",
        u"光伏×5 | 风电×4 | 储能×12",
    ]
    for i, s in enumerate(stats):
        ax.text(lx, ly3 - (i + 0.7) * 0.55, s, fontsize=11, color=C_SUBTLE, va="center")

    # Title
    fig.text(0.5, 0.97,
             u"IEEE 33节点配电网拓扑结构 — 智能体与储能分布",
             fontsize=26, fontweight="bold", color=C_TEXT_DARK, ha="center", va="top")
    fig.text(0.5, 0.935,
             u"P = 产消者内置储能（表后）  |  S = 独立网侧储能  |  "
             u"12.66 kV  |  T=96时段（15分钟）",
             fontsize=14, color=C_SUBTLE, ha="center", va="top")

    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(top=0.90, right=0.70)

    fig.savefig("topology_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig("topology_diagram.svg", bbox_inches="tight", facecolor="white")
    print("Saved: topology_diagram.png, topology_diagram.svg")
    plt.close()


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    draw_rl_diagram()
    draw_load_curves()
    draw_architecture()
    draw_rl_concept()
    draw_topology()
    print("\nAll 5 diagrams generated.")
