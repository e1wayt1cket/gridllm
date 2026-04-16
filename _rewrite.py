"""
agent_trading.py 重写脚本
任务1: 风电Agent + 任务2: ε-约束法 + 任务3: 多节点多场景
保留原函数完全不变，只新增/扩展
"""

import re

with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"原始文件: {len(lines)} 行")

# ============================================================
# 改动1: Agent 数据结构 - 新增 wind_forecast / wind_real 字段
# ============================================================
new_agent = '''@dataclass
class Agent:
    name: str
    bus: int                      # 接入节点
    # True: 既有负荷又有PV和储能；False: 纯负荷 prosumer=producer+consumer 产消者
    is_prosumer: bool
    load_forecast: np.ndarray     # (T,)
    pv_forecast: np.ndarray       # (T,) for prosumer, else zeros
    load_real: np.ndarray         # (T,)
    pv_real: np.ndarray           # (T,)
    # ---- 新增：风电字段（任务1）----
    wind_forecast: np.ndarray = None   # (T,) 风电预测出力; None表示无风电
    wind_real: np.ndarray = None       # (T,) 风电实际出力
    # 报价参数（由策略/智能体动作给出）
    bid_value: float              # £/MWh（买方边际价值上限的基准）
    offer_cost: float             # £/MWh（卖方边际成本基准；对PV可理解为机会成本/磨损等）
    # 储能（只有 prosumer 1/2 有）
    storage: Optional[StorageSpec] = None   # 如果没有储能，storage=None；否则提供储能参数

    def has_wind(self) -> bool:
        """判断该Agent是否配备风电"""
        return self.wind_forecast is not None and self.wind_real is not None

    @property
    def total_re_forecast(self) -> np.ndarray:
        """总可再生预测 = PV + Wind"""
        if self.has_wind():
            return self.pv_forecast + self.wind_forecast
        return self.pv_forecast

    @property
    def total_re_real(self) -> np.ndarray:
        """总可再生实际 = PV + Wind"""
        if self.has_wind():
            return self.pv_real + self.wind_real
        return self.pv_real
'''

old_agent_start = '@dataclass\nclass Agent:'
# 找到Agent类定义的起始位置
for i, line in enumerate(lines):
    if line.strip().startswith('@dataclass') and i + \
            1 < len(lines) and 'class Agent:' in lines[i + 1]:
        agent_start = i
        break

# 找到Agent类的结束位置(下一个@dataclass或顶层def)
agent_end = agent_start
brace_depth = 0
in_agent = False
for i in range(agent_start, len(lines)):
    if '@dataclass' in lines[i] and 'class Agent' in lines[i + \
        1] if i + 1 < len(lines) else False:
        in_agent = True
    if in_agent and ('@dataclass' in lines[i] or (
            lines[i].strip().startswith('class ') and 'Agent' not in lines[i])):
        agent_end = i
        break
    if in_agent and i > agent_start + 30:
        agent_end = i
        break

# 简单方式：用文本替换整个Agent类
old_agent_text = ''
j = agent_start
while j < len(lines) and not (lines[j].strip() == '' and j > agent_start + 20):
    old_agent_text += lines[j]
    if 'storage: Optional[StorageSpec]' in lines[j]:
        # 再读两行到类结束
        for k in range(j + 1, min(j + 3, len(lines))):
            old_agent_text += lines[k]
        break
    j += 1

# 用更精确的方式：替换从 @dataclass class Agent 到 storage那行之后
agent_lines_new = new_agent.split('\n')
# 找精确范围
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if line.strip() == '@dataclass' and i + \
            1 < len(lines) and 'class Agent:' in lines[i + 1]:
        start_idx = i
    if start_idx is not None and i > start_idx and 'storage: Optional[StorageSpec]' in line:
        end_idx = i + 1  # 包含这行
        break

if start_idx is not None and end_idx is not None:
    print(f"Agent类: 替换行 {start_idx +
                         1}~{end_idx +
                             1} → {len(agent_lines_new)} 行")
    lines[start_idx:end_idx] = [l + '\n' for l in agent_lines_new]

# ============================================================
# 改动2: Network 数据结构通用化（任务3）
# ============================================================
new_network = '''@dataclass
class Network:
    """
    通用配网拓扑结构（任务3：从固定2支路→任意支路列表）

    径向配网示例：
      0(外部电网) --cap01--> 1 --cap12--> 2 --cap23--> 3 ...
    潮流近似：下游注入累加
      flow_ij = Σ net_inj(bus_k) for all k downstream of i toward j

    branches: list of dict, each {"from": int, "to": int, "cap": float}
              cap 单位 MW，双向容量约束 |flow| <= cap
    """
    branches: List[Dict[str, any]]  # 支路列表

    @classmethod
    def simple_2bus(cls, cap01: float, cap12: float) -> 'Network':
        """兼容旧接口：创建 0-1-2 两支路网络"""
        return cls(branches=[
            {"from": 0, "to": 1, "cap": cap01},
            {"from": 1, "to": 2, "cap": cap12},
        ])

    @classmethod
    def from_edges(cls, edge_list: list) -> 'Network':
        """
        从边列表创建网络
        edge_list: [(from_bus, to_bus, capacity_MW), ...]
        示例: [(0,1,8.0), (1,2,5.0), (2,3,4.0), (3,4,6.0)]
        """
        branches = [{"from": e[0], "to": e[1], "cap": e[2]} for e in edge_list]
        return cls(branches=branches)
'''

# 替换Network类
net_start = None
net_end = None
for i, line in enumerate(lines):
    if '@dataclass' in line and i + \
            1 < len(lines) and 'class Network:' in lines[i + 1]:
        net_start = i
    if net_start is not None and i > net_start + 2 and ('@dataclass' in line.strip() or (line.strip(
    ).startswith('#') and '---' in line) or (line.strip().startswith('def ') and i > net_start + 10)):
        net_end = i
        break
    if net_start is not None and i > net_start + 15:
        # 找到 cap12 那行之后的空行
        if 'cap12' in line:
            net_end = i + 2
            break

if net_start is not None and net_end is not None:
    net_lines_new = new_network.split('\n')
    print(f"Network类: 替换行 {net_start +
                           1}~{net_end +
                               1} → {len(net_lines_new)} 行")
    lines[net_start:net_end] = [l + '\n' for l in net_lines_new]

# ============================================================
# 写入中间结果检查
# ============================================================
with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\n第一轮修改后: {len(lines)} 行")
print("Agent类 + Network类 已更新")
