# GridLLM — 配电网电力市场智能体仿真平台

GridLLM 是一个基于 IEEE 33 节点配电网的智能体电力市场仿真平台，支持日前（DA）与实时（RT）双结算市场出清。产消者配有分布式光伏、风电和储能系统，通过最优潮流（OPF）进行市场出清，并集成了 LLM 自然语言配置与洞察分析功能。

## 架构总览

```
run.py / dashboard.py / batch_export.py
  ├─ config/defaults.yaml + config/scenarios.yaml  （外部化配置）
  ├─ config_loader.py                              （YAML → 类型化访问）
  ├─ scenarios.py → grid.py → models.py            （配置 → 网络 → 智能体）
  ├─ market.py → dispatch.py / dispatch_*.py → models.py （出清 → OPF → 约束）
  ├─ nash.py → market.py → dispatch.py             （博弈论 → 市场出清）
  ├─ outputs.py                                     （CSV + PNG 图表导出）
  └─ llm.py → Ollama API                           （AI 顾问）
```

## 核心功能

- **三种 OPF 模式**：DC-OPF（无损线性）、LinDistFlow（辐射网支路潮流模型）和 SOCP-OPF（二阶锥松弛），均通过 Gurobi MILP/QCQP 求解
- **多目标优化**：支持加权求和与约束法两种方法，目标涵盖碳排放、可再生能源消纳率和弃电率
- **竞价策略**：随机探索与基于节点边际电价（LMP）的自适应最优响应竞价
- **双结算体系**：日前金融结算 + 实时不平衡结算，按各智能体所在节点的 LMP 进行结算
- **滚动实时市场**：MPC 式滚动时域出清，支持 perfect、DA-as-forecast、noisy-DA 多种预测模式
- **储能自调度**：MPC 预计算储能充放电计划，OPF 阶段作为固定注入，消除储能跨期套利导致的 LMP 尖峰
- **DA 滚动时域**：限制储能价格预知能力，更真实地反映储能在实际市场中的行为
- **八种内置场景**：baseline、高比例可再生能源（2x PV 和风电）、高峰负荷（1.5x）、网络阻塞（线路容量减半）、无阻塞验证（5x）、紧瓶颈（指定线路限流）、可再生能源骤降/骤升
- **纳什均衡分析**：对角化（Gauss-Seidel）、Jacobi 和虚拟对弈三种方法，支持并行多进程
- **批量导出**：`--export-all` 运行所有默认场景并生成 CSV + PNG 图表
- **LLM 顾问**：通过 Ollama 实现自然语言场景配置和仿真洞察生成，LLM 不可用时自动降级为规则引擎
- **交互式仪表板**：Plotly Dash 应用（端口 8050），支持拓扑可视化、LMP 热力图、时序曲线、储能 SOC、KPI 卡片、结算表格和 AI 洞察面板
- **伪实时仿真**：96 时段逐步执行，增量更新储能状态，可配置壁钟速度
- **Stackelberg 博弈**：供电商作为领导者、产消者作为跟随者的主从博弈模型
- **强化学习竞价**：基于 PPO 的竞价策略训练，Gym 式环境接口

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# CLI：运行单个场景
python run.py --scenario baseline --strategy best_response --opf-mode lindistflow

# CLI：运行纳什均衡测试
python run.py --scenario high_re --nash --nash-method diagonalization --nash-iters 5

# CLI：批量导出所有场景（CSV + PNG 图表 + 纳什测试）
python run.py --export-all

# 或使用独立批量导出器
python batch_export.py

# 启动交互式仪表板
python dashboard.py

# 比较多目标优化方法
python compare_methods.py

# 运行所有测试
python -m pytest tests/ -v
```

## 环境要求

- Python 3.10+
- Gurobi（需有效许可证，所有 OPF 路径均依赖 Gurobi 求解器；DC-OPF 有 HiGHS 回退）
- Ollama（可选，用于 LLM 顾问功能）

核心依赖：`gurobipy`、`pandapower`、`dash`、`plotly`、`numpy`、`scipy`、`pandas`、`matplotlib`、`ortools`、`pyyaml`

## 模块说明

| 模块 | 职责 |
|------|------|
| `models.py` | 数据类：`MarketConfig`（所有市场/OPF/多目标参数）、`StorageSpec`（储能参数与 SOC 可行性校验）、`Agent`（负荷/PV/风电预测与实际值、储能引用） |
| `grid.py` | IEEE 33 节点网络构建；智能体种群生成（居民/商业/工业产消者），含合成负荷/PV/风电/储能曲线；生成中国日前电价曲线。网络按线路容量倍数缓存。所有可调参数通过 `config_loader.py` 从 `config/defaults.yaml` 读取 |
| `config_loader.py` | YAML 配置加载器，提供 `get_default(key_path)`、`get_scenario_cfg(name)`、`get_prosumer_cfg(load_type)` 等点号分隔键路径访问 |
| `config/defaults.yaml` | 外部化默认配置：网络拓扑、负荷类型、产消者规格、储能参数、曲线生成参数、电价曲线 |
| `config/scenarios.yaml` | 场景专属倍乘因子与描述，新增场景只需添加 YAML 条目，无需修改 Python 代码 |
| `dispatch.py` | OPF 统一入口，路由至 DC/LDF/SOCP 引擎 |
| `dispatch_core.py` | `StorageConstraints`：储能可行性校验与出清后 SOC 更新 |
| `dispatch_dc.py` | DC-OPF 引擎：无损线性最优潮流（Gurobi，HiGHS 回退可用） |
| `dispatch_ldf.py` | LinDistFlow 引擎：辐射网支路潮流模型，含多时段联合优化、储能约束、多目标（加权法与约束法） |
| `dispatch_socp.py` | SOCP-OPF 引擎：二阶锥松弛，含线路容量菱形约束、网损迭代 |
| `market.py` | 市场出清编排：构建 OPF 问题并调用调度求解；竞价策略分发（`random_actions`、`best_response_bidding`）；双结算计算；MPC 式滚动 RT 出清 |
| `nash.py` | 纳什均衡测试器：对角化、Jacobi、虚拟对弈三种迭代方法，并行多进程（`Pool`），支持分块级策略参数配置 |
| `scenarios.py` | 场景注册表，YAML 驱动：`get_scenario(name, T)` 构建智能体与电价曲线 |
| `outputs.py` | 出清结果导出：10 类曲线 CSV + 6 类 PNG 图表，图表风格与仪表板一致 |
| `llm.py` | Ollama LLM 集成：自然语言 → 场景配置解析 + 仿真后洞察生成（<200 字），LLM 不可用时自动降级为规则引擎 |
| `pseudo_realtime.py` | 伪实时仿真器：逐步执行 RT 出清，增量更新储能状态，可配置壁钟速度 |
| `dashboard.py` | 交互式 Plotly Dash 仪表板（端口 8050）：拓扑可视化、LMP 热力图、时序曲线、储能 SOC、KPI 卡片、结算表格、AI 洞察面板、伪实时控制、纳什触发 |
| `run.py` | CLI 入口：单场景运行、批量导出、纳什测试、多尺度 MPC 等 |
| `batch_export.py` | 独立批量运行器：4 场景 + 快速纳什测试 |
| `compare_methods.py` | 多目标方法对比：遍历不同碳排上限与 RE 占比，输出福利/排放/影子价格对比表 |
| `stackelberg.py` | 供电商-产消者主从博弈模型 |
| `rl_env.py` | 强化学习环境：Gym 式接口，用于训练竞价策略 |
| `rl_bidding.py` | 强化学习竞价策略：基于 PPO 的智能体竞价训练 |
| `price_forecaster.py` | 电价预测：合成正弦曲线法与基于供需栈的 merit-order 法 |
| `export_analysis.py` | 数据质量分析：智能体能量平衡、SOC 边界、异常检测 |
| `mpc_storage.py` | MPC 储能自调度：滚动时域优化储能充放电计划 |
| `export_curves.py` | 负荷曲线与电价曲线可视化导出 |
| `export_dashboard_html.py` | 仪表板静态 HTML 导出 |
| `plot_diagrams.py` | 系统架构图、负荷曲线图、参数表可视化 |
| `topology_data.py` | IEEE 33 节点拓扑坐标数据 |

## 场景一览

| 场景名 | 描述 |
|--------|------|
| `baseline` | 基准配置：标准风电/光伏/储能 |
| `high_re` | 高比例可再生能源：2x 光伏与风电容量 |
| `peak_load` | 高峰负荷：1.5x 所有负荷与储能 |
| `congestion` | 网络阻塞：线路热容量减半 |
| `no_congestion` | 无阻塞：5x 线路容量，用于算法验证 |
| `tight_bottleneck` | 紧瓶颈：指定线路（11→12，15→16）容量降至 20%，用于 LMP 阻塞研究 |
| `re_ramp_drop` | 可再生能源骤降：中点后 PV/风电输出降至 10% |
| `re_ramp_surge` | 可再生能源骤升：中点后输出从 10% 升至满载 |

## 仪表板

运行 `python dashboard.py`，浏览器打开 `http://localhost:8050`。功能包括：

- 场景快捷选择（中文标签）与 OPF 模式切换（DC / LinDistFlow / SOCP）
- 竞价策略选择（random / rl / best_response）
- 静态市场出清与伪实时仿真控制
- 纳什均衡测试触发，可配置方法与迭代次数
- 自然语言输入驱动 LLM 配置
- IEEE 33 节点拓扑图：节点按 LMP 着色，产消者以星标标注
- LMP 时序曲线：IQR 包络 + 代表性节点轨迹
- 聚合交易（购/售电）、可再生能源发电（PV/风电）和负荷（已供应/未供应）曲线
- 储能 SOC 与充/放电双面板曲线
- KPI 摘要卡片（社会福利、RE 消纳率、碳排放、弃电量）与结算明细表
- 每次仿真完成后自动生成 AI 洞察

## 批量导出

运行 `python run.py --export-all` 或 `python batch_export.py`，每场景生成：

```
exports/<timestamp>/
  baseline/      （PNG 图表 + CSV 数据文件）
  high_re/
  peak_load/
  congestion/
```

PNG 图表与仪表板风格一致：LMP 曲线、交易曲线、储能 SOC、可再生能源发电、负荷曲线、IEEE 33 节点拓扑和纳什测试汇总。

## 关键设计要点

- **Gurobi 为必需依赖**：`_HAS_GUROBI` 标志控制导入，但所有 OPF 路径均依赖 Gurobi；DC-OPF 有 HiGHS 回退，LinDistFlow 和 SOCP 无开源替代
- **T=96 为标准时域**：24 小时 × 15 分钟分辨率（dt=0.25h），所有模块均假定此值
- **`MarketConfig` 集中控制**：OPF 模式、多目标方法（`use_constraint_multi_obj` + `carbon_cap_tco2` / `re_min_rate` 或加权法 `lambda_re` / `lambda_curtail` / `lambda_carbon`）、线路容量、储能阈值、RT 时域/步长等
- **结果字典格式**：`clear_market` 返回 `{price, lmp (96×33), schedules, welfare, re_consumption_rate, carbon_emissions, carbon_intensity, total_curtailment, shadow_prices}`
- **竞价策略**：基于 `bid_mult`（缩放支付意愿）和 `offer_adder`（叠加边际成本）；`best_response_bidding` 根据历史 LMP 信号自适应调整
- **仪表板 UI**：标签使用中文，物理量与缩写保持英文
