# 11 未解决问题（Open Issues）

> 记录 **所有尚未解决** 的问题，包括已知缺陷、未验证量、外部阻塞与设计待裁决项。
> 不记录已确认的决定（那些进 `10_DECISION_LOG.md`）。
>
> 字段：`ID` / `严重度`（`BLOCKER` / `HIGH` / `MEDIUM` / `LOW`）/ `状态`
> （`OPEN` / `PARTIAL` / `BLOCKED_EXTERNAL`）/ `证据` / `下一步`。
> **任何一条被关闭时，必须在 `M1_decision_report.md` 或后续阶段报告中给出证据，不得静默删除。**

---

## A. 契约与文档冲突

### I-01 CLAUDE.md 与规则文件把 v3 写成 live manifest

- **严重度**：HIGH ・ **状态**：本次已修 CLAUDE.md；规则文件仍 OPEN
- **证据**：M0 审计 §0.1 —— 工作树与 v3 的 19 个 pin 有 8 个不一致，与 v4 的 22 个 pin 全部一致；
  `formal_training_v4/formal_training_manifest.json` 为 `finalized` / `GO_FORMAL_TRAINING`。
- **冲突点**：`CLAUDE.md` 原写 v3 live；`.claude/rules/model_constraints.md` 写
  `storage_quote_v3`、`fixed_scale 1000`，而 v4 是 `storage_quote_v4`、`fixed_scale 100`。
- **下一步**：`CLAUDE.md` 已在本次修正为 v4。`training_contract.md` / `model_constraints.md`
  的对应表述需用户裁决后单独修正，本次未改（越权）。

### I-02 「日终 SOC 不强制 0.5」与冻结协议冲突

- **严重度**：BLOCKER（对 M2 而言）・ **状态**：OPEN
- **证据**：`formal_training_v4/formal_training_config.yaml`
  `terminal_soc.final: 0.5`、`day_end_window_only: true`；`10_DECISION_LOG.md` D-004 要求不强制。
- **影响**：任何新协议必须先定义终端 SOC 处理方式，否则预算契约与结算口径都无处对齐。
- **下一步**：见 I-07（库存价值方法）与 M1 的 `episode_independence_spec.md` §4。

### I-03 Episode 是「日内滚动」而不是「日前」

- **严重度**：BLOCKER ・ **状态**：OPEN
- **证据**：现有环境按 24 个 4 时段窗口顺序决策，每块只提交 4 段；观察含"已发生的 LMP"。
  M0 审计 L1 / L3、`clearing_model_current.md` §9.4。
- **影响**：日前市场的定义是 **D-1 一次性提交 96 时段申报**。顺序决策 + 日内价格反馈
  属于日内滚动市场，与 D-002 直接冲突。
- **下一步**：M1 的 `clearing_mathematical_model.md` §2 给出三种 Episode 结构的取舍；
  需用户裁决后再谈实现。**这是本次 M1 最重要的待裁决项。**

### I-04 报价接口不是量价曲线

- **严重度**：BLOCKER ・ **状态**：OPEN
- **证据**：`src/rl_spec.py` `ACTION_STORAGE_QUOTE_V4`：两个标量（`charge_price` /
  `discharge_price`），无功率轴、无分段；真值点在箱的下边界 `(0, 0)`。
- **影响**：规则要求的是「充电/放电量价曲线」（多段、单调、步长、端点约束），
  单点价格无法表达"报多少量"。
- **下一步**：`bid_schema_spec.json` 给出目标 schema；动作参数化方案见
  `clearing_mathematical_model.md` §7。

### I-05 申报价格范围与规则不一致

- **严重度**：HIGH ・ **状态**：OPEN
- **证据**：动作箱 charge (0, 150)、discharge (0, 300) CNY/MWh；
  规则侧申报限值为 **0–1500 CNY/MWh**（见 traceability `R-31`）。
- **影响**：项目动作箱是算法设计量，不是政策限值；两者不得互相引用，也不得以收益结果回调。
- **下一步**：新接口建议把策略输出映射到 [0, 1500] 内，由 quote map 决定可达子集，
  并在报告中说明"可达子集 ≠ 规则允许集"。需用户裁决。

### I-06 内部成本参数（循环成本）进入出清目标

- **严重度**：HIGH ・ **状态**：OPEN
- **证据**：M0 `cycle_cost_dataflow.md`：`c·(ch+dis)` 同时进入出清目标与结算账本；
  M0 §4 的两处不对称（prosumer 分支带 `γ^t`，独立储能不带）。
- **影响**：如果循环成本是参与者的真实成本，它应进入报价曲线（由策略决定），
  而不是由出清器替参与者记账。否则"报价"与"成本"混在同一目标里。
- **下一步**：`clearing_mathematical_model.md` §4 给出"成本归参与者、出清只认报价"的形式；
  需用户裁决。

---

## B. 未验证的测量

### I-07 终端库存价值方法未定

- **严重度**：HIGH ・ **状态**：OPEN
- **证据**：D-004；现有 `storage.terminal_value: null` 时取 `TV = mean(wholesale)`，
  且日终被钉住后退化为常数（M0 §5）。
- **待选方案**：见 `clearing_mathematical_model.md` §6（影子价格法 / 次日价格法 / 声明参数法）。
- **下一步**：用户选定一种；选定前不得写入任何生产路径。

### I-08 LMP 的有限差分验证未做

- **严重度**：MEDIUM ・ **状态**：**NOT_TESTED**
- **证据**：M0 §2.2 结论为 INFERRED（解析推导 slack 母线 LMP = `w_t`），未重算验证。
- **下一步**：M2 前的独立测量任务；需授权重算一天。

### I-09 报价对出清量与价格的弹性未测

- **严重度**：MEDIUM ・ **状态**：**NOT_TESTED**
- **证据**：M0 §3.2。
- **影响**：任何"储能是否行使市场力"的表述目前都没有幅度证据。

### I-10 末窗口终端钉住回退是否发生过

- **严重度**：MEDIUM ・ **状态**：**NOT_TESTED**
- **证据**：M0 F4 / L5：`dispatch_socp` 返回 `None` 时静默换引擎；
  是否发生需读既有 run 的 `train.log` / `episodes.csv`，本次未授权。

### I-11 `_extract_lmp` 对偶缺失回退是否触发过

- **严重度**：MEDIUM ・ **状态**：**NOT_TESTED**
- **证据**：M0 §2.2：回退时该格 LMP 直接等于外部价，空间差异被抹平。

---

## C. 外部阻塞

### I-12 江苏现货规则 V2.1 / V2.2 原文未取得

- **严重度**：BLOCKER ・ **状态**：**BLOCKED_EXTERNAL**
- **证据**：M1 `market_rule_traceability.csv`。本次可访问到的官方原文只有
  《江苏省电力现货市场运营规则（V1.0版）》（140 页）与《（试行）》版（146 页）。
  V2.1 的 PDF 在检索命中的托管地址返回 404；V2.2 只有媒体转述。
- **影响**：V2.1/V2.2 的条款号、页码、生效日期无法逐条核对；相关条目一律 `PARTIAL` 或 `BLOCKED`。
- **下一步**：向江苏电力交易中心 / 江苏能源监管办索取，或从交易平台会员区获取。

### I-13 储能日前结算条款未核实

- **严重度**：BLOCKER ・ **状态**：**BLOCKED_EXTERNAL**
- **证据**：V1.0 第九章只给出通用结算原则（第二百一十九、二百二十二、二百二十四条），
  不含储能条目；V2.1 附件四"市场结算公式明细"未能取得原文。
- **影响**：`settlement_spec.md` 的储能侧符号与结算价口径属于 **推断**，不是核对结果。

### I-14 储能主体资格与本项目资产不匹配

- **严重度**：HIGH ・ **状态**：**BLOCKED_EXTERNAL**
- **证据**：规则允许参与现货的是 **电网侧储能（以场站为单位）**；本项目是 4 台 0.4 MW
  配电节点储能，未核实是否满足备案、直接接入公用电网、可被调度监控等准入条件。
- **影响**：这是 **算例假设**，必须显式声明；不得写成"独立储能已具备江苏日前准入"。

### I-15 省级 SCUC/SCED、分区电价、备用与省间接口无法复现

- **严重度**：HIGH ・ **状态**：**BLOCKED_EXTERNAL**（数据不可得）
- **证据**：V1.0 第七十三条（SCUC/SCED + 社会福利最大化）、第十七/十八条（LMP 与系统边际价）、
  第二十三条（分区电价）；本项目没有省级机组参数、输电断面、备用需求、省间计划数据。
- **影响**：只能做配电网等值仿真；价格形成的主体过程不在模型内。

---

## D. 数据

### I-16 价格曲线身份与单位

- **严重度**：HIGH ・ **状态**：OPEN
- **证据**：四条曲线全部是 `spatial_mean`；`261009` 单日、其余为 7/30/180 日窗口均值；
  `price_unit_confirmed: False`（`src/wholesale_price.py`）；窗口结束日期不可核验。
- **影响**：不构成逐日价格序列；不能支撑跨日泛化结论；不能作为 D-1 可得的"预测"。

### I-17 没有 D-1 时点可得的日前价格预测

- **严重度**：HIGH ・ **状态**：OPEN
- **证据**：`01_RESEARCH_SCOPE.md` §5。
- **影响**：真实日前报价者在使用预测。若模型里策略无法获得任何 D 日价格信息，
  则"学习报价"的可行域与真实市场不同；若给它 D 日真实价格，则构成泄漏。
- **下一步**：必须在 `data_information_timing_spec.md` 里明确选择一种，并声明其代价。

### I-18 负荷数据口径

- **严重度**：MEDIUM ・ **状态**：PARTIAL
- **证据**：`01_RESEARCH_SCOPE.md` §6；行业曲线只保证归一化形状与经审计的绝对功率口径。

---

## E. 历史结论的适用性

### I-19 现有 v4 运行的结论不能外推到重构后的设计

- **严重度**：HIGH ・ **状态**：OPEN
- **证据**：v4 的 Episode 是滚动窗口、动作是单点报价、日终 SOC 钉住 0.5。
  宪章要求的是日前一次性申报、量价曲线、日终 SOC 自由。二者不是同一机制。
- **下一步**：重构后必须重新训练、重新评估；不得引用旧 Δπ 数值作为新设计的证据。

### I-20 Z10 的定位

- **严重度**：LOW ・ **状态**：OPEN（保持候选状态）
- **证据**：D-008。Z10 是离线候选，不是已批准的生产映射。

---

## F. 流程

### I-21 M1 之后没有任何已授权的下一步

- **严重度**：—（流程约束）・ **状态**：OPEN
- **证据**：D-009。
- **下一步**：等待用户审查 M1 交付物后再立项。

### I-22 `.claude/rules/model_constraints.md` 的 pin 数已过时

- **严重度**：低（文档一致性）・ **状态**：OPEN
- **证据**：该文件写"15 pinned sources"，live v4 清单
  `reports/v2_cy_jn_2025/formal_training_v4/formal_training_code_manifest.json`
  实为 22 条 `source_sha256` + 7 条 tests。`CLAUDE.md` 已同步更正为 22/7，
  规则文件本身未改（改动治理规则不在本次授权内）。
- **风险**：读规则文件的人会低估冻结范围。
- **下一步**：由用户决定是否修改规则文件。
