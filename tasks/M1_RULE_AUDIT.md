# M1 任务书：江苏日前市场规则审计

- **立项日期**：2026-11-12
- **状态**：本次执行中，完成后停止等待审查（`10_DECISION_LOG.md` D-009）
- **上游**：M0 当前出清机制审计（`reports/market_mechanism_audit/M0_current_clearing_audit_2026-11-12/`）
- **下游**：M2 出清器实现（**未授权**）

---

## 1. 目标

以 **当前有效的江苏官方市场规则** 为依据，核实日前市场的十个问题，并据此提出
新的数学模型方案。产出 **规则规格 + 可追溯性表 + 申报 schema + 出清模型 + 结算/时序/验证规格**。

**不做**：不改代码、不实现出清器、不训练。

## 2. 必须核实的十个问题

| # | 问题 | 对应产物 |
| --- | --- | --- |
| Q1 | 当前有效的日前市场规则文件、版本、生效日期 | `jiangsu_da_market_spec.md` §1 |
| Q2 | 储能参与日前市场的主体资格与参与方式 | §2 |
| Q3 | 储能充电、放电的申报要求 | §3 |
| Q4 | 申报量价曲线的字段、分段数、价格区间、功率区间、单调性 | §3 + `bid_schema_spec.json` |
| Q5 | 申报时间、关闸时间、出清时间顺序 | §4 + `data_information_timing_spec.md` |
| Q6 | 日前市场出清目标与约束 | §5 + `clearing_mathematical_model.md` |
| Q7 | 储能中标量与节点价格的适用规则 | §6 |
| Q8 | 储能日前充电、放电的结算规则 | §7 + `settlement_spec.md` |
| Q9 | 与 IEEE33 等值模型的差异 | §8 |
| Q10 | 因缺少省级机组与输电网数据而无法复现的部分 | §8 |

## 3. 证据规则

每条规则必须记录：来源名称、发布机构、版本、发布日期、生效日期、条款/页码、
对本项目的影响、是否可直接实现。

**证据分级**（三档，不得互相替代）：

| 级别 | 定义 | 可否写入结论 |
| --- | --- | --- |
| `OFFICIAL_PRIMARY` | 官方发布页或官方 PDF 原文，可核对到条款 | 可以引用，须标文件与日期 |
| `OFFICIAL_SECONDARY` | 交易机构转发稿、官方渠道的转载，但未取得原文 PDF | **只能标 PARTIAL**，不得作为唯一依据 |
| `TRADE_PRESS` | 行业媒体、聚合页、公众号 | **不得作为结论依据**，只能作为核查线索 |
| `PROJECT_ASSUMPTION` | 研究设定 | 可以陈述，但必须称为假设 |

**无法获得官方依据时标 `BLOCKED`。不得猜测，不得引用不可靠二手说法作为确定规则。**

## 4. 交付物清单

目录：`reports/market_mechanism_audit/M1_jiangsu_da_rule_audit_2026-11-12/`

| 文件 | 内容 |
| --- | --- |
| `jiangsu_da_market_spec.md` | 十个问题的核实结果 |
| `market_rule_traceability.csv` | 逐条规则的来源与证据等级 |
| `bid_schema_spec.json` | 申报量价曲线的机器可读 schema |
| `clearing_mathematical_model.md` | 96 时段申报 + 双向量价 + 配电网出清 + SOC + PCC + 节点价 |
| `settlement_spec.md` | 日前结算模型 |
| `episode_independence_spec.md` | 日间独立 Episode 管理 |
| `data_information_timing_spec.md` | 信息集与时序 |
| `verification_plan.md` | 验证计划（M2 前的可验证门槛） |
| `M1_decision_report.md` | 裁决材料：已确认 / 未确认 / 有争议 / 未完成 |
| `hash_manifest.json` | 本次新建与修改文件的 SHA-256 |

## 5. 验收标准

1. 十个问题 **每一项** 都有明确结论或明确的 `BLOCKED`/`PARTIAL` 标记，不允许留空、不允许含糊。
2. `market_rule_traceability.csv` 的每一行都能回溯到一个来源条目。
3. 规则事实与研究假设在文档中 **物理分离**（不同小节、不同列）。
4. 新模型方案区分 **官方规则 / 研究者假设 / 工程简化** 三类输入。
5. 不出现"江苏官方规则"与"项目动作空间"的混称。
6. 不修改任何正式代码、配置、冻结产物；不启动训练或清算。
7. `git status --porcelain` 的新增项只包含本次允许的目录与 `CLAUDE.md`。

## 6. 明确不做

- 不实现出清器、不写 Python、不改 `src/`
- 不重新清洗、不重算任何历史日
- 不读 Test / Holiday / H2
- 不 commit、不 push
