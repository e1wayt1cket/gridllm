# MASAC 与 MATD3+Physics 快速验证（2026-09-09）

> 状态：两处代码修复 + 2 条 200 集 rotation seed42 训练 + 3-dir K3 三层 eval 完成。

## 1. 本次代码修复

1. **`train_rl.py` 放开 masac+physics**：`--physics` 拒绝名单去 masac（仅 td3/mad3pg 拒绝）；masac 分支开 physics 时经 `_physics_factory(1)` 注入 critic。3-ep 冒烟 EXIT=0。
2. **`surplus_metrics.py` 修正消费者剩余**：`qbuy = max(served − (pv_used+wind_used), 0)`，**移除 p_dis 抵消**（储能放电价值在储能层单独结算，不再抵扣消费者购电）。加单测锁定。

**重要（口径影响）**：此修正让 `cs/cp/lmp_markup` 的**绝对/增量符号都变化** —— 此前三层文档里"RL 令消费者受损（CS 为负）"的判断是 **p_dis 抵消假象**；修正后 **RL 在四场景 CS delta 均为正、CP delta 为负**（消费者支付反而下降）。`profit/genuine_welfare/market_power_power` 不受影响。**旧三层文档的 cs/cp/markup 数值不再与本次可比；跨 run 比较 cs/cp 一律用新语义重评。** 本次三条 eval 均在**新语义**下。

## 2. 结果（K3 三层，新 CS 语义，rotation 200ep seed42，ALL 行，k CNY）

| 场景 | 指标 | masac | matd3+physics | matd3(ref) |
|---|---|---|---|---|
| baseline | profit_d | 15.07 | 14.23 | 15.08 |
| | genuine_d | 7.73 | 7.48 | 7.51 |
| | cs_d / cp_d | **+0.43 / −0.43** | +0.41 / −0.41 | +0.43 / −0.43 |
| | power | 0.33 | 0.32 | 0.33 |
| high_re | profit_d | 10.13 | 10.16 | 10.31 |
| | genuine_d | 30.28 | 29.97 | 29.90 |
| | cs_d / cp_d | +0.36 / −0.36 | +0.36 / −0.36 | +0.37 / −0.37 |
| peak_load | profit_d | 14.21 | 14.53 | 14.51 |
| | genuine_d | −16.56 | −16.72 | −16.41 |
| | cs_d / cp_d | +1.89 / −2.13 | +2.82 / −3.07 | **+3.64 / −3.85** |
| congestion | profit_d | 17.08 | 17.65 | 18.01 |
| | genuine_d | 1.52 | 1.29 | 1.55 |
| | cs_d / cp_d | +4.93 / −5.08 | +6.79 / −6.92 | **+7.67 / −7.68** |
| | power | 3.24 | 3.94 | 3.72 |

markup（load 支付加权，base→rl）：baseline/high_re 三策略几乎一致；peak_load masac 0.2294→0.2205（回落最多）、phys→0.2167、ref→0.2145；congestion masac→0.3266（回落最少）、phys→0.3174、ref→0.3154。

## 3. 训练健康（200ep rotation seed42，M1 多场景 best=4 场景均值）

| run | final_r | best_eval_r | best_ep | critic loss | 单集 |
|---|---|---|---|---|---|
| masac_rot | 5645 | 3236 | 175 | 1.93→1.43 | ~26 s（+60% vs plain matd3） |
| matd3_phys | 6274 | 3247 | 175 | 2.10→1.51 | ~19 s（+~15% vs plain） |

对照：plain `matd3_cc_rot_seed42` final_r≈6856（同 200ep、旧单场景 best 口径不同不可直接比 best_eval）。masac 200ep 相比其 100ep baseline pilot（final 5147）有提升但仍低于 matd3。

## 4. 判读 / 结论

1. **消费者口径翻转是最重要发现**：修正 p_dis 归属后，RL 储能出价在**全部四场景**降低消费者净支付（CS>0），congestion 最明显（+4.9~+7.7k）、baseline/high_re 弱正（+0.4k 左右）—— 旧"RL 让用户受损"结论主要来自指标口径，需在论文中按此口径重述（genuine 仍正、peak_load 仍负的结论不变）。
2. **MASAC（200ep）仍不构成主线替代**：等集数 final reward 仍低于 matd3（5645 vs ~6274/6856）、单集成本 +60%；评估上 profit 略低、消费者 CS 增益在 peak_load/congestion 反而更小、power 更低 —— 没有兑现"更稳健/更利消费者"。维持"暂不采纳"。
3. **matd3+physics（单 seed）≈ 中性**：profit/genuine 与 plain 相当，congestion 的 power 略高（3.94 vs 3.72）、CS 增益略小 —— 单 seed@capacity 1.5 未显示消费者/市场力差异。**physics 的价值需在 capacity 消融（E8：coupling 强弱）下检验**（何时网络耦合让 CTDE/physics 有意义），而非固定 1.5 单点。
4. 主线维持 full-obs MATD3；qmatd3≈matd3 的等价结论在 profit/genuine/power 上不受本次语义修正影响。

## 5. 产物与下一步

- eval：`results/quickval_{masac_rot,matd3_phys,matd3_cc_ref}_eval.csv`（新 CS 语义参考基线）。
- 下一步（建议）：**E8 capacity 消融** `C∈{1.5,1.0,0.8}` × {matd3, matd3+physics}（或许加 TD3 独立）跑 rotation、K3 三层（新口径）—— 回答"网络耦合何时让 CTDE / physics 有价值 + 消费者侧如何变化"。注意 `get_scenario` 会对四场景重钉 capacity，需在训练各 episode 用显式场景或重设（见 plan 风险记录）。代码未提交。
