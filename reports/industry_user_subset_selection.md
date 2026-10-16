# IEEE33 真实行业用户子集筛选

目标：在不缩放、不修改任何原始负荷曲线的前提下，从实测行业用户中选出一个子集，使最终 IEEE33 网络在 **2022-11-30** 上的全网同时有功峰值接近 **3.715 MW**。

## 结论

| | |
|---|---|
| 场景 | `typical_day_industry_subset` |
| 负荷日期 | `2022-11-30` |
| 最终系统同时峰值 | **3.6534 MW**（t=75） |
| 相对误差 | **-1.658%** |
| 匹配容差 | ±5%（**本项目选择标准，不是 IEEE33 标准参数**） |
| 入选用户 | **17** |
| 覆盖行业 | 9 — 化工业、商业综合体、汽车制造业、电制氢、电动汽车（充换电）、电子设备制造业、纺织业、通用设备制造业、钢铁 |
| 入选母线（`replace`） | [3, 16, 23, 24, 25, 26, 27, 30, 31] |
| 其余母线 | `preserve`，保留原生 IEEE33 负荷 |
| 全日平均功率 | 2.324753 MW |
| 全日电量 | 55.7941 MWh |
| 搜索空间 | 8,388,608 个子集（候选池穷举） |

取到该结果的最大可行子集有 **1** 个，没有并列。

## 为什么不是最小化误差优先

峰值是 96 个时段的**最大值**，因此把所有母线留在原生负荷上就已经复现 3.715 MW（到机器精度）—— 空集的误差为 0。若把「最小化误差」当作第一优先级，选出的将是 0 个用户，那不是一次筛选。

所以 ±5% 容差定义了「误差相近」：容差之内峰值误差按定义可比，真正携带信息的是子集能解释多少个真实用户。选择次序为：

1. 可行 = 相对误差 ≤ 容差；
2. 可行集内用户数最多；
3. 再取 |误差| 最小；
4. 再取行业数最多；
5. 再取用户 id 字典序最小。

## 搜索是穷举的，候选池是证明出来的

候选池 = 自身峰值 ≤ 目标 ×(1+容差) = **3.9007 MW** 的用户，共 **23** 个。

依据：若子集含有用户 u，则在 u 自身峰值所处时段，系统总量**至少**是 u 的贡献；其余各母线的贡献都非负，故系统峰值必然超过上限，该子集不可能落在容差内。因此非候选者不可入选，穷举候选池即穷举了整个可行空间。

该论证的前提（所有曲线与原生形状逐点非负）由 `require_non_negative` 运行时检查；一旦出现负值即报错，而不是静默剪枝出一个「并非最大」的子集。

## 每条母线的政策

| 母线 | 政策 | 依据 |
|---|---|---|
| 0 | `preserve` | slack，不映射任何行业用户 |
| 1 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 2 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 3 | `replace` | 入选用户 U656f86497702 的绝对 MW 曲线替换原生负荷 |
| 4 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 5 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 6 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 7 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 8 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 9 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 10 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 11 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 12 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 13 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 14 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 15 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 16 | `replace` | 入选用户 U0abfaa684d80 的绝对 MW 曲线替换原生负荷 |
| 17 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 18 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 19 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 20 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 21 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 22 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 23 | `replace` | 入选用户 U39e8a17aba10、Ue971298ed105 的绝对 MW 曲线替换原生负荷 |
| 24 | `replace` | 入选用户 Ub1da4ce43b04 的绝对 MW 曲线替换原生负荷 |
| 25 | `replace` | 入选用户 U018990a98eb1、U3a097bfc3f9e、Ue9e499c141dd 的绝对 MW 曲线替换原生负荷 |
| 26 | `replace` | 入选用户 U1f2d028ac65f、U5765bec968a1 的绝对 MW 曲线替换原生负荷 |
| 27 | `replace` | 入选用户 U6e0c9c380ad3、U73fcc2041874、Ucf65d248ce93、Uf67c750d6c55 的绝对 MW 曲线替换原生负荷 |
| 28 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 29 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |
| 30 | `replace` | 入选用户 U2aa2cf305ab6、U6e5b7a2d2097 的绝对 MW 曲线替换原生负荷 |
| 31 | `replace` | 入选用户 U46342f0b3576 的绝对 MW 曲线替换原生负荷 |
| 32 | `preserve` | 没有入选用户，恢复原生 IEEE33 负荷 |

政策由入选用户重新推导，**没有沿用**全量 41 用户场景的旧政策：旧政策把若干母线标为 `replace`，而本子集已把它们清空。配置、账本与映射三者逐项一致由 `scenarios.single_day_case` 校验 —— 它逐母线比对配置声明的政策与账本 manifest，不一致即 raise。

## 无功负荷

现有代码**按配置的功率因数推导无功**，不是保留原生无功、也不是别的规则：`dispatch_socp`、`dispatch_ldf`、`diagnose_typical_day` 三处都用 `q = p · tanφ`。

| | |
|---|---|
| 功率因数 | 0.8503 |
| 性质 | **仿真假设，不是实测值** |
| tanφ | 0.618956 |
| 峰值处无功 | 2.2613 MVAr |
| 全日无功电量 | 34.5341 MVArh |
| 基准无功 | 2.3 MVAr |

功率因数 `0.8503` 是按 `3.715 / √(3.715² + 2.3²)` 标定的，即**基准有功正好 3.715 MW 时**无功等于基准的 2.3 MVAr。本子集的峰值是 3.6534 MW，所以那条标定理由在本场景下只是**近似**成立，无功为 2.2613 MVAr。

**没有为了凑 2.3 MVAr 去调功率因数。** 那样做的代价是把一个声明调到一个数字上，与项目其余部分一贯拒绝的做法同类。

## 数据与边界

| | |
|---|---|
| source workbook sha256 | `63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb` |
| frozen mapping (v2) sha256 | `a5fae0850b2bd6a4d0f9db7412e71b1aac2c6771a867f11cb1f4c426386eb38b` |
| derived subset mapping sha256 | `35e4ef933980fa4d7a3a806d11d933814b83acbea5bce5b910161c8016aab9a2` |
| subset table sha256 | `ca6a4527cff7aa0336d220090c4d8588c4fa2b36cb431181e25cd12e2858ca37` |
| ledger sha256 | `da1441706242d8b635de8574cdf9d951be533627439723a8467463078688e46d` |
| 读数单位 | `kW`，证据 `ruling_only` |
| 单位状态 | **`unit_unverified`** |
| 选择规则版本 | `adace51084c8415a` |

- **账本是 audit ledger，不是网络的形式负荷。** 读数单位仍是 `unit_unverified`，`build_industry_node_loads --formal` 继续拒绝。
- **子集是针对 2022-11-30 这一天拟合的。** 换一天，峰值与入选集合都会变；这不是可泛化的结论，也不描述任何其他日期。
- **用户→母线是 simulation mapping，不是地理接线。** 本轮只筛选既有映射，没有改动、重排或新增任何映射关系。
- 账本 manifest 的 `mapping_version` 写作 `industry_bus_map_v2`，而实际读取的是**派生**映射文件；精确绑定由 manifest 的 `mapping_sha256` （派生文件的摘要）承担。
- 重建会覆盖 `reports/bus_load_composition.csv` 中全量 41 用户的数值；41 用户账本本体仍在原目录，可用原命令重建。`reports/node_loads_summary.csv` 按场景累积，两个场景各占一行。
- 账本 manifest 的 `excluded_cells` 是 **60**，比全量场景的 1 大得多，但这不是数据质量问题：该字段统计的是「在**本场景的映射**下未被放置到任何母线的表计」，而派生映射只保留了入选用户。未被选中的用户的表计由此计入，它们是本场景不使用、而非有任何缺陷。

## 逐用户

| 用户 | 母线 | 行业 | 入选 | 自身峰值 MW | 理由 |
|---|---|---|---|---|---|
| `U018990a98eb1` | 25 | 通用设备制造业 | **是** | 0.003600 | included |
| `U046530842875` | 16 | 钢铁 | 否 | 263.623925 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U074a148fa4c7` | 26 | 电子设备制造业 | 否 | 8.813248 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U0abfaa684d80` | 16 | 钢铁 | **是** | 0.000036 | included |
| `U11128e1b6964` | 22 | 水泥制造 | 否 |  | no qualified reading on the selected date |
| `U12592b9d53aa` | 32 | 数据中心 | 否 | 61.529600 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U134c5ea1227f` | 24 | 汽车制造业 | 否 | 10.164000 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U1f2af72e4165` | 27 | 纺织业 | 否 | 4.676510 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U1f2d028ac65f` | 26 | 电子设备制造业 | **是** | 0.004400 | included |
| `U1f95b8d57d1c` | 1 | 商业综合体 | 否 | 6.202800 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U22609eff325a` | 22 | 水泥制造 | 否 |  | no qualified reading on the selected date |
| `U2aa2cf305ab6` | 30 | 电制氢 | **是** | 0.198281 | included |
| `U2de6c0028077` | 23 | 化工业 | 否 | 60.121219 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U39e8a17aba10` | 23 | 化工业 | **是** | 0.152600 | included |
| `U3a097bfc3f9e` | 25 | 通用设备制造业 | **是** | 0.021000 | included |
| `U45f85cc7e9db` | 2 | 商业综合体 | 否 | 4.330282 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U46342f0b3576` | 31 | 电动汽车（充换电） | **是** | 1.302900 | included |
| `U481b69555b99` | 23 | 化工业 | 否 | 43.957018 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U4ea136e58032` | 24 | 汽车制造业 | 否 |  | no qualified reading on the selected date |
| `U5765bec968a1` | 26 | 电子设备制造业 | **是** | 0.086840 | included |
| `U579fe8c94614` | 25 | 通用设备制造业 | 否 | 12.096000 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U656f86497702` | 3 | 商业综合体 | **是** | 0.276928 | included |
| `U6ae415563f9f` | 32 | 数据中心 | 否 |  | no qualified reading on the selected date |
| `U6e0c9c380ad3` | 27 | 纺织业 | **是** | 0.040882 | included |
| `U6e5b7a2d2097` | 30 | 电制氢 | **是** | 0.005886 | included |
| `U73fcc2041874` | 27 | 纺织业 | **是** | 0.047664 | included |
| `U9b80870a7722` | 28 | 机械 | 否 | 3.051610 | eligible, but the largest tolerance-feasible subset does not contain it: adding it to the selected subset leaves the tolerance |
| `Uac6f64bb506e` | 24 | 汽车制造业 | 否 | 24.502500 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `Ub1da4ce43b04` | 24 | 汽车制造业 | **是** | 0.442700 | included |
| `Ub4e7e479aa6f` | 29 | 印刷业 | 否 |  | no qualified reading on the selected date |
| `Uc2a51b8e47ff` | 22 | 水泥制造 | 否 | 45.856800 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `Ucf65d248ce93` | 27 | 纺织业 | **是** | 0.031381 | included |
| `Ud7aaf2cc7bd7` | 28 | 机械 | 否 | 2.133600 | eligible, but the largest tolerance-feasible subset does not contain it: adding it to the selected subset leaves the tolerance |
| `Udfed49bf706b` | 31 | 电动汽车（充换电） | 否 | 1.339800 | eligible, but the largest tolerance-feasible subset does not contain it: adding it to the selected subset leaves the tolerance |
| `Ue5f08866ae4e` | 31 | 电动汽车（充换电） | 否 | 1.471200 | eligible, but the largest tolerance-feasible subset does not contain it: adding it to the selected subset leaves the tolerance |
| `Ue7a335c557ca` | 4 | 商业综合体 | 否 | 0.828000 | eligible, but the largest tolerance-feasible subset does not contain it: adding it to the selected subset leaves the tolerance |
| `Ue971298ed105` | 23 | 化工业 | **是** | 0.004000 | included |
| `Ue9e499c141dd` | 25 | 通用设备制造业 | **是** | 0.105840 | included |
| `Uf67c750d6c55` | 27 | 纺织业 | **是** | 0.044976 | included |
| `Ufbfdfb7de5b9` | 29 | 印刷业 | 否 | 1.280400 | eligible, but the largest tolerance-feasible subset does not contain it: adding it to the selected subset leaves the tolerance |
| `Ufe8fb8bfad44` | 5 | 商业综合体 | 否 | 7.101600 | own peak exceeds the tolerance ceiling; the system peak is at least this user's own contribution, so no tolerance-feasible subset contains it |
| `U007aebb8ae8c` | — | 5G基站 | 否 | 0.003629 | the mapping does not place this user on any bus |

## 复现

```bash
PYTHONPATH=src python src/select_industry_user_subset.py --date 2022-11-30
PYTHONPATH=src python src/build_industry_node_loads.py \
    --scenario typical_day_industry_subset --date 2022-11-30 \
    --mapping data\processed\industry_user_subset_mapping_v2.csv
```
