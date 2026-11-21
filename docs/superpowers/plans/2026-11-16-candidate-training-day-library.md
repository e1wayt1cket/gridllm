# 候选训练日负荷库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从冻结的真实表计来源池中，为 53 个训练负荷日逐日选出表计子集，使全网同时峰值落在 3.715 MW ±10%，并产出可复核的日库与证据。

**Architecture:** 一个纯函数选源核心（无随机、无求解器）→ 逐日母线矩阵 → 证据写入器（拒绝覆盖）→ 独立复核器（不复用选源判定）。全部只用默认 Python + NumPy，不依赖 Gurobi、pandapower 或 pytest。

**Tech Stack:** Python 3.12（默认解释器）、NumPy、标准库 `csv`/`json`/`hashlib`/`unittest`。

**Spec:** `docs/superpowers/specs/2026-11-16-candidate-training-day-library-design.md`

## Global Constraints

- 目标锚点 `3.715 MW`，容差 `±10%`，带宽 `[3.3435, 4.0865] MW`，全部逐字来自 spec。
- 每用户每日至多一个表计；同源不重复；同日 96 点；不缩放、不拆分、不复制、不跨日拼接、无合成背景。
- 代码标识符、注释、字符串只用英文；`SPEC.md`/`RESULTS.md` 用中文。
- 只写 `reports/candidate_days_2026-11-16/`；`reports/load_selection_2026-11-13/`、`reports/multi_user_load_2026-11-13/`、`data/processed/v2_cy_jn_2025/` 只读。
- 本仓库未经用户明确要求不得 commit / push。计划中的提交步骤只在用户授权后执行；未授权时停在"产物生成 + 测试通过"。
- 提交信息不得出现进度词（Phase/Step/Section/AC-x）或 AI 工具名。
- 运行命令统一带 `-B`（不写 `__pycache__`），使用默认 `python`。

## Review Focus

1. `meter_curves_train.npz` 与 `meter_sources.csv` 键集不一致（有曲线无行、或有行无曲线）——静默丢表计或崩溃。→ Task 1 测试。
2. 某用户当日全部表计零峰值——零负荷母线被算成"有载"。→ Task 1 测试。
3. 某日填充后已进带宽、无需上换——实现仍执行上换，把峰值推到上沿。→ Task 1 测试。
4. 两个可行上换得到完全相同的峰值——重复运行选源不同，产物不可重建。→ Task 4 测试。
5. 某日所有表计单条即超上界——空选源必须判 `OUT_OF_BAND` 而不是抛异常或产出空矩阵。→ Task 4 测试。

---

### Task 1: 来源池加载与逐日选源核心

**Files:**
- Create: `reports/candidate_days_2026-11-16/prototype/select_daily_sources.py`
- Test: `reports/candidate_days_2026-11-16/prototype/test_library_contract.py`

**Interfaces:**
- Consumes: `reports/load_selection_2026-11-13/evidence/source_pool/meter_curves_train.npz`（键 `"<date>|<source_id>"`，值 `(96,)` float64 MW）、`.../meter_sources.csv`（列 `date,source_id,user_id,curve_sha256`）、`data/processed/v2_cy_jn_2025/common_split.json`
- Produces:
  - `class Pool`: 属性 `curves: dict[str, np.ndarray]`、`user_of: dict[str, str]`、`curve_sha: dict[str, str]`，键均为 `"<date>|<source_id>"`
  - `load_pool(root: Path) -> Pool`
  - `class Selection`（frozen dataclass）: `date: str`、`sources: tuple[str, ...]`、`users: tuple[str, ...]`、`peak_mw: float`、`total: np.ndarray`、`swaps: int`、`status: str`（`"IN_BAND"` / `"OUT_OF_BAND"`）、`reason: str`（`"initial_in_band"` / `"raised_in_band"` / `"no_swap_below_band"` / `"nothing_selectable"`）
  - `select_day(date: str, pool: Pool, anchor_mw: float = 3.715, tolerance: float = 0.10) -> Selection`
  - `train_dates(split_path: Path) -> tuple[str, ...]`（取 `near.train` 的 `"<load>|<price>"` 中 `load` 部分，升序去重）

- [ ] **Step 1: Write the failing test**

```python
# All tests live in `class ContractTests(unittest.TestCase)`; run with
# `python -B -m unittest test_library_contract` from the prototype directory.
ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "reports/load_selection_2026-11-13/evidence/source_pool"
SPLIT = ROOT / "data/processed/v2_cy_jn_2025/common_split.json"

def test_entry_check_reproduces_the_declared_rule(self):
    pool = load_pool(POOL)
    selections = [select_day(d, pool) for d in train_dates(SPLIT)]
    peaks = [s.peak_mw for s in selections]
    nodes = [len(s.sources) for s in selections]
    assert len(selections) == 53
    assert all(s.status == "IN_BAND" for s in selections)
    assert abs(min(peaks) - 3.3476) < 5e-4
    assert abs(max(peaks) - 4.0849) < 5e-4
    assert abs(statistics.median(peaks) - 3.9248) < 5e-4
    assert sum(p > 3.715 for p in peaks) == 39
    assert (min(nodes), statistics.median(nodes), max(nodes)) == (28, 31, 32)

def test_anchor_day_matches_its_recorded_selection(self):
    pool = load_pool(POOL)
    s = select_day("2022-07-23", pool)
    assert abs(s.peak_mw - 3.8463) < 5e-4
    assert len(s.sources) == 28 and s.swaps == 1

def test_day_already_in_band_takes_no_swap(self):
    pool = load_pool(POOL)
    s = select_day("2022-12-09", pool)
    assert s.reason == "initial_in_band" and s.swaps == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: FAIL，`ImportError: cannot import name 'load_pool'`

- [ ] **Step 3: Implement `load_pool`, `train_dates`, `select_day` in `select_daily_sources.py`**

选源按 spec 第 3–4 步：用户按其当日最小峰值表计升序（并列按 `source_id`）依次填充，
加入后峰值 `<= 1.1 * anchor` 才保留；随后 `while peak < 0.9 * anchor` 做单次上换，
候选为同一用户另一条峰值更大的表计，换后峰值 `<= 1.1 * anchor` 才可行；
取换后峰值最大者，并列取 `(旧 source_id, 新 source_id)` 字典序最小。进入带宽立即停止。

表计当日峰值 `<= 0` 的曲线不参与（见 spec 第 1 步防御性规则）。
`load_pool` 在 `meter_curves_train.npz` 与 `meter_sources.csv` 键集不一致时抛 `ValueError`，
消息含缺失键数量。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: 3 tests PASS。若入口核对数字不符，停止并报告——说明规则仍有未写死的自由度，不得改期望值迁就实现。

- [ ] **Step 5: Commit（仅在用户授权提交后）**

```bash
git add reports/candidate_days_2026-11-16/prototype/select_daily_sources.py \
        reports/candidate_days_2026-11-16/prototype/test_library_contract.py
git commit -m "Add the per-day real-load source selection"
```

---

### Task 2: 逐日母线矩阵

**Files:**
- Modify: `reports/candidate_days_2026-11-16/prototype/select_daily_sources.py`
- Test: `reports/candidate_days_2026-11-16/prototype/test_library_contract.py`

**Interfaces:**
- Consumes: Task 1 的 `Pool`、`Selection`
- Produces:
  - `bus_matrix(selection: Selection, pool: Pool) -> tuple[np.ndarray, list[dict]]`：矩阵 `(96, 33)` MW；第二项为映射行，每行含 `date,source_id,user_id,bus_zero_based,meter_peak_mw,curve_sha256`

- [ ] **Step 1: Write the failing test**

```python
def test_bus_matrix_places_sources_lexicographically_after_the_slack(self):
    pool = load_pool(POOL)
    s = select_day("2022-12-22", pool)
    matrix, rows = bus_matrix(s, pool)
    assert matrix.shape == (96, 33)
    assert not matrix[:, 0].any()                     # slack carries no load
    n = len(s.sources)
    assert all(matrix[:, b].any() for b in range(1, n + 1))
    assert not matrix[:, n + 1:].any()
    ordered = sorted(s.sources)
    for offset, source in enumerate(ordered):
        assert rows[offset]["bus_zero_based"] == offset + 1
        assert np.array_equal(matrix[:, offset + 1], pool.curves[f"{s.date}|{source}"])
    assert np.array_equal(matrix.sum(axis=1), s.total)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: FAIL，`ImportError: cannot import name 'bus_matrix'`

- [ ] **Step 3: Implement `bus_matrix` in `select_daily_sources.py`**

母线 0 恒为零；入选表计按 `source_id` 字典序依次落到零基母线 `1..N`；`N+1..32` 恒为零。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit（仅在用户授权提交后）**

```bash
git add reports/candidate_days_2026-11-16/prototype/select_daily_sources.py \
        reports/candidate_days_2026-11-16/prototype/test_library_contract.py
git commit -m "Map selected sources onto the load buses"
```

---

### Task 3: 日库写入器

**Files:**
- Modify: `reports/candidate_days_2026-11-16/prototype/select_daily_sources.py`
- Test: `reports/candidate_days_2026-11-16/prototype/test_library_contract.py`

**Interfaces:**
- Consumes: Task 1–2
- Produces:
  - `write_library(out_dir: Path, pool_root: Path, split_path: Path, pool: Pool) -> dict`：写 `day_selection.csv`、`source_bus_mapping.csv`、`bus_active_load_mw.npz`、`manifest.json`；返回 manifest 字典
  - 目录已存在且非空时抛 `FileExistsError`，不覆盖、不删除
  - `manifest.json` 字段：`dataset_version`、`selection_rule`、`anchor_mw`、`tolerance`、`allowed_dates`、`inputs`（来源池三文件 + split 的 sha256）、`outputs`（三产物 sha256）、`retained_dates`、`status`、`rescaled: false`、`source_duplicated: false`、`cross_day_spliced: false`、`synthetic_background: false`、`one_meter_per_user: true`、`unit_status: "unit_basis_ruling"`、`clearing_tested: false`
  - `day_selection.csv` 列：`date,month,season,peak_mw,gap_percent,in_band,nodes_loaded,sources_used,users_available,users_excluded,energy_mwh,status,reason`
  - NPZ 键为 ISO 日期，值 `(96,33)` float64 MW

- [ ] **Step 1: Write the failing test**

```python
def test_writer_refuses_a_populated_directory(self):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "library"
        out.mkdir()
        (out / "leftover.txt").write_text("evidence", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            write_library(out, POOL, SPLIT, load_pool(POOL))

def test_written_library_round_trips(self):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "library"
        manifest = write_library(out, POOL, SPLIT, load_pool(POOL))
        assert manifest["status"] == "LOAD_ONLY_PASS"
        assert len(manifest["retained_dates"]) == 53
        assert manifest["clearing_tested"] is False
        with np.load(out / "bus_active_load_mw.npz") as archive:
            assert archive["2022-12-22"].shape == (96, 33)
```

（仓库无 pytest：测试类继承 `unittest.TestCase`，用 `tempfile.TemporaryDirectory()`。）

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: FAIL，`ImportError: cannot import name 'write_library'`

- [ ] **Step 3: Implement `write_library` and a `main()` CLI**

`main()` 参数 `--out <new dir> [--pool <dir>] [--split <path>]`；无 `--out` 时只打印逐日摘要不写文件。
所有文件用 `encoding="utf-8"`、CSV `lineterminator="\n"`、NPZ 用 `np.savez`（非压缩，
保证字节可重建）。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit（仅在用户授权提交后）**

```bash
git add reports/candidate_days_2026-11-16/prototype/select_daily_sources.py \
        reports/candidate_days_2026-11-16/prototype/test_library_contract.py
git commit -m "Write the day library with its manifest"
```

---

### Task 4: 规则契约测试

**Files:**
- Test: `reports/candidate_days_2026-11-16/prototype/test_library_contract.py`

**Interfaces:**
- Consumes: Task 1–3 的 `Pool`、`Selection`、`select_day`、`bus_matrix`
- Produces: 无（测试专用）

- [ ] **Step 1: Write the failing tests**

用一个构造 `Pool` 的辅助函数 `synthetic_pool(days: dict[str, dict[str, dict]]) -> Pool`，
形态 `{date: {source_id: {"user": str, "curve": ndarray}}}`，`curve_sha` 由曲线现算。

```python
def test_swap_above_the_ceiling_is_never_taken(self):
    # 2 MW below band plus one 3 MW meter: 3+2 超上界，必须留在 2 MW 并判 OUT
    pool = synthetic_pool(...)
    s = select_day("2022-01-01", pool)
    assert s.peak_mw <= 4.0865 + 1e-9 and s.status == "OUT_OF_BAND"

def test_two_sources_of_one_user_are_rejected(self):
    # 同一用户两条 2 MW 表计不能同时入选
    s = select_day("2022-01-01", pool)
    assert len(s.users) == len(set(s.users)) == len(s.sources)

def test_a_curve_from_another_date_is_never_used(self):
    # 只借别的日期曲线才够带的算例必须判 OUT
    assert select_day("2022-01-02", pool).status == "OUT_OF_BAND"

def test_an_undersized_day_is_not_scaled(self):
    # 0.5 MW 的曲线不得被放大到 3.715；判 OUT 且 bus 矩阵和为 0.5 MW
    s = select_day("2022-01-03", pool)
    assert s.status == "OUT_OF_BAND"
    assert abs(float(np.max(s.total)) - 0.5) < 1e-9

def test_zero_peak_only_day_yields_no_loaded_bus(self):
    s = select_day("2022-01-04", pool)
    assert s.sources == () and s.reason == "nothing_selectable"
    matrix, _ = bus_matrix(s, pool)
    assert not matrix.any()

def test_selection_is_deterministic_across_runs(self):
    pool = load_pool(POOL)
    a = select_day("2022-07-23", pool)
    b = select_day("2022-07-23", pool)
    assert a.sources == b.sources and np.array_equal(a.total, b.total)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: FAIL（`synthetic_pool` 未定义或规则未生效）

- [ ] **Step 3: Make the rule satisfy the contracts**

修 `select_day` / `bus_matrix`，不改测试期望。

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: 12 tests PASS

- [ ] **Step 5: Commit（仅在用户授权提交后）**

```bash
git add reports/candidate_days_2026-11-16/prototype/test_library_contract.py \
        reports/candidate_days_2026-11-16/prototype/select_daily_sources.py
git commit -m "Pin the selection contract with rejection tests"
```

---

### Task 5: 独立复核器

**Files:**
- Create: `reports/candidate_days_2026-11-16/prototype/verify_library.py`

**Interfaces:**
- Consumes: 已产出的 `reports/candidate_days_2026-11-16/evidence/`、来源池、split、v1 日库 `reports/multi_user_load_2026-11-13/evidence/manifest.json`
- Produces: `verify(library_dir: Path, pool_root: Path, split_path: Path) -> dict`，写 `evidence/verification.json` 并打印一行结论

**不得 import `select_daily_sources`**：用自己的 `np.load` / `csv.DictReader` 重建。

- [ ] **Step 1: Write the failing test**

在 `test_library_contract.py` 增加：

```python
def test_verifier_accepts_a_fresh_library_and_rejects_a_tampered_one(self):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "library"
        write_library(out, POOL, SPLIT, load_pool(POOL))
        assert verify(out, POOL, SPLIT)["status"] == "PASS_LIBRARY"
        with np.load(out / "bus_active_load_mw.npz") as archive:
            tampered = {k: archive[k].copy() for k in archive.files}
        tampered["2022-12-22"][:, 1] *= 0.5
        np.savez(out / "bus_active_load_mw.npz", **tampered)
        assert verify(out, POOL, SPLIT)["status"] != "PASS_LIBRARY"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: FAIL，`ImportError: cannot import name 'verify'`

- [ ] **Step 3: Implement `verify` and `main()`**

逐项检查并写入 `verification.json` 的 `checks` 字典，任一为假则 `status` 非 PASS：

1. `day_selection.csv` 53 行，日期集合等于 split 的训练日期集合
2. 每个 `in_band` 行的峰值在 `[3.3435, 4.0865]`，且等于矩阵该日行和的最大值
3. 矩阵逐点等于映射表中入选曲线之和，`max_abs_diff == 0.0`
4. 每行 `curve_sha256` 等于来源池记录，且等于曲线现算 sha256
5. 每日 `source_id` 不重复、`user_id` 不重复
6. 母线 0 全零；`N+1..32` 全零；有载母线数 = 入选表计数
7. `manifest.json` 的 `outputs` 哈希等于磁盘文件现算哈希
8. 来源池三文件哈希仍等于 `manifest.inputs`；v1 日库 `manifest.json` 记录的自身 outputs 哈希仍与磁盘一致；`data/processed/v2_cy_jn_2025/common_split.json` 哈希仍等于 `manifest.inputs`

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: 13 tests PASS

- [ ] **Step 5: Commit（仅在用户授权提交后）**

```bash
git add reports/candidate_days_2026-11-16/prototype/verify_library.py \
        reports/candidate_days_2026-11-16/prototype/test_library_contract.py
git commit -m "Verify the day library without reusing the selection rule"
```

---

### Task 6: 正式运行与交付文档

**Files:**
- Create: `reports/candidate_days_2026-11-16/SPEC.md`
- Create: `reports/candidate_days_2026-11-16/RESULTS.md`
- Create: `reports/candidate_days_2026-11-16/delivery_manifest.json`

**Interfaces:**
- Consumes: Task 1–5 全部产物与命令
- Produces: 交付入口与结论

- [ ] **Step 1: 生成证据**

Run: `python -B reports/candidate_days_2026-11-16/prototype/select_daily_sources.py --out reports/candidate_days_2026-11-16/evidence`
Expected: 打印 53 行摘要，末行 `IN_BAND 53 OUT_OF_BAND 0`；目录首次创建成功

- [ ] **Step 2: 跑复核**

Run: `python -B reports/candidate_days_2026-11-16/prototype/verify_library.py`
Expected: 输出 `PASS_LIBRARY`，`evidence/verification.json` 生成

- [ ] **Step 3: 写 `SPEC.md`**

冻结执行规格：来源池与 split 路径及哈希、6 步规则原文、映射规则、容差与锚点、
拒绝覆盖策略、运行命令。不写结果。

- [ ] **Step 4: 写 `RESULTS.md` 与 `delivery_manifest.json`**

`RESULTS.md` 用报告的裁决词汇：数据 `PASS_DATA_ONLY`、出清 `NOT_TESTED`；
列出 53 天的月度分布与峰值范围、有载母线下限、每项验收的命令与结果摘要、
以及 spec 的五条风险（峰值偏上沿、同日两版本、物理未验、价格门禁、无春季、非 MATD3 数据）。
`delivery_manifest.json` 记录全部产物与脚本的 sha256。

- [ ] **Step 5: 全量测试 + 收尾**

Run: `cd reports/candidate_days_2026-11-16/prototype && python -B -m unittest test_library_contract -v`
Expected: 13 tests PASS

Run: `git status --short reports/load_selection_2026-11-13 reports/multi_user_load_2026-11-13`
Expected: 无输出（冻结目录未被触碰）

---

## Self-Review

- **Spec coverage:** 意图/目标→Task 1–3；映射规则→Task 2；产物结构→Task 3/6；
  验收 8 项→Task 5（1–7 项）+ Task 6 Step 5（冻结证据）；5 条规则拒绝→Task 4；
  风险与非目标→Task 6 Step 4 的 RESULTS 内容。
- **Step scan:** 每步单一动作；无"处理边界情况"类空话。
- **Type consistency:** `Pool`/`Selection`/`select_day`/`bus_matrix`/`write_library`/`verify`
  在各自 Interfaces 中定义，后续任务只引用不重定义。
- **Review Focus:** 5 条各对应 Task 1 或 Task 4 的测试。
- **Proportion:** 计划约 6 个任务、13 个测试；实现预计 250–350 行。
