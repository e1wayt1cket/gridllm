# Candidate training day library v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `reports/candidate_days_v3_2026-11-17/` — a 136-day load library drawn from June-December of 2021-2022, mapped onto buses by native share, with a pre-registered split.

**Architecture:** Four modules under `prototype/`. `extract_window_pool.py` mirrors the frozen v1 extraction without its train-only date guard and writes a source pool for the 428 window dates. `library_rules.py` holds the pure rules — selection with a 32-meter ceiling, native-share assignment, reactive derivation, seeded split — and is tested without I/O. `build_library.py` orchestrates them and writes the artifacts. `verify_library.py` re-reads the workbook and the written library and recomputes every gating invariant independently of the builder.

**Tech Stack:** Python 3, numpy, pandas not required, `lxml` for the workbook, `pandapower` for native `case33bw` node loads, `unittest` for the contract tests (the repo's `pytest` root does not collect `reports/`).

**Spec:** `docs/superpowers/specs/2026-11-17-candidate-day-library-v3-design.md`

## Global Constraints

- Anchor `3.715 MW`, tolerance `0.10`, band `[3.3435, 4.0865] MW`.
- `96` periods per day, `33` buses, `BUS_COUNT - 1 = 32` load buses; at most `32` selected meters per day.
- Window: June-December of 2021 and 2022, inclusive — `428` dates.
- Power factor `0.8503`; reactive ratio `0.6190 = tan(arccos(0.8503))`.
- Library size ceiling `136`; split pattern `[train] x 5, [validation], [test]`, holiday = 01-05 October of each load year.
- No curve is scaled, smoothed, resampled or aggregated. One meter per bus.
- No clearing, settlement, price or reward computation in any task.
- Do not modify `reports/load_selection_2026-11-13/`, `reports/candidate_days_2026-11-16/` or `reports/multi_user_load_2026-11-13/`; every one is read-only evidence and v1's extraction is copied, not edited.
- Do not commit until the user has approved executing this plan.
- Run every script from the repository root with `PYTHONPATH=src` and the `energy_env` interpreter, e.g.
  `PYTHONPATH=src C:/Users/26036/.conda/envs/energy_env/python.exe -B reports/candidate_days_v3_2026-11-17/prototype/<script>.py ...`

## Review Focus

Inputs the design implies but no rule names, most likely to bite first:

1. A day whose eligible meters exceed 32. v1 raised `ValueError` here; v3 must cap at 32, record `hit_bus_ceiling`, and still select. Tested in Task 2.
2. A month whose in-band count exceeds its 2025 price count (Jun 14, Jul 15, Aug 14, Sep 16, Oct 31, Nov 15, Dec 31). Must draw with the recorded seed and list the discarded dates, never truncate by date order. Tested in Task 4.
3. A bus with `P_b(t) = 0` in some period. `Q/P` is undefined there; `Q` must be `0` and the verifier must skip the ratio. Tested in Tasks 3 and 6.
4. `N < 32`. Target weights are renormalized over the target buses, so `target_max_share` is not `0.1131`; the excluded native share must be recorded and the report must compare realized to target, never to `0.1131`. Tested in Task 3.
5. A day with no eligible meter, or whose every meter alone breaks the band. Must be `OUT_OF_BAND / nothing_selectable` in the manifest, never dropped and never an exception. Tested in Task 2.

---

### Task 1: Window extraction

**Files:**
- Create: `reports/candidate_days_v3_2026-11-17/prototype/extract_window_pool.py`
- Create: `reports/candidate_days_v3_2026-11-17/prototype/test_extract_window_pool.py`

**Interfaces:**
- Consumes: `src/industry_load.py` — `require_unit_basis()`, `PV_INDUSTRY`, `anon_id(prefix, raw)`, `_make_meter_day(raw, excel_row)`, `flag_meter_day(md)`, `reading_to_mw(md)`; `data/processed/industry_block_screening.csv`; `docs/16类行业用户.xlsx`.
- Produces: `window_dates(screening_path: Path) -> tuple[str, ...]` (428 ISO dates, ascending); `extract(pool_dir: Path) -> dict` writing `meter_curves.npz`, `meter_sources.csv`, `meter_inventory.csv`, `manifest.json`.

- [ ] **Step 1: Write the failing test**

```python
import unittest
from pathlib import Path
import extract_window_pool as w

class WindowDatesTest(unittest.TestCase):
    def test_window_is_june_to_december_of_two_years(self):
        dates = w.window_dates(Path("data/processed/industry_block_screening.csv"))
        self.assertEqual(len(dates), 428)
        self.assertEqual(dates[0], "2021-06-01")
        self.assertEqual(dates[-1], "2022-12-31")
        self.assertTrue(all(d[5:7] in {"06", "07", "08", "09", "10", "11", "12"} for d in dates))

    def test_empty_date_rows_are_excluded(self):
        dates = w.window_dates(Path("data/processed/industry_block_screening.csv"))
        self.assertNotIn("", dates)
        self.assertEqual(len(set(dates)), len(dates))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_extract_window_pool -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extract_window_pool'`

- [ ] **Step 3: Implement `window_dates` and `extract`**

`window_dates` reads the CSV's `date` column, keeps ISO strings whose month is in `06..12` and year in `2021..2022`, and returns them sorted unique. It does not read `qualified_B`; the 428-date count already accounts for the screening gate.

`extract` is a copy of `reports/load_selection_2026-11-13/prototype/select_real_loads.py:extract()` with three changes: `allowed_dates()` is replaced by `window_dates(...)`, `require_train(...)` is deleted, and the output goes to `pool_dir`. Keep the admissibility predicate, the per-curve rejection set, the duplicate-conflict removal, the unit-basis call and the manifest fields byte-for-byte identical to the frozen version, so the two pools are comparable. Rename the manifest counters `train_meter_days` to `window_meter_days` and drop `"SOURCE_POOL_ONLY"`'s train wording.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_extract_window_pool -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Build the pool and record its yield**

Run:
```bash
PYTHONPATH=src python -B reports/candidate_days_v3_2026-11-17/prototype/extract_window_pool.py \
    --out reports/candidate_days_v3_2026-11-17/evidence/source_pool
```
Record `window_meter_days`, `rejected`, and the per-month `eligible_distinct_users` from `meter_inventory.csv` into a note for the final report. Expected: 428 rows in `meter_inventory.csv`; a per-month eligible-user count above 0 for all seven months in both years. If a month has fewer than its price count (Jun 14 / Jul 15 / Aug 14 / Sep 16 / Oct 31 / Nov 15 / Dec 31) of days with usable users, report `MISSING_MONTH` for it and stop before Task 5.

- [ ] **Step 6: Commit**

```bash
git add reports/candidate_days_v3_2026-11-17/prototype/extract_window_pool.py \
        reports/candidate_days_v3_2026-11-17/prototype/test_extract_window_pool.py
git commit -m "Add the June-December window extraction for the v3 load pool"
```

---

### Task 2: Selection rule with the bus ceiling

**Files:**
- Create: `reports/candidate_days_v3_2026-11-17/prototype/library_rules.py`
- Create: `reports/candidate_days_v3_2026-11-17/prototype/test_library_rules.py`

**Interfaces:**
- Consumes: nothing from Task 1 at import time; the tests use in-memory `{source_id: np.ndarray}` curves.
- Produces:
  - `ANCHOR_MW = 3.715`, `TOLERANCE = 0.10`, `PERIODS = 96`, `BUS_COUNT = 33`, `LOAD_BUSES = 32`, `MAX_METERS = 32`, `POWER_FACTOR = 0.8503`, `REACTIVE_RATIO = 0.6190`, `LOW_MW`, `HIGH_MW`.
  - `DaySelection` frozen dataclass: `date: str`, `sources: tuple[str, ...]` (ascending `source_id`), `users: tuple[str, ...]`, `peak_mw: float`, `total: np.ndarray`, `status: str`, `reason: str`, `hit_bus_ceiling: bool`.
  - `select_day(date: str, curves: Mapping[str, np.ndarray], user_of: Mapping[str, str]) -> DaySelection`.
  - `array_sha256(array: np.ndarray) -> str` — sha256 of the `<f8` bytes, the same digest v1 records in `curve_sha256`.

- [ ] **Step 1: Write the failing tests**

```python
import unittest
import numpy as np
from library_rules import LOW_MW, HIGH_MW, MAX_METERS, select_day

def flat(peak):
    return np.full(96, float(peak))

class SelectDayTest(unittest.TestCase):
    def test_caps_selection_at_the_load_bus_count(self):
        curves = {f"s{i:02d}": flat(0.05) for i in range(40)}
        users = {k: f"u{k}" for k in curves}
        sel = select_day("2021-07-01", curves, users)
        self.assertEqual(len(sel.sources), MAX_METERS)
        self.assertTrue(sel.hit_bus_ceiling)
        self.assertEqual(sel.status, "OUT_OF_BAND")
        self.assertEqual(sel.reason, "no_swap_below_band")

    def test_upward_swap_keeps_the_user_set(self):
        curves = {"a1": flat(1.0), "a2": flat(2.4), "b1": flat(1.0)}
        users = {"a1": "A", "a2": "A", "b1": "B"}
        sel = select_day("2021-07-02", curves, users)
        self.assertEqual(set(sel.users), {"A", "B"})
        self.assertIn("a2", sel.sources)
        self.assertNotIn("a1", sel.sources)
        self.assertAlmostEqual(sel.peak_mw, 3.4, places=6)
        self.assertEqual(sel.status, "IN_BAND")
        self.assertEqual(sel.reason, "raised_in_band")

    def test_initial_band_is_reported_without_raising(self):
        sel = select_day("2021-07-03", {"a": flat(2.0), "b": flat(1.5)}, {"a": "A", "b": "B"})
        self.assertEqual(sel.reason, "initial_in_band")
        self.assertEqual(sel.status, "IN_BAND")

    def test_oversized_meter_leaves_nothing_selectable(self):
        sel = select_day("2021-07-04", {"a": flat(HIGH_MW + 0.1)}, {"a": "A"})
        self.assertEqual(sel.sources, ())
        self.assertEqual(sel.reason, "nothing_selectable")
        self.assertEqual(sel.status, "OUT_OF_BAND")

    def test_zero_peak_meter_is_not_the_smallest(self):
        curves = {"a1": flat(0.0), "a2": flat(2.0), "b": flat(1.5)}
        sel = select_day("2021-07-05", curves, {"a1": "A", "a2": "A", "b": "B"})
        self.assertIn("a2", sel.sources)
        self.assertNotIn("a1", sel.sources)

    def test_selection_is_deterministic(self):
        curves = {"a": flat(2.0), "b": flat(1.5), "c": flat(1.4)}
        users = {"a": "A", "b": "B", "c": "C"}
        self.assertEqual(select_day("2021-07-06", curves, users),
                         select_day("2021-07-06", curves, users))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_library_rules -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_rules'`

- [ ] **Step 3: Implement `select_day`**

Port `reports/candidate_days_2026-11-16/prototype/select_daily_sources.py:select_day` with these differences:

- candidates are given as a `{source_id: curve}` mapping already sliced to this date (Task 5 passes the day's slice), so no date prefix filtering; they are filtered to `peak > 0` before L0.4, so a zero-peak meter is never "the smallest" for its user;
- the ascending add loop stops when `len(chosen) == MAX_METERS`, setting `hit_bus_ceiling`;
- the upward swap candidate rule stays **within the same user** (`user_of[other] == user_of[k]`), tie-break `(-trial_peak, k, other)`, and the loop bound stays `swaps < len(candidates)`;
- the returned `peak_mw` is `float(total.max())` where `total` is the sum of the selected curves, not a second recomputation;
- `sources` is sorted; `users` follows the same order.

`LOW_MW = ANCHOR_MW * (1 - TOLERANCE)`, `HIGH_MW = ANCHOR_MW * (1 + TOLERANCE)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_library_rules -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add reports/candidate_days_v3_2026-11-17/prototype/library_rules.py \
        reports/candidate_days_v3_2026-11-17/prototype/test_library_rules.py
git commit -m "Add the v3 selection rule with the load-bus ceiling"
```

---

### Task 3: Native-share assignment and reactive derivation

**Files:**
- Modify: `reports/candidate_days_v3_2026-11-17/prototype/library_rules.py`
- Modify: `reports/candidate_days_v3_2026-11-17/prototype/test_library_rules.py`

**Interfaces:**
- Consumes: Task 2's constants.
- Produces:
  - `BusAssignment` frozen dataclass: `bus_of_source: dict[str, int]`, `target_share: dict[int, float]`, `d_init: float`, `d_final: float`, `excluded_native_share: float`, `target_max_share: float`.
  - `native_p0() -> np.ndarray` — shape `(33,)`, `p0[0] = 0.0`.
  - `assign_buses(pi: Mapping[str, float], p0: np.ndarray) -> BusAssignment`.
  - `bus_matrix(bus_of_source: Mapping[str, int], curves: Mapping[str, np.ndarray]) -> np.ndarray` — shape `(96, 33)`.
  - `reactive_from_active(active: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
from library_rules import (assign_buses, native_p0, reactive_from_active,
                           POWER_FACTOR, REACTIVE_RATIO)

class AssignBusesTest(unittest.TestCase):
    def setUp(self):
        self.p0 = native_p0()

    def test_assignment_uses_distinct_target_buses(self):
        pi = {f"m{i}": 0.2 for i in range(5)}
        a = assign_buses(pi, self.p0)
        self.assertEqual(len(set(a.bus_of_source.values())), 5)
        self.assertTrue(all(b in a.target_share for b in a.bus_of_source.values()))

    def test_target_max_share_equals_native_at_full_coverage(self):
        pi = {f"m{i}": 1.0 for i in range(32)}
        a = assign_buses(pi, self.p0)
        self.assertAlmostEqual(a.target_max_share, 0.1131, places=3)
        self.assertAlmostEqual(a.excluded_native_share, 0.0, places=9)

    def test_partial_coverage_records_excluded_share(self):
        pi = {f"m{i}": 1.0 for i in range(28)}
        a = assign_buses(pi, self.p0)
        self.assertGreater(a.target_max_share, 0.1131)
        self.assertAlmostEqual(a.excluded_native_share, 0.0606, places=3)

    def test_swap_iteration_never_increases_deviation(self):
        pi = {"big": 0.30, "mid": 0.25, "small": 0.20, "tiny": 0.15, "dot": 0.10}
        a = assign_buses(pi, self.p0)
        self.assertLessEqual(a.d_final, a.d_init)

    def test_reactive_ratio_holds_and_zero_stays_zero(self):
        active = np.zeros((2, 33))
        active[0, 1] = 1.234
        reactive = reactive_from_active(active)
        self.assertAlmostEqual(reactive[0, 1] / active[0, 1], REACTIVE_RATIO, places=9)
        self.assertEqual(reactive[0, 0], 0.0)
        self.assertTrue((reactive[0, 2:] == 0.0).all())

    def test_declared_ratio_matches_the_power_factor(self):
        self.assertAlmostEqual(np.tan(np.arccos(POWER_FACTOR)), REACTIVE_RATIO, places=3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_library_rules -v`
Expected: FAIL with `NameError: name 'assign_buses' is not defined` (or ImportError)

- [ ] **Step 3: Implement the four functions**

- `native_p0`: `pandapower.networks.case33bw()`, build `np.zeros(33)`, `p0[int(row.bus)] = row.p_mw` over `net.load.itertuples()`.
- `assign_buses`: target set = the `N` largest `p0` indices, ties by smaller index; `w_b = p0[b] / sum over target`; `pi` is the day's meter shares. Initial assignment pairs meters sorted by `(pi desc, source_id asc)` with target buses sorted by `(w desc, bus asc)`. Then loop: enumerate all `<i, j>` occupied pairs `i < j` ascending, take the exchange with the largest `D` reduction subject to `> 1e-12`, tie-break by the lexicographically smallest `(bus_i, bus_j)`, stop when none qualifies. `D = sum((assigned_b / w_b - 1) ** 2)` with `assigned_b = sum of pi_m` on that bus.
- `bus_matrix`: zeros `(96, 33)`; `matrix[:, bus] = curves[source]`.
- `reactive_from_active`: `REACTIVE_RATIO * active`; must leave zeros at zero.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_library_rules -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add reports/candidate_days_v3_2026-11-17/prototype/library_rules.py \
        reports/candidate_days_v3_2026-11-17/prototype/test_library_rules.py
git commit -m "Add native-share bus assignment and the reactive derivation"
```

---

### Task 4: Seeded split

**Files:**
- Modify: `reports/candidate_days_v3_2026-11-17/prototype/library_rules.py`
- Modify: `reports/candidate_days_v3_2026-11-17/prototype/test_library_rules.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SPLIT_SEED = 20261117`.
  - `SplitPlan` frozen dataclass: `blocks: dict[str, tuple[str, ...]]` keyed `train|validation|test|holiday`, `counts: dict[str, int]`, `pairs: dict[str, tuple[str, ...]]` (each `"load|price"`), `discarded: dict[str, tuple[str, ...]]`, `library_size: int`, `holiday_shortfall: int`.
  - `draw_split(in_band_dates: Sequence[str], price_dates_by_month: Mapping[str, Sequence[str]], seed: int = SPLIT_SEED) -> SplitPlan`.

- [ ] **Step 1: Write the failing tests**

```python
class DrawSplitTest(unittest.TestCase):
    def _dates(self, months, n, year=2022):
        """Round-robin the months so every generated day number stays valid."""
        return [f"{year}-{months[i % len(months)]:02d}-{i // len(months) + 1:02d}"
                for i in range(n)]

    def _prices(self):
        return {f"{m:02d}": tuple(f"2025-{m:02d}-{d:02d}" for d in range(1, 32))
                for m in range(6, 13)}

    def test_126_days_give_90_18_18(self):
        plan = draw_split(self._dates([6, 7, 8, 9, 11, 12], 126), self._prices())
        self.assertEqual(plan.library_size, 126)
        self.assertEqual(plan.counts, {"train": 90, "validation": 18, "test": 18, "holiday": 0})

    def test_holiday_block_is_the_national_day_window(self):
        plan = draw_split(self._dates([10], 31), self._prices())
        self.assertEqual(plan.blocks["holiday"],
                         tuple(f"2022-10-0{d}" for d in range(1, 6)))

    def test_month_over_price_count_draws_and_records_discards(self):
        prices = {"06": tuple(f"2025-06-{d:02d}" for d in range(1, 15))}
        plan = draw_split(self._dates([6], 30), prices)
        self.assertEqual(plan.library_size, 14)
        self.assertEqual(len(plan.discarded["06"]), 16)
        self.assertEqual(sum(plan.counts.values()), 14)

    def test_blocks_are_disjoint_by_date_and_by_price(self):
        plan = draw_split(self._dates([6, 7, 8, 9, 10, 11, 12], 140), self._prices())
        loads = [p.split("|")[0] for pair in plan.pairs.values() for p in pair]
        prices_used = [p.split("|")[1] for pair in plan.pairs.values() for p in pair]
        self.assertEqual(len(set(loads)), len(loads))
        self.assertEqual(len(set(prices_used)), len(prices_used))
        self.assertEqual(len(loads), plan.library_size)

    def test_same_seed_gives_the_same_split(self):
        dates = self._dates([9], 20)
        self.assertEqual(draw_split(dates, self._prices()),
                         draw_split(dates, self._prices()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_library_rules -v`
Expected: FAIL with `NameError: name 'draw_split' is not defined`

- [ ] **Step 3: Implement `draw_split`**

1. Holiday window: every date in the input whose `MM-DD` is in `10-01..10-05`. `H = min(len(holiday_window), 10)`, take the first `H` in ascending order.
2. Per month `m`, `P_m = len(price_dates_by_month[m])`, `E_m` = in-band dates in that month after the holiday dates are removed. `keep_m = min(E_m, P_m)`, allocated across load years by largest remainder on `E_{m,year}`; within each `(month, year)` take that year's quota by `numpy.random.default_rng(seed + 100 * month + year).permutation` over the date-sorted list. Non-kept dates go to `discarded[m]`.
3. Group the kept non-holiday dates by `(month, year)`; shuffle each group with `default_rng(seed + 100 * month + year)`; interleave the groups round-robin in `(month asc, year asc)` order to get `order`.
4. `pattern = (["train"] * 5 + ["validation", "test"])`, truncated to `len(order)`; zip it with `order`.
5. Pair each month's selected load dates, sorted, with that month's price dates, sorted, one-to-one; a month with more price dates than loads leaves the extras unused.
6. `counts` is `Counter` over `pattern`; `library_size = len(order) + H`.

The `default_rng` call is the only randomness; every loop iterates over sorted keys.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_library_rules -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add reports/candidate_days_v3_2026-11-17/prototype/library_rules.py \
        reports/candidate_days_v3_2026-11-17/prototype/test_library_rules.py
git commit -m "Add the seeded month-and-year stratified split"
```

---

### Task 5: Library build

**Files:**
- Create: `reports/candidate_days_v3_2026-11-17/prototype/build_library.py`
- Create: `reports/candidate_days_v3_2026-11-17/prototype/test_build_library.py`

**Interfaces:**
- Consumes: `library_rules.select_day`, `assign_buses`, `bus_matrix`, `reactive_from_active`, `draw_split`, `native_p0`, and every constant.
- Produces: `load_pool(pool_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str], dict[str, str]]` (curves by date, `user_of` by `"date|source"`, `curve_sha`); `mv_price_dates(npz_path: Path) -> dict[str, tuple[str, ...]]` keyed `"MM"`; `run(pool_dir: Path, out_dir: Path, price_dates: Mapping[str, tuple[str, ...]]) -> dict`; `main(argv)` with `--pool`, `--out`, `--price-curves`.

- [ ] **Step 1: Write the failing test**

```python
class BuildTest(unittest.TestCase):
    def test_build_writes_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pool = tmp / "pool"
            pool.mkdir()
            day = np.full(96, 2.0)
            np.savez(pool / "meter_curves.npz", **{"2022-11-01|m1": day,
                                                   "2022-11-01|m2": np.full(96, 1.5)})
            (pool / "meter_sources.csv").write_text(
                "date,source_id,user_id,curve_sha256\n"
                f"2022-11-01,m1,u1,{library_rules._array_sha256(day)}\n"
                f"2022-11-01,m2,u2,{library_rules._array_sha256(np.full(96, 1.5))}\n",
                encoding="utf-8")
            out = tmp / "out"
            manifest = build_library.run(pool, out, {"11": ("2025-11-01", "2025-11-02")})
            for name in ("day_selection.csv", "source_bus_mapping.csv",
                         "bus_active_load_mw.npz", "bus_reactive_load_mvar.npz",
                         "split.json", "manifest.json"):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(manifest["library_size"], 1)
            with np.load(out / "bus_reactive_load_mvar.npz") as npz:
                q = npz["2022-11-01"]
            with np.load(out / "bus_active_load_mw.npz") as npz:
                p = npz["2022-11-01"]
            self.assertTrue((np.abs(q - 0.6190 * p) < 1e-9).all())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_build_library -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_library'`

- [ ] **Step 3: Implement `load_pool`, `mv_price_dates`, `run`**

- `load_pool` mirrors `select_daily_sources.load_pool`: keys `"date|source"` in `meter_sources.csv`, curves in `meter_curves.npz`, and it raises if the two key sets differ or if any curve's recomputed `_array_sha256` differs from the CSV.
- `mv_price_dates` reads `np.load(npz_path).files`, keeps keys starting `mv/`, strips the prefix, groups by `MM`, sorted.
- `run`:
  1. `dates = sorted(curves_by_date)`.
  2. For each date, `select_day(date, curves, user_of)`.
  3. `library = [sel for sel in selections if sel.status == "IN_BAND"]`; this is the only filter.
  4. Per library day, `pi = {source: curve.mean() / total_mean}`; `assign_buses`; write `(96, 33)` active and reactive matrices.
  5. `draw_split([s.date for s in library], price_dates)`.
  6. Write `day_selection.csv` (all 428 candidate rows, `columns = date, month, peak_mw, gap_percent, in_band, nodes_loaded, hit_bus_ceiling, status, reason`), `source_bus_mapping.csv` (`date, source_id, user_id, bus_zero_based, meter_mean_mw, meter_share, target_share, curve_sha256`), both npz files, `split.json`, `manifest.json`.
  7. `manifest.json` records: input sha256 (workbook, screening, pool files, design doc), `universe_dates`, `window_dates`, `extraction` counters, `band`, `power_factor`, `reactive_ratio`, `library_size`, `month_counts`, `hit_bus_ceiling_count`, `out_of_band_dates` with their peaks and reasons, `split_seed`, `applied_counts`, `discarded_dates`, `holiday_shortfall`, output sha256, and the four `content_sha256` values for the npz arrays (hash the arrays, not the zip bytes).
  8. Refuse a non-empty `out_dir` with `FileExistsError`.

`manifest["status"]` is `LOAD_ONLY_PASS` when every library day is in band and all seven months appear; otherwise `PARTIAL`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_build_library -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reports/candidate_days_v3_2026-11-17/prototype/build_library.py \
        reports/candidate_days_v3_2026-11-17/prototype/test_build_library.py
git commit -m "Add the v3 library builder"
```

---

### Task 6: Independent verifier

**Files:**
- Create: `reports/candidate_days_v3_2026-11-17/prototype/verify_library.py`
- Create: `reports/candidate_days_v3_2026-11-17/prototype/test_verify_library.py`

**Interfaces:**
- Consumes: the written library directory and the workbook; it must not import `library_rules.select_day` or `build_library.run`.
- Produces: `verify(library_dir: Path) -> dict` with `{"checks": {name: bool}, "report": {...}}`; `main(argv)` writing `verification.json` and exiting non-zero if any gating check is false.

- [ ] **Step 1: Write the failing test**

```python
class VerifyTest(unittest.TestCase):
    def test_verifier_flags_two_meters_on_one_bus(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp)
            for name in ("day_selection.csv", "source_bus_mapping.csv",
                         "bus_active_load_mw.npz", "bus_reactive_load_mvar.npz",
                         "split.json", "manifest.json"):
                shutil.copy(REAL_LIBRARY / name, lib / name)
            rows = list(csv.DictReader(open(lib / "source_bus_mapping.csv", encoding="utf-8")))
            first_date = rows[0]["date"]
            same_day = [r for r in rows if r["date"] == first_date]
            same_day[1]["bus_zero_based"] = same_day[0]["bus_zero_based"]
            with open(lib / "source_bus_mapping.csv", "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = verify_library.verify(lib)
            self.assertFalse(result["checks"]["one_meter_per_bus"])
```

`REAL_LIBRARY` is `Path("reports/candidate_days_v3_2026-11-17/evidence")`, built in Task 7. Mark the test `@unittest.skipUnless(REAL_LIBRARY.exists(), "library not built")`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_verify_library -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verify_library'`

- [ ] **Step 3: Implement `verify`**

Gating checks, each computed from the files on disk:

- `curve_preservation`: for each date, the multiset of nonzero per-bus columns in `bus_active_load_mw.npz` equals the multiset of curves re-read from the workbook for that day's mapped sources, by `_array_sha256`; a source not found re-derivable from the workbook fails the check.
- `one_meter_per_bus`: no bus appears twice in a date's mapping rows; no source appears twice.
- `reactive_proportionality`: for every bus and period with `P > 0`, `|Q/P - 0.6190| <= 1e-9`; where `P == 0`, `Q == 0`.
- `band_library_only`: every `IN_BAND` row's `peak_mw` in `[3.3435, 4.0865]`; every `OUT_OF_BAND` row carries a `reason` in the four codes.
- `split_integrity`: blocks disjoint by load date and by price date; counts equal `Counter` over the pattern the builder recorded in `split.json`; `sum(counts) == library_size`.
- `month_coverage`: `{06..12}` present in the library months.
- `determinism`: recompute `content_sha256` of both npz files and compare with `manifest.json`.

Reported, not gating, under `report["concentration"]`: per day `realized_max_share`, `target_max_share`, `native_max_share`, `d_init`, `d_final`, `excluded_native_share`; plus `median_realized_over_target` and, when it exceeds `1.1`, `M_A_QUALITY_WARNING` with the offending dates.

- [ ] **Step 4: Run test to verify it fails for the injected defect only**

Run: `cd reports/candidate_days_v3_2026-11-17/prototype && python -B -m unittest test_verify_library -v`
Expected: PASS once the real library exists; it must FAIL `one_meter_per_bus` in the mutated copy and pass every other check there.

- [ ] **Step 5: Commit**

```bash
git add reports/candidate_days_v3_2026-11-17/prototype/verify_library.py \
        reports/candidate_days_v3_2026-11-17/prototype/test_verify_library.py
git commit -m "Add the independent v3 library verifier"
```

---

### Task 7: Build, verify, and report

**Files:**
- Create: `reports/candidate_days_v3_2026-11-17/SPEC.md`
- Create: `reports/candidate_days_v3_2026-11-17/RESULTS.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the delivered library and its report.

- [ ] **Step 1: Build the library**

Run:
```bash
PYTHONPATH=src python -B reports/candidate_days_v3_2026-11-17/prototype/build_library.py \
    --pool reports/candidate_days_v3_2026-11-17/evidence/source_pool \
    --out reports/candidate_days_v3_2026-11-17/evidence \
    --price-curves data/processed/v2_cy_jn_2025/price_curves.npz
```
Expected: `LOAD_ONLY_PASS`, `library_size` 136 or the smaller measured `L`, all seven months present. If `library_size < 136`, write the applied counts and the recomputed `episodes_per_seed` into the manifest note before continuing.

- [ ] **Step 2: Freeze the spec copy**

Copy `docs/superpowers/specs/2026-11-17-candidate-day-library-v3-design.md` byte-for-byte to `reports/candidate_days_v3_2026-11-17/SPEC.md` and record both sha256 in `manifest.json` under `spec_copy`.

- [ ] **Step 3: Run the verifier**

Run:
```bash
PYTHONPATH=src python -B reports/candidate_days_v3_2026-11-17/prototype/verify_library.py \
    --library reports/candidate_days_v3_2026-11-17/evidence
```
Expected: exit 0, every gating check `true`, `verification.json` written.

- [ ] **Step 4: Rebuild into a scratch directory and compare**

Run the builder into `reports/candidate_days_v3_2026-11-17/rebuild/`, then compare the sha256 of `day_selection.csv`, `source_bus_mapping.csv`, both npz files and `split.json` against `manifest.json`'s recorded values.
Expected: every hash equal. Record the comparison as `reproducibility.json` beside the evidence. Delete the scratch directory afterwards.

- [ ] **Step 5: Write `RESULTS.md`**

Use the verdict vocabulary `PASS` / `FAIL` / `NOT_TESTED` / `INFERRED` / `BLOCKED`, one row per claim, each with the command or artifact path it came from. Cover: library size and month counts; out-of-band candidate count; `hit_bus_ceiling_count`; the concentration table's median and `M_A_QUALITY_WARNING` status; split counts and discarded dates; verifier result; reproducibility result. State explicitly that clearing feasibility is `NOT_TESTED` here and is measured separately.

- [ ] **Step 6: Commit**

```bash
git add reports/candidate_days_v3_2026-11-17/
git commit -m "Build and verify the v3 candidate day library"
```

---

## Self-Review Notes

- Every design rule maps to a task: L0.1-L0.4 and the 428-date window to Task 1; L1 and the bus ceiling to Task 2; L2 including reactive and the reported concentration fields to Tasks 3 and 6; L3 to Task 4; artifacts and manifest to Task 5; acceptance to Tasks 6 and 7.
- The design's "reported, not gating" concentration section is Task 6's `report`, never a check.
- No task computes a price, a clearing or a reward.
