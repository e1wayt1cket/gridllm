# Local day-ahead RL stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move training onto `local_da_clearing` with a price-and-quantity storage action, a ledger-defined reward, and the measurements that decide whether protocol v7 may be frozen.

**Architecture:** A new window/adapter layer (`local_da_days`, `rl_da_env`) builds each block's `DayAheadInput` from the v3 library and clears it with the exact MIQCP plus the direction-conditional price solve. A new action map (`da_quote_map_v5`, `rl_spec_v7`) widens the storage quote from two prices to two prices and two quantities, and a new trainer (`rl_da_gate_v7`) reuses the existing replay buffer, critic and straight-through gate while anchoring on the v5 declared point. Nothing in the pinned v6 manifest set is edited; v7 file names are new so no v4/v6 artifact can be mistaken for a v7 one.

**Tech Stack:** Python, Gurobi (`NonConvex=2` MIQCP + `QCPDual=1` QCP), PyTorch, NumPy, pytest, the v3 candidate-day library.

**Spec:** `docs/superpowers/specs/2026-11-17-local-da-rl-stack-design.md`

## Global Constraints

- Frozen and not to be edited: the 22 `source_sha256` entries of `reports/v2_cy_jn_2025/formal_training_v6/formal_training_code_manifest.json` — including `rl_env.py`, `rl_spec.py`, `da_quote_map.py`, `rl_b_c2_gate.py`, `rl_da_gate.py`, `training_contract.py` — and its 7 pinned tests/audits. v7 code is new files that import from them. `src/local_da_clearing.py`, `src/local_da_contracts.py`, `src/local_da_accounting.py`, `src/run_local_da.py`, `src/calibrate_dev_penalty.py` are not pinned and may be modified.
- One episode is one day: 96 periods, 24 blocks, 4 committed periods per block, 20-period window `[t_start, min(t_start+20, 96))`. Pre-declared fallback: lookahead 16 → 8 (window 20 → 12), never below 8.
- Storage fleet: four `StorageUnit`s at buses `[9, 14, 17, 27]`, 0.4 MW, 1.6 MWh, eta 0.92 both ways, SOC band `[0.1, 0.9]`, initial 0.5, degradation 100 CNY/MWh. Terminal SOC is free in every block.
- Network: native case33bw radial tree, slack 1.0 pu, `0.93 <= V <= 1.07`, no `EXTERNAL33` in the training day, reactive load from the library's own `Q = 0.6190 P` basis.
- Action `storage_quote_v5`, 16 quote dimensions: per unit `(charge_price, charge_quantity, discharge_price, discharge_quantity)`, bounds `(0,150),(0,0.4),(0,300),(0,0.4)`, truthful `(0, 0.4, 0, 0.4)`.
- Observation `v7_12d`: unique `load_feat, re_feat, soc, lmp_bus_norm, avg_lmp_bus_hist_norm`; shared `block_pos, slr_norm, avg_other_dev, all_dev_std, ema_dev, price_trend, pred_lmp_norm`. Every price feature is past-only.
- Gate head unchanged: 2 gates per unit, straight-through Bernoulli; actor 256-128, centralized twin critic 256-256; reward mode `B_C2`, gamma 0.99, fixed scale 100.
- Probe block 11, one probe per training day. Aborts above 5% of a round's episodes mark the round `DEGRADED`.
- Only train days may be dropped for feasibility, and the drop is recorded as `feasibility_selected`. Validation, test and holiday days are never used for training, replay or selection; the H2 holdout stays sealed. `reports/load_selection_2026-11-13/`, `reports/candidate_days_2026-11-16/`, `reports/multi_user_load_2026-11-13/` and `reports/candidate_days_v3_clearing_2026-11-17/` are read-only.
- Code, comments, tests, commit messages in English; no AI tool names and no progress words (FIXED, Step, Phase, Section, Week, AC-x) in them.
- Every verdict in a report is `PASS` / `FAIL` / `NOT_TESTED` / `INFERRED` / `BLOCKED` with the command or artifact it came from.

## Review Focus

- **A block that clears but cannot be priced.** A reasonable user expects a named abort, never a zero reward: `PRICE_SOLVE_FAILED` aborts the episode and writes nothing to the buffer. Task 4's `test_an_unpriced_block_names_its_failure` pins it.
- **The last block, `t_start = 92`.** The window is 4 periods with no lookahead. A user expects block 23 to commit exactly what it clears, carry no phantom lookahead reward, and add the inventory term once. Task 5's `test_one_inventory_term_per_episode` pins it.
- **A policy that declares zero quantity on both legs.** A user expects never a crash and never a payoff from energy that was not dispatched: either the window clears with zero storage dispatch and a zero payoff, or it aborts with a named failure code. Task 4's `test_zero_quantity_quotes_either_clear_or_name_the_failure` pins it.
- **A carried SOC sitting on the band edge (0.1 or 0.9).** A user expects a named `CLEAR_INFEASIBLE`, not a `ValueError` from validation or a corrupt SOC path. Task 5's `test_a_carried_soc_on_the_band_edge_keeps_the_band` pins it.
- **Validation day 2022-08-02 is exact-infeasible.** A user expects 18 validation days attempted, 17 settled, and the infeasible one named — never a silently shortened count. Task 9's `test_the_eval_counts_name_the_infeasible_day` pins it.

---

### Task 1: Price a fixed-direction window

**Files:**
- Modify: `src/local_da_clearing.py` (`ClearingModel.__init__`, the solver-parameter block)
- Test: `tests/test_local_da_pricing_tolerance.py`
- Evidence: `reports/candidate_days_v3_clear_tol_2026-11-18/` (new)

**Interfaces:**
- Consumes: `run_local_da.build_day(date) -> (DayAheadInput, dict)`; `local_da_clearing.clear_day(day, *, seconds, price_probes, log_path) -> dict` (unchanged signature).
- Produces: the same `clear_day` contract, with `pricing["optimal"]` true on library days whose exact dispatch is optimal.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_da_pricing_tolerance.py
from local_da_clearing import ClearingModel, clear_day
from run_local_da import build_day


def test_the_conditional_price_solve_is_optimal_on_a_library_day():
    day, _ = build_day("2021-09-02")
    result = clear_day(day, seconds=120, price_probes=False)
    assert result["dispatch"]["optimal"] is True
    assert result["pricing"]["optimal"] is True
    assert result["pricing"]["solver_status"] == 2
    assert result["conditional_objective_difference_cny"] <= max(
        .01, abs(result["dispatch"]["clearing_objective"]) * 1e-6)


def test_the_exact_model_keeps_the_tight_barrier_tolerance():
    day, _ = build_day("2021-06-04")
    exact = ClearingModel(day, exact=True)
    convex = ClearingModel(day, exact=False)
    try:
        assert exact.model.Params.BarQCPConvTol == 1e-10
        assert convex.model.Params.BarQCPConvTol == 1e-6
    finally:
        exact.close()
        convex.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_local_da_pricing_tolerance.py -v`
Expected: both FAIL — `pricing["optimal"]` is `False` with `solver_status == 13`, and the convex model reports `BarQCPConvTol == 1e-10`.

- [ ] **Step 3: Set the barrier tolerance per branch**

In `ClearingModel.__init__`, replace the single `m.Params.BarQCPConvTol = 1e-10` with a branch and a one-line reason:

```python
# The exact model's quadratics are the exclusivity products and need the
# tight barrier. The fixed-direction pricing model has no product left to
# converge, and at 1e-10 Gurobi returns SUBOPTIMAL with a solution whose
# objective is ~0.005 CNY off the exact one; its default 1e-6 returns OPTIMAL.
m.Params.BarQCPConvTol = 1e-10 if exact else 1e-6
```

Do not relax `FeasibilityTol` or `OptimalityTol`, and do not accept `solver_status == 13` anywhere.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_local_da_pricing_tolerance.py -v`
Expected: 2 passed.

- [ ] **Step 5: Re-screen the library into a new evidence directory**

Run, first on the four days the previous screen could not price at all, then on the whole library:

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clear_tol_2026-11-18/subset \
    --dates 2021-06-04 2021-06-13 2021-09-02 2021-10-18
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clear_tol_2026-11-18/evidence
```

Probes are on unless `--no-probes` is passed; the full run is the expensive one.

Expected: the subset run reports `price_qualification` other than `NOT_TESTED`-by-failure on every date, i.e. no `"Direction-conditional prices unavailable or solve not optimal"`. The full run reports `feasible` 132 of 136 (the four exact-infeasible days unchanged: 2021-12-24, 2022-08-02, 2022-10-11, 2022-10-14) and a `qualified_for_settlement` count that is now bounded by `NONSMOOTH_OR_FAILED`, not by an unpriced solve. Record both counts in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/local_da_clearing.py tests/test_local_da_pricing_tolerance.py
git commit -m "Price the fixed-direction solve at the tolerance Gurobi reports optimal

The exact model keeps 1e-10 on its exclusivity products. The convex pricing
model has no product left and returned SUBOPTIMAL with a solution inside the
clear_day objective tolerance, which refused every library day's prices."
```

---

### Task 2: Build a window from a library pair

**Files:**
- Create: `src/local_da_days.py`
- Modify: `src/run_local_da.py` (extract the network builder out of `_assemble_day`)
- Test: `tests/test_local_da_v7_days.py`

**Interfaces:**
- Consumes: `run_local_da`'s `V3`, `PRICE_CURVES`, `sha`, `production_renewable_curves`, and the new `radial_network()`.
- Produces:
  - `local_da_days.load_pair(load_date: str) -> dict` — keys `load_date`, `price_date`, `block`, `active` (96, 33), `reactive` (96, 33), `external_price` (96,).
  - `local_da_days.train_pairs(screen_dir: Path) -> tuple[list[dict], dict]` — feasible train pairs, and `{"dropped": {date: reason}}`.
  - `local_da_days.window_day(pair, t_start: int, t_end: int, *, soc: dict[str, float], quotes: dict[str, np.ndarray]) -> DayAheadInput`.
  - `run_local_da.radial_network() -> RadialNetwork`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_da_v7_days.py
import numpy as np
from local_da_contracts import validate_day
from local_da_days import load_pair, train_pairs, window_day
from run_local_da import build_day

PAIR = "2021-06-04"


def test_a_pair_carries_the_library_curves_and_the_paired_price():
    pair = load_pair(PAIR)
    assert pair["active"].shape == (96, 33)
    assert pair["reactive"].shape == (96, 33)
    assert pair["external_price"].shape == (96,)
    assert pair["price_date"] == "2025-06-02"
    assert np.allclose(pair["reactive"], 0.6190 * pair["active"], atol=1e-3)


def test_a_window_slices_the_day_and_carries_the_soc():
    pair = load_pair(PAIR)
    soc = {f"ESS{b}": 0.55 for b in (9, 14, 17, 27)}
    quotes = {"ESS9": np.array([10., 0.3, 120., 0.2])}
    day = window_day(pair, 44, 64, soc=soc, quotes=quotes)
    validate_day(day)
    assert day.periods == 20
    assert day.terminal_soc_equal is False
    assert [u.soc_initial for u in day.storage] == [0.55] * 4
    assert day.bids["ESS9"].discharge_price_cny_mwh.tolist() == [120.] * 20
    assert day.bids["ESS9"].discharge_quantity_mw.tolist() == [0.2] * 20
    assert np.allclose(day.active_load_mw, pair["active"][44:64])
    day_of = build_day(PAIR)[0]
    assert np.allclose(day.network.r_mw_base, day_of.network.r_mw_base)
    assert day.network.branches == day_of.network.branches


def test_the_train_pairs_drop_only_exact_infeasible_days_and_name_them():
    pairs, selection = train_pairs(
        "reports/candidate_days_v3_clear_tol_2026-11-18/evidence")
    assert len(pairs) == 87
    assert set(selection["dropped"]) == {"2021-12-24", "2022-10-11", "2022-10-14"}
    assert all(p["block"] == "train" for p in pairs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_local_da_v7_days.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_da_days'`.

- [ ] **Step 3: Implement**

In `run_local_da.py`, move the topology/`zbase` block of `_assemble_day` into `radial_network()` returning `RadialNetwork(33, 0, branches, r, x)`; `_assemble_day` calls it. No behaviour change.

In `local_da_days.py`:

- `LIBRARY = ROOT / "reports/candidate_days_v3_2026-11-17/evidence"`, `DT_HOURS = 0.25`, `STORAGE_BUSES = (9, 14, 17, 27)`.
- `load_pair` verifies `manifest.json`'s sha for both npz and the `split.json` membership exactly as `run_local_da.build_day` does, and reads `archive[f"south/{price_date}"]` from `PRICE_CURVES`.
- `train_pairs(screen_dir)` reads `split.json`'s `pairs["train"]` and each date's `screen_dir/<date>.json`; a day whose `failure` starts with `"Exact complementary dispatch was not optimal"` is dropped with that reason, every other day is kept. Never reads the holiday block.
- `window_day` slices `active`, `reactive`, `external_price`; builds four `StorageUnit(f"ESS{b}", b, soc_initial=soc[f"ESS{b}"])`; slices `production_renewable_curves()` to `[t_start:t_end]`; expands each unit's 4-vector `(charge_price, charge_quantity, discharge_price, discharge_quantity)` to constant per-period arrays in `quotes` order, defaulting a missing unit to `ACTION_STORAGE_QUOTE_V5.truthful_array()`; `DayAheadInput(..., bid_price_lower=0., bid_price_upper=600., terminal_soc_equal=False)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_local_da_v7_days.py tests/test_local_da_v3_inputs.py -v`
Expected: 3 new tests plus the existing v3-input tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/local_da_days.py src/run_local_da.py tests/test_local_da_v7_days.py
git commit -m "Build a clearing window from one v3 library pair"
```

---

### Task 3: Price-and-quantity action and the v7 observation

**Files:**
- Create: `src/rl_spec_v7.py`, `src/da_quote_map_v5.py`
- Test: `tests/test_da_quote_map_v5.py`, `tests/test_rl_spec_v7.py`

**Interfaces:**
- Consumes: `rl_spec.register_action_spec`, `register_obs_spec`, `ActionSpec`, `ObservationSpec`, `FeatureSpec`, `get_action_spec`, `get_obs_spec`; `da_quote_map.bound_gate_logits`, `da_quote_map.map_fraction`.
- Produces:
  - `rl_spec_v7.ACTION_STORAGE_QUOTE_V5` (`storage_quote_v5`, dim 4), `rl_spec_v7.OBS_V7` (`v7_12d`, 12 dims).
  - `da_quote_map_v5.POLICY_DIM = 6`, `QUOTE_DIM = 4`, `ACTION_BOX_HIGH = (150.0, 0.4, 300.0, 0.4)`, `TRUTHFUL = (0.0, 0.4, 0.0, 0.4)`, `forward_policy_v5(policy_out, *, train=False, generator=None, gate_temperature=1.0, straight_through=True) -> dict`, `target_quotes_v5(policy_out) -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_da_quote_map_v5.py
import torch
from da_quote_map_v5 import QUOTE_DIM, TRUTHFUL, forward_policy_v5, target_quotes_v5

CLOSED = torch.tensor([[-1., -1., 2., -2., 3., 0.]])
OPEN_MID = torch.tensor([[1., 1., 0., 0., 0., 0.]])


def test_a_closed_gate_declares_the_truthful_vector():
    out = forward_policy_v5(CLOSED)
    assert out["quotes"].shape == (1, QUOTE_DIM)
    assert out["quotes"][0].tolist() == list(TRUTHFUL)


def test_an_open_gate_at_zero_logit_withholds_half():
    out = forward_policy_v5(OPEN_MID)
    q = out["quotes"][0]
    assert abs(q[0].item() - 75.) < 1e-4 and abs(q[2].item() - 150.) < 1e-4
    assert abs(q[1].item() - 0.2) < 1e-4 and abs(q[3].item() - 0.2) < 1e-4


def test_withholding_is_reachable_and_differentiable():
    z = torch.tensor([[1., 1., 0., -3., 0., 3.]], requires_grad=True)
    q = forward_policy_v5(z)["quotes"][0]
    # A low quantity logit withholds more than a high one, and neither is 0.
    assert 0. < q[1].item() < q[3].item() < 0.4
    q.sum().backward()
    assert z.grad[0, 3].item() > 0. and z.grad[0, 5].item() > 0.


def test_the_target_action_is_the_hard_gate_map():
    assert torch.allclose(target_quotes_v5(OPEN_MID),
                          forward_policy_v5(OPEN_MID)["quotes"])
```

```python
# tests/test_rl_spec_v7.py
import rl_spec_v7  # noqa: F401  registers the specs
from rl_spec import get_action_spec, get_obs_spec


def test_the_v5_box_and_its_declared_point():
    spec = get_action_spec("storage_quote_v5")
    assert spec.dim == 4
    assert spec.truthful_array().tolist() == [0., 0.4, 0., 0.4]
    assert spec.bounds == {"charge_price": (0., 150.), "charge_quantity": (0., 0.4),
                           "discharge_price": (0., 300.), "discharge_quantity": (0., 0.4)}


def test_the_v7_observation_is_five_unique_and_seven_shared():
    spec = get_obs_spec("v7_12d")
    assert spec.total_dim == 12 and spec.unique_dim == 5
    assert spec.feature_order[:5] == ("load_feat", "re_feat", "soc",
                                      "lmp_bus_norm", "avg_lmp_bus_hist_norm")
    assert spec.feature_order[5:] == ("block_pos", "slr_norm", "avg_other_dev",
                                      "all_dev_std", "ema_dev", "price_trend",
                                      "pred_lmp_norm")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_da_quote_map_v5.py tests/test_rl_spec_v7.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`da_quote_map_v5.forward_policy_v5` splits the last axis into `gate_logits[..., :2]` and `quote_logits[..., 2:]`, applies `bound_gate_logits`, `probs = sigmoid(bound / gate_temperature)`, and the same straight-through/hard draw as `da_quote_map.forward_policy_v3` (train: `hard + (probs - probs.detach())`; eval: `bound > 0`). Then, with `frac = map_fraction(quote_logits)`:

```python
charge_price = gate[..., 0] * 150.0 * frac[..., 0]
charge_qty = 0.4 * (1.0 - gate[..., 0] * (1.0 - frac[..., 1]))
discharge_price = gate[..., 1] * 300.0 * frac[..., 2]
discharge_qty = 0.4 * (1.0 - gate[..., 1] * (1.0 - frac[..., 3]))
quotes = torch.stack([charge_price, charge_qty, discharge_price, discharge_qty], -1)
```

Return `quotes`, `gates`, `probs`, `gate_logits`, `quote_logits`, `mode` (`gate0 + 2*gate1`, as a long). `rl_spec_v7` registers both specs with the exact names, versions, bounds and truthful values in Global Constraints.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_da_quote_map_v5.py tests/test_rl_spec_v7.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rl_spec_v7.py src/da_quote_map_v5.py tests/test_da_quote_map_v5.py tests/test_rl_spec_v7.py
git commit -m "Declare a storage quote as a price and a quantity per leg"
```

---

### Task 4: Clear one block on the local engine

**Files:**
- Create: `src/rl_da_env.py`
- Test: `tests/test_local_da_env_v7.py`

**Interfaces:**
- Consumes: `local_da_days.load_pair`, `window_day`, `train_pairs`; `local_da_clearing.ClearingModel`, `choose_directions`; `rl_spec_v7.OBS_V7`, `ACTION_STORAGE_QUOTE_V5`; `rl_env.BiddingEnv` (its `spec_for`, `_compute_opponent_features`, `_default_actions`, `get_state_dim`).
- Produces:
  - `BLOCK_SIZE = 4`, `N_BLOCKS = 24`, `ROLL_HORIZON = 20`, `PROBE_BLOCK = 11`.
  - `FAILURE_CODES = ("CLEAR_INFEASIBLE", "DIRECTIONS_FAILED", "PRICE_SOLVE_FAILED", "BASELINE_INFEASIBLE")`.
  - `EpisodeClearingError(RuntimeError)` with `failure_code`, `date`, `block`.
  - `clear_window(day, *, seconds=120.0) -> dict` — keys `dispatch`, `pricing`, `directions`, `idle`, `failure` (`None` or one of `FAILURE_CODES`), `direction_unstable` (bool).
  - `directions_consistent(directions, idle, pricing, *, tolerance=1e-6) -> bool`.
  - `LocalDABiddingEnv(BiddingEnv)` with `__init__(*, seconds=120.0, roll_horizon=20)`, `reset(pair) -> dict[str, np.ndarray]`, `step(actions) -> (obs, rewards, done, info)`, `agent_names` (property: `[a.name for a in self.rl_agents]`), `bus_lmp_history: dict[str, list[float]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_da_env_v7.py
import numpy as np
import pytest

import rl_da_env
from local_da_days import load_pair
from rl_da_env import EpisodeClearingError, LocalDABiddingEnv, directions_consistent

PAIR = "2021-06-04"
TRUTHFUL = np.array([0., 0.4, 0., 0.4])


def truthful(env):
    return {nm: TRUTHFUL.copy() for nm in env.agent_names}


def test_a_truthful_block_returns_finite_rewards_and_committed_lmps():
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    obs, rewards, done, info = env.step(truthful(env))
    assert set(rewards) == set(env.agent_names)
    assert all(np.isfinite(v) for v in rewards.values())
    assert info["lmp"].shape == (4, 33)
    assert all(0.1 <= s <= 0.9 for s in env.prev_soc.values())
    assert obs["ESS9"].shape == (12,)
    assert done is False


def test_the_observation_reads_no_price_after_the_decision():
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    env.step(truthful(env))
    agent = env.all_agents[0]
    hist = np.array(list(env.hist_lmp))
    before = env._compute_obs_features(agent, 1, hist, float(hist.mean()))
    saved = env.pair["external_price"].copy()
    env.pair["external_price"][8:] = 9999.
    after = env._compute_obs_features(agent, 1, hist, float(hist.mean()))
    env.pair["external_price"] = saved
    assert set(before) == set(after)
    assert all(np.allclose(before[k], after[k]) for k in before)
    assert before["lmp_bus_norm"] == pytest.approx(
        env.bus_lmp_history["ESS9"][-1] / rl_da_env.PRICE_REF_CNY_MWH)


def test_the_bus_history_holds_only_committed_periods():
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    env.step(truthful(env))
    assert len(env.bus_lmp_history["ESS9"]) == 4
    assert np.allclose(env.bus_lmp_history["ESS9"],
                       env._last_result["pricing"]["lmp_cny_mwh"][:4, 9])


def test_an_unpriced_block_names_its_failure(monkeypatch):
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    monkeypatch.setattr(rl_da_env, "clear_window",
                        lambda day, **kw: {"failure": "PRICE_SOLVE_FAILED"})
    with pytest.raises(EpisodeClearingError) as caught:
        env.step(truthful(env))
    assert caught.value.failure_code == "PRICE_SOLVE_FAILED"


def test_zero_quantity_quotes_either_clear_or_name_the_failure():
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    zero = {nm: np.array([0., 0., 0., 0.]) for nm in env.agent_names}
    try:
        _, _, _, info = env.step(zero)
    except EpisodeClearingError as error:
        assert error.failure_code in ("CLEAR_INFEASIBLE", "BASELINE_INFEASIBLE")
    else:
        assert all(abs(v) < 1e-9 for v in info["payoff"].values())
        assert np.allclose(info["charge_mwh"].values(), 0.)


def test_direction_consistency_flags_a_flipped_leg():
    directions = {"ESS9": np.array([1, -1])}
    idle = {"ESS9": np.array([False, False])}
    same = {"charge": {"ESS9": np.array([0.2, 0.0])},
            "discharge": {"ESS9": np.array([0.0, 0.3])}}
    flipped = {"charge": {"ESS9": np.array([0.0, 0.3])},
               "discharge": {"ESS9": np.array([0.0, 0.0])}}
    assert directions_consistent(directions, idle, same) is True
    assert directions_consistent(directions, idle, flipped) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_local_da_env_v7.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_da_env'`.

- [ ] **Step 3: Implement the clearing adapter**

`clear_window` builds `ClearingModel(day, exact=True)`, solves, and maps outcomes to codes: not optimal → `CLEAR_INFEASIBLE`; `choose_directions` raising `ValueError` → `DIRECTIONS_FAILED`; the convex solve not optimal or missing `lmp_cny_mwh` → `PRICE_SOLVE_FAILED`. `directions_consistent` compares `sign(charge - discharge)` on non-idle periods and requires both legs within `tolerance` on idle periods.

- [ ] **Step 4: Implement the environment**

`LocalDABiddingEnv.__init__` builds the four `Agent` objects (`name=f"ESS{bus}"`, `bus`, `is_prosumer=False`, zero load/PV/real arrays of length 96, `bid_value=0.`, `offer_cost=0.`, `StorageSpec(e_max=1.6, p_ch_max=.4, p_dis_max=.4, eta_ch=.92, eta_dis=.92, soc0=.5, soc_min=.1, soc_max=.9)`, `participant_type="storage"`) and calls `super().__init__(agents, None, roll_horizon=roll_horizon, obs_spec=OBS_V7, action_spec=ACTION_STORAGE_QUOTE_V5)`. `self.pair` and the SOC state are set in `reset`.

`reset(pair)` sets `self.pair`, clears `_last_result`, `_last_base_result`, `_prev_block_actions`, `_prev_block_index`, `block_idx = 0`, `prev_soc`, `hist_lmp = deque(maxlen=96)`, `bus_lmp_history = {nm: [] for nm in agent_names}`, `probe_records = []`, resets each forecaster's `_ema`/`_history`, sets `current_actions = self._default_actions()`, and returns block 0 observations built with `hist_lmp=None` and `avg_lmp_hist=PRICE_REF_CNY_MWH`.

`step(actions)`:
1. Snapshot the block's quotes into `self._prev_block_actions` as full-day arrays (`np.full(self.T, value)`), so the inherited `_compute_opponent_features` and `_get_agent_obs` read the block's slice.
2. `t_start = block * BLOCK_SIZE`; `n_commit = min(BLOCK_SIZE, self.T - t_start)`; `t_end = min(t_start + self.roll_horizon, self.T)`.
3. Build `window_day(self.pair, t_start, t_end, soc=self.prev_soc, quotes={nm: actions[nm]})` — `prev_soc` is `0.5` for every unit on block 0 — and `clear_window(day)`; on `failure` raise `EpisodeClearingError(failure_code, self.pair["load_date"], block)`.
4. Re-clear the same window with every unit's `quotes` replaced by `ACTION_STORAGE_QUOTE_V5.truthful_array()`; a failure raises `EpisodeClearingError("BASELINE_INFEASIBLE", ...)`.
5. Ledger payoff per unit over the committed periods from the conditional pricing result: `revenue = dt * sum(lmp[t,bus]*dis[t])`, `purchase = dt * sum(lmp[t,bus]*ch[t])`, `degradation = dt * 100 * sum(ch + dis)`, `payoff = revenue - purchase - degradation`. The baseline arm uses its own clearing the same way. (The eta-weighted degradation in the design's section D is not the ledger's; `local_da_accounting.settle_day` charges throughput, and the reward must be the settled payoff.)
6. `prev_soc[nm] = pricing["energy"][nm][n_commit] / 1.6`; update each forecaster with the committed cross-node mean LMP (`self.forecasters[nm].update(mean)`).
7. `_update_price_history(pricing, n_commit)` — override it to append the cross-node mean to `hist_lmp` exactly as the parent does and, per agent, the committed LMPs at that agent's bus to `bus_lmp_history`.
8. `block_idx += 1`; `done = block_idx >= N_BLOCKS`; build the next observation (or zeros when done).

`_compute_obs_features(agent, block_idx, hist_lmp, avg_lmp_hist, system_load_re_ratio=1.0, avg_other_dev=0.0, all_dev_std=0.0)` calls the inherited method and adds `lmp_bus_norm = clip(last_bus / PRICE_REF_CNY_MWH, -1., 3.)` and `avg_lmp_bus_hist_norm = clip(mean(bus_history) / PRICE_REF_CNY_MWH, -1., 3.)`, each falling back to `1.0` when the unit has no committed period yet (the same no-history convention as `last_lmp_norm`).

`info` carries `payoff`, `baseline_payoff`, `charge_mwh`, `discharge_mwh`, `lmp` (committed periods × 33), `directions`, `direction_unstable`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_local_da_env_v7.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rl_da_env.py tests/test_local_da_env_v7.py
git commit -m "Clear a training block on the local day-ahead engine"
```

---

### Task 5: Pay the settled payoff, once per day

**Files:**
- Modify: `src/rl_da_env.py`
- Test: `tests/test_local_da_env_v7_reward.py`

**Interfaces:**
- Consumes: Task 4's `LocalDABiddingEnv`, and Global Constraints' 1.6 MWh / 0.5 SOC.
- Produces:
  - `block_payoffs(day, clearing, n_commit) -> dict[str, dict]` — per unit `revenue_cny`, `purchase_cny`, `degradation_cny`, `payoff_cny`.
  - `inventory_value(day, clearing, *, e0_mwh) -> dict[str, float]` — `(energy[-1] - e0_mwh) * external_price[-1]` per unit.
  - `step` adds the inventory term to `payoff` and `baseline_payoff` in the day-end block only, and exposes it as `info["inventory_cny"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_da_env_v7_reward.py
import numpy as np

from local_da_days import load_pair
from rl_da_env import BLOCK_SIZE, LocalDABiddingEnv

TRUTHFUL = np.array([0., 0.4, 0., 0.4])


def truthful(env):
    return {nm: TRUTHFUL.copy() for nm in env.agent_names}


def walk(env):
    infos = []
    env.reset(load_pair("2021-06-04"))
    for _ in range(24):
        _, _, done, info = env.step(truthful(env))
        infos.append(info)
        if done:
            break
    return infos


def test_the_declared_point_pays_exactly_its_baseline():
    env = LocalDABiddingEnv()
    infos = walk(env)
    assert len(infos) == 24
    for info in infos:
        for nm in env.agent_names:
            assert abs(info["payoff"][nm] - info["baseline_payoff"][nm]) < 1e-9


def test_one_inventory_term_per_episode_at_the_declared_price():
    env = LocalDABiddingEnv()
    infos = walk(env)
    non_zero = [i for i, info in enumerate(infos)
                if any(abs(v) > 1e-9 for v in info["inventory_cny"].values())]
    assert non_zero == [23]
    e96 = infos[23]["inventory_cny"]["ESS9"] / env.pair["external_price"][-1] + 1.6 * 0.5
    assert 0.1 * 1.6 <= e96 <= 0.9 * 1.6


def test_the_degradation_is_throughput_times_the_declared_rate():
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    _, _, _, info = env.step(truthful(env))
    for nm in env.agent_names:
        assert abs(info["degradation_cny"][nm]
                   - 100. * 0.25 * (info["charge_mwh"][nm]
                                    + info["discharge_mwh"][nm])) < 1e-6


def test_a_carried_soc_on_the_band_edge_keeps_the_band():
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    env.prev_soc = {nm: 0.9 for nm in env.agent_names}
    _, _, _, info = env.step(truthful(env))
    assert all(0.1 - 1e-6 <= s <= 0.9 + 1e-6 for s in env.prev_soc.values())
    assert "ESS9" in info["payoff"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_local_da_env_v7_reward.py -v`
Expected: FAIL — `KeyError: 'inventory_cny'` and, before the inventory term exists, `test_the_declared_point_pays_exactly_its_baseline` passes only by accident; it must be re-read after the term lands.

- [ ] **Step 3: Implement**

Move the per-period ledger arithmetic out of `step` into `block_payoffs(day, clearing, n_commit)` (it must not import `settle_day`, which refuses an unqualified day), and add `inventory_value` with `e0_mwh = 1.6 * 0.5` taken from the environment, not recomputed from the window. Add both to the payoff only when `t_start + n_commit == self.T`; `info["inventory_cny"]` is `{}`-free zeros on every other block. Apply the same term to the baseline arm from its own `energy[-1]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_local_da_env_v7_reward.py tests/test_local_da_env_v7.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rl_da_env.py tests/test_local_da_env_v7_reward.py
git commit -m "Value terminal inventory once, in the day-end block"
```

---

### Task 6: Probe block 11 and count aborts

**Files:**
- Modify: `src/rl_da_env.py`
- Test: `tests/test_local_da_env_v7_probe.py`

**Interfaces:**
- Consumes: `local_da_clearing.clear_day(day, *, seconds, price_probes, log_path)`.
- Produces: `LocalDABiddingEnv.probe_block(block: int = PROBE_BLOCK) -> dict` — keys `date`, `block`, `price_qualification`, `idle_direction_stable`, `qualified_for_settlement`, `failure` (`None` or `"PROBE_FAILED"`); `LocalDABiddingEnv.probe_records: list[dict]`; `abort_count(records: Sequence[dict]) -> dict` with `episodes`, `aborts`, `degraded` (aborts > 5% of episodes).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_da_env_v7_probe.py
import numpy as np

import rl_da_env
from local_da_days import load_pair
from rl_da_env import PROBE_BLOCK, LocalDABiddingEnv, abort_count

TRUTHFUL = np.array([0., 0.4, 0., 0.4])


def test_a_day_records_exactly_one_probe_at_block_eleven():
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    for _ in range(24):
        _, _, done, _ = env.step({nm: TRUTHFUL.copy() for nm in env.agent_names})
        if done:
            break
    assert len(env.probe_records) == 1
    assert env.probe_records[0]["block"] == PROBE_BLOCK
    assert env.probe_records[0]["date"] == "2021-06-04"


def test_a_failed_probe_is_recorded_and_does_not_abort(monkeypatch):
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    monkeypatch.setattr(rl_da_env, "clear_day",
                        lambda day, **kw: {"price_qualification": "FAILED",
                                           "idle_direction_stable": False,
                                           "qualified_for_settlement": False})
    record = env.probe_block()
    assert record["failure"] == "PROBE_FAILED"
    assert env.probe_records == []


def test_more_than_five_percent_aborts_degrade_the_round():
    assert abort_count([{"abort": False}] * 19 + [{"abort": True}])["degraded"] is True
    assert abort_count([{"abort": False}] * 20 + [{"abort": True}])["degraded"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_local_da_env_v7_probe.py -v`
Expected: FAIL with `AttributeError: 'LocalDABiddingEnv' object has no attribute 'probe_block'`.

- [ ] **Step 3: Implement**

`rl_da_env` imports `clear_day` at module level so the test can replace it. `probe_block` rebuilds the block's window from the current quotes and calls `clear_day(day, seconds=self.seconds, price_probes=True)`, returning the record with `failure="PROBE_FAILED"` when `qualified_for_settlement` is false; it never raises. `step` calls it once, when `block == PROBE_BLOCK`, and appends to `probe_records`. `abort_count` is a pure function of per-episode records with `abort` flags; `degraded = aborts / episodes > 0.05`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_local_da_env_v7_probe.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rl_da_env.py tests/test_local_da_env_v7_probe.py
git commit -m "Sample one probed block per training day"
```

---

### Task 7: Train on the v5 action

**Files:**
- Create: `src/rl_da_gate_v7.py`
- Test: `tests/test_rl_da_gate_v7.py`

**Interfaces:**
- Consumes: `rl_b_c2_gate.GateReplayBuffer`, `rl_b_c2_gate.MATD3BC2Gate`, `rl_da_gate.DAExperimentConfig`, `rl_bidding.CentralizedCritic`, `checkpoint_loader_v2.load_checkpoint`, `c2_reward.c2_reward`, `c2_reward.C2Config`, `da_quote_map_v5.*`, `rl_spec_v7.*`, `rl_da_env.LocalDABiddingEnv`.
- Produces:
  - `GateActorV5(nn.Module)` — obs → 256 → 128 → 6, ReLU.
  - `DAV7ExperimentConfig(DAExperimentConfig)` — `experiment_id="MATD3_DA_GATE_v7"`, `action_schema="storage_quote_v5"`, and its own `__post_init__`: the parent raises unless the schema is `storage_quote_v4`, so the subclass repeats those checks with the v5 literal and inherits every other field.
  - `DAGateTrainerV7(MATD3BC2Gate)` with `act_dim = QUOTE_DIM` (4), `train_day(env, pair, *, blocks=N_BLOCKS) -> dict`, `save`, `load` (refuses any checkpoint whose `manifest.specs.action_spec.name != "storage_quote_v5"` or `manifest.specs.obs_spec.name != "v7_12d"`), `_anchor_tensor(quotes) -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_da_gate_v7.py
import numpy as np
import torch

from checkpoint_writer_v2 import save_checkpoint
from da_quote_map_v5 import ACTION_BOX_HIGH, QUOTE_DIM, TRUTHFUL
from local_da_days import load_pair
from rl_da_env import LocalDABiddingEnv
from rl_da_gate_v7 import DAGateTrainerV7, DAV7ExperimentConfig, GateActorV5
from rl_spec_v7 import ACTION_STORAGE_QUOTE_V5, OBS_V7


def trainer(blocks=2):
    env = LocalDABiddingEnv()
    return DAGateTrainerV7(env, DAV7ExperimentConfig()), env


def test_the_actor_emits_six_logits_with_a_four_wide_box():
    env = LocalDABiddingEnv()
    trainer = DAGateTrainerV7(env, DAV7ExperimentConfig())
    assert trainer.act_dim == QUOTE_DIM == 4
    assert list(trainer.actors["ESS9"](torch.zeros(1, 12)).shape) == [1, 6]
    assert isinstance(trainer.actors["ESS9"], GateActorV5)
    assert trainer.action_spec.name == "storage_quote_v5"
    assert trainer.obs_spec.name == "v7_12d"


def test_the_anchor_is_zero_at_the_declared_point_and_one_at_the_far_corner():
    t, _ = trainer()
    truth = torch.tensor([list(TRUTHFUL)], dtype=torch.float32)
    corner = torch.tensor([[ACTION_BOX_HIGH[0], 0., ACTION_BOX_HIGH[2], 0.]])
    assert t._anchor_tensor(truth).item() == 0.
    assert t._anchor_tensor(corner).item() == 4.


def test_one_day_writes_every_block_to_the_buffer():
    t, env = trainer()
    record = t.train_day(env, load_pair("2021-06-04"), blocks=2)
    assert len(t.buffer) == 2
    assert len(record["per_block"]["ESS9"]) == 2
    assert t.episodes == 1


def test_a_v4_checkpoint_is_refused(tmp_path):
    t, _ = trainer()
    save_checkpoint(
        tmp_path, level="inference", run_id="v4", dataset="d", seed=1,
        code_revision="rev", config_hash="cfg",
        specs={"obs_spec": {"name": "v6_12d", "version": 6},
               "action_spec": {"name": "storage_quote_v4", "version": 4}},
        actors=t.actors, actor_targets=t.actor_targets, critics=t.critics,
        critic_targets=t.critic_targets,
        optimizers={}, counters={"total_steps": 0, "episode": 0, "updates": 0},
        reward={"definition": "B_C2", "gamma": 0.99, "scale": 100.})
    try:
        t.load(tmp_path)
    except ValueError as error:
        assert "storage_quote_v5" in str(error)
    else:
        raise AssertionError("a v4 checkpoint must not load into a v7 trainer")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_rl_da_gate_v7.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_da_gate_v7'`.

- [ ] **Step 3: Implement the trainer**

`DAGateTrainerV7.__init__` calls `super().__init__(env, experiment, obs_spec=OBS_V7, action_spec=ACTION_STORAGE_QUOTE_V5)`, then replaces the actor maps with `GateActorV5`, the critic maps with `CentralizedCritic(n_agents, obs_dim, QUOTE_DIM)`, and the optimizers in `self.actor_opts` / `self.critic_opts`, and sets `self.act_dim = QUOTE_DIM`. The parent sizes those heads from the v4 constants; rebuilding here is why the pinned file is not edited.

Override `_policy_action` (the inherited noise already lands on `policy_out[..., 2:]`, which is the four quote logits; only the epsilon and calibration exploration need the v5 redraw — a price over `LEG_HIGHS` and a quantity over `(0, 0.4)`, then `forward_policy_v5`), `_target_quotes` (same noise slice, then `target_quotes_v5`), and `_actor_loss_quotes` (`forward_policy_v5(train=True, gate_temperature=current)`).

Override `update()` by copying the parent's body with three changes: the anchor is `self._anchor_tensor(quotes_i)`, i.e. `((quotes - TRUTHFUL) / ACTION_BOX_HIGH) ** 2` summed over the four dimensions and averaged over the batch; the hinge's truthful counterfactual writes `TRUTHFUL` into the agent's slice instead of zeros; the twin-gap hinge and critic losses are unchanged. Comment says the parent hard-codes the two-dimension truth and is pinned.

`train_day(env, pair, *, blocks=N_BLOCKS)` resets the env on `pair`, increments `self.episodes`, walks `blocks` blocks, collects `info["payoff"]`, `info["baseline_payoff"]` and the SOC path, and builds the day reward as `c2_reward(diff, soc_path, done=done_flags, config=C2Config(gamma=self.experiment.gamma))` divided by `fixed_scale`, where `diff[nm][b] = payoff[nm][b] - baseline_payoff[nm][b]`. It writes one transition per block with the state before the decision and the next observations. `blocks` exists so a test can run two blocks without paying for a day; a formal day is the default. An `EpisodeClearingError` propagates; the buffer keeps nothing from that day.

`save` mirrors the parent with `policy_dim=6`, `quote_dim=4`, `leg_highs=[150, 0.4, 300, 0.4]`. `load` calls `load_checkpoint(root)`, compares `bundle["manifest"]["specs"]` against `storage_quote_v5` / `v7_12d`, and raises a `ValueError` naming the checkpoint's spec before delegating to `super().load(root)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_rl_da_gate_v7.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rl_da_gate_v7.py tests/test_rl_da_gate_v7.py
git commit -m "Train the v5 quote with the settled payoff and the v5 anchor"
```

---

### Task 8: Smoke three days and measure the cost

**Files:**
- Create: `src/smoke_local_da_v7.py`
- Test: `tests/test_smoke_local_da_v7.py`
- Evidence: `reports/local_da_v7_2026-11-18/smoke.json`, `RESULTS.md`

**Interfaces:**
- Consumes: `local_da_days.load_pair`, `train_pairs`; `rl_da_env.LocalDABiddingEnv`, `ROLL_HORIZON`; `rl_spec_v7.ACTION_STORAGE_QUOTE_V5`.
- Produces: `smoke(days: Sequence[str], *, seconds=120.0) -> dict` — keys `episodes` (one record per day with `rewards_finite`, `clears`, `soc_min`, `soc_max`, `declared_point_deviation`, `inventory_terms`, `probe`, `block_seconds`), `latency` (`exact`, `price`, `baseline`, `probe_day`, `episode`), and `summarize(record) -> dict` with the four acceptance flags `rewards_finite`, `no_none_clears`, `soc_in_band`, `one_inventory_term`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke_local_da_v7.py
from smoke_local_da_v7 import summarize


def record(**over):
    base = {"rewards_finite": True, "clears": 24, "none_clears": 0,
            "soc_min": 0.32, "soc_max": 0.71, "declared_point_deviation": 0.0,
            "inventory_terms": 1}
    base.update(over)
    return base


def test_a_clean_day_passes_every_acceptance_flag():
    out = summarize(record())
    assert out == {"rewards_finite": True, "no_none_clears": True,
                   "soc_in_band": True, "one_inventory_term": True,
                   "declared_point_deviation": 0.0}


def test_a_missing_clear_and_a_second_inventory_term_fail_their_flags():
    out = summarize(record(none_clears=1, inventory_terms=2, soc_max=0.95))
    assert out["no_none_clears"] is False
    assert out["one_inventory_term"] is False
    assert out["soc_in_band"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_smoke_local_da_v7.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smoke_local_da_v7'`.

- [ ] **Step 3: Implement**

The CLI takes `--days` (default: the first three dates of `train_pairs`, in manifest order so a rerun picks the same days), `--seconds`, `--out`. For each day it walks the environment with the truthful action, times one exact solve, one convex price solve, one baseline re-clear and one `probe_block`, and writes `smoke.json`. `summarize` implements the four flags above; `declared_point_deviation` is passed through and must be `0.0`. Compare `ROLL_HORIZON`'s measured episode time against the v6 budget in `training_contract.py`; if it exceeds it, re-run at `roll_horizon=12` (lookahead 8) and record both numbers rather than picking one.

- [ ] **Step 4: Run the test, then the smoke**

Run: `python -m pytest tests/test_smoke_local_da_v7.py -v`
Expected: 2 passed.

Run: `PYTHONPATH=src python -B src/smoke_local_da_v7.py --out reports/local_da_v7_2026-11-18`
Expected: three episode records, every acceptance flag true, and a `latency` block with five measured entries. Write `RESULTS.md` with one row per acceptance item and a verdict from the command output.

- [ ] **Step 5: Commit**

```bash
git add src/smoke_local_da_v7.py tests/test_smoke_local_da_v7.py reports/local_da_v7_2026-11-18
git commit -m "Measure the local-engine block cost on three training days"
```

---

### Task 9: Audit the price stack and the deviation weight

**Files:**
- Create: `src/local_da_audit.py` (the pure comparisons), `tools/audit_local_da_v7.py` (the CLI that measures and writes), `src/calibrate_dev_penalty_v7.py`
- Test: `tests/test_local_da_audit_v7.py`
- Evidence: `reports/local_da_v7_2026-11-18/audit/`, `RESULTS.md` (appended)

`tools/` is not a package and no test imports it, so the aggregations live in `src/` where `pythonpath = src` reaches them; the tool only runs them.

**Interfaces:**
- Consumes: `local_da_days.load_pair`, `window_day`; `local_da_operating.support_day`, `OperatingRules`; `da_market_data.MarketRules`, `DEGRADATION_CNY_MWH`; `da_market_model.MarketModel`; `da_market_pricing.qualify_prices`; `run_local_da.build_day`; `local_da_clearing.clear_day`; `calibrate_dev_penalty.LAMBDA_GRID`, `select`; `rl_da_gate_v7.DAGateTrainerV7`.
- Produces:
  - `compare_stacks(training_lmp, audit_lmp, *, floor=1.0) -> dict` — `sign_agreement`, `median_rel_gap`, `cells` over the (period, bus) pairs whose audit price exceeds `floor`.
  - `rolling_gap(day_lmp, block_lmps, committed) -> dict` — per-period relative gap plus `median`, `p95`.
  - `eval_counts(days, results) -> dict` — `attempted`, `evaluated`, `infeasible` (named).
  - `feasibility_selected(screen_dir) -> dict` — `counts` per block plus the `dropped` map from `train_pairs`; it is the recorded form of that same reading, not a second pass over the screen.
  - `calibrate_dev_penalty_v7.calibrate(seed=123) -> dict` with `gradient_share` and `selected_weight`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_da_audit_v7.py
import numpy as np

from local_da_audit import compare_stacks, eval_counts, rolling_gap


def test_sign_and_gap_bounds_are_computed_on_the_cells_above_the_floor():
    training = np.array([[0.5, -2.0], [100.0, 300.0]])
    audit = np.array([[0.4, -2.5], [110.0, 250.0]])
    out = compare_stacks(training, audit, floor=1.0)
    # The 0.4 cell falls under the floor; the other three agree in sign and
    # their relative gaps are 0.2, 0.0909..., 0.2.
    assert out["cells"] == 3
    assert out["sign_agreement"] == 1.0
    assert abs(out["median_rel_gap"] - 0.2) < 1e-9


def test_a_sign_flip_drops_the_agreement_below_one():
    out = compare_stacks(np.array([[100.0]]), np.array([[-100.0]]))
    assert out["sign_agreement"] == 0.0


def test_the_rolling_gap_reports_median_and_p95():
    day = np.array([100.0, 100.0, 100.0, 100.0])
    blocks = np.array([110.0, 90.0, 120.0, 80.0])
    out = rolling_gap(day, blocks, committed=4)
    assert out["median"] == 0.15
    assert abs(out["p95"] - 0.185) < 1e-9


def test_the_eval_counts_name_the_infeasible_day():
    out = eval_counts(["2022-08-01", "2022-08-02"],
                      {"2022-08-01": {"settled": True},
                       "2022-08-02": {"settled": False}})
    assert out["attempted"] == 2 and out["evaluated"] == 1
    assert out["infeasible"] == ["2022-08-02"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_local_da_audit_v7.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the comparison and counts**

`compare_stacks` masks cells by `abs(audit) > floor`, computes `mean(sign(training) == sign(audit))` and the median of `|training - audit| / |audit|`. `rolling_gap` takes the absolute relative gap per committed period and returns `median` and the 95th percentile (linear interpolation). `eval_counts` and `feasibility_selected` are pure aggregations over recorded results; neither invents a verdict.

- [ ] **Step 4: Implement the audit run**

For each smoke day, build the full day `[0, 96)` from the same pair, with SOC 0.5 and truthful quotes, keeping the library's own active and reactive curves so only the execution layer differs. Then `day = dataclasses.replace(day, terminal_soc_equal=True, storage=tuple(dataclasses.replace(u, degradation_cny_mwh=da_market_data.DEGRADATION_CNY_MWH) for u in day.storage))`, `day, ratings = support_day(day, OperatingRules())`, append `Generator("EXTERNAL33", rules.external_q_bus, np.zeros(day.periods), reactive_limit_mvar=rules.external_q_capacity_mvar)` with `rules = da_market_data.MarketRules()`, and `rules.validate(day)`. Do not route this through `da_market_data.load_day`, which rebuilds the curves from a different source. Clear it with `MarketModel(day, ratings, rules, exact=True)`, then `qualify_prices(day, ratings, rules, dispatch, backend="cutting")`. Record `audit_day_construction` (terminal SOC equal, external reactive device, degradation 80, inverter ratings) so the report never presents the two constructions as one. Compare against the training-path LMPs at the storage buses over the same periods with `compare_stacks`.

Acceptance 4 passes only if `sign_agreement >= 0.95` and `median_rel_gap <= 0.10`; anything else is `FAIL`. If `qualify_prices` refuses the dispatch, record `BLOCKED` with its `failure` string and do not substitute another backend.

For Acceptance 5, rebuild every block of the three smoke days with truthful quotes and record its window LMPs; clear each day with `build_day` + `clear_day` and compare the same periods. For Acceptance 6, run `calibrate_dev_penalty_v7`, which imports `LAMBDA_GRID` from `calibrate_dev_penalty` and measures the anchor-gradient share against the critic term on a fixed v7 day with no learning; a share above 0.50 is `FAIL`. For Acceptance 7 and 8, read the smoke records and `feasibility_selected`.

- [ ] **Step 5: Run the audit and append the verdicts**

Run: `python -m pytest tests/test_local_da_audit_v7.py -v`
Expected: 4 passed.

Run: `PYTHONPATH=src python -B tools/audit_local_da_v7.py --smoke reports/local_da_v7_2026-11-18/smoke.json --out reports/local_da_v7_2026-11-18/audit`
Expected: `audit/summary.json` with the four acceptance blocks. Append Acceptance 4–8 rows to `RESULTS.md`, each with its verdict and the value read from `summary.json`.

- [ ] **Step 6: Run the whole suite and commit**

Run: `python -m pytest tests/ -q`
Expected: all pass, including the pinned tests. Record the count in the commit.

```bash
git add src/local_da_audit.py tools/audit_local_da_v7.py src/calibrate_dev_penalty_v7.py tests/test_local_da_audit_v7.py reports/local_da_v7_2026-11-18
git commit -m "Audit the training price stack and the re-derived anchor weight"
```

## Handoff

The plan ends before the protocol freeze. It produces the code, the three-day smoke, the latency measurement, the price-stack audit and the failure counts that a v7 manifest and a re-freeze decision need; it does not create `reports/v2_cy_jn_2025/formal_training_v7/`, does not edit `training_contract.py`, and does not launch a formal run.
