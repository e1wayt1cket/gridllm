# P2-WINTER-RANK — does the critic rank the payoff it is trained on

Protocol `P2_CONTROLLED_DIAGNOSTICS_V2`, stage 3 (offline, read-only).
Written 2026-11-08. Interpreter
`C:\Users\26036\.conda\envs\energy_env\python.exe` (CPython 3.10.20),
torch `2.11.0+cpu`, numpy `1.24.3`, scipy `1.15.3`, gurobipy `13.0.1`.

Collector: `src/diagnose_p2.py` (new, unpinned), functions `trace_day`,
`fixed_states`, `activity_states`, `sweep_states`, `summarize_rank`.
Commands:

```
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" \
  src/diagnose_p2.py rank
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" \
  src/diagnose_p2.py rankact
```

Artifacts: `P2_evidence.json` → `rank`, `rank_activity`;
`P2_winter_critic_ranking.csv`; `P2_winter_critic_ranking_activity.csv`.

Checkpoint: seed 23 best (`runs/FORMAL_1b47e0ed_seed23_best`, step 5,088,
episode 212) for every measurement below. Training data only
(`2022-12-09|2025-12-09`, `2022-12-10|2025-12-07`). No validation label is
fitted, no test or holiday pair is read, the buffer is not written, and no
checkpoint is saved.

## Method

For each selected state the deterministic policy is run forward from
`env.reset()` to that block, recording observation, SOC and the policy's own
action. One leg of one unit is then swept over its five-point quote grid
(charge `{0, 37.5, 75, 112.5, 150}` CNY/MWh, discharge `{0, 75, 150, 225,
300}`), every other decision staying with the policy, and the **remaining day**
is rolled out with the real clearing and the terminal SOC pinned at 0.5.

Two quantities are compared at the same state and the same swept quote:

- `Q_min(state, a)` — the twin critic's minimum at the pre-action state with
  the swept quote substituted, no rollout;
- `payoff_remaining_unit` — the swept unit's own settled profit over the rest
  of the day, summed from the environment's per-block `raw_profit`.

`Spearman(Q_min, payoff)` within a state, top-1 agreement (argmax quote, tie
tolerance `1e-6`), pairwise ordering accuracy over points whose payoff differs
by more than `1e-6`, the twin gap `|Q1 - Q2|`, and the sign agreement between
the midpoint `dQ/dquote` and the payoff direction. Thresholds are the
pre-registered ones in `thresholds.json` and are not re-chosen here.

## Arm 1 — the pre-registered fixed blocks (primary)

States: 2 days × blocks `{4, 10, 16}` × units `{ESS14, ESS27}`, leg chosen by
the block (charge at 4 and 16, discharge at 10) and switched to the other leg
only if the policy's chosen leg had a shut gate. All 12 states ended
gate-open; none required a switch.

| metric | value |
|---|---|
| states | 12 open, 0 gate-closed |
| median within-state Spearman | `0.15` |
| top-1 agreement | `0.1667` (2 / 12) |
| pairwise accuracy | `0.4860` (52 / 107) |
| mean twin gap `|Q1-Q2|` | `0.004024` |
| gradient-sign agreement | `0.4146` (34 / 82) |
| median payoff range across the grid | `0.000201` CNY/day |
| states with payoff range ≥ 1.0 CNY/day | **2 / 12** |
| pre-registered `no_signal` rule | **TRIGGERED** (10 / 12 = 83% below 1.0) |
| `ranking_verdict` field | `invalid` |

### This arm does not measure ranking — it measures that the sweep has no economic effect

The pre-registered no-signal rule fires, so the `ranking_verdict: invalid`
value is **not** evidence that the critic cannot rank. At 10 of the 12 states
the quote change leaves the cleared schedule untouched: `accept_mwh` at the
swept leg is `1e-9`–`1e-7` MWh at all five grid points, i.e. the leg clears no
energy at any quote, so the rest of the day is bitwise the same day and the
payoff is constant to numerical noise. Per-state Spearman computed on those
five constants is a function of `1e-12` rounding, not of the critic, and is
reported only to show it exists.

The scale mismatch is the point of this arm:

| quantity | median over the 12 states |
|---|---|
| `Q_min` range across the five quotes | `0.00711` |
| payoff range across the five quotes | `0.000201` |
| ratio | `46.3` |
| pooled within-state-centred Spearman (all 60 points) | `0.078` |

`Q` moves by ~46× more than the payoff it is supposed to rank, and the twin
critics disagree with each other (`0.00402`) by more than half of that `Q`
range. A ranking test cannot be built on top of a payoff that does not vary.

### The two states where the payoff does vary

Both are ESS27 discharge states, both of which clear energy at the low quotes:

| state | payoff at quotes 0 / 75 / 150 / 225 / 300 | Q_min at the same points |
|---|---|---|
| `2022-12-09` block 16 | `18.24, 15.78, -14.28, -14.28, -14.28` | `-0.00826, -0.00711, -0.01067, -0.00777, -0.00930` |
| `2022-12-10` block 4 | `82.99, 73.27, 73.27, 73.26, 73.27` | `-0.00827, -0.00738, -0.00679, -0.00615, -0.00949` |

In both, the payoff-maximising quote is `0.0` — the ESS sells at the cheapest
declared price and the market accepts the energy — while the critic's argmax is
a **rejected** quote (`75` in the first, `225` in the second). The payoff-best
point ranks 3rd of 5 in Q in the first state and 4th of 5 in the second.

That is a qualitative inversion at the only two comparisons with real money in
them, and it is the reading that survives the no-signal rule. It is **2
states**, so on its own it is not a phase-level conclusion; it motivates
arm 2 rather than replacing it.

## Arm 2 — states selected where the policy's own quote clears (exploratory)

Arm 1's fixed blocks were chosen before measurement but turn out to sit where
the swept leg never clears. Arm 2 selects, per day and per unit and leg, the
two blocks with the largest energy the policy itself commits there (commitment
≥ `0.05` MWh), so the sweep starts from a state where a quote demonstrably
changes the schedule. Selection is **outcome-based** and therefore labelled
exploratory: it is not a pre-registered sample and its rates are not
population estimates.

**Verdict: BLOCKED.** The run selected 11 states (4 of them on
`2022-12-09`) and was stopped by the environment's memory-pressure reaper after
completing 4 states, before it wrote any evidence. `rank_activity` is absent
from `P2_evidence.json`; `P2_winter_critic_ranking_activity.csv` holds 23 of
the expected 55 rows and is **not evidential** — a partial sweep has no
per-state summary and an incomplete state contributes individual points
without its ordering. The file will be rewritten in full if the arm is re-run.

Consequence for this report: the only measured ranking evidence is arm 1's
two informative states, which is below the pre-registered `top1` sample and
below the "10–20 states" the protocol asked for. The critic line is therefore
**NOT_TESTED**, not "the critic cannot rank".

What would settle it: the same sweep on the activity-selected states, i.e.

```
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" \
  src/diagnose_p2.py rankact
```

## Summary

| arm | states | payoff varies | ranking measurable | verdict |
|---|---:|---:|---|---|
| 1, pre-registered fixed blocks | 12 | 2 | no | **NOT_TESTED** (pre-registered `no_signal`) |
| 2, activity-selected | 11 selected, 4 completed | — | — | **BLOCKED** |

What the critic evidence does support (**INFERRED**, from 2 states): at both
places where the swept quote changed the cleared energy, the twin critics
preferred a quote that captures no energy to the quote that maximises the
day's payoff, and the payoff-maximising quote ranked 3rd and 4th of 5 in Q.
That is consistent with candidate B (critic ranking) and is not yet a
measurement of it.

