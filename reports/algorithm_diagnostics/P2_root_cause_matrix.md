# P2 root-cause matrix

Protocol `P2_CONTROLLED_DIAGNOSTICS_V2`, closing stage (offline, read-only).
Written 2026-11-08. Interpreter
`C:\Users\26036\.conda\envs\energy_env\python.exe` (CPython 3.10.20),
torch `2.11.0+cpu`, numpy `1.24.3`, scipy `1.15.3`, gurobipy `13.0.1`.

**Exit: `NO_GO_ROOT_CAUSE_UNRESOLVED`.**

The phase produced one large measured effect and two negative results, but no
candidate cause is supported by two independent evidence lines. P2-TERMINAL
was deferred as instructed and is not a missing line: none of its three
triggers appeared.

## Evidence lines

| # | line | artifact | measured value | verdict |
|---|---|---|---|---|
| L1 | τ mechanism | `P2_temperature_mechanism.md`, `P2_evidence.json` → `temp` | gate forward, hard draw and both gradients depend on τ exactly as `p(1−p)/τ`; eval/target bitwise τ-free; max error `1.07e-14` | **PASS** |
| L2 | τ as a control | `P2_control_matrix_report.md`, cell A1 | τ contrasts `+0.04 [−0.10,+0.20]`, `+0.34 [−0.03,+0.84]`, `+0.30 [+0.02,+0.67]` CNY/day against a `−5.1` mean; sign of τ4−τ2.5 flipped between the two runs (`−0.09` → `+0.30`) | **PASS** (no stable effect) |
| L3 | replay buffer | same report, cells A2/A3 | content at fixed size `+4.19 [−0.63,+10.55]`; size at fixed content `+2.05 [−0.03,+4.94]`; winter days `+17.0`/`+19.1` CNY, autumn `−2.5` | **PASS** (effect exists), **INFERRED** (direction of causation) |
| L4 | critic ranking | `P2_winter_critic_ranking.md`, `P2_evidence.json` → `rank` | 12 fixed states: no-signal rule triggered (payoff range `< 1` CNY in 10/12); Q range 46× the payoff range; 2 informative states both rank the payoff-best quote 3rd/4th of 5 | **NOT_TESTED**; exploratory arm **BLOCKED** |
| L5 | market signal | `P2_control_matrix.csv`, `P2_winter_critic_ranking.csv` | 5 of 8 Train days move `< 0.01` CNY under a 10-step policy update; `accept_mwh < 1e-6` at 10 of 12 swept states | **PASS** (measured property) |

## Candidates

### A — actor gradient / gate straight-through

Two independent measurements contradict A as the driver of the degradation:

1. the temperature is a real multiplicative factor on the gate gradient
   (L1, exact), and varying it over `1.0 … 4.0` at fixed everything moves the
   day's payoff by at most `0.34` CNY against a `−5.1` mean (L2), with no
   stable sign;
2. the actor step itself does not fail uniformly: it degrades the two winter
   days by `17–22` CNY while five other Train days move by `< 0.01` (L5).

The second point is not a defence of the actor — it says the actor's step
*direction* is harmful where the market responds — but it rules out "the gate
gradient is mis-scaled by τ" as the mechanism. Verdict for A: **not
supported**; the remaining actor-side question (does the ascent direction
lower true payoff) is L3's, and L3 is attributed to the buffer rather than to
the map.

The P1-B actor-step measurements that would speak to A directly were
collected with the harness defect described below and stay **INCONCLUSIVE**.

### B — critic ranking / value estimation

One line only, and it is short: at the two states where the swept quote
changes cleared energy, the twin critics' argmax is a quote that captures
nothing while the payoff-maximising quote ranks 3rd and 4th of 5 (L4). Q moves
`0.0071` across a five-point sweep where the payoff moves `0.0002`, and the
two critics disagree by `0.0040` — over half the Q range being ranked.

That is consistent with B and nowhere near sufficient: 2 states, no
pre-registered threshold met, and the arm designed to give a testable sample
did not complete. **NOT_TESTED.**

### C — reward / C2 objective mismatch

**NOT_TESTED** in this phase. No P2 stage computed the training return or
compared it with the payoff; the analytic identity in the plan
(`γ²³·Δ/1000` with the C2 shaping telescoping to zero) was not re-measured
here. Nothing contradicts C and nothing supports it.

### D — market environment / state representation

L5 is a measured property of the environment: the fleet's quote changes almost
never reach the clearing on most Train days, and at most swept states the leg
declares a price and clears no energy at all. This is a plausible amplifier of
B — a critic trained on returns that barely respond to actions has little
signal to rank — but it is a description of the signal, not a measurement that
it caused the degradation. **INFERRED**, one line.

## Why the phase does not name a cause

The pre-registered rule requires at least two independent evidence lines for a
chosen cause. The counts are:

| candidate | independent lines | strongest |
|---|---|---|
| A | 0 supporting, 2 contradicting (as a τ-scaled gate defect) | L2, L5 |
| B | 1 (2 states, INFERRED) | L4 |
| C | 0 | — |
| D | 1 (INFERRED) | L5 |

The one large, reproducible, paired effect is L3 — the buffer the actor steps
from changes the winter-day payoff by ~20 CNY at fixed checkpoint, τ, step
count and batch seeds. It is a strong *place to look*, and it is not a root
cause yet: the design separates buffer size from buffer content but not
content from exploration history, and a 10-step horizon is not the formal
runs' horizon. Naming it would be inference presented as measurement.

**`NO_GO_ROOT_CAUSE_UNRESOLVED`.**

## Defect found in the harness, and its scope

`torch.optim.Optimizer.load_state_dict` keeps references to the incoming
tensors. `P1.restore_state` restored the actor optimizer that way, so the first
actor step mutated the snapshot's Adam moments in place and later replicates
did not start from the recorded state. Detected by `optimizer_sha` drifting
within one process (`9c75c505… → 1e70c0af… → 29d67916…`) while fresh-process
cells held one value. Fixed by restoring from a `deepcopy`; verified by three
consecutive restore-and-step arms returning a bitwise-identical hash.

Scope: the first P2-MATRIX pass (superseded, kept in
`P2_evidence.json` → `matrix.superseded`) and **all P1-B actor-step
measurements**, which are therefore not a usable evidence line for A. The
replay, checkpoints, thresholds and frozen sources are untouched.

## Not tested / blocked

| item | status | reason | risk carried |
|---|---|---|---|
| critic ranking on activity-selected states | **BLOCKED** | process stopped by the environment's memory reaper after 4 of 11 states | the critic line cannot be closed; B stays at 2 states |
| reward / C2 return identity on these rollouts | **NOT_TESTED** | out of this phase's scope | C stays open |
| P1-B actor step under a correct optimizer restore | **NOT_TESTED** | not re-run | A's direct test is missing; ~35 min of clearing time |
| buffer date coverage | **NOT_TESTED** | replay extras carry no date field | L3 cannot be attributed to which days each buffer holds |
| buffer content vs exploration history | **NOT_TESTED** | size was controlled, history was not | L3's mechanism is undetermined |
| horizons beyond 10 steps | **NOT_TESTED** | short-horizon diagnostic by design | the link from a 20 CNY/day 10-step effect to the formal runs' degradation is **INFERRED** only |
| P2-TERMINAL | **NOT_TESTED** (deferred) | none of its three triggers appeared | terminal-SOC effects remain unexamined |

## What would move this out of NO_GO

In order of evidence gained per unit of compute:

1. re-run the activity-selected ranking sweep (11 states, ~15 min of clearing)
   with the memory pressure resolved — this is the only missing piece that
   could give B a second line;
2. re-run P1-B under the fixed restore (~35 min) — A's direct test;
3. a buffer-content/exploration-history split: same size, same day coverage,
   different exploration history.

None of these is a change to the algorithm, and none requires formal training.
