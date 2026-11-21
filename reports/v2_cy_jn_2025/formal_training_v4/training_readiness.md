# Training readiness — protocol v4 (`7834d0aa`)

Status: **GO_FORMAL_TRAINING**, finalized 2026-11-09.

Scope: formal training on NEAR80 / common_split near view under
protocol_version 4. Test is LEGACY_SEEN_TEST; no independent-generalisation
claim; H2 stays sealed.

## What v4 adds over v3

Three actor-loss mechanisms, each selected by a non-profit rule and covered
by `tests/test_v4_mechanics.py`:

1. **W7 advantage hinge** — the actor loss is
   `-ReLU(Q(a) - Q(a_truthful) - delta_eff) + anchor`. `delta = 0.10` was
   picked by the rule recorded in
   `reports/algorithm_diagnostics/v4_hinge_calibration.json`: round up to
   0.05 the maximum Q variation over states whose payoff range over the
   three lowest quotes is below 0.5 CNY. Never profit-selected.
2. **W8 calibration exposure** — every 8th episode resamples the open-leg
   quotes uniformly over the whole box (`calibration_interval = 8`), so the
   critic sees the full quote range and not only the policy's current mode.
3. **W9 twin-gap-adaptive hinge floor** —
   `delta_eff = clamp(twin_gap_weight * |Q1 - Q2|, min=delta)` with
   `twin_gap_weight = 1.0`: where the twin critics disagree more than the
   noise floor, the hinge widens instead of trusting a point estimate.

Protocol repairs closed with this freeze: TD-01 (log hashed after the
completion line), TD-02 (atomic episode skip via `EpisodeClearingError`),
TD-10 (interpreter recorded in `run_manifest.json`; resume-refusal on
interpreter mismatch still open), TD-11 (launcher `--resume`).

## Pre-training gates (G1 wave, four generations)

| generation | mechanisms | G1b ranking (median / top-1 / verdict) | G1c multi-day |
|---|---|---|---|
| V4A | anchor only | 0.70 / 0.80 / valid | seed23 CUMULATIVE_DEGRADATION |
| V4B | +W7 hinge | -0.10 / 0.14 / invalid | no degradation |
| V4C | +W8 calibration | -0.30 / 0.29 / invalid | no degradation |
| V4D | +W9 twin-gap floor | **0.80 / 0.50 / valid** | **PASS, both seeds** |

G1b is judged under the user-approved 口径 A (2026-11-09): a top-1 vote is
counted only on states whose payoff range over the quote grid clears the
calibration's own flat-payoff tolerance (`flat_payoff_tol_cny = 0.5`), and
the critic's pick counts as agreement whenever its payoff is within that
tolerance of the true best quote. On the V4D evidence that removes 1 of 9
states (payoff range 0.014 CNY) and converts one strict miss into a hit:
top-1 = 4/8 = 0.50, exactly on the gate line; median Spearman 0.80.
The strict count (3/9 = 0.33) stays in the evidence JSON for transparency.
The residual failure modes are documented in
`b/W9双分歧铰链_G1重测_2026-11-09.md`: correlated-error false peaks
(s02/s06, twin_gap < delta, not gated by design) and the autumn day
2022-10-13 (about -30 CNY/day).

## Verification at freeze time

| check | verdict |
|---|---|
| `make_formal_training_manifests.py --check` | ok, no problems |
| `run_formal_training.py --check` | all_pass true, budget_contract_ok |
| `training_contract.py --check` | ok, no violations |
| focused suites (frozen guard, contract, launch guards, resume, v4 mechanics) | 96 passed |
| full suite | 1452 passed, 54 deselected in 468.19s |

The full suite deselects
`tests/test_formal_ensemble.py::test_every_consumer_reads_the_declared_comparison_set`,
a pre-existing environment failure (matplotlib stub in `energy_env`),
unrelated to this protocol.

## Launch

`PYTHONPATH=src python src/run_formal_training.py --seed 11`, then
`--seed 23`, sequentially. Run ids `FORMAL_7834d0aa_seed{11,23}`. The hinge
delta, twin-gap weight and calibration interval are the trainer defaults and
are pinned in the config; the launcher passes only the budget and stop-rule
flags.
