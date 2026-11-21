# Baseline recompute under protocol v6

Verdicts here use `PASS` / `FAIL` / `NOT_TESTED` / `INFERRED` / `BLOCKED`.
Every number names the artifact it was read from.

## Why a recompute, and why an A/B tree

`F1` corrected `dispatch_core._select_radial_lines` to admit only in-service
lines. Every clearing number produced before it describes a network that is not
`case33bw`'s, so the ladder, the typical-day acceptance and the benchmark table
all have to be re-measured.

Comparing a fresh run against an old *artifact* does not isolate `F1`: the
scenarios and the engines also moved between the artifact and now.
`outputs/ladder.json` is from 2026-09-18 and already disagrees with the
pre-`F1` tree (B2/primary `vmin` 0.9583 there, 0.9547 on the pre-`F1` tree), so
an old-vs-new table over that gap would attribute to `F1` a difference `F1` did
not make.

The attribution instrument is therefore two trees that differ in one file:

```
cp -r src _pre_f1_src
git show HEAD~1:src/dispatch_core.py > _pre_f1_src/dispatch_core.py
PYTHONPATH=_pre_f1_src python _pre_f1_src/<script>.py ...
PYTHONPATH=src        python src/<script>.py ...
```

The copy is placed beside `src/`, not under `tmp/`, because these scripts
resolve data paths from their own module's directory; a copy under `tmp/`
looked for the price workbook in `tmp/docs/`. Both copies were deleted after
the runs; the two commands above rebuild them.

Approval precondition: the head of the corrected tree is `32384ad`, and
`HEAD~1:src/dispatch_core.py` is the selector before the correction
(`in_service` filter and the spanning-tree check absent).

## Ladder (B0-B3, both voltage bands)

`PYTHONPATH=src python src/baseline_ladder.py --rung all --bands both --json
reports/baseline_recompute_2026-11-16/ladder_v6.json`
→ wrote the artifact, exit 0.
The pre-`F1` run is `ladder_pre_f1.json` in the same directory, same command.

| rung / band | `vmin` pre-F1 | `vmin` v6 | shed MWh pre-F1 | shed MWh v6 | curtailment MWh pre-F1 | curtailment MWh v6 | LMP mean pre-F1 | LMP mean v6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 primary | 0.9131 | 0.9131 | 0.0000 | 0.0000 | — | — | — | — |
| B0 strict | 0.9131 | 0.9131 | 0.0000 | 0.0000 | — | — | — | — |
| B1 primary | 0.9459 | 0.9300 | 0.0000 | 0.1007 | 0.0000 | 0.0000 | 246.99 | 287.48 |
| B1 strict | 0.9500 | 0.9500 | 0.0317 | 1.0819 | 0.0000 | 0.0000 | 262.23 | 444.05 |
| B2 primary | 0.9547 | 0.9314 | 0.0000 | 0.0000 | 13.1328 | 13.1341 | 246.18 | 249.44 |
| B2 strict | 0.9547 | 0.9500 | 0.0000 | 0.2110 | 13.1330 | 12.2296 | 246.18 | 315.56 |
| B3 primary | 0.9300 | 0.9300 | 0.0000 | 0.0000 | 13.0628 | 13.1384 | 245.63 | 248.52 |
| B3 strict | 0.9500 | 0.9500 | 0.0000 | 0.0000 | 13.0595 | 13.1384 | 245.66 | 248.64 |

`PASS` — both artifacts written, both runs exit 0.

What the table says:

- **B0 is unchanged**, as it must be: B0 is a static power flow on the native
  complete network with no radial selection, so it never touched the corrected
  function.
- **The market rungs are now voltage-bound.** `vmin` falls from 0.9459-0.9547
  to 0.9300-0.9314 under the primary band and pins at 0.9500 under the strict
  band on every rung except B3/primary. Before the correction the primary band
  was never reached on any rung; the cleared day simply did not stress the
  network.
- **Shed appears where it was absent.** B1 under the primary band goes from
  0.0000 to 0.1007 MWh of load the market declined to buy, and under the strict
  band from 0.0317 to 1.0819 MWh. B2/strict goes from 0.0000 to 0.2110 MWh.
- **The band choice now costs money.** B1's `delta_lmp_95_vs_93` moves from
  3.88 to 156.57 CNY/MWh and its `delta_shed_95_vs_93` from 0.0013 to 0.9812
  MWh. A 0.02 p.u. lower bound was a labelling choice on the old tree; on the
  corrected one it is a real constraint.
- **The physical and strategic comparisons keep their shape.** On the v6 tree
  `attributable_to_bidding` is `false` for B2→B3 (`delta_shed` 0.0000,
  `delta_curtailment` +0.0043 MWh) and `true` at fixed B3 facilities
  (`pi_baseline_truthful` 190.25, `pi_rule` 176.95, `delta_pi_rule`
  −13.30 CNY/day, −0.07%). Pre-`F1` the same block read 205.72 / 188.05 /
  −17.67, so the rule policy's shortfall against truthful narrows but does not
  change sign.

## Typical-day industry acceptance

`PYTHONPATH=src python src/accept_typical_day_industry.py --json
reports/baseline_recompute_2026-11-16/typical_day_acceptance_v6.json`
→ wrote the artifact, exit 0. Pre-`F1` run beside it.

| quantity | pre-F1 | v6 | change |
|---|---:|---:|---:|
| coincident peak (MW) | 526.2085 | 526.2085 | 0 |
| daily energy (MWh) | 10,995.64 | 10,995.64 | 0 |
| served (MWh) | 354.81 | 332.50 | −22.31 (−6.3%) |
| unserved, share of forecast | 96.773% | 96.976% | +0.203 pp |
| grid import (MWh) | 357.88 | 334.47 | −23.41 |
| fleet profit (CNY/day) | −0.5213 | −2.23e−05 | ≈0 |
| LMP min/max at ESS9 (CNY/MWh) | 1337.77 / 1556.51 | 2531.36 / 2830.32 | +89% |
| LMP min/max at ESS27 (CNY/MWh) | 2592.67 / 2750.18 | 2624.59 / 2890.43 | +1.2% |
| curtailment (MWh) | 7.01e−05 | 7.31e−09 | ≈0 |
| both settlement paths agree | true | true | — |
| repeatable over 2 runs | true | true | — |

`PASS` — the artifact was written and the run exits 0 on both trees; both
settlement paths agree in both runs and both repeat identically.

Two findings that must not be reported as market outcomes:

1. **The scenario is not served, on either tree.** `typical_day_industry`
   declares `load_scale_mode: ledger` and the real industry meters sum to a
   526.2085 MW coincident peak against a 3.715 MW feeder — 141.7×. Under a
   clearing that may shed, 96.98% of the declared energy is unserved on the
   corrected tree and 96.77% on the pre-`F1` one. `F1` is not the cause: it
   moves served energy by −6.3% and the fleet's settled profit from −0.52 CNY
   to −0.00002 CNY a day. Any economic statement about this scenario is a
   statement about a shed-dominated day on an overloaded feeder.
2. **`clean: true` is solver health, not service.** The acceptance's `clean`
   flag means the clear was optimal, the duals were usable and no fallback ran.
   It was `true` on a day that served 3% of its demand, before and after the
   correction. Reading it as "the day cleared" is the error the flag cannot
   prevent, and it is why the served/unserved block is reported separately.

The archived `outputs/phase6_acceptance.json` (2026-10-12) is **not** a
comparable baseline for this scenario: it records `load_scale_mode: native_peak`
and `load_shape: industry` with a 3.715 MW peak, i.e. a different scenario
definition from the one `config/scenarios.yaml` declares today (`ledger` /
`native`). Comparing across that pair would attribute a scenario redefinition to
the radial-tree correction.

## Benchmark table — `BLOCKED`

The benchmark table (`reports/v2_cy_jn_2025/results.csv`,
`benchmark_results.md`) was not recomputed.

- `run_benchmarks.py` produces `benchmark_raw.csv` (B0, B2A, B2B), the tuned
  `rule_params.json` and `b3_lp_bound.csv`, and takes roughly the 36 minutes
  its own `run.log` records. It is not the blocker.
- The blocker is the **B1 = JN-Base rows** of `results.csv`: 96 one-shot
  whole-day clears with a terminal pin, carrying columns
  (`served_mwh`, `pcc_import_max_mw`, `reverse_within_limit`) that neither
  `benchmark_raw.csv` nor `b1_terminal_neutral_results.csv` has. No maintained
  script writes them — `audit_b1_equivalence.py` verifies and annotates the
  stored rows, and `b1_daily_baseline.py` produces the *rolling-interface* B1
  (a different mechanism, and the ancestor of `run_terminal_neutral_b1.py`).
  Writing a new producer would put a quantity I cannot check against the
  original under the original's name.
- `make_benchmark_report.load_rows()` also reads `RAW` and `COMMON` from
  hardcoded module constants rather than from its own `--raw` argument, so the
  assembler cannot be pointed at a new raw file without editing it.

Consequence: the training-readiness headroom quoted from this table
(B1 167.4, B2A 160.3, B2B 84.3, B3 177.9 CNY/day on the 11 common test days,
"arbitrage headroom is small") is also `BLOCKED` — it is a reading of a table
that no longer describes the live network.

Next unit of work: a JN-Base producer with an explicit, documented mechanism
(one 96-period clear, terminal SOC pinned, truthful quotes) plus a `results.csv`
writer, then `run_benchmarks.py --segments all` and the assembler.

## Not re-measured

| item | verdict | reason |
|---|---|---|
| `baseline` / `high_re` / `peak_load` / `congestion` / `re_ramp_*` per-scenario clears | `NOT_TESTED` | The ladder measures B0-B3 on `typical_day`; the legacy scenarios are `explicit_peak` at 10 MW and nothing in this recompute clears them. |
| `typical_day_industry_subset` | `NOT_TESTED` | The acceptance clears `typical_day_industry`. |
| the 53 training days' day-mean LMP distribution | `NOT_TESTED` | `TECHNICAL_DEBT.md` TD-12 quotes it (76.3 / 224.8 / 490.4 CNY/MWh); that measurement predates the `F4`/`F6` patches and was not repeated here. |
| `JN-Base` three-day settlement record | `NOT_TESTED` | Same JN-Base producer gap as above. |
| B1 baseline under the formal gate | `PASS` | `data/processed/v2_cy_jn_2025_v6/b1_terminal_neutral_results.csv`: 96/96 feasible, 0 fallbacks, 0 voltage-violation cells, max terminal error 0.0. Its before/after is in `REFREEZE_V6.md`. |
