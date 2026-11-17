# v3 library clearing screen: results

Date: 2026-11-17. Library: `reports/candidate_days_v3_2026-11-17/evidence`
(136 days, split 90/18/18/10). Rule declared in `evidence/rule.json` before the
first solve:

- **feasible** = the exact complementary dispatch is optimal and every
  bus-period stays inside the network's own band `0.93 <= V <= 1.07`;
- **`qualified_for_settlement`** is a separate and stricter claim (physical
  audits, idle-direction stability, smooth prices) and is reported on its own.

Run command:

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/screen_library.py \
    --out reports/candidate_days_v3_clearing_2026-11-17/evidence
```

## Verdicts

| Claim | Verdict | Evidence |
| --- | --- | --- |
| One clearing per library day, 136 rows | PASS | `evidence/summary.csv` row count 136; dates equal the union of `split.json` `blocks` |
| Feasible days | PASS | 132 of 136; `evidence/summary.json` `counts` |
| Feasible by block: train 87/90, validation 17/18, test 18/18, holiday 10/10 | PASS | `summary.json` `counts` |
| Library not modified | PASS | `summary.json` `library_unchanged` = true, from comparing all `outputs` hashes before and after |
| Deterministic | PASS | `rerun_check/summary.csv` for 2021-07-18 and 2021-12-24: `optimal`, `feasible`, `price_qualification`, `qualified_for_settlement`, `voltage_min_pu`, `voltage_max_pu` identical to the first run |
| Runtime | PASS | `summary.json` `runtime_seconds_total` 386.4 s; per day median 2.74 s, max 5.52 s |
| `qualified_for_settlement` | PASS (measured) | 0 of 136; `counts` per block |
| Settlement, payoff, or training behaviour of any day | NOT_TESTED | The screen stops at qualification; no ledger was produced |

Nothing in the library was edited or regenerated. The screen writes only under
this directory.

## The four infeasible days

| Date | Block | Solver status | Solutions | Peak MW | Peak rank (1 = highest of 136) |
| --- | --- | --- | --- | --- | --- |
| 2021-12-24 | train | 3 (INFEASIBLE) | 0 | 4.0779 | 6 |
| 2022-08-02 | validation | 3 (INFEASIBLE) | 0 | 4.0333 | 33 |
| 2022-10-11 | train | 3 (INFEASIBLE) | 0 | 4.0718 | 9 |
| 2022-10-14 | train | 3 (INFEASIBLE) | 0 | 4.0847 | 1 |

All four are exact-dispatch infeasibilities, not voltage-band failures: the
solver proves no feasible dispatch exists at all. Three sit in the top 7% of
the library's peaks and the fourth at rank 33, all within 0.06 MW of the
declared band ceiling (4.0865 MW), while the library's median day peaks at
3.91 MW and clears. Feasibility is otherwise uniform across months and load
years (2021: 69/70, 2022: 63/66; the worst month is 2022-10 at 13/15).

## Voltage

| Quantity | Value | Source |
| --- | --- | --- |
| Days whose cleared minimum sits exactly on 0.93 | 119 of 136 | `summary.csv` `voltage_at_lower_bound` |
| Cleared `Vmin` range | 0.9300 - 0.9479 | `summary.csv` `voltage_min_pu` |
| Cleared `Vmax` range | 1.0060 - 1.0654 | `summary.csv` `voltage_max_pu` |
| Days that would fail a `0.93 <= V <= 1.05` band | 41 of 136 | `voltage_max_pu` > 1.05 |

The lower bound is the binding limit on almost every day, as the library design
predicted; the 1.07 upper bound never binds. The band change from 1.05 to 1.07
is what makes 41 of these 136 days feasible: under the old band they would have
failed on the high side.

## Price qualification: two different failures

`qualified_for_settlement` is 0 of 136 for two distinct reasons, and they must
not be merged.

1. **108 days never reached a price.** The conditional price solve returns
   Gurobi status 13 (SUBOPTIMAL) with `optimal: false` while its objective
   still agrees with the exact dispatch objective to 1.3e-4 CNY on an
   18,264 CNY day (`evidence/2021-09-02.json`, `pricing` vs `dispatch`
   `clearing_objective`), so the *status* is what refuses the price, not a
   numerical divergence. The solver's own log says why:
   `Warning: max constraint violation (1.8542e-07) exceeds tolerance`
   (`dry_run/2021-09-02.log`; solver logs are on disk but git-ignored).
   `clear_day` requires `optimal`, so `price_qualification` stays
   `NOT_TESTED`.
2. **24 days reached prices and failed smoothness.** Their pricing solve is
   optimal, `probe_prices` runs, and every one returns
   `NONSMOOTH_OR_FAILED`.

The same pattern predates this library: the v1 screen recorded `NOT_TESTED`
for 102 of 106 rows. It is a property of the price path on this network, not a
consequence of the v3 reactive basis.

## LMP sanity

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_clearing_2026-11-17/prototype/lmp_sanity.py
```

| Claim | Verdict | Evidence |
| --- | --- | --- |
| The slack bus reproduces the exogenous price | PASS | 132 priced days; median `|LMP_slack - external|` 0.0000, max 0.0005 CNY/MWh |
| No negative prices | PASS | 0 of 418,176 bus-periods |
| The ordinary level is anchored | PASS | LMP median 306.51 against an external median of 300; mean bus markup over the slack runs 0.00% - 5.86%, rising with distance from the slack |
| Prices stay inside the declared 0-600 CNY/MWh research interval | FAIL | max 1621.85; 26 days carry a price above 600 and 16 above 800 |
| The extreme tail is a voltage rent, not an energy price | PASS | 654 of 12,672 periods carry a bus above 3x the external price (1.94% of bus-periods), and 100% of those periods have a bus exactly on 0.93; 95.6% of them sit where the external price is at or below 60 CNY/MWh |
| The duals are internally consistent | PASS | 41 of 405,504 branch-periods show a price fall above 20 CNY across a branch; the worst is -28.10, a 2% step between two buses both above 1,400 |
| Those prices are certified optimal duals | NOT_TESTED | 108 of the 132 day sets come from a pricing solve with status 13 (SUBOPTIMAL). The above-3x share is the same in the 24 optimal ones (1.92% against 1.94%), so the pattern belongs to the mechanism, not to the status |
| The four unservable days | n/a | no pricing dict exists for 2021-12-24, 2022-08-02, 2022-10-11, 2022-10-14 |

Read together: the price is anchored at the slack, positive everywhere, and
its ordinary variation is the loss markup of a radial feeder. Its extreme tail
is the lower voltage bound. With `generator_q_limits_mvar = 0`, no load
shedding and a storage fleet bidding zero, the only relief from the far-end
0.93 limit is more import, and at the 30 CNY/MWh price floor that relief is
worth 20-40x the energy price. The maxima therefore land on floor-price
periods, i.e. exactly the periods an arbitraging agent wants to charge in.

## Decision point

1. **Train days.** 3 of 90 train days cannot be served. The RL design permits
   dropping train days for feasibility and requires them to be named, so these
   three can be handled without touching the library.
2. **Validation.** 1 of 18 validation days (2022-08-02) cannot be served, and
   the design requires every validation day to be evaluated. This needs a
   decision before any training: keep the day and record its evaluation as
   failed, or treat an unservable validation day as a mechanism problem.
3. **Price qualification is empty.** With 0 of 136 days qualified, it cannot
   serve as a training-day admission criterion until the SUBOPTIMAL status and
   the smoothness failures are settled as their own question. Both are market
   model decisions, not library decisions.
4. **Price level.** Cleared LMP reaches 1,621.85 CNY/MWh on voltage-bound
   periods at the price floor, outside the declared 0-600 research interval.
   Any consumer of the price signal, the RL observation included, will read
   those spikes as the cheapest hours of the day.

## Notes on the artifacts

- `evidence/` holds `rule.json`, `summary.csv`, `summary.json`, one
  `{date}.json` / `{date}.npz` / `{date}.log` triple per day, 411 files and
  32 MB. It is regenerable by the command above and is not versioned
  (`*.csv` and `*.npz` are ignored repository-wide).
- `dry_run/` holds the three-day timing run, `rerun_check/` the determinism
  rerun.
- `run.log` is the full-run console output.
