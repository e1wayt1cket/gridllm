# Candidate training day library v3: design

Date: 2026-11-17. Supersedes the v1 library at
`reports/candidate_days_2026-11-16/` for every purpose except historical audit.
v1 stays read-only.

## Intent

The v1 library holds 53 days, all in July-December 2022, and the clearing
screen found two separate problems:

- spatial: its mapping places complete meters onto buses 1..N in `source_id`
  order, which is blind to feeder position. The bus carrying the largest share
  of a day's coincident peak ranges 25.9%-72.2%, against 11.3% in the native
  `case33bw`. Twelve of 53 days do not clear and 36 of the 41 that do sit
  exactly on the 0.93 V lower bound.
- temporal: five calendar months carry days, all in one year. June and
  September carry none even though price curves exist for both.

v3 changes the mapping rule and the date set together, and re-draws the split.
This document freezes the data-building rules only. It computes no clearing,
no settlement, no price, no benchmark and no reward.

## Goals

1. A load library whose per-bus spatial profile approximates the native
   `case33bw` share, and whose meter curves are unmodified.
2. Date coverage across every month the 2025 price pool can pair with.
3. A pre-registered split that no clearing or price outcome can influence.

## Non-goals

- No scaling, smoothing or re-sampling of any meter curve.
- No clearing, settlement or pricing run.
- No reactive-support device added to the model by this work.
- No change to the price pool, the network parameters or the voltage band by
  this work.

## Anti-goals

- Selecting or discarding a day because it clears, or because clearing it
  produces an attractive price or arbitrage.
- Discarding a day because its realized spatial concentration misses a
  threshold. Concentration is a property of the mapping rule over the whole
  library; it is reported, never used to edit the date set.
- Aggregating meters onto a bus so that the curve-level correspondence is lost.
- Growing the library past what a same-month unique price pairing can serve.

## Constraints

- **Universe.** `data/processed/industry_block_screening.csv` holds 103,739
  rows over 730 distinct ISO dates spanning 2021-01-01 to 2022-12-31, each
  date with at least 20 `qualified_B` meter-days. 68 rows carry an empty
  `date` field; all 68 have `qualified_B == False`, so they never enter. The
  count is recorded in the manifest.
- **Seasonal ceiling.** `data/processed/v2_cy_jn_2025/price_curves.npz` holds
  136 unique 2025 dates under the `mv/` prefix, covering June-December (seven
  months) only, identical for `mv` and `south`. Per-month availability:
  Jun 14, Jul 15, Aug 14, Sep 16, Oct 31, Nov 15, Dec 31. Under same-month
  pairing the library can cover June-December of the load years and nothing
  outside it.
- **Size ceiling.** 136 days, one unique price date each.
- **Candidate supply.** 428 dates lie in June-December of 2021 or 2022 (214
  each year, 60-62 per month). Every one carries at least 20 qualified-B
  meter-days. Per-month supply is at least 1.9x that month's price
  availability, so the size ceiling is reachable if the workbook supplies
  curves; October and December are the two tight months at 62 available days
  against 31 required.
- **Price pairing.** `common_split.json` pairs load and price by calendar
  month in 78 of its 80 pairs. v3 keeps that rule: a load date in month M pairs
  with an unused 2025 date in month M.
- **Band.** Per-day coincident peak within `[3.3435, 4.0865] MW`, i.e. the
  3.715 MW anchor with 10% tolerance. This is the v1 rule, unchanged.
- **Voltage band.** `0.93-1.07 pu`. Recorded here for reference only; this
  work runs no power flow. The working tree still carries 1.05 in
  `src/models.py`, `config/defaults.yaml`, `src/local_da_contracts.py` and the
  hardcoded literal in `src/local_da_operating.py`; the 1.07 edits are a
  separate pending change and are not a prerequisite for this library.

## Approaches

Three ways to fix the v1 library, all outcome-blind:

1. **Filter v1 to its feasible days.** 41 of 53 days under `operating`. Keeps
   every rule and every artifact; costs all of July, most of August, both empty
   months and any second year. Rejected: it removes exactly the cross-season
   coverage the library exists for, and it selects days by a clearing outcome.
2. **Mapping only (v1.2).** Keep the 53-date pool, replace the bus rule. Fixes
   concentration and recovers the three diagnosed days. Rejected: the spatial
   rule and the date set were chosen by the same person at the same time, and
   changing one while freezing the other leaves a library whose date draw was
   never evaluated against the new mapping. It also keeps five months and one
   year.
3. **Mapping and date set together, with a fresh split (v3).** Chosen. It is
   the only option that serves Goals 1 and 2 at once, and it is the option the
   split rule is designed for: partition after eligibility, before any
   clearing.

## Decided

| Decision | Value | Basis |
| --- | --- | --- |
| Mapping | M-a, native-share maximisation | v1's `source_id` ordering is blind to feeder position; three of twelve infeasible days recover when the same daily total is respread on native shares |
| Reactive load basis | One declared power factor, attached to the meter | `docs/2026-11-14-da-market-system-spec.md` declares a uniform PF 0.8503; the per-bus native `Q0/P0` basis lets a meter's reactive demand change when its bus changes, which contradicts M-a |
| Split | Redrawn in full | v3 changes both the pool and the mapping; preserving the old 80 assignments would carry v1's date draw into a library it no longer describes |
| Library size | Up to 136 days | Bounded by unique price dates |
| Split sizes | 90 / 18 / 18 / 10 at `L = 136`, proportional otherwise | train / validation / test / holiday block; 1.67x the v6 train count, accepted |
| Budget | `episodes_per_seed` scales with `train_days` | The training contract multiplies rounds by train days; doubling the exposure is accepted |
| Outcome filtering | Forbidden at this layer | A day's clearing feasibility is measured after the market mechanism is final, and reported, never used to edit the library |
| Spatial quality | Reported, never gated | No defensible threshold exists; a gate would turn a rule property into a per-day filter |
| Bus ceiling | At most 32 selected meters per day | There are 32 load buses above the slack and the curve-preservation invariant forbids two meters on one bus |

## Rules

### L0 eligibility (outcome-blind)

**L0.1 date universe.** The 730 dates in the screening CSV, restricted by
L0.2. No curve data is read at this step.

**L0.2 window.** June-December of 2021 or 2022, inclusive: 428 dates.

**L0.3 meter-day eligibility.** Per date, from the workbook
`docs/16类行业用户.xlsx`, a meter-day is eligible exactly when the existing
extraction accepts it: the screening row passes `qualified_B == True`,
`industry != PV_INDUSTRY`, and the flags `{negative_load, duplicate_conflict,
capacity_inconsistent}`; and its curve survives the per-curve rejection list
already in that function (`missing_slots`, `all_zero`, `placeholder`,
`other_phase`, `negative_load`, `capacity_inconsistent`, `ratio_missing`),
finite, non-negative, with a peak above zero. This document adds no filter of
its own. The zero-peak exclusion is applied before L0.4 so that a zero-peak
meter is never "the smallest".

**L0.4 one meter per user.** That user's smallest-peak eligible meter of the
day; ties by the smaller `source_id`.

Read the curves through the existing extraction in
`reports/load_selection_2026-11-13/prototype/select_real_loads.py`, with the
train-only date guard removed and the output written to a new directory. That
extraction is the only source of curve values; this document adds no
transform.

Eligibility depends on no clearing, price, voltage or economic quantity, and
is computed for the whole window before any split exists.

### L1 per-day source selection

The v1 rule applies unchanged, quoted from
`reports/candidate_days_2026-11-16/SPEC.md`:

> Candidates on a date are the pool's meter-days with peak > 0; at most one
> meter per user (that user's smallest-peak meter, ties by the smaller
> `source_id`); add candidates in ascending peak order (ties by `source_id`)
> while the fleet coincident peak stays at or below `anchor*(1+tolerance)`;
> while the peak is below `anchor*(1-tolerance)`, take the single upward swap
> yielding the largest peak at or below `anchor*(1+tolerance)` and stop as soon
> as the lower edge is reached; status is `IN_BAND` when the final peak is
> inside the band and `OUT_OF_BAND` otherwise, with the final peak and the
> reason recorded; no randomness, clock or solver tolerance enters the rule.

Two additions, both about the bus count rather than the selection:

1. **Ceiling.** At most 32 meters may be selected. The ascending add stops at
   32 even if the fleet peak is still below the cap; the upward swap then runs
   on that 32-member set. The day records `hit_bus_ceiling: true|false`.
2. **No placement at build time.** L1 returns the selected source set only;
   buses are assigned by L2. v1's placement step, and its `raise` when the set
   exceeds the 32 load buses, is not carried over.

Reason codes are v1's: `initial_in_band`, `raised_in_band`,
`no_swap_below_band`, `nothing_selectable`. A day outside the band stays in
the manifest with its final peak and code; it never enters the library.

### L2 mapping (M-a)

Per day, with `N` selected meters and native `P0` over the 32 load buses:

1. **Target buses.** The `N` buses with the largest native `P0`, ties by the
   smaller bus index. Weight `w_b = P0_b / sum(P0 over target buses)`. These
   weights are renormalized over the target set, so `w_b` is not the native
   32-bus share when `N < 32`; the excluded native share is recorded.
2. **Meter size.** `s_m` is the meter's 96-period mean; `pi_m = s_m / sum(s)`
   over the day's selected meters.
3. **Initial assignment.** Sort meters by `pi_m` descending (ties by
   `source_id` ascending) and target buses by `w_b` descending (ties by bus
   index ascending); pair them in order. Each bus carries exactly one meter.
   This is an initialization, not an optimisation: with one meter per bus the
   assignment is a permutation, so no incremental greedy can improve on the
   ordering.
4. **Local improvement.** With `assigned_b = sum_{m on b} pi_m`, define
   `D = sum_b (assigned_b / w_b - 1)^2`. Enumerate every pair of occupied
   target buses `(bus_i, bus_j)` with `i < j`, exchange the two meters, and
   apply the exchange that most reduces `D` by more than `1e-12`; repeat until
   no exchange qualifies. Ties in the maximum reduction take the
   lexicographically smallest pair. Record `D_init` and `D_final`.
5. **Reactive load.** `Q_b(t) = 0.6190 * P_b(t)` where
   `0.6190 = tan(arccos(0.8503))`, applied after mapping and stored as its own
   array. On periods where `P_b(t) = 0` the ratio is undefined; those entries
   are zero and are skipped by the verifier.

Note on step 4: because `N` meters occupy exactly `N` of the 32 buses, the
permutation has `P(32, N)` members and the swap neighbourhood is
"exchange the meters on two target buses". Swaps involving a non-target bus
are not enumerated: the target set is fixed by step 1 and moving a meter off
it would raise `D` by construction.

This rule declares an assumption: the day's users occupy the same buses as the
native heavily-loaded set. It does not claim a measured topology, and it does
not account for the PV at buses 21, 23, 30 or the ESS at 9, 14, 17, 27 in the
clearing template, which change voltage sensitivity. Native share is chosen
because it is the only outcome-blind proxy available.

### L3 partition

Draw the split after L0-L1 are complete and before any clearing, settlement or
price computation.

1. **Scale.** `E_m` is the count of in-band dates in month `m`, `P_m` that
   month's 2025 price count. `L = sum_m min(E_m, P_m)`.
2. **Draw.** Within each month, if `E_m` exceeds `P_m`, take `P_m` dates by a
   seeded draw stratified by load year, and record the discarded dates. If
   `E_m <= P_m`, take all of them.
3. **Holiday block.** The National Day window, 01-05 October of each load year.
   `H = min(10, available dates in that window)`; a shortfall is recorded as
   `holiday_shortfall`.
4. **Split.** The non-holiday dates are dealt in order into a repeating block
   pattern `[train] x 5, [validation], [test]`, truncated to `L - H` entries.
   At `L = 136, H = 10` this is exactly 90 / 18 / 18. Any other `L` scales the
   same pattern and the block counts are whatever the truncated pattern
   contains; `episodes_per_seed` is recomputed from the applied counts. The
   counts are written to `split.json`.
5. **Integrity.** Blocks are disjoint by date and by price date. The holiday
   date set may overlap the October block's draw only by exclusion: holiday
   dates are removed from the October pool before step 2.

Assignments are stratified by month and year from a fixed recorded seed, so
two runs give the same split.

## Invariants a verifier must check

Gating:

- **Curve preservation.** For every date, the multiset of non-zero per-bus
  96-period curves equals the multiset of that day's selected meter curves,
  each matching its source-pool sha256. A mapping that aggregates or scales
  fails this check.
- **One meter per bus.** No bus carries two meters; no meter appears twice.
- **Reactive proportionality.** On every loaded bus and period,
  `Q_b(t) / P_b(t)` equals 0.6190 to `1e-9`; periods with `P_b(t) = 0` are
  skipped and must have `Q_b(t) = 0`.
- **Band, library only.** Every library day's coincident peak lies in
  `[3.3435, 4.0865] MW`. Manifests may hold out-of-band days; the library may
  not.
- **Split integrity.** Disjoint by date and by price date, every block
  populated, counts equal to the rule applied in L3 step 4.
- **Determinism.** Two runs produce identical hashes for the CSV, JSON and
  `npz` artifacts. `np.savez` is byte-stable on the stack in use (numpy 1.24.3,
  measured across a 2 s gap) and v1's `reproducibility.json` records three
  identical builds.

Reported, not gating:

- **Concentration.** Per day: `realized_max_share`, `target_max_share`,
  `native_max_share` (0.1131 on the full 32 buses), `D_init`, `D_final`, and
  the excluded native share when `N < 32`. If the median
  `realized_max_share` exceeds `1.1 x target_max_share`, the report marks
  `M_A_QUALITY_WARNING` and lists the days. That is a signal, not a verdict.

## Artifacts

New directory `reports/candidate_days_v3_2026-11-17/`, following the v1 layout:

| Path | Content |
| --- | --- |
| `SPEC.md` | This rule set, frozen at build time |
| `evidence/day_selection.csv` | One row per date: month, `N`, peak, status, reason, `hit_bus_ceiling` |
| `evidence/source_bus_mapping.csv` | One row per meter: source, user, target bus, target and realized share, `D_init`, `D_final` |
| `evidence/bus_active_load_mw.npz` | `date -> (96, 33)` MW |
| `evidence/bus_reactive_load_mvar.npz` | `date -> (96, 33)` MVAr |
| `evidence/split.json` | Blocks, seed, applied counts, `L`, discarded dates, `holiday_shortfall` |
| `evidence/manifest.json` | Input hashes, rule, universe counts, all candidate dates, status |
| `evidence/verification.json` | Independent verifier output, gating and reported sections |
| `prototype/` | Builder, verifier, contract tests |

## Acceptance

`PASS` requires a command and its result. The library is accepted only when:

1. Contract tests pass and every gating invariant above is reported.
2. Every library day is in band; the manifest records every out-of-band
   candidate with its peak and reason code.
3. Library size `L`, the applied block counts and `episodes_per_seed` agree
   with L3 for the measured `E_m`, exactly rather than within a tolerance.
4. All seven months June-December appear in the library. A month with zero
   library days is `MISSING_MONTH` and is not accepted.

Clearing feasibility is **not** an acceptance criterion here. It is measured
in a separate screening unit against the final market model and reported as
its own result.

## Risks and unmeasured items

- **Workbook yield is unmeasured.** 428 candidate dates exist and 136 are
  needed, so a 32% build rate suffices. v1's pool build dropped 40 of 93 days
  for missing sources, a 57% rate, which would leave roughly 244. This is
  `INFERRED`; the actual rate is `NOT_TESTED` until L0.3 runs.
- **Every v1 clearing number is under a different reactive basis.** The
  0.9131 V native minimum, the 41/53 and 36/41 counts and the three recovered
  days were all measured with the native per-bus `Q0/P0` ratio and, for the
  screen, the old engine. Under `Q = 0.6190 P`, the total reactive demand is
  preserved (2.2994 vs 2.3000 MVAr at 3.715 MW) but its distribution moves, so
  the voltage profile moves. All of it is `NOT_TESTED` for v3.
- **The native minimum sits below the band.** Native `case33bw` at 3.715 MW has
  `Vmin = 0.9131 pu` under the native basis and 14 buses below 0.93. A day
  mapped to native shares therefore clears only by using reactive support, and
  feasibility is decided by the daily shape. Binding is the mechanism, not a
  defect, but it means the library's feasibility cannot be assumed.
- **`N = 32` is tight.** 11 of v1's 53 days selected exactly 32 meters. A day
  with more small meters would have raised instead of building, so the ceiling
  is a real path, not a theoretical one.
- **Same-month pairing caps the claim.** June-December is seven of twelve
  months. No statement about January-May is supported by this library, and no
  result on it is a statement about a full year.
- **2021 load days are new.** No v3 date outside 2022 has ever been
  materialised or cleared. Their curve quality is bounded by the same L0.3
  filter, but their aggregate behaviour is unverified.
- **M-a targets a proxy.** Native `P0` share ignores the DER and storage in the
  clearing template, so a low `realized_max_share` is not evidence that the
  day will clear.

## Left to the builder

- The exact seeded-draw construction in L3 step 2, provided it is stratified by
  month and load year and its seed is recorded.
- The verifier's independent reading of the workbook (it must not reuse the
  builder's selection function) and its hashing scheme.
- The `prototype/` test list.
