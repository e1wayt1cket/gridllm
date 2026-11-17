# Candidate training day library v3: results

Date: 2026-11-17. Built from `SPEC.md` (byte-identical to
`docs/superpowers/specs/2026-11-17-candidate-day-library-v3-design.md`, both
sha256 `11cff0f3…`). Verdicts use `PASS` / `FAIL` / `NOT_TESTED` / `INFERRED` /
`BLOCKED`, each with the artifact or command it came from.

## Build

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Library built, manifest status `LOAD_ONLY_PASS` | PASS | `evidence/manifest.json` `status`; build command below |
| Library size 136 | PASS | `manifest.json` `library_size` = 136; `split.json` `library_size` |
| All seven months June-December present | PASS | `manifest.json` `month_counts` = {06: 14, 07: 15, 08: 14, 09: 16, 10: 31, 11: 15, 12: 31} |
| Every library day in band | PASS | `manifest.json` `in_band_count` = 428, `out_of_band_dates` = [] |
| Bus ceiling reached on some days and recorded | PASS | `manifest.json` `hit_bus_ceiling_count` = 17 of 428 |
| Source pool yield | PASS | `source_pool/manifest.json`: 28,336 meter-days over 428 dates, 0 curves rejected, 117,272 workbook rows scanned |
| Split counts 90 / 18 / 18 / 10 | PASS | `split.json` `counts`; verifier `split_integrity` |
| Holiday shortfall 0 | PASS | `split.json` `holiday_shortfall` = 0 (both 01-05 October windows available) |
| Discarded dates recorded, not dropped | PASS | `manifest.json` `discarded_dates`: 06: 46, 07: 47, 08: 48, 09: 44, 10: 31, 11: 45, 12: 31 |
| `episodes_per_seed` under the frozen round budget | INFERRED | 90 train days x 10 rounds = 900, x 15 rounds = 1,350, from the training contract's `minimum_rounds * train_days`; no run was launched |

Build command:

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_2026-11-17/prototype/build_library.py \
    --pool reports/candidate_days_v3_2026-11-17/source_pool \
    --out reports/candidate_days_v3_2026-11-17/evidence \
    --price-curves data/processed/v2_cy_jn_2025/price_curves.npz
```

## Verification

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Curve preservation (multiset of bus columns vs workbook curves) | PASS | `verification.json` `checks.curve_preservation` = true |
| One meter per bus, no meter twice | PASS | `checks.one_meter_per_bus` = true |
| Reactive proportionality `Q = 0.6190 P`, zeros stay zero | PASS | `checks.reactive_proportionality` = true |
| Band, library rows only; out-of-band rows carry a valid code | PASS | `checks.band_library_only` = true |
| Split integrity: disjoint by date and price, counts match the pattern | PASS | `checks.split_integrity` = true |
| Month coverage 06-12 | PASS | `checks.month_coverage` = true |
| Determinism against the manifest's content hashes | PASS | `checks.determinism` = true |
| Verifier flagship test: injected duplicate bus flips only `one_meter_per_bus` | PASS | `python -B -m unittest test_verify_library -v` → 1 test OK |
| Contract tests | PASS | `python -B -m unittest test_library_rules test_build_library test_extract_window_pool` → 20 tests OK |
| Rebuild reproducibility | PASS | `reproducibility.json` `all_equal` = true over `day_selection.csv`, `source_bus_mapping.csv`, both npz files and `split.json` |

Verifier command:

```bash
PYTHONPATH=src python -B reports/candidate_days_v3_2026-11-17/prototype/verify_library.py \
    --library reports/candidate_days_v3_2026-11-17/evidence
```

Exit 0, seven gating checks true.

## Concentration: reported, not gating

| Quantity | Value | Source |
| --- | --- | --- |
| Median `realized_max_share / target_max_share` | 2.8889 | `verification.json` `report.concentration.median_realized_over_target` |
| `M_A_QUALITY_WARNING` | raised, 136 of 136 days | `report.concentration.warning` |
| Native max share over the 32 load buses | 0.1131 | `report.concentration.per_day[*].native_max_share` |

The warning is not a defect of the rule: with one meter per bus, a day whose
meters are far more unequal than the native bus weights cannot reproduce the
native concentration, and the rule minimizes squared relative deviation rather
than the maximum share. It is recorded so a later clearing screen can weigh it.

## Not measured here

| Item | Verdict | Why |
| --- | --- | --- |
| Clearing feasibility of any library day | NOT_TESTED | This unit runs no clearing, settlement, price or reward; it is measured separately against the final market model |
| Voltage behaviour under the 0.6190 reactive basis | NOT_TESTED | No power flow was run |
| 2021 load-day aggregate behaviour | NOT_TESTED | No 2021 date has been materialised before this library |
| January-May coverage | NOT_TESTED | Same-month pairing caps the library at June-December |

## Notes on the artifacts

- `source_pool/` (26 MB, 28,336 curves) is a regenerable intermediate from
  `prototype/extract_window_pool.py` plus the workbook, and is not versioned.
- `evidence/*.csv` is matched by the repository-wide `*.csv` ignore rule, so it
  is present on disk but not versioned; every value read from it above is also
  recorded in `manifest.json`, `split.json` or `verification.json`.
