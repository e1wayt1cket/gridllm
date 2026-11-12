# H1/H2 holdout protocol — verification review

Date: 2026-11-08. Read-only review; no frozen file, data artifact, run
directory or historical report was modified. Every value below was re-measured
this session from the cited artifact.

Scope: the H1/H2 holdout protocol as defined in
`reports/v2_cy_jn_2025/holdout_exposure_audit.md` and its artifacts
(`data/processed/v2_cy_jn_2025/holdout_manifest.json`, `holdout_pairs.csv`,
`holdout_exposure_ledger.csv`, `holdout_status.json`,
`new_holdout_manifest.json`, `pilot/holdout_status.json`), the G10 lineage
(`formal_training_go_no_go.md`, `new_holdout_data_audit.md`,
`formal_training_protocol.md`, `future_holdout_protocol.md`), and the
enforcement chain (`formal_training_data_manifest.json`,
`src/training_contract.py`, `src/run_formal_training.py`).

## Verification verdicts

| # | item | verdict | evidence |
|---|---|---|---|
| 1 | H1 = 0 days is justified | **PASS** | `holdout_manifest.json` → `h1`: 27 unused V2 loads are spring/spring-festival with no 2025 spring price; 28 non-V2 17-user days would change the load system; relaxed upper bound 18 pairs / 2 seasons < 30 days / 3 seasons |
| 2 | H2 = 40 distinct pairs, 3 seasons | **PASS** | 40/40 distinct ids; seasons by price month: summer 22, autumn 10, winter 8; `meets_30_days_and_3_seasons: true` |
| 3 | H2 price-side isolation | **PASS** | 0 of 40 H2 price dates appear in NEAR (80), FULL96 (96) or the 16 LEGACY_SEEN_TEST pairs; ledger shows all 40 with `raw_viewed: True` and every exposure flag `False` (`holdout_h2_eligible: True`) |
| 4 | Load-side disclosure | **PASS** | 38 distinct loads, `2022-08-12` used 3× (= 2 reuses); pairing types 33 same-type / 5 day-type-mismatch / 2 reused = 40; smoke load `2022-07-22` absent from the H2 pool; per-pair flags reproduce the audit table (pilot training 15, pilot eval 2, legacy 2, used-for-selection 8, reserved 17) |
| 5 | Artifact integrity, no drift | **PASS** | sha256 of ledger, pairs and status match `holdout_manifest.json`; all 14 recorded input hashes unchanged; `holdout_status.json` hash `4f31263d…84e5e85` matches the pin in the live v3 data manifest |
| 6 | Seal enforcement chain | **PASS** | data manifest `h2_opened: false` → `training_contract.py` violation `data_h2_opened` if flipped → `run_formal_training.py --check` reports `h2_sealed: true`; runs record `h2_holdout_read: false`; gate re-verified green in P0 today |
| 7 | G10 lineage consistent | **PASS** | `new_holdout_manifest.json` (`is_new_independent_holdout: false`), `formal_training_seeds.json` (`formal_training_authorised: false`), v3 config `old_g10_status`, data manifest `independent_generalisation_claim: false` tell one story: protocol runs are GO, independent-generalisation claims are NO-GO |

## Problems found

### P1 — Documentation tension across the same-day artifacts (fixed by this note)

- `future_holdout_protocol.md` states "no unseen 2025 day remains" and that all
  complete days "have been viewed", and `pilot/holdout_status.json` records
  `unseen_2025_holdout_available: false` — neither mentions the H2 construction.
- `holdout_exposure_audit.md` and `holdout_status.json`, same `as_of` date,
  construct H2 = 40 price dates with `meets_30_days_and_3_seasons: true`.

Read without this note, the two pairs of artifacts contradict each other. The
canonical reading, which this review fixes as the reference:

1. "Unseen / blind" in `future_holdout_protocol.md` and
   `pilot/holdout_status.json` means a **blind, joint source-load holdout**
   (the H1 class). None exists: H1 = 0, G10 stands, and no re-split of the 2025
   file may be called blind. Those statements remain true in that sense.
2. H2 is a **price-date adaptation set**, not a blind joint holdout: its 40
   price dates were raw-inspected by the data-quality audit (`raw_viewed`,
   declared not leakage) but never paired, trained, validated, piloted,
   benchmarked or selected. Its load side is historical and disclosed per pair.
3. Even H2 is not a ready formal holdout: `holdout_status.json` itself records
   `formal_holdout_ready: false`; the allowed claim is exactly
   "limited price-date adaptation on 40 unpaired 2025 price dates", and the
   disallowed claim is any joint source-load generalisation.

A secondary nit in the same audit: the H2 summary table's exposure rows
overlap (pilot-training 15, pilot-eval 2, legacy 2, used-for-selection 8,
reserved-for-selection 17 sum to 44 > 40 pairs; measured overlaps:
used∩reserved = 4, pilot∩reserved = 9). The per-pair columns in
`holdout_pairs.csv` are the authoritative disclosure; the summary rows are not
a partition.

### P2 — The H2 seal is logical, not physical (reported; not fixable in place)

All 40 H2 price curves are physically present in
`data/processed/v2_cy_jn_2025/price_curves.npz` (136 `south/<date>` keys; all
40 H2 dates verified present), the same file the training environment loads.
Isolation therefore rests on the pairing CSV not naming those dates, the
`h2_opened` manifest flag and the contract check — not on any barrier in the
load path. This was declared by the original audit (the seal is `INFERRED`, no
artifact records a read). Closing it would require a code-level guard in
`v2_cy_jn_env.py` / `train_da_gate.py`, both pinned by the live manifest; that
is a protocol change for the next version directory, not an edit.

### P3 — `h2_holdout_read: false` is declared, not measured (reported; not fixable in place)

`src/train_da_gate.py` writes `h2_holdout_read: False` as a constant into each
run manifest (two sites). The value is a promise by the code author, not a
check the code performs. `train_da_gate.py` is pinned by the live manifest, so
this is recorded here rather than edited. Cleared by: a future protocol version
whose environment refuses to build a pair listed in `holdout_pairs.csv`, or
whose run manifest records the hash of the pair set actually loaded.

## What was not tested

- **NOT_TESTED** — reproduction of the H2 candidate construction from
  `src/audit_holdout_exposure.py` (the generator was not rerun; this review
  verified the artifacts against each other and against the raw pairing files,
  not the script's internal logic).
- **NOT_TESTED** — the 5 H2 pairs whose loads sit in the formal test block and
  the 22 whose loads sit in the train block as an evaluation-design question;
  the disclosure records them, and whether price-adaptation evaluation should
  stratify by load-side split membership is a decision for the phase that opens
  H2, not for this review.
- **INFERRED** — the H2 seal itself, for the reason in P2; no artifact would
  record a read.

## Disposition

P1 is resolved by this note (a new dated artifact; no historical file was
edited, per the data-governance and reporting rules). P2 and P3 are registered
as known limits with their clearing conditions; both touch pinned sources and
therefore belong to the next protocol version, alongside TD-01, TD-10 and
TD-11.
