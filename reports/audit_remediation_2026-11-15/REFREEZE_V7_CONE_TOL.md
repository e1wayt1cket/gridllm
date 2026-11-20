# Re-freeze v7: the cone-tightness bound

Protocol `v7`, candidate `V7_candidate_004`, config hash `0f35fed8`. This
record supersedes `REFREEZE_V7_BUNDLE.md` (`V7_candidate_003`, `b8b61d8d`) as
the live manifest. No formal run and no formal round exists under any v7 hash.

## Why

Re-measuring the 136-day library on the disjunction tree
(`reports/local_da_v7_2026-11-20/qualification/`) returned:

| | bilinear, `BarQCPConvTol=1e-10` | disjunction, `1e-8` |
| --- | --- | --- |
| `SMOOTH_PASS` | 22 | 19 |
| `SUBGRADIENT_PASS` | 55 | 62 |
| `NONSMOOTH_OR_FAILED` | 59 | 55 |
| label-qualified | 77 | 81 |
| settle-qualified | 62 | **38** |
| label-qualified but not settle-qualified | 15 | **43** |

Twenty days that settled before stopped settling; six started. Every dispatch
was optimal and every price solve optimal on all 136 days; the loss is entirely
in the settlement audit.

## What was actually wrong

`BarQCPConvTol` was `1e-10` because the retired bilinear encoding needed it:
`1e6 * ch * dis == 0` had to hold tightly enough for the audit's
`complementarity_product_max_mw2 <= 1e-6`. One barrier tolerance governs every
quadratic in the model, so that setting also held the cone residual below
`1e-6`. The disjunction made exclusivity structural and the product check
disappeared with it; the barrier moved to `1e-8` on the strength of one
reference day, and the cone check inherited a margin nobody had derived.

That margin does not cover the library. Sampling fifteen of the twenty
regressed days, each fails on `cone_tightness_max_mw2` and on nothing else:

- gap range `1.05e-6` to `6.58e-6`, median `1.46e-6`;
- `p_balance`, `q_balance`, the SOC recursion, bounds, exclusivity and the
  voltage-drop recurrence all stay below `1e-6` on every sampled day;
- `ac_replay` passes on all fifteen, with a maximum voltage difference of
  `8.0e-10` pu and a maximum import difference of `3.6e-8` MW.

So the cleared operating points are AC solutions to `1e-10`. The gap is a
residual in the auxiliary `current_sq`, which enters the voltage recursion only
through `(r^2 + x^2) * l` and therefore does not move the physical state.

## What changed

`src/local_da_accounting.physical_audit` takes `cone_tolerance=1e-5` and keeps
`tolerance=1e-6` for every other check. The cone bound is now the same order as
the independent nonlinear check that `qualified_for_settlement` already
requires, instead of three orders tighter.

`src/make_formal_training_manifests.py` — `CANDIDATE` is `V7_candidate_004`,
which moves the config hash to `0f35fed8`.

The barrier tolerance was not changed. Both directions were measured on three
regressed days: at `1e-9` the audits pass but two of the three lose their price
label (`SMOOTH_PASS` becomes `NONSMOOTH_OR_FAILED`), and at `1e-10` the exact
model returns before the audits are computed. Tightening buys settlement
qualification at the cost of price qualification and wall clock.

## What this does not do

The bound is looser than before. It does not become a licence to ship a slack
relaxation: `qualified_for_settlement` still requires `ac_replay`, which
re-solves the nonlinear radial flow from the cleared injections and fails when
the cone is structurally open. What the change stops demanding is that a
bookkeeping variable be resolved finer than the solver was asked to resolve it.

`1e-5` clears the sampled maximum by 1.5x. Days outside the sample may exceed
it; that is a measured limit, not a proof of coverage.

## Gate proof

`PYTHONPATH=src python -B src/run_formal_training.py --config-dir
reports/v2_cy_jn_2025/formal_training_v7 --check`:

```
data_payload_matches true     code_sources_match true
code_tests_match true         config_hash_matches true
data_manifest_hash_matches true   code_manifest_hash_matches true
budget_contract_ok true       library_counts true
all_disjoint true             legacy_label_declared true
h2_declared true              all_pass true
status: null    accepted_candidate: V7_candidate_004
config_sha256: 0f35fed8    contract_violations: []
```

`status` stays null: the readiness status is written only by `--finalize`, and
no `pretraining_validation_results.json` exists for v7.

## Tests

`pytest tests/test_local_da_market.py tests/test_s8_physical_audit.py
tests/test_local_da_env_v7.py tests/test_checkpoint_schema.py
tests/test_training_contract_v7.py tests/test_frozen_guard.py -q` — `105
passed`. The mutation test that perturbs `qcc_mvar` by 0.1 still fails the
audit.

A single day (`2021-06-23`) re-cleared under the new bound reports
`SUBGRADIENT_PASS`, both audits passing, `qualified_for_settlement` true.

## Rollback

The superseded manifest set is at `tmp/manifest_superseded/v7_b8b61d8d/`.

## Not validated

- The 136-day qualification has not been re-run under the new bound. The
  recovery of the twenty days is measured on fifteen of them; the count of
  settle-qualified days under this candidate is `NOT_TESTED`.
- The barrier/tolerance pair was not tuned jointly. `1e-9` and `1e-10` were
  sampled on three days each, not swept.
- No formal run exists under `0f35fed8`.
