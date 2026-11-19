# Re-freeze v7: the charge/discharge disjunction

Protocol `v7`, candidate `V7_candidate_002`. This record supersedes
`REFREEZE_V7.md` (`V7_candidate_001`, config hash `ea92b0af`) as the live
manifest. Neither a formal run nor a formal round exists under either hash, so
nothing is invalidated retrospectively; what is invalidated is the acceptance
evidence measured on the earlier encoding, which describes the model this record
replaces.

## Why

The exact clearing model enforced charge/discharge exclusivity with a bilinear
equality, `1e6 * ch * dis == 0`, one per storage unit and period. With presolve
and presolve's linearization switched off — so the model declared no integer
variables — the solver had to separate that product by spatial branching over
continuous variables.

Measured on `2021-06-04`, block 5, at the quotes the seeded policy actually
produced:

| encoding | limit | result |
| --- | --- | --- |
| bilinear equality, presolve off | 60 s | status 9, 0 solutions |
| bilinear equality, presolve off | 300 s | status 9, 0 solutions (the launcher abort) |
| binary disjunction | 60 s | status 9, 4 solutions, gap 5.30e-4 |
| binary disjunction, `BarQCPConvTol` 1e-8 | 60 s | **status 2, optimal in 2.8 s** |

The abort was never a proof of infeasibility: the solver reports status 9 (time
limit) with no incumbent, and the failure code names that `CLEAR_INFEASIBLE`.
For the same window the relaxation wants simultaneous flows — ESS9 charges and
discharges 0.140 MW at one period, ESS27 0.252 MW at another — which is exactly
what the equality has to remove.

## What changed

`src/local_da_clearing.py`:

- The exact model adds one binary per storage unit and period and encodes the
  disjunction as `ch <= declared_charge_quantity * z` and
  `dis <= declared_discharge_quantity * (1 - z)`. The bound is the unit's own
  declared quantity, so the disjunction is exact and carries no big-M margin.
- `NonConvex` returns to the default: the only quadratics left are the convex
  cones. Presolve and `PreQLinearize` are no longer disabled.
- `BarQCPConvTol` is 1e-8 for both models. The 1e-10 setting existed to make the
  product tolerance imply sub-micro-MW overlap; with the product gone it only
  stalls certification, and 1e-8 keeps the cone inside the 1e-6 physical audit
  (1e-6 leaves that gap at 1.8e-6 and fails the audit).
- The integer guard now protects the pricing model instead of the dispatch: a
  direction-fixed pricing solve must declare no integer variables, because a MIP
  reports no duals and every LMP is read from the balance dual. Pricing keeps
  its continuous model with one leg bounded at zero, so the duals are unchanged
  in method.

`src/make_formal_training_manifests.py`: `CANDIDATE` is `V7_candidate_002`, the
declared `clearing_engine` names the disjunction, and the window-limit comment
records what the disjunction buys.

`src/rl_da_env.py`: the gap tolerance is now handed to the solver as its own
stopping rule instead of only being applied to an incumbent found after the
time limit. A window certifies to the declared 1% and stops there; before, the
solver kept closing the gap toward 1e-6 and spent the whole limit doing it. A
dispatch whose reported gap exceeds `CERTIFIED_GAP` is recorded as accepted on
the relaxed tolerance, which is what widens the pricing comparison, so
`gap_solved_blocks` counts every block whose payoff is known only to 1%.

`tests/test_local_da_market.py`: the dispatch now declares binaries and the
pricing solve still declares none; a bid that pays on both legs at once must
clear rather than consume its limit.

## Acceptance claim that changes

The market no longer carries "the local market must not declare integer
variables". It carries the narrower statement that matters for pricing: the
conditional pricing solve declares none, so its duals exist. The dispatch
declares one binary per storage unit and period — 80 for a 20-period window over
four units.

## Cost

One seeded day (`2021-06-04`, 24 blocks) driven by the launcher's own policy
path, per block, wall clock including the policy window, its truthful-baseline
window and the pricing solve:

| encoding | mean s/block | day |
| --- | --- | --- |
| bilinear equality | 4.6 | one block aborted after 300 s with no incumbent |
| binary disjunction | 1.65 | 40 s, every block optimal |

The reduced launcher run (`--train-days 2`, one full validation round) took
48 s and 59 s for its two training episodes. The block-step budget of
21,600-32,400 per seed is therefore back inside the recorded 11.95-17.92 h
projection rather than the 69-104 h the bilinear encoding implied.

## Gate proof

`PYTHONPATH=src python -B src/run_formal_training.py --config-dir
reports/v2_cy_jn_2025/formal_training_v7 --check`:

```
data_payload_matches true     budget_contract_ok true
code_sources_match true       code_tests_match true
config_hash_matches true      data_manifest_hash_matches true
code_manifest_hash_matches true   library_counts true
all_disjoint true             legacy_label_declared true
h2_declared true              all_pass true
status: null    accepted_candidate: V7_candidate_002
config_sha256: 8db03638    contract_violations: []
```

`status` is null for the same reason it was null before: the readiness status is
written only by `--finalize`, from a `pretraining_validation_results.json`
produced on these sources, and no producer for v7 exists.

## Rollback

The superseded manifest set is preserved at
`tmp/manifest_superseded/v7_ea92b0af/`; it is not tracked, and it describes a
source tree that no longer exists, so restoring it would require reverting this
record's commit as well.

## Not validated

- No formal run and no formal round exist under this hash either. The reduced
  launcher run is a smoke, and `formal_launch` is false for it.
- The qualification and latency tables in the earlier acceptance report were
  measured on the bilinear encoding. They are not carried over as evidence about
  this one; the feasibility-selection count and the price qualification counts
  have to be re-measured before a GO.
