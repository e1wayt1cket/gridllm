# The single-day data entry point

One path by which a single-day case's load and price are loaded, validated and
paired. It exists because the load ledger and the price curve describe the same
15-minute grid and **label their periods at opposite ends of it**, and each one
is validated for its own axis where it is read — neither check can see the
other. Pairing them is a third thing, and before this module it lived in one
script with nothing to stop a second consumer building its own, unaligned merge.

## What it is

`src/single_day_data.py`. One constructor, `load_case`, and three refusals.

```
load_case(case_type, ledger_csv, ledger_manifest, price_sheet)
    ├── apply_date_rule          the case type selects the date rule
    ├── load_load_axis           the ledger's own axis, from the ledger
    ├── load_price_axis          the curve's own axis, from the curve
    └── check_alignment          the two together, or a refusal
```

Every failure raises. **Nothing is shifted, interpolated, resampled or
date-substituted to make a pair fit** — a mismatch is a refusal, and the
`alignment` block records `conversion_applied: none`, `interpolation_applied:
none`, `resampling_applied: none` so a reader can see that it was a choice.

## The case type, and why it has no default

Whether two different dates are acceptable depends on what the case claims to
be, so the claim is an input rather than an inference:

| case type | `load_date` vs `price_date` | what it means |
|---|---|---|
| `cross_date_diagnostic` | **may differ** | a real load shape priced against a real price shape; not a historical replay |
| `same_day_replay` | **must be equal** | a replay of one actual day, which needs that day's own market |

A caller that states neither, or something unrecognised, gets
`CaseTypeError` — not the permissive rule. A missing or malformed date is an
error of the same kind, so an absent field cannot satisfy a rule that needs one.
Both dates are recorded as they are and never rewritten to match.

## The period axis, checked on both sides

| what | load ledger | price curve |
|---|---|---|
| periods per day | 96 | 96 |
| sampling interval | 15 min | 15 min |
| label convention | **interval start** (`00:00`…`23:45`) | **interval end** (`00:15`…`24:00`) |
| enforced by | `industry_load._check_time_header` | `dianyicha_price._check_slot_axis` |
| timezone | not stated | not stated |

`PeriodAxis.interval(t)` returns the same `[15t, 15(t+1))` for both conventions,
so a start label and an end label describe **one interval, not two** — which is
why an index-to-index pairing is correct here and why a one-period rotation
would have been invisible.

Refused, each by its own test:

| defect | what happens |
|---|---|
| a repeated period | `DataAlignmentError`, naming the repeated indices |
| a missing period | `DataAlignmentError` |
| a misordered period | `DataAlignmentError` — checked on the file's own order, **not** after sorting, because sorting first would accept a file assembled by concatenating two runs |
| a different period count | `DataAlignmentError` |
| a different sampling interval | `DataAlignmentError`, naming the resampling it will not do |
| periods covering different ground | `DataAlignmentError`, naming the interval each side covers |

## Two ledger states, one of which may become a network input

| state | may it feed the network as absolute MW? |
|---|---|
| `audit_ledger` | **no** — the reading unit is declared, not verified; every row says so |
| `formal_ledger` | yes, and only once `validate_formal_ledger` passes it |

`SingleDayCase.require_formal_load_mw()` is the gate. It refuses an audit ledger,
and it refuses a ledger whose `ledger_kind` claims `formal` while its
`unit_status` is still unverified — the two fields have to agree. It reads the
manifest's own status fields, the ones the validator reads, rather than trusting
a caller's claim.

**The unit gate is still closed** (`unit_status() == unit_unverified`), so
nothing here feeds the real load into an IEEE33 power flow or a formal
settlement. The current case is an `audit_ledger` and says so.

## The validator, split by case type

`validate_formal_ledger.validate(..., case_type=...)`:

| `--case-type` | outcome on the current audit ledger |
|---|---|
| *omitted* | `unit_not_verified` + **`case_type_unstated`** + `price_not_date_aligned` — the strict rule **and** a violation naming the omission |
| `cross_date_diagnostic` | `unit_not_verified` only |
| `same_day_replay` | `unit_not_verified` + `price_not_date_aligned` |

So `price_not_date_aligned` no longer stands in the way of a cross-date
diagnostic, `unit_not_verified` still does, and there is no path by which
omitting the case type buys the laxer rule. Verified by running all three, above.

## Which callers are on the entry point

| caller | status |
|---|---|
| `single_day_case.py` | **wired** — reads the curve, writes the artifacts, delegates every pairing decision |
| `tests/test_single_day_data.py` | the acceptance suite |
| everything else | does not pair a ledger with a curve, so has nothing to route |

The guard is a test, not a convention:
`test_every_module_that_pairs_a_ledger_with_a_curve_uses_the_entry_point` scans
`src/` and fails on any module that reads both a node-load ledger and a price
curve without importing `single_day_data`. It currently finds none; the predicate
was checked to fire when the import is removed from `single_day_case.py`, and
not to fire on `market.py`, which consumes a price but never a ledger.

`test_the_alignment_rule_has_exactly_one_implementation` asserts `PeriodAxis` and
`check_alignment` are defined in that one module and nowhere else.

### What is **not** routed through it, stated rather than implied

The ordinary multi-scenario price path is unchanged: `scenarios.py` →
`grid.day_ahead_price_china` → `wholesale_price.get_provider` →
`DianyichaPriceProvider` → `dianyicha_price.read_curves`. That path has no load
ledger in it, so there is no cross-curve alignment to check; its own intra-curve
axis **is** validated on every read by `read_curves`, which raises on a repeated,
misordered or non-interval-end label. Routing that path through this module would
mean inventing a second axis check for series that are not paired with anything,
and it was not done.

Three audit modules read the curves directly — `accept_typical_day_industry`,
`phase7_strategy_comparison`, `phase9_sensitivity`. They use a curve as an
exogenous reference price series alongside `get_scenario(..., day_id=...)`, not
as a load-paired day, and none of them reads a node-load ledger. They are not
formal clearing paths.

## Provenance kept

`case_manifest.json` carries, and `SingleDayCase.describe()` returns:

```
load_date            2022-11-30
price_date           2026-10-09
same_day             false
case_type            cross_date_diagnostic
load_source          ledger path, ledger_sha256, selection_rule,
                     selection_evidence, users_with_data, users_per_bus,
                     industry_aggregation
price_source         workbook, sha256, sheet, and how the date was derived
unit_status          unit_unverified   (reading_unit, reading_unit_evidence)
load_kind            audit_ledger
mapping_version      industry_bus_map_v2
period_alignment     both axes, the method, and the three "none"s
input_fingerprints   five hashes, both workbooks and both derived files
integrity_checks     load coverage, users, exclusions and the aggregation;
                     the price's enforced checks
```

Reproduce:

```
PYTHONPATH=src python src/build_industry_node_loads.py --audit-dates
PYTHONPATH=src python src/build_industry_node_loads.py --date-from-audit
PYTHONPATH=src python src/single_day_case.py
PYTHONPATH=src python src/validate_formal_ledger.py \
    --ledger data/processed/node_loads/typical_day_industry/bus_loads_96x33_audit.csv \
    --case-type cross_date_diagnostic
```

## Test results

```
PYTHONPATH=src pytest -q tests/test_single_day_data.py                 24 passed
PYTHONPATH=src pytest -q tests/test_single_day_case.py     tests/test_single_day_data.py                                      35 passed
PYTHONPATH=src pytest -q tests/test_formal_ledger_validator.py         15 passed

PYTHONPATH=src python src/single_day_case.py          # case_type cross_date_diagnostic
PYTHONPATH=src python src/single_day_case.py --case-type same_day_replay
    -> CaseTypeError: a same_day_replay needs one actual day …

PYTHONPATH=src python src/validate_formal_ledger.py --ledger …/bus_loads_96x33_audit.csv
    (no --case-type)          REJECTED: 3   unit_not_verified, case_type_unstated,
                                            price_not_date_aligned
    --case-type cross_date_diagnostic   REJECTED: 1   unit_not_verified
    --case-type same_day_replay         REJECTED: 2   unit_not_verified,
                                            price_not_date_aligned
```

Full suite, exit code 0:

```
PYTHONPATH=src pytest -q
1137 passed, 53 deselected, 17 warnings in 802.40s (0:13:22)
```

1137 against the previous run's 1113: +24 for the entry-point suite, and the
two alignment tests that moved out of `test_single_day_case.py` were replaced by
one that asserts the case script no longer owns a second implementation. The 53
deselected are the `slow` tests, not skips.

## What this does not claim

- It is **not** a physical clearing experiment. No market has been cleared on
  this case; the ledger is an audit ledger and the scenario builder does not read
  it.
- The two curves are from **different dates**. Nothing computed from this case is
  a historical replay, nor a statement about what either day's market would have
  paid.
- The price date is a **parsed sheet label**, the workbook states **no timezone**,
  and the single-day reading is a **project ruling** — all three travel with the
  value.
