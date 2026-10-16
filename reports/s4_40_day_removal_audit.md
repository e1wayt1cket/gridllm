# S4: removing the fixed 40-day system

What the 40-day assumption was, where it lived, what was deleted, what was kept
because it is generic, and what still holds the formal path shut. Every claim
here is from a search or a test run in this session; nothing is recorded as done
that was not executed.

**Baseline**: `HEAD = 9f61c9a`, branch `master`, 29 uncommitted paths before this
stage. Nothing was committed or pushed.

## What the assumption actually was

Not a number scattered through the code. It was a **frozen 40-day synthetic
price library** whose length was declared project-wide, and a **20/10/10 split
over day ids 0..39** that everything downstream was expressed against:

```
config/defaults.yaml   wholesale_market.number_of_days: 40, master_seed,
                       day_id: 11, library_file
wholesale_price.py     SyntheticWholesalePriceProvider, defaults 40 / 20260920
src/day_rotation.py    LIBRARY_DAYS = 40, SPLIT_SIZES = 20/10/10, SPLIT
market_opportunity.py  ENSEMBLE_SIZE = 40, MASTER_SEED, SPLIT_FRACTIONS
train_rl.py            DEFAULT_TRAIN_SCENARIO, `auto` gate on "synthetic_v1"
config/scenarios.yaml  typical_day.wholesale_source: synthetic_v1
```

The audit looked for **dependence on the structure**, not for the string `40`: a
caller that divides by a training-set length, a default that restores a size, a
provider selectable without anyone asking. That is how the items below were
found even though several of them no longer contained the number.

## Deleted

| what | where | why it had to go |
|---|---|---|
| `LIBRARY_DAYS`, `SPLIT_SIZES`, `SPLIT`, `split_of`, `validation_days`, `test_days`, `assert_protocol`, `episode_plan`'s fixed size | `day_rotation.py` | the whole fixed split |
| `number_of_days: 40`, `master_seed`, `day_id: 11`, `library_file` | `config/defaults.yaml` | a dataset length declared project-wide is a length every real dataset is then measured against |
| `ENSEMBLE_SIZE = 40`, `MASTER_SEED` | `market_opportunity.py` | defaults that decide an ensemble the caller never asked for |
| `DEFAULT_TRAIN_SCENARIO`, the `auto` rotation mode, the `"synthetic_v1"` string gate, `--day-rotation` | `train_rl.py` | a default research scenario, and rotation keyed on a provider name |
| `wholesale_source: synthetic_v1` as an unqualified choice | `config/scenarios.yaml` | replaced by an explicit `formal_data_ready: false` marker plus the source rule |
| the old rotation tests | `tests/test_day_rotation.py` | rewritten for the length-agnostic scheduler |

## Kept, because it is not the 40-day system

| kept | where | why |
|---|---|---|
| `cycle_seed`, `cycle_days`, `episode_day`, `is_cycle_end`, `episode_plan` | `day_rotation.py` | a permutation of whatever days the caller supplies; no size, no split |
| `assert_days` | `day_rotation.py` | new: refuses an empty, repeated or blank day list |
| `SPLIT_FRACTIONS = (0.5, 0.25, 0.25)` | `market_opportunity.py` | proportions, not a day count; sizes follow from what the caller supplies |
| `partition_days`, `assign_partitions`, `underfilled_strata` | `market_opportunity.py` | a stratified deal over whatever labels exist |
| `SyntheticWholesalePriceProvider`, `LibraryPriceProvider`, `build_library` | `wholesale_price.py` | a market with a known shape, needed by tests and audits — now **gated** |
| `ensemble_specs`, `series_specs`, `sweep_specs` | `market_opportunity.py` | `n` and `seed` are now required arguments rather than module defaults |

The scheduler's contract is now: *it does not know how many days there are*.
`day_rotation.py` takes the days that exist and decides an order; splitting them
is a protocol decision that belongs with the real date set and its coverage.

## Provider sources, by what they read

| source | reads | formal? | reachable without asking? |
|---|---|---|---|
| `dianyicha` | the measured Jiangsu workbook | **yes** | yes |
| `legacy` | the retired net-load map — an accounting identity over this network's own dispatch | no | yes, but flagged |
| `synthetic_v1` | a generated external market | **no** | **no** — refused unless `ALLOW_NON_FORMAL_SOURCES` is set |
| `library` | the committed 40-day CSV of the above | **no** | **no** — same gate |
| `guangdong` | — | raises `NotImplementedError` | n/a |

`get_provider` refuses a non-formal source **before** the provider cache, not
after: a provider built while the gate was open would otherwise be handed to
every later caller with the gate shut, which was the one way the restriction
could be bypassed. That ordering was wrong in the first draft and a test caught
it.

## Which scenarios may enter formal evaluation

The rule is over the **source**, not a flag on each scenario, so a scenario
added later cannot be formal-ready by omission.

| scenario | source | formal-ready |
|---|---|---|
| `typical_day_industry` | `dianyicha` | by source **yes** — and still blocked by the unit gate and the price-date gap (S3) |
| `typical_day` | `synthetic_v1` | **no**, marked |
| `baseline`, `high_re`, `peak_load`, `congestion`, `no_congestion`, `tight_bottleneck`, `re_ramp_drop`, `re_ramp_surge` | `legacy` | **no**, marked |

The legacy scenarios were **not** marked when this stage began. A test written
for this stage found them: they price from the retired net-load map, which is
computed from the distribution network's own dispatch and is therefore not a
measured market. All eight now carry the marker and a reason.

## Real-date intersection

Re-checked this stage, unchanged from S3:

| | |
|---|---|
| real load dates | 730, 2021-01-01 to 2022-12-31 |
| dates with a dated real price curve | **0** |
| the only dated real price curve | `261009` = 2026-10-09, outside the load range |
| the other three curve roles | window means, `window_start`/`window_end` = `None` |

No date was copied, no price was fabricated, and nothing falls back to a
generated series. There is no date on which a real load day and a real price
curve are both in hand.

## Tests

`tests/test_day_rotation.py` (18) and `tests/test_forty_day_removal.py` (20),
written for this stage, plus the existing suites re-run.

| what the new tests pin |
|---|
| every removed constant is absent from every source file, matched on word boundaries so the namespaced non-formal constants are not mistaken for them |
| `defaults.yaml` declares no dataset length (asserted against the file, because `day_id` is a runtime knob `scenarios.py` writes) |
| a training run must name its scenarios; there is no default |
| the audit ensemble size is the caller's, and `ensemble_specs()` with no size is a `TypeError` |
| a synthetic provider is refused when the gate is closed, and builds when a caller opens it deliberately |
| every scenario priced by a non-formal source is marked not formal-ready |
| `load_shape: diversified` does not bypass the data requirement |
| the scheduler works for 1, 2, 5, 37 and 400 days; three days stay three days across seven cycles; nothing is padded, reused or fabricated |
| a cycle is a permutation; the order depends on the seed and not on the data; the stream is stateless |

Exit codes, counts and failures are recorded below from the actual runs.

### Actual results

```
PYTHONPATH=src pytest -q tests/test_day_rotation.py
18 passed in 0.15s

PYTHONPATH=src pytest -q tests/test_forty_day_removal.py
20 passed

PYTHONPATH=src pytest -q            # full suite, exit code 0
1005 passed, 53 deselected, 17 warnings in 421.47s (0:07:01)
```

The 53 deselected are the `slow` tests (`pytest.ini`), not skips. Two
intermediate runs during this stage are worth recording rather than hiding:
the first full run after the code changes was **1003 passed, 2 failed**, and
both failures were the new tests doing their job — one caught a `day_id` that
`scenarios.py` writes at runtime with `set_default` (so the check had to move to
the file), the other was an old assertion of the removed default scenario. A
test in this stage also caught a real defect: the synthetic gate was checked
**after** the provider cache, so a cached synthetic provider bypassed it.

## Still open

1. **The reading unit is `unit_unverified`** — unchanged, and S4 was not
   permitted to resolve it.
2. **No real price date intersects a real load date** — unchanged.
3. **Only 30 days (all December 2022) cover all 17 mapped buses** — the formal
   date-selection rule is still undecided.
4. **The committed 40-day library CSV** (`data/wholesale/synthetic_da_v1.csv`)
   and its manifest are **retained**. They are no longer reachable from any
   formal path — `library` is behind the same gate as `synthetic_v1` — but the
   file itself is still in the tree. Deleting a tracked artefact that the
   non-formal audits read is a larger change than this stage required, so it is
   reported rather than removed. Say the word and it goes.

## Gate status

| | |
|---|---|
| S3 structure | complete; **formal ledger not frozen** |
| S4 | 40-day system removed; tests below |
| S8 network validation | **blocked** — on the unit gate and the date gap, not on S4 |
| formal MATD3 training | **blocked** |
