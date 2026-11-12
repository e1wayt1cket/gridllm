# Reporting

Applies to every phase report, audit, validation report and status update.

## Verdict vocabulary

Every claim carries exactly one of:

- `PASS` — a command, test or artifact demonstrates it.
- `FAIL` — a command, test or artifact contradicts it.
- `NOT_TESTED` — no evidence exists; say why.
- `INFERRED` — derived from other evidence; name that evidence.
- `BLOCKED` — an external condition prevents verification.

## Required evidence

Each `PASS` names the command and a result summary, or the artifact path and
the value read from it. Each `FAIL` does the same. A count such as "1365 tests
passed" does not establish that a budget rule is correct; the budget check is
its own line item.

## Four different things

Keep these distinct, in this order:

1. code exists;
2. code ran;
3. a test covers it and passed;
4. the scientific conclusion follows.

A later item never implies an earlier one. Never present inference as
measurement. Never report a skipped run as completed.

## Unverified items

List them explicitly under `NOT_TESTED` or `BLOCKED` with the reason and the
risk they carry. Do not omit a known gap because the rest of the phase passed.

## Frozen and historical material

Reports about a run cite its `run_manifest.json`, `eval_history.json`,
`episodes.csv` and the audit files beside them. They do not restate a number
without the path it came from, and they never rewrite the run's own artifacts.
