"""Name the library days the market cannot serve, from the screened evidence.

The library is never edited: the exclusions are their own artifact, so a day can
be dropped for feasibility while the frozen split and its hashes stay intact.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "reports/candidate_days_v3_clearing_2026-11-17/evidence_audits_slack1.00"
OUTPUT = ROOT / "reports/candidate_days_v3_clearing_2026-11-17/excluded_days.json"
BLOCKS = ("train", "validation", "test", "holiday")


def main():
    rows = list(csv.DictReader((EVIDENCE / "summary.csv").open(encoding="utf-8")))
    excluded = [{"date": row["date"], "block": row["block"], "price_date": row["price_date"],
                 "solver_status": int(row["solver_status"]), "failure": row["failure"]}
                for row in rows if row["feasible"] != "True"]
    rule = json.loads((EVIDENCE / "rule.json").read_text(encoding="utf-8"))
    record = {
        "source": f"{EVIDENCE.relative_to(ROOT).as_posix()}/summary.csv",
        "feasible_rule": rule["feasible"],
        "declared_inputs": {"slack_vm_pu": rule["slack_vm_pu"],
                            "shunts_mvar": rule["shunts_mvar"]},
        "excluded": excluded,
        "counts": {block: {"screened": sum(row["block"] == block for row in rows),
                           "excluded": sum(day["block"] == block for day in excluded)}
                   for block in BLOCKS},
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["counts"], indent=2))
    for day in excluded:
        print(day["date"], day["block"], day["solver_status"])


if __name__ == "__main__":
    main()
