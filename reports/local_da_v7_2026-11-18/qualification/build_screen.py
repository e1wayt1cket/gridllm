"""Write the per-day feasibility screen from the full-library measurement.

`local_da_days.train_pairs` reads one record per day and drops a day whose
exact dispatch was not optimal. The v3 screen it used predates the market's
declared reactive asset, so this rewrites it from the re-measurement: the
records here are the same reading the qualification table reports, not a
second pass over the library.
"""

import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

records = {}
for name in sorted(glob.glob(str(HERE / "library_*.json"))):
    for record in json.loads(Path(name).read_text(encoding="utf-8"))["sample"]:
        records[record["date"]] = record
if not records:
    raise SystemExit("No library measurement to build the screen from")

out = HERE / "screen"
out.mkdir(exist_ok=True)
for date, record in records.items():
    (out / f"{date}.json").write_text(json.dumps({
        "date": date,
        "dispatch_status": record["dispatch_status"],
        "failure": record.get("failure"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"{len(records)} screen records in {out}")
