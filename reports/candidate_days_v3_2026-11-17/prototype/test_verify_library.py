import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import verify_library

ROOT = Path(__file__).resolve().parents[3]
REAL_LIBRARY = ROOT / "reports/candidate_days_v3_2026-11-17/evidence"


class VerifyTest(unittest.TestCase):
    @unittest.skipUnless((REAL_LIBRARY / "split.json").exists(), "library not built")
    def test_verifier_flags_two_meters_on_one_bus(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp)
            for name in ("day_selection.csv", "source_bus_mapping.csv",
                         "bus_active_load_mw.npz", "bus_reactive_load_mvar.npz",
                         "split.json", "manifest.json"):
                shutil.copy(REAL_LIBRARY / name, lib / name)
            with open(lib / "source_bus_mapping.csv", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            first_date = rows[0]["date"]
            same_day = [r for r in rows if r["date"] == first_date]
            same_day[1]["bus_zero_based"] = same_day[0]["bus_zero_based"]
            with open(lib / "source_bus_mapping.csv", "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = verify_library.verify(lib)
            self.assertFalse(result["checks"]["one_meter_per_bus"])
            for name, value in result["checks"].items():
                if name != "one_meter_per_bus":
                    self.assertTrue(value, name)


if __name__ == "__main__":
    unittest.main()
