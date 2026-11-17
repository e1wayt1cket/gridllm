import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_library
import library_rules


class BuildTest(unittest.TestCase):
    def test_build_writes_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pool = tmp / "pool"
            pool.mkdir()
            day = np.full(96, 2.0)
            np.savez(pool / "meter_curves.npz", **{"2022-11-01|m1": day,
                                                   "2022-11-01|m2": np.full(96, 1.5)})
            (pool / "meter_sources.csv").write_text(
                "date,source_id,user_id,curve_sha256\n"
                f"2022-11-01,m1,u1,{library_rules._array_sha256(day)}\n"
                f"2022-11-01,m2,u2,{library_rules._array_sha256(np.full(96, 1.5))}\n",
                encoding="utf-8")
            out = tmp / "out"
            manifest = build_library.run(pool, out, {"11": ("2025-11-01", "2025-11-02")})
            for name in ("day_selection.csv", "source_bus_mapping.csv",
                         "bus_active_load_mw.npz", "bus_reactive_load_mvar.npz",
                         "split.json", "manifest.json"):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(manifest["library_size"], 1)
            with np.load(out / "bus_reactive_load_mvar.npz") as npz:
                q = npz["2022-11-01"]
            with np.load(out / "bus_active_load_mw.npz") as npz:
                p = npz["2022-11-01"]
            self.assertTrue((np.abs(q - 0.6190 * p) < 1e-9).all())


if __name__ == "__main__":
    unittest.main()
