"""The forward day builder reads the v3 library at its own bus assignment."""

import json
import unittest

import numpy as np
import pandapower.networks as networks

from local_da_operating import build_historical_day
from run_local_da import PRICE_CURVES, V3, build_day, build_legacy_day


def v3_pair(date):
    split = json.loads((V3 / "split.json").read_text(encoding="utf-8"))
    pairs = [p for rows in split["pairs"].values() for p in rows
             if p.split("|")[0] == date]
    if len(pairs) != 1:
        raise AssertionError(f"{date} is not a unique v3 pair")
    return pairs[0].split("|")[1]


class InputTests(unittest.TestCase):
    def test_library_day_declares_the_default_substation_voltage(self):
        day, _ = build_day("2021-06-04")
        self.assertEqual(day.network.slack_vm_pu, 1.)

    def test_v3_day_reads_its_own_curves_and_declared_price_date(self):
        date = "2021-06-04"
        price_date = v3_pair(date)
        day, context = build_day(date)
        self.assertEqual((context["load_date"], context["price_date"]), (date, price_date))
        self.assertEqual(day.active_load_mw.shape, (96, 33))
        with np.load(V3 / "bus_active_load_mw.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(day.active_load_mw, archive[date])
        with np.load(PRICE_CURVES, allow_pickle=False) as archive:
            np.testing.assert_array_equal(day.external_price_cny_mwh, archive[f"south/{price_date}"])

    def test_v3_day_carries_its_declared_reactive_basis(self):
        date = "2021-06-04"
        day, _ = build_day(date)
        with np.load(V3 / "bus_reactive_load_mvar.npz", allow_pickle=False) as archive:
            loaded_q = archive[date]
        np.testing.assert_array_equal(day.reactive_load_mvar, loaded_q)
        loaded = day.active_load_mw > 0
        np.testing.assert_allclose(day.reactive_load_mvar[loaded] / day.active_load_mw[loaded],
                                   0.6190, atol=1e-9)

    def test_date_outside_the_v3_library_is_refused(self):
        with self.assertRaises(ValueError):
            build_day("2022-12-22")

    def test_historical_reproduction_path_keeps_its_own_library(self):
        day, context = build_legacy_day("2022-12-22")
        self.assertEqual(day.active_load_mw.shape, (96, 33))
        self.assertEqual(context["load_date"], "2022-12-22")
        self.assertIn("pair_key", context)

    def test_legacy_day_keeps_its_native_reactive_basis(self):
        date = "2022-12-22"
        day, _ = build_legacy_day(date)
        net = networks.case33bw()
        ratio = np.zeros(33)
        for _, row in net.load.iterrows():
            ratio[int(row.bus)] = row.q_mvar / row.p_mw
        np.testing.assert_array_equal(day.reactive_load_mvar, day.active_load_mw * ratio[None, :])

    def test_historical_remapping_still_resolves_its_old_library(self):
        day, context, kept = build_historical_day("2022-12-22")
        self.assertEqual(day.active_load_mw.shape, (96, 33))
        self.assertTrue(kept)
        self.assertEqual(context["load_date"], "2022-12-22")


if __name__ == "__main__":
    unittest.main()
