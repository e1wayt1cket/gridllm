import unittest

import extract_window_pool as w

SCREENING = w.ROOT / "data/processed/industry_block_screening.csv"


class WindowDatesTest(unittest.TestCase):
    def test_window_is_june_to_december_of_two_years(self):
        dates = w.window_dates(SCREENING)
        self.assertEqual(len(dates), 428)
        self.assertEqual(dates[0], "2021-06-01")
        self.assertEqual(dates[-1], "2022-12-31")
        self.assertTrue(all(d[5:7] in {"06", "07", "08", "09", "10", "11", "12"} for d in dates))

    def test_empty_date_rows_are_excluded(self):
        dates = w.window_dates(SCREENING)
        self.assertNotIn("", dates)
        self.assertEqual(len(set(dates)), len(dates))


if __name__ == "__main__":
    unittest.main()
