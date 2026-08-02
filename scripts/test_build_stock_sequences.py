from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_stock_sequences import build_sequences


class BuildStockSequencesTest(unittest.TestCase):
    def test_groups_by_stock_and_sorts_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            a_path, b_path = root / "a.csv", root / "b.csv"
            keys = [
                ("2024-01-02", "000002.SZ"),
                ("2024-01-01", "000001.SZ"),
                ("2024-01-01", "000002.SZ"),
                ("2024-01-02", "000001.SZ"),
            ]
            for path, feature, offset in [(a_path, "a", 0), (b_path, "b", 10)]:
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["date", "stock_id", feature])
                    for index, key in enumerate(keys):
                        writer.writerow([*key, index + offset])

            output = root / "sequences"
            build_sequences(a_path, b_path, output, progress_every=0)

            stocks = np.load(output / "stock_ids.npy").tolist()
            dates = np.load(output / "dates.npy").tolist()
            values = np.load(output / "values.npy")
            self.assertEqual(stocks, ["000001.SZ", "000002.SZ"])
            self.assertEqual(dates, ["2024-01-01", "2024-01-02"])
            np.testing.assert_array_equal(values[0], [[1, 11], [3, 13]])
            np.testing.assert_array_equal(values[1], [[2, 12], [0, 10]])

    def test_rejects_misaligned_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            a_path, b_path = root / "a.csv", root / "b.csv"
            a_path.write_text("date,stock_id,a\n2024-01-01,A,1\n")
            b_path.write_text("date,stock_id,b\n2024-01-01,B,2\n")
            with self.assertRaisesRegex(ValueError, "key mismatch"):
                build_sequences(a_path, b_path, root / "output", progress_every=0)


if __name__ == "__main__":
    unittest.main()
