from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_scheme_a_glm_sequences import build


class SchemeAGlmSequenceTest(unittest.TestCase):
    def test_aligns_by_key_and_groups_by_stock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = root / "a.csv"
            glm = root / "features_48d_2025-05.csv"
            a_features = [f"a{i}" for i in range(19)]
            glm_values = [f"g{i}" for i in range(48)]
            glm_masks = [f"g{i}__observed" for i in range(48)]
            keys = [("2025-05-02", "B"), ("2025-05-01", "A"),
                    ("2025-05-01", "B"), ("2025-05-02", "A")]
            with a.open("w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow([*KEYS, *a_features])
                for n, key in enumerate(keys): writer.writerow([*key, *([n] * 19)])
            with glm.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([*KEYS, *sum(zip(glm_values, glm_masks), ())])
                for n, key in enumerate(keys):
                    row = []
                    for _ in range(48): row.extend([10 + n, 1])
                    writer.writerow([*key, *row])
            output = root / "output"
            build(a, [glm], output, overwrite=False)
            values = np.load(output / "values.npy")
            self.assertEqual(values.shape, (2, 2, 115))
            np.testing.assert_array_equal(values[0, 0, :19], [1] * 19)
            np.testing.assert_array_equal(values[0, 0, 19:67], [11] * 48)
            np.testing.assert_array_equal(values[0, 0, 67:], [1] * 48)


KEYS = ("date", "stock_id")


if __name__ == "__main__":
    unittest.main()
