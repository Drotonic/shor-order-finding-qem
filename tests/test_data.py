import unittest

import numpy as np

from shor_qem.data import bits_to_counts, bits_to_outcomes


class BitOrderTests(unittest.TestCase):
    def test_column_zero_is_lsb(self):
        bits = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(bits_to_outcomes(bits), [0, 1, 2, 5, 15])
        counts = bits_to_counts(bits, 4)
        self.assertEqual(int(counts.sum()), 5)
        for outcome in (0, 1, 2, 5, 15):
            self.assertEqual(int(counts[outcome]), 1)

    def test_rejects_nonbinary_input(self):
        with self.assertRaises(ValueError):
            bits_to_outcomes(np.array([[0, 2]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
