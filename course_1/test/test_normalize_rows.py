import unittest

import numpy as np

from course_1.src import normalize_rows


class NormalizeRowsTest(unittest.TestCase):

    def test_normalize_rows(self):
        x = np.array([[0, 3, 4], [1, 6, 4]])
        expected = np.array([[0., 0.6, 0.8], [0.13736056, 0.82416338, 0.54944226]])
        self.assertTrue(np.allclose(normalize_rows.normalize_rows(x), expected))


if __name__ == '__main__':
    unittest.main()
