import unittest

import numpy as np

from course_1.src import L1


class L1Test(unittest.TestCase):

    def test_L1(self):
        y_hat = np.array([.9, 0.2, 0.1, .4, .9])
        y = np.array([1, 0, 0, 1, 1])
        expected = 1.1
        self.assertEqual(L1.L1(y_hat, y), expected)


if __name__ == '__main__':
    unittest.main()
