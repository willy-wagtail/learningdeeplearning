import unittest

import numpy as np

from course_1.src import L2


class L2Test(unittest.TestCase):

    def test_L2(self):
        yhat = np.array([.9, 0.2, 0.1, .4, .9])
        y = np.array([1, 0, 0, 1, 1])
        expected = 0.43
        self.assertEqual(L2.L2(yhat, y), expected)


if __name__ == '__main__':
    unittest.main()
