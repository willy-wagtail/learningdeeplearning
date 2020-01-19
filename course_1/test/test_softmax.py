import unittest

import numpy as np

from course_1.src import softmax


class SoftmaxTest(unittest.TestCase):

    def test_softmax(self):
        x = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])

        expected = np.array([[9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04],
                             [8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]])

        self.assertTrue(np.allclose(softmax.softmax(x), expected))


if __name__ == '__main__':
    unittest.main()
