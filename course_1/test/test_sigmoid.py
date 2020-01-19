import unittest

import numpy as np

from course_1.src import sigmoid


class SigmoidTest(unittest.TestCase):

    def setUp(self) -> None:
        self.x = np.array([1, 2, 3])

    def test_sigmoid(self):
        expected = np.array([0.73105858, 0.88079708, 0.95257413])
        self.assertTrue(np.allclose(sigmoid.sigmoid(self.x), expected))

    def test_sigmoid_derivative(self):
        expected = np.array([0.19661193, 0.10499359, 0.04517666])
        self.assertTrue(np.allclose(sigmoid.sigmoid_derivative(self.x), expected))


if __name__ == '__main__':
    unittest.main()
