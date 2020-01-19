import unittest

import numpy as np

from course_1.src import logistic_regression


class LogisticRegressionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.w, self.b, self.X, self.Y = np.array([[1.], [2.]]), 2., np.array(
            [[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])

    def test_initialize_with_zeros_returns_b_as_zero(self):
        w, b = logistic_regression.initialize_with_zeros(2)
        self.assertEqual(b, 0)

    def test_initialize_with_zeros_returns_w_as_zero_vector(self):
        w, b = logistic_regression.initialize_with_zeros(2)
        expected = np.array([[0.], [0.]])
        self.assertTrue((w == expected).all())

    def test_propagation_returns_dw(self):
        grads, cost = logistic_regression.propagate(self.w, self.b, self.X, self.Y)
        expected_dw = np.array([[0.99845601], [2.39507239]])
        self.assertTrue(np.allclose(grads["dw"], expected_dw))

    def test_propagation_returns_cost(self):
        grads, cost = logistic_regression.propagate(self.w, self.b, self.X, self.Y)
        expected_cost = 5.80154531939
        self.assertTrue(np.allclose(cost, expected_cost))

    def test_propagation_returns_db(self):
        grads, cost = logistic_regression.propagate(self.w, self.b, self.X, self.Y)
        expected_db = 0.00145557813678
        self.assertTrue(np.allclose(grads["db"], expected_db))

    def test_optimize_returns_updated_w(self):
        params, grads, costs = logistic_regression.optimize(self.w, self.b, self.X, self.Y, num_iterations=100,
                                                            learning_rate=0.009, print_cost=False)
        expected_w = np.array([[0.19033591], [0.12259159]])
        self.assertTrue(np.allclose(params["w"], expected_w))

    def test_optimize_returns_updated_b(self):
        params, grads, costs = logistic_regression.optimize(self.w, self.b, self.X, self.Y, num_iterations=100,
                                                            learning_rate=0.009, print_cost=False)
        expected_b = 1.92535983008
        self.assertTrue(np.allclose(params["b"], expected_b))

    def test_optimize_returns_dw(self):
        params, grads, costs = logistic_regression.optimize(self.w, self.b, self.X, self.Y, num_iterations=100,
                                                            learning_rate=0.009, print_cost=False)
        expected_dw = np.array([[0.67752042], [1.41625495]])
        self.assertTrue(np.allclose(grads["dw"], expected_dw))

    def test_optimize_returns_db(self):
        params, grads, costs = logistic_regression.optimize(self.w, self.b, self.X, self.Y, num_iterations=100,
                                                            learning_rate=0.009, print_cost=False)
        expected_db = 0.219194504541
        self.assertTrue(np.allclose(grads["db"], expected_db))

    def test_optimize_returns_costs(self):
        params, grads, costs = logistic_regression.optimize(self.w, self.b, self.X, self.Y, num_iterations=100,
                                                            learning_rate=0.009, print_cost=False)
        expected_costs = [5.8015453193945534]
        self.assertTrue(np.allclose(costs, expected_costs))

    def test_predict_returns_prediction(self):
        w = np.array([[0.1124579], [0.23106775]])
        b = -0.3
        X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
        expected_predictions = np.array([[1., 1., 0.]])
        self.assertTrue((logistic_regression.predict(w, b, X) == expected_predictions).all())


if __name__ == '__main__':
    unittest.main()
