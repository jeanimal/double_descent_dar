import numpy as np
import pandas as pd
import unittest

from parameterized import parameterized
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model

from double_descent_dar import sample_eval

class TestSampleEval(unittest.TestCase):
    @parameterized.expand([
        (False),
        (True),
    ])
    def test_expected_shapes(self, replace):
        X = pd.DataFrame({'a': range(0, 9), 'b': range(10, 19)})
        y = pd.DataFrame({'target': X['a'] * 10 + X['b']})
        random_state = np.random.default_rng()
        num_rows = 3
        num_cols = 2
        # precondition: y should have one column
        self.assertEqual(y.shape[1], 1)
        X_sub, y_sub = sample_eval.sample_rows_and_cols(X, y, num_rows, num_cols, random_state=random_state, replace=replace)
        # Confirm shape.
        self.assertEqual(y_sub.shape, (num_rows, 1))
        self.assertEqual(X_sub.shape, (num_rows, num_cols))

    def test_different_number_of_rows(self):
        X = pd.DataFrame({'a': range(0, 9), 'b': range(10, 19)})
        y = pd.DataFrame({'target': range(0, 8)})
        with self.assertRaisesRegex(ValueError, 'same number of rows'):
            sample_eval.sample_rows_and_cols(X, y, 2, 2, replace=True)

    def test_sample_too_many_rows(self):
        X = pd.DataFrame({'a': range(0, 9), 'b': range(10, 19)})
        y = pd.DataFrame({'target': range(0, 9)})
        with self.assertRaisesRegex(ValueError, 'larger sample'):
            sample_eval.sample_rows_and_cols(X, y, 11, 2, replace=False)

    def test_sample_too_many_cols(self):
        X = pd.DataFrame({'a': range(0, 9), 'b': range(10, 19)})
        y = pd.DataFrame({'target': range(0, 9)})
        with self.assertRaisesRegex(ValueError, 'larger sample'):
            sample_eval.sample_rows_and_cols(X, y, 3, 3, replace=False)

    def test_split_and_calc_metric(self):
        X = pd.DataFrame({'a': [1, 5, 3, 4, 2, 6, 0, 7, 8],
                          'b': [8, 7, 0, 6, 3, 4, 3, 5, 1]})
        y = pd.DataFrame({'target': range(0, 9)})
        model = linear_model.LinearRegression()
        train_test_metric = sample_eval.split_and_calc_metric(X, y, 0.5, model, metric_func=mean_absolute_error)
        self.assertGreaterEqual(train_test_metric['train'], 0)
        self.assertGreater(train_test_metric['test'], train_test_metric['train'])

    def test_sample_and_calc_metric(self):
        X = pd.DataFrame({'a': [1, 5, 3, 4, 2, 6, 0, 9, 7, 8],
                          'b': [8, 7, 9, 0, 6, 3, 4, 3, 5, 1]})
        y = pd.DataFrame({'target': range(0, 10)})
        num_rows = 6
        num_cols = 2
        model = linear_model.LinearRegression()
        train_test_metric = sample_eval.sample_and_calc_metric(X, y, num_rows, num_cols, 0.5, model, mean_absolute_error, replace=False)
        self.assertGreaterEqual(train_test_metric['train'], 0)
        self.assertGreater(train_test_metric['test'], train_test_metric['train'])


if __name__ == '__main__':
    unittest.main()
