import pandas as pd
import unittest
from double_descent_dar import sample_eval
from parameterized import parameterized

class TestSampleEval(unittest.TestCase):
    @parameterized.expand([
        (False),
        (True),
    ])
    def test_expected_shapes(self, replace):
        X = pd.DataFrame({'a': range(0, 9), 'b': range(10, 19)})
        y = pd.DataFrame({'target': X['a'] * 10 + X['b']})
        random_state = 1
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
            sample_eval.sample_rows_and_cols(X, y, 2, 2, random_state=1, replace=True)

    def test_sample_too_many_rows(self):
        X = pd.DataFrame({'a': range(0, 9), 'b': range(10, 19)})
        y = pd.DataFrame({'target': range(0, 9)})
        with self.assertRaisesRegex(ValueError, 'larger sample'):
            sample_eval.sample_rows_and_cols(X, y, 11, 2, random_state=1, replace=False)

    def test_sample_too_many_cols(self):
        X = pd.DataFrame({'a': range(0, 9), 'b': range(10, 19)})
        y = pd.DataFrame({'target': range(0, 9)})
        with self.assertRaisesRegex(ValueError, 'larger sample'):
            sample_eval.sample_rows_and_cols(X, y, 3, 3, random_state=1, replace=False)



if __name__ == '__main__':
    unittest.main()
