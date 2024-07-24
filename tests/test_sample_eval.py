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
        X = pd.DataFrame({'a': range(1, 10), 'b': range(2, 11), 'c': range(3, 12)})
        y = pd.DataFrame({'target': range(100, 110)})
        random_state = 1
        num_rows = 3
        num_cols = 2
        # precondition: y should have one column
        self.assertEqual(y.shape[1], 1)
        X_sub, y_sub = sample_eval.sample_rows_and_cols(X, y, num_rows, num_cols, random_state=random_state, replace=replace)
        self.assertEqual(y_sub.shape, (num_rows, 1))
        self.assertEqual(X_sub.shape, (num_rows, num_cols))


if __name__ == '__main__':
    unittest.main()
