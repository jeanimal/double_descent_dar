import pandas as pd
from double_descent_dar import sample_eval


def test_train_test_split_by_rows_and_cols():
    """Check output shapes"""
    X = pd.DataFrame({'a': [1, 5, 3, 4, 2, 6, 0, 9, 7, 8],
                      'b': [8, 7, 9, 0, 6, 3, 4, 3, 5, 1]})
    y = pd.DataFrame({'target': range(0, 10)})
    num_rows = 6
    num_cols = 2
    X_train, X_test, y_train, y_test = sample_eval.train_test_split_by_rows_and_cols(X, y, num_rows, num_cols)
    assert y_train.shape == (num_rows, 1), 'y_train shape is not correct'
    assert X_train.shape == (num_rows, num_cols), 'X_train shape is not correct'
    expected_rows = X.shape[0] - num_rows
    assert y_test.shape == (expected_rows, 1), 'y_test shape is not correct'
    assert X_test.shape == (expected_rows, num_cols), 'X_test shape is not correct'