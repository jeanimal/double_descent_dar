import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error

from double_descent_dar import sample_eval


def test_train_test_split_by_rows_and_cols():
    """Check output shapes"""
    X = pd.DataFrame({'a': [1, 5, 3, 4, 2, 6, 0, 9, 7, 8],
                      'b': [8, 7, 9, 0, 6, 3, 4, 3, 5, 1],
                      'c': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10]})
    y = pd.DataFrame({'target': range(0, 10)})
    num_rows = 6
    num_cols = 2
    X_train, X_test, y_train, y_test = sample_eval.train_test_split_by_rows_and_cols(X, y, num_rows, num_cols)
    assert y_train.shape == (num_rows, 1), 'y_train shape is not correct'
    assert X_train.shape == (num_rows, num_cols), 'X_train shape is not correct'
    expected_rows = X.shape[0] - num_rows
    assert y_test.shape == (expected_rows, 1), 'y_test shape is not correct'
    assert X_test.shape == (expected_rows, num_cols), 'X_test shape is not correct'

def test_train_test_split_by_rows_and_cols_err_too_many_rows():
    """Check output shapes"""
    X = pd.DataFrame({'a': [1, 5, 3, 4, 2, 6, 0, 9, 7, 8],
                      'b': [8, 7, 9, 0, 6, 3, 4, 3, 5, 1]})
    y = pd.DataFrame({'target': range(0, 10)})
    with pytest.raises(ValueError, match=r".*num_train_rows must.*"):
        _ = sample_eval.train_test_split_by_rows_and_cols(X, y, num_train_rows=99, num_columns=1)

def test_train_test_split_by_rows_and_cols_err_too_many_cols():
    """Check output shapes"""
    X = pd.DataFrame({'a': [1, 5, 3, 4, 2, 6, 0, 9, 7, 8],
                      'b': [8, 7, 9, 0, 6, 3, 4, 3, 5, 1]})
    y = pd.DataFrame({'target': range(0, 10)})
    with pytest.raises(ValueError, match=r".*larger sample.*"):
        _ = sample_eval.train_test_split_by_rows_and_cols(X, y, num_train_rows=6, num_columns=99, replace=False)


class EstimatorForTesting():

    def __init__(self, value_to_return):
        self.value_to_return = value_to_return
        self.fitted = False

    def fit(self, X_train, y_train, replace=True, verbose=False):
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError('You must fit before predict')
        return np.full((X.shape[0], 1), self.value_to_return)

def test_estimator_for_testing():
    estimator = EstimatorForTesting(value_to_return=1.1)
    estimator.fit(pd.DataFrame(),pd.DataFrame())
    df = estimator.predict(pd.DataFrame([[1], [2], [3]]))
    assert df.shape == (3, 1)
    for row in df:
        assert row[0] == 1.1

def test_eval_metric_on_chosen_datatype():
    X_train = pd.DataFrame(
        {'a': [1, 5, 3, 4, 2, 6, 0],
         'b': [8, 7, 9, 0, 6, 3, 4]})
    X_test = pd.DataFrame(
        {'a': [9, 7, 8],
         'b': [3, 5, 1]})
    y_train = pd.DataFrame({'target': range(0, 7)})
    y_test = pd.DataFrame({'target': range(100, 103)})
    # The predicted value will always be 123.
    predicted_values= 123
    estimator = EstimatorForTesting(value_to_return=predicted_values)
    estimator.fit(X_train, y_train)

    # y_train has low actual values.
    out = sample_eval._eval_metric_on_chosen_datatype(
        estimator,
        sample_eval.MetricTuple('train_y_true', mean_absolute_error, sample_eval.DatasetType.train),
        X_train, X_test, y_train, y_test)
    assert out == (123-3), 'training set used'

    # y_test has high actual values.
    out = sample_eval._eval_metric_on_chosen_datatype(
        estimator,
        sample_eval.MetricTuple('test_y_true', mean_absolute_error, sample_eval.DatasetType.test),
        X_train, X_test, y_train, y_test)
    assert out == (123-101), 'test set used'

def test_sample_and_calc_metric_by_rows_and_cols():
    X = pd.DataFrame({'a': [1, 5, 3, 4, 2, 6, 0, 9, 7, 8],
                      'b': [8, 7, 9, 0, 6, 3, 4, 3, 5, 1]})
    y = pd.DataFrame({'target': range(0, 10)})
    model = EstimatorForTesting(0.0)
    results_dict = sample_eval.sample_and_calc_metric_by_rows_and_cols(
        X, y,
        num_train_rows=6, num_columns=1,
        model=model,
        metric_func=mean_absolute_error)
    assert 'train' in results_dict
    assert 'test' in results_dict

def test_sample_and_calc_metrics_by_rows_and_cols_one_metric_two_cols():
    X = pd.DataFrame({'a': [1, 5, 3, 4, 2, 6, 0, 9, 7, 8],
                      'b': [8, 7, 9, 0, 6, 3, 4, 3, 5, 1]})
    y = pd.DataFrame({'target': range(0, 10)})
    model = EstimatorForTesting(0.0)
    metric_tuple = sample_eval.MetricTuple('a', mean_absolute_error, sample_eval.DatasetType.train)
    results_dict = sample_eval.sample_and_calc_metrics_by_rows_and_cols(
        X, y,
        num_train_rows=6, num_columns_list=[1,2],
        model=model,
        metric_tuples=[metric_tuple],
        num_samples=3)
    assert 'a' in results_dict
    # Shape should have num_samples rows, len(num_columns_list) columns.
    assert results_dict['a'].shape == (3, 2), "metric 'a' shape is not correct"
