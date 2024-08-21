import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from sklearn.model_selection import train_test_split
from typing import Callable, Dict, Optional

def train_test_split_by_rows_and_cols(X: pd.DataFrame, y: pd.DataFrame, num_train_rows: int, num_columns: int,
                                      replace: bool = True, random_state: Optional[np.random.RandomState] = None,
                                      verbose: bool=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Use sklearn train_test_split with a fixed number of rows plus column sampling.

    This function has the same output as sklearn train_test_split: a tuple of X_train, X_test,
    y_train, y_test, where X_train has shape (num_train_rows, num_columns) and y_train has shape
    (num_train_rows, 1).

    Sampling of columns can be with or without replacement based on the boolean value of replace.

    Why?

    sklearn's train_test_split takes a proportion of taining rows as an input and always
    uses all columns.  But controlling the training rows and columns is helpful for controlling the
    overparametrization ratio, defined as:

      overparametrization ratio = num_parameters / num_rows

    In many estimation models, the number of parameters is closely related to the number of columns.
    For example, in ordinary least squares (OLS) regression without an intercept, the number of
    parameters equals the number of columns, and when there is an intercept, the number of
    parameters is number of columns + 1.

    The overparameterization ratio is described in this paper:
    Hastie, T., Montanari, A., Rosset, S., & Tibshirani, R. J. (2020). Surprises in High-Dimensional
    Ridgeless Least Squares Interpolation. http://arxiv.org/abs/1903.08560

    The overparametrization ratio distinguish these classes of behavior:
    * less than 1: uncerparameterized
    * equal to 1: interpolating
    * greater than 1: overparameterized

    Parameters
    ----------
    X
        DataFrame to sample. If replace=False, must have shape >= (num_sampled_rows, num_sampled_columns).
    y
        Single-column dataFrame to sample. If replace=False, must have shape > (num_sampled_rows, 1).
    num_train_rows
        Integer number of rows to sample for X_train.
    num_columns
        Integer number of columns to sample.
    replace
        Whether to sample with replacement
    random_state
        Optional random state for the random sample, settable for reproducibility.


    Returns
    -------
    X_train, X_test, y_train, y_test
        Sampled versions of the input dataframes.
    """
    if num_train_rows > X.shape[0] or num_train_rows <= 0:
        raise ValueError("num_train_rows must be in the range (0, num_rows_in_X) which"
                         f"for this input is (0, {X.shape[0]}] but num_train_rows was {num_train_rows}.")
    train_size = num_train_rows / X.shape[0]
    if verbose:
        # print(f"num_train_rows: {num_train_rows}")
        # print(f"X.shape[0]: {X.shape[0]}")
        print(f'using train_size {train_size}')
    X_subset = X.sample(n=num_columns, random_state=random_state, replace=replace, axis=1)
    return train_test_split(
        X_subset, y, train_size=train_size, random_state=random_state)


DatasetType = Enum('DatasetType', ['train', 'test'])

@dataclass
class MetricTuple:
    """Metric name, function to apply, data type to apply it to."""
    name: str
    metric_func: Callable[[pd.DataFrame, np.ndarray], float]
    """The metric func implements the sklearn.metric function, e.g. mae, rmse, etc."""
    dataset_type: DatasetType

def _eval_metric_on_chosen_datatype(
        model: Callable[[pd.DataFrame], np.ndarray],
        metric_tuple: MetricTuple,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame):
    if metric_tuple.dataset_type == DatasetType.train:
        df_to_predict = X_train
        df_actuals = y_train
    elif metric_tuple.dataset_type == DatasetType.test:
        df_to_predict = X_test
        df_actuals = y_test
    else:
        raise ValueError(f"Unrecognized dataset type {metric_tuple.dataset_type}")
    prediction = model.predict(df_to_predict)
    return metric_tuple.metric_func(df_actuals, prediction)

def sample_and_calc_metrics_by_rows_and_cols(
        X: pd.DataFrame,
        y: pd.DataFrame,
        num_train_rows: int,
        num_columns_list: list[int],
        model,
        metric_tuples: list[MetricTuple],
        num_samples: int,
        replace: bool = True,
        random_state: Optional[np.random.RandomState] = None,
        verbose: bool=False
) -> Dict[str, np.ndarray]:
    """Repeatedly sample rows and columns to train a model and collect metrics.

    Parameters
    ----------
    X
        DataFrame to sample. If replace=False, must have shape >= (num_sampled_rows, num_sampled_columns).
    y
        Single-column dataFrame to sample. If replace=False, must have shape > (num_sampled_rows, 1).
    num_train_rows
        Integer number of rows to sample for X_train.
    num_columns_list
        List of integer number of columns to sample.  Each number must range from 1 to the
        total number of columns in X. The output arrays have `num_samples` rows and
        `len(num_columns_list)` columns.
    model
        Estimator model to use.  Must have `fit` and `predict` methods implemented (as in sklearn).
    metric_tuples
        List of evaluation metrics to use and how.  The output dictionary will have an entry for each
        tuple, using the metric name as the key.
    num_samples
        Integer number of times to sample rows and columns, fit the model, and collect metrics.
        The output arrays have `num_samples` rows and `len(num_columns_list)` columns.
    replace
        Whether to sample with replacement
    random_state
        Optional random state for the random sample, settable for reproducibility.
    verbose
        Boolean to request some debug print statements.


    Returns
    -------
    dict of metric name to a numpy array of metric values
        The dict has as many keys as metric_tuples, with the metric names as the keys.
        The associates arrays  have `num_samples` rows and `len(num_columns_list)` columns.
    """
    out_dict = {}
    # Initialize.
    for metric_tuple in metric_tuples:
        out_dict[metric_tuple.name] = np.zeros((num_samples, len(num_columns_list)))
    def sample_and_eval(metric_dict, sample_index: int, num_columns_index: int, num_columns: int):
        X_train, X_test, y_train, y_test = train_test_split_by_rows_and_cols(
            X, y, num_train_rows, num_columns, replace, random_state, verbose)
        model.fit(X_train, y_train)
        for metric_tuple in metric_tuples:
            arr = metric_dict[metric_tuple.name]
            val = _eval_metric_on_chosen_datatype(
                model, metric_tuple, X_train, X_test, y_train, y_test)
            arr[sample_index, num_columns_index] = val
    for i, num_columns in enumerate(num_columns_list):
        [sample_and_eval(out_dict, sample_index, i, num_columns) for sample_index in range(num_samples)]
    return out_dict

def sample_and_calc_metric_by_rows_and_cols(
        X: pd.DataFrame,
        y: pd.DataFrame,
        num_train_rows: int,
        num_columns: int,
        model,
        metric_func: Callable[[pd.DataFrame, np.ndarray], float],
        replace: bool = True,
        random_state: Optional[np.random.RandomState] = None,
        verbose: bool=False
) -> Dict[str, float]:
    """Wrapper to return {'train': metric_train, 'test': metric_test}.
    Hard-coded to run just one sample with train and test data for the metric.
    """
    metrics = [
        MetricTuple('train', metric_func, DatasetType.train),
        MetricTuple('test', metric_func, DatasetType.test)
    ]
    out_dict = sample_and_calc_metrics_by_rows_and_cols(
        X, y, num_train_rows, [num_columns], model, metrics,
        num_samples=1, replace=replace, random_state=random_state, verbose=verbose)
    return dict((k, array[0, 0]) for k, array in out_dict.items())
