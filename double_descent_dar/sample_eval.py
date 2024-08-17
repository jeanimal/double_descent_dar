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
    name: str
    metric_func: Callable[[pd.DataFrame, np.ndarray], float]
    dataset_type: DatasetType

def sample_and_calc_metrics_by_rows_and_cols(
        X: pd.DataFrame,
        y: pd.DataFrame,
        num_train_rows: int,
        num_columns: int,
        model,
        metric_tuples: list[MetricTuple],
        replace: bool = True,
        random_state: Optional[np.random.RandomState] = None,
        verbose: bool=False
) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split_by_rows_and_cols(X, y, num_train_rows, num_columns, replace, random_state, verbose)
    model.fit(X_train, y_train)
    out_dict = {}
    for metric_tuple in metric_tuples:
        if metric_tuple.dataset_type == DatasetType.train:
            df_to_predict = X_train
            df_actuals = y_train
        elif metric_tuple.dataset_type == DatasetType.test:
            df_to_predict = X_test
            df_actuals = y_test
        else:
            raise ValueError(f"Unrecognized dataset type {metric_tuple.dataset_type}")
        prediction = model.predict(df_to_predict)
        out_dict[metric_tuple.name] = metric_tuple.metric_func(df_actuals, prediction)
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
    metrics = [
        MetricTuple('train', metric_func, DatasetType.train),
        MetricTuple('test', metric_func, DatasetType.test)
    ]
    return sample_and_calc_metrics_by_rows_and_cols(X, y, num_train_rows, num_columns, model, metrics, replace, random_state, verbose)
    # return {'train': metric_train, 'test': metric_test}


def run_multiple_samples(
        X: pd.DataFrame,
        y: pd.DataFrame,
        num_train_rows: int,
        num_columns: int,
        model,
        metric_func: Callable[[pd.DataFrame, np.ndarray], float],
        N: int,
        replace: bool = True,
        random_state: Optional[np.random.RandomState] = None
) -> pd.DataFrame:
    results = [
        sample_and_calc_metric_by_rows_and_cols(
            X, y, num_train_rows, num_columns, model, metric_func, replace, random_state, verbose=False
        )
        for _ in range(N)
    ]
    return pd.DataFrame(results)
