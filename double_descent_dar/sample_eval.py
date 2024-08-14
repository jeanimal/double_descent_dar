import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Callable, Dict, Optional


def sample_rows_and_cols(X: pd.DataFrame, y: pd.DataFrame, num_sampled_rows: int, num_sampled_columns: int,
                         replace: bool, random_state: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples the same num_sampled_rows from X and y and sample num_sampled_columns from X.

    Returns a tuple of the sampled X and y with matched rows, so they must have the same number of rows.

    Sampling is can be with or without replacement based on the boolean value of replace.

    Parameters
    ----------
    X
        DataFrame to sample. If replace=False, must have shape >= (num_sampled_rows, num_sampled_columns).
    y
        Single-column dataFrame to sample. If replace=False, must have shape > (num_sampled_rows, 1).
    num_sampled_rows
        Integer number of rows to sample.
    num_sampled_columns
        Integer number of columns to sample.
    replace
        Whether to sample with replacement
    random_state
        Optional random state for the random sample, settable for reproducible testing.  Leave None in production.


    Returns
    -------
    X, y
        Sampled version of the input dataframes where X.shape == (num_sampled_rows, num_sampled_columns) and y.shape == (num_sampled_rows, 1)
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X had and y must have the same number of rows but had {X.shape[0]} and {y.shape[0]}.")
    # TODO: Change function since np.random uses np.random.seed(42) rather than the passed-in state.
    # New code should use the `~numpy. random. Generator. choice` method of a `~numpy. random. Generator` instance
    indices = np.random.choice(X.index, num_sampled_rows, replace=replace)
    X_subset = X.iloc[indices]
    y_subset = y.iloc[indices]
    X_subset = X_subset.sample(n=num_sampled_columns, random_state=random_state, replace=replace, axis=1)
    return X_subset, y_subset


def split_and_calc_metric(X, y, test_size, model, metric_func, random_state: Optional[int] = None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    metric_train = metric_func(y_train, model.predict(X_train))
    metric_test = metric_func(y_test, model.predict(X_test))
    return {'train': metric_train, 'test': metric_test}


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
    train_size = num_train_rows / X.shape[0]
    if verbose:
        # print(f"num_train_rows: {num_train_rows}")
        # print(f"X.shape[0]: {X.shape[0]}")
        print(f'using train_size {train_size}')
    X_subset = X.sample(n=num_columns, random_state=random_state, replace=replace, axis=1)
    return train_test_split(
        X_subset, y, train_size=train_size, random_state=random_state)


# Combine the above two functions.
def sample_and_calc_metric_by_rows_and_cols(X: pd.DataFrame, y: pd.DataFrame, num_train_rows: int, num_columns: int,
                                            model, metric_func, replace: bool = True, random_state: Optional[np.random.RandomState] = None,
                                            verbose: bool=False) -> \
Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split_by_rows_and_cols(X, y, num_train_rows, num_columns, replace, random_state, verbose)
    model.fit(X_train, y_train)
    metric_train = metric_func(y_train, model.predict(X_train))
    metric_test = metric_func(y_test, model.predict(X_test))
    return {'train': metric_train, 'test': metric_test}


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


def sample_and_calc_metric(X, y, num_sampled_rows, num_sampled_columns, test_size, model, metric_func, replace,
                           random_state: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_sub, y_sub = sample_rows_and_cols(X, y, num_sampled_rows, num_sampled_columns, replace=replace,
                                        random_state=random_state)
    metric_tuple = split_and_calc_metric(X_sub, y_sub, test_size, model, metric_func, random_state=random_state)
    return metric_tuple


def sample_dataframes(X, y, rows_per_sample, cols_per_sample, num_samples, rng: Optional[np.random.RandomState] = None):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X had and y must have the same number of rows but had {X.shape[0]} and {y.shape[0]}.")

    if rng is None:
        rng = np.random.default_rng()
    total_rows, total_cols = X.shape

    row_indices = rng.choice(total_rows, size=(num_samples, rows_per_sample), replace=True)
    col_indices = rng.choice(total_cols, size=(num_samples, cols_per_sample), replace=True)

    sampled_X = [X.iloc[row_idx, col_idx].values for row_idx, col_idx in zip(row_indices, col_indices)]
    sampled_y = [y.iloc[row_idx].values for row_idx in row_indices]
    return zip(sampled_X, sampled_y)
