import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional

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
    return {'train':metric_train, 'test':metric_test}


# Wrap sklearn's train_test_split with train_rows and add column sampling.
# Returns X_train, X_test, y_train, y_test.
def train_test_split_by_rows_and_cols(X, y, num_train_rows, num_sampled_columns, replace: bool = True, random_state: Optional[np.random.RandomState] = None):
    train_size = num_train_rows / X.shape[0]
    print(f'using train_size {train_size}')
    X_subset = X.sample(n=num_sampled_columns, random_state=random_state, replace=replace, axis=1)
    return train_test_split(
        X_subset, y, train_size=train_size, random_state=random_state)

# Combine the above two functions.
def split_sample_and_calc_metric(X, y, train_rows, num_sampled_columns, model, metric_func, random_state: Optional[int] = None):
    train_size = train_rows / X.shape[0]
    print(f'using train_size {train_size}')
    replace = True
    X_subset = X.sample(n=num_sampled_columns, random_state=random_state, replace=replace, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, train_size=train_size, random_state=random_state)
    model.fit(X_train, y_train)
    metric_train = metric_func(y_train, model.predict(X_train))
    metric_test = metric_func(y_test, model.predict(X_test))
    return {'train':metric_train, 'test':metric_test}

def sample_and_calc_metric(X, y, num_sampled_rows, num_sampled_columns, test_size, model, metric_func, replace, random_state: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_sub, y_sub = sample_rows_and_cols(X, y, num_sampled_rows, num_sampled_columns, replace=replace, random_state=random_state)
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