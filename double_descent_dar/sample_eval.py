import numpy as np
import pandas as pd


def sample_rows_and_cols(X: pd.DataFrame, y: pd.DataFrame, num_sampled_rows: int, num_sampled_columns: int,
                         random_state: int, replace: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    random_state
        Integer random state for the random sample, useful for reproducible testing.  Do not set in production.
    replace
        Whether to sample with replacement

    Returns
    -------
    X, y
        Sampled version of the input dataframes where X.shape == (num_sampled_rows, num_sampled_columns) and y.shape == (num_sampled_rows, 1)
    """
    # TODO: Change function since np.random uses np.random.seed(42) rather than the passed-in state.
    indices = np.random.choice(X.index, num_sampled_rows, replace=replace)
    X_subset = X.iloc[indices]
    y_subset = y.iloc[indices]
    X_subset = X_subset.sample(n=num_sampled_columns, random_state=random_state, replace=replace, axis=1)
    return X_subset, y_subset