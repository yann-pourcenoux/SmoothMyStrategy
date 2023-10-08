"""Module to work with correlations."""

import pandas as pd


def correlation_matrix(
    dataframe: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Calculate correlation matrix.

    Args:
        dataframe (pd.DataFrame): Dataframe with data.
        method (str, optional): Method to calculate correlation. Defaults to "pearson".

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    scaled = dataframe / dataframe.iloc[0]
    return scaled.corr(method=method)
