"""Module to test the functions used in data loading."""

import unittest

import numpy as np
import pandas as pd

import data.loader


class TestDataLoading(unittest.TestCase):
    """Class to test the functions used in data loading."""

    def test_select_tickers_column(self):
        """Test the select_tickers_column function."""
        dataframes = {
            "AAPL": pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=["Open", "Close"]),
            "MSFT": pd.DataFrame([14, 15, 16]),
        }
        dataframe = data.loader.select_tickers_columns(dataframes, ["AAPL"], ["Open"])

        self.assertEqual(len(dataframe.columns), 1)
        self.assertEqual(dataframe.columns[0], "AAPL_Open")
        self.assertEqual(dataframe["AAPL_Open"].values.tolist(), [1, 2, 3])

    def test_select_time_range(self):
        """Test the select_time_range function."""
        # Three days list from today
        today = pd.Timestamp.today()
        days = [today - pd.Timedelta(days=i) for i in range(5)][::-1]
        assert days[-1] == today

        start = today - pd.Timedelta(days=3)
        end = today - pd.Timedelta(days=1)
        dataframe = pd.DataFrame([k for k in range(5)], index=days)
        dataframe = data.loader.select_time_range(dataframe, start, end)

        self.assertTrue(dataframe.index[0] == start)
        self.assertTrue(dataframe.index[-1] == end)
        self.assertEqual(len(dataframe), 3)

    def test_fill_nan(self):
        """Test the fill_nan function."""
        dataframe = pd.DataFrame([np.NaN, 1, np.NaN, 2], columns=["value"])
        dataframe = data.loader.fill_nan(dataframe)
        self.assertEqual(dataframe.value.values.tolist(), [1, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
