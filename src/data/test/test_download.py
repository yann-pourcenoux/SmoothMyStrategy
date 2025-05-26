"""Test the download module."""

import json
import os
import tempfile
import unittest

import pytest

import data.download


class TestDownload(unittest.TestCase):
    """Test the download module."""

    directory: tempfile.TemporaryDirectory
    directory_name: str
    ticker: str

    def setUp(self) -> None:
        """Set ups a temporary directory for the test."""
        self.directory = tempfile.TemporaryDirectory()
        self.directory_name = self.directory.name
        self.ticker = "AAPL"

    @pytest.mark.unparallel
    def test_download_tickers(self):
        """Download one ticker and verify that the file exists."""
        data.download.download_tickers([self.ticker], self.directory_name)
        self.assertTrue(os.path.exists(f"{self.directory_name}/{self.ticker}.csv"))

    def test_load_ticker_list(self):
        """Test that the load_ticker_list function works."""
        filepath = os.path.join(self.directory_name, "test.json")

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump([self.ticker], file)

        tickers = data.download.load_ticker_list(filepath)
        self.assertEqual(tickers, [self.ticker])

    def tearDown(self) -> None:
        """Cleans the temporary directory."""
        self.directory.cleanup()
