"""Utility functions to download data from Yahoo Finance."""

import argparse
import json
import os
from multiprocessing import Pool

import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from data.constants import DATA_CONFIG_PATH, DATASET_PATH


def save_ticker_data(args: tuple[str, pd.DataFrame, str]) -> None:
    """Save individual ticker data to a CSV file.

    Args:
        args: Tuple containing:
            - ticker (str): The ticker symbol
            - tickers_data (pd.DataFrame): Complete DataFrame containing all ticker data
            - data_path (Path): Path to save the CSV files
    """
    ticker, tickers_data, data_path = args
    ticker_data = tickers_data[tickers_data["Ticker"] == ticker].copy()
    ticker_data.drop(columns=["Ticker"], inplace=True)
    ticker_data.dropna(inplace=True)
    ticker_data.to_csv(f"{data_path}/{ticker}.csv", index=False)


def download_tickers(tickers: list[str], data_path: str) -> None:
    """Download the data from Yahoo Finance and save it to the data_path.

    Args:
        tickers (list[str]): List of tickers to donwload.
        data_path (str): Path to save the data to.
    """
    tickers_data = yf.download(tickers, period="max", auto_adjust=False)
    tickers_data = tickers_data.stack(level=1).reset_index()

    # Some ticker donwloading may fail for various reasons
    tickers = tickers_data["Ticker"].unique()

    with Pool() as pool:
        args = [
            (ticker, tickers_data, data_path)
            for ticker in tickers_data["Ticker"].unique()
        ]
        list(
            tqdm(
                pool.imap(save_ticker_data, args),
                total=len(args),
                desc="Saving files",
            )
        )


def load_ticker_list(path: str) -> list[str]:
    """Load the list of tickers from a json file.

    Args:
        path (str): Path to the file with the tickers.

    Raises:
        FileNotFoundError: If the file does not exist.

    Returns:
        list[str]: List of tickers.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _parse_args():
    """Parse the command line arguments.

    Returns:
        argparse.Namespace: Namespace with the arguments.
    """
    parser = argparse.ArgumentParser(description="Download data from Yahoo Finance.")

    parser.add_argument(
        "--output_path",
        metavar="output_path",
        type=str,
        help="Path to save the data to.",
        default=DATASET_PATH,
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="files",
        type=str,
        help="List of json files containing tickers. "
        + "The files are separated by spaces.",
        default=[
            os.path.join(DATA_CONFIG_PATH, file)
            for file in os.listdir(DATA_CONFIG_PATH)
        ],
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = _parse_args()

    logger.info(f"Downloading data to {args.output_path} ...")

    tickers = []
    for file in args.files:
        tickers.extend(load_ticker_list(file))

    download_tickers(tickers, args.output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
