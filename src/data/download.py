import argparse
import json
import logging
import os

import yfinance as yf
from tqdm import tqdm

from data.constants import CONFIG_PATH, FINANCE_DATA_PATH

LOGGER = logging.getLogger(__file__)


def download_tickers(tickers: list[str], data_path: str) -> None:
    """Download the data from Yahoo Finance and save it to the data_path.

    Args:
        tickers (list[str]): List of tickers to donwload.
        data_path (str): Path to save the data to.
    """
    progress_bar = tqdm(tickers, leave=False)
    for ticker in progress_bar:
        progress_bar.set_description(f"Downloading {ticker}")
        ticker_data = yf.Ticker(ticker)
        ticker_data.history(period="max").to_csv(f"{data_path}/{ticker}.csv")


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

    with open(path, "r") as file:
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
        default=FINANCE_DATA_PATH,
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="files",
        type=str,
        help="List of json files containing tickers. "
        + "The files are separated by spaces.",
        default=[os.path.join(CONFIG_PATH, file) for file in os.listdir(CONFIG_PATH)],
    )

    return parser.parse_args()


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    LOGGER.info(f"Downloading data to {args.output_path} ...")
    progress_bar = tqdm(args.files, leave=False)
    for file in progress_bar:
        progress_bar.set_description(f"Downloading tickers from file {file}:")
        download_tickers(load_ticker_list(file), args.output_path)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
