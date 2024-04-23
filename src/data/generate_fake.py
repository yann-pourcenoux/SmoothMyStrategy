"""Standalone script to generate fake data."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.constants import DATASET_PATH


def main(period: float = 10) -> None:
    """Main function of the script."""

    # Load Apple stock
    df = pd.read_csv(os.path.join(DATASET_PATH, "AAPL.csv"))

    index = np.arange(len(df))
    price = 1 + 0.1 * np.sin(2 * np.pi * index / period)

    # Replace all prices with the generated one
    df["Open"] = price
    df["High"] = price
    df["Low"] = price
    df["Close"] = price
    df.to_csv(os.path.join(DATASET_PATH, f"COS_{period}.csv"), index=False)

    df["Close"][:100].plot()
    plt.show()


if __name__ == "__main__":
    main()
