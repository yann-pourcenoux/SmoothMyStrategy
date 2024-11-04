import matplotlib.pyplot as plt
import yfinance as yf


def download_data():
    data = yf.download("AAPL", period="max")

    # Keep onlyt the "Adj Close" column and rename it to "close"
    data = data[["Adj Close"]]
    data.rename(columns={"Adj Close": "close"}, inplace=True)

    # Keep only the last 1000 rows

    # Compute a EMA of 12 and 26 days
    data.loc[:, "ema_12"] = data["close"].ewm(span=12, adjust=True).mean()
    data.loc[:, "ema_26"] = data["close"].ewm(span=26, adjust=True).mean()
    data.loc[:, "macd"] = data["ema_12"] - data["ema_26"]

    data.loc[:, "ema_12_ret"] = data["ema_12"] / data["close"].shift(26)
    data.loc[:, "ema_26_ret"] = data["ema_26"] / data["close"].shift(26)
    data.loc[:, "close_ret"] = data["close"] / data["close"].shift(26)
    data.loc[:, "macd_ret"] = data["ema_12_ret"] - data["ema_26_ret"]

    data.loc[:, "ema_12_mean"] = data["ema_12"] / data["close"].rolling(26).mean()
    data.loc[:, "ema_26_mean"] = data["ema_26"] / data["close"].rolling(26).mean()
    data.loc[:, "close_mean"] = data["close"] / data["close"].rolling(26).mean()
    data.loc[:, "macd_mean"] = data["ema_12_mean"] - data["ema_26_mean"]

    data.loc[:, "ema_12_med"] = data["ema_12"] / data["close"].rolling(26).median()
    data.loc[:, "ema_26_med"] = data["ema_26"] / data["close"].rolling(26).median()
    data.loc[:, "close_med"] = data["close"] / data["close"].rolling(26).median()
    data.loc[:, "macd_med"] = data["ema_12_med"] - data["ema_26_med"]
    return data


if __name__ == "__main__":
    data = download_data()

    fig, ax = plt.subplots(4, 6, figsize=(10, 10))
    data["close_ret"].plot(ax=ax[0, 0], label="Close ret")
    data["ema_12_ret"].plot(ax=ax[0, 0], label="EMA 12 ret")
    data["ema_26_ret"].plot(ax=ax[0, 0], label="EMA 26 ret")
    ax[0, 0].legend()

    data["close_ret"].hist(ax=ax[0, 1], label="Close ret", bins=100, density=True)
    ax[0, 1].legend()
    data["ema_12_ret"].hist(ax=ax[0, 2], label="EMA 12 ret", bins=100, density=True)
    ax[0, 2].legend()
    data["ema_26_ret"].hist(ax=ax[0, 3], label="EMA 26 ret", bins=100, density=True)
    ax[0, 3].legend()

    data["close_ret"].plot(ax=ax[0, 4], label="Close ret")
    data["macd_ret"].plot(ax=ax[0, 4], label="MACD ret")
    ax[0, 4].legend()

    data["macd_ret"].hist(ax=ax[0, 5], label="MACD ret", bins=100, density=True)
    ax[0, 5].legend()

    data["close_mean"].plot(ax=ax[1, 0], label="Close mean")
    data["ema_12_mean"].plot(ax=ax[1, 0], label="EMA 12 mean")
    data["ema_26_mean"].plot(ax=ax[1, 0], label="EMA 26 mean")
    ax[1, 0].legend()

    data["close_mean"].hist(ax=ax[1, 1], label="Close mean", bins=100, density=True)
    ax[1, 1].legend()
    data["ema_12_mean"].hist(ax=ax[1, 2], label="EMA 12 mean", bins=100, density=True)
    ax[1, 2].legend()
    data["ema_26_mean"].hist(ax=ax[1, 3], label="EMA 26 mean", bins=100, density=True)
    ax[1, 3].legend()

    data["close_mean"].plot(ax=ax[1, 4], label="Close mean")
    data["macd_mean"].plot(ax=ax[1, 4], label="MACD mean")
    ax[1, 4].legend()

    data["macd_mean"].hist(ax=ax[1, 5], label="MACD mean", bins=100, density=True)
    ax[1, 5].legend()

    data["close_med"].plot(ax=ax[2, 0], label="Close med")
    data["ema_12_med"].plot(ax=ax[2, 0], label="EMA 12 med")
    data["ema_26_med"].plot(ax=ax[2, 0], label="EMA 26 med")
    ax[2, 0].legend()

    data["close_med"].hist(ax=ax[2, 1], label="Close med", bins=100, density=True)
    ax[2, 1].legend()
    data["ema_12_med"].hist(ax=ax[2, 2], label="EMA 12 med", bins=100, density=True)
    ax[2, 2].legend()
    data["ema_26_med"].hist(ax=ax[2, 3], label="EMA 26 med", bins=100, density=True)
    ax[2, 3].legend()

    data["close_med"].plot(ax=ax[2, 4], label="Close med")
    data["macd_med"].plot(ax=ax[2, 4], label="MACD med")
    ax[2, 4].legend()

    data["macd_med"].hist(ax=ax[2, 5], label="MACD med", bins=100, density=True)
    ax[2, 5].legend()

    data["close"].plot(ax=ax[3, 0], label="Close")
    data["ema_12"].plot(ax=ax[3, 0], label="EMA 12")
    data["ema_26"].plot(ax=ax[3, 0], label="EMA 26")
    ax[3, 0].legend()

    data["close"].hist(ax=ax[3, 1], label="Close", bins=100, density=True)
    ax[3, 1].legend()
    data["ema_12"].hist(ax=ax[3, 2], label="EMA 12", bins=100, density=True)
    ax[3, 2].legend()
    data["ema_26"].hist(ax=ax[3, 3], label="EMA 26", bins=100, density=True)
    ax[3, 3].legend()

    data["close"].plot(ax=ax[3, 4], label="Close")
    data["macd"].plot(ax=ax[3, 4], label="MACD")
    ax[3, 4].legend()

    data["macd"].hist(ax=ax[3, 5], label="MACD", bins=100, density=True)
    ax[3, 5].legend()

    plt.tight_layout()
    plt.show()
