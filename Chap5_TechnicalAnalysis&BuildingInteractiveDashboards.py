# ---------------------------------Please run these for all chapters, as those plotting settings are standard throughout the book.------------------------
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# feel free to modify, for example, change the context to "notebook"
sns.set_theme(context="talk", style="whitegrid",
              palette="colorblind", color_codes=True,
              rc={"figure.figsize": [12, 8]})

# ----------------------------------------------------5.1 Calculating the most popular technical indicators-----------------------------
# Import the libraries:
import pandas as pd
import yfinance as yf
import talib

# Download IBM's stock prices from 2020:
df = yf.download("IBM",
                 start="2020-01-01",
                 end="2020-12-31",
                 progress=False,
                 auto_adjust=True)
df

# Calculate and plot the Simple Moving Average:
df["sma_20"] = talib.SMA(df["Close"], timeperiod=20)
(
    df[["Close", "sma_20"]]
    .plot(title="20-day Simple Moving Average (SMA)")
)

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_5_1", dpi=200)

# Calculate and plot the Bollinger bands:
df["bb_up"], df["bb_mid"], df["bb_low"] = talib.BBANDS(df["Close"])

fig, ax = plt.subplots()

(
    df.loc[:, ["Close", "bb_up", "bb_mid", "bb_low"]]
    .plot(ax=ax, title="Bollinger Bands")
)

ax.fill_between(df.index, df["bb_low"], df["bb_up"],
                color="gray",
                alpha=.4)

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_5_2", dpi=200)

# Calculate and plot the RSI:
df["rsi"] = talib.RSI(df["Close"])

fig, ax = plt.subplots()
df["rsi"].plot(ax=ax,
               title="Relative Strength Index (RSI)")
ax.hlines(y=30,
          xmin=df.index.min(),
          xmax=df.index.max(),
          color="red")
ax.hlines(y=70,
          xmin=df.index.min(),
          xmax=df.index.max(),
          color="red")

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_5_3", dpi=200)

# Calculate and plot the MACD:
df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(
    df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
)

with sns.plotting_context("notebook"):
    fig, ax = plt.subplots(2, 1, sharex=True)

    (
        df[["macd", "macdsignal"]].
        plot(ax=ax[0],
             title="Moving Average Convergence Divergence (MACD)")
    )
    ax[1].bar(df.index, df["macdhist"].values, label="macd_hist")
    ax[1].legend()

    sns.despine()
    plt.tight_layout()
    # plt.savefig("images/figure_5_4", dpi=200)

# There's more
# Import the libraries:
from ta import add_all_ta_features

# Discard the previously calculated indicators and keep only the required columns:
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

# Calculate all the technical indicators available in the ta library:
df = add_all_ta_features(df, open="Open", high="High",
                         low="Low", close="Close",
                         volume="Volume")
df.shape

df.columns

# -----------------------------------------------------------5.3 Recognizing candlestick patterns---------------------------------------
# Import the libraries:
import pandas as pd
import yfinance as yf
import talib
import mplfinance as mpf
from IPython.display import display

# DownloadBitcoin's hourly prices from the last 3 months:
df = yf.download("BTC-USD",
                 period="9mo",
                 interval="1h",
                 progress=False)
df

# Identify the "Three Line Strike" pattern:
df["3_line_strike"] = talib.CDL3LINESTRIKE(
    df["Open"], df["High"], df["Low"], df["Close"]
)

# Locate and plot the bearish pattern:
df[df["3_line_strike"] == -100].head().round(2)

mpf.plot(df["2023-07-16 05:00:00":"2023-07-16 16:00:00"],
         type="candle")

# Locate and plot the bullish pattern:
df[df["3_line_strike"] == 100].head().round(2)

mpf.plot(df["2023-07-10 10:00:00":"2023-07-10 23:00:00"],
         type="candle")

# Get all available pattern names:
candle_names = talib.get_function_groups()["Pattern Recognition"]

# Iterate over the list of patterns and try identifying them all:
for candle in candle_names:
    df[candle] = getattr(talib, candle)(df["Open"], df["High"],
                                            df["Low"], df["Close"])

# Inspect the summary statistics of the patterns:
with pd.option_context("display.max_rows", len(candle_names)):
    display(df[candle_names].describe().transpose().round(2))

# Locate and plot the "Evening Star" pattern:
df[df["CDLEVENINGSTAR"] == -100].head()

mpf.plot(df["2023-09-21 12:00:00":"2023-09-22 03:00:00"], type="candle")

