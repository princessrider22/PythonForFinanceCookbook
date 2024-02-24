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

# ----------------------------------------------------2.1 Converting prices to returns-----------------------------
# Import the libraries:
import pandas as pd
import numpy as np
import yfinance as yf

# Download the data and keep the adjusted close prices only:
df = yf.download("AAPL",
                 start="2010-01-01",
                 end="2020-12-31",
                 progress=False)

df = df.loc[:, ["Adj Close"]]

# Convert adjusted close prices to simple and log returns:
df["simple_rtn"] = df["Adj Close"].pct_change()
df["log_rtn"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

# Inspect the output:
df.head()

# --------------------------------------------------------2.2 Adjusting the returns for inflation----------------------------
# creating the steps from the previous recipe:
import yfinance as yf

df = yf.download("AAPL",
                 start="2010-01-01",
                 end="2020-12-31",
                 progress=False)

df = df.loc[:, ["Adj Close"]]
df.index.name = "date"
df.head()

# Resample daily prices to monthly:
df_resampled = df.resample('BMS').first()

# Import the library:
import cpi

# in the case of seeing the `StaleDataWarning: CPI data is out of date`
cpi.update()

# Obtain the default CPI series:
cpi_series = cpi.series.get()
cpi_series

print(cpi_series)

# Convert the object into a pandas DataFrame:
df_cpi_2 = cpi_series.to_dataframe()

# Filter the DataFrame and view the top 12 observations:
df_cpi_2_filtered = df_cpi_2.query("period_type == 'monthly' and year >= 2010").loc[:, ["date", "value"]].set_index("date") \
    .rename(columns={"value": "cpi"})

df_cpi_2_filtered.index = pd.to_datetime(df_cpi_2_filtered.index)

df_cpi_2_filtered.head(12)

# Join inflation data to prices:
# Calculate simple returns and inflation rate:
# Adjust the returns for inflation:
df_j = df_resampled.join(df_cpi_2_filtered, how="left")
df_j["simple_rtn"] = df_j["Adj Close"].pct_change()
df_j["inflation_rate"] = df_j["cpi"].pct_change()
df_j["real_rtn"] = (
        (df_j["simple_rtn"] + 1) / (df_j["inflation_rate"] + 1) - 1
)
df_j.head()

# ------------------------------------------------------2.3 Changing the frequency of time series data----------------------------------
# Obtain the log returns in case of starting in this recipe:
import pandas as pd
import yfinance as yf
import numpy as np

# download data
df = yf.download("AAPL",
                 start="2000-01-01",
                 end="2010-12-31",
                 auto_adjust=False,
                 progress=False)

# keep only the adjusted close price
df = df.loc[:, ["Adj Close"]].rename(columns={"Adj Close": "adj_close"})

# calculate simple returns
df["log_rtn"] = np.log(df["adj_close"] / df["adj_close"].shift(1))

# remove redundant data
df = df.drop("adj_close", axis=1).dropna(axis=0)

df.head()

# Import the libraries:
import pandas as pd
import numpy as np

# Define the function for calculating the realized volatility:
def realized_volatility(x):
        return np.sqrt(np.sum(x ** 2))

# Calculate monthly realized volatility:
df_rv = (
    df.groupby(pd.Grouper(freq="M"))
    .apply(realized_volatility)
    .rename(columns={"log_rtn": "rv"})
)

# Annualize the values:
df_rv.rv = df_rv["rv"] * np.sqrt(12)

# Plot the results:
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df)
ax[0].set_title("Apple's log returns (2000-2012)")
ax[1].plot(df_rv)
ax[1].set_title("Annualized realized volatility")

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_2_5', dpi=200)

# ------------------------------------------------------------2.4 Different ways of imputing missing data------------------------
# Import the libraries:
import pandas as pd
import numpy as np
import cpi

# Download the inflation data:
cpi.update()
cpi_series = cpi.series.get()
df = cpi_series.to_dataframe().query("period_type == 'monthly' and year >= 2015") \
         .loc[:, ["date", "value"]] \
    .set_index("date") \
    .rename(columns={"value": "cpi"})
df.index = pd.to_datetime(df.index)

# Introduce 5 missing values at random:
np.random.seed(42)
rand_indices = np.random.choice(df.index, 5, replace=False)

df["cpi_missing"] = df.loc[:, "cpi"]
df.loc[rand_indices, "cpi_missing"] = np.nan
df.head()

# Fill the missing values using different methods:
for method in ["bfill", "ffill"]:
    df[f"method_{method}"] = (
        df[["cpi_missing"]].fillna(method=method)
    )

# Inspect the results by displaying the rows in which we created the missing values:
df.loc[rand_indices].sort_index()

# Plot the results for years 2015 - 2016:
df.loc[:"2017-01-01"] \
        .drop(columns=["cpi_missing"]) \
        .plot(title="Different ways of filling missing values");

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_2_8', dpi=200)

# Use linear interpolation to fill the missing values:
df["method_interpolate"] = df[["cpi_missing"]].interpolate()

# Inspect the results:
df.loc[rand_indices].sort_index()

# Plot the results:
df.loc[:"2017-01-01"] \
    .drop(columns=["cpi_missing"]) \
    .plot(title="Different ways of filling missing values");

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_2_10', dpi=200)
#
# # ----------------------------------2.6 Different ways of aggregating trade data--------------------------------------
# # Import the libraries:
# from binance.spot import Spot as Client
# import pandas as pd
# import numpy as np
#
# # Instantiate the Binance client and download the last 500 BTCEUR trades:
# spot_client = Client(base_url="https://api3.binance.com")
# r = spot_client.trades("BTCEUR")
#
# # Process the downloaded trades into a pandas DataFrame:
# df = (
#     pd.DataFrame(r)
#     .drop(columns=["isBuyerMaker", "isBestMatch"])
# )
# df["time"] = pd.to_datetime(df["time"], unit="ms")
#
# for column in ["price", "qty", "quoteQty"]:
#     df[column] = pd.to_numeric(df[column])
# print(df)
#
#
# # Define a function aggregating the raw trades information:
# def get_bars(df, add_time=False):
#     """[summary]
#
#     Args:
#         df ([type]): [description]
#
#     Returns:
#         [type]: [description]
#     """
#     ohlc = df["price"].ohlc()
#     vwap = (
#         df.apply(lambda x: np.average(x["price"], weights=x["qty"]))
#         .to_frame("vwap")
#     )
#     vol = df["qty"].sum().to_frame("vol")
#     cnt = df["qty"].size().to_frame("cnt")
#
#     if add_time:
#         time = df["time"].last().to_frame("time")
#         res = pd.concat([time, ohlc, vwap, vol, cnt], axis=1)
#     else:
#         res = pd.concat([ohlc, vwap, vol, cnt], axis=1)
#     return res
#
# # Get time bars:
# df_grouped_time = df.groupby(pd.Grouper(key="time", freq="1Min"))
# time_bars = get_bars(df_grouped_time)
# time_bars
#
# # Get tick bars:
# bar_size = 50
# df["tick_group"] = (
#     pd.Series(list(range(len(df))))
#     .div(bar_size)
#     .apply(np.floor)
#     .astype(int)
#     .values
# )
# df_grouped_ticks = df.groupby("tick_group")
# tick_bars = get_bars(df_grouped_ticks, add_time=True)
# tick_bars
#
# # Get volume bars:
# bar_size = 1
# df["cum_qty"] = df["qty"].cumsum()
# df["vol_group"] = (
#     df["cum_qty"]
#     .div(bar_size)
#     .apply(np.floor)
#     .astype(int)
#     .values
# )
# df_grouped_ticks = df.groupby("vol_group")
# volume_bars = get_bars(df_grouped_ticks, add_time=True)
# volume_bars
#
# # Get dollar bars:
# bar_size = 50000
# df["cum_value"] = df["quoteQty"].cumsum()
# df["value_group"] = (
#     df["cum_value"]
#     .div(bar_size)
#     .apply(np.floor)
#     .astype(int)
#     .values
# )
# df_grouped_ticks = df.groupby("value_group")
# dollar_bars = get_bars(df_grouped_ticks, add_time=True)
# dollar_bars
