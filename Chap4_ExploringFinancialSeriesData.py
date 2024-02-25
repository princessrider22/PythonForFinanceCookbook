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

# ----------------------------------------------------4.1 Outlier detection using rolling statistics-----------------------------
# Import the libraries:
import pandas as pd
import yfinance as yf

# Download Tesla's stock prices from 2019-2020 and calculate simple returns:
df = yf.download("TSLA",
                 start="2019-01-01",
                 end="2020-12-31",
                 progress=False)

df["rtn"] = df["Adj Close"].pct_change()
df = df[["rtn"]].copy()
df.head()

# Calculate the rolling mean and standard deviation:
df_rolling = df[["rtn"]].rolling(window=21).agg(["mean", "std"])
df_rolling.columns = df_rolling.columns.droplevel()
df_rolling

# Join the rolling data back to the initial DataFrame:
df = df.join(df_rolling)
df.tail()

# Calculate the upper and lower thresholds:
N_SIGMAS = 3
df["upper"] = df["mean"] + N_SIGMAS * df["std"]
df["lower"] = df["mean"] - N_SIGMAS * df["std"]

# Identify the outliers using the previously calculated thresholds:
df["outlier"] = (
        (df["rtn"] > df["upper"]) | (df["rtn"] < df["lower"])
)

# Plot the returns together with the thresholds and mark the outliers:
fig, ax = plt.subplots()

df[["rtn", "upper", "lower"]].plot(ax=ax)
ax.scatter(df.loc[df["outlier"]].index,
           df.loc[df["outlier"], "rtn"],
           color="black", label="outlier")
ax.set_title("Tesla's stock returns")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_1", dpi=200)

# Define a function for identify outliers using the steps described in the previous section.
def identify_outliers(df, column, window_size, n_sigmas):
    """Function for identifying outliers using rolling statistics"""

    df = df[[column]].copy()
    df_rolling = df.rolling(window=window_size) \
        .agg(["mean", "std"])
    df_rolling.columns = df_rolling.columns.droplevel()
    df = df.join(df_rolling)
    df["upper"] = df["mean"] + n_sigmas * df["std"]
    df["lower"] = df["mean"] - n_sigmas * df["std"]

    return ((df[column] > df["upper"]) | (df[column] < df["lower"]))


identify_outliers(df, "rtn", 21, 3)

# --------------------------------------------------------------4.2 Outlier detection with the Hampel filter------------------------------------------
# Import the libraries:
import yfinance as yf
from sktime.transformations.series.outlier_detection import HampelFilter

# Download Tesla's stock prices from 2019-2020 and calculate simple returns:
df = yf.download("TSLA",
                 start="2019-01-01",
                 end="2020-12-31",
                 progress=False)
df["rtn"] = df["Adj Close"].pct_change()

# Instantiate the HampelFilter class and use it for detecting the outliers:
hampel_detector = HampelFilter(window_length=10,
                                   return_bool=True)

df["outlier"] = hampel_detector.fit_transform(df["Adj Close"])
df.head()

# Plot Tesla's stock price and mark the outliers:
fig, ax = plt.subplots()

df[["Adj Close"]].plot(ax=ax)
ax.scatter(df.loc[df["outlier"]].index,
           df.loc[df["outlier"], "Adj Close"],
           color="black", label="outlier")
ax.set_title("Tesla's stock price")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_2", dpi=200)

# Identify the outliers among the stock returns:
df["outlier_rtn"] = hampel_detector.fit_transform(df["rtn"])
df.head()

# Plot Tesla's daily returns and mark the outliers:
fig, ax = plt.subplots()

df[["rtn"]].plot(ax=ax)
ax.scatter(df.loc[df["outlier_rtn"]].index,
           df.loc[df["outlier_rtn"], "rtn"],
           color="black", label="outlier")
ax.set_title("Tesla's stock returns")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_3", dpi=200)

# Investigate the overlap in outliers identified for the prices and returns:
df.query("outlier == True and outlier_rtn == True").round(2)

# ------------------------------------------------------------------4.5 Detecting patterns in a time series using the Hurst exponent---------------------------------------------------------
# Import the libraries:
import yfinance as yf
import numpy as np
import pandas as pd

# Download S & P 500's historical prices from the years 2000-2019:
df = yf.download("^GSPC",
                 start="2000-01-01",
                 end="2019-12-31",
                 progress=False)
df["Adj Close"].plot(title="S&P 500 (years 2000-2019)")

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_9", dpi=200);

# Define a function calculating the Hurst exponent:
def get_hurst_exponent(ts, max_lag=20):
    """Returns the Hurst Exponent of the time series"""

    lags = range(2, max_lag)

    # standard deviations of the lagged differences
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    hurst_exp = np.polyfit(np.log(lags), np.log(tau), 1)[0]

    return hurst_exp


# Calculate the values of the Hurst exponent using different values for the max_lag parameter:
for lag in [20, 100, 250, 500, 1000]:
        hurst_exp = get_hurst_exponent(df["Adj Close"].values, lag)
        print(f"Hurst exponent with {lag} lags: {hurst_exp:.4f}")

# Narrow down the data to the years 2005 - 2007 and calculate the exponents one more time:
df.loc["2005":"2007", "Adj Close"].plot(title="S&P 500 (years 2005-2007)")

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_10", dpi=200);

shorter_series = df.loc["2005":"2007", "Adj Close"].values
for lag in [20, 100, 250, 500]:
    hurst_exp = get_hurst_exponent(shorter_series, lag)
    print(f"Hurst exponent with {lag} lags: {hurst_exp:.4f}")

# -------------------------------------------------------------------4.6 Investigating stylized facts of asset returns-----------------------------------------
# Import the libraries:
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt

# Download the S & P 500 data and calculate the returns:
df = yf.download("^GSPC",
                 start="2000-01-01",
                 end="2020-12-31",
                 progress=False)

df = df[["Adj Close"]].rename(
    columns={"Adj Close": "adj_close"}
)
df["log_rtn"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
df = df[["adj_close", "log_rtn"]].dropna()
df


# -------------------Fact1 - Non - Gaussian distribution of returns
# Calculate the Normal PDF using the mean and standard deviation of the observed returns:
r_range = np.linspace(min(df["log_rtn"]),
                      max(df["log_rtn"]),
                      num=1000)
mu = df["log_rtn"].mean()
sigma = df["log_rtn"].std()
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

# Plot the histogram and the Q - Q Plot:
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# histogram
sns.histplot(df.log_rtn, kde=False, ax=ax[0])
ax[0].set_title("Distribution of S&P 500 returns",
                fontsize=16)
ax[0].plot(r_range, norm_pdf, "g", lw=2,
           label=f"N({mu:.2f}, {sigma ** 2:.4f})")
ax[0].legend(loc="upper left");

# Q-Q plot
qq = sm.qqplot(df.log_rtn.values, line="s", ax=ax[1])
ax[1].set_title("Q-Q plot", fontsize=16)

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_11", dpi=200);

# Print the summary statistics of the log returns:
jb_test = scs.jarque_bera(df["log_rtn"].values)

print("---------- Descriptive Statistics ----------")
print("Range of dates:", min(df.index.date), "-", max(df.index.date))
print("Number of observations:", df.shape[0])
print(f"Mean: {df.log_rtn.mean():.4f}")
print(f"Median: {df.log_rtn.median():.4f}")
print(f"Min: {df.log_rtn.min():.4f}")
print(f"Max: {df.log_rtn.max():.4f}")
print(f"Standard Deviation: {df.log_rtn.std():.4f}")
print(f"Skewness: {df.log_rtn.skew():.4f}")
print(f"Kurtosis: {df.log_rtn.kurtosis():.4f}")
print(f"Jarque-Bera statistic: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}")

# ------------------------------Fact2 - Volatility Clustering-------------------------
# Run the following code to visualize the log returns series:
(
    df["log_rtn"]
    .plot(title="Daily S&P 500 returns", figsize=(10, 6))
)

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_12", dpi=200);

# -------------------------------Fact3 - Absence of autocorrelation in returns
# Define the parameters for creating the autocorrelation plots:
N_LAGS = 50
SIGNIFICANCE_LEVEL = 0.05

# Run the following code to create ACF plot of log returns:
acf = smt.graphics.plot_acf(df["log_rtn"],
                            lags=N_LAGS,
                            alpha=SIGNIFICANCE_LEVEL)

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_13", dpi=200);

# -------------------------------Fact4 - Small and decreasing autocorrelation in squared / absolute returns-----------------------
# Create the ACF plots:
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

smt.graphics.plot_acf(df["log_rtn"] ** 2, lags=N_LAGS,
                      alpha=SIGNIFICANCE_LEVEL, ax=ax[0])
ax[0].set(title="Autocorrelation Plots",
          ylabel="Squared Returns")

smt.graphics.plot_acf(np.abs(df["log_rtn"]), lags=N_LAGS,
                      alpha=SIGNIFICANCE_LEVEL, ax=ax[1])
ax[1].set(ylabel="Absolute Returns",
          xlabel="Lag")

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_14", dpi=200);

# --------------------------------Fact 5 - Leverage effect------------------------------
# Calculate volatility measures as moving standard deviations
df["rolling_std_252"] = df[["log_rtn"]].rolling(window=252).std()
df["rolling_std_21"] = df[["log_rtn"]].rolling(window=21).std()

# Plot all the series:
fig, ax = plt.subplots(3, 1, figsize=(18, 15),
                       sharex=True)

df["adj_close"].plot(ax=ax[0])
ax[0].set(title="S&P 500 time series",
          ylabel="Price ($)")

df["log_rtn"].plot(ax=ax[1])
ax[1].set(ylabel="Log returns")

df["rolling_std_252"].plot(ax=ax[2], color="r",
                           label="Rolling Volatility 252d")
df["rolling_std_21"].plot(ax=ax[2], color="g",
                          label="Rolling Volatility 21d")
ax[2].set(ylabel="Moving Volatility",
          xlabel="Date")
ax[2].legend()

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_15", dpi=200);


# Download and preprocess the prices of the S & P 500 and VIX:
df = yf.download(["^GSPC", "^VIX"],
                 start="2000-01-01",
                 end="2020-12-31",
                 progress=False)
df = df[["Adj Close"]]
df.columns = df.columns.droplevel(0)
df = df.rename(columns={"^GSPC": "sp500", "^VIX": "vix"})

# Calculate the log returns(we can just as well use percentage change - simple returns):
df["log_rtn"] = np.log(df["sp500"] / df["sp500"].shift(1))
df["vol_rtn"] = np.log(df["vix"] / df["vix"].shift(1))
df.dropna(how="any", axis=0, inplace=True)

# Plot a scatterplot with the returns on the axes and fit a regression line to identify the trend:
corr_coeff = df.log_rtn.corr(df.vol_rtn)

ax = sns.regplot(x="log_rtn", y="vol_rtn", data=df,
                 line_kws={"color": "red"})
ax.set(title=f"S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})",
       ylabel="VIX log returns",
       xlabel="S&P 500 log returns")

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_4_16", dpi=200);



