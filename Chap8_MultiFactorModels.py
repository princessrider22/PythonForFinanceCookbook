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

# --------------------------------------------------------------8.1 Estimating the CAPM------------------------------------------
# Import the libraries:
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# Specify the risky asset, the benchmark, and the time horizon:
RISKY_ASSET = "AMZN"
MARKET_BENCHMARK = "^GSPC"
START_DATE = "2016-01-01"
END_DATE = "2020-12-31"

# Download data from Yahoo Finance:
df = yf.download([RISKY_ASSET, MARKET_BENCHMARK],
                 start=START_DATE,
                 end=END_DATE,
                 progress=False)

print(f'Downloaded {df.shape[0]} rows of data.')

# Resample to monthly data and calculate simple returns:
X = (
    df["Adj Close"]
    .rename(columns={RISKY_ASSET: "asset",
                     MARKET_BENCHMARK: "market"})
    .resample("M")
    .last()
    .pct_change()
    .dropna()
)
X.head()

# Calculate beta using the covariance approach:
covariance = X.cov().iloc[0, 1]
benchmark_variance = X.market.var()
beta = covariance / benchmark_variance

# Prepare the input and estimate CAPM as a linear regression:
# separate target
y = X.pop("asset")

# add constant
X = sm.add_constant(X)

# define and fit the regression model
capm_model = sm.OLS(y, X).fit()

# print results
print(capm_model.summary())


# Or, using the formula notation:
import statsmodels.formula.api as smf

# rerun step 4 to have a DF with columns: `asset` and `market`
X = df["Adj Close"].rename(columns={RISKY_ASSET: "asset",
                                    MARKET_BENCHMARK: "market"}) \
    .resample("M") \
    .last() \
    .pct_change() \
    .dropna()

# define and fit the regression model
capm_model = smf.ols(formula="asset ~ market", data=X).fit()

# print results
print(capm_model.summary())


# ---------------------------------------Risk - free rate(13 Week Treasury Bill)------------------------------
# period length in days
N_DAYS = 90

# download data from Yahoo finance
df_rf = yf.download("^IRX",
                    start=START_DATE,
                    end=END_DATE,
                    progress=False)

# resample to monthly by taking last value from each month
rf = df_rf.resample("M").last().Close / 100

# calculate the corresponding daily risk-free return
rf = (1 / (1 - rf * N_DAYS / 360)) ** (1 / N_DAYS)

# convert to monthly and subtract 1
rf = (rf ** 30) - 1

# plot the risk-free rate
rf.plot(title="Risk-free rate (13 Week Treasury Bill)")

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_9_2", dpi=200)

# ---------------------------------------Risk - free rate(3 - Month Treasury Bill)-----------------------------------
import pandas_datareader.data as web

# download the data
rf = web.DataReader(
    "TB3MS", "fred", start=START_DATE, end=END_DATE
)

# convert to monthly
rf = (1 + (rf / 100)) ** (1 / 12) - 1

# plot the risk-free rate
rf.plot(title="Risk-free rate (3-Month Treasury Bill)")

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_9_3", dpi=200)

# ------------------------------------------------------------------8.2 Estimating the Fama - French three - factor model--------------------------------------
# Import the libraries:
import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
import pandas_datareader.data as web

# Define parameters:
RISKY_ASSET = "AAPL"
START_DATE = "2016-01-01"
END_DATE = "2020-12-31"

# Download the dataset containing the risk factors:
ff_dict = web.DataReader("F-F_Research_Data_Factors",
                         "famafrench",
                         start=START_DATE,
                         end=END_DATE)
ff_dict.keys()

print(ff_dict['DESCR'])

# Select the appropriate dataset and divide the values by 100:
factor_3_df = ff_dict[0].rename(columns={"Mkt-RF": "MKT"}).div(100)

factor_3_df.head()

# Download the prices of the risky asset:
asset_df = yf.download(RISKY_ASSET,
                       start=START_DATE,
                       end=END_DATE,
                       progress=False)

print(f"Downloaded {asset_df.shape[0]} rows of data.")

# Calculate monthly returns on the risky asset:
y = asset_df["Adj Close"].resample("M") \
    .last() \
    .pct_change() \
    .dropna()

y.index = y.index.to_period("m")
y.name = "rtn"
y.head()

# Merge the datasets and calculate excess returns:
factor_3_df = factor_3_df.join(y)
factor_3_df["excess_rtn"] = (
        factor_3_df["rtn"] - factor_3_df["RF"]
)
factor_3_df.head()

# Estimate the three - factor model:
# define and fit the regression model
ff_model = smf.ols(formula="excess_rtn ~ MKT + SMB + HML",
                   data=factor_3_df).fit()

# print results
print(ff_model.summary())

# Print available datasets(here only first 5):

from pandas_datareader.famafrench import get_available_datasets

get_available_datasets()[:5]


# -------------------------------------------------------8.3 Estimating the rolling three - factor model on a portfolio of assets -----------------------------------------------------
# Import the libraries:
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.formula.api as smf
import pandas_datareader.data as web

# Define the parameters:
ASSETS = ["AMZN", "GOOG", "AAPL", "MSFT"]
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
START_DATE = "2010-01-01"
END_DATE = "2020-12-31"

# Download the factor related data:
factor_3_df = web.DataReader("F-F_Research_Data_Factors",
                             "famafrench",
                             start=START_DATE,
                             end=END_DATE)[0]
factor_3_df = factor_3_df.div(100)

# Download the prices of risky assets from Yahoo Finance:
asset_df = yf.download(ASSETS,
                       start=START_DATE,
                       end=END_DATE,
                       progress=False)

print(f"Downloaded {asset_df.shape[0]} rows of data.")

# Calculate the monthly returns on the risky assets:
asset_df = asset_df["Adj Close"].resample("M") \
    .last() \
    .pct_change() \
    .dropna()
# reformat index for joining
asset_df.index = asset_df.index.to_period("m")

# Calculate the portfolio returns:
asset_df["portfolio_returns"] = np.matmul(
    asset_df[ASSETS].values,
    WEIGHTS
)
asset_df.head()
asset_df.plot();

# Merge the datasets:
factor_3_df = asset_df.join(factor_3_df).drop(ASSETS, axis=1)
factor_3_df.columns = ["portf_rtn", "mkt", "smb", "hml", "rf"]
factor_3_df["portf_ex_rtn"] = (
        factor_3_df["portf_rtn"] - factor_3_df["rf"]
)

# Define a function for the rolling n-factor model
def rolling_factor_model(input_data, formula, window_size):
    """
    Function for estimating the Fama-French (n-factor) model using a rolling window of fixed size.

    Parameters
    ------------
    input_data : pd.DataFrame
        A DataFrame containing the factors and asset/portfolio returns
    formula : str
        `statsmodels` compatible formula representing the OLS regression
    window_size : int
        Rolling window length.

    Returns
    -----------
    coeffs_df : pd.DataFrame
        DataFrame containing the intercept and the three factors for each iteration.
    """

    coeffs = []

    for start_ind in range(len(input_data) - window_size + 1):
        end_ind = start_ind + window_size

        # define and fit the regression model
        ff_model = smf.ols(
            formula=formula,
            data=input_data[start_ind:end_ind]
        ).fit()

        # store coefficients
        coeffs.append(ff_model.params)

    coeffs_df = pd.DataFrame(
        coeffs,
        index=input_data.index[window_size - 1:]
    )

    return coeffs_df


# Estimate the rolling three - factor model and plot the results:
MODEL_FORMULA = "portf_ex_rtn ~ mkt + smb + hml"
results_df = rolling_factor_model(factor_3_df,
                                  MODEL_FORMULA,
                                  window_size=60)
(
    results_df
    .plot(title="Rolling Fama-French Three-Factor model",
          style=["-", "--", "-.", ":"])
    .legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
)

sns.despine()
plt.tight_layout()
# plt.savefig("images/figure_9_6", dpi=200)

# ------------------------------------------------------------8.4 Estimating the four - and five - factor models-------------------------------------------------------
# Import the libraries:
import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
import pandas_datareader.data as web

# Specify the risky asset and the time horizon:
RISKY_ASSET = "AMZN"
START_DATE = "2016-01-01"
END_DATE = "2020-12-31"

# Download the risk factors from prof.French's website:
# three factors
factor_3_df = web.DataReader("F-F_Research_Data_Factors",
                             "famafrench",
                             start=START_DATE,
                             end=END_DATE)[0]

# momentum factor
momentum_df = web.DataReader("F-F_Momentum_Factor",
                             "famafrench",
                             start=START_DATE,
                             end=END_DATE)[0]

# five factors
factor_5_df = web.DataReader("F-F_Research_Data_5_Factors_2x3",
                             "famafrench",
                             start=START_DATE,
                             end=END_DATE)[0]
# Download the data of the risky asset from Yahoo Finance:
asset_df = yf.download(RISKY_ASSET,
                       start=START_DATE,
                       end=END_DATE,
                       progress=False)

print(f"Downloaded {asset_df.shape[0]} rows of data.")

# Calculate monthly returns:
y = asset_df["Adj Close"].resample("M") \
    .last() \
    .pct_change() \
    .dropna()

y.index = y.index.to_period("m")
y.name = "rtn"

# Merge the datasets for the four - factor models:
# join all datasets on the index
factor_4_df = factor_3_df.join(momentum_df).join(y)

# rename columns
factor_4_df.columns = ["mkt", "smb", "hml", "rf", "mom", "rtn"]

# divide everything (except returns) by 100
factor_4_df.loc[:, factor_4_df.columns != "rtn"] /= 100

# calculate excess returns
factor_4_df["excess_rtn"] = (
        factor_4_df["rtn"] - factor_4_df["rf"]
)

factor_4_df.head()

# Merge the datasets for the five - factor models:
# join all datasets on the index
factor_5_df = factor_5_df.join(y)

# rename columns
factor_5_df.columns = [
    "mkt", "smb", "hml", "rmw", "cma", "rf", "rtn"
]

# divide everything (except returns) by 100
factor_5_df.loc[:, factor_5_df.columns != "rtn"] /= 100

# calculate excess returns
factor_5_df["excess_rtn"] = (
        factor_5_df["rtn"] - factor_5_df["rf"]
)

factor_5_df.head()

# Estimate the four - factor model:
four_factor_model = smf.ols(
    formula="excess_rtn ~ mkt + smb + hml + mom",
    data=factor_4_df
).fit()

print(four_factor_model.summary())

# Estimate the five - factor model:
five_factor_model = smf.ols(
    formula="excess_rtn ~ mkt + smb + hml + rmw + cma",
    data=factor_5_df
).fit()

print(five_factor_model.summary())

# -----------------------------------------------------------8.5 Estimating cross - sectional factor models using the Fama - MacBeth regression
# Import the libraries:
import pandas as pd
import pandas_datareader.data as web
from linearmodels.asset_pricing import LinearFactorModel

# Specify the time horizon:
START_DATE = "2010"
END_DATE = "2020-12"

# Download and adjust the risk factors from prof.French's website:
factor_5_df = (
    web.DataReader("F-F_Research_Data_5_Factors_2x3",
                   "famafrench",
                   start=START_DATE,
                   end=END_DATE)[0]
    .div(100)
)
factor_5_df.head()

# Download and adjust the returns of 12 Industry Portfolios from prof.French's website:
portfolio_df = (
    web.DataReader("12_Industry_Portfolios",
                   "famafrench",
                   start=START_DATE,
                   end=END_DATE)[0]
    .div(100)
    .sub(factor_5_df["RF"], axis=0)
)
portfolio_df.head()

# Drop the risk - free rate from the factor data set:
factor_5_df = factor_5_df.drop("RF", axis=1)
factor_5_df.head()

# Estimate the Fama - MacBeth regression and print the summary:
five_factor_model = LinearFactorModel(
    portfolios=portfolio_df,
    factors=factor_5_df
)
result = five_factor_model.fit()
print(result)


# We can also print the full summary(1 aggregate and 12 individual ones for each portfolio separately).

print(result.full_summary)

# Import the libraries:
from statsmodels.api import OLS, add_constant

# First step - estimate the factor loadings:
factor_loadings = []
for portfolio in portfolio_df:
    reg_1 = OLS(
        endog=portfolio_df.loc[:, portfolio],
        exog=add_constant(factor_5_df)
    ).fit()
    factor_loadings.append(reg_1.params.drop("const"))

# Store the factor loadings in a DataFrame:
factor_load_df = pd.DataFrame(
    factor_loadings,
    columns=factor_5_df.columns,
    index=portfolio_df.columns
)
factor_load_df.head()
print(factor_load_df)

# Second step - estimate the risk premia:
risk_premia = []
for period in portfolio_df.index:
    reg_2 = OLS(
        endog=portfolio_df.loc[period, factor_load_df.index],
        exog=factor_load_df
    ).fit()
    risk_premia.append(reg_2.params)

# Store the risk premia in a DataFrame:
risk_premia_df = pd.DataFrame(
    risk_premia,
    index=portfolio_df.index,
    columns=factor_load_df.columns.tolist())
risk_premia_df.head()

print(risk_premia_df)

# Calculate the average risk premia:
risk_premia_df.mean()

print (risk_premia)