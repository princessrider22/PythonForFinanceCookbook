# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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


# ------------------------------------------Chapter 1: Acquiring Financial Data-----------------------------------------------------

# 1. Import the libraries:
import pandas as pd
import yfinance as yf
import nasdaqdatalink as ndq

#
# # 2. Download the data from Yahoo Finance:
# df = yf.download("AAPL",
#                  start="2011-01-01",
#                  end="2021-12-31",
#                  progress=False)
#
# # 3. Inspect the data:
# print(f"Downloaded {len(df)} rows of data.")
#
# print(df)
#
# # 4. We can also use the Ticker class to download the historical prices and much more.
# aapl_data = yf.Ticker("AAPL")
#
# # 5. get the last month of historical prices
# aapl_data.history()
#
# # 6. get stock's info
# aapl_data.info
#
# # 7. show corporate actions
# aapl_data.actions
#
# # 8. show financials
# aapl_data.financials

# # 9. show quarterly financials
# aapl_data.quarterly_financials

# Authenticate using the personal API key:
ndq.ApiConfig.api_key = "S6zt5z25_TfsUqWV1H5B"

# Download the data:
df = ndq.get(dataset="WIKI/AAPL",
                        start_date="2011-01-01",
                        end_date="2021-12-31")
# Inspect the data:
print(f"Downloaded {len(df)} rows of data from Nasdaq.")

print(df.head())

print(df)


COLUMNS = ["ticker", "date", "adj_close"]
df = ndq.get_table("WIKI/PRICES",
                              ticker=["AAPL", "MSFT", "INTC"],
                              qopts={"columns": COLUMNS},
                              date={"gte": "2011-01-01",
                                    "lte": "2021-12-31"},
                              paginate=True)
print(df.head())

print(df)


# Pivot the data from long to wide:
# set the date as index
df = df.set_index("date")

# use the pivot function to reshape the data
df_wide = df.pivot(columns="ticker")
df_wide.head()

print(df_wide)



# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')


