# ------------------------------------------------------------3.1 Basic visualization of time series data How to do it...
# Import the libraries:
# %matplotlib inline
# %config
# InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# feel free to modify, for example, change the context to "notebook"
sns.set_theme(context="talk", style="whitegrid",
              palette="colorblind", color_codes=True,
              rc={"figure.figsize": [12, 8]})

import pandas as pd
import numpy as np
import yfinance as yf

# Download Microsoft's stock prices from 2020 and calculate simple returns:
df = yf.download("MSFT",
                 start="2020-01-01",
                 end="2020-12-31",
                 auto_adjust=False,
                 progress=False)

df["simple_rtn"] = df["Adj Close"].pct_change()
df = df.dropna()
# Plot the adjusted close prices:
df["Adj Close"].plot(title="MSFT stock in 2020");

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_3_1', dpi=200)

# It is the same as running the following:

df.plot.line(y="Adj Close", title="MSFT stock in 2020");

sns.despine()
plt.tight_layout()


# Plot the adjusted close prices and simple returns in one plot:
(
    df[["Adj Close", "simple_rtn"]]
    .plot(subplots=True, sharex=True,
          title="MSFT stock in 2020")
);

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_3_2', dpi=200)

# Create a similar plot to the previous one using matplotlib's object-oriented interface:
fig, ax = plt.subplots(2, 1, sharex=True)

# add prices
df["Adj Close"].plot(ax=ax[0])
ax[0].set(title="MSFT time series",
          ylabel="Stock price ($)")

# add volume
df["simple_rtn"].plot(ax=ax[1])
ax[1].set(ylabel="Return")

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_3_3', dpi=200)

# Change the plotting backend of pandas to plotly:
df["Adj Close"].plot(title="MSFT stock in 2020", backend="plotly")
# df["Adj Close"].plot(backend="altair")


# ---------------------------------------------------------------3.2 Visualizing seasonal patterns How to do it...
# Import the libraries and authenticate:
import pandas as pd
import nasdaqdatalink
import seaborn as sns

nasdaqdatalink.ApiConfig.api_key = "S6zt5z25_TfsUqWV1H5B"

# Download and display unemployment data from Nasdaq Data Link:
df = (
    nasdaqdatalink.get(dataset="FRED/UNRATENSA",
                       start_date="2014-01-01",
                       end_date="2019-12-31")
    .rename(columns={"Value": "unemp_rate"})
)
df.head()

print ("Succeed at line 95")

df.plot(title="Unemployment rate in years 2014-2019");
sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_3_6', dpi=200)

print ("Succeed at line 102")

# Create new columns with year and month:
df["year"] = df.index.year
df["month"] = df.index.strftime("%b")

print ("Succeed at line 108")

print(df)

# Create the seasonal plot:
# sns.lineplot(data=df,
#              x="month",
#              y="unemp_rate",
#              hue="year",
#              style="year",
#              legend="full",
#              palette="colorblind")
#
# print ("Succeed at line 119")
#
# plt.title("Unemployment rate - Seasonal plot")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
#
# print ("Succeed at line 120")
#
# sns.despine()
# plt.tight_layout()
# # plt.savefig('images/figure_3_7', dpi=200)
#
# print ("Succeed at line 124")

# There's more Import the libraries:
from statsmodels.graphics.tsaplots import month_plot, quarter_plot
import plotly.express as px

# Create a month plot:
month_plot(df["unemp_rate"], ylabel="Unemployment rate (%)")
plt.title("Unemployment rate - Month plot");

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_3_8', dpi=200)

# Create a quarter plot:
quarter_plot(df["unemp_rate"].resample("Q").mean(),
             ylabel="Unemployment rate (%)")
plt.title("Unemployment rate - Quarter plot");

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_3_9', dpi=200)

print ("Succeed at line 147")

# Create a polar seasonal plot using plotly.express:
fig = px.line_polar(
    df, r="unemp_rate", theta="month",
    color="year", line_close=True,
    title="Unemployment rate - Polar seasonal plot",
    width=600, height=500,
    range_r=[3, 7]
)

fig.show()

# # -------------------------------------------------------------3.3 Creating interactive visualizations-------------
# # Import the libraries and initialize Notebook display:
# import pandas as pd
# import yfinance as yf
#
# from plotly.offline import iplot, init_notebook_mode
# import plotly.express as px
#
# # Download Microsoft's stock prices from 2020 and calculate simple returns:
# df = yf.download("MSFT",
#                  start="2020-01-01",
#                  end="2020-12-31",
#                  auto_adjust=False,
#                  progress=False)
#
# df["simple_rtn"] = df["Adj Close"].pct_change()
# df = df.loc[:, ["Adj Close", "simple_rtn"]].dropna()
# df = df.dropna()
#
# # Create the plot using plotly.express:
# fig = px.line(data_frame=df,
#               y="Adj Close",
#               title="MSFT time series")
# fig.show()
#
# # Import the libraries:
# from datetime import date
#
# # Define the annotations for the plotly plot:
# selected_date_1 = date(2020, 2, 19)
# selected_date_2 = date(2020, 3, 23)
#
# selected_y_1 = (
#     df
#     .query(f"index == '{selected_date_1}'")
#     .loc[:, "Adj Close"]
#     .squeeze()
# )
# selected_y_2 = (
#     df
#     .query(f"index == '{selected_date_2}'")
#     .loc[:, "Adj Close"]
#     .squeeze()
# )
#
# first_annotation = {
#     "x": selected_date_1,
#     "y": selected_y_1,
#     "arrowhead": 5,
#     "text": "COVID decline starting",
#     "font": {"size": 15, "color": "red"},
# }
#
# second_annotation = {
#     "x": selected_date_2,
#     "y": selected_y_2,
#     "arrowhead": 5,
#     "text": "COVID recovery starting",
#     "font": {"size": 15, "color": "green"},
#     "ax": 150,
#     "ay": 10
# }
#
# # Update the layout of the plot and show it:
# fig.update_layout(
#     {"annotations": [first_annotation, second_annotation]}
# )
# fig.show()
#
# # --------------------------------------------------------3.4 Creating a candlestick chart---------------------------
# # Import the libraries:
# import pandas as pd
# import yfinance as yf
#
# # Download the adjusted prices from Yahoo Finance:
# df = yf.download("MSFT",
#                  start="2018-01-01",
#                  end="2018-12-31",
#                  progress=False,
#                  auto_adjust=True)
#
# import plotly.graph_objects as go
# import mplfinance as mpf
#
# # Create a candlestick chart using plotly:
# fig = go.Figure(data=
#                 go.Candlestick(x=df.index,
#                                open=df["Open"],
#                                high=df["High"],
#                                low=df["Low"],
#                                close=df["Close"])
#                 )
#
# fig.update_layout(
#     title="Twitter's stock prices in 2018",
#     yaxis_title="Price ($)"
# )
#
# fig.show()
#
# # Create a candlestick chart using mplfinance:
# mpf.plot(df, type="candle",
#          mav=(10, 20),
#          volume=True,
#          style="yahoo",
#          title="Twitter's stock prices in 2018",
#          figsize=(8, 4));
#
# sns.despine()
# plt.tight_layout()





