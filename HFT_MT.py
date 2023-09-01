# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:40:42 2021

@author: Mark Marner-Hausen
"""


import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
import scipy.optimize
import statistics
import collections
from datetime import datetime
from scipy import stats
import time
import glob
from python_wip.create_pdf import create_pdf
from python_wip.lineplot_price import lineplot_price


# Start to stop the time
start_time = time.time()

# Load data
data_market = pd.read_csv("data\Pilot2.0_CDA_Market.csv", sep=",")
data_trader = pd.read_csv("data\Pilot2.0_CDA_Trader.csv", sep=",")
external = pd.read_csv("data\external_feed\external_feed_T0_May2021.csv", sep=",")
investor = pd.read_csv("data\investor_flow\investor_focal_T0_May2021.csv", sep=",")

# Session Overview
print(
    "Number of Participants",
    max(data_trader.market_id) - min(data_trader.market_id) + 1,
)
print(
    "Number of Rounds",
    (
        max(data_trader.player_id)
        - min(data_trader.player_id[data_trader.player_id > 1])
        + 1
    )
    / (max(data_trader.market_id) - min(data_trader.market_id) + 1),
)


# =============================================================================
# Trader Outcomes

# Clean Timestamp
data_trader.timestamp = pd.to_datetime(data_trader.timestamp)
data_trader = data_trader.reset_index(drop=True)

# Delete undesired variables
data_trader = data_trader[
    [
        "timestamp",
        "market_id",
        "player_id",
        "trigger_event_type",
        "speed_cost",
        "net_worth",
        "inventory",
        "tax_paid",
        "trader_model_name",
        "slider_a_y",
        "slider_a_z",
    ]
]

# Consider Trader Choices (Strategy and Sensitivities)
# To do so, meed to consider data on player level using a for loop
data_trader_cleaned = pd.DataFrame()

for i in range(
    min(data_trader.player_id[data_trader.player_id > 1]),
    max(data_trader.player_id) + 1,
):
    tmp = data_trader[data_trader.player_id == i]

    # Delete every action before the start of the market in order to stay
    # at 240 seconds
    tmp_help = tmp.index[tmp.trigger_event_type == "market_start"]
    tmp = tmp[tmp.index >= tmp_help[0]]

    # Calculate the timedelta on player level
    tmp["timedelta"] = pd.Series(tmp.timestamp.shift(-1), index=tmp.index)
    tmp.timedelta.replace({pd.NaT: max(tmp.timestamp)}, inplace=True)
    tmp.timedelta = tmp.timedelta - tmp.timestamp
    tmp.timedelta = tmp.timedelta.dt.total_seconds()

    # Create the new dataset to work with
    data_trader_cleaned = pd.concat([data_trader_cleaned, tmp], sort=True)

# Time on each Strategy and Sensitivity (slider_a_y inventory, slider_a_z external).
data_trader_cleaned["out"] = data_trader_cleaned.timedelta[
    data_trader_cleaned.trader_model_name == "out"
]
data_trader_cleaned["automated"] = data_trader_cleaned.timedelta[
    data_trader_cleaned.trader_model_name == "automated"
]
data_trader_cleaned["manual"] = data_trader_cleaned.timedelta[
    data_trader_cleaned.trader_model_name == "manual"
]
data_trader_cleaned = data_trader_cleaned.fillna(0)

# Consider weighted averages for the Sensitivities
data_trader_cleaned["inventory_sensitivity"] = np.multiply(
    data_trader_cleaned.slider_a_y, data_trader_cleaned.timedelta
)
data_trader_cleaned["externalfeed_sensitivity"] = np.multiply(
    data_trader_cleaned.slider_a_z, data_trader_cleaned.timedelta
)
# Consider relative Strategies and weighted average Sensitivities
data_trader_cleaned = data_trader_cleaned[
    [
        "market_id",
        "manual",
        "out",
        "automated",
        "inventory_sensitivity",
        "externalfeed_sensitivity",
    ]
]
summary_trader = data_trader_cleaned.groupby("market_id").agg(["sum", "count"])

# total_seconds = 6 * 240 = 1440 per round
tmp = [("manual", "sum"), ("out", "sum"), ("automated", "sum")]
summary_trader[("total_seconds", "sum")] = summary_trader[tmp].sum(axis=1)

# Scale by dividing by total_seconds
tmp = [
    ("manual", "sum"),
    ("out", "sum"),
    ("automated", "sum"),
    ("inventory_sensitivity", "sum"),
    ("externalfeed_sensitivity", "sum"),
]
summary_trader[tmp] = summary_trader[tmp].div(
    summary_trader.loc[:, ("total_seconds", "sum")], axis=0
)
summary_trader = summary_trader.drop(
    [
        ("manual", "count"),
        ("out", "count"),
        ("automated", "count"),
        ("inventory_sensitivity", "count"),
        ("externalfeed_sensitivity", "count"),
    ],
    axis=1,
)

# Consider Performance of Traders
data_trader = data_trader[data_trader.trigger_event_type == "market_end"]
data_trader["end_inventory_sensitivity"] = data_trader.slider_a_y
data_trader = data_trader[
    [
        "market_id",
        "speed_cost",
        "net_worth",
        "inventory",
        "tax_paid",
        "end_inventory_sensitivity",
    ]
]
tmp = data_trader.groupby("market_id").agg(["min", "max", "mean", "std"])
tmp = tmp.drop(
    [("end_inventory_sensitivity", "min"), ("end_inventory_sensitivity", "max")], axis=1
)

# Merge Performance and Choices
summary_trader = pd.concat([summary_trader, tmp], axis=1)
tmp = [
    ("speed_cost", "min"),
    ("speed_cost", "max"),
    ("speed_cost", "mean"),
    ("speed_cost", "std"),
    ("net_worth", "min"),
    ("net_worth", "max"),
    ("net_worth", "mean"),
    ("net_worth", "std"),
    ("tax_paid", "min"),
    ("tax_paid", "max"),
    ("tax_paid", "mean"),
    ("tax_paid", "std"),
]
summary_trader[tmp] = summary_trader[tmp].div(10000, axis=0)


# =============================================================================
# Market outcomes
#
# Clean Timestamp
data_market.timestamp = pd.to_datetime(data_market.timestamp)
data_market = data_market.reset_index(drop=True)

# Delete undesired variables
data_market = data_market[
    [
        "timestamp",
        "market_id",
        "trigger_event_type",
        "reference_price",
        "best_bid",
        "best_offer",
        "volume_at_best_bid",
        "volume_at_best_offer",
    ]
]

# Create Timedelta
data_market["timedelta"] = data_market.timestamp.shift(-1)
data_market.timedelta.replace({pd.NaT: max(data_market.timestamp)}, inplace=True)
data_market.timedelta = data_market.timedelta - data_market.timestamp
data_market.timedelta = data_market.timedelta.dt.total_seconds()

# With time passed weighted Marketspread
data_market["weighted_ms"] = np.multiply(
    (np.subtract(data_market.best_offer, data_market.best_bid)), data_market.timedelta
)

# Exclude Market start and end as it creates outliers
data_market = data_market[
    (data_market.trigger_event_type != "market_end")
    & (data_market.trigger_event_type != "market_start")
]

# Market Data for liquid Market
# Use another Name here cause otherwise oputliers are always excluded from now on
data_market_cleaned = data_market[
    (data_market.best_bid != 0) & (data_market.best_offer != 2147483647)
]

tmp = data_market_cleaned.drop(
    ["timestamp", "trigger_event_type", "reference_price", "timedelta"], axis=1
)
summary_market_liq = tmp.groupby("market_id").agg(["min", "max", "mean", "std"])

# We need weighted average for ms and thus need to scale the weighted_ms. Divide sum of weighthed_ms
# by total_seconds (240 seconds in this case)
tmp_help = data_market_cleaned[["market_id", "timedelta", "weighted_ms"]]
tmp_help = tmp_help.groupby("market_id").sum()
summary_market_liq[("total_seconds", "sum")] = tmp_help.loc[:, "timedelta"]
summary_market_liq.loc[:, ("weighted_ms", "mean")] = np.divide(
    tmp_help.loc[:, "weighted_ms"], summary_market_liq.loc[:, ("total_seconds", "sum")]
)

# Include Reference price unequal 0. Reference price equal 0 might occure in an instances
# where no trade took place.
tmp = data_market_cleaned[["market_id", "reference_price"]]
tmp = tmp[data_market_cleaned.reference_price != 0]
tmp = tmp.groupby("market_id").agg(["min", "max", "mean", "std"])

summary_market_liq = pd.concat([summary_market_liq, tmp], axis=1)
tmp = [
    ("weighted_ms", "min"),
    ("weighted_ms", "max"),
    ("weighted_ms", "mean"),
    ("weighted_ms", "std"),
    ("best_bid", "min"),
    ("best_bid", "max"),
    ("best_bid", "mean"),
    ("best_bid", "std"),
    ("best_offer", "min"),
    ("best_offer", "max"),
    ("best_offer", "mean"),
    ("best_offer", "std"),
    ("reference_price", "min"),
    ("reference_price", "max"),
    ("reference_price", "mean"),
    ("reference_price", "std"),
]
summary_market_liq[tmp] = summary_market_liq[tmp].div(10000, axis=0)


# Market dry up (Exclude instances where both sides are dry, might be technical!)
data_market_cleaned = data_market[
    (data_market.best_offer == 2147483647) | (data_market.best_bid == 0)
]

data_market_cleaned = data_market_cleaned.drop(
    data_market_cleaned[
        (data_market_cleaned["best_bid"] == 0)
        & (data_market_cleaned["best_offer"] == 2147483647)
    ].index,
    axis=0,
)

# Count the dry instances on each side
data_market_cleaned.loc[data_market_cleaned["best_bid"] == 0, "dry_buyside"] = 1
data_market_cleaned.loc[
    data_market_cleaned["best_offer"] == 2147483647, "dry_sellside"
] = 1
data_market_cleaned = data_market_cleaned.fillna(0)

# Exclude parameters that are not of interest for a dry market
data_market_cleaned = data_market_cleaned.drop(
    ["reference_price", "best_bid", "best_offer", "weighted_ms"], axis=1
)

summary_market_dry = data_market_cleaned.groupby("market_id").agg(["mean", "sum"])

summary_market_dry = summary_market_dry.drop(
    [
        ("volume_at_best_bid", "mean"),
        ("volume_at_best_offer", "mean"),
        ("timedelta", "mean"),
        ("dry_buyside", "mean"),
        ("dry_sellside", "mean"),
    ],
    axis=1,
)

# Add Round number to the summary Tables
tmp_help = pd.DataFrame(
    range(1, max(data_trader.market_id) - min(data_trader.market_id) + 2)
)
summary_market_liq = summary_market_liq.reset_index(drop=True)
summary_market_liq[("Round", "Number")] = tmp_help
tmp = summary_market_liq.columns.tolist()
tmp = tmp[-1:] + tmp[:-1]
summary_market_liq = summary_market_liq[tmp]

summary_market_dry = summary_market_dry.reset_index(drop=True)
summary_market_dry[("Round", "Number")] = tmp_help
tmp = summary_market_dry.columns.tolist()
tmp = tmp[-1:] + tmp[:-1]
summary_market_dry = summary_market_dry[tmp]

summary_trader = summary_trader.reset_index(drop=True)
summary_trader[("Round", "Number")] = tmp_help
tmp = summary_trader.columns.tolist()
tmp = tmp[-1:] + tmp[:-1]
summary_trader = summary_trader[tmp]


# Create Graphs

# Barplot for the mean of best_offer, best_bid and reference_price
tmp = summary_market_liq[
    [("best_bid", "mean"), ("reference_price", "mean"), ("best_offer", "mean")]
]
tmp.columns = tmp.columns.droplevel(1)
tmp = tmp.reset_index(drop=True)
tmp["round"] = pd.DataFrame(
    range(1, max(data_trader.market_id) - min(data_trader.market_id) + 2)
)
# Plots usually only take one column per X and Y axis. Thus we need to melt the data
tmp = tmp.melt("round", var_name="Prices", value_name="vals")

# sns.set_theme(style="whitegrid")
fig1 = sns.barplot(x="round", y="vals", hue="Prices", data=tmp, palette="tab10")
fig1.set(xlabel="Round", ylabel="Mean Price")
fig1 = fig1.get_figure()
fig1.savefig("graph_realizedprices.pdf")

# Horizontal Barplot for Strategy Choice
tmp = summary_trader[[("out", "sum"), ("manual", "sum"), ("automated", "sum")]]
tmp.columns = tmp.columns.droplevel(1)
tmp = tmp.reset_index(drop=True)
tmp["round"] = pd.DataFrame(
    range(1, max(data_trader.market_id) - min(data_trader.market_id) + 2)
)
tmp = tmp.melt("round", var_name="Strategies", value_name="vals")

fig2 = tmp.pivot(index="round", columns="Strategies", values="vals").plot(
    kind="barh", stacked=True, colormap="flare"
)
fig2.set(xlabel="Strategy Choice in %", ylabel="Rounds")
fig2 = fig2.get_figure()
fig2.savefig("graph_strategychoices.pdf")


# Create pdf for Tables
# Trader Outcomes

table = summary_trader[
    [
        ("Round", "Number"),
        ("externalfeed_sensitivity", "sum"),
        ("inventory_sensitivity", "sum"),
        ("end_inventory_sensitivity", "mean"),
    ]
]
table.columns = table.columns.droplevel(1)
table = table.round(3)
create_pdf(table, 1, "summary_sensitivity.pdf", 11, 2, 11)

# Tmp is only for having the right column Names
tmp = summary_trader
tmp = tmp.rename(
    columns={
        "speed_cost": "Speed Investment",
        "net_worth": "Profits",
        "best_bid": "Best Bid",
        "best_offer": "Best Ask",
        "reference_price": "Reference Price",
    }
)
table = tmp[
    [
        ("Round", "Number"),
        ("Speed Investment", "min"),
        ("Speed Investment", "max"),
        ("Speed Investment", "mean"),
        ("Speed Investment", "std"),
        ("Profits", "min"),
        ("Profits", "max"),
        ("Profits", "mean"),
        ("Profits", "std"),
    ]
]
# table = table.rename(columns={'speed_cost': 'Speed Investment', 'net_worth': 'Profits'})
table = table.round(1)
create_pdf(table, 2, "summary_speedworth.pdf", 11, 1, 11)

table = summary_trader[
    [
        ("Round", "Number"),
        ("inventory", "min"),
        ("inventory", "max"),
        ("inventory", "std"),
        ("tax_paid", "min"),
        ("tax_paid", "max"),
        ("tax_paid", "mean"),
        ("tax_paid", "std"),
    ]
]
table = table.round(1)
create_pdf(table, 2, "summary_inventorytax.pdf", 11, 1, 11)


# Market Outcomes

table = summary_market_liq[
    [
        ("Round", "Number"),
        ("volume_at_best_bid", "min"),
        ("volume_at_best_bid", "max"),
        ("volume_at_best_bid", "mean"),
        ("volume_at_best_bid", "std"),
        ("volume_at_best_offer", "min"),
        ("volume_at_best_offer", "max"),
        ("volume_at_best_offer", "mean"),
        ("volume_at_best_offer", "std"),
    ]
]
table = table.round(2)
create_pdf(table, 2, "volume_liq.pdf", 11, 1, 11)

table = summary_market_liq[
    [
        ("Round", "Number"),
        ("weighted_ms", "min"),
        ("weighted_ms", "max"),
        ("weighted_ms", "mean"),
        ("weighted_ms", "std"),
        ("total_seconds", "sum"),
    ]
]
table = table.round(2)
create_pdf(table, 2, "summary_ms_liq.pdf", 11, 1, 11)


tmp = summary_market_liq
tmp = tmp.rename(
    columns={
        "best_bid": "Best Bid",
        "best_offer": "Best Ask",
        "reference_price": "Reference Price",
    }
)
table = tmp[
    [
        ("Round", "Number"),
        ("Best Bid", "min"),
        ("Best Bid", "max"),
        ("Best Bid", "std"),
        ("Best Ask", "min"),
        ("Best Ask", "max"),
        ("Best Ask", "std"),
        ("Reference Price", "min"),
        ("Reference Price", "max"),
        ("Reference Price", "std"),
    ]
]
# table = table.rename(columns={'best_bid': 'Best Bid', 'best_offer': 'Best Ask', 'reference_price': 'Reference Price'})
table = table.round(2)
create_pdf(table, 2, "summary_prices.pdf", 11, 1, 11)

table = summary_market_dry[
    [
        ("Round", "Number"),
        ("dry_buyside", "sum"),
        ("dry_sellside", "sum"),
        ("volume_at_best_bid", "sum"),
        ("volume_at_best_offer", "sum"),
        ("timedelta", "sum"),
    ]
]
table.columns = table.columns.droplevel(1)
table["timedelta"] = table["timedelta"].round(2)
create_pdf(table, 1, "summary_dry.pdf", 11, 2, 11)


# =============================================================================
#
# Pilot 2 Fundamentals are equivalent except T4
#


# # Load data Investor Flow
path = r"C:\Users\Mark\Google Drive\Uni\M.Sc\Economics\Master_Thesis\WorkInProgress\Figures\data\investor_flow_p2"
files = glob.glob(path + "/*.csv")

investor_flow = []
i = 1
for f in files:
    tmp = pd.read_csv(f, index_col=None, header=0)
    tmp = tmp.drop(["time_in_force", "market_id_in_subsession"], axis=1)
    tmp["rounds"] = i

    # Include identifier
    tmp_help = tmp[["arrival_time", "fundamental_value", "rounds"]]
    tmp_help = tmp_help.rename(columns={"fundamental_value": "price"})
    tmp_help.insert(2, "buy_sell_indicator", "fundamental_value")
    tmp = tmp.drop(["fundamental_value"], axis=1)
    tmp = tmp.append(tmp_help)

    tmp["price"] = tmp["price"].div(10000, axis=0)
    investor_flow.append(tmp)
    i = i + 1

# Load data External Feed
path = r"C:\Users\Mark\Google Drive\Uni\M.Sc\Economics\Master_Thesis\WorkInProgress\Figures\data\external_feed_p2"
files = glob.glob(path + "/*.csv")

i = 1
for f in files:
    tmp = pd.read_csv(f, index_col=None, header=0)
    tmp = tmp[(tmp.e_best_bid != 0) & (tmp.e_best_offer != 2147483647)]
    tmp_help = ["e_best_bid", "e_best_offer"]
    tmp[tmp_help] = tmp[tmp_help].div(10000, axis=0)
    tmp_help = np.multiply(np.subtract(tmp.e_best_offer, tmp.e_best_bid), 1 / 2)
    tmp["price"] = np.add(tmp.e_best_bid, tmp_help)
    tmp = tmp.drop(
        ["e_best_bid", "e_best_offer", "e_signed_volume", "market_id_in_subsession"],
        axis=1,
    )

    # Include identifier
    tmp.insert(2, "buy_sell_indicator", "external_feed")

    tmp["rounds"] = i
    investor_flow.append(tmp)
    i = i + 1


# Combine Both
investor_flow = pd.concat(investor_flow, axis=0, ignore_index=True)
investor_flow = investor_flow.sort_values("buy_sell_indicator", ascending=False)
# price_data = price_data.reset_index(drop=True)

# =============================================================================
# # Plot All Pilot 2
# lineplot_price(investor_flow, 3, 2, 16 ,17, 'p2_prices_all', ['Fundamental', 'Fundamental Extern', 'Ask', 'Bid'],
#                                                              [1, 2, 3, 4, 5, 6],
#                '16', 'arrival_time', 'price', 'buy_sell_indicator', 'Time of Arrival', 'Price', 'tab10')
#
# =============================================================================

# # Plot All Pilot 2 fundamentals
tmp = investor_flow[
    (investor_flow.buy_sell_indicator == "external_feed")
    | (investor_flow.buy_sell_indicator == "fundamental_value")
]
lineplot_price(
    tmp,
    3,
    2,
    16,
    17,
    "p2_prices_all_fundamentals",
    ["Fundamental", "Fundamental Extern"],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "18",
    "arrival_time",
    "price",
    "buy_sell_indicator",
    "Time of Arrival",
    "Price",
    "tab10",
)

# =============================================================================
#
# # Stress Pilot 2
# tmp = investor_flow[(investor_flow.rounds == 5) |
#                     (investor_flow.rounds == 6)]
# tmp = tmp[(tmp.buy_sell_indicator == 'external_feed') |
#           (tmp.buy_sell_indicator == 'fundamental_value')]
# lineplot_price(tmp, 1, 2, 14 , 6, 'p2_prices_short_stress_fundamentals', ['Fundamental', 'Fundamental Extern'], [5, 6],
#                '14', 'arrival_time', 'price', 'buy_sell_indicator', 'Time of Arrival', 'Price', 'tab10')
# =============================================================================


# =============================================================================
# Pilot 3
# =============================================================================
# Load data Investor Flow
path = r"C:\Users\Mark\Google Drive\Uni\M.Sc\Economics\Master_Thesis\WorkInProgress\Figures\data\investor_flow"
files = glob.glob(path + "/*.csv")

investor_flow = []
i = 1
for f in files:
    tmp = pd.read_csv(f, index_col=None, header=0)
    tmp = tmp.drop(["time_in_force", "market_id_in_subsession"], axis=1)
    tmp["rounds"] = i

    # Include identifier
    tmp_help = tmp[["arrival_time", "fundamental_value", "rounds"]]
    tmp_help = tmp_help.rename(columns={"fundamental_value": "price"})
    tmp_help.insert(2, "buy_sell_indicator", "fundamental_value")
    tmp = tmp.drop(["fundamental_value"], axis=1)
    tmp = tmp.append(tmp_help)

    tmp["price"] = tmp["price"].div(10000, axis=0)
    investor_flow.append(tmp)
    i = i + 1


# Load data Normal External Feed
path = r"C:\Users\Mark\Google Drive\Uni\M.Sc\Economics\Master_Thesis\WorkInProgress\Figures\data\external_feed"
files = glob.glob(path + "/*.csv")


i = 1
for f in files:
    tmp = pd.read_csv(f, index_col=None, header=0)
    tmp = tmp[(tmp.e_best_bid != 0) & (tmp.e_best_offer != 2147483647)]
    tmp_help = ["e_best_bid", "e_best_offer"]
    tmp[tmp_help] = tmp[tmp_help].div(10000, axis=0)
    tmp_help = np.multiply(np.subtract(tmp.e_best_offer, tmp.e_best_bid), 1 / 2)
    tmp["price"] = np.add(tmp.e_best_bid, tmp_help)
    tmp = tmp.drop(
        ["e_best_bid", "e_best_offer", "e_signed_volume", "market_id_in_subsession"],
        axis=1,
    )

    # Include identifier
    tmp.insert(2, "buy_sell_indicator", "external_feed")

    tmp["rounds"] = i
    investor_flow.append(tmp)
    i = i + 1


# Combine Both
investor_flow = pd.concat(investor_flow, axis=0, ignore_index=True)
investor_flow = investor_flow.sort_values("buy_sell_indicator", ascending=False)
# price_data = price_data.reset_index(drop=True)


# =============================================================================
# # Plot All Pilot 3
# lineplot_price(investor_flow, 5, 2, 21 ,28, 'p3_prices_all', ['Fundamental', 'Fundamental Extern', 'Ask', 'Bid'],
#                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                '18', 'arrival_time', 'price', 'buy_sell_indicator', 'Time of Arrival', 'Price', 'tab10')
# =============================================================================


# # Plot All Pilot 3 fundamentals
tmp = investor_flow[
    (investor_flow.buy_sell_indicator == "external_feed")
    | (investor_flow.buy_sell_indicator == "fundamental_value")
]
lineplot_price(
    tmp,
    5,
    2,
    21,
    28,
    "p3_prices_all_fundamentals",
    ["Fundamental", "Fundamental Extern"],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "24",
    "arrival_time",
    "price",
    "buy_sell_indicator",
    "Time of Arrival",
    "Price",
    "tab10",
)


# Round 1 + 2 / No External
tmp = investor_flow[(investor_flow.rounds == 1) | (investor_flow.rounds == 2)]
tmp = tmp[tmp.buy_sell_indicator != "external_feed"]
lineplot_price(
    tmp,
    1,
    2,
    14,
    6,
    "p3_prices_short",
    ["Fundamental", "Ask", "Bid"],
    [1, 2],
    "18",
    "arrival_time",
    "price",
    "buy_sell_indicator",
    "Time of Arrival",
    "Price",
    "tab10",
)


# Round 1 + 2 / Fundamentals
tmp = investor_flow[(investor_flow.rounds == 1) | (investor_flow.rounds == 2)]
tmp = tmp[
    (tmp.buy_sell_indicator == "external_feed")
    | (tmp.buy_sell_indicator == "fundamental_value")
]
lineplot_price(
    tmp,
    1,
    2,
    14,
    6,
    "p3_prices_short_fundamentals",
    ["Fundamental", "Fundamental Extern"],
    [1, 2],
    "18",
    "arrival_time",
    "price",
    "buy_sell_indicator",
    "Time of Arrival",
    "Price",
    "tab10",
)


# Stress 4 + 8 / No External
tmp = investor_flow[(investor_flow.rounds == 4) | (investor_flow.rounds == 8)]
tmp = tmp[tmp.buy_sell_indicator != "external_feed"]
lineplot_price(
    tmp,
    1,
    2,
    14,
    6,
    "p3_prices_short_stress",
    ["Fundamental", "Ask", "Bid"],
    [4, 8],
    "18",
    "arrival_time",
    "price",
    "buy_sell_indicator",
    "Time of Arrival",
    "Price",
    "tab10",
)


# Stress 4 + 8 / Fundamentals
tmp = investor_flow[(investor_flow.rounds == 4) | (investor_flow.rounds == 8)]
tmp = tmp[
    (tmp.buy_sell_indicator == "external_feed")
    | (tmp.buy_sell_indicator == "fundamental_value")
]
lineplot_price(
    tmp,
    1,
    2,
    14,
    6,
    "p3_prices_short_stress_fundamentals",
    ["Fundamental", "Fundamental Extern"],
    [4, 8],
    "18",
    "arrival_time",
    "price",
    "buy_sell_indicator",
    "Time of Arrival",
    "Price",
    "tab10",
)


# Possible Color Schemes:
#   gist_heat


# Print the time passed and delte unwanted variables
print("--- %s seconds ---" % (time.time() - start_time))
del i
del f
del tmp
del tmp_help
del table
del start_time
del path
del files
