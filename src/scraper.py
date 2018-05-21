#!/usr/bin/env python3

from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import progressbar
import time

import api_keys

def fetch_data(interval="1min", num_stocks=10, cache_data=False):
    """
    Fetch stock data from the alphavantage API.

    Arguments:
        <interval> -- a string detailing how detailed we want the
        stock data we obtain to be, assuming an intraday timeseries.
        Can be "1min", "5min", "15min", "30min", or "60min".

        <num_stocks> -- an int specifying how many stocks should be
        fetched from alphavantage's api.

    Returns:
        <data> -- a list of time-series stock data for the constituent
        stocks of the S&P 500, with reporting frequency specified by
        <interval>.
    """
    print("==> Fetching data...")
    start_time = time.time()

    # Get our API key for alphavantage
    alphavantage_key = api_keys.alphavantage_key

    # Initialize the timeseries data object
    ts = TimeSeries(key=alphavantage_key, output_format="pandas")

    # Read in a csv of the data for all the S&P 500 stocks. Data
    # sourced from
    # https://github.com/datasets/s-and-p-500-companies
    names = pd.read_csv("data/S&P_500_stocks.csv", sep=",")

    # Slice the array to just get the symbols for the stocks, so that
    # we can iterate through pulling from the api
    symbols = names["Symbol"].values

    # Pulling all 500 isn't super desirable for our current testing
    # stuff, so let's just take a random subset of size <num_stocks>
    symbol_subset = np.random.choice(symbols, size=(num_stocks,))

    # Initialize data
    data = []

    # Iterate through the symbols, getting intraday stock info,
    # removing each from the pandas dataframe and converting to a data
    # array. We use progressbar to determine how far along it is.

    # Each call from the timeseries thing gets a tuple <datum> of two
    # values:
    # <datum>[0] contains a pandas dataframe
    # <datum>[1] contains a dictionary of information about the data

    # When we take the values, we get an array of values <x> with the
    # following structure:

    # x[0] : open
    # x[1] : high
    # x[2] : low
    # x[3] : close
    # x[4] : volume

    # Lastly, keep the symbol for the data with the data itself, so
    # that we can graph things easily later
    rows = [] # the number of rows for each stock data set
    for sym in progressbar.progressbar(symbol_subset):
        temp = ts.get_intraday(sym, interval=interval, outputsize="full")[0]
        row, col = temp.shape
        rows.append(row)
        data += [(ts.get_intraday(sym, interval=interval, outputsize =
                                  "full")[0], sym)]

    # Make all the data the same length
    new_data = []
    min_rows = min(rows)
    for dat, sym in data:
        new_data.append((dat[:min_rows], sym))
    data = new_data

    # Cut the data to ensure the data for each stock is the same
    # length
    for dat, sym in data:
        dat = dat.reset_index()

    print("Data ingested successfully!")
    print("Total elapsed time {: 4.4f}".format(time.time() -
        start_time))

    # Cache downloaded data to data directory
    if cache_data:
        cached_data_filename = "data/cached_downloaded_data.p"
        pickle.dump(data, open(cached_data_filename, "wb"))
        print("Cached downloaded stock data in " + cached_data_filename + ".")

    return data


def slice_windows(data, window_size=20, shift_size=1, normalize=True):
    """
    slice_windows converts time-series stock data into small windows.

    In order to perform polynomial fits on small windows of data, we
    need to first slice ingested data into said windows. slice_windows
    performs this task.

    Arguments:
        <data> -- (array_like) a list of time-series stock data,
        formatted as a numpy array structured like [<open_price>,
        <high_price>, <low_price>, <close_price>, <volume>].

        <window_size> -- (int) describes how many points of stock data we want
        each window to contain.

        <shift_size> -- (int) describes how much to shift with each next
        window.

        <cache_data> -- (bool) describes whether the downloaded data should be
        cached.

    Returns:
        <windowed_data> -- (array_like, int) a tuple containing
        (array_like), a list of the time-series stock data whose
        entries are arrays containing stock data that has been sliced
        into windows, and (int) <window_size>. Each array in
        (array_like) corresponds to one stock from <data>.
    """

    print("==> Began slicing data into windows...")

    start_time = time.time()

    # Convert pandas dataframes to numpy arrays
    data = [(x.values, y) for (x, y) in data]

    # initialize windowed_data
    windowed_data = []

    for data_tup in progressbar.progressbar(data):

        data_array = data_tup[0]

        # Each item in data_array represent the following:
        # x[0] : open
        # x[1] : high
        # x[2] : low
        # x[3] : close
        # x[4] : volume

        # Get average price over each interval
        avg_arr = [(x[1] + x[2])/2 for x in data_array]

        if normalize:
            tot_avg = sum(avg_arr) / (len(avg_arr))
            avg_arr /= tot_avg

        # Figure out how many windows there are
        end_idx = len(avg_arr) - window_size

        # Get the windows
        windows = []
        for idx in range(0, end_idx, shift_size):
            windows.append(avg_arr[idx:idx + window_size])

        windowed_data.append((np.array(windows), data_tup[1]))

    print("Slicing successful.")
    print("Elapsed time{: 4.4f}".format(time.time() - start_time))

    return (windowed_data, window_size)
