import numpy as np
import pandas as pd
import api_keys
from alpha_vantage.timeseries import TimeSeries

import progressbar
import time

def fetch_data(interval="1min", num_stocks=5):
    """
    Fetch stock data from alphavantage api.

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

    # Get our api key for alphavantage
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
    # <datum>[1] contains a dictionary of information about the data.

    # When we take the .values, we get an array of values <x> with the
    # following structure:

    # x[0] : open
    # x[1] : high
    # x[2] : low
    # x[3] : close
    # x[4] : volume

    # Lastly, keep the symbol for the data with the data itself, so
    # that we can graph things easily later
    for sym in progressbar.progressbar(symbol_subset):
        data += [(ts.get_intraday(sym, interval=interval, outputsize =
                                  "full")[0].values, sym)]

    print("Data ingested successfully!")
    print("Total elapsed time {: 4.4f}".format(time.time() -
    start_time))
    return data


def slice_windows(data, window_size=7):
    """
    slice_windows converts time-series stock data into small windows.

    In order to perform polynomial fits on small windows of data, we
    need to first slice ingested data into said windows. slice_windows
    performs this task.

    Arguments:
        <data> -- (array_like) a list of time-series stock data,
        formatted as a numpy array structured like [<open_price>,
        <high_price>, <low_price>, <close_price>, <volume>].

        <window_size> -- (int) an integer describing how many points
        of stock data we  want each window to contain.

    Returns:
        <windowed_data> -- (array_like, int) a tuple containing
        (array_like), a list of the time-series stock data whose
        entries are arrays containing stock data that has been sliced
        into windows, and (int) <window_size>. Each array in
        (array_like) corresponds to one stock from <data>.
    """

    print("==> Began slicing data into windows...")

    start_time = time.time()

    # initialize windowed_data
    windowed_data = []

    for data_tup in progressbar.progressbar(data):

        data_array = data_tup[0]

        # Get average price over each interval
        avg_arr = np.array([(x[1] + x[2])/2 for x in data_array])

        # Find out how many extraneous data points we have n
        end_ind = avg_arr.shape[0] % window_size

        # Remove extra data so that the windows can be equally-sized
        avg_arr = avg_arr[:-end_ind]

        if avg_arr.shape[0] <= window_size:
            print("\n data from {0} was too small ({1}); threw it"
                  "out".format(data_tup[1], avg_arr.shape))
            continue

        # Get the windows
        windows = np.array(np.split(avg_arr, avg_arr.shape[0] /
                                    window_size))

        windowed_data.append((windows, data_tup[1]))

    print("Slicing successful.")
    print("Elapsed time{: 4.4f}".format(time.time() - start_time))
    return (windowed_data, window_size)
