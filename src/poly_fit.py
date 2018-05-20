#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import sys

import scraper

def func(x, *args):
    """
    func gives a polynomial, to be used during the fitting procedure.

    Arguments:
        <x> -- (numeric) x is a numeric input, int or float, to be fed
        into our polynoial below.

    Positional Arguments:
        <*args> -- (list > float) *args is a list of floats to be used
        as coefficients in
    """
    result = 0.0
    args = list(*args)
    for i in range(len(args)):
        result += (args[i] * (x**i))

    return result

def main(degree=5, cache_data=False, use_cached_data=False):
    """
    main performs polynomial fits on windowed stock data (see
    scraper.py) and plots the coefficients in R^3.

    Arguments:
        <degree> -- (int) describes the degree of the polynomial to be fitted
        onto the data.

        <cache_data> -- (bool) describes whether the data should be cached when
        downloaded.

        <use_cached_data> -- (bool) describes whether cached data should be
        used.
    """

    # Get windowed data for S&P 500 stocks, together with the
    # window_size
    if use_cached_data:
        cached_data_filename = "data/cached_stock_data.p"
        try:
            with open(cached_data_filename) as cached_data:
                print("Using cached data found in " + cached_data_filename \
                        + "...")
                data, window_size = pickle.load(open(cached_data_filename, "rb"))
        except FileNotFoundError:
            print("Cached data not found in " + cached_data_filename + "...")
            print("Downloading data instead...")
            data, window_size = \
            scraper.slice_windows(scraper.fetch_data(), cache_data=cache_data)
    else:
        data, window_size = \
        scraper.slice_windows(scraper.fetch_data(), cache_data=cache_data)
        print

    fig = plt.figure()

    i = 1
    tot = len(data)
    side = np.ceil(np.sqrt(tot))

    for windows, symbol in data:

        # Initialize empty list for polynomial coefficients
        z_list = []

        for window in windows:
            x = np.array(list(range(window_size)))
            z_list.append(np.polyfit(x, window, degree))

            # Uncomment these to see what our fits are looking like
            # plt.plot(x, window, "ro")
            # smooth_x = np.linspace(0, scraper.windows_size, 200)
            # plt.plot(smooth_x, func(smooth_x, *z))
            # plt.show()
            # plt.close()

        xs, ys, zs = [], [], []

        for vec in z_list:
            xs += [vec[0]]
            ys += [vec[1]]
            zs += [vec[2]]


        ax = fig.add_subplot(side, side, i, projection="3d")
        ax.set_title("Polynomial coefficients for windowed fits of"
                     "{}".format(symbol))

        ax.plot(xs, ys, zs)

        i += 1

    plt.show()


if __name__ == "__main__":
    cache_data = "--cache-data" in sys.argv[1:]
    use_cached_data = "--use-cached-data" in sys.argv[1:]
    main(cache_data=cache_data, use_cached_data=use_cached_data)
