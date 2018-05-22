#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pickle
import sys

import scraper

import progressbar

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
    # print(args)
    args = list(args)
    for i in range(len(args)):
        result += (args[i] * (x**i))

    return result

def main(degree=3, cache_data=False, use_cached_data=False, plot=True):
    """
    main performs polynomial fits on windowed stock data (see
    scraper.py) and plots the coefficients in R^3.

    Arguments:
        <degree> -- (int) describes the degree of the polynomial to be
        fitted onto the data.

        <cache_data> -- (bool) describes whether the data should be
        cached when downloaded.

        <use_cached_data> -- (bool) describes whether cached data
        should be used.
    """

    # Get windowed data for S&P 500 stocks, together with the
    # window_size
    if use_cached_data:
        cached_data_filename = "data/cached_downloaded_data.p"
        try:
            with open(cached_data_filename) as cached_data:
                print("Using cached data found in " + cached_data_filename \
                        + "...")
                fetched_data = pickle.load(open(cached_data_filename, "rb"))
        except FileNotFoundError:
            print("Cached data not found in " + cached_data_filename + "...")
            print("Downloading data instead...")
            fetched_data = scraper.fetch_data(cache_data=cache_data)
    else:
        fetched_data = scraper.fetch_data(cache_data=cache_data)

    # Slice data into windows
    data, window_size = scraper.slice_windows(fetched_data)

    # Initialize empty array for holding the fast fourier transform
    total_fft_data = []

    print("==> Applying n-dimensional fast fourier transform...")

    transformed = []

    # Iterate through each of the windowed datasets, extracting the
    # stock symbol used to represent each.
    for windows, symbol in progressbar.progressbar(data):
        # print(len(windows))
        # Initialize empty list for vectors of polynomial coefficients
        # at each window
        z_list = []

        # Get the polynomial coefficients for each window
        for window in windows:
            x = np.array(list(range(window_size)))
            z = np.polyfit(x, window, degree)
            z_list.append(z)

            # Uncomment these to see what our fits are looking like
            # plt.plot(x, window, "ro")
            # smooth_x = np.linspace(0, window_size, 200)
            # p = np.poly1d(z)
            # plt.plot(smooth_x, p(smooth_x))
            # plt.show()
            # plt.close()

        # In sep vec, we just group all of the coefficients of a given
        # order together. E.g., all the coefficients for the constant
        # term, etc. into one single vector.
        sep_vec = []

        # If we have more than one coefficient, we want to group
        # coefficients of the same degree together. The code below
        # basically transposes z_list to achieve this.
        if degree > 1:
            # ...we want to group coefficients for the same degree
            # term into a single vector. So we initialize that
            for i in range(degree):
                sep_vec += [[]]
            # And then add the correct entry for each thing. Again,
            # we're really just transposing to get the columns to be
            # coefficients for the same degree.
            for vec in z_list:
                for i in range(degree):
                    sep_vec[i] += [vec[i]]
        else:
            # In the one-dimensional case, our work is pretty much
            # already done. We just wrap it in an extra list to make
            # the code below work in this case too.
            sep_vec = [z_list]

        # Apply the n-dimensional fast fourier transform by columns,
        # so that we can do frequency analysis on the stuff.
        transformed = np.array([np.fft.fft(vec) for vec in sep_vec])

        # Label it, and append it to the output list.
        total_fft_data += [(transformed, symbol)]

        # If we don't want to plot things, skip all of the steps
        # below.
        if not plot:
            continue

        # Else, we initialize the figure to plot onto.
        fig = plt.figure()

        # Get an array that we can plot things against.
        t = np.array(list(range(len(sep_vec[0]))))

        # If our degree is >= 2, we want to plot in 3d. So we make it
        # happen.
        if degree >= 2:
            ax = fig.add_subplot(111, projection="3d")
            # We plot the constant, linear, and quadratic coefficients since
            # they are more likely to give low frequency noise.
            line = ax.plot(sep_vec[-1], sep_vec[-2], sep_vec[-3])
            ax.plot(sep_vec[-1], sep_vec[-2], sep_vec[-3], "k<")

        # Else, we want to plot the two parameters against each other.
        elif degree == 1:
            ax = fig.add_subplot(111)
            line = ax.plot(sep_vec[1], sep_vec[0])
            ax.plot(sep_vec[1], sep_vec[0], "k<")

        # Else
        else:
            ax = fig.add_subplot(111)
            line = ax.plot(t, sep_vec[0])
            ax.plot(t, sep_vec[0], "k<")


        plt.setp(line, linewidth=.5, color='r')

        ax.set_title("Polynomial coefficients for windowed fits of "
                     "{}".format(symbol))
        plt.show()

        fig = plt.figure()


        size = int(np.ceil(np.sqrt(len(sep_vec))))

        plot_vec = []
        line_vec = []

        for i, vec in enumerate(sep_vec):
            plot_vec += [fig.add_subplot(size, size, i+1)]
            line_vec += [plot_vec[i].plot(t, vec)]
            plt.setp(line_vec[i], linewidth=.5)
            if i == 1:
                plot_vec[i].set_title(
                    "Time series data of polynomial coefficients for "
                    "windowed fits of {}".format(symbol))

        plt.show()

        fig = plt.figure()

        fft_vec = []

        for vec in sep_vec:
            fft_vec += [np.fft.fft(vec)]

        freq = np.fft.fftfreq(t.shape[-1])
        plot_vec = []
        line_vec = []

        for i, vec in enumerate(fft_vec):
            plot_vec += [fig.add_subplot(size, size, i+1)]
            line_vec += [plot_vec[i].plot(freq, np.abs(vec)/len(sep_vec[0]))]
            plt.setp(line_vec[i], linewidth=.5)


        plot_vec[1].set_title(
            "Time series data of fourier transform of polynomial "
            "coefficients for {}".format(symbol))

        plt.show()

        fig = plt.figure()

        plot_vec = []
        line_vec = []

        N = len(sep_vec[0])
        T = 1.0
        xf = np.linspace(0.0, 1.0/(2.0 * T), N//2)


        for i, vec in enumerate(fft_vec):
            plot_vec += [fig.add_subplot(size, size, i+1)]
            line_vec += [plot_vec[i].plot(xf, 2.0/N * np.abs(vec[0:N//2]))]
            plt.setp(line_vec[i], linewidth=.5)

        plt.show()

    shapes = [x[0].shape[1] for x in total_fft_data]
    max_x = max(shapes)

    new_fft_data = []

    for matrix, symbol in total_fft_data:
        zeros = np.zeros((degree, max_x))
        zeros[:matrix.shape[0],:matrix.shape[1]] = matrix
        new_fft_data += [(zeros, symbol)]

    cov_list = [[[] for _ in range(len(total_fft_data))] for _ in\
    range(len(total_fft_data))]

    print("==> Calculating covariance data...")

    for i in range(len(new_fft_data)):
        for j in range(len(new_fft_data)-i):
            cov_list[i][j] = cov_list[j][i] = (
                np.cov(new_fft_data[i][0], new_fft_data[j][0]),
                "Covariance of {0} and {1}".format(
                    new_fft_data[i][1], new_fft_data[j][1])
            )
    print("Done.")

    cov_list = np.array(cov_list)
    det_list = np.zeros(cov_list.shape)
    for i in range(cov_list.shape[0]):
        for j in range(cov_list.shape[0] - i):
            det_list[i][j] = det_list[j][i] = np.linalg.det(cov_list[i][j][0])

    return transformed

if __name__ == "__main__":
    cache_data = "--cache-data" in sys.argv[1:]
    use_cached_data = "--use-cached-data" in sys.argv[1:]
    transformed = main(cache_data=cache_data, use_cached_data=use_cached_data, plot=False)
