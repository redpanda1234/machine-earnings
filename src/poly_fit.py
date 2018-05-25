#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pickle
import sys

import scraper
from k_means import *

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

def get_fetched_data(cache_data=False, use_cached_data=False, use_all_data=False):
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
    elif use_all_data:
        plot=False
        all_data_filename = "data/cached_snp_data.p"
        try:
            with open(all_data_filename) as all_data:
                print("Using cached data found in " + all_data_filename \
                        + "...")
                fetched_data = pickle.load(open(all_data_filename, "rb"))
        except FileNotFoundError:
            print("Cached data not found in " + all_data_filename + "...")
            print("Downloading data instead...")
            fetched_data = scraper.fetch_data(cache_data=cache_data)
    else:
        fetched_data = scraper.fetch_data(cache_data=cache_data)

    return fetched_data


def plot_coefficients(degree, sep_vec, symbol):
    """

    """
    # initialize the figure to plot onto.
    fig = plt.figure()

    # Get an array that we can plot things against.
    t = np.array(list(range(len(sep_vec[0]))))

    # If our degree is >= 2, we want to plot in 3d. So we make it
    # happen.
    if degree >= 2:
        ax = fig.add_subplot(111, projection="3d")
        line = ax.plot(sep_vec[0], sep_vec[1], sep_vec[2])
        ax.plot(sep_vec[0], sep_vec[1], sep_vec[2], "k<")

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


def plot_fourier(sep_vec):
    # Apply the n-dimensional fast fourier transform by columns,
    # so that we can do frequency analysis on the stuff.
    transformed = np.array([np.fft.fft(vec) for vec in sep_vec])

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


def poly_fit(data, window_size, degree=5, plot=False):
    """
    data should be pre-windowed
    """
    # Initialize empty array for holding the fast fourier transform
    total_fft_data = []

    print("==> Fitting to windows...")

    total_z_list = []

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
            for i in range(degree+1):
                sep_vec += [[]]
            # And then add the correct entry for each thing. Again,
            # we're really just transposing to get the columns to be
            # coefficients for the same degree.
            for vec in z_list:
                for i in range(degree+1):
                    sep_vec[i] += [vec[i]]
        else:
            # In the one-dimensional case, our work is pretty much
            # already done. We just wrap it in an extra list to make
            # the code below work in this case too.
            sep_vec = [z_list]


        # If we don't want to plot things, skip all of the steps
        # below.
        if plot:
            plot_coefficients(degree, sep_vec, symbol)
            plot_fourier(sep_vec)

        transformed = np.array([np.fft.fft(vec) for vec in sep_vec])
        # print(transformed)
        total_fft_data += [transformed]
        total_z_list += [(z_list, symbol)]

    # shapes = [x[0].shape[1] for x in total_fft_data]
    # max_x = max(shapes)

    # new_fft_data = []

    # for matrix, symbol in total_fft_data:
    #     zeros = np.zeros((degree, max_x))
    #     zeros[:matrix.shape[0],:matrix.shape[1]] = matrix
    #     new_fft_data += [(zeros, symbol)]

    # cov_list = [[[] for _ in range(len(total_fft_data))] for _ in\
    # range(len(total_fft_data))]

    # print("==> Calculating covariance data...")

    # for i in range(len(total_fft_data)):
    #     for j in range(len(total_fft_data)-i):
    #         cov_list[i][j] = cov_list[j][i] = (
    #             np.cov(total_fft_data[i][0], total_fft_data[j][0]),
    #             "Covariance of {0} and {1}".format(
    #                 total_fft_data[i][1], total_fft_data[j][1])
    #         )
    # print("Done.")

    # cov_list = np.array(cov_list)
    # det_list = np.zeros(cov_list.shape)
    # for i in range(cov_list.shape[0]):
    #     for j in range(cov_list.shape[0] - i):
    #         det_list[i][j] = det_list[j][i] = np.linalg.det(cov_list[i][j][0])

    # print(det_list)
    # print(total_fft_data)
    return np.array(total_fft_data), z_list



def main(degree=4, cache_data=False, use_cached_data=True,
         use_all_data=False, plot=True, windowed=True):
    """
    """
    fetched_data = get_fetched_data(
        cache_data=cache_data,
        use_cached_data=use_cached_data,
        use_all_data=use_all_data)

    # Slice data into windows
    data, window_size = scraper.slice_windows(fetched_data)
    fft_data, z_list = poly_fit(data, window_size)
    # freq_vecs = [fft_vec for fft_vec in fft_data]

    # num_freqs = 10
    # total_freqs_data = np.zeros((num_freqs, len(data), 6))
    # print("freq_stuff")
    # for freq_index in range(num_freqs):
    #     this_freq_vec = np.zeros((len(data), 6))
    #     for i, fft_vec in enumerate(fft_data):
    #         this_freq_vec[i] = fft_vec[:, freq_index]
    #     total_freqs_data[freq_index] = this_freq_vec
    # print("finished")
    total_freqs_data = np.zeros((6, len(data), fft_data[0].shape[1]))
    for v_ind, fft_vec in enumerate(fft_data):
        for c_ind, coeff_vec in enumerate(fft_vec):
            total_freqs_data[c_ind, v_ind] = np.abs(fft_vec[c_ind])


    # k_step = 10
    k_min = 1
    k_max = 50
    for w, freq_data in enumerate(total_freqs_data):
        print("working on coefficient {}".format(w))
        cost_k_list = []
        for k in progressbar.progressbar(range(k_min, k_max)):
            clusters, label, cost_list = k_means(freq_data, k)
            cost = cost_list[-1]
            cost_k_list.append(cost)
        opt_k = np.argmin(cost_k_list) + 1


        plt.plot(range(k_min, k_max), cost_k_list, 'g^')
        plt.plot(opt_k, min(cost_k_list), 'rD')

        plt.title('Cost vs Number of Clusters')
        plt.savefig('plots/kmeans_{0}_ks.png'.format(w), format='png')
        plt.close()

        X = freq_data
        clusters, label, cost_list = k_means(X, opt_k)
        # pt_cluster = clusters[label.flatten()]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        data_plot = ax.plot(X[:, 2], X[:, 50], X[:, 25], "bo", markersize=1)

        center_plot = plt.plot(clusters[:, 0], clusters[:, 1], "g^", markersize=1)


        # set up legend and save the plot to the current folder
        plt.legend((data_plot, center_plot), \
                   ('data', 'clusters'), loc = 'best')
        plt.title('Visualization with {} clusters'.format(opt_k))
        plt.show()
        plt.savefig('plots/kmeans_{0}_{1}.png'.format(w, opt_k), format='png')
        plt.close()



if __name__ == "__main__":
    cache_data = "--cache-data" in sys.argv[1:]
    use_cached_data = "--use-cached-data" in sys.argv[1:]
    use_all_data = "--use-all-data" in sys.argv[1:]
    main(cache_data=cache_data, use_cached_data=use_cached_data, use_all_data = use_all_data)
