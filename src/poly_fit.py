#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import itertools
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

def binify(x):
    """
    put values into bins
    """
    if x > 7.5:
        return 3
    elif x > 4.5:
        return 2
    elif x > 1.5:
        return 1
    elif x > -1.5:
        return 0
    elif x > -4.5:
        return -1
    elif x > -7.5:
        return -2
    else:
        return -3

def slope_estimate(data, plot=False):
    """
    data should be pre-windowed
    """

    # Initialize slope data list. Each element of slope_data is a tuple of the
    # form (slopes, binned_slopes, sym).
    slope_data = []

    # Specifiy the number of subplots per plot.
    num_subplots = 4

    # Set up subplots for plotting data num_subplots stocks at a time.
    f, axarr = plt.subplots(2, num_subplots)
    # Counts how many plots we have so far.
    counter = 0
    # Determines whether there are some plots we have not shown yet after we
    # exit the loop.
    more_to_show = True

    # Iterate through each stock dataset.
    for dat, sym in data:

        # Initialize slope array for one stock. Both are numpy arrays of the
        # slopes of each window.
        slopes = []
        binned_slopes = []

        # Calculate the slope of each window.

        for i in dat:
            # Use the difference between the start and end points of each
            # window as an estimate of the slope in that window. The slope is
            # scaled by a factor of 1000.
            diff = i[-1] - i[0]
            diff *= 1000

            # Bin the slopes according to the follow scheme:
            # x > 7.5           ->   3
            # 4.5 < x < 7.5     ->   2
            # 1.5 < x < 4.5     ->   1
            # -1.5 < x < 1.5    ->   0
            # -4.5 < x < -1.5   ->  -1
            # -7.5 < x < -4.5   ->  -2
            # x < -7.5          ->  -3
            slopes.append(diff)
            binned_slopes.append(binify(diff))

        # Convert lists to numpy arrays.
        slopes = np.asarray(slopes)
        binned_slopes = np.asarray(binned_slopes)

        # Append data to slope_data list.
        slope_data.append((slopes, binned_slopes, sym))

        # Plot the data.

        # Plot the slope data and the binned slope data.
        axarr[0, counter].plot(range(1, len(slopes) + 1), slopes, 'b.')
        axarr[0, counter].set_title('Slopes of Windows of ' + sym)
        axarr[1, counter].hist(slopes, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        axarr[1, counter].set_title('Binned Slopes of Windows of ' + sym)
        axarr[1, counter].set_xlim([-4,4])

        # Determines whether there are some plots we have not shown yet after
        # we exit the loop.
        if not more_to_show:
            more_to_show = True

        # Determine whether we have 5 plots.
        counter = (counter + 1) % num_subplots
        if counter == 0 and plot:
            # If we have num_subplots plots, plot it and make a new plot for
            # new data.
            plt.show()
            f, axarr = plt.subplots(2, num_subplots)
            more_to_show = False

    if more_to_show and plot:
        plt.show()

    return slope_data

def threes(iterator):
    """
    Given an iterable object, iterate through it in a window of three items.
    """
    a, b, c = itertools.tee(iterator, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)

def generate_markov(slope_data):
    """
    Given slope data, returns a markov model used to predict future stock
    trends.  Each element of slope_data is a tuple of the form (slopes,
    binned_slopes, sym).
    """

    # Initialize the Markov dictionary.
    markov = {}

    for slopes, binned_slopes, sym in slope_data:
        # Iterate through slopes in windows of three items, i.e.
        # (x[0], x[1], x[2]), (x[1], x[2], x[3]), etc.
        for i, j, k in threes(binned_slopes):
            # Use the previous two slopes as the key and the next slope as the
            # value.
            try:
                markov[(i, j)].append(k)
            except KeyError:
                markov[(i, j)] = [k]

    return markov

def predict(markov, key):
    """
    returns a distribution of observed slopes based on recent data
    """
    dist = []
    try: # Needed in case we haven't seen this slope sequence before.
        for i in range(-3, 4):
            dist.append(markov[key].count(i))
    except KeyError:
        print("No data available for this series of slopes: " + str(key))

    # This array is of the form [c[-3], c[-2], ..., c[3]] where c[i] is the
    # number of times the slope i appeared in the Markov model.
    return dist

def suggest(dist):
    """
    returns a vector of probability distributions for whether the user should
    sell, do nothing, or buy
    """
    # Collapse the slope counts into probabilities of decisions by this scheme:
    # -3    ->  sell
    # -2    ->  sell
    # -1    ->  do nothing
    #  0    ->  do nothing
    #  1    ->  do nothing
    #  2    ->  buy
    #  3    ->  buy

    # Handle the case where no data was collected.
    if dist == []:
        print("Could not provide suggestions without data...")
        return [0] * 3

    counts = sum(dist)

    sell = (dist[0] + dist[1]) / counts
    do_nothing = (dist[2] + dist[3] + dist[4]) / counts
    buy = (dist[5] + dist[6]) / counts

    return [sell, do_nothing, buy]

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

def get_data():
    fetched_data = get_fetched_data(use_cached_data=True)
    data, window_size = scraper.slice_windows(fetched_data)
    return data, window_size

def give_probs(key):
    data, window_size = get_data()
    slope_data = slope_estimate(data)
    markov = generate_markov(slope_data)
    dist = predict(markov, key)
    vec = suggest(dist)
    return vec

def main(degree=4, cache_data=False, use_cached_data=False,
         use_all_data=True, plot=True, windowed=True):
    """
    """
    total_start = time.time()
    fetched_data = get_fetched_data(
        cache_data=cache_data,
        use_cached_data=use_cached_data,
        use_all_data=use_all_data)

    # Slice data into windows
    data, window_size = scraper.slice_windows(fetched_data)
    fft_data, z_list = poly_fit(data, window_size)
    # freq_vecs = [fft_vec for fft_vec in fft_data]

    symbol_dict = {i : data[i][1] for i in range(len(data))}

    total_freqs_data = np.zeros((6, len(data), fft_data[0].shape[1]))
    for v_ind, fft_vec in enumerate(fft_data):
        for c_ind, coeff_vec in enumerate(fft_vec):
            total_freqs_data[c_ind, v_ind] = np.abs(fft_vec[c_ind])

    k_min = 1000
    k_max = 1050
    label_list = []

    k_list = []
    for c_ind, freq_data in enumerate(total_freqs_data):
        # print("working on coefficient {}".format(c_ind))
        cost_k_list = []
        for k in range(k_min, k_max):
            if k % 5 == 0:
                print("k is {0} of {1} for coefficient {2}".format(k - k_min, k_max - k_min, c_ind))
            clusters, label, cost_list = k_means(freq_data, k)
            cost = cost_list[-1]
            cost_k_list.append(cost)
        opt_k = np.argmin(cost_k_list) + 1 + k_min

        plt.plot(range(k_min, k_max), cost_k_list, 'g^')
        plt.plot(opt_k, min(cost_k_list), 'rD')

        plt.title('Cost vs Number of Clusters')
        plt.savefig('plots/kmeans_{0}_ks.png'.format(c_ind), format='png')
        plt.close()

        X = freq_data
        clusters, label, cost_list = k_means(X, opt_k)
        label_list += [np.array(label)]
        k_list += [opt_k]
        # pt_cluster = clusters[label.flatten()]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # data_plot = ax.plot(X[:, 2], X[:, 50], X[:, 25], "bo", markersize=1)

        # center_plot = plt.plot(clusters[:, 0], clusters[:, 1], "g^", markersize=1)

        # # set up legend and save the plot to the current folder
        # plt.legend((data_plot, center_plot), ('data', 'clusters'), loc = 'best')
        # plt.title('Visualization with {} clusters'.format(opt_k))
        # # plt.show()
        # plt.savefig('plots/kmeans_{0}_{1}.png'.format(c_ind, opt_k), format='png')
        # plt.close()

    label_list = np.array(label_list)
    data_list = []
    for d_ind in range(label_list.shape[1]):
        datum = label_list[:, d_ind]
        out_vec = np.zeros((sum(k_list)))
        offset = 0
        for c_ind in range(len(datum)):
            out_vec[offset+datum[c_ind]] = 1
            offset += k_list[c_ind]
        data_list += [out_vec]

    data_list = np.array(data_list)
    cost_k_list = []
    k_min = 1000
    k_max = 1050
    for k in range(k_min, k_max):
        if k % 50 == 0:
            print("on {0} of {1}".format(k, k_max))
        clusters, label, cost_list = k_means(data_list, k)
        cost = cost_list[-1]
        cost_k_list.append(cost)
    opt_k = np.argmin(cost_k_list) + 1

    clusters, label, cost_list = k_means(X, opt_k)

    plt.plot(range(k_min, k_max), cost_k_list, 'g^')
    plt.plot(opt_k, min(cost_k_list), 'rD')

    plt.title('Cost vs Number of Clusters')
    plt.savefig('plots/kmeans_layer2_ks.png'.format(c_ind), format='png')
    plt.close()

    label_symbols = []
    # print(opt_k)
    total = 0
    for i in range(opt_k):
        ind_arr = np.where(label == i)
        ind_arr = ind_arr[0]
        if len(ind_arr) > 0:
            cluster_syms = []
            for ind in ind_arr:
                cluster_syms += [symbol_dict[ind]]
            label_symbols += [cluster_syms]
            total += len(cluster_syms)
    print(total)
    label_symbols = np.array(label_symbols)
    print(label_symbols)
    print("================================\n Total elapsed time: {}".format(time.time() - total_start))
    return label_symbols

# if __name__ == "__main__":
#     cache_data = "--cache-data" in sys.argv[1:]
#     use_cached_data = "--use-cached-data" in sys.argv[1:]
#     use_all_data = "--use-all-data" in sys.argv[1:]
#     main(cache_data=cache_data, use_cached_data=use_cached_data, use_all_data=use_all_data)
