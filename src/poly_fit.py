#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.signal import butter, lfilter, freqz

import numpy as np
import pickle
import sys

import scraper

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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
    print(args)
    args = list(args)
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

    fourier_space_data = []

    for windows, symbol in data:

        # Initialize empty list for polynomial coefficients
        z_list = []

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

        sep_vec = []

        if degree > 1:
            for i in range(degree):
                sep_vec += [[]]
            for vec in z_list:
                for i in range(degree):
                    sep_vec[i] += [vec[i]]
        else:
            sep_vec = [z_list]

        t = np.array(list(range(len(sep_vec[0]))))

        fig = plt.figure()

        if degree >= 3:
            ax = fig.add_subplot(111, projection="3d")
            line = ax.plot(sep_vec[0], sep_vec[1], sep_vec[2])
            ax.plot(sep_vec[0], sep_vec[1], sep_vec[2], "k<")

        elif degree == 2:
            ax = fig.add_subplot(111)
            line = ax.plot(sep_vec[0], sep_vec[1])
            ax.plot(sep_vec[0], sep_vec[1], "k<")

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

        plt.show()

        fig = plt.figure()

        fft_vec = []

        for vec in sep_vec:
            fft_vec += [np.fft.fft(vec)]

        freq = np.fft.fftfreq(t.shape[-1])

        # order = 6
        # fs = 30.0
        # cutoff = 3.667

        # print(freq)

        # W_b, W_a = butter_lowpass(cutoff, fs, order)

        # w, h = freqz(W_b, W_a, worN=8000)

        # plt.subplot(2, 1, 1)
        # plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        # plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        # plt.axvline(cutoff, color='k')
        # plt.xlim(0, 0.5*fs)
        # plt.title("Lowpass Filter Frequency Response")
        # plt.xlabel('Frequency [Hz]')
        # plt.grid()

        # N = len(ws)
        # T = 1.0
        # xf = np.linspace(0.0, 1.0/(2.0 * T), N//2)

        size = int(np.ceil(np.sqrt(len(sep_vec))))

        plot_vec = []
        line_vec = []

        for i, vec in enumerate(fft_vec):
            plot_vec += [fig.add_subplot(size, size, i+1)]
            line_vec += [plot_vec[i].plot(freq, vec.real, freq, vec.imag)]
            plt.setp(line_vec[i], linewidth=.5)

        # W_line = W_plot.plot(freq, np.abs(W))
        # X_line = X_plot.plot(freq, np.abs(X))
        # Y_line = Y_plot.plot(freq, np.abs(Y))
        # Z_line = Z_plot.plot(freq, np.abs(Z))

        # W_line = W_plot.plot(xf, 2.0/N * np.abs(ws[0:N//2]))
        # X_line = X_plot.plot(xf, 2.0/N * np.abs(xs[0:N//2]))
        # Y_line = Y_plot.plot(xf, 2.0/N * np.abs(ys[0:N//2]))
        # Z_line = Z_plot.plot(xf, 2.0/N * np.abs(zs[0:N//2]))

        # plt.setp(W_line, linewidth=.5, color='r')
        # plt.setp(X_line, linewidth=.5, color='b')
        # plt.setp(Y_line, linewidth=.5, color='g')
        # plt.setp(Z_line, linewidth=.5)

        plt.show()





if __name__ == "__main__":
    cache_data = "--cache-data" in sys.argv[1:]
    use_cached_data = "--use-cached-data" in sys.argv[1:]
    main(cache_data=cache_data, use_cached_data=use_cached_data)
