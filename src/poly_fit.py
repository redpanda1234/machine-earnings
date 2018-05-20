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
    args = list(*args)
    for i in range(len(args)):
        result += (args[i] * (x**i))

    return result

def main(degree=5, cache_data=False, use_cached_data=False, normalize=False):
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



    fourier_space_data = []

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

        ws, xs, ys, zs = [], [], [], []

        for vec in z_list:
            ws += [vec[0]]
            xs += [vec[1]]
            ys += [vec[2]]
            zs += [vec[3]]

        t = np.array(list(range(len(xs))))

        # fig = plt.figure()
        # # ax = fig.add_subplot(side, side, i, projection="3d")
        # ax = fig.add_subplot(111, projection="3d")
        # ax.set_title("Polynomial coefficients for windowed fits of "
        #              "{}".format(symbol))

        # line = ax.plot(xs, ys, zs)
        # plt.setp(line, linewidth=.5, color='r')

        # ax.plot(xs, ys, zs, "k<")

        # plt.show()

        # fig = plt.figure()


        # w_plot = fig.add_subplot(221)
        # x_plot = fig.add_subplot(222)
        # y_plot = fig.add_subplot(223)
        # z_plot = fig.add_subplot(224)

        # w_line = w_plot.plot(t, ws)
        # x_line = x_plot.plot(t, xs)
        # y_line = y_plot.plot(t, ys)
        # z_line = z_plot.plot(t, zs)

        # plt.setp(w_line, linewidth=.5, color='r')
        # plt.setp(x_line, linewidth=.5, color='b')
        # plt.setp(y_line, linewidth=.5, color='g')
        # plt.setp(z_line, linewidth=.5)

        # plt.show()

        fig = plt.figure()

        W = np.fft.fft(ws)
        X = np.fft.fft(xs)
        Y = np.fft.fft(ys)
        Z = np.fft.fft(zs)

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

        N = len(ws)
        T = 1.0
        xf = np.linspace(0.0, 1.0/(2.0 * T), N//2)

        W_plot = fig.add_subplot(221)
        X_plot = fig.add_subplot(222)
        Y_plot = fig.add_subplot(223)
        Z_plot = fig.add_subplot(224)

        W_line = W_plot.plot(freq, W.real, freq, W.imag)
        X_line = X_plot.plot(freq, X.real, freq, X.imag)
        Y_line = Y_plot.plot(freq, Y.real, freq, Y.imag)
        Z_line = Z_plot.plot(freq, Z.real, freq, Z.imag)

        # W_line = W_plot.plot(freq, np.abs(W))
        # X_line = X_plot.plot(freq, np.abs(X))
        # Y_line = Y_plot.plot(freq, np.abs(Y))
        # Z_line = Z_plot.plot(freq, np.abs(Z))

        # W_line = W_plot.plot(xf, 2.0/N * np.abs(ws[0:N//2]))
        # X_line = X_plot.plot(xf, 2.0/N * np.abs(xs[0:N//2]))
        # Y_line = Y_plot.plot(xf, 2.0/N * np.abs(ys[0:N//2]))
        # Z_line = Z_plot.plot(xf, 2.0/N * np.abs(zs[0:N//2]))

        plt.setp(W_line, linewidth=.5, color='r')
        plt.setp(X_line, linewidth=.5, color='b')
        plt.setp(Y_line, linewidth=.5, color='g')
        plt.setp(Z_line, linewidth=.5)

        plt.show()




if __name__ == "__main__":
    cache_data = "--cache-data" in sys.argv[1:]
    use_cached_data = "--use-cached-data" in sys.argv[1:]
    main(cache_data=cache_data, use_cached_data=use_cached_data)
