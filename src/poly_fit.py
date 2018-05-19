import scraper
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def main(degree=4):
    """

    """

    # Get windowed data for S&P 500 stocks, together with the
    # window_size
    data, window_size = scraper.slice_windows(scraper.fetch_data())

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
            xs += [vec[1]]
            ys += [vec[2]]
            zs += [vec[3]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Polynomial coefficients for windowed fits of"
                     "{}".format(symbol))

        ax.plot(xs, ys, zs)

        plt.show()
        plt.clf()

if __name__ == '__main__':
    main()
