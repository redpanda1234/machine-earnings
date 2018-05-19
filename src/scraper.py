import numpy as np
from alpha_vantage.timeseries import TimeSeries

import api_keys

alphavantage_key = api_keys.alphavantage_key

ts = TimeSeries(key=alphavantage_key, output_format = "pandas")

data = ts.get_intraday("GOOGL", interval = "1min", outputsize =
                       "full")

# data is a tuple of two values
# data[0] contains the pandas dataframe
# data[1] contains the dictionary of information about the data

data_array = data[0].values

# each element x in np_array is an array with
# x[0] : open
# x[1] : high
# x[2] : low
# x[3] : close
# x[4] : volume

# Size of windows we'll use to split the data into subintervals
windows_size = 7

# Get average price over each interval
avg_arr = np.array([(x[1] + x[2])/2 for x in data_array])

# Find out how many extraneous data points we have n
end_ind = avg_arr.shape[0] % windows_size

# Remove extra data so that the windows can be equally-sized
avg_arr = avg_arr[:-end_ind]

# Get the windows
windows = np.array(np.split(avg_arr, avg_arr.shape[0] //\
                            windows_size))
