import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

import api_keys

ts = TimeSeries(key=alphavantage_key, output_format = "pandas")

data = ts.get_intraday("GOOGL", interval = "60min", outputsize =
                       "full")

# data is a tuple of two values
# data[0] contains the pandas dataframe
# data[1] contains the dictionary of information about the data

np_array = data[0].values

# each element x in np_array is an array with
# x[0] : open
# x[1] : high
# x[2] : low
# x[3] : close
# x[4] : volume
