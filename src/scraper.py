from alpha_vantage.timeseries import TimeSeries
import api_keys

ts = TimeSeries(key=alphavantage_key, output_format = "pandas")

data = ts.get_intraday("GOOGL", interval = "60min", outputsize = "full")
