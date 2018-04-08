import numpy as np
import talib as tl
import pandas as pd

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('complete_data_set/AXISBANK.NS.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
print(data[1,:].head())

#utput = tl.SMA(data[1,], timeperiod=25)

#print(output)

