import pandas as pd
import re
import numpy as np


Demand = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/Damand/PUB_Demand.csv', header=3))

Demand.Date = pd.to_datetime(Demand.Date)
print (Demand.head(10))
begining_date = pd.to_datetime('2010-01-01')
ending_date = pd.to_datetime('2019-04-30')
Demand = Demand.loc[(Demand['Date'] > begining_date) & (Demand['Date'] < ending_date)]


print (Demand.head(10))
print (Demand.dtypes)
