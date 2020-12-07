import pandas as pd
import re
import numpy as np


Demand = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/Damand/PUB_Demand.csv', header=3,))
GOC = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/GOC/GOC.csv'))
#rename DATE
GOC.rename(columns = {'DATE':'Date'}, inplace=True)


#change to datetime
Demand.Date = pd.to_datetime(Demand['Date']) + pd.to_timedelta(Demand.pop('Hour'), unit='H')
GOC.Date = pd.to_datetime(GOC['Date']) + pd.to_timedelta(GOC.pop('HOUR'), unit='H')


#line up dates
begining_date = pd.to_datetime('2010-01-01')
ending_date = pd.to_datetime('2019-04-30')
Demand = Demand.loc[(Demand['Date'] > begining_date) & (Demand['Date'] < ending_date)]
GOC = GOC.loc[(GOC['Date'] > begining_date) & (GOC['Date'] < ending_date)]

#set date as index
Demand = Demand.set_index('Date')
GOC = GOC.set_index('Date')
#remove Unnamed: 0 coloum
GOC.pop('Unnamed: 0')

#merge Demand and GOC dataframes
Demand_GOC = pd.merge(Demand,GOC, how='inner', left_index=True, right_index=True)
print(Demand_GOC.head(10))

Large_Demand_GOC = Demand_GOC[[i for i in Demand_GOC.columns if int(Demand_GOC[i].sum()) > 1000000]]

print (Large_Demand_GOC.head(10))

#pd.options.display.max_columns = 14
corr = Large_Demand_GOC.corrwith(Large_Demand_GOC['Market Demand'], method='spearman')

corr.to_csv('/home/donald/Dropbox/Energy_analysis/GOC/market_demand_spearman.csv')

print(corr)
'''

#print (Demand.tail(10))






#print (GOC.shape)
#print (Demand.shape)
'''
