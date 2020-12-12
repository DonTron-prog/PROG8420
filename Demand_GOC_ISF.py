import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime


Demand_GOC = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/Damand/Dem_GOC_high_correlated40.csv', header=0))
ISF = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/IntertieFlows/HourlyImportExportSchedules_2002-2017.csv', header=0))
#rename DATE



ISF.Date = pd.to_datetime(ISF['Date']) + pd.to_timedelta(ISF['Hour'], unit='H')
Demand_GOC.Date = pd.to_datetime(Demand_GOC['Date'])

print(ISF.dtypes)
print(ISF.head())
print(ISF.tail())

#line up dates
begining_date = pd.to_datetime('2010-01-01')
ending_date = pd.to_datetime('2019-04-30')
Demand_GOC = Demand_GOC.loc[(Demand_GOC['Date'] > begining_date) & (Demand_GOC['Date'] < ending_date)]
ISF = ISF.loc[(ISF['Date'] > begining_date) & (ISF['Date'] < ending_date)]

print(Demand_GOC['Date'].tail())

#set date as index
Demand_GOC = Demand_GOC.set_index('Date')
ISF = ISF.set_index('Date')
#remove Unnamed: 0 coloum
#ISF.pop('Unnamed')

#merge Demand and GOC dataframes
Demand_GOC = pd.merge(Demand_GOC,ISF, how='inner', left_index=True, right_index=True)
print(Demand_GOC.head())
print(Demand_GOC.tail())

Demand_GOC.plot(y='Exports', use_index=True)
plt.show()

Demand_GOC.to_csv('/home/donald/Dropbox/Energy_analysis/IntertieFlows/Dem_GOC_ISF.csv')

'''



#print (Demand.tail(10))
#corr.to_csv('/home/donald/Dropbox/Energy_analysis/GOC/market_demand_spearman.csv')





#print (GOC.shape)
#print (Demand.shape)
'''
