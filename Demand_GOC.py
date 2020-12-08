import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt


Demand = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/Damand/Demand.csv', header=0))
GOC = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/GOC/GOC.csv'))
#rename DATE
GOC.rename(columns = {'DATE':'Date'}, inplace=True)


#change to datetime
Demand.Date = pd.to_datetime(Demand['Date']) + pd.to_timedelta(Demand.pop('Hour'), unit='H')
GOC.Date = pd.to_datetime(GOC['Date']) + pd.to_timedelta(GOC.pop('HOUR'), unit='H')
#print(GOC['Date'].tail())

#line up dates
begining_date = pd.to_datetime('2010-01-01')
ending_date = pd.to_datetime('2019-04-30')
Demand = Demand.loc[(Demand['Date'] > begining_date) & (Demand['Date'] < ending_date)]
GOC = GOC.loc[(GOC['Date'] > begining_date) & (GOC['Date'] < ending_date)]

#print(Demand['Date'].tail())

#set date as index
Demand = Demand.set_index('Date')
GOC = GOC.set_index('Date')
#remove Unnamed: 0 coloum
GOC.pop('Unnamed: 0')

#merge Demand and GOC dataframes
Demand_GOC = pd.merge(Demand,GOC, how='inner', left_index=True, right_index=True)
#print(Demand_GOC.tail())


Large_Demand_GOC = Demand_GOC[[i for i in Demand_GOC.columns if int(Demand_GOC[i].sum()) > 100]]

#print (Large_Demand_GOC)

#corelation coefficents for market demand
corr = Large_Demand_GOC.corrwith(Large_Demand_GOC['Market Demand'], method='spearman')

#the plants that correlated the most
pd.options.display.max_columns = 10
corr = corr.sort_values(ascending=False)
print(corr.head(20))
Large_Names = corr.index[0:20]
#print (Large_Names)
#LargeCorr_DF = GOC.index
Dem_GOC_high_corr = pd.DataFrame()
for i in Large_Names:
    Dem_GOC_high_corr[i] = Large_Demand_GOC[i]

#print(Dem_GOC_high_corr)
#Dem_GOC_high_corr.plot(y='Market Demand', use_index=True)
#plt.show()

#Dem_GOC_high_corr.to_csv('/home/donald/Dropbox/Energy_analysis/Damand/Dem_GOC_high_correlated.csv')

'''



#print (Demand.tail(10))
#corr.to_csv('/home/donald/Dropbox/Energy_analysis/GOC/market_demand_spearman.csv')





#print (GOC.shape)
#print (Demand.shape)
'''
