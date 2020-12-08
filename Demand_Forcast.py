import pandas as pd
import re
import numpy as np
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm


Dem_GOC_high_corr = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/Damand/Dem_GOC_high_correlated.csv', header=0))

Dem_GOC_high_corr['Date'] = pd.to_datetime(Dem_GOC_high_corr['Date'])

Demand = Dem_GOC_high_corr.set_index('Date')

#Demand.plot(y='Market Demand', use_index=True)
#plt.show()

#auto corelation of demand
#acf = pd.plotting.autocorrelation_plot(Demand['Market Demand'])
#acf.plot()
#plt.show()


#resample by month
y = Demand['Market Demand']
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Weekly')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Orders')
ax.legend();
plt.show()
#y = Dem_GOC_high_corr['Market Demand'].resample('M').mean()

def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

seasonal_decompose(y)
