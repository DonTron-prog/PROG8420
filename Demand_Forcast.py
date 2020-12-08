import pandas as pd
import re
import numpy as np
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


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
y = Demand['Market Demand'].resample('M').mean()
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Weekly')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Market Demand')
ax.legend();
plt.show()
#y = Dem_GOC_high_corr['Market Demand'].resample('M').mean()
'''
#sesonal decompostion
def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

seasonal_decompose(y)
'''
### plot for Rolling Statistic for testing Stationarity
def test_stationarity(timeseries, title):

    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label= title)
    ax.plot(rolmean, label='rolling mean');
    ax.plot(rolstd, label='rolling std (x10)');
    ax.legend()
    plt.show()

pd.options.display.float_format = '{:.8f}'.format
test_stationarity(y,'raw data')

# Augmented Dickey-Fuller Test
def ADF_test(timeseries, dataDesc):
    print(' > Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))

ADF_test(y,'raw data')

#Detrending
y_detrend =  (y - y.rolling(window=12).mean())/y.rolling(window=12).std()

test_stationarity(y_detrend,'de-trended data')
ADF_test(y_detrend,'de-trended data')

# Differencing
y_12lag =  y - y.shift(12)

test_stationarity(y_12lag,'12 lag differenced data')
ADF_test(y_12lag,'12 lag differenced data')


# Detrending + Differencing

y_12lag_detrend =  y_detrend - y_detrend.shift(12)

test_stationarity(y_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y_12lag_detrend,'12 lag differenced de-trended data')


y_to_train = y[:'2019-05-26'] # dataset to train
y_to_val = y['2019-06-02':] # last X months for test
predict_date = len(y) - len(y[:'2019-06-02']) # the number of data points for the test set
