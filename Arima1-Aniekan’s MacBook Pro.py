import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(font='IPAGothic')
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

#load data
series = pd.read_csv('../Data/Scenario1.csv', header=0, parse_dates=['datetime'], index_col='datetime', infer_datetime_format=True)
data = series.values
print(data.shape)
# data split
tr_start,tr_end = '2016-04-01 00:00:00','2016-04-28 23:55:00'
te_start,te_end = '2016-04-29 00:00:00','2016-04-30 23:55:00'
train = series['vol'][tr_start:tr_end].dropna()
test = series['vol'][te_start:te_end].dropna()
res = sm.tsa.seasonal_decompose(series.vol.dropna(),freq=365)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()
train = pd.DataFrame(train)

#dickey fuller test on train with ct component
res = sm.tsa.adfuller(train['vol'].dropna(),regression='ct')
print('p-value:{}'.format(res[1]))

#ADF-test(differenced-time-series)
res = sm.tsa.adfuller(train['vol'].diff().dropna(),regression='c')
print('p-value:{}'.format(res[1]))

#we use tra.diff()(differenced data), because this time series is unit root process.
fig,ax = plt.subplots(2,1,figsize=(20,10))
fig = sm.graphics.tsa.plot_acf(train.diff().dropna(), lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(train.diff().dropna(), lags=50, ax=ax[1])
plt.show()

resDiff = sm.tsa.arma_order_select_ic(train, max_ar=3, max_ma=7, ic='aic', trend='c')
print('ARMA(p,q) =',resDiff['aic_min_order'],'is the best.')

train = train.values
arima = sm.tsa.statespace.SARIMAX(train,order=(9,1,9),seasonal_order=(9,1,9,12), enforce_stationarity=False, enforce_invertibility=False).fit()
arima.summary()

res = arima.resid
#fig,ax = plt.subplots(2,1,figsize=(15,8))
#fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
#fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
#plt.show()

#predict and compute RMSE
pred = arima.predict(8065, 8640)
actual = np.array(test)
print('ARIMA model RMSE:{}'.format(np.sqrt(mean_squared_error(test,pred))))
plt.figure(figsize = (16, 8))
plt.plot(actual)
plt.plot(pred)
plt.xlabel('# Unit', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('RUL', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['True Vol (veh/hr)','Predicted Vol (veh/hr)'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.show()