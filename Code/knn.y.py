import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values


# define dataset
data = read_csv('../Data/Scenario1.csv', header=0, index_col=0, parse_dates=[0], infer_datetime_format=True)
#split into test and train
X = data.values
n_test = 1100
train, test = X[0:-n_test], X[-n_test:]

df = series_to_supervised(train, 5)
df_test = series_to_supervised(test, 5)
#get X and y
trainX, trainy = df[:, :-1], df[:, -1]
scaler = MinMaxScaler(feature_range=(0, 1))
testX, testy = df_test[:, :-1], df_test[:, -1]

trainX_scaled = scaler.fit_transform(trainX)
trainX_scaled = pd.DataFrame(trainX_scaled )

testX_scaled = scaler.fit_transform(testX)
testX_scaled = pd.DataFrame(testX_scaled)

#to store rmse values for different k
rmse_val = [] 
for K in range(6):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(trainX_scaled, trainy)  #fit the model
    pred=model.predict(testX_scaled) #make prediction on test set
    error = sqrt(mean_squared_error(testy,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()

model = neighbors.KNeighborsRegressor(n_neighbors=4562)
model.fit(trainX, trainy)
pred=model.predict(testX)
rmse = sqrt(mean_squared_error(testy,pred))
mae = mean_absolute_error(testy, pred)


plt.figure(figsize = (16, 8))
plt.plot(testy)
plt.plot(pred)
plt.xlabel('Time Step', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Traffic Flow (Vehicles)', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['True Flow','Predicted Flow'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.show()