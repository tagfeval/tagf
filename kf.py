from __future__ import division
from pandas import read_csv
from math import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from sklearn.metrics import mean_squared_error, mean_absolute_error




data  = read_csv('../Data/Scenario2.csv', header=0, index_col=0)



#update function
def update(mean1, var1, mean2, var2):
    new_mean = (var2*mean1 +var1*mean2)/(var2 + var1)
    new_var = 1/(1/var2 +1/var1)
    return [new_mean, new_var]

#flow update/predict function
def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

flow = data.iloc[:5000,-1]
measurements = data.iloc[:5000,:-1]
measurements = np.array(measurements)
flow = np.array(flow)
measurement_sig = 0.1
flow_sig = 0.3
mu = 0
sig = 1000
pred = list()
for n in range(len(measurements)):
    mu ,sig = update(mu, sig, measurements[n], measurement_sig)
    print('Update: [{}, {}]'.format(mu, sig))
    mu, sig = predict(mu, sig, flow[n], flow_sig)
    print('Predict: [{}, {}]'.format(mu, sig))
    pred.append(mu[1])
    
print('\n')
print('Final result: [{}, {}]'.format(mu, sig))

plt.figure(figsize = (16, 8))
plt.plot(flow[-500:])
plt.plot(pred[-500:])
plt.xlabel('# TimeStep', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Vol', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['True Vol','Predicted Vol'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.show()
rmse = np.sqrt(mean_squared_error(flow, pred))
mae = mean_absolute_error(flow, pred)
print(rmse, mae)