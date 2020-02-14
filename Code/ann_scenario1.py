# univariate ANN for scenario 1
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.metrics import mean_squared_error
 

'''
this code recreates Scenario 1 of the evaluation process. 

Grid search methodology is applied for hyper-parameter optimisation (credit Jason Brownlee 
from machinelearningmastery.com)

'''

# split the univariate timeseries using sliding window approach
def lookback(inputseq, w_size):
	X, y = list(), list()
	for i in range(len(inputseq)):
		# n_steps lookback to the sliding window
		look_back_x = i + w_size
		# check if sufficient samples remain
		if look_back_x > len(inputseq)-1:
			break
		# assemble input and output
		_x, _y = inputseq[i:look_back_x], inputseq[look_back_x]
		X.append(_x)
		y.append(_y)
	return np.array(X), np.array(y)
 
    
# function to split the univariate timeseries into train/test partitions
def split_data(data, data_split=0.2):
    data = np.array(data)
    #if number of observations specified as real-value number
    if data_split > 1:
        # split using observation number
        train, test = data[0:-data_split], data[-data_split:]        
    #else use data percentage
    else:
        data_index = round(len(data)*(1-data_split))
        train = data[0:data_index, :]
        test = data[data_index:, :]
    return train, test


#function to create base single layer neural network with dropout layer
def single_neuron(w_size, model, n_units, n_drop):
    model.add(Dense(n_units, input_dim = w_size, activation = 'relu'))
    model.add(Dropout(n_drop))
    return model


# make a single step forecast
def forecast(model, history, n_input):
    data = np.array(history)
    # retrieve last w_size observations for input data to serve as X
    w_size_x = data[-n_input:, :]
    # reshape data
    w_size_x  = w_size_x .reshape((1, n_input))
    # forecast the next timestep(s)
    yhat = model.predict(w_size_x , verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

#function to run each experimental setup
def experiment(train, test, config):
    # unpack config
    w_size, n_layers, n_units, n_epochs, n_dropout = config
    # split into samples and targets
    trainX, trainy = lookback(train, w_size)
    trainX = trainX.reshape(trainX.shape[0], (trainX.shape[1]*trainX.shape[2]))
    # define model
    model = Sequential()
    #add hidden layers
    for i in range(n_layers+1):
        model = single_neuron(w_size, model, n_units, n_dropout)
    #specify output
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(trainX, trainy, epochs=n_epochs, verbose=0, validation_split=0.2, batch_size=1000, shuffle=False)
    # predict and evaluate
    #take previous observations
    history = [x for x in train]
    # walk-forward validation over each time step
    predictions = list()
    for i in range(len(test)):
        # predict the time step
        yhat_sequence = forecast(model, history, w_size)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = np.mean(np.abs(test - predictions))
    print('MAE for Config: ', config, 'is ', mae)
    print('RMSE for Config: ', config, 'is ', rmse)
    return rmse



# create a list of configs to try
def model_configs():
    # define scope of configs
    n_input = [5,10]
    n_layers = [8]
    n_units = [16]
    n_epochs = [500]
    n_dropout = [0]
    # create configs
    configs = list()
    for a in n_input:
        for b in n_layers:
            for c in n_units:
                for d in n_epochs:
                    for e in n_dropout:
                        cfg = [a,b,c,d,e]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs)) 
    return configs


# grid search configs
def grid_search(cfg_list):
    # evaluate configs
    result = [repeat_evaluate(train, test, cfg) for cfg in cfg_list]
#    pd.DataFrame(result).to_csv('../Data/scores_scenario1.csv')
    # sort configs by error, asc
    result.sort(key=lambda tup: tup[1])
    return (result)


# score a model, return None on failure
def repeat_evaluate(train, test, config, n_repeats=3):
    # unpack config
    w_size, n_layers, n_units, n_epochs, n_dropout = config
    # convert config to a key
    key = str(config)
    print(key)
    # run the experiment n times
    scores = [experiment(train, test, config) for _ in range(n_repeats)] 
    # summarize score
    result = np.mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)



# define input sequence
data  = pd.read_csv('../Data/Scenario1.csv', header=0, index_col=0)
train, test = split_data(data, 1100)

# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(cfg_list) 
scores = pd.DataFrame(scores, columns=['Config', 'RMSE'])
print('done')
# list top 10 configs
print(scores.iloc[:10, :])
