# univariate ANN for scenario 1
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.metrics import mean_squared_error
import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
 
# split a nultivariate sequence into samples
def to_supervised(train, n_input, n_out=1):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, :]
            x_input = x_input.reshape((len(x_input), train.shape[2]))
            X.append(x_input)
            y.append(data[in_end:out_end, -1])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)
    



# split a univariate dataset into train/test sets
def split_dataset(data, n_test):
    # split into standard weeks
    train, test = data[:-n_test, :], data[-n_test:, :] 
    # restructure into windows of weekly data 
    train = np.array(np.split(train, len(train)/12)) 
    test = np.array(np.split(test, len(test)/12)) 
    return train, test


#create base single layer
def single_neuron(w_size, model, n_units, n_drop):
    model.add(Dense(n_units, input_dim = 25))
    #keras.layers.LeakyReLU(alpha=0.3)
    model.add(Dropout(n_drop))
    return model


# make a single step forecast
def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:]
    # reshape data
    input_x = input_x.reshape((1, 25))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

#function to run each experimental run
def experiment(train, test, config):
    # unpack config
    w_size, n_layers, n_units, n_epochs, n_dropout = config
    # split into samples and targets
    trainX, trainy = to_supervised(train, w_size)
    #flatten trainX
    trainX = trainX.reshape(trainX.shape[0], (trainX.shape[1]*trainX.shape[2]))
    # define model
    model = Sequential()
    #add hidden layers
    for i in range(n_layers+1):
        model = single_neuron(w_size, model, n_units, n_dropout)
    #specify output
    model.add(Dense(1))
    model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3), loss='mse')

    # fit model
    model.fit(trainX, trainy, epochs=n_epochs, verbose=2, validation_split=0.1)
    # predict and evaluate
    #take previous observations
    history = [x for x in train[-100:,]]
    # walk-forward validation over each time step
    predictions = list()
    for i in range(len(test)):
        # predict the time step
        yhat_sequence = forecast(model, history, w_size)
        # store the predictions
        predictions.append(yhat_sequence)
        if i >12:
            for s in range(12):
                dat = test[(i-s)]
                history.append(dat)
            i=1
        i+=1
    # evaluate predictions days for each timestep
    predictions = np.array(predictions)
    print(predictions.shape, test.shape)
    test = test[:, :, -1]
    rmse = np.sqrt(mean_squared_error(test, predictions[:, :, -1]))
    print('Config: '+str(config))
    print('The RMSE for Model with '+str(n_layers)+' hidden layers is: '+str(rmse))
    return rmse



# create a list of configs to try
def model_configs():
    # define scope of configs
    n_input = [5]
    n_layers = [8]
    n_units = [16]
    n_epochs = [1]
    n_dropout = [0, 0.1]
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
    pd.DataFrame(result).to_csv('../Data/scores_scenario2.csv')
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
dataset  = pd.read_csv('../Data/Scenario3.csv', header=0, index_col=0)
# split into train and test
dataset = dataset.iloc[:-1, :]
train, test = split_dataset(dataset.values, 5376)
train, trainy = train[:, :, :-1], train[:, :, -1]
test, testy = test[:, :, :-1], test[:, :, -1]
# standardize the data
mu = np.mean(train)
sig = np.std(train)

dataTrainStandardized = (train - mu) / sig

test_res = (test - mu) / sig

train = pd.DataFrame(dataTrainStandardized.reshape((dataTrainStandardized.shape[0]*dataTrainStandardized.shape[1]), dataTrainStandardized.shape[2]))
train['y'] = pd.DataFrame(trainy.reshape(trainy.shape[0]*trainy.shape[1], 1))

test = pd.DataFrame(test_res.reshape((test_res.shape[0]*test_res.shape[1]), test_res.shape[2]))
test['y'] = pd.DataFrame(testy.reshape(testy.shape[0]*testy.shape[1], 1))

#convert to DF
train = np.array(train) 
train = train.reshape(train.shape[0]//12, 12, train.shape[1])
test = np.array(test) 
test = test.reshape(test.shape[0]//12, 12, test.shape[1])
    

del trainy, testy, dataTrainStandardized, test_res



# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(cfg_list) 
scores = pd.DataFrame(scores, columns=['Config', 'RMSE'])
print('done')
# list top 10 configs
print(scores.iloc[:10, :])
