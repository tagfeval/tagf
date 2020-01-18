import pandas as pd
import numpy as np
import datetime
from math import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error 
from matplotlib import pyplot
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler



# split a univariate dataset into train/test sets
def split_dataset(data, n_test):
    # split into standard weeks
    train, test = data[:-n_test, :], data[-n_test:, :] 
    # restructure into windows of weekly data 
    train = np.array(np.split(train, len(train)/12)) 
    test = np.array(np.split(test, len(test)/12)) 
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day 
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s=0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1])) 
    return score, scores
    

# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores]) 
    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=12):
    train = np.array(train)
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2])) 
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance 
        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1)) 
            X.append(x_input) 
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)



# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 2, 10, 128
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1] 
    # define model
    model = Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))) 
    model.add(Dropout(0.2))
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))) 
    model.add(Dropout(0.2))
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))) 
    model.add(Dropout(0.2))
    model.add(LSTM(256, activation='relu', return_sequences=False, input_shape=(n_timesteps, n_features))) 
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-5))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    checkpointer = ModelCheckpoint(filepath="../best_weights.hdf5", 
                               monitor = 'val_loss',
                               verbose=2, 
                               save_best_only=True)
    callbacks_list = [checkpointer, es] #early
   
    # fit network
    model.fit(train_x, train_y, epochs=epochs, callbacks=callbacks_list, batch_size=batch_size, verbose=verbose, validation_split=0.2)
    model.load_weights('../best_weights.hdf5')
    model.save('../model.h5')
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1)) # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


 #evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data 
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions) 
    return score, scores


# load the new file
dataset = pd.read_csv('../Data/Scenario3.csv', header=0, infer_datetime_format=True, parse_dates=['datDateTime'], index_col=['datDateTime'])
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
# evaluate model and get scores
n_input = 12
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['00:05', '00:10', '00:15', '00:20', '00:25', '00:30', '00:35', '00:40', '00:45', '00:50', '00:55', '00:00'] 
pyplot.plot(days, scores, marker='o', label='lstm') 
pyplot.show()

