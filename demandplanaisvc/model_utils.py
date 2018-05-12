#imports
import os
import gc
from math import sqrt
import numpy as np
import pandas as pd
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

import tensorflow as tf

import itertools

import matplotlib
import matplotlib.pyplot as plt

class TimeSeriesPredictorBase():
    """TimeSeriesPredictorBase class

    Base class for Time Series Predictor Models
    """

    def __init__(self):
        self.model_prefix = 'base_model'
        self.verbose = 0
        self.is_lstm = False
        self.use_time_steps = False
        self.params = {}
        self.params['epochs'] = 50
        self.params['look_back'] = 1
        self.params['batch_size'] = 4
        self.model_base_dir = ''
        self.skip_train_if_cached = False

    def model_key(self):
        model_key_str = self.model_prefix
        for key in self.params:
            model_key_str = model_key_str + '_' + key + '_' + str(self.params[key])
        return model_key_str

    def save_model(self):
        model_name = self.model_key()
        self.model.save('models/' +self.model_base_dir +'/' +model_name +'.h5')
        # Save history
        pickle.dump(self.history, open('models/' +self.model_base_dir +'/' +model_name +'.pkl', 'wb'))

    def load_model(self):
        model_name = self.model_key()
        self.model = load_model('models/' +self.model_base_dir +'/' +model_name +'.h5')
        # Load history
        self.history = pickle.load(open('models/' +self.model_base_dir +'/' +model_name +'.pkl', 'rb'))

    def load_model2(self,model):
        self.model = model

    def model_exists(self):
        model_name = self.model_key()
        return os.path.isfile('models/' +self.model_base_dir +'/' +model_name +'.h5')

    def prepare_time_series_data(self ,data):
        x = []
        y = []
        for i in range(len(data ) -self.params['look_back']):
            x.append(data[i:(i + self.params['look_back']), 0])
            y.append(data[i + self.params['look_back'], 0])
        x = np.array(x)
        y = np.array(y)
        if self.is_lstm == False:
            return x ,y
        else:
            # Reshape for time steps
            if self.use_time_steps == False:
                x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
            else:
                x = np.reshape(x, (x.shape[0], x.shape[1], 1))
            return x ,y

    def normalize_data(self ,input_data):
        data = np.copy(input_data)
        # calculate the mean on data set
        mean = data.mean(axis=0)
        # subtract it from ALL OF THE DATA
        data -= mean
        # calculate the standard deviation
        std = data.std(axis=0)
        # normalize data
        data /= std
        # save the mean and std
        self.mean = mean
        self.std = std
        return data

    def inverse_normalize_data(self ,input_data):
        data = np.copy(input_data)
        # Multiply by std
        data *= self.std
        # add the mean back to the data
        data += self.mean

        # remove negative values
        data[data < 0] = 0
        return data

    def test_train_split(self ,data ,train_pecent = 0.7):
        train_size = int(len(data) * train_pecent)
        test_size = len(data) - train_size
        train, test = data[0:train_size ,:], data[train_size:len(data) ,:]
        return train, test

class TimeSeriesPredictorMLP(TimeSeriesPredictorBase):
    """TimeSeriesPredictorMLP class

    Multi Layer Perceptron class for Time Series Predictor Models
    """

    def __init__(self, model_prefix='mlp_model'):
        TimeSeriesPredictorBase.__init__(self)
        self.model_prefix = model_prefix
        self.params['ip_units'] = 12
        self.params['hidden_units'] = [8]
        self.params['op_units'] = 1

    def model(self):
        self.model = Sequential()
        # Input
        self.model.add(Dense(self.params['ip_units'], input_dim=self.params['look_back'], activation='relu'))
        # Hidden
        for units in self.params['hidden_units']:
            self.model.add(Dense(units, activation='relu'))
        # Output
        self.model.add(Dense(self.params['op_units']))

        # Compile
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def train(self, train_x, train_y, test_x, test_y):
        return self.model.fit(train_x, train_y, epochs=self.params['epochs']
                              , validation_data=(test_x, test_y)
                              , batch_size=self.params['batch_size'], verbose=self.verbose, shuffle=False)

    def evaluate(self, x, y):
        # Model performance
        return self.model.evaluate(x, y, batch_size=self.params['batch_size'], verbose=self.verbose)

    def predict(self, data):
        return self.model.predict(data, batch_size=self.params['batch_size'])

class TimeSeriesPredictorLSTM(TimeSeriesPredictorBase):
    """TimeSeriesPredictorLSTM class

    LSTM class for Time Series Predictor Models
    """

    def __init__(self, model_prefix='lstm_model'):
        TimeSeriesPredictorBase.__init__(self)
        self.is_lstm = True
        self.is_stateful = False
        self.model_prefix = model_prefix
        self.params['stacked_units'] = [4]
        self.params['op_units'] = 1

    def model(self):
        self.model = Sequential()
        # Stacked
        if self.is_stateful == False:
            if self.use_time_steps == False:
                self.model.add(LSTM(self.params['stacked_units'][0], input_shape=(1, self.params['look_back'])))
            else:
                self.model.add(LSTM(self.params['stacked_units'][0], input_shape=(self.params['look_back'], 1)))
        else:
            stack = len(self.params['stacked_units'])
            if stack == 1:
                self.model.add(
                    LSTM(self.params['stacked_units'][0], batch_input_shape=(self.params['batch_size']
                                                                             , self.params['look_back'], 1)
                         , stateful=True))
            else:
                self.model.add(
                    LSTM(self.params['stacked_units'][0], batch_input_shape=(self.params['batch_size']
                                                                             , self.params['look_back'], 1)
                         , stateful=True
                         , return_sequences=True))
                for i in range(1, stack):
                    self.model.add(
                        LSTM(self.params['stacked_units'][i], batch_input_shape=(self.params['batch_size']
                                                                                 , self.params['look_back'], 1)
                             , stateful=True))

        # Output
        self.model.add(Dense(self.params['op_units']))

        # Compile
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def train(self, train_x, train_y, test_x, test_y):
        if self.is_stateful == False:
            return self.model.fit(train_x, train_y, epochs=self.params['epochs']
                                  , validation_data=(test_x, test_y)
                                  , batch_size=self.params['batch_size'], verbose=self.verbose, shuffle=False)
        else:
            history = None
            for i in range(self.params['epochs']):
                hist = self.model.fit(train_x, train_y, epochs=1
                                      , validation_data=(test_x, test_y)
                                      , batch_size=self.params['batch_size'], verbose=self.verbose,
                                      shuffle=False)
                # Save history manually
                if history == None:
                    history = hist
                else:
                    for key in history.history:
                        history.history[key].append(hist.history[key][0])

                # reset state
                self.model.reset_states()
            return history

    def evaluate(self, x, y):
        # Model performance
        return self.model.evaluate(x, y, batch_size=self.params['batch_size'], verbose=self.verbose)

    def predict(self, data):
        return self.model.predict(data, batch_size=self.params['batch_size'])

# Utility Methods
def build_train_evaluate_model(commodity_data, model, debug=True):
    ###### Data Preparation ######
    # 1 - Normalize
    data = model.normalize_data(commodity_data)
    # 2 - Test Train Split
    train, test = model.test_train_split(data, train_pecent=0.7)
    # 3 - Prepare time series data
    train_x, train_y = model.prepare_time_series_data(train)
    test_x, test_y = model.prepare_time_series_data(test)

    print(model.model_exists(), model.model_key())
    ## Check if model alreasdy trained
    if ((model.skip_train_if_cached != True) | (model.model_exists() == False)):

        if debug == True:
            print('Building Model from scratch...')
        ###### Model ######
        model.model()

        ###### Train ######
        history = model.train(train_x, train_y, test_x, test_y)
        model.history = history.history
    else:
        ###### Load Model ######
        if debug == True:
            print('Loading Model from file...')
        model.load_model()
        history = model.history

    ###### Evaluate ######
    train_metrics = model.evaluate(train_x, train_y)
    if debug == True:
        print('Train Score: %.2f MSE (%.4f RMSE)' % (train_metrics[0], sqrt(train_metrics[0])))
    test_metrics = model.evaluate(test_x, test_y)
    if debug == True:
        print('Test Score: %.2f MSE (%.4f RMSE)' % (test_metrics[0], sqrt(test_metrics[0])))

    ###### Predict ######
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)

    return {
        'train': train, 'test': test
        , 'train_pred': train_pred, 'test_pred': test_pred
        , 'history': history, 'test_metrics': test_metrics, 'train_metrics': train_metrics
    }


def predict_future_time_series(commodity_data, model, num_predict_points):
    ###### Data Preparation ######
    # Normalize
    data = model.normalize_data(commodity_data)

    for i in range(num_predict_points):
        data_len = len(data)
        # For prediction we will us the last "look_back"
        # number of points to predict for the next point
        look_back_data = data[data_len - (model.params['look_back'] + 1):len(data), :]

        # Prepare time series data
        data_x, _ = model.prepare_time_series_data(look_back_data)

        # Make prediction
        pred = model.predict(data_x)

        # Add the prediction to our data for the next prediction
        data = np.concatenate((data, pred), axis=0)

    data_len = len(data)
    return data[data_len - num_predict_points:data_len, :]


def plot_results(model, train, test, train_pred, test_pred, history, future_pred):
    # Plot actual data with predictions
    fig, axes = plt.subplots(2, 1, figsize=(20, 15))

    train_actuals = model.inverse_normalize_data(train)
    test_actuals = model.inverse_normalize_data(test)
    train_pred_actuals = model.inverse_normalize_data(train_pred)
    test_pred_actuals = model.inverse_normalize_data(test_pred)
    future_pred_actuals = model.inverse_normalize_data(future_pred)

    # Actuals
    axes[0].plot([x for x in train_actuals] + [None for i in future_pred_actuals], label='Train'
                 , marker='o', markersize=5, alpha=0.8, linestyle='-', linewidth=.5)
    axes[0].plot([None for i in train_actuals] + [x for x in test_actuals] + [None for i in future_pred_actuals]
                 , label='Test', marker='o', markersize=5, alpha=0.8, linestyle='-', linewidth=.5)
    # Predictions
    axes[0].plot([None for i in range(model.params['look_back'])]
                 + [x for x in train_pred_actuals] + [None for i in future_pred_actuals], label='Predicted'
                 , marker='o', markersize=5, alpha=0.8, linestyle='--', linewidth=.5)
    axes[0].plot([None for i in train_actuals]
                 + [None for i in range(model.params['look_back'])]
                 + [x for x in test_pred_actuals] + [None for i in future_pred_actuals], label='Predicted'
                 , marker='o', markersize=5, alpha=0.8, linestyle='--', linewidth=.5)

    # Future predictions
    axes[0].plot([None for i in train_actuals] + [None for i in test_actuals]
                 + [x for x in future_pred_actuals], label='Future'
                 , marker='o', markersize=5, alpha=0.8, linestyle='--', linewidth=.5)

    axes[0].legend()

    # Plot Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, model.params['epochs'] + 1)

    # Loss
    min_val_loss = epochs[np.argmin(val_loss)]
    axes[1].plot(epochs, loss, 'bo', label='Training loss')
    axes[1].plot(epochs, val_loss, 'b', label='Validation loss')
    axes[1].set_title('Training and validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    # Plot minimum loss
    axes[1].axvline(x=min_val_loss)
    print('Epoch for lowest validation loss:', min_val_loss)

    plt.show()