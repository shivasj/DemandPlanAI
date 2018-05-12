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

import model_utils


def main():
    # Item demand
    commodity_demand_pattern_melt = pd.read_csv('datasets/commodity_demand_pattern_melt.csv')
    # commodity_demand_pattern_melt = pd.read_csv('datasets/commodity_demand_pattern_melt_irregular.csv')
    print('Shape:', commodity_demand_pattern_melt.shape)

    def get_item_time_series_data(commodity):
        filtered_data = commodity_demand_pattern_melt.loc[commodity_demand_pattern_melt['commodity'] == commodity]
        return filtered_data.sort_values(by=['day'])[['qty']]

    # Tuning to Maximize Model Performance
    performance_metric_list = []

    def store_performance_metric(model, test_metrics):
        metric = {
            'Model': model.model_prefix,
            'Type': model.model_base_dir,
            'key': model.model_key(),
            'epochs': model.params['epochs'],
            'look_back': model.params['look_back'],
            'batch_size': model.params['batch_size'],
            'Test MSE': test_metrics[0],
            'Test RMSE': sqrt(test_metrics[0])
        }
        if 'ip_units' in model.params:
            metric['ip_units'] = model.params['ip_units']
        else:
            metric['ip_units'] = 0
        if 'stacked_units' in model.params:
            metric['stacked_units'] = model.params['stacked_units']
        else:
            metric['stacked_units'] = 0
        performance_metric_list.append(metric)

    # Make model save directories for every commodity
    commodity_list = commodity_demand_pattern_melt.commodity.unique()
    # Set the commodity list to just top 2
    commodity_list = ['Nut','Supplement']
    for c in commodity_list:
        model_directory = 'models/' + c
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

    mlp_model = False
    lstm_model = False
    timestep_lstm_model = True
    state_lstm_model = False


    # MLP Model
    if mlp_model == True:
        # Various combinations of parameters
        params_combinations = {
            'look_back': [1, 10, 20, 30, 40, 50],
            'ip_units': [12, 24, 48],
            'epochs': [25, 50, 100, 200]
        }
        performance_metric_list = []

        for c in commodity_list:
            print(c)
            # Get data
            commodity_df = get_item_time_series_data(c)
            commodity_data = commodity_df.values
            commodity_data = commodity_data.astype('float32')

            params_list = []

            # Iterate though params
            for key in params_combinations:
                params_list.append(params_combinations[key])

            for params in itertools.product(*params_list):
                # Build MLP Model
                model = model_utils.TimeSeriesPredictorMLP(model_prefix='mlp_model')
                model.model_base_dir = c
                for index, key in enumerate(params_combinations):
                    model.params[key] = params[index]

                # Build, Train, Evaluate
                results = model_utils.build_train_evaluate_model(commodity_data, model)
                # Save model
                model.save_model()
                # Record performance
                store_performance_metric(model, results['test_metrics'])
                # clear memory
                gc.collect()

        performance_metric = pd.DataFrame(data=performance_metric_list)
        ### Write the dataframe to a csv file
        performance_metric.to_csv('datasets/performance_metric_mlp_model.csv', sep=',', index=False, encoding='utf-8')


    # LSTM Model
    if lstm_model == True:
        # Various cobinations of parameters
        params_combinations = {
            'look_back': [1, 10, 20, 30, 40, 50],
            'epochs': [50, 100, 200, 300, 600]
        }
        performance_metric_list = []

        for c in commodity_list:
            print(c)
            # Get data
            commodity_df = get_item_time_series_data(c)
            commodity_data = commodity_df.values
            commodity_data = commodity_data.astype('float32')

            params_list = []

            # Iterate though params
            for key in params_combinations:
                params_list.append(params_combinations[key])

            for params in itertools.product(*params_list):
                # Build LSTM Model
                model = model_utils.TimeSeriesPredictorLSTM(model_prefix='lstm_model')
                model.model_base_dir = c
                model.use_time_steps = False
                for index, key in enumerate(params_combinations):
                    model.params[key] = params[index]

                # Build, Train, Evaluate
                results = model_utils.build_train_evaluate_model(commodity_data, model)

                # Save model
                model.save_model()

                # Record performance
                store_performance_metric(model, results['test_metrics'])

                # clear memory
                gc.collect()

        performance_metric = pd.DataFrame(data=performance_metric_list)
        ### Write the dataframe to a csv file
        performance_metric.to_csv('datasets/performance_metric_lstm_model.csv', sep=',', index=False, encoding='utf-8')


    # LSTM Model with Timestep
    if timestep_lstm_model == True:
        # Various combinations for parameters
        params_combinations = {
            'look_back': [1, 10, 30, 50],
            'stacked_units':[4, 8,16],
            'epochs': [ 50, 100, 300]
        }
        performance_metric_list = []

        for c in commodity_list:
            print(c)
            # Get data
            commodity_df = get_item_time_series_data(c)
            commodity_data = commodity_df.values
            commodity_data = commodity_data.astype('float32')

            params_list = []
            # Iterate though params
            for key in params_combinations:
                params_list.append(params_combinations[key])

            for params in itertools.product(*params_list):
                # Build LSTM Model
                model = model_utils.TimeSeriesPredictorLSTM(model_prefix='timestep_lstm_model')
                model.model_base_dir = c
                model.use_time_steps = True
                for index, key in enumerate(params_combinations):
                    model.params[key] = params[index]

                # Build, Train, Evaluate
                results = model_utils.build_train_evaluate_model(commodity_data, model)
                # Save model
                model.save_model()
                # Record performance
                store_performance_metric(model, results['test_metrics'])
                # clear memory
                gc.collect()

        performance_metric = pd.DataFrame(data=performance_metric_list)
        ### Write the dataframe to a csv file
        performance_metric.to_csv('datasets/performance_metric_timestep_lstm_model.csv', sep=',', index=False, encoding='utf-8')


    # LSTM Model with State
    if state_lstm_model == True:
        # Various cobinations of parameters
        params_combinations = {
            'look_back': [1, 30, 50],
            'epochs': [100, 300, 600]
        }
        performance_metric_list = []

        for c in commodity_list:
            print(c)
            # Get data
            commodity_df = get_item_time_series_data(c)
            commodity_data = commodity_df.values
            commodity_data = commodity_data.astype('float32')

            params_list = []

            # Iterate though params
            for key in params_combinations:
                params_list.append(params_combinations[key])

            for params in itertools.product(*params_list):
                # Build LSTM Model
                model = model_utils.TimeSeriesPredictorLSTM(model_prefix='state_lstm_model')
                model.model_base_dir = c
                model.use_time_steps = True
                model.is_stateful = True
                model.params['batch_size'] = 1
                for index, key in enumerate(params_combinations):
                    model.params[key] = params[index]

                # Build, Train, Evaluate
                results = model_utils.build_train_evaluate_model(commodity_data, model)

                # Save model
                model.save_model()

                # Record performance
                store_performance_metric(model, results['test_metrics'])

                # clear memory
                gc.collect()

        performance_metric = pd.DataFrame(data=performance_metric_list)
        ### Write the dataframe to a csv file
        performance_metric.to_csv('datasets/performance_metric_state_lstm_model.csv', sep=',', index=False, encoding='utf-8')

    # # Top Performing Models
    # # Overall Metric Dataframe
    # performance_metric = pd.DataFrame(data=performance_metric_list)
    # ### Write the dataframe to a csv file
    # performance_metric.to_csv('datasets/performance_metric.csv', sep=',', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()