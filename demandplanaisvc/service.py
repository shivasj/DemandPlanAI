import json
import gc
from flask import Flask, jsonify,make_response,request
from flask_cors import CORS
import bson
import pandas as pd
from keras.models import load_model
from keras import backend as K
import traceback

import common
import appconfig as config
import model_utils

# Service
app = Flask('mfganalyticspysvc');
CORS(app);

logger = common.Logger()
settings = config.Settings()
keras_model = {}

# Item demand
commodity_demand_pattern_melt = pd.read_csv('datasets/commodity_demand_pattern_melt.csv')
def get_item_time_series_data(commodity):
    filtered_data = commodity_demand_pattern_melt.loc[commodity_demand_pattern_melt['commodity'] == commodity]
    return filtered_data.sort_values(by=['day'])[['qty']]

@app.route('/')
def root():
    return jsonify({'message':'Flask Service is running...'})

@app.route('/demand_history')
def demand_history():
    try:
        # Item demand
        demand_history = pd.read_csv('datasets/demand_history.csv')
        resp = make_response(demand_history.to_csv(index=False, encoding='utf-8'))
        resp.headers["Content-Disposition"] = "attachment; filename=demand_history.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    except Exception as e:
        logger.log("Error......")
        logger.log(e)

@app.route('/commodity')
def commodity():
    try:
        # Item demand
        commodity_demand_pattern_melt = pd.read_csv('datasets/commodity.csv')
        resp = make_response(commodity_demand_pattern_melt.to_csv(index=False, encoding='utf-8'))
        resp.headers["Content-Disposition"] = "attachment; filename=commodity.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    except Exception as e:
        logger.log("Error......")
        logger.log(e)

@app.route('/top_models')
def top_models():
    try:
        # Item demand
        top_models_all = pd.read_csv('datasets/top_models_all.csv')
        resp = make_response(top_models_all.to_csv(index=False, encoding='utf-8'))
        resp.headers["Content-Disposition"] = "attachment; filename=top_models_all.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    except Exception as e:
        logger.log("Error......")
        logger.log(e)

@app.route('/predict')
def predict():
    try:
        key = request.args.get("key");
        epochs = request.args.get("epochs");
        look_back = request.args.get("look_back");
        commodity = request.args.get("commodity");
        epochs = int(epochs)
        look_back = int(look_back)
        batch_size = key[key.find('batch_size')+11]
        commodity_df = get_item_time_series_data(commodity)
        commodity_data = commodity_df.values
        commodity_data = commodity_data.astype('float32')

        print(commodity,key, epochs, look_back,batch_size);

        if key.startswith('lstm_model'):
            # LSTM Regression model with lookback
            model = model_utils.TimeSeriesPredictorLSTM(model_prefix='lstm_model')
            model.skip_train_if_cached = False
            model.use_time_steps = False
        if key.startswith('mlp_model'):
            # Build MLP Model
            model = model_utils.TimeSeriesPredictorMLP(model_prefix='mlp_model')
            model.skip_train_if_cached = False
        if key.startswith('timestep_lstm_model'):
            # Build LSTM Model
            model = model_utils.TimeSeriesPredictorLSTM(model_prefix='timestep_lstm_model')
            model.use_time_steps = True
        if key.startswith('state_lstm_model'):
            # Build LSTM Model
            model = model_utils.TimeSeriesPredictorLSTM(model_prefix='state_lstm_model')
            model.use_time_steps = True
            model.is_stateful = True
            model.params['batch_size'] = 1


        model.params['look_back'] = look_back
        model.params['epochs'] = epochs

        if key not in keras_model:
            keras_model[key] = load_model('top_models/'+commodity+'/' + key + '.h5')
        model.load_model2(keras_model[key])

        # Make future predictions
        future_pred = model_utils.predict_future_time_series(commodity_data, model, 10)
        future_pred_actuals = model.inverse_normalize_data(future_pred)

        # Clear session
        K.clear_session()

        return jsonify({'future_pred_actuals': future_pred_actuals.tolist()})

    except Exception as e:
        logger.log("Error......")
        logger.log(e)
        logger.log(traceback.print_exc())

if __name__ == '__main__':
    logger.log("Starting Flask Service......")
    app.run(
        host=settings.host,
        port=settings.port,
        threaded=True,
        debug=False
    )