import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import backtrader as bt
from Strategies import *
from datetime import datetime

# import functions from RNN module
import sys
import os 
# setting path
this_folder = os.path.dirname(os.path.realpath('__file__'))
root_folder = os.path.dirname(current)
sys.path.append(root_folder)
from predictive_analytics.RNN.functions import transform_raw_data, last_step_accuracy, get_last_step_predictions_with_confidence


def change_time_format(timestamp): # so that backtrader logs are correct
    reg = "%Y-%m-%d %H:%M:%S"
    time_as_str = timestamp.strftime(reg)
    reformatted_time = datetime.strptime(time_as_str, reg)
    return reformatted_time


# Load model
MODEL_NAME = 'beta_1'
PATH_TO_MODEL = "../predictive_analytics/RNN/models/{}".format(MODEL_NAME)
model = tf.keras.models.load_model(PATH_TO_MODEL, custom_objects={'last_step_accuracy': last_step_accuracy})


# Parameters Setting
ticker = 'TSLA'
START_DATE = '2022-11-1'
END_DATE = '2022-11-20'

EVAL_RANGE=24
PREDICT_RANGE=3
INTERVAL = 5

# for normalization
train_means = np.load('../data/transformed/{}min/{}_train_means.npy'.format(INTERVAL, ticker))
train_stds = np.load('../data/transformed/{}min/{}_train_stds.npy'.format(INTERVAL, ticker))

# get stock data
tickerData = yf.Ticker(ticker)
df = tickerData.history(interval="{}m".format(INTERVAL),  start=START_DATE, end=END_DATE)
df = df.reset_index().rename({'Datetime':'Time'}, axis=1)

# transform
transformed_data = transform_raw_data(df, INTERVAL, EVAL_RANGE=24, 
                                      PREDICT_RANGE=3, NO_CHANGE_THRESHOLD=0.01,
                                      TRAIN_RATIO=None, train_means=train_means,
                                      train_stds=train_stds)
# predict
data, targets, time_stamps = transformed_data
y_pred = get_last_step_predictions_with_confidence(model, data)


# create data file for backtrade
df = df.loc[EVAL_RANGE-1:df.shape[0]-PREDICT_RANGE-1] # data points with labels
df.drop(['Dividends','Stock Splits'], axis=1, inplace=True) 
df.rename(columns = {'Open':'open','High':'high','Low':'low','Adj Close':'close','Volume':'volume',
                         }, inplace=True)
df[['prediction','confidence']] = y_pred
df['Time'] = df['Time'].apply(change_time_format)
df.set_index('Time', inplace=True)



# instantiate SignalData class
data = SignalData(dataname=df)
# instantiate Cerebro, add strategy, data, initial cash, commission 
cerebro = bt.Cerebro(stdstats = False, cheat_on_open=True)
cerebro.addstrategy(MyStrategy)
cerebro.adddata(data, name=ticker)
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0)
# cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

# run the backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
backtest_result = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
