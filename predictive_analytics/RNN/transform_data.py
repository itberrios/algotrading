

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from functions import get_target, transform_data_from_ticker
from sklearn.preprocessing import StandardScaler

# num_tickers = 100
# tickers_csv = pd.read_csv('tickers.csv')['Electronic Technology'].values
# tickers= tickers_csv[:num_tickers]

tickers = ['AAPL', 'GOOG', 'TSLA', 'QCOM']

EVAL_RANGE = 24
PREDICT_RANGE = 4
START_DATE = '2022-3-1'
END_DATE = '2022-4-22'
RAW_INTERVAL = "15m" # Interval to retrieve data from YFinance (can be 15m, 1h, 1d, etc.)
NO_CHANGE_THRESHOLD = 0.1 # percentage of ticker's price
TRAIN_RATIO = 0.80

train_data = []
train_targets = []
test_data = []
test_targets = []

ticker_no = 1
for ticker in tickers:          
    print('Working on ticker {}/{}'.format(ticker_no, 4))
    ticker_no += 1
    
    # reformat data to match input for RNN
    transformed_data = transform_data_from_ticker(ticker, START_DATE, END_DATE, RAW_INTERVAL, EVAL_RANGE, PREDICT_RANGE, NO_CHANGE_THRESHOLD, TRAIN_RATIO)
    if transformed_data != None:
        train_ticker_data, train_ticker_targets = transformed_data['train']
        test_ticker_data, test_ticker_targets = transformed_data['test']
    else:
        continue     # penny stock or missing stock 
    
    # concatenate dataframes of different tickers
    if len(train_data) == 0:
        train_data = train_ticker_data
        train_targets = train_ticker_targets
        test_data = test_ticker_data
        test_targets = test_ticker_targets
    else:
        train_data = np.concatenate((train_data, train_ticker_data), axis=0)
        train_targets = np.concatenate((train_targets, train_ticker_targets), axis=0)
        test_data = np.concatenate((test_data, test_ticker_data), axis=0)
        test_targets = np.concatenate((test_targets, test_ticker_targets), axis=0)

np.save('train_data.npy', train_data)
np.save('train_targets.npy', train_targets)
np.save('test_data.npy', test_data)
np.save('test_targets.npy', test_targets)



        
    

        
        

    
        




