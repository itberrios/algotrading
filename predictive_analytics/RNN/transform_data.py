

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from functions import get_target, transform_raw_data
from sklearn.preprocessing import StandardScaler

# num_tickers = 100
# tickers_csv = pd.read_csv('tickers.csv')['Electronic Technology'].values
# tickers= tickers_csv[:num_tickers]

tickers = ['AAPL', 'GOOG', 'TSLA', 'QCOM']

EVAL_RANGE = 24
PREDICT_RANGE = 3
INTERVAL = 5 # minutes

NO_CHANGE_THRESHOLD = 0.1 # percentage of ticker's price
TRAIN_RATIO = 0.95

train_data = None
train_targets = None
test_data = None
test_targets = None

ticker_no = 1
for ticker in tickers:          
    print('Working on ticker {}/{}'.format(ticker_no, 4))
    ticker_no += 1
    
    # load data
    ticker_csv = '../../data/clean/{}min/{}_{}min.csv'.format(INTERVAL, ticker, INTERVAL)
    df = pd.read_csv(ticker_csv, 
                     dtype={'Open':np.float32, 'High':np.float32, 'Low':np.float32, 
                            'Close':np.float32, 'Volume':np.float32},
                     parse_dates=['Time']
                     )
    
    # reformat data to match input for RNN
    transformed_data = transform_raw_data(df, INTERVAL, EVAL_RANGE, PREDICT_RANGE, NO_CHANGE_THRESHOLD, TRAIN_RATIO)
    if transformed_data != None:
        train_ticker_data, train_ticker_targets = transformed_data['train']
        test_ticker_data, test_ticker_targets = transformed_data['test']
        train_means = transformed_data['train_means']
        train_stds = transformed_data['train_stds']
    else:
        continue     # penny stock or missing stock 
    
    # concatenate dataframes of different tickers
    if train_data is None:
        train_data = train_ticker_data
        train_targets = train_ticker_targets
        test_data = test_ticker_data
        test_targets = test_ticker_targets
    else:
        train_data = np.concatenate((train_data, train_ticker_data), axis=0)
        train_targets = np.concatenate((train_targets, train_ticker_targets), axis=0)
        test_data = np.concatenate((test_data, test_ticker_data), axis=0)
        test_targets = np.concatenate((test_targets, test_ticker_targets), axis=0)
        
    # save means and stds 
    np.save('../../data/transformed/{}min/{}_train_means.npy'.format(INTERVAL, ticker), train_means)
    np.save('../../data/transformed/{}min/{}_train_stds.npy'.format(INTERVAL, ticker), train_stds)

tot_train_labels = train_targets.shape[0]*train_targets.shape[1]        
train_0_perc = np.count_nonzero(train_targets == 0)*100/tot_train_labels
train_1_perc = np.count_nonzero(train_targets == 1)*100/tot_train_labels
train_2_perc = np.count_nonzero(train_targets == 2)*100/tot_train_labels
print("Train set has:")
print("{:.2f}% label 0, {:.2f}% label 1, {:.2f}% label 2".format(train_0_perc, train_1_perc, train_2_perc))

np.save('../../data/transformed/{}min/train_data.npy'.format(INTERVAL), train_data)
np.save('../../data/transformed/{}min/train_targets.npy'.format(INTERVAL), train_targets)
np.save('../../data/transformed/{}min/test_data.npy'.format(INTERVAL), test_data)
np.save('../../data/transformed/{}min/test_targets.npy'.format(INTERVAL), test_targets)



        
    

        
        

    
        




