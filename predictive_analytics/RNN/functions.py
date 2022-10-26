

import yfinance as yf
import numpy as np
import pandas as pd

# return a sample's label
def get_target(cur_avg, next_avg, threshold=0.05):
    abs_price_change = abs(next_avg-cur_avg)*100/cur_avg
    if abs_price_change <= threshold:
        target = 2   # price change is less than threshold (percent) --> neutral target
    elif next_avg > cur_avg:
        target = 1
    else: # lower
        target = 0 
    return target


# Reformat raw data from YFinance
def transform_data_from_ticker(ticker, START_DATE, END_DATE, RAW_INTERVAL, EVAL_RANGE, PREDICT_RANGE, NO_CHANGE_THRESHOLD, TRAIN_RATIO):
    ticker_csv = '../../data/clean/{}_15min.csv'.format(ticker)
    df = pd.read_csv(ticker_csv, dtype={'Open':np.float32, 'High':np.float32, 'Low':np.float32, 'Close':np.float32, 'Volume':np.float32})
    
    cur_data = []
    cur_targets = []
    # each day corresponds to a time series of EVAL_RANGE elemetns
    first_run = True
    for current_day in range(EVAL_RANGE-1, df.shape[0]-PREDICT_RANGE): # such that past EVAL_RANGE and future PREDICT_RANGE are available 
        if first_run:   # first run for this sub_df
            time_series = [None]*EVAL_RANGE
            time_series_targets = [None]*EVAL_RANGE
            first_run = False
            for past_day in range(0, EVAL_RANGE): # e.g: from 0 -> 9 
                past_day_ix = current_day - (EVAL_RANGE-1) + past_day # Go from the oldest day to the current day 
                past_day_data = df.iloc[past_day_ix][['Open', 'High', 'Low', 'Close', 'Volume']].values
                time_series[past_day] = past_day_data
                
                cur_avg = (df.iloc[past_day_ix]['Open'] + df.iloc[past_day_ix]['Close']) / 2
                next_avg = (df.iloc[past_day_ix+1]['Open'] + df.iloc[past_day_ix+1]['Close']) / 2
                
                target = get_target(cur_avg, next_avg, threshold=NO_CHANGE_THRESHOLD)
                time_series_targets[past_day] = target
                
            time_series = np.array(time_series)
            time_series_targets = np.array(time_series_targets)

        else:   # reuse 9 days from last instance
            cur_data_ix = current_day - (EVAL_RANGE-1)
            time_series = cur_data[cur_data_ix-1][1:]
            time_series_targets = cur_targets[cur_data_ix-1][1:]
            
            cur_day_data = df.iloc[current_day][['Open', 'High', 'Low', 'Close', 'Volume']].values
            cur_avg = (df.iloc[current_day]['Open'] + df.iloc[current_day]['Close']) / 2
            next_avg = (df.iloc[current_day+1]['Open'] + df.iloc[current_day+1]['Close']) / 2
            target = get_target(cur_avg, next_avg, threshold=NO_CHANGE_THRESHOLD)
            
            time_series = np.concatenate((time_series, cur_day_data.reshape(1,len(cur_day_data))))
            time_series_targets = np.append(time_series_targets, target)
        
        # window normalization
        # time_series = np.array(time_series, dtype=np.float32)
        # normalized_time_series = (time_series - time_series.mean(axis=0))/time_series.std(axis=0)
        
        cur_data.append(time_series)
        cur_targets.append(time_series_targets)
        
    ticker_data = np.array(cur_data, dtype=np.float32)
    ticker_targets = np.array(cur_targets)

    train_test_split_idx = int(ticker_data.shape[0]*TRAIN_RATIO)
    train_ticker_data, train_ticker_targets = ticker_data[:train_test_split_idx], ticker_targets[:train_test_split_idx]
    test_ticker_data, test_ticker_targets = ticker_data[train_test_split_idx:], ticker_targets[train_test_split_idx:]

    # Normalization, only use info from train set
    train_data_means = train_ticker_data.mean(axis=0)
    train_data_stds = train_ticker_data.std(axis=0)
    scaled_train_ticker_data = (train_ticker_data - train_data_means)/train_data_stds
    scaled_test_ticker_data = (test_ticker_data - train_data_means)/train_data_stds
    
    return {
            'train': [scaled_train_ticker_data, train_ticker_targets], 
            'test': [scaled_test_ticker_data, test_ticker_targets]
                }

# def extract_last_step_labels(y_pred): # y_pred: output matrix of RNN. Output: 1D matrix of last step predictions
#     return np.array([np.argmax(pred[-1]) for pred in y_pred])

def get_last_step_predictions(model, X): # X: an input Tensor to RNN. Output: 1D matrix of last step predictions
    y_pred = model.predict(X)
    return np.array([np.argmax(pred[-1]) for pred in y_pred])

# same as get_last_step_predictions, but also return a second column, which contains probabilities
def get_last_step_predictions_with_confidence(model, X):
    y_pred = model.predict(X)
    labels = np.array([np.argmax(pred[-1]) for pred in y_pred])
    probabilities = np.array([pred[-1,np.argmax(pred[-1])] for pred in y_pred])
    return np.c_[labels.reshape((-1,1)), probabilities.reshape((-1,1))]


# measure accuracy for only predictions with probablity >= confidence
def get_last_step_accuracy_based_on_confidence(model, X, y_true, conf_threshold=0.8):
    y_pred = get_last_step_predictions_with_confidence(model, X)
    y_true = y_true[:, -1] # last step only
    
    satisfied_indicies = y_pred[:,1] > conf_threshold
    y_pred = y_pred[satisfied_indicies, 0]
    y_true = y_true[satisfied_indicies]
    
    accuracy = sum(y_pred==y_true)*100/y_pred.shape[0]
    num_all_labels = X.shape[0]
    num_filtered_labels = y_pred.shape[0]
    out_of_ratio = num_filtered_labels/num_all_labels
    num_0_label = np.count_nonzero(y_pred==0)
    num_1_label = np.count_nonzero(y_pred==1)
    num_2_label = np.count_nonzero(y_pred==2)
    
    print("For predictions with confidence >= {}%, accuracy = {:.2f}".format(conf_threshold*100, accuracy))
    print("\t - {}/{} ({:.2f}%) of all predictions".format(num_filtered_labels, num_all_labels, out_of_ratio*100))
    print("\t - {}/{} ({:.2f}%) of which have labels of val 0 (down)".format(num_0_label,num_filtered_labels, num_0_label*100/num_filtered_labels))
    print("\t - {}/{} ({:.2f}%) of which have labels of val 1 (up)".format(num_1_label,num_filtered_labels, num_1_label*100/num_filtered_labels))
    print("\t - {}/{} ({:.2f}%) of which have labels of val 2 (same)".format(num_2_label,num_filtered_labels, num_2_label*100/num_filtered_labels))

def evaluate(model, X, y_true, binary=False):
    from sklearn.metrics import precision_score, recall_score
    y_pred = get_last_step_predictions(model, X)
    y_true = y_true[:, -1] # last step only
    acc = sum(y_pred==y_true)/y_true.shape[0]
    if not binary:
        for i in range(len(y_true)):
            y_pred[i] = y_pred[i] if y_pred[i] != 2 else 0 
            y_true[i] = y_true[i] if y_true[i] != 2 else 0 
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    

def evaluate_on_ticker(model, ticker):
    EVAL_RANGE = 30
    PREDICT_RANGE = 1
    START_DATE = '2020-7-30'
    END_DATE = '2020-8-2'
    NO_CHANGE_THRESHOLD = 0.1
    RAW_INTERVAL = "15m"
    transformed_data = transform_data_from_ticker(ticker, START_DATE, END_DATE, EVAL_RANGE, PREDICT_RANGE, NO_CHANGE_THRESHOLD)
    if transformed_data != None:
        cur_data, cur_targets = transformed_data
        print(cur_data.shape)
        y_p = model.predict(cur_data)
        evaluate(model, cur_targets, y_p, binary=False)
    else:
        print('Ticker not available or is penny stock')
    