

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

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
def transform_raw_data(df, INTERVAL, EVAL_RANGE, PREDICT_RANGE, NO_CHANGE_THRESHOLD, 
                       TRAIN_RATIO, train_means=None, train_stds=None, CONTINUOUS_ONLY=False):
    MIN_HOUR_OF_DAY = 9.5 + EVAL_RANGE*INTERVAL/60 # minimum hour of the day needed to get continous data
    MAX_HOUR_OF_DAY = 16.5 - PREDICT_RANGE*INTERVAL/60 # max hour of day to have continous future data for a label 
    cur_data = []
    cur_targets = []
    time_stamps = []
    # each day corresponds to a time series of EVAL_RANGE elemetns
    last_continuous_index = None # contains index of transformed data of the last sample if it is exactly 1 time unit before the current --> reusesable for faster transformation
    
    for current_day in range(EVAL_RANGE-1, df.shape[0]-PREDICT_RANGE): # such that past EVAL_RANGE and future PREDICT_RANGE are available 
        # only transform and save a sample if continous past data is guaranteed
        cur_time = df.iloc[current_day]['Time']    
        cur_hour_of_day = cur_time.hour + cur_time.minute/60
        
        # if CONTINUOUS_ONLY=True, only transform & save continous data in a day
        if CONTINUOUS_ONLY and cur_hour_of_day < MIN_HOUR_OF_DAY or cur_hour_of_day > MAX_HOUR_OF_DAY:
            last_continuous_index = None
            continue
        time_stamps.append(cur_time)
        
        if last_continuous_index is None:   # there is a gap --> have to get data of all time units again
            time_series = [None]*EVAL_RANGE
            time_series_targets = [None]*EVAL_RANGE
            for past_day in range(0, EVAL_RANGE): # e.g: from 0 -> 9 
                past_day_ix = current_day - (EVAL_RANGE-1) + past_day # Go from the oldest sample to the current sample 
                past_day_data = df.iloc[past_day_ix][['Open', 'High', 'Low', 'Close', 'Volume']].values
                time_series[past_day] = past_day_data
                
                cur_avg = (df.iloc[past_day_ix]['Open'] + df.iloc[past_day_ix]['Close']) / 2
                next_avg = (df.iloc[past_day_ix+1]['Open'] + df.iloc[past_day_ix+1]['Close']) / 2
                
                target = get_target(cur_avg, next_avg, threshold=NO_CHANGE_THRESHOLD)
                time_series_targets[past_day] = target
            time_series = np.array(time_series)
            time_series_targets = np.array(time_series_targets)
                
        else:   # reuse EVAL_RANGE-1 time units from last sample (for faster transformation)
            time_series = cur_data[last_continuous_index][1:]
            time_series_targets = cur_targets[last_continuous_index][1:]
            
            cur_day_data = df.iloc[current_day][['Open', 'High', 'Low', 'Close', 'Volume']].values
            cur_avg = (df.iloc[current_day]['Open'] + df.iloc[current_day]['Close']) / 2
            next_avg = (df.iloc[current_day+1]['Open'] + df.iloc[current_day+1]['Close']) / 2
            target = get_target(cur_avg, next_avg, threshold=NO_CHANGE_THRESHOLD)
            time_series = np.concatenate((time_series, cur_day_data.reshape(1,len(cur_day_data))))
            time_series_targets = np.append(time_series_targets, target)
        
        
        # window normalization
        # time_series = np.array(time_series, dtype=np.float32)
        # normalized_time_series = (time_series - time_series.mean(axis=0))/time_series.std(axis=0)
        
        last_continuous_index = len(cur_data)
        cur_data.append(time_series)
        cur_targets.append(time_series_targets)
    
        
    ticker_data = np.array(cur_data, dtype=np.float32)
    ticker_targets = np.array(cur_targets)
    
    if TRAIN_RATIO is not None: # split data to train and test set
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
                'test': [scaled_test_ticker_data, test_ticker_targets],
                'train_means': train_data_means,
                'train_stds': train_data_stds,
                'time_stamps': time_stamps
                    }
    
    # normalize using input means and stds, used for testing on new data
    elif train_means is not None and train_stds is not None: 
        scaled_data = (ticker_data - train_means)/train_stds
        return (scaled_data, ticker_targets, time_stamps)


# def extract_last_step_labels(y_pred): # y_pred: output matrix of RNN. Output: 1D matrix of last step predictions
#     return np.array([np.argmax(pred[-1]) for pred in y_pred])

def get_last_step_predictions(model, X): # X: an input Tensor to RNN. Output: 1D matrix of last step predictions
    y_pred = model.predict(X)
    return np.array([np.argmax(pred) for pred in y_pred])

# same as get_last_step_predictions, but also return a second column, which contains probabilities
def get_last_step_predictions_with_confidence(model, X):
    y_pred = model.predict(X)
    labels = np.array([np.argmax(pred) for pred in y_pred])
    probabilities = np.array([pred[-1, np.argmax(pred)] for pred in y_pred])
    return np.c_[labels.reshape((-1,1)), probabilities.reshape((-1,1))]


# measure accuracy for only predictions with probablity >= confidence
def get_last_step_performance_based_on_confidence(model, X, y_true, conf_threshold=0, printout=True):
    X, y_true = X.copy(), y_true.copy()
    y_pred = get_last_step_predictions_with_confidence(model, X)
    y_true = y_true[:, -1] # last step only
    
    satisfied_indicies = y_pred[:,1] > conf_threshold
    y_pred = y_pred[satisfied_indicies, 0]
    y_true = y_true[satisfied_indicies]
    
    if y_pred.shape[0] == 0:
        print("No predictions have confidence >= {}%".format(conf_threshold))
        return
    
    accuracy = sum(y_pred==y_true)/y_pred.shape[0]
    num_all_labels = X.shape[0]
    num_filtered_labels = y_pred.shape[0]
    out_of_ratio = num_filtered_labels/num_all_labels
    num_0_label = np.count_nonzero(y_pred==0)
    num_1_label = np.count_nonzero(y_pred==1)
    num_2_label = np.count_nonzero(y_pred==2)
    
    if printout: 
        print("For predictions with confidence >= {}%, accuracy = {:.2f}".format(conf_threshold*100, accuracy))
        print("\t - {}/{} ({:.2f}%) of all predictions".format(num_filtered_labels, num_all_labels, out_of_ratio*100))
        print("\t - {}/{} ({:.2f}%) of which have labels of val 0 (down)".format(num_0_label,num_filtered_labels, num_0_label*100/num_filtered_labels))
        print("\t - {}/{} ({:.2f}%) of which have labels of val 1 (up)".format(num_1_label,num_filtered_labels, num_1_label*100/num_filtered_labels))
        print("\t - {}/{} ({:.2f}%) of which have labels of val 2 (same)".format(num_2_label,num_filtered_labels, num_2_label*100/num_filtered_labels))
        
        print("Performance:")
        print(classification_report(y_true, y_pred, labels=[0,1,2], target_names=["Down", "Up", "Approx. Same"]))
        

def evaluate_on_ticker(model, ticker, START_DATE, END_DATE, INTERVAL=5, 
                       EVAL_RANGE=24, PREDICT_RANGE=6, NO_CHANGE_THRESHOLD=0.1,
                       conf_thresholds=[0,90,95], printout=True):
    
    # for normalization
    train_means = np.load('../../data/transformed/{}min/{}_train_means.npy'.format(INTERVAL, ticker))
    train_stds = np.load('../../data/transformed/{}min/{}_train_stds.npy'.format(INTERVAL, ticker))
    
    # get stock data
    tickerData = yf.Ticker(ticker)
    df = tickerData.history(interval="{}m".format(INTERVAL),  start=START_DATE, end=END_DATE)
    df = df.reset_index().rename({'Datetime':'Time'}, axis=1)
    
    # transform
    transformed_data = transform_raw_data(df, INTERVAL, EVAL_RANGE, 
                                          PREDICT_RANGE, NO_CHANGE_THRESHOLD,
                                          TRAIN_RATIO=None, train_means=train_means,
                                          train_stds=train_stds)
    # evaluate
    data, targets, time_stamps = transformed_data
    for threshold in conf_thresholds:
        get_last_step_performance_based_on_confidence(model, data, targets, threshold, printout=printout)
    
    
    # return predictions with timestamp & confidence
    y_pred = get_last_step_predictions_with_confidence(model, data)
    result = pd.DataFrame(columns=['Time','Prediction', 'Actual', 'Confidence'])
    result['Time'] = time_stamps
    result[['Prediction','Confidence']] = y_pred
    result['Actual'] = targets[:,-1]
    return result

    