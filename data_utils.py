"""
Created on Sat Oct  1 14:07:43 2022

@author: itber

Basic Data Clening functions
"""

import os
from glob import glob
import re
import numpy as np 
import pandas as pd
import tensorflow as tf


def get_trading_times(df):
    ''' Obtains a cleaned stock price DataFrame, with trading times ranging
        from 9:45 - 16:00
       '''
    
    # ensure that all trading times are sequential, pad missing data with NaNs
    # df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='15min'))
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='5min'))
       
    # get regular trading times
    dayofweek = df.index.dayofweek
    hour = df.index.hour
    minute = df.index.minute
    
    df = df.iloc[(dayofweek <= 4)                  # only get M-F
                 & ~((hour == 9) & (minute < 45))  # remove less than 9:45
                 & ((hour >= 9) & (hour <= 16))    # hours 9-16
                 & ~((hour == 16) & (minute > 0))] # remove greater than 16:00
    
    # remove NaNs
    df = df[~df.isna().all(axis=1)]
    
    return df


def get_numeric_price_trend(df, n=4):
    ''' Obtains target variable for stock data 
        Discards the final n-1 rows.
        Inputs:
            df - Stock DataFrame
            n - range of target window to compute target variable
        Outputs:
            df - DataFrame with final n-1 rows removed with target variable inserted for each row
    '''
    # supress setting with copy warning
    # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    pd.options.mode.chained_assignment = None


    midpoints = []
    for i in range(1, n + 1):
        midpoints.append(df[['Open', 'Close']].shift(-i).mean(axis=1))

    midpoints = pd.concat(midpoints, axis=1) # .mean(axis=1)
    nan_locs = midpoints.isna().any(axis=1)

    # remvove NaNs
    df = df[~nan_locs]
    midpoints = midpoints[~nan_locs]

    # compute mean numeric price trend 
    price_trend = midpoints.mean(axis=1)

    # add targets to DataFrame
    df['price_trend'] = price_trend - df[['Open', 'Close']].mean(axis=1)

    # return setting with copy warning to default 
    pd.options.mode.chained_assignment = 'warn'

    return df


def get_iqr(ser):
    q1 = ser.quantile(0.25)
    q3 = ser.quantile(0.75)
    iqr = q3 - q1

    return q1, q3, iqr

def get_iqr_thresholds(x, lim=1.):
    q1, q3, iqr = get_iqr(x)

    lower = q1 - lim*iqr
    upper = q3 + lim*iqr

    return lower, upper

def get_percent_change_targets(price_trend, thresh=0.1):
    percent_change = price_trend.diff() / (np.abs(price_trend.shift(1)) + 1e-6)
    up = percent_change > thresh
    down = percent_change < -thresh

    return up, down


def mcc_metric(y_true, y_pred, num_classes=3, threshold=0.5):
    ''' Custom Mathew Correlation Coefficient for multiclass 
        For more details see: 
            "https://en.wikipedia.org/wiki/Phi_coefficient"
        Inputs: 
            y_true (tensor)
            y_pred (predicted class probabilities) (tensor)
            num_classes - number of classes
        Outputs:
            mcc - Mathews Correlation Coefficient
        '''
    # obtain predictions here, we can add in a threshold if we would like to
    y_pred = tf.argmax(y_pred, axis=-1)

    # cast to int64
    y_true = tf.squeeze(tf.cast(y_true, tf.int64), axis=-1)
    y_pred = tf.cast(y_pred, tf.int64)

    # total number of samples
    s = tf.size(y_true, out_type=tf.int64)

    # total number of correctly predicted labels
    c = s - tf.math.count_nonzero(y_true - y_pred)
    
    # number of times each class truely occured
    t = []

    # number of times each class was predicted
    p = []

    for k in range(num_classes):
        k = tf.cast(k, tf.int64)
        
        # number of times that the class truely occured
        t.append(tf.reduce_sum(tf.cast(tf.equal(k, y_true), tf.int32)))

        # number of times that the class was predicted
        p.append(tf.reduce_sum(tf.cast(tf.equal(k, y_pred), tf.int32)))


    t = tf.expand_dims(tf.stack(t), 0)
    p = tf.expand_dims(tf.stack(p), 0)

    s = tf.cast(s, tf.int32)
    c = tf.cast(c, tf.int32)
    
    num = tf.cast(c*s - tf.matmul(t, tf.transpose(p)), tf.float32)
    dem = tf.math.sqrt(tf.cast(s**2 - tf.matmul(p, tf.transpose(p)), tf.float32)) \
          * tf.math.sqrt(tf.cast(s**2 - tf.matmul(t, tf.transpose(t)), tf.float32))


    mcc = tf.divide(num, dem + 1e-6)

    return mcc
    


def numpy_mcc_metric(y_true, y_pred, num_classes=3, threshold=0.5):
    ''' Custom Mathew Correlation Coefficient for multiclass 
        For more details see: 
            "https://en.wikipedia.org/wiki/Phi_coefficient"
        Inputs: 
            y_true (tensor)
            y_pred (tensor)
            num_classes - number of classes
        Outputs:
            mcc - Mathews Correlation Coefficient
        '''
    # obtain predictions here, we can add in a threshold if we would like to
    y_pred = np.argmax(y_pred, axis=-1)

    # cast to int64
    # y_true = tf.squeeze(tf.cast(y_true, tf.int64), axis=-1)
    # y_pred = tf.cast(y_pred, tf.int64)

    # total number of samples
    s = tf.size(y_true, out_type=tf.int64)

    # total number of correctly predicted labels
    c = s - tf.math.count_nonzero(y_true - y_pred)
    
    # number of times each class truely occured
    t = []

    # number of times each class was predicted
    p = []

    for k in range(num_classes):
        k = tf.cast(k, tf.int64)
        
        # number of times that the class truely occured
        t.append(tf.reduce_sum(tf.cast(tf.equal(k, y_true), tf.int32)))

        # number of times that the class was predicted
        p.append(tf.reduce_sum(tf.cast(tf.equal(k, y_pred), tf.int32)))


    t = tf.expand_dims(tf.stack(t), 0)
    p = tf.expand_dims(tf.stack(p), 0)

    s = tf.cast(s, tf.int32)
    c = tf.cast(c, tf.int32)
    
    num = tf.cast(c*s - tf.matmul(t, tf.transpose(p)), tf.float32)
    dem = tf.math.sqrt(tf.cast(s**2 - tf.matmul(p, tf.transpose(p)), tf.float32)) \
          * tf.math.sqrt(tf.cast(s**2 - tf.matmul(t, tf.transpose(t)), tf.float32))


    mcc = tf.divide(num, dem + 1e-6)

    return mcc


def get_class_weights(dfs, name):
    class_counts = np.bincount(dfs[name][0].price_change)
    total = class_counts.sum()
    n_classes = len(class_counts)

    weights = []
    for idx, count in enumerate(class_counts):
        # compute balanced weights
        weights.append(total / (n_classes*count))

        # get inverse frequency class weighting
        # weights.append(1/np.power(count, 1))


    weights = np.array(weights) 
    # weights = weights / weights.sum()
    return weights