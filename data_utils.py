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


def get_trading_times(df):
    ''' Obtains a cleaned stock price DataFrame, with trading times ranging
        from 9:45 - 4:00
       '''
    
    # ensure that all trading times are sequential, pad missing data with NaNs
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='15min'))
       
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
    for i in range(n):
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


