''' 
    Main data pipeline
'''

import os
from glob import glob
import re
import numpy as np 
import pandas as pd

from data_utils import *

# helper for getting stock names
get_name_15 = lambda x : re.search('\w+(?=_15min)', x).group()
get_name_5 = lambda x : re.search('\w+(?=_5min)', x).group()
get_name_1 = lambda x : re.search('\w+(?=_1min)', x).group()


def add_encoded_timestamp(df):
    ''' Encodes temporal information into an interval format via sinusoids '''
    # add days, hours, and minutes to the dataset
    dayofweek = df.index.dayofweek
    hour = df.index.hour
    minute = df.index.minute

    # encode the days, hours, and minutes with sin and cos functions
    days_in_week = 7
    hours_in_day = 24
    minutes_in_hour = 60

    df['sin_day'] = np.sin(2*np.pi*dayofweek/days_in_week)
    df['cos_day'] = np.cos(2*np.pi*dayofweek/days_in_week)
    df['sin_hour'] = np.sin(2*np.pi*hour/hours_in_day)
    df['cos_hour'] = np.cos(2*np.pi*hour/hours_in_day)
    df['sin_minute'] = np.sin(2*np.pi*minute/minutes_in_hour)
    df['cos_minute'] = np.cos(2*np.pi*minute/minutes_in_hour)

    return df


def add_engineered_features(df):
    ''' Obtains Engineered features '''

    # get price differences
    # df['open_diff'] = df['Open'].diff()
    df['close_diff'] = df['Close'].diff()
    # df['high_diff'] = df['High'].diff()
    # df['low_diff'] = df['Low'].diff()
    df['log_vol_diff'] = df['log_volume']

    # possibly obtain other features

    # drop all NaNs
    df = df.dropna() 
    return df


def add_technical_indicators(df):
    return df

def get_stocks(data_paths, tgt_window=4, iqr_lim=0.25, encode_timestamp=True,
               engineer_features=True, tech_indicators=True):
    ''' Obtains base DataFrames for each stock '''

    # supress setting with copy warning
    # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    pd.options.mode.chained_assignment = None

    stock_dfs = {}
    for _path in data_paths:
        df = pd.read_csv(_path, index_col=0, parse_dates=True, 
                        infer_datetime_format=True).dropna()

        # add price_trend to each stock DataFrame
        df = get_numeric_price_trend(df, n=tgt_window)

        # use log volume instead of volume
        df['log_volume'] = np.log(df['Volume'])

        # add time encoding
        if encode_timestamp:
            df = add_encoded_timestamp(df)

        # add engineered features
        if engineer_features:
            df = add_engineered_features(df)

        # add technical indicators
        if tech_indicators:
            df = add_technical_indicators(df)

        # remove volume
        df = df.drop(columns=['Volume'])

        # get upper/lower thresholds for target variables
        lower, upper = get_iqr_thresholds(df['price_trend'], iqr_lim)

        df['price_change'] = 1 # price stays the same
        df['price_change'][df['price_trend'] < lower] = 0 # downward price movement
        df['price_change'][df['price_trend'] > upper] = 2 # upward prive movement

        # add stock to dict
        if '15min' in _path:
            stock_dfs.update({get_name_15(_path) : df})
        elif '5min' in _path:
            stock_dfs.update({get_name_5(_path) : df})
        elif '1min' in _path:
            stock_dfs.update({get_name_1(_path) : df})

    # return setting with copy warning to default 
    pd.options.mode.chained_assignment = 'warn'

    return stock_dfs


def get_split_dfs(stock_dfs, train_loc, valid_loc_1, valid_loc_2, test_loc):
    ''' Obtains Tran/Valid/Test DataFrames for each stock '''
    split_dfs = {}

    for name in stock_dfs.keys():

        # get stock DataFrame
        df = stock_dfs[name]

        # get splits
        train = df.loc[:train_loc]
        valid = df.loc[valid_loc_1:valid_loc_2]
        test = df.loc[test_loc:]
        
        split_dfs.update({name : [train, valid, test]})

    return split_dfs
