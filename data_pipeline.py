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
get_name = lambda x : re.search('\w+(?=_15min)', x).group()


def get_stocks(data_paths, tgt_window=4, iqr_lim=0.25):
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

        # add time encoding
        # TBD

        # add engineered features
        # TBD

        # add technical indicators
        # TBD

        # get upper/lower thresholds for target variables
        lower, upper = get_iqr_thresholds(df['price_trend'], iqr_lim)

        df['price_change'] = 1 # price stays the same
        df['price_change'][df['price_trend'] < lower] = 0 # downward price movement
        df['price_change'][df['price_trend'] > upper] = 2 # upward prive movement

        # add stock to dict
        stock_dfs.update({get_name(_path) : df})


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
