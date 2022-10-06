"""
Created on Sat Oct  1 14:07:43 2022

@author: itber

Basic Data Clening functions
"""

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


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