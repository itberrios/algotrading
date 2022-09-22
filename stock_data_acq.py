"""
functions to obtain stock data via Alpha Vantage API 
"""



import os
import time
import csv
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import alpha_vantage
import matplotlib.pyplot as plt



API_KEY = 'K1GZTUTQ65SVGLNU' # ALpha Vantage API Key goes here
BASE_URL = 'https://www.alphavantage.co/query?'
FUNCTION = 'TIME_SERIES_INTRADAY_EXTENDED'  # for short term intervals


# =============================================================================
# functions

def get_stock_data(symbol, tslice, interval='15min', adjusted='true'):
    ''' Obtains Stock DataFrame for a single ticker symbol '''
    
    # get url for Alpha Vantage API
    csv_url = (f'{BASE_URL}function={FUNCTION}' 
               + f'&symbol={symbol}'
               + f'&slice={tslice.strip()}'
               + f'&interval={interval}'
               + f'&apikey={API_KEY}')

    # query Alpha Vantage and get _csv.reader object
    with requests.Session() as s:
        download = s.get(csv_url)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')

    # place _csv.read object into a DataFrame
    df = pd.DataFrame(list(cr))
    df.columns = df.iloc[0, :]
    df.drop(axis=0, index=0, inplace=True)
    df.index = df.time
    df.drop(axis=1, columns=['time'], inplace=True)
    df = df.astype(np.float64).sort_index()

    return df


def get_multi_periods(symbol, interval='15min', adjusted='true'):
    ''' Obtains DataFrame for all periods in a given short term interval 
        (15min) 
        '''

    # these can be edited if desired
    year = [1, 2]
    month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    stock_dfs = []

    for y in year:
        for m in month:
            try:
                _df = get_stock_data(symbol, tslice=f'year{y}month{m}')
                stock_dfs.append(_df)
            except AttributeError as err:
                print(err)
                print('Error: ', f'year{y}month{m}')

            # pause exceution for API (5 calls per minute limit)
            time.sleep(15)


    # get single stock DataFrame
    stock_df = pd.concat(stock_dfs, axis=0).sort_index()

    return stock_df


# example usage to get 2 year time slcie of 15min intervals
# stock_df = get_multi_periods(symbol='AAPL')

def main():
    symbols = [
        'AAPL',
        'TSM',
        'NVDA',
        'AVGO',
        'ASML',
        'CSCO',
        'TXN',
        'QCOM',
        'RTX',
        'AMD',
        'INTC',
        'LMT'
        ]
    
    for symbol in symbols:
        print(f'Acquiring {symbol}')
        stock_df = get_multi_periods(symbol)
        stock_df.to_csv(f'{symbol}_15min.csv')
        
        
if __name__ == '__main__':
    main()
    
