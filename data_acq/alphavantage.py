# import yfinance as yf
# tickerData = yf.Ticker('AAPL')
# df = tickerData.history(interval='15m',  start='2022-8-', end='2022-9-25')

import requests
import csv
import pandas as pd
import numpy as np
import time

tickers = ['AAPL','QCOM']

periods = []
for year in [2,1]:
    for month in range(12,0,-1):
        periods.append('year{}month{}'.format(year, month))


for ticker in tickers[:10]:
    data = []
    print("getting data for ticker {}...".format(ticker))
    for period in periods:
        with requests.Session() as s:
            CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval=15min&slice={}&apikey=MYK5M86CMOZ6XX86'.format(ticker,period)
            download = s.get(CSV_URL)
            decoded_content = download.content.decode('utf-8')
            
        lines = decoded_content.split('\n')[:-1]
        df = pd.DataFrame([line.split(',') for line in lines[1:]], 
                          columns = lines[0].split(','),
                          dtype = np.float64
                          )
        df = df.rename({'time':'Time','low':'Low','high':'High', 'close':'Close','open':'Open','volume':'Volume'}, axis=1)
        data.append(df)    
        time.sleep(12) # 5 calls per minute 
    
    # concat, sort, and save df
    df = pd.concat(data, axis=0)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')
    df.to_csv('data/raw/{}_15min.csv'.format(ticker), index=False)

