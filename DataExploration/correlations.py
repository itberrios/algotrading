import yfinance as yf
import pandas as pd
import glob

tickers = ['AAPL','TSLA','GOOG','QCOM']

def get_price_history(ticker, save=True):
    START_DATE = '2022-8-10'
    END_DATE = '2022-10-6'
    RAW_INTERVAL = "15m" 

    tickerData = yf.Ticker(ticker)
    df = tickerData.history(interval=RAW_INTERVAL,  start=START_DATE, end=END_DATE)
    
    if save:
        df.to_csv("data/yahoo/{}_15m.csv".format(ticker))
        

for ticker in tickers:
    get_price_history(ticker, save=True)

for csv_file in glob.glob("data/yahoo/*.csv"):
    df = pd.read_csv(csv_file)
    print(csv_file, df.shape[0])
    

aapl = pd.read_csv("data/yahoo/AAPL_15m.csv")
goog = pd.read_csv("data/yahoo/GOOG_15m.csv")
tsla = pd.read_csv("data/yahoo/TSLA_15m.csv")
qcom = pd.read_csv("data/yahoo/QCOM_15m.csv")