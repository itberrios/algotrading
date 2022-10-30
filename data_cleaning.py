import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

INTERVAL = 5

def is_within_range(timestamp):
    MIN_DATE = datetime(2020,10,23,9,30) # 2020-1-2 9:30:00
    MAX_DATE = datetime(2022,10,12,16,0) # 2022-1-2 16:00:00
    
    if type(timestamp) == str:
        # only between min_date and max_date
        sample_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    else:
        sample_date = timestamp
    
    if sample_date < MIN_DATE or sample_date > MAX_DATE:
        return False
    
    # only regular trading hours (9:30 AM - 4:00 PM)
    hour = sample_date.hour +  sample_date.minute/60 # for convenience
    if hour < 9.5 or hour > 16:
        return False

    return True
    
data = []
for csv_file in glob.glob("data/raw/{}min/*.csv".format(INTERVAL)):
    price_df = pd.read_csv(csv_file) 
    print("[RAW]", csv_file, "---", price_df['Time'].iloc[0], "---", price_df['Time'].iloc[-1], "---", price_df.shape[0])
    price_df['Time'] = pd.to_datetime(price_df['Time'])
    within_range_indices = price_df['Time'].apply(is_within_range)
    price_df = price_df[within_range_indices]
    data.append(price_df)
    print("[CLEAN]", csv_file, "---", price_df['Time'].iloc[0], "---", price_df['Time'].iloc[-1], "---", price_df.shape[0])
    ticker = csv_file.split('\\')[1].split('.')[0]
    price_df.to_csv("data/clean/{}min/{}.csv".format(INTERVAL, ticker), index=False)

    

# Note: Apple missing row for {'2020-11-27 13:30:00'}
    