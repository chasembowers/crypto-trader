import ccxt
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Download historical candlestick data into a local SQL database.
# Then stream live candlestick data into the database.

DATABASE = 'postgresql:///BitcoinHistory'
TABLE = 'BTC/USDT'
PAIR = 'BTC/USDT'

engine = create_engine(DATABASE)
binance = ccxt.binance()
binance.load_markets()
read_table = pd.read_sql_table(TABLE, engine, index_col='index')

# Get the index of the most recent candle.
last_time = read_table.index.max()

if np.isnan(last_time): last_time = 0
if (read_table.index.duplicated().sum() > 0): raise Exception('Database has repeat entries.')
while True:
    print('Latest timestamp:',last_time)

    # Get 1 minute candlesticks that aren't already downloaded.
    new_candles = binance.fetch_ohlcv(PAIR, '1m', since=int(last_time + 1))
    if new_candles:

        # Update the index of the most recent candle.
        last_time = new_candles[-1][0]

        new_candles_array = np.array(new_candles)
        new_candles_df = pd.DataFrame(
            new_candles_array[:, 1:],
            index=new_candles_array[:, 0],
            columns=['open', 'high', 'low', 'close', 'volume'])
        new_candles_df.to_sql(TABLE, engine, if_exists='append')

    # Rate limit is in milliseconds.
    time.sleep(binance.rateLimit / 1000)
