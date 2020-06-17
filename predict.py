import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from model import BisectingClassifier
from sklearn.ensemble import RandomForestClassifier

# Every time a new candle(s) appears in the local database,
# predict a distribution of price for the next candle,
# and re-train the model on the last month.

DATABASE = 'postgresql:///BitcoinHistory'

TABLE = 'BTC/USDT'

MINUTES_IN_MONTH = 30 * 24 * 60

def predict(recent_prices, model):
    return model.predict(np.array(recent_prices))

engine = create_engine(DATABASE)
history = pd.read_sql_table(TABLE, engine, index_col='index')
close = history['close']

# Create a model which predicts probability price will change by
# ratio of (-inf,-.002],(-.002, 0],(0,.002],(.002, inf).
# Use RandomForest as base classifier for each of 3 splits of (-inf,inf).
# Generate features from the 3 most recent price changes (4 prices).
model = BisectingClassifier(
    [-.002, 0, .002],
    [RandomForestClassifier(max_depth=10, n_estimators=20) for i in range(3)],
    3)
model.fit(close.iloc[-MINUTES_IN_MONTH:])

last_time = history.index.max()
print('Waiting for first candle...')
while True:
    new_candles = pd.read_sql_query(
        """SELECT * FROM "BTC/USDT" WHERE index > {}""".format(last_time),
        engine,
        index_col='index')
    if not new_candles.empty:
        last_time = new_candles.index.max()
        close = close.append(new_candles['close'])
        print('Prediction for timestamp {:d}:'.format(int(last_time)))
        print(predict(close.iloc[-4:], model), '\n')
        print('Fitting model...')
        model.fit(close.iloc[-MINUTES_IN_MONTH:])
        print('Waiting for next candle...')

    time.sleep(1)
