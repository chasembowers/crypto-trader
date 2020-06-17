from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import BisectingClassifier
from trader import SimpleTrader
from sklearn.ensemble import RandomForestClassifier
from os.path import expanduser

# Simulate trading on all of local historical closing prices,
# periodically retraining the trader's model,
# and periodically replacing saved graph of profit/losss and monthly sharpe ratio.
# Every order size is the same.

DATABASE = 'postgresql:///BitcoinHistory'
TABLE = 'BTC/USDT'
PLOT_DIR = expanduser("~") + '/trader_experiments/3_lag.png'

# ratio of order size lost in transaction fee
FEE_RATIO = .002

MINUTES_IN_MONTH = 30 * 24 * 60

# number of recent candle prices to train a model on - 30 days
TRAIN_WINDOW = MINUTES_IN_MONTH

# how outdated the most recent price trained on is before re-training
MAX_STALENESS = int(TRAIN_WINDOW * .05)

def backtest():
    engine = create_engine(DATABASE)

    history = pd.read_sql_table(TABLE, engine, index_col='index')
    minutes = history.shape[0]
    close = history['close']

    # Create a model which predicts probability price will change by
    # ratio of (-inf,-.002],(-.002, 0],(0,.002],(.002, inf).
    # Use RandomForest as base classifier for each of 3 splits of (-inf,inf).
    # Generate features from the 3 most recent price changes (4 prices).
    model = BisectingClassifier(
        [-.002, 0, .002],
        [RandomForestClassifier(max_depth=10, n_estimators=20) for i in range(3)],
        3)

    # Create a trader which maximizes expected value,
    # assuming trade is exited at instant of price prediction.
    trader = SimpleTrader(model, TRAIN_WINDOW, FEE_RATIO)

    time = TRAIN_WINDOW
    last_train = None
    curr_position = None
    enter_price = None
    profits = []
    long_profits = []
    short_profits = []
    plt.ioff()
    f, (ax1, ax2) = plt.subplots(1,2)
    while time < minutes:

        # Periodically update saved graph and re-train model on recent prices.
        if last_train is None or time - last_train > MAX_STALENESS:

            # Plot profit/loss for all trades.
            ax1.clear()
            ax1.set_title('Bankroll')
            ax1.plot(range(len(profits)), np.cumsum(profits))

            # Plot the last 1000 candles of the monthly sharpe ratio.
            ax2.clear()
            profits_series = pd.Series(profits)
            monthly_sharpe = profits_series.expanding().mean() / profits_series.expanding().std() * np.sqrt(
                MINUTES_IN_MONTH)
            monthly_sharpe = monthly_sharpe.iloc[-1000:]
            ax2.plot(range(len(monthly_sharpe)), monthly_sharpe)
            ax2.set_title('Monthly Sharpe Ratio')

            plt.savefig(PLOT_DIR, bbox_inches='tight')

            # Re-train the trader's model on recent prices.
            print('Training...')
            trader.train(close.iloc[time - TRAIN_WINDOW:time])
            last_train = time
            print('Done training.')

        # Get a price prediction and suggested trading position.
        prediction, new_pos = trader.predict_and_trade(close.iloc[time - 4:time], curr_position)

        last_price = close.iloc[time - 1]

        # TODO- fee should be fraction of current + future price rather than buying price
        if curr_position == None:
            if new_pos != None:
                enter_price = last_price
        elif curr_position == 'long':
            if new_pos == None:
                profit = last_price / enter_price - 1. - FEE_RATIO
                profits.append(profit)
                long_profits.append(profit)
                enter_price = None
            elif new_pos == 'short':
                profit = last_price / enter_price - 1. - FEE_RATIO
                profits.append(profit)
                long_profits.append(profit)
                enter_price = last_price
        elif curr_position == 'short':
            if new_pos == None:
                profit = enter_price / last_price - 1. - FEE_RATIO
                profits.append(profit)
                short_profits.append(profit)
                enter_price = None
            elif new_pos == 'long':
                profit = enter_price / last_price - 1. - FEE_RATIO
                profits.append(profit)
                short_profits.append(profit)
                enter_price = last_price
        curr_position = new_pos
        time += 1


backtest()
