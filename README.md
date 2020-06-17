crypto-trader
========
Crypto-currency backtesting and price prediction built around ML prediction of probability distribution of instantaneous price 

## Model

Instead of using regression to predict a price at a future instant or using classification to predict the probability of the price being below/above a number at a future instant, approximate probability distributions over price at a future instant are predicted. A binary tree of classifiers is created, with each node splitting the probability distribution at a certain price. In this way, an arbitrarily granular approximation of a probability distribution over price can be created.

Model Construction Example
```python
# Create a model which predicts probability that price will change by
# ratio of (-inf,-.002],(-.002, 0],(0,.002],(.002, inf).
# Use RandomForest as base classifier for each of 3 splits of (-inf,inf).
# Generate features from the 3 most recent price changes (4 prices).
model = BisectingClassifier(
    [-.002, 0, .002],
    [RandomForestClassifier(max_depth=10, n_estimators=20) for i in range(3)],
    3)
```

## Trader

While the Model outputs price predictions, the Trader outputs suggested trading positions based on those predictions. So the Trader might maximize expected value of profit greedily, or it might wait for a better time to enter. The trader might vary position sizes based on the shape of the predicted price distribution. An example of an expected value maximizing Trader can be found in 'trader.py'.

## Backtest

An example of backtesting can be found in 'backtest.py'.

Trading is simulated on historical closing prices from a local database. Periodically, the model is retrained with more "recent" data, and a graph of profit/losss and monthly sharpe ratio are updated.

![alt tag](https://raw.githubusercontent.com/chasembowers/crypto-trader/master/3_lag.png)

## Local Database

Historical candlestick data is downloaded/streamed to a local SQL database using the 'ccxt' cryptocurrency API library. An example of this can be found in 'download_candles.py'.

## Predict

An example of live prediction can be found in 'predict.py'. 'download_candles.py' must be running in a separate thread. Each time a new candle is streamed to the local database, a probability distribution over the price in 1 future minute is output
in the following form:

```
[DistributionBin<0.0,9431.38853925509,9446.23966>(0.004693331982555838), DistributionBin<9446.23966,9461.434238012913,9465.17>(0.4824197677209406), DistributionBin<9465.17,9469.044872569932,9484.10034>(0.5082717689732247), DistributionBin<9484.10034,9493.683703288838,inf>(0.00461513132327911)] 
```

## Next Steps

Backtesting over months/years takes a long time (hours) because this is written in Python. Backtesting could be sharded into many smaller time periods and run in parallel in a cloud computing environment with very little loss of statistical value. Eventually, latency-sensitive parts of this system would be ported to another language like C/++,
