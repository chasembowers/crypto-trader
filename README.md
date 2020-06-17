crypto-trader
========
Crypto-currency backtesting and price prediction built around ML prediction of probability distribution of instantaneous price 

## Model

Instead of using regression to predict a price at a future instant or using classification to predict the probability of the price being below/above a number at a future instant, approximate probability distributions over price at a future instant are are predicted. A binary tree of classifiers is created, with each node splitting the probability distribution at a certain price. In this way, an arbitrarily granular approximation of a probability distribution over price can be created.

Model Construction Example
```python
# Create a model which predicts probability price will change by
# ratio of (-inf,-.002],(-.002, 0],(0,.002],(.002, inf).
# Use RandomForest as base classifier for each of 3 splits of (-inf,inf).
# Generate features from the 3 most recent price changes (4 prices).
model = BisectingClassifier(
    [-.002, 0, .002],
    [RandomForestClassifier(max_depth=10, n_estimators=20) for i in range(3)],
    3)
```
