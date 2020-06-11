import pandas as pd
import numpy as np


class DistributionBin(object):

    """Describes an interval ['lower','upper'] of price in a probability distribution over price"""

    # 'mean' - the expected value of price given that price is between 'lower' and 'upper'
    # 'probability' - probability that price is between 'lower' and 'upper'
    def __init__(self, lower, upper, mean, probability):
        if not ((mean >= lower) and (upper >= mean)): raise ValueError("'lower' <= 'mean' <= 'upper' violated.")
        if probability < 0 or probability > 1: raise ValueError("probability must be 0<=p<=1")
        self._lower = lower
        self._upper = upper
        self._mean = mean
        self._probability = probability

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, lower):
        if lower > self._upper or lower > self._mean:
            raise ValueError("'lower' cannot be greater than 'upper' or 'mean'")
        self._lower = lower

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        if mean < self._lower or mean > self._upper: raise ValueError("'lower' <= 'mean' <= 'upper' violated.")
        self._mean = mean

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, upper):
        if upper < self._lower or upper < self._mean: raise ValueError("'upper' cannot be less than 'lower' or 'mean'")
        self._upper = upper

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, probability):
        if probability < 0 or probability > 1: raise ValueError("probability must be 0<=p<=1")
        self._probability = probability

    def __eq__(self, other):
        if isinstance(other, DistributionBin):
            return self.lower == other.lower and \
                   self.mean == other.mean and \
                   self.upper == other.upper and \
                   self.probability == other.probability
        return False

    def __repr__(self):
        return 'DistributionBin<{},{},{}>({})'.format(self.lower, self.mean, self.upper, self.probability)


class Model(object):

    def fit(self, prices):
        raise NotImplementedError()

    # predict probability distribution, represented by a list of DistributionBins, over price at some future instant
    def predict(self, last_prices):
        raise NotImplementedError()

# TODO- rename to 'BinaryClassifier'
class DistributionClassifier(object):

    """Predict probability distributions, split at some price into two DistributionBins, over price at some future instant"""

    # 'clf' is a classifier that implments 'fit' and 'predict' (like an sklearn classifier)
    def __init__(self, clf):
        self._clf = clf
        self._split = None
        self._below_mean = None
        self._above_mean = None

    # fit classifier 'clf' on X,(y>split)
    def fit(self, X, y, split):

        self._split = split
        above = y[y > split]
        below = y[y <= split]
        if len(above) > 0:
            self._above_mean = above.mean()
        else:
            self._above_mean = None
        if len(below) > 0:
            self._below_mean = below.mean()
        else:
            self._below_mean = None
        y = y > split
        self._clf.fit(X, y)

    def predict_proba(self, X):

        y_pred = self._clf.predict_proba(X)
        mean_pred = []
        for i in range(len(y_pred)):
            if list(self._clf.classes_) == [0] or (len(y_pred[i]) == 2 and y_pred[i][0] == 1):
                mean_pred.append((DistributionBin(-np.inf, self._split, self._below_mean, 1), None))
            elif list(self._clf.classes_) == [1] or (len(y_pred[i]) == 2 and y_pred[i][1] == 1):
                mean_pred.append((None, DistributionBin(self._split, np.inf, self._above_mean, 1)))
            else:
                mean_pred.append((DistributionBin(-np.inf, self._split, self._below_mean, y_pred[i][0]),
                                  DistributionBin(self._split, np.inf, self._above_mean, y_pred[i][1])))
        return mean_pred


class BinaryClassifierTree(object):

    """Tree of BinaryClassifiers, where left/right sub-trees predict price distributions given that the price will be below/above 'split'"""

    def __init__(self, clf, split, below=None, above=None):
        self.clf = clf
        self.split = split
        self.below = below
        self.above = above


class BisectingClassifier(Model):

    """Generate a BinaryClassifierTree, and use it to predict a probability distribution over price at some future instant"""

    def _create_tree(self, splits, base_models):
        if splits == []: return None
        if len(splits) != len(base_models): raise ValueError()
        num_splits = len(splits)
        bisector = int(num_splits / 2.)

        return BinaryClassifierTree(
            DistributionClassifier(base_models[bisector]),
            splits[bisector],
            self._create_tree(splits[:bisector], base_models[:bisector]),
            self._create_tree(splits[bisector + 1:], base_models[bisector + 1:]))

    
    # 'splits'- prices at which probabiilty distribution over price will be split into DistributionBins
    # 'base_models'- list of len('splits') classifiers with 'fit' and 'predict' methods for predicting price distributions
    # 'lags'- the number of previous time steps used to predict future price ('lags' + 1 instants)
    def __init__(self, splits, base_models, lags):

        if not len(splits) > 0: raise ValueError('No splits provided.')
        if sorted(splits) != splits: raise ValueError("'splits' must be monotonically increasing.")
        if len(set(splits)) is not len(splits): raise ValueError("'splits' must have only unique values.")
        if len(base_models) != len(splits): raise ValueError('Must have same number of splits and base models.')

        log_splits = [np.log(split + 1) for split in splits]
        self._tree = self._create_tree(log_splits, base_models)
        self._lags = lags

    def _fit_tree(self, tree, train_data):
        tree.clf.fit(train_data.drop('y', 1), train_data['y'], tree.split)
        if tree.below:
            below_index = train_data['y'] <= tree.split
            if below_index.any(): self._fit_tree(tree.below, train_data[below_index])
        if tree.above:
            above_index = train_data['y'] > tree.split
            if above_index.any(): self._fit_tree(tree.above, train_data[above_index])

    def fit(self, prices):
        prices = np.array(prices)
        if 0. in prices: raise ValueError('Encountered 0 price.')
        prices_series = pd.Series(prices)
        train_df = pd.DataFrame()
        train_df['y'] = np.log(prices_series / prices_series.shift(1))
        for l in range(self._lags):
            train_df['x' + str(l)] = train_df['y'].shift(l + 1)
        train_df.dropna(inplace=True)
        self._fit_tree(self._tree, train_df)

    def _tree_predict(self, tree, features):
        below_dist, above_dist = tree.clf.predict_proba([features])[0]

        if below_dist !=  None:
            prob_below = below_dist.probability
            below_dist = [below_dist]
            if tree.below:
                below_dist = self._tree_predict(tree.below, features)
                for dbin in below_dist:
                    dbin.probability *= prob_below
                if below_dist[-1].upper == np.inf: below_dist[-1].upper = tree.split
        else:
            below_dist = []

        if above_dist != None:
            prob_above = above_dist.probability
            above_dist = [above_dist]

            if tree.above:
                above_dist = self._tree_predict(tree.above, features)
                for dbin in above_dist:
                    dbin.probability *= prob_above
                if above_dist[0].lower == -np.inf: above_dist[0].lower = tree.split
        else:
            above_dist = []

        dist = below_dist + above_dist
        return dist

    # Use the last 'lags' + 1 prices to predict a price distribution, represented as a list of DistributionBins
    def predict(self, last_prices):
        last_prices = np.array(last_prices)
        features = np.flip(np.log(last_prices[1:] / last_prices[:-1]))
        pred = self._tree_predict(self._tree, features)
        abs_price_pred = []
        for dbin in pred:
            abs_price_pred.append(
                DistributionBin(
                    np.exp(dbin.lower) * last_prices[-1],
                    np.exp(dbin.upper) * last_prices[-1],
                    np.exp(dbin.mean) * last_prices[-1],
                    dbin.probability))
        return abs_price_pred
