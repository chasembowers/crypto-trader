import numpy as np

class Trader(object):

    """Effectively a function from model prediction and current position (long/short/neither) to desired position"""

    def train(self, prices):
        raise NotImplementedError()
	
    # Accepts recent prices and returns list of DistributionBins and desired position ('long','short',None)
    def predict_and_trade(self, prices, curr_position):
        raise  NotImplementedError()

class SimpleTrader(Trader):

    """Compress predicted price distributions into expected value, and maximize expected value of returns"""

    # 'train_window' - number of prices in training window
    # 'fee_ratio' - exchange fee as fraction of price (not a percentage)
    def __init__(self, model, train_window, fee_ratio):
        self._model = model
        self._train_window = train_window
        self._fee_ratio = fee_ratio

    def train(self, prices):
        prices = np.array(prices)
        if len(prices) < self._train_window: raise ValueError('Training prices cannot be less than train window.')
        self._model.fit(prices[-self._train_window:])

    def predict_and_trade(self, prices, curr_position):
        prices = np.array(prices)
        prediction = self._model.predict(prices)
        new_position = None
        expected_price = 0
        curr_price = prices[-1]
        for dist_bin in prediction:
            if dist_bin.probability == 0: continue
            expected_price += dist_bin.mean * dist_bin.probability
	# TODO- fee should be fraction of current + future price rather than buying price
        if expected_price / curr_price - 1 > self._fee_ratio:
            new_position = 'long'
        if curr_price / expected_price - 1 > self._fee_ratio:
            new_position = 'short'
        if not new_position and curr_position is not None:
            if curr_position is 'long' and expected_price / curr_price > 1:
                new_position = 'long'
            if curr_position is 'short' and expected_price / curr_price < 1:
                new_position = 'short'
        return (prediction, new_position)


