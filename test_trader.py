from unittest import TestCase
from unittest import mock
from trader import SimpleTrader
from model import Model, DistributionBin
import numpy as np


class TestSimpleTrader(TestCase):

    def setUp(self):
        self.model = mock.create_autospec(Model)
        self.train_window = 3
        self.fee_ratio = .01
        self.trader = SimpleTrader(self.model, self.train_window, self.fee_ratio)

    def test_train(self):
        prices = np.array([3, 6, 3, 8, 4, 2, 1, 4, 6, 7])
        self.trader.train(prices)
	# model should be fit on last 'train_window'=3 prices
        np.testing.assert_array_equal([4, 6, 7], self.model.fit.call_args[0][0])

    def test_new_long(self):
	# The model will predict that the price will be 7 with 100% certainty
	# The price change exceeds fees
        prediction = [DistributionBin(7, 7, 7, 1)]
        self.model.predict.return_value = prediction
        self.assertEqual(self.trader.predict_and_trade([4,3,6], None), (prediction, 'long'))

    def test_new_short(self):
	# The model will predict that the price will be 5 with 100% certainty
	# The price change exceeds fees
        prediction = [DistributionBin(5, 5, 5, 1)]
        self.model.predict.return_value = prediction
        self.assertEqual(self.trader.predict_and_trade([4,3,6], None), (prediction, 'short'))

    # TODO- Test that trade is not entered when fees exceed the price change

    def test_stay_long(self):
	# The model will predict that the price will be 6.001 with 100% certainty
	# The are no fees for staying long
        prediction = [DistributionBin(6.001, 6.001, 6.001, 1)]
        self.model.predict.return_value = prediction
        self.assertEqual(self.trader.predict_and_trade([4,3,6], 'long'), (prediction, 'long'))

    def test_stay_short(self):
	# The model will predict that the price will be 5.999 with 100% certainty
	# The are no fees for staying short
        prediction = [DistributionBin(5.999, 5.999, 5.999, 1)]
        self.model.predict.return_value = prediction
        self.assertEqual(self.trader.predict_and_trade([4,3,6], 'short'), (prediction, 'short'))

    def test_multiple_bins(self):
	# The expected value of the model's prediction is 6.445
	# The expected price change exceeds fees
        prediction = [DistributionBin(5.95, 5.95, 5.95, .1), DistributionBin(6.5, 6.5, 6.5, .9)]
        self.model.predict.return_value = prediction
        self.assertEqual(self.trader.predict_and_trade([4,3,6], None), (prediction, 'long'))
