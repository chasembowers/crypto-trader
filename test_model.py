from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from model import BisectingClassifier, DistributionBin


class TestBisectingClassifier(TestCase):

    def setUp(self):
        self.base_models = [mock.create_autospec(SGDClassifier) for i in range(4)]
        self.splits = [-.2, -.1, .1, .2]
        self.model = BisectingClassifier(self.splits, self.base_models, 3)

    def test_fit(self):
        prices = pd.Series([10, 10, 10, 10, 11.5, 11, 9.5])
        log_prices = np.log(prices / prices.shift(1))
        self.model.fit(prices)

        X2, y2 = self.base_models[2].fit.call_args[0]
	
	# The third base classifier should be fit on all samples, and samples should be re-labeled with y>.1 (bool)
	np.testing.assert_array_equal(X2,
                                      [np.flip(log_prices[1:4]), np.flip(log_prices[2:5]), np.flip(log_prices[3:6])])
        np.testing.assert_array_equal(y2, log_prices[4:] > .1)

	X3, y3 = self.base_models[3].fit.call_args[0]
	
	# The fourth base classifier should be fit on only the sample with y>.1, and samples should be re-labeled with y>.2 (bool)
        np.testing.assert_array_equal(X3,
                                      [np.flip(log_prices[1:4])])
        np.testing.assert_array_equal(y3, log_prices[4] > .2)

	
        X1, y1 = self.base_models[1].fit.call_args[0]
	
	# The second base classifier should be fit on only samples with y<.1, and samples should be re-labeled with y>-.1 (bool)
        np.testing.assert_array_equal(X1,
                                      [np.flip(log_prices[2:5]), np.flip(log_prices[3:6])])
        np.testing.assert_array_equal(y1, log_prices[5:] > -.1)

        X0, y0 = self.base_models[0].fit.call_args[0]
       
	# The first base classifier should be fit on only samples with y<-.1, and samples should be re-labeled with y>-.2 (bool)
	np.testing.assert_array_equal(X0,
                                      [np.flip(log_prices[3:6])])
        np.testing.assert_array_equal(y0, log_prices[6] > -.2)

    def test_predict(self):

	# recent prices
        prices = pd.Series([10, 11.5, 11, 9.5])
        log_prices = np.log(prices / prices.shift(1))

	# base model's predictions are of the form [[probability that price is less than split, probability that price is not]]
        self.base_models[2].predict_proba.return_value = [[1]]
        self.base_models[1].predict_proba.return_value = [[.6, .4]]
        self.base_models[0].predict_proba.return_value = [[.1, .9]]

        for model in self.base_models: model.classes_ = [0, 1]
        self.base_models[2].classes_ = [0]

	# convert splits from [ratio of future price difference to current price] to [future price]
        abs_price_splits = (1 + np.array(self.splits)) * 9.5

	# the model is fit on old training prices
	self.model.fit([1.25, 1.25, 1.25, 1.25, 1.25, 1, .85])	

	# A DistributionBin should be included iff there is probability>0 that price will be in that interval
	# DistributionBin.mean should be [the mean label for samples with corresponding labels] * [current price]
	# DistributionBin.probability should be the product of base model predictions corresponding to adjacent/parent nodes in split tree
        expected_pred = [DistributionBin(0, abs_price_splits[0], 1 / 1.25 * 9.5, .6 * .1),
                         DistributionBin(abs_price_splits[0], abs_price_splits[1], .85 * 9.5, .6 * .9),
                         DistributionBin(abs_price_splits[1], abs_price_splits[2], 9.5, .4)]
        
        self.assertEqual(expected_pred, self.model.predict(prices))

	# The fourth base model should not be fit because there are no samples with labels greater than the third split
	self.base_models[3].fit.assert_not_called()

	# Base models should be called with log of difference of recent prices
        np.testing.assert_array_equal(self.base_models[2].predict_proba.call_args[0][0],
                                      [np.flip(log_prices[1:])])
