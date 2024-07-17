# backend/test/predictors/measure_enhanced_logistic_regression_predictor.py

from unittest import TestCase

from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.results_provider import training_results, validation_results
from matchpredictor.predictors.enhanced_logistic_regression_predictor import train_enhanced_logistic_regression_predictor
from test.predictors import csv_location

class TestEnhancedLogisticRegressionPredictor(TestCase):
    def test_accuracy(self) -> None:
        training_data = training_results(csv_location, 2021, result_filter=lambda result: result.season >= 2018)
        validation_data = validation_results(csv_location, 2021)
        predictor = train_enhanced_logistic_regression_predictor(training_data)

        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, .43)
