# backend/test/predictors/measure_enhanced_gradient_boosting_predictor.py

from unittest import TestCase
from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.results_provider import training_results, validation_results
from matchpredictor.predictors.enhanced_gradient_boosting_predictor import train_enhanced_gradient_boosting_predictor
from test.predictors import csv_location

class TestEnhancedGradientBoostingPredictor(TestCase):
    """
    Tests for the EnhancedGradientBoostingPredictor.
    """
    def test_accuracy(self) -> None:
        """
        Tests that the EnhancedGradientBoostingPredictor achieves at least 50% accuracy.
        """
        training_data = training_results(csv_location, 2021, result_filter=lambda result: result.season >= 2018)
        validation_data = validation_results(csv_location, 2021)
        predictor = train_enhanced_gradient_boosting_predictor(training_data)

        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, .43)
