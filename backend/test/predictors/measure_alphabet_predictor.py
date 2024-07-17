# backend/test/predictors/measure_alphabet_predictor.py

from unittest import TestCase

from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.results_provider import validation_results
from matchpredictor.predictors.alphabet_predictor import AlphabetPredictor
from test.predictors import csv_location

class TestAlphabetPredictor(TestCase):
    def test_accuracy(self) -> None:
        validation_data = validation_results(csv_location, 2019)
        predictor = AlphabetPredictor()
        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, .33)
