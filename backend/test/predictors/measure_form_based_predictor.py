from unittest import TestCase

from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.results_provider import training_results, validation_results
from matchpredictor.predictors.form_based_predictor import train_form_based_predictor
from test.predictors import csv_location

class TestFormBasedPredictor(TestCase):
    def test_accuracy(self) -> None:
        training_data = training_results(csv_location, 2021, result_filter=lambda result: result.season >= 2018)
        validation_data = validation_results(csv_location, 2021)
        predictor = train_form_based_predictor(training_data)

        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, 0.40)

    def test_championship_accuracy(self) -> None:
        training_data = training_results(csv_location, 2021, result_filter=lambda result: result.season >= 2018 and result.fixture.league == "English League Championship")
        validation_data = validation_results(csv_location, 2021, result_filter=lambda result: result.fixture.league == "English League Championship")
        predictor = train_form_based_predictor(training_data)

        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, 0.40)
