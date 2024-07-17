# backend/matchpredictor/predictors/alphabet_predictor.py

from matchpredictor.matchresults.result import Fixture, Outcome
from matchpredictor.predictors.predictor import Prediction, Predictor

class AlphabetPredictor(Predictor):
    """
    A predictor that uses alphabetical order of team names to make predictions.
    Predicts a HOME win if the home team name comes alphabetically before the away team name, otherwise predicts an AWAY win.
    """
    def predict(self, fixture: Fixture) -> Prediction:
        # Predict HOME if the home team name comes alphabetically before the away team name
        if fixture.home_team.name < fixture.away_team.name:
            return Prediction(outcome=Outcome.HOME)
        else:
            return Prediction(outcome=Outcome.AWAY)
