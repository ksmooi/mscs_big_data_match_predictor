# backend/matchpredictor/predictors/enhanced_logistic_regression_predictor.py

from typing import List, Tuple, Optional

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Prediction, Predictor

class EnhancedLogisticRegressionPredictor(Predictor):
    """
    A predictor that uses logistic regression to make predictions based on encoded team names.
    """
    def __init__(self, model: LogisticRegression, team_encoding: OneHotEncoder) -> None:
        """
        Initializes the EnhancedLogisticRegressionPredictor with a trained logistic regression model
        and a OneHotEncoder for team names.
        
        :param model: Trained logistic regression model.
        :param team_encoding: OneHotEncoder for team names.
        """
        self.model = model
        self.team_encoding = team_encoding

    def predict(self, fixture: Fixture) -> Prediction:
        """
        Predicts the outcome of a fixture using the logistic regression model.
        
        :param fixture: The fixture to predict the outcome for.
        :return: Prediction containing the predicted outcome.
        """
        # Encode the home and away team names
        encoded_home_name = self.__encode_team(fixture.home_team)
        encoded_away_name = self.__encode_team(fixture.away_team)

        # Handle cases where team encoding is not available
        if encoded_home_name is None:
            return Prediction(outcome=Outcome.AWAY)
        if encoded_away_name is None:
            return Prediction(outcome=Outcome.HOME)

        # Concatenate the encoded names and predict the outcome
        x: NDArray[float64] = np.concatenate([encoded_home_name, encoded_away_name], 1)
        pred = self.model.predict(x)

        # Map the prediction to the corresponding outcome
        if pred > 0:
            return Prediction(outcome=Outcome.HOME)
        elif pred < 0:
            return Prediction(outcome=Outcome.AWAY)
        else:
            return Prediction(outcome=Outcome.DRAW)

    def __encode_team(self, team: Team) -> Optional[NDArray[float64]]:
        """
        Encodes the team name using the OneHotEncoder.
        
        :param team: The team to encode.
        :return: Encoded team name as a NumPy array, or None if encoding fails.
        """
        try:
            result: NDArray[float64] = self.team_encoding.transform(np.array(team.name).reshape(-1, 1))
            return result
        except ValueError:
            return None

def build_enhanced_model(results: List[Result]) -> Tuple[LogisticRegression, OneHotEncoder]:
    """
    Builds and trains a logistic regression model using the given match results.
    
    :param results: List of match results to train the model on.
    :return: Trained logistic regression model and the corresponding OneHotEncoder.
    """
    # Extract team names and goals from the results
    home_names = np.array([r.fixture.home_team.name for r in results])
    away_names = np.array([r.fixture.away_team.name for r in results])
    home_goals = np.array([r.home_goals for r in results])
    away_goals = np.array([r.away_goals for r in results])

    # Prepare the team name encoding
    team_names = np.array(list(home_names) + list(away_names)).reshape(-1, 1)
    team_encoding = OneHotEncoder(sparse_output=False).fit(team_names)

    # Encode the team names
    encoded_home_names = team_encoding.transform(home_names.reshape(-1, 1))
    encoded_away_names = team_encoding.transform(away_names.reshape(-1, 1))

    # Prepare the feature matrix and target vector
    x: NDArray[float64] = np.concatenate([encoded_home_names, encoded_away_names], 1)
    y = np.sign(home_goals - away_goals)

    # Train the logistic regression model
    model = LogisticRegression(penalty="l2", fit_intercept=False, multi_class="ovr", C=1)
    model.fit(x, y)

    return model, team_encoding

def train_enhanced_logistic_regression_predictor(results: List[Result]) -> Predictor:
    """
    Trains and returns an EnhancedLogisticRegressionPredictor.
    
    :param results: List of match results to train the model on.
    :return: Trained EnhancedLogisticRegressionPredictor.
    """
    model, team_encoding = build_enhanced_model(results)
    return EnhancedLogisticRegressionPredictor(model, team_encoding)
