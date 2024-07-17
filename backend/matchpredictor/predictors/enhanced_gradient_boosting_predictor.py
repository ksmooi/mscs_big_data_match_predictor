# backend/matchpredictor/predictors/enhanced_gradient_boosting_predictor.py

from typing import List, Tuple, Optional
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore
from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Prediction, Predictor

class EnhancedGradientBoostingPredictor(Predictor):
    """
    A predictor that uses gradient boosting to make predictions based on encoded team names and additional features.
    """
    def __init__(self, model: GradientBoostingClassifier, team_encoding: OneHotEncoder, results: List[Result]) -> None:
        """
        Initializes the EnhancedGradientBoostingPredictor with a trained gradient boosting model,
        a OneHotEncoder for team names, and match results for feature extraction.
        
        :param model: Trained gradient boosting model.
        :param team_encoding: OneHotEncoder for team names.
        :param results: List of match results for feature extraction.
        """
        self.model = model
        self.team_encoding = team_encoding
        self.results = results

    def predict(self, fixture: Fixture) -> Prediction:
        """
        Predicts the outcome of a fixture using the gradient boosting model.
        
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

        # Gather additional features for the prediction
        home_form_score = self.get_form_score(fixture.home_team)
        away_form_score = self.get_form_score(fixture.away_team)
        recent_home_goals, recent_home_conceded = self.get_recent_goals(fixture.home_team)
        recent_away_goals, recent_away_conceded = self.get_recent_goals(fixture.away_team)
        home_win_streak = self.get_win_streak(fixture.home_team)
        away_win_streak = self.get_win_streak(fixture.away_team)

        # Concatenate the encoded names and additional features, then predict the outcome
        additional_features = np.array([
            home_form_score, away_form_score, recent_home_goals, recent_away_goals,
            recent_home_conceded, recent_away_conceded, home_win_streak, away_win_streak
        ]).reshape(1, -1)
        x: NDArray[float64] = np.concatenate([encoded_home_name, encoded_away_name, additional_features], axis=1)
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

    def get_form_score(self, team: Team) -> float:
        """
        Calculate the form score for a team based on the last 5 matches.
        
        :param team: The team to calculate the form score for.
        :return: Form score for the team.
        """
        form_score = 0
        count = 0
        for result in self.results:
            if result.fixture.home_team == team:
                if result.outcome == Outcome.HOME:
                    form_score += 3
                elif result.outcome == Outcome.DRAW:
                    form_score += 1
                count += 1
            elif result.fixture.away_team == team:
                if result.outcome == Outcome.AWAY:
                    form_score += 3
                elif result.outcome == Outcome.DRAW:
                    form_score += 1
                count += 1
            if count == 5:
                break
        return form_score / 15  # Normalize form score to be between 0 and 1

    def get_recent_goals(self, team: Team) -> Tuple[int, int]:
        """
        Calculate the recent goals scored and conceded by a team based on the last 5 matches.
        
        :param team: The team to calculate the goals for.
        :return: Tuple of recent goals scored and recent goals conceded.
        """
        goals_scored = 0
        goals_conceded = 0
        count = 0
        for result in self.results:
            if result.fixture.home_team == team:
                goals_scored += result.home_goals
                goals_conceded += result.away_goals
                count += 1
            elif result.fixture.away_team == team:
                goals_scored += result.away_goals
                goals_conceded += result.home_goals
                count += 1
            if count == 5:
                break
        return goals_scored, goals_conceded

    def get_win_streak(self, team: Team) -> int:
        """
        Calculate the win streak for a team based on the last 5 matches.
        
        :param team: The team to calculate the win streak for.
        :return: Win streak for the team.
        """
        win_streak = 0
        for result in self.results:
            if result.fixture.home_team == team and result.outcome == Outcome.HOME:
                win_streak += 1
            elif result.fixture.away_team == team and result.outcome == Outcome.AWAY:
                win_streak += 1
            else:
                break
        return win_streak

def build_enhanced_model(results: List[Result]) -> Tuple[GradientBoostingClassifier, OneHotEncoder]:
    """
    Builds and trains a gradient boosting model using the given match results.
    
    :param results: List of match results to train the model on.
    :return: Trained gradient boosting model and the corresponding OneHotEncoder.
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

    # Create an instance of the predictor to access its methods
    temp_predictor = EnhancedGradientBoostingPredictor(None, team_encoding, results)

    # Add additional features: recent performance metrics and form scores
    home_form_scores = np.array([temp_predictor.get_form_score(r.fixture.home_team) for r in results])
    away_form_scores = np.array([temp_predictor.get_form_score(r.fixture.away_team) for r in results])
    recent_home_goals = np.array([temp_predictor.get_recent_goals(r.fixture.home_team)[0] for r in results])
    recent_away_goals = np.array([temp_predictor.get_recent_goals(r.fixture.away_team)[0] for r in results])
    recent_home_conceded = np.array([temp_predictor.get_recent_goals(r.fixture.home_team)[1] for r in results])
    recent_away_conceded = np.array([temp_predictor.get_recent_goals(r.fixture.away_team)[1] for r in results])
    home_win_streak = np.array([temp_predictor.get_win_streak(r.fixture.home_team) for r in results])
    away_win_streak = np.array([temp_predictor.get_win_streak(r.fixture.away_team) for r in results])

    additional_features = np.column_stack((
        home_form_scores, away_form_scores, recent_home_goals, recent_away_goals,
        recent_home_conceded, recent_away_conceded, home_win_streak, away_win_streak
    ))
    x = np.concatenate((x, additional_features), axis=1)

    # Train the gradient boosting model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(x, y)

    return model, team_encoding

def train_enhanced_gradient_boosting_predictor(results: List[Result]) -> Predictor:
    """
    Trains and returns an EnhancedGradientBoostingPredictor.
    
    :param results: List of match results to train the model on.
    :return: Trained EnhancedGradientBoostingPredictor.
    """
    model, team_encoding = build_enhanced_model(results)
    return EnhancedGradientBoostingPredictor(model, team_encoding, results)
