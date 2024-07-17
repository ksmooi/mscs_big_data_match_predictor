from typing import List, Dict
from collections import defaultdict
from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction

class FormBasedPredictor(Predictor):
    def __init__(self, results: List[Result], form_length: int = 5) -> None:
        self.form_length = form_length
        self.team_forms: Dict[str, List[Outcome]] = defaultdict(list)
        self._calculate_team_forms(results)

    def _calculate_team_forms(self, results: List[Result]) -> None:
        for result in results:
            home_team = result.fixture.home_team.name
            away_team = result.fixture.away_team.name
            
            if result.outcome == Outcome.HOME:
                self._update_form(home_team, Outcome.HOME)
                self._update_form(away_team, Outcome.AWAY)
            elif result.outcome == Outcome.AWAY:
                self._update_form(home_team, Outcome.AWAY)
                self._update_form(away_team, Outcome.HOME)
            else:
                self._update_form(home_team, Outcome.DRAW)
                self._update_form(away_team, Outcome.DRAW)

    def _update_form(self, team: str, outcome: Outcome) -> None:
        self.team_forms[team].append(outcome)
        if len(self.team_forms[team]) > self.form_length:
            self.team_forms[team] = self.team_forms[team][-self.form_length:]

    def predict(self, fixture: Fixture) -> Prediction:
        home_form = self.team_forms.get(fixture.home_team.name, [])
        away_form = self.team_forms.get(fixture.away_team.name, [])

        home_score = self._calculate_form_score(home_form)
        away_score = self._calculate_form_score(away_form)

        if home_score > away_score:
            return Prediction(outcome=Outcome.HOME)
        elif away_score > home_score:
            return Prediction(outcome=Outcome.AWAY)
        else:
            return Prediction(outcome=Outcome.DRAW)

    def _calculate_form_score(self, form: List[Outcome]) -> float:
        score = 0
        for i, outcome in enumerate(reversed(form), 1):
            if outcome == Outcome.HOME:
                score += 3 * i
            elif outcome == Outcome.DRAW:
                score += 1 * i
        return score / (len(form) * (len(form) + 1) / 2) if form else 0

def train_form_based_predictor(results: List[Result]) -> Predictor:
    return FormBasedPredictor(results)
