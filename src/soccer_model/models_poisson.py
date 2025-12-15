from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.stats import poisson
import math

from soccer_model.config import DEFAULT_MODEL_CONFIG


@dataclass
class TeamStrength:
    attack: float
    defense: float


class PoissonGoalModel:
    """
    Classic Poisson goal model using log-linear form:
      log(lambda_home) = base_rate + home_advantage + attack_home - defense_away
      log(lambda_away) = base_rate + attack_away - defense_home
    """
    def __init__(self,
                 base_rate: float,
                 home_advantage: float,
                 team_strengths: Dict[str, TeamStrength],
                 max_goals: int = DEFAULT_MODEL_CONFIG.max_goals):
        self.base_rate = base_rate
        self.home_advantage = home_advantage
        self.team_strengths = team_strengths
        self.max_goals = max_goals

    def get_lambdas(self, home_team: str, away_team: str) -> Tuple[float, float]:
        home = self.team_strengths[home_team]
        away = self.team_strengths[away_team]
        lam_home = np.exp(self.base_rate + self.home_advantage + home.attack - away.defense)
        lam_away = np.exp(self.base_rate + away.attack - home.defense)
        return lam_home, lam_away

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        lam_home, lam_away = self.get_lambdas(home_team, away_team)
        goals = np.arange(0, self.max_goals + 1)
        p_home = poisson.pmf(goals, lam_home)
        p_away = poisson.pmf(goals, lam_away)
        return np.outer(p_home, p_away)

    def result_probabilities(self, home_team: str, away_team: str):
        pmatrix = self.score_matrix(home_team, away_team)
        home_win = np.tril(pmatrix, k=-1).sum()
        draw = np.trace(pmatrix)
        away_win = np.triu(pmatrix, k=1).sum()
        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    def total_goals_probabilities(self, home_team: str, away_team: str):
        pmatrix = self.score_matrix(home_team, away_team)
        totals: Dict[int, float] = {}
        for tg in range(0, 2 * self.max_goals + 1):
            prob = 0.0
            for i in range(0, min(self.max_goals, tg) + 1):
                j = tg - i
                if 0 <= j <= self.max_goals:
                    prob += pmatrix[i, j]
            totals[tg] = prob
        return totals


class BivariatePoissonModel:
    """
    Bivariate Poisson with parameters (lambda1, lambda2, lambda3).

    Home goals = X1 + X3
    Away goals = X2 + X3
    where X1, X2, X3 ~ independent Poisson.
    """
    def __init__(self,
                 lam1: float,
                 lam2: float,
                 lam3: float,
                 max_goals: int = DEFAULT_MODEL_CONFIG.max_goals):
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.max_goals = max_goals

    def pmf(self, x: int, y: int) -> float:
        total = 0.0
        for k in range(0, min(x, y) + 1):
            term = (
                math.exp(-(self.lam1 + self.lam2 + self.lam3))
                * self.lam1 ** (x - k) / math.factorial(x - k)
                * self.lam2 ** (y - k) / math.factorial(y - k)
                * self.lam3 ** k / math.factorial(k)
            )
            total += term
        return total

    def score_matrix(self) -> np.ndarray:
        pm = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                pm[i, j] = self.pmf(i, j)
        return pm

    def result_probabilities(self):
        pmatrix = self.score_matrix()
        home_win = np.tril(pmatrix, k=-1).sum()
        draw = np.trace(pmatrix)
        away_win = np.triu(pmatrix, k=1).sum()
        return {"home_win": home_win, "draw": draw, "away_win": away_win}
