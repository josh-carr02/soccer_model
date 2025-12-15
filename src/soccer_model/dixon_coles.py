from __future__ import annotations
import numpy as np

from soccer_model.config import DEFAULT_MODEL_CONFIG


class DixonColesWrapper:
    """
    Wrap a goal model and apply Dixon–Coles adjustments to low scorelines.

    The base model must implement:
      - score_matrix(home_team, away_team) -> np.ndarray of shape (max_goals+1, max_goals+1)
    """

    def __init__(self, base_model, rho: float = DEFAULT_MODEL_CONFIG.dixon_coles_rho):
        self.base_model = base_model
        self._rho = rho

    def _correction(self, i: int, j: int) -> float:
        """
        Dixon–Coles correction factor for low scores (0-0, 1-0, 0-1, 1-1).
        """
        if i == 0 and j == 0:
            return 1 - self._rho
        if i == 0 and j == 1:
            return 1 + self._rho
        if i == 1 and j == 0:
            return 1 + self._rho
        if i == 1 and j == 1:
            return 1 - self._rho
        return 1.0

    def adjusted_score_matrix(self, *args, **kwargs) -> np.ndarray:
        """
        Get the base score matrix and apply Dixon–Coles corrections, then renormalize.
        """
        if not hasattr(self.base_model, "score_matrix"):
            raise ValueError("Base model must implement score_matrix().")

        pmatrix = self.base_model.score_matrix(*args, **kwargs)
        adjusted = np.zeros_like(pmatrix)

        for i in range(pmatrix.shape[0]):
            for j in range(pmatrix.shape[1]):
                corr = self._correction(i, j)
                adjusted[i, j] = pmatrix[i, j] * corr

        total = adjusted.sum()
        if total > 0:
            adjusted /= total
        return adjusted

    def result_probabilities(self, *args, **kwargs):
        """
        Return probabilities of home win, draw, away win based on adjusted matrix.
        """
        pmatrix = self.adjusted_score_matrix(*args, **kwargs)
        home_win = np.tril(pmatrix, k=-1).sum()
        draw = np.trace(pmatrix)
        away_win = np.triu(pmatrix, k=1).sum()
        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    def total_goals_probabilities(self, *args, **kwargs):
        """
        Return probability distribution over total goals.
        """
        pmatrix = self.adjusted_score_matrix(*args, **kwargs)
        max_goals = pmatrix.shape[0] - 1
        totals: dict[int, float] = {}

        for tg in range(0, 2 * max_goals + 1):
            prob = 0.0
            for i in range(0, min(max_goals, tg) + 1):
                j = tg - i
                if 0 <= j <= max_goals:
                    prob += pmatrix[i, j]
            totals[tg] = prob
        return totals
