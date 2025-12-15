from __future__ import annotations
from pathlib import Path

from soccer_model.data_loading import load_matches
from soccer_model.models_bayesian import BayesianXGModel
from soccer_model.models_poisson import PoissonGoalModel
from soccer_model.dixon_coles import DixonColesWrapper
from soccer_model.betting import best_1x2_ev, best_total_ev, best_asian_ev


def build_models(matches_csv: str | Path):
    """
    Load match data, fit Bayesian xG model, convert to Poisson model,
    and wrap with Dixon–Coles.
    """
    matches = load_matches(matches_csv)
    bayes_model = BayesianXGModel()
    bayes_model.fit(matches)
    poisson_model: PoissonGoalModel = bayes_model.to_poisson_model()
    dc_model = DixonColesWrapper(poisson_model)
    return matches, bayes_model, poisson_model, dc_model


def evaluate_match_markets(
    dc_model: DixonColesWrapper,
    home_team: str,
    away_team: str,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    total_line: float,
    odds_over: float,
    odds_under: float,
    ah_line: float,
    odds_fav: float,
    odds_dog: float,
    home_is_favored: bool = True,
):
    """
    Evaluate Moneyline (formerly 1X2), totals, and Asian handicap markets for a given match.

    Returns a dict with keys:
      - "moneyline": {"probs": ..., "bets": [...]}
      - "totals":    {"line": total_line, "bets": [...]}
      - "asian":     {"line": ah_line,    "bets": [...]}
    """
    # Moneyline Probabilities, Adjusted by Dixon–Coles
    probs_1x2 = dc_model.result_probabilities(home_team, away_team)
    bets_1x2 = best_1x2_ev(probs_1x2, odds_home, odds_draw, odds_away)

    # Totals probabilities
    totals_probs = dc_model.total_goals_probabilities(home_team, away_team)
    bets_totals = best_total_ev(totals_probs, total_line, odds_over, odds_under)

    # Asian handicap using adjusted score matrix
    pmatrix = dc_model.adjusted_score_matrix(home_team, away_team)
    bets_asian = best_asian_ev(
        pmatrix,
        ah_line,
        odds_fav,
        odds_dog,
        home_is_favored=home_is_favored,
    )

    return {
        "moneyline": {
            "probs": probs_1x2,
            "bets": bets_1x2,
        },
        "totals": {
            "line": total_line,
            "bets": bets_totals,
        },
        "asian": {
            "line": ah_line,
            "bets": bets_asian,
        },
    }
