from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class BetResult:
    market: str
    selection: str
    prob: float
    odds_decimal: float
    ev: float
    ev_percent: float


def decimal_odds_to_implied_prob(odds: float) -> float:
    return 1.0 / odds


def expected_value(prob_win: float, odds: float) -> float:
    """
    EV per 1 unit stake.

    Win -> profit = odds - 1
    Lose -> profit = -1
    EV = p*(odds-1) + (1-p)*(-1)
    """
    return prob_win * (odds - 1.0) - (1.0 - prob_win)


def ev_percent(ev: float) -> float:
    return ev * 100.0


def make_bet_result(market: str, selection: str, prob: float, odds: float) -> BetResult:
    ev_val = expected_value(prob, odds)
    return BetResult(
        market=market,
        selection=selection,
        prob=prob,
        odds_decimal=odds,
        ev=ev_val,
        ev_percent=ev_percent(ev_val),
    )

# ----------------------------------------------------------------------
# 1X2 market
# ----------------------------------------------------------------------


def best_1x2_ev(result_probs: Dict[str, float],
                odds_home: float,
                odds_draw: float,
                odds_away: float):
    bets = [
        make_bet_result("1X2", "Home", result_probs["home_win"], odds_home),
        make_bet_result("1X2", "Draw", result_probs["draw"], odds_draw),
        make_bet_result("1X2", "Away", result_probs["away_win"], odds_away),
    ]
    return sorted(bets, key=lambda b: b.ev, reverse=True)

# ----------------------------------------------------------------------
# Totals (Over/Under) for half-goal lines
# ----------------------------------------------------------------------


def over_under_probabilities(total_goals_probs: Dict[int, float],
                             line: float) -> Tuple[float, float]:
    """
    Given P(TG = k), compute P(Over line) and P(Under line)
    for a half-goal line (e.g. 2.5).

    Over 2.5 -> TG >= 3
    """
    threshold = int(np.floor(line)) + 1
    p_over = sum(prob for tg, prob in total_goals_probs.items() if tg >= threshold)
    p_under = 1.0 - p_over
    return p_over, p_under


def best_total_ev(total_goals_probs: Dict[int, float],
                  line: float,
                  odds_over: float,
                  odds_under: float):
    p_over, p_under = over_under_probabilities(total_goals_probs, line)
    bets = [
        make_bet_result(f"Total {line}", "Over", p_over, odds_over),
        make_bet_result(f"Total {line}", "Under", p_under, odds_under),
    ]
    return sorted(bets, key=lambda b: b.ev, reverse=True)

# ----------------------------------------------------------------------
# Asian handicap (simple, single-line)
# ----------------------------------------------------------------------


def asian_handicap_probabilities(score_matrix: np.ndarray,
                                 handicap: float,
                                 home_is_favored: bool = True):
    """
    Compute win/push/lose probabilities for an Asian handicap line (full/half).

    score_matrix[i,j] = P(Home=i, Away=j)
    handicap is applied to the favored side's goal difference.
    """
    max_goals = score_matrix.shape[0] - 1
    p_fav_win = 0.0
    p_push = 0.0
    p_dog_win = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            diff = (i - j) if home_is_favored else (j - i)
            prob = score_matrix[i, j]
            result = diff + handicap
            if result > 0:
                p_fav_win += prob
            elif result == 0:
                p_push += prob
            else:
                p_dog_win += prob

    return {
        "fav_win": p_fav_win,
        "push": p_push,
        "dog_win": p_dog_win,
    }


def asian_ev(prob_win: float, prob_push: float, odds: float) -> float:
    """
    EV per 1 unit stake with possibility of push (refund).
    """
    p_lose = 1.0 - prob_win - prob_push
    return prob_win * (odds - 1.0) - p_lose


def best_asian_ev(score_matrix: np.ndarray,
                  handicap: float,
                  odds_fav: float,
                  odds_dog: float,
                  home_is_favored: bool = True):
    probs = asian_handicap_probabilities(score_matrix, handicap,
                                         home_is_favored=home_is_favored)

    ev_fav = asian_ev(probs["fav_win"], probs["push"], odds_fav)
    ev_dog = asian_ev(probs["dog_win"], probs["push"], odds_dog)

    fav_bet = BetResult(
        market=f"AH {handicap}",
        selection="Favorite",
        prob=probs["fav_win"],
        odds_decimal=odds_fav,
        ev=ev_fav,
        ev_percent=ev_percent(ev_fav),
    )
    dog_bet = BetResult(
        market=f"AH {handicap}",
        selection="Underdog",
        prob=probs["dog_win"],
        odds_decimal=odds_dog,
        ev=ev_dog,
        ev_percent=ev_percent(ev_dog),
    )

    return sorted([fav_bet, dog_bet], key=lambda b: b.ev, reverse=True)
