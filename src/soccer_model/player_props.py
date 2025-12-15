from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from scipy.stats import poisson

from soccer_model.betting import BetResult, expected_value, ev_percent


@dataclass
class PlayerRates:
    xg_per_90: float
    xT_per_90: float
    lambda_goals: float


def compute_player_rates(player_match_stats: pd.DataFrame,
                         xg_col: str = "xg",
                         xT_col: str = "xT_added",
                         minutes_col: str = "minutes",
                         player_col: str = "player_id",
                         xt_weight: float = 0.5) -> Dict[str, PlayerRates]:
    """
    Compute per-player scoring rates from historical data.

    For each player:
      xg_per_90 = total_xg / total_minutes * 90
      xT_per_90 = total_xT / total_minutes * 90
      lambda_goals = xg_per_90 + xt_weight * xT_per_90
    """
    df = player_match_stats.copy()

    grouped = df.groupby(player_col).agg(
        total_xg=(xg_col, "sum"),
        total_xT=(xT_col, "sum"),
        total_minutes=(minutes_col, "sum"),
    )

    grouped = grouped[grouped["total_minutes"] > 0]

    grouped["xg_per_90"] = grouped["total_xg"] / grouped["total_minutes"] * 90.0
    grouped["xT_per_90"] = grouped["total_xT"] / grouped["total_minutes"] * 90.0

    grouped["lambda_goals"] = grouped["xg_per_90"] + xt_weight * grouped["xT_per_90"]

    rates: Dict[str, PlayerRates] = {}
    for pid, row in grouped.iterrows():
        lam = max(row["lambda_goals"], 1e-6)
        rates[pid] = PlayerRates(
            xg_per_90=float(row["xg_per_90"]),
            xT_per_90=float(row["xT_per_90"]),
            lambda_goals=float(lam),
        )
    return rates


def goal_prop_probabilities(player_rate: PlayerRates,
                            minutes_expected: float = 90.0):
    """
    Probability distribution for goals using Poisson.

    Scale lambda by expected minutes for the match:
      lambda_effective = lambda_goals * (minutes_expected / 90)
    """
    lam = player_rate.lambda_goals * (minutes_expected / 90.0)

    p0 = poisson.pmf(0, lam)
    p1 = poisson.pmf(1, lam)
    p2plus = 1.0 - p0 - p1
    return {"p0": p0, "p1": p1, "p2plus": p2plus, "lambda": lam}


def player_goal_prop_ev(player_rate: PlayerRates,
                        odds_anytime: float,
                        minutes_expected: float = 90.0) -> BetResult:
    """
    EV for 'Anytime Goalscorer'.

    p_score = 1 - P(0 goals) under Poisson model.
    """
    probs = goal_prop_probabilities(player_rate, minutes_expected=minutes_expected)
    p_score = 1.0 - probs["p0"]
    ev_val = expected_value(p_score, odds_anytime)

    return BetResult(
        market="Anytime Goalscorer",
        selection="Player",
        prob=p_score,
        odds_decimal=odds_anytime,
        ev=ev_val,
        ev_percent=ev_percent(ev_val),
    )
