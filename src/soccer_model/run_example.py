from __future__ import annotations
from pathlib import Path

import pandas as pd

from soccer_model.pipeline import build_models, evaluate_match_markets


def american_to_decimal(american_odds: float) -> float:
    """
    Convert American odds to decimal odds.

    Examples:
      -125 -> 1 + 100/125 = 1.80
      +200 -> 1 + 200/100 = 3.00
    """
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    else:
        return 1.0 + 100.0 / abs(american_odds)


def main():
    matches_csv = Path("data/matches.csv")
    odds_csv = Path("data/odds.csv")

    # Fit models on all matches in the CSV
    matches, bayes_model, poisson_model, dc_model = build_models(matches_csv)

    if matches.empty:
        raise RuntimeError("matches.csv is empty – add at least one match row.")
    if not odds_csv.exists():
        raise FileNotFoundError(f"odds.csv not found at {odds_csv}")

    odds_df = pd.read_csv(odds_csv)

    print("Evaluating all matches in matches.csv with per-game odds from odds.csv\n")

    for _, row in matches.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        # Find the corresponding odds row for this matchup
        mask = (odds_df["home_team"] == home_team) & (odds_df["away_team"] == away_team)
        if not mask.any():
            print(f"WARNING: No odds found for {home_team} vs {away_team}, skipping.\n")
            continue

        o = odds_df.loc[mask].iloc[0]

        # Moneyline odds (American -> decimal)
        home_ml = float(o["home_ml"])
        away_ml = float(o["away_ml"])
        draw_ml = float(o["draw"])

        odds_home = american_to_decimal(home_ml)
        odds_away = american_to_decimal(away_ml)
        odds_draw = american_to_decimal(draw_ml)

        # Totals odds and line from CSV
        total_line = float(o["total"])
        odds_over = float(o["over_odds"])
        odds_under = float(o["under_odds"])

        # Asian handicap line and odds from CSV
        ah_line = float(o["ah_line"])
        odds_fav = float(o["ah_fav_odds"])
        odds_dog = float(o["ah_dog_odds"])
        home_is_favored = True  # can be extended with a column if needed

        results = evaluate_match_markets(
            dc_model,
            home_team,
            away_team,
            odds_home,
            odds_draw,
            odds_away,
            total_line,
            odds_over,
            odds_under,
            ah_line,
            odds_fav,
            odds_dog,
            home_is_favored,
        )

        print(f"Match: {home_team} vs {away_team}\n")

        print("  1X2 probabilities:")
        for k, v in results["1x2"]["probs"].items():
            print(f"    {k}: {v:.3f}")

        print("\n  1X2 EV (best to worst):")
        for b in results["1x2"]["bets"]:
            print(
                f"    {b.selection}: p={b.prob:.3f}, "
                f"odds={b.odds_decimal:.3f}, "
                f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)"
            )

        print("\n  Totals EV (best to worst):")
        for b in results["totals"]["bets"]:
            print(
                f"    {b.selection} {results['totals']['line']}: "
                f"p={b.prob:.3f}, odds={b.odds_decimal:.3f}, "
                f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)"
            )

        print("\n  Asian Handicap EV (best to worst):")
        for b in results["asian"]["bets"]:
            print(
                f"    {b.selection} {results['asian']['line']}: "
                f"p={b.prob:.3f}, odds={b.odds_decimal:.3f}, "
                f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)"
            )

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
