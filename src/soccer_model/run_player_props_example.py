from __future__ import annotations
from pathlib import Path
import pandas as pd

from soccer_model.player_props import compute_player_rates, player_goal_prop_ev
from soccer_model.betting import BetResult


def main():
    stats_path = Path("data/player_match_stats.csv")
    odds_path = Path("data/player_odds.csv")

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing player stats file: {stats_path}")
    if not odds_path.exists():
        raise FileNotFoundError(f"Missing player odds file: {odds_path}")

    player_stats = pd.read_csv(stats_path)
    player_odds = pd.read_csv(odds_path)

    rates = compute_player_rates(player_stats)

    results: list[BetResult] = []

    for _, row in player_odds.iterrows():
        player_id = row["player_id"]
        player_name = row.get("player_name", player_id)
        odds_anytime = float(row["odds_anytime"])

        if "minutes_expected" in row and not pd.isna(row["minutes_expected"]):
            minutes_expected = float(row["minutes_expected"])
        else:
            minutes_expected = 90.0

        if player_id not in rates:
            continue

        rate = rates[player_id]
        bet_result = player_goal_prop_ev(
            rate,
            odds_anytime=odds_anytime,
            minutes_expected=minutes_expected,
        )

        bet_result.selection = player_name
        results.append(bet_result)

    if not results:
        print("No overlap between stats and odds.")
        return

    results.sort(key=lambda br: br.ev, reverse=True)
    top_n = 10

    print(f"\nTop {top_n} Anytime Goalscorer EV bets:\n")
    for br in results[:top_n]:
        print(
            f"{br.selection}: "
            f"p(score)={br.prob:.3f}, "
            f"odds={br.odds_decimal:.2f}, "
            f"EV={br.ev:.3f} ({br.ev_percent:.1f}%)"
        )


if __name__ == "__main__":
    main()
