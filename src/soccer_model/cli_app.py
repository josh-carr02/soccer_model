from __future__ import annotations
import argparse
from pathlib import Path

from soccer_model.pipeline import build_models, evaluate_match_markets


def main():
    parser = argparse.ArgumentParser(description="Soccer betting model CLI")
    parser.add_argument("--matches_csv", type=str, default="data/matches.csv",
                        help="Path to matches CSV")
    parser.add_argument("--home_team", type=str, required=True)
    parser.add_argument("--away_team", type=str, required=True)

    parser.add_argument("--odds_home", type=float, required=True)
    parser.add_argument("--odds_draw", type=float, required=True)
    parser.add_argument("--odds_away", type=float, required=True)

    parser.add_argument("--total_line", type=float, default=2.5)
    parser.add_argument("--odds_over", type=float, required=True)
    parser.add_argument("--odds_under", type=float, required=True)

    parser.add_argument("--ah_line", type=float, default=-0.5)
    parser.add_argument("--odds_fav", type=float, required=True)
    parser.add_argument("--odds_dog", type=float, required=True)
    parser.add_argument("--home_is_favored", action="store_true", default=True)

    args = parser.parse_args()

    matches_csv = Path(args.matches_csv)
    _, _, _, dc_model = build_models(matches_csv)

    results = evaluate_match_markets(
        dc_model,
        args.home_team,
        args.away_team,
        args.odds_home,
        args.odds_draw,
        args.odds_away,
        args.total_line,
        args.odds_over,
        args.odds_under,
        args.ah_line,
        args.odds_fav,
        args.odds_dog,
        args.home_is_favored,
    )

    print(f"\nMatch: {args.home_team} vs {args.away_team}\n")

    print("1X2 probabilities:")
    for k, v in results["1x2"]["probs"].items():
        print(f"  {k}: {v:.3f}")

    print("\nTop 1X2 EV bets:")
    for b in results["1x2"]["bets"]:
        print(f"  {b.selection}: p={b.prob:.3f}, odds={b.odds_decimal:.2f}, "
              f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)")

    print("\nTop Totals EV bets:")
    for b in results["totals"]["bets"]:
        print(f"  {b.selection} {results['totals']['line']}: "
              f"p={b.prob:.3f}, odds={b.odds_decimal:.2f}, "
              f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)")

    print("\nTop Asian Handicap EV bets:")
    for b in results["asian"]["bets"]:
        print(f"  {b.selection} {results['asian']['line']}: "
              f"p={b.prob:.3f}, odds={b.odds_decimal:.2f}, "
              f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)")


if __name__ == "__main__":
    main()
