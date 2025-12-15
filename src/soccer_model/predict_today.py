from __future__ import annotations
from pathlib import Path
import pickle
from datetime import datetime
import re

import pandas as pd

from soccer_model.models_poisson import PoissonGoalModel, TeamStrength
from soccer_model.dixon_coles import DixonColesWrapper
from soccer_model.pipeline import evaluate_match_markets


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def find_latest_model(models_dir: Path) -> Path:
    """
    Find the latest versioned model file named 'poisson_model_YYYYMMDD_HHMMSS.pkl'.
    Fallback to 'poisson_model_latest.pkl' if no versioned files exist.
    """
    pattern = re.compile(r"poisson_model_(\d{8}_\d{6})\.pkl")

    version_files = []
    for p in models_dir.glob("poisson_model_*.pkl"):
        m = pattern.match(p.name)
        if m:
            version_files.append((m.group(1), p))

    if version_files:
        # sort by timestamp string; lexicographic equals chronological here
        version_files.sort(key=lambda x: x[0])
        latest_ts, latest_path = version_files[-1]
        return latest_path

    # fallback
    latest_path = models_dir / "poisson_model_latest.pkl"
    if latest_path.exists():
        return latest_path

    raise FileNotFoundError(
        f"No model files found in {models_dir}. "
        f"Run `python -m soccer_model.train_only` first."
    )


def load_poisson_model(project_root: Path) -> tuple[PoissonGoalModel, str]:
    models_dir = project_root / "models"
    model_path = find_latest_model(models_dir)

    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    team_strengths = {
        team: TeamStrength(attack=v["attack"], defense=v["defense"])
        for team, v in model_data["team_strengths"].items()
    }

    poisson_model = PoissonGoalModel(
        base_rate=model_data["base_rate"],
        home_advantage=model_data["home_advantage"],
        team_strengths=team_strengths,
        max_goals=model_data["max_goals"],
    )
    return poisson_model, model_path.name


def main():
    project_root = Path(__file__).resolve().parents[2]  # .../soccer_model
    odds_csv = project_root / "data" / "odds.csv"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    print(f"Loading saved model...")
    poisson_model, model_version = load_poisson_model(project_root)
    dc_model = DixonColesWrapper(poisson_model)
    print(f"Using model version: {model_version}")

    print(f"Loading odds from: {odds_csv}")
    odds_df = pd.read_csv(odds_csv)

    required_cols = [
        "home_team", "away_team",
        "home_ml", "away_ml", "draw_ml",
        "total", "over_ml", "under_ml",
        "ah_line", "ah_fav_ml", "ah_dog_ml",
    ]
    missing = [c for c in required_cols if c not in odds_df.columns]
    if missing:
        raise ValueError(f"Missing columns in odds.csv: {missing}")

    run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    log_rows = []

    print("\nEvaluating EV for all matches in odds.csv using saved model:\n")

    for _, row in odds_df.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        # Moneyline American -> decimal
        home_ml = float(row["home_ml"])
        away_ml = float(row["away_ml"])
        draw_ml = float(row["draw_ml"])

        odds_home = american_to_decimal(home_ml)
        odds_away = american_to_decimal(away_ml)
        odds_draw = american_to_decimal(draw_ml)

        # Totals
        total_line = float(row["total"])
        over_ml = float(row["over_ml"])
        under_ml = float(row["under_ml"])
        odds_over = american_to_decimal(over_ml)
        odds_under = american_to_decimal(under_ml)

        # Asian handicap
        ah_line = float(row["ah_line"])
        ah_fav_ml = float(row["ah_fav_ml"])
        ah_dog_ml = float(row["ah_dog_ml"])
        odds_fav = american_to_decimal(ah_fav_ml)
        odds_dog = american_to_decimal(ah_dog_ml)
        home_is_favored = True  # can add a column later if needed

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

        # Moneyline
        print("  Moneyline probabilities:")
        for k, v in results["1x2"]["probs"].items():
            print(f"    {k}: {v:.3f}")

        print("\n  Moneyline EV (best to worst):")
        for b in results["1x2"]["bets"]:
            print(
                f"    {b.selection}: p={b.prob:.3f}, "
                f"odds_decimal={b.odds_decimal:.3f}, "
                f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)"
            )
            log_rows.append({
                "run_timestamp_utc": run_ts,
                "model_version": model_version,
                "home_team": home_team,
                "away_team": away_team,
                "market": "Moneyline",
                "selection": b.selection,
                "line": "",
                "prob": b.prob,
                "odds_decimal": b.odds_decimal,
                "ev": b.ev,
                "ev_percent": b.ev_percent,
            })

        # Totals
        print("\n  Totals EV (best to worst):")
        for b in results["totals"]["bets"]:
            print(
                f"    {b.selection} {results['totals']['line']}: "
                f"p={b.prob:.3f}, odds_decimal={b.odds_decimal:.3f}, "
                f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)"
            )
            log_rows.append({
                "run_timestamp_utc": run_ts,
                "model_version": model_version,
                "home_team": home_team,
                "away_team": away_team,
                "market": "Total",
                "selection": b.selection,
                "line": results["totals"]["line"],
                "prob": b.prob,
                "odds_decimal": b.odds_decimal,
                "ev": b.ev,
                "ev_percent": b.ev_percent,
            })

        # Asian Handicap
        print("\n  Asian Handicap EV (best to worst):")
        for b in results["asian"]["bets"]:
            print(
                f"    {b.selection} {results['asian']['line']}: "
                f"p={b.prob:.3f}, odds_decimal={b.odds_decimal:.3f}, "
                f"EV={b.ev:.3f} ({b.ev_percent:.1f}%)"
            )
            log_rows.append({
                "run_timestamp_utc": run_ts,
                "model_version": model_version,
                "home_team": home_team,
                "away_team": away_team,
                "market": "Asian Handicap",
                "selection": b.selection,
                "line": results["asian"]["line"],
                "prob": b.prob,
                "odds_decimal": b.odds_decimal,
                "ev": b.ev,
                "ev_percent": b.ev_percent,
            })

        print("\n" + "-" * 60 + "\n")

    # ----- write log file -----
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        ts_file = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"ev_log_{ts_file}.csv"
        log_df.to_csv(log_path, index=False)
        print(f"Saved EV log to: {log_path}")
    else:
        print("No rows to log.")


if __name__ == "__main__":
    main()
