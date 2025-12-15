from __future__ import annotations
import pandas as pd
from pathlib import Path


def load_matches(csv_path: str | Path) -> pd.DataFrame:
    """
    Load match-level data from CSV.

    Required columns:
      date, home_team, away_team, home_goals, away_goals, home_xg, away_xg
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Matches CSV not found: {path}")

    df = pd.read_csv(path)
    required_cols = [
        "date", "home_team", "away_team",
        "home_goals", "away_goals",
        "home_xg", "away_xg",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in matches data: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    return df


def load_events(csv_path: str | Path) -> pd.DataFrame:
    """
    Load event-level data for xT (passes, shots).

    Required columns:
      match_id, team, x_start, y_start, x_end, y_end, event_type
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Events CSV not found: {path}")

    df = pd.read_csv(path)
    required_cols = [
        "match_id", "team",
        "x_start", "y_start",
        "x_end", "y_end",
        "event_type",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in events data: {missing}")
    return df
