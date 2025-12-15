from __future__ import annotations
from pathlib import Path
import pickle
from datetime import datetime
import re

import pandas as pd
import streamlit as st

from soccer_model.models_poisson import PoissonGoalModel, TeamStrength
from soccer_model.dixon_coles import DixonColesWrapper
from soccer_model.pipeline import evaluate_match_markets


# ----------------------- helpers ----------------------- #

def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def internal_to_display(name: str) -> str:
    return name.replace("_", " ")


def display_to_internal(name: str) -> str:
    return name.strip().replace(" ", "_")


def list_model_versions(models_dir: Path) -> list[str]:
    """
    Return a sorted list of available model files (versioned), plus 'latest'
    if poisson_model_latest.pkl exists.
    """
    pattern = re.compile(r"poisson_model_(\d{8}_\d{6})\.pkl")
    versions = []

    for p in models_dir.glob("poisson_model_*.pkl"):
        m = pattern.match(p.name)
        if m:
            versions.append(p.name)

    versions.sort()  # oldest -> newest

    # Add 'latest' alias if exists
    latest = models_dir / "poisson_model_latest.pkl"
    if latest.exists():
        versions.append("poisson_model_latest.pkl")

    return versions


def load_poisson_model(models_dir: Path, model_file: str) -> PoissonGoalModel:
    """
    Load a PoissonGoalModel from a specific model file name in models_dir.
    """
    model_path = models_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model file not found: {model_path}. "
            f"Train a model with `python -m soccer_model.train_only`."
        )

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
    return poisson_model


def load_teams(matches_csv: Path) -> list[str]:
    df = pd.read_csv(matches_csv)
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
    teams = sorted(str(t) for t in teams)
    return teams


# ----------------------- app ----------------------- #

def main():
    project_root = Path(__file__).resolve().parents[2]  # .../soccer_model
    models_dir = project_root / "models"
    matches_csv = project_root / "data" / "matches.csv"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    st.title("Soccer Betting Model")

    # ----- model version picker -----
    if not models_dir.exists():
        st.error(f"Models directory not found: {models_dir}")
        return

    model_files = list_model_versions(models_dir)
    if not model_files:
        st.error(
            f"No model versions found in {models_dir}. "
            f"Run `python -m soccer_model.train_only` to create one."
        )
        return

    st.sidebar.header("Model Version")
    default_index = len(model_files) - 1  # last one (Newest / 'Latest')
    selected_model_file = st.sidebar.selectbox(
        "Select Model Version",
        options=model_files,
        index=default_index,
    )

    st.sidebar.write(f"Using model: `{selected_model_file}`")

    @st.cache_resource
    def get_dc_model(models_dir_str: str, model_file_str: str):
        md = Path(models_dir_str)
        pm = load_poisson_model(md, model_file_str)
        return DixonColesWrapper(pm)

    dc_model = get_dc_model(str(models_dir), selected_model_file)

    # ----- team list -----
    if not matches_csv.exists():
        st.error(f"matches.csv not found at {matches_csv}")
        return

    teams_internal = load_teams(matches_csv)
    teams_display = [internal_to_display(t) for t in teams_internal]

    st.subheader("Match Selection")

    col1, col2 = st.columns(2)
    with col1:
        home_team_display = st.selectbox("Home Team", teams_display, index=0)
    with col2:
        # avoid defaulting to same team; simple safe index handling
        default_away_index = 1 if len(teams_display) > 1 else 0
        away_team_display = st.selectbox("Away Team", teams_display, index=default_away_index)

    home_team = display_to_internal(home_team_display)
    away_team = display_to_internal(away_team_display)

    # ----- Moneyline (American odds) -----
    st.subheader("Moneyline Odds")

    col1, col2, col3 = st.columns(3)
    with col1:
        home_ml = st.number_input("Home ML Odds", value=-125)
    with col2:
        draw_ml = st.number_input("Draw ML Odds", value=310)
    with col3:
        away_ml = st.number_input("Away ML Odds", value=290)

    odds_home = american_to_decimal(home_ml)
    odds_draw = american_to_decimal(draw_ml)
    odds_away = american_to_decimal(away_ml)

    # ----- Totals (American odds) -----
    st.subheader("Totals")

    col1, col2, col3 = st.columns(3)
    with col1:
        total_line = st.number_input("Total Line", value=2.5)
    with col2:
        over_ml = st.number_input("Over Odds", value=-110)
    with col3:
        under_ml = st.number_input("Under Odds", value=-110)

    odds_over = american_to_decimal(over_ml)
    odds_under = american_to_decimal(under_ml)

    # ----- Asian Handicap / Spread (American odds) -----
    st.subheader("Asian Handicap")

    col1, col2, col3 = st.columns(3)
    with col1:
        ah_line = st.number_input("Line For Favorite", value=-0.5)
    with col2:
        fav_ml = st.number_input("Favorite Odds", value=-110)
    with col3:
        dog_ml = st.number_input("Underdog Odds", value=-110)

    odds_fav = american_to_decimal(fav_ml)
    odds_dog = american_to_decimal(dog_ml)

    home_is_favored = st.checkbox("Home Is Favorite (uncheck if neutral site)", value=True)

    # ----- Evaluate + logging -----
    if st.button("Find The Best Picks"):
        try:
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
        except KeyError as e:
            st.error(
                f"Team {e} not found in model. "
                f"Make sure the team exists in data/matches.csv."
            )
            return

        st.subheader(
            f"Results for {home_team_display} vs {away_team_display} "
            f"(model: {selected_model_file})"
        )

        run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_rows = []

        # Moneyline
        st.markdown("### Moneyline – Probabilities")
        st.write(results["Moneyline"]["Probabilities"])

        st.markdown("### Moneyline – EV (Best First)")
        for b in results["Moneyline"]["Bets"]:
            row = {
                "run_timestamp_utc": run_ts,
                "model_version": selected_model_file,
                "home_team": home_team,
                "away_team": away_team,
                "market": "Moneyline",
                "selection": b.selection,
                "line": "",
                "prob": b.prob,
                "odds_decimal": b.odds_decimal,
                "ev": b.ev,
                "ev_percent": b.ev_percent,
            }
            log_rows.append(row)

            st.write({
                "selection": b.selection,
                "prob": round(b.prob, 3),
                "odds_decimal": round(b.odds_decimal, 3),
                "EV": round(b.ev, 3),
                "EV%": round(b.ev_percent, 1),
            })

        # Totals
        st.markdown("### Totals – EV (Best First)")
        for b in results["totals"]["bets"]:
            row = {
                "run_timestamp_utc": run_ts,
                "model_version": selected_model_file,
                "home_team": home_team,
                "away_team": away_team,
                "market": "Total",
                "selection": b.selection,
                "line": results["totals"]["line"],
                "prob": b.prob,
                "odds_decimal": b.odds_decimal,
                "ev": b.ev,
                "ev_percent": b.ev_percent,
            }
            log_rows.append(row)

            st.write({
                "selection": f"{b.selection} {results['totals']['line']}",
                "prob": round(b.prob, 3),
                "odds_decimal": round(b.odds_decimal, 3),
                "EV": round(b.ev, 3),
                "EV%": round(b.ev_percent, 1),
            })

        # Asian Handicap
        st.markdown("### Asian Handicap – EV (Best First)")
        for b in results["asian"]["bets"]:
            row = {
                "run_timestamp_utc": run_ts,
                "model_version": selected_model_file,
                "home_team": home_team,
                "away_team": away_team,
                "market": "Asian Handicap",
                "selection": b.selection,
                "line": results["asian"]["line"],
                "prob": b.prob,
                "odds_decimal": b.odds_decimal,
                "ev": b.ev,
                "ev_percent": b.ev_percent,
            }
            log_rows.append(row)

            st.write({
                "selection": f"{b.selection} {results['asian']['line']}",
                "prob": round(b.prob, 3),
                "odds_decimal": round(b.odds_decimal, 3),
                "EV": round(b.ev, 3),
                "EV%": round(b.ev_percent, 1),
            })

        # write log file for this UI run
        if log_rows:
            log_df = pd.DataFrame(log_rows)
            ts_file = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            log_path = logs_dir / f"ev_log_app_{ts_file}.csv"
            log_df.to_csv(log_path, index=False)
            st.success(f"Logged EV output to: {log_path}")
        else:
            st.info("Nothing to log.")


if __name__ == "__main__":
    main()
