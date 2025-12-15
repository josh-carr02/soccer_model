from __future__ import annotations
from pathlib import Path
import pickle
from datetime import datetime
import re
import os

import pandas as pd
import requests
import streamlit as st

from soccer_model.models_poisson import PoissonGoalModel, TeamStrength
from soccer_model.dixon_coles import DixonColesWrapper
from soccer_model.pipeline import evaluate_match_markets
from soccer_model import train_only  # we will call train_only.main()


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

    latest = models_dir / "poisson_model_latest.pkl"
    if latest.exists():
        versions.append("poisson_model_latest.pkl")

    return versions


def load_poisson_model(models_dir: Path, model_file: str) -> PoissonGoalModel:
    model_path = models_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model file not found: {model_path}. "
            f"Run `python -m soccer_model.train_only` first."
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


def load_model_index(models_dir: Path) -> pd.DataFrame | None:
    index_path = models_dir / "model_index.csv"
    if index_path.exists():
        return pd.read_csv(index_path)
    return None


def get_secrets():
    # two places: Streamlit Cloud secrets or environment variables (local)
    api_key = ""
    base_url = ""

    try:
        api_key = st.secrets.get("5a45b2088c3c95f5ed5e6d43d5b9deec", "")
        base_url = st.secrets.get("https://api.the-odds-api.com/v4", "")
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("5a45b2088c3c95f5ed5e6d43d5b9deec", "")

    if not base_url:
        base_url = os.environ.get("https://api.the-odds-api.com/v4", "")

    return api_key, base_url


def fetch_live_odds_from_api(
    api_key: str,
    base_url: str,
    sport_key: str,
    home_team_display: str,
    away_team_display: str,
    region: str = "us",
) -> dict | None:
    """
    Example live-odds fetch using a generic REST API.
    This is written in a way that is compatible with providers like The Odds API.
    You MUST adapt sport_key, base_url, and name matching to your provider.

    Returns a dict with keys:
      'home_ml', 'away_ml', 'draw_ml', 'total', 'over_ml', 'under_ml'
    or None if nothing found.
    """
    if not api_key or not base_url:
        return None

    # Example for "h2h" (moneyline) odds
    # You must adapt 'soccer_epl' and parameters for your provider
    try:
        # Moneyline
        h2h_url = f"{base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": region,
            "markets": "h2h,totals",
            "oddsFormat": "american",
        }
        resp = requests.get(h2h_url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    # Try to find the specific match
    home_lower = home_team_display.lower()
    away_lower = away_team_display.lower()

    match_obj = None
    for game in data:
        # These keys vary across APIs; adapt as needed.
        teams = [t.lower() for t in game.get("teams", [])]
        home_team_api = str(game.get("home_team", "")).lower()
        if home_lower in home_team_api and away_lower in " ".join(teams):
            match_obj = game
            break
        if away_lower in home_team_api and home_lower in " ".join(teams):
            # Might be flipped; still treat as found
            match_obj = game
            break

    if not match_obj:
        return None

    # Parse markets from the match object. This is API-specific.
    home_ml = away_ml = draw_ml = None
    total_line = None
    over_ml = under_ml = None

    bookmakers = match_obj.get("bookmakers", [])
    if not bookmakers:
        return None

    # Just take the first bookmaker for now
    bm = bookmakers[0]
    markets = bm.get("markets", [])

    for m in markets:
        mkey = m.get("key")
        outcomes = m.get("outcomes", [])
        if mkey == "h2h":
            # moneyline market
            for o in outcomes:
                name = str(o.get("name", "")).lower()
                price = o.get("price")
                if price is None:
                    continue
                if home_lower in name:
                    home_ml = price
                elif away_lower in name:
                    away_ml = price
                elif "draw" in name or "tie" in name:
                    draw_ml = price
        elif mkey == "totals":
            # totals market
            for o in outcomes:
                name = str(o.get("name", "")).lower()
                price = o.get("price")
                point = o.get("point")
                if price is None or point is None:
                    continue
                total_line = float(point)
                if "over" in name:
                    over_ml = price
                elif "under" in name:
                    under_ml = price

    if home_ml is None or away_ml is None:
        return None

    # Some markets do not support draw (knockout, etc.)
    result = {
        "home_ml": home_ml,
        "away_ml": away_ml,
        "draw_ml": draw_ml,
        "total": total_line,
        "over_ml": over_ml,
        "under_ml": under_ml,
    }
    return result


# ----------------------- app ----------------------- #

def main():
    project_root = Path(__file__).resolve().parents[2]  # .../soccer_model
    models_dir = project_root / "models"
    matches_csv = project_root / "data" / "matches.csv"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    st.title("Soccer Betting Model – Train & Predict - The Book Beaters")

    tab_train, tab_predict = st.tabs(["Train Model", "Predict & Pick"])

    # -------------------- TRAIN TAB -------------------- #
    with tab_train:
        st.subheader("Offline Training (Bayesian xG → Poisson)")

        if not matches_csv.exists():
            st.error(f"matches.csv not found at {matches_csv}")
        else:
            matches_df = pd.read_csv(matches_csv)
            st.write(f"Matches in training set: {len(matches_df)}")
            st.dataframe(matches_df.head())

        st.markdown("#### Current Model Versions")
        if models_dir.exists():
            index_df = load_model_index(models_dir)
            if index_df is not None and not index_df.empty:
                st.dataframe(index_df.tail(10))
            else:
                st.info("No model_index.csv yet – train at least once.")
        else:
            st.info("Models directory does not exist yet.")

        if st.button("Train New Model Now"):
            with st.spinner("Running Bayesian Training via train_only.main()..."):
                # This calls your existing training script (versioning + index)
                train_only.main()
            st.success("Training Complete. Model Updated.")
            # Clear cached models in prediction tab
            st.cache_resource.clear()

    # -------------------- PREDICT TAB -------------------- #
    with tab_predict:
        st.subheader("Predict Markets & Calculate EV")

        # --- Model Version Selection ---
        if not models_dir.exists():
            st.error(f"Models directory not found: {models_dir}")
            return

        model_files = list_model_versions(models_dir)
        if not model_files:
            st.error(
                f"No model versions found in {models_dir}. "
                f"Run training in the 'Train model' tab or locally with train_only."
            )
            return

        default_index = len(model_files) - 1  # last
        selected_model_file = st.sidebar.selectbox(
            "Model Version",
            options=model_files,
            index=default_index,
        )
        st.sidebar.write(f"Using Model: `{selected_model_file}`")

        @st.cache_resource
        def get_dc_model(models_dir_str: str, model_file_str: str):
            md = Path(models_dir_str)
            pm = load_poisson_model(md, model_file_str)
            return DixonColesWrapper(pm)

        dc_model = get_dc_model(str(models_dir), selected_model_file)

        # --- Team selection ---
        if not matches_csv.exists():
            st.error(f"matches.csv not found at {matches_csv}")
            return

        teams_internal = load_teams(matches_csv)
        teams_display = [internal_to_display(t) for t in teams_internal]

        st.markdown("#### Match Selection")
        col1, col2 = st.columns(2)
        with col1:
            home_team_display = st.selectbox("Home Team", teams_display, index=0)
        with col2:
            default_away_index = 1 if len(teams_display) > 1 else 0
            away_team_display = st.selectbox(
                "Away Team", teams_display, index=default_away_index
            )

        home_team = display_to_internal(home_team_display)
        away_team = display_to_internal(away_team_display)

        # --- Live odds API settings (optional) ---
        st.markdown("#### Live Odds API")
        api_key_default, base_url_default = get_secrets()

        col_api1, col_api2, col_api3 = st.columns(3)
        with col_api1:
            api_key_input = st.text_input(
                "API key", value=api_key_default, type="password"
            )
        with col_api2:
            sport_key = st.text_input("Sport Key", value="soccer_epl")
        with col_api3:
            region = st.text_input("Region", value="us")

        live_odds = None
        if st.button("Gather Live Odds For This Match"):
            if not api_key_input or not base_url_default:
                st.warning("No API key/base URL configured. Set ODDS_API_KEY/ODDS_API_BASE or enter them.")
            else:
                with st.spinner("Gathering Live Odds..."):
                    live_odds = fetch_live_odds_from_api(
                        api_key_input,
                        base_url_default,
                        sport_key,
                        home_team_display,
                        away_team_display,
                        region=region,
                    )
                if live_odds is None:
                    st.warning("No Live Odds Found For This Match")
                else:
                    st.success("Live Odds Gathered")
                    st.json(live_odds)

        # --- Moneyline ---
        st.markdown("#### Moneyline")

        default_home_ml = -125
        default_away_ml = 290
        default_draw_ml = 310

        if live_odds:
            if live_odds.get("home_ml") is not None:
                default_home_ml = live_odds["home_ml"]
            if live_odds.get("away_ml") is not None:
                default_away_ml = live_odds["away_ml"]
            if live_odds.get("draw_ml") is not None:
                default_draw_ml = live_odds["draw_ml"]

        col1, col2, col3 = st.columns(3)
        with col1:
            home_ml = st.number_input("Home ML Odds", value=float(default_home_ml))
        with col2:
            draw_ml = st.number_input("Draw ML Odds", value=float(default_draw_ml))
        with col3:
            away_ml = st.number_input("Away ML Odds", value=float(default_away_ml))

        odds_home = american_to_decimal(home_ml)
        odds_draw = american_to_decimal(draw_ml)
        odds_away = american_to_decimal(away_ml)

        # --- Totals ---
        st.markdown("#### Totals")

        default_total_line = 2.5
        default_over_ml = -110
        default_under_ml = -110

        if live_odds:
            if live_odds.get("total") is not None:
                default_total_line = live_odds["total"]
            if live_odds.get("over_ml") is not None:
                default_over_ml = live_odds["over_ml"]
            if live_odds.get("under_ml") is not None:
                default_under_ml = live_odds["under_ml"]

        col1, col2, col3 = st.columns(3)
        with col1:
            total_line = st.number_input("Total Line", value=float(default_total_line))
        with col2:
            over_ml = st.number_input("Over Odds", value=float(default_over_ml))
        with col3:
            under_ml = st.number_input("Under Odds", value=float(default_under_ml))

        odds_over = american_to_decimal(over_ml)
        odds_under = american_to_decimal(under_ml)

        # --- Asian Handicap ---
        st.markdown("#### Asian Handicap")

        col1, col2, col3 = st.columns(3)
        with col1:
            ah_line = st.number_input("Line For Favorite", value=-0.5)
        with col2:
            fav_ml = st.number_input("Favorite Odds", value=-110.0)
        with col3:
            dog_ml = st.number_input("Underdog Odds", value=-110.0)

        odds_fav = american_to_decimal(fav_ml)
        odds_dog = american_to_decimal(dog_ml)

        home_is_favored = st.checkbox("Home Is Favored (uncheck if played at neutral site)", value=True)

        # --- Evaluate & log ---
        if st.button("Run Through Model and See Best Bets"):
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

            # logging to CSV
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
