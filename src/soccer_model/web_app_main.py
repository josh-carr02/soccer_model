from __future__ import annotations

import os
import re
from datetime import datetime, date
from pathlib import Path
import pickle
from typing import Tuple, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

from soccer_model.models_poisson import PoissonGoalModel, TeamStrength
from soccer_model.dixon_coles import DixonColesWrapper
from soccer_model.pipeline import evaluate_match_markets
from soccer_model import train_only


# ----------------------- PAGE CONFIG ----------------------- #

st.set_page_config(
    page_title="Soccer Betting Model – Train & Predict",
    page_icon="⚽",
    layout="wide",
)


# ----------------------- HELPERS ----------------------- #

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def internal_to_display(name: str) -> str:
    return name.replace("_", " ")


def display_to_internal(name: str) -> str:
    return name.strip().replace(" ", "_")


def list_model_versions(models_dir: Path) -> List[str]:
    """
    Return a sorted list of available Poisson model files.
    Includes poisson_model_latest.pkl if present.
    """
    pattern = re.compile(r"poisson_model_(\d{8}_\d{6})\.pkl")
    versions: List[str] = []

    for p in models_dir.glob("poisson_model_*.pkl"):
        m = pattern.match(p.name)
        if m:
            versions.append(p.name)

    versions.sort()

    latest = models_dir / "poisson_model_latest.pkl"
    if latest.exists():
        versions.append("poisson_model_latest.pkl")

    return versions


def load_poisson_model(models_dir: Path, model_file: str) -> PoissonGoalModel:
    model_path = models_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model file not found: {model_path}. "
            f"Train a model with the Train tab or with train_only.py."
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


def load_teams(matches_csv: Path) -> List[str]:
    df = pd.read_csv(matches_csv)
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
    teams = sorted(str(t) for t in teams)
    return teams


def load_teams_by_league(matches_csv: Path) -> Dict[str, List[str]]:
    """
    Return mapping: league_label -> list of internal team names.

    Requires a 'league' column in matches.csv.
    If 'league' is missing, everything is grouped under 'Other'.
    """
    df = pd.read_csv(matches_csv)

    if "league" not in df.columns:
        all_teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
        return {"Other": sorted(str(t) for t in all_teams)}

    teams_by_league: Dict[str, set] = {}
    for _, row in df.iterrows():
        league = str(row["league"])
        ht = str(row["home_team"])
        at = str(row["away_team"])
        teams_by_league.setdefault(league, set()).update([ht, at])

    return {lg: sorted(list(ts)) for lg, ts in teams_by_league.items()}


def load_model_index(models_dir: Path) -> Optional[pd.DataFrame]:
    index_path = models_dir / "model_index.csv"
    if index_path.exists():
        return pd.read_csv(index_path)
    return None


def get_league_options() -> Dict[str, str]:
    """
    Map friendly league names to ESPN league codes.
    These are used by the ESPN fixtures scraper.
    """
    return {
        "EPL": "eng.1",
        "La Liga": "esp.1",
        "Serie A": "ita.1",
        "Ligue 1": "fra.1",
        "Bundesliga": "ger.1",
        "Other": "",  # no direct ESPN mapping; will use manual selection
    }

def get_teams_for_league_label(
    teams_by_league: Dict[str, List[str]],
    all_teams: List[str],
    league_label: str,
) -> List[str]:
    """
    Map the UI league label (EPL, La Liga, ...) to one or more league
    values in matches.csv and return the teams that belong to that league.

    Adjust the strings in label_to_csv to match whatever you actually use
    in the 'league' column of data/matches.csv.
    """
    label_to_csv = {
        "EPL": ["EPL", "Premier League", "eng.1"],
        "La Liga": ["La Liga", "LaLiga", "Spain La Liga", "esp.1"],
        "Serie A": ["Serie A", "SerieA", "Italy Serie A", "ita.1"],
        "Ligue 1": ["Ligue 1", "France Ligue 1", "fra.1"],
        "Bundesliga": ["Bundesliga", "Germany Bundesliga", "ger.1"],
        "Other": [],
    }

    # "Other" = all teams (or you can make this an empty list if you prefer)
    if league_label == "Other":
        return sorted(all_teams)

    targets = label_to_csv.get(league_label, [])
    if not targets:
        return []

    teams = set()
    for csv_league, team_list in teams_by_league.items():
        if csv_league in targets:
            teams.update(team_list)

    return sorted(teams)

def normalize_team_name(name: str) -> str:
    """
    Normalize a team name for fuzzy matching:
    lowercase, remove spaces, underscores and punctuation.
    """
    name = name.lower()
    name = name.replace("_", " ")
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


def map_api_team_to_internal(api_name: str, internal_team_list: List[str]) -> Optional[str]:
    """
    Try to map a scraped ESPN team name (e.g. 'Manchester United') to one of
    your internal team strings (e.g. 'Manchester_United').
    Returns the internal team name or None if not found.
    """
    target = normalize_team_name(api_name)
    for t in internal_team_list:
        if normalize_team_name(t) == target:
            return t
        if normalize_team_name(internal_to_display(t)) == target:
            return t
    return None


def scrape_espn_schedule(league_code: str, date_str: str) -> List[Dict]:
    """
    Scrape ESPN match schedules.

    league_code examples:
        'eng.1' = Premier League
        'esp.1' = La Liga
        'ita.1' = Serie A
        'fra.1' = Ligue 1
        'ger.1' = Bundesliga

    date_str format: 'YYYYMMDD' (ESPN format)
    e.g. '20251216'

    Returns list of fixtures:
        [
            {
                "home": "Manchester United",
                "away": "Bournemouth",
                "time": "3:00 PM",
            },
            ...
        ]
    """
    if not league_code:
        return []

    url = f"https://www.espn.com/soccer/fixtures/_/league/{league_code}/date/{date_str}"

    fixtures: List[Dict] = []
    try:
        page = requests.get(url, timeout=10)
        page.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(page.text, "html.parser")

    # ESPN uses TR rows with various classes; grab all small-table rows
    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        home = cells[0].get_text(strip=True)
        away = cells[1].get_text(strip=True)
        time_txt = cells[2].get_text(strip=True)

        # crude filter to avoid header or empty rows
        if not home or not away or home == "MATCH" or away == "MATCH":
            continue

        fixtures.append(
            {
                "home": home,
                "away": away,
                "time": time_txt,
            }
        )

    return fixtures


def format_ev_line(selection: str, line_str: Optional[str], prob: float, ev: float, ev_percent: float) -> str:
    """
    Return an HTML-formatted line with color-coded EV%.
    """
    if ev_percent > 0:
        color = "limegreen"
    elif ev_percent < 0:
        color = "crimson"
    else:
        color = "gray"

    label = selection if not line_str else f"{selection} {line_str}"

    return (
        f"<span style='font-weight:600'>{label}</span> — "
        f"Probability: <code>{prob:.3f}</code> | "
        f"EV: <code>{ev:.3f}</code> "
        f"(<span style='color:{color}'>{ev_percent:.1f}%</span>)"
    )


# ----------------------- MAIN APP ----------------------- #

def main():
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    matches_csv = project_root / "data" / "matches.csv"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    st.title("Soccer Betting Model – Train & Predict")
    st.caption("Poisson / Dixon–Coles / Bayesian xG · EV-Based Pricing · The Book Beaters")

    st.sidebar.markdown("### Model Version & Settings")

    tab_train, tab_predict = st.tabs(["Train Model", "Predict & Pick"])

    # -------------------- TRAIN TAB -------------------- #
    with tab_train:
        st.markdown("## Training")

        if not matches_csv.exists():
            st.error(f"`matches.csv` not found at `{matches_csv}`.")
        else:
            df_matches = pd.read_csv(matches_csv)
            st.markdown("#### Training Data Overview")
            st.write(f"Total Matches: **{len(df_matches)}**")
            st.dataframe(df_matches.head(), use_container_width=True)

        st.markdown("#### Existing Model Versions")
        if models_dir.exists():
            index_df = load_model_index(models_dir)
            if index_df is not None and not index_df.empty:
                st.dataframe(index_df.tail(10), use_container_width=True)
            else:
                st.info("No `model_index.csv` yet. Train at least one model.")
        else:
            st.info("Models directory does not exist yet.")

        if st.button("Train Model"):
            if not matches_csv.exists():
                st.error("Cannot train: `matches.csv` is missing.")
            else:
                with st.spinner("Running Bayesian training via `train_only.main()`..."):
                    train_only.main()
                st.success("Training complete. Model files and index updated.")
                st.cache_resource.clear()

    # -------------------- PREDICT TAB -------------------- #
    with tab_predict:
        st.markdown("## Predict Markets & Calculate EV")

        # Model version picker
        if not models_dir.exists():
            st.error(f"Models directory not found: `{models_dir}`.")
            return

        model_files = list_model_versions(models_dir)
        if not model_files:
            st.error(
                "No model versions found. Train a model in the **Train Model** tab "
                "or locally with `python -m soccer_model.train_only`."
            )
            return

        default_index = len(model_files) - 1
        selected_model_file = st.sidebar.selectbox(
            "Select Model Version",
            options=model_files,
            index=default_index,
        )
        st.sidebar.info(f"Using Model: **{selected_model_file}**")

        @st.cache_resource
        def get_dc_model(models_dir_str: str, model_file_str: str):
            md = Path(models_dir_str)
            pm = load_poisson_model(md, model_file_str)
            return DixonColesWrapper(pm)

        dc_model = get_dc_model(str(models_dir), selected_model_file)

        if not matches_csv.exists():
            st.error(f"`matches.csv` not found at `{matches_csv}`.")
            return

  # ---------------- MATCH SELECTION ---------------- #
        st.markdown("### Match Selection")

        league_options = get_league_options()
        league_labels = list(league_options.keys())
        league_label = st.selectbox("League", league_labels, index=0)
        league_code = league_options[league_label]

        # Teams grouped by the actual 'league' values in matches.csv
        teams_by_league = load_teams_by_league(matches_csv)
        all_teams_internal = load_teams(matches_csv)

        internal_team_list = get_teams_for_league_label(
            teams_by_league=teams_by_league,
            all_teams=all_teams_internal,
            league_label=league_label,
        )

        if not internal_team_list:
            st.warning(
                "No teams found in data/matches.csv for this league label. "
                "Showing all teams instead."
            )
            internal_team_list = all_teams_internal


        input_mode = st.radio(
            "Match Input Mode",
            ["From ESPN schedule", "Manual team selection"],
            index=0,
            horizontal=True,
        )

        today = date.today()
        match_date = st.date_input("Match Date", value=today)
        espn_date_str = match_date.strftime("%Y%m%d")

        home_team_display = ""
        away_team_display = ""
        home_team = ""
        away_team = ""

        if input_mode == "From ESPN schedule":
            if not league_code:
                st.warning(
                    "Selected league does not have an ESPN code mapping. "
                    "Please use manual team selection."
                )
                input_mode = "Manual team selection"
            else:
                with st.spinner("Scraping ESPN schedule for this league and date..."):
                    fixtures = scrape_espn_schedule(league_code, espn_date_str)

                if not fixtures:
                    st.warning(
                        "No fixtures found on ESPN for this league/date. "
                        "Switching to manual team selection."
                    )
                    input_mode = "Manual team selection"
                else:
                    labels = [
                        f"{f['home']} vs {f['away']} ({f['time']})"
                        for f in fixtures
                    ]
                    selected = st.selectbox("Select Fixture", labels, index=0)
                    idx = labels.index(selected)
                    fixture = fixtures[idx]

                    home_api = fixture["home"]
                    away_api = fixture["away"]

                    mapped_home = map_api_team_to_internal(home_api, internal_team_list)
                    mapped_away = map_api_team_to_internal(away_api, internal_team_list)

                    if not mapped_home or not mapped_away:
                        st.warning(
                            "Could not map ESPN team names to your internal team list. "
                            "Please use manual team selection."
                        )
                        input_mode = "Manual team selection"
                    else:
                        home_team = mapped_home
                        away_team = mapped_away
                        home_team_display = internal_to_display(home_team)
                        away_team_display = internal_to_display(away_team)

                        st.success(
                            f"Using fixture: **{home_api} vs {away_api}** "
                            f"→ mapped to **{home_team_display}** vs **{away_team_display}**"
                        )

        if input_mode == "Manual team selection":
            teams_display = [internal_to_display(t) for t in internal_team_list]
            col_match1, col_match2 = st.columns(2)
            with col_match1:
                home_team_display = st.selectbox("Home Team", teams_display, index=0)
            with col_match2:
                default_away_index = 1 if len(teams_display) > 1 else 0
                away_team_display = st.selectbox(
                    "Away Team", teams_display, index=default_away_index
                )

            home_team = display_to_internal(home_team_display)
            away_team = display_to_internal(away_team_display)

        st.markdown("---")

        # ---------------- MONEYLINE INPUTS ---------------- #
        st.markdown("### Moneyline (American Odds)")

        col_ml1, col_ml2, col_ml3 = st.columns(3)
        with col_ml1:
            home_ml = st.number_input("Home ML", value=-125.0)
        with col_ml2:
            draw_ml = st.number_input("Draw ML", value=310.0)
        with col_ml3:
            away_ml = st.number_input("Away ML", value=290.0)

        odds_home = american_to_decimal(home_ml)
        odds_draw = american_to_decimal(draw_ml)
        odds_away = american_to_decimal(away_ml)

        # ---------------- TOTALS INPUTS ---------------- #
        st.markdown("### Totals (Over/Under – American Odds)")

        col_tot1, col_tot2, col_tot3 = st.columns(3)
        with col_tot1:
            total_line = st.number_input("Total Line", value=2.5)
        with col_tot2:
            over_ml = st.number_input("Over ML", value=-110.0)
        with col_tot3:
            under_ml = st.number_input("Under ML", value=-110.0)

        odds_over = american_to_decimal(over_ml)
        odds_under = american_to_decimal(under_ml)

        # ---------------- ASIAN HANDICAP INPUTS ---------------- #
        st.markdown("### Asian Handicap / Spread (American Odds)")

        col_ah1, col_ah2, col_ah3 = st.columns(3)
        with col_ah1:
            ah_line = st.number_input("Favorite Handicap Line", value=-0.5)
        with col_ah2:
            fav_ml = st.number_input("Favorite ML", value=-110.0)
        with col_ah3:
            dog_ml = st.number_input("Underdog ML", value=-110.0)

        odds_fav = american_to_decimal(fav_ml)
        odds_dog = american_to_decimal(dog_ml)

        home_is_favored = st.checkbox("Home Team Is Favorite", value=True)

        st.markdown("---")

        # ---------------- RUN MODEL ---------------- #
        if st.button("Run Model And See Best Bets"):
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
                    f"Team `{e}` not found in model. "
                    f"Ensure it exists in `data/matches.csv` and the model was trained with it."
                )
                return

            st.markdown(
                f"### Results for {home_team_display} vs {away_team_display}  "
                f"*(Model: `{selected_model_file}`)*"
            )

            run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            log_rows: List[Dict] = []

            # Moneyline probabilities
            st.markdown("#### Moneyline – Probabilities")
            money_probs = results["moneyline"]["probs"]
            prob_df = pd.DataFrame(
                {
                    "Outcome": list(money_probs.keys()),
                    "Probability": list(money_probs.values()),
                }
            )
            st.table(prob_df)

            # Moneyline EV
            st.markdown("#### Moneyline – Expected Value (Best First)")
            for bet in results["moneyline"]["bets"]:
                row = {
                    "run_timestamp_utc": run_ts,
                    "model_version": selected_model_file,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market": "Moneyline",
                    "selection": bet.selection,
                    "line": "",
                    "prob": bet.prob,
                    "odds_decimal": bet.odds_decimal,
                    "ev": bet.ev,
                    "ev_percent": bet.ev_percent,
                }
                log_rows.append(row)

                st.markdown(
                    format_ev_line(
                        selection=bet.selection,
                        line_str=None,
                        prob=bet.prob,
                        ev=bet.ev,
                        ev_percent=bet.ev_percent,
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Totals EV
            st.markdown("#### Totals (Over/Under) – Expected Value")
            totals_line_str = str(results["totals"]["line"])
            for bet in results["totals"]["bets"]:
                row = {
                    "run_timestamp_utc": run_ts,
                    "model_version": selected_model_file,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market": "Totals",
                    "selection": bet.selection,
                    "line": results["totals"]["line"],
                    "prob": bet.prob,
                    "odds_decimal": bet.odds_decimal,
                    "ev": bet.ev,
                    "ev_percent": bet.ev_percent,
                }
                log_rows.append(row)

                st.markdown(
                    format_ev_line(
                        selection=bet.selection,
                        line_str=totals_line_str,
                        prob=bet.prob,
                        ev=bet.ev,
                        ev_percent=bet.ev_percent,
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Asian handicap EV
            st.markdown("#### Asian Handicap – Expected Value")
            asian_line_str = str(results["asian"]["line"])
            for bet in results["asian"]["bets"]:
                row = {
                    "run_timestamp_utc": run_ts,
                    "model_version": selected_model_file,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market": "Asian Handicap",
                    "selection": bet.selection,
                    "line": results["asian"]["line"],
                    "prob": bet.prob,
                    "odds_decimal": bet.odds_decimal,
                    "ev": bet.ev,
                    "ev_percent": bet.ev_percent,
                }
                log_rows.append(row)

                st.markdown(
                    format_ev_line(
                        selection=bet.selection,
                        line_str=asian_line_str,
                        prob=bet.prob,
                        ev=bet.ev,
                        ev_percent=bet.ev_percent,
                    ),
                    unsafe_allow_html=True,
                )

            # Logging
            if log_rows:
                log_df = pd.DataFrame(log_rows)
                ts_file = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                log_path = logs_dir / f"ev_log_app_{ts_file}.csv"
                log_df.to_csv(log_path, index=False)
                st.success(f"Logged EV output to: `{log_path}`")
            else:
                st.info("No rows to log.")


if __name__ == "__main__":
    main()