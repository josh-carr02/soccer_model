from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pickle
import re
import os

import pandas as pd
import requests
import streamlit as st

from soccer_model.models_poisson import PoissonGoalModel, TeamStrength
from soccer_model.dixon_coles import DixonColesWrapper
from soccer_model.pipeline import evaluate_match_markets
from soccer_model import train_only


# ----------------------- PAGE CONFIG ----------------------- #

st.set_page_config(
    page_title="Soccer Betting Model",
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

import json
import math
import re

def normalize_team_name(name: str) -> str:
    """
    Normalize a team name for fuzzy matching:
    lowercase, remove spaces, underscores and punctuation.
    """
    name = name.lower()
    name = name.replace("_", " ")
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


def map_api_team_to_internal(api_name: str, internal_team_list: list[str]) -> str | None:
    """
    Try to map an API team name (e.g. 'Manchester United') to one of
    your internal team strings (e.g. 'Manchester_United').
    Returns the internal team name or None if not found.
    """
    target = normalize_team_name(api_name)
    for t in internal_team_list:
        if normalize_team_name(t) == target:
            return t
        # also compare to display version
        if normalize_team_name(internal_to_display(t)) == target:
            return t
    return None


def fetch_fixtures_from_api(
    api_key: str,
    base_url: str,
    sport_key: str,
    region: str = "us",
) -> list[dict]:
    """
    Fetch upcoming fixtures for a league using the odds API.
    We call the h2h odds endpoint and treat each game as a fixture.

    Returns a list of dicts:
    [
      {
        "home": "Manchester United",
        "away": "Bournemouth",
        "commence_time": "2025-12-15T18:00:00Z",
      },
      ...
    ]
    """
    if not api_key or not base_url:
        return []

    try:
        url = f"{base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": region,
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    fixtures = []
    for game in data:
        home_team_api = str(game.get("home_team", ""))
        teams = game.get("teams", [])
        if len(teams) != 2:
            continue
        # figure out which is away
        if teams[0] == home_team_api:
            away_team_api = teams[1]
        else:
            away_team_api = teams[0]
        fixtures.append(
            {
                "home": home_team_api,
                "away": away_team_api,
                "commence_time": game.get("commence_time", ""),
            }
        )
    return fixtures



def list_model_versions(models_dir: Path) -> list[str]:
    """
    Return a sorted list of available model files (versioned), plus
    'poisson_model_latest.pkl' if present.
    """
    pattern = re.compile(r"poisson_model_(\d{8}_\d{6})\.pkl")
    versions: list[str] = []

    for p in models_dir.glob("poisson_model_*.pkl"):
        m = pattern.match(p.name)
        if m:
            versions.append(p.name)

    versions.sort()  # chronological order

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

def load_teams_by_league(matches_csv: Path) -> dict[str, list[str]]:
    """
    Return mapping: league_label -> list of internal team names.

    Requires a 'league' column in matches.csv.
    If 'league' is missing, everything is grouped under 'Other'.
    """
    df = pd.read_csv(matches_csv)

    if "league" not in df.columns:
        # Fallback: all teams under "Other"
        all_teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
        return {"Other": sorted(str(t) for t in all_teams)}

    teams_by_league: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        league = str(row["league"])
        ht = str(row["home_team"])
        at = str(row["away_team"])
        teams_by_league.setdefault(league, set()).update([ht, at])

    # Convert sets to sorted lists
    return {lg: sorted(list(ts)) for lg, ts in teams_by_league.items()}


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

def get_secrets() -> tuple[str, str]:
    """
    Retrieve ODDS_API_KEY and ODDS_API_BASE from Streamlit secrets or env vars.
    For Streamlit Cloud, set them in Settings → Secrets.
    """
    api_key = ""
    base_url = ""

    # Streamlit secrets (cloud)
    try:
        if "ODDS_API_KEY" in st.secrets:
            api_key = st.secrets["ODDS_API_KEY"]
        if "ODDS_API_BASE" in st.secrets:
            base_url = st.secrets["ODDS_API_BASE"]
    except Exception:
        pass

    # Environment variables (local)
    if not api_key:
        api_key = os.environ.get("ODDS_API_KEY", "")
    if not base_url:
        base_url = os.environ.get("ODDS_API_BASE", "")

    return api_key, base_url


def get_league_options() -> dict[str, str]:
    """
    Map friendly league names to API sport keys.
    Adjust these depending on your odds provider.
    """
    return {
        "English PL": "soccer_epl",
        "La Liga": "soccer_spain_la_liga",
        "Serie A": "soccer_italy_serie_a",
        "Ligue 1": "soccer_france_ligue_one",
        "Bundesliga": "soccer_germany_bundesliga",
        "Other": "soccer",
    }


def fetch_live_odds_from_api(
    api_key: str,
    base_url: str,
    sport_key: str,
    home_team_display: str,
    away_team_display: str,
    region: str = "us",
) -> dict | None:
    """
    Example live odds fetch using a generic REST API (like The Odds API).

    Returns a dict:
      {
        "home_ml": ...,
        "away_ml": ...,
        "draw_ml": ... or None,
        "total": ... or None,
        "over_ml": ... or None,
        "under_ml": ... or None,
      }
    or None if not found / error.
    """
    if not api_key or not base_url:
        return None

    try:
        url = f"{base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": region,
            "markets": "h2h,totals",
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    home_lower = home_team_display.lower()
    away_lower = away_team_display.lower()

    match_obj = None
    for game in data:
        teams = [str(t).lower() for t in game.get("teams", [])]
        home_team_api = str(game.get("home_team", "")).lower()

        if home_lower in home_team_api and any(away_lower in t for t in teams):
            match_obj = game
            break
        if away_lower in home_team_api and any(home_lower in t for t in teams):
            match_obj = game
            break

    if not match_obj:
        return None

    home_ml = away_ml = draw_ml = None
    total_line = None
    over_ml = under_ml = None

    bookmakers = match_obj.get("bookmakers", [])
    if not bookmakers:
        return None

    bm = bookmakers[0]
    markets = bm.get("markets", [])

    for mkt in markets:
        key = mkt.get("key")
        outcomes = mkt.get("outcomes", [])
        if key == "h2h":
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
        elif key == "totals":
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

    return {
        "home_ml": home_ml,
        "away_ml": away_ml,
        "draw_ml": draw_ml,
        "total": total_line,
        "over_ml": over_ml,
        "under_ml": under_ml,
    }


def format_ev_line(selection: str, line_str: str | None, prob: float, ev: float, ev_percent: float) -> str:
    """
    Return an HTML-formatted line with color-coded EV%.
    """
    if ev_percent > 0:
        color = "limegreen"
    elif ev_percent < 0:
        color = "crimson"
    else:
        color = "gray"

    if line_str:
        label = f"{selection} {line_str}"
    else:
        label = selection

    return (
        f"<span style='font-weight:600'>{label}</span> — "
        f"Probability: <code>{prob:.3f}</code> | "
        f"EV: <code>{ev:.3f}</code> "
        f"(<span style='color:{color}'>{ev_percent:.1f}%</span>)"
    )


# ----------------------- APP ----------------------- #

def main():
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    matches_csv = project_root / "data" / "matches.csv"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    st.title("Soccer Betting Model")
    st.caption("Bayesian xG · Poisson · Dixon–Coles · EV-Based Picks")

    # Sidebar logo / info
    st.sidebar.markdown("### The Book Beaters")
    st.sidebar.markdown("Professional Soccer Pricing Dashboard")

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
                with st.spinner("Training Models..."):
                    train_only.main()
                st.success("Training complete. Model files and index updated.")
                st.cache_resource.clear()

    # -------------------- PREDICT TAB -------------------- #
    with tab_predict:
        st.markdown("## Predict & Pick")

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

        st.sidebar.markdown("### Model Version")
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

        # Match selection
# Match selection
# Match selection
if not matches_csv.exists():
    st.error(f"`matches.csv` not found at `{matches_csv}`.")
    return

st.markdown("### Match Selection")

# 1) Choose league (this drives both fixtures and sport_key)
league_options = get_league_options()
league_labels = list(league_options.keys())
league_label = st.selectbox("League", league_labels, index=0)
sport_key = league_options[league_label]  # used for live odds & fixtures

# 2) Load internal teams by league
teams_by_league = load_teams_by_league(matches_csv)
all_teams_internal = load_teams(matches_csv)  # fallback
internal_team_list = teams_by_league.get(league_label, all_teams_internal)

# 3) Choose input mode: from fixtures or manual
input_mode = st.radio(
    "Match Input Mode",
    ["From today's fixtures (API)", "Manual team selection"],
    index=0,
    horizontal=True,
)

home_team_display = ""
away_team_display = ""
home_team = ""
away_team = ""

api_key_default, base_url_default = get_secrets()

if input_mode == "From today's fixtures (API)":
    if not api_key_default or not base_url_default:
        st.warning(
            "Live odds API is not configured (no ODDS_API_KEY/ODDS_API_BASE). "
            "Switch to manual selection or set Secrets."
        )
        # fallback to manual selection
        input_mode = "Manual team selection"
    else:
        with st.spinner("Fetching today's fixtures for this league..."):
            fixtures = fetch_fixtures_from_api(
                api_key_default, base_url_default, sport_key
            )

        if not fixtures:
            st.warning(
                "No fixtures returned from API for this league. "
                "Switching to manual team selection."
            )
            input_mode = "Manual team selection"
        else:
            # build label strings for dropdown
            fixture_labels = [
                f"{f['home']} vs {f['away']} ({f.get('commence_time','')})"
                for f in fixtures
            ]
            selected_label = st.selectbox("Select Fixture", fixture_labels, index=0)
            idx = fixture_labels.index(selected_label)
            fixture = fixtures[idx]

            home_api = fixture["home"]
            away_api = fixture["away"]

            # Map API names to internal team names
            mapped_home = map_api_team_to_internal(home_api, internal_team_list)
            mapped_away = map_api_team_to_internal(away_api, internal_team_list)

            if not mapped_home or not mapped_away:
                st.warning(
                    "Could not map fixture team names to your internal team list. "
                    "Please use manual selection instead."
                )
                input_mode = "Manual team selection"
            else:
                home_team = mapped_home
                away_team = mapped_away
                home_team_display = internal_to_display(home_team)
                away_team_display = internal_to_display(away_team)

                st.success(
                    f"Using fixture: **{home_api} vs {away_api}**  \n"
                    f"Mapped to internal teams: **{home_team_display}** vs "
                    f"**{away_team_display}**"
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

        ## League + API config
# Live odds API config (reuse league_label, sport_key, api_key_default from above)
col_league2, col_league3 = st.columns(2)
with col_league2:
    if api_key_default and base_url_default:
        st.success(
            f"Live odds API configured. League: **{league_label}** "
            f"({sport_key})"
        )
    else:
        st.warning(
            "Live odds API is not fully configured. "
            "Set ODDS_API_KEY and ODDS_API_BASE in the app Secrets."
        )
with col_league3:
    region = st.text_input("Region", value="us")


# Live odds fetch
with st.expander("Live Odds (Optional)"):
    st.write("Gather Live odds and Pre-Fill Odds Selections")
    if st.button("Gather Live Odds For This Match"):
        if not api_key_default or not base_url_default:
            st.warning(
                "No API key/base URL configured. "
                "Add ODDS_API_KEY and ODDS_API_BASE in the app Secrets."
            )
        else:
            with st.spinner("Fetching Live Odds..."):
                live_odds = fetch_live_odds_from_api(
                    api_key_default,
                    base_url_default,
                    sport_key,
                    home_team_display,
                    away_team_display,
                    region=region,
                )
            if live_odds is None:
                st.warning("No Live Odds Found For This Match (or API format mismatch).")
            else:
                st.success("Live Odds Fetched")
                for key in [
                    "home_ml",
                    "away_ml",
                    "draw_ml",
                    "total",
                    "over_ml",
                    "under_ml",
                ]:
                    if live_odds.get(key) is not None:
                        st.session_state[key] = live_odds[key]
                st.json(live_odds)

        # Moneyline inputs
        st.markdown("### Moneyline")

        def ss_get(name: str, default: float) -> float:
            return float(st.session_state.get(name, default))

        col_ml1, col_ml2, col_ml3 = st.columns(3)
        with col_ml1:
            home_ml = st.number_input(
                "Home ML",
                value=ss_get("home_ml", 110),
                key="home_ml_input",
            )
        with col_ml2:
            draw_ml = st.number_input(
                "Draw ML",
                value=ss_get("draw_ml", 110),
                key="draw_ml_input",
            )
        with col_ml3:
            away_ml = st.number_input(
                "Away ML",
                value=ss_get("away_ml", 110),
                key="away_ml_input",
            )

        odds_home = american_to_decimal(home_ml)
        odds_draw = american_to_decimal(draw_ml)
        odds_away = american_to_decimal(away_ml)

        # Totals inputs
        st.markdown("### Totals (Over/Under)")

        col_tot1, col_tot2, col_tot3 = st.columns(3)
        with col_tot1:
            total_line = st.number_input(
                "Total Line",
                value=ss_get("total", 2.5),
                key="total_line_input",
            )
        with col_tot2:
            over_ml = st.number_input(
                "Over Odds",
                value=ss_get("over_ml", 110),
                key="over_ml_input",
            )
        with col_tot3:
            under_ml = st.number_input(
                "Under Odds",
                value=ss_get("under_ml", 110),
                key="under_ml_input",
            )

        odds_over = american_to_decimal(over_ml)
        odds_under = american_to_decimal(under_ml)

        # Asian handicap inputs
        st.markdown("### Asian Handicap")

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

        # Evaluate and log
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
                    f"Ensure it exists in `data/matches.csv`."
                )
                return

            st.markdown(
                f"### Results for {home_team_display} vs {away_team_display}  "
                f"*(Model: `{selected_model_file}`)*"
            )

            run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            log_rows: list[dict] = []

            # Moneyline probabilities (table)
            st.markdown("#### Moneyline – Probabilities")
            money_probs = results["moneyline"]["probs"]
            prob_df = pd.DataFrame(
                {"Outcome": list(money_probs.keys()), "Probability": list(money_probs.values())}
            )
            st.table(prob_df)

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

            # Totals
            st.markdown("#### Totals (Over/Under) – Expected Value")
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
                        line_str=str(results["totals"]["line"]),
                        prob=bet.prob,
                        ev=bet.ev,
                        ev_percent=bet.ev_percent,
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Asian handicap
            st.markdown("#### Asian Handicap – Expected Value")
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
                        line_str=str(results["asian"]["line"]),
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
