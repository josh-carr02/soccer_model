from __future__ import annotations
from pathlib import Path

import streamlit as st

from soccer_model.pipeline import build_models, evaluate_match_markets


# ----------------------- helpers ----------------------- #

def american_to_decimal(odds: float) -> float:
    """
    Convert American odds to decimal odds.

    Examples:
      -125 -> 1 + 100/125 = 1.80
      +200 -> 1 + 200/100 = 3.00
    """
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def normalize_team_name(name: str) -> str:
    """
    Convert display name ('Manchester United') to internal name
    used in matches.csv ('Manchester_United').
    """
    return name.strip().replace(" ", "_")


# ----------------------- app ----------------------- #

def main():
    st.title("Soccer Betting Analytics Model")

    # --- matches CSV path and model building ---
    matches_csv = st.text_input("Matches CSV path", "data/matches.csv")

    if "dc_model" not in st.session_state:
        st.session_state["dc_model"] = None

    if st.button("Build / Rebuild Models"):
        with st.spinner("Fitting Bayesian xG model..."):
            _, _, _, dc_model = build_models(matches_csv)
            st.session_state["dc_model"] = dc_model
        st.success("Models built, let's win.")

    dc_model = st.session_state.get("dc_model")
    if dc_model is None:
        st.info("Build The Models To Start Winning")
        return

    # --- match inputs ---
    st.subheader("Match inputs")
    col1, col2 = st.columns(2)
    with col1:
        home_team_display = st.text_input("Home Team", "Manchester United")
    with col2:
        away_team_display = st.text_input("Away Team", "Bournemouth")

    # convert display names -> internal names with underscores
    home_team = normalize_team_name(home_team_display)
    away_team = normalize_team_name(away_team_display)

    # --- Moneyline odds (American) ---
    st.subheader("Moneyline Odds")

    col1, col2, col3 = st.columns(3)
    with col1:
        home_ml = st.number_input("Home ML Odds", value=-125)
    with col2:
        draw_ml = st.number_input("Draw Odds", value=310)
    with col3:
        away_ml = st.number_input("Away ML Odds", value=290)

    # convert to decimal for the model
    odds_home = american_to_decimal(home_ml)
    odds_draw = american_to_decimal(draw_ml)
    odds_away = american_to_decimal(away_ml)

    # --- Totals market (American odds) ---
    st.subheader("Total Goals Scored")

    col1, col2, col3 = st.columns(3)
    with col1:
        total_line = st.number_input("Total Line", value=2.5)
    with col2:
        over_ml = st.number_input("Over Odds", value=-110)
    with col3:
        under_ml = st.number_input("Under Odds", value=-110)

    odds_over = american_to_decimal(over_ml)
    odds_under = american_to_decimal(under_ml)

    # --- Asian handicap / spread (American odds) ---
    st.subheader("Asian Handicap")

    col1, col2, col3 = st.columns(3)
    with col1:
        ah_line = st.number_input("Asian Handicap Spread For Favorite", value=-0.5)
    with col2:
        fav_ml = st.number_input("Favorite Odds", value=-110)
    with col3:
        dog_ml = st.number_input("Underdog Odds", value=-110)

    odds_fav = american_to_decimal(fav_ml)
    odds_dog = american_to_decimal(dog_ml)

    home_is_favored = st.checkbox("Home is favorite", value=True)

    # --- evaluate button ---
    if st.button("Evaluate markets"):
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
                f"Make sure the team names exist in matches.csv "
                f"(internal format uses underscores, e.g. Manchester_United)."
            )
            return

        st.subheader(f"Results for {home_team_display} vs {away_team_display}")

        # 1) Moneyline probabilities and EV
        st.markdown("### Moneyline (1X2)")
        st.write(results["1x2"]["probs"])

        st.markdown("#### Moneyline EV (best first)")
        for b in results["1x2"]["bets"]:
            st.write({
                "selection": b.selection,
                "prob": round(b.prob, 3),
                "odds_decimal": round(b.odds_decimal, 3),
                "EV": round(b.ev, 3),
                "EV%": round(b.ev_percent, 1),
            })

        # 2) Totals EV
        st.markdown("### Totals EV (best first)")
        for b in results["totals"]["bets"]:
            st.write({
                "selection": f"{b.selection} {results['totals']['line']}",
                "prob": round(b.prob, 3),
                "odds_decimal": round(b.odds_decimal, 3),
                "EV": round(b.ev, 3),
                "EV%": round(b.ev_percent, 1),
            })

        # 3) Asian Handicap EV
        st.markdown("### Asian Handicap / Spread EV (best first)")
        for b in results["asian"]["bets"]:
            st.write({
                "selection": f"{b.selection} {results['asian']['line']}",
                "prob": round(b.prob, 3),
                "odds_decimal": round(b.odds_decimal, 3),
                "EV": round(b.ev, 3),
                "EV%": round(b.ev_percent, 1),
            })


if __name__ == "__main__":
    main()