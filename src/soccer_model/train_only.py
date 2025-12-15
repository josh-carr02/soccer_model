from __future__ import annotations
from pathlib import Path
import pickle
from datetime import datetime

import pandas as pd

from soccer_model.data_loading import load_matches
from soccer_model.models_bayesian import BayesianXGModel
from soccer_model.models_poisson import PoissonGoalModel


def main():
    project_root = Path(__file__).resolve().parents[2]  # .../soccer_model
    matches_csv = project_root / "data" / "matches.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Loading matches from: {matches_csv}")
    matches = load_matches(matches_csv)
    n_matches = len(matches)
    print(f"Fitting Bayesian xG model on {n_matches} matches...")

    bayes_model = BayesianXGModel()
    bayes_model.fit(matches)

    print("Converting to Poisson goal model...")
    poisson_model: PoissonGoalModel = bayes_model.to_poisson_model()

    # Serialize only the parameters we need for fast prediction
    model_data = {
        "base_rate": poisson_model.base_rate,
        "home_advantage": poisson_model.home_advantage,
        "max_goals": poisson_model.max_goals,
        "team_strengths": {
            team: {"attack": ts.attack, "defense": ts.defense}
            for team, ts in poisson_model.team_strengths.items()
        },
    }

    # ----- versioning -----
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version_name = f"poisson_model_{ts}.pkl"
    version_path = models_dir / version_name
    latest_path = models_dir / "poisson_model_latest.pkl"

    # Save versioned model
    with version_path.open("wb") as f:
        pickle.dump(model_data, f)

    # Overwrite latest pointer file
    with latest_path.open("wb") as f:
        pickle.dump(model_data, f)

    print(f"Saved new model version: {version_path.name}")
    print(f"Updated latest model: {latest_path.name}")

    # ----- model index -----
    index_path = models_dir / "model_index.csv"
    new_row = {
        "version_file": version_name,
        "trained_at_utc": ts,
        "n_matches": n_matches,
        "models_dir": str(models_dir),
    }

    if index_path.exists():
        index_df = pd.read_csv(index_path)
        index_df = pd.concat([index_df, pd.DataFrame([new_row])],
                             ignore_index=True)
    else:
        index_df = pd.DataFrame([new_row])

    index_df.to_csv(index_path, index=False)
    print(f"Updated model index: {index_path}")


if __name__ == "__main__":
    main()
