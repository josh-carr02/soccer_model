import numpy as np
import pandas as pd
from typing import Dict
import pymc as pm
import arviz as az

from .config import DEFAULT_BAYESIAN_CONFIG, DEFAULT_MODEL_CONFIG
from .models_poisson import TeamStrength, PoissonGoalModel


class BayesianXGModel:
    """
    Bayesian hierarchical model to estimate team attack/defense based on xG.

    Model:
      base_rate ~ Normal(0, 0.5)
      home_advantage ~ Normal(mu_home_adv, sd_home_adv)
      attack[team] ~ Normal(league_attack_mean, team_effect_sd)
      defense[team] ~ Normal(league_defense_mean, team_effect_sd)

      log(lambda_home) = base_rate + home_advantage + attack[home] - defense[away]
      log(lambda_away) = base_rate + attack[away] - defense[home]

      home_xg ~ Poisson(lambda_home)
      away_xg ~ Poisson(lambda_away)
    """

    def __init__(self,
                 bayes_config=DEFAULT_BAYESIAN_CONFIG,
                 model_config=DEFAULT_MODEL_CONFIG):
        self.bayes_config = bayes_config
        self.model_config = model_config

        self.id2team: dict[int, str] | None = None
        self.team2id: dict[str, int] | None = None

        self.trace: az.InferenceData | None = None
        self.base_rate: float | None = None
        self.home_advantage: float | None = None
        self.team_strengths: Dict[str, TeamStrength] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _encode_teams(self, matches: pd.DataFrame):
        teams = pd.unique(matches[["home_team", "away_team"]].values.ravel("K"))
        self.id2team = dict(enumerate(sorted(teams)))
        self.team2id = {t: i for i, t in self.id2team.items()}

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self,
            matches: pd.DataFrame,
            draws: int = 1000,
            tune: int = 1000,
            random_seed: int = 42):
        """
        Fit the hierarchical model using match-level xG.

        Parameters:
          matches: DataFrame with columns home_team, away_team, home_xg, away_xg
        """
        self._encode_teams(matches)
        assert self.team2id is not None

        n_teams = len(self.team2id)

        home_idx = matches["home_team"].map(self.team2id).values
        away_idx = matches["away_team"].map(self.team2id).values

        home_xg = matches["home_xg"].values
        away_xg = matches["away_xg"].values

        cfg = self.bayes_config

        with pm.Model() as model:
            base_rate = pm.Normal("base_rate", mu=0.0, sigma=0.5)

            home_advantage = pm.Normal(
                "home_advantage",
                mu=cfg.league_attack_mean,
                sigma=cfg.team_effect_sd,
            )

            attack = pm.Normal(
                "attack",
                mu=cfg.league_attack_mean,
                sigma=cfg.team_effect_sd,
                shape=n_teams,
            )
            defense = pm.Normal(
                "defense",
                mu=cfg.league_defense_mean,
                sigma=cfg.team_effect_sd,
                shape=n_teams,
            )

            lambda_home = pm.math.exp(
                base_rate + home_advantage + attack[home_idx] - defense[away_idx]
            )
            lambda_away = pm.math.exp(
                base_rate + attack[away_idx] - defense[home_idx]
            )

            pm.Poisson("home_xg_obs", mu=lambda_home, observed=home_xg)
            pm.Poisson("away_xg_obs", mu=lambda_away, observed=away_xg)

            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=0.9,
                random_seed=random_seed,
                progressbar=True,
            )

        # posterior means as point estimates
        assert self.trace is not None
        base_rate_mean = self.trace.posterior["base_rate"].mean().item()
        home_advantage_mean = self.trace.posterior["home_advantage"].mean().item()
        attack_mean = self.trace.posterior["attack"].mean(dim=("chain", "draw")).values
        defense_mean = self.trace.posterior["defense"].mean(dim=("chain", "draw")).values

        self.base_rate = float(base_rate_mean)
        self.home_advantage = float(home_advantage_mean)

        assert self.id2team is not None
        for i in range(len(self.id2team)):
            team_name = self.id2team[i]
            self.team_strengths[team_name] = TeamStrength(
                attack=float(attack_mean[i]),
                defense=float(defense_mean[i]),
            )

    # ------------------------------------------------------------------
    # conversion to Poisson goal model
    # ------------------------------------------------------------------
    def to_poisson_model(self) -> PoissonGoalModel:
        if self.base_rate is None or self.home_advantage is None:
            raise RuntimeError("Bayesian model not fitted yet.")
        return PoissonGoalModel(
            base_rate=self.base_rate,
            home_advantage=self.home_advantage,
            team_strengths=self.team_strengths,
            max_goals=self.model_config.max_goals,
        )
