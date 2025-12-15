from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Maximum number of goals per team when building probability matrices
    max_goals: int = 10

    # Dixon–Coles correlation parameter for low-score adjustments
    dixon_coles_rho: float = 0.13

    # Prior for home advantage (on log-scale) in Bayesian model
    home_advantage_prior_mean: float = 0.25
    home_advantage_prior_sd: float = 0.1


@dataclass
class BayesianConfig:
    # Prior standard deviation for team attack/defense strengths
    team_effect_sd: float = 0.3

    # League-level mean attack and defense strengths
    league_attack_mean: float = 0.0
    league_defense_mean: float = 0.0


# Global default configs used by other modules
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_BAYESIAN_CONFIG = BayesianConfig()
