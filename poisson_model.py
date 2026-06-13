"""
Dixon-Coles Poisson Model
Predicts score distributions, match winner, over/under, and BTTS probabilities.
Reference: Dixon & Coles (1997) "Modelling Association Football Scores"
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Tuple


def _dc_rho(x: int, y: int, mu: float, nu: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor."""
    if x == 0 and y == 0:
        return 1 - mu * nu * rho
    elif x == 0 and y == 1:
        return 1 + mu * rho
    elif x == 1 and y == 0:
        return 1 + nu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _log_likelihood(params: np.ndarray, matches: pd.DataFrame, teams: list) -> float:
    """Negative log-likelihood for Dixon-Coles model."""
    n = len(teams)
    attack  = dict(zip(teams, params[:n]))
    defence = dict(zip(teams, params[n:2*n]))
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    ll = 0.0
    for _, row in matches.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        mu = np.exp(attack[ht] + defence[at] + home_adv)
        nu = np.exp(attack[at] + defence[ht])

        ll += (
            np.log(poisson.pmf(hg, mu) + 1e-10)
            + np.log(poisson.pmf(ag, nu) + 1e-10)
            + np.log(_dc_rho(hg, ag, mu, nu, rho) + 1e-10)
        )
    return -ll


class DixonColesModel:
    """
    Fits a Dixon-Coles Poisson model on completed World Cup matches
    and generates match predictions.
    """

    def __init__(self):
        self.teams = []
        self.attack = {}
        self.defence = {}
        self.home_adv = 0.0
        self.rho = 0.0
        self.fitted = False

    def fit(self, matches: pd.DataFrame):
        """Fit the model on completed matches."""
        completed = matches[matches["status"] == "FINISHED"].copy()
        if len(completed) < 5:
            print("  Warning: fewer than 5 completed matches — model may be unreliable")

        self.teams = sorted(
            set(completed["home_team"].tolist() + completed["away_team"].tolist())
        )
        n = len(self.teams)

        # Initial params: attack=0, defence=0, home_adv=0.1, rho=-0.1
        x0 = np.zeros(2 * n + 2)
        x0[2 * n]     = 0.1   # home advantage
        x0[2 * n + 1] = -0.1  # rho

        result = minimize(
            _log_likelihood,
            x0,
            args=(completed, self.teams),
            method="L-BFGS-B",
            options={"maxiter": 200},
        )

        params = result.x
        self.attack  = dict(zip(self.teams, params[:n]))
        self.defence = dict(zip(self.teams, params[n:2*n]))
        self.home_adv = params[2 * n]
        self.rho      = params[2 * n + 1]
        self.fitted = True
        print(f"  Model fitted on {len(completed)} matches, {n} teams")

    def _expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        """Expected goals (lambda) for each team."""
        mu = np.exp(self.attack[home] + self.defence[away] + self.home_adv)
        nu = np.exp(self.attack[away] + self.defence[home])
        return mu, nu

    def score_matrix(self, home: str, away: str, max_goals: int = 8) -> np.ndarray:
        """Return probability matrix P[home_goals, away_goals]."""
        mu, nu = self._expected_goals(home, away)
        matrix = np.outer(
            poisson.pmf(range(max_goals + 1), mu),
            poisson.pmf(range(max_goals + 1), nu),
        )
        # Apply Dixon-Coles correction for low scores
        for i in range(2):
            for j in range(2):
                matrix[i, j] *= _dc_rho(i, j, mu, nu, self.rho)
        matrix /= matrix.sum()
        return matrix

    def predict(self, home: str, away: str) -> dict:
        """
        Generate full prediction for a match.

        Returns probabilities for:
        - home_win, draw, away_win
        - over_1_5, over_2_5, over_3_5 (total goals)
        - btts (both teams to score)
        - expected_home_goals, expected_away_goals
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        for team, label in [(home, "home"), (away, "away")]:
            if team not in self.teams:
                raise ValueError(f"Unknown team '{team}' ({label}). "
                                 f"Available: {self.teams[:5]}...")

        matrix = self.score_matrix(home, away)
        n = matrix.shape[0]

        home_win = float(np.tril(matrix, -1).sum())
        draw     = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())

        total_goals = np.array(
            [[i + j for j in range(n)] for i in range(n)]
        )
        over_1_5 = float(matrix[total_goals > 1.5].sum())
        over_2_5 = float(matrix[total_goals > 2.5].sum())
        over_3_5 = float(matrix[total_goals > 3.5].sum())

        btts = float(matrix[1:, 1:].sum())

        mu, nu = self._expected_goals(home, away)

        return {
            "home_team":            home,
            "away_team":            away,
            "home_win":             round(home_win, 4),
            "draw":                 round(draw, 4),
            "away_win":             round(away_win, 4),
            "over_1_5":             round(over_1_5, 4),
            "over_2_5":             round(over_2_5, 4),
            "over_3_5":             round(over_3_5, 4),
            "btts":                 round(btts, 4),
            "expected_home_goals":  round(mu, 3),
            "expected_away_goals":  round(nu, 3),
        }

    def most_likely_score(self, home: str, away: str) -> Tuple[int, int]:
        matrix = self.score_matrix(home, away)
        idx = np.unravel_index(matrix.argmax(), matrix.shape)
        return int(idx[0]), int(idx[1])
