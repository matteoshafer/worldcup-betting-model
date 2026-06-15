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


def _elo_predict(home: str, away: str, league_avg_goals: float = 1.35) -> dict:
    """
    Pure Elo-based prediction when tournament match data is unavailable.
    Uses Poisson distribution parameterised by Elo-derived expected goals.
    """
    from team_ratings import ELO_RATINGS
    home_elo = ELO_RATINGS.get(home, 1650)
    away_elo = ELO_RATINGS.get(away, 1650)
    avg_elo  = 1750

    # Scale Elo to expected goals: stronger team scores more, concedes less
    mu = league_avg_goals * np.exp((home_elo - avg_elo) / 600 - (away_elo - avg_elo) / 800)
    nu = league_avg_goals * np.exp((away_elo - avg_elo) / 600 - (home_elo - avg_elo) / 800)
    mu = max(0.3, min(4.0, mu))
    nu = max(0.3, min(4.0, nu))

    max_g = 8
    matrix = np.outer(
        poisson.pmf(range(max_g + 1), mu),
        poisson.pmf(range(max_g + 1), nu),
    )
    # Dixon-Coles low-score correction (rho=-0.1 as neutral prior)
    for i in range(2):
        for j in range(2):
            matrix[i, j] *= _dc_rho(i, j, mu, nu, -0.1)
    matrix /= matrix.sum()

    home_win = float(np.tril(matrix, -1).sum())
    draw     = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())

    total = np.array([[i + j for j in range(max_g + 1)] for i in range(max_g + 1)])
    return {
        "home_team":           home,
        "away_team":           away,
        "home_win":            round(home_win, 4),
        "draw":                round(draw, 4),
        "away_win":            round(away_win, 4),
        "over_1_5":            round(float(matrix[total > 1.5].sum()), 4),
        "over_2_5":            round(float(matrix[total > 2.5].sum()), 4),
        "over_3_5":            round(float(matrix[total > 3.5].sum()), 4),
        "btts":                round(float(matrix[1:, 1:].sum()), 4),
        "expected_home_goals": round(mu, 3),
        "expected_away_goals": round(nu, 3),
        "source":              "Elo prior (limited tournament data)",
    }


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


def _log_likelihood(params: np.ndarray, matches: pd.DataFrame, teams: list,
                    weights: np.ndarray = None,
                    elo_priors: dict = None,
                    reg_strength: float = 1.5) -> float:
    """
    Negative log-likelihood for Dixon-Coles model with time-decay weights
    and L2 regularization toward Elo-derived priors.

    reg_strength controls how strongly parameters are pulled toward Elo priors.
    Higher = more Elo influence, less data-driven extremes.
    """
    n = len(teams)
    attack   = dict(zip(teams, params[:n]))
    defence  = dict(zip(teams, params[n:2*n]))
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    if weights is None:
        weights = np.ones(len(matches))

    ll = 0.0
    for i, (_, row) in enumerate(matches.iterrows()):
        ht, at = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        mu = np.exp(np.clip(attack[ht] + defence[at] + home_adv, -4, 4))
        nu = np.exp(np.clip(attack[at] + defence[ht], -4, 4))

        ll += weights[i] * (
            np.log(poisson.pmf(hg, mu) + 1e-10)
            + np.log(poisson.pmf(ag, nu) + 1e-10)
            + np.log(_dc_rho(hg, ag, mu, nu, rho) + 1e-10)
        )

    # L2 regularization toward Elo priors — prevents extreme params from weak opponents
    if elo_priors:
        for i, team in enumerate(teams):
            prior_atk, prior_def = elo_priors.get(team, (0.0, 0.0))
            ll -= reg_strength * (params[i] - prior_atk) ** 2
            ll -= reg_strength * (params[n + i] - prior_def) ** 2

    return -ll


def _time_decay_weights(dates: pd.Series, half_life_days: int = 120) -> np.ndarray:
    """
    Exponential time decay: recent matches weight 1.0, older ones decay.
    WC matches always get weight 2.0 (double) — most relevant signal.
    """
    today = pd.Timestamp.today()
    days_ago = (today - pd.to_datetime(dates)).dt.days.clip(lower=0).values
    decay = np.exp(-np.log(2) * days_ago / half_life_days)
    return decay


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

    def fit(self, matches: pd.DataFrame, historical: pd.DataFrame = None):
        """
        Fit the model on completed matches + optional historical competitive data.
        Historical matches are time-decay weighted; WC matches get double weight.
        Falls back to Elo priors for teams with no data at all.
        """
        from team_ratings import get_team_params, ELO_RATINGS
        self._elo_fallback = True

        wc_completed = matches[matches["status"] == "FINISHED"].copy()

        if historical is not None and not historical.empty:
            hist_completed = historical[historical["status"] == "FINISHED"].copy()
            # Keep matches where at least one team is a known WC/Elo team
            known_teams = set(ELO_RATINGS.keys()) | set(wc_completed["home_team"]) | set(wc_completed["away_team"])
            hist_completed = hist_completed[
                hist_completed["home_team"].isin(known_teams) |
                hist_completed["away_team"].isin(known_teams)
            ].copy()

            # Combine — WC matches get 2× weight via duplication, handled in weights below
            wc_completed["_source"] = "wc"
            hist_completed["_source"] = "historical"
            completed = pd.concat([wc_completed, hist_completed], ignore_index=True)
            print(f"  Training on {len(wc_completed)} WC matches + {len(hist_completed)} historical matches")
        else:
            wc_completed["_source"] = "wc"
            completed = wc_completed.copy()
            print(f"  Training on {len(completed)} WC matches only")

        if len(completed) < 5:
            print("  Warning: fewer than 5 completed matches — using Elo priors for unseen teams")

        self.teams = sorted(
            set(completed["home_team"].tolist() + completed["away_team"].tolist())
        )
        self._fitted_from_data = set(self.teams)
        n = len(self.teams)

        # Time-decay weights — WC matches get 2× boost as most relevant signal
        weights = _time_decay_weights(completed["date"])
        weights[completed["_source"].values == "wc"] *= 2.0

        # Elo priors for regularization — keeps teams with thin/unrepresentative data sane
        elo_priors = {team: get_team_params(team) for team in self.teams}

        # Initial params seeded from Elo priors so optimizer starts in a sensible place
        x0 = np.zeros(2 * n + 2)
        for i, team in enumerate(self.teams):
            prior_atk, prior_def = elo_priors.get(team, (0.0, 0.0))
            x0[i]     = prior_atk
            x0[n + i] = prior_def
        x0[2 * n]     = 0.0   # no home advantage at neutral WC venues
        x0[2 * n + 1] = -0.1  # rho

        result = minimize(
            _log_likelihood,
            x0,
            args=(completed, self.teams, weights, elo_priors, 2.5),
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

        params = result.x
        self.attack   = dict(zip(self.teams, params[:n]))
        self.defence  = dict(zip(self.teams, params[n:2*n]))
        self.home_adv = params[2 * n]
        self.rho      = params[2 * n + 1]
        self.fitted   = True
        print(f"  {n} teams fitted")

        # Add Elo-based params for any team not in completed matches.
        # Scale Elo deltas to match the fitted model's parameter space.
        from team_ratings import get_team_params, ELO_RATINGS
        if self.attack:
            fitted_atk_mean = np.mean(list(self.attack.values()))
            fitted_atk_std  = max(np.std(list(self.attack.values())), 0.01)
        else:
            fitted_atk_mean, fitted_atk_std = 0.0, 0.3

        for team in ELO_RATINGS:
            if team not in self.attack:
                raw_atk, raw_dfc = get_team_params(team)
                # Scale to match fitted distribution
                self.attack[team]  = fitted_atk_mean + raw_atk * fitted_atk_std / 0.3
                self.defence[team] = -raw_dfc * fitted_atk_std / 0.3
                if team not in self.teams:
                    self.teams.append(team)

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

        # If either team wasn't seen in completed matches, use pure Elo prediction
        fitted_from_data = getattr(self, '_fitted_from_data', set())
        if home not in fitted_from_data or away not in fitted_from_data:
            return _elo_predict(home, away)

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
