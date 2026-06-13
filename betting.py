"""
Betting Strategy
Value bet detection and Kelly criterion stake sizing.
"""

import pandas as pd
import numpy as np
from typing import Optional


def implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / odds


def value_edge(model_prob: float, odds: float) -> float:
    """
    Return edge = model_prob - implied_prob.
    Positive edge = value bet (model thinks it's more likely than the market does).
    """
    return model_prob - implied_prob(odds)


def kelly_fraction(model_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly criterion stake as % of bankroll.
    Uses quarter-Kelly by default (safer for model uncertainty).

    Returns 0 if the bet has no edge.
    """
    edge = value_edge(model_prob, odds)
    if edge <= 0:
        return 0.0
    b = odds - 1  # net odds
    kelly = (b * model_prob - (1 - model_prob)) / b
    return max(0.0, round(kelly * fraction, 4))


def find_value_bets(
    predictions: list[dict],
    odds_df: pd.DataFrame,
    min_edge: float = 0.05,
    min_odds: float = 1.5,
) -> pd.DataFrame:
    """
    Compare model probabilities against bookmaker odds to find value bets.

    Args:
        predictions: list of dicts from DixonColesModel.predict()
        odds_df: DataFrame with columns [home_team, away_team, market, odds]
                 markets: 'home_win', 'draw', 'away_win', 'over_2_5', 'under_2_5', 'btts_yes', 'btts_no'
        min_edge: minimum required edge to flag as value (default 5%)
        min_odds: minimum decimal odds to consider (default 1.5)

    Returns:
        DataFrame of value bets sorted by edge descending
    """
    value_bets = []

    market_map = {
        "home_win":  "home_win",
        "draw":      "draw",
        "away_win":  "away_win",
        "over_2_5":  "over_2_5",
        "btts_yes":  "btts",
    }

    for pred in predictions:
        home, away = pred["home_team"], pred["away_team"]
        match_odds = odds_df[
            (odds_df["home_team"] == home) & (odds_df["away_team"] == away)
        ]

        for market_label, pred_key in market_map.items():
            market_odds_row = match_odds[match_odds["market"] == market_label]
            if market_odds_row.empty:
                continue

            odds = float(market_odds_row["odds"].iloc[0])
            if odds < min_odds:
                continue

            model_prob = pred.get(pred_key, 0)
            edge = value_edge(model_prob, odds)

            if edge >= min_edge:
                stake = kelly_fraction(model_prob, odds)
                value_bets.append({
                    "match":        f"{home} vs {away}",
                    "market":       market_label,
                    "model_prob":   round(model_prob, 4),
                    "implied_prob": round(implied_prob(odds), 4),
                    "odds":         odds,
                    "edge":         round(edge, 4),
                    "kelly_stake":  f"{stake*100:.1f}% of bankroll",
                })

    df = pd.DataFrame(value_bets)
    if not df.empty:
        df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    return df


def simulate_bankroll(
    bets: list[dict],
    initial_bankroll: float = 1000.0,
) -> pd.DataFrame:
    """
    Simulate bankroll growth over a sequence of bets.

    Each bet dict needs: stake_pct, odds, won (bool).
    """
    bankroll = initial_bankroll
    history = [{"bet": 0, "bankroll": bankroll}]

    for i, bet in enumerate(bets, 1):
        stake = bankroll * bet["stake_pct"]
        if bet["won"]:
            bankroll += stake * (bet["odds"] - 1)
        else:
            bankroll -= stake
        history.append({"bet": i, "bankroll": round(bankroll, 2)})

    return pd.DataFrame(history)
