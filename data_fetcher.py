"""
World Cup 2026 Data Fetcher
Fetches match results, team stats, and odds from free APIs.
Primary source: football-data.org (free tier, no key needed for WC)
"""

import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

BASE_URL = "https://api.football-data.org/v4"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Free tier doesn't need a key for World Cup data, but add yours here for higher rate limits
API_KEY = ""

HEADERS = {"X-Auth-Token": API_KEY} if API_KEY else {}
WC_2026_ID = 2000  # football-data.org competition ID for FIFA World Cup


def _get(endpoint: str) -> dict:
    resp = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_matches(save: bool = True) -> pd.DataFrame:
    """Fetch all World Cup 2026 matches (played and scheduled)."""
    data = _get(f"competitions/{WC_2026_ID}/matches")
    matches = data.get("matches", [])

    rows = []
    for m in matches:
        rows.append({
            "match_id":       m["id"],
            "date":           m["utcDate"][:10],
            "stage":          m["stage"],
            "group":          m.get("group", ""),
            "home_team":      m["homeTeam"]["name"],
            "away_team":      m["awayTeam"]["name"],
            "home_goals":     m["score"]["fullTime"]["home"],
            "away_goals":     m["score"]["fullTime"]["away"],
            "status":         m["status"],
        })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(DATA_DIR / "matches.csv", index=False)
        print(f"Saved {len(df)} matches to data/matches.csv")
    return df


def fetch_standings(save: bool = True) -> pd.DataFrame:
    """Fetch current group stage standings."""
    data = _get(f"competitions/{WC_2026_ID}/standings")
    rows = []
    for standing in data.get("standings", []):
        group = standing.get("group", "")
        for entry in standing.get("table", []):
            team = entry["team"]["name"]
            rows.append({
                "group":       group,
                "team":        team,
                "played":      entry["playedGames"],
                "won":         entry["won"],
                "drawn":       entry["draw"],
                "lost":        entry["lost"],
                "goals_for":   entry["goalsFor"],
                "goals_against": entry["goalsAgainst"],
                "goal_diff":   entry["goalDifference"],
                "points":      entry["points"],
            })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(DATA_DIR / "standings.csv", index=False)
        print(f"Saved standings for {len(df)} teams")
    return df


def fetch_team_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute attack/defence strength per team from completed matches.
    Used as features for the Poisson model.
    """
    completed = matches_df[matches_df["status"] == "FINISHED"].copy()
    if completed.empty:
        return pd.DataFrame()

    league_avg_home = completed["home_goals"].mean()
    league_avg_away = completed["away_goals"].mean()
    league_avg = (league_avg_home + league_avg_away) / 2

    teams = pd.concat([completed["home_team"], completed["away_team"]]).unique()
    rows = []
    for team in teams:
        home = completed[completed["home_team"] == team]
        away = completed[completed["away_team"] == team]

        goals_scored   = home["home_goals"].sum() + away["away_goals"].sum()
        goals_conceded = home["away_goals"].sum() + away["home_goals"].sum()
        games          = len(home) + len(away)

        if games == 0:
            continue

        attack_str  = (goals_scored   / games) / (league_avg or 1)
        defence_str = (goals_conceded / games) / (league_avg or 1)

        rows.append({
            "team":         team,
            "games":        games,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "attack_strength":  round(attack_str, 4),
            "defence_strength": round(defence_str, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "team_stats.csv", index=False)
    return df


def load_matches() -> pd.DataFrame:
    path = DATA_DIR / "matches.csv"
    if path.exists():
        return pd.read_csv(path)
    return fetch_matches()


def load_team_stats() -> pd.DataFrame:
    path = DATA_DIR / "team_stats.csv"
    if path.exists():
        return pd.read_csv(path)
    matches = load_matches()
    return fetch_team_stats(matches)


if __name__ == "__main__":
    print("Fetching World Cup 2026 data...")
    matches = fetch_matches()
    print(matches[["date", "home_team", "away_team", "home_goals", "away_goals", "status"]].head(10))
    stats = fetch_team_stats(matches)
    print("\nTeam stats:")
    print(stats.sort_values("attack_strength", ascending=False).head(10))
