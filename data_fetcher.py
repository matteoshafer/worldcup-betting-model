"""
World Cup 2026 Data Fetcher
Fetches match results, team stats, and odds from free APIs.
Primary source: football-data.org (free tier, no key needed for WC)
"""

import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
ESPN_SOCCER = "https://site.api.espn.com/apis/site/v2/sports/soccer"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Competitive national-team competitions to pull historical data from.
# Friendlies are excluded by filtering competition type from the ESPN response.
HISTORICAL_COMPETITIONS = [
    # WC Qualifiers — Sept/Oct/Nov 2025 window + March 2026 window
    ("fifa.worldq.uefa",        "2025-09-01", "2025-11-21"),
    ("fifa.worldq.uefa",        "2026-03-20", "2026-03-26"),
    ("fifa.worldq.conmebol",    "2025-09-01", "2025-11-21"),
    ("fifa.worldq.conmebol",    "2026-03-20", "2026-03-26"),
    ("fifa.worldq.caf",         "2025-09-01", "2025-11-21"),
    ("fifa.worldq.caf",         "2026-03-20", "2026-03-26"),
    ("fifa.worldq.afc",         "2025-09-01", "2025-11-21"),
    ("fifa.worldq.concacaf",    "2025-09-01", "2025-11-21"),
    ("fifa.worldq.ofc",         "2025-09-01", "2025-11-21"),
    # AFC 3rd-round qualifiers 2024 (Australia, Japan, South Korea, Saudi Arabia etc.)
    ("fifa.worldq.afc",         "2024-03-01", "2024-06-15"),
    # AFC Asian Cup 2024 (Jan-Feb, Qatar) — key tournament for AFC nations
    ("afc.cup",                 "2024-01-12", "2024-02-11"),
    # AFCON 2025 (Morocco)
    ("caf.nations",             "2025-12-21", "2026-02-01"),
    # CONCACAF Gold Cup 2025
    ("concacaf.gold.cup",       "2025-06-14", "2025-07-07"),
    # UEFA Nations League finals
    ("uefa.nations",            "2025-06-01", "2025-06-09"),
    # Copa America 2024
    ("conmebol.america",        "2024-06-20", "2024-07-15"),
]


def fetch_matches(save: bool = True) -> pd.DataFrame:
    """Fetch all available World Cup 2026 matches from ESPN (no API key needed)."""
    from datetime import datetime, timedelta

    rows = []
    seen = set()

    # Scan from tournament start through end of group stage
    start = datetime(2026, 6, 11)
    for i in range(30):
        dt = (start + timedelta(days=i)).strftime("%Y%m%d")
        try:
            resp = requests.get(f"{ESPN_BASE}/scoreboard?dates={dt}", timeout=10)
            events = resp.json().get("events", [])
        except Exception:
            continue

        for e in events:
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            comp = e["competitions"][0]
            competitors = {c.get("homeAway", ""): c for c in comp.get("competitors", [])}
            home = competitors.get("home", {})
            away = competitors.get("away", {})
            status_type = comp["status"]["type"]

            rows.append({
                "match_id":   e["id"],
                "date":       e["date"][:10],
                "stage":      e.get("season", {}).get("slug", ""),
                "home_team":  home.get("team", {}).get("displayName", ""),
                "away_team":  away.get("team", {}).get("displayName", ""),
                "home_goals": int(home.get("score", 0)) if status_type.get("completed") else None,
                "away_goals": int(away.get("score", 0)) if status_type.get("completed") else None,
                "status":     "FINISHED" if status_type.get("completed") else status_type.get("name", "SCHEDULED"),
            })

    df = pd.DataFrame(rows)
    if save:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(DATA_DIR / "matches.csv", index=False)
        print(f"Saved {len(df)} matches ({(df['status']=='FINISHED').sum()} completed)")
    return df


def fetch_standings(save: bool = True) -> pd.DataFrame:
    """Fetch current group stage standings from ESPN."""
    try:
        resp = requests.get(f"{ESPN_BASE}/standings", timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"  Could not fetch standings: {e}")
        return pd.DataFrame()

    rows = []
    for group in data.get("standings", []):
        group_name = group.get("name", "")
        for entry in group.get("standings", {}).get("entries", []):
            team = entry.get("team", {}).get("displayName", "")
            stats = {s["name"]: s["value"] for s in entry.get("stats", [])}
            rows.append({
                "group":         group_name,
                "team":          team,
                "played":        int(stats.get("gamesPlayed", 0)),
                "won":           int(stats.get("wins", 0)),
                "drawn":         int(stats.get("ties", 0)),
                "lost":          int(stats.get("losses", 0)),
                "goals_for":     int(stats.get("pointsFor", 0)),
                "goals_against": int(stats.get("pointsAgainst", 0)),
                "points":        int(stats.get("points", 0)),
            })

    df = pd.DataFrame(rows)
    if save and not df.empty:
        df.to_csv(DATA_DIR / "standings.csv", index=False)
        print(f"Saved standings for {len(df)} teams")
    return df


def fetch_historical_competitive_matches(save: bool = True) -> pd.DataFrame:
    """
    Fetch ~1 year of competitive national-team results (no friendlies)
    from ESPN across WC qualifiers, AFCON, Gold Cup, and Nations League.
    """
    rows = []
    seen = set()

    for slug, start_str, end_str in HISTORICAL_COMPETITIONS:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_str,   "%Y-%m-%d")
        days     = (end_dt - start_dt).days + 1
        url_base = f"{ESPN_SOCCER}/{slug}/scoreboard"
        fetched  = 0

        for i in range(days):
            dt = (start_dt + timedelta(days=i)).strftime("%Y%m%d")
            try:
                resp   = requests.get(f"{url_base}?dates={dt}", timeout=8)
                events = resp.json().get("events", [])
            except Exception:
                continue

            for e in events:
                key = (slug, e["id"])
                if key in seen:
                    continue
                seen.add(key)

                comp        = e["competitions"][0]
                status_type = comp["status"]["type"]
                if not status_type.get("completed"):
                    continue

                competitors = {c.get("homeAway", ""): c for c in comp.get("competitors", [])}
                home = competitors.get("home", {})
                away = competitors.get("away", {})

                home_name = home.get("team", {}).get("displayName", "")
                away_name = away.get("team", {}).get("displayName", "")
                if not home_name or not away_name:
                    continue

                rows.append({
                    "match_id":   f"{slug}_{e['id']}",
                    "date":       e["date"][:10],
                    "competition": slug,
                    "home_team":  home_name,
                    "away_team":  away_name,
                    "home_goals": int(home.get("score", 0)),
                    "away_goals": int(away.get("score", 0)),
                    "status":     "FINISHED",
                })
                fetched += 1

        if fetched:
            print(f"  {slug}: {fetched} matches")

    df = pd.DataFrame(rows)
    if save and not df.empty:
        df.to_csv(DATA_DIR / "historical_matches.csv", index=False)
        print(f"Saved {len(df)} historical competitive matches")
    return df


def load_historical_matches() -> pd.DataFrame:
    path = DATA_DIR / "historical_matches.csv"
    if path.exists():
        return pd.read_csv(path)
    return fetch_historical_competitive_matches()


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
