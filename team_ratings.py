"""
Pre-tournament team ratings for World Cup 2026.
Based on FIFA rankings and historical Elo ratings (May 2026).
Used as priors when tournament match data is sparse.
"""

# Elo ratings (approx, May 2026). Higher = stronger.
ELO_RATINGS = {
    # Group A
    "United States":       1850,
    "Canada":              1720,
    "Mexico":              1730,
    "Uruguay":             1810,
    # Group B
    "Argentina":           2050,
    "Chile":               1680,
    "Peru":                1630,
    "Australia":           1650,
    # Group C
    "Spain":               1990,
    "Morocco":             1740,
    "Croatia":             1880,
    "Belgium":             1900,
    # Group D
    "France":              2000,
    "England":             1950,
    "Netherlands":         1920,
    "Egypt":               1680,
    # Group E
    "Brazil":              1980,
    "Portugal":            1950,
    "Colombia":            1770,
    "Senegal":             1720,
    # Group F
    "Germany":             1960,
    "Japan":               1800,
    "South Korea":         1730,
    "Saudi Arabia":        1640,
    # Group G
    "Italy":               1910,
    "Switzerland":         1840,
    "Ivory Coast":         1700,
    "Nigeria":             1710,
    # Group H
    "Qatar":               1540,
    "Ecuador":             1690,
    "Czechia":             1740,
    "Bosnia-Herzegovina":  1680,
    # Group I / Other WC teams
    "Türkiye":             1760,
    "Hungary":             1700,
    "Romania":             1690,
    "Norway":              1730,
    "Iran":                1680,
    "New Zealand":         1540,
    "Iraq":                1620,
    "Sweden":              1750,
    "Tunisia":             1650,
    "Algeria":             1670,
    "Jordan":              1580,
    "Austria":             1770,
    "Venezuela":           1640,
    "Indonesia":           1530,
    "United Arab Emirates": 1580,
    # Other teams
    "Paraguay":            1640,
    "South Africa":        1580,
    "Haiti":               1490,
    "Scotland":            1700,
    "Curaçao":             1500,
    "Cape Verde":          1550,
    "Georgia":             1670,
    "Slovenia":            1660,
    "Slovakia":            1660,
}


def elo_to_attack_defence(elo: float, league_avg_elo: float = 1750) -> tuple[float, float]:
    """
    Convert Elo rating to attack/defence parameters for Poisson model.
    Stronger teams get higher attack and lower (better) defence.
    """
    delta = (elo - league_avg_elo) / 400
    attack  = delta * 0.6
    defence = -delta * 0.4
    return round(attack, 4), round(defence, 4)


def get_team_params(team: str) -> tuple[float, float]:
    """Return (attack, defence) Poisson params for a team based on Elo."""
    elo = ELO_RATINGS.get(team, 1650)
    return elo_to_attack_defence(elo)


def elo_win_probability(home_elo: float, away_elo: float) -> tuple[float, float, float]:
    """
    Simple Elo-based match outcome probabilities.
    Returns (home_win, draw, away_win).
    """
    diff = home_elo - away_elo
    home_exp = 1 / (1 + 10 ** (-diff / 400))
    away_exp = 1 - home_exp

    # Estimate draw probability based on closeness of teams
    draw = 0.28 - 0.005 * abs(diff) / 50
    draw = max(0.10, min(0.32, draw))

    home_win = home_exp * (1 - draw)
    away_win = away_exp * (1 - draw)
    return round(home_win, 4), round(draw, 4), round(away_win, 4)
