# World Cup 2026 Betting Model

A sports betting model for the 2026 FIFA World Cup using Dixon-Coles Poisson regression. Generates win/draw/loss probabilities, over/under and BTTS predictions, and identifies value bets against bookmaker odds using the Kelly criterion.

## How It Works

### Model — Dixon-Coles Poisson

Each team is assigned an **attack strength** and **defence strength** parameter, fit by maximum likelihood on completed match results. Goals scored by each team follow a Poisson distribution:

```
λ_home = exp(attack_home + defence_away + home_advantage)
λ_away = exp(attack_away + defence_home)
```

A low-score correction (the Dixon-Coles ρ parameter) adjusts probabilities for 0-0, 1-0, 0-1, and 1-1 scorelines, which are systematically mis-priced by naive Poisson models.

### Betting — Kelly Criterion

For each market (home win, draw, away win, over 2.5, BTTS), the model compares its probability estimate against the bookmaker's implied probability. If there's a positive edge, a **quarter-Kelly** stake is recommended:

```
Kelly stake = (b * p - (1 - p)) / b  ×  0.25
```

where `p` = model probability, `b` = net decimal odds.

## Quick Start

```bash
pip install -r requirements.txt
```

Fetch data and predict all upcoming matches:
```bash
python main.py --refresh   # first run: fetch from API
python main.py             # subsequent runs: use cached data
```

Predict a specific match:
```bash
python main.py --match "Brazil" "France"
```

Backtest on completed matches:
```bash
python main.py --backtest
```

### Example Output

```
=======================================================
  Brazil  vs  France
  Expected: 1.42 - 1.21
=======================================================
  Match Result                      Prob    Fair Odds
  --------------------------------------------------
  Home Win (Brazil)               43.2%         2.31
  Draw                            26.1%         3.83
  Away Win (France)               30.7%         3.26

  Goals Markets                     Prob    Fair Odds
  --------------------------------------------------
  Over 1.5                        72.4%         1.38
  Over 2.5                        48.9%         2.04
  Over 3.5                        27.3%         3.66
  Both Teams to Score             54.1%         1.85
=======================================================
```

## Finding Value Bets

To use the value betting module, provide a CSV of bookmaker odds (`data/odds.csv`) with columns:

```
home_team, away_team, market, odds
```

Markets: `home_win`, `draw`, `away_win`, `over_2_5`, `btts_yes`

Then run:
```python
from data_fetcher import load_matches
from poisson_model import DixonColesModel
from betting import find_value_bets
import pandas as pd

matches = load_matches()
model = DixonColesModel()
model.fit(matches)

upcoming = matches[matches["status"].isin(["SCHEDULED", "TIMED"])]
predictions = [model.predict(r["home_team"], r["away_team"])
               for _, r in upcoming.iterrows()]

odds = pd.read_csv("data/odds.csv")
value_bets = find_value_bets(predictions, odds, min_edge=0.05)
print(value_bets)
```

## Data Source

Match data is fetched from [football-data.org](https://football-data.org) (free tier). For higher rate limits, add your API key to `data_fetcher.py`:

```python
API_KEY = "your_key_here"
```

## Project Structure

```
├── main.py           # CLI — predict matches, backtest, refresh data
├── poisson_model.py  # Dixon-Coles model fitting and prediction
├── betting.py        # Kelly criterion, value bet detection, bankroll sim
├── data_fetcher.py   # football-data.org API wrapper
├── data/             # Cached CSVs (matches, standings, team stats, odds)
└── requirements.txt
```

## Limitations

- Model accuracy improves as more group stage matches complete
- World Cup has no home advantage (neutral venues) — the home_adv parameter will converge toward 0
- Small sample sizes early in the tournament increase uncertainty
- Always gamble responsibly
