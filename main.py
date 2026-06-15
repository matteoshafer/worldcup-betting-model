"""
World Cup 2026 Betting Model
Main entry point — fetches data, fits model, generates predictions and value bets.

Usage:
    python main.py                          # Predict all upcoming matches
    python main.py --match "Brazil" "France"  # Predict a specific match
    python main.py --backtest               # Backtest on completed matches
"""

import argparse
import pandas as pd
from data_fetcher import (load_matches, fetch_matches, fetch_standings, fetch_team_stats,
                          load_historical_matches, fetch_historical_competitive_matches)
from poisson_model import DixonColesModel
from betting import find_value_bets, kelly_fraction, value_edge


def _verdict(model_prob: float, book_odds: float) -> str:
    """Return edge + Kelly stake verdict string for a market."""
    edge = value_edge(model_prob, book_odds)
    kelly = kelly_fraction(model_prob, book_odds)
    if edge >= 0.04:
        return f"  +{edge:.1%} edge  ✅ BET  ({kelly*100:.1f}% Kelly)"
    elif edge >= 0.02:
        return f"  +{edge:.1%} edge  ⚠️  MARGINAL"
    else:
        return f"  {edge:+.1%} edge  ❌ No value"


def print_prediction(pred: dict, book_odds: dict = None):
    """
    Print match prediction with optional bookmaker value analysis.

    book_odds keys: home_win, draw, away_win, over_1_5, over_2_5, over_3_5, btts
    Values are decimal odds (e.g. 1.85, 3.40, 4.50).
    """
    home, away = pred["home_team"], pred["away_team"]
    hs, as_ = pred["expected_home_goals"], pred["expected_away_goals"]
    W = 65 if book_odds else 55

    print(f"\n{'='*W}")
    print(f"  {home}  vs  {away}")
    print(f"  Expected: {hs:.2f} - {as_:.2f}")
    if book_odds:
        print(f"  [Bookmaker odds supplied — value analysis shown]")
    print(f"{'='*W}")

    markets = [
        ("Match Result", None, None),
        (f"Home Win ({home})", "home_win", pred["home_win"]),
        ("Draw",               "draw",     pred["draw"]),
        (f"Away Win ({away})", "away_win", pred["away_win"]),
        (None, None, None),
        ("Goals Markets", None, None),
        ("Over 1.5",           "over_1_5", pred["over_1_5"]),
        ("Over 2.5",           "over_2_5", pred["over_2_5"]),
        ("Over 3.5",           "over_3_5", pred["over_3_5"]),
        ("Both Teams to Score","btts",     pred["btts"]),
    ]

    for label, key, prob in markets:
        if label is None:
            print()
            continue
        if prob is None:
            print(f"  {label}")
            print(f"  {'-'*(W-4)}")
            continue

        fair = 1 / prob
        line = f"  {label:32} {prob:>7.1%}   {fair:>8.2f}"
        if book_odds and key in book_odds:
            bo = book_odds[key]
            line += f"   {bo:>8.2f}{_verdict(prob, bo)}"
        print(line)

    print(f"{'='*W}")


def backtest(matches: pd.DataFrame, model: DixonColesModel) -> dict:
    """
    Rolling backtest: for each completed match, fit on all prior matches
    and predict the outcome. Report calibration and ROI at fair odds.
    """
    completed = matches[matches["status"] == "FINISHED"].copy().reset_index(drop=True)
    if len(completed) < 10:
        print("Not enough completed matches for a meaningful backtest (need 10+).")
        return {}

    correct = 0
    total   = 0
    log_loss_sum = 0.0

    print(f"\nBacktesting on {len(completed)} completed matches...")
    print(f"{'Match':<45} {'Predicted':>12} {'Actual':>10} {'Correct':>8}")
    print("-" * 80)

    for i in range(5, len(completed)):
        train = completed.iloc[:i]
        test  = completed.iloc[i]

        m = DixonColesModel()
        try:
            m.fit(train)
            pred = m.predict(test["home_team"], test["away_team"])
        except Exception:
            continue

        hg, ag = int(test["home_goals"]), int(test["away_goals"])
        actual = "home_win" if hg > ag else ("draw" if hg == ag else "away_win")
        predicted = max(["home_win", "draw", "away_win"], key=lambda k: pred[k])

        is_correct = predicted == actual
        correct += int(is_correct)
        total   += 1
        log_loss_sum += -pd.np.log(pred[actual] + 1e-10) if hasattr(pd, 'np') else 0

        label = f"{test['home_team'][:18]} vs {test['away_team'][:18]}"
        print(f"{label:<45} {predicted:>12} {actual:>10} {'✓' if is_correct else '✗':>8}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")
    return {"accuracy": accuracy, "total_matches": total}


def predict_upcoming(matches: pd.DataFrame, model: DixonColesModel):
    """Print predictions for all scheduled (not yet played) matches."""
    upcoming = matches[matches["status"].isin(["SCHEDULED", "TIMED", "STATUS_SCHEDULED", "STATUS_TIMED"])].copy()
    if upcoming.empty:
        print("No upcoming matches found.")
        return

    print(f"\nPredictions for {len(upcoming)} upcoming matches:")
    for _, row in upcoming.iterrows():
        try:
            pred = model.predict(row["home_team"], row["away_team"])
            print_prediction(pred)
        except ValueError as e:
            print(f"\n  Skipping {row['home_team']} vs {row['away_team']}: {e}")


def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 Betting Model")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predict a specific match")
    parser.add_argument("--backtest", action="store_true",
                        help="Run rolling backtest on completed matches")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch latest data from API")
    parser.add_argument("--refresh-history", action="store_true",
                        help="Re-fetch historical competitive match data")
    parser.add_argument("--odds", nargs=3, metavar=("HOME", "DRAW", "AWAY"),
                        type=float, help="Bookmaker decimal odds: home draw away")
    parser.add_argument("--over15",  type=float, metavar="ODDS", help="Book odds for Over 1.5")
    parser.add_argument("--over25",  type=float, metavar="ODDS", help="Book odds for Over 2.5")
    parser.add_argument("--over35",  type=float, metavar="ODDS", help="Book odds for Over 3.5")
    parser.add_argument("--btts",    type=float, metavar="ODDS", help="Book odds for Both Teams to Score")
    args = parser.parse_args()

    print("=" * 60)
    print("  WORLD CUP 2026 BETTING MODEL")
    print("=" * 60)

    # Load or refresh WC data
    if args.refresh:
        print("\nFetching latest World Cup data...")
        matches = fetch_matches()
        fetch_standings()
    else:
        matches = load_matches()

    print(f"\nLoaded {len(matches)} WC matches "
          f"({(matches['status'] == 'FINISHED').sum()} completed, "
          f"{(matches['status'].isin(['SCHEDULED','TIMED','STATUS_SCHEDULED','STATUS_TIMED'])).sum()} upcoming)")

    # Load or refresh historical competitive data
    if args.refresh_history:
        print("\nFetching historical competitive matches (this may take ~30s)...")
        historical = fetch_historical_competitive_matches()
    else:
        historical = load_historical_matches()
        if not historical.empty:
            print(f"Loaded {len(historical)} historical competitive matches")

    # Fit model
    print("\nFitting Dixon-Coles model...")
    model = DixonColesModel()
    model.fit(matches, historical=historical if not historical.empty else None)

    # Build book_odds dict from CLI flags
    book_odds = None
    if args.odds or args.over25 or args.over15 or args.over35 or args.btts:
        book_odds = {}
        if args.odds:
            book_odds["home_win"] = args.odds[0]
            book_odds["draw"]     = args.odds[1]
            book_odds["away_win"] = args.odds[2]
        if args.over15:  book_odds["over_1_5"] = args.over15
        if args.over25:  book_odds["over_2_5"] = args.over25
        if args.over35:  book_odds["over_3_5"] = args.over35
        if args.btts:    book_odds["btts"]     = args.btts

    if args.backtest:
        backtest(matches, model)
    elif args.match:
        home, away = args.match
        try:
            pred = model.predict(home, away)
            print_prediction(pred, book_odds=book_odds)
        except ValueError as e:
            print(f"\nError: {e}")
    else:
        predict_upcoming(matches, model)


if __name__ == "__main__":
    main()
