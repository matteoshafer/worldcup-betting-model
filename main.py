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
from betting import find_value_bets, kelly_fraction


def print_prediction(pred: dict):
    home, away = pred["home_team"], pred["away_team"]
    hs, as_ = pred["expected_home_goals"], pred["expected_away_goals"]

    print(f"\n{'='*55}")
    print(f"  {home}  vs  {away}")
    print(f"  Expected: {hs:.2f} - {as_:.2f}")
    print(f"{'='*55}")
    print(f"  {'Match Result':30} {'Prob':>8}  {'Fair Odds':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Home Win (' + home + ')':30} {pred['home_win']:>7.1%}  {1/pred['home_win']:>9.2f}")
    print(f"  {'Draw':30} {pred['draw']:>7.1%}  {1/pred['draw']:>9.2f}")
    print(f"  {'Away Win (' + away + ')':30} {pred['away_win']:>7.1%}  {1/pred['away_win']:>9.2f}")
    print(f"\n  {'Goals Markets':30} {'Prob':>8}  {'Fair Odds':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Over 1.5':30} {pred['over_1_5']:>7.1%}  {1/pred['over_1_5']:>9.2f}")
    print(f"  {'Over 2.5':30} {pred['over_2_5']:>7.1%}  {1/pred['over_2_5']:>9.2f}")
    print(f"  {'Over 3.5':30} {pred['over_3_5']:>7.1%}  {1/pred['over_3_5']:>9.2f}")
    print(f"  {'Both Teams to Score':30} {pred['btts']:>7.1%}  {1/pred['btts']:>9.2f}")
    print(f"{'='*55}")


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
    upcoming = matches[matches["status"].isin(["SCHEDULED", "TIMED"])].copy()
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
          f"{(matches['status'].isin(['SCHEDULED','TIMED'])).sum()} upcoming)")

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

    if args.backtest:
        backtest(matches, model)
    elif args.match:
        home, away = args.match
        try:
            pred = model.predict(home, away)
            print_prediction(pred)
        except ValueError as e:
            print(f"\nError: {e}")
    else:
        predict_upcoming(matches, model)


if __name__ == "__main__":
    main()
