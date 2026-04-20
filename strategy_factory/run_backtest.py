#!/usr/bin/env python3
"""
Quick Backtest Engine — main entry point.

Usage:
    python run_backtest.py --signal rsi_3 --condition quintile_1
    python run_backtest.py --signal rsi_3 --condition quintile_1 --hold 1,3,5,10,20
    python run_backtest.py --signal overnight_gap --condition below_threshold --threshold -0.5
    python run_backtest.py --signal rsi_3 --condition quintile_1 --test
"""
import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    NIFTY_200, TEST_UNIVERSE, START_DATE, END_DATE,
    NIFTY50_TICKER, DEFAULT_HOLD_DAYS,
)
from signals.scanner import download_universe, download_stock_data, build_panel
from backtest.engine import run_and_report


def main():
    parser = argparse.ArgumentParser(description="Quick Backtest Engine")
    parser.add_argument("--signal", type=str, required=True,
                        help="Feature/signal column name (e.g., rsi_3, overnight_gap)")
    parser.add_argument("--condition", type=str, default="quintile_1",
                        choices=["quintile_1", "quintile_5", "below_threshold", "above_threshold"],
                        help="Entry condition (default: quintile_1 = buy lowest quintile)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold value (for below/above_threshold conditions)")
    parser.add_argument("--hold", type=str, default=str(DEFAULT_HOLD_DAYS),
                        help="Hold period(s) in days, comma-separated (default: 5)")
    parser.add_argument("--test", action="store_true", help="Use small test universe")
    parser.add_argument("--force-download", action="store_true", help="Force re-download")
    args = parser.parse_args()

    universe = TEST_UNIVERSE if args.test else NIFTY_200
    universe_name = "Test (5 stocks)" if args.test else "Nifty 200"
    hold_days_list = [int(x) for x in args.hold.split(",")]

    print(f"\n{'#'*60}")
    print(f"  STRATEGY FACTORY — Quick Backtest")
    print(f"  Signal: {args.signal} | Condition: {args.condition}")
    print(f"  Universe: {universe_name} | Hold: {hold_days_list}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"{'#'*60}")

    # Download data
    stock_data = download_universe(universe, START_DATE, END_DATE, force=args.force_download)

    # Download index
    print(f"\nDownloading Nifty 50 index data...")
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE, force=args.force_download)

    # Build panel
    panel = build_panel(stock_data, index_data)

    if panel.empty:
        print("ERROR: No data. Check downloads.")
        sys.exit(1)

    # Verify signal column exists
    if args.signal not in panel.columns:
        print(f"\nERROR: Signal '{args.signal}' not found.")
        exclude = {"ticker", "date"} | {f"fwd_{d}d" for d in [1, 3, 5, 10, 20]}
        features = [c for c in panel.columns if c not in exclude]
        print(f"Available signals: {', '.join(sorted(features))}")
        sys.exit(1)

    # Feasibility check (Rule #9)
    print(f"\n  Feasibility check (Rule #9)...")
    total_days = panel["date"].nunique()
    years = total_days / 252
    # Quick estimate of qualifying trades
    qualifying = panel.dropna(subset=[args.signal])
    estimated_trades = len(qualifying) / 5  # ~1/5 stocks in each quintile
    trades_per_year = estimated_trades / max(years, 1)
    print(f"  Estimated qualifying universe: ~{estimated_trades:.0f} stock-days")
    print(f"  Estimated trades/year: ~{trades_per_year:.0f}")

    # Run backtest
    run_and_report(
        stock_data=stock_data,
        panel=panel,
        signal_col=args.signal,
        signal_condition=args.condition,
        threshold=args.threshold,
        hold_days_list=hold_days_list,
        universe_name=universe_name,
    )


if __name__ == "__main__":
    main()
