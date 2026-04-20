#!/usr/bin/env python3
"""
Signal Scanner — main entry point.

Usage:
    python run_scan.py                      # Full scan, Nifty 200, 5-day forward
    python run_scan.py --test               # Quick test with 5 stocks
    python run_scan.py --fwd-days 10        # Scan with 10-day forward returns
    python run_scan.py --force-download     # Re-download all data
    python run_scan.py --detail rsi_3       # Show detailed quintile breakdown for one feature
"""
import argparse
import os
import sys
import json
from datetime import datetime

# Ensure we run from the strategy_factory directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    NIFTY_200, TEST_UNIVERSE, START_DATE, END_DATE,
    NIFTY50_TICKER, FORWARD_RETURN_DAYS, DEFAULT_FORWARD_DAYS,
)
from signals.scanner import (
    download_universe, download_stock_data, build_panel,
    run_scan, print_scan_summary, print_quintile_result, classify_signal,
)


def main():
    parser = argparse.ArgumentParser(description="Signal Scanner — Quintile Sorter")
    parser.add_argument("--test", action="store_true", help="Use small test universe (5 stocks)")
    parser.add_argument("--fwd-days", type=int, default=DEFAULT_FORWARD_DAYS,
                        help=f"Forward return period in days (default: {DEFAULT_FORWARD_DAYS})")
    parser.add_argument("--force-download", action="store_true", help="Force re-download all data")
    parser.add_argument("--detail", type=str, default=None,
                        help="Show detailed quintile breakdown for a specific feature")
    parser.add_argument("--all-periods", action="store_true",
                        help="Scan across all forward return periods")
    parser.add_argument("--save", action="store_true", help="Save results to results/scans/")
    args = parser.parse_args()

    universe = TEST_UNIVERSE if args.test else NIFTY_200
    universe_name = "Test (5 stocks)" if args.test else "Nifty 200"

    print(f"\n{'#'*60}")
    print(f"  STRATEGY FACTORY — Signal Scanner")
    print(f"  Universe: {universe_name} ({len(universe)} stocks)")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Forward returns: {args.fwd_days}d")
    print(f"{'#'*60}")

    # Download data
    stock_data = download_universe(universe, START_DATE, END_DATE, force=args.force_download)

    # Download Nifty 50 index for relative strength
    print(f"\nDownloading Nifty 50 index data...")
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE, force=args.force_download)

    # Build panel
    panel = build_panel(stock_data, index_data)

    if panel.empty:
        print("ERROR: No data to scan. Check your data downloads.")
        sys.exit(1)

    if args.all_periods:
        # Scan all forward return periods
        for fwd in FORWARD_RETURN_DAYS:
            results = run_scan(panel, fwd_days=fwd)
            print_scan_summary(results, fwd_days=fwd)
    else:
        # Single period scan
        results = run_scan(panel, fwd_days=args.fwd_days)

        # Show detailed breakdown if requested
        if args.detail:
            matching = [r for r in results if r["feature"] == args.detail]
            if matching:
                print_quintile_result(matching[0], classify_signal(matching[0]))
            else:
                print(f"\nFeature '{args.detail}' not found. Available features:")
                for r in results:
                    print(f"  - {r['feature']}")
        else:
            # Show detailed breakdown for all strong/moderate signals
            strong_moderate = [r for r in results if r["classification"] in ("STRONG", "MODERATE")]
            if strong_moderate:
                print(f"\n{'='*60}")
                print("  DETAILED QUINTILE BREAKDOWN — Strong & Moderate Signals")
                print(f"{'='*60}")
                for r in strong_moderate:
                    print_quintile_result(r, r["classification"])

        # Print summary
        print_scan_summary(results, fwd_days=args.fwd_days)

    # Save results
    if args.save:
        os.makedirs("results/scans", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/scans/scan_{universe_name.replace(' ', '_')}_{args.fwd_days}d_{timestamp}.json"

        save_data = []
        for r in results:
            save_data.append({
                "feature": r["feature"],
                "spread": round(r["spread"], 4),
                "t_stat": round(r["t_stat"], 4),
                "direction": r["direction"],
                "classification": r["classification"],
                "monotonic_score": round(r["monotonic_score"], 4),
                "n_observations": r["n_observations"],
            })
        with open(filename, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()
