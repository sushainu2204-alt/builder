#!/usr/bin/env python3
"""
Week-by-week backtest for the last 3 months.
Strategy: RSI(3) bear dip + market breadth < 40%
"""
import os, sys, warnings
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel, zscore_normalize
from research_5k_v5 import get_nifty_regime, run_regime_filtered_backtest
from research_deep import add_advanced_features

CAPITAL = 600000
MAX_POS = 5
PER_POS = CAPITAL // MAX_POS

def main():
    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)
    nifty_regime = get_nifty_regime(index_data)
    panel = add_advanced_features(panel, stock_data, index_data)

    # Filter panel: breadth < 40%
    panel_filtered = panel[panel["mkt_breadth"] < 0.4].copy()

    # Run backtest on last 3 months
    result = run_regime_filtered_backtest(
        stock_data, panel_filtered, nifty_regime,
        signal_col="rsi_3", condition="lowest",
        hold_days=5, max_positions=5, num_buckets=10,
        allowed_regime=0, block_reentry_days=5,
        start_date="2025-12-22",
    )

    if result is None or result["trades"].empty:
        print("No trades in last 3 months.")
        return

    trades = result["trades"]
    trades["week_start"] = trades["entry_date"].dt.to_period("W").apply(lambda r: r.start_time)

    print(f"\n{'#'*90}")
    print(f"  WEEK-BY-WEEK BACKTEST — Last 3 Months")
    print(f"  Strategy: RSI(3) Bear Dip + Breadth < 40% | Capital: {CAPITAL:,}")
    print(f"{'#'*90}")

    weeks = sorted(trades["week_start"].unique())
    cum_pnl = 0
    total_trades = 0
    total_wins = 0
    total_stops = 0

    print(f"\n  {'Week':<12} {'Trades':>6} {'Stocks':<45} {'WR':>5} {'Stops':>5} {'Week P&L':>10} {'Cum P&L':>10}")
    print(f"  {'-'*82}")

    for week in weeks:
        wt = trades[trades["week_start"] == week].sort_values("entry_date")
        n = len(wt)
        wins = (wt["net_return"] > 0).sum()
        stops = wt["stopped_out"].sum()
        week_pnl = wt["net_return"].sum() / MAX_POS * CAPITAL
        cum_pnl += week_pnl
        total_trades += n
        total_wins += wins
        total_stops += stops

        tickers = ", ".join(wt["ticker"].tolist())
        if len(tickers) > 44:
            tickers = tickers[:41] + "..."
        wr = wins / n * 100 if n > 0 else 0

        bar = ""
        if week_pnl >= 10000:
            bar = " ██████"
        elif week_pnl >= 5000:
            bar = " ████"
        elif week_pnl >= 0:
            bar = " ██"
        elif week_pnl >= -5000:
            bar = " ▒"
        else:
            bar = " ░░░"

        print(f"  {week.strftime('%Y-%m-%d'):<12} {n:>6} {tickers:<45} {wr:>4.0f}% {stops:>5} {week_pnl:>+10,.0f} {cum_pnl:>+10,.0f}{bar}")

    print(f"  {'-'*82}")
    wr_total = total_wins / total_trades * 100 if total_trades > 0 else 0
    print(f"  {'TOTAL':<12} {total_trades:>6} {'':45} {wr_total:>4.0f}% {total_stops:>5} {cum_pnl:>+10,.0f}")

    print(f"\n  --- TRADE-BY-TRADE DETAIL ---")
    print(f"  {'Entry':<12} {'Ticker':<12} {'Entry':>8} {'Exit':>8} {'P&L':>7} {'INR':>9} {'Status':<8} {'Days':>4}")
    print(f"  {'-'*78}")

    for _, t in trades.sort_values("entry_date").iterrows():
        pnl_inr = t["net_return"] / MAX_POS * CAPITAL
        status = "STOPPED" if t["stopped_out"] else "CLOSED"
        days = (t["exit_date"] - t["entry_date"]).days
        marker = "+" if t["net_return"] > 0 else "*"
        print(f"  {t['entry_date'].strftime('%Y-%m-%d'):<12} {t['ticker']:<12} "
              f"{t['entry_price']:>8.1f} {t['exit_price']:>8.1f} "
              f"{t['net_return']*100:>+6.1f}% {pnl_inr:>+9,.0f} {status:<8} {days:>4} {marker}")

    # Summary stats
    avg_wk = cum_pnl / len(weeks)
    weeks_pos = sum(1 for w in weeks
                    if trades[trades["week_start"] == w]["net_return"].sum() / MAX_POS * CAPITAL > 0)
    weeks_5k = sum(1 for w in weeks
                   if trades[trades["week_start"] == w]["net_return"].sum() / MAX_POS * CAPITAL >= 5000)

    print(f"\n  {'='*50}")
    print(f"  SUMMARY — Last 3 Months")
    print(f"  {'='*50}")
    print(f"  Total P&L:        {cum_pnl:>+12,.0f} INR")
    print(f"  Avg per week:     {avg_wk:>+12,.0f} INR")
    print(f"  Weeks traded:     {len(weeks):>12}")
    print(f"  Weeks profitable: {weeks_pos:>12} ({weeks_pos/len(weeks)*100:.0f}%)")
    print(f"  Weeks >= 5K:      {weeks_5k:>12} ({weeks_5k/len(weeks)*100:.0f}%)")
    print(f"  Total trades:     {total_trades:>12}")
    print(f"  Win rate:         {wr_total:>11.0f}%")
    print(f"  Stop-losses:      {total_stops:>12} ({total_stops/total_trades*100:.0f}%)")
    print(f"  Avg return/trade: {trades['net_return'].mean()*100:>+11.2f}%")


if __name__ == "__main__":
    main()
