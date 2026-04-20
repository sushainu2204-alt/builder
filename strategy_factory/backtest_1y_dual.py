#!/usr/bin/env python3
"""
Week-by-week backtest for the last 1 year.
Dual Strategy:
  Bull (Nifty > SMA, breadth > 60%): Buy 5d dip bottom decile, hold 10d, 3 pos
  Bear (Nifty < SMA, breadth < 40%): RSI3 oversold bottom decile, hold 5d, 5 pos
"""
import os, sys, warnings
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel
from research_5k_v5 import get_nifty_regime, run_regime_filtered_backtest
from research_deep import add_advanced_features

CAPITAL = 600000

def main():
    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)
    nifty_regime = get_nifty_regime(index_data)
    panel = add_advanced_features(panel, stock_data, index_data)

    # Bull strategy: 5d dip + breadth > 60%, hold 10d, 3 pos
    bull_panel = panel[panel["mkt_breadth"] > 0.6].copy() if "mkt_breadth" in panel.columns else panel.copy()
    bull_result = run_regime_filtered_backtest(
        stock_data, bull_panel, nifty_regime,
        signal_col="return_5d", condition="lowest",
        hold_days=10, max_positions=3, num_buckets=10,
        allowed_regime=1, block_reentry_days=10,
        start_date="2025-04-10",
    )

    # Bear strategy: RSI3 + breadth < 40%, hold 5d, 5 pos
    bear_panel = panel[panel["mkt_breadth"] < 0.4].copy() if "mkt_breadth" in panel.columns else panel.copy()
    bear_result = run_regime_filtered_backtest(
        stock_data, bear_panel, nifty_regime,
        signal_col="rsi_3", condition="lowest",
        hold_days=5, max_positions=5, num_buckets=10,
        allowed_regime=0, block_reentry_days=5,
        start_date="2025-04-10",
    )

    # Combine
    all_trades = []
    if bull_result and not bull_result["trades"].empty:
        bt = bull_result["trades"].copy()
        bt["strategy"] = "BULL"
        bt["max_pos"] = 3
        all_trades.append(bt)
    if bear_result and not bear_result["trades"].empty:
        brt = bear_result["trades"].copy()
        brt["strategy"] = "BEAR"
        brt["max_pos"] = 5
        all_trades.append(brt)

    if not all_trades:
        print("No trades in last year.")
        return

    trades = pd.concat(all_trades).sort_values("entry_date").reset_index(drop=True)
    trades["week_start"] = trades["entry_date"].dt.to_period("W").apply(lambda r: r.start_time)
    trades["pnl_inr"] = trades.apply(lambda r: r["net_return"] / r["max_pos"] * CAPITAL, axis=1)

    print(f"\n{'#'*100}")
    print(f"  DUAL STRATEGY — WEEKLY BACKTEST (Last 1 Year)")
    print(f"  Bull: 5d dip + breadth>60%, hold 10d, 3 pos | Bear: RSI3 + breadth<40%, hold 5d, 5 pos")
    print(f"  Capital: {CAPITAL:,} INR")
    print(f"{'#'*100}")

    weeks = sorted(trades["week_start"].unique())
    cum_pnl = 0
    total_trades = 0
    total_wins = 0
    total_stops = 0
    weeks_pos = 0
    weeks_5k = 0

    print(f"\n  {'Week':<12} {'Mode':<5} {'Trd':>3} {'Stocks':<40} {'WR':>5} {'Stp':>3} {'Week P&L':>10} {'Cum P&L':>10}")
    print(f"  {'-'*95}")

    for week in weeks:
        wt = trades[trades["week_start"] == week].sort_values("entry_date")
        n = len(wt)
        wins = (wt["net_return"] > 0).sum()
        stops = wt["stopped_out"].sum()
        week_pnl = wt["pnl_inr"].sum()
        cum_pnl += week_pnl
        total_trades += n
        total_wins += wins
        total_stops += stops

        tickers = ", ".join(wt["ticker"].tolist())
        if len(tickers) > 39:
            tickers = tickers[:36] + "..."

        mode = wt["strategy"].iloc[0] if len(wt["strategy"].unique()) == 1 else "BOTH"
        wr = wins / n * 100 if n > 0 else 0

        if week_pnl >= 5000: weeks_5k += 1
        if week_pnl > 0: weeks_pos += 1

        bar = ""
        if week_pnl >= 20000: bar = " ██████████"
        elif week_pnl >= 10000: bar = " ██████"
        elif week_pnl >= 5000: bar = " ████"
        elif week_pnl >= 0: bar = " ██"
        elif week_pnl >= -10000: bar = " ░"
        elif week_pnl >= -20000: bar = " ░░░"
        else: bar = " ░░░░░░"

        print(f"  {week.strftime('%Y-%m-%d'):<12} {mode:<5} {n:>3} {tickers:<40} {wr:>4.0f}% {stops:>3} {week_pnl:>+10,.0f} {cum_pnl:>+10,.0f}{bar}")

    print(f"  {'-'*95}")

    wr_total = total_wins / total_trades * 100 if total_trades > 0 else 0
    n_weeks = len(weeks)

    print(f"\n  {'='*60}")
    print(f"  SUMMARY — Last 1 Year")
    print(f"  {'='*60}")
    print(f"  Total P&L:         {cum_pnl:>+12,.0f} INR")
    print(f"  Avg per week:      {cum_pnl/n_weeks:>+12,.0f} INR (across {n_weeks} traded weeks)")
    print(f"  Weeks traded:      {n_weeks:>12} out of ~52")
    print(f"  Weeks profitable:  {weeks_pos:>12} ({weeks_pos/n_weeks*100:.0f}%)")
    print(f"  Weeks >= 5K:       {weeks_5k:>12} ({weeks_5k/n_weeks*100:.0f}%)")
    print(f"  Total trades:      {total_trades:>12}")
    print(f"  Win rate:          {wr_total:>11.0f}%")
    print(f"  Stop-losses:       {total_stops:>12} ({total_stops/total_trades*100:.0f}%)")
    print(f"  Avg return/trade:  {trades['net_return'].mean()*100:>+11.2f}%")

    # Bull vs Bear breakdown
    bull_trades = trades[trades["strategy"] == "BULL"]
    bear_trades = trades[trades["strategy"] == "BEAR"]
    print(f"\n  BULL trades: {len(bull_trades)} | WR: {(bull_trades['net_return']>0).sum()/len(bull_trades)*100:.0f}% | Avg: {bull_trades['net_return'].mean()*100:+.2f}%" if len(bull_trades) > 0 else "  BULL trades: 0")
    print(f"  BEAR trades: {len(bear_trades)} | WR: {(bear_trades['net_return']>0).sum()/len(bear_trades)*100:.0f}% | Avg: {bear_trades['net_return'].mean()*100:+.2f}%" if len(bear_trades) > 0 else "  BEAR trades: 0")

    # Monthly summary
    trades["month"] = trades["entry_date"].dt.to_period("M")
    print(f"\n  MONTHLY SUMMARY:")
    print(f"  {'Month':<10} {'Trd':>4} {'B/R':>7} {'WR':>5} {'P&L':>10} {'Avg/wk':>9}")
    for month in sorted(trades["month"].unique()):
        mt = trades[trades["month"] == month]
        pnl = mt["pnl_inr"].sum()
        nb = (mt["strategy"] == "BULL").sum()
        nr = (mt["strategy"] == "BEAR").sum()
        wr = (mt["net_return"] > 0).sum() / len(mt) * 100
        wks_in_month = mt["week_start"].nunique()
        avg_wk = pnl / wks_in_month if wks_in_month > 0 else 0
        marker = " +" if pnl >= 20000 else (" *" if pnl < 0 else "")
        print(f"  {str(month):<10} {len(mt):>4} {nb:>3}/{nr:<3} {wr:>4.0f}% {pnl:>+10,.0f} {avg_wk:>+9,.0f}{marker}")


if __name__ == "__main__":
    main()
