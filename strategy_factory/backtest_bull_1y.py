#!/usr/bin/env python3
"""
Week-by-week backtest for Breakout + Accumulation strategy.
Nifty 500, bull regime only, hold 15d, 5 positions.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_500, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel
from backtest.metrics import compute_all_metrics
from research_5k_v5 import get_nifty_regime, run_regime_filtered_backtest
from research_deep import add_advanced_features
from research_bull_fresh import add_bull_features

CAPITAL = 600000
MAX_POS = 5
HOLD = 15

def main():
    stock_data = download_universe(NIFTY_500, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)
    nifty_regime = get_nifty_regime(index_data)
    panel = add_advanced_features(panel, stock_data, index_data)
    panel = add_bull_features(panel, stock_data, index_data)

    # Filter: accumulation > 0.5
    panel_filtered = panel[panel["accumulation"] > 0.5].copy() if "accumulation" in panel.columns else panel.copy()

    # Full period backtest
    result = run_regime_filtered_backtest(
        stock_data, panel_filtered, nifty_regime,
        signal_col="at_20d_high", condition="highest",
        hold_days=HOLD, max_positions=MAX_POS, num_buckets=10,
        allowed_regime=1, block_reentry_days=HOLD,
    )

    if result is None or result["trades"].empty:
        print("No trades.")
        return

    trades = result["trades"]
    trades["week_start"] = trades["entry_date"].dt.to_period("W").apply(lambda r: r.start_time)
    trades["pnl_inr"] = trades["net_return"] / MAX_POS * CAPITAL

    m = result["metrics"]

    print(f"\n{'#'*100}")
    print(f"  BREAKOUT + ACCUMULATION — Full Period Backtest")
    print(f"  Universe: Nifty 500 | Bull regime only | Hold: {HOLD}d | Positions: {MAX_POS}")
    print(f"  Capital: {CAPITAL:,} INR")
    print(f"{'#'*100}")

    print(f"\n  Full Period: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1f}% WR={m['win_rate']:.1f}% "
          f"PF={m['profit_factor']:.2f} MDD={m['max_drawdown']:.1f}%")
    print(f"  Trades: {m['trades']} ({m['trades']/5:.0f}/year)")

    # Week by week
    weeks = sorted(trades["week_start"].unique())
    cum_pnl = 0
    total_wins = 0
    total_stops = 0
    weeks_pos = 0
    weeks_5k = 0

    print(f"\n  {'Week':<12} {'Trd':>3} {'Stocks':<50} {'WR':>5} {'Stp':>3} {'Week P&L':>10} {'Cum P&L':>10}")
    print(f"  {'-'*100}")

    for week in weeks:
        wt = trades[trades["week_start"] == week].sort_values("entry_date")
        n = len(wt)
        wins = (wt["net_return"] > 0).sum()
        stops = wt["stopped_out"].sum()
        week_pnl = wt["pnl_inr"].sum()
        cum_pnl += week_pnl
        total_wins += wins
        total_stops += stops

        tickers = ", ".join(wt["ticker"].tolist())
        if len(tickers) > 49:
            tickers = tickers[:46] + "..."
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

        print(f"  {week.strftime('%Y-%m-%d'):<12} {n:>3} {tickers:<50} {wr:>4.0f}% {stops:>3} {week_pnl:>+10,.0f} {cum_pnl:>+10,.0f}{bar}")

    n_weeks = len(weeks)
    total_trades = len(trades)
    wr_total = total_wins / total_trades * 100

    print(f"  {'-'*100}")

    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Total P&L:         {cum_pnl:>+12,.0f} INR")
    print(f"  Avg per week:      {cum_pnl/n_weeks:>+12,.0f} INR (across {n_weeks} traded weeks)")
    print(f"  Weeks traded:      {n_weeks:>12}")
    print(f"  Weeks profitable:  {weeks_pos:>12} ({weeks_pos/n_weeks*100:.0f}%)")
    print(f"  Weeks >= 5K:       {weeks_5k:>12} ({weeks_5k/n_weeks*100:.0f}%)")
    print(f"  Total trades:      {total_trades:>12}")
    print(f"  Win rate:          {wr_total:>11.0f}%")
    print(f"  Stop-losses:       {total_stops:>12} ({total_stops/total_trades*100:.0f}%)")
    print(f"  Avg return/trade:  {trades['net_return'].mean()*100:>+11.2f}%")

    # Monthly
    trades_c = trades.copy()
    trades_c["month"] = trades_c["entry_date"].dt.to_period("M")
    print(f"\n  MONTHLY:")
    print(f"  {'Month':<10} {'Trd':>4} {'WR':>5} {'Stops':>5} {'P&L':>10} {'Avg/trade':>10}")
    mp, mt = 0, 0
    for month in sorted(trades_c["month"].unique()):
        mtrades = trades_c[trades_c["month"] == month]
        pnl = mtrades["pnl_inr"].sum()
        wr = (mtrades["net_return"] > 0).sum() / len(mtrades) * 100
        stops = mtrades["stopped_out"].sum()
        avg_ret = mtrades["net_return"].mean() * 100
        marker = " +" if pnl >= 10000 else (" *" if pnl < 0 else "")
        print(f"  {str(month):<10} {len(mtrades):>4} {wr:>4.0f}% {stops:>5} {pnl:>+10,.0f} {avg_ret:>+9.2f}%{marker}")
        mt += 1
        if pnl > 0: mp += 1
    print(f"  Profitable months: {mp}/{mt} ({mp/mt*100:.0f}%)")

    # Top trades
    trades_sorted = trades.sort_values("net_return", ascending=False)
    print(f"\n  TOP 10 WINNERS:")
    for _, t in trades_sorted.head(10).iterrows():
        print(f"    {t['entry_date'].strftime('%Y-%m-%d')} {t['ticker']:<12} {t['net_return']*100:>+6.1f}% = {t['pnl_inr']:>+8,.0f} INR")

    print(f"\n  WORST 10 LOSERS:")
    for _, t in trades_sorted.tail(10).iterrows():
        stopped = " (STOPPED)" if t["stopped_out"] else ""
        print(f"    {t['entry_date'].strftime('%Y-%m-%d')} {t['ticker']:<12} {t['net_return']*100:>+6.1f}% = {t['pnl_inr']:>+8,.0f} INR{stopped}")


if __name__ == "__main__":
    main()
