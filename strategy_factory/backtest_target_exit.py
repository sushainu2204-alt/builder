#!/usr/bin/env python3
"""
Target Exit Backtest — Hold until +1% profit, then exit.

Rules:
- Capital: 4L, split equally across finalists (max 5)
- Buy at next day's open
- Exit when close >= entry * 1.01 (+1% target)
- Stop-loss at -8% (checked intraday)
- If stock is stuck (hasn't hit target), capital stays locked
- Free capital redeploys on next signal day
- Max hold: 30 trading days (force exit if target not hit)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, NIFTY_500, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel, zscore_normalize
from research_5k_v5 import get_nifty_regime
from research_deep import add_advanced_features
from research_bull_fresh import add_bull_features
from backtest_combined_1y import (
    compute_regime_for_date, get_bear_picks, get_bull_breakout_picks,
    get_bull_accel_picks, get_bull_relstr_picks, get_bull_dip_picks,
    select_finalists,
)

CAPITAL = 400000
MAX_POS = 5
PER_POS = CAPITAL // MAX_POS
TARGET_PCT = 0.01       # +1% target
STOP_PCT = -0.08        # -8% stop
MAX_HOLD_DAYS = 30      # Force exit after 30 trading days
COST = 0.001            # 0.1% round trip


def main():
    print(f"Loading data...")
    stock_data_n500 = download_universe(NIFTY_500, START_DATE, END_DATE)
    stock_data_n200 = {k: v for k, v in stock_data_n500.items() if k in NIFTY_200}
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    nifty_data = index_data

    print(f"Building panels...")
    panel_n500 = build_panel(stock_data_n500, index_data)
    panel_n500 = add_advanced_features(panel_n500, stock_data_n500, index_data)
    panel_n500 = add_bull_features(panel_n500, stock_data_n500, index_data)
    panel_n200 = panel_n500[panel_n500["ticker"].isin(NIFTY_200)].copy()

    all_dates = sorted(panel_n200["date"].unique())
    one_year_ago = pd.Timestamp(datetime.now() - timedelta(days=365))
    test_dates = [d for d in all_dates if d >= one_year_ago]

    print(f"\nSimulating {len(test_dates)} trading days...")
    print(f"Capital: {CAPITAL:,} | Per position: {PER_POS:,} | Target: +{TARGET_PCT*100:.0f}% | Stop: {STOP_PCT*100:.0f}%")

    # Track open positions
    open_positions = {}  # ticker -> {entry_date, entry_price, entry_idx, stop_price, strategy, per_pos}
    closed_trades = []

    for date in test_dates:
        # Check and update open positions
        tickers_to_close = []
        for ticker, pos in open_positions.items():
            if ticker not in stock_data_n500:
                continue
            df = stock_data_n500[ticker]
            if date not in df.index:
                continue

            idx = df.index.get_loc(date)
            close_price = df.iloc[idx]["Close"]
            low_price = df.iloc[idx]["Low"]
            days_held = sum(1 for d in all_dates if pos["entry_date"] < d <= date)

            # Check stop-loss (intraday)
            if low_price <= pos["stop_price"]:
                exit_price = pos["stop_price"]
                ret = (exit_price / pos["entry_price"] - 1) - COST
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": round(exit_price, 1),
                    "return": round(ret * 100, 2),
                    "pnl_inr": round(ret * pos["per_pos"]),
                    "days_held": days_held,
                    "exit_reason": "STOP",
                    "strategy": pos["strategy"],
                    "n_strats": pos["n_strats"],
                })
                tickers_to_close.append(ticker)
                continue

            # Check target hit
            if close_price >= pos["entry_price"] * (1 + TARGET_PCT):
                exit_price = pos["entry_price"] * (1 + TARGET_PCT)  # Exit at target
                ret = TARGET_PCT - COST
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": round(exit_price, 1),
                    "return": round(ret * 100, 2),
                    "pnl_inr": round(ret * pos["per_pos"]),
                    "days_held": days_held,
                    "exit_reason": "TARGET",
                    "strategy": pos["strategy"],
                    "n_strats": pos["n_strats"],
                })
                tickers_to_close.append(ticker)
                continue

            # Check max hold
            if days_held >= MAX_HOLD_DAYS:
                exit_price = close_price
                ret = (exit_price / pos["entry_price"] - 1) - COST
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": round(exit_price, 1),
                    "return": round(ret * 100, 2),
                    "pnl_inr": round(ret * pos["per_pos"]),
                    "days_held": days_held,
                    "exit_reason": f"MAX_HOLD({MAX_HOLD_DAYS}d)",
                    "strategy": pos["strategy"],
                    "n_strats": pos["n_strats"],
                })
                tickers_to_close.append(ticker)

        for t in tickers_to_close:
            del open_positions[t]

        # How many slots free?
        available = MAX_POS - len(open_positions)
        if available <= 0:
            continue

        # Get regime and signals
        regime = compute_regime_for_date(date, nifty_data, panel_n200)
        if regime is None:
            continue

        all_picks = []
        all_picks.extend(get_bear_picks(date, panel_n200, regime))
        all_picks.extend(get_bull_breakout_picks(date, panel_n500, regime))
        all_picks.extend(get_bull_accel_picks(date, panel_n200, regime))
        all_picks.extend(get_bull_relstr_picks(date, panel_n500, regime))
        all_picks.extend(get_bull_dip_picks(date, panel_n200, regime))

        # Remove already held
        all_picks = [p for p in all_picks if p["ticker"] not in open_positions]

        finalists = select_finalists(all_picks, max_pos=available)

        for pick in finalists:
            ticker = pick["ticker"]
            if ticker not in stock_data_n500:
                continue

            df = stock_data_n500[ticker]
            try:
                idx = df.index.get_indexer([date], method="nearest")[0]
            except Exception:
                continue

            entry_idx = idx + 1
            if entry_idx >= len(df):
                continue

            entry_price = df.iloc[entry_idx]["Open"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            entry_date = df.index[entry_idx]
            stop_price = entry_price * (1 + STOP_PCT)

            open_positions[ticker] = {
                "entry_date": entry_date,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "strategy": ", ".join(pick["strategies"]),
                "n_strats": pick["n_strats"],
                "per_pos": PER_POS,
            }

    # Force close any remaining open positions at last available price
    for ticker, pos in open_positions.items():
        if ticker in stock_data_n500:
            df = stock_data_n500[ticker]
            exit_price = df["Close"].iloc[-1]
            days_held = sum(1 for d in all_dates if pos["entry_date"] < d <= all_dates[-1])
            ret = (exit_price / pos["entry_price"] - 1) - COST
            closed_trades.append({
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": df.index[-1],
                "entry_price": pos["entry_price"],
                "exit_price": round(exit_price, 1),
                "return": round(ret * 100, 2),
                "pnl_inr": round(ret * pos["per_pos"]) if not np.isnan(ret) else 0,
                "days_held": days_held,
                "exit_reason": "STILL_OPEN",
                "strategy": pos["strategy"],
                "n_strats": pos["n_strats"],
            })

    if not closed_trades:
        print("No trades.")
        return

    tdf = pd.DataFrame(closed_trades).sort_values("entry_date")
    tdf["week"] = tdf["entry_date"].dt.to_period("W").apply(lambda r: r.start_time)

    # ============================================================
    # REPORT
    # ============================================================
    print(f"\n{'#'*110}")
    print(f"  TARGET EXIT BACKTEST — Hold until +1%, stop at -8%")
    print(f"  Capital: {CAPITAL:,} | Per pos: {PER_POS:,} | Max hold: {MAX_HOLD_DAYS}d")
    print(f"{'#'*110}")

    # Exit reason breakdown
    target_trades = tdf[tdf["exit_reason"] == "TARGET"]
    stop_trades = tdf[tdf["exit_reason"] == "STOP"]
    maxhold_trades = tdf[tdf["exit_reason"].str.startswith("MAX_HOLD")]
    still_open = tdf[tdf["exit_reason"] == "STILL_OPEN"]

    print(f"\n  EXIT REASON BREAKDOWN:")
    print(f"  TARGET (+1% hit):   {len(target_trades):>4} trades ({len(target_trades)/len(tdf)*100:.0f}%)  Avg days: {target_trades['days_held'].mean():.1f}" if len(target_trades) > 0 else "  TARGET: 0")
    print(f"  STOP (-8% hit):     {len(stop_trades):>4} trades ({len(stop_trades)/len(tdf)*100:.0f}%)  Avg days: {stop_trades['days_held'].mean():.1f}" if len(stop_trades) > 0 else "  STOP: 0")
    print(f"  MAX HOLD (30d):     {len(maxhold_trades):>4} trades ({len(maxhold_trades)/len(tdf)*100:.0f}%)  Avg ret: {maxhold_trades['return'].mean():+.1f}%" if len(maxhold_trades) > 0 else "  MAX HOLD: 0")
    print(f"  STILL OPEN:         {len(still_open):>4}" if len(still_open) > 0 else "")

    # Avg days to hit target
    if len(target_trades) > 0:
        print(f"\n  DAYS TO HIT +1% TARGET:")
        for d in [1, 2, 3, 5, 10]:
            n = (target_trades["days_held"] <= d).sum()
            print(f"    Within {d:>2} day(s): {n:>4} ({n/len(target_trades)*100:.0f}%)")

    # Weekly report
    weeks = sorted(tdf["week"].unique())
    cum_pnl = 0
    w_pos, w_5k = 0, 0

    print(f"\n  {'Week':<12} {'New':>3} {'Exit':>4} {'Tgt':>3} {'Stp':>3} {'Avg D1':>7} {'Week PnL':>10} {'Cum':>10}")
    print(f"  {'-'*65}")

    for week in weeks:
        # Trades entered this week
        entered = tdf[(tdf["entry_date"] >= week) & (tdf["entry_date"] < week + timedelta(days=7))]
        # Trades exited this week
        exited = tdf[(tdf["exit_date"] >= week) & (tdf["exit_date"] < week + timedelta(days=7))]

        if len(exited) == 0 and len(entered) == 0:
            continue

        targets_hit = (exited["exit_reason"] == "TARGET").sum()
        stops_hit = (exited["exit_reason"] == "STOP").sum()
        week_pnl = exited["pnl_inr"].sum()
        cum_pnl += week_pnl

        # Day 1 return of entered trades
        d1_avg = ""
        if len(entered) > 0:
            d1_rets = []
            for _, t in entered.iterrows():
                ticker = t["ticker"]
                if ticker in stock_data_n500:
                    df = stock_data_n500[ticker]
                    if t["entry_date"] in df.index:
                        idx = df.index.get_loc(t["entry_date"])
                        d1_close = df.iloc[idx]["Close"]
                        d1_ret = (d1_close / t["entry_price"] - 1) * 100
                        d1_rets.append(d1_ret)
            if d1_rets:
                d1_avg = f"{np.mean(d1_rets):+.1f}%"

        if week_pnl > 0: w_pos += 1
        if week_pnl >= 5000: w_5k += 1

        bar = ""
        if week_pnl >= 5000: bar = " ████"
        elif week_pnl >= 0: bar = " ██"
        elif week_pnl >= -5000: bar = " ░"
        else: bar = " ░░░"

        print(f"  {week.strftime('%Y-%m-%d'):<12} {len(entered):>3} {len(exited):>4} {targets_hit:>3} {stops_hit:>3} {d1_avg:>7} {week_pnl:>+10,.0f} {cum_pnl:>+10,.0f}{bar}")

    n_weeks = len([w for w in weeks if tdf[(tdf["exit_date"] >= w) & (tdf["exit_date"] < w + timedelta(days=7))]["pnl_inr"].sum() != 0])

    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Total P&L:         {cum_pnl:>+12,.0f} INR")
    print(f"  Total trades:      {len(tdf):>12}")
    print(f"  Target hit rate:   {len(target_trades):>12} ({len(target_trades)/len(tdf)*100:.0f}%)")
    print(f"  Stop hit rate:     {len(stop_trades):>12} ({len(stop_trades)/len(tdf)*100:.0f}%)")
    print(f"  Avg days to target:{target_trades['days_held'].mean():>11.1f}d" if len(target_trades) > 0 else "")
    print(f"  Avg P&L per trade: {tdf['pnl_inr'].mean():>+11,.0f} INR")

    # The key question: is target exit better than fixed hold?
    target_pnl = target_trades["pnl_inr"].sum() if len(target_trades) > 0 else 0
    stop_pnl = stop_trades["pnl_inr"].sum() if len(stop_trades) > 0 else 0
    maxhold_pnl = maxhold_trades["pnl_inr"].sum() if len(maxhold_trades) > 0 else 0

    print(f"\n  P&L BREAKDOWN:")
    print(f"  From targets:      {target_pnl:>+12,.0f} ({len(target_trades)} trades × ~{PER_POS*TARGET_PCT:,.0f} each)")
    print(f"  From stops:        {stop_pnl:>+12,.0f} ({len(stop_trades)} trades × ~{PER_POS*abs(STOP_PCT):,.0f} each)")
    print(f"  From max hold:     {maxhold_pnl:>+12,.0f} ({len(maxhold_trades)} trades, mixed)")

    # Risk/reward
    if len(target_trades) > 0 and len(stop_trades) > 0:
        rr = (len(target_trades) * PER_POS * TARGET_PCT) / (len(stop_trades) * PER_POS * abs(STOP_PCT))
        print(f"\n  Risk/reward: {len(target_trades)} wins × {PER_POS*TARGET_PCT:,.0f} vs {len(stop_trades)} losses × {PER_POS*abs(STOP_PCT):,.0f}")
        print(f"  Need {abs(STOP_PCT)/TARGET_PCT:.0f} targets per stop to break even. Actual: {len(target_trades)/max(len(stop_trades),1):.1f}")


if __name__ == "__main__":
    main()
