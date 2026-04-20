#!/usr/bin/env python3
"""
Combined Strategy Backtest — 1 Year Weekly Report

Simulates running ALL strategies daily for the past year.
For each day, picks finalists using the same logic as run_signals.py:
  - Multi-strategy picks ranked first (HIGH conviction)
  - Then single-strategy CLEAN picks (MEDIUM)
  - Max 5 positions
  - Buy next day open, check day-1 return, hold for strategy's hold period
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
from backtest.metrics import compute_all_metrics
from research_5k_v5 import get_nifty_regime
from research_deep import add_advanced_features
from research_bull_fresh import add_bull_features

CAPITAL = 600000
MAX_POS = 5
PER_POS = CAPITAL // MAX_POS


def compute_regime_for_date(date, nifty_data, panel_n200):
    """Compute regime as of a specific date."""
    nifty_close = nifty_data["Close"]
    nifty_sma = nifty_close.rolling(20).mean()

    if date not in nifty_close.index:
        # Find nearest prior date
        prior = nifty_close.index[nifty_close.index <= date]
        if len(prior) == 0:
            return None
        date = prior[-1]

    nc = nifty_close.loc[date]
    ns = nifty_sma.loc[date]
    if pd.isna(ns):
        return None

    # Breadth from panel
    day_panel = panel_n200[panel_n200["date"] == date]
    if "mkt_breadth" in day_panel.columns and len(day_panel) > 0:
        breadth = day_panel["mkt_breadth"].iloc[0] * 100 if pd.notna(day_panel["mkt_breadth"].iloc[0]) else 50
    else:
        breadth = 50

    return {
        "nifty_close": nc, "nifty_sma": ns,
        "is_bear": nc < ns,
        "breadth": breadth,
    }


def get_bear_picks(date, panel_n200, regime):
    """Bear dip: RSI3 bottom decile, breadth<40%."""
    if not (regime["is_bear"] and regime["breadth"] < 40):
        return []

    day = panel_n200[panel_n200["date"] == date].dropna(subset=["rsi_3"])
    if len(day) < 20:
        return []

    try:
        day = day.copy()
        day["_q"] = pd.qcut(day["rsi_3"], 10, labels=False, duplicates="drop")
        picks = day[day["_q"] == 0].sort_values("rsi_3").head(5)
    except ValueError:
        return []

    return [{"ticker": r["ticker"], "strategy": "Bear Dip", "hold": 5} for _, r in picks.iterrows()]


def get_bull_breakout_picks(date, panel_n500, regime):
    """Bull breakout + accumulation: top decile at_20d_high with accum>0.5."""
    if regime["is_bear"]:
        return []

    day = panel_n500[(panel_n500["date"] == date)].copy()
    if "accumulation" not in day.columns or "at_20d_high" not in day.columns:
        return []

    day = day[(day["accumulation"] > 0.5)].dropna(subset=["at_20d_high"])
    if len(day) < 5:
        return []

    # Sort by at_20d_high (1 first) then accumulation
    day["_score"] = day["at_20d_high"] + day["accumulation"] * 0.1
    picks = day.sort_values("_score", ascending=False).head(5)

    return [{"ticker": r["ticker"], "strategy": "Breakout+Accum", "hold": 15} for _, r in picks.iterrows()]


def get_bull_accel_picks(date, panel_n200, regime):
    """Bull acceleration + breadth>60%."""
    if regime["is_bear"] or regime["breadth"] <= 60:
        return []

    if "price_accel" not in panel_n200.columns:
        return []

    day = panel_n200[panel_n200["date"] == date].dropna(subset=["price_accel"])
    if len(day) < 20:
        return []

    try:
        day = day.copy()
        day["_q"] = pd.qcut(day["price_accel"], 10, labels=False, duplicates="drop")
        picks = day[day["_q"] == day["_q"].max()].sort_values("price_accel", ascending=False).head(3)
    except ValueError:
        return []

    return [{"ticker": r["ticker"], "strategy": "Acceleration", "hold": 15} for _, r in picks.iterrows()]


def get_bull_relstr_picks(date, panel_n500, regime):
    """Bull relative strength + volume breakout."""
    if regime["is_bear"]:
        return []

    if "rel_str_z" not in panel_n500.columns or "vol_breakout" not in panel_n500.columns:
        return []

    day = panel_n500[panel_n500["date"] == date].copy()
    day = day[day["vol_breakout"] > 0].dropna(subset=["rel_str_z"])
    if len(day) < 20:
        return []

    day["_score"] = day["rel_str_z"] + day["vol_breakout"] * 0.3
    picks = day.sort_values("_score", ascending=False).head(3)

    return [{"ticker": r["ticker"], "strategy": "RelStr+VolBreak", "hold": 15} for _, r in picks.iterrows()]


def get_bull_dip_picks(date, panel_n200, regime):
    """Bull dip buy + breadth>60%."""
    if regime["is_bear"] or regime["breadth"] <= 60:
        return []

    day = panel_n200[panel_n200["date"] == date].dropna(subset=["return_5d"])
    if len(day) < 20:
        return []

    try:
        day = day.copy()
        day["_q"] = pd.qcut(day["return_5d"], 10, labels=False, duplicates="drop")
        picks = day[day["_q"] == 0].sort_values("return_5d").head(3)
    except ValueError:
        return []

    return [{"ticker": r["ticker"], "strategy": "Bull Dip", "hold": 10} for _, r in picks.iterrows()]


def select_finalists(all_picks, max_pos=5):
    """Rank by multi-strategy agreement, then singles. Return top max_pos."""
    if not all_picks:
        return []

    # Count strategies per ticker
    ticker_strats = defaultdict(list)
    ticker_hold = {}
    for p in all_picks:
        ticker_strats[p["ticker"]].append(p["strategy"])
        ticker_hold[p["ticker"]] = p["hold"]

    # Multi first, then single
    multi = [(t, len(s), s, ticker_hold[t]) for t, s in ticker_strats.items() if len(s) >= 2]
    single = [(t, len(s), s, ticker_hold[t]) for t, s in ticker_strats.items() if len(s) == 1]

    multi.sort(key=lambda x: -x[1])
    ranked = multi + single

    return [{"ticker": t, "n_strats": n, "strategies": strats, "hold": hold}
            for t, n, strats, hold in ranked[:max_pos]]


def main():
    print(f"Loading data...")
    stock_data_n500 = download_universe(NIFTY_500, START_DATE, END_DATE)
    stock_data_n200 = {k: v for k, v in stock_data_n500.items() if k in NIFTY_200}
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    nifty_data = index_data

    print(f"Building panels...")
    panel_n500 = build_panel(stock_data_n500, index_data)
    panel_n200 = panel_n500[panel_n500["ticker"].isin(NIFTY_200)].copy()

    panel_n500 = add_advanced_features(panel_n500, stock_data_n500, index_data)
    panel_n500 = add_bull_features(panel_n500, stock_data_n500, index_data)
    panel_n200 = panel_n500[panel_n500["ticker"].isin(NIFTY_200)].copy()

    # Get all trading dates in the last year
    all_dates = sorted(panel_n200["date"].unique())
    one_year_ago = pd.Timestamp(datetime.now() - timedelta(days=365))
    test_dates = [d for d in all_dates if d >= one_year_ago and d <= all_dates[-3]]  # -3 to have room for forward returns

    print(f"\nSimulating {len(test_dates)} trading days...")

    trades = []
    open_positions = {}  # ticker -> exit_date

    for date in test_dates:
        # Clean expired positions
        open_positions = {t: info for t, info in open_positions.items() if info["exit_date"] > date}

        available = MAX_POS - len(open_positions)
        if available <= 0:
            continue

        regime = compute_regime_for_date(date, nifty_data, panel_n200)
        if regime is None:
            continue

        # Get picks from all strategies
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
            hold = pick["hold"]
            if ticker not in stock_data_n500:
                continue

            df = stock_data_n500[ticker]
            try:
                idx = df.index.get_indexer([date], method="nearest")[0]
            except Exception:
                continue

            entry_idx = idx + 1
            if entry_idx >= len(df) - 1:
                continue

            entry_price = df.iloc[entry_idx]["Open"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Day 1 close
            day1_close = df.iloc[entry_idx]["Close"]
            day1_ret = (day1_close / entry_price - 1) * 100

            # Full hold return
            exit_idx = min(entry_idx + hold, len(df) - 1)
            exit_price = entry_price
            stopped = False
            stop_price = entry_price * 0.92

            for di in range(entry_idx, exit_idx + 1):
                if df.iloc[di]["Low"] <= stop_price:
                    exit_price = stop_price
                    stopped = True
                    break
                if di == exit_idx:
                    exit_price = df.iloc[di]["Close"]

            full_ret = (exit_price / entry_price - 1) * 100
            net_ret = full_ret - 0.1  # cost

            entry_date = df.index[entry_idx]
            exit_date = df.index[exit_idx]

            trades.append({
                "signal_date": date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "ticker": ticker,
                "n_strats": pick["n_strats"],
                "strategies": ", ".join(pick["strategies"]),
                "hold": hold,
                "entry_price": round(entry_price, 1),
                "day1_close": round(day1_close, 1),
                "day1_ret": round(day1_ret, 2),
                "exit_price": round(exit_price, 1),
                "full_ret": round(net_ret, 2),
                "stopped": stopped,
                "conv": "HIGH" if pick["n_strats"] >= 2 else "MEDIUM",
            })

            open_positions[ticker] = {"exit_date": exit_date}

    if not trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(trades)
    tdf["week"] = tdf["entry_date"].dt.to_period("W").apply(lambda r: r.start_time)
    tdf["pnl_inr"] = tdf["full_ret"] / 100 * PER_POS

    print(f"\n{'#'*105}")
    print(f"  COMBINED STRATEGY BACKTEST — Last 1 Year")
    print(f"  All 5 strategies | Finalists logic | Capital: {CAPITAL:,}")
    print(f"{'#'*105}")

    weeks = sorted(tdf["week"].unique())
    cum_pnl = 0
    w_pos, w_5k = 0, 0

    print(f"\n  {'Week':<12} {'Trd':>3} {'Stocks':<35} {'Conv':>5} {'D1 Avg':>7} {'D1>1%':>5} {'Full':>7} {'PnL INR':>9} {'Cum':>10}")
    print(f"  {'-'*100}")

    for week in weeks:
        wt = tdf[tdf["week"] == week]
        n = len(wt)
        tickers = ", ".join(wt["ticker"].tolist())
        if len(tickers) > 34: tickers = tickers[:31] + "..."

        high_n = (wt["conv"] == "HIGH").sum()
        conv_str = f"{high_n}H" if high_n > 0 else "M"

        d1_avg = wt["day1_ret"].mean()
        d1_above_1 = (wt["day1_ret"] >= 1).sum()
        full_avg = wt["full_ret"].mean()
        week_pnl = wt["pnl_inr"].sum()
        cum_pnl += week_pnl

        if week_pnl > 0: w_pos += 1
        if week_pnl >= 5000: w_5k += 1

        bar = ""
        if week_pnl >= 20000: bar = " ██████"
        elif week_pnl >= 10000: bar = " ████"
        elif week_pnl >= 5000: bar = " ███"
        elif week_pnl >= 0: bar = " █"
        elif week_pnl >= -10000: bar = " ░"
        else: bar = " ░░░"

        print(f"  {week.strftime('%Y-%m-%d'):<12} {n:>3} {tickers:<35} {conv_str:>5} {d1_avg:>+6.1f}% {d1_above_1:>3}/{n} {full_avg:>+6.1f}% {week_pnl:>+9,.0f} {cum_pnl:>+10,.0f}{bar}")

    n_weeks = len(weeks)
    total_trades = len(tdf)

    print(f"  {'-'*100}")

    # Day-1 return analysis
    d1_above_1_total = (tdf["day1_ret"] >= 1).sum()
    d1_above_0 = (tdf["day1_ret"] > 0).sum()

    # HIGH vs MEDIUM conviction
    high_trades = tdf[tdf["conv"] == "HIGH"]
    med_trades = tdf[tdf["conv"] == "MEDIUM"]

    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Total P&L:         {cum_pnl:>+12,.0f} INR")
    print(f"  Avg per week:      {cum_pnl/n_weeks:>+12,.0f} INR")
    print(f"  Weeks traded:      {n_weeks:>12}")
    print(f"  Weeks profitable:  {w_pos:>12} ({w_pos/n_weeks*100:.0f}%)")
    print(f"  Weeks >= 5K:       {w_5k:>12} ({w_5k/n_weeks*100:.0f}%)")
    print(f"  Total trades:      {total_trades:>12}")
    print(f"  Win rate (full):   {(tdf['full_ret']>0).sum():>12} ({(tdf['full_ret']>0).sum()/total_trades*100:.0f}%)")
    print(f"  Stop-losses:       {tdf['stopped'].sum():>12} ({tdf['stopped'].sum()/total_trades*100:.0f}%)")

    print(f"\n  DAY-1 RETURN ANALYSIS:")
    print(f"  Day-1 avg return:  {tdf['day1_ret'].mean():>+11.2f}%")
    print(f"  Day-1 > 0%:        {d1_above_0:>12} ({d1_above_0/total_trades*100:.0f}%)")
    print(f"  Day-1 >= 1%:       {d1_above_1_total:>12} ({d1_above_1_total/total_trades*100:.0f}%)")
    print(f"  Day-1 >= 2%:       {(tdf['day1_ret']>=2).sum():>12} ({(tdf['day1_ret']>=2).sum()/total_trades*100:.0f}%)")

    print(f"\n  BY CONVICTION:")
    if len(high_trades) > 0:
        print(f"  HIGH ({len(high_trades)} trades):  D1 avg={high_trades['day1_ret'].mean():+.2f}%  Full avg={high_trades['full_ret'].mean():+.2f}%  "
              f"D1>1%={((high_trades['day1_ret']>=1).sum()/len(high_trades)*100):.0f}%  WR={((high_trades['full_ret']>0).sum()/len(high_trades)*100):.0f}%")
    if len(med_trades) > 0:
        print(f"  MEDIUM ({len(med_trades)} trades): D1 avg={med_trades['day1_ret'].mean():+.2f}%  Full avg={med_trades['full_ret'].mean():+.2f}%  "
              f"D1>1%={((med_trades['day1_ret']>=1).sum()/len(med_trades)*100):.0f}%  WR={((med_trades['full_ret']>0).sum()/len(med_trades)*100):.0f}%")


if __name__ == "__main__":
    main()
