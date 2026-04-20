#!/usr/bin/env python3
"""
Live Signal Generator + Recent Backtest Validator.

Usage:
    python live_signal.py                  # Today's signals + last week backtest
    python live_signal.py --date 2026-03-10  # Signals as of specific date
"""
import os
import sys
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, NIFTY50_TICKER, CACHE_DIR
from signals.features import compute_all_features
from signals.scanner import download_stock_data, zscore_normalize


def get_fresh_data(tickers, days_back=400):
    """Download fresh data (bypass cache for recency)."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    data = {}
    failed = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] downloading...")
        yf_ticker = ticker if ticker.startswith("^") else f"{ticker}.NS"
        try:
            df = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 60:
                failed.append(ticker)
                continue
            data[ticker] = df
        except Exception:
            failed.append(ticker)

    print(f"  Downloaded: {len(data)} | Failed: {len(failed)}")
    return data


def compute_signals(stock_data, index_data):
    """Compute return_60d z-score for all stocks on all dates."""
    index_close = index_data["Close"] if index_data is not None else None
    all_signals = []

    for ticker, df in stock_data.items():
        try:
            close = df["Close"]
            # 60-day return
            ret_60d = close.pct_change(60)
            # Z-score normalize
            ret_60d_z = zscore_normalize(ret_60d, lookback=252)

            # RSI(3) for secondary filter
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/3, min_periods=3).mean()
            avg_loss = loss.ewm(alpha=1/3, min_periods=3).mean()
            rs = avg_gain / avg_loss
            rsi_3 = 100 - (100 / (1 + rs))

            for date in df.index:
                if pd.notna(ret_60d_z.get(date)):
                    all_signals.append({
                        "date": date,
                        "ticker": ticker,
                        "close": close.get(date),
                        "return_60d_z": ret_60d_z.get(date),
                        "return_60d_raw": ret_60d.get(date),
                        "rsi_3": rsi_3.get(date),
                    })
        except Exception:
            continue

    return pd.DataFrame(all_signals)


def get_top_momentum_stocks(signals_df, as_of_date, n=5, num_buckets=10):
    """Get the top momentum stocks as of a specific date."""
    day = signals_df[signals_df["date"] == as_of_date].dropna(subset=["return_60d_z"])
    if len(day) < 20:
        return pd.DataFrame()

    day = day.copy()
    day["rank"] = day["return_60d_z"].rank(pct=True)
    # Top decile
    top = day[day["rank"] >= (1 - 1/num_buckets)].sort_values("return_60d_z", ascending=False)
    return top.head(n)


def simulate_last_week(stock_data, signals_df, hold_days=10, max_pos=5):
    """
    Simulate trades that would have been entered in the last 2-3 weeks
    and show their current status.
    """
    all_dates = sorted(signals_df["date"].unique())
    today = all_dates[-1]

    # Find signals from ~10-15 trading days ago (trades that are currently open or just closed)
    lookback_entries = 20  # Check last 20 trading days for entries
    recent_dates = all_dates[-lookback_entries:]

    trades = []
    open_positions = {}

    for date in recent_dates:
        # Clean expired
        date_idx = all_dates.index(date) if date in all_dates else len(all_dates)
        open_positions = {t: info for t, info in open_positions.items()
                         if info["exit_idx"] > date_idx}

        available = max_pos - len(open_positions)
        if available <= 0:
            continue

        top = get_top_momentum_stocks(signals_df, date, n=max_pos)
        if top.empty:
            continue

        candidates = [t for t in top["ticker"].tolist() if t not in open_positions][:available]

        for ticker in candidates:
            if ticker not in stock_data:
                continue
            df = stock_data[ticker]
            try:
                entry_idx = df.index.get_indexer([date], method="nearest")[0]
            except Exception:
                continue

            actual_entry_idx = entry_idx + 1
            if actual_entry_idx >= len(df):
                continue

            entry_price = df.iloc[actual_entry_idx]["Open"]
            entry_date = df.index[actual_entry_idx]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            target_exit_idx = min(actual_entry_idx + hold_days, len(df) - 1)
            target_exit_date = df.index[target_exit_idx]

            # Check stop-loss
            current_price = entry_price
            stopped = False
            exit_date = target_exit_date
            stop_price = entry_price * 0.92  # -8%

            for day_idx in range(actual_entry_idx, min(target_exit_idx + 1, len(df))):
                low = df.iloc[day_idx]["Low"]
                if low <= stop_price:
                    current_price = stop_price
                    stopped = True
                    exit_date = df.index[day_idx]
                    break
                current_price = df.iloc[day_idx]["Close"]

            # Is this trade still open?
            still_open = (not stopped) and (target_exit_idx >= len(df) - 1 or target_exit_date >= today)

            if still_open:
                current_price = df.iloc[-1]["Close"]
                exit_date = None

            pnl_pct = (current_price / entry_price - 1) * 100

            signal_row = top[top["ticker"] == ticker]
            z_score = signal_row["return_60d_z"].values[0] if len(signal_row) > 0 else None

            trades.append({
                "signal_date": date,
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_price": round(entry_price, 2),
                "current_price": round(current_price, 2),
                "exit_date": exit_date,
                "target_exit": target_exit_date,
                "pnl_pct": round(pnl_pct, 2),
                "status": "OPEN" if still_open else ("STOPPED" if stopped else "CLOSED"),
                "z_score": round(z_score, 2) if z_score else None,
                "days_held": (df.index[-1] - entry_date).days if still_open else (exit_date - entry_date).days,
            })

            if not still_open:
                pass  # Already closed
            else:
                open_positions[ticker] = {"exit_idx": target_exit_idx}

    return pd.DataFrame(trades)


def main():
    parser = argparse.ArgumentParser(description="Live Signal Generator")
    parser.add_argument("--date", type=str, default=None, help="Signal date (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  LIVE SIGNAL GENERATOR — 3-Month Momentum Strategy")
    print(f"  Universe: Nifty 200 | Hold: 10 days | Max: 5 positions")
    print(f"{'#'*70}")

    # Download fresh data
    print(f"\n  Downloading fresh data for Nifty 200...")
    stock_data = get_fresh_data(NIFTY_200, days_back=400)
    index_data = download_stock_data(NIFTY50_TICKER,
                                      (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"),
                                      datetime.now().strftime("%Y-%m-%d"),
                                      force=True)

    # Compute signals
    print(f"\n  Computing signals...")
    signals_df = compute_signals(stock_data, index_data)
    all_dates = sorted(signals_df["date"].unique())
    latest_date = all_dates[-1]
    print(f"  Latest data date: {latest_date.strftime('%Y-%m-%d')}")

    # ============================================================
    # 1. LAST 2 WEEKS BACKTEST — what actually happened
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  RECENT TRADE HISTORY (last 2-3 weeks)")
    print(f"{'='*70}")

    trades_df = simulate_last_week(stock_data, signals_df, hold_days=10, max_pos=5)

    if trades_df.empty:
        print("  No trades in recent period")
    else:
        capital = 600000
        per_pos = capital // 5

        # Show closed trades first
        closed = trades_df[trades_df["status"] != "OPEN"].sort_values("entry_date")
        if len(closed) > 0:
            print(f"\n  CLOSED TRADES:")
            print(f"  {'Signal Date':<12} {'Ticker':<12} {'Entry':>8} {'Exit':>8} {'P&L':>7} {'Status':<8} {'Days':>4}")
            total_pnl = 0
            for _, t in closed.iterrows():
                pnl_inr = t["pnl_pct"] / 100 * per_pos
                total_pnl += pnl_inr
                marker = "+" if t["pnl_pct"] > 0 else "*"
                exit_str = t["exit_date"].strftime("%Y-%m-%d") if t["exit_date"] else "?"
                print(f"  {t['signal_date'].strftime('%Y-%m-%d'):<12} {t['ticker']:<12} "
                      f"{t['entry_price']:>8.1f} {t['current_price']:>8.1f} "
                      f"{t['pnl_pct']:>+6.1f}% {t['status']:<8} {t['days_held']:>4} {marker}")
            print(f"\n  Closed P&L: {total_pnl:+,.0f} INR (at {per_pos:,}/position)")
            print(f"  Win rate: {(closed['pnl_pct'] > 0).sum()}/{len(closed)} "
                  f"({(closed['pnl_pct'] > 0).sum()/len(closed)*100:.0f}%)")

        # Show open trades
        open_trades = trades_df[trades_df["status"] == "OPEN"].sort_values("entry_date")
        if len(open_trades) > 0:
            print(f"\n  CURRENTLY OPEN POSITIONS:")
            print(f"  {'Signal Date':<12} {'Ticker':<12} {'Entry':>8} {'Current':>8} {'P&L':>7} {'Z-score':>7} {'Days':>4} {'Exit By'}")
            for _, t in open_trades.iterrows():
                pnl_inr = t["pnl_pct"] / 100 * per_pos
                target = t["target_exit"].strftime("%Y-%m-%d") if pd.notna(t["target_exit"]) else "?"
                stop = t["entry_price"] * 0.92
                print(f"  {t['signal_date'].strftime('%Y-%m-%d'):<12} {t['ticker']:<12} "
                      f"{t['entry_price']:>8.1f} {t['current_price']:>8.1f} "
                      f"{t['pnl_pct']:>+6.1f}% {t['z_score']:>7.2f} {t['days_held']:>4}d  {target}")
                print(f"  {'':>36} Stop-loss: {stop:.1f} | P&L: {pnl_inr:+,.0f} INR")

            unrealized = sum(t["pnl_pct"] / 100 * per_pos for _, t in open_trades.iterrows())
            print(f"\n  Unrealized P&L: {unrealized:+,.0f} INR")
            print(f"  Open positions: {len(open_trades)}/5")

    # ============================================================
    # 2. TODAY'S SIGNALS — what to buy tomorrow
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  TODAY'S SIGNALS — Stocks to BUY at tomorrow's OPEN")
    print(f"  Signal date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

    top = get_top_momentum_stocks(signals_df, latest_date, n=20, num_buckets=10)

    if top.empty:
        print("  No signals generated")
    else:
        # Count open positions
        n_open = len(trades_df[trades_df["status"] == "OPEN"]) if not trades_df.empty else 0
        slots = 5 - n_open
        already_held = set(trades_df[trades_df["status"] == "OPEN"]["ticker"].tolist()) if not trades_df.empty else set()

        print(f"\n  Open slots: {slots}/5")
        print(f"  Already holding: {', '.join(already_held) if already_held else 'none'}")

        print(f"\n  TOP DECILE — Strongest 3-Month Momentum (z-scored):")
        print(f"  {'Rank':>4} {'Ticker':<12} {'Close':>8} {'60d Ret':>8} {'Z-score':>8} {'RSI(3)':>7} {'Action'}")

        buy_list = []
        for i, (_, row) in enumerate(top.iterrows(), 1):
            action = ""
            if row["ticker"] in already_held:
                action = "ALREADY HELD"
            elif len(buy_list) < slots:
                action = ">> BUY"
                buy_list.append(row)
            else:
                action = "(backup)"

            print(f"  {i:>4} {row['ticker']:<12} {row['close']:>8.1f} "
                  f"{row['return_60d_raw']*100:>+7.1f}% {row['return_60d_z']:>8.2f} "
                  f"{row['rsi_3']:>7.1f} {action}")

        if buy_list:
            per_pos = 600000 // 5
            print(f"\n  {'='*50}")
            print(f"  ACTION PLAN — Execute at market open tomorrow")
            print(f"  {'='*50}")
            for row in buy_list:
                stop = row["close"] * 0.92
                target_exit = (latest_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")  # ~10 trading days
                print(f"\n  BUY {row['ticker']}")
                print(f"    Amount: {per_pos:,} INR (~{int(per_pos / row['close'])} shares)")
                print(f"    Entry: Market open tomorrow")
                print(f"    Stop-loss: {stop:.1f} (-8%)")
                print(f"    Exit: Close on ~{target_exit} (10 trading days)")
                print(f"    Reason: 3m momentum z-score = {row['return_60d_z']:.2f} (top decile)")
        else:
            print(f"\n  No new buys — all 5 slots occupied")

    # ============================================================
    # 3. LAST WEEK SUMMARY
    # ============================================================
    if not trades_df.empty:
        # Trades that entered or exited in the last 5 trading days
        last_week = latest_date - pd.Timedelta(days=7)
        week_trades = trades_df[
            (trades_df["entry_date"] >= last_week) |
            ((trades_df["exit_date"] is not None) & (trades_df["status"] != "OPEN"))
        ]

        per_pos = 600000 // 5
        all_closed = trades_df[trades_df["status"] != "OPEN"]
        if len(all_closed) > 0:
            total_pnl = sum(t["pnl_pct"] / 100 * per_pos for _, t in all_closed.iterrows())
            print(f"\n{'='*70}")
            print(f"  WEEKLY SUMMARY")
            print(f"{'='*70}")
            print(f"  Realized P&L (recent closed): {total_pnl:+,.0f} INR")

            open_trades = trades_df[trades_df["status"] == "OPEN"]
            if len(open_trades) > 0:
                unrealized = sum(t["pnl_pct"] / 100 * per_pos for _, t in open_trades.iterrows())
                print(f"  Unrealized P&L (open):        {unrealized:+,.0f} INR")
                print(f"  Total (realized+unrealized):  {total_pnl + unrealized:+,.0f} INR")


if __name__ == "__main__":
    main()
