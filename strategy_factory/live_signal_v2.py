#!/usr/bin/env python3
"""
Live Signal Generator V2 — RSI(3) Bear Dip Strategy

Only generates signals when Nifty < 20d SMA (bear regime).
Buys most oversold stocks (lowest RSI-3 z-score decile).
Hold 5 days, -8% stop, max 5 positions.

Usage:
    python live_signal_v2.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, NIFTY50_TICKER
from signals.scanner import download_stock_data, zscore_normalize


CAPITAL = 600000
MAX_POS = 5
PER_POS = CAPITAL // MAX_POS
HOLD_DAYS = 5
STOP_LOSS_PCT = 0.08


def download_fresh(tickers, days=400):
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    total = len(tickers)
    for i, t in enumerate(tickers, 1):
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}]...")
        yf_t = t if t.startswith("^") else f"{t}.NS"
        try:
            df = yf.download(yf_t, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= 60:
                data[t] = df
        except Exception:
            pass
    return data


def check_regime(nifty_data, sma_period=20):
    close = nifty_data["Close"]
    sma = close.rolling(sma_period).mean()
    latest_close = close.iloc[-1]
    latest_sma = sma.iloc[-1]
    is_bear = latest_close < latest_sma
    return is_bear, latest_close, latest_sma


def compute_rsi3_zscore(stock_data):
    signals = []
    for ticker, df in stock_data.items():
        try:
            close = df["Close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/3, min_periods=3).mean()
            avg_loss = loss.ewm(alpha=1/3, min_periods=3).mean()
            rs = avg_gain / avg_loss
            rsi_3 = 100 - (100 / (1 + rs))
            rsi_z = zscore_normalize(rsi_3, lookback=252)

            latest = df.index[-1]
            signals.append({
                "ticker": ticker,
                "date": latest,
                "close": close.iloc[-1],
                "rsi_3": rsi_3.iloc[-1],
                "rsi_3_z": rsi_z.iloc[-1],
                "return_5d": close.pct_change(5).iloc[-1] * 100,
                "return_1d": close.pct_change(1).iloc[-1] * 100,
            })
        except Exception:
            pass
    return pd.DataFrame(signals)


def simulate_recent_trades(stock_data, all_signals_history, nifty_data, days_back=20):
    """Simulate what the strategy would have done recently."""
    nifty_close = nifty_data["Close"]
    nifty_sma = nifty_close.rolling(20).mean()

    # Get last N trading dates
    dates = sorted(nifty_close.index)[-days_back:]
    trades = []
    open_pos = {}
    blocked = {}

    for date in dates:
        # Check regime
        if date not in nifty_sma.index or pd.isna(nifty_sma[date]):
            continue
        is_bear = nifty_close[date] < nifty_sma[date]
        if not is_bear:
            continue

        # Clean expired
        open_pos = {t: info for t, info in open_pos.items() if info["exit_date"] > date}
        blocked = {t: d for t, d in blocked.items() if d > date}

        available = MAX_POS - len(open_pos)
        if available <= 0:
            continue

        # Compute RSI(3) z-score for all stocks on this date
        day_signals = []
        for ticker, df in stock_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < 60:
                continue
            close = df["Close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/3, min_periods=3).mean()
            avg_loss = loss.ewm(alpha=1/3, min_periods=3).mean()
            rs = avg_gain / avg_loss
            rsi_3 = 100 - (100 / (1 + rs))
            rsi_z = zscore_normalize(rsi_3, lookback=252)

            if pd.notna(rsi_z.iloc[idx]):
                day_signals.append({"ticker": ticker, "rsi_z": rsi_z.iloc[idx], "close": close.iloc[idx]})

        if len(day_signals) < 20:
            continue

        day_df = pd.DataFrame(day_signals).sort_values("rsi_z")
        # Bottom decile
        cutoff = len(day_df) // 10
        candidates = day_df.head(max(cutoff, 1))
        candidates = candidates[~candidates["ticker"].isin(open_pos) & ~candidates["ticker"].isin(blocked)]

        for _, row in candidates.head(available).iterrows():
            ticker = row["ticker"]
            df = stock_data[ticker]
            idx = df.index.get_loc(date)
            entry_idx = idx + 1
            if entry_idx >= len(df):
                continue

            entry_price = df.iloc[entry_idx]["Open"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            exit_idx = min(entry_idx + HOLD_DAYS, len(df) - 1)
            exit_price = entry_price
            stopped = False
            stop_price = entry_price * (1 - STOP_LOSS_PCT)

            for di in range(entry_idx, exit_idx + 1):
                if df.iloc[di]["Low"] <= stop_price:
                    exit_price = stop_price
                    stopped = True
                    break
                if di == exit_idx:
                    exit_price = df.iloc[di]["Close"]

            still_open = (not stopped) and (exit_idx >= len(df) - 1)
            if still_open:
                exit_price = df.iloc[-1]["Close"]

            pnl_pct = (exit_price / entry_price - 1) * 100

            trades.append({
                "signal_date": date,
                "ticker": ticker,
                "entry_date": df.index[entry_idx],
                "entry_price": round(entry_price, 2),
                "current_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "pnl_inr": round(pnl_pct / 100 * PER_POS),
                "status": "OPEN" if still_open else ("STOPPED" if stopped else "CLOSED"),
                "rsi_z": round(row["rsi_z"], 2),
                "days_held": (df.index[-1] - df.index[entry_idx]).days if still_open else (df.index[min(exit_idx, len(df)-1)] - df.index[entry_idx]).days,
                "stop_price": round(stop_price, 1),
                "target_exit": df.index[exit_idx] if exit_idx < len(df) else df.index[-1],
            })

            exit_date = df.index[exit_idx] if exit_idx < len(df) else df.index[-1]
            open_pos[ticker] = {"exit_date": exit_date}
            if stopped:
                blocked[ticker] = date + pd.Timedelta(days=8)

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def main():
    print(f"\n{'#'*70}")
    print(f"  RSI(3) BEAR DIP STRATEGY — Live Signal Generator")
    print(f"  Capital: {CAPITAL:,} | Positions: {MAX_POS} | Hold: {HOLD_DAYS}d")
    print(f"{'#'*70}")

    # Download data
    print(f"\n  Downloading Nifty 200 + index data...")
    stock_data = download_fresh(NIFTY_200)
    nifty_data = download_fresh([NIFTY50_TICKER])
    nifty_data = nifty_data.get(NIFTY50_TICKER, pd.DataFrame())

    if nifty_data.empty:
        print("  ERROR: Could not download Nifty data")
        sys.exit(1)

    # Check regime
    is_bear, nifty_close, nifty_sma = check_regime(nifty_data)

    print(f"\n  {'='*60}")
    print(f"  MARKET REGIME")
    print(f"  Nifty 50 Close: {nifty_close:.0f}")
    print(f"  Nifty 50 20d SMA: {nifty_sma:.0f}")
    if is_bear:
        print(f"  Status: BEAR (close < SMA) — STRATEGY IS ACTIVE")
    else:
        print(f"  Status: BULL (close > SMA) — STRATEGY IS PAUSED (no new entries)")
    print(f"  {'='*60}")

    # Recent trade simulation
    print(f"\n  Simulating recent trades...")
    recent = simulate_recent_trades(stock_data, None, nifty_data, days_back=25)

    if not recent.empty:
        closed = recent[recent["status"] != "OPEN"].sort_values("entry_date")
        opens = recent[recent["status"] == "OPEN"].sort_values("entry_date")

        if len(closed) > 0:
            print(f"\n  RECENTLY CLOSED TRADES:")
            print(f"  {'Date':<12} {'Ticker':<12} {'Entry':>8} {'Exit':>8} {'P&L':>7} {'INR':>8} {'Status':<8}")
            total_closed_pnl = 0
            for _, t in closed.iterrows():
                marker = "+" if t["pnl_pct"] > 0 else "*"
                print(f"  {t['signal_date'].strftime('%Y-%m-%d'):<12} {t['ticker']:<12} "
                      f"{t['entry_price']:>8.1f} {t['current_price']:>8.1f} "
                      f"{t['pnl_pct']:>+6.1f}% {t['pnl_inr']:>+8,} {t['status']:<8} {marker}")
                total_closed_pnl += t["pnl_inr"]
            print(f"  Realized P&L: {total_closed_pnl:+,} INR | WR: {(closed['pnl_pct']>0).sum()}/{len(closed)}")

        if len(opens) > 0:
            print(f"\n  OPEN POSITIONS:")
            print(f"  {'Date':<12} {'Ticker':<12} {'Entry':>8} {'Now':>8} {'P&L':>7} {'INR':>8} {'Stop':>8} {'Exit By'}")
            total_open_pnl = 0
            for _, t in opens.iterrows():
                exit_by = t["target_exit"].strftime("%Y-%m-%d") if pd.notna(t["target_exit"]) else "?"
                print(f"  {t['signal_date'].strftime('%Y-%m-%d'):<12} {t['ticker']:<12} "
                      f"{t['entry_price']:>8.1f} {t['current_price']:>8.1f} "
                      f"{t['pnl_pct']:>+6.1f}% {t['pnl_inr']:>+8,} {t['stop_price']:>8.1f} {exit_by}")
                total_open_pnl += t["pnl_inr"]
            print(f"  Unrealized P&L: {total_open_pnl:+,} INR")

        all_pnl = recent["pnl_inr"].sum()
        print(f"\n  Total P&L (all recent): {all_pnl:+,} INR")

    # Today's signals
    print(f"\n  {'='*60}")
    if is_bear:
        print(f"  TODAY'S BUY SIGNALS (regime: BEAR — active)")
    else:
        print(f"  NO SIGNALS TODAY (regime: BULL — paused)")
        print(f"  Strategy only enters during corrections.")
        print(f"  Will activate when Nifty drops below {nifty_sma:.0f}")
    print(f"  {'='*60}")

    if is_bear:
        signals = compute_rsi3_zscore(stock_data)
        signals = signals.dropna(subset=["rsi_3_z"]).sort_values("rsi_3_z")

        # Bottom decile
        cutoff = len(signals) // 10
        bottom = signals.head(max(cutoff, 1))

        # Filter out currently open positions
        held = set(recent[recent["status"] == "OPEN"]["ticker"].tolist()) if not recent.empty else set()
        n_open = len(held)
        slots = MAX_POS - n_open

        print(f"\n  Open slots: {slots}/{MAX_POS}")
        if held:
            print(f"  Already holding: {', '.join(held)}")

        print(f"\n  BOTTOM DECILE — Most Oversold (RSI-3 z-score):")
        print(f"  {'#':>3} {'Ticker':<12} {'Close':>8} {'RSI(3)':>7} {'Z-score':>8} {'5d Ret':>7} {'Action'}")

        buy_list = []
        for i, (_, row) in enumerate(bottom.iterrows(), 1):
            if row["ticker"] in held:
                action = "HELD"
            elif len(buy_list) < slots:
                action = ">> BUY"
                buy_list.append(row)
            else:
                action = "(backup)"
            print(f"  {i:>3} {row['ticker']:<12} {row['close']:>8.1f} {row['rsi_3']:>7.1f} "
                  f"{row['rsi_3_z']:>8.2f} {row['return_5d']:>+6.1f}% {action}")

        if buy_list:
            print(f"\n  {'='*60}")
            print(f"  ACTION PLAN — Execute at market OPEN tomorrow")
            print(f"  {'='*60}")
            for row in buy_list:
                stop = row["close"] * (1 - STOP_LOSS_PCT)
                exit_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                shares = int(PER_POS / row["close"])
                print(f"\n  BUY {row['ticker']}")
                print(f"    Shares: ~{shares} (amount: {PER_POS:,} INR)")
                print(f"    Entry: Market OPEN tomorrow")
                print(f"    Stop-loss: {stop:.1f} (-{STOP_LOSS_PCT*100:.0f}%)")
                print(f"    Target exit: ~{exit_date} (5 trading days)")
                print(f"    RSI(3): {row['rsi_3']:.1f} | Z-score: {row['rsi_3_z']:.2f}")
        elif slots == 0:
            print(f"\n  All {MAX_POS} slots occupied. Wait for exits.")
        else:
            print(f"\n  No stocks in bottom decile qualify today.")

    print(f"\n{'#'*70}")


if __name__ == "__main__":
    main()
