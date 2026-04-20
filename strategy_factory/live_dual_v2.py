#!/usr/bin/env python3
"""
Production Live Signal Generator V2 — Dual Strategy

Bull mode (Nifty > SMA, breadth > 60%): Buy 5d dip bottom decile, hold 10d, 3 pos
Bear mode (Nifty < SMA, breadth < 40%): RSI3 oversold bottom decile, hold 5d, 5 pos
No-trade zone: Breadth 40-60%

V1 fixes:
1. NSE 2026 holiday calendar
2. Overlap protection (max 5 total positions across modes)
3. Gap filter (skip stocks opening 5%+ below prev close)

V2 additions:
4. Crash filter: skip stocks that dropped >8% in a single day in last 10 days
5. Downtrend filter: skip stocks below 60-day SMA (structural downtrend, not a dip)
6. Volume spike crash filter: skip if any day had >4x avg volume with >5% drop (panic selling)
7. Flags on each pick: CLEAN, CAUTION, or SKIP with reasons

Usage:
    python live_dual.py              # Today's signals
    python live_dual.py --status     # Just show regime + open positions
"""
import os, sys, argparse, warnings, json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, NIFTY50_TICKER
from signals.scanner import zscore_normalize

# ============================================================
# NSE HOLIDAYS 2026
# ============================================================
NSE_HOLIDAYS_2026 = [
    "2026-01-26",  # Republic Day
    "2026-02-17",  # Mahashivratri
    "2026-03-10",  # Holi (tentative — verify)
    "2026-03-26",  # Holi
    "2026-03-31",  # Id-ul-Fitr
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-06-07",  # Id-ul-Adha (Bakri Id)
    "2026-07-07",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-08-18",  # Parsi New Year
    "2026-09-05",  # Milad-un-Nabi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-09",  # Diwali (Laxmi Puja)
    "2026-11-10",  # Diwali (Balipratipada)
    "2026-11-30",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
]
NSE_HOLIDAYS = set(pd.to_datetime(NSE_HOLIDAYS_2026).strftime("%Y-%m-%d"))

CAPITAL = 600000
MAX_TOTAL_POSITIONS = 5
STOP_LOSS_PCT = 0.08
POSITION_FILE = "data/open_positions.json"


def is_trading_day(date):
    """Check if a date is an NSE trading day."""
    d = pd.Timestamp(date)
    if d.weekday() >= 5:  # Saturday/Sunday
        return False
    if d.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        return False
    return True


def get_next_n_trading_days(from_date, n):
    """Get the next N trading days from a date (exclusive of from_date)."""
    days = []
    d = pd.Timestamp(from_date) + timedelta(days=1)
    while len(days) < n:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def get_exit_date(entry_date, hold_days):
    """Get the exit date (Nth trading day from entry, inclusive of entry as day 1)."""
    days = [pd.Timestamp(entry_date)]
    d = pd.Timestamp(entry_date) + timedelta(days=1)
    while len(days) < hold_days:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days[-1]


def load_positions():
    """Load open positions from file."""
    if os.path.exists(POSITION_FILE):
        try:
            with open(POSITION_FILE) as f:
                positions = json.load(f)
            # Clean expired
            today = datetime.now().strftime("%Y-%m-%d")
            active = [p for p in positions if p["exit_date"] >= today]
            return active
        except Exception:
            return []
    return []


def save_positions(positions):
    """Save positions to file."""
    os.makedirs(os.path.dirname(POSITION_FILE), exist_ok=True)
    with open(POSITION_FILE, "w") as f:
        json.dump(positions, f, indent=2)


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


def compute_regime(nifty_data, stock_data):
    """Compute regime and breadth."""
    close = nifty_data["Close"]
    sma20 = close.rolling(20).mean()
    nifty_close = close.iloc[-1]
    nifty_sma = sma20.iloc[-1]
    is_bear = nifty_close < nifty_sma

    # Breadth
    n_above = 0
    n_total = 0
    for ticker, df in stock_data.items():
        c = df["Close"]
        s = c.rolling(20).mean()
        if len(c) >= 20:
            n_total += 1
            if c.iloc[-1] > s.iloc[-1]:
                n_above += 1

    breadth = n_above / n_total * 100 if n_total > 0 else 50

    return {
        "nifty_close": nifty_close,
        "nifty_sma": nifty_sma,
        "is_bear": is_bear,
        "breadth": breadth,
        "n_above": n_above,
        "n_total": n_total,
    }


def compute_quality_metrics(df):
    """Compute quality/safety metrics for a stock. Returns dict or None."""
    close = df["Close"]
    volume = df["Volume"]
    if len(close) < 60:
        return None

    daily_ret = close.pct_change()
    last_10d_ret = daily_ret.tail(10)
    avg_vol = volume.rolling(20).mean()

    # Worst single-day drop in last 10 days
    worst_1d = last_10d_ret.min() * 100

    # Check for panic day: >4x volume AND >5% drop in last 10 days
    panic_days = 0
    for i in range(-10, 0):
        if i >= -len(df):
            d_ret = daily_ret.iloc[i]
            d_vol = volume.iloc[i]
            d_avg = avg_vol.iloc[i] if pd.notna(avg_vol.iloc[i]) and avg_vol.iloc[i] > 0 else 1
            if d_ret < -0.05 and d_vol > 4 * d_avg:
                panic_days += 1

    # Is stock below its 60-day SMA? (structural downtrend)
    sma60 = close.rolling(60).mean().iloc[-1]
    below_sma60 = close.iloc[-1] < sma60

    # Distance from 60d SMA
    dist_sma60 = (close.iloc[-1] / sma60 - 1) * 100 if sma60 > 0 else 0

    return {
        "worst_1d_10d": round(worst_1d, 1),
        "panic_days": panic_days,
        "below_sma60": below_sma60,
        "sma60": round(sma60, 1),
        "dist_sma60": round(dist_sma60, 1),
    }


def assess_quality(qm):
    """Return (flag, reasons) based on quality metrics."""
    reasons = []
    flag = "CLEAN"

    if qm["worst_1d_10d"] < -8:
        flag = "SKIP"
        reasons.append(f"Crashed {qm['worst_1d_10d']}% in a single day (last 10d)")
    elif qm["worst_1d_10d"] < -5:
        if flag != "SKIP":
            flag = "CAUTION"
        reasons.append(f"Dropped {qm['worst_1d_10d']}% in a day (last 10d)")

    if qm["panic_days"] > 0:
        flag = "SKIP"
        reasons.append(f"{qm['panic_days']} panic day(s): >4x volume + >5% drop")

    if qm["below_sma60"]:
        if qm["dist_sma60"] < -10:
            flag = "SKIP"
            reasons.append(f"Deep below 60d SMA ({qm['sma60']:.0f}, {qm['dist_sma60']:+.1f}%) — structural downtrend")
        elif qm["dist_sma60"] < -5:
            if flag != "SKIP":
                flag = "CAUTION"
            reasons.append(f"Below 60d SMA ({qm['sma60']:.0f}, {qm['dist_sma60']:+.1f}%)")

    if not reasons:
        reasons.append("No red flags")

    return flag, reasons


def compute_bear_signals(stock_data):
    """RSI(3) z-score for bear mode."""
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
            rsi3 = 100 - (100 / (1 + rs))
            rsi3_z = zscore_normalize(rsi3, lookback=252)

            qm = compute_quality_metrics(df)
            if qm is None:
                continue

            if pd.notna(rsi3_z.iloc[-1]):
                signals.append({
                    "ticker": ticker,
                    "close": close.iloc[-1],
                    "prev_close": close.iloc[-2] if len(close) >= 2 else close.iloc[-1],
                    "signal_value": rsi3_z.iloc[-1],
                    "rsi3": rsi3.iloc[-1],
                    "ret5d": close.pct_change(5).iloc[-1] * 100,
                    **qm,
                })
        except Exception:
            pass
    return pd.DataFrame(signals).dropna(subset=["signal_value"]).sort_values("signal_value")


def compute_bull_signals(stock_data):
    """5-day return z-score for bull mode."""
    signals = []
    for ticker, df in stock_data.items():
        try:
            close = df["Close"]
            ret5d = close.pct_change(5)
            ret5d_z = zscore_normalize(ret5d, lookback=252)

            qm = compute_quality_metrics(df)
            if qm is None:
                continue

            if pd.notna(ret5d_z.iloc[-1]):
                signals.append({
                    "ticker": ticker,
                    "close": close.iloc[-1],
                    "prev_close": close.iloc[-2] if len(close) >= 2 else close.iloc[-1],
                    "signal_value": ret5d_z.iloc[-1],
                    "ret5d": ret5d.iloc[-1] * 100,
                    "ret20d": close.pct_change(20).iloc[-1] * 100 if len(close) >= 20 else 0,
                    **qm,
                })
        except Exception:
            pass
    return pd.DataFrame(signals).dropna(subset=["signal_value"]).sort_values("signal_value")


def main():
    parser = argparse.ArgumentParser(description="Dual Strategy Live Signal Generator")
    parser.add_argument("--status", action="store_true", help="Show regime and positions only")
    parser.add_argument("--date", type=str, default=None,
                        help="Expected data date (YYYY-MM-DD). Warns if data is stale.")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    today_trading = is_trading_day(today)
    expected_date = args.date  # User can pass --date 2026-04-10 to verify

    print(f"\n{'#'*70}")
    print(f"  DUAL STRATEGY — Live Signal Generator")
    print(f"  Date: {today} ({'Trading day' if today_trading else 'MARKET CLOSED'})")
    print(f"{'#'*70}")

    # Check upcoming holidays
    upcoming = [h for h in sorted(NSE_HOLIDAYS) if h >= today and h <= (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")]
    if upcoming:
        print(f"  Upcoming holidays: {', '.join(upcoming)}")

    # Download data
    print(f"\n  Downloading data...")
    stock_data = download_fresh(NIFTY_200)
    nifty_data = download_fresh([NIFTY50_TICKER]).get(NIFTY50_TICKER)

    # Data staleness check
    if nifty_data is not None and not nifty_data.empty:
        data_date = nifty_data.index[-1].strftime("%Y-%m-%d")
        print(f"\n  Data as of: {data_date}")
        if expected_date and data_date != expected_date:
            print(f"  WARNING: Expected {expected_date} but got {data_date}. Data may be STALE!")
            print(f"  Re-run after 5 PM or try again later.")
            return
        elif not expected_date:
            # Auto-check: if today is a trading day and data is not from today, warn
            if today_trading and data_date != today:
                print(f"  WARNING: Today is a trading day but data is from {data_date}.")
                print(f"  If market has closed, data may be delayed. Re-run after 5 PM.")
                print(f"  To proceed anyway, re-run with: python live_dual.py --date {data_date}")

    if nifty_data is None or nifty_data.empty:
        print("  ERROR: Could not download Nifty data")
        return

    # Regime
    regime = compute_regime(nifty_data, stock_data)

    print(f"\n  {'='*60}")
    print(f"  MARKET REGIME")
    print(f"  Nifty: {regime['nifty_close']:.0f} | 20d SMA: {regime['nifty_sma']:.0f}")
    print(f"  Breadth: {regime['n_above']}/{regime['n_total']} = {regime['breadth']:.1f}%")

    if regime["is_bear"] and regime["breadth"] < 40:
        mode = "BEAR"
        print(f"  Mode: BEAR (Nifty < SMA, breadth < 40%)")
        print(f"  Strategy: RSI3 oversold → buy 5, hold 5 trading days")
    elif not regime["is_bear"] and regime["breadth"] > 60:
        mode = "BULL"
        print(f"  Mode: BULL (Nifty > SMA, breadth > 60%)")
        print(f"  Strategy: 5d dip buy → buy 3, hold 10 trading days")
    else:
        mode = "NONE"
        print(f"  Mode: NO TRADE ZONE (breadth {regime['breadth']:.0f}% — between 40-60%)")
    print(f"  {'='*60}")

    # Load positions
    positions = load_positions()
    held_tickers = set(p["ticker"] for p in positions)
    n_open = len(positions)

    if positions:
        print(f"\n  OPEN POSITIONS ({n_open}/{MAX_TOTAL_POSITIONS}):")
        print(f"  {'Ticker':<12} {'Mode':<5} {'Entry':>8} {'Stop':>8} {'Entry Date':<12} {'Exit Date':<12}")
        for p in sorted(positions, key=lambda x: x["exit_date"]):
            # Get current price
            curr = stock_data.get(p["ticker"])
            curr_price = curr["Close"].iloc[-1] if curr is not None else 0
            pnl = (curr_price / p["entry_price"] - 1) * 100 if p["entry_price"] > 0 else 0
            print(f"  {p['ticker']:<12} {p['mode']:<5} {p['entry_price']:>8.1f} {p['stop']:>8.1f} {p['entry_date']:<12} {p['exit_date']:<12}  Now: {curr_price:.1f} ({pnl:+.1f}%)")

    # Check exits due
    exits_due = [p for p in positions if p["exit_date"] <= today]
    if exits_due:
        print(f"\n  EXITS DUE TODAY:")
        for p in exits_due:
            curr = stock_data.get(p["ticker"])
            curr_price = curr["Close"].iloc[-1] if curr is not None else 0
            pnl_pct = (curr_price / p["entry_price"] - 1) * 100
            pnl_inr = pnl_pct / 100 * p["amount"]
            print(f"  >> SELL {p['ticker']} at close | Entry: {p['entry_price']:.1f} | Now: {curr_price:.1f} | P&L: {pnl_pct:+.1f}% ({pnl_inr:+,.0f} INR)")

    if args.status:
        return

    # Signals
    if mode == "NONE":
        print(f"\n  No signals — sitting in cash until breadth moves beyond 40-60% zone.")
        return

    slots = MAX_TOTAL_POSITIONS - n_open
    # Account for positions that exit today — those slots free up
    slots += len(exits_due)

    if slots <= 0:
        print(f"\n  All {MAX_TOTAL_POSITIONS} slots occupied. Wait for exits.")
        return

    if mode == "BEAR":
        max_new = min(slots, 5)
        hold_days = 5
        signals = compute_bear_signals(stock_data)
        signal_label = "RSI(3) z-score"
    else:  # BULL
        max_new = min(slots, 3)
        hold_days = 10
        signals = compute_bull_signals(stock_data)
        signal_label = "5d return z-score"

    # Bottom decile
    cutoff = max(len(signals) // 10, 3)
    candidates = signals.head(cutoff)

    # Filter: skip already held
    candidates = candidates[~candidates["ticker"].isin(held_tickers)]

    # Gap filter: skip if today's close is 5%+ below prev close (would open even lower)
    candidates = candidates[
        (candidates["close"] / candidates["prev_close"] - 1) > -0.05
    ].copy()

    # Quality assessment
    candidates["flag"] = ""
    candidates["flag_reasons"] = ""
    for idx, row in candidates.iterrows():
        qm = {k: row[k] for k in ["worst_1d_10d", "panic_days", "below_sma60", "sma60", "dist_sma60"] if k in row}
        flag, reasons = assess_quality(qm)
        candidates.at[idx, "flag"] = flag
        candidates.at[idx, "flag_reasons"] = "; ".join(reasons)

    # Show all candidates with flags before filtering
    skipped = candidates[candidates["flag"] == "SKIP"]
    if len(skipped) > 0:
        print(f"\n  FILTERED OUT (quality issues):")
        for _, row in skipped.iterrows():
            print(f"    {row['ticker']:<12} — {row['flag_reasons']}")

    caution = candidates[candidates["flag"] == "CAUTION"]
    if len(caution) > 0:
        print(f"\n  CAUTION (proceed with awareness):")
        for _, row in caution.iterrows():
            print(f"    {row['ticker']:<12} — {row['flag_reasons']}")

    # Remove SKIP stocks, keep CAUTION with warning
    candidates = candidates[candidates["flag"] != "SKIP"]

    picks = candidates.head(max_new)

    per_pos = CAPITAL // MAX_TOTAL_POSITIONS

    # Calculate exit date using holiday calendar
    # Entry is next trading day
    next_td = datetime.now() + timedelta(days=1)
    while not is_trading_day(next_td.strftime("%Y-%m-%d")):
        next_td += timedelta(days=1)
    entry_date = next_td.strftime("%Y-%m-%d")
    exit_date = get_exit_date(entry_date, hold_days).strftime("%Y-%m-%d")

    print(f"\n  {'='*60}")
    print(f"  {'BEAR' if mode == 'BEAR' else 'BULL'} SIGNALS — Buy at {entry_date} open")
    print(f"  Slots available: {slots} | New picks: {len(picks)}")
    print(f"  Hold: {hold_days} trading days → Exit at {exit_date} close")
    print(f"  {'='*60}")

    if len(picks) == 0:
        print(f"\n  No qualifying stocks after filters.")
        return

    print(f"\n  {'#':>3} {'Ticker':<12} {'CMP':>8} {'Shares':>7} {'Amount':>9} {'Stop':>8} {signal_label:>16} {'5d Ret':>7} {'Flag':<8}")
    print(f"  {'-'*85}")

    new_positions = []
    for i, (_, row) in enumerate(picks.iterrows(), 1):
        shares = int(per_pos / row["close"])
        amount = shares * row["close"]
        stop = round(row["close"] * (1 - STOP_LOSS_PCT), 1)

        extra = ""
        if mode == "BEAR":
            extra = f" RSI3={row.get('rsi3', 0):.0f}"

        flag_display = row.get("flag", "")
        print(f"  {i:>3} {row['ticker']:<12} {row['close']:>8.1f} {shares:>7} {amount:>9,.0f} {stop:>8.1f} {row['signal_value']:>16.2f} {row.get('ret5d', 0):>+6.1f}% {flag_display:<8}{extra}")
        if flag_display == "CAUTION":
            print(f"      ^^ {row.get('flag_reasons', '')}")

        new_positions.append({
            "ticker": row["ticker"],
            "mode": mode,
            "entry_price": round(row["close"], 2),  # Approximate — actual is tomorrow's open
            "stop": stop,
            "shares": shares,
            "amount": round(amount, 0),
            "entry_date": entry_date,
            "exit_date": exit_date,
        })

    # Trading day count verification
    print(f"\n  TRADING DAY COUNT:")
    td = pd.Timestamp(entry_date)
    for d in range(1, hold_days + 1):
        label = "ENTRY" if d == 1 else ("EXIT at close" if d == hold_days else "")
        print(f"    Day {d}: {td.strftime('%Y-%m-%d')} ({td.strftime('%A')}){' — ' + label if label else ''}")
        next_d = td + timedelta(days=1)
        while not is_trading_day(next_d.strftime("%Y-%m-%d")):
            skip_date = next_d
            reason = "Weekend" if next_d.weekday() >= 5 else "HOLIDAY"
            if d < hold_days:
                print(f"           {skip_date.strftime('%Y-%m-%d')} ({skip_date.strftime('%A')}) — {reason}, skip")
            next_d += timedelta(days=1)
        td = next_d

    # Save positions
    print(f"\n  Save these positions? Run with --save to confirm.")
    # Remove exited positions and add new ones
    if "--save" in sys.argv:
        active = [p for p in positions if p["exit_date"] > today]
        active.extend(new_positions)
        save_positions(active)
        print(f"  Saved {len(new_positions)} new + {len(active) - len(new_positions)} existing = {len(active)} total positions")

    print(f"\n{'#'*70}")


if __name__ == "__main__":
    main()
