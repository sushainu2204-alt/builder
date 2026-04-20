#!/usr/bin/env python3
"""
Per-Strategy Backtest — Uses actual strategy modules with quality filters.

Simulates each strategy INDEPENDENTLY over the past year.
Uses the real compute_signals() and get_picks() from each strategy module,
so quality filters are applied exactly as in live signal generation.

Usage:
    python backtest_per_strategy.py                # All strategies, 1 year
    python backtest_per_strategy.py --months 6     # Last 6 months
    python backtest_per_strategy.py --strategy 2   # Only strategy 2 (bull breakout)
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, NIFTY_500, NIFTY50_TICKER
import strategy_bear
import strategy_bull
import strategy_bull_accel
import strategy_bull_relstr
import strategy_bull_dip

CAPITAL = 600000


def download_fresh(tickers, days=500):
    end = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    total = len(tickers)
    for i, t in enumerate(tickers, 1):
        if i % 100 == 0 or i == total:
            print(f"    [{i}/{total}]...")
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


def compute_regime(nifty_data, stock_data_n200, as_of_idx):
    """Compute regime using data up to as_of_idx."""
    close = nifty_data["Close"].iloc[:as_of_idx+1]
    sma20 = close.rolling(20).mean()
    if pd.isna(sma20.iloc[-1]):
        return None

    n_above, n_total = 0, 0
    for t, df in stock_data_n200.items():
        c = df["Close"]
        # Align to same date
        mask = c.index <= close.index[-1]
        c_sub = c[mask]
        if len(c_sub) < 20:
            continue
        s = c_sub.rolling(20).mean()
        n_total += 1
        if c_sub.iloc[-1] > s.iloc[-1]:
            n_above += 1

    return {
        "nifty_close": close.iloc[-1],
        "nifty_sma": sma20.iloc[-1],
        "is_bear": close.iloc[-1] < sma20.iloc[-1],
        "breadth": n_above / n_total * 100 if n_total > 0 else 50,
        "n_above": n_above, "n_total": n_total,
    }


def trim_data(stock_data, up_to_date):
    """Return stock_data trimmed to up_to_date (inclusive)."""
    trimmed = {}
    for t, df in stock_data.items():
        mask = df.index <= up_to_date
        sub = df[mask]
        if len(sub) >= 60:
            trimmed[t] = sub
    return trimmed


def sim_trade(stock_df, signal_date, hold_days, stop_pct):
    """Simulate a single trade: enter next day open, hold, check stop daily."""
    dates = stock_df.index
    try:
        sig_loc = dates.get_indexer([signal_date], method="nearest")[0]
    except Exception:
        return None

    entry_idx = sig_loc + 1
    if entry_idx >= len(stock_df) - 1:
        return None

    entry_price = stock_df.iloc[entry_idx]["Open"]
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    stop_price = entry_price * (1 - stop_pct)
    exit_idx = min(entry_idx + hold_days, len(stock_df) - 1)
    exit_price = entry_price
    stopped = False

    for di in range(entry_idx, exit_idx + 1):
        low = stock_df.iloc[di]["Low"]
        if not pd.isna(low) and low <= stop_price:
            exit_price = stop_price
            stopped = True
            break
        if di == exit_idx:
            exit_price = stock_df.iloc[di]["Close"]

    if pd.isna(exit_price) or exit_price <= 0:
        return None

    gross_ret = (exit_price / entry_price - 1) * 100
    net_ret = gross_ret - 0.1  # 0.1% round trip cost

    # Day-1 return
    day1_close = stock_df.iloc[entry_idx]["Close"]
    day1_ret = (day1_close / entry_price - 1) * 100 if not pd.isna(day1_close) else 0

    return {
        "entry_date": dates[entry_idx],
        "exit_date": dates[min(exit_idx, len(dates)-1)],
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "day1_ret": round(day1_ret, 2),
        "full_ret": round(net_ret, 2),
        "stopped": stopped,
    }


def backtest_strategy(strat, stock_data, nifty_data, stock_data_n200, extra_data, test_dates, per_pos):
    """Backtest a single strategy over test_dates. Returns trades DataFrame."""
    import inspect

    trades = []
    open_positions = {}  # ticker -> exit_date (pd.Timestamp)
    skipped_inactive = 0
    skipped_no_picks = 0

    for date in test_dates:
        # Clean expired positions
        open_positions = {t: ed for t, ed in open_positions.items() if ed > date}

        available = strat.MAX_POSITIONS - len(open_positions)
        if available <= 0:
            continue

        # Trim data to simulate point-in-time
        trimmed = trim_data(stock_data, date)
        trimmed_n200 = {k: v for k, v in trimmed.items() if k in NIFTY_200}

        # Compute regime
        nifty_trimmed = nifty_data[nifty_data.index <= date]
        if len(nifty_trimmed) < 20:
            continue

        close = nifty_trimmed["Close"]
        sma20 = close.rolling(20).mean()
        n_above, n_total = 0, 0
        for t, df in trimmed_n200.items():
            c = df["Close"]
            if len(c) < 20:
                continue
            s = c.rolling(20).mean()
            n_total += 1
            if c.iloc[-1] > s.iloc[-1]:
                n_above += 1

        regime = {
            "nifty_close": close.iloc[-1],
            "nifty_sma": sma20.iloc[-1],
            "is_bear": close.iloc[-1] < sma20.iloc[-1],
            "breadth": n_above / n_total * 100 if n_total > 0 else 50,
            "n_above": n_above, "n_total": n_total,
        }

        if not strat.check_active(regime):
            skipped_inactive += 1
            continue

        # Compute signals using actual strategy module
        if extra_data is not None:
            trimmed_extra = nifty_trimmed
            sig = inspect.signature(strat.compute_signals)
            if len(sig.parameters) > 1:
                signals = strat.compute_signals(trimmed, trimmed_extra)
            else:
                signals = strat.compute_signals(trimmed)
        else:
            signals = strat.compute_signals(trimmed)

        if signals is None or len(signals) == 0:
            skipped_no_picks += 1
            continue

        held_tickers = set(open_positions.keys())
        picks, skipped = strat.get_picks(signals, held_tickers=held_tickers, max_new=available)
        if len(picks) == 0:
            skipped_no_picks += 1
            continue

        for _, row in picks.iterrows():
            ticker = row["ticker"]
            if ticker not in stock_data:
                continue

            result = sim_trade(stock_data[ticker], date, strat.HOLD_DAYS, strat.STOP_PCT)
            if result is None:
                continue

            result["signal_date"] = date
            result["ticker"] = ticker
            result["flag"] = row["flag"]
            result["pnl_inr"] = result["full_ret"] / 100 * per_pos
            trades.append(result)
            open_positions[ticker] = result["exit_date"]

    return pd.DataFrame(trades), skipped_inactive, skipped_no_picks


def print_strategy_report(name, tdf, per_pos, skipped_inactive, skipped_no_picks, test_days):
    """Print detailed report for one strategy."""
    if len(tdf) == 0:
        print(f"\n  ┌{'─'*68}┐")
        print(f"  │  {name:<66}│")
        print(f"  └{'─'*68}┘")
        print(f"    No trades. Inactive {skipped_inactive}/{test_days} days. No picks {skipped_no_picks} days.")
        return

    tdf = tdf.copy()
    tdf["week"] = tdf["entry_date"].dt.to_period("W").apply(lambda r: r.start_time)

    print(f"\n  ┌{'─'*68}┐")
    print(f"  │  {name:<66}│")
    print(f"  └{'─'*68}┘")

    # Weekly summary
    weeks = sorted(tdf["week"].unique())
    cum_pnl = 0
    w_pos = 0

    print(f"\n    {'Week':<12} {'Trd':>3} {'Stocks':<30} {'Flg':>4} {'D1Avg':>6} {'Full':>6} {'PnL':>9} {'Cum':>10}")
    print(f"    {'-'*85}")

    for week in weeks:
        wt = tdf[tdf["week"] == week]
        n = len(wt)
        tickers = ", ".join(wt["ticker"].tolist())
        if len(tickers) > 29:
            tickers = tickers[:26] + "..."

        cautions = (wt["flag"] == "CAUTION").sum()
        flg = f"{cautions}C" if cautions > 0 else "ok"

        d1_avg = wt["day1_ret"].mean()
        full_avg = wt["full_ret"].mean()
        week_pnl = wt["pnl_inr"].sum()
        cum_pnl += week_pnl
        if week_pnl > 0:
            w_pos += 1

        bar = ""
        if week_pnl >= 20000: bar = " ██████"
        elif week_pnl >= 10000: bar = " ████"
        elif week_pnl >= 5000: bar = " ███"
        elif week_pnl >= 0: bar = " █"
        elif week_pnl >= -10000: bar = " ░"
        else: bar = " ░░░"

        print(f"    {week.strftime('%Y-%m-%d'):<12} {n:>3} {tickers:<30} {flg:>4} {d1_avg:>+5.1f}% {full_avg:>+5.1f}% {week_pnl:>+9,.0f} {cum_pnl:>+10,.0f}{bar}")

    n_weeks = len(weeks)
    total_trades = len(tdf)
    total_pnl = tdf["pnl_inr"].sum()
    stopped = tdf["stopped"].sum()
    wr = (tdf["full_ret"] > 0).sum() / total_trades * 100
    d1_wr = (tdf["day1_ret"] > 0).sum() / total_trades * 100

    # By flag
    clean = tdf[tdf["flag"] == "CLEAN"]
    caution = tdf[tdf["flag"] == "CAUTION"]

    print(f"    {'-'*85}")
    print(f"\n    STATS:")
    print(f"    Total P&L:       {total_pnl:>+10,.0f} INR  |  Avg/week: {total_pnl/max(n_weeks,1):>+8,.0f}")
    print(f"    Trades:          {total_trades:>10}     |  Avg/week: {total_trades/max(n_weeks,1):>8.1f}")
    print(f"    Win rate:        {wr:>9.0f}%     |  Day-1 WR: {d1_wr:>7.0f}%")
    print(f"    Avg full ret:    {tdf['full_ret'].mean():>+9.2f}%    |  Avg D1:   {tdf['day1_ret'].mean():>+7.2f}%")
    print(f"    Stop-losses:     {stopped:>10} ({stopped/total_trades*100:.0f}%)")
    print(f"    Weeks profitable:{w_pos:>10}/{n_weeks} ({w_pos/max(n_weeks,1)*100:.0f}%)")
    print(f"    Active days:     {test_days - skipped_inactive:>10}/{test_days} (paused {skipped_inactive}d)")

    if len(clean) > 0 and len(caution) > 0:
        print(f"\n    BY FLAG:")
        print(f"    CLEAN   ({len(clean):>3}):  avg={clean['full_ret'].mean():+.2f}%  WR={((clean['full_ret']>0).sum()/len(clean)*100):.0f}%  PnL={clean['pnl_inr'].sum():+,.0f}")
        print(f"    CAUTION ({len(caution):>3}):  avg={caution['full_ret'].mean():+.2f}%  WR={((caution['full_ret']>0).sum()/len(caution)*100):.0f}%  PnL={caution['pnl_inr'].sum():+,.0f}")

    # Last 3 months performance
    three_months_ago = pd.Timestamp(datetime.now() - timedelta(days=90))
    recent = tdf[tdf["entry_date"] >= three_months_ago]
    if len(recent) > 0:
        print(f"\n    LAST 3 MONTHS ({len(recent)} trades):")
        print(f"    P&L: {recent['pnl_inr'].sum():+,.0f}  |  WR: {(recent['full_ret']>0).sum()/len(recent)*100:.0f}%  |  Avg: {recent['full_ret'].mean():+.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--strategy", type=int, default=0, help="1-5 for specific, 0 for all")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  PER-STRATEGY BACKTEST — Last {args.months} months")
    print(f"  Using actual strategy modules with quality filters")
    print(f"  Capital: {CAPITAL:,} INR")
    print(f"{'#'*70}")

    # Download
    days_needed = args.months * 30 + 300  # extra for rolling calculations
    print(f"\n  Downloading N500 + Nifty ({days_needed} days)...")
    stock_data_n500 = download_fresh(NIFTY_500, days=days_needed)
    stock_data_n200 = {k: v for k, v in stock_data_n500.items() if k in NIFTY_200}
    nifty_data = download_fresh([NIFTY50_TICKER], days=days_needed).get(NIFTY50_TICKER)

    if nifty_data is None:
        print("  ERROR: Could not download Nifty data")
        return

    print(f"  Got {len(stock_data_n500)} N500 stocks, {len(stock_data_n200)} N200 stocks")
    print(f"  Data range: {nifty_data.index[0].strftime('%Y-%m-%d')} to {nifty_data.index[-1].strftime('%Y-%m-%d')}")

    # Test dates: every trading day in the period, leaving room for forward returns
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=args.months * 30))
    all_dates = nifty_data.index
    test_dates = [d for d in all_dates if d >= cutoff and d <= all_dates[-20]]
    print(f"  Test dates: {len(test_dates)} trading days")

    strategies = [
        (1, strategy_bear, stock_data_n200, None),
        (2, strategy_bull, stock_data_n500, None),
        (3, strategy_bull_accel, stock_data_n200, None),
        (4, strategy_bull_relstr, stock_data_n500, nifty_data),
        (5, strategy_bull_dip, stock_data_n200, None),
    ]

    if args.strategy > 0:
        strategies = [s for s in strategies if s[0] == args.strategy]

    all_results = {}

    for idx, strat, sdata, extra in strategies:
        per_pos = CAPITAL // strat.MAX_POSITIONS
        print(f"\n  Running {strat.STRATEGY_NAME}...")
        tdf, skip_inactive, skip_no_picks = backtest_strategy(
            strat, sdata, nifty_data, stock_data_n200, extra, test_dates, per_pos
        )
        all_results[strat.STRATEGY_NAME] = tdf
        print_strategy_report(strat.STRATEGY_NAME, tdf, per_pos, skip_inactive, skip_no_picks, len(test_dates))

    # Combined summary
    if len(all_results) > 1:
        print(f"\n\n  {'='*70}")
        print(f"  COMPARISON ACROSS STRATEGIES")
        print(f"  {'='*70}")
        print(f"\n    {'Strategy':<40} {'Trades':>6} {'WR':>5} {'AvgRet':>7} {'PnL':>10} {'3mo PnL':>10}")
        print(f"    {'─'*80}")

        three_months_ago = pd.Timestamp(datetime.now() - timedelta(days=90))
        grand_total = 0
        for name, tdf in all_results.items():
            if len(tdf) == 0:
                print(f"    {name:<40} {'—':>6} {'—':>5} {'—':>7} {'—':>10} {'—':>10}")
                continue
            wr = (tdf["full_ret"] > 0).sum() / len(tdf) * 100
            pnl = tdf["pnl_inr"].sum()
            recent = tdf[tdf["entry_date"] >= three_months_ago]
            r_pnl = recent["pnl_inr"].sum() if len(recent) > 0 else 0
            grand_total += pnl
            print(f"    {name:<40} {len(tdf):>6} {wr:>4.0f}% {tdf['full_ret'].mean():>+6.2f}% {pnl:>+10,.0f} {r_pnl:>+10,.0f}")

        print(f"    {'─'*80}")
        print(f"    {'TOTAL':<40} {'':>6} {'':>5} {'':>7} {grand_total:>+10,.0f}")

    print(f"\n{'#'*70}")


if __name__ == "__main__":
    main()
