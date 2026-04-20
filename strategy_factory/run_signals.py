#!/usr/bin/env python3
"""
Master Signal Runner — Runs ALL strategies, shows each, then picks finalists.

Strategies:
  1. Bear Dip (RSI3 + breadth<40%) — N200, hold 5d, 5 pos
  2. Bull Breakout + Accumulation — N500, hold 15d, 5 pos
  3. Bull Acceleration + Breadth>60% — N200, hold 15d, 3 pos
  4. Bull Rel Strength + Vol Breakout — N500, hold 15d, 3 pos
  5. Bull Dip Buy + Breadth>60% — N200, hold 10d, 3 pos

Usage:
    python run_signals.py
    python run_signals.py --date 2026-04-13
    python run_signals.py --capital 300000
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, NIFTY_500, NIFTY50_TICKER
import strategy_bear
import strategy_bull
import strategy_bull_accel
import strategy_bull_relstr
import strategy_bull_dip

NSE_HOLIDAYS_2026 = {
    "2026-01-26", "2026-02-17", "2026-03-10", "2026-03-26", "2026-03-31",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-06-07", "2026-07-07",
    "2026-08-15", "2026-08-18", "2026-09-05", "2026-10-02", "2026-10-20",
    "2026-11-09", "2026-11-10", "2026-11-30", "2026-12-25",
}


def is_trading_day(date_str):
    d = pd.Timestamp(date_str)
    return d.weekday() < 5 and date_str not in NSE_HOLIDAYS_2026


def get_exit_date(entry_date, hold_days):
    days = [pd.Timestamp(entry_date)]
    d = pd.Timestamp(entry_date) + timedelta(days=1)
    while len(days) < hold_days:
        if is_trading_day(d.strftime("%Y-%m-%d")):
            days.append(d)
        d += timedelta(days=1)
    return days[-1]


def download_fresh(tickers, days=400):
    # Use tomorrow as end date to force yfinance to include today
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


def compute_regime(nifty_data, stock_data_n200):
    close = nifty_data["Close"]
    sma20 = close.rolling(20).mean()
    n_above, n_total = 0, 0
    for _, df in stock_data_n200.items():
        c = df["Close"]
        s = c.rolling(20).mean()
        if len(c) >= 20:
            n_total += 1
            if c.iloc[-1] > s.iloc[-1]:
                n_above += 1
    return {
        "nifty_close": close.iloc[-1],
        "nifty_sma": sma20.iloc[-1],
        "is_bear": close.iloc[-1] < sma20.iloc[-1],
        "breadth": n_above / n_total * 100 if n_total > 0 else 50,
        "n_above": n_above, "n_total": n_total,
    }


def print_strategy_picks(picks, skipped, strategy_mod, per_pos, entry_date, exit_date):
    """Print picks for one strategy with full trade details."""
    if len(skipped) > 0:
        print(f"\n    Filtered out:")
        for _, row in skipped.iterrows():
            print(f"      {row['ticker']:<12} — {row['flag_reasons']}")

    if len(picks) == 0:
        print(f"    No qualifying stocks after filters.")
        return

    print(f"\n    {'#':>3} {'Ticker':<12} {'CMP':>8} {'Qty':>6} {'Amount':>9} {'Stop':>8} {'Exit Date':<12} {'Flag':<8}")
    print(f"    {'-'*75}")

    for i, (_, row) in enumerate(picks.iterrows(), 1):
        shares = int(per_pos / row["close"])
        amount = shares * row["close"]
        stop = round(row["close"] * (1 - strategy_mod.STOP_PCT), 1)

        print(f"    {i:>3} {row['ticker']:<12} {row['close']:>8.1f} {shares:>6} {amount:>9,.0f} {stop:>8.1f} {exit_date:<12} {row['flag']:<8}")
        if row["flag"] == "CAUTION":
            print(f"        ^^ {row['flag_reasons']}")

    print(f"\n    Entry: {entry_date} open | Hold: {strategy_mod.HOLD_DAYS} trading days | Stop: -{strategy_mod.STOP_PCT*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Master Signal Runner — All Strategies")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--capital", type=int, default=600000)
    args = parser.parse_args()

    CAPITAL = args.capital
    today = datetime.now().strftime("%Y-%m-%d")
    PER_POS = CAPITAL // 5

    print(f"\n{'#'*70}")
    print(f"  SIGNAL RUNNER — All Strategies")
    print(f"  {today} ({datetime.now().strftime('%A')}) | Capital: {CAPITAL:,} INR")
    print(f"{'#'*70}")

    upcoming = [h for h in sorted(NSE_HOLIDAYS_2026)
                if today <= h <= (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")]
    if upcoming:
        print(f"  Upcoming holidays: {', '.join(upcoming)}")

    # Download
    print(f"\n  Downloading Nifty 500 + index...")
    stock_data_n500 = download_fresh(NIFTY_500)
    stock_data_n200 = {k: v for k, v in stock_data_n500.items() if k in NIFTY_200}
    nifty_data = download_fresh([NIFTY50_TICKER]).get(NIFTY50_TICKER)

    if nifty_data is None or nifty_data.empty:
        print("  ERROR: Could not download Nifty data")
        return

    # Check data date from a major stock (more reliable than index)
    check_stock = stock_data_n200.get("RELIANCE", stock_data_n200.get("TCS", nifty_data))
    data_date = check_stock.index[-1].strftime("%Y-%m-%d")
    nifty_date = nifty_data.index[-1].strftime("%Y-%m-%d")
    print(f"  Data as of: {data_date} (Nifty: {nifty_date})")
    if args.date and data_date != args.date:
        print(f"  ERROR: Expected {args.date} but got {data_date}. STALE.")
        return

    # Regime
    regime = compute_regime(nifty_data, stock_data_n200)
    print(f"\n  {'='*60}")
    print(f"  MARKET REGIME")
    print(f"  Nifty: {regime['nifty_close']:.0f} | 20d SMA: {regime['nifty_sma']:.0f} | "
          f"{'BEAR' if regime['is_bear'] else 'BULL'}")
    print(f"  Breadth: {regime['n_above']}/{regime['n_total']} = {regime['breadth']:.1f}%")
    print(f"  {'='*60}")

    # Entry date
    entry_d = datetime.now() + timedelta(days=1)
    while not is_trading_day(entry_d.strftime("%Y-%m-%d")):
        entry_d += timedelta(days=1)
    entry_date = entry_d.strftime("%Y-%m-%d")

    # ============================================================
    # RUN ALL STRATEGIES
    # ============================================================
    all_strategies = [
        (strategy_bear, stock_data_n200, None, "N200"),
        (strategy_bull, stock_data_n500, None, "N500"),
        (strategy_bull_accel, stock_data_n200, None, "N200"),
        (strategy_bull_relstr, stock_data_n500, nifty_data, "N500"),
        (strategy_bull_dip, stock_data_n200, None, "N200"),
    ]

    strategy_results = []  # (module, picks, exit_date, active)

    for idx, (strat, sdata, extra, universe) in enumerate(all_strategies, 1):
        print(f"\n  {'─'*60}")
        print(f"  STRATEGY {idx}: {strat.STRATEGY_NAME}")

        active = strat.check_active(regime)

        if not active:
            reasons = []
            if regime["is_bear"] and "Bull" in strat.STRATEGY_NAME:
                reasons.append(f"Nifty below SMA")
            if not regime["is_bear"] and "Bear" in strat.STRATEGY_NAME:
                reasons.append(f"Nifty above SMA")
            if hasattr(strat, 'check_active'):
                if "Breadth" in strat.STRATEGY_NAME and regime["breadth"] <= 60:
                    reasons.append(f"Breadth {regime['breadth']:.0f}% ≤ 60%")
                if "Breadth" in strat.STRATEGY_NAME and regime["breadth"] >= 40:
                    reasons.append(f"Breadth {regime['breadth']:.0f}% ≥ 40%")
            print(f"  Status: PAUSED — {'; '.join(reasons) if reasons else 'conditions not met'}")
            strategy_results.append((strat, pd.DataFrame(), None, False))
            continue

        print(f"  Status: ACTIVE | Universe: {universe} | Hold: {strat.HOLD_DAYS}d | Max: {strat.MAX_POSITIONS}")

        # Compute signals — handle extra args for relstr
        if extra is not None and hasattr(strat, 'compute_signals'):
            import inspect
            sig = inspect.signature(strat.compute_signals)
            if len(sig.parameters) > 1:
                signals = strat.compute_signals(sdata, extra)
            else:
                signals = strat.compute_signals(sdata)
        else:
            signals = strat.compute_signals(sdata)

        picks, skipped = strat.get_picks(signals, max_new=strat.MAX_POSITIONS)
        exit_date = get_exit_date(entry_date, strat.HOLD_DAYS).strftime("%Y-%m-%d")

        print_strategy_picks(picks, skipped, strat, PER_POS, entry_date, exit_date)
        strategy_results.append((strat, picks, exit_date, True))

    # ============================================================
    # CONSOLIDATE ALL PICKS
    # ============================================================
    print(f"\n\n  {'='*70}")
    print(f"  CONSOLIDATION — All Signals Across Strategies")
    print(f"  {'='*70}")

    all_picks = []
    for strat, picks, exit_date, active in strategy_results:
        if not active or len(picks) == 0:
            continue
        for _, row in picks.iterrows():
            all_picks.append({
                "ticker": row["ticker"],
                "close": row["close"],
                "strategy": strat.STRATEGY_NAME,
                "strategy_short": strat.STRATEGY_NAME.split("(")[0].strip().replace("Bull ", "").replace("Bear ", ""),
                "hold": strat.HOLD_DAYS,
                "exit_date": exit_date,
                "signal_value": row.get("signal_value", 0),
                "flag": row["flag"],
                "flag_reasons": row["flag_reasons"],
                "ret5d": row.get("ret5d", 0),
            })

    if not all_picks:
        print(f"\n  No signals from any strategy. Sit in cash.")
        print(f"\n{'#'*70}")
        return

    adf = pd.DataFrame(all_picks)

    # Count how many strategies picked each stock
    ticker_counts = adf.groupby("ticker").agg(
        n_strategies=("strategy", "count"),
        strategies=("strategy_short", lambda x: " + ".join(sorted(set(x)))),
        close=("close", "first"),
        flag=("flag", lambda x: "SKIP" if "SKIP" in x.values else ("CAUTION" if "CAUTION" in x.values else "CLEAN")),
        flag_reasons=("flag_reasons", lambda x: "; ".join(set(r for r in x if r != "No red flags")) or "No red flags"),
        min_hold=("hold", "min"),
    ).sort_values(["n_strategies", "close"], ascending=[False, True])

    multi = ticker_counts[ticker_counts["n_strategies"] >= 2]
    single = ticker_counts[ticker_counts["n_strategies"] == 1]

    if len(multi) > 0:
        print(f"\n  MULTI-STRATEGY (2+ strategies agree):")
        for ticker, row in multi.iterrows():
            print(f"    {ticker:<12} {row['close']:>8.1f}  [{row['n_strategies']} strategies: {row['strategies']}]  {row['flag']}")
            if row["flag"] != "CLEAN":
                print(f"                            ^^ {row['flag_reasons']}")

    if len(single) > 0:
        print(f"\n  SINGLE-STRATEGY:")
        for ticker, row in single.iterrows():
            print(f"    {ticker:<12} {row['close']:>8.1f}  [{row['strategies']}]  {row['flag']}")
            if row["flag"] != "CLEAN":
                print(f"                            ^^ {row['flag_reasons']}")

    # ============================================================
    # FINAL SIGNAL — Grouped by strategy
    # ============================================================
    print(f"\n\n  {'*'*70}")
    print(f"  FINAL SIGNALS — By Strategy")
    print(f"  {'*'*70}")

    # Build hypothesis map
    def get_hypothesis(strat_name):
        if "Breakout" in strat_name and "Accumulation" in strat_name:
            return "Stock at 20d high with rising volume — institutional buying, trend likely continues"
        elif "Acceleration" in strat_name:
            return "Recent momentum accelerating (20d outpacing 60d) in strong market — trend strengthening"
        elif "Rel Strength" in strat_name:
            return "Outperforming Nifty with volume breakout — relative winner, money rotating in"
        elif "Dip Buy" in strat_name:
            return "Biggest 5-day dipper in a strong bull market — temporary weakness, expect snap-back"
        elif "Bear" in strat_name or "RSI" in strat_name:
            return "Deeply oversold in a weak market — selling exhaustion, bounce expected"
        return "Signal-driven pick"

    # Multi-strategy tickers for tagging
    multi_tickers = set(multi.index)

    total_deployed = 0
    total_positions = 0
    global_pick = 0

    for strat, picks, exit_date, active in strategy_results:
        if not active or len(picks) == 0:
            continue

        print(f"\n  ┌{'─'*68}┐")
        print(f"  │  {strat.STRATEGY_NAME:<52} Hold: {strat.HOLD_DAYS}d  │")
        print(f"  └{'─'*68}┘")
        hyp = get_hypothesis(strat.STRATEGY_NAME)
        print(f"    Hypothesis: {hyp}")
        print()
        print(f"    {'#':>3} {'Ticker':<12} {'CMP':>8} {'Qty':>6} {'Amount':>9} {'Stop':>8} {'Exit':<12} {'Conv':<8}")
        print(f"    {'─'*75}")

        for _, row in picks.iterrows():
            global_pick += 1
            shares = int(PER_POS / row["close"])
            amount = shares * row["close"]
            stop = round(row["close"] * (1 - strat.STOP_PCT), 1)
            conv = "HIGH" if row["ticker"] in multi_tickers else "MEDIUM"
            flag_str = ""
            if row["flag"] == "CAUTION":
                conv += "*"
                flag_str = f"  ⚠ {row['flag_reasons']}"

            print(f"    {global_pick:>3} {row['ticker']:<12} {row['close']:>8.1f} {shares:>6} {amount:>9,.0f} {stop:>8.1f} {exit_date:<12} {conv:<8}{flag_str}")
            total_deployed += amount
            total_positions += 1

        print(f"    Entry: {entry_date} open | Stop: -{strat.STOP_PCT*100:.0f}%")

    # Summary
    print(f"\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")

    # Show multi-strategy overlaps
    if len(multi) > 0:
        print(f"\n  HIGH conviction (2+ strategies agree):")
        for ticker, row in multi.iterrows():
            print(f"    ★ {ticker:<12} {row['close']:>8.1f}  [{row['strategies']}]")

    unique_tickers = set()
    for strat, picks, exit_date, active in strategy_results:
        if active and len(picks) > 0:
            for _, row in picks.iterrows():
                unique_tickers.add(row["ticker"])

    print(f"\n  Unique stocks: {len(unique_tickers)} | Total positions: {total_positions}")
    print(f"  Per position: {PER_POS:,} | Total if all taken: {total_deployed:,.0f} / {CAPITAL:,}")

    print(f"\n{'#'*70}")


if __name__ == "__main__":
    main()
