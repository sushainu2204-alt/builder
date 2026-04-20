#!/usr/bin/env python3
"""
Research: Find setups that deliver 1-2% in a single day.

This is a different beast — not weekly swing trading.
Need: High probability, high average return, per-trade.

Patterns to explore:
1. Extreme RSI(3) — raw values below 5, 10, 15
2. Consecutive down days — 3, 4, 5, 6+ red days in a row
3. Large single-day drops — stocks down 5%+ in one day
4. Gap down opens — stock opens 2%+ below prev close
5. Selling exhaustion — extreme volume on down days
6. Combinations of the above
7. Overnight holds vs full next-day holds
8. Time analysis — does day of week matter?

For each: measure next-day open-to-close and close-to-close returns.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data


def build_daily_panel(stock_data):
    """Build a panel with daily features and forward returns for 1-day analysis."""
    print(f"  Building daily panel...")
    rows = []

    for ticker, df in stock_data.items():
        if len(df) < 100:
            continue
        close = df["Close"]
        opn = df["Open"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"]

        daily_ret = close.pct_change()
        vol_avg20 = vol.rolling(20).mean()

        # RSI(3) raw
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/3, min_periods=3).mean()
        avg_loss = loss.ewm(alpha=1/3, min_periods=3).mean()
        rs = avg_gain / avg_loss
        rsi3 = 100 - (100 / (1 + rs))

        # RSI(14)
        avg_gain14 = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss14 = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs14 = avg_gain14 / avg_loss14
        rsi14 = 100 - (100 / (1 + rs14))

        # Consecutive down days
        neg = (daily_ret < 0).astype(int)
        streak = neg.copy()
        for i in range(1, len(streak)):
            if streak.iloc[i] == 1:
                streak.iloc[i] = streak.iloc[i-1] + 1
            else:
                streak.iloc[i] = 0

        # Volume ratio
        vol_ratio = vol / vol_avg20

        # Gap: open vs prev close
        gap = opn / close.shift(1) - 1

        # Intraday range
        intraday = (high - low) / close

        # Distance from 20d high
        dd20 = close / close.rolling(20).max() - 1

        for i in range(60, len(df) - 2):  # Need i+1 and i+2 for forward returns
            fwd_close_to_close = (close.iloc[i+1] / close.iloc[i]) - 1  # Next day close
            fwd_close_to_open = (opn.iloc[i+1] / close.iloc[i]) - 1    # Overnight gap
            fwd_open_to_close = (close.iloc[i+1] / opn.iloc[i+1]) - 1  # Next day intraday
            fwd_2d = (close.iloc[i+2] / close.iloc[i]) - 1             # 2-day return

            rows.append({
                "date": df.index[i],
                "ticker": ticker,
                "close": close.iloc[i],
                "open": opn.iloc[i],
                "daily_ret": daily_ret.iloc[i],
                "rsi3": rsi3.iloc[i],
                "rsi14": rsi14.iloc[i],
                "consec_down": streak.iloc[i],
                "vol_ratio": vol_ratio.iloc[i] if pd.notna(vol_ratio.iloc[i]) else 1,
                "gap": gap.iloc[i],
                "intraday_range": intraday.iloc[i],
                "dd20": dd20.iloc[i],
                "day_of_week": df.index[i].dayofweek,
                # Forward returns
                "fwd_c2c": fwd_close_to_close,     # Buy at close, sell next close
                "fwd_overnight": fwd_close_to_open,  # Buy at close, sell next open
                "fwd_intraday": fwd_open_to_close,   # Buy at open, sell same day close
                "fwd_2d": fwd_2d,                     # 2-day hold
            })

    panel = pd.DataFrame(rows)
    print(f"  Panel: {len(panel):,} observations")
    return panel


def analyze_pattern(panel, mask, label, min_n=200):
    """Analyze a pattern's forward returns."""
    subset = panel[mask]
    if len(subset) < min_n:
        return None

    results = {}
    for ret_col, ret_label in [
        ("fwd_c2c", "Close→Close (1d)"),
        ("fwd_overnight", "Close→Open (overnight)"),
        ("fwd_intraday", "Open→Close (next day)"),
        ("fwd_2d", "Close→Close (2d)"),
    ]:
        vals = subset[ret_col].dropna()
        if len(vals) < min_n:
            continue
        avg = vals.mean() * 100
        med = vals.median() * 100
        wr = (vals > 0).sum() / len(vals) * 100
        t, p = stats.ttest_1samp(vals, 0)
        pct_above_1 = (vals > 0.01).sum() / len(vals) * 100
        pct_above_2 = (vals > 0.02).sum() / len(vals) * 100

        results[ret_col] = {
            "label": ret_label,
            "n": len(vals),
            "avg": avg,
            "median": med,
            "wr": wr,
            "t_stat": t,
            "pct_above_1pct": pct_above_1,
            "pct_above_2pct": pct_above_2,
        }

    return {"label": label, "n": len(subset), "results": results}


def main():
    print(f"\n{'#'*70}")
    print(f"  1-DAY RETURN RESEARCH")
    print(f"  Finding setups that deliver 1-2% in a single day")
    print(f"{'#'*70}")

    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    panel = build_daily_panel(stock_data)

    # ============================================================
    # 1. SCAN ALL PATTERNS
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PATTERN SCAN — Next-day returns")
    print(f"{'='*70}")

    patterns = [
        # RSI extremes
        ("RSI3 < 5", panel["rsi3"] < 5),
        ("RSI3 < 10", panel["rsi3"] < 10),
        ("RSI3 < 15", panel["rsi3"] < 15),
        ("RSI3 < 20", panel["rsi3"] < 20),
        ("RSI3 5-10", (panel["rsi3"] >= 5) & (panel["rsi3"] < 10)),
        ("RSI14 < 20", panel["rsi14"] < 20),
        ("RSI14 < 30", panel["rsi14"] < 30),

        # Consecutive down days
        ("3 consec down", panel["consec_down"] == 3),
        ("4 consec down", panel["consec_down"] == 4),
        ("5 consec down", panel["consec_down"] == 5),
        ("6+ consec down", panel["consec_down"] >= 6),
        ("4+ consec down", panel["consec_down"] >= 4),
        ("5+ consec down", panel["consec_down"] >= 5),

        # Single-day drops
        ("Dropped 3%+ today", panel["daily_ret"] < -0.03),
        ("Dropped 5%+ today", panel["daily_ret"] < -0.05),
        ("Dropped 7%+ today", panel["daily_ret"] < -0.07),

        # Gap downs
        ("Gapped down 1%+", panel["gap"] < -0.01),
        ("Gapped down 2%+", panel["gap"] < -0.02),
        ("Gapped down 3%+", panel["gap"] < -0.03),

        # Volume exhaustion
        ("Vol > 3x avg + down day", (panel["vol_ratio"] > 3) & (panel["daily_ret"] < 0)),
        ("Vol > 4x avg + down day", (panel["vol_ratio"] > 4) & (panel["daily_ret"] < 0)),
        ("Vol > 3x + drop 3%+", (panel["vol_ratio"] > 3) & (panel["daily_ret"] < -0.03)),

        # Drawdown
        ("Down 10%+ from 20d high", panel["dd20"] < -0.10),
        ("Down 15%+ from 20d high", panel["dd20"] < -0.15),
        ("Down 20%+ from 20d high", panel["dd20"] < -0.20),

        # Combos
        ("RSI3<10 + 3+ down days", (panel["rsi3"] < 10) & (panel["consec_down"] >= 3)),
        ("RSI3<10 + dropped 3%+", (panel["rsi3"] < 10) & (panel["daily_ret"] < -0.03)),
        ("RSI3<15 + 4+ down days", (panel["rsi3"] < 15) & (panel["consec_down"] >= 4)),
        ("RSI3<15 + vol>2x + down", (panel["rsi3"] < 15) & (panel["vol_ratio"] > 2) & (panel["daily_ret"] < 0)),
        ("5+ down days + RSI3<20", (panel["consec_down"] >= 5) & (panel["rsi3"] < 20)),
        ("Drop 5%+ + vol>3x", (panel["daily_ret"] < -0.05) & (panel["vol_ratio"] > 3)),
        ("Gap down 2%+ + RSI3<15", (panel["gap"] < -0.02) & (panel["rsi3"] < 15)),
        ("RSI3<10 + dd20>10%", (panel["rsi3"] < 10) & (panel["dd20"] < -0.10)),
        ("RSI3<5 + any consec down", (panel["rsi3"] < 5) & (panel["consec_down"] >= 1)),
        ("4+ down + drop 3%+ today", (panel["consec_down"] >= 4) & (panel["daily_ret"] < -0.03)),
        ("3+ down + vol>2x + RSI<15", (panel["consec_down"] >= 3) & (panel["vol_ratio"] > 2) & (panel["rsi3"] < 15)),
    ]

    # Analyze and collect
    all_results = []
    for label, mask in patterns:
        r = analyze_pattern(panel, mask, label, min_n=100)
        if r and "fwd_c2c" in r["results"]:
            all_results.append(r)

    # Sort by next-day close-to-close average return
    all_results.sort(key=lambda x: x["results"]["fwd_c2c"]["avg"], reverse=True)

    print(f"\n  {'Pattern':<35} {'N':>6} {'AvgRet':>7} {'Median':>7} {'WR':>5} {'t':>6} {'>1%':>5} {'>2%':>5}")
    print(f"  {'-'*80}")
    for r in all_results:
        c = r["results"]["fwd_c2c"]
        marker = " <<" if c["avg"] >= 0.8 and c["t_stat"] > 3 and c["wr"] > 52 else ""
        print(f"  {r['label']:<35} {c['n']:>6} {c['avg']:>+6.2f}% {c['median']:>+6.2f}% {c['wr']:>4.0f}% {c['t_stat']:>6.2f} {c['pct_above_1pct']:>4.0f}% {c['pct_above_2pct']:>4.0f}%{marker}")

    # ============================================================
    # 2. DEEP DIVE ON TOP PATTERNS
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  DEEP DIVE — Top patterns with all return types")
    print(f"{'='*70}")

    top = [r for r in all_results if r["results"]["fwd_c2c"]["avg"] >= 0.5 and r["results"]["fwd_c2c"]["t_stat"] > 2]
    top.sort(key=lambda x: x["results"]["fwd_c2c"]["avg"], reverse=True)

    for r in top[:10]:
        print(f"\n  {r['label']} (N={r['n']:,})")
        print(f"  {'Return Type':<25} {'Avg':>7} {'Med':>7} {'WR':>5} {'t':>6} {'>1%':>5} {'>2%':>5}")
        for ret_key in ["fwd_c2c", "fwd_overnight", "fwd_intraday", "fwd_2d"]:
            if ret_key in r["results"]:
                c = r["results"][ret_key]
                print(f"  {c['label']:<25} {c['avg']:>+6.2f}% {c['median']:>+6.2f}% {c['wr']:>4.0f}% {c['t_stat']:>6.2f} {c['pct_above_1pct']:>4.0f}% {c['pct_above_2pct']:>4.0f}%")

    # ============================================================
    # 3. WALK-FORWARD: 2025+ OUT OF SAMPLE
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  WALK-FORWARD: 2025+ Out-of-Sample")
    print(f"{'='*70}")

    panel_2025 = panel[panel["date"] >= pd.Timestamp("2025-01-01")]
    panel_pre = panel[panel["date"] < pd.Timestamp("2025-01-01")]

    # Re-run top patterns on 2025 data
    top_patterns_to_test = [p for p in patterns if any(
        r["label"] == p[0] and r["results"]["fwd_c2c"]["avg"] >= 0.5
        for r in all_results if "fwd_c2c" in r["results"]
    )]

    print(f"\n  {'Pattern':<35} {'Pre25 Avg':>9} {'Pre25 WR':>8} {'2025 N':>6} {'2025 Avg':>8} {'2025 WR':>7} {'2025 t':>6}")
    print(f"  {'-'*85}")

    for label, mask in top_patterns_to_test:
        mask_2025 = panel["date"] >= pd.Timestamp("2025-01-01")
        mask_pre = panel["date"] < pd.Timestamp("2025-01-01")

        sub_pre = panel[mask & mask_pre]
        sub_2025 = panel[mask & mask_2025]

        if len(sub_pre) < 50 or len(sub_2025) < 20:
            continue

        pre_avg = sub_pre["fwd_c2c"].mean() * 100
        pre_wr = (sub_pre["fwd_c2c"] > 0).sum() / len(sub_pre) * 100
        t25_avg = sub_2025["fwd_c2c"].mean() * 100
        t25_wr = (sub_2025["fwd_c2c"] > 0).sum() / len(sub_2025) * 100
        t25_t = stats.ttest_1samp(sub_2025["fwd_c2c"].dropna(), 0)[0] if len(sub_2025) > 5 else 0

        marker = " <<" if t25_avg >= 0.5 and t25_t > 1.5 else ""
        print(f"  {label:<35} {pre_avg:>+8.2f}% {pre_wr:>7.0f}% {len(sub_2025):>6} {t25_avg:>+7.2f}% {t25_wr:>6.0f}% {t25_t:>6.2f}{marker}")

    # ============================================================
    # 4. SIMULATE: How much would this make per week?
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  SIMULATION: Weekly P&L with 1-day hold strategies")
    print(f"{'='*70}")

    # Pick the best pattern that works in 2025
    best_patterns = [
        ("RSI3<10 + 3+ down days", (panel["rsi3"] < 10) & (panel["consec_down"] >= 3)),
        ("RSI3<10 + dropped 3%+", (panel["rsi3"] < 10) & (panel["daily_ret"] < -0.03)),
        ("RSI3 < 5", panel["rsi3"] < 5),
        ("5+ down days + RSI3<20", (panel["consec_down"] >= 5) & (panel["rsi3"] < 20)),
        ("RSI3<15 + 4+ down days", (panel["rsi3"] < 15) & (panel["consec_down"] >= 4)),
        ("Drop 5%+ + vol>3x", (panel["daily_ret"] < -0.05) & (panel["vol_ratio"] > 3)),
        ("4+ down + drop 3%+ today", (panel["consec_down"] >= 4) & (panel["daily_ret"] < -0.03)),
    ]

    for label, mask in best_patterns:
        trades = panel[mask & (panel["date"] >= pd.Timestamp("2025-01-01"))].copy()
        if len(trades) < 10:
            continue

        trades = trades.sort_values("date")

        # Simulate: buy at next day open, sell next day close
        # Max 3 positions per day, 1.2L per position
        capital = 600000
        per_pos = 120000
        max_pos = 3

        trades["pnl"] = trades["fwd_c2c"] * per_pos  # Approximate
        trades["week"] = trades["date"].dt.to_period("W")

        weekly = trades.groupby("week").agg(
            n_trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_ret=("fwd_c2c", "mean"),
            wr=("fwd_c2c", lambda x: (x > 0).sum() / len(x) * 100),
        )

        # Cap at max_pos per day
        capped_weekly = trades.groupby("date").head(max_pos).groupby(
            trades.groupby("date").head(max_pos)["date"].dt.to_period("W")
        ).agg(
            n_trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_ret=("fwd_c2c", "mean"),
        )

        avg_wk = capped_weekly["total_pnl"].mean()
        weeks_active = len(capped_weekly)
        total_pnl = capped_weekly["total_pnl"].sum()
        total_trades = capped_weekly["n_trades"].sum()

        print(f"\n  {label}")
        print(f"    Trades in 2025+: {len(trades)} | Weeks active: {weeks_active}")
        print(f"    Avg next-day return: {trades['fwd_c2c'].mean()*100:+.2f}% | WR: {(trades['fwd_c2c']>0).sum()/len(trades)*100:.0f}%")
        print(f"    Avg weekly P&L (at 1.2L/pos, max 3/day): {avg_wk:+,.0f} INR")
        print(f"    Total P&L: {total_pnl:+,.0f} INR")

        # Show week-by-week
        if weeks_active <= 30:
            print(f"    {'Week':<12} {'Trades':>6} {'P&L':>10} {'Avg Ret':>8}")
            for period, row in capped_weekly.iterrows():
                marker = " +" if row["total_pnl"] >= 5000 else (" *" if row["total_pnl"] < 0 else "")
                print(f"    {str(period):<12} {row['n_trades']:>6} {row['total_pnl']:>+10,.0f} {row['avg_ret']*100:>+7.2f}%{marker}")


if __name__ == "__main__":
    main()
