#!/usr/bin/env python3
"""
Deep dive into the gap-down pattern.
Stock gaps down 2%+ → buy at close → sell next day.

Questions:
1. Does the gap size matter? (2% vs 3% vs 5% gap)
2. Does RSI/oversold + gap work better?
3. Does market regime matter for gaps?
4. How many positions per day to maximize P&L?
5. Exact P&L simulation with realistic costs
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, zscore_normalize
from research_1day import build_daily_panel


def main():
    print(f"\n{'#'*70}")
    print(f"  GAP-DOWN STRATEGY — Deep Dive")
    print(f"{'#'*70}")

    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    panel = build_daily_panel(stock_data)

    # Add gap column (open vs prev close)
    # Already in panel as 'gap'

    # ============================================================
    # 1. GAP SIZE ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  GAP SIZE vs NEXT-DAY RETURNS")
    print(f"{'='*70}")

    gap_ranges = [
        ("Gap -1% to -2%", (-0.02, -0.01)),
        ("Gap -2% to -3%", (-0.03, -0.02)),
        ("Gap -3% to -4%", (-0.04, -0.03)),
        ("Gap -4% to -5%", (-0.05, -0.04)),
        ("Gap -5% to -7%", (-0.07, -0.05)),
        ("Gap -7%+", (-1.0, -0.07)),
    ]

    print(f"\n  {'Gap Range':<20} {'N':>6} {'C2C Avg':>8} {'C2C WR':>7} {'O/N Avg':>8} {'O/N WR':>7} {'t':>6}")
    for label, (lo, hi) in gap_ranges:
        mask = (panel["gap"] >= lo) & (panel["gap"] < hi)
        sub = panel[mask]
        if len(sub) < 50:
            continue
        c2c = sub["fwd_c2c"].dropna()
        ovn = sub["fwd_overnight"].dropna()
        t, _ = stats.ttest_1samp(c2c, 0)
        print(f"  {label:<20} {len(sub):>6} {c2c.mean()*100:>+7.2f}% {(c2c>0).sum()/len(c2c)*100:>6.0f}% "
              f"{ovn.mean()*100:>+7.2f}% {(ovn>0).sum()/len(ovn)*100:>6.0f}% {t:>6.2f}")

    # ============================================================
    # 2. GAP + OVERSOLD COMBOS
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  GAP-DOWN + OVERSOLD COMBOS")
    print(f"{'='*70}")

    combos = [
        ("Gap 2%+ only", panel["gap"] < -0.02),
        ("Gap 2%+ + RSI3<20", (panel["gap"] < -0.02) & (panel["rsi3"] < 20)),
        ("Gap 2%+ + RSI3<15", (panel["gap"] < -0.02) & (panel["rsi3"] < 15)),
        ("Gap 2%+ + RSI3<10", (panel["gap"] < -0.02) & (panel["rsi3"] < 10)),
        ("Gap 2%+ + 2+ down days", (panel["gap"] < -0.02) & (panel["consec_down"] >= 2)),
        ("Gap 2%+ + 3+ down days", (panel["gap"] < -0.02) & (panel["consec_down"] >= 3)),
        ("Gap 3%+ + RSI3<20", (panel["gap"] < -0.03) & (panel["rsi3"] < 20)),
        ("Gap 3%+ + RSI3<15", (panel["gap"] < -0.03) & (panel["rsi3"] < 15)),
        ("Gap 3%+ + 2+ down", (panel["gap"] < -0.03) & (panel["consec_down"] >= 2)),
        ("Gap 2%+ + dd20>10%", (panel["gap"] < -0.02) & (panel["dd20"] < -0.10)),
        ("Gap 2%+ + dd20>15%", (panel["gap"] < -0.02) & (panel["dd20"] < -0.15)),
        ("Gap 3%+ + dd20>10%", (panel["gap"] < -0.03) & (panel["dd20"] < -0.10)),
        ("Gap 2%+ + vol>2x", (panel["gap"] < -0.02) & (panel["vol_ratio"] > 2)),
    ]

    print(f"\n  {'Combo':<30} {'N':>6} {'C2C Avg':>8} {'Med':>7} {'WR':>5} {'t':>6} {'>1%':>5} {'>2%':>5}")
    for label, mask in combos:
        sub = panel[mask]
        if len(sub) < 50:
            continue
        c2c = sub["fwd_c2c"].dropna()
        t, _ = stats.ttest_1samp(c2c, 0)
        pct1 = (c2c > 0.01).sum() / len(c2c) * 100
        pct2 = (c2c > 0.02).sum() / len(c2c) * 100
        marker = " <<" if c2c.mean() > 0.008 and t > 3 else ""
        print(f"  {label:<30} {len(sub):>6} {c2c.mean()*100:>+7.2f}% {c2c.median()*100:>+6.2f}% "
              f"{(c2c>0).sum()/len(c2c)*100:>4.0f}% {t:>6.2f} {pct1:>4.0f}% {pct2:>4.0f}%{marker}")

    # ============================================================
    # 3. REGIME ANALYSIS
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  GAP-DOWN BY MARKET REGIME")
    print(f"{'='*70}")

    nifty = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    nifty_close = nifty["Close"]
    nifty_sma = nifty_close.rolling(20).mean()
    nifty_regime = (nifty_close > nifty_sma).astype(int)

    regime_map = nifty_regime.to_dict()
    panel["regime"] = panel["date"].apply(lambda d: regime_map.get(d, regime_map.get(d - pd.Timedelta(days=1), None)))

    gap_mask = panel["gap"] < -0.02

    for regime_label, regime_val in [("BULL (Nifty > SMA)", 1), ("BEAR (Nifty < SMA)", 0)]:
        sub = panel[gap_mask & (panel["regime"] == regime_val)]
        if len(sub) < 50:
            continue
        c2c = sub["fwd_c2c"].dropna()
        t, _ = stats.ttest_1samp(c2c, 0)
        print(f"\n  {regime_label}: N={len(sub):,}")
        print(f"    Close→Close: {c2c.mean()*100:+.2f}% | WR: {(c2c>0).sum()/len(c2c)*100:.0f}% | t: {t:.2f}")
        print(f"    >1%: {(c2c>0.01).sum()/len(c2c)*100:.0f}% | >2%: {(c2c>0.02).sum()/len(c2c)*100:.0f}%")

    # ============================================================
    # 4. FULL SIMULATION — 2025+
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  FULL SIMULATION: Gap-Down Strategy 2025+")
    print(f"  Buy at close of gap-down day, sell next day close")
    print(f"{'='*70}")

    # Best combos to simulate
    sim_configs = [
        ("Gap 2%+ (simple)", panel["gap"] < -0.02, 3),
        ("Gap 2%+ + RSI3<20", (panel["gap"] < -0.02) & (panel["rsi3"] < 20), 3),
        ("Gap 2%+ + dd20>10%", (panel["gap"] < -0.02) & (panel["dd20"] < -0.10), 3),
        ("Gap 3%+ (simple)", panel["gap"] < -0.03, 3),
        ("Gap 3%+ + RSI3<20", (panel["gap"] < -0.03) & (panel["rsi3"] < 20), 3),
    ]

    for label, mask, max_pos in sim_configs:
        trades = panel[mask & (panel["date"] >= pd.Timestamp("2025-01-01"))].copy()
        if len(trades) < 10:
            continue

        trades = trades.sort_values(["date", "gap"])  # Sort by biggest gap first

        CAPITAL = 600000
        PER_POS = CAPITAL // 5
        COST = 0.002  # 0.2% round trip (higher for overnight)

        # Cap positions per day
        daily_trades = []
        for date, group in trades.groupby("date"):
            day_picks = group.head(max_pos)  # Biggest gaps first
            for _, row in day_picks.iterrows():
                pnl_gross = row["fwd_c2c"]
                pnl_net = pnl_gross - COST
                pnl_inr = pnl_net * PER_POS
                daily_trades.append({
                    "date": date,
                    "ticker": row["ticker"],
                    "close": row["close"],
                    "gap": row["gap"] * 100,
                    "rsi3": row["rsi3"],
                    "fwd_ret": row["fwd_c2c"] * 100,
                    "pnl_net": pnl_net * 100,
                    "pnl_inr": pnl_inr,
                })

        tdf = pd.DataFrame(daily_trades)
        tdf["week"] = tdf["date"].dt.to_period("W")

        weekly = tdf.groupby("week").agg(
            n=("pnl_inr", "count"),
            pnl=("pnl_inr", "sum"),
            avg_ret=("fwd_ret", "mean"),
            wr=("fwd_ret", lambda x: (x > 0).sum() / len(x) * 100),
        )

        total_pnl = weekly["pnl"].sum()
        weeks_active = len(weekly)
        avg_wk = weekly["pnl"].mean()
        weeks_pos = (weekly["pnl"] > 0).sum()
        total_trades_n = tdf["pnl_inr"].count()
        overall_wr = (tdf["fwd_ret"] > 0).sum() / len(tdf) * 100

        print(f"\n  {label} (max {max_pos}/day, {PER_POS/1000:.0f}K/pos, 0.2% cost)")
        print(f"    Trades: {total_trades_n} | Weeks active: {weeks_active}/{52}")
        print(f"    WR: {overall_wr:.0f}% | Avg return/trade: {tdf['fwd_ret'].mean():+.2f}%")
        print(f"    Total P&L: {total_pnl:+,.0f} | Avg/week: {avg_wk:+,.0f}")
        print(f"    Weeks profitable: {weeks_pos}/{weeks_active} ({weeks_pos/weeks_active*100:.0f}%)")

        # Week-by-week
        if weeks_active <= 20:
            print(f"    {'Week':<12} {'Trades':>6} {'P&L':>10} {'Avg Ret':>8} {'WR':>5}")
            for period, row in weekly.iterrows():
                m = " +" if row["pnl"] > 3000 else (" *" if row["pnl"] < 0 else "")
                print(f"    {str(period):<12} {row['n']:>6} {row['pnl']:>+10,.0f} {row['avg_ret']:>+7.2f}% {row['wr']:>4.0f}%{m}")
        else:
            # Show monthly
            tdf["month"] = tdf["date"].dt.to_period("M")
            monthly = tdf.groupby("month").agg(
                n=("pnl_inr", "count"),
                pnl=("pnl_inr", "sum"),
                avg_ret=("fwd_ret", "mean"),
            )
            print(f"    Monthly:")
            for period, row in monthly.iterrows():
                m = " +" if row["pnl"] > 5000 else (" *" if row["pnl"] < 0 else "")
                print(f"    {str(period):<10} {row['n']:>4} trades  {row['pnl']:>+10,.0f}  avg={row['avg_ret']:>+.2f}%{m}")


if __name__ == "__main__":
    main()
