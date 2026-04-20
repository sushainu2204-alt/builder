#!/usr/bin/env python3
"""
Bull Strategy Research on Nifty 500.
More stocks = more trade opportunities = more reliable stats.

Top patterns from N200 research to validate on larger universe:
1. Price acceleration + breadth>60% (Sharpe 2.93 but only 60 trades)
2. Volume breakout (Sharpe 2.83, 168 trades)
3. Relative strength (Sharpe 3.05, 177 trades)
4. Breakout at 20d high (Sharpe 3.30, 186 trades)
5. Consolidation + rel strength (new combo)
6. Accumulation + breadth>60%

Also test new ideas:
7. Sector rotation: buy stocks in the hottest sector
8. Composite score: combine multiple bull signals
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_500, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel, zscore_normalize
from backtest.metrics import compute_all_metrics
from research_5k_v5 import get_nifty_regime, run_regime_filtered_backtest
from research_deep import add_advanced_features
from research_bull_fresh import add_bull_features


def create_composite_scores(panel):
    """Create composite bull scores combining multiple signals."""
    print(f"\n  Creating composite scores...")

    # Composite 1: Trend Strength
    # Combines: relative strength + price acceleration + at 20d high
    cols1 = ["rel_str_z", "price_accel", "at_20d_high"]
    valid1 = [c for c in cols1 if c in panel.columns]
    if valid1:
        panel["trend_strength"] = panel[valid1].mean(axis=1)

    # Composite 2: Breakout Quality
    # Combines: volume breakout + accumulation + gap-up trend
    cols2 = ["vol_breakout", "accumulation", "gap_up_trend"]
    valid2 = [c for c in cols2 if c in panel.columns]
    if valid2:
        panel["breakout_quality"] = panel[valid2].mean(axis=1)

    # Composite 3: Momentum Quality
    # High relative strength + low volatility + consolidation = quality trend
    if all(c in panel.columns for c in ["rel_str_z", "consolidation"]):
        panel["momentum_quality"] = panel["rel_str_z"] + panel["consolidation"]

    # Composite 4: Full Bull Score
    # Everything combined — kitchen sink
    all_bull = ["rel_str_z", "price_accel", "accumulation", "vol_breakout",
                "gap_up_trend", "consolidation"]
    valid_all = [c for c in all_bull if c in panel.columns]
    if valid_all:
        panel["full_bull_score"] = panel[valid_all].mean(axis=1)

    print(f"  Added: trend_strength, breakout_quality, momentum_quality, full_bull_score")
    return panel


def main():
    print(f"\n{'#'*70}")
    print(f"  BULL STRATEGY — NIFTY 500 UNIVERSE")
    print(f"  More stocks, more trades, more robust signals")
    print(f"{'#'*70}")

    # Download Nifty 500
    stock_data = download_universe(NIFTY_500, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)
    nifty_regime = get_nifty_regime(index_data)

    n_stocks = panel["ticker"].nunique()
    print(f"\n  Universe: {n_stocks} stocks | Panel: {len(panel):,} rows")

    # Add all features
    panel = add_advanced_features(panel, stock_data, index_data)
    panel = add_bull_features(panel, stock_data, index_data)
    panel = create_composite_scores(panel)

    # ============================================================
    # BACKTEST ALL STRATEGIES
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  BACKTESTING ON NIFTY 500")
    print(f"{'='*70}")

    configs = [
        # Singles
        ("Price acceleration", "price_accel", "highest", None, {}),
        ("Rel strength", "rel_str_z", "highest", None, {}),
        ("Volume breakout", "vol_breakout", "highest", None, {}),
        ("Breakout 20d high", "at_20d_high", "highest", None, {}),
        ("Accumulation", "accumulation", "highest", None, {}),
        ("Consolidation", "consolidation", "highest", None, {}),
        ("Gap-up trend", "gap_up_trend", "highest", None, {}),
        ("Higher lows", "higher_lows", "highest", None, {}),

        # Composites
        ("Trend strength", "trend_strength", "highest", None, {}),
        ("Breakout quality", "breakout_quality", "highest", None, {}),
        ("Momentum quality", "momentum_quality", "highest", None, {}),
        ("Full bull score", "full_bull_score", "highest", None, {}),

        # Filtered combos
        ("Accel + breadth>60%", "price_accel", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("Accel + breadth>70%", "price_accel", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.7}),
        ("RelStr + breadth>60%", "rel_str_z", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("RelStr + breadth>70%", "rel_str_z", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.7}),
        ("VolBreak + breadth>60%", "vol_breakout", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("TrendStr + breadth>60%", "trend_strength", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("FullBull + breadth>60%", "full_bull_score", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("Breakout + accum", "at_20d_high", "highest", None,
         {"filter_col": "accumulation", "filter_op": ">", "filter_val": 0.5}),
        ("Breakout + rel str", "at_20d_high", "highest", None,
         {"filter_col": "rel_str_z", "filter_op": ">", "filter_val": 0.5}),
        ("Consol + accel", "consolidation", "highest", None,
         {"filter_col": "price_accel", "filter_op": ">", "filter_val": 0.5}),
        ("MomQuality + breadth>60%", "momentum_quality", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
    ]

    results = []
    total_configs = len(configs) * 4 * 2  # hold × pos combinations
    tested = 0

    for label, signal, condition, threshold, filters in configs:
        if signal not in panel.columns:
            continue
        p = panel.copy()
        if filters:
            fc, fo, fv = filters["filter_col"], filters["filter_op"], filters["filter_val"]
            if fc in p.columns:
                if fo == ">": p = p[p[fc] > fv]
                elif fo == ">=": p = p[p[fc] >= fv]
                elif fo == "<": p = p[p[fc] < fv]

        for hold in [7, 10, 15, 20]:
            for max_pos in [3, 5]:
                tested += 1
                if tested % 20 == 0:
                    print(f"  [{tested}/{total_configs}] testing...")

                r = run_regime_filtered_backtest(
                    stock_data, p, nifty_regime,
                    signal_col=signal, condition=condition,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    allowed_regime=1, block_reentry_days=hold,
                )
                if r is None or r["metrics"]["trades"] < 30:
                    continue

                m = r["metrics"]
                weekly = r["weekly_returns"]

                t25 = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
                if len(t25) > 5:
                    t25c = t25.copy()
                    t25c["yearweek"] = t25c["year"].astype(str) + "-W" + t25c["week"].astype(str).str.zfill(2)
                    w25 = t25c.groupby("yearweek")["net_return"].sum() / max_pos
                    t25_m = compute_all_metrics(t25["net_return"],
                                                pd.Series([1.0] + list(1 + t25["net_return"].cumsum() / max_pos)))
                else:
                    w25 = pd.Series(dtype=float)
                    t25_m = {"sharpe": -99, "win_rate": 0}

                cap = 600000
                wk = weekly * cap
                w25_inr = w25 * cap if len(w25) > 0 else pd.Series(dtype=float)

                results.append({
                    "label": label, "hold": hold, "pos": max_pos,
                    "sharpe": m["sharpe"], "cagr": m["cagr"], "wr": m["win_rate"],
                    "pf": m["profit_factor"], "mdd": m["max_drawdown"],
                    "trades": m["trades"], "trades_yr": m["trades"] / 5,
                    "avg_wk": wk.mean(),
                    "t25_sh": t25_m["sharpe"],
                    "t25_avg_wk": w25_inr.mean() if len(w25_inr) > 0 else -99999,
                    "t25_pct_pos": (w25_inr > 0).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                    "t25_pct_5k": (w25_inr >= 5000).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                    "t25_weeks": len(w25_inr),
                    "result": r,
                })

    df = pd.DataFrame(results)

    # Filter: positive 2025 Sharpe + decent trade count
    top = df[(df["t25_sh"] > 0) & (df["trades"] >= 40)].sort_values("t25_avg_wk", ascending=False)
    if len(top) == 0:
        top = df[df["trades"] >= 30].sort_values("t25_avg_wk", ascending=False)

    print(f"\n  TOP STRATEGIES (2025 Sharpe > 0, trades >= 40):")
    print(f"  {'Label':<28} {'H':>3} {'P':>2} {'Sh':>5} {'WR':>5} {'PF':>4} {'25Sh':>5} {'25Wk':>7} {'25%>0':>5} {'25%5K':>5} {'Trd':>4} {'T/yr':>4}")
    for _, r in top.head(30).iterrows():
        tag = " <<" if r["t25_avg_wk"] >= 5000 and r["t25_sh"] > 1 else ""
        print(f"  {r['label']:<28} {r['hold']:>3} {r['pos']:>2} "
              f"{r['sharpe']:>5.2f} {r['wr']:>5.1f} {r['pf']:>4.2f} "
              f"{r['t25_sh']:>5.2f} {r['t25_avg_wk']:>+7,.0f} {r['t25_pct_pos']:>5.0f} {r['t25_pct_5k']:>5.0f} "
              f"{r['trades']:>4} {r['trades_yr']:>4.0f}{tag}")

    # ============================================================
    # CHAMPION DEEP DIVE
    # ============================================================
    if len(top) > 0:
        w = top.iloc[0]
        r = w["result"]
        cap = 600000
        print(f"\n\n{'*'*70}")
        print(f"  BULL CHAMPION (N500): {w['label']} | hold={w['hold']}d | pos={w['pos']}")
        print(f"{'*'*70}")

        m = r["metrics"]
        print(f"\n  Full: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1f}% WR={m['win_rate']:.1f}% PF={m['profit_factor']:.2f} MDD={m['max_drawdown']:.1f}%")
        print(f"  Trades: {m['trades']} ({m['trades']/5:.0f}/year)")

        t25 = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
        if len(t25) > 0:
            t25c = t25.copy()
            t25c["yearweek"] = t25c["year"].astype(str) + "-W" + t25c["week"].astype(str).str.zfill(2)
            w25 = t25c.groupby("yearweek")["net_return"].sum() / w["pos"] * cap
            print(f"\n  2025+: Avg/wk={w25.mean():+,.0f} | %>0={((w25>0).sum()/len(w25)*100):.0f}% | "
                  f"%>=5K={((w25>=5000).sum()/len(w25)*100):.0f}% | Weeks={len(w25)}")
            print(f"  Best week: {w25.max():+,.0f} | Worst: {w25.min():+,.0f}")

        # Monthly
        trades = r["trades"].copy()
        trades["month"] = trades["entry_date"].dt.to_period("M")
        print(f"\n  Monthly (last 24 months):")
        mp, mt = 0, 0
        for month in sorted(trades["month"].unique())[-24:]:
            mt_trades = trades[trades["month"] == month]
            pnl = mt_trades["net_return"].sum() / w["pos"] * cap
            wr = (mt_trades["net_return"] > 0).sum() / len(mt_trades) * 100
            stopped = mt_trades["stopped_out"].sum()
            marker = " +" if pnl >= 10000 else (" *" if pnl < 0 else "")
            print(f"    {str(month):<10} {len(mt_trades):>3} trades WR={wr:>4.0f}% Stops={stopped} {pnl:>+10,.0f}{marker}")
            mt += 1
            if pnl > 0: mp += 1
        print(f"  Profitable months: {mp}/{mt} ({mp/mt*100:.0f}%)")

        # Capital sensitivity
        print(f"\n  Capital for 5K/week:")
        if len(t25) > 0:
            w25_raw = t25c.groupby("yearweek")["net_return"].sum() / w["pos"]
            print(f"  {'Capital':>10} {'25Avg':>8} {'25%>=5K':>8} {'25%>0':>6}")
            for c in [300000, 400000, 500000, 600000, 750000]:
                tw = w25_raw * c
                print(f"  {c:>10,} {tw.mean():>+8,.0f} {(tw>=5000).sum()/len(tw)*100:>7.0f}% {(tw>0).sum()/len(tw)*100:>5.0f}%")

        # N500 vs N200 comparison
        from config import NIFTY_200
        n200_trades = trades[trades["ticker"].isin(NIFTY_200)]
        n500_only = trades[~trades["ticker"].isin(NIFTY_200)]
        print(f"\n  N200 vs N500-only stocks:")
        if len(n200_trades) > 0:
            print(f"    N200 stocks: {len(n200_trades)} trades, WR={((n200_trades['net_return']>0).sum()/len(n200_trades)*100):.0f}%, "
                  f"Avg={n200_trades['net_return'].mean()*100:+.2f}%")
        if len(n500_only) > 0:
            print(f"    N500-only:   {len(n500_only)} trades, WR={((n500_only['net_return']>0).sum()/len(n500_only)*100):.0f}%, "
                  f"Avg={n500_only['net_return'].mean()*100:+.2f}%")

        print(f"\n{'*'*70}")


if __name__ == "__main__":
    main()
