#!/usr/bin/env python3
"""
Deep Pattern Research — find what REALLY drives returns.

Instead of just scanning signals blindly, let's understand:
1. What conditions make oversold bounces work vs fail?
2. Does volume confirmation improve the signal?
3. Do sector clusters matter (PSU banks vs IT vs pharma)?
4. Multi-signal combos: which 2-3 signals stack best?
5. Market breadth as regime filter (% stocks above 20d SMA)
6. Intraday pattern: gap-down opens that recover
7. Consecutive down days (3+ red days → bounce?)
8. Walk-forward: rolling 6-month train, 1-month test
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel, zscore_normalize
from backtest.metrics import compute_all_metrics, format_metrics
from research_5k_v5 import get_nifty_regime, run_regime_filtered_backtest


def add_advanced_features(panel, stock_data, index_data):
    """Add features that capture deeper patterns."""
    print(f"\n  Computing advanced features...")

    # 1. Consecutive down days
    for ticker, df in stock_data.items():
        close = df["Close"]
        daily_ret = close.pct_change()
        # Count consecutive negative days
        neg = (daily_ret < 0).astype(int)
        # Rolling streak: count consecutive 1s
        streak = neg.copy()
        for i in range(1, len(streak)):
            if streak.iloc[i] == 1:
                streak.iloc[i] = streak.iloc[i-1] + 1
            else:
                streak.iloc[i] = 0

        mask = panel["ticker"] == ticker
        dates = panel.loc[mask, "date"]
        streaks_aligned = streak.reindex(dates.values)
        panel.loc[mask, "consec_down"] = streaks_aligned.values

    # 2. Volume on down days (selling exhaustion)
    for ticker, df in stock_data.items():
        vol = df["Volume"]
        close = df["Close"]
        ret = close.pct_change()
        vol_avg = vol.rolling(20).mean()
        # Volume ratio on down days only
        vol_on_down = vol.where(ret < 0, np.nan)
        vol_down_ratio = vol_on_down / vol_avg
        # 3-day avg volume ratio on down days
        vol_down_3d = vol_down_ratio.rolling(3, min_periods=1).mean()

        mask = panel["ticker"] == ticker
        dates = panel.loc[mask, "date"]
        panel.loc[mask, "vol_down_ratio"] = vol_down_3d.reindex(dates.values).values

    # 3. Market breadth: % of stocks above their 20d SMA
    all_dates = sorted(panel["date"].unique())
    breadth = {}
    for date in all_dates:
        day_data = panel[panel["date"] == date]
        if len(day_data) < 20:
            continue
        # dist_52w_high > some threshold means stock is doing well
        # Or simpler: use return_20d > 0
        if "return_20d" in day_data.columns:
            above = (day_data["return_20d"] > 0).sum()
            breadth[date] = above / len(day_data)

    panel["mkt_breadth"] = panel["date"].map(breadth)

    # 4. Normalized drawdown from recent high (how far has this stock fallen?)
    for ticker, df in stock_data.items():
        close = df["Close"]
        rolling_high_20 = close.rolling(20).max()
        drawdown_20 = (close / rolling_high_20) - 1  # negative number
        dd_z = zscore_normalize(drawdown_20, lookback=252)

        mask = panel["ticker"] == ticker
        dates = panel.loc[mask, "date"]
        panel.loc[mask, "drawdown_20d"] = drawdown_20.reindex(dates.values).values
        panel.loc[mask, "drawdown_20d_z"] = dd_z.reindex(dates.values).values

    # 5. RSI(3) + Volume combo score
    if "rsi_3" in panel.columns and "volume_ratio" in panel.columns:
        # Oversold + high volume = selling exhaustion (bullish)
        panel["oversold_volume"] = -panel["rsi_3"] + panel["volume_ratio"]

    # 6. Mean reversion composite: combine return_5d + rsi_3 + drawdown
    if all(c in panel.columns for c in ["return_5d", "rsi_3", "drawdown_20d_z"]):
        panel["mr_deep"] = (panel["return_5d"] + panel["rsi_3"] + panel["drawdown_20d_z"]) / 3

    print(f"  Added: consec_down, vol_down_ratio, mkt_breadth, drawdown_20d, oversold_volume, mr_deep")
    return panel


def analyze_conditional_performance(panel, stock_data):
    """Find conditions under which the oversold bounce is strongest."""
    print(f"\n{'='*70}")
    print(f"  CONDITIONAL ANALYSIS: When does oversold bounce work best?")
    print(f"{'='*70}")

    # Get oversold stocks (bottom decile RSI-3)
    panel_copy = panel.copy()
    results = []

    for date in sorted(panel_copy["date"].unique()):
        day = panel_copy[panel_copy["date"] == date].dropna(subset=["rsi_3"])
        if len(day) < 20:
            continue
        try:
            day = day.copy()
            day["_q"] = pd.qcut(day["rsi_3"], 10, labels=False, duplicates="drop")
            oversold = day[day["_q"] == 0]  # Bottom decile
        except ValueError:
            continue

        for _, row in oversold.iterrows():
            if pd.notna(row.get("fwd_5d")):
                results.append({
                    "date": date,
                    "ticker": row["ticker"],
                    "fwd_5d": row["fwd_5d"],
                    "rsi_3": row.get("rsi_3", np.nan),
                    "volume_ratio": row.get("volume_ratio", np.nan),
                    "consec_down": row.get("consec_down", np.nan),
                    "vol_down_ratio": row.get("vol_down_ratio", np.nan),
                    "mkt_breadth": row.get("mkt_breadth", np.nan),
                    "drawdown_20d": row.get("drawdown_20d", np.nan),
                    "return_5d": row.get("return_5d", np.nan),
                    "realized_vol_20d": row.get("realized_vol_20d", np.nan),
                    "overnight_gap": row.get("overnight_gap", np.nan),
                })

    rdf = pd.DataFrame(results)
    if len(rdf) == 0:
        print("  No data")
        return

    avg_fwd = rdf["fwd_5d"].mean() * 100
    print(f"\n  Oversold stocks (RSI3 bottom decile): {len(rdf):,} observations")
    print(f"  Average 5-day forward return: {avg_fwd:+.3f}%")

    # Test each condition
    conditions = [
        ("consec_down >= 3", rdf["consec_down"] >= 3, "3+ consecutive down days"),
        ("consec_down >= 4", rdf["consec_down"] >= 4, "4+ consecutive down days"),
        ("consec_down >= 5", rdf["consec_down"] >= 5, "5+ consecutive down days"),
        ("vol_down_ratio > 1.5", rdf["vol_down_ratio"] > 1.5, "High volume on down days"),
        ("vol_down_ratio > 2.0", rdf["vol_down_ratio"] > 2.0, "Very high volume on down"),
        ("mkt_breadth < 0.3", rdf["mkt_breadth"] < 0.3, "Market breadth < 30% (broad weakness)"),
        ("mkt_breadth < 0.4", rdf["mkt_breadth"] < 0.4, "Market breadth < 40%"),
        ("mkt_breadth < 0.5", rdf["mkt_breadth"] < 0.5, "Market breadth < 50%"),
        ("drawdown_20d < -0.10", rdf["drawdown_20d"] < -0.10, "Stock down >10% from 20d high"),
        ("drawdown_20d < -0.15", rdf["drawdown_20d"] < -0.15, "Stock down >15% from 20d high"),
        ("overnight_gap < -0.5", rdf["overnight_gap"] < -0.5, "Gapped down (z < -0.5)"),
        ("realized_vol > 0.5", rdf["realized_vol_20d"] > 0.5, "High volatility stock (z > 0.5)"),
        ("realized_vol < -0.5", rdf["realized_vol_20d"] < -0.5, "Low volatility stock (z < -0.5)"),
    ]

    print(f"\n  {'Condition':<45} {'N':>6} {'Avg 5d Ret':>10} {'WR':>6} {'t-stat':>7} {'vs base':>8}")
    for label, mask, desc in conditions:
        subset = rdf[mask]
        if len(subset) < 50:
            continue
        fwd = subset["fwd_5d"]
        avg = fwd.mean() * 100
        wr = (fwd > 0).sum() / len(fwd) * 100
        t, p = stats.ttest_1samp(fwd.dropna(), 0)
        diff = avg - avg_fwd
        marker = " <<" if avg > avg_fwd * 1.5 and abs(t) > 2 else ""
        print(f"  {desc:<45} {len(subset):>6} {avg:>+9.3f}% {wr:>5.1f}% {t:>7.2f} {diff:>+7.3f}%{marker}")

    # Combo conditions
    print(f"\n  --- COMBO CONDITIONS ---")
    combos = [
        ("Oversold + 3 down days + breadth<40%",
         (rdf["consec_down"] >= 3) & (rdf["mkt_breadth"] < 0.4)),
        ("Oversold + 3 down days + high sell vol",
         (rdf["consec_down"] >= 3) & (rdf["vol_down_ratio"] > 1.5)),
        ("Oversold + >10% drawdown + breadth<40%",
         (rdf["drawdown_20d"] < -0.10) & (rdf["mkt_breadth"] < 0.4)),
        ("Oversold + >10% drawdown + high sell vol",
         (rdf["drawdown_20d"] < -0.10) & (rdf["vol_down_ratio"] > 1.5)),
        ("Oversold + 3 down + >10% DD + breadth<40%",
         (rdf["consec_down"] >= 3) & (rdf["drawdown_20d"] < -0.10) & (rdf["mkt_breadth"] < 0.4)),
        ("Oversold + gap down + high sell vol",
         (rdf["overnight_gap"] < -0.5) & (rdf["vol_down_ratio"] > 1.5)),
        ("Oversold + low vol stock (quality dip)",
         (rdf["realized_vol_20d"] < -0.5) & (rdf["mkt_breadth"] < 0.5)),
        ("Oversold + low vol + 3 down days",
         (rdf["realized_vol_20d"] < -0.5) & (rdf["consec_down"] >= 3)),
        ("Oversold + breadth<30% (capitulation)",
         (rdf["mkt_breadth"] < 0.3)),
        ("Oversold + breadth<30% + 3 down days",
         (rdf["mkt_breadth"] < 0.3) & (rdf["consec_down"] >= 3)),
    ]

    for label, mask in combos:
        subset = rdf[mask]
        if len(subset) < 30:
            n_str = f"({len(subset)} obs - too few)"
            print(f"  {label:<50} {n_str}")
            continue
        fwd = subset["fwd_5d"]
        avg = fwd.mean() * 100
        wr = (fwd > 0).sum() / len(fwd) * 100
        t, p = stats.ttest_1samp(fwd.dropna(), 0)
        diff = avg - avg_fwd
        marker = " <<" if avg > avg_fwd * 1.5 and abs(t) > 2 else ""
        print(f"  {label:<50} N={len(subset):>5} Avg={avg:>+.3f}% WR={wr:.0f}% t={t:.2f} vs_base={diff:>+.3f}%{marker}")

    return rdf


def backtest_enhanced_strategies(panel, stock_data, nifty_regime):
    """Backtest the strongest conditional patterns found above."""
    print(f"\n\n{'='*70}")
    print(f"  BACKTESTING ENHANCED STRATEGIES")
    print(f"{'='*70}")

    results = []

    # Strategy configs to test
    configs = [
        # (label, signal, condition, threshold, secondary filters as dict)
        ("Baseline: RSI3 bear dip",
         "rsi_3", "lowest", None, 0, {}),
        ("RSI3 + consec_down>=3",
         "rsi_3", "lowest", None, 0,
         {"filter_col": "consec_down", "filter_op": ">=", "filter_val": 3}),
        ("RSI3 + drawdown>10%",
         "rsi_3", "lowest", None, 0,
         {"filter_col": "drawdown_20d", "filter_op": "<", "filter_val": -0.10}),
        ("RSI3 + breadth<40%",
         "rsi_3", "lowest", None, 0,
         {"filter_col": "mkt_breadth", "filter_op": "<", "filter_val": 0.4}),
        ("RSI3 + breadth<30%",
         "rsi_3", "lowest", None, 0,
         {"filter_col": "mkt_breadth", "filter_op": "<", "filter_val": 0.3}),
        ("mr_deep composite bear",
         "mr_deep", "lowest", None, 0, {}),
        ("drawdown_20d_z bear",
         "drawdown_20d_z", "lowest", None, 0, {}),
        ("oversold_volume bear",
         "oversold_volume", "highest", None, 0, {}),
        ("RSI3 + vol_down>1.5",
         "rsi_3", "lowest", None, 0,
         {"filter_col": "vol_down_ratio", "filter_op": ">", "filter_val": 1.5}),
        ("RSI3 + low_vol stock",
         "rsi_3", "lowest", None, 0,
         {"filter_col": "realized_vol_20d", "filter_op": "<", "filter_val": -0.5}),
    ]

    for label, signal, condition, threshold, regime, filters in configs:
        # Apply filters to panel
        p = panel.copy()
        if filters:
            fc = filters["filter_col"]
            fo = filters["filter_op"]
            fv = filters["filter_val"]
            if fc in p.columns:
                if fo == ">=":
                    p = p[p[fc] >= fv]
                elif fo == "<":
                    p = p[p[fc] < fv]
                elif fo == ">":
                    p = p[p[fc] > fv]
                elif fo == "<=":
                    p = p[p[fc] <= fv]

        for hold in [3, 5, 7]:
            for max_pos in [3, 5]:
                r = run_regime_filtered_backtest(
                    stock_data, p, nifty_regime,
                    signal_col=signal, condition=condition,
                    threshold=threshold,
                    hold_days=hold, max_positions=max_pos,
                    num_buckets=10, allowed_regime=0,
                    block_reentry_days=5,
                )
                if r is None or r["metrics"]["trades"] < 30:
                    continue

                m = r["metrics"]
                weekly = r["weekly_returns"]

                # 2025+ test
                t25 = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
                if len(t25) > 0:
                    t25c = t25.copy()
                    t25c["yearweek"] = t25c["year"].astype(str) + "-W" + t25c["week"].astype(str).str.zfill(2)
                    w25 = t25c.groupby("yearweek")["net_return"].sum() / max_pos
                    t25_m = compute_all_metrics(t25["net_return"],
                                                pd.Series([1.0] + list(1 + t25["net_return"].cumsum() / max_pos)))
                else:
                    w25 = pd.Series(dtype=float)
                    t25_m = {"sharpe": -99, "win_rate": 0}

                # Crash test
                crash = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2026-02-01")]
                crash_pnl_pct = crash["net_return"].sum() / max_pos * 100 if len(crash) > 0 else 0

                for cap in [400000, 600000]:
                    wk = weekly * cap
                    w25_inr = w25 * cap if len(w25) > 0 else pd.Series(dtype=float)

                    results.append({
                        "label": label, "hold": hold, "pos": max_pos, "cap": cap,
                        "sharpe": m["sharpe"], "cagr": m["cagr"], "wr": m["win_rate"],
                        "pf": m["profit_factor"], "mdd": m["max_drawdown"],
                        "trades": m["trades"],
                        "avg_wk": wk.mean(),
                        "pct_pos": (wk > 0).sum() / len(wk) * 100 if len(wk) > 0 else 0,
                        "t25_sh": t25_m["sharpe"], "t25_wr": t25_m["win_rate"],
                        "t25_avg_wk": w25_inr.mean() if len(w25_inr) > 0 else -99999,
                        "t25_pct_5k": (w25_inr >= 5000).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                        "t25_pct_pos": (w25_inr > 0).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                        "crash_pnl": crash_pnl_pct,
                        "crash_trades": len(crash),
                        "result": r,
                    })

    df = pd.DataFrame(results)

    # Sort by 2025 performance
    df_sorted = df[df["t25_sh"] > 0].sort_values("t25_avg_wk", ascending=False)
    if len(df_sorted) == 0:
        df_sorted = df.sort_values("t25_avg_wk", ascending=False)

    print(f"\n  TOP STRATEGIES (sorted by 2025 avg weekly):")
    print(f"  {'Label':<30} {'H':>2} {'P':>2} {'Cap':>4} {'Sh':>5} {'WR':>5} {'PF':>4} {'25Sh':>5} {'25Wk':>7} {'25%5K':>5} {'25%>0':>5} {'CrPnL':>6} {'Trd':>4}")
    for _, r in df_sorted.head(30).iterrows():
        tag = " <<" if r["t25_avg_wk"] >= 5000 and r["t25_sh"] > 1 else ""
        print(f"  {r['label']:<30} {r['hold']:>2} {r['pos']:>2} {r['cap']/1000:>3.0f}K "
              f"{r['sharpe']:>5.2f} {r['wr']:>5.1f} {r['pf']:>4.2f} "
              f"{r['t25_sh']:>5.2f} {r['t25_avg_wk']:>+7,.0f} {r['t25_pct_5k']:>5.0f} {r['t25_pct_pos']:>5.0f} "
              f"{r['crash_pnl']:>+5.1f}% {r['trades']:>4}{tag}")

    # ============================================================
    # WINNER DEEP DIVE
    # ============================================================
    if len(df_sorted) > 0:
        winner = df_sorted.iloc[0]
        r = winner["result"]
        cap = int(winner["cap"])

        print(f"\n\n{'*'*70}")
        print(f"  CHAMPION: {winner['label']}")
        print(f"  hold={winner['hold']}d | pos={winner['pos']} | cap={cap:,}")
        print(f"{'*'*70}")

        m = r["metrics"]
        print(f"\n  Full: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1f}% WR={m['win_rate']:.1f}% PF={m['profit_factor']:.2f} MDD={m['max_drawdown']:.1f}%")
        print(f"  Trades: {m['trades']} | Trades/yr: {m['trades'] / 5:.0f}")

        # 2025 weekly
        t25 = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
        if len(t25) > 0:
            t25c = t25.copy()
            t25c["yearweek"] = t25c["year"].astype(str) + "-W" + t25c["week"].astype(str).str.zfill(2)
            w25 = t25c.groupby("yearweek")["net_return"].sum() / winner["pos"] * cap
            print(f"\n  2025 Out-of-Sample:")
            print(f"    Avg weekly: {w25.mean():+,.0f} INR")
            print(f"    Weeks >= 5K: {(w25 >= 5000).sum()}/{len(w25)} ({(w25>=5000).sum()/len(w25)*100:.0f}%)")
            print(f"    Weeks > 0: {(w25 > 0).sum()}/{len(w25)} ({(w25>0).sum()/len(w25)*100:.0f}%)")
            print(f"    Best week: {w25.max():+,.0f} | Worst: {w25.min():+,.0f}")

        # Monthly
        trades = r["trades"].copy()
        trades["month"] = trades["entry_date"].dt.to_period("M")
        print(f"\n  Monthly P&L at {cap:,}:")
        mp = 0
        mt = 0
        for month in sorted(trades["month"].unique()):
            mt_trades = trades[trades["month"] == month]
            pnl = mt_trades["net_return"].sum() / winner["pos"] * cap
            wr = (mt_trades["net_return"] > 0).sum() / len(mt_trades) * 100
            stopped = mt_trades["stopped_out"].sum()
            marker = " +" if pnl >= 20000 else (" *" if pnl < 0 else "")
            print(f"    {str(month):<10} {len(mt_trades):>3} trades WR={wr:>4.0f}% Stops={stopped} {pnl:>+10,.0f}{marker}")
            mt += 1
            if pnl > 0: mp += 1
        print(f"  Profitable months: {mp}/{mt} ({mp/mt*100:.0f}%)")

        # Capital sensitivity
        wk_full = r["weekly_returns"]
        print(f"\n  Capital for 5K/week target:")
        print(f"  {'Capital':>10} {'FullAvg':>8} {'25Avg':>8} {'25%>=5K':>8} {'25%>0':>6}")
        for c in [300000, 400000, 500000, 600000, 750000, 1000000]:
            fw = wk_full * c
            if len(t25) > 0:
                tw = t25c.groupby("yearweek")["net_return"].sum() / winner["pos"] * c
                print(f"  {c:>10,} {fw.mean():>+8,.0f} {tw.mean():>+8,.0f} {(tw>=5000).sum()/len(tw)*100:>7.0f}% {(tw>0).sum()/len(tw)*100:>5.0f}%")

        print(f"\n{'*'*70}")

    return df


def main():
    print(f"\n{'#'*70}")
    print(f"  DEEP PATTERN RESEARCH")
    print(f"  Finding what makes oversold bounces work vs fail")
    print(f"{'#'*70}")

    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)
    nifty_regime = get_nifty_regime(index_data)

    # Add advanced features
    panel = add_advanced_features(panel, stock_data, index_data)

    # Conditional analysis
    analyze_conditional_performance(panel, stock_data)

    # Backtest enhanced strategies
    backtest_enhanced_strategies(panel, stock_data, nifty_regime)


if __name__ == "__main__":
    main()
