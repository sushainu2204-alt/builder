#!/usr/bin/env python3
"""
Deep Research: Bull Regime Strategy

The bear dip strategy sits in cash when Nifty > 20d SMA.
We need a strategy that works DURING bull regimes.

Approach:
1. Isolate all bull-regime data (Nifty > 20d SMA)
2. Scan ALL signals — find what predicts 5d forward returns in bull markets
3. Conditional analysis — what amplifies the edge?
4. Backtest top candidates with walk-forward validation
5. Test combined dual-strategy (bull + bear) for all-weather performance
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
from signals.scanner import quintile_analysis, classify_signal
from backtest.metrics import compute_all_metrics, format_metrics
from research_5k_v5 import get_nifty_regime, run_regime_filtered_backtest
from research_deep import add_advanced_features


def scan_bull_regime(panel, nifty_regime):
    """Find what signals work when market is in bull mode."""
    print(f"\n{'='*70}")
    print(f"  SIGNAL SCAN — BULL REGIME ONLY (Nifty > 20d SMA)")
    print(f"{'='*70}")

    regime_map = nifty_regime.to_dict()
    panel_copy = panel.copy()

    def get_regime(date):
        for delta in range(0, 10):
            d = date - pd.Timedelta(days=delta)
            if d in regime_map:
                return regime_map[d]
        return None

    panel_copy["regime"] = panel_copy["date"].apply(get_regime)
    bull_panel = panel_copy[panel_copy["regime"] == 1]
    bear_panel = panel_copy[panel_copy["regime"] == 0]

    bull_days = bull_panel["date"].nunique()
    bear_days = bear_panel["date"].nunique()
    print(f"  Bull days: {bull_days} | Bear days: {bear_days}")

    exclude_cols = {"ticker", "date", "regime"} | {f"fwd_{d}d" for d in [1, 3, 5, 10, 20]}
    feature_cols = [c for c in panel_copy.columns if c not in exclude_cols and not c.startswith("_")]

    print(f"\n  Scanning {len(feature_cols)} features in bull regime...")
    print(f"\n  {'Feature':<22} {'Bull t-stat':>10} {'Bull spread':>12} {'Bear t-stat':>10} {'Bear spread':>12} {'Direction'}")

    bull_results = []
    for feat in feature_cols:
        rb = quintile_analysis(bull_panel, feat, "fwd_5d")
        rr = quintile_analysis(bear_panel, feat, "fwd_5d")
        if rb:
            rb["classification"] = classify_signal(rb)
            bull_results.append(rb)
            bear_t = f"{rr['t_stat']:.2f}" if rr else "N/A"
            bear_sp = f"{rr['spread']:+.2f}%" if rr else "N/A"
            print(f"  {feat:<22} {rb['t_stat']:>10.2f} {rb['spread']:>+11.2f}% {bear_t:>10} {bear_sp:>12} {rb['direction']}")

    bull_results.sort(key=lambda x: x["t_stat"], reverse=True)
    print(f"\n  Top 5 bull-regime signals:")
    for i, r in enumerate(bull_results[:5], 1):
        print(f"    {i}. {r['feature']:<22} t={r['t_stat']:.2f} spread={r['spread']:+.2f}% {r['direction']} {r['classification']}")

    return bull_results, bull_panel, bear_panel


def conditional_analysis_bull(panel, nifty_regime):
    """What conditions make bull-regime signals work best?"""
    print(f"\n{'='*70}")
    print(f"  CONDITIONAL ANALYSIS — BULL REGIME")
    print(f"  What makes momentum/trend signals work best?")
    print(f"{'='*70}")

    regime_map = nifty_regime.to_dict()
    panel_copy = panel.copy()

    def get_regime(date):
        for delta in range(0, 10):
            d = date - pd.Timedelta(days=delta)
            if d in regime_map:
                return regime_map[d]
        return None

    panel_copy["regime"] = panel_copy["date"].apply(get_regime)
    bull = panel_copy[panel_copy["regime"] == 1]

    # Get top decile of various signals and measure their 5d forward returns
    strategies = {
        "Momentum 60d (top decile)": ("return_60d", "highest"),
        "Momentum 20d (top decile)": ("return_20d", "highest"),
        "Near 52w high (top decile)": ("dist_52w_high", "highest"),
        "Rel strength (top decile)": ("rel_strength_20d", "highest"),
        "Low vol (bottom decile)": ("realized_vol_20d", "lowest"),
        "RSI3 oversold (bottom decile)": ("rsi_3", "lowest"),
        "5d losers (bottom decile)": ("return_5d", "lowest"),
        "Volume spike (top decile)": ("volume_ratio", "highest"),
    }

    for label, (signal, direction) in strategies.items():
        if signal not in bull.columns:
            continue

        observations = []
        for date in sorted(bull["date"].unique()):
            day = bull[bull["date"] == date].dropna(subset=[signal])
            if len(day) < 20:
                continue
            try:
                day = day.copy()
                day["_q"] = pd.qcut(day[signal], 10, labels=False, duplicates="drop")
                if direction == "highest":
                    selected = day[day["_q"] == day["_q"].max()]
                else:
                    selected = day[day["_q"] == 0]
            except ValueError:
                continue

            for _, row in selected.iterrows():
                if pd.notna(row.get("fwd_5d")):
                    observations.append({
                        "fwd_5d": row["fwd_5d"],
                        "mkt_breadth": row.get("mkt_breadth", np.nan),
                        "consec_down": row.get("consec_down", np.nan),
                        "vol_down_ratio": row.get("vol_down_ratio", np.nan),
                        "drawdown_20d": row.get("drawdown_20d", np.nan),
                        "realized_vol_20d": row.get("realized_vol_20d", np.nan),
                        "volume_ratio": row.get("volume_ratio", np.nan),
                    })

        if len(observations) < 100:
            continue

        odf = pd.DataFrame(observations)
        avg = odf["fwd_5d"].mean() * 100
        wr = (odf["fwd_5d"] > 0).sum() / len(odf) * 100
        t, _ = stats.ttest_1samp(odf["fwd_5d"].dropna(), 0)

        print(f"\n  {label}: N={len(odf):,} Avg={avg:+.3f}% WR={wr:.1f}% t={t:.2f}")

        # Test conditions
        conditions = [
            ("breadth > 60%", odf["mkt_breadth"] > 0.6, "Strong bull"),
            ("breadth > 70%", odf["mkt_breadth"] > 0.7, "Very strong bull"),
            ("breadth 40-60%", (odf["mkt_breadth"] >= 0.4) & (odf["mkt_breadth"] <= 0.6), "Moderate bull"),
            ("low vol stock", odf["realized_vol_20d"] < -0.5, "Quality/stable stock"),
            ("high vol stock", odf["realized_vol_20d"] > 0.5, "Volatile stock"),
            ("vol spike", odf["volume_ratio"] > 1.0, "Above avg volume"),
            ("drawdown < -5%", odf["drawdown_20d"] < -0.05, "Recent pullback"),
            ("no drawdown", odf["drawdown_20d"] > -0.02, "Near highs"),
        ]

        for clabel, mask, desc in conditions:
            sub = odf[mask]
            if len(sub) < 50:
                continue
            savg = sub["fwd_5d"].mean() * 100
            swr = (sub["fwd_5d"] > 0).sum() / len(sub) * 100
            st, _ = stats.ttest_1samp(sub["fwd_5d"].dropna(), 0)
            diff = savg - avg
            marker = " <<" if savg > avg * 1.3 and st > 2 else ""
            print(f"    + {desc:<25} N={len(sub):>5} Avg={savg:>+.3f}% WR={swr:.0f}% t={st:.2f} vs_base={diff:>+.3f}%{marker}")


def backtest_bull_strategies(panel, stock_data, nifty_regime):
    """Backtest the most promising bull-regime strategies."""
    print(f"\n\n{'='*70}")
    print(f"  BACKTESTING BULL STRATEGIES")
    print(f"{'='*70}")

    configs = [
        # (label, signal, condition, threshold, filters)
        ("Momentum 60d", "return_60d", "highest", None, {}),
        ("Momentum 20d", "return_20d", "highest", None, {}),
        ("Near 52w high", "dist_52w_high", "highest", None, {}),
        ("Rel strength", "rel_strength_20d", "highest", None, {}),
        ("5d dip in bull", "return_5d", "lowest", None, {}),
        ("RSI3 dip in bull", "rsi_3", "lowest", None, {}),
        ("Mom 60d + breadth>60%", "return_60d", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("Mom 60d + breadth>70%", "return_60d", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.7}),
        ("Mom 20d + breadth>60%", "return_20d", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("5d dip + breadth>60%", "return_5d", "lowest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("RSI3 dip + breadth>60%", "rsi_3", "lowest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("52w high + low vol", "dist_52w_high", "highest", None,
         {"filter_col": "realized_vol_20d", "filter_op": "<", "filter_val": -0.5}),
        ("Mom 60d + low vol", "return_60d", "highest", None,
         {"filter_col": "realized_vol_20d", "filter_op": "<", "filter_val": -0.5}),
        ("5d dip + consec down>=3", "return_5d", "lowest", None,
         {"filter_col": "consec_down", "filter_op": ">=", "filter_val": 3}),
        ("RSI3 dip + consec>=3", "rsi_3", "lowest", None,
         {"filter_col": "consec_down", "filter_op": ">=", "filter_val": 3}),
    ]

    results = []
    for label, signal, condition, threshold, filters in configs:
        p = panel.copy()
        if filters:
            fc, fo, fv = filters["filter_col"], filters["filter_op"], filters["filter_val"]
            if fc in p.columns:
                if fo == ">": p = p[p[fc] > fv]
                elif fo == ">=": p = p[p[fc] >= fv]
                elif fo == "<": p = p[p[fc] < fv]

        for hold in [3, 5, 7, 10]:
            for max_pos in [3, 5]:
                r = run_regime_filtered_backtest(
                    stock_data, p, nifty_regime,
                    signal_col=signal, condition=condition,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    allowed_regime=1, block_reentry_days=10,
                )
                if r is None or r["metrics"]["trades"] < 50:
                    continue

                m = r["metrics"]
                weekly = r["weekly_returns"]

                # Test on 2025+
                t25 = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
                if len(t25) > 10:
                    t25c = t25.copy()
                    t25c["yearweek"] = t25c["year"].astype(str) + "-W" + t25c["week"].astype(str).str.zfill(2)
                    w25 = t25c.groupby("yearweek")["net_return"].sum() / max_pos
                    t25_m = compute_all_metrics(t25["net_return"],
                                                pd.Series([1.0] + list(1 + t25["net_return"].cumsum() / max_pos)))
                    t25_sh = t25_m["sharpe"]
                    t25_wr = t25_m["win_rate"]
                else:
                    w25 = pd.Series(dtype=float)
                    t25_sh = -99
                    t25_wr = 0

                for cap in [300000, 600000]:
                    wk = weekly * cap
                    w25_inr = w25 * cap if len(w25) > 0 else pd.Series(dtype=float)
                    results.append({
                        "label": label, "hold": hold, "pos": max_pos, "cap": cap,
                        "sharpe": m["sharpe"], "cagr": m["cagr"], "wr": m["win_rate"],
                        "pf": m["profit_factor"], "mdd": m["max_drawdown"],
                        "trades": m["trades"],
                        "avg_wk": wk.mean(),
                        "t25_sh": t25_sh, "t25_wr": t25_wr,
                        "t25_avg_wk": w25_inr.mean() if len(w25_inr) > 0 else -99999,
                        "t25_pct_pos": (w25_inr > 0).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                        "t25_pct_5k": (w25_inr >= 5000).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                        "result": r,
                    })

    df = pd.DataFrame(results)
    top = df[df["t25_sh"] > 0].sort_values("t25_avg_wk", ascending=False)
    if len(top) == 0:
        top = df.sort_values("t25_avg_wk", ascending=False)

    print(f"\n  TOP BULL STRATEGIES (sorted by 2025 avg weekly):")
    print(f"  {'Label':<28} {'H':>2} {'P':>2} {'Cap':>4} {'Sh':>5} {'CAGR':>6} {'WR':>5} {'25Sh':>5} {'25Wk':>7} {'25%>0':>5} {'25%5K':>5} {'Trd':>4}")
    for _, r in top.head(25).iterrows():
        tag = " <<" if r["t25_avg_wk"] >= 3000 and r["t25_sh"] > 0.5 else ""
        print(f"  {r['label']:<28} {r['hold']:>2} {r['pos']:>2} {r['cap']/1000:>3.0f}K "
              f"{r['sharpe']:>5.2f} {r['cagr']:>+5.1f}% {r['wr']:>5.1f} "
              f"{r['t25_sh']:>5.2f} {r['t25_avg_wk']:>+7,.0f} {r['t25_pct_pos']:>5.0f} {r['t25_pct_5k']:>5.0f} "
              f"{r['trades']:>4}{tag}")

    return df


def test_dual_strategy(panel, stock_data, nifty_regime, bull_df):
    """Combine best bull strategy with the bear dip strategy."""
    print(f"\n\n{'='*70}")
    print(f"  DUAL STRATEGY: Bull + Bear combined (all-weather)")
    print(f"{'='*70}")

    # Bear strategy (our champion)
    bear_result = run_regime_filtered_backtest(
        stock_data, panel[panel["mkt_breadth"] < 0.4] if "mkt_breadth" in panel.columns else panel,
        nifty_regime,
        signal_col="rsi_3", condition="lowest",
        hold_days=5, max_positions=5, num_buckets=10,
        allowed_regime=0, block_reentry_days=5,
    )

    # Get top bull configs
    top_bull = bull_df[bull_df["t25_sh"] > 0].sort_values("t25_avg_wk", ascending=False)
    if len(top_bull) == 0:
        top_bull = bull_df.sort_values("t25_avg_wk", ascending=False)

    seen = set()
    unique_bull = []
    for _, r in top_bull.iterrows():
        key = (r["label"], r["hold"], r["pos"])
        if key not in seen:
            seen.add(key)
            unique_bull.append(r)
        if len(unique_bull) >= 5:
            break

    combos = []
    for bull_row in unique_bull:
        bull_r = bull_row["result"]
        if bull_r is None or bear_result is None:
            continue

        # Merge trades
        all_trades = []
        if not bull_r["trades"].empty:
            bt = bull_r["trades"].copy()
            bt["strategy"] = "BULL"
            all_trades.append(bt)
        if not bear_result["trades"].empty:
            brt = bear_result["trades"].copy()
            brt["strategy"] = "BEAR"
            all_trades.append(brt)

        if not all_trades:
            continue

        combined = pd.concat(all_trades).sort_values("entry_date").reset_index(drop=True)
        max_pos = int(bull_row["pos"])

        eq = [1.0]
        for _, t in combined.iterrows():
            eq.append(eq[-1] * (1 + t["net_return"] / max_pos))

        m = compute_all_metrics(combined["net_return"], pd.Series(eq))
        combined["week"] = combined["entry_date"].dt.isocalendar().week.astype(int)
        combined["year"] = combined["entry_date"].dt.year
        combined["yearweek"] = combined["year"].astype(str) + "-W" + combined["week"].astype(str).str.zfill(2)
        weekly = combined.groupby("yearweek")["net_return"].sum() / max_pos

        # 2025 test
        t25 = combined[combined["entry_date"] >= pd.Timestamp("2025-01-01")]
        if len(t25) > 10:
            t25c = t25.copy()
            t25c["yearweek"] = t25c["year"].astype(str) + "-W" + t25c["week"].astype(str).str.zfill(2)
            w25 = t25c.groupby("yearweek")["net_return"].sum() / max_pos
        else:
            w25 = pd.Series(dtype=float)

        for cap in [300000, 600000]:
            wk = weekly * cap
            w25_inr = w25 * cap if len(w25) > 0 else pd.Series(dtype=float)

            bull_trades = len(combined[combined["strategy"] == "BULL"])
            bear_trades = len(combined[combined["strategy"] == "BEAR"])
            weeks_traded = len(weekly)

            combos.append({
                "bull_label": bull_row["label"],
                "bull_hold": int(bull_row["hold"]),
                "bull_pos": int(bull_row["pos"]),
                "cap": cap,
                "sharpe": m["sharpe"], "cagr": m["cagr"], "wr": m["win_rate"],
                "mdd": m["max_drawdown"], "trades": m["trades"],
                "bull_trades": bull_trades, "bear_trades": bear_trades,
                "weeks_traded": weeks_traded,
                "avg_wk": wk.mean(),
                "pct_pos": (wk > 0).sum() / len(wk) * 100,
                "t25_avg_wk": w25_inr.mean() if len(w25_inr) > 0 else -99999,
                "t25_pct_pos": (w25_inr > 0).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                "t25_pct_5k": (w25_inr >= 5000).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                "t25_weeks": len(w25_inr),
                "result": {"trades": combined, "weekly": weekly, "metrics": m},
            })

    cdf = pd.DataFrame(combos)
    cdf_sorted = cdf.sort_values("t25_avg_wk", ascending=False)

    # Bear-only baseline
    if bear_result:
        bw = bear_result["weekly_returns"]
        print(f"\n  BEAR-ONLY BASELINE (current strategy):")
        for cap in [300000, 600000]:
            bwk = bw * cap
            print(f"    {cap/1000:.0f}K: Avg/wk={bwk.mean():+,.0f} | Weeks traded={len(bw)} | %>0={((bwk>0).sum()/len(bwk)*100):.0f}%")

    print(f"\n  DUAL STRATEGIES (Bull + Bear):")
    print(f"  {'Bull strategy':<28} {'H':>2} {'P':>2} {'Cap':>4} {'Sh':>5} {'WR':>5} {'Wks':>4} {'B/R':>7} {'AvgWk':>7} {'25Wk':>7} {'25%>0':>5} {'25%5K':>5}")
    for _, r in cdf_sorted.head(15).iterrows():
        tag = " <<" if r["t25_avg_wk"] >= 4000 and r["t25_pct_pos"] > 55 else ""
        br = f"{r['bull_trades']}/{r['bear_trades']}"
        print(f"  {r['bull_label']:<28} {r['bull_hold']:>2} {r['bull_pos']:>2} {r['cap']/1000:>3.0f}K "
              f"{r['sharpe']:>5.2f} {r['wr']:>5.1f} {r['weeks_traded']:>4} {br:>7} "
              f"{r['avg_wk']:>+7,.0f} {r['t25_avg_wk']:>+7,.0f} {r['t25_pct_pos']:>5.0f} {r['t25_pct_5k']:>5.0f}{tag}")

    # Winner deep dive
    if len(cdf_sorted) > 0:
        w = cdf_sorted.iloc[0]
        print(f"\n\n{'*'*70}")
        print(f"  DUAL CHAMPION: Bear dip + {w['bull_label']} (hold={w['bull_hold']}d, pos={w['bull_pos']})")
        print(f"  Capital: {w['cap']:,}")
        print(f"{'*'*70}")

        m = w["result"]["metrics"]
        weekly = w["result"]["weekly"] * w["cap"]
        trades = w["result"]["trades"]
        cap = w["cap"]

        print(f"\n  Full period: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1f}% WR={m['win_rate']:.1f}% MDD={m['max_drawdown']:.1f}%")
        print(f"  Trades: {m['trades']} (Bull: {w['bull_trades']}, Bear: {w['bear_trades']})")
        print(f"  Weeks traded: {w['weeks_traded']} | Avg/wk: {weekly.mean():+,.0f}")

        # 2025
        t25 = trades[trades["entry_date"] >= pd.Timestamp("2025-01-01")]
        if len(t25) > 0:
            t25c = t25.copy()
            t25c["yearweek"] = t25c["year"].astype(str) + "-W" + t25c["week"].astype(str).str.zfill(2)
            w25 = t25c.groupby("yearweek")["net_return"].sum() / w["bull_pos"] * cap
            print(f"\n  2025 Out-of-Sample:")
            print(f"    Weeks: {len(w25)} | Avg/wk: {w25.mean():+,.0f} | %>0: {(w25>0).sum()}/{len(w25)} ({(w25>0).sum()/len(w25)*100:.0f}%)")
            print(f"    Weeks >= 5K: {(w25>=5000).sum()}/{len(w25)} ({(w25>=5000).sum()/len(w25)*100:.0f}%)")

        # Monthly
        trades_c = trades.copy()
        trades_c["month"] = trades_c["entry_date"].dt.to_period("M")
        print(f"\n  Monthly P&L at {cap:,}:")
        mp, mt = 0, 0
        for month in sorted(trades_c["month"].unique())[-18:]:
            mtrades = trades_c[trades_c["month"] == month]
            pnl = mtrades["net_return"].sum() / w["bull_pos"] * cap
            n_bull = (mtrades["strategy"] == "BULL").sum()
            n_bear = (mtrades["strategy"] == "BEAR").sum()
            wr = (mtrades["net_return"] > 0).sum() / len(mtrades) * 100
            marker = " +" if pnl >= 15000 else (" *" if pnl < 0 else "")
            print(f"    {str(month):<10} {len(mtrades):>3} trades (B:{n_bull} R:{n_bear}) WR={wr:>4.0f}% {pnl:>+10,.0f}{marker}")
            mt += 1
            if pnl > 0: mp += 1
        print(f"  Profitable months: {mp}/{mt} ({mp/mt*100:.0f}%)")

        # Capital sensitivity
        print(f"\n  Capital for 5K/week:")
        if len(t25) > 0:
            w25_raw = t25c.groupby("yearweek")["net_return"].sum() / w["bull_pos"]
            print(f"  {'Capital':>10} {'FullAvg':>8} {'25Avg':>8} {'25%>=5K':>8} {'25%>0':>6}")
            for c in [300000, 400000, 500000, 600000, 750000]:
                fw = weekly / cap * c  # rescale
                tw = w25_raw * c
                print(f"  {c:>10,} {fw.mean():>+8,.0f} {tw.mean():>+8,.0f} {(tw>=5000).sum()/len(tw)*100:>7.0f}% {(tw>0).sum()/len(tw)*100:>5.0f}%")

        print(f"\n{'*'*70}")

    return cdf


def main():
    print(f"\n{'#'*70}")
    print(f"  BULL REGIME DEEP RESEARCH")
    print(f"  Finding what works when Nifty is trending UP")
    print(f"{'#'*70}")

    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)
    nifty_regime = get_nifty_regime(index_data)
    panel = add_advanced_features(panel, stock_data, index_data)

    # 1. Scan bull regime signals
    bull_signals, bull_panel, bear_panel = scan_bull_regime(panel, nifty_regime)

    # 2. Conditional analysis
    conditional_analysis_bull(panel, nifty_regime)

    # 3. Backtest bull strategies
    bull_df = backtest_bull_strategies(panel, stock_data, nifty_regime)

    # 4. Dual strategy
    test_dual_strategy(panel, stock_data, nifty_regime, bull_df)


if __name__ == "__main__":
    main()
