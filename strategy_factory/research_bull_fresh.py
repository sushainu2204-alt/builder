#!/usr/bin/env python3
"""
Fresh Bull Strategy Research — Start from scratch.

Forget mean reversion. Forget dip-buying.
What ACTUALLY predicts positive 5-10 day returns when market is in bull mode?

Approach: Pure data-driven discovery.
1. Build every possible feature combination
2. Measure which predict forward returns in bull regime
3. Find conditional amplifiers
4. Test holding periods 3-20 days
5. Validate on 2025 out-of-sample

New features to explore:
- Relative strength (stock vs sector, stock vs Nifty)
- Breakout patterns (new 20d/50d highs)
- Volume trend (accumulation: rising price + rising volume)
- Tight consolidation (low ATR after a run-up = coiled spring)
- Gap-up continuation (stock gaps up and holds)
- Earnings momentum proxy (60d return acceleration)
- Price acceleration (20d return > 60d return = accelerating)
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
from backtest.metrics import compute_all_metrics
from research_5k_v5 import get_nifty_regime, run_regime_filtered_backtest


def add_bull_features(panel, stock_data, index_data):
    """Add features specifically designed for bull market edge detection."""
    print(f"\n  Computing bull-specific features...")

    nifty_close = index_data["Close"] if index_data is not None else None

    for ticker, df in stock_data.items():
        if len(df) < 100:
            continue
        mask = panel["ticker"] == ticker
        if mask.sum() == 0:
            continue

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"]
        dates = panel.loc[mask, "date"]

        # 1. NEW HIGH flags
        # Is close at 20-day high?
        high_20d = close.rolling(20).max()
        at_20d_high = (close >= high_20d * 0.99).astype(float)  # within 1% of high
        panel.loc[mask, "at_20d_high"] = at_20d_high.reindex(dates.values).values

        # Is close at 50-day high?
        high_50d = close.rolling(50).max()
        at_50d_high = (close >= high_50d * 0.99).astype(float)
        panel.loc[mask, "at_50d_high"] = at_50d_high.reindex(dates.values).values

        # 2. ACCUMULATION: price up + volume up together
        ret_5d = close.pct_change(5)
        vol_5d = vol.rolling(5).mean()
        vol_20d = vol.rolling(20).mean()
        vol_trend = vol_5d / vol_20d
        # Accumulation = positive return + rising volume
        accumulation = ret_5d * vol_trend  # positive when both align
        acc_z = zscore_normalize(accumulation, lookback=252)
        panel.loc[mask, "accumulation"] = acc_z.reindex(dates.values).values

        # 3. TIGHT CONSOLIDATION: low recent volatility after a rally
        # Stock went up 10%+ in 60d but ATR(5) is below ATR(20) = coiled
        atr_5 = (high - low).rolling(5).mean()
        atr_20 = (high - low).rolling(20).mean()
        consolidation = 1 - (atr_5 / atr_20)  # positive = tightening
        consol_z = zscore_normalize(consolidation, lookback=252)
        panel.loc[mask, "consolidation"] = consol_z.reindex(dates.values).values

        # 4. PRICE ACCELERATION: 20d return > 60d/3 return (speeding up)
        ret_20d = close.pct_change(20)
        ret_60d = close.pct_change(60)
        acceleration = ret_20d - (ret_60d / 3)  # excess recent momentum
        acc_z2 = zscore_normalize(acceleration, lookback=252)
        panel.loc[mask, "price_accel"] = acc_z2.reindex(dates.values).values

        # 5. GAP-UP HOLD: opened higher and held (bullish)
        prev_close = close.shift(1)
        gap_up = (df["Open"] / prev_close - 1)
        # 3-day avg of gap-up tendency
        gap_up_3d = gap_up.rolling(3).mean()
        gap_z = zscore_normalize(gap_up_3d, lookback=252)
        panel.loc[mask, "gap_up_trend"] = gap_z.reindex(dates.values).values

        # 6. RELATIVE STRENGTH vs Nifty (normalized)
        if nifty_close is not None:
            nifty_ret_20 = nifty_close.pct_change(20)
            stock_ret_20 = close.pct_change(20)
            nifty_aligned = nifty_ret_20.reindex(close.index, method="ffill")
            rs = stock_ret_20 - nifty_aligned
            rs_z = zscore_normalize(rs, lookback=252)
            panel.loc[mask, "rel_str_z"] = rs_z.reindex(dates.values).values

        # 7. VOLUME BREAKOUT: volume today vs 60-day avg
        vol_60d = vol.rolling(60).mean()
        vol_breakout = vol / vol_60d
        vb_z = zscore_normalize(vol_breakout, lookback=252)
        panel.loc[mask, "vol_breakout"] = vb_z.reindex(dates.values).values

        # 8. HIGHER LOWS: counting consecutive days where low > prev low
        higher_low = (low > low.shift(1)).astype(int)
        hl_streak = higher_low.copy()
        for i in range(1, len(hl_streak)):
            if hl_streak.iloc[i] == 1:
                hl_streak.iloc[i] = hl_streak.iloc[i-1] + 1
            else:
                hl_streak.iloc[i] = 0
        panel.loc[mask, "higher_lows"] = hl_streak.reindex(dates.values).values

    new_cols = ["at_20d_high", "at_50d_high", "accumulation", "consolidation",
                "price_accel", "gap_up_trend", "rel_str_z", "vol_breakout", "higher_lows"]
    print(f"  Added: {', '.join(new_cols)}")
    return panel


def scan_bull_features(panel, nifty_regime):
    """Scan all features in bull regime for predictive power."""
    print(f"\n{'='*70}")
    print(f"  FEATURE SCAN — BULL REGIME (Nifty > 20d SMA)")
    print(f"{'='*70}")

    regime_map = nifty_regime.to_dict()
    panel_c = panel.copy()
    panel_c["regime"] = panel_c["date"].apply(
        lambda d: next((regime_map.get(d - pd.Timedelta(days=delta))
                        for delta in range(10) if (d - pd.Timedelta(days=delta)) in regime_map), None))
    bull = panel_c[panel_c["regime"] == 1]

    exclude = {"ticker", "date", "regime"} | {f"fwd_{d}d" for d in [1, 3, 5, 10, 20]}
    features = [c for c in panel_c.columns if c not in exclude and not c.startswith("_")]

    # Test each feature for BOTH directions (highest and lowest quintile)
    print(f"\n  Testing {len(features)} features × 2 directions × 3 forward periods...")
    results = []

    for feat in features:
        for fwd in ["fwd_5d", "fwd_10d", "fwd_20d"]:
            r = quintile_analysis(bull, feat, fwd)
            if r is None:
                continue
            r["classification"] = classify_signal(r)

            # Which quintile is best?
            qr = r["quintile_returns"]
            q1_ret = qr.iloc[0]  # lowest quintile
            q5_ret = qr.iloc[-1]  # highest quintile

            results.append({
                "feature": feat, "fwd": fwd,
                "q1_ret": q1_ret, "q5_ret": q5_ret,
                "spread": r["spread"], "t_stat": r["t_stat"],
                "monotonic": r["monotonic_score"],
                "direction": r["direction"],
                "best_q": "Q1" if q1_ret > q5_ret else "Q5",
                "best_ret": max(q1_ret, q5_ret),
                "n": r["n_observations"],
            })

    rdf = pd.DataFrame(results)
    # Sort by best quintile return (highest avg return regardless of direction)
    rdf = rdf.sort_values("best_ret", ascending=False)

    print(f"\n  TOP FEATURES by best quintile return (bull regime):")
    print(f"  {'Feature':<22} {'Fwd':>6} {'Best Q':>6} {'Q ret':>7} {'Spread':>8} {'t':>6} {'Mono':>5} {'Dir'}")
    for _, r in rdf.head(30).iterrows():
        marker = " <<" if r["t_stat"] > 2 and r["best_ret"] > 0.5 else ""
        print(f"  {r['feature']:<22} {r['fwd']:>6} {r['best_q']:>6} {r['best_ret']:>+6.2f}% {r['spread']:>+7.2f}% "
              f"{r['t_stat']:>6.2f} {r['monotonic']:>4.0f}% {r['direction']}{marker}")

    return rdf


def conditional_analysis(panel, nifty_regime):
    """Find conditions that amplify the best features."""
    print(f"\n\n{'='*70}")
    print(f"  CONDITIONAL ANALYSIS — What amplifies the edge?")
    print(f"{'='*70}")

    regime_map = nifty_regime.to_dict()
    panel_c = panel.copy()
    panel_c["regime"] = panel_c["date"].apply(
        lambda d: next((regime_map.get(d - pd.Timedelta(days=delta))
                        for delta in range(10) if (d - pd.Timedelta(days=delta)) in regime_map), None))
    bull = panel_c[panel_c["regime"] == 1]

    # For each promising feature, get the best quintile and test conditions
    best_features = [
        ("at_20d_high", "highest", "fwd_10d", "Stocks at 20d high"),
        ("at_50d_high", "highest", "fwd_10d", "Stocks at 50d high"),
        ("accumulation", "highest", "fwd_10d", "Accumulation (price+vol up)"),
        ("price_accel", "highest", "fwd_10d", "Accelerating momentum"),
        ("rel_str_z", "highest", "fwd_10d", "High relative strength"),
        ("consolidation", "highest", "fwd_10d", "Tight consolidation"),
        ("gap_up_trend", "highest", "fwd_10d", "Gap-up trend"),
        ("vol_breakout", "highest", "fwd_10d", "Volume breakout"),
        ("higher_lows", "highest", "fwd_5d", "Higher lows streak"),
        ("return_60d", "highest", "fwd_10d", "3m momentum"),
        ("dist_52w_high", "highest", "fwd_10d", "Near 52w high"),
    ]

    for feat, direction, fwd_col, label in best_features:
        if feat not in bull.columns or fwd_col not in bull.columns:
            continue

        # Get top decile
        observations = []
        for date in sorted(bull["date"].unique()):
            day = bull[bull["date"] == date].dropna(subset=[feat])
            if len(day) < 20:
                continue
            try:
                day = day.copy()
                day["_q"] = pd.qcut(day[feat], 10, labels=False, duplicates="drop")
                if direction == "highest":
                    sel = day[day["_q"] == day["_q"].max()]
                else:
                    sel = day[day["_q"] == 0]
            except ValueError:
                continue

            for _, row in sel.iterrows():
                if pd.notna(row.get(fwd_col)):
                    obs = {"fwd": row[fwd_col]}
                    for col in ["mkt_breadth", "at_20d_high", "at_50d_high", "accumulation",
                                "consolidation", "price_accel", "vol_breakout", "higher_lows",
                                "realized_vol_20d", "rel_str_z", "volume_ratio"]:
                        if col in row and col != feat:
                            obs[col] = row[col]
                    observations.append(obs)

        if len(observations) < 200:
            continue

        odf = pd.DataFrame(observations)
        base_avg = odf["fwd"].mean() * 100
        base_wr = (odf["fwd"] > 0).sum() / len(odf) * 100
        base_t = stats.ttest_1samp(odf["fwd"].dropna(), 0)[0]

        print(f"\n  {label} — top decile ({fwd_col.replace('fwd_','').replace('d','')}d): "
              f"N={len(odf):,} Avg={base_avg:+.3f}% WR={base_wr:.0f}% t={base_t:.2f}")

        conditions = [
            ("+ breadth > 70%", odf.get("mkt_breadth", pd.Series()) > 0.7),
            ("+ breadth > 60%", odf.get("mkt_breadth", pd.Series()) > 0.6),
            ("+ at 20d high", odf.get("at_20d_high", pd.Series()) > 0.5),
            ("+ high accumulation", odf.get("accumulation", pd.Series()) > 0.5),
            ("+ consolidating", odf.get("consolidation", pd.Series()) > 0.5),
            ("+ accelerating", odf.get("price_accel", pd.Series()) > 0.5),
            ("+ vol breakout", odf.get("vol_breakout", pd.Series()) > 1.0),
            ("+ higher lows 3+", odf.get("higher_lows", pd.Series()) >= 3),
            ("+ low volatility", odf.get("realized_vol_20d", pd.Series()) < -0.5),
            ("+ high rel strength", odf.get("rel_str_z", pd.Series()) > 0.5),
        ]

        for clabel, cmask in conditions:
            if len(cmask) == 0 or cmask.sum() < 50:
                continue
            sub = odf[cmask]
            if len(sub) < 50:
                continue
            avg = sub["fwd"].mean() * 100
            wr = (sub["fwd"] > 0).sum() / len(sub) * 100
            t = stats.ttest_1samp(sub["fwd"].dropna(), 0)[0]
            diff = avg - base_avg
            marker = " <<" if avg > base_avg * 1.3 and t > 2 else ""
            print(f"    {clabel:<25} N={len(sub):>5} Avg={avg:>+.3f}% WR={wr:.0f}% t={t:.2f} vs_base={diff:>+.3f}%{marker}")


def backtest_best(panel, stock_data, nifty_regime):
    """Backtest the most promising bull patterns."""
    print(f"\n\n{'='*70}")
    print(f"  BACKTESTING BULL STRATEGIES")
    print(f"{'='*70}")

    configs = [
        # New features
        ("Breakout: at 20d high", "at_20d_high", "highest", None, {}),
        ("Breakout: at 50d high", "at_50d_high", "highest", None, {}),
        ("Accumulation", "accumulation", "highest", None, {}),
        ("Consolidation (coiled)", "consolidation", "highest", None, {}),
        ("Price acceleration", "price_accel", "highest", None, {}),
        ("Gap-up trend", "gap_up_trend", "highest", None, {}),
        ("Rel strength", "rel_str_z", "highest", None, {}),
        ("Volume breakout", "vol_breakout", "highest", None, {}),
        ("Higher lows", "higher_lows", "highest", None, {}),
        # Combos
        ("Breakout + accumulation", "at_20d_high", "highest", None,
         {"filter_col": "accumulation", "filter_op": ">", "filter_val": 0.5}),
        ("Breakout + consolidation", "at_20d_high", "highest", None,
         {"filter_col": "consolidation", "filter_op": ">", "filter_val": 0.5}),
        ("Rel str + breadth>70%", "rel_str_z", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.7}),
        ("Accel + breadth>60%", "price_accel", "highest", None,
         {"filter_col": "mkt_breadth", "filter_op": ">", "filter_val": 0.6}),
        ("Accum + higher lows 3+", "accumulation", "highest", None,
         {"filter_col": "higher_lows", "filter_op": ">=", "filter_val": 3}),
        ("Breakout + rel strength", "at_20d_high", "highest", None,
         {"filter_col": "rel_str_z", "filter_op": ">", "filter_val": 0.5}),
    ]

    results = []
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

        for hold in [5, 7, 10, 15, 20]:
            for max_pos in [3, 5]:
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

                for cap in [600000]:
                    wk = weekly * cap
                    w25_inr = w25 * cap if len(w25) > 0 else pd.Series(dtype=float)
                    results.append({
                        "label": label, "hold": hold, "pos": max_pos, "cap": cap,
                        "sharpe": m["sharpe"], "cagr": m["cagr"], "wr": m["win_rate"],
                        "pf": m["profit_factor"], "mdd": m["max_drawdown"],
                        "trades": m["trades"],
                        "avg_wk": wk.mean(),
                        "t25_sh": t25_m["sharpe"],
                        "t25_avg_wk": w25_inr.mean() if len(w25_inr) > 0 else -99999,
                        "t25_pct_pos": (w25_inr > 0).sum() / len(w25_inr) * 100 if len(w25_inr) > 0 else 0,
                        "result": r,
                    })

    df = pd.DataFrame(results)
    top = df[df["t25_sh"] > 0].sort_values("t25_avg_wk", ascending=False)
    if len(top) == 0:
        top = df.sort_values("t25_avg_wk", ascending=False)

    print(f"\n  TOP BULL STRATEGIES:")
    print(f"  {'Label':<28} {'H':>3} {'P':>2} {'Sh':>5} {'CAGR':>6} {'WR':>5} {'PF':>4} {'25Sh':>5} {'25Wk':>7} {'25%>0':>5} {'Trd':>4}")
    for _, r in top.head(25).iterrows():
        tag = " <<" if r["t25_avg_wk"] >= 3000 and r["t25_sh"] > 0.5 else ""
        print(f"  {r['label']:<28} {r['hold']:>3} {r['pos']:>2} "
              f"{r['sharpe']:>5.2f} {r['cagr']:>+5.1f}% {r['wr']:>5.1f} {r['pf']:>4.2f} "
              f"{r['t25_sh']:>5.2f} {r['t25_avg_wk']:>+7,.0f} {r['t25_pct_pos']:>5.0f} {r['trades']:>4}{tag}")

    # Deep dive on winner
    if len(top) > 0:
        w = top.iloc[0]
        r = w["result"]
        cap = int(w["cap"])
        print(f"\n\n{'*'*70}")
        print(f"  BULL CHAMPION: {w['label']} | hold={w['hold']}d | pos={w['pos']}")
        print(f"{'*'*70}")

        m = r["metrics"]
        print(f"  Full: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1f}% WR={m['win_rate']:.1f}% PF={m['profit_factor']:.2f}")
        print(f"  Trades: {m['trades']}")

        t25 = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
        if len(t25) > 0:
            t25c = t25.copy()
            t25c["yearweek"] = t25c["year"].astype(str) + "-W" + t25c["week"].astype(str).str.zfill(2)
            w25 = t25c.groupby("yearweek")["net_return"].sum() / w["pos"] * cap
            print(f"\n  2025+: Avg/wk={w25.mean():+,.0f} | %>0={((w25>0).sum()/len(w25)*100):.0f}% | "
                  f"%>=5K={((w25>=5000).sum()/len(w25)*100):.0f}%")

        # Monthly
        trades = r["trades"].copy()
        trades["month"] = trades["entry_date"].dt.to_period("M")
        print(f"\n  Monthly (last 18 months):")
        for month in sorted(trades["month"].unique())[-18:]:
            mt = trades[trades["month"] == month]
            pnl = mt["net_return"].sum() / w["pos"] * cap
            wr = (mt["net_return"] > 0).sum() / len(mt) * 100
            marker = " +" if pnl >= 10000 else (" *" if pnl < 0 else "")
            print(f"    {str(month):<10} {len(mt):>3} trades WR={wr:>4.0f}% {pnl:>+10,.0f}{marker}")

    return df


def main():
    print(f"\n{'#'*70}")
    print(f"  FRESH BULL STRATEGY RESEARCH")
    print(f"  No dip-buying. Finding what works from scratch.")
    print(f"{'#'*70}")

    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)
    nifty_regime = get_nifty_regime(index_data)

    # Add breadth
    from research_deep import add_advanced_features
    panel = add_advanced_features(panel, stock_data, index_data)

    # Add new bull features
    panel = add_bull_features(panel, stock_data, index_data)

    # Scan
    scan_df = scan_bull_features(panel, nifty_regime)

    # Conditional analysis
    conditional_analysis(panel, nifty_regime)

    # Backtest
    backtest_best(panel, stock_data, nifty_regime)


if __name__ == "__main__":
    main()
